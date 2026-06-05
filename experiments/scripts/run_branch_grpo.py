#!/usr/bin/env python3
"""Run Branch-Dr.GRPO: health gate (plan.txt section 4/23) or training to plateau.

Examples:
  # Phase 4 loop-health gate (smoke config, a few updates, assert health):
  PYTHONPATH=python python experiments/scripts/run_branch_grpo.py \
      --model experiments/sft/qwen3_4b_base_v1/model --config smoke \
      --output-dir experiments/runs/health_gate --health-gate --updates 6

  # Training run with periodic held-out eval + checkpoints:
  PYTHONPATH=python python experiments/scripts/run_branch_grpo.py \
      --model experiments/sft/qwen3_4b_base_v1/model --config main \
      --output-dir experiments/runs/branch_main --updates 2000 \
      --eval-every 50 --checkpoint-every 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

# cuDNN SDPA JIT backend is broken on this image; force prebuilt flash/mem-efficient.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from minisgl.branch_grpo.config import (  # noqa: E402
    BranchGRPOConfig, main_config, main_v2_config, round3_config, smoke_config,
)
from minisgl.branch_grpo.data import iter_openr1_train_rows  # noqa: E402
from minisgl.branch_grpo.loop import BranchGRPOLoop  # noqa: E402
from minisgl.llm import LLM  # noqa: E402
from minisgl.utils import ensure_local_model_path  # noqa: E402

PROBE_PROMPT = (
    "Solve the math problem. Write your reasoning inside <think>...</think> and the final "
    "short answer inside <answer>...</answer>.\n\nProblem:\nWhat is 7 times 6?\n\nSolution:\n"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_config(name: str) -> BranchGRPOConfig:
    return {"smoke": smoke_config, "main": main_config, "main_v2": main_v2_config, "round3": round3_config}[name]()


def build_loop(args: argparse.Namespace) -> BranchGRPOLoop:
    model_path = ensure_local_model_path(args.model)
    cfg = make_config(args.config)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    llm = LLM(
        model_path, dtype=torch.bfloat16,
        max_running_req=args.max_running_req, memory_ratio=args.memory_ratio,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
    )
    return BranchGRPOLoop(
        llm=llm, trainer_model=model, hf_config=hf_config, config=cfg,
        max_packed_tokens=args.max_packed_tokens, device="cuda",
        on_policy_old_logprobs=not args.use_xsglang_old_logprobs,
        use_wave_rollout=args.wave_rollout,
    )


def prompt_batches(heldout_path: Path, total_prompts: int, per_update: int, skip_prompts: int = 0):
    rows, batch = [], []
    seen = 0
    for row in iter_openr1_train_rows(heldout_path=heldout_path if heldout_path.exists() else None):
        seen += 1
        if seen <= skip_prompts:  # deterministic skip of already-consumed prompts on resume
            continue
        batch.append(row)
        if len(batch) == per_update:
            rows.append(batch)
            batch = []
        if len(rows) * per_update >= total_prompts:
            break
    return rows


def run_health_gate(loop: BranchGRPOLoop, args: argparse.Namespace, out: Path) -> None:
    cfg = loop.config
    report: dict[str, Any] = {"config": cfg.name, "denominator_tokens": cfg.denominator_tokens}

    init_parity = loop.parity_probe(PROBE_PROMPT, n_tokens=24)
    report["init_parity"] = init_parity
    init_probe_lp = _probe_logprobs(loop)

    heldout = Path(args.heldout)
    batches = prompt_batches(heldout, cfg.prompts_per_update * args.updates, cfg.prompts_per_update)

    per_update = []
    denom_constant = True
    any_weight_move = False
    for uid, batch in enumerate(batches):
        m = loop.run_update(uid, batch, capture_weight_delta=True)
        per_update.append(m.to_json())
        if abs(m.fields["denominator_tokens"] - cfg.denominator_tokens) > 1e-6:
            denom_constant = False
        if m.fields["weight_delta_l2"] > 0:
            any_weight_move = True
        print(f"[gate] u{uid} reward_slot={m.fields['reward_mean_slot']:.3f} "
              f"grad={m.fields['grad_norm']:.3e} wΔ={m.fields['weight_delta_l2']:.3e} "
              f"mixed={m.fields['branch_mixed_node_rate_all']:.2f} "
              f"peak={m.fields['sys_peak_gpu_gb']:.1f}GB roll={m.fields['sys_rollout_s']:.1f}s "
              f"train={m.fields['sys_train_s']:.1f}s", flush=True)

    final_parity = loop.parity_probe(PROBE_PROMPT, n_tokens=24)
    final_probe_lp = _probe_logprobs(loop)
    xsglang_drift = sum(abs(a - b) for a, b in zip(init_probe_lp, final_probe_lp))
    peaks = [u["sys_peak_gpu_gb"] for u in per_update]
    # Leak detector: the live-memory baseline at each update start (xsglang+trainer, before
    # transient training memory) must be flat. Skip u0 (allocator warmup / first CUDA-graph
    # pools). Per-update PEAK varies with microbatch packing and is reported but not gated.
    baselines = [u["sys_mem_alloc_start_gb"] for u in per_update]
    steady = baselines[1:] if len(baselines) > 1 else baselines
    baseline_growth = (max(steady) - min(steady)) if steady else 0.0
    grads_finite = all(_finite(u["grad_norm"]) and _finite(u["loss"]) for u in per_update)

    checks = {
        "parity_healthy": final_parity["mean_abs_diff"] < args.parity_mean_max
                          and final_parity["max_abs_diff"] < args.parity_max_max,
        "denominator_constant": denom_constant,
        "trainer_weights_moved": any_weight_move,
        "xsglang_generation_changed": xsglang_drift > 1e-6,
        "grads_and_loss_finite": grads_finite,
        "no_memory_leak": baseline_growth < args.mem_growth_gb,
    }
    report.update({
        "final_parity": final_parity,
        "xsglang_probe_logprob_drift_l1": xsglang_drift,
        "peak_gpu_gb": {"min": min(peaks), "max": max(peaks)} if peaks else {},
        "baseline_alloc_gb": {"values": [round(b, 2) for b in baselines], "growth": baseline_growth},
        "checks": checks,
        "all_green": all(checks.values()),
        "per_update": per_update,
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "health_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"checks": checks, "all_green": report["all_green"],
                      "init_parity": init_parity, "final_parity": final_parity,
                      "xsglang_drift": xsglang_drift}, indent=2), flush=True)
    if not report["all_green"]:
        raise SystemExit(1)


@torch.no_grad()
def _probe_logprobs(loop: BranchGRPOLoop) -> list[float]:
    from minisgl.core import OUTPUT_LOGPROBS, OUTPUT_TOKENS, SamplingParams
    req = loop.llm.open_continuation(
        PROBE_PROMPT,
        SamplingParams(temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True, max_tokens=24),
        requested_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS),
    )
    try:
        res = req.run_block(max_new_tokens=16, min_new_tokens=16,
                            request_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS))
        return [float(v) for v in res.continuation_results[0].logprobs.tolist()]
    finally:
        loop.llm.free_continuation(req)


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def run_training(loop: BranchGRPOLoop, args: argparse.Namespace, out: Path,
                 start_update: int = 0, best_acc: float = -1.0) -> None:
    cfg = loop.config
    out.mkdir(parents=True, exist_ok=True)
    metrics_fp = (out / "metrics.jsonl").open("a")
    eval_fp = (out / "eval.jsonl").open("a")
    heldout = Path(args.heldout)
    eval_rows = _read_jsonl(heldout)[: args.eval_limit]

    init_parity = loop.parity_probe(PROBE_PROMPT, n_tokens=24)
    (out / "init_parity.json").write_text(json.dumps(init_parity, indent=2))
    print(f"[init u{start_update}] parity mean={init_parity['mean_abs_diff']:.4f} max={init_parity['max_abs_diff']:.4f}", flush=True)

    remaining = args.updates - start_update
    batches = prompt_batches(heldout, cfg.prompts_per_update * remaining, cfg.prompts_per_update,
                             skip_prompts=cfg.prompts_per_update * start_update)
    for i, batch in enumerate(batches):
        uid = start_update + i
        m = loop.run_update(uid, batch)
        metrics_fp.write(json.dumps(m.to_json()) + "\n"); metrics_fp.flush()
        if uid % args.log_every == 0:
            print(f"[u{uid}] rew={m.fields['reward_mean_slot']:.3f} acc_vc={m.fields['accuracy_per_verifier_call']:.3f} "
                  f"grad={m.fields['grad_norm']:.2e} mixed={m.fields['branch_mixed_node_rate_all']:.2f} "
                  f"sib={m.fields['branch_sibling_disagreement_rate_all']:.2f} "
                  f"roll={m.fields['sys_rollout_s']:.0f}s peak={m.fields['sys_peak_gpu_gb']:.0f}GB", flush=True)
        if args.eval_every and uid % args.eval_every == 0:
            ev = loop.eval_greedy(eval_rows, max_new_tokens=cfg.max_generation_tokens)
            ev["update_id"] = uid
            eval_fp.write(json.dumps(ev) + "\n"); eval_fp.flush()
            print(f"[eval u{uid}] greedy_acc={ev['greedy_accuracy']:.3f} "
                  f"invalid_fmt={ev['invalid_format_rate']:.3f} len={ev['mean_response_length']:.0f}", flush=True)
            if ev["greedy_accuracy"] > best_acc:
                best_acc = ev["greedy_accuracy"]
                _save_checkpoint(loop, out / "best_model", uid, ev)
        if args.checkpoint_every and uid > 0 and uid % args.checkpoint_every == 0:
            _save_last(loop, out, uid + 1, best_acc)
    metrics_fp.close(); eval_fp.close()
    _save_last(loop, out, start_update + len(batches), best_acc)
    print(f"[done] reached update {start_update + len(batches)} best_acc={best_acc:.3f}", flush=True)


def _save_last(loop: BranchGRPOLoop, out: Path, next_update: int, best_acc: float) -> None:
    """Cheap resume checkpoint: model weights + run state (Adam reinit on resume; OK at lr 1e-6)."""
    path = out / "last_model"
    path.mkdir(parents=True, exist_ok=True)
    loop.model.save_pretrained(path, safe_serialization=True)
    loop.llm.tokenizer.save_pretrained(path)
    _save_run_state(out, next_update, best_acc)
    print(f"[ckpt] saved last_model, next_update={next_update}", flush=True)


def _save_checkpoint(loop: BranchGRPOLoop, path: Path, uid: int, ev: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    loop.model.save_pretrained(path, safe_serialization=True)
    loop.llm.tokenizer.save_pretrained(path)
    (path / "checkpoint_meta.json").write_text(json.dumps({"update_id": uid, "eval": ev}, indent=2))
    print(f"[ckpt] saved best model @ u{uid} acc={ev['greedy_accuracy']:.3f}", flush=True)


def _save_run_state(out: Path, next_update: int, best_acc: float) -> None:
    (out / "run_state.json").write_text(json.dumps(
        {"next_update": next_update, "best_greedy_acc": best_acc, "ts": time.time()}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--config", choices=["smoke", "main", "main_v2", "round3"], default="smoke")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--heldout", default="experiments/data/openr1_heldout/openr1_heldout_eval.jsonl")
    p.add_argument("--health-gate", action="store_true")
    p.add_argument("--updates", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-limit", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--max-packed-tokens", type=int, default=8192)
    p.add_argument("--memory-ratio", type=float, default=0.35)
    p.add_argument("--max-running-req", type=int, default=256)
    p.add_argument("--cuda-graph-max-bs", type=int, default=128)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--use-xsglang-old-logprobs", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from <output-dir>/last_model + run_state")
    p.add_argument("--wave-rollout", action="store_true", help="cross-prompt level-major rollout batching")
    p.add_argument("--parity-mean-max", type=float, default=0.05)
    p.add_argument("--parity-max-max", type=float, default=0.3)
    p.add_argument("--mem-growth-gb", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    start_update, best_acc = 0, -1.0
    if args.resume:
        last_model = out / "last_model"
        state_fp = out / "run_state.json"
        if last_model.exists() and state_fp.exists():
            st = json.loads(state_fp.read_text())
            start_update = int(st.get("next_update", 0))
            best_acc = float(st.get("best_greedy_acc", -1.0))
            args.model = str(last_model)  # reload trainer + xsglang from the resumed weights
            print(f"[resume] from {last_model} at update {start_update} best_acc={best_acc:.3f}", flush=True)
        else:
            print("[resume] no checkpoint found; starting fresh", flush=True)
    loop = build_loop(args)
    if args.health_gate:
        run_health_gate(loop, args, out)
    else:
        run_training(loop, args, out, start_update=start_update, best_acc=best_acc)


if __name__ == "__main__":
    main()
