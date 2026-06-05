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
from typing import Any, Iterable

import torch

# cuDNN SDPA JIT backend is broken on this image; force prebuilt flash/mem-efficient.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from minisgl.branch_grpo.config import (  # noqa: E402
    BranchGRPOConfig, bigmath128_config, fixed128_config, main_config, main_v2_config,
    round3_config, smoke_config,
)
from minisgl.branch_grpo.data import iter_big_math_rows, iter_openr1_train_rows  # noqa: E402
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
    return {
        "smoke": smoke_config,
        "main": main_config,
        "main_v2": main_v2_config,
        "round3": round3_config,
        "fixed128": fixed128_config,
        "bigmath128": bigmath128_config,
    }[name]()


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((int(value) + multiple - 1) // multiple) * multiple


def build_loop(args: argparse.Namespace) -> BranchGRPOLoop:
    model_path = ensure_local_model_path(args.model)
    cfg = make_config(args.config)
    wave_rollout = args.wave_rollout and not args.prompt_major_rollout
    required_table_slots = (
        cfg.retained_continuation_slots_per_wave
        if wave_rollout
        else cfg.retained_continuation_slots_per_prompt
    )
    # Forked continuations retain their parents until the rollout tree is freed. A 128-leaf
    # wave can therefore need far more request-table slots than the final active frontier.
    max_running_req = int(args.max_running_req)
    recommended_table_slots = _round_up(required_table_slots, 128)
    if max_running_req < recommended_table_slots:
        print(
            f"[config] bump max_running_req {max_running_req} -> {recommended_table_slots} "
            f"for retained branch table slots (required={required_table_slots})",
            flush=True,
        )
        max_running_req = recommended_table_slots
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    llm = LLM(
        model_path, dtype=torch.bfloat16,
        max_running_req=max_running_req, memory_ratio=args.memory_ratio,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
    )
    return BranchGRPOLoop(
        llm=llm, trainer_model=model, hf_config=hf_config, config=cfg,
        max_packed_tokens=args.max_packed_tokens, device="cuda",
        on_policy_old_logprobs=not args.use_xsglang_old_logprobs,
        use_wave_rollout=wave_rollout,
        entropy_sample_tokens=args.entropy_sample_tokens,
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def train_row_iter(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.train_source == "openr1":
        heldout_path = Path(args.heldout)
        return iter_openr1_train_rows(heldout_path=heldout_path if heldout_path.exists() else None)
    if args.train_source == "big-math-file":
        if args.train_data is None:
            raise ValueError("--train-data is required for --train-source big-math-file")
        return _iter_jsonl(Path(args.train_data))
    if args.train_source == "big-math-stream":
        return iter_big_math_rows(
            dataset_name=args.big_math_dataset,
            split=args.big_math_split,
            min_llama8b_solve_rate=args.min_llama8b_solve_rate,
            seed=args.big_math_seed,
        )
    raise AssertionError(args.train_source)


def prompt_batches(rows_iter: Iterable[dict[str, Any]], total_prompts: int, per_update: int,
                   skip_prompts: int = 0) -> Iterable[list[dict[str, Any]]]:
    batch = []
    seen = 0
    emitted = 0
    for row in rows_iter:
        seen += 1
        if seen <= skip_prompts:  # deterministic skip of already-consumed prompts on resume
            continue
        batch.append(row)
        if len(batch) == per_update:
            yield batch
            emitted += per_update
            batch = []
        if emitted >= total_prompts:
            break


def run_health_gate(loop: BranchGRPOLoop, args: argparse.Namespace, out: Path) -> None:
    cfg = loop.config
    report: dict[str, Any] = {"config": cfg.name, "denominator_tokens": cfg.denominator_tokens}

    init_parity = loop.parity_probe(PROBE_PROMPT, n_tokens=24)
    report["init_parity"] = init_parity
    init_probe_lp = _probe_logprobs(loop)

    batches = prompt_batches(train_row_iter(args), cfg.prompts_per_update * args.updates, cfg.prompts_per_update)

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
    eval_fp = (out / "eval.jsonl").open("a") if args.eval_every else None
    heldout = Path(args.heldout)
    eval_rows = _read_jsonl(heldout)[: args.eval_limit] if args.eval_every else []
    trajectory_dir = out / "trajectories"
    outputs_dir = out / "outputs"
    live_plot_path = Path(args.live_plot) if args.live_plot else out / "live_reward.png"
    plot_data_path = Path(args.plot_data) if args.plot_data else out / "live_reward_data.json"
    plot_series: dict[str, list[float]] = {
        "update_id": [],
        "reward_mean_slot": [],
        "reward_mean_unique_leaf": [],
        "accuracy_per_verifier_call": [],
        "branch_mixed_node_rate_all": [],
        "grad_norm": [],
        "sys_rollout_s": [],
        "sys_train_s": [],
    }
    if plot_data_path.exists():
        try:
            loaded = json.loads(plot_data_path.read_text(encoding="utf-8"))
            for key in plot_series:
                if isinstance(loaded.get(key), list):
                    plot_series[key] = [float(v) for v in loaded[key]]
        except Exception:
            pass

    init_parity = loop.parity_probe(PROBE_PROMPT, n_tokens=24)
    (out / "init_parity.json").write_text(json.dumps(init_parity, indent=2))
    print(f"[init u{start_update}] parity mean={init_parity['mean_abs_diff']:.4f} max={init_parity['max_abs_diff']:.4f}", flush=True)

    remaining = args.updates - start_update
    batches = prompt_batches(train_row_iter(args), cfg.prompts_per_update * remaining, cfg.prompts_per_update,
                             skip_prompts=cfg.prompts_per_update * start_update)
    completed = 0
    for i, batch in enumerate(batches):
        uid = start_update + i
        capture_weight_delta = (
            args.weight_delta_every > 0
            and uid > start_update
            and uid % args.weight_delta_every == 0
        )
        m = loop.run_update(
            uid,
            batch,
            capture_weight_delta=capture_weight_delta,
            trajectory_dir=trajectory_dir,
            outputs_dir=outputs_dir,
        )
        completed += 1
        metrics_fp.write(json.dumps(m.to_json()) + "\n"); metrics_fp.flush()
        _append_plot_series(plot_series, m.to_json())
        _write_live_plot(plot_series, live_plot_path, plot_data_path)
        if uid % args.log_every == 0:
            print(f"[u{uid}] rew={m.fields['reward_mean_slot']:.3f} acc_vc={m.fields['accuracy_per_verifier_call']:.3f} "
                  f"grad={m.fields['grad_norm']:.2e} mixed={m.fields['branch_mixed_node_rate_all']:.2f} "
                  f"H={m.fields.get('mean_token_entropy', 0.0):.1f} "
                  f"sib={m.fields['branch_sibling_disagreement_rate_all']:.2f} "
                  f"roll={m.fields['sys_rollout_s']:.0f}s peak={m.fields['sys_peak_gpu_gb']:.0f}GB", flush=True)
        if args.eval_every and uid % args.eval_every == 0:
            ev = loop.eval_greedy(eval_rows, max_new_tokens=cfg.max_generation_tokens)
            ev["update_id"] = uid
            assert eval_fp is not None
            eval_fp.write(json.dumps(ev) + "\n"); eval_fp.flush()
            print(f"[eval u{uid}] greedy_acc={ev['greedy_accuracy']:.3f} "
                  f"invalid_fmt={ev['invalid_format_rate']:.3f} len={ev['mean_response_length']:.0f}", flush=True)
            score = float(ev["greedy_accuracy"])
            improvement = score - best_acc
            if score > best_acc:
                best_acc = score
                if (
                    not args.disable_checkpoints
                    and not args.disable_best_checkpoints
                    and improvement >= args.best_save_min_delta
                ):
                    _save_checkpoint(loop, out / "best_model", uid, ev)
        elif not args.eval_every:
            score = float(m.fields.get(args.best_metric, 0.0))
            improvement = score - best_acc
            if score > best_acc:
                best_acc = score
                if (
                    not args.disable_checkpoints
                    and not args.disable_best_checkpoints
                    and improvement >= args.best_save_min_delta
                ):
                    _save_checkpoint(loop, out / "best_model", uid, {
                        "selection_metric": args.best_metric,
                        args.best_metric: score,
                        "reward_mean_slot": m.fields.get("reward_mean_slot", 0.0),
                        "reward_mean_unique_leaf": m.fields.get("reward_mean_unique_leaf", 0.0),
                        "accuracy_per_verifier_call": m.fields.get("accuracy_per_verifier_call", 0.0),
                        "update_id": uid,
                    })
        if args.checkpoint_every and not args.disable_checkpoints and uid > 0 and uid % args.checkpoint_every == 0:
            _save_last(loop, out, uid + 1, best_acc)
    metrics_fp.close()
    if eval_fp is not None:
        eval_fp.close()
    if not args.disable_checkpoints:
        _save_last(loop, out, start_update + completed, best_acc)
    else:
        _save_run_state(out, start_update + completed, best_acc)
    print(f"[done] reached update {start_update + completed} best_score={best_acc:.3f}", flush=True)


def _append_plot_series(series: dict[str, list[float]], metrics: dict[str, Any]) -> None:
    for key in series:
        value = metrics.get(key, 0.0)
        try:
            series[key].append(float(value))
        except (TypeError, ValueError):
            series[key].append(0.0)


def _write_live_plot(series: dict[str, list[float]], plot_path: Path, data_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(series, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (plot_path.with_suffix(".plot_error.txt")).write_text(str(exc), encoding="utf-8")
        return
    xs = series["update_id"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=130, sharex=True)
    ax = axes[0]
    ax.plot(xs, series["reward_mean_slot"], label="slot reward", color="#1769aa", linewidth=2)
    ax.plot(xs, series["reward_mean_unique_leaf"], label="unique leaf reward", color="#2e7d32", linewidth=1.6)
    ax.plot(xs, series["accuracy_per_verifier_call"], label="verifier-call acc", color="#7b1fa2", linewidth=1.3)
    ax.set_ylabel("reward")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(xs, series["branch_mixed_node_rate_all"], label="mixed-node rate", color="#ef6c00", linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(xs, series["sys_rollout_s"], label="rollout s", color="#455a64", linewidth=1.2, alpha=0.85)
    ax2.plot(xs, series["sys_train_s"], label="train s", color="#8d6e63", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("update")
    ax.set_ylabel("branch signal")
    ax2.set_ylabel("seconds")
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.suptitle("Branch-Dr.GRPO Big-Math live training")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


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
    score = ev.get("greedy_accuracy", ev.get(ev.get("selection_metric", ""), ev.get("reward_mean_slot", 0.0)))
    print(f"[ckpt] saved best model @ u{uid} score={float(score):.3f}", flush=True)


def _save_run_state(out: Path, next_update: int, best_acc: float) -> None:
    (out / "run_state.json").write_text(json.dumps(
        {"next_update": next_update, "best_score": best_acc, "best_greedy_acc": best_acc, "ts": time.time()}, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--config", choices=["smoke", "main", "main_v2", "round3", "fixed128", "bigmath128"], default="smoke")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--heldout", default="experiments/data/openr1_heldout/openr1_heldout_eval.jsonl")
    p.add_argument("--health-gate", action="store_true")
    p.add_argument("--updates", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--disable-checkpoints", action="store_true")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--max-packed-tokens", type=int, default=2048)
    p.add_argument("--memory-ratio", type=float, default=0.35)
    p.add_argument("--max-running-req", type=int, default=256)
    p.add_argument("--cuda-graph-max-bs", type=int, default=128)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--train-source", choices=["openr1", "big-math-file", "big-math-stream"], default="openr1")
    p.add_argument("--train-data", default=None, help="Materialized JSONL for --train-source big-math-file")
    p.add_argument("--big-math-dataset", default="SynthLabsAI/Big-Math-RL-Verified")
    p.add_argument("--big-math-split", default="train")
    p.add_argument("--big-math-seed", type=int, default=20260605)
    p.add_argument("--min-llama8b-solve-rate", type=float, default=0.1)
    p.add_argument("--best-metric", default="reward_mean_slot")
    p.add_argument("--disable-best-checkpoints", action="store_true")
    p.add_argument("--best-save-min-delta", type=float, default=0.0)
    p.add_argument("--live-plot", default=None)
    p.add_argument("--plot-data", default=None)
    p.add_argument("--use-xsglang-old-logprobs", action="store_true")
    p.add_argument("--entropy-sample-tokens", type=int, default=1024,
                   help="sampled response-token rows per microbatch for entropy health telemetry")
    p.add_argument("--weight-delta-every", type=int, default=0,
                   help="capture full model L2 movement every N updates; 0 disables it")
    p.add_argument("--resume", action="store_true", help="resume from <output-dir>/last_model + run_state")
    p.add_argument("--wave-rollout", action="store_true", default=True, help="cross-prompt level-major rollout batching")
    p.add_argument("--prompt-major-rollout", action="store_true", help="debug fallback: roll out one prompt tree at a time")
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
            best_acc = float(st.get("best_score", st.get("best_greedy_acc", -1.0)))
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
