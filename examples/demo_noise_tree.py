#!/usr/bin/env python3
"""Show a clean path and an increasingly noised path in the same branch tree."""

from __future__ import annotations

import argparse
import time

from minisgl.core import BlockSpec, ChildContinuationSpec, HookProgram, SamplingParams
from minisgl.llm import LLM
from minisgl.research import compose_hook_programs, gaussian_noise, make_hook_program
from minisgl.utils import ensure_local_model_path


PROMPT = (
    "Write a field report from a Mars rover crossing rough terrain. "
    "Keep the prose concrete, plain, and sentence-based."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model id or local path")
    parser.add_argument("--levels", type=int, default=7, help="How many 20-token tree levels to run")
    parser.add_argument("--block-size", type=int, default=20, help="Tokens per block before each split")
    parser.add_argument("--noise-step", type=float, default=1.00, help="Extra noise added on the noisy child at each split")
    parser.add_argument("--tail-layers", type=int, default=12, help="How many final layers get the noise hook")
    parser.add_argument(
        "--max-running-req",
        type=int,
        default=256,
        help="Continuation table size. Raise this for deeper trees if your GPU memory allows it.",
    )
    return parser.parse_args()


def _peak_branches(levels: int) -> int:
    return 2 ** max(0, levels - 1)


def _peak_table_rows(levels: int) -> int:
    peak_branches = _peak_branches(levels)
    return peak_branches if levels <= 1 else peak_branches + 1


def _max_safe_levels(max_running_req: int) -> int:
    if max_running_req <= 1:
        return 1
    return max(1, (max_running_req - 1).bit_length())


def _print_capacity_error(args: argparse.Namespace) -> None:
    print("Tree too wide for the current continuation table.", flush=True)
    print(f"  requested levels: {args.levels}", flush=True)
    print(f"  final live branches: {_peak_branches(args.levels)}", flush=True)
    print(f"  peak table rows needed while forking: {_peak_table_rows(args.levels)}", flush=True)
    print(f"  current --max-running-req: {args.max_running_req}", flush=True)
    print(f"  safe levels with this setting: {_max_safe_levels(args.max_running_req)}", flush=True)
    print("  fix: lower --levels or raise --max-running-req if your GPU memory allows it", flush=True)


def _free_if_active(llm: LLM, req) -> None:
    try:
        llm.free_continuation(req)
    except Exception:
        pass


def _print_intro(args: argparse.Namespace) -> None:
    peak_branches = _peak_branches(args.levels)
    print("XSGLang Demo: Noise Tree", flush=True)
    print("  what it does: every split keeps one clean child and makes the other child noisier", flush=True)
    print(f"  shape: {args.levels} levels x {args.block_size} tokens, widening to {peak_branches} live branches", flush=True)
    print("  goal: compare the cleanest full path against the branch that keeps accumulating noise", flush=True)
    print(f"  continuation table: {args.max_running_req} rows", flush=True)
    print("  note: model loading plus first-time CUDA/kernel compile can take around 30 seconds on a fresh machine", flush=True)
    print()


def _resident_program(req) -> HookProgram | None:
    # A continuation can already carry resident hooks from earlier splits.
    # We re-wrap them here so a child can add another noise step instead of
    # replacing the old hook program.
    if (
        req.hook_spec is None
        and req.logit_processor is None
        and req.hook_config is None
        and req.hook_preset_name is None
    ):
        return None
    return HookProgram(
        hook_spec=req.hook_spec,
        logit_processor=req.logit_processor,
        hook_config=req.hook_config,
        hook_preset_name=req.hook_preset_name,
        label="resident-hooks",
    )


def _noise_program(llm: LLM, std: float, tail_layers: int, *, seed: int):
    if std <= 0.0:
        return None
    target_layers = range(max(0, llm.num_layers - tail_layers), llm.num_layers)
    # Two hook points keeps the demo compact while making the degradation visible:
    # late layers perturb the branch state, and pre-LM-head perturbation pushes on logits directly.
    layer_noise = gaussian_noise(std=std, only_last_token=True, seed=seed, state_key="noise_calls")
    head_noise = gaussian_noise(
        std=std,
        only_last_token=True,
        seed=seed + 1000,
        state_key="noise_calls",
    )
    return make_hook_program(
        layer_hooks={layer_idx: layer_noise for layer_idx in target_layers},
        pre_lm_head_hooks=head_noise,
        has_writes=True,
        label=f"noise-{std:.2f}",
    )


def _render_path(req) -> str:
    pieces = []
    for idx, segment in enumerate(req.metadata.get("segments", []), start=1):
        pieces.append(f"### split {idx}\n{(segment or '<no text>').strip()}\n")
    return "\n".join(pieces).strip()


def _warm_demo_path(llm: LLM, block_size: int, tail_layers: int, noise_step: float) -> None:
    # One tiny untimed pass makes the measured tree much easier to read on a
    # fresh machine because setup work is paid up front.
    warm = llm.open_continuation(
        "Write two short Mars rover sentences.",
        SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=block_size + 8,
        ),
        requested_outputs=("text",),
    )
    warm.run_block(max_new_tokens=1, min_new_tokens=1, request_outputs=("text",))
    children = warm.spawn_children(
        [
            ChildContinuationSpec(label="warm-clean"),
            ChildContinuationSpec(
                label="warm-noisy",
                hook_program=_noise_program(
                    llm,
                    std=noise_step,
                    tail_layers=tail_layers,
                    seed=7,
                ),
            ),
        ]
    )
    llm.run_block(
        BlockSpec(
            continuation_ids=tuple(child.continuation_id for child in children),
            max_new_tokens=1,
            min_new_tokens=1,
            request_outputs=("text",),
        )
    )
    for child in children:
        llm.free_continuation(child)
    llm.free_continuation(warm)


def _print_level(
    *,
    level: int,
    levels: int,
    live: int,
    emitted: int,
    wall_s: float,
    total_tps: float,
    per_branch_tps: float,
) -> None:
    print(f"[{level}/{levels}] running {live} live branches")
    print(
        f"  emitted {emitted} tokens in {wall_s:.2f}s"
        f" | total_tok/s {total_tps:7.2f}"
        f" | tok/s/branch {per_branch_tps:6.2f}"
    )


def main() -> None:
    args = parse_args()
    _print_intro(args)
    if _peak_table_rows(args.levels) > args.max_running_req:
        _print_capacity_error(args)
        return

    # Keep one root continuation alive, then decode a block on every leaf and
    # split each leaf into a clean child plus a noisier child.
    print("Loading model and preparing the engine...", flush=True)
    print("Checking the local model cache...", flush=True)
    local_model_path = ensure_local_model_path(args.model)
    if local_model_path != args.model:
        print(f"  using local snapshot: {local_model_path}", flush=True)
    llm = LLM(model_path=local_model_path, cuda_graph_max_bs=0, max_running_req=args.max_running_req)
    root = llm.open_continuation(
        PROMPT,
        SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.levels * args.block_size + 64,
        ),
        requested_outputs=("text",),
        metadata={"path": "", "noise_std": 0.0, "segments": []},
    )

    active = [root]
    total_tokens = 0
    total_wall_s = 0.0

    print(f"model: {args.model}")
    print(f"tree levels: {args.levels}")
    print(f"block size: {args.block_size} tokens")
    print(f"noise step: {args.noise_step:.2f}")
    print(f"peak live branches: {_peak_branches(args.levels)}")
    print()

    print("Warming once before the measured tree starts...", flush=True)
    _warm_demo_path(llm, block_size=args.block_size, tail_layers=args.tail_layers, noise_step=args.noise_step)
    print()

    for level in range(args.levels):
        started = time.perf_counter()
        result = llm.run_block(
            BlockSpec(
                continuation_ids=tuple(req.continuation_id for req in active),
                max_new_tokens=args.block_size,
                min_new_tokens=args.block_size,
                request_outputs=("text",),
            )
        )
        wall_s = time.perf_counter() - started
        total_wall_s += wall_s

        by_id = {req.continuation_id: req for req in active}
        block_tokens = 0
        for item in result.continuation_results:
            req = by_id[item.continuation_id]
            req.metadata.setdefault("segments", []).append(item.text or "")
            block_tokens += int(item.emitted_token_ids.numel())

        total_tokens += block_tokens
        total_tps = block_tokens / wall_s if wall_s > 0 else 0.0
        per_branch_tps = total_tps / len(active) if active else 0.0
        _print_level(
            level=level + 1,
            levels=args.levels,
            live=len(active),
            emitted=block_tokens,
            wall_s=wall_s,
            total_tps=total_tps,
            per_branch_tps=per_branch_tps,
        )

        if level == args.levels - 1:
            break

        # Every leaf becomes two children:
        # - the clean child keeps its current noise level
        # - the noisy child gets one more step of noise
        next_active = []
        try:
            for req in active:
                current_std = float(req.metadata.get("noise_std", 0.0))
                path = str(req.metadata.get("path", ""))
                segments = list(req.metadata.get("segments", []))
                depth = len(path)
                children = req.spawn_children(
                    [
                        ChildContinuationSpec(
                            label=f"{path}C",
                            metadata={
                                "path": path + "C",
                                "noise_std": current_std,
                                "segments": list(segments),
                            },
                        ),
                        ChildContinuationSpec(
                            label=f"{path}N",
                            metadata={
                                "path": path + "N",
                                "noise_std": current_std + args.noise_step,
                                "segments": list(segments),
                            },
                            hook_program=compose_hook_programs(
                                _resident_program(req),
                                _noise_program(
                                    llm,
                                    std=args.noise_step,
                                    tail_layers=args.tail_layers,
                                    seed=7 + depth,
                                ),
                                label="resident-plus-noise",
                            ),
                        ),
                    ]
                )
                next_active.extend(children)
                llm.free_continuation(req)
        except RuntimeError as exc:
            if "No free table slots left for fork" not in str(exc):
                raise
            print()
            print("Stopped early: the continuation table filled up during branching.", flush=True)
            _print_capacity_error(args)
            for req in next_active:
                _free_if_active(llm, req)
            for req in active:
                _free_if_active(llm, req)
            llm.shutdown()
            return
        active = next_active
        print()

    total_tps = total_tokens / total_wall_s if total_wall_s > 0 else 0.0
    top = min(active, key=lambda req: (float(req.metadata["noise_std"]), req.metadata["path"]))
    bottom = max(active, key=lambda req: (float(req.metadata["noise_std"]), req.metadata["path"]))

    print()
    print("Summary")
    print(f"  final leaves: {len(active)}")
    print(f"  total emitted tokens: {total_tokens}")
    print(f"  overall total_tok/s: {total_tps:.2f}")

    print()
    print("Clean path: never picked for extra noise")
    print(f"  branch path: {top.metadata['path'] or '<root>'}")
    print(f"  final noise std: {float(top.metadata['noise_std']):.2f}")
    print(_render_path(top))

    print()
    print("Noisiest path: picked for extra noise at every split")
    print(f"  branch path: {bottom.metadata['path'] or '<root>'}")
    print(f"  final noise std: {float(bottom.metadata['noise_std']):.2f}")
    print(_render_path(bottom))

    for req in active:
        llm.free_continuation(req)
    llm.shutdown()


if __name__ == "__main__":
    main()
