#!/usr/bin/env python3
"""Show pure shared-prefix branching throughput with no hooks in the path."""

from __future__ import annotations

import argparse
import time

import torch

from minisgl.core import BlockSpec, ChildContinuationSpec, SamplingParams
from minisgl.llm import LLM
from minisgl.utils import ensure_local_model_path


PROMPT = (
    "Write a plain field notebook from a Mars rover traverse. "
    "Keep the writing concrete, sentence-based, and operational."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model id or local path")
    parser.add_argument("--levels", type=int, default=8, help="How many tree levels to run")
    parser.add_argument("--block-size", type=int, default=40, help="Tokens per block on each live branch")
    parser.add_argument("--warmup-tokens", type=int, default=2, help="Untimed tokens to warm the engine once")
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
    print("XSGLang Demo: Clean Branch Stress", flush=True)
    print("  what it does: decode fixed-size blocks, then fork every live branch into two clean children", flush=True)
    print(f"  shape: {args.levels} levels x {args.block_size} tokens, widening to {peak_branches} live branches", flush=True)
    print(f"  continuation table: {args.max_running_req} rows", flush=True)
    print("  note: model loading plus first-time CUDA/kernel compile can take around 30 seconds on a fresh machine", flush=True)
    print()


def _warm_once(llm: LLM, warmup_tokens: int) -> None:
    # This tiny untimed pass keeps the printed throughput focused on steady-state
    # decode instead of one-time setup work.
    warm = llm.open_continuation(
        "Write two short rover sentences.",
        SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=warmup_tokens + 8,
        ),
        requested_outputs=("text",),
    )
    warm.run_block(max_new_tokens=warmup_tokens, min_new_tokens=warmup_tokens, request_outputs=("text",))
    llm.free_continuation(warm)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
        metadata={"path": ""},
    )

    active = [root]
    total_tokens = 0
    total_wall_s = 0.0
    best_total_tps = 0.0

    print(f"model: {args.model}")
    print(f"block size: {args.block_size} tokens")
    print(f"peak live branches: {_peak_branches(args.levels)}")
    print()

    print("Warming once before the measured tree starts...", flush=True)
    _warm_once(llm, warmup_tokens=args.warmup_tokens)
    print()

    for level in range(args.levels):
        live = len(active)
        _sync_cuda()
        started = time.perf_counter()
        result = llm.run_block(
            BlockSpec(
                continuation_ids=tuple(req.continuation_id for req in active),
                max_new_tokens=args.block_size,
                min_new_tokens=args.block_size,
                request_outputs=("text",),
            )
        )
        _sync_cuda()
        wall_s = time.perf_counter() - started
        emitted = sum(int(item.emitted_token_ids.numel()) for item in result.continuation_results)

        total_tokens += emitted
        total_wall_s += wall_s
        total_tps = emitted / wall_s if wall_s > 0 else 0.0
        per_branch_tps = total_tps / live if live else 0.0
        best_total_tps = max(best_total_tps, total_tps)
        _print_level(
            level=level + 1,
            levels=args.levels,
            live=live,
            emitted=emitted,
            wall_s=wall_s,
            total_tps=total_tps,
            per_branch_tps=per_branch_tps,
        )

        if level == args.levels - 1:
            break

        # Each leaf becomes two children that reuse the same live prefix state.
        next_active = []
        try:
            for req in active:
                path = str(req.metadata.get("path", ""))
                children = req.spawn_children(
                    [
                        ChildContinuationSpec(metadata={"path": path + "L"}),
                        ChildContinuationSpec(metadata={"path": path + "R"}),
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
    print()
    print("Summary")
    print(f"  final leaves: {len(active)}")
    print(f"  total emitted tokens: {total_tokens}")
    print(f"  overall total_tok/s: {total_tps:.2f}")
    print(f"  best single level:   {best_total_tps:.2f} tok/s")

    for req in active:
        llm.free_continuation(req)
    llm.shutdown()


if __name__ == "__main__":
    main()
