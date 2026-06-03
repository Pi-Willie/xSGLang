#!/usr/bin/env python3
"""Show pure shared-prefix branching throughput with no hooks in the path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from minisgl.core import OUTPUT_TEXT, BlockSpec, ChildContinuationSpec, SamplingParams
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
        "--request-text",
        action="store_true",
        help="Decode block text in results. Leave disabled for pure throughput measurement.",
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help="Largest CUDA graph batch size to capture. Use 0 to disable graphs.",
    )
    parser.add_argument("--json-output", help="Optional path for machine-readable benchmark results")
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


def _requested_outputs(args: argparse.Namespace) -> tuple[str, ...]:
    return (OUTPUT_TEXT,) if args.request_text else ()


def _warm_once(llm: LLM, warmup_tokens: int, requested_outputs: tuple[str, ...]) -> None:
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
        requested_outputs=requested_outputs,
    )
    warm.run_block(
        max_new_tokens=warmup_tokens,
        min_new_tokens=warmup_tokens,
        request_outputs=requested_outputs,
    )
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
    fork_wall_s: float | None = None,
    fork_children: int = 0,
) -> None:
    print(f"[{level}/{levels}] running {live} live branches")
    print(
        f"  emitted {emitted} tokens in {wall_s:.2f}s"
        f" | total_tok/s {total_tps:7.2f}"
        f" | tok/s/branch {per_branch_tps:6.2f}"
    )
    if fork_wall_s is not None:
        fork_rate = fork_children / fork_wall_s if fork_wall_s > 0 else 0.0
        print(
            f"  forked {fork_children} children in {fork_wall_s * 1000.0:.2f} ms"
            f" | children/s {fork_rate:8.2f}"
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
    requested_outputs = _requested_outputs(args)
    llm = LLM(
        model_path=local_model_path,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_running_req=args.max_running_req,
    )
    root = llm.open_continuation(
        PROMPT,
        SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.levels * args.block_size + 64,
        ),
        requested_outputs=requested_outputs,
        metadata={"path": ""},
    )

    active = [root]
    total_tokens = 0
    total_wall_s = 0.0
    best_total_tps = 0.0
    total_fork_wall_s = 0.0
    total_fork_children = 0
    level_stats = []

    print(f"model: {args.model}")
    print(f"block size: {args.block_size} tokens")
    print(f"peak live branches: {_peak_branches(args.levels)}")
    print()

    print("Warming once before the measured tree starts...", flush=True)
    _warm_once(llm, warmup_tokens=args.warmup_tokens, requested_outputs=requested_outputs)
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
                request_outputs=requested_outputs,
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

        if level == args.levels - 1:
            _print_level(
                level=level + 1,
                levels=args.levels,
                live=live,
                emitted=emitted,
                wall_s=wall_s,
                total_tps=total_tps,
                per_branch_tps=per_branch_tps,
            )
            level_stats.append(
                {
                    "level": level + 1,
                    "live_branches": live,
                    "emitted_tokens": emitted,
                    "decode_wall_s": wall_s,
                    "total_tok_s": total_tps,
                    "tok_s_per_branch": per_branch_tps,
                    "fork_wall_s": None,
                    "fork_children": 0,
                    "fork_children_s": None,
                }
            )
            break

        # Each leaf becomes two children that reuse the same live prefix state.
        next_active = []
        fork_started = time.perf_counter()
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
            fork_wall_s = time.perf_counter() - fork_started
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
        fork_children = len(next_active)
        total_fork_wall_s += fork_wall_s
        total_fork_children += fork_children
        _print_level(
            level=level + 1,
            levels=args.levels,
            live=live,
            emitted=emitted,
            wall_s=wall_s,
            total_tps=total_tps,
            per_branch_tps=per_branch_tps,
            fork_wall_s=fork_wall_s,
            fork_children=fork_children,
        )
        level_stats.append(
            {
                "level": level + 1,
                "live_branches": live,
                "emitted_tokens": emitted,
                "decode_wall_s": wall_s,
                "total_tok_s": total_tps,
                "tok_s_per_branch": per_branch_tps,
                "fork_wall_s": fork_wall_s,
                "fork_children": fork_children,
                "fork_children_s": fork_children / fork_wall_s if fork_wall_s > 0 else 0.0,
            }
        )
        active = next_active
        print()

    total_tps = total_tokens / total_wall_s if total_wall_s > 0 else 0.0
    print()
    print("Summary")
    print(f"  final leaves: {len(active)}")
    print(f"  total emitted tokens: {total_tokens}")
    print(f"  overall total_tok/s: {total_tps:.2f}")
    print(f"  best single level:   {best_total_tps:.2f} tok/s")
    if total_fork_children:
        fork_rate = total_fork_children / total_fork_wall_s if total_fork_wall_s > 0 else 0.0
        print(f"  forked children:     {total_fork_children}")
        print(f"  total fork wall:     {total_fork_wall_s:.4f}s")
        print(f"  overall children/s:  {fork_rate:.2f}")

    if args.json_output:
        payload = {
            "model": args.model,
            "levels": args.levels,
            "block_size": args.block_size,
            "max_running_req": args.max_running_req,
            "request_text": args.request_text,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "summary": {
                "final_leaves": len(active),
                "total_emitted_tokens": total_tokens,
                "overall_total_tok_s": total_tps,
                "best_single_level_tok_s": best_total_tps,
                "total_fork_children": total_fork_children,
                "total_fork_wall_s": total_fork_wall_s,
                "overall_fork_children_s": (
                    total_fork_children / total_fork_wall_s if total_fork_wall_s > 0 else 0.0
                ),
            },
            "levels_detail": level_stats,
        }
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote JSON: {output_path}", flush=True)

    for req in active:
        llm.free_continuation(req)
    llm.shutdown()


if __name__ == "__main__":
    main()
