#!/usr/bin/env python3
"""Measure continuation throughput and branch mechanics for branch-GRPO loops."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from minisgl.core import OUTPUT_TEXT, BlockSpec, ChildContinuationSpec, SamplingParams
from minisgl.llm import LLM
from minisgl.utils import ensure_local_model_path

PROMPT = (
    "You are solving a short reasoning task. Keep the answer concise and continue "
    "with the most useful next step."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model id or local path")
    parser.add_argument("--label", default="run", help="Label stored in the JSON output")
    parser.add_argument("--levels", type=int, default=8, help="Binary-tree levels")
    parser.add_argument("--block-size", type=int, default=32, help="Tokens per branch per level")
    parser.add_argument("--single-trace-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--max-running-req", type=int, default=256)
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--request-text", action="store_true")
    parser.add_argument("--json-output", required=True)
    return parser.parse_args()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_branches(levels: int) -> int:
    return 2 ** max(0, levels - 1)


def _peak_table_rows(levels: int) -> int:
    peak = _peak_branches(levels)
    return peak if levels <= 1 else peak + 1


def _requested_outputs(args: argparse.Namespace) -> tuple[str, ...]:
    return (OUTPUT_TEXT,) if args.request_text else ()


def _sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=max_tokens,
    )


def _run_block(llm: LLM, reqs: list[Any], tokens: int, outputs: tuple[str, ...]):
    _sync_cuda()
    started = time.perf_counter()
    result = llm.run_block(
        BlockSpec(
            continuation_ids=tuple(req.continuation_id for req in reqs),
            max_new_tokens=tokens,
            min_new_tokens=tokens,
            request_outputs=outputs,
        )
    )
    _sync_cuda()
    wall_s = time.perf_counter() - started
    emitted = sum(int(item.emitted_token_ids.numel()) for item in result.continuation_results)
    return result, emitted, wall_s


def _warm_once(llm: LLM, outputs: tuple[str, ...], warmup_tokens: int) -> None:
    warm = llm.open_continuation(
        "Warm the engine with a tiny deterministic decode.",
        _sampling_params(warmup_tokens + 8),
        requested_outputs=outputs,
    )
    _run_block(llm, [warm], warmup_tokens, outputs)
    llm.free_continuation(warm)


def _single_trace(llm: LLM, args: argparse.Namespace, outputs: tuple[str, ...]) -> dict[str, Any]:
    req = llm.open_continuation(
        PROMPT,
        _sampling_params(args.single_trace_tokens + 64),
        requested_outputs=outputs,
    )
    _, emitted, wall_s = _run_block(llm, [req], args.single_trace_tokens, outputs)
    llm.free_continuation(req)
    tok_s = emitted / wall_s if wall_s > 0 else 0.0
    print(
        f"single_trace: emitted={emitted} wall={wall_s:.4f}s "
        f"tok/s={tok_s:.2f}",
        flush=True,
    )
    return {
        "emitted_tokens": emitted,
        "wall_s": wall_s,
        "tok_s": tok_s,
    }


def _branch_tree(llm: LLM, args: argparse.Namespace, outputs: tuple[str, ...]) -> dict[str, Any]:
    if _peak_table_rows(args.levels) > args.max_running_req:
        raise ValueError(
            f"levels={args.levels} needs {_peak_table_rows(args.levels)} table rows, "
            f"but max_running_req={args.max_running_req}"
        )

    root = llm.open_continuation(
        PROMPT,
        _sampling_params(args.levels * args.block_size + 64),
        requested_outputs=outputs,
        metadata={"path": ""},
    )
    active = [root]
    level_stats: list[dict[str, Any]] = []
    total_tokens = 0
    total_decode_wall_s = 0.0
    total_spawn_wall_s = 0.0
    total_free_wall_s = 0.0
    total_children = 0
    best_total_tok_s = 0.0

    for level in range(args.levels):
        live = len(active)
        _, emitted, decode_wall_s = _run_block(llm, active, args.block_size, outputs)
        total_tokens += emitted
        total_decode_wall_s += decode_wall_s
        total_tok_s = emitted / decode_wall_s if decode_wall_s > 0 else 0.0
        tok_s_per_branch = total_tok_s / live if live else 0.0
        best_total_tok_s = max(best_total_tok_s, total_tok_s)

        spawn_wall_s = None
        free_wall_s = None
        children_count = 0
        if level < args.levels - 1:
            next_active = []
            _sync_cuda()
            spawn_started = time.perf_counter()
            for req in active:
                path = str(req.metadata.get("path", ""))
                next_active.extend(
                    req.spawn_children(
                        [
                            ChildContinuationSpec(metadata={"path": path + "L"}),
                            ChildContinuationSpec(metadata={"path": path + "R"}),
                        ]
                    )
                )
            _sync_cuda()
            spawn_wall_s = time.perf_counter() - spawn_started

            free_started = time.perf_counter()
            for req in active:
                llm.free_continuation(req)
            _sync_cuda()
            free_wall_s = time.perf_counter() - free_started

            children_count = len(next_active)
            total_children += children_count
            total_spawn_wall_s += spawn_wall_s
            total_free_wall_s += free_wall_s
            active = next_active

        print(
            f"level={level + 1:02d} branches={live:4d} emitted={emitted:6d} "
            f"decode={decode_wall_s:.4f}s total_tok/s={total_tok_s:9.2f} "
            f"tok/s/branch={tok_s_per_branch:8.2f}",
            flush=True,
        )
        if spawn_wall_s is not None:
            print(
                f"  spawn children={children_count:4d} wall={spawn_wall_s * 1000.0:8.2f}ms "
                f"children/s={children_count / spawn_wall_s if spawn_wall_s > 0 else 0.0:9.2f} "
                f"free_wall={free_wall_s * 1000.0 if free_wall_s is not None else 0.0:8.2f}ms",
                flush=True,
            )

        level_stats.append(
            {
                "level": level + 1,
                "live_branches": live,
                "emitted_tokens": emitted,
                "decode_wall_s": decode_wall_s,
                "total_tok_s": total_tok_s,
                "tok_s_per_branch": tok_s_per_branch,
                "spawn_wall_s": spawn_wall_s,
                "free_wall_s": free_wall_s,
                "spawn_children": children_count,
                "spawn_children_s": (
                    children_count / spawn_wall_s
                    if spawn_wall_s is not None and spawn_wall_s > 0
                    else None
                ),
            }
        )

    for req in active:
        llm.free_continuation(req)

    overall_tok_s = total_tokens / total_decode_wall_s if total_decode_wall_s > 0 else 0.0
    spawn_children_s = total_children / total_spawn_wall_s if total_spawn_wall_s > 0 else 0.0
    best_level = max(level_stats, key=lambda item: item["total_tok_s"])
    return {
        "summary": {
            "levels": args.levels,
            "block_size": args.block_size,
            "final_live_branches": level_stats[-1]["live_branches"],
            "total_emitted_tokens": total_tokens,
            "total_decode_wall_s": total_decode_wall_s,
            "overall_total_tok_s": overall_tok_s,
            "best_single_level_tok_s": best_total_tok_s,
            "best_level": best_level["level"],
            "best_level_branches": best_level["live_branches"],
            "total_spawn_children": total_children,
            "total_spawn_wall_s": total_spawn_wall_s,
            "overall_spawn_children_s": spawn_children_s,
            "total_free_wall_s": total_free_wall_s,
        },
        "levels_detail": level_stats,
    }


def main() -> None:
    args = parse_args()
    outputs = _requested_outputs(args)
    print(f"label: {args.label}", flush=True)
    print(f"model: {args.model}", flush=True)
    print(
        f"shape: levels={args.levels} block_size={args.block_size} "
        f"peak_branches={_peak_branches(args.levels)}",
        flush=True,
    )
    local_model_path = ensure_local_model_path(args.model)
    if local_model_path != args.model:
        print(f"using local snapshot: {local_model_path}", flush=True)

    llm = LLM(
        model_path=local_model_path,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_running_req=args.max_running_req,
        memory_ratio=args.memory_ratio,
    )
    try:
        _warm_once(llm, outputs, args.warmup_tokens)
        single = _single_trace(llm, args, outputs)
        tree = _branch_tree(llm, args, outputs)
    finally:
        llm.shutdown()

    payload = {
        "label": args.label,
        "model": args.model,
        "resolved_model_path": local_model_path,
        "request_text": args.request_text,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "memory_ratio": args.memory_ratio,
        "max_running_req": args.max_running_req,
        "single_trace": single,
        "branch_tree": tree,
    }
    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote JSON: {output_path}", flush=True)


if __name__ == "__main__":
    main()
