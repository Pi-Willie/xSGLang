#!/usr/bin/env python3
"""Benchmark shared-prefix branching in XSGLang against repeated HF generate calls."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--levels", type=int, default=8, help="How many branch levels to run")
    parser.add_argument("--block-size", type=int, default=40, help="Tokens generated per block")
    parser.add_argument("--warmup-tokens", type=int, default=2, help="Tiny untimed warmup decode")
    parser.add_argument("--max-running-req", type=int, default=256, help="XSGLang continuation table size")
    parser.add_argument("--out-dir", default="plots", help="Where plots and JSON should go")
    parser.add_argument("--summary-json", default="", help="Skip the benchmark and regenerate plots from an existing summary.json")
    parser.add_argument("--backend", choices=("driver", "xsglang", "hf"), default="driver", help=argparse.SUPPRESS)
    parser.add_argument("--json-output", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_mb(value: int) -> float:
    return value / (1024 * 1024)


def _peak_branches(levels: int) -> int:
    return 2 ** max(0, levels - 1)


def _pythonpath_env() -> str:
    root = Path(__file__).resolve().parents[1]
    repo_python = str(root / "python")
    existing = os.environ.get("PYTHONPATH")
    return repo_python if not existing else f"{repo_python}:{existing}"


def _xs_warm_once(llm: LLM, warmup_tokens: int) -> None:
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


def _hf_warm_once(model, tokenizer, warmup_tokens: int) -> None:
    warm = tokenizer("Write two short rover sentences.", return_tensors="pt")
    warm = {k: v.to("cuda") for k, v in warm.items()}
    with torch.inference_mode():
        _ = model.generate(
            **warm,
            do_sample=False,
            max_new_tokens=warmup_tokens,
            min_new_tokens=warmup_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    _sync_cuda()


def _pad_sequences(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        length = len(seq)
        input_ids[row, :length] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_xsglang(args: argparse.Namespace) -> dict:
    print("Running XSGLang branch benchmark...", flush=True)
    model_path = ensure_local_model_path(args.model)
    llm = LLM(model_path=model_path, cuda_graph_max_bs=0, max_running_req=args.max_running_req)
    prompt_ids = llm.tokenizer.encode(PROMPT)
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
    levels: list[dict] = []
    _xs_warm_once(llm, warmup_tokens=args.warmup_tokens)
    _sync_cuda()
    init_alloc_mb = _cuda_mb(torch.cuda.memory_allocated())
    init_reserved_mb = _cuda_mb(torch.cuda.memory_reserved())

    total_tokens = 0
    total_wall_s = 0.0

    for level in range(args.levels):
        live = len(active)
        before_alloc = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
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
        peak_alloc_delta_mb = _cuda_mb(torch.cuda.max_memory_allocated() - before_alloc)
        peak_reserved_delta_mb = _cuda_mb(torch.cuda.max_memory_reserved() - before_reserved)
        total_tokens += emitted
        total_wall_s += wall_s
        level_stat = {
            "level": level + 1,
            "branches": live,
            "emitted_tokens": emitted,
            "wall_s": wall_s,
            "total_tok_s": emitted / wall_s,
            "tok_s_per_branch": emitted / wall_s / live,
            "peak_alloc_delta_mb": peak_alloc_delta_mb,
            "peak_reserved_delta_mb": max(0.0, peak_reserved_delta_mb),
            "seq_len_tokens": len(prompt_ids) + (level + 1) * args.block_size,
        }
        levels.append(level_stat)
        print(
            f"  XS level {level + 1}: branches={live:3d} total_tok/s={level_stat['total_tok_s']:8.2f}"
            f" tok/s/branch={level_stat['tok_s_per_branch']:7.2f}",
            flush=True,
        )

        if level == args.levels - 1:
            break

        next_active = []
        for req in active:
            children = req.spawn_children([ChildContinuationSpec(), ChildContinuationSpec()])
            next_active.extend(children)
            llm.free_continuation(req)
        active = next_active

    for req in active:
        llm.free_continuation(req)
    llm.shutdown()

    return {
        "backend": "xsglang",
        "model": args.model,
        "levels": levels,
        "total_tokens": total_tokens,
        "total_wall_s": total_wall_s,
        "overall_tok_s": total_tokens / total_wall_s,
        "init_alloc_mb": init_alloc_mb,
        "init_reserved_mb": init_reserved_mb,
    }


def _run_hf(args: argparse.Namespace) -> dict:
    print("Running HF generate branch benchmark...", flush=True)
    model_path = ensure_local_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda").eval()
    model.generation_config.do_sample = False
    model.generation_config.use_cache = True
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    prompt_ids = tokenizer(PROMPT, add_special_tokens=True)["input_ids"]
    active = [list(prompt_ids)]
    levels: list[dict] = []

    _hf_warm_once(model, tokenizer, warmup_tokens=args.warmup_tokens)
    _sync_cuda()
    init_alloc_mb = _cuda_mb(torch.cuda.memory_allocated())
    init_reserved_mb = _cuda_mb(torch.cuda.memory_reserved())

    total_tokens = 0
    total_wall_s = 0.0

    for level in range(args.levels):
        live = len(active)
        batch_ids, attention_mask = _pad_sequences(active, tokenizer.pad_token_id)
        batch_ids = batch_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        before_alloc = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        _sync_cuda()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.block_size,
                min_new_tokens=args.block_size,
            )
        _sync_cuda()
        wall_s = time.perf_counter() - started
        peak_alloc_delta_mb = _cuda_mb(torch.cuda.max_memory_allocated() - before_alloc)
        peak_reserved_delta_mb = _cuda_mb(torch.cuda.max_memory_reserved() - before_reserved)
        output_cpu = output.cpu().tolist()
        emitted = len(output_cpu) * args.block_size
        total_tokens += emitted
        total_wall_s += wall_s
        level_stat = {
            "level": level + 1,
            "branches": live,
            "emitted_tokens": emitted,
            "wall_s": wall_s,
            "total_tok_s": emitted / wall_s,
            "tok_s_per_branch": emitted / wall_s / live,
            "peak_alloc_delta_mb": peak_alloc_delta_mb,
            "peak_reserved_delta_mb": max(0.0, peak_reserved_delta_mb),
            "seq_len_tokens": len(prompt_ids) + (level + 1) * args.block_size,
        }
        levels.append(level_stat)
        print(
            f"  HF level {level + 1}: branches={live:3d} total_tok/s={level_stat['total_tok_s']:8.2f}"
            f" tok/s/branch={level_stat['tok_s_per_branch']:7.2f}",
            flush=True,
        )

        if level != args.levels - 1:
            next_active: list[list[int]] = []
            for seq in output_cpu:
                next_active.append(list(seq))
                next_active.append(list(seq))
            active = next_active

        del batch_ids, attention_mask, output, output_cpu
        gc.collect()
        _sync_cuda()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "backend": "hf_generate",
        "model": args.model,
        "levels": levels,
        "total_tokens": total_tokens,
        "total_wall_s": total_wall_s,
        "overall_tok_s": total_tokens / total_wall_s,
        "init_alloc_mb": init_alloc_mb,
        "init_reserved_mb": init_reserved_mb,
    }


def _plot_series(out_dir: Path, xs: dict, hf: dict) -> None:
    levels = [item["level"] for item in xs["levels"]]
    branches = [item["branches"] for item in xs["levels"]]

    def save_plot(filename: str, title: str, ylabel: str, xs_key: str) -> None:
        plt.figure(figsize=(8, 4.5))
        plt.plot(levels, [item[xs_key] for item in xs["levels"]], marker="o", label="XSGLang")
        plt.plot(levels, [item[xs_key] for item in hf["levels"]], marker="o", label="HF generate")
        plt.title(title)
        plt.xlabel("Tree level")
        plt.ylabel(ylabel)
        plt.xticks(levels)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()

    def cumulative_points(items: list[dict]) -> tuple[list[float], list[int]]:
        wall: list[float] = []
        tokens: list[int] = []
        total_wall = 0.0
        total_tokens = 0
        for item in items:
            total_wall += float(item["wall_s"])
            total_tokens += int(item["emitted_tokens"])
            wall.append(total_wall)
            tokens.append(total_tokens)
        return wall, tokens

    save_plot("total_tok_s.png", "Total Throughput by Level", "tokens / second", "total_tok_s")
    save_plot("tok_s_per_branch.png", "Per-Branch Throughput by Level", "tokens / second / branch", "tok_s_per_branch")
    save_plot("peak_alloc_delta_mb.png", "Peak CUDA Allocation Increase by Level", "MB", "peak_alloc_delta_mb")
    save_plot("peak_reserved_delta_mb.png", "Peak CUDA Reserved Increase by Level", "MB", "peak_reserved_delta_mb")
    save_plot("wall_s.png", "Wall Time per Level", "seconds", "wall_s")

    plt.figure(figsize=(8, 4.5))
    speedup = [x["total_tok_s"] / h["total_tok_s"] for x, h in zip(xs["levels"], hf["levels"])]
    plt.plot(levels, speedup, marker="o")
    plt.title("XSGLang Speedup vs HF by Level")
    plt.xlabel("Tree level")
    plt.ylabel("speedup over HF")
    plt.xticks(levels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_vs_hf.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(branches, [item["total_tok_s"] for item in xs["levels"]], marker="o", label="XSGLang")
    plt.plot(branches, [item["total_tok_s"] for item in hf["levels"]], marker="o", label="HF generate")
    plt.title("Total Throughput vs Live Branches")
    plt.xlabel("Live branches")
    plt.ylabel("tokens / second")
    plt.xscale("log", base=2)
    plt.xticks(branches, branches)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "branch_scaling_total_tok_s.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    xs_base = xs["levels"][0]["tok_s_per_branch"]
    hf_base = hf["levels"][0]["tok_s_per_branch"]
    plt.plot(branches, [item["tok_s_per_branch"] / xs_base for item in xs["levels"]], marker="o", label="XSGLang")
    plt.plot(branches, [item["tok_s_per_branch"] / hf_base for item in hf["levels"]], marker="o", label="HF generate")
    plt.title("Per-Branch Throughput Retention")
    plt.xlabel("Live branches")
    plt.ylabel("relative tok/s/branch")
    plt.xscale("log", base=2)
    plt.xticks(branches, branches)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tok_s_per_branch_retention.png", dpi=160)
    plt.close()

    xs_wall, xs_tokens = cumulative_points(xs["levels"])
    hf_wall, hf_tokens = cumulative_points(hf["levels"])
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs_wall, xs_tokens, marker="o", label="XSGLang")
    plt.plot(hf_wall, hf_tokens, marker="o", label="HF generate")
    plt.title("Cumulative Tokens vs Cumulative Wall Time")
    plt.xlabel("cumulative wall time (s)")
    plt.ylabel("cumulative emitted tokens")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_tokens_vs_time.png", dpi=160)
    plt.close()


def _print_summary(xs: dict, hf: dict, out_dir: Path) -> None:
    print()
    print("Summary")
    print(f"  model: {xs['model']}")
    print(f"  total tokens: {xs['total_tokens']}")
    print(f"  XSGLang overall tok/s: {xs['overall_tok_s']:.2f}")
    print(f"  HF generate overall tok/s: {hf['overall_tok_s']:.2f}")
    print(f"  XSGLang init reserved MB: {xs['init_reserved_mb']:.2f}")
    print(f"  HF init reserved MB: {hf['init_reserved_mb']:.2f}")
    print(f"  plots: {out_dir}")


def _driver(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve()

    shared = [
        "--model",
        args.model,
        "--levels",
        str(args.levels),
        "--block-size",
        str(args.block_size),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--max-running-req",
        str(args.max_running_req),
    ]

    results: dict[str, dict] = {}
    for backend in ("xsglang", "hf"):
        json_path = out_dir / f"{backend}.json"
        cmd = [sys.executable, str(root), "--backend", backend, "--json-output", str(json_path), *shared]
        print(f"Launching {backend} benchmark...", flush=True)
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": _pythonpath_env()})
        with open(json_path, "r", encoding="utf-8") as f:
            results[backend] = json.load(f)

    combined = {
        "model": args.model,
        "levels": args.levels,
        "block_size": args.block_size,
        "peak_branches": _peak_branches(args.levels),
        "xsglang": results["xsglang"],
        "hf_generate": results["hf"],
    }
    _write_json(str(out_dir / "summary.json"), combined)
    _plot_series(out_dir, results["xsglang"], results["hf"])
    _print_summary(results["xsglang"], results["hf"], out_dir)


def main() -> None:
    args = parse_args()
    if args.backend == "driver":
        if args.summary_json:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(args.summary_json, "r", encoding="utf-8") as f:
                combined = json.load(f)
            _plot_series(out_dir, combined["xsglang"], combined["hf_generate"])
            _print_summary(combined["xsglang"], combined["hf_generate"], out_dir)
            return
        _driver(args)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects a CUDA GPU.")

    payload = _run_xsglang(args) if args.backend == "xsglang" else _run_hf(args)
    if args.json_output:
        _write_json(args.json_output, payload)


if __name__ == "__main__":
    main()
