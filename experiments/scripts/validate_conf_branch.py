#!/usr/bin/env python3
"""Validate confidence-gated branching: are forks landing on low-confidence tokens?

Builds a few real rollout trees and reports the fork stop-reason histogram, actual segment
lengths (how far past the nominal target forks defer), and sibling reward divergence. If most
branch edges stop at "branch_boundary" (top-1 prob <= threshold) rather than forced at the
lookahead cap, the mechanism is well-calibrated. Cheap (~a few trees, no training).
"""
from __future__ import annotations
import argparse, json, collections
from pathlib import Path
import torch
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
from minisgl.llm import LLM
from minisgl.utils import ensure_local_model_path
from minisgl.branch_grpo.config import smoke_config, main_config
from minisgl.branch_grpo.rollout import build_branch_rollout_tree
from minisgl.branch_grpo.records import compute_nominal_slot_q_values


def _read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", choices=["smoke", "main"], default="main")
    ap.add_argument("--heldout", default="experiments/data/openr1_heldout/openr1_heldout_eval.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--memory-ratio", type=float, default=0.6)
    args = ap.parse_args()
    cfg = smoke_config() if args.config == "smoke" else main_config()
    rows = _read_jsonl(Path(args.heldout))[: args.n]
    llm = LLM(ensure_local_model_path(args.model), dtype=torch.bfloat16,
              max_running_req=256, memory_ratio=args.memory_ratio, cuda_graph_max_bs=128)

    finish = collections.Counter()
    seg_by_depth = collections.defaultdict(list)
    mixed_nodes = tot_nodes = 0
    for i, row in enumerate(rows):
        tree = build_branch_rollout_tree(llm=llm, prompt_row=row, prompt_id=i, config=cfg)
        compute_nominal_slot_q_values(tree)
        for e in tree.edges.values():
            finish[e.finish_reason] += 1
            seg_by_depth[e.depth].append(int(e.tokens.shape[0]))
        for node in tree.nodes.values():
            ch = tree.child_edges(node.id)
            if len(ch) >= 2:
                tot_nodes += 1
                qs = [tree.nodes[c.child_node].q_value or 0.0 for c in ch]
                if max(qs) - min(qs) > 1e-8:
                    mixed_nodes += 1
    nominal = list(cfg.branch_targets)
    seg_nominal = [nominal[0]] + [nominal[k]-nominal[k-1] for k in range(1, len(nominal))]
    print(json.dumps({
        "config": cfg.name, "confidence_threshold": cfg.confidence_threshold,
        "boundary_lookahead": cfg.boundary_lookahead, "trees": len(rows),
        "finish_reason_hist": dict(finish),
        "branch_boundary_frac": finish.get("branch_boundary", 0) / max(1, sum(
            finish.get(r, 0) for r in ["branch_boundary", "block_limit", "max_tokens", "forced_branch"])),
        "mean_seg_len_by_depth": {d: round(sum(v)/len(v), 1) for d, v in sorted(seg_by_depth.items())},
        "nominal_seg_len_branch_stages": seg_nominal,
        "mixed_node_rate": round(mixed_nodes / max(1, tot_nodes), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
