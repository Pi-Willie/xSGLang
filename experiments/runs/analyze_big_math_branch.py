#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < x.shape[0]:
        end = start + 1
        while end < x.shape[0] and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _ols(y: np.ndarray, cols: list[np.ndarray]) -> tuple[np.ndarray, float, np.ndarray]:
    x = np.column_stack([np.ones_like(y), *cols])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2, resid


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def build_analysis(metrics: list[dict[str, Any]], data_rows: list[dict[str, Any]], prompts_per_update: int) -> dict[str, Any]:
    joined: list[dict[str, Any]] = []
    for m in metrics:
        uid = int(m["update_id"])
        start = uid * prompts_per_update
        batch = data_rows[start:start + prompts_per_update]
        rates = np.asarray([_safe_float(row.get("llama8b_solve_rate"), float("nan")) for row in batch], dtype=np.float64)
        problems = [str(row.get("problem", row.get("prompt", ""))) for row in batch]
        item = dict(m)
        item.update({
            "llama8b_solve_rate_mean": float(np.nanmean(rates)) if rates.size else float("nan"),
            "llama8b_solve_rate_median": float(np.nanmedian(rates)) if rates.size else float("nan"),
            "llama8b_solve_rate_min": float(np.nanmin(rates)) if rates.size else float("nan"),
            "llama8b_solve_rate_max": float(np.nanmax(rates)) if rates.size else float("nan"),
            "llama8b_solve_rate_std": float(np.nanstd(rates)) if rates.size else float("nan"),
            "prompt_char_len_mean": float(np.mean([len(p) for p in problems])) if problems else float("nan"),
        })
        joined.append(item)

    targets = {
        "accuracy_per_verifier_call": "acc_vc",
        "reward_mean_slot": "reward_slot",
    }
    features = [
        "update_id",
        "llama8b_solve_rate_mean",
        "llama8b_solve_rate_median",
        "llama8b_solve_rate_min",
        "llama8b_solve_rate_std",
        "actual_node_count",
        "actual_edge_count",
        "actual_leaf_count",
        "actual_branch_node_count",
        "unique_leaf_count",
        "actual_generated_tree_tokens",
        "tree_token_ratio",
        "mean_response_char_len",
        "zero_signal_prompt_fraction",
        "branch_mixed_node_rate_all",
        "branch_sibling_disagreement_rate_all",
        "nonzero_weighted_tokens",
        "microbatches",
        "sys_rollout_s",
        "sys_train_s",
        "sys_weight_refresh_s",
    ]
    correlations: dict[str, list[dict[str, float | str]]] = {}
    for target_key, target_name in targets.items():
        y = np.asarray([_safe_float(row.get(target_key), float("nan")) for row in joined], dtype=np.float64)
        rows = []
        for feature in features:
            x = np.asarray([_safe_float(row.get(feature), float("nan")) for row in joined], dtype=np.float64)
            rows.append({
                "feature": feature,
                "pearson": _pearson(x, y),
                "spearman": _spearman(x, y),
            })
        rows.sort(key=lambda r: abs(float(r["spearman"])) if math.isfinite(float(r["spearman"])) else -1.0, reverse=True)
        correlations[target_name] = rows

    update = np.asarray([_safe_float(row["update_id"]) for row in joined], dtype=np.float64)
    acc = np.asarray([_safe_float(row["accuracy_per_verifier_call"], float("nan")) for row in joined], dtype=np.float64)
    reward = np.asarray([_safe_float(row["reward_mean_slot"], float("nan")) for row in joined], dtype=np.float64)
    llama = np.asarray([_safe_float(row["llama8b_solve_rate_mean"], float("nan")) for row in joined], dtype=np.float64)
    mask = np.isfinite(update) & np.isfinite(acc) & np.isfinite(llama)
    trend: dict[str, Any] = {}
    if int(mask.sum()) >= 4:
        update_z = (update[mask] - float(np.mean(update[mask]))) / max(float(np.std(update[mask])), 1e-8)
        llama_z = (llama[mask] - float(np.mean(llama[mask]))) / max(float(np.std(llama[mask])), 1e-8)
        acc_coef, acc_r2, acc_resid = _ols(acc[mask], [llama_z, update_z])
        rew_coef, rew_r2, rew_resid = _ols(reward[mask], [llama_z, update_z])
        trend = {
            "n_updates": int(mask.sum()),
            "raw_acc_update_pearson": _pearson(update[mask], acc[mask]),
            "raw_acc_update_spearman": _spearman(update[mask], acc[mask]),
            "raw_reward_update_pearson": _pearson(update[mask], reward[mask]),
            "difficulty_adjusted_acc_update_coef_per_update_std": float(acc_coef[2]),
            "difficulty_adjusted_acc_llama_coef_per_llama_std": float(acc_coef[1]),
            "difficulty_adjusted_acc_r2": float(acc_r2),
            "difficulty_adjusted_reward_update_coef_per_update_std": float(rew_coef[2]),
            "difficulty_adjusted_reward_llama_coef_per_llama_std": float(rew_coef[1]),
            "difficulty_adjusted_reward_r2": float(rew_r2),
            "acc_residuals": [
                {"update_id": int(u), "residual": float(r)}
                for u, r in zip(update[mask], acc_resid)
            ],
            "reward_residuals": [
                {"update_id": int(u), "residual": float(r)}
                for u, r in zip(update[mask], rew_resid)
            ],
        }
    return {
        "prompts_per_update": prompts_per_update,
        "updates": joined,
        "correlations": correlations,
        "trend": trend,
    }


def write_plot(analysis: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = analysis["updates"]
    x = np.asarray([row["update_id"] for row in rows], dtype=np.float64)
    acc = np.asarray([row["accuracy_per_verifier_call"] for row in rows], dtype=np.float64)
    reward = np.asarray([row["reward_mean_slot"] for row in rows], dtype=np.float64)
    llama = np.asarray([row["llama8b_solve_rate_mean"] for row in rows], dtype=np.float64)
    leaves = np.asarray([row["unique_leaf_count"] for row in rows], dtype=np.float64)
    tokens = np.asarray([row["actual_generated_tree_tokens"] for row in rows], dtype=np.float64)
    wall = np.asarray([
        row["sys_rollout_s"] + row["sys_train_s"] + row.get("sys_weight_refresh_s", 0.0)
        for row in rows
    ], dtype=np.float64)

    trend = analysis.get("trend", {})
    resid_map = {item["update_id"]: item["residual"] for item in trend.get("acc_residuals", [])}
    resid = np.asarray([resid_map.get(int(row["update_id"]), np.nan) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=150)
    ax = axes[0, 0]
    ax.plot(x, acc, marker="o", linewidth=1.4, label="acc per verifier call")
    ax.plot(x, reward, marker="o", linewidth=1.2, label="slot reward", alpha=0.8)
    ax.set_title("Training signal by update")
    ax.set_xlabel("update")
    ax.set_ylabel("reward / acc")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    sc = ax.scatter(llama, acc, c=x, cmap="viridis", s=34)
    ax.set_title("Batch difficulty vs acc")
    ax.set_xlabel("mean llama8b_solve_rate")
    ax.set_ylabel("acc per verifier call")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="update")

    ax = axes[0, 2]
    ax.plot(x, llama, marker="o", linewidth=1.2, color="#6a4c93", label="mean llama pass")
    ax2 = ax.twinx()
    ax2.plot(x, acc, marker=".", linewidth=1.0, color="#1982c4", alpha=0.75, label="acc")
    ax.set_title("Difficulty drift")
    ax.set_xlabel("update")
    ax.set_ylabel("mean llama8b_solve_rate")
    ax2.set_ylabel("acc")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(leaves, acc, c=x, cmap="plasma", s=34)
    ax.set_title("Unique leaves vs acc")
    ax.set_xlabel("unique leaves")
    ax.set_ylabel("acc")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.scatter(tokens, acc, c=wall, cmap="magma", s=34)
    ax.set_title("Generated tokens vs acc")
    ax.set_xlabel("actual generated tree tokens")
    ax.set_ylabel("acc")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.plot(x, resid, marker="o", linewidth=1.2, color="#2a9d8f")
    ax.set_title("Difficulty-adjusted acc residual")
    ax.set_xlabel("update")
    ax.set_ylabel("acc residual after llama pass-rate")
    ax.grid(alpha=0.25)

    fig.suptitle("Big-Math Branch-Dr.GRPO relationships (train batches, not held-out eval)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-plot", type=Path, required=True)
    parser.add_argument("--prompts-per-update", type=int, default=8)
    args = parser.parse_args()

    analysis = build_analysis(
        _read_jsonl(args.metrics),
        _read_jsonl(args.data),
        prompts_per_update=args.prompts_per_update,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_plot(analysis, args.out_plot)

    top = {
        target: rows[:8]
        for target, rows in analysis["correlations"].items()
    }
    print(json.dumps({"trend": analysis["trend"], "top_correlations": top}, indent=2))


if __name__ == "__main__":
    main()
