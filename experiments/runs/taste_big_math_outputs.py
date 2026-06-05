#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def short(text: str, n: int = 900) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    outputs_dir = args.run_dir / "outputs"
    traj_dir = args.run_dir / "trajectories"
    output_files = sorted(outputs_dir.glob("update_*.jsonl"))
    traj_files = sorted(traj_dir.glob("update_*.json"))
    if not output_files:
        raise SystemExit(f"no outputs found in {outputs_dir}")

    total = 0
    reward_sum = 0.0
    nominal_weight_sum = 0
    nominal_reward_sum = 0.0
    empty_pred = 0
    finish = Counter()
    pred_counter = Counter()
    answer_counter = Counter()
    response_hashes = Counter()
    per_update: dict[int, dict[str, Any]] = {}
    examples_correct: list[dict[str, Any]] = []
    examples_wrong: list[dict[str, Any]] = []
    examples_long: list[tuple[int, dict[str, Any]]] = []

    for fp in output_files:
        rows = list(read_jsonl(fp))
        if not rows:
            continue
        uid = int(rows[0].get("update_id", fp.stem.rsplit("_", 1)[-1]))
        upd_reward = 0.0
        upd_nominal = 0
        upd_nominal_reward = 0.0
        upd_preds = Counter()
        upd_empty = 0
        for row in rows:
            total += 1
            reward = float(row.get("reward", 0.0))
            weight = int(row.get("nominal_slot_count", 1) or 1)
            pred = str(row.get("normalized_pred", row.get("pred_answer", "")) or "")
            ans = str(row.get("normalized_answer", row.get("answer", "")) or "")
            text = str(row.get("response_text", ""))
            h = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            response_hashes[h] += 1
            reward_sum += reward
            nominal_weight_sum += weight
            nominal_reward_sum += reward * weight
            upd_reward += reward
            upd_nominal += weight
            upd_nominal_reward += reward * weight
            finish[str(row.get("finish_reason", ""))] += 1
            pred_counter[pred] += 1
            answer_counter[ans] += 1
            upd_preds[pred] += 1
            if not pred:
                empty_pred += 1
                upd_empty += 1
            if reward > 0 and len(examples_correct) < 8:
                examples_correct.append(row)
            if reward <= 0 and len(examples_wrong) < 8:
                examples_wrong.append(row)
            examples_long.append((len(text), row))
        per_update[uid] = {
            "leaf_count": len(rows),
            "reward_mean_unique_leaf": upd_reward / max(1, len(rows)),
            "reward_mean_nominal_slot": upd_nominal_reward / max(1, upd_nominal),
            "empty_pred_fraction": upd_empty / max(1, len(rows)),
            "top_pred_share": max(upd_preds.values()) / max(1, len(rows)),
        }

    rep_trajs = []
    for fp in traj_files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        rep_trajs.append({
            "update_id": d.get("update_id"),
            "reward": d.get("reward"),
            "problem": d.get("problem"),
            "answer": d.get("answer"),
            "pred_answer": d.get("pred_answer"),
            "finish_reason": d.get("finish_reason"),
            "response_excerpt": short(d.get("response_text", ""), 1200),
        })

    examples_long.sort(key=lambda item: item[0], reverse=True)
    repeated = response_hashes.most_common(5)
    summary = {
        "updates_with_outputs": len(output_files),
        "representative_trajectories": len(rep_trajs),
        "total_unique_leaf_outputs": total,
        "reward_mean_unique_leaf_all_outputs": reward_sum / max(1, total),
        "reward_mean_nominal_slot_all_outputs": nominal_reward_sum / max(1, nominal_weight_sum),
        "empty_pred_fraction": empty_pred / max(1, total),
        "finish_reason_counts": dict(finish.most_common()),
        "top_pred_answers": pred_counter.most_common(20),
        "top_ground_truth_answers": answer_counter.most_common(20),
        "max_exact_response_repeat": repeated[0][1] if repeated else 0,
        "top_exact_response_repeats": repeated,
        "per_update": per_update,
        "correct_examples": [
            {
                "update_id": r.get("update_id"),
                "problem": r.get("problem"),
                "answer": r.get("answer"),
                "pred_answer": r.get("pred_answer"),
                "response_excerpt": short(r.get("response_text", "")),
            }
            for r in examples_correct
        ],
        "wrong_examples": [
            {
                "update_id": r.get("update_id"),
                "problem": r.get("problem"),
                "answer": r.get("answer"),
                "pred_answer": r.get("pred_answer"),
                "response_excerpt": short(r.get("response_text", "")),
            }
            for r in examples_wrong
        ],
        "longest_examples": [
            {
                "chars": n,
                "update_id": r.get("update_id"),
                "reward": r.get("reward"),
                "problem": r.get("problem"),
                "answer": r.get("answer"),
                "pred_answer": r.get("pred_answer"),
                "finish_reason": r.get("finish_reason"),
                "response_excerpt": short(r.get("response_text", ""), 1200),
            }
            for n, r in examples_long[:5]
        ],
        "representative_samples": rep_trajs[:12],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Big-Math Branch-GRPO Taste Summary",
        "",
        f"- updates with outputs: {summary['updates_with_outputs']}",
        f"- representative trajectories: {summary['representative_trajectories']}",
        f"- total unique leaf outputs: {summary['total_unique_leaf_outputs']}",
        f"- unique-leaf reward mean: {summary['reward_mean_unique_leaf_all_outputs']:.3f}",
        f"- nominal-slot reward mean: {summary['reward_mean_nominal_slot_all_outputs']:.3f}",
        f"- empty prediction fraction: {summary['empty_pred_fraction']:.4f}",
        f"- max exact response repeat count: {summary['max_exact_response_repeat']}",
        f"- finish reasons: {summary['finish_reason_counts']}",
        "",
        "## Correct Samples",
    ]
    for item in summary["correct_examples"][:4]:
        lines.extend([
            "",
            f"### u{item['update_id']} correct",
            f"Problem: {item['problem']}",
            f"Answer: {item['answer']} | Pred: {item['pred_answer']}",
            "",
            item["response_excerpt"],
        ])
    lines.append("")
    lines.append("## Wrong Samples")
    for item in summary["wrong_examples"][:4]:
        lines.extend([
            "",
            f"### u{item['update_id']} wrong",
            f"Problem: {item['problem']}",
            f"Answer: {item['answer']} | Pred: {item['pred_answer']}",
            "",
            item["response_excerpt"],
        ])
    lines.append("")
    lines.append("## Representative Trajectories")
    for item in summary["representative_samples"][:6]:
        lines.extend([
            "",
            f"### u{item['update_id']} reward={item['reward']}",
            f"Problem: {item['problem']}",
            f"Answer: {item['answer']} | Pred: {item['pred_answer']}",
            "",
            item["response_excerpt"],
        ])
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "updates_with_outputs",
        "representative_trajectories",
        "total_unique_leaf_outputs",
        "reward_mean_unique_leaf_all_outputs",
        "reward_mean_nominal_slot_all_outputs",
        "empty_pred_fraction",
        "finish_reason_counts",
        "max_exact_response_repeat",
    )}, indent=2))


if __name__ == "__main__":
    main()
