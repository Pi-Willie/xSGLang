#!/usr/bin/env python3
"""SFT Qwen3-4B Base into the Branch-GRPO math answer schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# This H100 image has a cudart/nvrtc version mismatch that breaks cuDNN's SDPA JIT
# backend ("No valid engine configs ... RUNTIME_PREREQUISITE_MISSING"). Disable the
# cuDNN SDPA backend so PyTorch uses the prebuilt flash / mem-efficient kernels.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from minisgl.branch_grpo.verifier import binary_tag_reward, extract_answer_tag, normalize_answer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _stable_float(*parts: object, seed: int) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest(), "big") / float(1 << 64)


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    eval_rows = []
    for row in rows:
        key = row.get("id") or row.get("problem") or row.get("prompt")
        if _stable_float(key, seed=seed) < eval_fraction:
            eval_rows.append(row)
        else:
            train.append(row)
    if not eval_rows and train:
        eval_rows.append(train.pop())
    return train, eval_rows


@dataclass
class SFTFeature:
    input_ids: list[int]
    labels: list[int]


def encode_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    prompt_max_tokens: int,
) -> SFTFeature | None:
    prompt_ids = tokenizer(
        row["prompt"],
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_max_tokens,
    ).input_ids
    remaining = max_length - len(prompt_ids)
    if remaining <= 1:
        return None
    completion = row["completion"]
    if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
        completion = completion + tokenizer.eos_token
    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
        truncation=True,
        max_length=remaining,
    ).input_ids
    if not completion_ids:
        return None
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return SFTFeature(input_ids=input_ids, labels=labels)


def pack_features(features: list[SFTFeature], max_length: int) -> list[SFTFeature]:
    packed: list[SFTFeature] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    for feature in features:
        if cur_ids and len(cur_ids) + len(feature.input_ids) > max_length:
            packed.append(SFTFeature(input_ids=cur_ids, labels=cur_labels))
            cur_ids = []
            cur_labels = []
        if len(feature.input_ids) > max_length:
            packed.append(
                SFTFeature(
                    input_ids=feature.input_ids[:max_length],
                    labels=feature.labels[:max_length],
                )
            )
            continue
        cur_ids.extend(feature.input_ids)
        cur_labels.extend(feature.labels)
    if cur_ids:
        packed.append(SFTFeature(input_ids=cur_ids, labels=cur_labels))
    return packed


class SFTDataset(Dataset):
    def __init__(self, features: list[SFTFeature]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> SFTFeature:
        return self.features[idx]


def collate(features: list[SFTFeature], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(feature.input_ids) for feature in features)
    input_ids = []
    labels = []
    attention_mask = []
    for feature in features:
        pad = max_len - len(feature.input_ids)
        input_ids.append(feature.input_ids + [pad_id] * pad)
        labels.append(feature.labels + [-100] * pad)
        attention_mask.append([1] * len(feature.input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def build_features(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    prompt_max_tokens: int,
    pack: bool,
) -> list[SFTFeature]:
    features = [
        feature
        for row in rows
        if (
            feature := encode_row(
                row,
                tokenizer,
                max_length=max_length,
                prompt_max_tokens=prompt_max_tokens,
            )
        )
        is not None
    ]
    if pack:
        features = pack_features(features, max_length=max_length)
    return features


@torch.no_grad()
def evaluate_generation(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    limit: int,
) -> dict[str, float]:
    model.eval()
    rows = rows[:limit] if limit else rows
    valid = 0
    correct = 0
    total = 0
    total_len = 0
    for row in tqdm(rows, desc="eval", leave=False):
        encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output[0, encoded.input_ids.shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        answer = extract_answer_tag(text)
        total += 1
        total_len += int(generated.numel())
        if answer is not None and normalize_answer(answer):
            valid += 1
        correct += int(binary_tag_reward(text, row.get("answer")) > 0.0)
    denom = max(1, total)
    return {
        "eval_rows": float(total),
        "format_valid_rate": valid / denom,
        "answer_accuracy": correct / denom,
        "mean_generated_tokens": total_len / denom,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(Path(args.train_jsonl))
    train_rows, eval_rows = split_rows(rows, eval_fraction=args.eval_fraction, seed=args.seed)
    # Preemption-safe resume: warm-restart from <out>/ckpt (weights + step), fresh optimizer.
    resume_dir = output_dir / "ckpt"
    start_model = args.model
    resumed_step = 0
    if args.resume and (resume_dir / "config.json").exists():
        start_model = str(resume_dir)
        sp = resume_dir / "ckpt_step.json"
        if sp.exists():
            resumed_step = int(json.loads(sp.read_text()).get("global_step", 0))
        print(f"[sft-resume] from {start_model} at step {resumed_step}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(start_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_features = build_features(
        train_rows,
        tokenizer,
        max_length=args.max_length,
        prompt_max_tokens=args.prompt_max_tokens,
        pack=not args.no_pack,
    )
    if not train_features:
        raise RuntimeError("No SFT train features were built")

    model = AutoModelForCausalLM.from_pretrained(
        start_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.train()
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    loader = DataLoader(
        SFTDataset(train_features),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    metrics_path = output_dir / "sft_metrics.jsonl"
    global_step = resumed_step
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with metrics_path.open("a" if resumed_step else "w", encoding="utf-8") as metrics_file:
        for epoch in range(args.epochs):
            for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}")):
                batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss / args.grad_accum_steps
                loss.backward()
                if (step + 1) % args.grad_accum_steps != 0 and step + 1 < len(loader):
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float((loss.detach() * args.grad_accum_steps).cpu().item()),
                    "grad_norm": float(grad_norm.detach().cpu().item())
                    if torch.is_tensor(grad_norm)
                    else float(grad_norm),
                    "lr": args.lr,
                    "elapsed_s": time.perf_counter() - started,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                if args.save_every and global_step % args.save_every == 0:
                    resume_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(resume_dir, safe_serialization=True)
                    tokenizer.save_pretrained(resume_dir)
                    (resume_dir / "ckpt_step.json").write_text(json.dumps({"global_step": global_step}))
                    print(f"[sft-ckpt] step {global_step}", flush=True)
                if args.max_steps and global_step >= args.max_steps:
                    break
            if args.max_steps and global_step >= args.max_steps:
                break

    eval_metrics = evaluate_generation(
        model,
        tokenizer,
        eval_rows,
        max_new_tokens=args.eval_max_new_tokens,
        limit=args.eval_limit,
    )
    model.save_pretrained(output_dir / "model", safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "model")
    summary = {
        "model": args.model,
        "train_jsonl": args.train_jsonl,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_features": len(train_features),
        "packed": not args.no_pack,
        "max_length": args.max_length,
        "prompt_max_tokens": args.prompt_max_tokens,
        "global_steps": global_step,
        "eval": eval_metrics,
    }
    (output_dir / "sft_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-jsonl", default="experiments/data/sft_math_fireworks.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--prompt-max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--no-pack", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="warm-restart from <output-dir>/ckpt")
    parser.add_argument("--save-every", type=int, default=200, help="checkpoint <out>/ckpt every N optim steps")
    args = parser.parse_args()
    if not (0.0 < args.eval_fraction < 0.5):
        raise ValueError("--eval-fraction must be between 0 and 0.5")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
