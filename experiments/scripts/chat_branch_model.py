#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


DEFAULT_INSTRUCTION = (
    "Solve the math problem. Write your reasoning inside <think>...</think> and the final "
    "short answer inside <answer>...</answer>."
)


def format_prompt(question: str, *, raw: bool = False) -> str:
    if raw:
        return question
    return f"{DEFAULT_INSTRUCTION}\n\nProblem:\n{question.strip()}\n\nSolution:\n"


def generate_stream(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        timeout=120.0,
    )
    kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": max(temperature, 1e-5),
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }
    thread = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
    thread.start()
    for text in streamer:
        print(text, end="", flush=True)
    thread.join()
    print("\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive streaming chat for the trained Branch-GRPO model.")
    parser.add_argument("--model", required=True, help="Remote model path, e.g. experiments/runs/.../last_model")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--raw", action="store_true", help="Send input directly instead of wrapping in the math prompt.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model path does not exist: {model_path}")
    print(f"[load] {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).cuda()
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    print(
        "[ready] Type a math question and press enter. Commands: /exit, /raw, /prompt, "
        "/temp 0.7, /top_p 0.95, /max 1536",
        flush=True,
    )

    raw = bool(args.raw)
    temperature = float(args.temperature)
    top_p = float(args.top_p)
    max_new_tokens = int(args.max_new_tokens)
    while True:
        try:
            question = input("\nbranch-grpo> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]", flush=True)
            return
        if not question:
            continue
        if question in {"/exit", "/quit", ":q"}:
            print("[bye]", flush=True)
            return
        if question == "/raw":
            raw = True
            print("[mode] raw prompt", flush=True)
            continue
        if question == "/prompt":
            raw = False
            print("[mode] math prompt wrapper", flush=True)
            continue
        if question.startswith("/temp "):
            temperature = float(question.split(maxsplit=1)[1])
            print(f"[temperature] {temperature}", flush=True)
            continue
        if question.startswith("/top_p "):
            top_p = float(question.split(maxsplit=1)[1])
            print(f"[top_p] {top_p}", flush=True)
            continue
        if question.startswith("/max "):
            max_new_tokens = int(question.split(maxsplit=1)[1])
            print(f"[max_new_tokens] {max_new_tokens}", flush=True)
            continue
        prompt = format_prompt(question, raw=raw)
        try:
            generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:
            print(f"\n[error] {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
