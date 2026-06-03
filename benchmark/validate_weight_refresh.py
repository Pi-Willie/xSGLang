#!/usr/bin/env python3
"""Validate full-weight refresh, LoRA hot-swap, and a tiny sample-train-sample loop."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from minisgl.utils import ensure_local_model_path


PROMPT = "Answer with one short word. Continue:"
TARGET_TEXT = " yes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--max-running-req", type=int, default=32)
    parser.add_argument("--memory-ratio", type=float, default=0.25)
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--train-lr", type=float, default=0.01)
    parser.add_argument("--advantage", type=float, default=1.0)
    return parser.parse_args()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_tokens(llm: LLM, prompt: str, tokens: int) -> list[int]:
    req = llm.open_continuation(
        prompt,
        SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=tokens + 8,
        ),
    )
    result = req.run_block(max_new_tokens=tokens, min_new_tokens=tokens)
    out = result.continuation_results[0].emitted_token_ids.tolist()
    llm.free_continuation(req)
    return [int(v) for v in out]


def _clone_native_state(llm: LLM) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in llm.engine.model.state_dict().items()
    }


def _zero_float_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if tensor.is_floating_point():
            out[name] = torch.zeros_like(tensor)
        else:
            out[name] = tensor.clone()
    return out


def _validate_corrupt_refresh(llm: LLM, args: argparse.Namespace) -> dict[str, Any]:
    before = _sample_tokens(llm, PROMPT, args.tokens)
    base_state = _clone_native_state(llm)
    corrupt_state = _zero_float_state(base_state)
    corrupt_info = llm.refresh_model_weights_from_state_dict(
        corrupt_state,
        source="zeroed_native_state",
        preserve_adapter=False,
    )
    corrupt = _sample_tokens(llm, PROMPT, args.tokens)
    restore_info = llm.refresh_model_weights_from_state_dict(
        base_state,
        source="restored_native_state",
        preserve_adapter=False,
    )
    restored = _sample_tokens(llm, PROMPT, args.tokens)
    return {
        "before_tokens": before,
        "corrupt_tokens": corrupt,
        "restored_tokens": restored,
        "corrupt_changed_output": corrupt != before,
        "restore_recovered_output": restored == before,
        "corrupt_refresh": corrupt_info,
        "restore_refresh": restore_info,
    }


def _write_tiny_lora_adapter(llm: LLM, adapter_dir: Path, model_path: str) -> Path:
    import safetensors.torch

    cfg = llm.engine.config.model_config
    rank = 1
    hidden = cfg.hidden_size
    q_rows = cfg.num_qo_heads * cfg.head_dim
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    tensors = {
        f"{prefix}.lora_A.weight": torch.randn(rank, hidden, dtype=torch.float32) * 1e-4,
        f"{prefix}.lora_B.weight": torch.randn(q_rows, rank, dtype=torch.float32) * 1e-4,
    }
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": model_path,
                "r": rank,
                "lora_alpha": rank,
                "target_modules": ["q_proj"],
                "fan_in_fan_out": False,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    safetensors.torch.save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    return adapter_dir


def _validate_lora_hot_swap(llm: LLM, model_path: str, work_dir: Path) -> dict[str, Any]:
    adapter_dir = _write_tiny_lora_adapter(llm, work_dir / "tiny_lora", model_path)
    _sync_cuda()
    started = time.perf_counter()
    active = llm.engine.lora_manager.load(str(adapter_dir))
    _sync_cuda()
    load_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    llm.engine.lora_manager.unload()
    _sync_cuda()
    unload_ms = (time.perf_counter() - started) * 1000.0
    return {
        "adapter_dir": str(adapter_dir),
        "active_adapter": active,
        "load_ms": load_ms,
        "unload_ms": unload_ms,
    }


def _target_logprob(model, input_ids: torch.Tensor, target_id: int) -> float:
    with torch.no_grad():
        logits = model(input_ids).logits[:, -1, :].float()
        return float(torch.log_softmax(logits, dim=-1)[0, target_id].item())


def _tiny_policy_step(model_path: str, work_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    output_embeddings = model.get_output_embeddings()
    output_embeddings.weight.requires_grad_(True)

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()
    target_ids = tokenizer(TARGET_TEXT, add_special_tokens=False).input_ids
    if not target_ids:
        raise RuntimeError("Target text did not tokenize to any ids")
    target_id = int(target_ids[0])
    before_logprob = _target_logprob(model, input_ids, target_id)

    optimizer = torch.optim.SGD([output_embeddings.weight], lr=args.train_lr)
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids).logits[:, -1, :].float()
    logprob = torch.log_softmax(logits, dim=-1)[0, target_id]
    loss = -float(args.advantage) * logprob
    loss.backward()
    optimizer.step()
    after_logprob = _target_logprob(model, input_ids, target_id)

    trained_dir = work_dir / "trained_policy_step"
    trained_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(trained_dir, safe_serialization=True)
    tokenizer.save_pretrained(trained_dir)
    del model
    torch.cuda.empty_cache()
    return {
        "trained_dir": str(trained_dir),
        "target_text": TARGET_TEXT,
        "target_id": target_id,
        "before_logprob": before_logprob,
        "after_logprob": after_logprob,
        "improved_logprob": after_logprob > before_logprob,
        "loss": float(loss.detach().cpu().item()),
    }


def _validate_train_refresh(
    llm: LLM,
    model_path: str,
    work_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    before_tokens = _sample_tokens(llm, PROMPT, args.tokens)
    train = _tiny_policy_step(model_path, work_dir, args)
    refresh = llm.refresh_model_weights(train["trained_dir"], preserve_adapter=False)
    after_tokens = _sample_tokens(llm, PROMPT, args.tokens)
    return {
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "output_changed": after_tokens != before_tokens,
        "train": train,
        "refresh": refresh,
    }


def main() -> None:
    args = parse_args()
    model_path = ensure_local_model_path(args.model)
    llm = LLM(
        model_path=model_path,
        cuda_graph_max_bs=0,
        max_running_req=args.max_running_req,
        memory_ratio=args.memory_ratio,
    )
    payload: dict[str, Any] = {
        "model": args.model,
        "resolved_model_path": model_path,
    }
    with tempfile.TemporaryDirectory(prefix="xsglang-refresh-") as tmp:
        work_dir = Path(tmp)
        try:
            payload["corrupt_refresh"] = _validate_corrupt_refresh(llm, args)
            payload["lora_hot_swap"] = _validate_lora_hot_swap(llm, model_path, work_dir)
            payload["tiny_policy_step"] = _validate_train_refresh(llm, model_path, work_dir, args)
        finally:
            llm.shutdown()

    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote JSON: {output_path}", flush=True)


if __name__ == "__main__":
    main()
