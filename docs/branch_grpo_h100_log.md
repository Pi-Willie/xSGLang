# Branch-GRPO H100 Hardening Log

Branch: `branch-grpo-h100-hardening`

Current pushed commits:

- `a7632f2` - Batch sibling fork setup and cache resets.
- `e733613` - Add model refresh and branch metrics.
- `c7e95d4` - Handle tied LM head during weight refresh.

## Design Calls So Far

- Full-model refresh should be in-place for same-architecture checkpoints. This keeps model tensor addresses stable, avoids constructing a second engine, preserves CUDA graph validity for shape-identical weights, and is the right fast path for sample-train-sample loops.
- Refresh invalidates all live continuations and KV/prefix cache state. KV belongs to a specific weight version, so carrying it across a full-weight update is incorrect.
- LoRA remains an additive hot-swap path. Full-model refresh unloads any active adapter first, copies base weights, then optionally reloads the adapter.
- Branch throughput measurement should not request decoded text by default. Text is result formatting, not a decode dependency.
- Branch cost is measured as spawn-only wall time, with parent-free time tracked separately in the standalone harness.

## H100 Setup

- Active instance: `h100-spot`, zone `us-central1-a`, machine `a3-highgpu-1g`.
- Image family used: `common-cu129-ubuntu-2404-nvidia-580`.
- Remote repo: `~/xSGLang`.
- Remote branch: `branch-grpo-h100-hardening`.
- Remote venv: `~/xSGLang/.venv`.
- HF token file: `~/.hf_env`.

## Validation So Far

### Small Model Smoke Benchmark

Command:

```bash
python examples/demo_branch_stress.py \
  --model Qwen/Qwen3-0.6B \
  --levels 8 \
  --block-size 40 \
  --max-running-req 256 \
  --warmup-tokens 2 \
  --json-output benchmark/out/qwen06b_improved_l8_b40.json
```

Result:

- Overall decode throughput: `10124.17 tok/s`.
- Best level throughput: `18432.36 tok/s` at 128 live branches.
- Final level per-branch throughput: `144.00 tok/s/branch`.
- Spawned children: 254.
- Total spawn wall time: `0.0702s`.
- Overall spawn rate: `3617.89 children/s`.

Earlier old-path demo result on the same H100 and shape, before the benchmark/demo cleanup, was:

- Overall decode throughput: `4368.97 tok/s`.
- Best level throughput: `12643.53 tok/s`.
- Final level per-branch throughput: `98.78 tok/s/branch`.

This is only a smoke comparison on `Qwen3-0.6B`; the required Qwen3-4B baseline/improved benchmark is still pending.

### Full-Model Refresh Smoke

Command shape:

```bash
llm = LLM(model_path=local_qwen06b, cuda_graph_max_bs=0, max_running_req=16, memory_ratio=0.25)
sample_before = req.run_block(max_new_tokens=4, min_new_tokens=4)
info = llm.refresh_model_weights(local_qwen06b)
sample_after = req2.run_block(max_new_tokens=4, min_new_tokens=4)
```

Result:

- Same-checkpoint refresh elapsed: `314.33 ms` on Qwen3-0.6B.
- Released continuations during refresh: 1.
- Greedy tokens before and after same-checkpoint refresh matched: `[576, 11652, 1265, 387]`.

## Pending

- Run Qwen3-4B baseline from pre-change `main`/`29f4977` using the standalone harness.
- Run Qwen3-4B improved branch with the same harness.
- Document plateau/cap and why scaling stops.
- Add and validate corrupted in-memory full-weight refresh.
- Add and validate LoRA hot-swap timing.
- Add and validate a tiny real train/sample loop.
- Final cleanup/review notes and completion audit.
