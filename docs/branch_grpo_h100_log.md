# Branch-GRPO H100 Hardening Log

Branch: `branch-grpo-h100-hardening`

Key pushed commits:

- `a7632f2` - Batch sibling fork setup and cache resets.
- `e733613` - Add model refresh and branch metrics.
- `c7e95d4` - Handle tied LM head during weight refresh.
- `5ec6dfc` - Add H100 branch benchmark harness and log.
- `02f8a29` - Record Qwen3-4B branch scaling benchmarks.
- `985cba2` - Add in-memory weight refresh validation.
- `a22cfa8` - Stabilize tiny policy validation prompt.
- `a3b9640` - Allow partial weight refresh validation.
- `f384356` - Add branch GRPO loop validation harness.

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

### Qwen3-4B Baseline and Improved Branch Scaling

Baseline was run from detached worktree `~/xSGLang-baseline` at `29f4977`.
That historical tree needed the missing `python/minisgl/llm/hf_compat.py` import stub copied in so it could import `minisgl.llm`; the stub only routes to the native runtime and does not touch the hot path.

Common settings:

```bash
python benchmark/benchmark_branch_scaling.py \
  --model Qwen/Qwen3-4B \
  --block-size 32 \
  --single-trace-tokens 128 \
  --warmup-tokens 4 \
  --memory-ratio 0.75
```

Results, 128-branch run (`--levels 8 --max-running-req 256 --cuda-graph-max-bs 128`):

| Metric | Baseline | Improved | Ratio |
| --- | ---: | ---: | ---: |
| Single-trace TPS | 205.90 | 206.60 | 1.003x |
| Overall tree TPS | 5068.97 | 5060.95 | 0.998x |
| Best level TPS | 12070.06 | 11979.97 | 0.993x |
| Overall spawn children/s | 4961.63 | 7128.62 | 1.437x |
| 64 live branches -> 128 children spawn | 26.44 ms | 18.39 ms | 1.438x faster |

Results, 256-branch cap run (`--levels 9 --max-running-req 512 --cuda-graph-max-bs 256`):

| Metric | Baseline | Improved | Ratio |
| --- | ---: | ---: | ---: |
| Single-trace TPS | 206.60 | 206.23 | 0.998x |
| Overall tree TPS | 7609.11 | 7544.73 | 0.992x |
| Best level TPS | 15036.22 | 14908.01 | 0.991x |
| Overall spawn children/s | 5005.85 | 7078.83 | 1.414x |
| 128 live branches -> 256 children spawn | 53.61 ms | 36.19 ms | 1.482x faster |

Observed scaling/cap:

- Net decode throughput still rises through 256 live branches: `15036 tok/s` baseline and `14908 tok/s` improved.
- Per-branch throughput falls from about `207 tok/s/branch` at one branch to about `58 tok/s/branch` at 256 branches.
- At this shape, the cap is not table slots, KV memory, or CUDA graph memory. Graph capture to batch 256 fits with about `32.38 GiB` free after capture.
- The visible plateau pressure is decode saturation: larger batches keep improving net TPS but with sharply diminishing per-branch retention after 128 branches.
- The improvement intentionally targets the act of branching. Decode math is model-bound and remains effectively unchanged; spawn setup is now about `1.4x` faster overall.

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

This is only a smoke comparison on `Qwen3-0.6B`; the required Qwen3-4B baseline/improved benchmark is recorded above.

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

### Native Weight Refresh and Train Validation

Validation harness:

```bash
python benchmark/validate_weight_refresh.py \
  --model Qwen/Qwen3-0.6B \
  --json-output benchmark/out/qwen06b_weight_refresh_validation.json \
  --memory-ratio 0.25 \
  --max-running-req 32 \
  --tokens 6
```

Qwen3-0.6B full validation:

- Corrupt native state refresh changed greedy output from `[279, 1372, 315, 1251, 879, 614]` to `[0, 0, 0, 0, 0, 0]`.
- Restoring the original native state recovered `[279, 1372, 315, 1251, 879, 614]`.
- Corrupt refresh elapsed: `246.74 ms`; restore refresh elapsed: `226.99 ms`.
- Tiny LoRA load/unload: `1.48 ms` / `0.23 ms`.
- Tiny policy step target: token id `9834` for text `" yes"`.
- Target logprob improved from `-8.5903` to `0.0`; generated tokens changed from `[279, 1372, 315, 1251, 879, 614]` to `[9834, 9834, 9834, 9834, 9834, 9834]`.
- Reloading the trained checkpoint into the live engine elapsed: `276.67 ms`.

Qwen3-4B corrupt-refresh and LoRA validation:

```bash
python benchmark/validate_weight_refresh.py \
  --model Qwen/Qwen3-4B \
  --json-output benchmark/out/qwen4b_weight_refresh_corrupt_lora.json \
  --memory-ratio 0.25 \
  --max-running-req 32 \
  --tokens 4 \
  --skip-train-step
```

Result:

- Corrupt native state refresh changed greedy output from `[330, 785, 4226, 374]` to `[0, 0, 0, 0]`.
- Restoring the original native state recovered `[330, 785, 4226, 374]`.
- Corrupt refresh elapsed: `931.81 ms`; restore refresh elapsed: `942.35 ms`.
- Tiny LoRA load/unload on the 4B engine: `1.90 ms` / `0.27 ms`.

### Branch-GRPO Sample-Train-Refresh-Sample Validation

Validation harness:

```bash
python benchmark/validate_branch_grpo_loop.py \
  --model Qwen/Qwen3-0.6B \
  --json-output benchmark/out/qwen06b_branch_grpo_validation.json \
  --memory-ratio 0.25 \
  --max-running-req 32 \
  --tokens 6 \
  --train-lr 0.01
```

Result:

- The native engine sampled a two-sibling branch group from one prompt using `spawn_children` with forced first tokens.
- Target branch: token id `9834` (`" yes"`), reward `1.0`, normalized advantage `1.0`.
- Reject branch: token id `902` (`" no"`), reward `0.0`, normalized advantage `-1.0`.
- The tiny branch-GRPO loss improved target logprob from `-8.5903` to `0.0`.
- The same loss pushed reject logprob from `-7.7778` to `-107.5`.
- Full checkpoint refresh into the live engine elapsed: `289.50 ms`.
- Greedy sample after refresh changed from `[279, 1372, 315, 1251, 879, 614]` to `[9834, 9834, 9834, 9834, 9834, 9834]`.
- Validation checks all passed: nonzero branch reward, mixed advantages, target improvement, reject degradation, and changed post-refresh output.

## Remaining

- Close the H100 instance after the completion audit passes.
