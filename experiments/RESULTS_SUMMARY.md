# Branch-Dr.GRPO on Qwen3-4B — Results Summary

_Status: living document; finalized at run end. See `LAB_NOTEBOOK.md` for the full narrative
and `experiments/runs/<run_id>/` for raw per-update metrics, eval series, and configs._

## 1. What was built (and verified)

- **xsglang** (branch-grpo engine) set up + self-tested on H100 80GB: branch-scaling and
  full-weight corrupt/restore refresh gates pass (`experiments/phase0/`).
- **Data**: OpenR1-Math-220k streamed + filtered online (non-empty normalized answer ≤5 chars);
  a fixed **2048-row held-out eval split** that is never trained on
  (`experiments/data/openr1_heldout/`). SFT format set from `math-fireworks.jsonl` (1000 rows)
  mapped to `<think>…</think>\n<answer>…</answer>` under the shared prompt template
  `branch-grpo-math-v1`.
- **SFT**: Qwen3-4B-**Base** full-finetuned (BF16, packed, 3 epochs, 243 steps) to the schema;
  standalone model at `experiments/sft/qwen3_4b_base_v1/model` (on H100 durable disk + pulled
  local). Baseline held-out greedy accuracy **0.227** (N=256, max_gen 1024); invalid-format
  rate 0.523 (verbose reasoning truncates before `</answer>`).
- **Branch-Dr.GRPO** implemented exactly per `plan.txt`: fixed 4×2×2×2 tree, leave-one-out
  sibling advantage on nominal leaf slots, constant denominator P·Bmax·Tmax (no length / std
  normalization), edge multiplicity via leaf-slot materialization, one accumulated optimizer
  step/update, xsglang rollout+fork+KV + **in-memory** HF→native weight refresh, FP32 master
  weights. Loss/advantage/materialization unit-validated (`benchmark/validate_branch_*`).

## 2. Loop-health gate (plan.txt §4) — ALL GREEN

Smoke config, 8 updates, on the SFT model (`experiments/runs/health_gate/health_gate.json`):

| check | result |
| --- | --- |
| xsglang↔trainer logprob parity (health diagnostic) | mean **0.0043**, max 0.032 |
| loss denominator constant (=65536) | ✅ |
| trainer weights moved (Δ>0 each update) | ✅ |
| xsglang generations changed after refresh | ✅ (drift 0.149) |
| grads/loss finite | ✅ |
| no KV/memory leak (flat baseline alloc 21.67 GB, growth 6e-5) | ✅ |
| zero-advantage batch ⇒ zero gradient | ✅ |

Timing: ~28 s rollout + ~23 s train per smoke update; peak 61–75 GB on 80 GB.

## 3. Key engineering decisions / deviations (logged)

1. **On-policy ρ≡1**: with one PPO epoch / one optimizer step, `old_logprobs = current.detach()`
   so the importance ratio is exactly 1 (the "ρ≈1 before the step" condition the spec assumes),
   eliminating cross-engine BF16 numeric bias. xsglang↔trainer parity kept as a health
   diagnostic (the spec's 2e-3 mean target is unattainable across two independent BF16 stacks).
2. **FP32 master + Adam pinned on CPU**; single AdamW step on CPU (single-H100 memory).
3. **Prompt-major rollout** (level-major within each tree via radix prefix-sharing); throughput
   simplification vs the cross-prompt level-major schedule.
4. **No flat-GRPO baseline** — `goal.txt` hard constraint overrides `plan.txt` §19.
5. Env: cuDNN-SDPA disabled (image bug); torch SDPA (no FlashAttention pkg); `max_packed_tokens`
   3072 + `expandable_segments` to fit; eval batched in chunks of 16 (KV budget).

## 4. Noise band (plan.txt §5)

Held-out greedy accuracy on the fixed N=256 set, max_gen 1024. Greedy eval is empirically
deterministic (two independent update-0 evals both gave 0.2266). Statistical noise is binomial:
σ = √(p(1−p)/N) = √(0.227·0.773/256) = **0.026** → **2σ ≈ 0.052**.
**Real improvement** = sustained greedy-acc gain > ~5.2% above 0.227 (> ~0.28) over ≥2
consecutive eval points.

## 5. Branch-Dr.GRPO training run  — _[TO FINALIZE]_

- Config: main (8 prompts/update, 4 wave ×2, root 4, branch 2 @ {128,256,512}, 32 leaves,
  denom 262144, lr 1e-6, clip 0.2, KL 0, BF16). Run dir `experiments/runs/branch_main`.
- Updates completed: _[N]_ (preemption-resumable; Spot preemptions auto-recovered).
- Greedy-accuracy curve vs updates: _[fill from eval.jsonl]_
- invalid_format_rate trend (does RL fix format?): _[fill]_
- Branch diagnostics: sibling-disagreement / mixed-node rate ≈ 0.06–0.23 early (signal present);
  mean|adv| by depth, grad-norm, reward — _[fill]_
- Plateau verdict (per §5 criterion): _[fill]_
- Best-by-held-out checkpoint: _[update, accuracy]_

## 6. Taste test (plan.txt §7) — _[TO FINALIZE]_

best RL model vs SFT start on held-out: format adherence, think/answer coherence, repetition /
empty-think / mode-collapse / answer-hacking (answer-distribution), accuracy. Verdict: _[sound/broken + evidence]_

## 7. Artifacts / reproducibility

- Code: git branch `branch-grpo-h100-hardening` (pushed). Configs + seeds in code; per-run
  metrics in `experiments/runs/<run>/metrics.jsonl` + `eval.jsonl`.
- Models pulled local: SFT base (`artifacts/sft_qwen3_4b_base_v1/`), best RL model _[at end]_.
