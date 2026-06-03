# Branch-GRPO Qwen3-4B Research Notebook

## 2026-06-03 - Session restart and scope audit

Authoritative files read:

- `goal.txt`
- `plan.txt`

Scope correction:

- The earlier H100 engine-hardening work is Phase 0 groundwork only.
- The completion contract requires a full Branch-GRPO research run: data prep, SFT, exact
  Branch-Dr.GRPO implementation, health gates, noise characterization, training to plateau,
  taste test, local artifact mirrors, and H100 shutdown.

Important spec conflict:

- `goal.txt` hard constraint says Branch-GRPO only and explicitly forbids flat/normal GRPO.
- `plan.txt` section 19 requests a Flat Dr.GRPO baseline.
- Decision: obey the hard constraint in `goal.txt`. No flat GRPO implementation or run unless
  the goal file changes.

Current evidence:

- Repo branch: `branch-grpo-h100-hardening`.
- Engine baseline commit required by `goal.txt`: `36ec59b`.
- Local repo was clean before new research scaffolding.
- Local SFT source exists at `/Users/wilhelmhedenskog/qagent/math/math-fireworks.jsonl`.
- Local SFT source row count: 1000.
- First inspected rows have `messages` with `user` problem and `assistant` completion that
  already contains `<think>...</think>` and `<answer>...</answer>`.
- No `experiments/` directory or `LAB_NOTEBOOK.md` existed before this entry.

H100 status:

- `h100-box`: `TERMINATED`.
- `h100-spot`: `TERMINATED`.
- Starting `h100-spot` failed with `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` for
  `a3-highgpu-1g` with one H100 and two local SSDs in `us-central1-a`.
- This is not treated as a stop condition. Local data/code scaffolding continues while H100
  capacity is retried or relocated.

Work added in this step:

- Copied authoritative `goal.txt` and `plan.txt` into the repo root.
- Added Branch-GRPO config primitives for the smoke and main fixed-tree settings.
- Added answer extraction/normalization and binary tag reward helpers.
- Added rollout record dataclasses, nominal-slot Q backup, sibling leave-one-out advantages,
  and exact leaf-slot materialization.
- Added constant-denominator Branch-Dr.GRPO PPO loss.
- Added data prep utilities for:
  - local `math-fireworks.jsonl` SFT conversion to exact prompt/completion schema,
  - streamed OpenR1 filtering into RL train plus fixed held-out eval split.
- Added a model-free core validator for denominators, advantage backup, zero-adv masking, and
  loss normalization.

Validation run locally:

```bash
PYTHONPATH=python python3 -m py_compile python/minisgl/branch_grpo/*.py \
  benchmark/validate_branch_grpo_core.py
PYTHONPATH=python python3 benchmark/validate_branch_grpo_core.py
```

Result:

- Smoke denominator: `65536`.
- Main denominator: `262144`.
- Synthetic mixed tree materialized 8 exact leaf-slot examples.
- Nonzero weighted tokens: `8.0`.
- All-equal reward tree had zero advantage mask sum: `0.0`.

Data prep run locally:

```bash
PYTHONPATH=python python3 -m minisgl.branch_grpo.data sft \
  --source /Users/wilhelmhedenskog/qagent/math/math-fireworks.jsonl \
  --output experiments/data/sft_math_fireworks.jsonl \
  --summary experiments/data/sft_math_fireworks_summary.json
```

Result:

- Source rows: `1000`.
- Written SFT rows: `1000`.
- Skipped rows: `0`.
- Prompt template version: `branch-grpo-math-v1`.
- Completion schema now uses exactly `<think>...</think>\n<answer>...</answer>`.

OpenR1 streaming smoke:

```bash
PYTHONPATH=python python3 -m minisgl.branch_grpo.data openr1 \
  --output-dir experiments/data/openr1_smoke \
  --eval-size 20 \
  --max-rows 200
```

Result:

- Dataset streamed: `open-r1/OpenR1-Math-220k`, split `train`.
- Scanned rows: `200`.
- Filtered rows with non-empty normalized answer length <= 5: `119`.
- Smoke train rows: `99`.
- Smoke held-out eval rows: `20`.
- Inspected schema includes `problem`, `solution`, `answer`, `messages`, and correctness metadata.
- The Python process hung during shutdown after writing outputs; it was terminated manually.
  Outputs were present and inspected.

Next gates:

1. Build the full OpenR1 filtered train/eval split, preferably on the H100 env once capacity
   is available.
2. Retry H100 capacity and/or find a viable H100 zone.
3. Once H100 is available, verify environment, run xsglang self-tests, transfer data, and pin
   trainer dependencies before SFT or RL.
