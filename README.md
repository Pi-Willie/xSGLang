# XSGLang

XSGLang is a Linux-only CUDA inference engine built for block-wise generation, branching continuations, activation hooks, and other workflows that are awkward or inefficient in a normal inference stack.

Standard inference engines are often fast, but they expose generation as a black box. You submit a prompt, ask for tokens, and get a completion back. That works for serving, but it becomes restrictive once you want to stop partway through generation, inspect the current state, branch from it, or run many related continuations without repeatedly rebuilding the same prefix.

At the other extreme, `transformers.generate()` gives more direct access, but in practice it is a poor fit for shared-prefix, multi-block research loops. If the goal is to branch, compare trajectories, intervene on activations, or keep a whole continuation tree resident, it is easy to end up with a Python control loop that throws away most of the model's real performance.

XSGLang is meant to sit in that gap. It keeps the model resident, exposes generation as bounded blocks, and makes continuation control part of the engine itself rather than something wrapped around it afterwards.

## Block-wise generation

The central idea in XSGLang is that generation does not have to be treated as one opaque stream.

Instead of only asking the model to generate until it finishes, you can run a bounded block such as 20, 30, or 40 tokens, inspect the result, then decide what to do next. You can continue the same state, fork it into children, or attach hooks before the next block.

That makes it possible to do things like:

* generate a short block
* inspect the local behavior of the model
* branch into several child continuations
* keep all of them warm for the next step
* reuse KV state efficiently instead of restarting each branch from scratch

That is the main use case for this repository.

## What XSGLang is for

XSGLang is built for inference workloads where control matters as much as raw throughput. That includes:

* branching generation and search
* shared-prefix continuation trees
* activation capture and intervention
* hook-based experiments
* repeated experiments over one prompt prefix
* data-generation workflows where many related continuations need to stay live at once

The point is not to bolt more Python on top of a server. The point is to make these workflows native to the runtime.

## What it is built on

XSGLang is a fork of [Mini-SGLang](https://github.com/sgl-project/mini-sglang), and the base runtime foundation comes from that project.

Mini-SGLang provides the compact low-level serving core this fork builds on, including:

* paged KV cache
* prefix reuse
* chunked prefill
* overlap scheduling
* tensor parallelism
* CUDA graph decode
* a codebase small enough to read without getting lost immediately

XSGLang keeps that base and extends the control surface on top of it.

## What is different here

This version is centered on the engine rather than on a larger pile of surrounding tooling.

The main additions in this fork are around controllable inference:

* bounded block execution
* continuation-oriented offline control
* branching from shared live state
* efficient KV-backed continuation reuse
* hook-aware execution for capture and intervention
* research-oriented workflows that need direct runtime control

The model stays loaded. The continuation stays live. You can advance it a block at a time, inspect it, fork it, and continue from there.

## Two demos

This repo only keeps two public demos in [`examples/`](./examples). They are meant to be easy to run from a terminal and easy to read afterward.

### `demo_branch_stress.py`

This is the clean-path throughput demo.

It does three simple things:

* decode a fixed-size block on every live branch
* fork every live branch into two clean children
* print total `tok/s` and `tok/s/branch` as the tree widens

There are no hooks in the hot path. The point is to show the thing this engine is mainly built for: keep one shared prefix resident, widen the tree, and keep throughput high instead of recomputing the same prefix over and over.

### `demo_noise_tree.py`

This is the branching-plus-hooks demo.

It also grows a binary tree, but at every split:

* one child stays clean
* one child gets another layer of activation noise

At the end it prints:

* throughput as the tree widened
* the cleanest full path
* the noisiest full path

The point is to show two things at once: branching still works while hooks are active, and the noisiest path actually drifts and degrades as interventions accumulate.

## Rough performance

The numbers below are from local runs on `Qwen/Qwen3-0.6B` on an RTX 4080 SUPER.

### Clean shared-prefix branching

This is where the engine is strongest.

In benchmark comparisons against rerunning Hugging Face `generate`, XSGLang landed around **2.7x to 3.1x faster** on the tested multi-block workloads:

* single stream blocks: `48.75 tok/s` vs HF `15.91 tok/s`
* clean branch workload: `357.62 tok/s` vs HF `117.27 tok/s`
* steering branch workload: `327.27 tok/s` vs HF `118.39 tok/s`
* wider stress branch workload: `674.27 tok/s` vs HF `246.54 tok/s`

Memory behavior was also good for a resident-KV design: XSGLang pays a large upfront reservation, but incremental allocation stayed very small while HF grew much more during generation.

The clearest public demo result is from `demo_branch_stress.py`:

* `8` levels, `40` tokens per block, widening to `128` live branches
* final level: `5120` tokens in `0.85s`
* best single-level throughput: **`5997.78 tok/s`**
* overall throughput: **`1648.07 tok/s`**

That is the pattern you want to see: as the tree widens, total emitted tokens grow rapidly while wall time grows much more slowly. Shared-prefix reuse is doing real work.

### Hooked / noisy branching

The hook path works, but it is much slower than the clean path.

In the steering benchmark, the steered branch case was:

* XSGLang steering workload: `327.27 tok/s`
* clean branch workload: `357.62 tok/s`

So that particular intervention path was only about **`1.09x`** slower than the clean branch benchmark.

But the heavier CPU-side noise-hook demo is much more expensive. In one `demo_noise_tree.py` run:

* `7` levels, `20` tokens per block, widening to `64` live branches
* final level throughput: **`182.34 tok/s`**
* overall throughput: **`180.99 tok/s`**

The clean path stayed coherent, while the noised path degraded progressively into nonsense. So the intervention path appears to be doing the right thing, but the hook-heavy path currently dominates runtime once you lean on it hard.

The short honest summary is:

**XSGLang performs very well on resident, shared-prefix, multi-block branching, and much worse once CPU-side activation hooks are heavily in the loop.**

## Repository layout

This trimmed repository is intentionally small. The important parts are:

Main runtime code:

* [`python/minisgl/engine`](./python/minisgl/engine)
* [`python/minisgl/scheduler`](./python/minisgl/scheduler)
* [`python/minisgl/models`](./python/minisgl/models)
* [`python/minisgl/kvcache`](./python/minisgl/kvcache)
* [`python/minisgl/attention`](./python/minisgl/attention)
* [`python/minisgl/hooks.py`](./python/minisgl/hooks.py)
* [`python/minisgl/research`](./python/minisgl/research)
* [`python/minisgl/llm`](./python/minisgl/llm)
* [`python/minisgl/core.py`](./python/minisgl/core.py)

Public demos:

* [`examples/demo_branch_stress.py`](./examples/demo_branch_stress.py)
* [`examples/demo_noise_tree.py`](./examples/demo_noise_tree.py)

Packaging:

* [`pyproject.toml`](./pyproject.toml)
* [`requirements.txt`](./requirements.txt)

## Installation

Requirements:

* Linux only
* Python 3.10+
* NVIDIA GPU with CUDA available to PyTorch

Suggested setup:

```bash
git clone https://github.com/Pi-Willie/XSGLang.git
cd XSGLang
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If your machine needs a specific CUDA build of `torch`, install that first.

The first model load can take around 30 seconds on a fresh machine because weights, kernels, and CUDA code may need to initialize or compile once.

The demos now prefetch Hugging Face models into the local cache before engine startup, and the default install includes `hf_transfer` so first-time downloads use the faster hub transfer path when available.

## Quick start

Run the clean branching demo:

```bash
python examples/demo_branch_stress.py --model Qwen/Qwen3-0.6B
```

Run the branching-plus-noise demo:

```bash
python examples/demo_noise_tree.py --model Qwen/Qwen3-0.6B
```

For deeper trees, both demos expose `--max-running-req` so you can raise the continuation-table size if your GPU memory allows it.
