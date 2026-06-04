"""Integrated Branch-Dr.GRPO training loop on a single H100.

Ties together:
  * xsglang (minisgl) LLM for rollout / fork / KV reuse and in-memory weight refresh,
  * an HF trainer model for exact autograd logprob training,
  * leaf-slot materialization + constant-denominator Dr.GRPO loss,
  * FP32-master AdamW (master/m/v pinned on CPU),
  * a logprob-parity health probe and weight-movement verification.

See plan.txt sections 15-17 for the loop, and LAB_NOTEBOOK.md for the documented
deviations (on-policy rho==1, CPU-resident optimizer state, prompt-major rollout).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import BranchGRPOConfig
from .records import (
    RolloutTree,
    compute_leave_one_out_sibling_advantages,
    compute_nominal_slot_q_values,
    materialize_leaf_slot_paths,
)
from .rollout import build_branch_rollout_tree
from .trainer import FP32MasterAdamW, branch_grpo_train_step
from .verifier import binary_tag_reward, extract_answer_tag, normalize_answer


def hf_to_native_state_dict(model: torch.nn.Module, hf_config: Any) -> dict[str, torch.Tensor]:
    """Convert an HF Qwen3 state dict to xsglang's fused (qkv_proj/gate_up_proj) layout."""
    from minisgl.models.weight import _merge_state_dict

    model_type = str(getattr(hf_config, "model_type", "qwen3"))
    hf_sd = {name: tensor.detach() for name, tensor in model.state_dict().items()}
    return _merge_state_dict(hf_sd, model_type=model_type, hf_config=hf_config)


@dataclass
class UpdateMetrics:
    update_id: int
    fields: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = {"update_id": self.update_id}
        out.update(self.fields)
        return out


class BranchGRPOLoop:
    def __init__(
        self,
        *,
        llm: Any,
        trainer_model: torch.nn.Module,
        hf_config: Any,
        config: BranchGRPOConfig,
        max_packed_tokens: int,
        device: torch.device | str = "cuda",
        on_policy_old_logprobs: bool = True,
    ) -> None:
        self.llm = llm
        self.model = trainer_model
        self.hf_config = hf_config
        self.config = config
        self.max_packed_tokens = max_packed_tokens
        self.device = torch.device(device)
        self.on_policy_old_logprobs = on_policy_old_logprobs
        self.pad_token_id = int(getattr(llm.tokenizer, "pad_token_id", 0) or 0)
        self.policy_version = 0

        self.model.eval()
        if hasattr(getattr(self.model, "config", None), "use_cache"):
            self.model.config.use_cache = False

        self.optimizer = FP32MasterAdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            master_device="cpu",
        )

    # ---- weight lifecycle -------------------------------------------------
    def refresh_xsglang_from_trainer(self) -> dict[str, Any]:
        native_sd = hf_to_native_state_dict(self.model, self.hf_config)
        info = self.llm.refresh_model_weights_from_state_dict(
            native_sd,
            source=f"trainer_bf16_v{self.policy_version}",
            preserve_adapter=False,
        )
        del native_sd
        gc.collect()
        torch.cuda.empty_cache()
        return info

    # ---- health probes ----------------------------------------------------
    @torch.no_grad()
    def xsglang_greedy_tokens(self, prompt: str, n_tokens: int) -> list[int]:
        from minisgl.core import OUTPUT_TOKENS, SamplingParams

        req = self.llm.open_continuation(
            prompt,
            SamplingParams(temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True, max_tokens=n_tokens + 8),
            requested_outputs=(OUTPUT_TOKENS,),
        )
        try:
            result = req.run_block(max_new_tokens=n_tokens, min_new_tokens=n_tokens,
                                   request_outputs=(OUTPUT_TOKENS,))
            return [int(v) for v in result.continuation_results[0].emitted_token_ids.tolist()]
        finally:
            self.llm.free_continuation(req)

    @torch.no_grad()
    def parity_probe(self, prompt: str, n_tokens: int = 16) -> dict[str, float]:
        """plan.txt section 16: compare xsglang selected-token logprobs to the trainer's."""
        from minisgl.core import OUTPUT_LOGPROBS, OUTPUT_TOKENS, SamplingParams

        req = self.llm.open_continuation(
            prompt,
            SamplingParams(temperature=0.0, top_k=1, top_p=1.0, ignore_eos=True, max_tokens=n_tokens + 8),
            requested_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS),
        )
        try:
            res = req.run_block(max_new_tokens=n_tokens, min_new_tokens=n_tokens,
                                request_outputs=(OUTPUT_TOKENS, OUTPUT_LOGPROBS))
            cont = res.continuation_results[0]
            prompt_ids = [int(v) for v in req.materialize_input_ids()[: req.prompt_len].tolist()]
            token_ids = [int(v) for v in cont.emitted_token_ids.tolist()]
            x_lp = [float(v) for v in cont.logprobs.tolist()]
        finally:
            self.llm.free_continuation(req)

        full = torch.tensor([prompt_ids + token_ids], dtype=torch.long, device=self.device)
        logits = self.model(full[:, :-1], use_cache=False).logits.float()[0]
        positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(token_ids), device=self.device)
        targets = full[0, len(prompt_ids):]
        t_lp = torch.log_softmax(logits[positions], dim=-1).gather(-1, targets.view(-1, 1)).view(-1)
        t_lp = [float(v) for v in t_lp.cpu().tolist()]
        diffs = [abs(a - b) for a, b in zip(x_lp, t_lp)]
        return {
            "mean_abs_diff": float(np.mean(diffs)) if diffs else 0.0,
            "max_abs_diff": float(np.max(diffs)) if diffs else 0.0,
            "n_tokens": float(len(diffs)),
        }

    # ---- one update -------------------------------------------------------
    def run_update(self, update_id: int, prompt_rows: list[dict[str, Any]],
                   *, capture_weight_delta: bool = False) -> UpdateMetrics:
        cfg = self.config
        m = UpdateMetrics(update_id=update_id)
        # Persistent-baseline live memory BEFORE any rollout/train transient. This is the
        # true leak detector: xsglang weights+KV + trainer weights should be flat across
        # updates; a KV/continuation/python leak shows up as a growing baseline. (Per-update
        # peak varies with microbatch packing and is NOT a leak signal.)
        mem_alloc_start_gb = torch.cuda.memory_allocated(self.device) / 1e9
        torch.cuda.reset_peak_memory_stats(self.device)

        # ---- B/C. rollout + reward (rewards attached inside rollout builder) ----
        t0 = time.perf_counter()
        trees: list[RolloutTree] = []
        rollout_gen_tokens = 0
        skipped_prompts = 0
        for offset, row in enumerate(prompt_rows):
            try:
                tree = build_branch_rollout_tree(
                    llm=self.llm, prompt_row=row,
                    prompt_id=update_id * 1000 + offset, config=cfg,
                )
            except Exception as exc:
                # Robustness: an occasional OpenR1 prompt exceeds prompt_max_tokens (the
                # answer-length filter does not bound prompt length), or a rollout hiccups.
                # Skip that prompt rather than killing the run. Denominator stays constant.
                skipped_prompts += 1
                print(f"[warn] u{update_id} skip prompt offset={offset}: {exc}", flush=True)
                continue
            trees.append(tree)
            rollout_gen_tokens += int(sum(int(e.tokens.shape[0]) for e in tree.edges.values()))
        rollout_s = time.perf_counter() - t0
        m.fields["skipped_prompts"] = float(skipped_prompts)
        if not trees:
            # Whole batch unusable; skip the optimizer step but keep xsglang weights current.
            print(f"[warn] u{update_id}: all {len(prompt_rows)} prompts skipped, no update", flush=True)
            m.fields["no_update"] = 1.0
            return m

        # ---- E. advantage backup ----
        all_rewards, unique_rewards, leaf_lengths = [], [], []
        zero_signal_prompts = 0
        for tree in trees:
            compute_nominal_slot_q_values(tree)
            compute_leave_one_out_sibling_advantages(tree)
            tree_leaf_rewards = [leaf.reward for leaf in tree.leaves.values()]
            unique_rewards.extend(tree_leaf_rewards)
            for leaf in tree.leaves.values():
                all_rewards.extend([leaf.reward] * int(leaf.nominal_slot_count))
                leaf_lengths.append(len(leaf.answer_text))
            if max(abs(e.advantage or 0.0) for e in tree.edges.values()) < 1e-8:
                zero_signal_prompts += 1

        # ---- F. materialize ----
        train_examples = []
        for tree in trees:
            train_examples.extend(materialize_leaf_slot_paths(tree))

        # ---- weight snapshot (optional, for health gate) ----
        pre_snapshot = None
        if capture_weight_delta:
            pre_snapshot = [p.detach().float().cpu().clone() for p in self.optimizer.model_params]

        # ---- G/H. one accumulated optimizer step ----
        t1 = time.perf_counter()
        stats = branch_grpo_train_step(
            model=self.model,
            optimizer=self.optimizer,
            train_examples=train_examples,
            denominator_tokens=cfg.denominator_tokens,
            max_packed_tokens=self.max_packed_tokens,
            clip_epsilon=cfg.ppo_clip,
            grad_clip=cfg.grad_clip,
            pad_token_id=self.pad_token_id,
            device=self.device,
            shuffle=True,
            seed=update_id,
            on_policy_old_logprobs=self.on_policy_old_logprobs,
        )
        train_s = time.perf_counter() - t1

        weight_delta = -1.0
        if pre_snapshot is not None:
            sq = 0.0
            for before, p in zip(pre_snapshot, self.optimizer.model_params):
                sq += float(((p.detach().float().cpu() - before) ** 2).sum())
            weight_delta = float(sq ** 0.5)
            del pre_snapshot

        # ---- I. refresh xsglang weights in-memory ----
        t2 = time.perf_counter()
        refresh_info = self.refresh_xsglang_from_trainer()
        refresh_s = time.perf_counter() - t2
        self.policy_version = update_id + 1

        n_unique = max(1, len(unique_rewards))
        n_slots = max(1, len(all_rewards))
        m.fields.update({
            "reward_mean_slot": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "reward_mean_unique_leaf": float(np.mean(unique_rewards)) if unique_rewards else 0.0,
            "accuracy_per_verifier_call": float(np.mean([1.0 if r > 0 else 0.0 for r in unique_rewards])) if unique_rewards else 0.0,
            "unique_leaf_count": float(len(unique_rewards)),
            "nominal_leaf_slots": float(len(all_rewards)),
            "zero_signal_prompt_fraction": zero_signal_prompts / max(1, len(trees)),
            "actual_generated_tree_tokens": float(rollout_gen_tokens),
            "denominator_tokens": float(cfg.denominator_tokens),
            "tree_token_ratio": rollout_gen_tokens / float(cfg.denominator_tokens),
            "mean_response_char_len": float(np.mean(leaf_lengths)) if leaf_lengths else 0.0,
            "loss": stats.loss,
            "loss_sum_before_div": stats.loss_sum_before_div,
            "grad_norm": stats.grad_norm,
            "clip_fraction": stats.clip_fraction,
            "approx_kl_old_current": stats.approx_kl_old_current,
            "mean_logratio": stats.mean_logratio,
            "max_abs_logratio": stats.max_abs_logratio,
            "nonzero_weighted_tokens": stats.nonzero_weighted_tokens,
            "microbatches": float(stats.microbatches),
            "optimizer_steps": float(stats.optimizer_steps),
            "weight_delta_l2": weight_delta,
            "sys_rollout_s": rollout_s,
            "sys_train_s": train_s,
            "sys_weight_refresh_s": refresh_s,
            "sys_rollout_tokens_per_s": rollout_gen_tokens / rollout_s if rollout_s > 0 else 0.0,
            "sys_peak_gpu_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
            "sys_mem_alloc_start_gb": mem_alloc_start_gb,
            "wall_ts": time.time(),  # for accuracy-per-wall-hour accounting
            "lr": cfg.lr,
        })
        # branch diagnostics by depth
        self._attach_branch_diagnostics(trees, m)
        return m

    def _attach_branch_diagnostics(self, trees: list[RolloutTree], m: UpdateMetrics) -> None:
        depth_mixed: dict[int, list[float]] = {}
        depth_absadv: dict[int, list[float]] = {}
        mixed_all, sib_disagree_all = [], []
        for tree in trees:
            for node in tree.nodes.values():
                child_edges = tree.child_edges(node.id)
                if len(child_edges) < 2:
                    continue
                qs = [tree.nodes[e.child_node].q_value or 0.0 for e in child_edges]
                mixed = 1.0 if (max(qs) - min(qs)) > 1e-8 else 0.0
                mixed_all.append(mixed)
                sib_disagree_all.append(1.0 if len({round(q, 6) for q in qs}) > 1 else 0.0)
                d = node.depth
                depth_mixed.setdefault(d, []).append(mixed)
                for e in child_edges:
                    depth_absadv.setdefault(d, []).append(abs(e.advantage or 0.0))
        m.fields["branch_mixed_node_rate_all"] = float(np.mean(mixed_all)) if mixed_all else 0.0
        m.fields["branch_sibling_disagreement_rate_all"] = float(np.mean(sib_disagree_all)) if sib_disagree_all else 0.0
        for d in sorted(depth_mixed):
            m.fields[f"branch_mixed_node_rate_depth_{d}"] = float(np.mean(depth_mixed[d]))
        for d in sorted(depth_absadv):
            m.fields[f"branch_mean_abs_adv_depth_{d}"] = float(np.mean(depth_absadv[d]))

    # ---- evaluation -------------------------------------------------------
    @torch.no_grad()
    def eval_greedy(self, eval_rows: list[dict[str, Any]], *, max_new_tokens: int,
                    chunk_size: int = 16) -> dict[str, float]:
        """Batched greedy held-out eval via xsglang continuous batching (chunked by KV budget)."""
        from minisgl.core import OUTPUT_TOKENS, BlockSpec, SamplingParams

        correct = valid = rep = 0
        total = len(eval_rows)
        lengths = []
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0,
                                ignore_eos=False, max_tokens=max_new_tokens)
        for start in range(0, total, chunk_size):
            chunk = eval_rows[start:start + chunk_size]
            conts = [self.llm.open_continuation(str(r["prompt"]), params,
                                                requested_outputs=(OUTPUT_TOKENS,)) for r in chunk]
            try:
                res = self.llm.run_block(BlockSpec(
                    continuation_ids=tuple(c.continuation_id for c in conts),
                    max_new_tokens=max_new_tokens, stop_on_eos=True,
                    request_outputs=(OUTPUT_TOKENS,)))
                by_id = {item.continuation_id: item for item in res.continuation_results}
                for r, c in zip(chunk, conts):
                    toks = [int(v) for v in by_id[c.continuation_id].emitted_token_ids.tolist()]
                    text = self.llm.tokenizer.decode(toks, skip_special_tokens=False)
                    lengths.append(len(toks))
                    ans = extract_answer_tag(text)
                    if ans is not None and normalize_answer(ans):
                        valid += 1
                    correct += int(binary_tag_reward(text, r.get("answer")) > 0.0)
                    if _has_repetition(toks):
                        rep += 1
            finally:
                for c in conts:
                    try:
                        self.llm.free_continuation(c)
                    except Exception:
                        pass
        d = max(1, total)
        return {
            "greedy_accuracy": correct / d,
            "invalid_format_rate": 1.0 - valid / d,
            "repetition_rate": rep / d,
            "mean_response_length": float(np.mean(lengths)) if lengths else 0.0,
            "eval_rows": float(total),
        }


def _has_repetition(tokens: list[int], run: int = 200) -> bool:
    if len(tokens) < run:
        return False
    count = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            count += 1
            if count >= run:
                return True
        else:
            count = 1
    return False
