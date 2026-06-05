from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .loss import BranchLossStats, branch_drgrpo_loss
from .records import TrainExample


@dataclass(frozen=True)
class PackedTrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_batch_indices: torch.Tensor
    response_positions: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    response_mask: torch.Tensor
    repeat_weights: torch.Tensor
    example_count: int
    packed_tokens: int

    @property
    def response_tokens(self) -> int:
        return int(self.response_positions.numel())

    @property
    def weighted_response_tokens(self) -> float:
        return float(self.repeat_weights.detach().sum().cpu().item())


@dataclass(frozen=True)
class BranchTrainStepStats:
    denominator_tokens: int
    examples: int
    microbatches: int
    response_tokens: int
    weighted_response_tokens: float
    nonzero_weighted_tokens: float
    loss_sum_before_div: float
    loss: float
    grad_norm: float
    clip_fraction: float
    approx_kl_old_current: float
    mean_logratio: float
    max_abs_logratio: float
    mean_selected_logprob: float
    mean_selected_prob: float
    mean_token_entropy: float
    mean_advantage: float
    mean_abs_advantage: float
    optimizer_steps: int


class FP32MasterAdamW:
    """AdamW wrapper that updates FP32 master weights then refreshes model params."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        master_device: torch.device | str = "cpu",
    ) -> None:
        # FP32 master weights + Adam moment buffers live on master_device (CPU by default).
        # On a single 80GB H100 we cannot hold xsglang weights+KV, the BF16 trainer, BF16
        # grads, AND 48GB of FP32 optimizer state at once. Keeping master/m/v pinned on CPU
        # and running the (single) AdamW step on CPU costs only a grad copy down + param copy
        # up per update, which is cheap relative to rollout. Matches plan.txt's memory schedule.
        self.master_device = torch.device(master_device)
        self.model_params = [param for param in parameters if param.requires_grad]
        self.master_params = [
            param.detach().float().to(self.master_device).clone().requires_grad_(True)
            for param in self.model_params
        ]
        self.optimizer = torch.optim.AdamW(
            self.master_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.step_count = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.model_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def copy_model_grads_to_master(self) -> None:
        for model_param, master_param in zip(self.model_params, self.master_params):
            if model_param.grad is None:
                master_param.grad = None
            else:
                master_param.grad = (
                    model_param.grad.detach().float().to(self.master_device)
                )

    def refresh_model_from_master(self) -> None:
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.master_params):
                model_param.copy_(master_param.to(dtype=model_param.dtype, device=model_param.device))

    def step(self, *, grad_clip: float | None = None) -> float:
        self.copy_model_grads_to_master()
        active_master_params = [param for param in self.master_params if param.grad is not None]
        if active_master_params and grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(active_master_params, grad_clip)
            grad_norm_value = float(grad_norm.detach().cpu().item())
        elif active_master_params:
            grad_norm_value = float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [param.grad.detach().float().norm() for param in active_master_params]
                    )
                )
                .detach()
                .cpu()
                .item()
            )
        else:
            grad_norm_value = 0.0
        self.optimizer.step()
        self.refresh_model_from_master()
        self.step_count += 1
        return grad_norm_value

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "master_params": [param.detach().cpu().clone() for param in self.master_params],
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        master_params = state_dict.get("master_params")
        if not isinstance(master_params, list) or len(master_params) != len(self.master_params):
            raise ValueError("master_params in state_dict do not match optimizer parameters")
        with torch.no_grad():
            for target, source in zip(self.master_params, master_params):
                if not isinstance(source, torch.Tensor):
                    raise TypeError("master_params entries must be tensors")
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        optimizer_state = state_dict.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ValueError("optimizer state is missing")
        self.optimizer.load_state_dict(optimizer_state)
        self.step_count = int(state_dict.get("step_count", 0))
        self.refresh_model_from_master()


def pack_train_examples(
    examples: Sequence[TrainExample],
    *,
    max_packed_tokens: int,
    shuffle: bool = False,
    seed: int | None = None,
    group_by_length: bool = True,
) -> Iterable[list[TrainExample]]:
    if max_packed_tokens <= 0:
        raise ValueError("max_packed_tokens must be positive")
    items = list(examples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)
    if group_by_length:
        # Keep the per-update stochastic order, but bucket by length inside the update so each
        # dense HF forward pads less. This does not change the objective; it only reduces
        # useless pad-token compute in long/short mixed leaf batches.
        items.sort(key=lambda example: int(example.input_ids.shape[0]), reverse=True)

    batch: list[TrainExample] = []
    packed_tokens = 0
    for example in items:
        example_tokens = int(example.input_ids.shape[0])
        if batch and packed_tokens + example_tokens > max_packed_tokens:
            yield batch
            batch = []
            packed_tokens = 0
        batch.append(example)
        packed_tokens += example_tokens
    if batch:
        yield batch


def collate_train_examples(
    examples: Sequence[TrainExample],
    *,
    device: torch.device | str,
    pad_token_id: int = 0,
    include_old_logprobs: bool = True,
) -> PackedTrainBatch:
    if not examples:
        raise ValueError("cannot collate an empty microbatch")
    device = torch.device(device)
    max_len = max(int(example.input_ids.shape[0]) for example in examples)
    input_ids = torch.full(
        (len(examples), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    response_lens = [int(example.input_ids.shape[0]) - int(example.response_start) for example in examples]
    total_response = sum(response_lens)
    response_batch_indices_np = np.empty(total_response, dtype=np.int64)
    response_positions_np = np.empty(total_response, dtype=np.int64)
    old_logprobs_parts = [] if include_old_logprobs else None
    advantages_parts = []
    response_mask_parts = []
    repeat_weights_np = np.empty(total_response, dtype=np.float32)

    packed_tokens = 0
    cursor = 0
    for batch_idx, example in enumerate(examples):
        ids = torch.as_tensor(example.input_ids, dtype=torch.long, device=device)
        seq_len = int(ids.numel())
        packed_tokens += seq_len
        input_ids[batch_idx, :seq_len] = ids
        attention_mask[batch_idx, :seq_len] = 1
        response_start = int(example.response_start)
        response_len = seq_len - response_start
        if response_len < 0:
            raise ValueError("response_start is outside input_ids")
        end = cursor + response_len
        response_batch_indices_np[cursor:end] = batch_idx
        response_positions_np[cursor:end] = np.arange(response_start, seq_len, dtype=np.int64)
        if old_logprobs_parts is not None:
            old_logprobs_parts.append(example.old_logprobs)
        advantages_parts.append(example.advantages)
        response_mask_parts.append(example.response_mask)
        repeat_weights_np[cursor:end] = float(example.repeat_weight)
        cursor = end

    if cursor != total_response:
        raise RuntimeError("response collation cursor mismatch")
    old_logprobs_np = (
        np.concatenate(old_logprobs_parts).astype(np.float32, copy=False)
        if old_logprobs_parts is not None and old_logprobs_parts
        else np.zeros((total_response,), dtype=np.float32)
    )
    advantages_np = np.concatenate(advantages_parts).astype(np.float32, copy=False)
    response_mask_np = np.concatenate(response_mask_parts).astype(np.float32, copy=False)

    return PackedTrainBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_batch_indices=torch.as_tensor(response_batch_indices_np, dtype=torch.long, device=device),
        response_positions=torch.as_tensor(response_positions_np, dtype=torch.long, device=device),
        old_logprobs=torch.as_tensor(old_logprobs_np, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages_np, dtype=torch.float32, device=device),
        response_mask=torch.as_tensor(response_mask_np, dtype=torch.float32, device=device),
        repeat_weights=torch.as_tensor(repeat_weights_np, dtype=torch.float32, device=device),
        example_count=len(examples),
        packed_tokens=packed_tokens,
    )


@dataclass(frozen=True)
class SelectedTokenStats:
    logprobs: torch.Tensor
    sampled_entropy: torch.Tensor | None
    entropy_indices: torch.Tensor | None


def trainer_selected_logprobs(
    model: torch.nn.Module,
    batch: PackedTrainBatch,
) -> torch.Tensor:
    return trainer_selected_logprobs_and_entropy(
        model,
        batch,
        max_entropy_tokens=0,
    ).logprobs


def trainer_selected_logprobs_and_entropy(
    model: torch.nn.Module,
    batch: PackedTrainBatch,
    *,
    max_entropy_tokens: int = 1024,
) -> SelectedTokenStats:
    if batch.response_tokens == 0:
        return SelectedTokenStats(
            logprobs=batch.old_logprobs.new_empty((0,)),
            sampled_entropy=None,
            entropy_indices=None,
        )
    if bool((batch.response_positions <= 0).any()):
        raise ValueError("response tokens must have at least one context token")
    previous_positions = batch.response_positions - 1
    logits_start = 0
    forward_kwargs = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "use_cache": False,
    }
    if _supports_logits_to_keep(model):
        min_previous = int(previous_positions.min().detach().cpu().item())
        logits_to_keep = int(batch.input_ids.shape[1]) - min_previous
        if 0 < logits_to_keep < int(batch.input_ids.shape[1]):
            forward_kwargs["logits_to_keep"] = logits_to_keep
            logits_start = int(batch.input_ids.shape[1]) - logits_to_keep
    outputs = model(**forward_kwargs)
    logits = outputs.logits
    if int(logits.shape[1]) == int(batch.input_ids.shape[1]):
        logits_start = 0
    # Gather only the response-token rows, then compute logprob via logsumexp instead of a
    # full log_softmax. log_softmax materialises an extra [n_resp, vocab] fp32 tensor (the
    # alloc that OOM'd on this single-H100 setup); logsumexp is a reduction, so we hold one
    # fp32 [n_resp, vocab] copy instead of two. Mathematically identical.
    local_previous_positions = previous_positions - logits_start
    selected_logits = logits[batch.response_batch_indices, local_previous_positions].float()
    targets = batch.input_ids[batch.response_batch_indices, batch.response_positions]
    target_logits = selected_logits.gather(dim=-1, index=targets.view(-1, 1)).view(-1)
    log_z = torch.logsumexp(selected_logits, dim=-1)
    current_logprobs = target_logits - log_z

    sampled_entropy = None
    entropy_indices = None
    if max_entropy_tokens > 0 and int(selected_logits.shape[0]) > 0:
        n_rows = int(selected_logits.shape[0])
        if n_rows > max_entropy_tokens:
            entropy_indices = torch.linspace(
                0,
                n_rows - 1,
                steps=int(max_entropy_tokens),
                device=selected_logits.device,
            ).long()
            entropy_logits = selected_logits.index_select(0, entropy_indices)
            entropy_log_z = log_z.index_select(0, entropy_indices)
        else:
            entropy_indices = torch.arange(n_rows, device=selected_logits.device)
            entropy_logits = selected_logits
            entropy_log_z = log_z
        # Health-only entropy: sample rows to avoid materialising a second full
        # [response_tokens, vocab] tensor on long packed batches.
        probs = torch.softmax(entropy_logits, dim=-1)
        sampled_entropy = entropy_log_z - (probs * entropy_logits).sum(dim=-1)
    return SelectedTokenStats(
        logprobs=current_logprobs,
        sampled_entropy=sampled_entropy,
        entropy_indices=entropy_indices,
    )


def _supports_logits_to_keep(model: torch.nn.Module) -> bool:
    cached = getattr(model, "_branch_grpo_supports_logits_to_keep", None)
    if cached is not None:
        return bool(cached)
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        supported = False
    else:
        supported = "logits_to_keep" in signature.parameters
    setattr(model, "_branch_grpo_supports_logits_to_keep", supported)
    return supported


def _optimizer_step(
    optimizer: FP32MasterAdamW | torch.optim.Optimizer,
    model: torch.nn.Module,
    *,
    grad_clip: float,
) -> tuple[float, int]:
    if isinstance(optimizer, FP32MasterAdamW):
        before = optimizer.step_count
        grad_norm = optimizer.step(grad_clip=grad_clip)
        return grad_norm, optimizer.step_count - before
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(grad_norm_tensor.detach().cpu().item()), 1


def branch_grpo_train_step(
    *,
    model: torch.nn.Module,
    optimizer: FP32MasterAdamW | torch.optim.Optimizer,
    train_examples: Sequence[TrainExample],
    denominator_tokens: int,
    max_packed_tokens: int,
    clip_epsilon: float,
    grad_clip: float,
    pad_token_id: int = 0,
    device: torch.device | str | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    on_policy_old_logprobs: bool = True,
    entropy_sample_tokens: int = 1024,
) -> BranchTrainStepStats:
    if denominator_tokens <= 0:
        raise ValueError("denominator_tokens must be positive")
    if not train_examples:
        raise ValueError("train_examples must not be empty")
    if device is None:
        first_param = next(model.parameters())
        device = first_param.device
    device = torch.device(device)
    model.eval()
    if hasattr(getattr(model, "config", None), "use_cache"):
        model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)

    examples_seen = 0
    microbatches = 0
    response_tokens = 0
    weighted_response_tokens = 0.0
    nonzero_weighted_tokens = 0.0
    loss_sum_before_div = 0.0
    clip_fraction_sum = 0.0
    approx_kl_sum = 0.0
    mean_logratio_sum = 0.0
    max_abs_logratio = 0.0
    metric_weight = 0.0
    selected_logprob_sum = 0.0
    selected_prob_sum = 0.0
    advantage_sum = 0.0
    abs_advantage_sum = 0.0
    entropy_sum = 0.0
    entropy_weight = 0.0

    for examples in pack_train_examples(
        train_examples,
        max_packed_tokens=max_packed_tokens,
        shuffle=shuffle,
        seed=seed,
    ):
        batch = collate_train_examples(
            examples,
            device=device,
            pad_token_id=pad_token_id,
            include_old_logprobs=not on_policy_old_logprobs,
        )
        token_stats = trainer_selected_logprobs_and_entropy(
            model,
            batch,
            max_entropy_tokens=entropy_sample_tokens,
        )
        current_logprobs = token_stats.logprobs
        # On-policy old-logprobs: with exactly one PPO epoch / one optimizer step, the
        # behaviour policy IS the current trainer policy. Setting old = current.detach()
        # makes the importance ratio rho == 1 exactly (the "rho ~= 1 before the step"
        # condition plan.txt assumes), which removes cross-engine BF16 numeric bias that
        # would otherwise enter via xsglang-stored old logprobs. The PPO clip path is kept
        # intact for future multi-epoch experiments. See LAB_NOTEBOOK deviation note.
        old_logprobs = current_logprobs.detach() if on_policy_old_logprobs else batch.old_logprobs
        loss, stats = branch_drgrpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            advantages=batch.advantages,
            response_mask=batch.response_mask,
            denominator_tokens=denominator_tokens,
            clip_epsilon=clip_epsilon,
            repeat_weight=batch.repeat_weights,
        )
        loss.backward()

        examples_seen += batch.example_count
        microbatches += 1
        response_tokens += batch.response_tokens
        weighted_response_tokens += batch.weighted_response_tokens
        nonzero_weighted_tokens += stats.nonzero_weighted_tokens
        loss_sum_before_div += stats.loss_sum_before_div
        max_abs_logratio = max(max_abs_logratio, stats.max_abs_logratio)
        weight = max(stats.nonzero_weighted_tokens, 0.0)
        if weight > 0.0:
            metric_weight += weight
            clip_fraction_sum += stats.clip_fraction * weight
            approx_kl_sum += stats.approx_kl_old_current * weight
            mean_logratio_sum += stats.mean_logratio * weight
            active_weight = batch.response_mask * batch.repeat_weights
            selected_logprob_sum += float((current_logprobs.detach() * active_weight).sum().cpu().item())
            selected_prob_sum += float((current_logprobs.detach().exp() * active_weight).sum().cpu().item())
            advantage_sum += float((batch.advantages.detach() * active_weight).sum().cpu().item())
            abs_advantage_sum += float((batch.advantages.detach().abs() * active_weight).sum().cpu().item())
            if token_stats.sampled_entropy is not None and token_stats.entropy_indices is not None:
                ew = active_weight.index_select(0, token_stats.entropy_indices)
                ew_sum = float(ew.sum().detach().cpu().item())
                if ew_sum > 0.0:
                    entropy_sum += float((token_stats.sampled_entropy.detach() * ew).sum().cpu().item())
                    entropy_weight += ew_sum

    grad_norm, optimizer_steps = _optimizer_step(optimizer, model, grad_clip=grad_clip)
    optimizer.zero_grad(set_to_none=True)
    mean_clip_fraction = clip_fraction_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_approx_kl = approx_kl_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_logratio = mean_logratio_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_selected_logprob = selected_logprob_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_selected_prob = selected_prob_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_advantage = advantage_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_abs_advantage = abs_advantage_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_token_entropy = entropy_sum / entropy_weight if entropy_weight > 0.0 else 0.0
    return BranchTrainStepStats(
        denominator_tokens=int(denominator_tokens),
        examples=examples_seen,
        microbatches=microbatches,
        response_tokens=response_tokens,
        weighted_response_tokens=weighted_response_tokens,
        nonzero_weighted_tokens=nonzero_weighted_tokens,
        loss_sum_before_div=loss_sum_before_div,
        loss=loss_sum_before_div / float(denominator_tokens),
        grad_norm=grad_norm,
        clip_fraction=mean_clip_fraction,
        approx_kl_old_current=mean_approx_kl,
        mean_logratio=mean_logratio,
        max_abs_logratio=max_abs_logratio,
        mean_selected_logprob=mean_selected_logprob,
        mean_selected_prob=mean_selected_prob,
        mean_token_entropy=mean_token_entropy,
        mean_advantage=mean_advantage,
        mean_abs_advantage=mean_abs_advantage,
        optimizer_steps=optimizer_steps,
    )
