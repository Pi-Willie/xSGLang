from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

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
        return int(self.old_logprobs.numel())

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
) -> Iterable[list[TrainExample]]:
    if max_packed_tokens <= 0:
        raise ValueError("max_packed_tokens must be positive")
    items = list(examples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

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
    response_batch_indices: list[int] = []
    response_positions: list[int] = []
    old_logprobs: list[float] = []
    advantages: list[float] = []
    response_mask: list[float] = []
    repeat_weights: list[float] = []

    packed_tokens = 0
    for batch_idx, example in enumerate(examples):
        ids = torch.as_tensor(example.input_ids, dtype=torch.long, device=device)
        seq_len = int(ids.numel())
        packed_tokens += seq_len
        input_ids[batch_idx, :seq_len] = ids
        attention_mask[batch_idx, :seq_len] = 1
        response_len = seq_len - int(example.response_start)
        if response_len < 0:
            raise ValueError("response_start is outside input_ids")
        response_batch_indices.extend([batch_idx] * response_len)
        response_positions.extend(range(int(example.response_start), seq_len))
        old_logprobs.extend(float(v) for v in example.old_logprobs.tolist())
        advantages.extend(float(v) for v in example.advantages.tolist())
        response_mask.extend(float(v) for v in example.response_mask.tolist())
        repeat_weights.extend([float(example.repeat_weight)] * response_len)

    return PackedTrainBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_batch_indices=torch.tensor(response_batch_indices, dtype=torch.long, device=device),
        response_positions=torch.tensor(response_positions, dtype=torch.long, device=device),
        old_logprobs=torch.tensor(old_logprobs, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
        response_mask=torch.tensor(response_mask, dtype=torch.float32, device=device),
        repeat_weights=torch.tensor(repeat_weights, dtype=torch.float32, device=device),
        example_count=len(examples),
        packed_tokens=packed_tokens,
    )


def trainer_selected_logprobs(model: torch.nn.Module, batch: PackedTrainBatch) -> torch.Tensor:
    if batch.response_tokens == 0:
        return batch.old_logprobs.new_empty((0,))
    if bool((batch.response_positions <= 0).any()):
        raise ValueError("response tokens must have at least one context token")
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    logits = outputs.logits
    previous_positions = batch.response_positions - 1
    # Gather only the response-token rows, then compute logprob via logsumexp instead of a
    # full log_softmax. log_softmax materialises an extra [n_resp, vocab] fp32 tensor (the
    # alloc that OOM'd on this single-H100 setup); logsumexp is a reduction, so we hold one
    # fp32 [n_resp, vocab] copy instead of two. Mathematically identical.
    selected_logits = logits[batch.response_batch_indices, previous_positions].float()
    targets = batch.input_ids[batch.response_batch_indices, batch.response_positions]
    target_logits = selected_logits.gather(dim=-1, index=targets.view(-1, 1)).view(-1)
    log_z = torch.logsumexp(selected_logits, dim=-1)
    return target_logits - log_z


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

    for examples in pack_train_examples(
        train_examples,
        max_packed_tokens=max_packed_tokens,
        shuffle=shuffle,
        seed=seed,
    ):
        batch = collate_train_examples(examples, device=device, pad_token_id=pad_token_id)
        current_logprobs = trainer_selected_logprobs(model, batch)
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

    grad_norm, optimizer_steps = _optimizer_step(optimizer, model, grad_clip=grad_clip)
    optimizer.zero_grad(set_to_none=True)
    mean_clip_fraction = clip_fraction_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_approx_kl = approx_kl_sum / metric_weight if metric_weight > 0.0 else 0.0
    mean_logratio = mean_logratio_sum / metric_weight if metric_weight > 0.0 else 0.0
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
        optimizer_steps=optimizer_steps,
    )
