from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BranchLossStats:
    denominator_tokens: int
    loss_sum_before_div: float
    nonzero_weighted_tokens: float
    clip_fraction: float
    approx_kl_old_current: float
    mean_logratio: float
    max_abs_logratio: float


def branch_drgrpo_loss(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    denominator_tokens: int,
    clip_epsilon: float = 0.2,
    repeat_weight: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, BranchLossStats]:
    if denominator_tokens <= 0:
        raise ValueError("denominator_tokens must be positive")
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current and old logprobs must have the same shape")
    if advantages.shape != current_logprobs.shape:
        raise ValueError("advantages must match logprobs")
    if response_mask.shape != current_logprobs.shape:
        raise ValueError("response_mask must match logprobs")

    logratio = current_logprobs.float() - old_logprobs.float()
    ratio = torch.exp(logratio)
    unclipped = ratio * advantages.float()
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.float()
    objective = torch.minimum(unclipped, clipped)
    weighted_mask = response_mask.float()
    if repeat_weight is not None:
        weighted_mask = weighted_mask * torch.as_tensor(
            repeat_weight,
            dtype=weighted_mask.dtype,
            device=weighted_mask.device,
        )
    token_loss = -objective * weighted_mask
    loss_sum = token_loss.sum()
    loss = loss_sum / float(denominator_tokens)

    with torch.no_grad():
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        active = weighted_mask > 0
        clipped_active = (ratio != clipped_ratio) & active
        active_count = active.float().sum().clamp_min(1.0)
        stats = BranchLossStats(
            denominator_tokens=int(denominator_tokens),
            loss_sum_before_div=float(loss_sum.detach().cpu().item()),
            nonzero_weighted_tokens=float(weighted_mask.detach().sum().cpu().item()),
            clip_fraction=float(clipped_active.float().sum().div(active_count).cpu().item()),
            approx_kl_old_current=float(((-logratio).exp() - 1.0 + logratio)[active].mean().cpu().item())
            if bool(active.any())
            else 0.0,
            mean_logratio=float(logratio[active].mean().cpu().item()) if bool(active.any()) else 0.0,
            max_abs_logratio=float(logratio[active].abs().max().cpu().item())
            if bool(active.any())
            else 0.0,
        )
    return loss, stats
