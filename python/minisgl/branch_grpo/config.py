from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BranchGRPOConfig:
    name: str
    prompts_per_update: int
    rollout_wave_prompts: int
    root_samples: int
    branch_factor: int
    branch_targets: Tuple[int, ...]
    max_generation_tokens: int
    prompt_max_tokens: int = 512
    boundary_lookahead: int = 0
    confidence_threshold: float | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    ppo_clip: float = 0.2
    lr: float = 1e-6
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    kl_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.prompts_per_update <= 0:
            raise ValueError("prompts_per_update must be positive")
        if self.rollout_wave_prompts <= 0:
            raise ValueError("rollout_wave_prompts must be positive")
        if self.prompts_per_update % self.rollout_wave_prompts != 0:
            raise ValueError("prompts_per_update must be divisible by rollout_wave_prompts")
        if self.root_samples <= 0:
            raise ValueError("root_samples must be positive")
        if self.branch_factor <= 1:
            raise ValueError("branch_factor must be greater than one")
        if tuple(sorted(self.branch_targets)) != self.branch_targets:
            raise ValueError("branch_targets must be sorted")
        if any(target <= 0 or target >= self.max_generation_tokens for target in self.branch_targets):
            raise ValueError("branch targets must be inside the generation window")
        if self.boundary_lookahead < 0:
            raise ValueError("boundary_lookahead must be non-negative")
        if self.confidence_threshold is not None and not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in (0, 1] or None")

    @property
    def waves_per_update(self) -> int:
        return self.prompts_per_update // self.rollout_wave_prompts

    @property
    def leaves_per_prompt(self) -> int:
        return self.root_samples * (self.branch_factor ** len(self.branch_targets))

    @property
    def denominator_tokens(self) -> int:
        return self.prompts_per_update * self.leaves_per_prompt * self.max_generation_tokens

    @property
    def max_active_continuations_per_wave(self) -> int:
        return self.rollout_wave_prompts * self.leaves_per_prompt

    @property
    def retained_continuation_slots_per_prompt(self) -> int:
        """Continuation-table slots held while building one full prompt tree.

        xsglang forks keep parent continuations alive until the whole tree is freed. The table
        therefore needs room for the prompt root plus every retained branch continuation, not
        just the final active leaf frontier.
        """
        branch_levels = sum(
            self.branch_factor**level for level in range(len(self.branch_targets) + 1)
        )
        return 1 + self.root_samples * branch_levels

    @property
    def retained_continuation_slots_per_wave(self) -> int:
        return self.rollout_wave_prompts * self.retained_continuation_slots_per_prompt


def smoke_config() -> BranchGRPOConfig:
    return BranchGRPOConfig(
        name="smoke",
        prompts_per_update=4,
        rollout_wave_prompts=4,
        root_samples=4,
        branch_factor=2,
        branch_targets=(128, 512),
        max_generation_tokens=1024,
    )


def main_config() -> BranchGRPOConfig:
    return BranchGRPOConfig(
        name="main",
        prompts_per_update=8,
        rollout_wave_prompts=4,
        root_samples=4,
        branch_factor=2,
        branch_targets=(128, 256, 512),
        max_generation_tokens=1024,
    )


def fixed128_config() -> BranchGRPOConfig:
    # Current clean default: iid stochastic forks at fixed token intervals. The r3-best model's
    # typical completion length is around 1040 tokens, so five evenly spaced fork points turn
    # 4 root samples into a nominal 128 leaves right around that natural stopping length.
    interval = 208
    branch_count = 5
    return BranchGRPOConfig(
        name="fixed128",
        prompts_per_update=8,
        rollout_wave_prompts=4,
        root_samples=4,
        branch_factor=2,
        branch_targets=tuple(interval * (idx + 1) for idx in range(branch_count)),
        max_generation_tokens=1536,
        boundary_lookahead=0,
        confidence_threshold=None,
        lr=2e-6,
    )


def bigmath128_config() -> BranchGRPOConfig:
    interval = 208
    branch_count = 5
    return BranchGRPOConfig(
        name="bigmath128",
        prompts_per_update=8,
        rollout_wave_prompts=2,
        root_samples=4,
        branch_factor=2,
        branch_targets=tuple(interval * (idx + 1) for idx in range(branch_count)),
        max_generation_tokens=1536,
        boundary_lookahead=0,
        confidence_threshold=None,
        lr=5e-6,
    )


def round3_config() -> BranchGRPOConfig:
    # Round 3: the cap is policy sharpness (pass@8 0.56 >> greedy 0.23), not truncation.
    # Sharpen harder: root_samples 4->8 (64 leaves/prompt) surfaces more correct branches per
    # prompt => denser leave-one-out advantage to concentrate mass on correct reasoning.
    return BranchGRPOConfig(
        name="round3",
        prompts_per_update=8,
        rollout_wave_prompts=4,
        root_samples=8,
        branch_factor=2,
        branch_targets=(128, 256, 512),
        max_generation_tokens=1536,
        confidence_threshold=0.6,
        boundary_lookahead=48,
        lr=2e-6,
    )


def main_v2_config() -> BranchGRPOConfig:
    # v2 bets vs the 0.262 baseline: confidence-gated branching (mandated), longer outputs
    # (max_gen 1024->1536, fewer truncated correct traces), faster climb (lr 1e-6->2e-6).
    return BranchGRPOConfig(
        name="main_v2",
        prompts_per_update=8,
        rollout_wave_prompts=4,
        root_samples=4,
        branch_factor=2,
        branch_targets=(128, 256, 512),
        max_generation_tokens=1536,
        confidence_threshold=0.6,
        boundary_lookahead=48,
        lr=2e-6,
    )
