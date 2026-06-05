"""Branch-Dr.GRPO research utilities."""

from .config import BranchGRPOConfig, bigmath128_config, fixed128_config, main_config, smoke_config
from .loss import BranchLossStats, branch_drgrpo_loss
from .records import Edge, Leaf, Node, RolloutTree, TrainExample, materialize_leaf_slot_paths
from .rollout import build_branch_rollout_tree, build_branch_rollout_trees, rollout_sampling_params
from .trainer import (
    BranchTrainStepStats,
    FP32MasterAdamW,
    PackedTrainBatch,
    branch_grpo_train_step,
    collate_train_examples,
    pack_train_examples,
    trainer_selected_logprobs,
)
from .verifier import binary_tag_reward, extract_answer_tag, normalize_answer

__all__ = [
    "BranchGRPOConfig",
    "BranchLossStats",
    "BranchTrainStepStats",
    "Edge",
    "FP32MasterAdamW",
    "Leaf",
    "Node",
    "PackedTrainBatch",
    "RolloutTree",
    "TrainExample",
    "binary_tag_reward",
    "bigmath128_config",
    "branch_drgrpo_loss",
    "branch_grpo_train_step",
    "build_branch_rollout_tree",
    "build_branch_rollout_trees",
    "collate_train_examples",
    "extract_answer_tag",
    "fixed128_config",
    "main_config",
    "materialize_leaf_slot_paths",
    "normalize_answer",
    "pack_train_examples",
    "rollout_sampling_params",
    "smoke_config",
    "trainer_selected_logprobs",
]
