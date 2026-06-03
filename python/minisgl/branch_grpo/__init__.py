"""Branch-Dr.GRPO research utilities."""

from .config import BranchGRPOConfig, main_config, smoke_config
from .loss import BranchLossStats, branch_drgrpo_loss
from .records import Edge, Leaf, Node, RolloutTree, TrainExample, materialize_leaf_slot_paths
from .verifier import binary_tag_reward, extract_answer_tag, normalize_answer

__all__ = [
    "BranchGRPOConfig",
    "BranchLossStats",
    "Edge",
    "Leaf",
    "Node",
    "RolloutTree",
    "TrainExample",
    "binary_tag_reward",
    "branch_drgrpo_loss",
    "extract_answer_tag",
    "main_config",
    "materialize_leaf_slot_paths",
    "normalize_answer",
    "smoke_config",
]
