from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]
RELEASE_SLOT_FN: TypeAlias = Callable[[int], None]


class HybridRadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, HybridRadixTreeNode] = {}
        self._parent: HybridRadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = HybridRadixTreeNode.counter
        HybridRadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()
        self.state_slot: int | None = None

        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: HybridRadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> HybridRadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> HybridRadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = HybridRadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        # Hybrid recurrent state only exists at aligned chunk boundaries.
        new_node.state_slot = None
        return new_node

    def __lt__(self, other: HybridRadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class HybridRadixCacheHandle(BaseCacheHandle):
    node: HybridRadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list) if value_list else torch.empty(0, dtype=torch.int32)

    @property
    def snapshot_slot(self) -> int | None:
        return self.node.state_slot


class HybridRadixPrefixCache(BasePrefixCache):
    def __init__(
        self,
        *,
        device: torch.device,
        release_state_slot: RELEASE_SLOT_FN,
        segment_size: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.segment_size = max(segment_size, self.page_size)
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self._release_state_slot = release_state_slot
        self.root_node = HybridRadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1

    def align_cached_len(self, length: int) -> int:
        return align_down(length, self.segment_size)

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, HybridRadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len, best_node, best_prefix_len = self._tree_walk(input_ids)
        _ = node, prefix_len
        return MatchResult(HybridRadixCacheHandle(best_prefix_len, best_node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        raise NotImplementedError(
            "HybridRadixPrefixCache requires insert_tracked_prefix() with state snapshots"
        )

    def insert_tracked_prefix(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        tracked_slots: List[Tuple[int, int]],
    ) -> InsertResult:
        if len(tracked_slots) == 0:
            return InsertResult(0, HybridRadixCacheHandle(0, self.root_node))

        insert_len = tracked_slots[-1][0]
        input_ids = input_ids[:insert_len]
        indices = indices[:insert_len]
        existing = self.match_prefix(input_ids).cuda_handle
        assert isinstance(existing, HybridRadixCacheHandle)
        cached_len = existing.cached_len
        handle = existing
        for boundary_len, slot in tracked_slots:
            if boundary_len <= cached_len:
                self._release_state_slot(slot)
                continue
            handle = self._insert_boundary(input_ids[:boundary_len], indices[:boundary_len], slot)
        return InsertResult(cached_len, handle)

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        if size > self.evictable_size:
            raise RuntimeError(f"Cannot evict {size}, only {self.evictable_size} is evictable")

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            if not leave_nodes:
                raise RuntimeError(
                    f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
                )
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            if node.state_slot is not None:
                self._release_state_slot(node.state_slot)
                node.state_slot = None
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("HybridRadixPrefixCache.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def _insert_boundary(
        self,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
        state_slot: int,
    ) -> HybridRadixCacheHandle:
        insert_len = len(input_ids)
        node, prefix_len, _, _ = self._tree_walk(input_ids)
        if prefix_len != insert_len:
            new_node = HybridRadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.state_slot = state_slot
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        elif node.state_slot is None:
            node.state_slot = state_slot
        else:
            self._release_state_slot(state_slot)
        return HybridRadixCacheHandle(insert_len, node)

    def _collect_leave_nodes_for_evict(self) -> List[HybridRadixTreeNode]:
        nodes: List[HybridRadixTreeNode] = [self.root_node]
        leave_nodes: List[HybridRadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[HybridRadixTreeNode, int, HybridRadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        best_node = self.root_node
        best_prefix_len = 0
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len, best_node, best_prefix_len
            node = child_node
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            if match_len != node.length:
                node = node.split_at(match_len)
                return node, prefix_len, best_node, best_prefix_len

            node.timestamp = tic
            if node.state_slot is not None:
                best_node = node
                best_prefix_len = prefix_len

        return node, prefix_len, best_node, best_prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
