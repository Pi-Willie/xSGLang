from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import PinnedRingBuffer, div_ceil

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

    from .utils import PendingReq


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        *,
        model_config: "ModelConfig | None" = None,
        state_cache=None,
    ):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(
            device=device,
            type=type,
            model_config=model_config,
            state_cache=state_cache,
        )
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.state_cache = state_cache
        self._table_idx_buffer = PinnedRingBuffer(device, torch.int64)
        self._position_buffer = PinnedRingBuffer(device, torch.int64)
        # Per-page ref counts for fork/snapshot tracked states.
        # Key is page start index in token-space (always aligned by page_size).
        self._tracked_page_refs: Dict[int, int] = {}

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            self._write_page_table(allocated, allocation_info)

    def allocate_for_range(self, table_idx: int, start_pos: int, end_pos: int) -> None:
        if end_pos <= start_pos:
            return
        first_page = div_ceil(start_pos, self.page_size)
        last_page = div_ceil(end_pos, self.page_size)
        if last_page <= first_page:
            return
        allocated = self._page_to_token(self._allocate(last_page - first_page))
        self._write_page_table(allocated, [(table_idx, first_page, last_page)])

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.materialize_input_ids()[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        if hasattr(self.prefix_cache, "insert_tracked_prefix") and self.state_cache is not None:
            insert_len = int(self.prefix_cache.align_cached_len(req.cached_len))
            tracked_slots = self.state_cache.consume_tracked_prefixes(req.table_idx, upto=insert_len)
            if insert_len > old_handle.cached_len and tracked_slots:
                cached_len, new_handle = self.prefix_cache.insert_tracked_prefix(
                    insert_ids[:insert_len],
                    page_indices[:insert_len],
                    tracked_slots,
                )
                self.unlock(old_handle)
                self._free(page_indices[old_handle.cached_len : cached_len])
                if finished:
                    self._free(page_indices[new_handle.cached_len :])
                else:
                    req.cache_handle = new_handle
                    self.lock(new_handle)
            else:
                self.state_cache.discard_tracked_prefixes(req.table_idx)
                if finished:
                    self.unlock(old_handle)
                    self._free(page_indices[old_handle.cached_len :])
            return

        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.unlock(old_handle)
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:
            self._free(page_indices[new_handle.cached_len :])
        else:
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices)

        lazy_free_list: List[torch.Tensor] = []
        real_free = self._free
        try:
            self._free = lazy_free
            yield
        finally:
            self._free = real_free
            for indices in lazy_free_list:
                self._free(indices)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_page_starts(self._indices_to_page_starts(evicted))
            if len(self.free_slots) < needed_pages:
                raise RuntimeError(
                    "KV cache exhausted during decode: "
                    f"need {needed_pages} pages, only {len(self.free_slots)} free after eviction."
                )
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        self.free_page_starts(self._indices_to_page_starts(indices))

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()

    def _indices_to_page_starts(self, indices: torch.Tensor) -> List[int]:
        if len(indices) == 0:
            return []
        page_starts = (
            torch.div(indices.to(torch.int64), self.page_size, rounding_mode="floor") * self.page_size
        )
        return [int(v) for v in torch.unique(page_starts).cpu().tolist()]

    def state_page_starts(self, req: Req, *, upto: int | None = None) -> List[int]:
        length = req.device_len if upto is None else upto
        if length <= 0:
            return []
        indices = self.page_table[req.table_idx, :length]
        return self._indices_to_page_starts(indices)

    def track_fork_from_state(self, req: Req) -> None:
        for page_start in self.state_page_starts(req):
            if page_start in self._tracked_page_refs:
                self._tracked_page_refs[page_start] += 1
            else:
                # Parent + child references.
                self._tracked_page_refs[page_start] = 2

    def track_clone_page_starts(self, page_starts: Iterable[int]) -> None:
        for page_start in page_starts:
            if page_start in self._tracked_page_refs:
                self._tracked_page_refs[page_start] += 1
            else:
                self._tracked_page_refs[page_start] = 2

    def release_state_tracking(self, req: Req, *, upto: int | None = None) -> None:
        self.release_page_starts_tracking(self.state_page_starts(req, upto=upto))

    def release_page_starts_tracking(self, page_starts: Iterable[int]) -> None:
        for page_start in page_starts:
            refs = self._tracked_page_refs.get(page_start)
            if refs is None:
                continue
            refs -= 1
            if refs <= 0:
                self._tracked_page_refs.pop(page_start, None)
            else:
                self._tracked_page_refs[page_start] = refs

    def tracked_ref_count(self, page_start: int) -> int:
        return self._tracked_page_refs.get(page_start, 0)

    def replace_tracked_page(self, old_page_start: int, new_page_start: int) -> None:
        refs = self._tracked_page_refs.get(old_page_start)
        if refs is None or refs <= 0:
            raise RuntimeError(f"Cannot replace untracked page: {old_page_start}")
        refs -= 1
        if refs == 0:
            self._tracked_page_refs.pop(old_page_start, None)
        else:
            self._tracked_page_refs[old_page_start] = refs
        # New page is now uniquely owned by the writing state.
        self._tracked_page_refs[new_page_start] = 1

    def free_page_starts(self, page_starts: Iterable[int]) -> None:
        slots = [p for p in page_starts if self._tracked_page_refs.get(p, 0) == 0]
        if len(slots) == 0:
            return
        slot_tensor = torch.tensor(slots, dtype=torch.int32, device=self.device)
        self.free_slots = torch.cat([self.free_slots, slot_tensor])

    def _write_page_table(
        self,
        allocated: torch.Tensor,
        allocation_info: List[Tuple[int, int, int]],
    ) -> None:
        needed_tokens = len(allocated)
        table_idx_host, table_idxs = self._table_idx_buffer.acquire(needed_tokens)
        positions_host, positions = self._position_buffer.acquire(needed_tokens)
        offset = 0
        for table_idx, first_page, last_page in allocation_info:
            first_pos, last_pos = first_page * self.page_size, last_page * self.page_size
            length = last_pos - first_pos
            table_idx_host[offset : offset + length].fill_(table_idx)
            torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
            offset += length
        assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
        if needed_tokens > 0:
            table_idxs.copy_(table_idx_host, non_blocking=True)
            positions.copy_(positions_host, non_blocking=True)
        self.page_table[table_idxs, positions] = allocated
