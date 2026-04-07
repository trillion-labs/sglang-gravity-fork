from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator, page_size):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("Streaming requests should not delegate to inner cache")

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeReq:
    def __init__(self, session_id: str, req_pool_idx: int, committed: int, allocated: int):
        self.session = SimpleNamespace(session_id=session_id, streaming=True)
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = committed
        self.kv_allocated_len = allocated
        self.swa_evicted_seqlen = 0
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.pop_overallocated_calls = 0

    def pop_overallocated_kv_cache(self):
        self.pop_overallocated_calls += 1
        return self.kv_committed_len, self.kv_allocated_len


def test_streaming_release_kv_cache_trims_overallocated_tail(monkeypatch):
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = SessionAwareCache(_FakeInnerCache(req_to_token_pool, allocator, page_size))
    req = _FakeReq("session-a", req_pool_idx=0, committed=17, allocated=40)

    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.get_global_server_args",
        lambda: SimpleNamespace(page_size=page_size, speculative_algorithm="eagle"),
    )

    release_kv_cache(req, tree_cache)

    slot = tree_cache.slots["session-a"]
    assert req.pop_overallocated_calls == 1
    assert req.req_pool_idx is None
    assert slot.kv_committed_len == 17
    assert slot.kv_allocated_len == 17
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(32, 40))
