"""crit's kernel expert-directory on its own: a fill copies a plan's bundles byte for byte at every vector width and
grid, a range of a longer job list (job_first) only, in a CUDA graph too; a plan of a fresh directory admits the routed
experts in route order into the clock's slots and lists their copies. The plans themselves are checked against
device_cache.LayerDirectory, plan for plan with the clock, the admission limits, prompt rings and recent residency, by
device-expert-cache's tests (tests/test_device.py)."""
import pytest
import torch

from crit_kernels import build as kbuild

ed = kbuild.load("expert-directory")
dev = torch.device("cuda")
i32 = dict(dtype=torch.int32, device=dev)


def store_of(keys, bundle):
    k = torch.arange(keys, device=dev)
    return ((k[:, None] * 37 + torch.arange(bundle, device=dev)[None, :] * 11) % 253).to(torch.uint8)


def jobs_of(pairs):
    jobs = torch.zeros(len(pairs) + 1, 2, **i32)
    if pairs:
        jobs[:len(pairs)] = torch.tensor(pairs, **i32)
    return jobs, torch.tensor([len(pairs)], **i32)


@pytest.mark.parametrize("vector", [4, 8, 16])
@pytest.mark.parametrize("blocks", [0, 1, 7])
def test_a_fill_copies_its_bundles(vector, blocks):
    source, store = store_of(9, 4096 + 48), torch.zeros(6, 4096 + 48, dtype=torch.uint8, device=dev)
    pairs = [(8, 0), (2, 5), (4, 3)]
    jobs, count = jobs_of(pairs)
    ed.fill(source, store, jobs, count, blocks, vector, torch.empty(0, **i32))
    want = torch.zeros_like(store)
    for key, slot in pairs:
        want[slot] = source[key]
    assert torch.equal(store, want)


def test_a_fill_of_a_range_copies_that_range_only():
    source, store = store_of(9, 256), torch.zeros(6, 256, dtype=torch.uint8, device=dev)
    jobs, _ = jobs_of([(1, 0), (2, 1), (3, 2), (4, 3)])
    ed.fill(source, store, jobs, torch.tensor([2], **i32), 0, 16, torch.tensor([1], **i32))
    assert torch.equal(store[1:3], source[2:4]) and not store[0].any() and not store[3:].any()


def test_a_captured_fill_replays_its_copies():
    source, store = store_of(9, 1024), torch.zeros(6, 1024, dtype=torch.uint8, device=dev)
    jobs, count = jobs_of([(3, 1), (5, 4)])
    empty = torch.empty(0, **i32)
    ed.fill(source, store, jobs, count, 0, 16, empty)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ed.fill(source, store, jobs, count, 0, 16, empty)
    store.zero_()
    jobs[:2] = torch.tensor([[7, 0], [2, 2]], **i32)   # the same addresses, new copies
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(store[0], source[7]) and torch.equal(store[2], source[2]) and not store[1].any()


def test_a_fresh_directorys_plan_admits_the_routes_in_order():
    slots, experts, keys = 4, 8, 8   # one layer: keys are the layer's experts
    key_slot = torch.full((keys + 1,), -1, **i32)
    slot_key = torch.full((slots + 1,), -1, **i32)
    slot_ref = torch.zeros(slots + 1, dtype=torch.uint8, device=dev)
    slot_pin = torch.full((slots + 1,), -1, **i32)
    state = torch.zeros(4, **i32)
    counters = torch.zeros(8, dtype=torch.int64, device=dev)
    out = torch.full((experts,), -7, **i32)
    jobs = torch.zeros(experts + 1, 2, **i32)
    count = torch.zeros(1, **i32)
    e32, e64, e8 = torch.empty(0, **i32), torch.empty(0, dtype=torch.int64, device=dev), torch.empty(0, dtype=torch.uint8,
                                                                                                    device=dev)
    ids = torch.tensor([[5, 2], [5, -1]], dtype=torch.int64, device=dev)
    ed.plan(key_slot, slot_key, slot_ref, slot_pin, state, counters, e8, e64, e32, -1, 0, experts, ids.view(-1), 2,
            e64, e64, e32, out, jobs, count, e32, e32, e32, e32, e32, False)
    torch.cuda.synchronize()
    table = out.tolist()
    assert 0 <= table[5] < slots and 0 <= table[2] < slots and table[5] != table[2]
    assert all(table[e] == -1 for e in range(experts) if e not in (2, 5))
    n = int(count.item())
    assert n == 2 and [tuple(j) for j in jobs[:n].tolist()] == [(5, table[5]), (2, table[2])]
