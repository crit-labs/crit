"""crit's kernel expert-directory on its own: a fill copies a plan's bundles byte for byte at every vector width and
grid (more copies than blocks, a tail shorter than the grid's words, one key into several slots, none past the count,
nothing for no copies), from mapped host memory too, a range of a longer job list (job_first) only, in a CUDA graph
too and with the range's first and count changed between replays, and refuses what it cannot copy; a plan of a fresh
directory admits the routed experts in route order into the clock's slots and lists their copies. The plans themselves are checked against
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


def test_a_fill_of_no_copies_writes_nothing():
    source, store = store_of(4, 512), torch.zeros(3, 512, dtype=torch.uint8, device=dev)
    jobs, _ = jobs_of([(1, 0), (2, 1)])
    ed.fill(source, store, jobs, torch.tensor([0], **i32), 0, 16, torch.empty(0, **i32))
    torch.cuda.synchronize()
    assert not store.any()


@pytest.mark.parametrize("blocks", [1, 3, 300])
def test_a_fill_of_more_copies_than_blocks_and_a_short_tail(blocks):
    # 13 copies of a bundle that is no multiple of the grid's words: each copy's tail goes word by word
    bundle = 16 * 1031
    source, store = store_of(20, bundle), torch.zeros(14, bundle, dtype=torch.uint8, device=dev)
    pairs = [(19 - i, (5 * i) % 13) for i in range(13)]
    jobs, count = jobs_of(pairs)
    for vector in (4, 8, 16):
        store.zero_()
        ed.fill(source, store, jobs, count, blocks, vector, torch.empty(0, **i32))
        want = torch.zeros_like(store)
        for key, slot in pairs:
            want[slot] = source[key]
        assert torch.equal(store, want), vector


def test_a_fill_reads_only_its_counted_copies_and_one_key_fills_several_slots():
    source, store = store_of(6, 1024), torch.zeros(5, 1024, dtype=torch.uint8, device=dev)
    jobs, _ = jobs_of([(3, 0), (3, 2), (5, 4), (1, 1)])   # the last copy is past the count
    ed.fill(source, store, jobs, torch.tensor([3], **i32), 2, 8, torch.empty(0, **i32))
    torch.cuda.synchronize()
    assert torch.equal(store[0], source[3]) and torch.equal(store[2], source[3]) and torch.equal(store[4], source[5])
    assert not store[1].any() and not store[3].any()


def test_a_captured_fill_of_a_range_follows_its_first_and_count():
    source, store = store_of(9, 2048), torch.zeros(6, 2048, dtype=torch.uint8, device=dev)
    jobs, _ = jobs_of([(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)])
    first, count = torch.tensor([0], **i32), torch.tensor([1], **i32)
    ed.fill(source, store, jobs, count, 4, 16, first)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ed.fill(source, store, jobs, count, 4, 16, first)
    store.zero_()
    first.fill_(2)
    count.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(store[2:5], source[3:6]) and not store[:2].any() and not store[5].any()


def test_a_fill_from_mapped_host_memory_copies_its_bundles():
    # an engine's store: pinned host memory read over the link at its device address (bench.mapped_host)
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("expert_directory_bench",
                                                  pathlib.Path(__file__).resolve().parents[1] / "bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    host = store_of(7, 4096 + 16).cpu()
    found = bench.mapped_host(host, dev)
    if found is None:
        pytest.skip("this platform maps no pinned host memory for the device")
    pinned, source = found
    store = torch.zeros(4, 4096 + 16, dtype=torch.uint8, device=dev)
    pairs = [(6, 1), (0, 3), (4, 0)]
    jobs, count = jobs_of(pairs)
    for blocks in (0, 4, 32):
        store.zero_()
        ed.fill(source, store, jobs, count, blocks, 16, torch.empty(0, **i32))
        torch.cuda.synchronize()
        for key, slot in pairs:
            assert torch.equal(store[slot].cpu(), pinned[key])
        assert not store[2].any()


def test_a_fill_refuses_a_bundle_or_vector_it_cannot_copy():
    source, store = store_of(2, 256), torch.zeros(2, 256, dtype=torch.uint8, device=dev)
    jobs, count = jobs_of([(0, 0)])
    with pytest.raises(RuntimeError):
        ed.fill(source, store, jobs, count, 0, 12, torch.empty(0, **i32))
    odd = torch.zeros(2, 200, dtype=torch.uint8, device=dev)
    with pytest.raises(RuntimeError):
        ed.fill(store_of(2, 200), odd, jobs, count, 0, 16, torch.empty(0, **i32))
