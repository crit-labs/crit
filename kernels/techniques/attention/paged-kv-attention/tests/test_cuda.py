"""The CUDA flash tile (attend(..., flash="cuda"), prefill(..., flash="cuda"); structures cuda-window, cuda-fused,
cuda-prefill) against a float64 reference: windows of every page kind, a column's bits the same alone, in any window,
packed, fused or split and whatever the warps, trees, fresh rows, CUDA graph replays at other lengths over poisoned
positions past the row, prompt chunks split on the device or not, and the knobs (speed knobs keep every bit, the
order knobs reach every launch)."""
import pytest
import torch

from crit_kernels.build import load, select
from test_paged_kv_attention import Cache, _q, dev, reference

pk = load("paged-kv-attention")
CUDA = dict(flash="cuda")
SHAPES = [(8, 2, 128, 128), (24, 2, 256, 256), (56, 8, 128, 128), (8, 4, 64, 128), (12, 2, 256, 128)]

# FUSED_SERIAL (a fused launch's form): every launch one program a tile; none (a program a split, through the ring
# or as the window launch and its merge)
SERIAL, RING = 1, 100000
_SERIAL_STRUCTURE = 3


def _launched(monkeypatch):
    """The structures the package launches from here on (cuda_window.cu's S_*)."""
    seen, launch = [], pk._cuda_launch

    def recorded(lib, structure, *args):
        seen.append(structure)
        return launch(lib, structure, *args)
    monkeypatch.setattr(pk, "_cuda_launch", recorded)
    return seen


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp16", "fp8", "fp8:row", "int8"])
@pytest.mark.parametrize("hq,hkv,d,dv", SHAPES)
@pytest.mark.parametrize("page", [16, 48])
def test_cuda_windows_match_float64_attention(kind, hq, hkv, d, dv, page):
    cache = Cache(kind, hkv, d, dv, page, [130, 64, 1, 2517])
    rows, starts, firsts = [0, 1, 2, 3], [0, 5, 10, 15], [125, 59, 0, 2400]
    counts = [5, 5, 1, 3]
    q = _q(20, hq, d, 3)
    scale = d ** -0.5
    qd = q.to(torch.bfloat16 if kind != "fp16" else torch.float16).double()
    for fused in (False, True):
        lse = torch.empty(20, hq, device=dev)
        out = cache.attend(q, rows, starts, firsts, 5, lse=lse, fused=fused, chunk=256 if fused else None,
                           counts=torch.tensor(counts, dtype=torch.int32, device=dev), **CUDA)
        for s, start, first, count in zip(rows, starts, firsts, counts):
            keys, values = cache.rows(s, first + count)
            for c in range(count):
                seen = torch.arange(first + count, device=dev) <= first + c
                want, want_lse = reference(qd[start + c], keys, values, seen, scale)
                torch.testing.assert_close(out[start + c].double(), want, atol=8e-3, rtol=2e-2)
                torch.testing.assert_close(lse[start + c].double(), want_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8:row", "int8", "fp8"])
@pytest.mark.parametrize("hq,hkv,d", [(24, 2, 256), (56, 8, 128)])
def test_a_cuda_column_has_its_bits_alone_in_any_window_fused_or_split(kind, hq, hkv, d):
    # Every column has the bits of the same column decoded alone (log-sum-exps too), in windows of every width, packed
    # by tokens, fused or split, in tiles of any warps, at lengths of one split and of many.
    for length, chunk in ((40, None), (2100, None), (5000, 256)):
        cache = Cache(kind, hkv, d, d, 16, [length, length], seed=1)
        kw = dict(chunk=chunk, **CUDA)
        T = min(64, length)
        pos0 = length - T
        q = _q(T, hq, d, 2)
        one_lse = torch.empty(T, hq, device=dev)
        alone = cache.attend(q, [0] * T, list(range(T)), [pos0 + i for i in range(T)], 1, lse=one_lse, **kw)
        for cols in (2, 5, 16, 64):
            starts = list(range(0, T, cols))
            counts = torch.tensor([min(cols, T - s) for s in starts], dtype=torch.int32, device=dev)
            for fused, warps in ((False, None), (True, None), (False, 2), (True, 8), (False, 8)):
                lse = torch.empty(T, hq, device=dev)
                got = cache.attend(q, [0] * len(starts), starts, [pos0 + s for s in starts], cols, counts=counts,
                                   lse=lse, fused=fused, num_warps=warps, **kw)
                assert torch.equal(got, alone), (kind, length, cols, fused, warps)
                assert torch.equal(lse, one_lse), (kind, length, cols, fused, warps)
        offsets = torch.tensor([0, 3], dtype=torch.int32, device=dev)
        parts = pk.cuda_parts(d, d, hq, hkv, 16, kind, cache.tables.shape[1] * 16, chunk)   # with the ramp
        packed = cache.attend(q, [0, 1], [0, 3], [pos0, pos0 + 3], 8, offsets=offsets, tokens=8,
                              counts=torch.tensor([3, 5], dtype=torch.int32, device=dev),
                              workspace=torch.empty(pk.workspace_floats(8, hq, parts, d), device=dev), **kw)
        assert torch.equal(packed[:3], alone[:3])
        other = cache.attend(q[3:8], [1], [0], [pos0 + 3], 5, **kw)     # the second sequence holds other keys
        assert torch.equal(packed[3:8], other)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "int8"])
@pytest.mark.parametrize("group", [1, 2, 3, 16])
def test_a_fused_cuda_launch_has_the_split_launchs_bits_in_groups_of_any_size(kind, group, monkeypatch):
    # The fused launch's programs fold their tile's split states in groups (a ring of two groups' slots, a running
    # state between groups): the group is scheduling only, every size gives the split launch's bits, and the sync words
    # a launch leaves serve the next one (launched twice, graph-replayed). FUSED_PARTIALS 0: every fused launch through
    # its ring (not as the window launch and its merge).
    monkeypatch.setattr(pk, "FUSED_GROUP", group)
    monkeypatch.setattr(pk, "FUSED_PARTIALS", 0)
    monkeypatch.setattr(pk, "FUSED_SERIAL", RING)
    cache = Cache(kind, 2, 256, 256, 16, [3000, 700, 9], seed=4)
    q = _q(40, 24, 256, 5)
    rows, starts, firsts = [0, 1, 2], [0, 16, 32], [2984, 690, 1]
    counts = torch.tensor([16, 10, 8], dtype=torch.int32, device=dev)
    lse, want_lse = torch.empty(40, 24, device=dev), torch.empty(40, 24, device=dev)
    want = cache.attend(q, rows, starts, firsts, 16, counts=counts, lse=want_lse, chunk=128, **CUDA)
    for _ in range(2):
        got = cache.attend(q, rows, starts, firsts, 16, counts=counts, lse=lse, chunk=128, fused=True, **CUDA)
        assert torch.equal(got[:26], want[:26]) and torch.equal(got[32:40], want[32:40])
        assert torch.equal(lse[:26], want_lse[:26]) and torch.equal(lse[32:40], want_lse[32:40])
    qq = q.to(torch.bfloat16)
    out = torch.empty(40, 24, 256, dtype=torch.bfloat16, device=dev)
    args = (qq, cache.k, cache.v, kind, cache.tables, torch.tensor(rows, dtype=torch.int32, device=dev),
            torch.tensor(starts, dtype=torch.int32, device=dev), torch.tensor(firsts, device=dev), 16, 16)
    kw = dict(key_scales=cache.ks, value_scales=cache.vs, counts=counts, out=out, chunk=128, fused=True, **CUDA)
    pk.attend(*args, **kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pk.attend(*args, **kw)
    for _ in range(3):
        out.zero_()
        graph.replay()
        assert torch.equal(out[:26], want[:26]) and torch.equal(out[32:40], want[32:40])


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "int8"])
def test_a_cuda_tree_sees_its_ancestors_and_a_chain_is_causal(kind):
    cache = Cache(kind, 2, 128, 128, 16, [400])
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], dtype=torch.int64, device=dev)
    q = _q(8, 8, 128, 6)
    out = cache.attend(q, [0], [0], [392], 8, parents=parents, **CUDA)
    keys, values = cache.rows(0, 400)
    qd = q.to(torch.bfloat16).double()
    for c in range(8):
        found, p = {c}, int(parents[c])
        while p >= 0:
            found.add(p)
            p = int(parents[p])
        seen = torch.tensor([n < 392 or (n - 392) in found for n in range(400)], device=dev)
        want, _ = reference(qd[c], keys, values, seen, 128 ** -0.5)
        torch.testing.assert_close(out[c].double(), want, atol=8e-3, rtol=2e-2)
    chain = cache.attend(q, [0], [0], [392], 8, parents=torch.arange(-1, 7, device=dev), **CUDA)
    assert torch.equal(chain, cache.attend(q, [0], [0], [392], 8, **CUDA))
    assert torch.equal(cache.attend(q, [0], [0], [392], 8, parents=parents.int(), fused=True, **CUDA), out)
    assert torch.equal(cache.attend(q, [0], [0], [392], 8, parents=parents, num_stages=1, **CUDA), out)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "bf16"])
def test_a_fused_cuda_chunk_reads_its_own_keys_and_values_fresh(kind):
    cache = Cache(kind, 2, 128, 128, 16, [700])
    g = torch.Generator().manual_seed(21)
    T = 100
    q = _q(T, 8, 128, 22).to(torch.bfloat16)
    fk = torch.randn(T, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    fv = torch.randn(T, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    frontier = torch.zeros(1, dtype=torch.int64, device=dev)
    first = torch.zeros(1, dtype=torch.int64, device=dev)
    starts = torch.tensor([0, 64], dtype=torch.int32, device=dev)
    offsets = torch.tensor([0, 64], dtype=torch.int64, device=dev)
    counts = torch.tensor([64, T - 64], dtype=torch.int32, device=dev)
    rows = torch.zeros(2, dtype=torch.int32, device=dev)
    out, lse = torch.empty_like(q), torch.empty(T, 8, device=dev)
    firsts = torch.empty(2, dtype=torch.int64, device=dev)

    def launch():
        torch.add(offsets, frontier, out=firsts)
        pk.attend(q, cache.k, cache.v, kind, cache.tables, rows, starts, firsts, 64, 16, counts=counts,
                  key_scales=cache.ks, value_scales=cache.vs, out=out, lse=lse, fused=True, chunk=256,
                  fresh=(fk, fv, frontier, first), **CUDA)
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    qd = q.double()
    for at in (0, 200, 600):
        frontier.fill_(at)
        for split in (0, 37, 100):
            first.fill_(at + split)
            graph.replay()
            keys, values = cache.rows(0, at + split)
            keys, values = torch.cat([keys, fk[split:].double()]), torch.cat([values, fv[split:].double()])
            for t in (0, 41, 99):
                seen = torch.arange(at + T, device=dev) <= at + t
                want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
                torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2, msg=(at, split, t))
                torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(at, split, t))
            if split == T:
                paged = cache.attend(q, [0, 0], [0, 64], [at, at + 64], 64, counts=counts, chunk=256, **CUDA)
                assert torch.equal(out, paged), at


@pytest.mark.kernels_ci
def test_a_cuda_graph_replays_at_every_length_reading_the_rows_positions_only():
    # One graph over a 200K-position table, replayed at a long and a short length with every position past the row
    # poisoned first (NaN scales): the replay reads the row's own positions only and equals an eager launch.
    cache = Cache("fp8:row", 4, 256, 256, 64, [200_000])
    q = _q(5, 24, 256, 1).to(torch.bfloat16)
    firsts = torch.tensor([0], device=dev)
    args = (q, cache.k, cache.v, "fp8:row", cache.tables, torch.zeros(1, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev), firsts, 5, 64)
    kw = dict(key_scales=cache.ks, value_scales=cache.vs, **CUDA)
    pk.attend(*args, **kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pk.attend(*args, **kw)
    for first in (199_000, 4_000, 1):
        firsts.fill_(first)
        want = pk.attend(*args, **kw)
        past = cache.slots(0, torch.arange(first + 5, 200_000, device=dev))
        cache.ks.view(-1, 4)[past] = float("nan")
        cache.vs.view(-1, 4)[past] = float("nan")
        graph.replay()
        assert bool(torch.isfinite(out).all()), first
        assert torch.equal(out, want), first


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "bf16", "fp8"])
@pytest.mark.parametrize("T", [16, 100, 600])
def test_cuda_prompt_chunks_match_float64_attention(kind, T):
    cache = Cache(kind, 2, 128, 128, 16, [200 + T])
    q = _q(T, 8, 128, 4).to(torch.bfloat16)
    keys, values = cache.rows(0, 200 + T)
    qd = q.double()
    table = cache.tables[0]
    for splits in (0, 1, 16):
        cuda = _tuned({"cuda-prefill": {"SPLITS": splits}}) if splits else pk
        lse = torch.empty(T, 8, device=dev)
        out = cuda.prefill(q, cache.k, cache.v, kind, table, 200, 16, key_scales=cache.ks, value_scales=cache.vs,
                           multipliers=cache.multipliers, lse=lse, **CUDA)
        for t in (0, T // 3, T - 1):
            seen = torch.arange(200 + T, device=dev) <= 200 + t
            want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
            torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2, msg=(splits, t))
            torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(splits, t))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "bf16"])
@pytest.mark.parametrize("T", [16, 64])
def test_a_cuda_prompt_chunk_at_a_device_position_replays_from_one_cuda_graph(kind, T):
    cache = Cache(kind, 4, 256, 256, 16, [700])
    q = _q(T, 24, 256, 12).to(torch.bfloat16)
    table = cache.tables[0]
    position = torch.zeros(1, dtype=torch.int64, device=dev)
    out = torch.empty_like(q)

    def launch():
        pk.prefill(q, cache.k, cache.v, kind, table, position, 16, key_scales=cache.ks, value_scales=cache.vs,
                   multipliers=cache.multipliers, out=out, **CUDA)
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for at in (0, 5, 300, 700 - T):
        position.fill_(at)
        graph.replay()
        want = pk.prefill(q, cache.k, cache.v, kind, table, at, 16, key_scales=cache.ks, value_scales=cache.vs,
                          multipliers=cache.multipliers, **CUDA)
        assert torch.equal(out, want), at


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "bf16"])
def test_a_cuda_prompt_chunk_reads_its_own_keys_and_values_fresh(kind):
    cache = Cache(kind, 2, 128, 128, 16, [700])
    g = torch.Generator().manual_seed(21)
    q = _q(100, 8, 128, 22).to(torch.bfloat16)
    fk = torch.randn(100, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    fv = torch.randn(100, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    table = cache.tables[0]
    position = torch.zeros(1, dtype=torch.int64, device=dev)
    first = torch.zeros(1, dtype=torch.int64, device=dev)
    out, lse = torch.empty_like(q), torch.empty(100, 8, device=dev)
    graphs = []
    for fresh in ((fk, fv), (fk, fv, first)):
        def launch(fresh=fresh):
            pk.prefill(q, cache.k, cache.v, kind, table, position, 16, key_scales=cache.ks, value_scales=cache.vs,
                       multipliers=cache.multipliers, out=out, lse=lse, fresh=fresh, **CUDA)
        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graphs.append(graph)
    qd = q.double()
    for at in (0, 200, 600):
        position.fill_(at)
        for graph, split in ((graphs[0], 0), (graphs[1], 0), (graphs[1], 37), (graphs[1], 100), (graphs[1], 140)):
            first.fill_(at + split)
            graph.replay()
            own = min(split, 100)
            keys, values = cache.rows(0, at + own)
            keys, values = torch.cat([keys, fk[own:].double()]), torch.cat([values, fv[own:].double()])
            for t in (0, 41, 99):
                seen = torch.arange(at + 100, device=dev) <= at + t
                want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
                torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2, msg=(at, split, t))
                torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(at, split, t))


def _biased_reference(q, keys, values, bias, position, t, sliding, scale):
    """Token t of a chunk at `position` in float64: positions d < extent behind it get bias[t][h][d] after the scale."""
    end = position + t + 1
    behind = end - 1 - torch.arange(end, device=dev)
    seen = behind >= 0 if not sliding else behind < sliding
    extent = bias.shape[2]
    out, lse = [], []
    hq, hkv = q.shape[1], keys.shape[1]
    for h in range(hq):
        k, v = keys[:end, h // (hq // hkv)], values[:end, h // (hq // hkv)]
        s = (k @ q[t, h].double()) * scale
        s = s + torch.where(behind < extent, bias[t, h].double()[behind.clamp(max=extent - 1)], 0.0)
        s = torch.where(seen, s, torch.tensor(float("-inf"), dtype=torch.float64, device=dev))
        lse.append(torch.logsumexp(s, 0))
        out.append(torch.softmax(s, 0) @ v)
    return torch.stack(out), torch.stack(lse)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "bf16"])
@pytest.mark.parametrize("sliding,extent", [(0, 48), (64, 64), (0, 1000)])
def test_a_cuda_prompt_chunk_adds_each_tokens_distance_bias(kind, sliding, extent):
    # Each token's scores for the positions d < extent behind it get its bias[d] after the scale (nothing further
    # back), over fresh rows, a sliding window, split or in one range; a bias of zeros keeps every bit of none, and a
    # chunk captured at a device position replays the eager chunk's bits.
    T, at = 100, 200
    cache = Cache(kind, 2, 128, 128, 16, [at + T])
    g = torch.Generator().manual_seed(31)
    q = _q(T, 8, 128, 32).to(torch.bfloat16)
    fk = torch.randn(T, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    fv = torch.randn(T, 2, 128, generator=g).to(dev).to(torch.bfloat16)
    bias = (torch.randn(T, 8, extent, generator=g) * 2).to(dev)
    table = cache.tables[0]
    keys, values = cache.rows(0, at)
    keys, values = torch.cat([keys, fk.double()]), torch.cat([values, fv.double()])

    def chunk(position, **given):
        return pk.prefill(q, cache.k, cache.v, kind, table, position, 16, key_scales=cache.ks, value_scales=cache.vs,
                          multipliers=cache.multipliers, fresh=(fk, fv), sliding=sliding, **CUDA, **given)
    for splits in (1, 4, None):
        lse = torch.empty(T, 8, device=dev)
        out = chunk(at, bias=bias, lse=lse, splits=splits)
        for t in (0, 37, T - 1):
            want, want_lse = _biased_reference(q, keys, values, bias, at, t, sliding, 128 ** -0.5)
            torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2, msg=(splits, t))
            torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(splits, t))
        assert torch.equal(chunk(at, bias=torch.zeros_like(bias), splits=splits), chunk(at, splits=splits)), splits
    position = torch.full((1,), at, dtype=torch.int64, device=dev)
    out = torch.empty_like(q)
    chunk(position, bias=bias, out=out, splits=1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chunk(position, bias=bias, out=out, splits=1)
    for where in (0, 150, at):
        position.fill_(where)
        graph.replay()
        assert torch.equal(out, chunk(where, bias=bias, splits=1)), where


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "bf16"])
@pytest.mark.parametrize("sliding,extent", [(0, 0), (0, 64), (64, 64), (0, 1024), (1024, 1024)])
def test_a_cuda_prompt_chunk_in_one_range_gives_a_prefix_its_bits_at_every_length(kind, sliding, extent):
    # The chunk of 17 tokens and the chunk of 629 at the same position: the 17 rows equal bit for bit (splits 1 reads
    # a token's positions in one range, whatever the chunk), the bias the longer chunk's rows of it.
    T, short = 629, 17
    cache = Cache(kind, 2, 128, 128, 16, [T])
    g = torch.Generator().manual_seed(7)
    q = _q(T, 8, 128, 9).to(torch.bfloat16)
    biased = extent > 0
    bias = (torch.randn(T, 8, extent, generator=g) * 2).to(dev) if biased else None
    table = cache.tables[0]

    def chunk(n):
        given = {"bias": bias[:n].contiguous()} if biased else {}
        return pk.prefill(q[:n].contiguous(), cache.k, cache.v, kind, table, 0, 16, key_scales=cache.ks,
                          value_scales=cache.vs, multipliers=cache.multipliers, sliding=sliding, splits=1, **CUDA,
                          **given)
    whole, part = chunk(T), chunk(short)
    assert torch.equal(part, whole[:short]), float((part.float() - whole[:short].float()).abs().max())


def test_a_prompt_chunks_bias_is_the_cuda_tiles_and_of_its_shape():
    cache = Cache("bf16", 2, 128, 128, 16, [64])
    q = _q(16, 8, 128, 3).to(torch.bfloat16)
    for flash, bias in ((False, torch.zeros(16, 8, 5, device=dev)), ("cuda", torch.zeros(16, 4, 5, device=dev)),
                        ("cuda", torch.zeros(16, 8, 5, device=dev, dtype=torch.bfloat16))):
        with pytest.raises(ValueError):
            pk.prefill(q, cache.k, cache.v, "bf16", cache.tables[0], 0, 16, flash=flash, bias=bias)


def _tuned(knobs):
    module = load("paged-kv-attention", configs={s: [k] for s, k in knobs.items()})
    for structure in knobs:
        select(module, structure, 1)
    return module


@pytest.mark.kernels_ci
def test_cuda_speed_knobs_keep_every_bit_and_the_order_knobs_reach_every_launch():
    speed = _tuned({"cuda-window": {"STAGES": 3, "WARPS": 2}, "cuda-fused": {"STAGES": 1, "VSPLIT": 2, "WARPS": 8}})
    shared = _tuned({"cuda-window": {"VWARPS": 1, "CONVERT": 1, "STAGES": 3},
                     "cuda-fused": {"WARPS": 2, "VWARPS": 1, "CONVERT": 1}})
    order = _tuned({"cuda-window": {"CHUNK": 256, "BLOCK_N": 64}})
    assert order.cuda_chunk(128, 128, 24, 4, 16, "int8") == 256
    assert order.cuda_block(128, 128, 24, 4, 16, "int8") == 64
    ramped = _tuned({"cuda-window": {"CHUNK": 512, "RAMP": 2}})
    assert ramped.cuda_ramp(256, 256, 24, 2, 16, "bf16") == (2, ramped.CUDA_RAMP_SPLITS)
    assert pk.cuda_ramp(256, 256, 24, 2, 16, "bf16")[0] == 1   # the default: RAMP 1 (the first splits half as long)
    # The untuned rules: blocks of 16 over 16-bit pages, 32 over 8-bit ones; a split from the GPU's SMs and the KV heads
    # (CUDA_REACH positions over six programs an SM)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    assert pk.cuda_block(256, 256, 24, 2, 16, "bf16") == 16 and pk.cuda_block(128, 128, 56, 8, 16, "int8") == 32
    for hkv, kind, block in ((2, "bf16", 16), (8, "int8", 32), (1, "fp8", 32)):
        want = min(2048, max(128, round(pk.CUDA_REACH * hkv ** 0.5 / (6 * sms) / 128) * 128)) // block * block
        assert pk.cuda_chunk(128, 128, 8 * hkv, hkv, 16, kind) == want
    for kind in ("bf16", "int8"):
        cache = Cache(kind, 4, 128, 128, 16, [700, 1500])
        q = _q(16, 24, 128, 13).to(torch.bfloat16)
        args = (q, cache.k, cache.v, kind, cache.tables, torch.tensor([0, 1], dtype=torch.int32, device=dev),
                torch.tensor([0, 8], dtype=torch.int32, device=dev),
                torch.tensor([690, 1490], dtype=torch.int64, device=dev), 8, 16)
        kw = dict(key_scales=cache.ks, value_scales=cache.vs, **CUDA)
        want = pk.attend(*args, **kw)
        assert torch.equal(speed.attend(*args, **kw), want)
        assert torch.equal(speed.attend(*args, fused=True, **kw), want)
        assert torch.equal(shared.attend(*args, **kw), want)
        assert torch.equal(shared.attend(*args, fused=True, **kw), want)
        got = order.attend(*args, **kw)
        assert torch.equal(order.attend(*args, fused=True, **kw), got)
        assert not torch.equal(got, want)


@pytest.mark.kernels_ci
def test_a_sweep_build_of_several_window_configurations_launches_each_as_a_build_of_it_alone():
    # A sweep build holds the window table's diagonal only (select() gives the call's rows and one row the same
    # configuration): selecting configuration i plans and launches what a build of i alone does, bit for bit, the
    # order knobs (CHUNK, BLOCK_N) included.
    configs = [{"STAGES": 3, "WARPS": 2}, {"CHUNK": 256, "BLOCK_N": 64}, {"VWARPS": 1, "CONVERT": 1, "STAGES": 3}]
    swept = load("paged-kv-attention", configs={"cuda-window": configs})
    for kind in ("bf16", "int8"):
        cache = Cache(kind, 4, 128, 128, 16, [700, 1500])
        q = _q(16, 24, 128, 13).to(torch.bfloat16)
        args = (q, cache.k, cache.v, kind, cache.tables, torch.tensor([0, 1], dtype=torch.int32, device=dev),
                torch.tensor([0, 8], dtype=torch.int32, device=dev),
                torch.tensor([690, 1490], dtype=torch.int64, device=dev), 8, 16)
        kw = dict(key_scales=cache.ks, value_scales=cache.vs, **CUDA)
        for i, knobs in enumerate(configs, 1):
            select(swept, "cuda-window", i)
            alone = _tuned({"cuda-window": knobs})
            for fn in ("cuda_block", "cuda_chunk", "cuda_ramp"):
                assert getattr(swept, fn)(128, 128, 24, 4, 16, kind) == getattr(alone, fn)(128, 128, 24, 4, 16, kind)
            assert torch.equal(swept.attend(*args, **kw), alone.attend(*args, **kw)), (kind, knobs)
    select(swept, "cuda-window", -1)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8", "int8"])
@pytest.mark.parametrize("hq,hkv,d,dv", [(24, 2, 256, 256), (56, 8, 128, 128), (16, 2, 64, 64)])
def test_long_cuda_rows_merge_in_split_order_staged_or_fused(kind, hq, hkv, d, dv, monkeypatch):
    # Rows of many splits (the staged merge's batches of splits, its programs fewer or more than the GPU's SMs): the
    # window launch, the fused launch as the window launch and its merge, and the fused launch through its ring give
    # every column the bits it has decoded alone.
    cache = Cache(kind, hkv, d, dv, 16, [20000, 9000, 33], seed=5)
    for cols in (1, 3, 16):
        q = _q(3 * cols, hq, d, 7)
        firsts, starts = [20000 - cols, 8000, 33 - cols], [0, cols, 2 * cols]
        for chunk in (128, None):
            lse = torch.empty(3 * cols, hq, device=dev)
            alone, alone_lse = [], []
            for r in range(3):
                for c in range(cols):
                    one = torch.empty(1, hq, device=dev)
                    alone.append(cache.attend(q[r * cols + c:r * cols + c + 1], [r], [0], [firsts[r] + c], 1,
                                              lse=one, chunk=chunk, **CUDA))
                    alone_lse.append(one)
            alone, alone_lse = torch.cat(alone), torch.cat(alone_lse)
            for fused, partials in ((False, None), (True, None), (True, 0)):
                if partials is not None:
                    monkeypatch.setattr(pk, "FUSED_PARTIALS", partials)
                got = cache.attend(q, [0, 1, 2], starts, firsts, cols, lse=lse, chunk=chunk, fused=fused, **CUDA)
                assert torch.equal(got, alone) and torch.equal(lse, alone_lse), (cols, chunk, fused, partials)
            monkeypatch.undo()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8", "int8"])
@pytest.mark.parametrize("ramp", [(1, 16), (2, 16), (3, 8)])
def test_a_cuda_ramp_splits_short_positions_finer_and_keeps_a_columns_bits_alone(kind, ramp, monkeypatch):
    # A ramp (an order knob): a row's first positions in bands of smaller splits. Every column has the bits of the same
    # column decoded alone in windows of every width, fused (as the window launch or through its ring) or split, and
    # matches float64 attention; no ramp (RAMP 0) reads the same columns in other splits.
    hq, hkv, d = 24, 2, 256
    chunk = 512
    for length in (40, 2100, 9000):
        cache = Cache(kind, hkv, d, d, 16, [length, length], seed=3)
        T = min(48, length)
        pos0 = length - T
        q = _q(T, hq, d, 4)
        kw = dict(chunk=chunk, ramp=ramp, **CUDA)
        one_lse = torch.empty(T, hq, device=dev)
        alone = cache.attend(q, [0] * T, list(range(T)), [pos0 + i for i in range(T)], 1, lse=one_lse, **kw)
        keys, values = cache.rows(0, length)
        qd = q.to(torch.float16 if kind == "fp16" else torch.bfloat16).double()
        for t in (0, T - 1):
            seen = torch.arange(length, device=dev) <= pos0 + t
            want, _ = reference(qd[t], keys, values, seen, d ** -0.5)
            torch.testing.assert_close(alone[t].double(), want, atol=8e-3, rtol=2e-2)
        for cols in (5, 16):
            starts = list(range(0, T, cols))
            counts = torch.tensor([min(cols, T - s) for s in starts], dtype=torch.int32, device=dev)
            for fused, partials in ((False, None), (True, None), (True, 0)):
                if partials is not None:
                    monkeypatch.setattr(pk, "FUSED_PARTIALS", partials)
                lse = torch.empty(T, hq, device=dev)
                got = cache.attend(q, [0] * len(starts), starts, [pos0 + s for s in starts], cols, counts=counts,
                                   lse=lse, fused=fused, **kw)
                assert torch.equal(got, alone) and torch.equal(lse, one_lse), (length, cols, fused, partials)
                monkeypatch.undo()
        if length > chunk:
            plain = cache.attend(q, [0] * T, list(range(T)), [pos0 + i for i in range(T)], 1, chunk=chunk,
                                 ramp=(0, pk.CUDA_RAMP_SPLITS), **CUDA)
            assert not torch.equal(plain, alone)
    assert pk.cuda_splits(4096, 384, (3, 16)) > pk.cuda_splits(4096, 384, (0, 16))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fused", [False, True])
def test_each_owner_launches_in_workspaces_of_its_own(fused, monkeypatch):
    # Two engines of a process must never share a launch's partials, ring or sync words: launches inside owner(key)
    # take the module's workspaces of that key, the same bits as outside, and release(key) drops them.
    if fused:
        monkeypatch.setattr(pk, "FUSED_PARTIALS", 0)
        monkeypatch.setattr(pk, "FUSED_SERIAL", RING)
    monkeypatch.setattr(pk, "_workspaces", {})
    monkeypatch.setattr(pk, "_fused_workspaces", {})
    cache = Cache("bf16", 2, 128, 128, 16, [700, 700])
    q = _q(8, 8, 128, 3).to(torch.bfloat16)
    rows = torch.arange(2, dtype=torch.int32, device=dev)
    starts, firsts = rows * 4, torch.full((2,), 640, dtype=torch.int64, device=dev)

    def launch():
        return pk.attend(q, cache.k, cache.v, "bf16", cache.tables, rows, starts, firsts, 4, 16, fused=fused, **CUDA)
    held = pk._fused_workspaces if fused else pk._workspaces
    want = launch()
    with pk.owner("a"):
        a = launch()
    with pk.owner("b"):
        b = launch()
    assert torch.equal(a, want) and torch.equal(b, want)
    assert sorted(str(key[1]) for key in held) == ["None", "a", "b"]
    assert len({held[key][0].data_ptr() for key in held}) == 3   # three workspaces, none shared
    pk.release("a")
    assert sorted(str(key[1]) for key in held) == ["None", "b"]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fused", [False, True])
def test_a_graph_captured_before_the_workspace_grows_still_replays_its_own_bits(fused, monkeypatch):
    # The module's workspace grows for a larger launch after a capture: the captured launch keeps the one it read,
    # whose blocks no other tensor may take (a replay would read them, and write over that tensor).
    if fused:   # every fused launch fused (not the window launch and its merge)
        monkeypatch.setattr(pk, "FUSED_PARTIALS", 0)
    monkeypatch.setattr(pk, "_workspaces", {})   # the module's workspaces as before any launch
    monkeypatch.setattr(pk, "_fused_workspaces", {})
    cache = Cache("bf16", 4, 256, 256, 16, [3100, 3100, 3100, 3100])

    def launcher(columns, rows):
        q = _q(columns * rows, 24, 256, 7).to(torch.bfloat16)
        out = torch.empty(columns * rows, 24, 256, dtype=torch.bfloat16, device=dev)
        args = dict(rows=torch.arange(rows, dtype=torch.int32, device=dev),
                    starts=torch.arange(rows, dtype=torch.int32, device=dev) * columns,
                    firsts=torch.full((rows,), 3000, dtype=torch.int64, device=dev))
        counts = torch.full((rows,), columns, dtype=torch.int32, device=dev)
        return out, lambda: pk.attend(q, cache.k, cache.v, "bf16", cache.tables, args["rows"], args["starts"],
                                      args["firsts"], columns, 16, counts=counts, out=out, fused=fused, **CUDA)
    out, small = launcher(8, 1)
    small()
    want = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        small()
    held = pk._fused_workspaces if fused else pk._workspaces
    before = [t.numel() for t in held[next(iter(held))]]
    _, large = launcher(64, 4)
    large()
    assert [t.numel() for t in held[next(iter(held))]] != before   # it grew
    torch.cuda.synchronize()
    junk = [torch.full((n,), float("nan"), device=dev) for n in before]   # the sizes a freed workspace would give back
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)
    assert all(torch.isnan(t).all() for t in junk)   # and other tensors keep their values


# -- the fused launch's serial form (one program a tile, walking its splits in turn) --------------------------------------
@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8:row", "int8", "fp8", "fp16"])
@pytest.mark.parametrize("hq,hkv,d,dv", SHAPES)
def test_a_serial_cuda_launch_has_the_split_launchs_bits(kind, hq, hkv, d, dv, monkeypatch):
    # One program a tile, walking its row's splits in turn and folding each into the tile's running state (in registers
    # up to 128 value columns a program, in the workspace past them): every column has the split launch's bits and
    # log-sum-exps, in rows of one split and of many, in tiles whose rows end in different splits, in windows of 1 to
    # 64 columns of other counts, under a ramp and with a value split.
    monkeypatch.setattr(pk, "FUSED_SERIAL", SERIAL)
    seen = _launched(monkeypatch)
    for length, chunk, ramp, vsplit in ((40, None, None, None), (2100, None, None, 2), (5000, 128, None, None),
                                        (3000, 256, (2, 4), None)):
        cache = Cache(kind, hkv, d, dv, 16, [length, max(3, length // 3)], seed=2)
        kw = dict(chunk=chunk, **CUDA)
        if ramp:
            kw["ramp"] = ramp
        T = min(64, length // 3)
        q = _q(2 * T, hq, d, 9)
        for cols in (1, 5, 16, 64):
            cols = min(cols, T)
            rows, starts, firsts, counts = [], [], [], []
            for s, n in enumerate(cache.lengths):
                for at in range(0, T, cols):
                    rows.append(s), starts.append(s * T + at), firsts.append(n - T + at)
                    counts.append(min(cols, T - at))
            counts = torch.tensor(counts, dtype=torch.int32, device=dev)
            want_lse, lse = torch.empty(2 * T, hq, device=dev), torch.empty(2 * T, hq, device=dev)
            want = cache.attend(q, rows, starts, firsts, cols, counts=counts, lse=want_lse, **kw)
            del seen[:]
            got = cache.attend(q, rows, starts, firsts, cols, counts=counts, lse=lse, fused=True, vsplit=vsplit, **kw)
            assert seen == [_SERIAL_STRUCTURE], (length, cols)
            assert torch.equal(got, want) and torch.equal(lse, want_lse), (kind, length, chunk, ramp, cols)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "int8"])
@pytest.mark.parametrize("hq,hkv,d", [(24, 2, 256), (56, 8, 128)])
def test_a_serial_cuda_launch_keeps_sinks_and_sliding_windows(kind, hq, hkv, d, monkeypatch):
    # Sinks fold into a row's final state after its splits, and a sliding window skips the splits and blocks below it:
    # the serial form keeps the split launch's bits with both, in windows narrower than a block and wider than splits.
    monkeypatch.setattr(pk, "FUSED_SERIAL", SERIAL)
    seen = _launched(monkeypatch)
    cache = Cache(kind, hkv, d, d, 16, [2600, 900], seed=6)
    T = 64
    q = _q(2 * T, hq, d, 10)
    sinks = torch.randn(hq, generator=torch.Generator().manual_seed(3)).to(dev)
    rows, starts, firsts = [0, 0, 1], [0, 40, T], [2600 - T, 2600 - T + 40, 900 - T]
    counts = torch.tensor([40, 24, 64], dtype=torch.int32, device=dev)
    for sliding in (0, 3, 100, 700, 5000):
        for given in (None, sinks):
            kw = dict(counts=counts, chunk=128, sinks=given, sliding=sliding, **CUDA)
            want_lse, lse = torch.empty(2 * T, hq, device=dev), torch.empty(2 * T, hq, device=dev)
            want = cache.attend(q, rows, starts, firsts, 64, lse=want_lse, **kw)
            del seen[:]
            got = cache.attend(q, rows, starts, firsts, 64, lse=lse, fused=True, **kw)
            assert seen == [_SERIAL_STRUCTURE]
            assert torch.equal(got, want) and torch.equal(lse, want_lse), (kind, sliding, given is not None)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "bf16"])
def test_a_serial_cuda_chunk_reads_its_own_keys_and_values_fresh_and_replays(kind, monkeypatch):
    # A captured chunk's pieces through the serial form: one graph replayed at other positions and first fresh
    # positions gives the bits of the ring's programs over the same rows (fresh rows from `first` on, the pages below).
    cache = Cache(kind, 2, 256, 256, 16, [1500])
    g = torch.Generator().manual_seed(21)
    T = 100
    q = _q(T, 24, 256, 22).to(torch.bfloat16)
    fk = torch.randn(T, 2, 256, generator=g).to(dev).to(torch.bfloat16)
    fv = torch.randn(T, 2, 256, generator=g).to(dev).to(torch.bfloat16)
    frontier = torch.zeros(1, dtype=torch.int64, device=dev)
    first = torch.zeros(1, dtype=torch.int64, device=dev)
    starts = torch.tensor([0, 64], dtype=torch.int32, device=dev)
    offsets = torch.tensor([0, 64], dtype=torch.int64, device=dev)
    counts = torch.tensor([64, T - 64], dtype=torch.int32, device=dev)
    rows = torch.zeros(2, dtype=torch.int32, device=dev)
    firsts = torch.empty(2, dtype=torch.int64, device=dev)

    def launcher(out, lse):
        def launch():
            torch.add(offsets, frontier, out=firsts)
            pk.attend(q, cache.k, cache.v, kind, cache.tables, rows, starts, firsts, 64, 16, counts=counts,
                      key_scales=cache.ks, value_scales=cache.vs, out=out, lse=lse, fused=True, chunk=128,
                      fresh=(fk, fv, frontier, first), **CUDA)
        return launch
    out, lse = torch.empty_like(q), torch.empty(T, 24, device=dev)
    want, want_lse = torch.empty_like(q), torch.empty(T, 24, device=dev)
    monkeypatch.setattr(pk, "FUSED_PARTIALS", 0)
    monkeypatch.setattr(pk, "FUSED_SERIAL", SERIAL)
    seen = _launched(monkeypatch)
    serial = launcher(out, lse)
    serial()
    assert seen == [_SERIAL_STRUCTURE]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        serial()
    monkeypatch.setattr(pk, "FUSED_SERIAL", RING)
    ring = launcher(want, want_lse)
    for at in (0, 200, 1300):
        frontier.fill_(at)
        for split in (0, 37, 100):
            first.fill_(at + split)
            out.zero_()
            graph.replay()
            ring()
            assert torch.equal(out, want) and torch.equal(lse, want_lse), (kind, at, split)


@pytest.mark.kernels_ci
def test_a_serial_cuda_row_keeps_its_bits_beside_a_later_tokens_nan(monkeypatch):
    # A tile's rows end at different positions: a later token's value (NaN here) lies in a split an earlier row's merge
    # never folds, so that row keeps the bits it has decoded alone, as the split launch gives it.
    monkeypatch.setattr(pk, "FUSED_SERIAL", SERIAL)
    seen = _launched(monkeypatch)
    cache = Cache("bf16", 2, 256, 256, 16, [300])
    q = _q(8, 24, 256, 4)
    first = 124   # columns 0..3 end in the split of positions 0..127, columns 4..7 in the next one
    cache.v.view(-1, 2, 256)[cache.slots(0, [130])] = float("nan")
    want = cache.attend(q, [0], [0], [first], 8, chunk=128, **CUDA)
    del seen[:]
    got = cache.attend(q, [0], [0], [first], 8, chunk=128, fused=True, **CUDA)
    assert seen == [_SERIAL_STRUCTURE]
    assert bool(torch.isfinite(want[:4].float()).all()) and bool(torch.isnan(want[6:].float()).any())
    assert torch.equal(got[:4], want[:4])
    assert torch.equal(torch.isnan(got.float()), torch.isnan(want.float()))


@pytest.mark.kernels_ci
def test_the_serial_knob_chooses_the_form_by_the_programs_the_gpu_holds_and_keeps_every_bit(monkeypatch):
    # cuda-fused's SERIAL: the launch's tiles as a percent of the programs the device holds at once from which each
    # tile is one program (0: FUSED_SERIAL; 1: every launch; a large value: none). A speed knob: the same bits.
    seen = _launched(monkeypatch)
    cache = Cache("bf16", 2, 256, 256, 16, [1200, 1200, 1200, 1200])
    hq = 24

    def launch(module, pieces):   # `pieces` rows of 64 columns: 12 tiles of 64 query rows each, over 2 KV heads
        q = _q(64 * pieces, hq, 256, 5).to(torch.bfloat16)
        rows = torch.arange(pieces, dtype=torch.int32, device=dev) % 4
        starts = torch.arange(pieces, dtype=torch.int32, device=dev) * 64
        firsts = torch.full((pieces,), 1100, dtype=torch.int64, device=dev)
        del seen[:]
        got = module.attend(q, cache.k, cache.v, "bf16", cache.tables, rows, starts, firsts, 64, 16, chunk=256,
                            fused=True, **CUDA)
        want = module.attend(q, cache.k, cache.v, "bf16", cache.tables, rows, starts, firsts, 64, 16, chunk=256,
                             **CUDA)
        assert torch.equal(got, want)
        return seen[0]
    lib = pk._cuda_lib(dev, T=0, KIND=0, D=256, DV=256, W=0, S=0, VS=0)
    held = pk._cuda_resident(lib, _SERIAL_STRUCTURE, pk._cuda_plan(lib, _SERIAL_STRUCTURE, hq, 2, 16, 64)[0], dev)
    assert held >= torch.cuda.get_device_properties(dev).multi_processor_count   # a tile an SM at least
    fills = -(-held * pk.FUSED_SERIAL // (100 * 24))   # pieces whose tiles (12 x 2 KV heads) reach the default share
    assert launch(pk, fills) == _SERIAL_STRUCTURE
    if fills > 1:
        assert launch(pk, fills - 1) != _SERIAL_STRUCTURE
    always = _tuned({"cuda-fused": {"SERIAL": 1}})
    never = _tuned({"cuda-fused": {"SERIAL": 100000}})
    for module in (always, never):
        monkeypatch.setattr(module, "_cuda_launch", pk._cuda_launch)
    assert launch(always, 1) == _SERIAL_STRUCTURE
    assert launch(never, fills + 4) != _SERIAL_STRUCTURE
