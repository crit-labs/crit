"""Against torch's encoding rules and a float64 reference: page writes, decode and verify windows (row and column
invariance, trees, block lists, log-sum-exp), CUDA graph replay at other lengths, prompt chunks, and the knobs: every
warps choice a tuning can make lowers to mma.sync on every GPU, speed knobs keep every bit, the split knobs reach every
launch."""
import math

import pytest
import torch

from crit_kernels.build import load, select

pk = load("paged-kv-attention")
dev = torch.device("cuda")
STORE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.float8_e4m3fn, "fp8:row": torch.float8_e4m3fn,
         "int8": torch.int8}
KINDS = list(STORE)


class Cache:
    """One layer's pages for several sequences, each over shuffled pages, written by pk.write."""

    def __init__(self, kind, hkv, d, dv, page, lengths, seed=0, multipliers=(0.02, 0.03)):
        g = torch.Generator().manual_seed(seed)
        self.kind, self.hkv, self.d, self.dv, self.page = kind, hkv, d, dv, page
        self.multipliers = multipliers if kind == "fp8" else (1.0, 1.0)
        width = max(math.ceil(n / page) for n in lengths) + 1
        n_pages = width * len(lengths) + 3
        self.k = torch.zeros(n_pages, page, hkv, d, dtype=STORE[kind], device=dev)
        self.v = torch.zeros(n_pages, page, hkv, dv, dtype=STORE[kind], device=dev)
        self.ks = self.vs = None
        if kind in ("fp8:row", "int8"):
            self.ks = torch.zeros(n_pages, page, hkv, dtype=torch.float32, device=dev)
            self.vs = torch.zeros_like(self.ks)
        order = torch.randperm(n_pages, generator=g)
        self.tables = order[:width * len(lengths)].view(len(lengths), width).to(torch.int32).to(dev)
        self.lengths = list(lengths)
        for s, n in enumerate(lengths):
            keys = torch.randn(n, hkv, d, generator=g).to(dev)
            values = torch.randn(n, hkv, dv, generator=g).to(dev)
            self.write(s, 0, keys, values)

    def slots(self, s, positions):
        positions = torch.as_tensor(positions, device=dev)
        return self.tables[s][positions // self.page].long() * self.page + positions % self.page

    def write(self, s, start, keys, values):
        slots = self.slots(s, torch.arange(start, start + keys.shape[0], device=dev))
        pk.write(keys, values, slots, self.k, self.v, self.kind, self.ks, self.vs, multipliers=self.multipliers)

    def rows(self, s, n):
        """The stored keys and values of positions 0..n-1 of sequence s, decoded to float64."""
        slots = self.slots(s, torch.arange(n, device=dev))
        k = self.k.view(-1, self.hkv, self.d)[slots].double()
        v = self.v.view(-1, self.hkv, self.dv)[slots].double()
        if self.kind == "fp8":
            k, v = k * self.multipliers[0], v * self.multipliers[1]
        if self.ks is not None:
            k = k * self.ks.view(-1, self.hkv)[slots].double()[..., None]
            v = v * self.vs.view(-1, self.hkv)[slots].double()[..., None]
        return k, v

    def attend(self, q, rows, starts, firsts, columns, **kw):
        q_dtype = torch.float16 if self.kind == "fp16" else torch.bfloat16
        return pk.attend(q.to(q_dtype), self.k, self.v, self.kind, self.tables,
                         torch.tensor(rows, dtype=torch.int32, device=dev),
                         torch.tensor(starts, dtype=torch.int32, device=dev),
                         torch.tensor(firsts, dtype=torch.int64, device=dev), columns, self.page,
                         key_scales=self.ks, value_scales=self.vs, multipliers=self.multipliers, **kw)


def reference(q, keys, values, seen, scale):
    """q [hq][d] float64 over keys [n][hkv][d] where seen [n] -> (out [hq][dv], lse [hq])."""
    hq, hkv = q.shape[0], keys.shape[1]
    out, lse = [], []
    for h in range(hq):
        k, v = keys[:, h // (hq // hkv)], values[:, h // (hq // hkv)]
        s = (k @ q[h]) * scale
        s = torch.where(seen, s, torch.tensor(float("-inf"), dtype=torch.float64, device=dev))
        lse.append(torch.logsumexp(s, 0))
        out.append(torch.softmax(s, 0) @ v)
    return torch.stack(out), torch.stack(lse)


def _q(tokens, hq, d, seed):
    return torch.randn(tokens, hq, d, generator=torch.Generator().manual_seed(seed)).to(dev)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("d,dv", [(128, 128), (64, 128)])
def test_writes_encode_as_torch_does_and_skip_negative_slots(kind, d, dv):
    from torch import float8_e4m3fn as e4m3
    g = torch.Generator().manual_seed(1)
    keys, values = torch.randn(40, 3, d, generator=g).to(dev) * 3, torch.randn(40, 3, dv, generator=g).to(dev)
    keys[5] = 0.0
    k = torch.zeros(64, 3, d, dtype=STORE[kind], device=dev)
    v = torch.zeros(64, 3, dv, dtype=STORE[kind], device=dev)
    ks = vs = None
    if kind in ("fp8:row", "int8"):
        ks, vs = torch.zeros(64, 3, device=dev), torch.zeros(64, 3, device=dev)
    slots = torch.randperm(64, device=dev)[:40]
    slots[7] = -1
    mult = (0.05, 0.07)
    pk.write(keys, values, slots, k, v, kind, ks, vs, multipliers=mult)
    keep = slots >= 0
    for x, store, scales, m in ((keys, k, ks, mult[0]), (values, v, vs, mult[1])):
        x = x[keep].float()
        if kind in ("bf16", "fp16"):
            want, want_scale = x.to(STORE[kind]), None
        elif kind == "fp8":
            want, want_scale = (x / torch.full_like(x, m)).clamp(-448, 448).to(e4m3), None
        else:
            # IEEE divisions (torch divides by a Python number with a reciprocal multiply on CUDA)
            top = 448.0 if kind == "fp8:row" else 127.0
            amax = x.abs().amax(-1)
            want_scale = (amax / torch.full_like(amax, top)).clamp(min=1e-30)
            y = (x / want_scale[..., None]).clamp(-top, top)
            want = y.to(e4m3) if kind == "fp8:row" else torch.round(y).to(torch.int8)
        got = store[slots[keep]]
        assert torch.equal(got.view(torch.uint8) if got.dtype == e4m3 else got,
                           want.view(torch.uint8) if want.dtype == e4m3 else want)
        if want_scale is not None:
            assert torch.equal(scales[slots[keep]], want_scale)
        untouched = torch.ones(64, dtype=torch.bool, device=dev)
        untouched[slots[keep]] = False
        assert not bool(store[untouched].float().abs().sum())   # the skipped token went nowhere


SHAPES = [(8, 2, 128, 128), (24, 4, 256, 256), (56, 8, 128, 128), (12, 2, 64, 128)]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("hq,hkv,d,dv", SHAPES)
@pytest.mark.parametrize("page", [16, 48])
def test_windows_match_float64_attention(kind, hq, hkv, d, dv, page):
    cache = Cache(kind, hkv, d, dv, page, [130, 64, 1, 517])
    cols = 5
    rows, starts, firsts = [0, 1, 2, 3], [0, 5, 10, 15], [125, 59, 0, 400]   # every window inside its written keys
    q = _q(20, hq, d, 3)
    lse = torch.empty(20, hq, device=dev)
    out = cache.attend(q, rows, starts, firsts, cols, lse=lse,
                       counts=torch.tensor([5, 5, 1, 3], dtype=torch.int32, device=dev))
    scale = d ** -0.5
    qd = q.to(torch.bfloat16 if kind != "fp16" else torch.float16).double()
    for r, (s, start, first, count) in enumerate(zip(rows, starts, firsts, [5, 5, 1, 3])):
        keys, values = cache.rows(s, first + count)
        for c in range(count):
            seen = torch.arange(first + count, device=dev) <= first + c
            want, want_lse = reference(qd[start + c], keys, values, seen, scale)
            torch.testing.assert_close(out[start + c].double(), want, atol=8e-3, rtol=2e-2)
            torch.testing.assert_close(lse[start + c].double(), want_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8:row", "int8"])
def test_rows_and_columns_are_invariant_bit_for_bit(kind):
    cache = Cache(kind, 4, 256, 256, 16, [700, 300, 2100])
    q = _q(24, 24, 256, 5)
    together = cache.attend(q, [0, 1, 2], [0, 8, 16], [690, 290, 2090], 8)
    for r, (s, start, first) in enumerate(zip([0, 1, 2], [0, 8, 16], [690, 290, 2090])):
        alone = cache.attend(q[start:start + 8], [s], [0], [first], 8)
        assert torch.equal(alone, together[start:start + 8])
        for c in range(8):    # each column decoded alone, in a launch of one column
            one = cache.attend(q[start + c:start + c + 1], [s], [0], [first + c], 1)
            assert torch.equal(one[0], together[start + c])
        # ... and inside a wider launch whose rows are shorter than its columns
        wide = cache.attend(q[start:start + 8], [s], [0], [first], 16,
                            counts=torch.tensor([8], dtype=torch.int32, device=dev))
        assert torch.equal(wide, together[start:start + 8])
        # ... and its partial rows packed by the rows' tokens (offsets), in a workspace of those tokens only
        rows3 = [0, 1, 2]
        counts = torch.tensor([3, 1, 4], dtype=torch.int32, device=dev)
        offsets = torch.tensor([0, 3, 4], dtype=torch.int32, device=dev)
        floats = pk.workspace_floats(8, 24, pk.partitions(dev, 4, kind), 256)
        packed = cache.attend(q, rows3, [0, 8, 16], [690, 290, 2090], 8, counts=counts, offsets=offsets, tokens=8,
                              workspace=torch.empty(floats, dtype=torch.float32, device=dev))
        for start, count in zip([0, 8, 16], [3, 1, 4]):
            assert torch.equal(packed[start:start + count], together[start:start + count])


@pytest.mark.kernels_ci
def test_a_count_past_the_columns_reads_as_the_columns():
    cache = Cache("bf16", 2, 128, 128, 16, [300, 300])
    q = _q(8, 8, 128, 11)
    want = cache.attend(q, [0, 1], [0, 4], [290, 280], 4)
    got = cache.attend(q, [0, 1], [0, 4], [290, 280], 4, counts=torch.tensor([9, 4], dtype=torch.int32, device=dev))
    assert torch.equal(got, want)    # row 0's partials stay its own (no row writes past its columns)


def _lowering(arch, group, d, kind, columns, fused=False, knob=0):
    """The MMA instructions Triton gives the window launch tile_for() picks for `columns` columns on `arch` (compiled
    offline, no GPU of that kind needed), its warps a tuning's `knob` where attend takes it: from the TTGIR, wgmma
    (warp_group_dot), tcgen05 (tc_gen5_mma) or mma.sync (nvidia_mma v2)."""
    import triton
    from triton.backends.compiler import GPUTarget
    rows, warps, stages = pk.tile_for(columns, group, (arch // 10, arch % 10), fused)
    warps = pk._warps(knob, warps, rows, (arch // 10, arch % 10))
    kernel = pk._kernels._window_kernel
    store = {0: "*bf16", 1: "*u8", 2: "*u8", 3: "*i8"}[kind]
    signature = {"q_ptr": "*bf16", "k_ptr": store, "v_ptr": store, "ks_ptr": "*fp32", "vs_ptr": "*fp32",
                 "tab_ptr": "*i32", "trow_ptr": "*i32", "start_ptr": "*i32", "count_ptr": "*i32", "first_ptr": "*i64",
                 "par_ptr": "*i64", "off_ptr": "*i32", "po_ptr": "*fp32", "pm_ptr": "*fp32", "pl_ptr": "*fp32",
                 "out_ptr": "*bf16", "lse_ptr": "*fp32", "fk_ptr": "*bf16", "fv_ptr": "*bf16", "fshift_ptr": "*i64",
                 "ffirst_ptr": "*i64", "stride_qt": "i32", "stride_qh": "i32", "stride_ot": "i32", "stride_oh": "i32",
                 "stride_fk": "i32", "stride_fv": "i32", "stride_fr": "i32", "tab_stride": "i32", "tab_width": "i32",
                 "n_parts": "i32", "sm_scale": "fp32", "k_mult": "fp32", "v_mult": "fp32", "count_all": "i32",
                 "sink_ptr": "*fp32", "win": "i32"}
    constants = {"COLS": 8, "CPT": rows // group, "HQ": group * 2, "HKV": 2, "G": group, "D": d, "DV": d,
                 "BLOCK_M": rows, "PIECE": pk.piece_size(d, d, ["bf16", "fp8", "fp8:row", "int8"][kind]), "PAGE": 16,
                 "KIND": kind, "HW_FP8": arch >= 89, "TREE": False, "COUNTS": True, "OFFSETS": True, "DC": min(64, d),
                 "FUSED": fused, "LSE": fused, "FRESH": fused and kind >= 1,
                 "DVP": d // pk.FUSED_VSPLIT if fused and d > 128 else d, "SINKS": False}
    names = kernel.arg_names
    full = {n: signature.get(n, "constexpr") for n in names}
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if full[n].startswith("*")}
    source = triton.compiler.ASTSource(fn=kernel, signature=full, constexprs=constants, attrs=attrs)
    ir = triton.compile(source, target=GPUTarget("cuda", arch, 32),
                        options={"num_warps": warps, "num_stages": stages}).asm["ttgir"]
    return {name for name, key in (("wgmma", "warp_group_dot"), ("tcgen05", "tc_gen5_mma"),
                                   ("mma.sync", "versionMajor = 2")) if key in ir}


@pytest.mark.parametrize("arch", [80, 90, 100, 110, 120])
@pytest.mark.parametrize("group,d,kind,columns", [(6, 256, 2, 1), (6, 256, 2, 5), (6, 256, 2, 8), (12, 256, 0, 1),
                                                  (12, 256, 0, 5), (7, 128, 3, 5), (64, 128, 0, 1)])
def test_every_window_tile_multiplies_with_mma_sync_on_every_gpu(arch, group, d, kind, columns):
    # Four warps over 64 rows lower to wgmma on sm_90 and tcgen05 on sm_100 and sm_110, whose summation order differs
    # from mma.sync's: there tile_for() keeps windows in 32-row two-warp tiles, elsewhere it takes 64 rows in four
    # warps, which those GPUs multiply with mma.sync: on each GPU every tile has one instruction, so a column's bits are
    # the same in any tile.
    assert _lowering(arch, group, d, kind, columns) == {"mma.sync"}


@pytest.mark.parametrize("arch", [80, 90, 100, 110, 120])
@pytest.mark.parametrize("group,d,kind", [(6, 256, 2), (6, 256, 3), (7, 128, 3), (12, 256, 0), (6, 256, 1)])
def test_a_fused_launch_multiplies_with_mma_sync_on_every_gpu(arch, group, d, kind):
    # A fused launch's tile (up to 32 rows in four warps, fresh rows included) is mma.sync on every GPU too.
    assert _lowering(arch, group, d, kind, 64, fused=True) == {"mma.sync"}


def test_a_wide_window_takes_a_64_row_tile_only_where_it_stays_mma_sync():
    assert pk.tile_for(5, 12, (12, 0)) == (64, 4, 4)
    assert pk.tile_for(5, 12, (9, 0)) == (32, 2, 3)
    assert pk.tile_for(5, 12, (10, 0)) == (32, 2, 3)
    assert pk.tile_for(1, 6, (12, 0)) == (16, 2, 3)
    assert pk.tile_for(5, 6, (12, 0)) == (32, 2, 3)
    assert pk.tile_for(64, 6, (12, 0), fused=True) == (64, 4, 3)     # a fused launch: two output sets, four warps
    assert pk.tile_for(64, 6, (9, 0), fused=True) == (32, 4, 3)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8:row", "int8", "fp8"])
@pytest.mark.parametrize("hq,hkv,d", [(24, 2, 256), (24, 4, 256), (56, 8, 128)])
def test_every_tile_gives_a_column_the_one_column_decodes_bits(kind, hq, hkv, d):
    # Every tile tile_for() picks (16 or 32 rows in two warps, 64 rows in four on this GPU) gives each column the bits
    # of the column decoded alone, in windows of every width, trees and log-sum-exps included.
    cache = Cache(kind, hkv, d, d, 16, [700, 1500])
    q = _q(16, hq, d, 13)
    lse = torch.empty(16, hq, device=dev)
    for cols in (2, 5, 8):
        window = cache.attend(q, [0, 1], [0, 8], [690, 1490], cols, lse=lse)
        for r, (s, start, first) in enumerate(zip([0, 1], [0, 8], [690, 1490])):
            for c in range(cols):
                one_lse = torch.empty(1, hq, device=dev)
                one = cache.attend(q[start + c:start + c + 1], [s], [0], [first + c], 1, lse=one_lse)
                assert torch.equal(one[0], window[start + c]), (cols, r, c)
                assert torch.equal(one_lse[0], lse[start + c]), (cols, r, c)
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], dtype=torch.int64, device=dev)
    tree = cache.attend(q[:8], [0], [0], [690], 8, parents=parents)
    alone = cache.attend(q[:8], [0], [0], [690], 8, parents=parents, tile=16, num_warps=2, num_stages=3)
    assert torch.equal(tree, alone)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["int8", "fp8:row", "fp8", "bf16"])
@pytest.mark.parametrize("hq,hkv,d", [(24, 4, 256), (56, 8, 128)])
def test_a_fused_launch_gives_a_column_the_split_launchs_bits(kind, hq, hkv, d):
    # Prompt rows as window rows of 64 columns, fused (each program folding its partitions as the merge does): every
    # column has the bits of the same column decoded alone through the split launch and merge, its log-sum-exp too,
    # at lengths of one piece, of more pieces than partitions and between; a tree window too.
    for length in (40, 700, 5000):
        cache = Cache(kind, hkv, d, d, 64, [length], seed=1)
        T = min(130, length)
        pos0 = length - T
        q = _q(T, hq, d, 2)
        starts = list(range(0, T, 64))
        counts = torch.tensor([min(64, T - s) for s in starts], dtype=torch.int32, device=dev)
        lse, one_lse = torch.empty(T, hq, device=dev), torch.empty(T, hq, device=dev)
        fused = cache.attend(q, [0] * len(starts), starts, [pos0 + s for s in starts], 64, counts=counts, fused=True,
                             lse=lse)
        alone = cache.attend(q, [0] * T, list(range(T)), [pos0 + i for i in range(T)], 1, lse=one_lse)
        assert torch.equal(fused, alone), (kind, length)
        assert torch.equal(lse, one_lse), (kind, length)
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], dtype=torch.int64, device=dev)
    tree = cache.attend(q[:8], [0], [0], [pos0], 8, parents=parents)
    assert torch.equal(cache.attend(q[:8], [0], [0], [pos0], 8, parents=parents, fused=True), tree)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "fp8"])
def test_a_fused_chunk_reads_its_own_keys_and_values_fresh(kind):
    # The chunk's positions from a first one on (shift and first on the device: one CUDA graph) from fresh rows, the
    # earlier ones from the pages: float64 attention over [the pages' prefix, the fresh rows]; with the first fresh
    # position past the chunk the pages' bits exactly.
    cache = Cache(kind, 2, 128, 128, 16, [700], multipliers=(0.05, 0.04))
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
                  key_scales=cache.ks, value_scales=cache.vs, multipliers=cache.multipliers, out=out, lse=lse,
                  fused=True, fresh=(fk, fv, frontier, first))
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
                paged = cache.attend(q, [0, 0], [0, 64], [at, at + 64], 64, counts=counts)
                assert torch.equal(out, paged), at


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8"])
def test_a_tree_sees_its_ancestors_and_a_chain_is_causal(kind):
    cache = Cache(kind, 2, 128, 128, 16, [400])
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], dtype=torch.int64, device=dev)
    q = _q(8, 8, 128, 6)
    out = cache.attend(q, [0], [0], [392], 8, parents=parents)
    keys, values = cache.rows(0, 400)
    anc = []
    for c in range(8):
        found, p = {c}, int(parents[c])
        while p >= 0:
            found.add(p)
            p = int(parents[p])
        anc.append(found)
    qd = q.to(torch.bfloat16).double()
    for c in range(8):
        seen = torch.tensor([n < 392 or (n - 392) in anc[c] for n in range(400)], device=dev)
        want, _ = reference(qd[c], keys, values, seen, 128 ** -0.5)
        torch.testing.assert_close(out[c].double(), want, atol=8e-3, rtol=2e-2)
    chain = cache.attend(q, [0], [0], [392], 8, parents=torch.arange(-1, 7, device=dev))
    assert torch.equal(chain, cache.attend(q, [0], [0], [392], 8))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "int8"])
@pytest.mark.parametrize("block_size", [4, 16, 80])
def test_block_lists_see_only_their_blocks(kind, block_size):
    cache = Cache(kind, 2, 128, 128, 16, [1000])
    g = torch.Generator().manual_seed(9)
    q = _q(4, 8, 128, 7)
    nblk = math.ceil(1000 / block_size)
    blocks = torch.full((4, 12), -1, dtype=torch.int32)
    for t in range(4):
        chosen = torch.randperm(nblk, generator=g)[:10 - t].sort().values
        blocks[t, :len(chosen)] = chosen.to(torch.int32)
    blocks = blocks.to(dev)
    lse = torch.empty(4, 8, device=dev)
    out = cache.attend(q, [0], [0], [996], 4, blocks=blocks, block_size=block_size, lse=lse)
    keys, values = cache.rows(0, 1000)
    qd = q.to(torch.bfloat16).double()
    n = torch.arange(1000, device=dev)
    for t in range(4):
        listed = blocks[t][blocks[t] >= 0].long()
        seen = (n // block_size)[:, None].eq(listed[None, :]).any(1) & (n <= 996 + t)
        if not bool(seen.any()):
            assert bool((out[t] == 0).all()) and bool(torch.isinf(lse[t]).all())
            continue
        want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
        torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2)
        torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.kernels_ci
def test_one_cuda_graph_replays_at_every_length():
    cache = Cache("fp8:row", 4, 256, 256, 16, [5000, 5000])
    q = _q(10, 24, 256, 8).to(torch.bfloat16)
    rows = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    starts = torch.tensor([0, 5], dtype=torch.int32, device=dev)
    firsts = torch.tensor([10, 20], dtype=torch.int64, device=dev)

    def launch():
        return pk.attend(q, cache.k, cache.v, "fp8:row", cache.tables, rows, starts, firsts, 5, 16,
                         key_scales=cache.ks, value_scales=cache.vs)
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = launch()
    for a, b in ((10, 20), (4000, 63), (64, 4990), (1, 2)):
        firsts.copy_(torch.tensor([a, b], device=dev))
        graph.replay()
        assert torch.equal(out, launch())


@pytest.mark.kernels_ci
def test_e4m3_decoded_in_software_equals_the_hardware_conversion(monkeypatch):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("the hardware conversion needs sm_89+")
    cache = Cache("fp8:row", 2, 128, 128, 16, [300])
    q = _q(4, 8, 128, 2)
    hardware = cache.attend(q, [0], [0], [296], 4)
    monkeypatch.setattr(pk, "_hardware_fp8", lambda device: False)
    assert torch.equal(cache.attend(q, [0], [0], [296], 4), hardware)


@pytest.mark.kernels_ci
def test_work_follows_the_rows_length_not_the_capacity():
    # No timing (the tests share the GPU): one graph over a 200K-position table, replayed at a long and then a short
    # length with every position past it poisoned first (NaN scales): a read of any of them would make the output NaN
    # or move it, so a replay reads the row's own positions only, whatever the table holds.
    cache = Cache("fp8:row", 4, 256, 256, 64, [200_000])
    q = _q(1, 24, 256, 1).to(torch.bfloat16)
    firsts = torch.tensor([0], device=dev)
    args = (q, cache.k, cache.v, "fp8:row", cache.tables, torch.zeros(1, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev), firsts, 1, 64)
    kw = dict(key_scales=cache.ks, value_scales=cache.vs)
    pk.attend(*args, **kw)
    graph = torch.cuda.CUDAGraph()      # one graph, its work read from the row's length on the device
    with torch.cuda.graph(graph):
        out = pk.attend(*args, **kw)
    for first in (199_000, 4_000):
        firsts.fill_(first)
        want = pk.attend(*args, **kw)
        past = cache.slots(0, torch.arange(first + 1, 200_000, device=dev))
        cache.ks.view(-1, 4)[past] = float("nan")
        cache.vs.view(-1, 4)[past] = float("nan")
        graph.replay()
        assert bool(torch.isfinite(out).all()), first
        assert torch.equal(out, want), first


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "bf16", "fp8"])
def test_prompt_chunks_match_float64_attention(kind):
    cache = Cache(kind, 2, 128, 128, 16, [300])
    q = _q(100, 8, 128, 4)
    lse = torch.empty(100, 8, device=dev)
    table = cache.tables[0]
    q_dtype = torch.bfloat16
    out = pk.prefill(q.to(q_dtype), cache.k, cache.v, kind, table, 200, 16, key_scales=cache.ks,
                     value_scales=cache.vs, multipliers=cache.multipliers, lse=lse)
    keys, values = cache.rows(0, 300)
    qd = q.to(q_dtype).double()
    for t in (0, 37, 99):
        seen = torch.arange(300, device=dev) <= 200 + t
        want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
        torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2)
        torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4)
    if kind == "int8":
        approx = pk.prefill(q.to(q_dtype), cache.k, cache.v, kind, table, 200, 16, key_scales=cache.ks,
                            value_scales=cache.vs, int8_qk=True)
        assert float((approx.float() - out.float()).abs().max()) < 0.06


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp8:row", "int8"])
def test_key_only_writes_leave_no_values(kind):
    g = torch.Generator().manual_seed(3)
    keys = torch.randn(20, 1, 64, generator=g).to(dev)
    k = torch.zeros(32, 1, 64, dtype=STORE[kind], device=dev)
    ks = torch.zeros(32, 1, device=dev) if kind != "bf16" else None
    slots = torch.arange(20, device=dev) + 5
    pk.write(keys, None, slots, k, None, kind, ks, None)
    both = torch.zeros_like(k)
    other = torch.zeros(32, 1, 64, dtype=STORE[kind], device=dev)
    both_s = torch.zeros(32, 1, device=dev) if ks is not None else None
    pk.write(keys, keys, slots, both, other, kind, both_s, torch.zeros_like(both_s) if ks is not None else None)
    view = (lambda t: t.view(torch.uint8)) if kind == "fp8:row" else (lambda t: t)
    assert torch.equal(view(k), view(both))
    if ks is not None:
        assert torch.equal(ks, both_s)


@pytest.mark.kernels_ci
def test_a_given_workspace_holds_the_partials_and_a_small_one_is_refused():
    cache = Cache("bf16", 2, 128, 128, 16, [300])
    q = _q(4, 8, 128, 9)
    floats = pk.workspace_floats(4, 8, 16, 128)
    own = cache.attend(q, [0], [0], [296], 4, partitions=16)
    given = cache.attend(q, [0], [0], [296], 4, partitions=16,
                         workspace=torch.empty(floats, dtype=torch.float32, device=dev))
    assert torch.equal(own, given)
    with pytest.raises(ValueError, match="workspace"):
        cache.attend(q, [0], [0], [296], 4, partitions=16,
                     workspace=torch.empty(floats - 1, dtype=torch.float32, device=dev))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "bf16"])
def test_a_prompt_chunk_at_a_device_position_replays_from_one_cuda_graph(kind):
    cache = Cache(kind, 4, 256, 256, 16, [700])
    q = _q(64, 24, 256, 12).to(torch.bfloat16)
    table = cache.tables[0]
    position = torch.zeros(1, dtype=torch.int64, device=dev)
    out = torch.empty_like(q)

    def launch():
        pk.prefill(q, cache.k, cache.v, kind, table, position, 16, key_scales=cache.ks, value_scales=cache.vs,
                   multipliers=cache.multipliers, out=out)
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for at in (0, 5, 300, 636):
        position.fill_(at)
        graph.replay()
        want = pk.prefill(q, cache.k, cache.v, kind, table, at, 16, key_scales=cache.ks, value_scales=cache.vs,
                          multipliers=cache.multipliers)
        assert torch.equal(out, want), at


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "fp8", "bf16"])
def test_a_prompt_chunk_reads_its_own_keys_and_values_fresh(kind):
    # The positions before the chunk from the pages (at their kind's precision), the chunk's own from the fresh keys
    # and values (here other values than its pages hold, so a read of its pages would show); with a first fresh
    # position (on the device) the chunk's positions before it from the pages too, none fresh past the chunk.
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
                       multipliers=cache.multipliers, out=out, lse=lse, fresh=fresh)
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


@pytest.mark.parametrize("arch", [80, 90, 100, 120])
@pytest.mark.parametrize("knob", [1, 4, 8])
@pytest.mark.parametrize("group,d,kind,columns", [(6, 256, 2, 5), (12, 256, 0, 5), (64, 128, 0, 1)])
def test_every_warps_knob_multiplies_with_mma_sync_on_every_gpu(arch, knob, group, d, kind, columns):
    # A tuning's warps for a 32-row tile, a 64-row window tile and a 64-row tile of one column's 64 heads: attend takes
    # the knob except where four warps (or more) over 64 rows would be warpgroup MMA, so every choice stays mma.sync.
    assert _lowering(arch, group, d, kind, columns, knob=knob) == {"mma.sync"}


SPEED = {"write": {"WARPS": 4}, "window": {"WARPS": 4, "STAGES": 2}, "merge": {"WARPS": 1, "CHUNK": 4},
         "fused": {"WARPS": 8, "STAGES": 2, "VSPLIT": 4}, "sparse": {"WARPS": 4, "STAGES": 2},
         "prefill": {"BLOCK_M": 128, "BLOCK_N": 32, "WARPS": 8, "STAGES": 2}}


def _tuned(knobs):
    """A build holding `knobs` beside each structure's defaults, the given ones selected for every call."""
    module = load("paged-kv-attention", configs={s: [k] for s, k in knobs.items()})
    for structure in knobs:
        select(module, structure, 1)
    return module


@pytest.mark.kernels_ci
def test_speed_knobs_keep_every_bit_and_prompt_blocks_stay_within_bounds():
    tuned = _tuned(SPEED)
    global pk
    default = pk
    try:
        for kind in ("bf16", "int8"):
            cache = Cache(kind, 4, 256, 256, 16, [700, 1500])
            q = _q(16, 24, 256, 13)
            blocks = torch.tensor([[0, 3, 10, -1], [1, 2, 40, 43]], dtype=torch.int32, device=dev)
            launches = [((q, [0, 1], [0, 8], [690, 1490], cols), {}) for cols in (1, 5, 8)]
            launches.append(((q, [0, 1], [0, 8], [690, 1490], 8), dict(fused=True)))
            launches.append(((q[:2], [0, 1], [0, 1], [690, 1490], 1), dict(blocks=blocks, block_size=16)))
            found = []
            for module in (default, tuned):
                pk = module
                # Into zeroed outputs: a launch writes its rows' tokens only.
                found.append([cache.attend(*args, out=torch.zeros(args[0].shape[0], 24, 256, dtype=torch.bfloat16,
                                                                  device=dev), **kw) for args, kw in launches])
                k = torch.zeros_like(cache.k)
                v = torch.zeros_like(cache.v)
                ks = None if cache.ks is None else torch.zeros_like(cache.ks)
                vs = None if cache.vs is None else torch.zeros_like(cache.vs)
                x = _q(40, 4, 256, 3)
                pk.write(x, x * 2, torch.arange(40, device=dev) * 3, k, v, kind, ks, vs)
                found[-1].append(_bits_of(k, v, ks, vs))
            for a, b in zip(*found):
                assert torch.equal(a, b)
        pk = tuned
        test_prompt_chunks_match_float64_attention("bf16")
        test_prompt_chunks_match_float64_attention("int8")
    finally:
        pk = default


def _bits_of(*tensors):
    return torch.cat([t.reshape(-1).view(torch.uint8) for t in tensors if t is not None])


@pytest.mark.kernels_ci
def test_the_split_knobs_reach_every_launch_over_a_cache():
    split = _tuned({"split": {"PIECE": 64, "PARTS": 16}})
    assert split.partitions(dev, 4, "int8", 256, 256) == 16 and split.piece_size(256, 256, "int8", 4) == 64
    # Without the cache's geometry the rule's: a tuning is keyed by it.
    assert split.partitions(dev, 4, "int8") == pk.partitions(dev, 4, "int8")
    assert split.piece_size(256, 256, "int8") == pk.piece_size(256, 256, "int8")
    cache = Cache("int8", 4, 256, 256, 16, [700, 1500])
    q = _q(16, 24, 256, 13)
    args = (q.to(torch.bfloat16), cache.k, cache.v, "int8", cache.tables, torch.tensor([0, 1], dtype=torch.int32,
            device=dev), torch.tensor([0, 8], dtype=torch.int32, device=dev),
            torch.tensor([690, 1490], dtype=torch.int64, device=dev), 8, 16)
    kw = dict(key_scales=cache.ks, value_scales=cache.vs)
    want = pk.attend(*args, partitions=16, piece=64, **kw)
    assert torch.equal(split.attend(*args, **kw), want)
    assert torch.equal(split.attend(*args, fused=True, **kw), want)
