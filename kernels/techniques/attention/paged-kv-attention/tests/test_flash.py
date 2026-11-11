"""The flash structures (attend(..., flash=True), prefill(..., flash=True)) against a float64 reference: windows of
every page kind, a column's bits the same alone, in any window, packed, fused or split, through trees, fresh rows,
CUDA graph replays at other lengths, work that follows a row's length; prompt chunks split on the device or not; every
tile a GPU gets is one instruction; speed knobs keep every bit and the order knobs reach every launch."""
import pytest
import torch

from crit_kernels.build import load, select
from test_paged_kv_attention import Cache, SHAPES, _q, dev, reference

pk = load("paged-kv-attention")
FLASH = dict(flash=True)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "fp16", "fp8", "fp8:row", "int8"])
@pytest.mark.parametrize("hq,hkv,d,dv", SHAPES)
@pytest.mark.parametrize("page", [16, 48])
def test_flash_windows_match_float64_attention(kind, hq, hkv, d, dv, page):
    cache = Cache(kind, hkv, d, dv, page, [130, 64, 1, 2517])
    rows, starts, firsts = [0, 1, 2, 3], [0, 5, 10, 15], [125, 59, 0, 2400]
    counts = [5, 5, 1, 3]
    q = _q(20, hq, d, 3)
    scale = d ** -0.5
    qd = q.to(torch.bfloat16 if kind != "fp16" else torch.float16).double()
    for fused in (False, True):
        lse = torch.empty(20, hq, device=dev)
        out = cache.attend(q, rows, starts, firsts, 5, lse=lse, fused=fused, chunk=256 if fused else None,
                           counts=torch.tensor(counts, dtype=torch.int32, device=dev), **FLASH)
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
def test_a_flash_column_has_its_bits_alone_in_any_window_fused_or_split(kind, hq, hkv, d):
    # Every column of a flash launch has the bits of the same column decoded alone (log-sum-exps too), in windows of
    # every width (tiles of 16 to 128 rows), packed by tokens, fused or split, at lengths of one split and of many.
    for length, chunk in ((40, None), (2100, None), (5000, 256)):
        cache = Cache(kind, hkv, d, d, 16, [length, length], seed=1)
        kw = dict(chunk=chunk, **FLASH)
        T = min(64, length)
        pos0 = length - T
        q = _q(T, hq, d, 2)
        one_lse = torch.empty(T, hq, device=dev)
        alone = cache.attend(q, [0] * T, list(range(T)), [pos0 + i for i in range(T)], 1, lse=one_lse, **kw)
        for cols in (2, 5, 16, 64):
            starts = list(range(0, T, cols))
            counts = torch.tensor([min(cols, T - s) for s in starts], dtype=torch.int32, device=dev)
            for fused in (False, True):
                lse = torch.empty(T, hq, device=dev)
                got = cache.attend(q, [0] * len(starts), starts, [pos0 + s for s in starts], cols, counts=counts,
                                   lse=lse, fused=fused, **kw)
                assert torch.equal(got, alone), (kind, length, cols, fused)
                assert torch.equal(lse, one_lse), (kind, length, cols, fused)
        # rows packed by their tokens (offsets), in a workspace of those tokens only
        offsets = torch.tensor([0, 3], dtype=torch.int32, device=dev)
        parts = pk.flash_parts(cache.tables.shape[1] * 16, chunk or pk.flash_chunk(d, d, hq, hkv, 16, kind))
        packed = cache.attend(q, [0, 1], [0, 3], [pos0, pos0 + 3], 8, offsets=offsets, tokens=8,
                              counts=torch.tensor([3, 5], dtype=torch.int32, device=dev),
                              workspace=torch.empty(pk.workspace_floats(8, hq, parts, d), device=dev), **kw)
        assert torch.equal(packed[:3], alone[:3])
        other = cache.attend(q[3:8], [1], [0], [pos0 + 3], 5, **kw)     # the second sequence holds other keys
        assert torch.equal(packed[3:8], other)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["bf16", "int8"])
def test_a_flash_tree_sees_its_ancestors_and_a_chain_is_causal(kind):
    cache = Cache(kind, 2, 128, 128, 16, [400])
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 5, 3], dtype=torch.int64, device=dev)
    q = _q(8, 8, 128, 6)
    out = cache.attend(q, [0], [0], [392], 8, parents=parents, **FLASH)
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
    chain = cache.attend(q, [0], [0], [392], 8, parents=torch.arange(-1, 7, device=dev), **FLASH)
    assert torch.equal(chain, cache.attend(q, [0], [0], [392], 8, **FLASH))
    assert torch.equal(cache.attend(q, [0], [0], [392], 8, parents=parents, fused=True, **FLASH), out)
    assert torch.equal(cache.attend(q, [0], [0], [392], 8, parents=parents, num_stages=2, **FLASH), out)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8"])
def test_a_fused_flash_chunk_reads_its_own_keys_and_values_fresh(kind):
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
                  fresh=(fk, fv, frontier, first), **FLASH)
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
                paged = cache.attend(q, [0, 0], [0, 64], [at, at + 64], 64, counts=counts, chunk=256, **FLASH)
                assert torch.equal(out, paged), at


@pytest.mark.kernels_ci
def test_a_flash_graph_replays_at_every_length_reading_the_rows_positions_only():
    # One graph over a 200K-position table, replayed at a long and a short length with every position past the row
    # poisoned first (NaN scales): the replay reads the row's own positions only and equals an eager launch.
    cache = Cache("fp8:row", 4, 256, 256, 64, [200_000])
    q = _q(5, 24, 256, 1).to(torch.bfloat16)
    firsts = torch.tensor([0], device=dev)
    args = (q, cache.k, cache.v, "fp8:row", cache.tables, torch.zeros(1, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev), firsts, 5, 64)
    kw = dict(key_scales=cache.ks, value_scales=cache.vs, **FLASH)
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
def test_flash_prompt_chunks_match_float64_attention(kind, T):
    cache = Cache(kind, 2, 128, 128, 16, [200 + T])
    q = _q(T, 8, 128, 4).to(torch.bfloat16)
    keys, values = cache.rows(0, 200 + T)
    qd = q.double()
    table = cache.tables[0]
    for splits in (0, 1, 16):
        flash = _tuned({"flash-prefill": {"SPLITS": splits}}) if splits else pk
        lse = torch.empty(T, 8, device=dev)
        out = flash.prefill(q, cache.k, cache.v, kind, table, 200, 16, key_scales=cache.ks, value_scales=cache.vs,
                            multipliers=cache.multipliers, lse=lse, **FLASH)
        for t in (0, T // 3, T - 1):
            seen = torch.arange(200 + T, device=dev) <= 200 + t
            want, want_lse = reference(qd[t], keys, values, seen, 128 ** -0.5)
            torch.testing.assert_close(out[t].double(), want, atol=8e-3, rtol=2e-2, msg=(splits, t))
            torch.testing.assert_close(lse[t].double(), want_lse, atol=1e-3, rtol=1e-4, msg=(splits, t))
        if kind == "int8":
            approx = flash.prefill(q, cache.k, cache.v, kind, table, 200, 16, key_scales=cache.ks,
                                   value_scales=cache.vs, int8_qk=True, **FLASH)
            assert float((approx.float() - out.float()).abs().max()) < 0.06


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "bf16"])
@pytest.mark.parametrize("T", [16, 64])
def test_a_flash_prompt_chunk_at_a_device_position_replays_from_one_cuda_graph(kind, T):
    cache = Cache(kind, 4, 256, 256, 16, [700])
    q = _q(T, 24, 256, 12).to(torch.bfloat16)
    table = cache.tables[0]
    position = torch.zeros(1, dtype=torch.int64, device=dev)
    out = torch.empty_like(q)

    def launch():
        pk.prefill(q, cache.k, cache.v, kind, table, position, 16, key_scales=cache.ks, value_scales=cache.vs,
                   multipliers=cache.multipliers, out=out, **FLASH)
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for at in (0, 5, 300, 700 - T):
        position.fill_(at)
        graph.replay()
        want = pk.prefill(q, cache.k, cache.v, kind, table, at, 16, key_scales=cache.ks, value_scales=cache.vs,
                          multipliers=cache.multipliers, **FLASH)
        assert torch.equal(out, want), at


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", ["fp8:row", "int8", "bf16"])
def test_a_flash_prompt_chunk_reads_its_own_keys_and_values_fresh(kind):
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
                       multipliers=cache.multipliers, out=out, lse=lse, fresh=fresh, **FLASH)
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


def _flash_lowering(arch, group, d, kind, fused=False):
    """(instructions, warpsPerCTA of every MMA layout) Triton gives the flash window launch flash_tile() picks on `arch`
    (compiled offline, no GPU of that kind needed): wgmma (warp_group_dot), tcgen05 (tc_gen5_mma) or mma.sync
    (nvidia_mma v2)."""
    import re
    import triton
    from triton.backends.compiler import GPUTarget
    capability = (arch // 10, arch % 10)
    kinds = ["bf16", "fp8", "fp8:row", "int8"]
    rows, warps = pk.flash_tile(d, d, group * 2, 2, 16, kinds[kind], capability)
    kernel = pk._flash._flash_kernel
    store = {0: "*bf16", 1: "*u8", 2: "*u8", 3: "*i8"}[kind]
    types = {"k_ptr": store, "v_ptr": store, "q_ptr": "*bf16", "out_ptr": "*bf16", "fk_ptr": "*bf16",
             "fv_ptr": "*bf16", "first_ptr": "*i64", "par_ptr": "*i64", "fshift_ptr": "*i64", "ffirst_ptr": "*i64",
             "ks_ptr": "*fp32", "vs_ptr": "*fp32", "po_ptr": "*fp32", "pm_ptr": "*fp32", "pl_ptr": "*fp32",
             "lse_ptr": "*fp32", "qk_scale": "fp32", "k_mult": "fp32", "v_mult": "fp32", "first0": "i64",
             "ffirst0": "i64"}
    constants = {"HQ": group * 2, "HKV": 2, "G": group, "D": d, "DV": d, "DVP": d // 2 if fused and d > 128 else d,
                 "BLOCK_M": rows, "BLOCK_N": pk.flash_block(d, d, group * 2, 2, 16, kinds[kind]), "PAGE": 16,
                 "KIND": kind, "HW_FP8": arch >= 89, "TREE": False, "TREE_COLS": 1, "COUNTS": True, "OFFSETS": True,
                 "FUSED": fused, "ONE_SPLIT": False, "LSE": True, "FRESH": fused, "ONE_ROW": False, "POS_T": False,
                 "FIRST_T": False, "DEV_CHUNK": False, "INT8_QK": False, "SKIP": False}
    names = kernel.arg_names
    full = {n: "constexpr" if n in constants else types.get(n, "*i32" if n.endswith("_ptr") else "i32")
            for n in names}
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(names) if full[n].startswith("*")}
    source = triton.compiler.ASTSource(fn=kernel, signature=full, constexprs=constants, attrs=attrs)
    ir = triton.compile(source, target=GPUTarget("cuda", arch, 32),
                        options={"num_warps": warps, "num_stages": 1}).asm["ttgir"]
    found = {name for name, key in (("wgmma", "warp_group_dot"), ("tcgen05", "tc_gen5_mma"),
                                    ("mma.sync", "versionMajor = 2")) if key in ir}
    layouts = set(re.findall(r"nvidia_mma<\{[^}]*warpsPerCTA = \[(\d+), (\d+)\]", ir))
    return found, layouts


@pytest.mark.parametrize("arch", [80, 89, 120])
@pytest.mark.parametrize("group,d,kind", [(12, 256, 0), (12, 256, 2), (7, 128, 3), (12, 128, 0)])
def test_the_flash_tile_multiplies_with_mma_sync_in_one_layout_split_or_fused(arch, group, d, kind):
    # One tile per cache: the split and the fused launch lower it to mma.sync in the same warp layout, so a row's sum
    # (the layout's reduction) and products are the same in both and in every tile of the launch.
    split, split_layouts = _flash_lowering(arch, group, d, kind)
    fused, fused_layouts = _flash_lowering(arch, group, d, kind, fused=True)
    assert split == fused == {"mma.sync"}
    assert split_layouts and split_layouts == fused_layouts


@pytest.mark.parametrize("arch", [90, 100, 110])
def test_a_gpu_with_warpgroup_mma_takes_one_flash_tile_for_every_cache(arch):
    capability = (arch // 10, arch % 10)
    tiles = {pk.flash_tile(d, d, hq, hkv, 16, kind, capability) for d, hq, hkv in ((256, 24, 2), (128, 56, 8))
             for kind in ("bf16", "int8")}
    assert tiles == {pk.FLASH_WARPGROUP_TILE}
    split, split_layouts = _flash_lowering(arch, 12, 128, 0)
    fused, fused_layouts = _flash_lowering(arch, 12, 128, 0, fused=True)
    assert split == fused and split_layouts == fused_layouts


def _tuned(knobs):
    module = load("paged-kv-attention", configs={s: [k] for s, k in knobs.items()})
    for structure in knobs:
        select(module, structure, 1)
    return module


@pytest.mark.kernels_ci
def test_flash_speed_knobs_keep_every_bit_and_the_order_knobs_reach_every_launch():
    speed = _tuned({"flash-window": {"STAGES": 2}, "flash-fused": {"STAGES": 2, "VSPLIT": 4}})
    order = _tuned({"flash-window": {"CHUNK": 256, "ROWS": 64, "WARPS": 8}})
    block = _tuned({"flash-window": {"BLOCK_N": 64}})
    assert order.flash_chunk(128, 128, 24, 4, 16, "int8") == 256
    assert block.flash_block(128, 128, 24, 4, 16, "int8") == 64
    assert order.flash_tile(128, 128, 24, 4, 16, "int8", (12, 0)) == (64, 8)
    for kind in ("bf16", "int8"):
        cache = Cache(kind, 4, 128, 128, 16, [700, 1500])
        q = _q(16, 24, 128, 13).to(torch.bfloat16)
        args = (q, cache.k, cache.v, kind, cache.tables, torch.tensor([0, 1], dtype=torch.int32, device=dev),
                torch.tensor([0, 8], dtype=torch.int32, device=dev),
                torch.tensor([690, 1490], dtype=torch.int64, device=dev), 8, 16)
        kw = dict(key_scales=cache.ks, value_scales=cache.vs, **FLASH)
        want = pk.attend(*args, **kw)
        assert torch.equal(speed.attend(*args, **kw), want)
        assert torch.equal(speed.attend(*args, fused=True, **kw), want)
        got = order.attend(*args, **kw)
        assert torch.equal(order.attend(*args, fused=True, **kw), got)
        assert torch.equal(got, pk.attend(*args, chunk=256, tile=64, num_warps=8, **kw))
        assert torch.equal(block.attend(*args, fused=True, **kw), block.attend(*args, **kw))
