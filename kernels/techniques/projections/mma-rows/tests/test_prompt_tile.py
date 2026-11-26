"""mm_prompt, the prompt tile, against mm_rows, the decode tile, bit for bit: for every weight format and activation
quantization, with and without splits of K, every output of a prompt chunk of any length equals the same row's output
from a decode call (alone, or in a call of up to 16 rows), a tile's splits run in one CTA or in several (partial and
counters given: each later split's chain stored and added in split order), whatever partial buffer bounds them; at the
role shapes of real checkpoints with plan()'s splits, prompt chunks of 1 to 4,096 rows; and the call captured in a CUDA
graph, its counters left zero."""
import math

import pytest
import torch

from crit_kernels import build as kbuild

from test_mma_rows import FORMATS, INPUT_SCALE, activations, call, matrix, mr
dev = torch.device("cuda")

# (role, weights, N, K): stacked matrices as model plugins load them (a llama-layout 33B NVFP4 checkpoint, a hybrid
# 27B one with FP8 attention projections and NVFP4 MLP), read from the checkpoints' headers.
ROLES = [("qkv", "nvfp4", 9216, 7168), ("o", "nvfp4", 7168, 7168), ("gate_up", "nvfp4", 44032, 7168),
         ("down", "nvfp4", 7168, 22016), ("gate_up", "nvfp4", 34816, 5120), ("down", "nvfp4", 5120, 17408),
         ("qkvz", "fp8", 16384, 5120), ("out", "fp8", 5120, 6144), ("qkv", "fp8", 14336, 5120)]


PARTIAL, COUNTERS = 16 << 20, 1 << 14   # quantized-projections' shared partial sums (64 MB) and counters


def parallel_buffers(partial=PARTIAL):
    """partial and counters for mm_prompt (counters zero): a tile's splits in several CTAs where they fit."""
    return {"partial": torch.empty(max(1, partial), dtype=torch.float32, device=dev),
            "counters": torch.zeros(COUNTERS, dtype=torch.int32, device=dev)}


def prompt(x, stored, weights, acts="bf16", blocked=False, splits=1, dtype=torch.float32, partial=PARTIAL):
    """mm_prompt's outputs; partial: the FP32 elements its partial sums may take (None: no partial or counters, a
    CTA runs every split of its tile)."""
    w, scales, row_scale = stored
    out = torch.empty(x.shape[0], w.shape[0], dtype=dtype, device=dev)
    scratch = torch.empty(max(16, mr.scratch_bytes(x.shape[0], x.shape[1])), dtype=torch.uint8, device=dev)
    extra = {} if partial is None else parallel_buffers(partial)
    found = mr.mm_prompt(x, w, out, weights, scales, row_scale, acts, INPUT_SCALE, blocked, splits, scratch, **extra)
    if extra:
        assert not extra["counters"].any(), "the counters are left zero"
    return found


def decoded(x, stored, weights, acts, blocked, splits, per_call, dtype=torch.float32):
    """The decode tile's outputs of x's rows, `per_call` rows a call."""
    return torch.cat([call(x[i:i + per_call], stored, weights, acts, blocked, dtype=dtype, splits=splits)
                      for i in range(0, x.shape[0], per_call)])


def same_bits(a, b):
    return torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a.view(torch.int16),
                       b.view(torch.int32) if b.dtype == torch.float32 else b.view(torch.int16))


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", [(96, 256), (1000, 5120), (130, 2048), (40, 640), (300, 1056), (257, 96)])
def test_a_prompt_rows_bits_are_the_decode_tiles_at_every_row_count(fmt, shape):
    weights, acts, blocked = fmt
    n, k = shape
    stored, _ = matrix(weights, n, k, blocked)
    steps = math.ceil(k / mr.VALUES[weights])
    x = activations(150, k)
    for splits in sorted({1, 2, 3, min(5, steps), min(steps, 32)}):
        if splits > steps:
            continue
        found = prompt(x, stored, weights, acts, blocked, splits)
        alone = decoded(x, stored, weights, acts, blocked, splits, 1)
        assert same_bits(found, alone), (fmt, shape, splits, (found - alone).abs().max().item())
        assert same_bits(prompt(x, stored, weights, acts, blocked, splits, partial=None), found), (fmt, shape, splits)
        for per_call in range(2, 17):   # decode calls of 2 .. 16 rows (and the tile's 64)
            assert same_bits(found[:3 * per_call], decoded(x[:3 * per_call], stored, weights, acts, blocked, splits,
                                                           per_call)), (fmt, shape, splits, per_call)
        assert same_bits(found, decoded(x, stored, weights, acts, blocked, splits, 64))
        for rows in (1, 2, 7, 8, 9, 63, 64, 65, 129):   # any chunk length: a row's bits are its own
            assert same_bits(prompt(x[:rows], stored, weights, acts, blocked, splits), found[:rows]), rows


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_every_output_type_rounds_the_same_sum(dtype):
    stored, _ = matrix("nvfp4", 300, 1056)
    x = activations(70, 1056)
    assert same_bits(prompt(x, stored, "nvfp4", splits=3, dtype=dtype),
                     decoded(x, stored, "nvfp4", "bf16", False, 3, 16, dtype=dtype))


@pytest.mark.parametrize("role", ROLES, ids=lambda r: f"{r[0]}-{r[2]}x{r[3]}")
def test_real_shapes_with_their_plans_from_one_row_to_4096(role):
    _, weights, n, k = role
    stored, _ = matrix(weights, n, k, seed=11)
    _, _, splits = mr.plan(n, k, weights)
    x = activations(4096, k, seed=13)
    found = prompt(x, stored, weights, splits=splits)
    assert same_bits(found, decoded(x, stored, weights, "bf16", False, splits, 64)), role
    assert same_bits(found[:48], decoded(x[:48], stored, weights, "bf16", False, splits, 16)), role
    for rows in (1, 5, 16, 17, 100, 1000):
        assert same_bits(prompt(x[:rows], stored, weights, splits=splits), found[:rows]), (role, rows)


@pytest.mark.parametrize("weights, acts, n, k", [("fp8", "fp8_rows", 336, 10240), ("bf16", "bf16", 1, 2560),
                                                 ("nvfp4", "nvfp4", 320, 10240), ("mxfp8", "bf16", 96, 2560),
                                                 ("fp8", "bf16", 2560, 6144)])
def test_a_tiles_splits_in_several_ctas_keep_the_bits_whatever_buffer_bounds_them(weights, acts, n, k):
    # A narrow matrix's few tiles: the plan's splits spread over CTAs (more CTAs than tiles), every output the decode
    # tile's bits; a partial buffer too small for them takes fewer parts, none takes one CTA a tile: the same bits.
    blocked = acts == "nvfp4"
    stored, _ = matrix(weights, n, k, blocked, seed=3)
    _, _, splits = mr.plan(n, k, weights)
    assert splits > 1
    for rows in (1, 17, 64, 200, 512):
        x = activations(rows, k, seed=rows)
        want = decoded(x, stored, weights, acts, blocked, splits, 64)
        tiles = mr.prompt_ctas(rows, n, k, weights, acts, splits)
        if rows <= 64:   # few tiles: their splits run in several CTAs
            assert mr.prompt_ctas(rows, n, k, weights, acts, splits, PARTIAL, COUNTERS) > tiles, (rows, tiles)
        for partial in (PARTIAL, None, 1, 4096 * 8, 4096 * 40):
            found = prompt(x, stored, weights, acts, blocked, splits, partial=partial)
            assert same_bits(found, want), (weights, n, k, rows, partial)
        assert same_bits(prompt(x, stored, weights, acts, blocked, splits, dtype=torch.bfloat16),
                         decoded(x, stored, weights, acts, blocked, splits, 64, dtype=torch.bfloat16))


def test_a_captured_call_replays_with_the_same_bits():
    stored, _ = matrix("fp8", 520, 2048)
    x = activations(200, 2048)
    expected = prompt(x, stored, "fp8", "fp8_rows", splits=4)
    w, scales, row_scale = stored
    out = torch.empty(200, 520, dtype=torch.float32, device=dev)
    scratch = torch.empty(mr.scratch_bytes(200, 2048), dtype=torch.uint8, device=dev)
    extra = parallel_buffers()
    assert mr.prompt_ctas(200, 520, 2048, "fp8", "fp8_rows", 4, PARTIAL, COUNTERS) > \
        mr.prompt_ctas(200, 520, 2048, "fp8", "fp8_rows", 4)   # the replays run the splits in several CTAs

    def run():
        mr.mm_prompt(x, w, out, "fp8", scales, row_scale, "fp8_rows", INPUT_SCALE, False, 4, scratch, **extra)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert same_bits(out, expected)
        assert not extra["counters"].any()


def test_shapes_the_tile_refuses():
    stored, _ = matrix("nvfp4", 64, 256)
    with pytest.raises(RuntimeError, match="multiple of"):
        prompt(activations(4, 250)[:, :240].contiguous(), (stored[0][:, :120].contiguous(), stored[1], stored[2]),
               "nvfp4")
    with pytest.raises(RuntimeError, match="splits"):
        prompt(activations(4, 256), stored, "nvfp4", splits=33)
    w, scales, row_scale = stored
    out = torch.empty(4, 64, dtype=torch.float32, device=dev)
    with pytest.raises(RuntimeError, match="together"):
        mr.mm_prompt(activations(4, 256), w, out, "nvfp4", scales, row_scale, splits=2,
                     partial=torch.empty(64, dtype=torch.float32, device=dev))


FP4 = pytest.mark.skipif(not mr.fp4_mma(dev), reason="no FP4 MMA on this device or build (the sm_120 family)")


def fp4_prompt(x, stored, splits=1, dtype=torch.float32, glu=False, partial=PARTIAL):
    """The FP4 prompt tile's outputs; partial as prompt()'s (None: a CTA runs every split of its tile)."""
    w, scales, row_scale = stored
    out = torch.empty(x.shape[0], w.shape[0] // 2 if glu else w.shape[0], dtype=dtype, device=dev)
    scratch = torch.empty(mr.scratch_bytes(x.shape[0], x.shape[1], "nvfp4_mma"), dtype=torch.uint8, device=dev)
    extra = {} if partial is None else parallel_buffers(partial)
    found = mr.mm_prompt(x, w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, splits, scratch,
                         glu=glu, **extra)
    if extra:
        assert not extra["counters"].any(), "the counters are left zero"
    return found


def fp4_decoded(x, stored, splits, per_call, dtype=torch.float32):
    """The FP4 MMA decode tile's outputs of x's rows, `per_call` rows a call."""
    from test_mma_rows import fp4_call
    return torch.cat([fp4_call(x[i:i + per_call], stored, splits, dtype=dtype) for i in range(0, x.shape[0], per_call)])


@FP4
@pytest.mark.parametrize("shape", [(96, 256), (1000, 5120), (130, 2048), (40, 640), (300, 1152), (257, 128)])
def test_an_fp4_prompt_rows_bits_are_the_fp4_decode_tiles_at_every_row_count(shape):
    # The FP4 prompt tile against the FP4 MMA decode tile, bit for bit: every split count, decode calls of 1 to 64
    # rows, prompt chunks of any length (tiles partly filled, rows and tokens past the tiles).
    n, k = shape
    stored, _ = matrix("nvfp4", n, k, True)
    steps = k // mr.FP4_STEP
    x = activations(150, k)
    for splits in sorted({1, 2, 3, min(5, steps), min(steps, 32)}):
        if splits > steps:
            continue
        found = fp4_prompt(x, stored, splits)
        alone = fp4_decoded(x, stored, splits, 1)
        assert same_bits(found, alone), (shape, splits, (found - alone).abs().max().item())
        assert same_bits(fp4_prompt(x, stored, splits, partial=None), found), (shape, splits)
        for per_call in (2, 3, 7, 8, 9, 16, 17, 33, 64):
            assert same_bits(found, fp4_decoded(x, stored, splits, per_call)), (shape, splits, per_call)
        for rows in (1, 2, 7, 8, 9, 63, 64, 65, 129):   # any chunk length: a row's bits are its own
            assert same_bits(fp4_prompt(x[:rows], stored, splits), found[:rows]), rows


@FP4
@pytest.mark.parametrize("role", [r for r in ROLES if r[1] == "nvfp4" and r[3] % 128 == 0],
                         ids=lambda r: f"{r[0]}-{r[2]}x{r[3]}")
def test_real_fp4_shapes_with_their_plans_from_one_row_to_4096(role):
    _, _, n, k = role
    stored, _ = matrix("nvfp4", n, k, True, seed=11)
    _, _, splits = mr.plan(n, k, "nvfp4", dev, acts="nvfp4_mma")
    x = activations(4096, k, seed=13)
    found = fp4_prompt(x, stored, splits, dtype=torch.bfloat16)
    assert same_bits(found[:256], fp4_decoded(x[:256], stored, splits, 64, dtype=torch.bfloat16)), role
    assert same_bits(found[-48:], fp4_decoded(x[-48:], stored, splits, 16, dtype=torch.bfloat16)), role
    for rows in (1, 5, 16, 17, 100, 1000):
        assert same_bits(fp4_prompt(x[:rows], stored, splits, dtype=torch.bfloat16), found[:rows]), (role, rows)


@FP4
@pytest.mark.parametrize("splits", [1, 3])
def test_the_fp4_prompt_tile_gives_the_fp4_decode_tiles_glu_bits(splits):
    from test_glu import stored_pair
    _, _, whole = stored_pair("nvfp4", 136, 1024, True)
    x = activations(70, 1024)
    w, scales, row_scale = whole
    want = []
    for i in range(0, 70, 16):
        m = min(16, 70 - i)
        out = torch.empty(m, w.shape[0] // 2, dtype=torch.bfloat16, device=dev)
        scratch = torch.empty(mr.scratch_bytes(m, 1024, "nvfp4_mma"), dtype=torch.uint8, device=dev)
        partial = torch.empty(max(1, mr.partial_elements(m, w.shape[0], splits)), dtype=torch.float32, device=dev)
        counters = torch.zeros(mr.row_tiles(w.shape[0], 1, 4), dtype=torch.int32, device=dev)
        want.append(mr.mm_rows(x[i:i + m], w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, 1, 4,
                               splits, scratch, partial, counters, glu=True))
    assert same_bits(fp4_prompt(x, whole, splits, dtype=torch.bfloat16, glu=True), torch.cat(want))


@FP4
@pytest.mark.parametrize("n, k", [(320, 10240), (324, 10240)])
def test_an_fp4_tiles_splits_in_several_ctas_keep_the_bits_whatever_buffer_bounds_them(n, k):
    # The FP4 prompt tile's few tiles at short prompts (a sixteenth of the resident CTAs or fewer): the plan's splits
    # spread over CTAs, every output the FP4 decode tile's bits; smaller partial buffers take fewer parts, none takes
    # one CTA a tile: the same bits.
    stored, _ = matrix("nvfp4", n, k, True, seed=4)
    _, _, splits = mr.plan(n, k, "nvfp4", dev, acts="nvfp4_mma")
    if splits < 2:
        pytest.skip("one split: nothing to spread")
    for rows in (1, 17, 64, 200, 512):
        x = activations(rows, k, seed=rows + 1)
        want = fp4_decoded(x, stored, splits, 64)
        if rows <= 64:
            assert mr.prompt_ctas(rows, n, k, "nvfp4", "nvfp4_mma", splits, PARTIAL, COUNTERS) > mr.prompt_ctas(
                rows, n, k, "nvfp4", "nvfp4_mma", splits), rows
        for partial in (PARTIAL, None, 1, 16384 * 3):
            assert same_bits(fp4_prompt(x, stored, splits, partial=partial), want), (n, k, rows, partial)
        assert same_bits(fp4_prompt(x, stored, splits, dtype=torch.bfloat16),
                         fp4_decoded(x, stored, splits, 64, dtype=torch.bfloat16)), (n, k, rows)


@FP4
def test_the_fp4_tiles_parts_knob_keeps_the_bits_at_every_shape():
    # PK 2, 4 and 32 (one build, select()): wider matrices and longer prompts in parts, the FP4 decode tile's bits.
    built = kbuild.load("mma-rows", configs={"fp4-prompt-tile": [{"PK": 2}, {"PK": 4}, {"PK": 32}]})
    for n, k in ((2560, 2560), (5120, 17408), (520, 2048)):
        stored, _ = matrix("nvfp4", n, k, True, seed=6)
        _, _, splits = mr.plan(n, k, "nvfp4", dev, acts="nvfp4_mma")
        x = activations(300, k, seed=n)
        want = fp4_decoded(x, stored, splits, 64)
        w, scales, row_scale = stored
        for index in (1, 2, 3):
            kbuild.select(built, "fp4-prompt-tile", index)
            for rows in (16, 300):
                out = torch.empty(rows, n, dtype=torch.float32, device=dev)
                scratch = torch.empty(mr.scratch_bytes(rows, k, "nvfp4_mma"), dtype=torch.uint8, device=dev)
                extra = parallel_buffers()
                built.mm_prompt(x[:rows], w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, splits,
                                scratch, **extra)
                assert same_bits(out, want[:rows]), (n, k, index, rows)
                assert not extra["counters"].any()
    kbuild.select(built, "fp4-prompt-tile", -1)


@FP4
def test_a_captured_fp4_prompt_call_replays_with_the_same_bits_and_refuses_what_it_cannot_read():
    stored, _ = matrix("nvfp4", 520, 2048, True)
    x = activations(200, 2048)
    expected = fp4_prompt(x, stored, 4)
    w, scales, row_scale = stored
    out = torch.empty(200, 520, dtype=torch.float32, device=dev)
    scratch = torch.empty(mr.scratch_bytes(200, 2048, "nvfp4_mma"), dtype=torch.uint8, device=dev)
    extra = parallel_buffers()
    assert mr.prompt_ctas(200, 520, 2048, "nvfp4", "nvfp4_mma", 4, PARTIAL, COUNTERS) > \
        mr.prompt_ctas(200, 520, 2048, "nvfp4", "nvfp4_mma", 4)   # the replays run the splits in several CTAs

    def run():
        mr.mm_prompt(x, w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, 4, scratch, **extra)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert same_bits(out, expected)
        assert not extra["counters"].any()
    row_major, _ = matrix("nvfp4", 520, 2048)
    with pytest.raises(RuntimeError, match="blocked"):   # the FP4 MMA reads blocked scales
        mr.mm_prompt(x, row_major[0], out, "nvfp4", row_major[1], row_major[2], "nvfp4_mma", INPUT_SCALE, False, 1,
                     scratch)
    with pytest.raises(RuntimeError, match="scratch"):
        mr.mm_prompt(x, w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, 1, scratch[:64])
