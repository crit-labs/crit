"""glu: an MLP's gate and up rows interleaved by eights (interleave_glu) in one matrix give out [M][N / 2] =
silu(gate) * up of the two projections as they are stored: against the two projections called apart and the product
applied to their 16-bit outputs, every format, with and without splits of K, in the decode tile, the prompt tile and
on the FP4 MMA; each row's bits the same alone and beside others."""
import pytest
import torch

from test_mma_rows import FORMATS, INPUT_SCALE, activations, buffers, dev, matrix, mr, to_blocked

FP4 = pytest.mark.skipif(not mr.fp4_mma(torch.empty(1, device=dev)), reason="no FP4 MMA on this device or build")


def interleaved(weights, gate, up, blocked):
    """The glu matrix of two stored matrices (w, scales, row_scale), scales row major from matrix(blocked=False)."""
    w = mr.interleave_glu(gate[0], up[0])
    scales = None if gate[1] is None else mr.interleave_glu(gate[1].view(-1, up[1].numel() // up[0].shape[0]),
                                                            up[1].view(-1, up[1].numel() // up[0].shape[0]))
    if scales is not None and blocked:
        scales = to_blocked(scales.view(torch.float8_e4m3fn).cpu()).view(torch.uint8).to(dev)
    row_scale = None if gate[2] is None else mr.interleave_glu(gate[2], up[2])
    return w, scales, row_scale


def apart(x, stored, weights, acts, blocked, splits, prompt=False):
    w, scales, row_scale = stored
    m, n, k = x.shape[0], w.shape[0], x.shape[1]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    if prompt:
        scratch = torch.empty(mr.scratch_bytes(m, k), dtype=torch.uint8, device=dev)
        return mr.mm_prompt(x, w, out, weights, scales, row_scale, acts, INPUT_SCALE, blocked, splits, scratch)
    return mr.mm_rows(x, w, out, weights, scales, row_scale, acts, INPUT_SCALE, blocked, 1, 4, splits,
                      *buffers(m, n, k, splits))


def fused(x, stored, weights, acts, blocked, splits, prompt=False):
    w, scales, row_scale = stored
    m, n, k = x.shape[0], w.shape[0], x.shape[1]
    out = torch.empty(m, n // 2, dtype=torch.bfloat16, device=dev)
    if prompt:
        scratch = torch.empty(mr.scratch_bytes(m, k), dtype=torch.uint8, device=dev)
        return mr.mm_prompt(x, w, out, weights, scales, row_scale, acts, INPUT_SCALE, blocked, splits, scratch,
                            glu=True)
    return mr.mm_rows(x, w, out, weights, scales, row_scale, acts, INPUT_SCALE, blocked, 1, 4, splits,
                      *buffers(m, n, k, splits), glu=True)


def product(g, u):
    g, u = g.float(), u.float()
    return (g * torch.sigmoid(g) * u).to(torch.bfloat16)


def within_activation(got, want):
    """got within one BF16 step (the activation's exp may round otherwise) of want."""
    step = want.float().abs().clamp_min(1e-30) * 2.0 ** -7
    bad = (got.float() - want.float()).abs() > step + 1e-6
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} outputs beyond one BF16 step"


def stored_pair(weights, inter, k, blocked):
    gate, _ = matrix(weights, inter, k, False, seed=5)
    up, _ = matrix(weights, inter, k, False, seed=6)
    whole = interleaved(weights, gate, up, blocked)
    if blocked:
        gate = (gate[0], to_blocked(gate[1].view(torch.float8_e4m3fn).cpu()).view(torch.uint8).to(dev), gate[2])
        up = (up[0], to_blocked(up[1].view(torch.float8_e4m3fn).cpu()).view(torch.uint8).to(dev), up[2])
    return gate, up, whole


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("splits", [1, 3])
def test_glu_is_the_product_of_the_two_projections(fmt, splits):
    weights, acts, blocked = fmt
    inter, k = 136, 1024
    gate, up, whole = stored_pair(weights, inter, k, blocked)
    x = activations(9, k)
    want = product(apart(x, gate, weights, acts, blocked, splits), apart(x, up, weights, acts, blocked, splits))
    got = fused(x, whole, weights, acts, blocked, splits)
    within_activation(got, want)
    alone = fused(x[2:3].clone(), whole, weights, acts, blocked, splits)
    assert torch.equal(alone.view(torch.int16), got[2:3].view(torch.int16))


@pytest.mark.parametrize("fmt", FORMATS)
def test_the_prompt_tile_gives_the_decode_tiles_glu_bits(fmt):
    weights, acts, blocked = fmt
    inter, k = 136, 1024
    gate, up, whole = stored_pair(weights, inter, k, blocked)
    x = activations(150, k)
    want = product(apart(x, gate, weights, acts, blocked, 1, True), apart(x, up, weights, acts, blocked, 1, True))
    got = fused(x, whole, weights, acts, blocked, 1, True)
    within_activation(got, want)
    decode = fused(x[:40].clone(), whole, weights, acts, blocked, 1)
    assert torch.equal(decode.view(torch.int16), got[:40].view(torch.int16))


@FP4
@pytest.mark.parametrize("splits", [1, 2])
def test_glu_on_the_fp4_mma(splits):
    inter, k = 136, 1024
    gate, up, whole = stored_pair("nvfp4", inter, k, True)
    x = activations(5, k)

    def run(stored, glu):
        w, scales, row_scale = stored
        m, n = x.shape[0], w.shape[0]
        out = torch.empty(m, n // 2 if glu else n, dtype=torch.bfloat16, device=dev)
        scratch = torch.empty(mr.scratch_bytes(m, k, "nvfp4_mma"), dtype=torch.uint8, device=dev)
        partial = torch.empty(max(1, mr.partial_elements(m, n, splits)), dtype=torch.float32, device=dev)
        counters = torch.zeros(mr.row_tiles(n, 1, 4), dtype=torch.int32, device=dev)
        return mr.mm_rows(x, w, out, "nvfp4", scales, row_scale, "nvfp4_mma", INPUT_SCALE, True, 1, 4, splits,
                          scratch, partial, counters, glu=glu)

    within_activation(run(whole, True), product(run(gate, False), run(up, False)))


def test_glu_refuses_what_it_cannot_pair():
    w, scales, row_scale = matrix("bf16", 24, 256)[0]
    x = activations(2, 256)
    with pytest.raises(RuntimeError):
        mr.mm_rows(x, w, torch.empty(2, 12, dtype=torch.bfloat16, device=dev), "bf16", glu=True)
    with pytest.raises(ValueError):
        mr.interleave_glu(torch.zeros(12, 4), torch.zeros(12, 4))
