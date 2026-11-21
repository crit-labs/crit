"""The FP8 tensor-core arithmetic (arithmetic="fp8", structures fp8-decode and fp8-prompt) for fp8:channel and
fp8:block bundles against the FP64 reference of the same quantized operands (crit's declared group quantizer: each 128
values of a row in E4M3 at amax / 448), and the bits a route keeps: the same in decode-width (8) and prompt-width (32,
64) items, alone and beside other tokens, through both plans, over long reductions, in a captured CUDA graph and in a
build with other knobs."""
import pytest
import torch

from crit_kernels.build import load
from test_grouped_experts import Experts, dev, gx, routes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not gx.fp8_available(),
                                reason="the E4M3 tensor cores (8.9 on)")
FORMATS = ["fp8:channel", "fp8:block"]


def quantize(v):
    """crit's declared FP8 group quantizer as values: each 128 values of a row in E4M3 at scale amax / 448 (FP32), the
    nearest code (ties to even), the value code x scale."""
    rows, cols = v.shape
    g = v.float().reshape(rows, cols // 128, 128)
    scale = g.abs().amax(-1, keepdim=True) / 448.0
    codes = torch.where(scale > 0, g / torch.where(scale > 0, scale, torch.ones_like(scale)), torch.zeros_like(g))
    codes = codes.clamp(-448, 448).to(torch.float8_e4m3fn).double()
    return (codes * scale.double()).reshape(rows, cols)


def reference(experts, x, ids, w):
    x = x.bfloat16().double().cpu()
    out = torch.zeros(x.shape[0], experts.H, dtype=torch.float64)
    for t in range(x.shape[0]):
        xq = quantize(x[t:t + 1])[0]
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            if e < 0:
                continue
            gu = experts.gate_up[e] @ xq
            h = torch.nn.functional.silu(gu[:experts.I]) * gu[experts.I:]
            h = quantize(h[None].float().bfloat16().double())[0]
            out[t] += float(w[t, j]) * (experts.down[e] @ h)
    return out


def rows(experts, x, ids, width=None, module=gx):
    ws = module.Workspace(ids.shape[0], ids.shape[1], experts.E, experts.H, experts.I, dev, width)
    return module.route_outputs(x, ids, experts.store, None, experts.layout, ws, False, "fp8").clone()


def forward(experts, x, ids, w):
    ws = gx.Workspace(ids.shape[0], ids.shape[1], experts.E, experts.H, experts.I, dev)
    return gx.forward(x, ids, w, experts.store, None, experts.layout, ws, False, None, "fp8")


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("H,I", [(256, 128), (384, 256)])
def test_fp8_matches_the_fp64_reference_of_the_quantized_operands(fmt, H, I):
    experts = Experts(fmt, H=H, I=I)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(13, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(13, 3, experts.E, g)
    out = forward(experts, x, ids, w).double().cpu()
    ref = reference(experts, x, ids.cpu(), w.cpu())
    scale = ref.abs().max()
    assert (out - ref).abs().max() <= 1e-2 * scale, (fmt, float((out - ref).abs().max() / scale))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("H,I,T,top", [(256, 128, 300, 4), (2048, 1152, 40, 2)])
def test_a_fp8_route_keeps_its_bits_in_decode_and_prompt_items(fmt, H, I, T, top):
    experts = Experts(fmt, E=16, H=H, I=I, seed=1)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(T, experts.H, generator=g).bfloat16().to(dev)
    ids, _ = routes(T, top, experts.E, g)
    full = rows(experts, x, ids)
    valid = (ids.reshape(-1) >= 0).cpu()
    assert bool(torch.isfinite(full[valid]).all())
    for width in (8, 32, 64):
        for first, count in ((0, 1), (17, 1), (3, 7), (T // 3, T - T // 3)):
            part = rows(experts, x[first:first + count].clone(), ids[first:first + count].clone(), width)
            sel = valid[first * top:(first + count) * top]
            assert torch.equal(part[sel], full[first * top:(first + count) * top][sel]), (fmt, width, first, count)


@pytest.mark.kernels_ci
def test_a_captured_fp8_call_replays_new_routes():
    experts = Experts("fp8:block", E=8, seed=11)
    g = torch.Generator().manual_seed(12)
    tokens = 40
    x = torch.zeros(tokens, experts.H, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(tokens, 2, dtype=torch.int64, device=dev)
    w = torch.zeros(tokens, 2, device=dev)
    ws = gx.Workspace(tokens, 2, experts.E, experts.H, experts.I, dev)
    out = torch.empty(tokens, experts.H, device=dev)
    gx.forward(x, ids, w, experts.store, None, experts.layout, ws, False, out, "fp8")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gx.forward(x, ids, w, experts.store, None, experts.layout, ws, False, out, "fp8")
    for _ in range(3):
        x.copy_(torch.randn(tokens, experts.H, generator=g).bfloat16())
        new_ids, new_w = routes(tokens, 2, experts.E, g)
        ids.copy_(new_ids), w.copy_(new_w)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, forward(experts, x.clone(), ids.clone(), w.clone()))


@pytest.mark.kernels_ci
def test_fp8_refuses_other_formats():
    experts = Experts("nvfp4", a4=True)
    x = torch.zeros(2, experts.H, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(2, 1, dtype=torch.int64, device=dev)
    with pytest.raises(ValueError, match="fp8:channel and fp8:block"):
        rows(experts, x, ids)


OTHER = {"fp8-decode": {"WARPS": 8, "TILES": 4, "DEPTH": 1, "FETCH": 1},
         "fp8-prompt": {"WARPS": 2, "TILES": 4, "DEPTH": 3, "CHUNK": 128}}


@pytest.mark.kernels_ci
def test_other_fp8_knobs_give_every_route_the_same_bits():
    other = load("grouped-experts", knobs=OTHER)
    for fmt in FORMATS:
        experts = Experts(fmt, E=5, H=1024, I=384, seed=7)
        g = torch.Generator().manual_seed(9)
        x = torch.randn(37, experts.H, generator=g).bfloat16().to(dev)
        ids, _ = routes(37, 3, experts.E, g, holes=False)
        for width in (8, 32, 64):
            want = rows(experts, x, ids, width)
            got = rows(experts, x, ids, width, other)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (fmt, width)
