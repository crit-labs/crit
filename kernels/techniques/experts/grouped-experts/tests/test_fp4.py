"""The FP4 tensor-core arithmetic (arithmetic="fp4", structures fp4-decode and fp4-prompt) against the FP64 reference
of the same quantized operands, and the bits a route keeps: the same in decode-width (8) and prompt-width (32, 64)
items, alone and beside other tokens, through the one-block plan and the sorted one, over long reductions, in a
captured CUDA graph and in a build with other knobs. Its operands are crit's declared NVFP4 quantizer's: the codes and
scales decode to the BF16 path's W4A4 operands exactly."""
import pytest
import torch

from crit_kernels.build import load
from test_grouped_experts import E2M1_SIGNED, Experts, dev, gx, routes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not gx.fp4_available(),
                                reason="the FP4 tensor cores (the sm_120 family)")


def fp4_rows(experts, x, ids, width=None, module=gx, tile=False):
    ws = module.Workspace(ids.shape[0], ids.shape[1], experts.E, experts.H, experts.I, dev, width, tile=tile)
    return module.route_outputs(x, ids, experts.store, None, experts.layout, ws, True, "fp4").clone()


def fp4_forward(experts, x, ids, w):
    ws = gx.Workspace(ids.shape[0], ids.shape[1], experts.E, experts.H, experts.I, dev)
    return gx.forward(x, ids, w, experts.store, None, experts.layout, ws, True, None, "fp4")


def decoded(codes, scales):
    """[routes][K] FP64 of E2M1 codes (two a byte, low nibble first) times their UE4M3 block scales."""
    c = codes.cpu()
    v = E2M1_SIGNED[torch.stack([(c & 15).long(), (c >> 4).long()], -1).reshape(c.shape[0], -1)]
    s = scales.cpu().view(torch.float8_e4m3fn).double().repeat_interleave(16, 1)
    return v * s


@pytest.mark.kernels_ci
@pytest.mark.parametrize("H,I", [(256, 128), (384, 256)])
def test_fp4_matches_the_fp64_reference_of_the_quantized_operands(H, I):
    experts = Experts("nvfp4", H=H, I=I, a4=True)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(13, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(13, 3, experts.E, g)
    out = fp4_forward(experts, x, ids, w).double().cpu()
    ref = experts.reference(x, ids.cpu(), w.cpu())
    scale = ref.abs().max()
    assert (out - ref).abs().max() <= 3e-2 * scale, float((out - ref).abs().max() / scale)
    # The BF16 path's W4A4 on the same operands: close, another order of the sums (not its bits)
    bf16 = experts.forward(x, ids, w).double().cpu()
    assert (out - bf16).abs().max() <= 2e-2 * scale


@pytest.mark.kernels_ci
def test_the_fp4_operands_are_the_declared_quantizers_codes():
    # The routes' input codes and scales decode to the operands a4_inputs gives the BF16 path (e2m1 x decoded scale,
    # exact in BF16), bit for bit.
    experts = Experts("nvfp4", E=4, H=512, I=128, a4=True, seed=4)
    g = torch.Generator().manual_seed(8)
    x = (torch.randn(9, experts.H, generator=g) * torch.rand(9, 1, generator=g) * 10).bfloat16().to(dev)
    x[2, 16:32] = 0.0   # a block of zeros: scale 0, codes 0
    ids, _ = routes(9, 2, experts.E, g, holes=False)
    ws = gx.Workspace(9, 2, experts.E, experts.H, experts.I, dev, 8)
    gx.route_outputs(x, ids, experts.store, None, experts.layout, ws, True, "fp4")
    f = ws.fp4()
    ref = gx.Workspace(9, 2, experts.E, experts.H, experts.I, dev, 8)
    gx.route_outputs(x, ids, experts.store, None, experts.layout, ref, True)
    assert torch.equal(decoded(f["x"], f["xs"]), ref.xa.double().cpu())


@pytest.mark.kernels_ci
@pytest.mark.parametrize("H,I,T,top", [(256, 128, 300, 4), (2048, 1152, 40, 2), (640, 384, 90, 8)])
def test_a_fp4_route_keeps_its_bits_in_decode_and_prompt_items(H, I, T, top):
    # Decode items (8 routes, operands straight from global memory) and prompt items (32 and 64, staged a chunk at a
    # time), the one-block plan and the sorted one: every route the same bits, long reductions included.
    experts = Experts("nvfp4", E=16, H=H, I=I, a4=True, seed=1)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(T, experts.H, generator=g).bfloat16().to(dev)
    ids, _ = routes(T, top, experts.E, g)
    full = fp4_rows(experts, x, ids)
    valid = (ids.reshape(-1) >= 0).cpu()
    assert bool(torch.isfinite(full[valid]).all())
    for width, tile in ((8, False), (32, False), (64, False), (64, True)):   # tile: fp4-tile
        for first, count in ((0, 1), (17, 1), (3, 7), (T // 3, T - T // 3)):
            part = fp4_rows(experts, x[first:first + count].clone(), ids[first:first + count].clone(), width, tile=tile)
            sel = valid[first * top:(first + count) * top]
            assert torch.equal(part[sel], full[first * top:(first + count) * top][sel]), (width, tile, first, count)
    assert torch.equal(fp4_rows(experts, x, ids)[valid], full[valid])


@pytest.mark.kernels_ci
@pytest.mark.parametrize("tokens", [3, 80])
def test_a_captured_fp4_call_replays_new_routes(tokens):
    experts = Experts("nvfp4", E=8, a4=True, seed=11)
    g = torch.Generator().manual_seed(12)
    x = torch.zeros(tokens, experts.H, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(tokens, 2, dtype=torch.int64, device=dev)
    w = torch.zeros(tokens, 2, device=dev)
    ws = gx.Workspace(tokens, 2, experts.E, experts.H, experts.I, dev)
    out = torch.empty(tokens, experts.H, device=dev)
    gx.forward(x, ids, w, experts.store, None, experts.layout, ws, True, out, "fp4")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gx.forward(x, ids, w, experts.store, None, experts.layout, ws, True, out, "fp4")
    for _ in range(3):
        x.copy_(torch.randn(tokens, experts.H, generator=g).bfloat16())
        new_ids, new_w = routes(tokens, 2, experts.E, g)
        ids.copy_(new_ids), w.copy_(new_w)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, fp4_forward(experts, x.clone(), ids.clone(), w.clone()))


@pytest.mark.kernels_ci
def test_a_gelu_gated_fp4_block_matches_its_reference():
    experts = Experts("nvfp4", a4=True, activation="gelu_tanh", seed=13)
    g = torch.Generator().manual_seed(14)
    x = torch.randn(21, experts.H, generator=g).bfloat16().to(dev)
    ids, w = routes(21, 3, experts.E, g)
    out = fp4_forward(experts, x, ids, w).double().cpu()
    ref = experts.reference(x, ids.cpu(), w.cpu())
    assert (out - ref).abs().max() <= 3e-2 * ref.abs().max()


@pytest.mark.kernels_ci
def test_a_non_finite_input_block_reaches_only_its_own_routes():
    # A NaN in a token's input: its scale code is the format's NaN, so the token's routes carry it; every other
    # token keeps the bits it has alone.
    experts = Experts("nvfp4", E=6, a4=True, seed=15)
    g = torch.Generator().manual_seed(16)
    x = torch.randn(10, experts.H, generator=g).bfloat16().to(dev)
    ids, _ = routes(10, 2, experts.E, g, holes=False)
    clean = fp4_rows(experts, x, ids, 8)
    x[4, 37] = float("nan")
    got = fp4_rows(experts, x, ids, 8)
    assert bool(torch.isnan(got[8:10]).all())
    keep = torch.ones(20, dtype=torch.bool)
    keep[8:10] = False
    assert torch.equal(got[keep], clean[keep])


@pytest.mark.kernels_ci
def test_fp4_refuses_other_formats_and_widths():
    experts = Experts("nvfp4", H=320, I=192, a4=True)   # 64 mod 128
    x = torch.zeros(2, experts.H, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(2, 1, dtype=torch.int64, device=dev)
    with pytest.raises(ValueError, match="multiples of 128"):
        fp4_rows(experts, x, ids)
    other = Experts("int4", a4=False)
    with pytest.raises(ValueError, match="nvfp4"):
        fp4_rows(other, torch.zeros(2, other.H, dtype=torch.bfloat16, device=dev), ids)


OTHER = {"fp4-decode": {"WARPS": 8, "TILES": 4, "DEPTH": 1, "FETCH": 1},
         "fp4-prompt": {"WARPS": 2, "TILES": 4, "DEPTH": 3, "CHUNK": 128}}


@pytest.mark.kernels_ci
def test_other_fp4_knobs_give_every_route_the_same_bits():
    other = load("grouped-experts", knobs=OTHER)
    experts = Experts("nvfp4", E=5, H=1024, I=384, seed=7, a4=True)
    g = torch.Generator().manual_seed(9)
    x = torch.randn(37, experts.H, generator=g).bfloat16().to(dev)
    ids, _ = routes(37, 3, experts.E, g, holes=False)
    for width in (8, 32, 64):
        want = fp4_rows(experts, x, ids, width)
        got = fp4_rows(experts, x, ids, width, other)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), width


@pytest.mark.kernels_ci
@pytest.mark.parametrize("H, I", [(1024, 384), (2560, 640)])
def test_fp4_tiles_two_mainloops_give_every_route_the_same_bits(H, I):
    # fp4-tile's block ring (PRODUCER 1) and its warp-specialized form (2: TMA boxes of the weights, mbarriers), each
    # pass, a partly filled last item and several k parts
    ring = load("grouped-experts", knobs={"fp4-tile": {"PRODUCER": 1}})
    ws = load("grouped-experts", knobs={"fp4-tile": {"PRODUCER": 2, "TILES": 2}})
    experts = Experts("nvfp4", E=6, H=H, I=I, seed=11, a4=True)
    g = torch.Generator().manual_seed(13)
    x = torch.randn(150, experts.H, generator=g).bfloat16().to(dev)
    ids, _ = routes(150, 4, experts.E, g, holes=True)
    want = fp4_rows(experts, x, ids, 64, gx, tile=True)
    for other in (ring, ws):
        got = fp4_rows(experts, x, ids, 64, other, tile=True)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))
