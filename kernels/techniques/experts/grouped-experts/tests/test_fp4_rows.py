"""The FP4 tensor-core arithmetic at a scale of each route's own (arithmetic="fp4-row": one FP4 term a value;
"fp4-row2": two, the first's remainder at the same scale), which needs no input scales: close to the FP64 reference of
the BF16 activations and the stored weights (two terms closer than one), and the bits a route keeps, the same in
decode-width (8) and prompt-width (32, 64) items, alone and beside other tokens."""
import pytest
import torch

from test_grouped_experts import Experts, dev, gx, routes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not gx.fp4_available(),
                                reason="the FP4 tensor cores (the sm_120 family)")


def rows(experts, x, ids, arithmetic, width=None, tile=False):
    ws = gx.Workspace(ids.shape[0], ids.shape[1], experts.E, experts.H, experts.I, dev, width, tile=tile)
    return gx.route_outputs(x, ids, experts.store, None, experts.layout, ws, False, arithmetic).clone()


def combined(experts, x, ids, w, arithmetic):
    y = rows(experts, x, ids, arithmetic).double().cpu().view(ids.shape[0], ids.shape[1], -1)
    keep = (ids >= 0).cpu().double()[..., None]
    return (y * keep * w.cpu().double()[..., None]).sum(1)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("H,I,caps", [(256, 128, None), (2304, 1152, None), (256, 128, (4.0, 25.0))])
def test_row_scaled_fp4_is_close_and_two_terms_closer(H, I, caps):
    # no input scales; caps: a SwiGLU whose gate and up are capped (Inkling 2's experts)
    experts = Experts("nvfp4", E=6, H=H, I=I, seed=2, **({"activation": "softcapped_silu", "caps": caps} if caps else {}))
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(21, H, generator=g) * torch.rand(21, 1, generator=g) * 4).bfloat16().to(dev)
    ids, w = routes(21, 3, experts.E, g)
    ref = experts.reference(x, ids.cpu(), w.cpu())
    scale = ref.abs().max()
    err = {a: float((combined(experts, x, ids, w, a) - ref).abs().max() / scale) for a in ("fp4-row", "fp4-row2")}
    assert err["fp4-row"] < 0.2, err    # W4A4: about 14% of the largest output at worst on random rows
    assert err["fp4-row2"] < 0.03, err   # about 8 bits a value: about 1.5%
    assert err["fp4-row2"] < err["fp4-row"] / 4, err


@pytest.mark.kernels_ci
@pytest.mark.parametrize("arithmetic", ["fp4-row", "fp4-row2"])
def test_a_row_scaled_fp4_route_keeps_its_bits_in_decode_and_prompt_items(arithmetic):
    experts = Experts("nvfp4", E=16, H=640, I=384, seed=1)
    g = torch.Generator().manual_seed(5)
    T, top = 90, 4
    x = torch.randn(T, experts.H, generator=g).bfloat16().to(dev)
    ids, _ = routes(T, top, experts.E, g)
    full = rows(experts, x, ids, arithmetic)
    valid = (ids.reshape(-1) >= 0).cpu()
    assert bool(torch.isfinite(full[valid]).all())
    for width, tile in ((8, False), (32, False), (64, False), (64, True)):   # tile: fp4-tile's items
        for first, count in ((0, 1), (17, 1), (3, 7), (T // 3, T - T // 3)):
            part = rows(experts, x[first:first + count].clone(), ids[first:first + count].clone(), arithmetic, width,
                        tile)
            sel = valid[first * top:(first + count) * top]
            assert torch.equal(part[sel], full[first * top:(first + count) * top][sel]), (width, tile, first, count)


def test_a_workspace_takes_the_block_tile_for_items_of_64_alone():
    # the width rule: 32 from 8 routes an expert, else 8 (64 on request); tile only with items of 64
    assert [gx.width_for(r, 128) for r in (128 * 128, 8 * 128, 7 * 128)] == [32, 32, 8]
    assert gx.Workspace(2048, 8, 128, 256, 256, dev, width=64, tile=True).tile
    assert not gx.Workspace(2048, 8, 128, 256, 256, dev, tile=True).tile   # 32-route items: fp4-prompt
