"""The starter matmul against FP64, and a column alone equal to the same column in a call of several. The kernel is
built for this GPU by crit_kernels.build."""
import torch

from crit_kernels.build import load

ws = load("workbench-starter")
dev = torch.device("cuda")


def test_bf16_and_fp8_against_fp64_and_columns_alone():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(96, 256, generator=g) * 0.05
    x = torch.randn(8, 256, generator=g)
    wb, xb = w.to(dev, torch.bfloat16), x.to(dev, torch.bfloat16)
    y = ws.run_bf16(xb, wb)
    exact = xb.double() @ wb.double().t()
    assert float((y.double() - exact).abs().max() / exact.abs().max()) < 1e-5
    for c in range(8):
        assert torch.equal(ws.run_bf16(xb[c:c + 1].contiguous(), wb)[0], y[c])
    w_scale, x_scale = w.abs().amax(1) / 448, x.abs().amax(1) / 448
    w8 = (w / w_scale[:, None]).to(torch.float8_e4m3fn).to(dev)
    x8 = (x / x_scale[:, None]).to(torch.float8_e4m3fn).to(dev)
    y8 = ws.run_fp8(x8, x_scale.to(dev), w8, w_scale.to(dev))
    exact8 = (x8.double() * x_scale.to(dev).double()[:, None]) @ (w8.double() * w_scale.to(dev).double()[:, None]).t()
    assert float((y8.double() - exact8).abs().max() / exact8.abs().max()) < 1e-5
