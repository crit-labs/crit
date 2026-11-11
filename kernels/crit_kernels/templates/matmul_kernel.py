"""{name}: y[columns, N] = x[columns, K] @ w[N, K]^T in {format} ({op}), on the workbench (see harness.py).

candidate() builds src/, a torch extension project (kernel-builder's template, or the project --from copied: crit's
kernels/starters/workbench-starter is a correct, slow starting point), with torch's extension builder and calls its op
run_{format}:
register that op in src/torch-ext/torch_binding.cpp (bf16: run_bf16(Tensor x, Tensor w) -> Tensor; fp8:
run_fp8(Tensor x, Tensor x_scale, Tensor w, Tensor w_scale) -> Tensor; nvfp4: run_nvfp4(Tensor x, Tensor w, Tensor
w_scale, Tensor w_global) -> Tensor, FP32 out). Keep inputs() and reference() as the contract, and keep yardstick() on
the best existing provider for the same arithmetic. A kernel that proves out belongs in crit's kernel templates,
with its speed constants as knobs (kernels/TEMPLATES.md).
"""
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FORMAT = "{format}"
_ext = None


def ext():
    """The kernel's ops (torch.ops.{module}): src/'s CUDA and C++ sources (not its tests or benchmarks) compiled once
    with torch's extension builder into build/, CUDA_KERNEL defined as kernel-builder defines it."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        src = HERE / "src"
        skip = {"tests", "benchmarks", "build", "result"}
        sources = sorted(str(p) for p in src.rglob("*") if p.suffix in (".cu", ".cpp")
                         and not skip & set(p.relative_to(src).parts))
        includes = sorted({str(src), str(src / "torch-ext")} | {str(Path(p).parent) for p in sources})
        includes.append(r"{core_include}")   # crit's kernel core: #include "crit_kcore/<area>.cuh"
        (HERE / "build").mkdir(exist_ok=True)
        load(name="{module}", sources=sources, extra_include_paths=includes, build_directory=str(HERE / "build"),
             extra_cflags=["-DCUDA_KERNEL"], extra_cuda_cflags=["-O3", "-DCUDA_KERNEL"], verbose=False)
        _ext = getattr(torch.ops, "{module}")
    return _ext


# E2M1 values by their 3-bit magnitude code; bit 3 is the sign.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4(w):
    """w as stored NVFP4 (compressed-tensors): E2M1 pairs, low nibble first, E4M3 scales per 16 inputs and an FP32
    global divisor, so w ~ e2m1 * scale / global."""
    n, k = w.shape
    global_ = (448 * 6 / w.abs().amax().clamp_min(1e-12)).float()
    blocks = w.reshape(n, k // 16, 16)
    scale = (blocks.abs().amax(2) / 6 * global_).to(torch.float8_e4m3fn)
    real = (scale.float() / global_).clamp_min(1e-12).unsqueeze(2)
    target = (blocks / real).clamp(-6, 6)
    codes = (target.abs().unsqueeze(3) - E2M1).abs().argmin(3) | ((target < 0).to(torch.int64) << 3)
    codes = codes.reshape(n, k).to(torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), scale, global_.reshape(1)


def widened(packed, scale, global_):
    """Stored NVFP4 decoded exactly, in float64."""
    codes = torch.stack([packed & 15, packed >> 4], 2).reshape(packed.shape[0], -1).long().cpu()
    values = E2M1.double()[codes & 7] * (1 - 2 * (codes >> 3)).double()
    return values * scale.cpu().double().repeat_interleave(16, 1) / global_.cpu().double()


def inputs(shape, columns, seed):
    n, k = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.05
    x = torch.randn(columns, k, generator=g)
    if FORMAT == "nvfp4":
        # W4A16: stored NVFP4 weights, bf16 activations; the yardstick's BF16 copy of the weights is made once.
        packed, scale, global_ = nvfp4(w)
        return {"w": packed.cuda(), "w_scale": scale.cuda(), "w_global": global_.cuda(), "x": x.bfloat16().cuda(),
                "w_bf16": widened(packed, scale, global_).bfloat16().cuda()}
    if FORMAT == "fp8":
        # W8A8: E4M3 weights with one scale per row, E4M3 activations with one scale per column (token).
        w_scale = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
        x_scale = x.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
        return {"w": (w / w_scale).to(torch.float8_e4m3fn).cuda(), "w_scale": w_scale.float().cuda(),
                "x": (x / x_scale).to(torch.float8_e4m3fn).cuda(), "x_scale": x_scale.float().cuda()}
    return {"w": w.bfloat16().cuda(), "x": x.bfloat16().cuda()}


def select(data, column):
    """The inputs of one column alone (for the column invariance check)."""
    out = dict(data)
    out["x"] = data["x"][column:column + 1]
    if "x_scale" in data:
        out["x_scale"] = data["x_scale"][column:column + 1]
    return out


def reference(data):
    if FORMAT == "nvfp4":
        return data["x"].double() @ widened(data["w"], data["w_scale"], data["w_global"]).to(data["x"].device).T
    if FORMAT == "fp8":
        x = data["x"].double() * data["x_scale"].double()
        w = data["w"].double() * data["w_scale"].double()
        return x @ w.T
    return data["x"].double() @ data["w"].double().T


def candidate(data):
    if FORMAT == "nvfp4":
        return ext().run_nvfp4(data["x"], data["w"], data["w_scale"], data["w_global"])
    if FORMAT == "fp8":
        return ext().run_fp8(data["x"], data["x_scale"].reshape(-1).contiguous(), data["w"],
                             data["w_scale"].reshape(-1).contiguous())
    return ext().run_bf16(data["x"], data["w"])


def yardstick(data):
    """{yardstick}"""
    if FORMAT == "nvfp4":
        return (data["x"] @ data["w_bf16"].T).float()
    if FORMAT == "fp8":
        return torch._scaled_mm(data["x"], data["w"].T, scale_a=data["x_scale"], scale_b=data["w_scale"].T,
                                out_dtype=torch.float32)
    return (data["x"] @ data["w"].T).float()


def work(shape, columns):
    n, k = shape
    if FORMAT == "nvfp4":   # half a byte per weight and an E4M3 scale per 16, bf16 activations
        return {"bytes": n * k // 2 + n * k // 16 + columns * k * 2 + columns * n * 4, "flops": 2 * n * k * columns}
    width = 1 if FORMAT == "fp8" else 2
    return {"bytes": n * k * width + columns * k * width + columns * n * 4, "flops": 2 * n * k * columns}
