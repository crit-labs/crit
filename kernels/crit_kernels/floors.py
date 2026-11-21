"""A call's floor on a device: the time the bytes it must move take at the device's memory bandwidth, or the time its
multiply-adds take at its arithmetic's tensor rate, whichever is longer. A kernel report states each structure's time
as a share of its floor (`of_floor`, floor / time: 1.0 is the device's limit), so a kernel far from its device's
limit reads that way whatever its yardstick library does.

A bench gives the work: `bytes(shape, rows)` (every bench) and, where the call is compute-bound at some row count,
`flops(shape, rows)` -> (multiply-adds x 2, arithmetic format: "bf16", "fp16", "fp8", "nvfp4", "fp32"). The device's
ceilings come from a `crit eval probe` result (its bandwidth_gbs, tflops and mma_tflops: the larger of the GEMM and
raw tensor-core rates per format) or, without one, from measuring here: a 1 GiB copy for the bandwidth and cuBLAS
matmuls for the BF16, FP16 and FP8 rates (a GEMM library's rate, below the raw tensor-core peak, so such a floor is a
little generous). A format with no rate leaves a compute-bound call's floor at its memory time, marked unpriced."""
import json
from pathlib import Path

FORMATS = ("bf16", "fp16", "fp8", "nvfp4", "fp32")


def from_probe(probe, device=0):
    """Ceilings from a `crit eval probe` result (a file or its dict; a multi-device result's device `device`):
    {"bandwidth_gbs", "tflops": {format: rate}, "source"}."""
    found = json.loads(Path(probe).read_text()) if isinstance(probe, (str, Path)) else dict(probe)
    if isinstance(found.get("devices"), list):
        found = found["devices"][device]
    found = found.get("device", found)
    rates = {}
    for key in ("tflops", "mma_tflops"):
        for name, value in (found.get(key) or {}).items():
            if value and value > rates.get(name, 0):
                rates[name] = float(value)
    if not found.get("bandwidth_gbs"):
        raise ValueError("the probe result has no bandwidth_gbs")
    return {"bandwidth_gbs": float(found["bandwidth_gbs"]), "tflops": rates, "source": "crit eval probe"}


def _best_us(call, reps=5):
    import torch
    call()
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        call()
        b.record()
        torch.cuda.synchronize()
        us = a.elapsed_time(b) * 1000
        best = us if best is None or us < best else best
    return best


def measure(device="cuda"):
    """Ceilings measured on `device`: the bandwidth the faster of a 1 GiB device-to-device copy (2 GiB moved) and a
    sum over 1 GiB (reads only, as most of the kernels' calls) reaches, cuBLAS matmuls of 8192 x 8192 x 8192 for BF16
    and FP16 and torch._scaled_mm for FP8 where it runs."""
    import torch
    x = torch.empty(1 << 30, dtype=torch.uint8, device=device)
    y = torch.empty_like(x)
    copy = 2 * x.numel() / _best_us(lambda: y.copy_(x)) / 1e3
    words = x.view(torch.float32).zero_()
    read = x.numel() / _best_us(lambda: words.sum()) / 1e3
    bandwidth = max(copy, read)
    x = y = words = None   # freed before the matmuls
    n = 8192
    rates = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        a = torch.randn(n, n, device=device).to(dtype)
        b = torch.randn(n, n, device=device).to(dtype)
        rates[name] = 2 * n ** 3 / _best_us(lambda: a @ b) / 1e6
    try:
        a = torch.randn(n, n, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=device).to(torch.float8_e4m3fn).t()
        one = torch.ones((), device=device)
        rates["fp8"] = 2 * n ** 3 / _best_us(lambda: torch._scaled_mm(a, b, one, one, out_dtype=torch.bfloat16)) / 1e6
    except Exception:   # noqa: BLE001 - no FP8 GEMM here: FP8 calls stay unpriced
        pass
    torch.cuda.empty_cache()
    return {"bandwidth_gbs": bandwidth, "tflops": rates, "source": "measured (copy, cuBLAS)"}


def floor(bench, shape, rows, ceilings):
    """{"floor_us", "bound": "memory" | "compute", "memory_us", "compute_us"?, "unpriced"?} of one call, or None when
    the bench gives no work."""
    if not hasattr(bench, "bytes"):
        return None
    memory = bench.bytes(shape, rows) / (ceilings["bandwidth_gbs"] * 1e3)
    out = {"memory_us": round(memory, 3), "floor_us": round(memory, 3), "bound": "memory"}
    if hasattr(bench, "flops"):
        flops, fmt = bench.flops(shape, rows)
        rate = ceilings["tflops"].get(fmt)
        if rate:
            compute = flops / (rate * 1e6)
            out["compute_us"] = round(compute, 3)
            if compute > memory:
                out["floor_us"], out["bound"] = round(compute, 3), "compute"
        else:
            out["unpriced"] = fmt
    return out


def stamp(entry, bench, shape, rows, ceilings):
    """Adds the floor and of_floor (floor / the entry's time) to a report or sweep entry timed at entry["us"] (or
    "tuned_us")."""
    found = floor(bench, shape, rows, ceilings)
    us = entry.get("us", entry.get("tuned_us"))
    if not found or not us:
        return entry
    entry.update(found)
    entry["of_floor"] = round(found["floor_us"] / us, 3)
    return entry


__all__ = ["FORMATS", "floor", "from_probe", "measure", "stamp"]
