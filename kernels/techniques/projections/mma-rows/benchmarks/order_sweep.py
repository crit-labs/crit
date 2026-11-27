"""mma-rows' order sweep: the splits of K each matrix's sums take (torch-ext/mma_rows/orders.py), chosen by shape.

    python benchmarks/order_sweep.py WEIGHTS:N:K [WEIGHTS:N:K ...] [--out sweep.json] [--write]

For each shape (weights nvfp4, fp8, mxfp8 or bf16) and each MMA it runs on (bf16; fp4 too for NVFP4 where the device
has the FP4 MMA) it times every allowed split count (each split at least two steps of K, at most MAX_SPLITS) on
decode calls of 1, 2, 4, 8 and 16 rows and prompt calls of 512 and 2,048 rows (CUDA events, the least of REPEATS
interleaved passes) and keeps the count whose worst ratio to the fastest count, over those seven calls, is least
(fewer splits on a tie): one count a matrix for every row count, since a row's bits must not follow the row count,
and no call much slower than its best. --write merges the choices into orders.py (shapes swept before keep theirs),
whose header names the device, the date and this tool.
The table is shipped and read on every device: the order is chosen once, never per GPU."""
import argparse
import datetime
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from crit_kernels import build  # noqa: E402

ROWS = (1, 2, 4, 8, 16)
PROMPTS = (512, 2048)
REPEATS = 3   # passes over every count, each call's least time kept
CANDIDATES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)
TABLE = HERE / "torch-ext" / "mma_rows" / "orders.py"


def matrix(mr, weights, n, k, blocked, device):
    g = torch.Generator(device="cpu").manual_seed(n * 7 + k)
    if weights == "nvfp4":
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g).to(device)
        scales = (torch.rand(n, k // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn).view(torch.uint8).to(device)
        if blocked:
            from bench import _blocked
            scales = _blocked(scales)
        return codes, scales, torch.full((n,), 0.01, device=device)
    if weights in ("fp8", "mxfp8"):
        codes = (torch.randn(n, k, generator=g) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8).to(device)
        scales = (torch.full((n, k // 32), 127, dtype=torch.uint8, device=device) if weights == "mxfp8" else None)
        return codes, scales, torch.full((n,), 0.01, device=device)
    if weights == "bf16":
        return (torch.randn(n, k, generator=g) * 0.02).to(device, torch.bfloat16), None, None
    raise ValueError(f"the order sweep makes nvfp4, fp8, mxfp8 and bf16 matrices, not {weights}")


def timed(fn, reps=60):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1000.0


def sweep(mr, shapes, device):
    partial = torch.empty(32 << 20, dtype=torch.float32, device=device)
    counters = torch.zeros(1 << 14, dtype=torch.int32, device=device)
    rows, chosen = [], {}
    for weights, n, k in shapes:
        for mma in (("bf16", "fp4") if weights == "nvfp4" and mr.fp4_mma(device) else ("bf16",)):
            acts = "nvfp4_mma" if mma == "fp4" else "bf16"
            blocked = mma == "fp4"
            w, scales, row_scale = matrix(mr, weights, n, k, blocked, device)
            groups, warps, _ = mr.plan(n, k, weights, device, acts=acts)
            steps = k // mr.FP4_STEP if mma == "fp4" else math.ceil(k / mr.VALUES[weights])
            scratch = torch.empty(max(16, mr.scratch_bytes(max(PROMPTS), k, acts if acts != "bf16" else "fp8_rows")),
                                  dtype=torch.uint8, device=device)
            allowed = [splits for splits in CANDIDATES if splits == 1 or splits <= steps // 2]
            measured = {splits: {"weights": weights, "mma": mma, "N": n, "K": k, "splits": splits, "us": {}}
                        for splits in allowed}
            # REPEATS passes over every count in turn, each call's least time kept: a pass the device ran slower
            # (clocks, another program) moves every count alike, and one outlier does not decide the order
            for _ in range(REPEATS):
                for splits in allowed:
                    times = measured[splits]["us"]
                    for m in ROWS:
                        x = torch.randn(m, k, generator=torch.Generator().manual_seed(m + k)).to(device,
                                                                                              torch.bfloat16)
                        out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
                        found = timed(lambda: mr.mm_rows(x, w, out, weights, scales, row_scale, acts, 500.0, blocked,
                                                         groups, warps, splits, scratch, partial, counters), 100)
                        times[m] = round(min(found, times.get(m, found)), 2)
                    for m in PROMPTS:
                        x = torch.randn(m, k, generator=torch.Generator().manual_seed(m + k)).to(device,
                                                                                              torch.bfloat16)
                        out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
                        found = timed(lambda: mr.mm_prompt(x, w, out, weights, scales, row_scale, acts, 500.0,
                                                           blocked, splits, scratch, partial=partial,
                                                           counters=counters), 20)
                        times[m] = round(min(found, times.get(m, found)), 1)
            measured = list(measured.values())
            for row in measured:
                rows.append(row)
                print(json.dumps(row), flush=True)
            chosen[(weights, mma, n, k)] = choose(measured)
    return rows, chosen


def choose(measured):
    """The split count of `measured` ([{splits, us: {call: microseconds}}]) whose worst ratio to the fastest count's
    time, over every call, is least; fewer splits on a tie."""
    calls = measured[0]["us"]
    best = {call: min(row["us"][call] for row in measured) for call in calls}
    regret = lambda row: max(row["us"][call] / best[call] for call in calls)
    return min(measured, key=lambda row: (round(regret(row), 3), row["splits"]))["splits"]


def write(chosen, device_name):
    namespace = {}
    exec(compile(TABLE.read_text(encoding="utf-8"), str(TABLE), "exec"), namespace)
    merged = {**namespace.get("ORDERS", {}), **chosen}
    doc = namespace.get("__doc__", "").split("\n\nSweep:")[0]
    lines = [f'"""{doc}', "", f"Sweep: {device_name}, {datetime.date.today().isoformat()}, benchmarks/order_sweep.py."
             '"""', "", "ORDERS = {"]
    lines += [f"    {key!r}: {value}," for key, value in sorted(merged.items())]
    TABLE.write_text("\n".join(lines + ["}", ""]), encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("shapes", nargs="+", help="WEIGHTS:N:K")
    parser.add_argument("--out", help="the timings as JSON")
    parser.add_argument("--write", action="store_true", help="merge the choices into orders.py")
    args = parser.parse_args(argv)
    shapes = [(w, int(n), int(k)) for w, n, k in (s.split(":") for s in args.shapes)]
    device = torch.device("cuda")
    mr = build.load("mma-rows")
    rows, chosen = sweep(mr, shapes, device)
    if args.out:
        Path(args.out).write_text(json.dumps({"device": torch.cuda.get_device_name(device), "rows": rows}, indent=1),
                                  encoding="utf-8")
    for key, value in chosen.items():
        print(f"{key}: {value}")
    if args.write:
        write(chosen, torch.cuda.get_device_name(device))


if __name__ == "__main__":
    main()
