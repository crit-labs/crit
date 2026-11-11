"""fused-norms against the torch chains a model writes by hand, at three models' shapes: per call, the host's own
time to issue it (the calling thread's CPU time, which other processes on a shared machine do not inflate), the GPU
time inside a CUDA graph (as a captured decode round pays it; the median of several replays) and the kernels each
launches. FlashInfer's same-dtype norms, rope and SiLU product as a yardstick where it has the operation. Run:
python benchmarks/shapes.py [--rows 1,8,1024,8192] [--tuning FILE] (crit's kernels package on the path; the
build uses the default knobs, or the tuning file crit kernels sweep wrote)."""
import argparse
import json
import statistics
import time

import torch

from crit_kernels.build import load, read_tuning

fn = None   # the kernel's package, built in main() with the default knobs or --tuning
try:
    import flashinfer
except ImportError:   # the yardstick is optional
    flashinfer = None

dev = torch.device("cuda")
BF16 = torch.bfloat16
EPS = 1e-6

# name: hidden, q heads, kv heads, head dim, rotary dim, norm offset, intermediate, value heads x dim of a gated norm
MODELS = {
    "qwen3.8-27b": dict(hidden=5120, q=24, kv=4, dim=256, rotary=64, offset=1.0, inter=17408, gated=(48, 128),
                        gate=True, qk_norm=True),
    "nanbeige4.3-33b": dict(hidden=7168, q=56, kv=8, dim=128, rotary=128, offset=0.0, inter=22016, gated=None,
                            gate=False, qk_norm=False),
    "qwen3.8-flash-next": dict(hidden=2560, q=24, kv=2, dim=256, rotary=64, offset=1.0, inter=640, gated=(48, 128),
                               gate=True, qk_norm=True, streams=4),
}


# -- the torch chains, as a model plugin writes them ----------------------------------------------------------------
def chain_norm(x, w, offset):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * (offset + w)).to(BF16)


def chain_residual_norm(a, r, w, offset):
    r = r + a
    return chain_norm(r, w, offset), r


def chain_rope(x, positions, inv_freq, rotary):
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs, freqs], -1).cos().to(x.dtype)[:, None, :]
    sin = torch.cat([freqs, freqs], -1).sin().to(x.dtype)[:, None, :]
    rot, rest = x[..., :rotary], x[..., rotary:]
    half = rotary // 2
    turned = torch.cat([-rot[..., half:], rot[..., :half]], -1)
    out = rot * cos + turned * sin
    return torch.cat([out, rest], -1) if rest.shape[-1] else out


def chain_gated(x, z, w):
    x32 = x.float()
    normed = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(BF16)
    return ((w * normed.float()) * torch.nn.functional.silu(z.float())).to(BF16)


def calls(model, rows):
    """op -> {variant: callable} at one model's shape and row count."""
    m = MODELS[model]
    g = torch.Generator(device="cpu").manual_seed(rows)
    normal = lambda *s, dtype=BF16: torch.randn(*s, generator=g).to(dev, dtype)  # noqa: E731
    H, D = m["hidden"], m["dim"]
    a, r, w = normal(rows, H), normal(rows, H, dtype=torch.float32), normal(H, dtype=torch.float32) * 0.1
    qg = normal(rows, m["q"], 2 * D if m["gate"] else D)
    q = qg[..., :D]
    k = normal(rows, m["kv"], D)
    qw, kw = normal(D, dtype=torch.float32) * 0.1, normal(D, dtype=torch.float32) * 0.1
    positions = torch.randint(0, 200000, (rows,), generator=g).to(dev)
    inv = (1.0 / 1e7 ** (torch.arange(0, m["rotary"], 2, dtype=torch.float64) / m["rotary"])).float().to(dev)
    gate_up = normal(rows, 2 * m["inter"])
    out = {}
    out["residual + rms_norm"] = {
        "chain": lambda: chain_residual_norm(a, r, w, m["offset"]),
        "fused": lambda: fn.rms_norm(a, w, EPS, offset=m["offset"], residual=r, out_dtype=BF16)}
    if flashinfer is not None:
        rb, wb = r.to(BF16), (w + m["offset"]).to(BF16)
        out["residual + rms_norm"]["flashinfer (bf16 residual)"] = lambda: flashinfer.norm.fused_add_rmsnorm(a, rb, wb, EPS)
    w16 = (w + m["offset"]).to(BF16) if m["offset"] else w.to(BF16)
    out["rms_norm, BF16 weight"] = {
        "chain": lambda: chain_norm(a, w16.float(), 0.0),
        "fused": lambda: fn.rms_norm(a, w16, EPS)}
    if flashinfer is not None:
        out["rms_norm, BF16 weight"]["flashinfer"] = lambda: flashinfer.norm.rmsnorm(a, w16, EPS)
    if m["qk_norm"]:
        out["q, k: head norm + rope"] = {
            "chain": lambda: (chain_rope(chain_norm(q, qw, m["offset"]), positions, inv, m["rotary"]),
                              chain_rope(chain_norm(k, kw, m["offset"]), positions, inv, m["rotary"])),
            "fused": lambda: (fn.rms_norm_rope(q, qw, EPS, positions, inv, offset=m["offset"]),
                              fn.rms_norm_rope(k, kw, EPS, positions, inv, offset=m["offset"]))}
    else:
        out["q, k: rope"] = {
            "chain": lambda: (chain_rope(q, positions, inv, m["rotary"]), chain_rope(k, positions, inv, m["rotary"])),
            "fused": lambda: (fn.rope(q, positions, inv), fn.rope(k, positions, inv))}
        if flashinfer is not None:
            qc = q.contiguous()
            out["q, k: rope"]["flashinfer (in place, q and k)"] = lambda: flashinfer.rope.apply_rope_pos_ids_inplace(
                qc, k, positions, rotary_dim=m["rotary"], rope_theta=1.2e8)
    if m["gated"]:
        vh, vd = m["gated"]
        x, z, gw = normal(rows, vh, vd), normal(rows, vh, vd), normal(vd, dtype=torch.float32)
        out["gated rms_norm"] = {"chain": lambda: chain_gated(x, z, gw),
                                 "fused": lambda: fn.gated_rms_norm(x, z, gw, EPS)}
    if m["gate"]:
        attn = normal(rows, m["q"] * D)
        gate = qg[..., D:]
        out["attention gate"] = {"chain": lambda: attn * torch.sigmoid(gate.reshape(rows, -1)),
                                 "fused": lambda: fn.sigmoid_mul(attn.view(rows, m["q"], D), gate)}
    out["silu product"] = {
        "chain": lambda: torch.nn.functional.silu(gate_up[:, :m["inter"]]) * gate_up[:, m["inter"]:],
        "fused": lambda: fn.silu_mul(gate_up)}
    if flashinfer is not None:
        out["silu product"]["flashinfer"] = lambda: flashinfer.activation.silu_and_mul(gate_up)
    if m.get("streams"):
        S = m["streams"]
        h = normal(rows, S, H, dtype=torch.float32)                # the residual streams, FP32
        sw = normal(S, H, dtype=torch.float32) * 0.1 + 1            # a norm weight per stream
        u = normal(rows, S, H, dtype=torch.float32)                 # the mixing gates (a matmul's output)
        block = normal(rows, H)                                     # the block's output
        inj = torch.sigmoid(normal(rows, S, dtype=torch.float32))   # its weight per stream

        def chain_mix():
            hn = torch.nn.functional.rms_norm(h, (H,), eps=EPS) * sw
            streams = torch.sigmoid(u) * hn
            mixed = streams[:, 0]
            for s in range(1, S):
                mixed = mixed + streams[:, s]
            mixed = mixed / S
            return mixed, h + (block.float()[:, None, :] * inj[:, :, None])

        def fused_mix():
            hn = fn.rms_norm(h, sw, EPS)
            mixed = fn.stream_sum(hn, u, "sigmoid", 1.0 / S)
            return mixed, fn.stream_add(h, block, inj)

        out["residual streams: norm, mix, add"] = {"chain": chain_mix, "fused": fused_mix}
    return out


def host_us(call, repeat=100, trials=3):
    """The calling thread's CPU time per call (Python, the dispatch and the launches), the least of a few trials."""
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    found = []
    for _ in range(trials):
        start = time.thread_time_ns()
        for _ in range(repeat):
            call()
        found.append((time.thread_time_ns() - start) / repeat / 1000)
        torch.cuda.synchronize()
    return min(found)


def graph_us(call, inner=20, replays=5, trials=7):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            call()
    graph.replay()
    torch.cuda.synchronize()
    found = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        found.append(start.elapsed_time(end) * 1000 / (inner * replays))
    return statistics.median(found)


def launches(call, calls=4, attempts=3):
    """Kernels per call, from the profiler over a few calls (CUPTI now and then misses a window: tried again)."""
    call()
    torch.cuda.synchronize()
    for _ in range(attempts):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                call()
            torch.cuda.synchronize()
        count = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                    and "memcpy" not in e.name.lower() and "memset" not in e.name.lower())
        if count:
            return round(count / calls, 2)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,1024,8192")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--json", default="")
    parser.add_argument("--tuning", default="", help="a tuning file crit kernels sweep wrote (default: the defaults)")
    args = parser.parse_args()
    global fn
    fn = load("fused-norms", tuning=read_tuning(args.tuning) if args.tuning else None)
    results = []
    for model in args.models.split(","):
        for rows in (int(r) for r in args.rows.split(",")):
            for op, variants in calls(model, rows).items():
                for variant, call in variants.items():
                    found = {"model": model, "rows": rows, "op": op, "variant": variant,
                             "host_us": round(host_us(call), 2), "graph_us": round(graph_us(call), 2),
                             "launches": launches(call)}
                    results.append(found)
                    print(f"{model:20s} {rows:5d} {op:34s} {variant:30s} host {found['host_us']:8.2f} us  graph "
                          f"{found['graph_us']:9.2f} us  launches {found['launches']}", flush=True)
            torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
