"""fused-norms' sweep benches (crit_kernels.sweep), one a tunable structure: the model's widths (its norm weights, head
size, MLP widths, residual streams, decay heads), BF16 rows (an FP32 residual stream beside them), each token's input
a slice of one cached tensor so a row never depends on the row count, and FP64 references (cached for the most rows,
sliced). The checks are the kernel tests' bounds: FP32 rounding plus one output rounding.

A state's first call runs eagerly (the call the sweep checks) and is then captured in a CUDA graph, which every later
call replays: the sweep times what a captured decode round pays for the call, and a call of a few microseconds is not
timed as Triton's host launch path (tens of microseconds of Python a launch on a loaded machine)."""
import torch

MOST = 4096            # the most tokens a structure's tune_rows asks for
EPS = 1e-6
BF16 = torch.bfloat16
WIDEST_BLOCK = 16384   # the package's: wider rows are long-rows'


def _block(width):
    return max(16, 1 << (width - 1).bit_length())


def _normal(shape, seed, device, dtype=torch.float32, scale=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device) * scale).to(dtype)


def _within(got, want, step=2.0 ** -8):
    """None when got is within 2e-5 + (2e-5 + step) |want| (+ 1% of a step of the largest value) of want."""
    got, want = got.double(), want.double()
    bound = 2e-5 + (2e-5 + step) * want.abs() + step * 0.01 * float(want.abs().max())
    bad = int(((got - want).abs() > bound).sum())
    return None if bad == 0 else f"{bad} values beyond FP32 rounding and one output rounding"


def _replay(s, call):
    """call() eagerly on the state's first use, then captured once and replayed (the outputs are the state's own
    buffers, so the graph's results are the eager call's)."""
    graph = s.get("graph")
    if graph is not None:
        graph.replay()
        return s["result"]
    out = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s["result"] = call()
    s["graph"] = graph
    return out


def _flashinfer():
    """FlashInfer, or None where it is not installed here."""
    try:
        import flashinfer
        return flashinfer
    except ImportError:
        return None


def _rms64(x, weight, offset):
    x = x.double()
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (offset + weight.double())


class _Bench:
    def __init__(self):
        self._data, self._references = {}, {}

    def _cached(self, key, make):
        if key not in self._data:
            self._data[key] = make()
        return self._data[key]

    def reference(self, s):
        if s["key"] not in self._references:
            self._references[s["key"]] = self._reference(s)
        whole = self._references[s["key"]]
        return tuple(t[:s["rows"]] for t in whole)

    def first_row(self, out):
        return out[0]


def _norm_widths(model):
    """{width: (rows a token, residual)} of the model's RMSNorm weights (1-D norm weights without a bias): the hidden
    width takes a residual, a narrower one (heads, gated heads) about hidden / width rows a token."""
    hidden = model.get("hidden_size")
    found = {}
    for name, (_, shape) in model.tensors.items():
        if len(shape) == 1 and "norm" in name and name.endswith(".weight") and name[:-6] + "bias" not in model.tensors:
            d = int(shape[0])
            per = max(1, int(hidden) // d) if hidden and d < int(hidden) else 1
            found[d] = (per, bool(hidden) and d == int(hidden))
    return found


class Rows(_Bench):
    """rms_norm at each norm width of the model: the hidden width with a residual (BF16 rows, FP32 residual in place),
    the narrower ones without (several heads a token), a zero-centred FP32 weight, BF16 out."""
    structure = "rows"

    def shapes(self, model):
        return [{"D": d, "per": per, "residual": res} for d, (per, res) in sorted(_norm_widths(model).items())
                if _block(d) <= WIDEST_BLOCK]

    def inputs(self, shape, rows, device):
        d, per, res = int(shape["D"]), int(shape.get("per", 1)), bool(shape.get("residual"))
        x, r, w = self._cached(("rows", d, per, res), lambda: (
            _normal((MOST, per, d), d, device, BF16, 2.0), _normal((MOST, per, d), d + 1, device) if res else None,
            _normal(d, d + 2, device, scale=0.1)))
        return {"x": x[:rows], "residual": r[:rows].clone() if res else None, "w": w, "rows": rows,
                "out": torch.empty(rows, per, d, dtype=BF16, device=device), "key": ("rows", d, per, res)}

    def run(self, module, s):
        def call():
            module.rms_norm(s["x"], s["w"], EPS, offset=1.0, residual=s["residual"], out_dtype=BF16, out=s["out"])
            return s["out"]
        return _replay(s, call)

    def _reference(self, s):
        x, r, w = self._data[s["key"]]
        if r is None:
            return (_rms64(x, w, 1.0).float(),)
        sums = r + x.float()                # the FP32 sum as the residual stores it
        return _rms64(sums, w, 1.0).float(), sums

    def check(self, got, want, s):
        if s["residual"] is not None and not torch.equal(s["residual"], want[1]):
            return "the residual is not the FP32 sum"
        return _within(got, want[0])

    def yardstick(self, s):
        """FlashInfer's Gemma-style norms (the weight stored as weight - 1, as offset=1.0 reads it): with a residual
        gemma_fused_add_rmsnorm, in place, which keeps the residual in the rows' dtype, so it runs on a BF16 copy of
        the FP32 residual (it reads and writes 2 bytes a residual value instead of 4); without one gemma_rmsnorm on the
        rows as [tokens x heads, D]. The weight is rounded to BF16 once (FlashInfer takes it in the rows' dtype)."""
        fi = _flashinfer()
        if fi is None:
            return None
        w = s["w"].to(BF16)
        if s["residual"] is not None:
            d = s["x"].shape[-1]
            x, r = s["x"].reshape(-1, d).clone(), s["residual"].reshape(-1, d).to(BF16)
            return lambda: fi.gemma_fused_add_rmsnorm(x, r, w, EPS)
        x, out = s["x"].reshape(-1, s["x"].shape[-1]), s["out"].view(-1, s["x"].shape[-1])
        return lambda: fi.gemma_rmsnorm(x, w, EPS, out=out)

    def bytes(self, shape, rows):
        d, per = int(shape["D"]), int(shape.get("per", 1))
        return rows * per * d * (2 + 2 + (8 if shape.get("residual") else 0)) + d * 4


class LongRows(Rows):
    """rms_norm at the model's norm widths past 16,384 values (rows of two passes)."""
    structure = "long-rows"

    def shapes(self, model):
        return [{"D": d, "per": per, "residual": res} for d, (per, res) in sorted(_norm_widths(model).items())
                if _block(d) > WIDEST_BLOCK]


def _inv_freq(rotary, theta, device):
    return (1.0 / float(theta) ** (torch.arange(0, rotary, 2, dtype=torch.float64) / rotary)).float().to(device)


def _rope64(x, positions, freqs, rotary):
    """FP64 rotation in half pairs at the FP32 angle fp32(position) * fp32(freq)."""
    angle = (positions.float()[:, None] * freqs[None, :]).double()
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    half = rotary // 2
    y = x.clone()
    a, b = x[..., :half], x[..., half:rotary]
    y[..., :half], y[..., half:rotary] = a * cos - b * sin, b * cos + a * sin
    return y


class Rope(_Bench):
    """A layer's queries and keys in one launch at the model's head size and head counts: rms_norm_rope_pair where the
    model has query/key norms, rope_pair otherwise; half pairs, the model's rotary values, BF16 out. A state's first
    call also runs the two separate calls: a configuration whose pair differs from them is dropped (check)."""
    structure = "rope"

    def shapes(self, model):
        heads = model.get("num_attention_heads")
        if not heads:
            return []
        hidden = model.get("hidden_size")
        d = int(model.get("head_dim") or int(hidden) // int(heads))
        params = model.get("rope_parameters") or {}
        factor = model.get("partial_rotary_factor") or params.get("partial_rotary_factor") or 1.0
        rotary = int(model.get("rotary_dim") or d * float(factor)) // 2 * 2
        theta = model.get("rope_theta") or params.get("rope_theta") or 10000.0
        norm = any(name.endswith(("q_norm.weight", "k_norm.weight")) for name in model.tensors)
        return [{"D": d, "H": int(heads), "KV": int(model.get("num_key_value_heads") or heads), "R": rotary,
                 "theta": float(theta), "norm": norm}]

    def inputs(self, shape, rows, device):
        d, h, kv, r = int(shape["D"]), int(shape["H"]), int(shape["KV"]), int(shape.get("R", shape["D"]))
        key = ("rope", d, h, kv, r, bool(shape.get("norm", True)))
        q, k, qw, kw, positions, freqs = self._cached(key, lambda: (
            _normal((MOST, h, d), d, device, BF16, 2.0), _normal((MOST, kv, d), d + 1, device, BF16, 2.0),
            _normal(d, d + 2, device, BF16, 0.1), _normal(d, d + 3, device, BF16, 0.1),
            torch.randint(0, 1 << 17, (MOST,), generator=torch.Generator(device=device).manual_seed(d),
                          device=device), _inv_freq(r, shape.get("theta", 1e7), device)))
        return {"q": q[:rows], "k": k[:rows], "qw": qw, "kw": kw, "positions": positions[:rows], "freqs": freqs,
                "R": r, "theta": float(shape.get("theta", 1e7)), "norm": key[-1], "rows": rows, "key": key,
                "fresh": True,
                "q_out": torch.empty(rows, h, d, dtype=BF16, device=device),
                "k_out": torch.empty(rows, kv, d, dtype=BF16, device=device)}

    def _pair(self, module, s):
        if s["norm"]:
            return module.rms_norm_rope_pair(s["q"], s["k"], s["qw"], s["kw"], EPS, s["positions"], s["freqs"], 1.0,
                                             s["R"], q_out=s["q_out"], k_out=s["k_out"])
        return module.rope_pair(s["q"], s["k"], s["positions"], s["freqs"], s["R"], q_out=s["q_out"], k_out=s["k_out"])

    def run(self, module, s):
        out = _replay(s, lambda: self._pair(module, s))
        if s["fresh"]:   # once a state (the call the sweep checks): the pair against the separate calls
            s["fresh"] = False
            if s["norm"]:
                one = [module.rms_norm_rope(t, w, EPS, s["positions"], s["freqs"], 1.0, s["R"])
                       for t, w in ((s["q"], s["qw"]), (s["k"], s["kw"]))]
            else:
                one = [module.rope(t, s["positions"], s["freqs"], s["R"]) for t in (s["q"], s["k"])]
            s["same"] = all(torch.equal(a, b) for a, b in zip(out, one))
        return out

    def _reference(self, s):
        q, k, qw, kw, positions, freqs = self._data[s["key"]]
        found = []
        for t, w in ((q, qw), (k, kw)):
            t = _rms64(t, w, 1.0) if s["norm"] else t.double()
            found.append(_rope64(t, positions, freqs, s["R"]).float())
        return tuple(found)

    def check(self, got, want, s):
        if not s.get("same", True):
            return "the pair differs from the separate calls"
        return _within(got[0], want[0]) or _within(got[1], want[1])

    def first_row(self, out):
        return torch.cat([out[0][0].flatten(), out[1][0].flatten()])

    def yardstick(self, s):
        """FlashInfer's calls for the same work: gemma_rmsnorm of the queries and of the keys (as [tokens x heads, D])
        and apply_rope_pos_ids_inplace on both (half pairs over the first R values, the same theta; FlashInfer makes
        the inverse frequencies itself), three launches; apply_rope_pos_ids alone without the norms. No FlashInfer
        call fuses a head norm with 1-D rotary positions."""
        fi = _flashinfer()
        if fi is None:
            return None
        d, theta = s["q"].shape[-1], s["theta"]
        positions = s["positions"].to(torch.int32)
        q_out, k_out = s["q_out"], s["k_out"]
        if not s["norm"]:
            q, k = s["q"], s["k"]
            return lambda: fi.apply_rope_pos_ids(q, k, positions, rotary_dim=s["R"], rope_theta=theta)
        q, k, qw, kw = s["q"].reshape(-1, d), s["k"].reshape(-1, d), s["qw"], s["kw"]

        def call():
            fi.gemma_rmsnorm(q, qw, EPS, out=q_out.view(-1, d))
            fi.gemma_rmsnorm(k, kw, EPS, out=k_out.view(-1, d))
            fi.apply_rope_pos_ids_inplace(q_out, k_out, positions, rotary_dim=s["R"], rope_theta=theta)
        return call

    def bytes(self, shape, rows):
        return rows * (int(shape["H"]) + int(shape["KV"])) * int(shape["D"]) * 4


class ActMul(_Bench):
    """silu_mul of packed [gate | up] BF16 halves at the model's MLP widths (the kernel serves them where no provider
    does), BF16 out."""
    structure = "act-mul"

    def shapes(self, model):
        found = {model.get(name) for name in ("intermediate_size", "moe_intermediate_size",
                                              "shared_expert_intermediate_size")}
        return [{"N": int(n)} for n in sorted(n for n in found if n)]

    def valid(self, knobs):
        return knobs["BLOCK"] >= 32 * knobs["WARPS"]

    def inputs(self, shape, rows, device):
        n = int(shape["N"])
        x = self._cached(("act", n), lambda: _normal((MOST, 2 * n), n, device, BF16, 2.0))
        return {"x": x[:rows], "rows": rows, "key": ("act", n),
                "out": torch.empty(rows, n, dtype=BF16, device=device)}

    def run(self, module, s):
        return _replay(s, lambda: module.silu_mul(s["x"], out=s["out"]))

    def _reference(self, s):
        x = self._data[s["key"]].double()
        n = x.shape[-1] // 2
        g = x[:, :n]
        return ((g * torch.sigmoid(g) * x[:, n:]).float(),)

    def check(self, got, want, s):
        return _within(got, want[0])

    def yardstick(self, s):
        """FlashInfer's silu_and_mul on the same packed [gate | up] rows."""
        fi = _flashinfer()
        if fi is None:
            return None
        x, out = s["x"], s["out"]
        return lambda: fi.silu_and_mul(x, out=out)

    def bytes(self, shape, rows):
        return rows * int(shape["N"]) * 6


def _streams(model):
    count = model.get("hc_count", "hc_mult", "num_residual_streams")
    hidden = model.get("hidden_size")
    return [{"S": int(count), "D": int(hidden)}] if count and hidden else []


class StreamSum(_Bench):
    """stream_sum of a token's residual streams (BF16) with an FP32 weight per stream, BF16 out."""
    structure = "stream-sum"

    def shapes(self, model):
        return _streams(model)

    def valid(self, knobs):
        return knobs["BLOCK"] >= 32 * knobs["WARPS"]

    def inputs(self, shape, rows, device):
        n, d = int(shape["S"]), int(shape["D"])
        x, g = self._cached(("sum", n, d), lambda: (_normal((MOST, n, d), d, device, BF16),
                                                    _normal((MOST, n), d + 1, device)))
        return {"x": x[:rows], "g": g[:rows], "rows": rows, "key": ("sum", n, d),
                "out": torch.empty(rows, d, dtype=BF16, device=device)}

    def run(self, module, s):
        return _replay(s, lambda: module.stream_sum(s["x"], s["g"], "none", out=s["out"]))

    def _reference(self, s):
        x, g = self._data[s["key"]]
        return (((x.double() * g.double()[..., None]).sum(1)).float(),)

    def check(self, got, want, s):
        return _within(got, want[0])

    def yardstick(self, s):
        """torch ops: (x * g[..., None]).sum(1) in FP32, cast to BF16 into the output (three kernels). No library
        fuses a weighted sum over residual streams."""
        x, g, out = s["x"], s["g"][..., None], s["out"]
        return lambda: out.copy_((x * g).sum(1))

    def bytes(self, shape, rows):
        return rows * int(shape["D"]) * (int(shape["S"]) + 1) * 2


class StreamAdd(_Bench):
    """stream_add of one BF16 row into a token's residual streams (BF16) with an FP32 weight per stream, in place."""
    structure = "stream-add"

    def shapes(self, model):
        return _streams(model)

    def valid(self, knobs):
        return knobs["BLOCK"] >= 32 * knobs["WARPS"]

    def inputs(self, shape, rows, device):
        n, d = int(shape["S"]), int(shape["D"])
        h, x, w = self._cached(("add", n, d), lambda: (_normal((MOST, n, d), d, device, BF16),
                                                       _normal((MOST, d), d + 1, device, BF16),
                                                       _normal((MOST, n), d + 2, device)))
        return {"h": h[:rows].clone(), "x": x[:rows], "w": w[:rows], "rows": rows, "key": ("add", n, d)}

    def run(self, module, s):
        return _replay(s, lambda: module.stream_add(s["h"], s["x"], s["w"]))

    def _reference(self, s):
        h, x, w = self._data[s["key"]]
        return ((h.double() + w.double()[..., None] * x.double()[:, None]).float(),)

    def check(self, got, want, s):
        return _within(got, want[0])

    def yardstick(self, s):
        """torch ops: h += w[..., None] * x[:, None] in place, the product in FP32 and one rounding to BF16 (two
        kernels). No library adds one row into weighted residual streams."""
        h, x, w = s["h"], s["x"][:, None], s["w"][..., None]
        return lambda: h.add_(w * x)

    def bytes(self, shape, rows):
        return rows * int(shape["D"]) * (2 * int(shape["S"]) + 1) * 2


class SoftplusDecay(_Bench):
    """softplus_decay of a projection's [gate | x] BF16 rows at the model's decay widths (its A_log tensors), FP32
    out."""
    structure = "softplus-decay"

    def shapes(self, model):
        found = {int(shape[0]) for name, (_, shape) in model.tensors.items()
                 if name.endswith("A_log") and len(shape) == 1}
        return [{"N": n} for n in sorted(found)]

    def valid(self, knobs):
        return knobs["BLOCK"] >= 32 * knobs["WARPS"]

    def inputs(self, shape, rows, device):
        n = int(shape["N"])
        x, bias, rate = self._cached(("decay", n), lambda: (
            _normal((MOST, 2 * n), n, device, BF16, 2.0), _normal(n, n + 1, device, BF16),
            torch.exp(_normal(n, n + 2, device, scale=0.5))))
        return {"x": x[:rows], "bias": bias, "rate": rate, "rows": rows, "key": ("decay", n),
                "y": torch.empty(rows, n, device=device), "beta": torch.empty(rows, n, device=device)}

    def run(self, module, s):
        n = s["bias"].shape[0]
        return _replay(s, lambda: module.softplus_decay(s["x"][:, n:], s["bias"], s["rate"], s["x"][:, :n], out=s["y"],
                                                        gate_out=s["beta"]))

    def _reference(self, s):
        x, bias, rate = self._data[s["key"]]
        n = bias.shape[0]
        v = (x[:, n:].float() + bias.float()).double()      # x + bias as the kernel adds it, in FP32
        return -rate.double() * torch.nn.functional.softplus(v), torch.sigmoid(x[:, :n].double())

    def check(self, got, want, s):
        return _within(got[0], want[0], 0.0) or _within(got[1], want[1], 0.0)

    def first_row(self, out):
        return torch.cat([out[0][0], out[1][0]])

    def yardstick(self, s):
        """torch ops: y = -rate * softplus(x + bias) and beta = sigmoid(gate), in FP32 into the same outputs (five
        kernels). No library computes a gated delta layer's decay and beta gate on their own (FlashInfer folds them
        into its own gated-delta calls)."""
        n = s["bias"].shape[0]
        x, gate, bias, rate, y, beta = s["x"][:, n:], s["x"][:, :n], s["bias"].float(), -s["rate"], s["y"], s["beta"]

        def call():
            torch.mul(torch.nn.functional.softplus(x.float() + bias), rate, out=y)
            torch.sigmoid(gate.float(), out=beta)
        return call

    def bytes(self, shape, rows):
        return rows * int(shape["N"]) * 12
