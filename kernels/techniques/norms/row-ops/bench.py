"""row-ops' sweep benches (crit_kernels.sweep), one a tunable structure: the model's widths (its hidden width, head
width and heads, and the small matrices its tensors hold), BF16 or FP32 rows as the techniques call them, each token's
input a slice of one cached tensor so a row never depends on the row count, and the operations' FP64 references
(crit_kernels.contracts), cached for the most rows and sliced. A shape the model does not have is not benched: a
structure finds its widths in the config's standard keys and in the shapes of the checkpoint's tensors, never in a
tensor's or config key's name.

A state's first call runs eagerly (the call the sweep checks) and is then captured in a CUDA graph, which every later
call replays."""
import torch

from crit_kernels import contracts

MOST = 4096          # the most tokens a structure's tune_rows asks for
BF16 = torch.bfloat16


def _normal(shape, seed, device, dtype=torch.float32, scale=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device) * scale).to(dtype)


def _replay(s, call):
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


def _hidden(model):
    return int(model.get("hidden_size") or 0)


def _heads(model):
    """(heads, head width) of the model's attention, from the config's standard keys."""
    heads, hidden = model.get("num_attention_heads"), _hidden(model)
    d = model.get("head_dim") or (hidden // int(heads) if heads and hidden else None)
    return (int(heads), int(d)) if heads and d else (None, None)


def _small_matrices(model, most_rows=64, most_columns=4096):
    """(R, E) of the checkpoint's 2-D tensors of at most most_rows rows and most_columns columns that are no
    projection of the hidden width: the small matrices a row's few values multiply."""
    hidden, found = _hidden(model), set()
    for dtype, shape in model.tensors.values():
        if len(shape) == 2 and 1 < shape[0] <= most_rows < shape[1] <= most_columns and hidden not in shape \
                and str(dtype).upper() in ("F32", "BF16", "F16", "FLOAT32", "BFLOAT16", "FLOAT16"):
            found.add((int(shape[0]), int(shape[1])))
    return sorted(found)


def _low_ranks(model, most=256):
    """The ranks of the checkpoint's [rank][2 hidden] matrices (a gate's low-rank product of two rows)."""
    hidden = _hidden(model)
    return sorted({int(shape[0]) for _, shape in model.tensors.values()
                   if hidden and len(shape) == 2 and shape[1] == 2 * hidden and shape[0] <= most})


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
        return tuple(t[:s["rows"]] for t in self._references[s["key"]])

    def first_row(self, out):
        return out[0]

    def yardstick(self, s):
        return None   # the torch chain each technique falls back to is timed beside it in its plugin's tests


class HeadClip(_Bench):
    """head_clip of a token's query heads with a learned scale and a per-token factor (BF16 in, FP32 out)."""
    structure = "head-clip"

    def shapes(self, model):
        heads, d = _heads(model)
        return [{"D": d, "heads": heads}] if heads else []

    def inputs(self, shape, rows, device):
        d, heads = int(shape["D"]), int(shape["heads"])
        x, scale, tau = self._cached(("clip", d, heads), lambda: (
            _normal((MOST, heads, d), d, device, BF16, 2.0), 1.0 + 0.3 * _normal((heads, d), d + 1, device),
            1.0 + 0.1 * _normal(MOST, d + 2, device).abs()))
        return {"x": x[:rows], "scale": scale, "tau": tau[:rows], "rows": rows, "key": ("clip", d, heads)}

    def run(self, module, s):
        return _replay(s, lambda: module.head_clip(s["x"], 1.5, s["scale"], s["tau"], round_bf16=True))

    def _reference(self, s):
        x, scale, tau = self._data[s["key"]]
        return (contracts.reference("head-clip")(x, 1.5, scale, tau, round_bf16=True).float(),)

    def check(self, got, want, s):
        return contracts.check("head-clip", got, want[0], arithmetic="fp32-output")

    def bytes(self, shape, rows):
        return rows * int(shape["heads"]) * int(shape["D"]) * 6


class SmallProduct(_Bench):
    """small_product of each token head's few values and a small matrix of the checkpoint (FP32 in and out)."""
    structure = "small-product"

    def shapes(self, model):
        heads, _ = _heads(model)
        return [{"R": r, "E": e, "heads": heads or 1} for r, e in _small_matrices(model)]

    def valid(self, knobs):
        return knobs["BLOCK"] >= 32 * knobs["WARPS"]

    def inputs(self, shape, rows, device):
        r, e, heads = int(shape["R"]), int(shape["E"]), int(shape["heads"])
        x, proj = self._cached(("small", r, e, heads), lambda: (_normal((MOST, heads, r), r, device),
                                                                _normal((r, e), e, device, scale=0.5)))
        return {"x": x[:rows], "proj": proj, "rows": rows, "key": ("small", r, e, heads)}

    def run(self, module, s):
        return _replay(s, lambda: module.small_product(s["x"], s["proj"]))

    def _reference(self, s):
        x, proj = self._data[s["key"]]
        return (contracts.reference("small-product")(x, proj).float(),)

    def check(self, got, want, s):
        return contracts.check("small-product", got, want[0])

    def bytes(self, shape, rows):
        return rows * int(shape["heads"]) * (int(shape["R"]) + int(shape["E"])) * 4


class SinkGate(_Bench):
    """sink_gate of an attention's BF16 rows with FP32 log-sum-exps, sinks and gates."""
    structure = "sink-gate"

    def shapes(self, model):
        heads, d = _heads(model)
        return [{"D": d, "heads": heads}] if heads else []

    def inputs(self, shape, rows, device):
        d, heads = int(shape["D"]), int(shape["heads"])
        a, lse, sink, gate = self._cached(("sink", d, heads), lambda: (
            _normal((MOST, heads, d), d, device, BF16), _normal((MOST, heads), d + 1, device),
            _normal((MOST, heads), d + 2, device), _normal((MOST, heads * d), d + 3, device)))
        return {"a": a[:rows], "lse": lse[:rows], "sink": sink[:rows], "gate": gate[:rows], "rows": rows,
                "key": ("sink", d, heads)}

    def run(self, module, s):
        return _replay(s, lambda: module.sink_gate(s["a"], s["lse"], s["sink"], s["gate"]))

    def _reference(self, s):
        return (contracts.reference("sink-gate")(*self._data[s["key"]]).float(),)

    def check(self, got, want, s):
        return contracts.check("sink-gate", got, want[0])

    def bytes(self, shape, rows):
        return rows * int(shape["heads"]) * int(shape["D"]) * 8


class ScaledAdd(_Bench):
    """scaled_add of BF16 rows of the hidden width into BF16."""
    structure = "scaled-add"

    def shapes(self, model):
        return [{"N": _hidden(model)}] if _hidden(model) else []

    def valid(self, knobs):
        return knobs["BLOCK"] >= 32 * knobs["WARPS"]

    def inputs(self, shape, rows, device):
        n = int(shape["N"])
        y, x = self._cached(("add", n), lambda: (_normal((MOST, n), n, device, BF16),
                                                 _normal((MOST, n), n + 1, device, BF16)))
        return {"y": y[:rows], "x": x[:rows], "rows": rows, "key": ("add", n)}

    def run(self, module, s):
        return _replay(s, lambda: module.scaled_add(s["y"], s["x"], 0.5, BF16))

    def _reference(self, s):
        y, x = self._data[s["key"]]
        return (contracts.reference("scaled-add")(y, x, 0.5).float(),)

    def check(self, got, want, s):
        return contracts.check("scaled-add", got.float(), want[0])

    def bytes(self, shape, rows):
        return rows * int(shape["N"]) * 6


class SummaryMix(_Bench):
    """summary_mix of a token's summaries (BF16, a few of them) and its FP32 row by one and three FP32 queries."""
    structure = "summary-mix"
    SUMMARIES = 4

    def shapes(self, model):
        return [{"H": _hidden(model), "Q": q} for q in (1, 3)] if _hidden(model) else []

    def inputs(self, shape, rows, device):
        h, q = int(shape["H"]), int(shape["Q"])
        blocks, row, query = self._cached(("mix", h, q), lambda: (
            _normal((MOST, self.SUMMARIES, h), h, device, BF16), _normal((MOST, h), h + 1, device),
            _normal((q, h), h + 2, device)))
        return {"blocks": blocks[:rows], "h": row[:rows], "query": query, "rows": rows, "key": ("mix", h, q)}

    def run(self, module, s):
        return _replay(s, lambda: module.summary_mix(s["blocks"], s["h"], s["query"]))

    def _reference(self, s):
        return (contracts.reference("summary-mix")(*self._data[s["key"]]).float(),)

    def check(self, got, want, s):
        return contracts.check("summary-mix", got, want[0])

    def bytes(self, shape, rows):
        return rows * int(shape["H"]) * (2 * self.SUMMARIES + 4 + 4 * int(shape["Q"]))


class PairNorm(_Bench):
    """pair_norm of two FP32 rows of the hidden width into BF16."""
    structure = "pair-norm"

    def shapes(self, model):
        return [{"H": _hidden(model)}] if _hidden(model) else []

    def inputs(self, shape, rows, device):
        h = int(shape["H"])
        x, m = self._cached(("pair", h), lambda: (_normal((MOST, h), h, device), _normal((MOST, h), h + 1, device)))
        return {"x": x[:rows], "m": m[:rows], "rows": rows, "key": ("pair", h)}

    def run(self, module, s):
        return _replay(s, lambda: module.pair_norm(s["x"], s["m"]))

    def _reference(self, s):
        return (contracts.reference("pair-norm")(*self._data[s["key"]]).float(),)

    def check(self, got, want, s):
        return contracts.check("pair-norm", got.float(), want[0])

    def bytes(self, shape, rows):
        return rows * int(shape["H"]) * 12


class GatedJoin(_Bench):
    """gated_join of FP32 rows, messages and gates of the hidden width."""
    structure = "gated-join"

    def shapes(self, model):
        return [{"H": _hidden(model)}] if _hidden(model) else []

    def valid(self, knobs):
        return knobs["BLOCK"] >= 32 * knobs["WARPS"]

    def inputs(self, shape, rows, device):
        h = int(shape["H"])
        x, m, fa = self._cached(("join", h), lambda: (_normal((MOST, h), h, device), _normal((MOST, h), h + 1, device),
                                                      _normal((MOST, 2 * h), h + 2, device)))
        return {"x": x[:rows], "m": m[:rows], "fa": fa[:rows], "rows": rows, "key": ("join", h)}

    def run(self, module, s):
        h = s["x"].shape[1]
        return _replay(s, lambda: module.gated_join(s["x"], s["m"], s["fa"][:, :h], s["fa"][:, h:]))

    def _reference(self, s):
        x, m, fa = self._data[s["key"]]
        h = x.shape[1]
        return (contracts.reference("gated-join")(x, m, fa[:, :h], fa[:, h:]).float(),)

    def check(self, got, want, s):
        return contracts.check("gated-join", got, want[0])

    def bytes(self, shape, rows):
        return rows * int(shape["H"]) * 20


class LowRankGates(_Bench):
    """low_rank_gates of three gates over FP32 rows at each low rank of the checkpoint's [rank][2 hidden] matrices."""
    structure = "low-rank-gates"
    GATES = 3

    def shapes(self, model):
        return [{"H": _hidden(model), "R": r} for r in _low_ranks(model)]

    def inputs(self, shape, rows, device):
        h, r, g = int(shape["H"]), int(shape["R"]), self.GATES
        x, m, down, up = self._cached(("gates", h, r), lambda: (
            _normal((MOST, g, h), h, device), _normal((MOST, g, h), h + 1, device),
            _normal((g, r, 2 * h), h + 2, device, BF16, (2 * h) ** -0.5),
            _normal((g, 2 * h, r), h + 3, device, BF16, r ** -0.5)))
        return {"x": x[:rows], "m": m[:rows], "down": down, "up": up, "rows": rows, "key": ("gates", h, r)}

    def run(self, module, s):
        return _replay(s, lambda: module.low_rank_gates(s["x"], s["m"], s["down"], s["up"]))

    def _reference(self, s):
        x, m, down, up = self._data[s["key"]]
        return (contracts.reference("gated-join")(x, m, down=down, up=up).float(),)

    def check(self, got, want, s):
        return contracts.check("gated-join", got, want[0], arithmetic="low-rank")

    def bytes(self, shape, rows):
        return rows * self.GATES * int(shape["H"]) * 12


class SpanAttention(_Bench):
    """span_attention of BF16 queries (16 and 64 a KV head) over a span of 1,024 cached BF16 positions."""
    structure = "span-attention"
    POSITIONS = 1024
    MOST_ROWS = 64

    def shapes(self, model):
        heads, d = _heads(model)
        kv = model.get("num_key_value_heads") or heads
        return [{"G": g, "D": d, "kv": int(kv)} for g in (16, 64)] if heads and d in (64, 128, 256) else []

    def inputs(self, shape, rows, device):
        rows = min(rows, self.MOST_ROWS)
        g, d, kv, t = int(shape["G"]), int(shape["D"]), int(shape["kv"]), self.POSITIONS
        q, k, v = self._cached(("span", g, d, kv), lambda: (
            _normal((self.MOST_ROWS, kv, g, d), d, device, BF16), _normal((self.MOST_ROWS, kv, t, d), d + 1, device,
                                                                          BF16),
            _normal((self.MOST_ROWS, kv, t, d), d + 2, device, BF16)))
        first = torch.zeros(self.MOST_ROWS, dtype=torch.int64, device=device)
        last = torch.full((self.MOST_ROWS,), t - 1, dtype=torch.int64, device=device)
        return {"q": q[:rows], "k": k[:rows], "v": v[:rows], "first": first[:rows], "last": last[:rows],
                "rows": rows, "key": ("span", g, d, kv)}

    def run(self, module, s):
        return _replay(s, lambda: module.span_attention(s["q"], s["k"], s["v"], s["first"], s["last"]))

    def _reference(self, s):
        q, k, v = self._data[s["key"]]
        n = q.shape[0]
        first = torch.zeros(n, dtype=torch.int64, device=q.device)
        last = torch.full((n,), k.shape[2] - 1, dtype=torch.int64, device=q.device)
        return (contracts.reference("span-attention")(q, k, v, first, last).float(),)

    def check(self, got, want, s):
        return contracts.check("span-attention", got.float(), want[0])

    def bytes(self, shape, rows):
        return min(rows, self.MOST_ROWS) * int(shape["kv"]) * self.POSITIONS * int(shape["D"]) * 4
