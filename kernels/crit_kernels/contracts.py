"""The operation contract, executable: each operation of crit's Inference and model techniques (their operations.json)
has here the FP64 reference a kernel's result is compared with and the tolerance it keeps, one definition that the
kernels' benches (so the sweep and the gates) and their tests call:

    from crit_kernels import contracts
    want = contracts.reference("paged-attention")(q, keys, values, seen, scale)    # FP64
    problem = contracts.check("paged-attention", got, want)                      # None, or what is wrong

A tolerance is one of:
    Close(atol, rtol, of_max)       |got - want| <= atol + rtol * |want| + of_max * max |want|
    Sums(rel, rounding, atol)       a projection's: |got - want| <= rel * magnitude + rounding * |want| + atol, with
                                    magnitude = |x| @ |w|^T (the sum of the products' magnitudes, which bounds FP32
                                    accumulation in any order), rounding the output type's half ulp
    Bits(what)                      the same bits as the reference (an encoder, a quantizer, a declared order, the
                                    sampling law): got equals want exactly, NaN where it is NaN

An operation computed in several arithmetics (INT8 queries, token-by-token decode rows, W4A4 operands) names each
one's tolerance among its variants: check(op, got, want, arithmetic=name).

operations.json's "verify" says the same in prose; a kernel.json's "operations" names the ids its structures compute
(a structure may narrow them with its own "operations"). problems() is CI's check that every operation has a contract
and every structure of every kernel maps to a defined operation.

The references are FP64 throughout (torch float64 on the inputs' device, or the technique's own FP64 host form where it
has one, imported from its file), the operation's arithmetic as its "what" defines it: quantized operands enter as the
values they decode to.
"""
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Bits", "Close", "Contract", "Sums", "catalog", "check", "contract", "contracts", "problems", "reference"]


# -- tolerances -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Close:
    atol: float
    rtol: float = 0.0
    of_max: float = 0.0   # times the largest |want|

    def check(self, got, want, magnitude=None):
        import torch
        g, w = _as64(got), _as64(want)
        if g.shape != w.shape:
            return f"shape {tuple(g.shape)} where the reference has {tuple(w.shape)}"
        finite = torch.isfinite(w)
        if not torch.equal(torch.isnan(g), torch.isnan(w)):
            return "NaN where the reference has none, or a number where it has NaN"
        top = float(w[finite].abs().max()) if self.of_max and bool(finite.any()) else 0.0
        bound = self.atol + self.rtol * w.abs() + self.of_max * top
        error = (g - w).abs()
        bad = int(((error > bound) & finite).sum())
        if bad:
            worst = float(((error - bound) * finite).max())
            return (f"{bad} values beyond {self.atol} + {self.rtol} |reference| + {self.of_max} max |reference|, "
                    f"worst by {worst:.3g}")
        return None


@dataclass(frozen=True)
class Sums:
    rel: float = 1e-5
    rounding: float = 2.0 ** -8
    atol: float = 1e-6

    def check(self, got, want, magnitude=None):
        import torch
        if isinstance(want, tuple):
            want, magnitude = want
        g, w = _as64(got), _as64(want)
        if g.shape != w.shape:
            return f"shape {tuple(g.shape)} where the reference has {tuple(w.shape)}"
        m = _as64(magnitude) if magnitude is not None else w.abs()
        bound = self.rel * m + self.rounding * w.abs() + self.atol
        bad = int(((g - w).abs() > bound).sum()) + int((torch.isnan(g) != torch.isnan(w)).sum())
        return None if bad == 0 else f"{bad} outputs beyond FP32 accumulation of the products and the output's rounding"


@dataclass(frozen=True)
class Bits:
    what: str

    def check(self, got, want, magnitude=None):
        import torch
        pairs = zip(got, want) if isinstance(got, (tuple, list)) else [(got, want)]
        for g, w in pairs:
            g, w = torch.as_tensor(g), torch.as_tensor(w).to(device=torch.as_tensor(g).device)
            if g.shape != w.shape or g.dtype != w.dtype:
                return f"{tuple(g.shape)} {g.dtype} where the reference has {tuple(w.shape)} {w.dtype}"
            same = g == w
            if g.is_floating_point():
                same |= torch.isnan(g) & torch.isnan(w)
                same &= torch.signbit(g) == torch.signbit(w)
            bad = int((~same).sum())
            if bad:
                return f"{bad} values differ from {self.what}"
        return None


def _as64(x):
    import torch
    return torch.as_tensor(x).double()


# -- FP64 references ------------------------------------------------------------------------------------------------
def attention(q, keys, values, seen, scale, sinks=None):
    """q [T][hq][d] over keys [n][hkv][d] and values [n][hkv][dv] where seen [T][n] -> (out [T][hq][dv], lse [T][hq]),
    FP64; sinks [hq]: one more position with no value in every softmax (natural log, the scaled scores' units)."""
    import torch
    q, k, v = q.double(), keys.double(), values.double()
    hq, hkv = q.shape[1], k.shape[1]
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    s = torch.einsum("thd,nhd->thn", q, k) * scale
    s = s.masked_fill(~seen.to(s.device)[:, None, :], float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.double().view(1, -1, 1).expand(s.shape[0], -1, 1)], dim=-1)
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    if sinks is not None:
        p = p[..., :-1]
    return torch.einsum("thn,nhd->thd", p, v), lse


def merge(outputs, maxima, sums):
    """Partitions' unnormalized outputs [P][...][dv], running maxima and sums [P][...] -> the merged attention [...][dv]
    and its log-sum-exp, FP64 (a partition with no position: maximum -inf, sum 0)."""
    import torch
    o, m, l = outputs.double(), maxima.double(), sums.double()
    top = m.max(0).values
    w = torch.where(torch.isfinite(m), torch.exp(m - top), torch.zeros_like(m))
    total = (l * w).sum(0)
    return (o * w[..., None]).sum(0) / total[..., None], top + torch.log(total)


def projection(x, w):
    """x [rows][K] @ w [N][K]^T in FP64, with the products' magnitudes |x| @ |w|^T (Sums' bound): (out, magnitude).
    The weights as the values their format decodes to (codes times their scales)."""
    x, w = x.double(), w.double()
    return x @ w.t(), x.abs() @ w.abs().t()


def experts(x, ids, weights, gate_up, down, activation="silu", inputs=None, intermediate=None):
    """A routed gated MLP block in FP64: out[t] = sum_j weights[t][j] * down[e] (act(gate) * up), e = ids[t][j] (routes
    with e < 0 skipped), gate_up[e] [2 inter][hidden] decoded (gate rows first), down[e] [hidden][inter]. inputs(e, x)
    and intermediate(e, h): the operands as the arithmetic quantizes them (W4A4: crit's declared NVFP4 quantizer at
    the expert's scales), None for the values themselves."""
    import torch
    x = x.double()
    out = torch.zeros(x.shape[0], down[0].shape[0], dtype=torch.float64, device=x.device)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            if e < 0:
                continue
            xin = inputs(e, x[t]) if inputs else x[t]
            gu = gate_up[e].double().to(x.device) @ xin
            inter = gu.shape[0] // 2
            g = gu[:inter]
            g = torch.nn.functional.silu(g) if activation == "silu" else torch.nn.functional.gelu(g, approximate="tanh")
            h = g * gu[inter:]
            h = intermediate(e, h) if intermediate else h
            out[t] += float(weights[t, j]) * (down[e].double().to(x.device) @ h)
    return out


def routing(logits, top, bias=None, softmax=True, before=False, renormalize=False, epsilon=0.0, floor=0.0,
            scaling=1.0):
    """Expert selection in FP64, heterogeneous-expert-dispatch's Routing (routing.py, route) for rows of logits: the
    selection on the scores (before: softmax or sigmoid of every logit) or on the raw logits, plus the correction bias
    (the selection only), ties to the lower id; the weights the chosen scores (before), the softmax over the chosen
    logits, or their sigmoids; renormalize divides them by max(sum + epsilon, floor), then scaling -> (ids int64
    [rows][top] in selection order, weights FP64)."""
    import torch
    z = logits.double()
    scores = (torch.softmax(z, -1) if softmax else torch.sigmoid(z)) if before else z
    choice = scores + bias.double() if bias is not None else scores
    ids = torch.sort(-choice, dim=-1, stable=True).indices[:, :top]   # stable: ties to the lower id
    if before:
        w = scores.gather(-1, ids)
    else:
        w = torch.softmax(z.gather(-1, ids), -1) if softmax else torch.sigmoid(z.gather(-1, ids))
    if renormalize:
        w = w / torch.clamp(w.sum(-1, keepdim=True) + epsilon, min=floor)
    return ids, w * scaling


def gated_delta(q, k, v, g, b, state, scale, eps=None):
    """The gated delta rule token by token in FP64 over [T][heads][...] inputs with a leading batch of sequences or
    none: per value head, delta = b (v - e^g S^T k); S' = e^g S + k delta^T; out = scale S'^T q, key head h // (vh /
    kh) serving value head h, keys and queries L2-normalized first with eps (None: as given). q, k [..][T][kh][dk],
    v [..][T][vh][dv], g, b [..][T][vh], state [..][vh][dk][dv] -> (outputs, final state)."""
    import torch
    q, k, v, g, b, S = (t.double() for t in (q, k, v, g, b, state))
    if eps is not None:
        k = k / (k.square().sum(-1, keepdim=True) + eps).sqrt()
        q = q / (q.square().sum(-1, keepdim=True) + eps).sqrt()
    ratio = v.shape[-2] // q.shape[-2]
    q, k = q.repeat_interleave(ratio, dim=-2), k.repeat_interleave(ratio, dim=-2)
    out = torch.empty(v.shape, dtype=torch.float64, device=v.device)
    for t in range(v.shape[-3]):
        e = g[..., t, :].exp()[..., None, None]
        kt, qt = k[..., t, :, :], q[..., t, :, :]
        delta = b[..., t, :, None] * (v[..., t, :, :] - e[..., 0] * torch.einsum("...hkv,...hk->...hv", S, kt))
        S = e * S + kt[..., :, None] * delta[..., None, :]
        out[..., t, :, :] = scale * torch.einsum("...hkv,...hk->...hv", S, qt)
    return out, S


def ssd(x, dt, B, C, A, D, state):
    """Mamba-2's state-space recurrence token by token in FP64: per head, H' = e^(dt A) H + dt x B^T; y = H' C (+ D x).
    x [T][heads][P], dt [T][heads], B and C [T][groups][N] (group h // (heads / groups)), A [heads], D [heads] or None,
    state [heads][P][N] -> (y, final state)."""
    import torch
    x, dt, B, C, A, H = (t.double() for t in (x, dt, B, C, A, state))
    ratio = x.shape[1] // B.shape[1]
    B, C = B.repeat_interleave(ratio, dim=1), C.repeat_interleave(ratio, dim=1)
    out = torch.empty_like(x)
    for t in range(x.shape[0]):
        H = torch.exp(dt[t] * A)[:, None, None] * H + dt[t][:, None, None] * x[t][:, :, None] * B[t][:, None, :]
        out[t] = torch.einsum("hpn,hn->hp", H, C[t])
        if D is not None:
            out[t] = out[t] + D.double()[:, None] * x[t]
    return out, H


def causal_conv(x, weight, window=None, bias=None):
    """The causal depthwise convolution in FP64: x [T][C] after its carried window [W - 1][C] (zeros without), weight
    [C][W] (tap W - 1 the current row) -> [T][C]."""
    import torch
    C, W = weight.shape
    pad = window.double() if window is not None else torch.zeros(W - 1, C, dtype=torch.float64, device=x.device)
    full = torch.cat([pad, x.double()], 0)
    out = sum(full[i:i + x.shape[0]] * weight.double()[:, i] for i in range(W))
    return out + bias.double() if bias is not None else out


def _from_file(relative, symbol):
    """A function of a technique's own FP64 host form, imported from its file (no package needed)."""
    def load(*args, **kwargs):
        path = _root() / relative
        key = f"crit_contract_{path.stem}_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, symbol)(*args, **kwargs)
    load.__name__ = symbol
    load.__qualname__ = f"{relative}:{symbol}"
    return load


# -- the contracts --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Contract:
    id: str
    reference: object       # the FP64 reference (a callable); None where the reference is a declared order's code
    tolerance: object       # Close, Sums or Bits
    note: str = ""
    variants: tuple = ()    # (arithmetic, tolerance) pairs: an arithmetic's own tolerance

    def check(self, got, want, magnitude=None, arithmetic=None):
        tolerance = self.tolerance if arithmetic is None else dict(self.variants)[arithmetic]
        return tolerance.check(got, want, magnitude)


_ATTENTION = Close(8e-3, 2e-2)   # BF16 probabilities times BF16 values, FP32 sums, one rounding of the output
_PROJECTION = Sums()
# FP32 arithmetic (2e-5), one rounding to the output's 16-bit type (2^-8 of a value, 1% of it of the row's largest);
# FP32 outputs keep the arithmetic's alone
_NORMS = Close(2e-5, 2e-5 + 2.0 ** -8, 0.01 * 2.0 ** -8)
_NORM_VARIANTS = (("fp32-output", Close(2e-5, 2e-5)),)
CONTRACTS = {c.id: c for c in (
    Contract("paged-attention", attention, _ATTENTION, "q, keys and values as the pages decode them",
             (("int8-queries", Close(7e-2, 2e-2)),)),
    Contract("partial-attention-merge", merge, Close(1e-6, 1e-5), "the merged rows before their output rounding"),
    Contract("latent-attention", _from_file("inference/techniques/attention/attention-state/code/latent_attention.py", "latent_attention"),
             Close(8e-3, 0.0, 2e-2), "the latent part of each cached row both key (with the rotary part) and value"),
    Contract("sparse-index", _from_file("inference/techniques/attention/attention-state/code/sparse_index.py", "sparse_scores"),
             Close(1e-5, 0.0, 1e-3), "scores; the picks those of sparse_index or within the scores' error"),
    Contract("kv-page-write", None, Bits("the page kind's encoder (paged-kv-attention's module docstring)"),
             "codes and scales of the encoding rule, checked against its torch form in the kernel's tests"),
    Contract("grouped-expert-projection", experts, Close(0.0, 0.0, 5e-3), "BF16 operands, FP32 sums",
             (("w4a4", Close(0.0, 0.0, 3e-2)),)),
    Contract("integer-block-rows", None, Bits("the declared integer block order (host_int.cpp)"),
             "the CPU tier's bits; its sums within Sums() of projection()"),
    Contract("expert-routing", routing, Close(2e-6, 0.0), "ids exactly the reference's, weights within FP32 rounding"),
    Contract("rms-norm", _from_file("inference/techniques/norms/fused-norms/code/rows.py", "rms_norm_rows"), _NORMS, "", _NORM_VARIANTS),
    Contract("rotary-positions", _from_file("inference/techniques/norms/fused-norms/code/rows.py", "rope_rows"), _NORMS, "", _NORM_VARIANTS),
    Contract("gated-products", _from_file("inference/techniques/norms/fused-norms/code/rows.py", "act_mul_rows"), _NORMS, "", _NORM_VARIANTS),
    Contract("stream-mixes", _from_file("inference/techniques/norms/fused-norms/code/rows.py", "stream_sum_rows"), _NORMS, "", _NORM_VARIANTS),
    Contract("softplus-decay", _from_file("inference/techniques/norms/fused-norms/code/rows.py", "softplus_decay_rows"), _NORMS, "", _NORM_VARIANTS),
    Contract("rms-norm-quantize", _from_file("inference/techniques/norms/fused-norms/code/rows.py", "rms_norm_rows"), _NORMS,
             "the norm's rows before they are quantized; the quantized codes those of the format's quantizer",
             _NORM_VARIANTS),
    Contract("nvfp4-projection", projection, _PROJECTION),
    Contract("nvfp4-activation-quantization", _from_file("inference/techniques/projections/quantized-projections/code/nvfp4.py",
                                                         "quantize_row"), Bits("crit's declared NVFP4 quantizer")),
    Contract("fp8-row-projection", projection, _PROJECTION),
    Contract("mxfp8-projection", projection, _PROJECTION),
    Contract("fp8-matched-pair", None, Bits("the declared FP8 block order (fp8_block.h)"),
             "its sums within Sums() of projection()"),
    Contract("int4-group-projection", projection, _PROJECTION, "(code - zero) x scale a group, exact in FP32"),
    Contract("gguf-projection", _from_file("inference/techniques/projections/gguf-projections/code/host.py", "project_rows"), _PROJECTION),
    Contract("exl3-projection", projection, Sums(rel=2e-3),
             "the trellis codes' decoded weights; FP16 rotated rows (2^-11 of the magnitudes)"),
    Contract("gated-delta", gated_delta, Close(1e-3, 1e-3),
             "a chunked form's FP32 products in another order than the scan, below a wrong tile's error",
             (("token-by-token", Close(1e-5, 1e-4)),)),
    Contract("ssd-scan", ssd, Close(1e-3, 1e-3)),
    Contract("causal-conv", causal_conv, Close(1e-4, 1e-4), "FP32 products and sums, the SiLU's expf"),
    Contract("sampling", _from_file("inference/techniques/sampling/token-sampling/code/sampling.py", "law"),
             Bits("the sampling law's draws (sampling.py law, select)")),
    Contract("cpu-route-handoff", None, Bits("heterogeneous-expert-dispatch's host path (DeviceDispatch, ext None)"),
             "the CPU routes, payloads and combined rows bit for bit, checked in the plugin's device tests"),
    Contract("head-clip", _from_file("models/techniques/mixers/relative-attention/code/reference.py", "head_clip"),
             _NORMS, "", _NORM_VARIANTS),
    Contract("small-product", _from_file("models/techniques/mixers/relative-attention/code/reference.py",
                                         "small_product"), _NORMS, "the product before its BF16 rounding"),
    Contract("sink-gate", _from_file("models/techniques/mixers/relative-attention/code/reference.py", "sink_gate"),
             _NORMS),
    Contract("scaled-add", _from_file("models/techniques/mixers/relative-attention/code/reference.py", "scaled_add"),
             _NORMS, "", _NORM_VARIANTS),
    Contract("summary-mix", _from_file("models/techniques/residuals/block-attention/code/reference.py", "summary_mix"),
             Close(2e-5, 2e-5)),
    Contract("pair-norm", _from_file("models/techniques/residuals/block-attention/code/reference.py", "pair_norm"),
             _NORMS),
    Contract("gated-join", _from_file("models/techniques/residuals/block-attention/code/reference.py", "gated_join"),
             Close(1e-4, 1e-4), "f and a given", (("low-rank", Close(0.0, 0.0, 2e-2)),)),
    Contract("span-attention", _from_file("models/techniques/drafters/block-drafter/code/reference.py",
                                          "span_attention"), _ATTENTION, "the span's keys and values as stored (BF16)"),
    Contract("expert-residency", None, Bits("device_cache.LayerDirectory's plans (device-expert-cache)"),
             "the slot tables, copies and counters plan for plan, the slots' bundles byte for byte, checked against "
             "the host twin in device-expert-cache's tests"),
)}


def contract(operation):
    """The operation's Contract (KeyError for an id without one)."""
    try:
        return CONTRACTS[operation]
    except KeyError:
        raise KeyError(f"no contract for operation {operation!r} (crit_kernels/contracts.py)") from None


def contracts():
    return dict(CONTRACTS)


def reference(operation):
    found = contract(operation).reference
    if found is None:
        raise ValueError(f"{operation}'s reference is a declared order's code, not an FP64 function: "
                         f"{contract(operation).tolerance.what}")
    return found


def check(operation, got, want, magnitude=None, arithmetic=None):
    """None where `got` keeps the operation's tolerance (an arithmetic's own, one of its variants) against the
    reference's `want`, else what is wrong."""
    return contract(operation).check(got, want, magnitude, arithmetic)


# -- the catalog and CI's check -------------------------------------------------------------------------------------
def _root():
    """The folder holding inference/ and kernels/ side by side: a checkout, or an installed crit's crit/library."""
    return Path(__file__).resolve().parents[2]


def catalog(root=None):
    """Every operation of crit's techniques: id -> (technique, its operations.json entry)."""
    root = Path(root) if root else _root()
    found = {}
    for path in sorted(p for pillar in ("inference", "models") for p in (root / pillar / "techniques").glob(
            "*/*/operations.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for op in data.get("operations", []):
            found[op["id"]] = (data.get("technique"), op)
    return found


def problems(root=None):
    """What breaks the contract, as sentences (empty: nothing): an operation without a contract here or a contract
    for no operation; a kernel that names no operation or one no technique defines; a structure whose own
    "operations" leave the kernel's."""
    from . import build
    root = Path(root) if root else _root()
    defined = catalog(root)
    out = [f"operation {op} ({defined[op][0]}/operations.json) has no contract in crit_kernels/contracts.py"
           for op in sorted(set(defined) - set(CONTRACTS))]
    out += [f"contract {op} names no operation of any technique's operations.json"
            for op in sorted(set(CONTRACTS) - set(defined))]
    for kid, (_, manifest) in sorted(build.manifests(root / "kernels").items()):
        listed = manifest.get("operations") or []
        if not listed:
            out.append(f"{kid}: kernel.json names no operation (\"operations\")")
        for op in listed:
            if op not in defined:
                out.append(f"{kid}: operation {op} is defined by no technique's operations.json")
        for sid, spec in manifest.get("structures", {}).items():
            own = spec.get("operations")
            if own is None:
                continue
            if not isinstance(own, list) or not own:
                out.append(f"{kid}/{sid}: a structure's operations are a non-empty list")
            for op in own or []:
                if op not in listed:
                    out.append(f"{kid}/{sid}: operation {op} is not one of the kernel's ({', '.join(listed)})")
    return out
