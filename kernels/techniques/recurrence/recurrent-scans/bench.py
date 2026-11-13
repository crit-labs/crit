"""recurrent-scans' sweep benches (crit_kernels.sweep): the model's gated-delta layer shape (key and value heads, key
and value dims from its config), random inputs of the plugins' layout seeded per token block or per sequence, an FP64
token-by-token reference of the recurrence (recurrent-state-recovery's gated-delta operation)."""
import torch

from crit_kernels.gates import shared_memory_optin

BLOCK = 256   # prefill tokens generated per seed, so a prompt's first tokens do not depend on its length


def delta_shapes(model):
    """The gated-delta layers' shape of a hybrid model (one per model: every recurrent layer shares it), or none."""
    kh, vh = model.get("linear_num_key_heads"), model.get("linear_num_value_heads")
    dk, dv = model.get("linear_key_head_dim"), model.get("linear_value_head_dim")
    if not all(isinstance(x, int) and x > 0 for x in (kh, vh, dk, dv)) or vh % kh:
        return []
    return [{"k_heads": kh, "v_heads": vh, "dk": dk, "dv": dv}]


def _key(shape):
    return tuple(int(shape[f]) for f in ("k_heads", "v_heads", "dk", "dv"))


def recurrence(q, k, v, g, b, state, scale, eps=None):
    """FP64 gated delta token by token over [T, heads, ...] inputs with a leading batch of sequences or none: per token
    delta = b (v - e^g S^T k); S' = e^g S + k delta^T; out = scale S'^T q (keys and queries L2-normalized first with
    eps). Returns (outputs, final state)."""
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


class DeltaPrefillRb:
    """gated_delta_prefill at 128 key dims (the register-blocked passes), FP32 inputs as the plugin passes them
    (normalized keys and queries), on a prompt of `rows` tokens from a random state (nan_rule causal: one sequence's
    tokens, each seeing the ones before it)."""
    structure = "delta-prefill-rb"
    nan_rule = "causal"
    # The chunked forms multiply a 64-token chunk's triangular matrices densely (0 x NaN is NaN): a NaN reaches its own
    # chunk's earlier tokens, never an earlier chunk's.
    nan_block = 64

    def __init__(self):
        self._tokens, self._states, self._references = {}, {}, {}

    def shapes(self, model):
        return [s for s in delta_shapes(model) if s["dk"] == 128 and s["dv"] % 64 == 0]

    def valid(self, knobs):
        ty = knobs["THREADS"] // (knobs["COLS"] // 4)
        return ty <= 64 and (knobs["MINB"] == 0 or knobs["THREADS"] * knobs["MINB"] <= 2048)

    def _inputs(self, shape, rows, device):
        kh, vh, dk, dv = _key(shape)
        have = self._tokens.get(_key(shape))
        if have is None or have[0].shape[0] < rows:
            blocks = []
            for i in range((rows + BLOCK - 1) // BLOCK):
                g = torch.Generator(device="cpu").manual_seed(7919 * i + 1)
                blocks.append((torch.nn.functional.normalize(torch.randn(BLOCK, kh, dk, generator=g), dim=-1),
                               torch.nn.functional.normalize(torch.randn(BLOCK, kh, dk, generator=g), dim=-1),
                               torch.randn(BLOCK, vh, dv, generator=g),
                               -torch.rand(BLOCK, vh, generator=g) * 0.2,
                               torch.rand(BLOCK, vh, generator=g)))
            have = self._tokens[_key(shape)] = tuple(torch.cat(p).to(device) for p in zip(*blocks))
        if _key(shape) not in self._states:
            g = torch.Generator(device="cpu").manual_seed(17)
            self._states[_key(shape)] = (torch.randn(vh, dk, dv, generator=g) * 0.1).to(device)
        return [t[:rows] for t in have], self._states[_key(shape)]

    def inputs(self, shape, rows, device):
        (q, k, v, g, b), state0 = self._inputs(shape, rows, device)
        return {"q": q, "k": k, "v": v, "g": g, "b": b, "state0": state0, "state": state0.clone(),
                "out": torch.empty_like(v), "scale": int(shape["dk"]) ** -0.5, "key": (_key(shape), rows)}

    def run(self, module, s):
        s["state"].copy_(s["state0"])
        module.gated_delta_prefill(s["q"], s["k"], s["v"], s["g"], s["b"], s["state"], scale=s["scale"], out=s["out"])
        return s["out"], s["state"]

    def reference(self, s):
        if s["key"] not in self._references:
            self._references[s["key"]] = recurrence(s["q"], s["k"], s["v"], s["g"], s["b"], s["state0"], s["scale"])
        return self._references[s["key"]]

    def check(self, got, want, s):
        # A chunked prefill sums in another order than the recurrence: FP32 rounding of the chunks' products, which grows
        # with the prompt's length but stays far below a wrong tile's error.
        for name, x, y in (("outputs", got[0], want[0]), ("final state", got[1], want[1])):
            bad = int(((x.double() - y).abs() > 1e-3 + 1e-3 * y.abs()).sum())
            if bad:
                return f"{bad} {name} beyond 1e-3 + 1e-3|x| of the FP64 recurrence"
        return None

    def first_row(self, out):
        return out[0][0]

    def yardstick(self, s):
        """FlashInfer's chunk_gated_delta_rule (its chunked gated-delta prefill) on the same prompt from the same
        state: the log-domain decays as they are (gate_domain="log"), the keys and queries already normalized as
        here; FlashInfer takes 16-bit q, k and v only, so they are rounded to BF16 once (half the input bytes), and
        its state is V-major ([heads, dv, dk]), so the start state is transposed once."""
        try:
            import flashinfer
        except ImportError:
            return None
        key = ("library", s["key"])
        if key not in self._states:
            rows = s["q"].shape[0]
            q, k, v = (t.to(torch.bfloat16).contiguous() for t in (s["q"], s["k"], s["v"]))
            start = s["state0"].transpose(-1, -2).contiguous()[None]
            self._states[key] = dict(q=q, k=k, v=v, g=s["g"].float().contiguous(), b=s["b"].float().contiguous(),
                                     start=start, final=torch.empty_like(start), out=torch.empty_like(v),
                                     cu=torch.tensor([0, rows], dtype=torch.int32, device=q.device))
        t = self._states[key]
        return lambda: flashinfer.chunk_gated_delta_rule(
            t["q"], t["k"], t["v"], t["g"], t["b"], scale=s["scale"], initial_state=t["start"],
            output_final_state=True, cu_seqlens=t["cu"], output=t["out"], output_state=t["final"], gate_domain="log")

    def bytes(self, shape, rows):
        kh, vh, dk, dv = _key(shape)
        return 4 * (rows * (2 * kh * dk + 2 * vh * dv + 2 * vh) + 2 * vh * dk * dv)


class DeltaPrefillTc(DeltaPrefillRb):
    """gated_delta_prefill(tensor_cores=True): the chunked WY form on tensor cores (FP16 operands, FP32 accumulation and
    state) on the same prompts, reference, tolerance and yardstick as delta-prefill-rb, at dk 64 or 128."""
    structure = "delta-prefill-tc"

    def shapes(self, model):
        return [s for s in delta_shapes(model) if s["dk"] in (64, 128) and s["dv"] % 16 == 0]

    def valid(self, knobs):
        return True

    def run(self, module, s):
        s["state"].copy_(s["state0"])
        module.gated_delta_prefill(s["q"], s["k"], s["v"], s["g"], s["b"], s["state"], scale=s["scale"], out=s["out"],
                                   tensor_cores=True)
        return s["out"], s["state"]


class PoolDelta:
    """pool_delta's deferred-commit (STASH) rows, one per sequence, each `tokens` tokens of BF16 q, k, v (default 1:
    decode; a verify window when the shape says so) from its own FP32 state in a pool; nothing pending, so the live
    state is read and not written and every call gives the same outputs."""
    structure = "pool-delta"
    scratch = ("stash",)   # STASH rows keep their inputs there for the deferred commit

    def __init__(self):
        self._rows, self._pools, self._references = {}, {}, {}

    def shapes(self, model):
        return [dict(s, tokens=1) for s in delta_shapes(model) if s["dk"] <= 256]

    def valid(self, knobs):
        threads = 1024 // knobs.get("LANE_COLS", 4) * knobs["SLOTS"]   # a slot's 32 columns: 32 / LANE_COLS warps
        return threads * max(knobs["MINB"], 1) <= 2048

    def _sequence(self, shape, r, device):
        """Sequence r's state and tokens: seeded by r alone."""
        kh, vh, dk, dv = _key(shape)
        tokens = int(shape.get("tokens", 1))
        key = (_key(shape), tokens, r)
        if key not in self._rows:
            g = torch.Generator(device="cpu").manual_seed(1000 + r)
            self._rows[key] = tuple(x.to(device) for x in (
                torch.randn(vh, dk, dv, generator=g) * 0.1,
                torch.randn(tokens, kh, dk, generator=g).bfloat16(),
                torch.randn(tokens, kh, dk, generator=g).bfloat16(),
                torch.randn(tokens, vh, dv, generator=g).bfloat16(),
                -torch.rand(tokens, vh, generator=g) * 0.2,
                torch.rand(tokens, vh, generator=g)))
        return self._rows[key]

    def inputs(self, shape, rows, device):
        kh, vh, dk, dv = _key(shape)
        tokens = int(shape.get("tokens", 1))
        key = (_key(shape), tokens, rows)
        if key not in self._pools:
            parts = [self._sequence(shape, r, device) for r in range(rows)]
            states = torch.stack([p[0] for p in parts])[None].contiguous()
            q, k, v, g, b = (torch.cat([p[i] for p in parts]).contiguous() for i in range(1, 6))
            blocks = vh * ((dv + 31) // 32)
            half = tokens * (kh * dk + vh + vh * dv)
            meta = torch.tensor([[r * tokens, tokens, r, 0, -1] for r in range(rows)], dtype=torch.int32, device=device)
            self._pools[key] = {
                "states": states, "q": q, "k": k, "v": v, "g": g, "b": b, "meta": meta, "rows": rows,
                "tokens": tokens, "stash": torch.zeros(1, rows, 2 * half, device=device),
                "control": torch.zeros(1, rows, 2 * blocks, dtype=torch.int32, device=device),
                "out": torch.empty(rows * tokens, vh, dv, device=device), "scale": dk ** -0.5, "key": key}
        return self._pools[key]

    def run(self, module, s):
        module.pool_delta(s["out"], s["states"], s["stash"], s["control"], None, s["q"], s["k"], s["v"], s["g"], s["b"],
                          meta=s["meta"], rows=s["rows"], kind=module.STASH, layer=0, capacity=s["tokens"],
                          epsilon=1e-6, scale=s["scale"])
        return s["out"].view(s["rows"], s["tokens"], *s["out"].shape[1:])

    def reference(self, s):
        if s["key"] not in self._references:
            n, t = s["rows"], s["tokens"]
            view = lambda x: x.view(n, t, *x.shape[1:])   # noqa: E731
            out, _ = recurrence(view(s["q"]), view(s["k"]), view(s["v"]), view(s["g"]), view(s["b"]), s["states"][0],
                                s["scale"], eps=1e-6)
            self._references[s["key"]] = out
        return self._references[s["key"]]

    def check(self, got, want, s):
        bad = int(((got.double() - want).abs() > 1e-5 + 1e-4 * want.abs()).sum())
        return None if bad == 0 else f"{bad} outputs beyond 1e-5 + 1e-4|x| of the FP64 recurrence"

    def first_row(self, out):
        return out[0]

    def yardstick(self, s):
        """FlashInfer's gated-delta decode on the same rows and FP32 states: gated_delta_rule_decode_pretranspose for
        one token a row (it writes the updated state back, which pool_delta's STASH rows defer), gated_delta_rule_mtp
        with disable_state_update for a verify window. FlashInfer takes the gate inputs before their activations
        (g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)), so A_log and dt_bias are zero and a and b the
        inverse activations of these g and beta, in BF16, made once; its states are V-major ([heads, dv, dk]),
        transposed once."""
        try:
            import flashinfer.gdn_decode as gdn
        except ImportError:
            return None
        key = ("library", s["key"])
        if key not in self._pools:
            n, t = s["rows"], s["tokens"]
            vh = s["v"].shape[1]
            view = lambda x: x.reshape(n, t, *x.shape[1:]).contiguous()   # noqa: E731
            g = s["g"].clamp(max=-1e-6)
            beta = s["b"].clamp(1e-4, 1 - 1e-4)
            self._pools[key] = dict(
                q=view(s["q"]), k=view(s["k"]), v=view(s["v"]),
                a=view(torch.log(torch.expm1(-g)).to(torch.bfloat16)),
                b=view(torch.log(beta / (1 - beta)).to(torch.bfloat16)),
                A_log=torch.zeros(vh, device=s["v"].device), dt_bias=torch.zeros(vh, device=s["v"].device),
                states=s["states"][0].transpose(-1, -2).contiguous(),
                index=torch.arange(n, dtype=torch.int32, device=s["v"].device),
                out=torch.empty(n, t, vh, s["v"].shape[2], dtype=torch.bfloat16, device=s["v"].device))
        p = self._pools[key]
        common = dict(scale=s["scale"], output=p["out"], use_qk_l2norm=True)
        if s["tokens"] == 1:
            return lambda: gdn.gated_delta_rule_decode_pretranspose(
                p["q"], p["k"], p["v"], p["states"], p["A_log"], p["a"], p["dt_bias"], p["b"], **common)
        return lambda: gdn.gated_delta_rule_mtp(
            p["q"], p["k"], p["v"], p["states"], p["index"], p["A_log"], p["a"], p["dt_bias"], p["b"],
            disable_state_update=True, **common)

    def bytes(self, shape, rows):
        kh, vh, dk, dv = _key(shape)
        tokens = int(shape.get("tokens", 1))
        return rows * (4 * vh * dk * dv + tokens * (2 * (2 * kh * dk + vh * dv) + 8 * vh + 4 * vh * dv * 2))


class PoolDeltaPipe(PoolDelta):
    """pool_delta(pipelined=True), the pool-delta-pipe structure: the same rows, reference, tolerance and yardstick as
    pool-delta (the same bits on another schedule)."""
    structure = "pool-delta-pipe"

    def valid(self, knobs):
        threads = 1024 // knobs["LANE_COLS"]
        return threads * max(knobs["MINB"], 1) <= 2048 and (knobs["STAGES"] == 1 or knobs["ITEMS"] > 1)

    def valid_for(self, knobs, shape):
        rows = 32 * (1 if shape["dk"] <= 32 else 2 if shape["dk"] <= 64 else 4 if shape["dk"] <= 128 else 8)
        stage = rows * 32 * 4 + 8 * (2 * rows + 32) * 2 + 8 * 4   # the tile, 8 BF16 tokens and their betas
        return knobs["STAGES"] * stage * max(knobs["MINB"], 1) <= shared_memory_optin()

    def run(self, module, s):
        module.pool_delta(s["out"], s["states"], s["stash"], s["control"], None, s["q"], s["k"], s["v"], s["g"], s["b"],
                          meta=s["meta"], rows=s["rows"], kind=module.STASH, layer=0, capacity=s["tokens"],
                          epsilon=1e-6, scale=s["scale"], pipelined=True)
        return s["out"].view(s["rows"], s["tokens"], *s["out"].shape[1:])


# -- the untuned structures: no knobs to sweep, benches for the gates (nonfinite, repeat, edges) on the model's shapes --
def ssd_shapes(model):
    """The Mamba-2 layers' shape of a hybrid model (heads, head_dim, groups, state), or none."""
    heads, dim = model.get("mamba_num_heads"), model.get("mamba_head_dim")
    groups, state = model.get("n_groups"), model.get("ssm_state_size")
    if not all(isinstance(x, int) and x > 0 for x in (heads, dim, groups, state)) or heads % groups:
        return []
    return [{"heads": heads, "head_dim": dim, "groups": groups, "state": state}]


def conv_shapes(model):
    """The causal convolutions of a hybrid model's recurrent layers (channels, width): Mamba-2's (its inner width
    plus B and C) and gated delta's (q, k and v)."""
    found = []
    for s in ssd_shapes(model):
        found.append({"channels": s["heads"] * s["head_dim"] + 2 * s["groups"] * s["state"],
                      "width": int(model.get("conv_kernel") or 4)})
    for s in delta_shapes(model):
        found.append({"channels": 2 * s["k_heads"] * s["dk"] + s["v_heads"] * s["dv"],
                      "width": int(model.get("linear_conv_kernel_dim") or 4)})
    return found


def _seeded(rows, make, device, seed):
    """Rows built BLOCK at a time, each block from its own seed: row t is the same at any row count."""
    blocks = [make(torch.Generator(device="cpu").manual_seed(seed * 7919 + i)) for i in range((rows + BLOCK - 1) // BLOCK)]
    return tuple(torch.cat(p)[:rows].to(device) for p in zip(*blocks))


def ssd_recurrence(x, dt, B, C, A, D, state):
    """FP64 Mamba-2 token by token: H' = e^(dt A) H + dt x B^T; y = H' C + D x (groups shared by heads / groups)."""
    x, dt, B, C, A, D, H = (t.double() for t in (x, dt, B, C, A, D, state))
    ratio = x.shape[1] // B.shape[1]
    B, C = B.repeat_interleave(ratio, dim=1), C.repeat_interleave(ratio, dim=1)
    out = torch.empty_like(x)
    for t in range(x.shape[0]):
        H = torch.exp(dt[t] * A)[:, None, None] * H + dt[t][:, None, None] * x[t][:, :, None] * B[t][:, None, :]
        out[t] = torch.einsum("hpn,hn->hp", H, C[t]) + D[:, None] * x[t]
    return out, H


class _Untuned:
    """The common part: one sequence's tokens as the rows (nan_rule causal: a token sees the ones before it), checked
    within 1e-3 + 1e-3|x| of the FP64 recurrence, outputs and the state or window after."""
    nan_rule = "causal"

    def __init__(self):
        self._cache, self._references = {}, {}

    def valid(self, knobs):
        return True

    def check(self, got, want, s):
        for name, x, y in (("outputs", got[0], want[0]), ("final state", got[1], want[1])):
            bad = int(((x.double() - y).abs() > 1e-3 + 1e-3 * y.abs()).sum())
            if bad:
                return f"{bad} {name} beyond 1e-3 + 1e-3|x| of the FP64 recurrence"
        return None

    def first_row(self, out):
        return out[0][0]

    def reference(self, s):
        if s["key"] not in self._references:
            self._references[s["key"]] = self.recurrence(s)
        return self._references[s["key"]]


class SsdPrefill(_Untuned):
    """ssd_prefill on a prompt of `rows` tokens from a random state, the plugin's FP32 inputs (dt after softplus)."""
    structure = "ssd-prefill"
    nan_block = 64   # 64-token chunks, as gated delta's prefill
    call = "ssd_prefill"

    def shapes(self, model):
        return ssd_shapes(model)

    def inputs(self, shape, rows, device):
        h, p, g, n = (int(shape[f]) for f in ("heads", "head_dim", "groups", "state"))
        key = (h, p, g, n)
        if key not in self._cache or self._cache[key][0][0].shape[0] < rows:
            tokens = _seeded(rows, lambda r: (torch.randn(BLOCK, h, p, generator=r), torch.rand(BLOCK, h, generator=r) * 0.1,
                                              torch.randn(BLOCK, g, n, generator=r), torch.randn(BLOCK, g, n, generator=r)),
                             device, 3)
            r = torch.Generator(device="cpu").manual_seed(5)
            fixed = ((-torch.rand(h, generator=r) - 0.1).to(device), torch.randn(h, generator=r).to(device),
                     (torch.randn(h, p, n, generator=r) * 0.1).to(device))
            self._cache[key] = (tokens, fixed)
        (x, dt, B, C), (A, Dk, state0) = self._cache[key]
        x, dt, B, C = x[:rows], dt[:rows], B[:rows], C[:rows]
        return {"x": x, "dt": dt, "B": B, "C": C, "A": A, "D": Dk, "state0": state0, "state": state0.clone(),
                "out": torch.empty_like(x), "key": (key, rows)}

    def run(self, module, s):
        s["state"].copy_(s["state0"])
        getattr(module, self.call)(s["x"], s["dt"], s["B"], s["C"], s["A"], s["state"], D=s["D"], out=s["out"])
        return s["out"], s["state"]

    def recurrence(self, s):
        return ssd_recurrence(s["x"], s["dt"], s["B"], s["C"], s["A"], s["D"], s["state0"])


class SsdColumns(SsdPrefill):
    """ssd_scan: the decode and verify columns, token by token."""
    structure = "ssd-columns"
    call = "ssd_scan"
    nan_block = 0


class DeltaColumns(_Untuned):
    """gated_delta_scan: one sequence's decode and verify columns, from the plugin's FP32 inputs."""
    structure = "delta-columns"

    def __init__(self):
        super().__init__()
        self._prefill = DeltaPrefillRb()

    def shapes(self, model):
        return delta_shapes(model)

    def inputs(self, shape, rows, device):
        return self._prefill.inputs(shape, rows, device)

    def run(self, module, s):
        s["state"].copy_(s["state0"])
        module.gated_delta_scan(s["q"], s["k"], s["v"], s["g"], s["b"], s["state"], scale=s["scale"], out=s["out"])
        return s["out"], s["state"]

    def recurrence(self, s):
        return recurrence(s["q"], s["k"], s["v"], s["g"], s["b"], s["state0"], s["scale"])


class DeltaPrefill64(DeltaColumns):
    """gated_delta_prefill at 64 key dims (the thread-a-row passes)."""
    structure = "delta-prefill-64"

    def shapes(self, model):
        return [s for s in delta_shapes(model) if s["dk"] == 64]

    def run(self, module, s):
        s["state"].copy_(s["state0"])
        module.gated_delta_prefill(s["q"], s["k"], s["v"], s["g"], s["b"], s["state"], scale=s["scale"], out=s["out"])
        return s["out"], s["state"]


class CausalConv(_Untuned):
    """causal_conv1d: one sequence's convolution with its carried window, SiLU, the model's channels and width."""
    structure = "causal-conv"

    def shapes(self, model):
        return conv_shapes(model)

    def inputs(self, shape, rows, device):
        c, w = int(shape["channels"]), int(shape["width"])
        key = (c, w)
        if key not in self._cache or self._cache[key][0][0].shape[0] < rows:
            tokens = _seeded(rows, lambda r: (torch.randn(BLOCK, c, generator=r),), device, 9)
            r = torch.Generator(device="cpu").manual_seed(11)
            fixed = (torch.randn(c, w, generator=r).to(device), torch.randn(c, generator=r).to(device),
                     torch.randn(c, w - 1, generator=r).to(device))
            self._cache[key] = (tokens, fixed)
        (x,), (weight, bias, window0) = self._cache[key]
        x = x[:rows]
        return {"x": x, "weight": weight, "bias": bias, "window0": window0, "window": window0.clone(),
                "after": torch.empty_like(window0), "out": torch.empty_like(x), "key": (key, rows)}

    def run(self, module, s):
        s["window"].copy_(s["window0"])
        module.causal_conv1d(s["x"], s["weight"], s["window"], bias=s["bias"], silu=True, out=s["out"])
        return s["out"], s["window"]

    def recurrence(self, s):
        w = s["weight"].shape[1]
        padded = torch.cat([s["window0"].T.double(), s["x"].double()])
        out = torch.stack([(padded[t:t + w].T * s["weight"].double()).sum(1) + s["bias"].double()
                           for t in range(s["x"].shape[0])])
        return torch.nn.functional.silu(out), padded[-(w - 1):].T


class ConvSegments(CausalConv):
    """conv_segments: one long row out of place (the window in, the window after it out), pool_conv's bits."""
    structure = "conv-segments"

    def run(self, module, s):
        module.conv_segments(s["out"], s["window0"], s["after"], s["x"], s["weight"], s["bias"], True)
        return s["out"], s["after"]
