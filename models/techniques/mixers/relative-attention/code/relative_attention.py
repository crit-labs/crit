"""Softmax attention with a relative-position bias a token computes for the positions behind it, learned sinks a token, L2-clipped queries and keys, short and dynamic convolutions over keys and values, a gated output.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `relative_attention`
and composes the block relative-attention a layout names (README.md)."""
import math
from ..template.blocks import _tensor, row_sum


def _clipped(torch, x, threshold, scale=None, tau=None, out_dtype=None, ops=None):
    """Each head vector of x [tokens, heads, dim] (times its learned scale [heads][dim], where given) shortened to
    `threshold` where its L2 norm is longer (norm + 1e-6), in FP32: the norm a row sum in one fixed order. tau
    [tokens]: the clipped rows rounded to BF16 and times their token's tau (log scaling). ops: crit's kernel row-ops
    (head_clip), else the torch form below."""
    if ops is not None:
        return ops.head_clip(x, threshold, scale, tau, round_bf16=tau is not None, out_dtype=out_dtype)
    x = x.float() if scale is None else x.float() * scale
    if tau is not None:
        norm = row_sum(x * x).sqrt()
        x = (x * (threshold / (norm + 1e-6)).clamp(max=1.0)[..., None]).to(torch.bfloat16).float()
        return x * tau[:, None, None]
    norm = row_sum(x * x).sqrt()
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)[..., None]


class RelativeAttention:
    """Softmax attention with a relative-position bias a token computes for itself, learned sinks a token, short
    convolutions over the keys and values, and a gated output. Its rows: queries, keys and values may each come from
    rows of their own (forward((q_in, k_in, v_in), ctx), each normed by the residual; a single row for all three):

        q = clip(q_proj(q_in) * q_scale)           each head shortened to the config's qk_l2_clip_threshold
        k = clip(conv(k_proj(k_in)))               the convolution: (taps over the last `kernel` rows + the row) /
        v = rms_norm(conv(v_proj(v_in)))           sqrt(kernel + 1); with stored static taps and a projection of
                                                   the input (dynamic), each token's taps are the static ones plus
                                                   its own, one set a key/value head, / sqrt(2 * kernel + 1)
        r = clip(relative_proj(q_in) * r_scale)    each head's relative state, d_rel values, clipped to the config's
        bias[d] = r . proj[:, d]                     rel_l2_clip_threshold: its score for the position d behind it
        out = sigmoid(lse - sink) * attention(q, k, v, bias)    sink = sink_proj(q_in), a logit with no value
        y = o_proj(out * sigmoid(gate_proj(q_in)))

    On CUDA the row operations between the products (the clips, the bias's small product, the sink and the gate, the
    convolutions' joins) run crit's kernel row-ops (the model's kernel("row-ops"), its knobs tuned by crit kernels
    sweep), else the torch forms here, which are its references' arithmetic.

    With `log_scaling` (the full layers) the queries and the relative states are scaled after their clip by
    1 + alpha * log(max((position + 1) / floor, 1)) (the config's log_scaling_alpha and log_scaling_n_floor). The
    bias reaches `extent` positions back: the sliding window, or the config's rel_extent. The block also holds the
    convolutions its layer's residual runs after the attention and after the FFN (`post_convolution(name, y, ctx)`):
    each convolution keeps its window in a pool the block declares (the recurrence slot's convolve), one a layer, so a
    looped layer's passes keep their own. Tensor names are the technique's own (entry.json reads; a layout's names
    rules map a checkpoint's spelling onto them); spec keys: heads, kv_heads and head_dim (the config keys of this
    kind's shape), log_scaling, post (the residual's convolutions: {name: tensor prefix under the layer})."""

    keeps_kv = True

    @classmethod
    def _config(cls, model, spec):
        found = {key: model.setting(name) for key, name in (
            ("kernel", "conv_kernel_size"), ("d_rel", "d_rel"), ("qk", "qk_l2_clip_threshold"),
            ("rel", "rel_l2_clip_threshold"), ("alpha", "log_scaling_alpha"), ("floor", "log_scaling_n_floor"),
            ("rel_extent", "rel_extent"))}
        if not found["kernel"]:
            found["kernel"] = model.setting("sconv_kernel_size")
        missing = [key for key in ("kernel", "d_rel", "qk", "rel") if not found[key]]
        if missing:
            raise ValueError(f"model-template: relative attention takes the config's {', '.join(missing)}")
        return found

    @classmethod
    def _dynamic(cls, model, layers, spec):
        p = model.layer_prefix(layers[0]) + spec.get("prefix", "self_attn.")
        return p + "k_proj.sconv.static_weight" in model.ck

    @classmethod
    def pools(cls, model, spec, layers):
        """The windows of the layers' convolutions: keys and values (a dynamic convolution keeps `kernel` copies of
        each input, one a tap), and the residual's after the attention and after the FFN."""
        found = cls._config(model, spec)
        kv_heads, head_dim = model.kv_shape(layers[0], spec)
        taps = found["kernel"] if cls._dynamic(model, layers, spec) else 1
        kind = spec.get("kind", "")
        if taps > 1:
            windows = [(f"{kind}.k", taps * kv_heads * head_dim), (f"{kind}.v", taps * kv_heads * head_dim)]
        else:   # a static convolution's keys and values side by side, one window
            windows = [(f"{kind}.kv", 2 * kv_heads * head_dim)]
        windows += [(f"{kind}.{name}", model.hidden) for name in (spec.get("post") or {})]
        return [{"key": key, "kind": "window", "layers": list(layers), "channels": channels,
                 "kernel": found["kernel"]} for key, channels in windows]

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        torch = m.torch
        self.layer = layer
        found = self._config(m, spec)
        self.kernel, self.d_rel = int(found["kernel"]), int(found["d_rel"])
        self.qk, self.rel = float(found["qk"]), float(found["rel"])
        self.log_scaling = bool(spec.get("log_scaling"))
        self.alpha, self.floor = float(found["alpha"] or 0.0), float(found["floor"] or 1.0)
        self.heads = int(m.setting(spec["heads"]) if spec.get("heads") else m.heads)
        self.kv_heads, self.head_dim = m.kv_shape(layer, spec)
        kind = spec.get("kind", "")
        window = spec.get("window")
        if window is None and kind == "sliding_attention":
            window = m.config.sliding_window
        self.window = int(window or 0)
        self.extent = self.window or int(found["rel_extent"] or 0)
        if not self.extent:
            raise ValueError("model-template: relative attention's bias reaches the sliding window or the config's "
                             "rel_extent")
        for fact in ("attention:distance-bias", "attention:lse") + (("attention:sliding",) if self.window else ()):
            if fact not in m.parts.facts("attention"):
                raise ValueError(f"model-template: relative attention needs the attention plugin's {fact}")
        if m.recurrence is None:
            raise ValueError("model-template: relative attention's convolutions keep their windows in the "
                             "recurrence slot: the selection leaves it empty")
        self.pool = ("kv", self.window) if self.window else "kv"
        names = ["q_proj.linear", "o_proj.gate_proj", "sink_proj", "relative_position_bias.linear"]
        q_rows = self.heads * self.head_dim
        rows = [q_rows, q_rows, self.heads, self.heads * self.d_rel]
        for name, want in zip(names, rows):
            if m.matrices.rows(prefix + name) != want:
                raise ValueError(f"model-template: {prefix}{name} gives {m.matrices.rows(prefix + name)} rows, the "
                                 f"layer's {self.heads} heads take {want}")
        self.query_width = sum(rows)
        dynamic = self.dynamic_names(m, prefix)
        span = getattr(m.residual, "span", None)
        # A layer outside a looped span always takes one input for its queries, keys and values (only a later pass's
        # message gates give them three): its static layers read all three as one product, and hold nothing else.
        single = not dynamic and not (span and span[0] <= m.sources[layer] < span[1])
        self.qkv_all = m.matrices.together([prefix + n for n in names] + [prefix + "k_proj.linear",
                                                                            prefix + "v_proj.linear"], "qkv") \
            if single else None
        self.query_rows = None if single else m.matrices.together([prefix + n for n in names], "qkv")
        # a dynamic layer's key and value rows come with their taps' projection (self.taps), one product each
        self.k, self.v = (None, None) if single or dynamic else (
            m.matrices.load([prefix + n + "_proj.linear"], "qkv") for n in "kv")
        # Keys and values from one row (a layer whose residual gives its mixer one input): one product of both
        self.kv = None if dynamic or single else m.matrices.together(
            [prefix + "k_proj.linear", prefix + "v_proj.linear"], "qkv")
        self.o = m.matrices.load([prefix + "o_proj.linear"], "o")
        self.q_scale = _tensor(m, prefix + "q_proj.norm.scale").float().view(self.heads, self.head_dim)
        self.r_scale = _tensor(m, prefix + "relative_position_bias.norm.scale").float().view(self.heads, self.d_rel)
        proj = _tensor(m, prefix + "relative_position_bias.proj").float()
        if tuple(proj.shape) != (self.d_rel, self.extent):
            raise ValueError(f"model-template: {prefix}relative_position_bias.proj is {list(proj.shape)}, the bias "
                             f"takes [{self.d_rel}, {self.extent}]")
        self.proj = proj.contiguous()
        channels = self.kv_heads * self.head_dim
        self.dynamic = prefix + "k_proj.sconv.static_weight" in m.ck
        self.taps = {}
        for n in "kv":
            if self.dynamic:   # static taps [kernel][channels] by lag (0: the token's own), a projection of the input
                static = _tensor(m, prefix + n + "_proj.sconv.static_weight").float()
                # static taps, and the rows with their taps' projection of one input as one product
                self.taps[n] = (static.contiguous(), m.matrices.together([prefix + n + "_proj.linear",
                                                                          prefix + n + "_proj.sconv.weight_proj"],
                                                                         "qkv"))
            else:              # [channels][kernel], the last tap the token's own: the recurrence slot's order
                self.taps[n] = _tensor(m, prefix + n + "_proj.sconv.conv1d.weight").float().reshape(channels, -1)
        if self.dynamic:   # x repeated once a tap, each copy read through one tap: copy i the input i rows back
            k = self.kernel
            pick = torch.zeros(k, channels, k, dtype=torch.float32, device=m.device)
            for lag in range(k):
                pick[lag, :, k - 1 - lag] = 1.0
            self.pick = pick.reshape(k * channels, k).contiguous()
        self.post = {name: _tensor(m, m.layer_prefix(layer) + at + "conv1d.weight").float().reshape(m.hidden, -1)
                     for name, at in (spec.get("post") or {}).items()}
        self.windows = ([f"{kind}.k", f"{kind}.v"] if self.dynamic else [f"{kind}.kv"]) + \
            [f"{kind}.{name}" for name in self.post]
        if not self.dynamic:   # the keys' taps, then the values': one convolution over [keys | values]
            self.taps["kv"] = torch.cat([self.taps["k"], self.taps["v"]]).contiguous()
        self.ones = torch.ones(self.head_dim, dtype=torch.float32, device=m.device)
        self.ops = m.kernel("row-ops")   # crit's row operations on CUDA, None for the torch forms

    def _convolve(self, key, x, weight, ctx):
        """(convolution + x) / sqrt(kernel + 1) of rows x [tokens][channels] over the window `key`, in FP32."""
        m = self.m
        y = m.recurrence.convolve(self.layer, x.float().contiguous(), weight, None, m.pools[key], ctx.batch,
                                  activation=None)
        return (y.float() + x.float()) / math.sqrt(self.kernel + 1)

    @staticmethod
    def dynamic_names(m, prefix):
        return prefix + "k_proj.sconv.static_weight" in m.ck

    def _static_keys_values(self, both, ctx):
        """Keys and values of a static convolution from their rows side by side ([tokens][keys | values]): one
        convolution over both, then the keys' clip and the values' norm per head."""
        m, torch = self.m, self.m.torch
        t, width = both.shape[0], self.kv_heads * self.head_dim
        y = self._joined(self.windows[0], both, self.taps["kv"], ctx)
        k = _clipped(torch, y[:, :width].view(t, self.kv_heads, self.head_dim), self.qk,
                     out_dtype=both.dtype, ops=self.ops).to(both.dtype)
        v = m.norms.rms_norm(y[:, width:].contiguous().view(t, self.kv_heads, self.head_dim), None, m.eps,
                             out_dtype=both.dtype)
        return k, v

    def _projected(self, n, x_in, ctx, rows=None):
        """Keys or values: the projection (or its rows, given), its convolution, then per head the clip (keys) or the
        norm (values)."""
        m, torch = self.m, self.m.torch
        own = None
        if rows is None and self.dynamic:   # the rows and the taps' projection in one product
            found = [ctx.mm(x_in, matrix) for matrix in self.taps[n][1]]
            found = found[0] if len(found) == 1 else torch.cat(found, dim=-1)
            rows, own = found[:, :self.kv_heads * self.head_dim], found[:, self.kv_heads * self.head_dim:]
        elif rows is None:
            rows = ctx.mm(x_in, self.k if n == "k" else self.v)
        t, channels = rows.shape[0], self.kv_heads * self.head_dim
        kind_key = self.windows[0 if n == "k" else 1]
        if self.dynamic:
            static = self.taps[n][0]
            own = own.float().view(t, self.kv_heads, 1, self.kernel)
            copies = rows.float().repeat(1, self.kernel)                                  # [t][kernel * channels]
            back = m.recurrence.convolve(self.layer, copies.contiguous(), self.pick, None, m.pools[kind_key],
                                         ctx.batch, activation=None).float()
            back = back.view(t, self.kernel, self.kv_heads, self.head_dim)               # back[:, lag]: lag rows back
            taps = static.view(self.kernel, self.kv_heads, self.head_dim) + own.permute(0, 3, 1, 2)
            y = (row_sum((back * taps).permute(0, 2, 3, 1)).reshape(t, channels) + rows.float()) \
                / math.sqrt(2 * self.kernel + 1)
        else:
            y = self._joined(kind_key, rows, self.taps[n], ctx)
        y = y.to(rows.dtype).view(t, self.kv_heads, self.head_dim)
        if n == "k":
            return _clipped(torch, y, self.qk, out_dtype=rows.dtype, ops=self.ops).to(rows.dtype)
        return m.norms.rms_norm(y.contiguous(), None, m.eps, out_dtype=rows.dtype)

    def post_convolution(self, name, y, ctx):
        """The residual's convolution `name` (after the attention, after the FFN) of rows y, in y's dtype."""
        key = f"{self.windows[0].rsplit('.', 1)[0]}.{name}"
        return self._joined(key, y, self.post[name], ctx)

    def _joined(self, key, x, weight, ctx):
        """(convolution + x) / sqrt(kernel + 1) of rows x over the window `key`, in x's dtype: with crit's row-ops
        the convolution reads x as it is where the model keeps 16-bit convolution rows (quality mode speed), FP32
        otherwise, and the join is one launch."""
        if self.ops is None:
            return self._convolve(key, x, weight, ctx).to(x.dtype)
        m = self.m
        xf = x.contiguous() if m.conv_rows_as_stored else x.float().contiguous()
        conv = m.recurrence.convolve(self.layer, xf, weight, None, m.pools[key], ctx.batch, activation=None)
        return self.ops.scaled_add(conv, xf, 1.0 / math.sqrt(self.kernel + 1), x.dtype)

    def forward(self, h, ctx):
        m, torch = self.m, self.m.torch
        q_in, k_in, v_in = h if isinstance(h, tuple) else (h, h, h)
        t, dtype = q_in.shape[0], q_in.dtype
        one = self.qkv_all is not None
        found = [ctx.mm(q_in, matrix) for matrix in (self.qkv_all if one else self.query_rows)]
        rows = found[0] if len(found) == 1 else torch.cat(found, dim=-1)
        if one:   # the query rows, then the keys' and values' rows
            rows, joined = rows[:, :self.query_width], rows[:, self.query_width:]
        qw = self.heads * self.head_dim
        tau = None
        if self.log_scaling:   # 1 + alpha * log(max((position + 1) / floor, 1)), on the clipped rows
            tau = (1.0 + self.alpha * torch.log(((ctx.positions.float() + 1.0) / self.floor).clamp(min=1.0)))
            tau = tau.contiguous()
        q = _clipped(torch, rows[:, :qw].view(t, self.heads, self.head_dim), self.qk, self.q_scale, tau, dtype,
                     ops=self.ops)
        r = _clipped(torch, rows[:, 2 * qw + self.heads:].view(t, self.heads, self.d_rel), self.rel, self.r_scale,
                     tau, ops=self.ops)
        q = q.to(dtype).contiguous()
        if self.ops is not None and dtype == torch.bfloat16:
            bias = self.ops.small_product(r, self.proj)
        else:
            bias = (r.to(dtype).float() @ self.proj).to(dtype).float().contiguous()      # [t][heads][extent]
        sink = rows[:, 2 * qw:2 * qw + self.heads].float()
        if one or (k_in is v_in and self.kv is not None):   # one product of the keys' and values' rows
            if one:
                both = joined
            else:
                found = [ctx.mm(k_in, matrix) for matrix in self.kv]
                both = found[0] if len(found) == 1 else torch.cat(found, dim=-1)
            k, v = self._static_keys_values(both, ctx)
        elif not self.dynamic:
            k, v = self._static_keys_values(torch.cat([ctx.mm(k_in, self.k), ctx.mm(v_in, self.v)], dim=-1), ctx)
        else:
            k, v = self._projected("k", k_in, ctx), self._projected("v", v_in, ctx)
        pool = m.pools[self.pool]
        pool.write(self.layer, ctx.slots[self.pool], k, v)
        a, lse = m.attention.forward(self.layer, q, pool, ctx.batch, scale=self.head_dim ** -0.5, lse=True,
                                     bias=bias, **({"sliding": self.window} if self.window else {}))
        if self.ops is not None and dtype == torch.bfloat16:
            gated = self.ops.sink_gate(a, lse, rows[:, 2 * qw:2 * qw + self.heads], rows[:, qw:2 * qw])
            return ctx.mm(gated, self.o)
        a = a.float() * torch.sigmoid(lse - sink)[..., None]                              # the sink: no value
        gated = (a.reshape(t, qw) * torch.sigmoid(rows[:, qw:2 * qw].float())).to(dtype)
        return ctx.mm(gated, self.o)

BLOCKS = {"relative-attention": RelativeAttention}
