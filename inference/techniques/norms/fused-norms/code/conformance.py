"""The checks a norms device form passes (crit_plugins.device.DEVICE_INTERFACES["norms"]), for any norms plugin's
tests: torch-norms', fused-norms' and an engine's own.

- against_rows: every method against the FP64 host form (rows.py) on the inputs as given, FP32 out within FP32
  rounding (TOLERANCE), 16-bit out within one rounding of the output dtype.
- rows_alone: every method gives each row the bits it gets alone, in calls of 1 to 8 rows and of a few hundred
  (decode, verify and prompt rows agree), at hidden widths where a sum over the row is long; odd_layouts the same at
  a width no multiple of 16 and on a head view with an odd token stride.
- strided: views of a fused projection's output (a head's query beside its gate, a slice of [q | k | v | z]) give the
  bits their contiguous copies give.
- captured: a CUDA graph of every method replays to the bits eager calls give on new inputs.
- dtypes: BF16, FP16 and FP32 inputs and outputs, an FP32 residual beside BF16 rows, out= and an in-place rope.
- streams: a weight per stream or head, the gated, weighted and plain sums over residual streams and a row added
  into every stream, in each of the checks above.
- prompt_chunks (prompts:norms): every method in a prompt forward over crit_plugins.prompt_graphs' padded buckets,
  eager and captured, against the unpadded chunk, with junk in the padding (the fused forms too where the plugin has
  them).
- fused_forms (norms:fused): each fused form gives every output the bits the plugin's separate calls give, at every
  row count, on views, with a weight per head or stream or none, into out= and in place, and captured.
"""
from . import rows

# FP32 arithmetic against FP64: |device - host| <= ATOL + RTOL * |host| with FP32 out.
TOLERANCE = {"atol": 2e-5, "rtol": 2e-5}
# One rounding to a 16-bit output on top (BF16 keeps 8 significant bits, FP16 11).
ROUNDED = {"bfloat16": 2.0 ** -8, "float16": 2.0 ** -11}


def _torch():
    import torch
    return torch


class Inputs:
    """Seeded inputs on a device."""

    def __init__(self, device, seed=0):
        self.torch = _torch()
        self.device = device
        self.generator = self.torch.Generator().manual_seed(seed)

    def normal(self, *shape, scale=1.0, dtype=None):
        t = self.torch.randn(*shape, generator=self.generator) * scale
        return t.to(device=self.device, dtype=dtype or self.torch.float32)

    def positions(self, count, top=1 << 18):
        return self.torch.randint(0, top, (count,), generator=self.generator).to(self.device)


def inv_freq(rotary_dim, theta=10000.0, device="cpu"):
    torch = _torch()
    return (1.0 / theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)).float().to(device)


def cos_sin_table(freqs, positions):
    """A [positions][rotary_dim] FP32 table: cos of every pair, then sin, from FP64 angles."""
    torch = _torch()
    p = torch.arange(positions, dtype=torch.float64)[:, None] * freqs.double().cpu()[None, :]
    return torch.cat([p.cos(), p.sin()], -1).float().to(freqs.device)


def _close(case, found, expected, out_dtype, what, scale=None):
    torch = _torch()
    found = found.detach().double().cpu()
    expected = torch.tensor(expected, dtype=torch.float64).reshape(found.shape)
    atol, rtol = TOLERANCE["atol"], TOLERANCE["rtol"]
    step = ROUNDED.get(str(out_dtype).replace("torch.", ""))
    if step:
        rtol += step
        atol += step * (float(expected.abs().max()) if scale is None else scale) * 0.01
    worst = (found - expected).abs() - (atol + rtol * expected.abs())
    if not torch.isfinite(found).all() or float(worst.max()) > 0:
        i = int(worst.flatten().argmax())
        case.fail(f"{what}: {float(found.flatten()[i])!r} differs from the FP64 host form's "
                  f"{float(expected.flatten()[i])!r} by more than atol {atol:g} + rtol {rtol:g}")


def _lists(t):
    return t.detach().double().cpu().tolist()


def _flat(t):
    return t.detach().double().cpu().reshape(-1, t.shape[-1]).tolist()


def against_rows(case, norms, device, dtypes=None):
    """Every method of the device form against the FP64 host form, for each input dtype, out FP32 and out 16-bit."""
    torch = _torch()
    dtypes = dtypes or (torch.bfloat16, torch.float16, torch.float32)
    for dtype in dtypes:
        g = Inputs(device, seed=11)
        for width in (64, 520, 2560):
            x, r = g.normal(3, width, scale=2.0, dtype=dtype), g.normal(3, width, scale=2.0, dtype=dtype)
            w = g.normal(width, scale=0.3) + 1.0
            for out_dtype in (torch.float32, dtype):
                what = f"rms_norm {dtype} width {width} out {out_dtype}"
                _close(case, norms.rms_norm(x, w, 1e-6, out_dtype=out_dtype),
                       rows.rms_norm_rows(_lists(x), _lists(w), 1e-6), out_dtype, what)
                _close(case, norms.rms_norm(x, w.to(dtype), 1e-5, offset=1.0, out_dtype=out_dtype),
                       rows.rms_norm_rows(_lists(x), _lists(w.to(dtype)), 1e-5, offset=1.0), out_dtype,
                       what + " offset 1")
            _close(case, norms.rms_norm(x, None, 1e-6, out_dtype=torch.float32),
                   rows.rms_norm_rows(_lists(x), None, 1e-6), torch.float32, f"rms_norm without a weight {dtype}")
            residual = r.float().clone()
            y, kept = norms.rms_norm(x, w, 1e-6, residual=residual, out_dtype=torch.float32)
            expected, sums = rows.rms_norm_rows(_lists(x), _lists(w), 1e-6, residual=_lists(r.float()))
            case.assertIs(kept, residual, "rms_norm updates the residual in place and returns it")
            _close(case, y, expected, torch.float32, f"rms_norm with an FP32 residual {dtype}")
            _close(case, kept, sums, torch.float32, f"the FP32 residual sums {dtype}")
            residual = r.clone()
            y, kept = norms.rms_norm(x, w, 1e-6, residual=residual)
            case.assertEqual(y.dtype, dtype, "y defaults to the residual's dtype")
            _, sums = rows.rms_norm_rows(_lists(x), _lists(w), 1e-6, residual=_lists(r))
            _close(case, kept, sums, dtype, f"the residual sums in {dtype}")
            # The rows normalized are the sums as the residual keeps them, rounded to its dtype.
            _close(case, y, rows.rms_norm_rows(_lists(kept), _lists(w), 1e-6), dtype, f"rms_norm with a {dtype} residual")
        zero = torch.zeros(2, 640, dtype=dtype, device=device)
        case.assertTrue(torch.equal(norms.rms_norm(zero, None, 1e-6), zero), "a zero row normalizes to zeros")
        for heads, dim, rotary in ((4, 64, 16), (3, 128, 128), (2, 256, 64)):
            x = g.normal(5, heads, dim, scale=2.0, dtype=dtype)
            w = g.normal(dim, scale=0.3) + 1.0
            positions = g.positions(5)
            heads_rows = _flat(x)
            per_head = [int(p) for p in positions.tolist() for _ in range(heads)]
            normed = rows.rms_norm_rows(heads_rows, _lists(w), 1e-6, offset=0.5)
            _close(case, norms.rms_norm_heads(x, w, 1e-6, offset=0.5, out_dtype=torch.float32), normed,
                   torch.float32, f"rms_norm_heads {dtype} {heads}x{dim}")
            for style in ("half", "interleaved"):
                freqs = inv_freq(rotary, 1e6, device)
                expected = rows.rope_rows(heads_rows, per_head, _lists(freqs), rotary, style, scale=1.5)
                for out_dtype in (torch.float32, dtype):
                    _close(case, norms.rope(x, positions, freqs, rotary, style, 1.5, out_dtype=out_dtype), expected,
                           out_dtype, f"rope {style} {dtype} {heads}x{dim} rotary {rotary} out {out_dtype}")
                _close(case, norms.rms_norm_rope(x, w, 1e-6, positions, freqs, 0.5, rotary, style,
                                                 out_dtype=torch.float32),
                       rows.rope_rows(normed, per_head, _lists(freqs), rotary, style), torch.float32,
                       f"rms_norm_rope {style} {dtype} {heads}x{dim}")
                small = positions % 64
                table = cos_sin_table(freqs, 64)
                _close(case, norms.rope(x, small, table, style=style, out_dtype=torch.float32),
                       rows.rope_rows(heads_rows, [int(p) for p in small.tolist() for _ in range(heads)],
                                      _lists(table), rotary, style), torch.float32, f"rope from a table {style}")
            z = g.normal(5, heads, dim, scale=2.0, dtype=dtype)
            for activation in ("silu", "sigmoid"):
                _close(case, norms.gated_rms_norm(x, z, w, 1e-6, activation=activation, out_dtype=torch.float32),
                       rows.gated_rms_norm_rows(heads_rows, _flat(z), _lists(w), 1e-6, activation=activation),
                       torch.float32, f"gated_rms_norm {activation} {dtype}")
            _close(case, norms.l2_norm(x, 1e-6, out_dtype=torch.float32), rows.l2_norm_rows(heads_rows, 1e-6),
                   torch.float32, f"l2_norm {dtype} {heads}x{dim}")
        x = g.normal(4, 2 * 384, scale=3.0, dtype=dtype)
        halves = (_lists(x[:, :384]), _lists(x[:, 384:]))
        pairs = (_lists(x[:, 0::2]), _lists(x[:, 1::2]))
        for out_dtype in (torch.float32, dtype):
            _close(case, norms.silu_mul(x, out_dtype=out_dtype), rows.act_mul_rows(*halves, "silu"), out_dtype,
                   f"silu_mul halves {dtype} out {out_dtype}")
        _close(case, norms.silu_mul(x, layout="interleaved", out_dtype=torch.float32),
               rows.act_mul_rows(*pairs, "silu"), torch.float32, f"silu_mul interleaved {dtype}")
        _close(case, norms.silu_mul(x[:, :384], x[:, 384:], out_dtype=torch.float32),
               rows.act_mul_rows(*halves, "silu"), torch.float32, f"silu_mul of two tensors {dtype}")
        _close(case, norms.gelu_mul(x, out_dtype=torch.float32), rows.act_mul_rows(*halves, "gelu"), torch.float32,
               f"gelu_mul {dtype}")
        _close(case, norms.gelu_mul(x, layout="interleaved", approximate="tanh", out_dtype=torch.float32),
               rows.act_mul_rows(*pairs, "gelu_tanh"), torch.float32, f"gelu_mul tanh interleaved {dtype}")
        _close(case, norms.sigmoid_mul(x[:, 384:], x[:, :384], out_dtype=torch.float32),
               rows.act_mul_rows(halves[0], halves[1], "sigmoid"), torch.float32, f"sigmoid_mul {dtype}")
        streams = g.normal(5, 4, 320, scale=2.0, dtype=dtype)               # 5 tokens of 4 residual streams
        grouped = g.normal(4, 320, scale=0.3) + 1.0
        _close(case, norms.rms_norm(streams, grouped, 1e-6, offset=1.0, out_dtype=torch.float32),
               rows.rms_norm_rows(_flat(streams), _lists(grouped), 1e-6, offset=1.0), torch.float32,
               f"rms_norm with a weight per stream {dtype}")
        heads = g.normal(3, 4, 64, scale=2.0, dtype=dtype)
        per_head = g.normal(4, 64, scale=0.3) + 1.0
        freqs = inv_freq(32, 1e4, device)
        positions = g.positions(3)
        _close(case, norms.rms_norm_rope(heads, per_head, 1e-6, positions, freqs, out_dtype=torch.float32),
               rows.rope_rows(rows.rms_norm_rows(_flat(heads), _lists(per_head), 1e-6),
                              [int(p) for p in positions.tolist() for _ in range(4)], _lists(freqs)), torch.float32,
               f"rms_norm_rope with a weight per head {dtype}")
        gates = g.normal(5, 4, 320, scale=2.0, dtype=dtype)
        per_stream = g.normal(5, 4, scale=2.0)
        for gate, activation, scale in ((None, "none", 0.25), (gates, "sigmoid", 1.0), (per_stream, "none", 1.0),
                                        (per_stream, "silu", 0.5), (gates, "none", 1.0)):
            found = norms.stream_sum(streams, gate, activation, scale, out_dtype=torch.float32)
            case.assertEqual(tuple(found.shape), (5, 320))
            _close(case, found, rows.stream_sum_rows(_lists(streams), None if gate is None else _lists(gate),
                                                     activation, scale), torch.float32,
                   f"stream_sum {activation} gate {None if gate is None else tuple(gate.shape)} {dtype}")
        row = g.normal(5, 320, scale=2.0, dtype=dtype)
        for weights in (None, per_stream):
            kept = streams.float().clone()
            expected = rows.stream_add_rows(_lists(kept), _lists(row), None if weights is None else _lists(weights))
            case.assertIs(norms.stream_add(kept, row, weights), kept, "stream_add advances the streams in place")
            _close(case, kept, expected, torch.float32, f"stream_add weights {weights is not None} {dtype}")
        ba = g.normal(4, 2 * 48, scale=4.0, dtype=dtype)                  # [beta gates | decay inputs] of a projection
        bias, rate = g.normal(48), g.normal(48).exp()
        decay, beta = rows.softplus_decay_rows(_lists(ba[:, 48:]), _lists(bias), _lists(rate), _lists(ba[:, :48]))
        found = norms.softplus_decay(ba[:, 48:], bias, rate, ba[:, :48])
        case.assertEqual([t.dtype for t in found], [torch.float32] * 2, "softplus_decay's outputs default to FP32")
        _close(case, found[0], decay, torch.float32, f"softplus_decay {dtype}")
        _close(case, found[1], beta, torch.float32, f"softplus_decay's sigmoid gate {dtype}")
        _close(case, norms.softplus_decay(ba[:, 48:], bias, rate, out_dtype=dtype), decay, dtype,
               f"softplus_decay out {dtype}")


def _calls(norms, g, width, heads, dim, rotary, dtype, rows_count=320):
    """name -> (function of a row count n -> the result for the first n rows) at one shape."""
    torch = _torch()
    x = g.normal(rows_count, width, scale=2.0, dtype=dtype)
    r = g.normal(rows_count, width, scale=2.0, dtype=torch.float32)
    w = g.normal(width, scale=0.3, dtype=dtype) + 1
    h = g.normal(rows_count, heads, dim, scale=2.0, dtype=dtype)
    z = g.normal(rows_count, heads, dim, scale=2.0, dtype=dtype)
    hw = g.normal(dim, scale=0.3) + 1
    hw16 = hw.to(dtype)
    positions = g.positions(rows_count)
    freqs = inv_freq(rotary, 1e7, h.device)
    gate_up = g.normal(rows_count, 2 * width, scale=3.0, dtype=dtype)
    streams = g.normal(rows_count, 4, width, scale=2.0, dtype=torch.float32)
    sw = g.normal(4, width, scale=0.3) + 1
    gates = g.normal(rows_count, 4, width, scale=2.0, dtype=dtype)
    per_stream = g.normal(rows_count, 4)
    ba = g.normal(rows_count, 96, scale=4.0, dtype=dtype)
    bias, rate = g.normal(48), g.normal(48).exp()

    def added(n):
        kept = streams[:n].clone()
        return norms.stream_add(kept, x[:n].clone(), per_stream[:n].clone())

    return {
        "rms_norm": lambda n: norms.rms_norm(x[:n].clone(), w, 1e-6),
        "rms_norm offset": lambda n: norms.rms_norm(x[:n].clone(), w, 1e-6, offset=1.0, out_dtype=torch.float16),
        "rms_norm residual": lambda n: torch.cat([t.float() for t in norms.rms_norm(
            x[:n].clone(), w, 1e-6, residual=r[:n].clone(), out_dtype=torch.bfloat16)], -1),
        "rms_norm_heads": lambda n: norms.rms_norm_heads(h[:n].clone(), hw, 1e-6, offset=1.0),
        "rms_norm_heads 16-bit weight": lambda n: norms.rms_norm_heads(h[:n].clone(), hw16, 1e-6, offset=1.0),
        "rms_norm 16-bit weight offset": lambda n: norms.rms_norm(x[:n].clone(), w, 1e-6, offset=1.0),
        "rms_norm_rope": lambda n: norms.rms_norm_rope(h[:n].clone(), hw, 1e-6, positions[:n].clone(), freqs,
                                                       rotary_dim=rotary),
        "rope": lambda n: norms.rope(h[:n].clone(), positions[:n].clone(), freqs, style="interleaved"),
        "gated_rms_norm": lambda n: norms.gated_rms_norm(h[:n].clone(), z[:n].clone(), hw, 1e-6),
        "l2_norm": lambda n: norms.l2_norm(h[:n].clone()),
        "silu_mul": lambda n: norms.silu_mul(gate_up[:n].clone()),
        "gelu_mul": lambda n: norms.gelu_mul(gate_up[:n].clone(), layout="interleaved"),
        "sigmoid_mul": lambda n: norms.sigmoid_mul(x[:n].clone(), gate_up[:n, :width].clone()),
        "rms_norm per stream": lambda n: norms.rms_norm(streams[:n].clone(), sw, 1e-6, offset=1.0,
                                                        out_dtype=torch.bfloat16),
        "stream_sum": lambda n: norms.stream_sum(streams[:n].clone(), gates[:n].clone(), "sigmoid", 0.25),
        "stream_add": added,
        "softplus_decay": lambda n: torch.cat(norms.softplus_decay(ba[:n, 48:].clone(), bias, rate,
                                                                   ba[:n, :48].clone()), -1),
    }


SHAPES = ((5120, 24, 256, 64), (7168, 56, 128, 128), (2560, 2, 256, 64))


COUNTS = (1, 2, 3, 4, 5, 7, 8, 64, 320)
CPU_COUNTS = (1, 3, 4, 7, 8, 40)   # the CPU's row counts: fewer and shorter calls, the same widths


def rows_alone(case, norms, device, shapes=SHAPES, counts=None):
    """Each method gives a row the same bits in calls of every row count."""
    torch = _torch()
    counts = counts or (COUNTS if str(device).startswith("cuda") else CPU_COUNTS)
    for width, heads, dim, rotary in shapes:
        calls = _calls(norms, Inputs(device, seed=width), width, heads, dim, rotary, torch.bfloat16, max(counts))
        for name, call in calls.items():
            whole = call(max(counts))
            for n in counts:
                if not torch.equal(call(n), whole[:n]):
                    case.fail(f"{name} at width {width} ({heads}x{dim}): the rows of a {n}-row call differ from the "
                              f"same rows in a {max(counts)}-row call")


def odd_layouts(case, norms, device, counts=(1, 2, 3, 8), rows_count=64):
    """Each row's bits whatever the row count where a layout is not aligned: a width no multiple of 16 (BF16 in, FP32
    out, an FP32 residual) and a head view whose token stride is odd."""
    torch = _torch()
    g = Inputs(device, seed=41)
    f32 = torch.float32
    width = 4100
    x = g.normal(rows_count, width, scale=2.0, dtype=torch.bfloat16)
    r = g.normal(rows_count, width)
    w = g.normal(width, scale=0.1) + 1
    z = g.normal(rows_count, width, dtype=torch.bfloat16)
    heads, dim = 8, 128
    big = g.normal(rows_count, heads * dim + 1, dtype=torch.bfloat16)        # token stride 1025
    q = big[:, :heads * dim].view(rows_count, heads, dim)
    hw = g.normal(dim, scale=0.1) + 1
    positions = g.positions(rows_count, 4096)
    freqs = inv_freq(dim, 1e4, device)
    calls = {
        "rms_norm": lambda n: norms.rms_norm(x[:n], w, 1e-6, out_dtype=f32),
        "l2_norm": lambda n: norms.l2_norm(x[:n], out_dtype=f32),
        "gated_rms_norm": lambda n: norms.gated_rms_norm(x[:n], z[:n], w, 1e-6, out_dtype=f32),
        "rms_norm residual": lambda n: torch.cat(norms.rms_norm(x[:n], w, 1e-6, residual=r[:n].clone(),
                                                                out_dtype=f32), -1),
        "rms_norm_heads view": lambda n: norms.rms_norm_heads(q[:n], hw, 1e-6, out_dtype=f32),
        "rms_norm_rope view": lambda n: norms.rms_norm_rope(q[:n], hw, 1e-6, positions[:n], freqs, out_dtype=f32),
        "l2_norm view": lambda n: norms.l2_norm(q[:n], out_dtype=f32),
    }
    for name, call in calls.items():
        whole = call(rows_count)
        for n in counts:
            if not torch.equal(call(n), whole[:n]):
                case.fail(f"{name}: the rows of a {n}-row call differ from the same rows in a {rows_count}-row call "
                          "(an unaligned layout)")
    if not torch.equal(calls["rms_norm_heads view"](rows_count), norms.rms_norm_heads(q.contiguous(), hw, 1e-6,
                                                                                         out_dtype=f32)):
        case.fail("rms_norm_heads of an odd-stride view differs from its contiguous copy")


def strided(case, norms, device):
    """Views of fused projection outputs give the bits of their contiguous copies."""
    torch = _torch()
    g = Inputs(device, seed=3)
    tokens, heads, dim = 6, 4, 64
    qg = g.normal(tokens, heads, 2 * dim, dtype=torch.bfloat16)          # each head's query beside its gate
    q, gate = qg[..., :dim], qg[..., dim:]
    w = g.normal(dim, scale=0.2) + 1
    positions = g.positions(tokens)
    freqs = inv_freq(16, 1e4, device)
    pairs = (
        (lambda t: norms.rms_norm_heads(t, w, 1e-6, offset=1.0), q),
        (lambda t: norms.rms_norm_rope(t, w, 1e-6, positions, freqs, offset=1.0), q),
        (lambda t: norms.rope(t, positions, freqs), q),
        (lambda t: norms.l2_norm(t), q),
        (lambda t: norms.sigmoid_mul(t, gate), q),
        (lambda t: norms.gated_rms_norm(t, gate, w, 1e-6), q),
    )
    for i, (call, view) in enumerate(pairs):
        if not torch.equal(call(view), call(view.contiguous())):
            case.fail(f"call {i}: a strided view gives other bits than its contiguous copy")
    fused = g.normal(tokens, 3 * 512, dtype=torch.bfloat16)               # [a | b | c] of one projection
    rows_view = fused[:, 512:1024]
    wide = g.normal(512, scale=0.2) + 1
    case.assertTrue(torch.equal(norms.rms_norm(rows_view, wide, 1e-6), norms.rms_norm(rows_view.contiguous(), wide, 1e-6)),
                    "rms_norm of a column slice")
    # the same with weights in the rows' dtype (calls a provider may take)
    for weight in (wide.to(torch.bfloat16),):
        for offset in (0.0, 1.0):
            case.assertTrue(torch.equal(norms.rms_norm(rows_view, weight, 1e-6, offset=offset),
                                        norms.rms_norm(rows_view.contiguous(), weight, 1e-6, offset=offset)),
                            f"rms_norm of a column slice, a BF16 weight, offset {offset}")
    w16 = w.to(torch.bfloat16)
    case.assertTrue(torch.equal(norms.rms_norm_heads(q, w16, 1e-6), norms.rms_norm_heads(q.contiguous(), w16, 1e-6)),
                    "rms_norm_heads of a view, a BF16 weight")
    packed = g.normal(tokens, 3 * 512, dtype=torch.bfloat16)[:, :1024]     # [gate | up] beside another matrix's rows
    case.assertTrue(torch.equal(norms.silu_mul(packed), norms.silu_mul(packed.contiguous())), "silu_mul of a slice")
    case.assertTrue(torch.equal(norms.silu_mul(fused[:, :512], fused[:, 1024:]),
                                norms.silu_mul(fused[:, :512].contiguous(), fused[:, 1024:].contiguous())),
                    "silu_mul of two column slices")
    split = g.normal(tokens, 2, 3 * 2 * 32, dtype=torch.bfloat16)        # per group [x | z], each 2 heads of 32
    x4 = split[..., :64].view(tokens, 2, 2, 32)
    z4 = split[..., 64:128].view(tokens, 2, 2, 32)
    hw = g.normal(32, scale=0.2) + 1
    case.assertTrue(torch.equal(norms.gated_rms_norm(x4, z4, hw, 1e-6),
                                norms.gated_rms_norm(x4.contiguous(), z4.contiguous(), hw, 1e-6)),
                    "gated_rms_norm of 4-dimensional views")
    ba = g.normal(tokens, 16 + 2 * 48, dtype=torch.bfloat16)[:, 16:]       # [beta | decay] beside another matrix's
    bias, rate = g.normal(48), g.normal(48).exp()
    found = norms.softplus_decay(ba[:, 48:], bias, rate, ba[:, :48])
    again = norms.softplus_decay(ba[:, 48:].contiguous(), bias, rate, ba[:, :48].contiguous())
    case.assertTrue(all(torch.equal(a, b) for a, b in zip(found, again)), "softplus_decay of column slices")


def outputs(case, norms, device):
    """out= receives the result, rope rotates in place, and the dtypes are the ones asked for."""
    torch = _torch()
    g = Inputs(device, seed=5)
    x = g.normal(3, 2, 64, dtype=torch.bfloat16)
    positions = g.positions(3)
    freqs = inv_freq(64, 1e4, device)
    expected = norms.rope(x, positions, freqs)
    out = torch.empty_like(x)
    case.assertIs(norms.rope(x, positions, freqs, out=out), out)
    case.assertTrue(torch.equal(out, expected), "rope into out")
    inplace = x.clone()
    case.assertIs(norms.rope(inplace, positions, freqs, out=inplace), inplace)
    case.assertTrue(torch.equal(inplace, expected), "rope in place")
    w = g.normal(64) + 1
    for out_dtype in (torch.float16, torch.float32, torch.bfloat16):
        case.assertEqual(norms.rms_norm(x, w, 1e-6, out_dtype=out_dtype).dtype, out_dtype)
        case.assertEqual(norms.silu_mul(x.view(3, 128), out_dtype=out_dtype).dtype, out_dtype)
    y = torch.empty(3, 2, 64, dtype=torch.float16, device=device)
    case.assertIs(norms.rms_norm_heads(x, w, 1e-6, out_dtype=torch.float16, out=y), y)
    case.assertTrue(torch.equal(y, norms.rms_norm_heads(x, w, 1e-6, out_dtype=torch.float16)))
    case.assertEqual(tuple(norms.silu_mul(x.view(3, 128)).shape), (3, 64))
    # outs whose rows are strided (a slice of a wider buffer), with weights of x's dtype too
    rows2 = x.view(6, 64)
    for weight in (w, w.to(x.dtype)):
        wide = torch.zeros(6, 96, dtype=x.dtype, device=device)
        case.assertEqual(norms.rms_norm(rows2, weight, 1e-6, out=wide[:, :64]).data_ptr(), wide.data_ptr())
        case.assertTrue(torch.equal(wide[:, :64], norms.rms_norm(rows2, weight, 1e-6)), "rms_norm into a strided out")
    wide = torch.zeros(3, 96, dtype=x.dtype, device=device)
    norms.silu_mul(x.view(3, 128), out=wide[:, :64])
    case.assertTrue(torch.equal(wide[:, :64], norms.silu_mul(x.view(3, 128))), "silu_mul into a strided out")
    case.assertEqual(tuple(norms.silu_mul(x, layout="interleaved").shape), (3, 2, 32))
    flat = g.normal(4, 64)
    for bad in (lambda: norms.rms_norm(x, g.normal(32), 1e-6),
                lambda: norms.silu_mul(x.view(3, 128)[:, :127]),
                lambda: norms.rope(x, positions[:2], freqs),
                lambda: norms.rope(x, positions, inv_freq(80, 1e4, device)),
                lambda: norms.gated_rms_norm(x, x[:2], w, 1e-6),
                # an out whose last dimension is not contiguous
                lambda: norms.rms_norm(x.view(6, 64), w, 1e-6, out=torch.empty(64, 6, dtype=x.dtype,
                                                                               device=device).t()),
                lambda: norms.silu_mul(x.view(3, 128), out=torch.empty(64, 3, dtype=x.dtype, device=device).t()),
                # a per-token weight for 2-D rope rows: heads are what a weight's rows go with
                lambda: norms.rms_norm_rope(flat, g.normal(4, 64), 1e-6, positions[:3].repeat(2)[:4], freqs),
                lambda: norms.softplus_decay(flat, g.normal(63), g.normal(64)),
                lambda: norms.softplus_decay(flat, g.normal(64), g.normal(64), flat[:2])):
        with case.assertRaises(ValueError):
            bad()
    # rope in place on shapes where many threads share a head: every value as the out-of-place rope gives it
    for dtype, shape in ((torch.float32, (512, 16, 256)), (torch.bfloat16, (256, 4, 1024))):
        h = g.normal(*shape, dtype=dtype)
        where = g.positions(shape[0])
        hw = g.normal(shape[-1]) + 1
        f = inv_freq(shape[-1], 1e4, device)
        for call in (lambda t, out=None: norms.rope(t, where, f, out=out),
                     lambda t, out=None: norms.rms_norm_rope(t, hw, 1e-6, where, f, out=out)):
            expected = call(h)
            for _ in range(5):
                again = h.clone()
                call(again, out=again)
                if not torch.equal(again, expected):
                    case.fail(f"an in-place rope of {shape} {dtype} differs from the out-of-place one")


def captured(case, norms, device="cuda"):
    """A CUDA graph of every method replays to the bits eager calls give, on new inputs at the same addresses."""
    torch = _torch()
    g = Inputs(device, seed=9)
    width, heads, dim, rotary, count = 2560, 4, 128, 32, 3
    x = g.normal(count, width, dtype=torch.bfloat16)
    residual = g.normal(count, width)
    w = g.normal(width) + 1
    h = g.normal(count, heads, dim, dtype=torch.bfloat16)
    z = g.normal(count, heads, dim, dtype=torch.bfloat16)
    hw = g.normal(dim) + 1
    positions = g.positions(count)
    freqs = inv_freq(rotary, 1e6, device)
    table = cos_sin_table(freqs, 128)
    small = positions % 128
    gate_up = g.normal(count, 2 * width, dtype=torch.bfloat16)
    streams = g.normal(count, 4, 128)
    sw = g.normal(4, 128) + 1
    h4 = g.normal(count, 4, 128, dtype=torch.bfloat16)
    per_stream = g.normal(count, 4)
    streams_kept = streams.clone()
    ba = g.normal(count, 96, dtype=torch.bfloat16)
    bias, rate = g.normal(48), g.normal(48).exp()

    def run():
        y, kept = norms.rms_norm(x, w, 1e-6, offset=1.0, residual=residual, out_dtype=torch.bfloat16)
        return [y, kept.clone(), norms.rms_norm(x, w, 1e-6), norms.rms_norm_heads(h, hw, 1e-6),
                norms.rms_norm_rope(h, hw, 1e-6, positions, freqs, rotary_dim=rotary),
                norms.rope(h, small, table), norms.rope(h, positions, freqs, style="interleaved"),
                norms.gated_rms_norm(h, z, hw, 1e-6), norms.l2_norm(h), norms.silu_mul(gate_up),
                norms.gelu_mul(gate_up, approximate="tanh"), norms.sigmoid_mul(x, gate_up[:, :width]),
                norms.rms_norm(x, w.to(torch.bfloat16), 1e-6), norms.rms_norm_heads(h, hw.to(torch.bfloat16), 1e-6,
                                                                                    offset=1.0),
                norms.rms_norm(streams, sw, 1e-6), norms.stream_sum(streams, h4, "sigmoid", 0.25),
                norms.stream_add(streams_kept, h4[:, 0], per_stream).clone(),
                *norms.softplus_decay(ba[:, 48:], bias, rate, ba[:, :48])]

    start, kept_start = residual.clone(), streams_kept.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):   # every shape compiled and warm before the capture
            residual.copy_(start)
            streams_kept.copy_(kept_start)
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    residual.copy_(start)
    streams_kept.copy_(kept_start)
    with torch.cuda.graph(graph):
        captured_out = run()
    fresh = Inputs(device, seed=10)
    for t in (x, residual, w, h, z, hw, gate_up, streams, sw, h4, per_stream, streams_kept, ba, bias):
        t.copy_(fresh.normal(*t.shape, dtype=t.dtype))
    positions.copy_(fresh.positions(count))
    small.copy_(positions % 128)
    start, kept_start = residual.clone(), streams_kept.clone()
    graph.replay()
    replayed = [t.clone() for t in captured_out]
    residual.copy_(start)
    streams_kept.copy_(kept_start)
    eager = run()
    for i, (a, b) in enumerate(zip(replayed, eager)):
        if not torch.equal(a, b):
            case.fail(f"output {i}: the replayed graph differs from eager calls on the same inputs")


def prompt_chunks(case, norms, device, buckets=(16, 64, 128, 512)):
    """prompts:norms: a prompt forward through every method over crit_plugins.prompt_graphs' padded buckets gives the
    unpadded eager chunk's bits on its real rows, padded eagerly (any device) and captured and replayed (CUDA), and junk
    in the padding (another token, a row of NaN) leaves the real rows as they are: no method lets a row see another."""
    from crit_plugins.device import StepBatch
    from crit_plugins.prompt_graphs import PromptGraphs
    torch = _torch()
    g = Inputs(device, seed=51)
    vocab, width, heads, dim, rotary = 97, 256, 4, 64, 32
    table = g.normal(vocab, width, scale=2.0)
    table[vocab - 1] = float("nan")                      # the row junk padding reads
    wf, w16 = g.normal(width, scale=0.2), (g.normal(width, scale=0.2) + 1).to(torch.bfloat16)
    hw, per_head = g.normal(dim, scale=0.2) + 1, g.normal(heads, dim, scale=0.2) + 1
    freqs = inv_freq(rotary, 1e4, device)
    cache = cos_sin_table(freqs, 4096)
    bf16 = torch.bfloat16
    bias, rate = g.normal(width // 2), g.normal(width // 2).exp()
    fused = callable(getattr(norms, "rms_norm_rope_pair", None))       # norms:fused

    def forward(batch):
        t = batch.tokens_t.shape[0]
        x = table.index_select(0, batch.tokens_t)
        residual = x.clone()
        h, residual = norms.rms_norm(x.to(bf16), wf, 1e-6, offset=1.0, residual=residual, out_dtype=bf16)
        plain = norms.rms_norm(x.to(bf16), w16, 1e-6)
        q = norms.rms_norm_rope(h.view(t, heads, dim), hw, 1e-6, batch.positions_t, freqs, rotary_dim=rotary)
        k = norms.rope(plain.view(t, heads, dim), batch.positions_t, cache, style="interleaved")
        gated = norms.gated_rms_norm(q, norms.rms_norm_heads(k, per_head, 1e-6), hw, 1e-6, activation="sigmoid")
        unit = norms.l2_norm(gated).view(t, width)
        product = norms.gelu_mul(norms.silu_mul(torch.cat([h, unit], -1)), unit, approximate="tanh")
        gate = norms.sigmoid_mul(product, plain)
        streams = torch.stack([residual, gate.float(), plain.float()], 1)
        mixed = norms.stream_sum(streams, streams.to(bf16), "silu", 1.0 / 3)
        norms.stream_add(streams, gate, None)
        decay, beta = norms.softplus_decay(gate[:, width // 2:], bias, rate, plain[:, :width // 2])
        out = [mixed, streams.reshape(t, -1), decay, beta]
        if fused:
            pq, pk = norms.rms_norm_rope_pair(h.view(t, heads, dim), plain.view(t, heads, dim)[:, :2], hw, per_head[:2],
                                              1e-6, batch.positions_t, freqs, rotary_dim=rotary)
            rq, rk = norms.rope_pair(h.view(t, heads, dim), plain.view(t, heads, dim)[:, 1:], batch.positions_t,
                                     cache, style="interleaved", out_dtype=torch.float32)
            normed, _ = norms.stream_add_norm(streams, mixed.to(bf16), streams[:, :, 0].contiguous(), None, 1e-6)
            out += [pq.reshape(t, -1).float(), pk.reshape(t, -1).float(), rq.reshape(t, -1), rk.reshape(t, -1),
                    normed.reshape(t, -1), streams.reshape(t, -1)]
        return torch.cat(out, -1)

    prompts = PromptGraphs(forward, device=device, buckets=buckets)
    captured = prompts.prepare()
    cuda = str(device).startswith("cuda")
    case.assertEqual(captured, sorted(buckets) if cuda else [])
    for frontier, count in ((0, 5), (5, 16), (21, 40), (61, 100), (161, 512), (700, 1)):
        tokens = [(frontier * 7 + 3 * i) % (vocab - 1) for i in range(count)]
        unpadded = StepBatch.build([("chunk", frontier, count)], device)
        unpadded.tokens_t = torch.tensor(tokens, dtype=torch.int64, device=device)
        expected = forward(unpadded)
        bucket = next(b for b in sorted(buckets) if b >= count)
        found = prompts.run_eager("chunk", frontier, tokens, bucket)[:count]
        case.assertTrue(torch.equal(found, expected), f"a padded chunk of {count} in {bucket}: its real rows")
        prompts.batch.fill(bucket, "chunk", frontier, tokens)
        prompts.batch.tokens_t[count:] = vocab - 1
        junk = forward(prompts.batch)[:count]
        case.assertTrue(torch.equal(junk, expected), f"NaN padding of a chunk of {count} in {bucket} reached a real row")
        if cuda:
            replayed = prompts.run("chunk", frontier, tokens)
            case.assertIsNotNone(replayed, f"a chunk of {count} replays a captured bucket")
            case.assertTrue(torch.equal(replayed[:count], expected), f"a replayed chunk of {count}: its real rows")


def _same(case, found, expected, what):
    torch = _torch()
    for i, (a, b) in enumerate(zip(found, expected)):
        if not torch.equal(a, b):
            case.fail(f"{what}: output {i} differs from the separate calls' bits")


def fused_forms(case, norms, device, shapes=SHAPES, counts=(1, 4, 9, 64)):
    """norms:fused: rms_norm_rope_pair, rope_pair and stream_add_norm give every output the bits the plugin's separate
    calls give (rms_norm_rope or rope of q and of k; stream_add, then rms_norm), at every row count (so a row has its
    bits alone and beside others, as the separate calls do), on views of a fused projection's output, with a weight per
    head or stream or none, FP32 and 16-bit streams, into out= tensors and in place; each refuses what the separate
    calls refuse; on CUDA a captured graph replays them to the eager bits."""
    torch = _torch()
    bf16, f32 = torch.bfloat16, torch.float32
    n = max(counts)
    for width, heads, dim, rotary in shapes:
        g = Inputs(device, seed=width + 1)
        kv = max(1, heads // 6)
        qkv = g.normal(n, heads * 2 * dim + 2 * kv * dim, scale=2.0, dtype=bf16)
        q = qkv[:, :heads * 2 * dim].view(n, heads, 2 * dim)[..., :dim]      # each head's query beside its gate
        k = qkv[:, heads * 2 * dim:heads * 2 * dim + kv * dim].view(n, kv, dim)
        positions = g.positions(n)
        freqs = inv_freq(rotary, 1e7, device)
        table = cos_sin_table(freqs, 4096)
        for wq, wk in ((g.normal(dim, scale=0.2), g.normal(dim, scale=0.2)),
                       (g.normal(heads, dim, scale=0.2) + 1, g.normal(kv, dim, scale=0.2) + 1), (None, None)):
            for f, style, out_dtype in ((freqs, "half", bf16), (table, "interleaved", f32), (freqs, "half", None)):
                for c in counts:
                    shape = None if wq is None else tuple(wq.shape)
                    where = f"{heads}/{kv} heads of {dim}, {c} tokens, {style}, weight {shape}"
                    _same(case, norms.rms_norm_rope_pair(q[:c], k[:c], wq, wk, 1e-6, positions[:c], f, 1.0, rotary,
                                                         style, 0.5, out_dtype),
                          (norms.rms_norm_rope(q[:c], wq, 1e-6, positions[:c], f, 1.0, rotary, style, 0.5, out_dtype),
                           norms.rms_norm_rope(k[:c], wk, 1e-6, positions[:c], f, 1.0, rotary, style, 0.5, out_dtype)),
                          "rms_norm_rope_pair " + where)
                    if wq is None:
                        _same(case, norms.rope_pair(q[:c], k[:c], positions[:c], f, rotary, style, 0.5, out_dtype),
                              (norms.rope(q[:c], positions[:c], f, rotary, style, 0.5, out_dtype),
                               norms.rope(k[:c], positions[:c], f, rotary, style, 0.5, out_dtype)),
                              "rope_pair " + where)
        qc, kc = q.contiguous(), k.contiguous()
        expected = norms.rms_norm_rope_pair(qc, kc, wq, wk, 1e-6, positions, freqs)
        found = norms.rms_norm_rope_pair(qc, kc, wq, wk, 1e-6, positions, freqs, q_out=qc, k_out=kc)
        case.assertTrue(found[0] is qc and found[1] is kc, "the pair returns its out= tensors")
        _same(case, (qc, kc), expected, "rms_norm_rope_pair in place")
        streams = g.normal(n, 4, width, scale=2.0)
        row = g.normal(n, width, scale=2.0, dtype=bf16)
        per_stream = g.normal(n, 4)
        for sdtype, weights, weight, out_dtype in ((f32, per_stream, g.normal(4, width, scale=0.2), bf16),
                                                   (bf16, None, (g.normal(width, scale=0.2) + 1).to(bf16), None),
                                                   (f32, per_stream, None, f32)):
            for c in counts:
                base = streams[:c].to(sdtype)
                a, b = base.clone(), base.clone()
                ws = None if weights is None else weights[:c]
                norms.stream_add(a, row[:c], ws)
                y = norms.rms_norm(a, weight, 1e-6, 1.0, out_dtype=out_dtype)
                found = norms.stream_add_norm(b, row[:c], ws, weight, 1e-6, 1.0, out_dtype)
                case.assertIs(found[1], b, "stream_add_norm advances the streams in place and returns them")
                _same(case, found, (y, a), f"stream_add_norm {sdtype} streams of {width}, {c} tokens")
        wide = g.normal(n, 4, width + 16)                                     # the streams a view of a wider buffer
        view = wide[..., :width]
        kept = view.clone()
        norms.stream_add(kept, row, per_stream)
        y, _ = norms.stream_add_norm(view, row, per_stream, None, 1e-6)
        _same(case, (y, view), (norms.rms_norm(kept, None, 1e-6), kept), "stream_add_norm of a strided view")
    g = Inputs(device, seed=7)
    q, k = g.normal(3, 4, 64), g.normal(3, 2, 64)
    positions, freqs = g.positions(3), inv_freq(64, 1e4, device)
    for bad in (lambda: norms.rms_norm_rope_pair(q, k[:2], None, None, 1e-6, positions, freqs),
                lambda: norms.rms_norm_rope_pair(q, k, g.normal(64), None, 1e-6, positions, freqs),
                lambda: norms.rope_pair(q, k[..., :32], positions, freqs),
                lambda: norms.stream_add_norm(q, k[:2, 0], None, None, 1e-6),
                lambda: norms.stream_add_norm(q, q[:, 0], g.normal(3, 3), None, 1e-6)):
        with case.assertRaises(ValueError):
            bad()
    if str(device).startswith("cuda"):
        _fused_captured(case, norms, device)


def _fused_captured(case, norms, device):
    torch = _torch()
    g = Inputs(device, seed=19)
    count = 5
    qkv = g.normal(count, 24 * 512 + 2048, dtype=torch.bfloat16)
    q, k = qkv[:, :12288].view(count, 24, 512)[..., :256], qkv[:, 12288:13312].view(count, 4, 256)
    wq, wk = g.normal(256, scale=0.2), g.normal(256, scale=0.2)
    positions = g.positions(count)
    freqs = inv_freq(64, 1e7, device)
    streams, row, weights = g.normal(count, 4, 2560), g.normal(count, 2560, dtype=torch.bfloat16), g.normal(count, 4)
    sw = g.normal(4, 2560, scale=0.2)

    def run():
        y, kept = norms.stream_add_norm(streams, row, weights, sw, 1e-6, 1.0, torch.bfloat16)
        return [*norms.rms_norm_rope_pair(q, k, wq, wk, 1e-6, positions, freqs, 1.0, 64),
                *norms.rope_pair(q, k, positions, freqs, 64, "interleaved"), y, kept.clone()]

    start = streams.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            streams.copy_(start)
            run()
    torch.cuda.current_stream().wait_stream(stream)
    streams.copy_(start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    fresh = Inputs(device, seed=20)
    for t in (qkv, streams, row, weights):
        t.copy_(fresh.normal(*t.shape, dtype=t.dtype))
    positions.copy_(fresh.positions(count))
    start = streams.clone()
    graph.replay()
    replayed = [t.clone() for t in out]
    streams.copy_(start)
    _same(case, replayed, run(), "a captured graph of the fused forms")


def run_all(case, norms, device):
    against_rows(case, norms, device)
    rows_alone(case, norms, device)
    odd_layouts(case, norms, device)
    strided(case, norms, device)
    outputs(case, norms, device)
    if str(device).startswith("cuda"):
        captured(case, norms, device)


def compare(case, norms, reference, device, atol=1e-5, rtol=1e-5):
    """Two norms plugins' device forms agree on the same inputs within one rounding of a 16-bit output (most of the
    calls round to BF16 or FP16 there: two FP32 results a hair apart may round to neighbouring values, 2^-7 apart
    relative at most); against_rows holds each to FP32 rounding with FP32 out."""
    torch = _torch()
    step = 2.0 ** -7
    for width, heads, dim, rotary in SHAPES:
        a = _calls(norms, Inputs(device, seed=width), width, heads, dim, rotary, torch.bfloat16)
        b = _calls(reference, Inputs(device, seed=width), width, heads, dim, rotary, torch.bfloat16)
        for name in a:
            found, expected = a[name](8).float(), b[name](8).float()
            worst = float(((found - expected).abs() - (atol + (rtol + step) * expected.abs())).max())
            if worst > 0:
                case.fail(f"{name} at width {width}: the plugins differ beyond one rounding of their output")


__all__ = ["TOLERANCE", "against_rows", "captured", "compare", "cos_sin_table", "fused_forms", "inv_freq",
           "odd_layouts", "outputs", "prompt_chunks", "rows_alone", "run_all", "strided"]
