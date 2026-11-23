"""The cases the norms slot's plugins are compared with their reference on (crit_plugins.testing: assertMatchesReference), which this base plugin's manifest names under defines.slots.norms.cases."""
import math
import random

from crit_plugins.slots import rows as _rows


def norms(parts):
    norms, rng, out = parts["norms"], random.Random(29), {}
    x, residual = _rows(rng, 4, 12), _rows(rng, 4, 12)
    weight = [round(rng.uniform(-0.5, 1.5), 3) for _ in range(12)]
    out["rms"] = norms.rms_norm_rows(x, weight, 1e-6)
    out["rms_plain"] = norms.rms_norm_rows(x[:1], None, 1e-5)
    out["rms_offset"] = norms.rms_norm_rows(x, [w - 1 for w in weight], 1e-6, offset=1.0)
    out["rms_residual"] = list(norms.rms_norm_rows(x, weight, 1e-6, residual=residual))
    out["rms_zero"] = norms.rms_norm_rows([[0.0] * 12], weight, 1e-6)
    gates = _rows(rng, 4, 12)
    out["gated_silu"] = norms.gated_rms_norm_rows(x, gates, weight, 1e-6)
    out["gated_sigmoid"] = norms.gated_rms_norm_rows(x, gates, weight, 1e-6, offset=0.5, activation="sigmoid")
    inv_freq = [1.0 / 10000 ** (2 * j / 8) for j in range(4)]
    positions = [0, 1, 17, 4099]
    out["rope_half"] = norms.rope_rows(x, positions, inv_freq, 8)
    out["rope_interleaved"] = norms.rope_rows(x, positions, inv_freq, style="interleaved", scale=1.25)
    table = [[math.cos(p * f) for f in inv_freq] + [math.sin(p * f) for f in inv_freq] for p in range(20)]
    out["rope_table"] = norms.rope_rows(x[:3], [0, 5, 19], table, 8)
    out["rope_full"] = norms.rope_rows(x, positions, [1.0 / 500 ** (2 * j / 12) for j in range(6)])
    for activation in ("silu", "gelu", "gelu_tanh", "sigmoid"):
        out[f"mul_{activation}"] = norms.act_mul_rows(gates, x, activation)
    out["l2"] = norms.l2_norm_rows(x, 1e-6)
    out["rms_grouped"] = norms.rms_norm_rows(x, [weight, [w * 0.5 for w in weight]], 1e-6, offset=1.0)
    streams = [_rows(rng, 3, 5) for _ in range(2)]                    # 2 tokens of 3 streams of 5
    out["stream_mean"] = norms.stream_sum_rows(streams, scale=1 / 3)
    out["stream_gated"] = norms.stream_sum_rows(streams, [_rows(rng, 3, 5) for _ in range(2)], "sigmoid")
    out["stream_weighted"] = norms.stream_sum_rows(streams, [[0.5, -1.0, 2.0], [1.0, 0.0, 0.25]])
    out["stream_add"] = norms.stream_add_rows(streams, _rows(rng, 2, 5), [[1.0, 0.5, -2.0], [0.0, 1.0, 3.0]])
    out["stream_add_plain"] = norms.stream_add_rows(streams, _rows(rng, 2, 5))
    bias, rate = [round(rng.uniform(-2, 2), 3) for _ in range(12)], [round(rng.uniform(0.01, 8), 3) for _ in range(12)]
    out["softplus_decay"] = norms.softplus_decay_rows([[v * 8 for v in row] for row in x], bias, rate)
    out["softplus_decay_gated"] = list(norms.softplus_decay_rows(x, bias, rate, gates))
    return out
