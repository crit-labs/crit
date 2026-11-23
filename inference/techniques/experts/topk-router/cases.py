"""The cases the router slot's plugins are compared with their reference on (crit_plugins.testing: assertMatchesReference), which this base plugin's manifest names under defines.slots.router.cases."""
import random

from crit_plugins.slots import rows as _rows


def router(parts):
    router, rng = parts["router"], random.Random(13)
    logits = _rows(rng, 5, 8)
    out = {}
    for top in (1, 2, 3):
        out[f"top{top}"] = [[list(ids), list(weights)] for ids, weights in router.route(logits, top)]
    bias = [round(rng.uniform(-0.5, 0.5), 3) for _ in range(8)]
    out["bias"] = [[list(ids), list(weights)] for ids, weights in router.route(logits, 2, bias)]
    return out
