"""The cases the experts slot's plugins are compared with their reference on (crit_plugins.testing: assertMatchesReference), which this base plugin's manifest names under defines.slots.experts.cases."""
import random

from crit_plugins.slots import rows as _rows


def expert_weights(layer, expert, hidden=6, inter=4):
    """The synthetic weights the experts cases add to the expert_source slot."""
    rng = random.Random(1000 * layer + expert)
    return _rows(rng, 2 * inter, hidden, 0.5), _rows(rng, hidden, inter, 0.5)


def experts(parts):
    source, experts, rng = parts["expert_source"], parts["experts"], random.Random(17)
    for layer in (0, 1):
        for expert in range(6):
            source.add(layer, expert, *expert_weights(layer, expert))
    x = _rows(rng, 5, 6, 1.0)
    ids = [[0, 3], [3, 5], [1, -1], [2, 4], [5, 0]]
    weights = [[0.7, 0.3], [0.5, 0.5], [1.0, 0.0], [0.6, 0.4], [0.9, 0.1]]
    out = {"layer0": experts.run(0, x, ids, weights), "layer1": experts.run(1, x[:3], ids[:3], weights[:3])}
    out["again"] = experts.run(0, x[2:], ids[2:], weights[2:])
    return out
