"""The cases the recurrence slot's plugins are compared with their reference on (crit_plugins.testing: assertMatchesReference), which this base plugin's manifest names under defines.slots.recurrence.cases."""
import random



def recurrence(parts):
    recurrence, rng = parts["recurrence"], random.Random(19)
    geometry = {"keys": 3, "values": 2, "epsilon": 1e-6, "query_scale": 0.5, "normalize_keys": True,
                "normalize_queries": True}
    steps = [{"query": [rng.uniform(-1, 1) for _ in range(3)], "key": [rng.uniform(-1, 1) for _ in range(3)],
              "value": [rng.uniform(-1, 1) for _ in range(2)], "log_decay": rng.uniform(-0.4, -0.01),
              "beta": rng.uniform(0.1, 0.9)} for _ in range(9)]
    state = tuple(0.0 for _ in range(6))
    outputs, state = recurrence.prefill(state, steps[:7], geometry)
    more, state = recurrence.prefill(state, steps[7:], geometry)
    return {"outputs": [list(o) for o in outputs + more], "state": list(state)}
