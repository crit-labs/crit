"""The cases the attention slot's plugins are compared with their reference on (crit_plugins.testing: assertMatchesReference), which this base plugin's manifest names under defines.slots.attention.cases."""
import random

from crit_plugins.slots import rows as _rows


def attention(parts):
    kv, attention, rng, out = parts["kv_cache"], parts["attention"], random.Random(5), {}
    kv.open(11, 8, 5, 64)
    kv.stage(11, _rows(rng, 13, 8, 1.0), _rows(rng, 13, 5))
    kv.commit(11)
    queries = _rows(rng, 4, 8, 1.0)
    out["decode"] = attention.attend(11, queries[:1], [12])
    out["prefill"] = attention.attend(11, queries, [0, 5, 9, 12])
    out["scaled"] = attention.attend(11, queries[1:3], [7, 12], scale=0.5)
    kv.stage(11, _rows(rng, 3, 8, 1.0), _rows(rng, 3, 5))
    out["pending"] = attention.attend(11, queries[:3], [13, 14, 15], pending=True)
    kv.rollback(11)
    kv.release(11)
    return out
