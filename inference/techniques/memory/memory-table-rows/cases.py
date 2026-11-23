"""The cases the embeddings slot's plugins are compared with their reference on (crit_plugins.testing: assertMatchesReference), which this base plugin's manifest names under defines.slots.embeddings.cases."""
import random

from crit_plugins.slots import rows as _rows


def embeddings(parts):
    embeddings, rng = parts["embeddings"], random.Random(23)
    embeddings.load(_rows(rng, 40, 6, 0.5))
    return {"decode": embeddings.rows([7]), "prefill": embeddings.rows([0, 39, 7, 7, 12, 3]),
            "again": embeddings.rows([12, 7, 25])}
