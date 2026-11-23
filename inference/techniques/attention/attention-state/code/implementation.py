"""Transactional causal KV/state reference; physical layout remains a backend choice."""
import math


class KVState:
    def __init__(self, capacity, key_width, value_width):
        if any(type(x) is not int or x <= 0 for x in (capacity, key_width, value_width)):
            raise ValueError("positive integer state geometry required")
        self.capacity, self.key_width, self.value_width = capacity, key_width, value_width
        self.committed = []
        self.pending = []

    def stage(self, key, value):
        key, value = tuple(key), tuple(value)
        if len(key) != self.key_width or len(value) != self.value_width:
            raise ValueError("KV geometry differs")
        if any(not math.isfinite(x) for x in key + value):
            raise ValueError("finite KV values required")
        if len(self.committed) + len(self.pending) >= self.capacity:
            raise ValueError("KV capacity exhausted")
        self.pending.append((key, value))

    def commit(self, count=None):
        count = len(self.pending) if count is None else count
        if type(count) is not int or not 0 <= count <= len(self.pending):
            raise ValueError("commit exceeds staged state")
        self.committed.extend(self.pending[:count])
        self.pending.clear()  # rejected speculative suffix is discarded

    def rollback(self):
        self.pending.clear()

    def attend(self, query, *, include_pending=False, scale=None, backend=None):
        query = tuple(query)
        if len(query) != self.key_width or any(not math.isfinite(x) for x in query):
            raise ValueError("query geometry/domain differs")
        scale = 1 / math.sqrt(self.key_width) if scale is None else scale
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("positive finite attention scale required")
        rows = tuple(self.committed + (self.pending if include_pending else []))
        if not rows:
            raise ValueError("attention requires an admitted causal prefix")
        if backend is not None:
            return backend(query, rows, scale)
        logits = [sum(q * k for q, k in zip(query, key)) * scale for key, _ in rows]
        top = max(logits)
        probabilities = [math.exp(x - top) for x in logits]
        total = sum(probabilities)
        return tuple(sum(p * value[column] for p, (_, value) in zip(probabilities, rows)) / total
                     for column in range(self.value_width))
