"""Toy KV: rows in Python lists as the host form, and a pool of torch tensors per sequence and position as the device
form (crit_plugins.device.KV_POOL), on whatever device torch is given."""


class Rows:
    def __init__(self):
        self.committed, self.pending = {}, {}

    def open(self, sequence, key_width, value_width, capacity):
        self.committed[sequence], self.pending[sequence] = [], []

    def stage(self, sequence, keys, values):
        self.pending[sequence] += [(list(k), list(v)) for k, v in zip(keys, values)]

    def commit(self, sequence, count=None):
        kept = self.pending[sequence] if count is None else self.pending[sequence][:count]
        self.committed[sequence] += kept
        self.pending[sequence] = []

    def rollback(self, sequence):
        self.pending[sequence] = []

    def rows(self, sequence, pending=False):
        rows = self.committed[sequence] + (self.pending[sequence] if pending else [])
        return [(p, k, v) for p, (k, v) in enumerate(rows)]

    def fork(self, sequence, new):
        self.committed[new], self.pending[new] = list(self.committed[sequence]), []

    def release(self, sequence):
        self.committed.pop(sequence, None)
        self.pending.pop(sequence, None)

    def pool(self, layers, kv_heads, head_dim, tokens, device, dtype="fp32", value_dim=None):
        """The device form: every sequence's KV as one [kv_heads][dim] tensor per layer and position (FP32)."""
        return Pool(layers, kv_heads, head_dim, device, value_dim)


class Pool:
    def __init__(self, layers, kv_heads, head_dim, device, value_dim=None):
        import torch
        self.torch, self.device = torch, torch.device(device)
        self.layers, self.kv_heads, self.head_dim = list(layers), int(kv_heads), int(head_dim)
        self.value_dim = int(value_dim or head_dim)
        self.lengths, self.kv, self.slots = {}, {}, []   # kv[sequence][layer] = {position: (key, value)}

    def open(self, sequence):
        if sequence in self.lengths:
            raise ValueError(f"sequence {sequence!r} is open already")
        self.lengths[sequence], self.kv[sequence] = 0, {layer: {} for layer in self.layers}

    def close(self, sequence):
        self.lengths.pop(sequence, None)
        self.kv.pop(sequence, None)

    def fork(self, sequence, new):
        if new in self.lengths:
            raise ValueError(f"sequence {new!r} is open already")
        length = self.lengths[sequence]
        self.lengths[new] = length
        self.kv[new] = {layer: {p: row for p, row in rows.items() if p < length}
                        for layer, rows in self.kv[sequence].items()}

    def length(self, sequence):
        return self.lengths[sequence]

    def reserve(self, batch):
        """Slot i of a step is (sequence, position) of its token i, until the next reserve."""
        self.slots = []
        for r in range(batch.rows):
            sequence, _, count, position = batch.row(r)
            if sequence not in self.lengths:
                raise ValueError(f"sequence {sequence!r} is not open")
            self.slots += [(sequence, position + i) for i in range(count)]
        return self.torch.arange(len(self.slots), dtype=self.torch.int64, device=self.device)

    def write(self, layer, slots, keys, values):
        for i, slot in enumerate(slots.tolist()):
            sequence, position = self.slots[slot]
            self.kv[sequence][layer][position] = (keys[i].float().clone(), values[i].float().clone())

    def keep(self, sequence, length):
        self.lengths[sequence] = int(length)
        for rows in self.kv[sequence].values():
            for position in [p for p in rows if p >= length]:
                del rows[position]

    def gather(self, layer, sequence, count):
        rows = self.kv[sequence][layer]
        if any(p not in rows for p in range(count)):
            raise ValueError(f"sequence {sequence!r} holds no KV at some of its first {count} positions")
        torch = self.torch
        if not count:
            return (torch.zeros(0, self.kv_heads, self.head_dim, device=self.device),
                    torch.zeros(0, self.kv_heads, self.value_dim, device=self.device))
        return (torch.stack([rows[p][0] for p in range(count)]), torch.stack([rows[p][1] for p in range(count)]))


def create(context, options, parts):
    return Rows()
