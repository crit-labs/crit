"""Toy softmax attention: over the toy KV rows (host form), and over any kv_cache pool's gathered KV in torch (device
form), each token attending its sequence's positions up to its own."""
import math


class Attention:
    def __init__(self, kv):
        self.kv = kv

    def attend(self, sequence, queries, positions, scale=None, pending=False):
        rows = self.kv.rows(sequence, pending)
        out = []
        for query, position in zip(queries, positions):
            seen = [(key, value) for p, key, value in rows if p <= position]
            s = scale if scale is not None else 1 / math.sqrt(len(query))
            weights = [math.exp(s * sum(q * k for q, k in zip(query, key))) for key, _ in seen]
            total = sum(weights)
            out.append([sum(w * value[j] for w, (_, value) in zip(weights, seen)) / total
                        for j in range(len(seen[0][1]))])
        return out

    def forward(self, layer, queries, pool, batch, scale=None):
        """queries [tokens][heads][dim] -> [tokens][heads][value_dim]; heads share the pool's kv heads in groups."""
        import torch
        out = queries.new_empty(queries.shape[0], queries.shape[1], pool.value_dim, dtype=torch.float32)
        s = scale if scale is not None else queries.shape[-1] ** -0.5
        group = queries.shape[1] // pool.kv_heads
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            keys, values = pool.gather(layer, sequence, position + count)
            keys = keys.float().repeat_interleave(group, dim=1)
            values = values.float().repeat_interleave(group, dim=1)
            for t in range(count):
                seen = position + t + 1
                weights = torch.softmax(torch.einsum("hd,phd->hp", queries[first + t].float(), keys[:seen]) * s, -1)
                out[first + t] = torch.einsum("hp,phd->hd", weights, values[:seen])
        return out


def create(context, options, parts):
    return Attention(parts.get("kv_cache"))
