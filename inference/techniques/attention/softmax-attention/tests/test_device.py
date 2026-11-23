"""softmax-attention's device form equals its host form over the same rows, and a batch equals its rows alone."""
from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}


class DeviceAttentionTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        self.parts = self.assemble(USE)
        self.kv, self.attention = self.parts["kv_cache"], self.parts["attention"]
        self.pool = self.kv.pool([0], 2, 8, 512, self.device(), "fp32", page=16)
        g = self.torch.Generator().manual_seed(11)
        self.keys = {s: self.torch.randn(40, 2, 8, generator=g) for s in "abc"}
        self.values = {s: self.torch.randn(40, 2, 8, generator=g) for s in "abc"}
        for s in "abc":
            self.pool.open(s)
            self.kv.open(s, 8, 8, 64)

    def write(self, rows):
        """Write rows of (sequence, first position, count) into the pool and the host form; returns the batch."""
        batch = StepBatch.build(rows, self.device())
        slots = self.pool.reserve(batch)
        keys = self.torch.cat([self.keys[s][p:p + n] for s, p, n in rows]).to(self.device())
        values = self.torch.cat([self.values[s][p:p + n] for s, p, n in rows]).to(self.device())
        self.pool.write(0, slots, keys, values)
        for s, p, n in rows:
            # The host form keeps one head per sequence here: each kv head is a sequence of its own.
            for head in range(2):
                name = f"{s}{head}"
                if name not in self.kv.states:
                    self.kv.open(name, 8, 8, 64)
                self.kv.stage(name, self.keys[s][p:p + n, head].tolist(), self.values[s][p:p + n, head].tolist())
                self.kv.commit(name)
        return batch

    def test_the_device_form_equals_the_host_form_with_grouped_heads(self):
        batch = self.write([("a", 0, 13), ("b", 0, 1), ("c", 0, 7)])
        queries = self.torch.randn(batch.tokens, 4, 8, generator=self.torch.Generator().manual_seed(3))
        out = self.attention.forward(0, queries.to(self.device()), self.pool, batch).cpu()
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            for t in range(count):
                for head in range(4):   # query heads 0, 1 read kv head 0; 2, 3 read kv head 1
                    host = self.attention.attend(f"{sequence}{head // 2}", [queries[first + t, head].tolist()],
                                                 [position + t])[0]
                    self.assertClose(out[first + t, head].tolist(), host, atol=1e-5, rtol=1e-4)

    def test_decode_rows_after_a_prefill_and_a_batch_equal_their_rows_alone(self):
        self.write([("a", 0, 20), ("b", 0, 9)])
        batch = self.write([("a", 20, 1), ("b", 9, 3), ("c", 0, 5)])   # a decode, a verify window, a prompt
        queries = self.torch.randn(batch.tokens, 4, 8, generator=self.torch.Generator().manual_seed(4)).to(self.device())
        together = self.attention.forward(0, queries, self.pool, batch)
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            alone = StepBatch.build([(sequence, position, count)], self.device())
            single = self.attention.forward(0, queries[first:first + count], self.pool, alone)
            self.assertTrue(self.torch.equal(single, together[first:first + count]), sequence)
