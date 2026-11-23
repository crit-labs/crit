"""attention:latent: forward(..., latent=R) over a pool of keys only with one KV head (each position's latent, then its
rotary key) against a float64 reference over the stored rows: decode tokens, verify windows and prompt rows of a
step on crit's latent-attention kernel (BF16 and fp8:row pages) and in torch (FP32 pages); a captured round's column
equal to the step's decode of the same token; a captured prompt chunk equal to the padded eager chunk; with
prompt_reader decode, a prompt row's tokens with the bits of the same tokens decoded."""
from types import SimpleNamespace

from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
R, P, H = 512, 64, 16
SCALE = 192 ** -0.5


class LatentTests(PluginTestCase):
    def setUp(self):
        if not self.device() or self.device() == "cpu":
            self.skipTest("the device form needs a GPU")
        self.torch = __import__("torch")
        parts = self.assemble(USE)
        self.kv, self.attention = parts["kv_cache"], parts["attention"]

    def pool(self, dtype, tokens=4096):
        return self.kv.pool([0], 1, R + P, tokens, self.device(), dtype, page=16, value_dim=0)

    def rows(self, positions, seed=1):
        """Each position's latent and rotary key: a function of the position alone."""
        torch = self.torch
        return torch.stack([torch.randn(1, R + P, generator=torch.Generator().manual_seed(seed * 7919 + int(p)))
                            for p in positions]).to(self.device()) * 0.5

    def queries(self, n, seed):
        torch = self.torch
        return (torch.randn(n, H, R + P, generator=torch.Generator().manual_seed(seed)) * 0.5).to(self.device())

    def write(self, pool, batch):
        if not hasattr(self, "_rows"):   # every position's row on the device once: a capture copies nothing
            self._rows = self.rows(range(1024))
        pool.write(0, pool.reserve(batch), self._rows.index_select(0, batch.positions_t), None)

    def reference(self, pool, batch, q, sinks=None):
        torch = self.torch
        out = []
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            c = pool.gather(0, sequence, position + count)[0][:, 0].double()
            qq = q[first:first + count].double()
            s = torch.einsum("thd,nd->thn", qq, c) * SCALE
            n = torch.arange(position + count, device=q.device)
            s = s.masked_fill(~(n[None, :] <= position + torch.arange(count, device=q.device)[:, None])[:, None, :],
                              float("-inf"))
            if sinks is not None:
                s = torch.cat([s, sinks.double().view(1, -1, 1).expand(count, -1, 1)], dim=-1)
            p = torch.softmax(s, dim=-1)[..., :position + count]
            out.append(torch.einsum("thn,nr->thr", p, c[:, :R]))
        return torch.cat(out)

    def test_a_steps_prompt_rows_in_one_launch_keep_each_rows_bits(self):
        """Several prompt rows of a step (the kernel's prefill_rows) give each token the bits of its row's step
        alone, beside a window row, on BF16 and fp8:row pages."""
        torch = self.torch
        for dtype in ("bf16", "fp8:row"):
            rows = [("a", 0, 100), ("b", 0, 130), ("c", 0, 3), ("d", 0, 65)]
            together = self.pool(dtype)
            for s in "abcd":
                together.open(s)
            batch = StepBatch.build(rows, self.device())
            self.write(together, batch)
            q = self.queries(batch.tokens, 41)
            out = self.attention.forward(0, q, together, batch, scale=SCALE, latent=R)
            first = 0
            for row in rows:
                alone = self.pool(dtype)
                alone.open(row[0])
                one = StepBatch.build([row], self.device())
                self.write(alone, one)
                got = self.attention.forward(0, q[first:first + row[2]], alone, one, scale=SCALE, latent=R)
                self.assertTrue(torch.equal(got, out[first:first + row[2]]), (dtype, row))
                first += row[2]

    def test_steps_match_float64_latent_attention_on_every_reader(self):
        torch = self.torch
        sinks = torch.randn(H, generator=torch.Generator().manual_seed(3)).to(self.device())
        for dtype in ("bf16", "fp8:row", "fp32"):
            pool = self.pool(dtype)
            for s in "ab":
                pool.open(s)
            for i, rows in enumerate(([("a", 0, 100), ("b", 0, 30)],      # a prompt row and a window row
                                      [("a", 100, 1), ("b", 30, 5)],      # a decode token and a verify window
                                      [("a", 101, 70), ("b", 35, 1)])):   # a prompt row past a cached prefix
                batch = StepBatch.build(rows, self.device())
                self.write(pool, batch)
                q = self.queries(batch.tokens, 10 + i)
                for s in (None, sinks):
                    out, lse = self.attention.forward(0, q, pool, batch, scale=SCALE, lse=True, latent=R, sinks=s)
                    self.assertEqual(tuple(out.shape), (batch.tokens, H, R))
                    want = self.reference(pool, batch, q.to(torch.bfloat16 if dtype != "fp32" else q.dtype), s)
                    self.assertLess(float((out.double() - want).abs().max()), 1.5e-2, (dtype, i))
                for sequence, _, count, position in (batch.row(r) for r in range(batch.rows)):
                    pool.keep(sequence, position + count)

    def test_a_captured_rounds_column_equals_the_steps_decode_of_the_token(self):
        torch = self.torch
        for dtype in ("bf16", "fp8:row"):
            pool = self.pool(dtype)
            pool.open("a")
            prompt = StepBatch.build([("a", 0, 300)], self.device())
            self.write(pool, prompt)
            pool.keep("a", 300)
            q = self.queries(4, 21)
            steps = []
            for c in range(4):   # each token decoded alone in a step (its KV the round's: a function of the position)
                batch = StepBatch.build([("a", 300 + c, 1)], self.device())
                self.write(pool, batch)
                steps.append(self.attention.forward(0, q[c:c + 1], pool, batch, scale=SCALE, latent=R))
                pool.keep("a", 301 + c)
            pool.keep("a", 300)
            pool.bind(0, "a", 400)
            batch = RoundBatch(1, 3, self.device())
            batch.fill(1, 3, torch.tensor([300], device=self.device()),
                       torch.zeros(1, 4, dtype=torch.int64, device=self.device()))
            self.write(pool, batch)
            out = self.attention.forward(0, q, pool, batch, scale=SCALE, latent=R)
            for c in range(4):
                self.assertTrue(torch.equal(out[c:c + 1], steps[c]), (dtype, c))
            pool.commit_round(SimpleNamespace(rows=1, frontier=torch.tensor([304], device=self.device()),
                                              n_emitted=torch.tensor([4], device=self.device())))
            self.assertEqual(pool.length("a"), 304)
            pool.unbind(0)

    def test_a_captured_prompt_chunk_equals_the_padded_eager_chunk(self):
        from crit_plugins.prompt_graphs import PromptGraphs
        torch = self.torch
        queries = self.queries(1024, 31)
        self._rows = self.rows(range(1024))
        readers = (self.attention, self.assemble(USE, options={"prompt_reader": "decode"})["attention"])
        for attention in readers:   # the prompt reader's chunks, and the decode reader's fused windows
            models = []
            for _ in range(2):
                pool = self.pool("bf16")
                pool.open(0)

                def forward(batch, pool=pool, attention=attention):
                    self.write(pool, batch)
                    q = queries.index_select(0, batch.positions_t)
                    out = attention.forward(0, q, pool, batch, scale=SCALE, latent=R)
                    last = batch.last_t if batch.prompt else torch.tensor([batch.tokens - 1], device=self.device())
                    return out.index_select(0, last)
                models.append((forward, pool))
            graphs = PromptGraphs(models[0][0], device="cuda", buckets=(128, 256), pools=[models[0][1]])
            eager = PromptGraphs(models[1][0], device="cuda", buckets=(128, 256), pools=[models[1][1]])
            graphs.prepare()
            for frontier, count in ((0, 100), (100, 200), (300, 60)):
                found = graphs.run(0, frontier, list(range(count))).clone()
                want = eager.run_eager(0, frontier, list(range(count)), 128 if count <= 128 else 256)
                self.assertTrue(torch.equal(found, want), (frontier, count, attention is readers[1]))

    def test_prompt_reader_decode_gives_prompt_rows_their_decode_bits(self):
        """prompt_reader decode: a step's prompt row reads through the latent kernel's fused launch as window rows of
        64 columns, in the decode's reduction, so every token's output is the bits of the same token decoded alone in
        a step; the default prompt reader's are another reduction."""
        torch = self.torch
        attention = self.assemble(USE, options={"prompt_reader": "decode"})["attention"]
        for dtype in ("bf16", "fp8:row"):
            pool = self.pool(dtype)
            pool.open("a")
            batch = StepBatch.build([("a", 0, 150)], self.device())
            self.write(pool, batch)
            q = self.queries(150, 41)
            out = attention.forward(0, q, pool, batch, scale=SCALE, latent=R)
            other = self.attention.forward(0, q, pool, batch, scale=SCALE, latent=R)
            self.assertFalse(torch.equal(out, other), dtype)
            self.assertLess(float((out.double() - other.double()).abs().max()), 2e-2, dtype)
            for t in (0, 1, 63, 64, 100, 149):   # the token decoded alone, in a sequence of the same rows before it
                pool.open(t)                      # (a position's row is a function of the position)
                if t:
                    self.write(pool, StepBatch.build([(t, 0, t)], self.device()))
                    pool.keep(t, t)
                one = StepBatch.build([(t, t, 1)], self.device())
                self.write(pool, one)
                alone = attention.forward(0, q[t:t + 1], pool, one, scale=SCALE, latent=R)
                self.assertTrue(torch.equal(alone[0], out[t]), (dtype, t))

    def test_what_latent_attention_cannot_read_is_refused(self):
        pool = self.kv.pool([0], 2, 64, 256, self.device(), "bf16", page=16)
        pool.open("a")
        batch = StepBatch.build([("a", 0, 1)], self.device())
        pool.reserve(batch)
        with self.assertRaisesRegex(ValueError, "a pool of keys only"):
            self.attention.forward(0, self.queries(1, 1)[:, :, :64], pool, batch, latent=32)
