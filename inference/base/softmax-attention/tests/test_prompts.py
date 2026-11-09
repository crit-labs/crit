"""Captured prompt chunks over attention-state's pool and softmax-attention (prompts:kv_cache, prompts:attention), at a
27B attention shape (24 query / 4 KV heads x 256, fp8:row and BF16 pages): a chunk replayed from its padded bucket's
graph gives exactly what the same padded chunk gives eagerly, at any frontier; the KV it leaves equals the unpadded
eager path's (the padding writes only the dump page); its output equals the unpadded eager path's, bit for bit where
prompt_reader decode (every row in the decode's reduction), else within the prompt readers' rounding;
and the selection's prompt facts hold, so the graphs are captured."""
from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
HQ, HKV, D = 24, 4, 256
LAYERS = (0, 1)
# Each sequence's chunks: (frontier, count); every chunk continues its sequence's valid KV.
CHUNKS = [[(0, 5), (5, 40), (45, 2)], [(0, 64)], [(0, 100), (100, 16), (116, 9)]]
BUCKETS = (16, 64, 128)


class Model:
    """A model plugin's prompt arithmetic over the pool and the attention plugin: per layer each token writes its
    position's keys and values and attends; the output is the last real token's row. The same forward serves a host
    StepBatch (the eager path) and a PromptBatch (the padded bucket, captured or not)."""

    def __init__(self, test, dtype, reader="prefill"):
        torch = test.torch
        parts = test.assemble(USE, options={"prompt_reader": reader}, use_options={"attention-state": {"page": 64}})
        self.parts, self.attention, self.torch, self.calls = parts, parts["attention"], torch, 0
        self.pool = parts["kv_cache"].pool(list(LAYERS), HKV, D, 4096, "cuda", dtype)
        for sequence in range(len(CHUNKS)):
            self.pool.open(sequence)
        g = torch.Generator().manual_seed(7)
        # Each position's query, key and value rows: a function of the position, so every path writes the same KV.
        self.inputs = torch.randn(1024, 3, HQ, D, generator=g).cuda().to(torch.bfloat16)

    def forward(self, batch):
        self.calls += 1
        torch = self.torch
        x = self.inputs.index_select(0, batch.positions_t)
        slots = self.pool.reserve(batch)
        q = x[:, 0]
        for layer in LAYERS:
            self.pool.write(layer, slots, x[:, 1, :HKV] * (layer + 1), x[:, 2, :HKV])
            q = (self.attention.forward(layer, q, self.pool, batch).float() + x[:, 0].float()).to(torch.bfloat16)
        last = batch.last_t if batch.prompt else torch.tensor([batch.tokens - 1], device="cuda")
        return q.index_select(0, last)

    def eager(self, sequence, frontier, count):
        """The model's own eager prompt path: an unpadded StepBatch of the chunk."""
        out = self.forward(StepBatch.build([(sequence, frontier, count)], "cuda"))
        self.pool.keep(sequence, frontier + count)
        return out


class PromptChunkTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("captured prompt chunks run on CUDA")
        self.torch = __import__("torch")

    def test_it_provides_the_prompt_facts(self):
        from crit_plugins.prompt_graphs import prompt_problems
        assembly, registry = self.assemble(USE), self.registry()

        class Parts:   # a factory's parts as prompt_problems reads them
            def slots(self):
                return list(assembly.use)

            def plugin(self, slot):
                return assembly.use.get(slot)

            def facts(self, slot):
                return tuple(registry[assembly.use[slot]]["provides"])

            def get(self, slot, default=None):
                return assembly.get(slot, default)
        self.assertEqual(prompt_problems(Parts(), uses=["kv_cache", "attention"]), [])
        self.assertImplements(assembly)

    def test_captured_chunks_give_the_padded_eager_chunks_output_and_the_eager_paths_kv(self):
        from crit_plugins.prompt_graphs import PromptGraphs
        torch = self.torch
        for dtype, reader in (("fp8:row", "prefill"), ("bf16", "prefill"), ("fp8:row", "decode"), ("int8", "decode"),
                              ("bf16", "decode")):
            graph, padded, plain = Model(self, dtype, reader), Model(self, dtype, reader), Model(self, dtype, reader)
            prompts = PromptGraphs(graph.forward, device="cuda", buckets=BUCKETS, pools=[graph.pool])
            # prompt_reader decode: every row in the decode's reduction, so the captured bucket and the unpadded eager
            # chunk (a window row up to the pool's window, a prompt row past it) agree bit for bit.
            exact = reader == "decode"
            eager = PromptGraphs(padded.forward, device="cuda", buckets=BUCKETS, pools=[padded.pool])
            self.assertEqual(prompts.prepare(), list(BUCKETS))
            calls = graph.calls
            for sequence, chunks in enumerate(CHUNKS):
                for frontier, count in chunks:
                    tokens = list(range(count))
                    bucket = min(b for b in BUCKETS if b >= count)
                    found = prompts.run(sequence, frontier, tokens).clone()
                    want = eager.run_eager(sequence, frontier, tokens, bucket)
                    self.assertTrue(torch.equal(found, want), (dtype, reader, sequence, frontier, count))
                    near = plain.eager(sequence, frontier, count)
                    if exact:
                        self.assertTrue(torch.equal(found, near), (dtype, reader, sequence, frontier, count))
                    else:
                        torch.testing.assert_close(found.float(), near.float(), atol=3e-2, rtol=3e-2,
                                                   msg=(dtype, sequence, frontier, count))
            self.assertEqual(graph.calls, calls, "every chunk replayed its bucket's graph")
            for sequence, chunks in enumerate(CHUNKS):
                total = sum(count for _, count in chunks)
                self.assertEqual((graph.pool.length(sequence), padded.pool.length(sequence)), (total, total))
                for layer in LAYERS:
                    for got, a, b in zip(graph.pool.gather(layer, sequence, total),
                                         padded.pool.gather(layer, sequence, total),
                                         plain.pool.gather(layer, sequence, total)):
                        self.assertTrue(torch.equal(got, a) and torch.equal(got, b), (dtype, sequence, layer))
