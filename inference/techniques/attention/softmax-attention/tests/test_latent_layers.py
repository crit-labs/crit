"""attention:latent-layers: forward_latent_layers(layers, queries, pool, batch, rank) over several layers of one pool
gives each layer's query heads the bits forward(layer, ..., latent=rank) gives them alone, on every path: prompt rows
with prompt_reader decode (the layers' pieces in one launch of the latent kernel's fused reader, BF16 and fp8:row
pages), a step of decode and prompt rows (a call a layer) and the torch form (FP32 pages, any device)."""
from crit_plugins import StepBatch
from crit_plugins.device import DECODE, PREFILL
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
R, P, GROUP, LAYERS = 256, 64, 4, (3, 4, 5)
SCALE = 192 ** -0.5


class LatentLayersTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        if self.torch is None:
            self.skipTest("torch")

    def parts(self, **options):
        found = self.assemble(USE, options=options)
        return found["kv_cache"], found["attention"]

    def fill(self, kv, dtype, rows):
        torch = self.torch
        pool = kv.pool(list(range(8)), 1, R + P, 2048, self.device(), dtype, page=16, value_dim=0)
        for sequence in {row[0] for row in rows}:
            pool.open(sequence)
        batch = StepBatch.build(rows, self.device())
        slots = pool.reserve(batch)
        for layer in LAYERS:
            g = torch.Generator().manual_seed(layer)
            pool.write(layer, slots, (torch.randn(batch.tokens, 1, R + P, generator=g) * 0.5).to(self.device()), None)
        q = (torch.randn(batch.tokens, len(LAYERS) * GROUP, R + P, generator=torch.Generator().manual_seed(9)) * 0.5)
        return pool, batch, q.to(self.device())

    def check(self, attention, pool, batch, q):
        torch = self.torch
        found = attention.forward_latent_layers(LAYERS, q, pool, batch, R, scale=SCALE)
        alone = torch.cat([attention.forward(layer, q[:, i * GROUP:(i + 1) * GROUP].contiguous(), pool, batch,
                                             scale=SCALE, latent=R) for i, layer in enumerate(LAYERS)], dim=1)
        self.assertEqual(tuple(found.shape), (batch.tokens, len(LAYERS) * GROUP, R))
        self.assertTrue(torch.equal(found, alone))

    def test_prompt_rows_through_the_fused_reader_take_one_launch_for_every_layer(self):
        if self.device() != "cuda":
            self.skipTest("the latent kernel's fused reader runs on a GPU")
        kv, attention = self.parts(prompt_reader="decode")
        for dtype in ("bf16", "fp8:row"):
            pool, batch, q = self.fill(kv, dtype, [("a", 0, 150, PREFILL), ("b", 0, 40, PREFILL)])
            launches = []
            kernels = attention.device_form._latent_module(pool, q)
            if kernels is None:
                self.skipTest("crit's latent-attention kernel does not build here")
            attend = kernels.attend
            kernels.attend = lambda *a, **k: launches.append(1) or attend(*a, **k)
            try:
                found = attention.forward_latent_layers(LAYERS, q, pool, batch, R, scale=SCALE)
            finally:
                kernels.attend = attend
            self.assertEqual(len(launches), 1, dtype)
            self.check(attention, pool, batch, q)
            self.assertTrue(self.torch.equal(found, attention.forward_latent_layers(LAYERS, q, pool, batch, R,
                                                                                     scale=SCALE)))

    def test_a_step_of_decode_and_prompt_rows_reads_a_layer_a_call(self):
        if self.device() != "cuda":
            self.skipTest("the latent kernel runs on a GPU")
        kv, attention = self.parts(prompt_reader="decode")
        pool, batch, q = self.fill(kv, "bf16", [("a", 0, 3, DECODE), ("b", 0, 70, PREFILL)])
        self.check(attention, pool, batch, q)

    def test_the_torch_form_reads_a_layer_a_call(self):
        kv, attention = self.parts(kernels="torch")
        pool, batch, q = self.fill(kv, "fp32", [("a", 0, 20, PREFILL), ("b", 0, 2, DECODE)])
        self.check(attention, pool, batch, q)

    def test_queries_that_are_not_a_whole_group_a_layer_are_refused(self):
        kv, attention = self.parts(kernels="torch")
        pool, batch, q = self.fill(kv, "fp32", [("a", 0, 4, PREFILL)])
        with self.assertRaisesRegex(ValueError, "a whole group each"):
            attention.forward_latent_layers(LAYERS, q[:, :GROUP * len(LAYERS) - 1], pool, batch, R)
