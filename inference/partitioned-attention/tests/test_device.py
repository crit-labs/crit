"""partitioned-attention's device form: its split-KV partitions equal softmax-attention's device form, its log-sum-exps
are each row's normalizer, and merge() of a sequence's shards equals attention over the whole sequence, as the host
form's merge_partials does."""
import math

from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}


class PartitionedDeviceTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch")
        self.dev = self.device()
        parts = self.assemble(USE)
        self.kv, self.attention = parts["kv_cache"], parts["attention"]

    def pool(self, rows, dtype="bf16", seed=0):
        """A pool holding `rows` (sequence -> keys, values) written as one step."""
        pool = self.kv.pool([0], 2, 64, 2048, self.dev, dtype if self.dev == "cuda" else "fp32")
        for sequence, (keys, values) in rows.items():
            pool.open(sequence)
            batch = StepBatch.build([(sequence, 0, keys.shape[0])], self.dev)
            pool.write(0, pool.reserve(batch), keys.to(self.dev), values.to(self.dev))
            pool.keep(sequence, keys.shape[0])
        return pool

    def test_it_has_the_attention_interface(self):
        self.assertImplements(self.assemble(USE))

    def test_merged_shards_equal_attention_over_the_whole_sequence(self):
        torch = self.torch
        g = torch.Generator().manual_seed(1)
        keys, values = torch.randn(300, 2, 64, generator=g), torch.randn(300, 2, 64, generator=g)
        q = torch.randn(1, 4, 64, generator=g).to(self.dev)
        if self.dev == "cuda":
            q = q.to(torch.bfloat16)
        whole = self.pool({"s": (keys, values)})
        shards = self.pool({"a": (keys[:120], values[:120]), "b": (keys[120:], values[120:])})
        # The query sees every position: the last of the whole sequence and of each shard (a step each: reserve()
        # first, as a model does, though the keys are written already).
        def step(pool, sequence, position):
            batch = StepBatch.build([(sequence, position, 1)], self.dev)
            pool.reserve(batch)
            return self.attention.forward(0, q, pool, batch, lse=True)
        out, lse = step(whole, "s", 299)
        parts = [step(shards, s, n - 1) for s, n in (("a", 120), ("b", 180))]
        merged, merged_lse = self.attention.merge([o for o, _ in parts], [x for _, x in parts])
        torch.testing.assert_close(merged.float(), out.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(merged_lse, lse, atol=2e-3, rtol=1e-4)
        # The normalizer is the log of the softmax's sum over the scaled scores.
        k = whole.gather(0, "s", 300)[0].double()
        for h in range(4):
            s = k[:, h // 2] @ q[0, h].double() / math.sqrt(64)
            self.assertAlmostEqual(float(lse[0, h]), float(torch.logsumexp(s, 0)), delta=2e-3)

    def test_merge_skips_parts_without_keys_like_merge_partials(self):
        torch = self.torch
        outs = [torch.ones(2, 1, 3), torch.zeros(2, 1, 3)]
        lses = [torch.tensor([[0.5], [float("-inf")]]), torch.full((2, 1), float("-inf"))]
        out, lse = self.attention.merge(outs, lses)
        self.assertEqual(out[0].tolist(), [[1.0, 1.0, 1.0]])
        self.assertEqual((out[1].tolist(), lse[1].tolist()), ([[0.0, 0.0, 0.0]], [float("-inf")]))
        self.assertAlmostEqual(float(lse[0, 0]), 0.5, places=6)

    def test_its_device_form_equals_softmax_attentions(self):
        torch = self.torch
        if self.dev != "cuda":
            self.skipTest("the split-KV reader runs on CUDA")
        reference = self.reference(USE, {"softmax-attention": {"decode": "split"}})["attention"]
        g = torch.Generator().manual_seed(2)
        rows = {s: (torch.randn(n, 2, 64, generator=g), torch.randn(n, 2, 64, generator=g))
                for s, n in (("a", 700), ("b", 64), ("c", 1))}
        pool = self.pool(rows)
        batch = StepBatch.build([("a", 699, 1), ("b", 60, 4), ("c", 0, 1)], "cuda")
        q = torch.randn(batch.tokens, 4, 64, generator=g).cuda().to(torch.bfloat16)
        self.assertTrue(torch.equal(self.attention.forward(0, q, pool, batch), reference.forward(0, q, pool, batch)))

    def test_memory_declares_exactly_what_the_device_form_allocates(self):
        from crit_plugins.assemble import load_module
        module = load_module(self.registry()["partitioned-attention"])
        base = {"kernels": "auto", "workspace_mib": 4, "flashinfer": "auto", "flashinfer_mib": 16,
                "prompt_reader": "prefill"}
        for change in ({}, {"flashinfer": "off"}, {"kernels": "torch"}, {"prompt_reader": "decode"}):
            options = dict(base, **change)
            declared = module.memory(None, options)
            if self.dev != "cuda":
                self.assertEqual(declared, {"device": 0, "host": 0})
                continue
            torch = self.torch
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            # attention-state's fresh buffer is its own declaration: here none, so only the attention's are counted
            form = self.assemble(USE, options=options, use_options={"attention-state": {"fresh_mib": 0}})[
                "attention"].device_form
            torch.cuda.synchronize()
            self.assertEqual(torch.cuda.memory_allocated() - before, declared["device"], change)
            self.assertEqual(sum(w._pin_memory_int_workspace_buffer.numel() for w in form._wrappers.values()),
                             declared["host"], change)
            self.assertNotIn(True, form._wrappers)    # decode is the split-KV reader's: no decode wrapper
            form.close()
