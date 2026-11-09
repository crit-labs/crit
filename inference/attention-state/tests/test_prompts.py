"""attention-state's prompts:kv_cache: stage_prompt(batch) takes a captured prompt chunk's pages up to its end, puts its
real tokens' KV slots and its page table in fixed buffers and its padding at the dump page; commit_prompt(batch) sets
the length; a chunk of padding alone takes nothing; a bound sequence's prompt is refused."""
from crit_plugins.device import PromptBatch
from crit_plugins.testing import PluginTestCase


class PromptStagingTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch")
        self.dev = self.device()
        self.kv = self.assemble(options={"page": 16})["kv_cache"]

    def test_a_staged_chunk_takes_its_pages_and_puts_its_padding_at_the_dump_page(self):
        pool = self.kv.pool([0], 2, 16, 512, self.dev, "bf16")
        pool.open("a")
        batch = PromptBatch(64, self.dev)
        free = pool.free_pages()
        pool.stage_prompt(batch.fill(32, None, 0, []))          # a capture's warm-up: padding alone
        self.assertEqual(pool.free_pages(), free)
        self.assertEqual(set(pool.reserve(batch).tolist()), {pool.dump * pool.page})
        pool.stage_prompt(batch.fill(32, "a", 0, list(range(20))))
        slots, table = pool.reserve(batch).tolist(), pool.table("a")
        self.assertEqual(len(table), 2)                          # 20 positions: two pages
        self.assertEqual(slots[:20], [table[p // 16] * 16 + p % 16 for p in range(20)])
        self.assertEqual(set(slots[20:]), {pool.dump * pool.page})
        self.assertEqual(pool.prompt_table[0, :2].tolist(), table)
        self.assertEqual(pool.prompt_table[0, 2:].unique().tolist(), [pool.dump])
        pool.commit_prompt(batch)
        self.assertEqual(pool.length("a"), 20)
        pool.stage_prompt(batch.fill(16, "a", 20, [7] * 15))   # the next chunk continues it: one page more
        self.assertEqual(len(pool.table("a")), 3)
        pool.commit_prompt(batch)
        self.assertEqual(pool.length("a"), 35)
        pool.bind(0, "a", 64)
        with self.assertRaisesRegex(ValueError, "round slot"):
            pool.stage_prompt(batch.fill(16, "a", 35, [1, 2]))


class FreshBufferTests(PluginTestCase):
    """The fresh buffer a pool of FP8 or INT8 pages keeps a prompt chunk's keys and values in: declared exactly, made
    once by the plugin, viewed by each such pool, used only in a step with prompt rows, and stated in the facts."""

    def setUp(self):
        self.torch = __import__("torch")
        self.dev = self.device()

    def test_memory_declares_exactly_the_fresh_buffer(self):
        from crit_plugins.assemble import load_module
        torch = self.torch
        module = load_module(self.registry()["attention-state"])
        for mib in (0, 4):
            declared = module.memory(None, {"fresh_mib": mib})
            if self.dev != "cuda":
                self.assertEqual(declared, {"device": 0, "host": 0})
                continue
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            kv = self.assemble(options={"fresh_mib": mib})["kv_cache"]
            torch.cuda.synchronize()
            self.assertEqual(torch.cuda.memory_allocated() - before, declared["device"], mib)
            self.assertEqual(declared["device"], mib << 20)
            del kv

    def test_a_staged_chunk_past_the_window_reads_itself_fresh_and_a_shorter_one_its_pages(self):
        if self.dev != "cuda":
            self.skipTest("the fresh buffer is a CUDA engine's")
        kv = self.assemble(options={"fresh_mib": 4, "page": 16, "window": 16})["kv_cache"]
        pool = kv.pool([0], 2, 64, 512, self.dev, "fp8:row")
        pool.open("a")
        batch = PromptBatch(64, self.dev)
        pool.stage_prompt(batch.fill(64, "a", 0, list(range(40))))       # 40 real tokens: a prompt row, fresh
        self.assertEqual(int(pool.prompt_fresh_first), 0)
        pool.commit_prompt(batch)
        pool.stage_prompt(batch.fill(16, "a", 40, list(range(16))))      # up to the window: its pages, as a step's
        self.assertEqual(int(pool.prompt_fresh_first), 40 + 16)
        self.assertIsNone(kv.pool([0], 2, 64, 512, self.dev, "bf16").prompt_fresh_first, "16-bit pages keep nothing")

    def test_a_prompt_step_keeps_its_keys_and_values_and_the_facts_say_so(self):
        from crit_plugins import StepBatch
        if self.dev != "cuda":
            self.skipTest("the fresh buffer is a CUDA engine's")
        torch = self.torch
        kv = self.assemble(options={"fresh_mib": 4, "page": 16})["kv_cache"]
        self.assertEqual(kv.facts(), {})
        pool = kv.pool([0, 1], 2, 64, 512, self.dev, "int8")
        self.assertEqual(pool.fresh_tokens, (4 << 20) // (2 * (64 + 64) * 2))
        pool.open("a")
        batch = StepBatch.build([("a", 0, 40)], self.dev)              # a prompt row (past the window)
        keys, values = torch.randn(40, 2, 64, device=self.dev), torch.randn(40, 2, 64, device=self.dev)
        pool.write(1, pool.reserve(batch), keys, values)
        found = pool.fresh_rows(1, 40)
        self.assertTrue(torch.equal(found[0], keys.to(torch.bfloat16)) and torch.equal(found[1], values.to(torch.bfloat16)))
        self.assertIsNone(pool.fresh_rows(0, 40), "another layer's write is not this one's")
        other = kv.pool([0], 2, 64, 512, self.dev, "fp8:row")      # the plugin's other pool, the same buffer
        other.open("b")
        other.write(0, other.reserve(StepBatch.build([("b", 0, 40)], self.dev)), values, keys)
        self.assertIsNone(pool.fresh_rows(1, 40), "another pool's write took the buffer: the pages are read")
        self.assertTrue(torch.equal(other.fresh_rows(0, 40)[0], values.to(torch.bfloat16)))
        pool.keep("a", 40)
        pool.write(1, pool.reserve(StepBatch.build([("a", 40, 1)], self.dev)), keys[:1], values[:1])
        self.assertIsNone(pool.fresh_rows(1, 1), "a decode step keeps nothing fresh")
        facts = kv.facts()
        self.assertIn("as the projections gave them", facts["prompt_kv"])
        self.assertEqual(facts["prompt_kv_fresh_tokens"], pool.fresh_tokens)
