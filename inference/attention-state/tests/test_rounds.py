"""attention-state's rounds:kv_cache: sequences bound to round slots with their pages reserved up to their capacity, a
RoundBatch's slots computed on the device (in a CUDA graph too), lengths committed on the device and read back."""
from types import SimpleNamespace

from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch
from crit_plugins.testing import PluginTestCase


class RoundSlotTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch")
        self.dev = self.device()
        self.kv = self.assemble(options={"page": 16})["kv_cache"]

    def pool(self, dtype="bf16", tokens=1024):
        return self.kv.pool([0, 1], 2, 16, tokens, self.dev, dtype)

    def prefill(self, pool, sequence, count, seed=0):
        pool.open(sequence)
        g = self.torch.Generator().manual_seed(seed)
        batch = StepBatch.build([(sequence, 0, count)], self.dev)
        slots = pool.reserve(batch)
        for layer in (0, 1):
            pool.write(layer, slots, self.torch.randn(count, 2, 16, generator=g).to(self.dev),
                       self.torch.randn(count, 2, 16, generator=g).to(self.dev))
        pool.keep(sequence, count)

    def round(self, rows, width, frontier):
        batch = RoundBatch(4, 7, self.dev)
        columns = self.torch.zeros(rows, width + 1, dtype=self.torch.int64, device=self.dev)
        return batch.fill(rows, width, self.torch.tensor(frontier, device=self.dev), columns)

    def expected(self, pool, sequence, positions):
        table = pool.table(sequence)
        return [table[p // pool.page] * pool.page + p % pool.page for p in positions]

    def test_it_provides_the_rounds_methods(self):
        self.assertImplements()

    def test_a_bound_sequence_reserves_on_the_device_where_its_table_says(self):
        pool = self.pool()
        self.prefill(pool, "a", 20)
        self.prefill(pool, "b", 33, seed=1)
        free = pool.free_pages()
        pool.bind(0, "a", 20 + 40)
        pool.bind(1, "b", 33 + 40)
        self.assertEqual(len(pool.table("a")), 4)          # 60 positions: 4 pages
        self.assertEqual(len(pool.table("b")), 5)          # 73 positions: 5 pages
        self.assertEqual(free - pool.free_pages(), 2 + 2)   # their new pages (the dump page is no sequence page)
        self.assertEqual(pool.dump, pool.pages)
        # The tables' entries past a sequence's pages point at the dump page, never at a page a sequence may own.
        self.assertEqual(pool.round_tables[0, 4:].unique().tolist(), [pool.dump])
        self.assertEqual(pool.round_tables[2].unique().tolist(), [pool.dump])
        batch = self.round(2, 3, [20, 33])
        slots = pool.reserve(batch).tolist()
        self.assertEqual(slots[:4], self.expected(pool, "a", range(20, 24)))
        self.assertEqual(slots[4:], self.expected(pool, "b", range(33, 37)))
        # A position past a slot's capacity goes to the dump page, which no sequence holds.
        late = pool.reserve(self.round(2, 3, [58, 33])).tolist()
        self.assertEqual(late[:2], self.expected(pool, "a", [58, 59]))
        self.assertEqual(late[2:4], [pool.dump * pool.page] * 2)
        for sequence in "ab":
            self.assertNotIn(pool.dump, pool.table(sequence))

    def test_a_round_commits_its_lengths_on_the_device_and_unbind_reads_them_back(self):
        pool = self.pool()
        self.prefill(pool, "a", 20)
        pool.bind(0, "a", 100)
        frontier = self.torch.tensor([23], device=self.dev)
        pool.commit_round(SimpleNamespace(rows=1, frontier=frontier, n_emitted=self.torch.tensor([3], device=self.dev)))
        self.assertEqual(pool.length("a"), 23)              # read from the device while bound
        pool.keep("a", 22)                                  # a bound sequence keeps its pages
        self.assertEqual((pool.length("a"), len(pool.table("a"))), (22, 7))
        pool.unbind(0)
        self.assertEqual((pool.length("a"), len(pool.table("a"))), (22, 2))
        with self.assertRaisesRegex(ValueError, "holds no sequence"):
            pool.unbind(0)
        pool.bind(0, "a", 40)                               # bound again where it was left
        self.assertEqual(int(pool.round_lengths[0]), 22)

    def test_binding_is_checked(self):
        pool = self.pool(tokens=64)
        self.prefill(pool, "a", 20)
        self.prefill(pool, "b", 5)
        with self.assertRaisesRegex(ValueError, "too small"):
            pool.bind(0, "a", 10)
        with self.assertRaisesRegex(ValueError, "outside the pool's 8 round slots"):
            pool.bind(8, "a", 30)
        with self.assertRaisesRegex(MemoryError, "takes"):
            pool.bind(0, "a", 64)
        pool.bind(0, "a", 32)
        with self.assertRaisesRegex(ValueError, "holds 'a' already"):
            pool.bind(0, "b", 8)
        with self.assertRaisesRegex(ValueError, "in round slot 0 already"):
            pool.bind(1, "a", 32)
        pool.move(0, 3)
        self.assertEqual((pool.bound, pool.slot_of), ({3: "a"}, {"a": 3}))
        self.assertEqual(pool.round_tables[0].unique().tolist(), [pool.dump])   # the slot it left
        self.assertEqual(pool.reserve(self.round(4, 0, [0, 0, 0, 20])).tolist()[3], self.expected(pool, "a", [20])[0])
        pool.close("a")                                     # closing a bound sequence frees its slot
        self.assertEqual(pool.bound, {})
        self.assertEqual(pool.round_tables[3].unique().tolist(), [pool.dump])

    def test_a_round_wider_than_the_round_rows_is_refused_on_the_host(self):
        torch = self.torch
        pool = self.pool()
        batch = RoundBatch(9, 0, self.dev).fill(9, 0, torch.zeros(9, dtype=torch.int64, device=self.dev),
                                                torch.zeros(9, 1, dtype=torch.int64, device=self.dev))
        with self.assertRaisesRegex(ValueError, "a round of 9 rows .* has 8"):
            pool.reserve(batch)

    def test_a_held_row_keeps_what_the_round_found_and_a_keep_between_rounds_stands(self):
        torch = self.torch
        pool = self.pool()
        self.prefill(pool, "a", 20)
        self.prefill(pool, "b", 30, seed=1)
        pool.bind(0, "a", 100)
        pool.bind(1, "b", 100)

        def view(frontier, emitted):
            return SimpleNamespace(rows=2, frontier=torch.tensor(frontier, device=self.dev),
                                   n_emitted=torch.tensor(emitted, device=self.dev), tree_path=None)
        first = view([23, 30], [3, 0])                      # b held: its frontier as the driver left it
        pool.start_round(first)
        pool.commit_round(first)
        pool.commit_round(view([22, 30], [2, 0]))           # a truncation commits again from the round's start
        self.assertEqual((pool.length("a"), pool.length("b")), (22, 30))
        pool.keep("b", 27)                                  # between rounds (a host rollback of b)
        second = view([24, 31], [2, 0])                     # b held again, its frontier stale
        pool.start_round(second)
        pool.commit_round(second)
        self.assertEqual((pool.length("a"), pool.length("b")), (24, 27))

    def test_a_write_skips_slots_below_zero(self):
        torch = self.torch
        for kernels in ("auto", "torch"):
            kv = self.assemble(options={"page": 16, "kernels": kernels})["kv_cache"]
            for dtype in ("bf16", "fp8:row", "int8"):
                pool = kv.pool([0, 1], 2, 16, 256, self.dev, dtype)
                self.prefill(pool, "a", 20)
                before = pool.gather(0, "a", 20)
                slots = torch.tensor([-1, -5, -1], device=self.dev)
                pool.write(0, slots, torch.randn(3, 2, 16, device=self.dev), torch.randn(3, 2, 16, device=self.dev))
                for got, was in zip(pool.gather(0, "a", 20), before):
                    self.assertTrue(torch.equal(got, was), (kernels, dtype))

    def test_a_fork_of_a_bound_sequence_owns_its_tail_page(self):
        pool = self.pool()
        self.prefill(pool, "a", 20)
        pool.bind(0, "a", 60)
        pool.fork("a", "b")
        self.assertEqual(pool.table("b")[0], pool.table("a")[0])     # the full page is shared
        self.assertNotEqual(pool.table("b")[1], pool.table("a")[1])  # the tail the round writes is not
        self.assertTrue(self.torch.equal(pool.gather(0, "b", 20)[0], pool.gather(0, "a", 20)[0]))

    def test_round_reservations_replay_from_a_cuda_graph(self):
        if self.dev != "cuda":
            self.skipTest("CUDA graphs need CUDA")
        torch = self.torch
        pool = self.pool()
        for s, n in (("a", 20), ("b", 70)):
            self.prefill(pool, s, n)
        pool.bind(0, "a", 120)
        pool.bind(1, "b", 120)
        batch = RoundBatch(2, 3, self.dev)
        frontier = torch.tensor([20, 70], device=self.dev)
        columns = torch.zeros(2, 4, dtype=torch.int64, device=self.dev)
        keys = torch.randn(8, 2, 16, device=self.dev)

        def run():
            batch.fill(2, 3, frontier, columns)
            slots = pool.reserve(batch)
            pool.write(0, slots, keys, keys)
            return slots
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for a, b in ((21, 71), (40, 100), (116, 75)):
            frontier.copy_(torch.tensor([a, b]))
            graph.replay()
            want = self.expected(pool, "a", range(a, a + 4)) + self.expected(pool, "b", range(b, b + 4))
            want = [w if p < 120 else pool.dump * pool.page for w, p in zip(want, [*range(a, a + 4), *range(b, b + 4)])]
            self.assertEqual(out.tolist(), want)
        pool.commit_round(SimpleNamespace(rows=2, frontier=frontier, n_emitted=frontier * 0 + 1))
        got = pool.gather(0, "a", 120)[0][116:120].float()
        self.assertTrue(torch.equal(got, keys[:4].to(pool.compute).float()))

    def test_commit_moves_an_accepted_tree_path_into_consecutive_positions(self):
        torch = self.torch
        for dtype in ("bf16", "int8", "fp8:row"):
            pool = self.pool(dtype)
            self.prefill(pool, "a", 30)
            pool.bind(0, "a", 80)
            batch = RoundBatch(1, 5, self.dev).fill(1, 5, torch.tensor([30], device=self.dev),
                                                    torch.zeros(1, 6, dtype=torch.int64, device=self.dev))
            slots = pool.reserve(batch)
            window = torch.randn(6, 2, 16, device=self.dev)
            for layer in (0, 1):
                pool.write(layer, slots, window * (layer + 1), window)
            before = pool.gather(1, "a", 36)
            # The pending token (column 0) and the drafts of columns 2 and 5 accepted: 3 tokens emitted (the two
            # drafts and the next one), whose inputs' KV stays, frontier 30 -> 33.
            # The round driver's view: the tree verifier's accepted nodes (view.tree_path, -1 past them).
            view = SimpleNamespace(rows=1, frontier=torch.tensor([33], device=self.dev),
                                   n_emitted=torch.tensor([3], device=self.dev),
                                   tree_path=torch.tensor([[0, 2, 5, -1, -1, -1]], device=self.dev))
            pool.start_round(view)
            pool.commit_round(view)
            pool.commit_round(view)      # a second commit of the round (a truncation's) finds the path moved
            after = pool.gather(1, "a", 36)
            for got, was in zip(after, before):
                self.assertTrue(torch.equal(got[:30], was[:30]), dtype)
                self.assertTrue(torch.equal(got[30:33], was[[30, 32, 35]]), dtype)
            self.assertEqual(pool.length("a"), 33)
            # A held row (n_emitted 0) keeps its length through a commit.
            pool.start_round(view)
            pool.commit_round(SimpleNamespace(rows=1, frontier=torch.tensor([33], device=self.dev),
                                              n_emitted=torch.tensor([0], device=self.dev), tree_path=view.tree_path))
            self.assertTrue(torch.equal(pool.gather(1, "a", 33)[0], after[0][:33]))
            self.assertEqual(pool.length("a"), 33)

    def test_the_dump_page_lies_past_what_a_borrower_is_lent_and_a_round_asks_no_lender(self):
        torch = self.torch
        pool = self.pool()
        calls = []
        pool.on_reserve(lambda floor: calls.append(floor) or (-1, None))
        info = pool.lendable()
        for first, per_page, pages in info["regions"]:
            self.assertEqual(pages, pool.pages)             # the dump page (index pages) is past every region
        batch = RoundBatch(2, 1, self.dev).fill(2, 1, torch.tensor([5, 9], device=self.dev),
                                                torch.zeros(2, 2, dtype=torch.int64, device=self.dev))
        pool.write(0, pool.reserve(batch), torch.randn(4, 2, 16, device=self.dev), torch.randn(4, 2, 16, device=self.dev))
        self.assertEqual((calls, pool.floor), ([0], 0))   # a round's junk writes take no page
        pool.on_reserve(None)

    def test_a_round_reserves_before_any_sequence_is_bound(self):
        # A capture's warm-ups may run before a bind: every token goes to the dump page.
        torch = self.torch
        pool = self.pool()
        batch = RoundBatch(2, 1, self.dev).fill(2, 1, torch.tensor([5, 9], device=self.dev),
                                                torch.zeros(2, 2, dtype=torch.int64, device=self.dev))
        self.assertEqual(set(pool.reserve(batch).tolist()), {pool.dump * pool.page})


class KeyRowsTests(PluginTestCase):
    """kv:keys and kv:scales: a pool of keys only, rows read at slots and placed at chosen positions of bound
    sequences inside a round, and static FP8 scales."""

    def setUp(self):
        self.torch = __import__("torch")
        self.dev = self.device()

    def test_a_pool_of_keys_only_writes_and_reads_its_keys(self):
        torch = self.torch
        for kernels in ("auto", "torch"):
            kv = self.assemble(options={"page": 16, "kernels": kernels})["kv_cache"]
            for dtype in ("bf16", "fp8:row", "int8"):
                pool = kv.pool([0], 1, 32, 256, self.dev, dtype, value_dim=0)
                self.assertEqual(pool.value_dim, 0)
                pool.open("a")
                batch = StepBatch.build([("a", 0, 40)], self.dev)
                keys = torch.randn(40, 1, 32, device=self.dev)
                slots = pool.reserve(batch)
                pool.write(0, slots, keys, None)
                got, values = pool.gather(0, "a", 40)
                self.assertEqual(tuple(values.shape), (40, 1, 0))
                torch.testing.assert_close(got.float(), keys, atol=0.07, rtol=0.07, msg=(kernels, dtype))
                at, empty = pool.rows_at(0, slots.view(8, 5))
                self.assertEqual((tuple(at.shape), tuple(empty.shape)), ((8, 5, 1, 32), (8, 5, 1, 0)))
                # rows_at in FP32, gather in the compute dtype (BF16)
                torch.testing.assert_close(at.view(40, 1, 32), got.float(), atol=0.02, rtol=0.01)

    def test_slots_at_places_rows_of_bound_sequences_without_the_host(self):
        torch = self.torch
        kv = self.assemble(options={"page": 4})["kv_cache"]
        blocks = kv.pool([0], 1, 16, 512, self.dev, "bf16", value_dim=0)   # a per-block row cache: one row a block
        for s in "ab":
            blocks.open(s)
        blocks.bind(0, "a", 64)
        blocks.bind(1, "b", 64)
        slot = torch.tensor([[0, 0, 0], [1, 1, 1]], device=self.dev)
        where = torch.tensor([[0, 5, 63], [7, 8, 64]], device=self.dev)
        found = blocks.slots_at(slot, where)
        table_a, table_b = blocks.table("a"), blocks.table("b")
        want = [[table_a[p // 4] * 4 + p % 4 for p in (0, 5, 63)], [table_b[p // 4] * 4 + p % 4 for p in (7, 8)]]
        self.assertEqual(found[0].tolist(), want[0])
        self.assertEqual(found[1, :2].tolist(), want[1])
        self.assertEqual(int(found[1, 2]), blocks.dump * 4)   # past b's capacity: the dump page
        rows = torch.randn(6, 1, 16, device=self.dev)
        blocks.write(0, found.reshape(-1), rows, None)
        got, _ = blocks.rows_at(0, found[:, :2])
        self.assertTrue(torch.equal(got, rows.view(2, 3, 1, 16)[:, :2].to(torch.bfloat16).float()))

    def test_static_fp8_scales_are_the_layers_multipliers(self):
        torch = self.torch
        kv = self.assemble(options={"page": 16})["kv_cache"]
        pool = kv.pool([0, 1], 2, 32, 128, self.dev, "fp8")
        pool.set_scales(1, 0.25, 4.0)
        pool.open("a")
        keys = torch.randn(10, 2, 32, device=self.dev)
        pool.write(1, pool.reserve(StepBatch.build([("a", 0, 10)], self.dev)), keys, keys)
        stored = pool.k[1].view(-1, 2, 32)[torch.tensor(pool.table("a"), device=self.dev).repeat_interleave(16)[:10] * 16
                                           + torch.arange(10, device=self.dev) % 16]
        want = (keys / torch.full_like(keys, 0.25)).clamp(-448, 448).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(stored.view(torch.uint8), want.view(torch.uint8)))
