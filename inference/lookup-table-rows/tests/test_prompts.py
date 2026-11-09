"""lookup-table-rows' tables in captured prompt chunks (crit_plugins.prompt_graphs): a padded bucket's lookup over a
device table, registered host memory or files read by the native reader captures, replays the chunk's rows exactly
(the padding reads row 0 and is masked by the forward), and keeps its buffer set out of the eager ring, so an eager
lookup of the bucket's size between replays never overwrites the ids a replay has not read yet; a table the Python
reader serves is refused through round_problems() and every chunk runs eagerly."""
import os
import tempfile

from crit_plugins.testing import PluginTestCase

# Host memory regions (a table opened from a tensor, or a file mapped) need libc's mmap, which Windows does not
# have; files read by the readers and device tables work everywhere.
HOST_PLACES = ("host", "mapped")
NO_HOST_MEMORY = "host memory regions need libc's mmap, which Windows does not have"

BUCKETS = (16, 64, 128)
CHUNKS = [[5, 299, 300, 301, 999], list(range(16)), [7] * 23, list(range(450, 550)), [998, 0, 301]]


class PromptChunkTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        g = self.torch.Generator().manual_seed(4)
        self.table = self.torch.randn(1000, 24, generator=g)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.paths = []
        for i, (first, last) in enumerate(((0, 300), (300, 301), (301, 1000))):
            path = os.path.join(self.folder.name, f"rows-{i}.bin")
            with open(path, "wb") as f:
                f.write(self.table[first:last].numpy().tobytes())
            self.paths.append(path)

    def selection(self, **options):
        assembly = self.assemble(options=options)
        return assembly, assembly["embeddings"]

    def open(self, rows, place):
        torch, device = self.torch, self.device()
        if place == "device":
            return rows.store(self.table.to(device))
        if place == "host":
            return rows.open(self.table, 24, torch.float32, device)
        return rows.open(self.paths, 24, torch.float32, device)

    def prompts(self, assembly, table):
        from crit_plugins.prompt_graphs import PromptGraphs

        def forward(batch):   # the embedding step of a prompt forward: the bucket's rows, the padding's masked
            return table.lookup(batch.tokens_t) * batch.valid_t[:, None]
        return PromptGraphs(forward, device=self.device(), buckets=BUCKETS, parts=assembly.parts, uses=["embeddings"])

    def expected(self, chunk, bucket):
        out = self.torch.zeros(bucket, 24)
        out[:len(chunk)] = self.table[chunk]
        return out

    def test_a_padded_buckets_lookup_captures_and_replays_the_chunks_rows(self):
        torch = self.torch
        cuda = self.device() == "cuda"
        for place in ("device", "host", "files"):
            with self.subTest(place=place):
                if place in HOST_PLACES and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                assembly, rows = self.selection(prefetch_rows=0)
                table = self.open(rows, place)
                if place == "files" and rows.facts()["embeddings"].get("reader") != "native":
                    self.skipTest("the native row reader did not build")
                prompts = self.prompts(assembly, table)
                self.assertEqual(prompts.problems, [])
                if not cuda:   # the host device captures nothing: the padded chunk run eagerly
                    for chunk in CHUNKS:
                        bucket = next(b for b in BUCKETS if b >= len(chunk))
                        out = prompts.run_eager(0, 0, chunk, bucket)
                        self.assertTrue(torch.equal(out.cpu(), self.expected(chunk, bucket)))
                    self.assertEqual(prompts.prepare(), [])
                    continue
                self.assertEqual(prompts.prepare(), list(BUCKETS))
                frontier, found, between = 0, [], []
                for chunk in CHUNKS:   # no synchronization: replays and eager lookups queue on the stream
                    found.append(prompts.run(0, frontier, chunk).clone())
                    other = [(len(chunk) * 37 + i) % 1000 for i in range(len(found[-1]))]
                    between.append((other, table.lookup(torch.tensor(other)).clone()))   # host ids, the bucket's size
                    frontier += len(chunk)
                torch.cuda.synchronize()
                for chunk, out in zip(CHUNKS, found):
                    self.assertTrue(torch.equal(out.cpu(), self.expected(chunk, out.shape[0])), (place, chunk))
                for other, out in between:
                    self.assertTrue(torch.equal(out.cpu(), self.table[other]), (place, len(other)))
                self.assertEqual(prompts.telemetry()["prompt_graphs"]["graph_chunks"], len(CHUNKS))

    def test_a_table_the_python_reader_serves_runs_chunks_eagerly(self):
        assembly, rows = self.selection(native=False)
        table = self.open(rows, "files")
        prompts = self.prompts(assembly, table)
        self.assertEqual(len(prompts.problems), 1)
        self.assertIn("lookup-table-rows in embeddings cannot run inside a captured graph as configured: the table "
                      "is read by Python threads", prompts.problems[0])
        self.assertEqual(prompts.prepare(), [])
        self.assertIsNone(prompts.run(0, 0, [1, 2, 3]))

    def test_captures_that_would_hold_every_buffer_set_are_refused(self):
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        assembly, rows = self.selection(ring=1)
        table = self.open(rows, "files")
        if not rows.device_rows.tables[-1].native:
            self.skipTest("the native row reader did not build")
        import warnings
        with warnings.catch_warnings(), self.assertRaisesRegex(RuntimeError, "larger ring"):
            warnings.simplefilter("ignore")   # the refused capture ends empty
            self.prompts(assembly, table).prepare()
