"""memory-table-rows in captured prompt chunks (crit_plugins.prompt_graphs): a padded bucket's lookup over the device
table or the registered host table captures and replays the chunk's rows exactly, the padding reading row 0."""
import os

from crit_plugins.testing import PluginTestCase

NO_HOST_MEMORY = "host memory regions need libc's mmap, which Windows does not have"

BUCKETS = (16, 64)
CHUNKS = [[3, 0, 499], list(range(16)), list(range(100, 140)), [42]]


class PromptChunkTests(PluginTestCase):
    def test_a_padded_buckets_lookup_captures_and_replays_the_chunks_rows(self):
        torch, device = __import__("torch"), self.device()
        from crit_plugins.prompt_graphs import PromptGraphs
        table = torch.randn(500, 32, generator=torch.Generator().manual_seed(6)).bfloat16()
        for host in (False, True):
            with self.subTest(host=host):
                if host and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                assembly = self.assemble(options={"host": host})
                rows = assembly["embeddings"]
                rows.store(table.to(device))

                def forward(batch):
                    return rows.lookup(batch.tokens_t) * batch.valid_t[:, None]
                prompts = PromptGraphs(forward, device=device, buckets=BUCKETS, parts=assembly.parts,
                                       uses=["embeddings"])
                self.assertEqual(prompts.problems, [])
                captured = prompts.prepare()
                self.assertEqual(captured, list(BUCKETS) if device == "cuda" else [])
                frontier = 0
                for chunk in CHUNKS:
                    bucket = next(b for b in BUCKETS if b >= len(chunk))
                    out = prompts.run(0, frontier, chunk) if captured else prompts.run_eager(0, frontier, chunk, bucket)
                    expected = torch.zeros(bucket, 32, dtype=torch.bfloat16)
                    expected[:len(chunk)] = table[chunk]
                    self.assertTrue(torch.equal(out.cpu(), expected), (host, chunk))
                    frontier += len(chunk)
