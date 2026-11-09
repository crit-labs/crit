"""Core's captured prompt chunks (crit_plugins.prompt_graphs): a sequence's prompt chunk runs in the smallest captured
bucket that holds it, padded, and gives exactly the eager prompt path's output and leaves exactly its state (the
padding changes none); the graphs slot's plugin chooses between the eager path and the buckets; a selection whose
plugins lack prompts:<slot>, or give a reason of their own, runs every chunk eagerly; the captures' memory is planned
through the placement plugin."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

try:
    import torch
except ImportError:  # pragma: no cover - torch is in crit's test environment
    torch = None

P = (1 << 31) - 1   # every value stays an exact int64 below it
LAYERS = 2
CUDA = torch is not None and torch.cuda.is_available()


class ToyPool:
    """Each sequence's per-layer state on the device, row 0 the scratch row: a running sum of the values its tokens
    wrote (the toy's KV) and a recurrent state. A prompt batch reads and writes the row stage_prompt put in row_t; a
    host StepBatch the row of its sequence."""

    def __init__(self, sequences, device):
        long = dict(dtype=torch.long, device=device)
        self.totals, self.states = torch.zeros(sequences + 1, LAYERS, **long), torch.zeros(sequences + 1, LAYERS, **long)
        self.row_t = torch.zeros(1, **long)
        self.rows, self.lengths, self.device = {}, {}, device

    def open(self, sequence):
        self.rows[sequence], self.lengths[sequence] = len(self.rows) + 1, 0

    def row_of(self, batch):
        if batch.prompt:
            return self.row_t
        return torch.tensor([self.rows[batch.sequences[0]]], device=self.device)

    def stage_prompt(self, batch):
        self.row_t.fill_(0 if batch.sequence is None else self.rows[batch.sequence])

    def commit_prompt(self, batch):
        if batch.sequence is not None:
            assert self.lengths[batch.sequence] == batch.frontier
            self.lengths[batch.sequence] += batch.count


class ToyModel:
    """A prompt forward over the pool, the same function on a host StepBatch (every token real) and on a PromptBatch
    (its padding masked): per layer each token writes a value into the sequence's running sum and reads the sum so
    far, and a recurrent state steps once per real token; the output is the last real token's hidden value."""

    def __init__(self, sequences, device):
        self.device = device
        self.embed = torch.randint(0, P, (97,), generator=torch.Generator().manual_seed(3)).to(device)
        self.pool = ToyPool(sequences, device)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        x = self.embed[batch.tokens_t % 97]
        real = batch.valid_t.long() if batch.prompt else torch.ones_like(x)
        row = self.pool.row_of(batch)
        for layer in range(LAYERS):
            v = (x * (layer + 3)) % P * real
            total = self.pool.totals[row, layer]
            seen = (torch.cumsum(v, 0) + total) % P
            self.pool.totals[row, layer] = (total + v.sum()) % P
            h, states = self.pool.states[row, layer], []
            for i in range(x.shape[0]):
                h = torch.where(real[i] > 0, (h * 31 + x[i]) % P, h)
                states.append(h)
            self.pool.states[row, layer] = h
            x = (x + seen + torch.cat(states)) % P
        last = batch.last_t if batch.prompt else torch.tensor([x.shape[0] - 1], device=self.device)
        return x.index_select(0, last)

    def eager(self, sequence, frontier, tokens):
        """The model's own eager prompt path: an unpadded StepBatch of the chunk."""
        from crit_plugins.device import StepBatch
        batch = StepBatch.build([(sequence, frontier, len(tokens))], self.device)
        batch.tokens_t = torch.tensor(tokens, dtype=torch.long, device=self.device)
        out = self.forward(batch)
        self.pool.lengths[sequence] += len(tokens)
        return out


class Parts:
    """A factory's parts for prompt_problems: the plugin of each slot, its facts and objects."""

    def __init__(self, use, provides, objects=None):
        self.use, self.provides, self.objects = use, provides, dict(objects or {})

    def slots(self):
        return list(self.use)

    def plugin(self, slot):
        return self.use.get(slot)

    def facts(self, slot):
        return tuple(self.provides.get(self.use.get(slot), ()))

    def get(self, slot, default=None):
        return self.objects.get(slot, default)


USE = {"model": "toy-model", "placement": "toy-placement", "graphs": "toy-graphs", "sampler": "toy-sampler",
       "kv_cache": "toy-kv", "attention": "toy-attention", "projections": "toy-projections"}
FACTS = {"toy-kv": ["prompts:kv_cache"], "toy-attention": ["prompts:attention"],
         "toy-projections": ["prompts:projections"]}
CHUNKS = [[[5, 9, 2, 7, 1]], [[3] * 23], [list(range(1, 17)), list(range(20, 60)), [8, 8]], [list(range(64))]]


@unittest.skipIf(torch is None, "torch is not installed")
class PromptGraphTests(unittest.TestCase):
    def build(self, device, **options):
        from crit_plugins.prompt_graphs import PromptGraphs
        self.eager_model, self.graph_model = ToyModel(8, device), ToyModel(8, device)
        for sequence in range(len(CHUNKS)):
            self.eager_model.pool.open(sequence)
            self.graph_model.pool.open(sequence)
        return PromptGraphs(self.graph_model.forward, device=device, pools=[self.graph_model.pool], **options)

    def compare(self, prompts, run):
        for sequence, chunks in enumerate(CHUNKS):
            frontier = 0
            for chunk in chunks:
                expected = self.eager_model.eager(sequence, frontier, chunk)
                found = run(prompts, sequence, frontier, chunk)
                self.assertEqual(found.tolist(), expected.tolist(), (sequence, frontier, len(chunk)))
                frontier += len(chunk)
        eager, graph = self.eager_model.pool, self.graph_model.pool
        self.assertEqual(graph.totals.tolist(), eager.totals.tolist(), "the padding wrote no state")
        self.assertEqual(graph.states.tolist(), eager.states.tolist())
        self.assertEqual(graph.lengths, eager.lengths)

    def test_a_padded_chunk_gives_the_eager_paths_output_and_state(self):
        # On the host device (no graphs): the padded batch, run eagerly, against the unpadded chunk.
        prompts = self.build("cpu", buckets=(16, 64))
        self.compare(prompts, lambda p, s, f, chunk: p.run_eager(s, f, chunk, 16 if len(chunk) <= 16 else 64))
        self.assertEqual(prompts.prepare(), [], "a host device captures nothing")
        self.assertIsNone(prompts.run(0, 0, [1, 2]), "and runs every chunk eagerly")

    def test_a_fill_moves_the_chunk_in_one_copy_and_derives_positions_and_validity(self):
        from crit_plugins.device import PromptBatch
        batch = PromptBatch(8, "cpu")
        batch.fill(8, "a", 10, [5, 6, 7, 8, 9, 4, 3])
        batch.fill(8, "b", 3, [1, 2], derive=False)
        self.assertEqual((batch.tokens_t.tolist(), batch.count_t.tolist(), batch.last_t.tolist(),
                          batch.frontier_t.tolist()), ([1, 2, 0, 0, 0, 0, 0, 0], [2], [1], [3]), "padding is token 0")
        self.assertEqual(batch.positions_t.tolist(), list(range(10, 18)), "derived on the device only by derive()")
        batch.derive()
        self.assertEqual(batch.positions_t.tolist(), list(range(3, 11)))
        self.assertEqual(batch.valid_t.tolist(), [True, True] + [False] * 6)
        self.assertEqual((batch.sequences, batch.starts, list(batch.positions)), (["b"], [0, 8], list(range(3, 11))))
        tokens = batch.tokens_t
        batch.fill(4, "c", 0, [9])
        batch.fill(8, "c", 1, [9, 9])
        self.assertEqual(batch.tokens_t.data_ptr(), tokens.data_ptr(), "a bucket's tensors stay where they were")

    def test_a_selection_lacking_prompt_facts_or_giving_a_reason_runs_chunks_eagerly(self):
        from crit_plugins.prompt_graphs import PromptsRefused, prompt_problems
        self.assertEqual(prompt_problems(Parts(USE, FACTS)), [], "the sampler and policy slots are not asked")
        lacking = prompt_problems(Parts(USE, {**FACTS, "toy-attention": ["rounds:attention"]}))
        self.assertEqual(lacking, ["toy-attention in attention does not provide prompts:attention: its prompt form is "
                                   "not known to run inside a captured graph"])

        class Tier:
            def round_problems(self):
                return ["its pages are read from a file tier"]
        reasons = prompt_problems(Parts(USE, FACTS, {"kv_cache": Tier()}), uses=["kv_cache"])
        self.assertIn("toy-kv in kv_cache cannot run inside a captured graph as configured", reasons[0])
        prompts = self.build("cuda" if CUDA else "cpu", parts=Parts(USE, FACTS, {"kv_cache": Tier()}))
        self.assertEqual(prompts.prepare(), [])
        self.assertIsNone(prompts.run(0, 0, [1, 2, 3]))
        with self.assertRaises(PromptsRefused):
            prompts.capture(16)

    @unittest.skipIf(not CUDA, "no CUDA device")
    def test_captured_chunks_give_the_eager_paths_output_and_state(self):
        plans = []

        class Placement:
            def plan(self, tensors, budget):
                plans.append(dict(tensors))
                return {name: "device" for name in tensors}
        prompts = self.build("cuda", buckets=(16, 64), parts=Parts(USE, FACTS, {"placement": Placement()}))
        self.assertEqual(prompts.prepare(), [16, 64])
        calls = self.graph_model.calls
        self.compare(prompts, lambda p, s, f, chunk: p.run(s, f, chunk).clone())
        self.assertEqual(self.graph_model.calls, calls, "every chunk replayed a graph: no forward call")
        self.assertIsNone(prompts.run(0, 66, list(range(65))), "a chunk past the largest bucket runs eagerly")
        counts = prompts.telemetry()["prompt_graphs"]
        self.assertEqual((counts["graph_chunks"], counts["eager_chunks"], counts["tokens"]), (6, 1, 5 + 23 + 58 + 64))
        self.assertEqual(sum(p["prompts:captures"] for p in plans), prompts.captured_bytes)
        self.assertGreater(prompts.captured_bytes, 0)
        for bucket in (16, 64):
            self.assertGreater(prompts.costs[bucket]["eager_ns"], 0)
            self.assertGreater(prompts.costs[bucket]["graph_ns"], 0)

    @unittest.skipIf(not CUDA, "no CUDA device")
    def test_the_offers_carry_the_buckets_steady_replay_times(self):
        import statistics
        import time
        prompts = self.build("cuda", buckets=(16, 64))
        prompts.prepare()
        for bucket in (16, 64):
            graph, _ = prompts.graphs[bucket]
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            steady = []
            for _ in range(60):
                started = time.perf_counter_ns()
                graph.replay()
                torch.cuda.synchronize()
                steady.append(time.perf_counter_ns() - started)
            steady = statistics.median(steady)
            offered = prompts.costs[bucket]["graph_ns"]
            self.assertTrue(steady / 2.5 <= offered <= steady * 2.5, (bucket, offered, steady))

    @unittest.skipIf(not CUDA, "no CUDA device")
    def test_the_graphs_plugin_chooses_between_the_eager_path_and_a_padded_bucket(self):
        seen = []

        class Eager:   # eager-graphs: the eager offer of the exact shape
            def choose(self, shape, offers, free_bytes, uses=1):
                seen.append((shape, [o["name"] for o in offers]))
                return next(o["name"] for o in offers if o["mode"] == "eager")

        class Fastest:   # padding allowed: the least measured run time
            def choose(self, shape, offers, free_bytes, uses=1):
                return min(offers, key=lambda o: o["run_ns"])["name"]
        prompts = self.build("cuda", buckets=(16, 64), parts=Parts(USE, FACTS, {"graphs": Eager()}))
        prompts.prepare()
        self.assertIsNone(prompts.run(0, 0, [1] * 10))
        shape, names = seen[0]
        self.assertEqual(shape, {"phase": "prefill", "tokens": 10, "rows": 1, "identity": ("prompt-graphs",)})
        self.assertEqual(names, ["eager", "graph:16", "graph:64"])
        self.assertEqual([o["shape"]["tokens"] for o in prompts.offers(20)[1]], [20, 64])
        self.assertEqual([o["bits"] for o in prompts.offers(16)[1]], ["eager", "padded:16", "padded:64"])
        prompts = self.build("cuda", buckets=(16, 64), parts=Parts(USE, FACTS, {"graphs": Fastest()}))
        prompts.capture(64)
        self.assertIsNotNone(prompts.run(0, 0, [1] * 10), "a replay beats the toy's eager loop")
        # The choice is asked once per chunk size while the offers stay as they are, and again after a capture.
        asked = []
        prompts._chooser = type("Counting", (), {"choose": lambda self, *a, **k: asked.append(a[0]["tokens"]) or
                                                 Fastest().choose(*a, **k)})()
        prompts._chosen.clear()
        for frontier in (10, 20):
            prompts.run(0, frontier, [1] * 10)
        self.assertEqual(asked, [10])
        prompts.capture(16)
        self.assertEqual(prompts.choose(10), 16)
        self.assertEqual(asked, [10, 10])


if __name__ == "__main__":
    unittest.main()
