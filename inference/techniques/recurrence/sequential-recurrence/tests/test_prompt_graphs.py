"""sequential-recurrence's prompt form (prompts:recurrence) in Core's captured prompt chunks
(crit_plugins.prompt_graphs): a chunk replayed in a padded bucket gives the bits of the step path's scan rows over the
same tokens (outputs, state and convolution window, at the gated-delta shapes of Qwen3.8-27B and Flash-Next and a
narrower one), its padding takes no step and leaves the window, padding alone (the captures' warm-ups) changes no
sequence, and decoding after a captured chunk continues as after the step path. A chunk the step path would hand to the
chunked kernel agrees with it within the stated prompt tolerance."""
from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

VOCAB = 97
LAYERS = 2
KERNEL = 4
PROMPT = dict(atol=2e-3, rtol=2e-2)
# (key heads, value heads, keys, values): Qwen3.8-27B's and Flash-Next's gated-delta layers (the same per layer: 16 key
# and 48 value heads of 128, a 4-wide convolution over 10,240 channels), and a narrower grouping.
SHAPES = ((16, 48, 128, 128), (2, 6, 64, 32))
# Each sequence's chunks: a short one, one past a bucket, a run of chunks (a two-token one a decode row's size).
CHUNKS = [[[5, 9, 2, 7, 1]], [[3] * 23], [list(range(1, 17)), list(range(20, 60)), [8, 8], list(range(70, 97))]]


class Layers:
    """A stack of gated-delta layers over the recurrence slot, as a hybrid model runs them: token ids give the
    convolution's inputs and the gates (fixed tables), each layer convolves and continues its state; forward(batch)
    returns every layer's output at the chunk's last real token, the same function on a host StepBatch and on a
    PromptBatch."""

    def __init__(self, torch, plugin, shape, device, seed=11):
        self.torch, self.plugin, self.device = torch, plugin, device
        self.kh, self.h, self.dk, self.dv = shape
        self.channels = 2 * self.kh * self.dk + self.h * self.dv
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(VOCAB, self.channels, generator=g).to(device, torch.bfloat16)
        self.gates = torch.randn(VOCAB, 2 * self.h, generator=g).to(device)
        self.weights = [(torch.randn(self.channels, KERNEL, generator=g) * 0.5).to(device) for _ in range(LAYERS)]
        self.pool = plugin.pool(list(range(LAYERS)), self.h, self.dk, self.dv, device, 4, 3, (self.channels, KERNEL),
                                self.kh, 2)
        self.geometry = {"keys": self.dk, "values": self.dv, "epsilon": 1e-6, "query_scale": self.dk ** -0.5,
                         "normalize_keys": True, "normalize_queries": True}
        self.calls = 0

    def forward(self, batch):
        torch = self.torch
        self.calls += 1
        x = self.x.index_select(0, batch.tokens_t)
        gates = self.gates.index_select(0, batch.tokens_t)
        beta, decay = torch.sigmoid(gates[:, :self.h]), -torch.nn.functional.softplus(gates[:, self.h:])
        t = x.shape[0]
        found = []
        for layer in range(LAYERS):
            y = self.plugin.convolve(layer, x, self.weights[layer], None, self.pool, batch)
            q, k, v = torch.split(y, [self.kh * self.dk, self.kh * self.dk, self.h * self.dv], -1)
            found.append(self.plugin.forward(layer, q.reshape(t, self.kh, self.dk), k.reshape(t, self.kh, self.dk),
                                             v.reshape(t, self.h, self.dv), decay, beta, self.pool, batch,
                                             self.geometry))
        last = batch.last_t if batch.prompt else torch.tensor([t - 1], device=self.device)
        return torch.stack(found).index_select(1, last)

    def eager(self, sequence, frontier, tokens):
        """The step path: an unpadded StepBatch of the chunk, then its accept."""
        batch = StepBatch.build([(sequence, frontier, len(tokens))], self.device)
        batch.tokens_t = self.torch.tensor(tokens, dtype=self.torch.int64, device=self.device)
        out = self.forward(batch)
        self.pool.accept(sequence, len(tokens))
        return out


class PromptGraphTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None

    def models(self, shape, **options):
        plugin = self.assemble(options=options)["recurrence"]
        step, graph = Layers(self.torch, plugin, shape, self.device()), Layers(self.torch, plugin, shape, self.device())
        for model in (step, graph):
            for sequence in range(len(CHUNKS) + 1):
                model.pool.open(sequence)
        return step, graph

    def assertSame(self, step, graph, sequences, exact=True):
        torch = self.torch
        for sequence in sequences:
            for layer in range(LAYERS):
                for name in ("state", "window"):
                    a, b = getattr(step.pool, name)(layer, sequence), getattr(graph.pool, name)(layer, sequence)
                    if exact or name == "window":
                        self.assertTrue(torch.equal(a, b), (sequence, layer, name))
                    else:
                        torch.testing.assert_close(b, a, **PROMPT)

    def test_it_provides_the_prompt_form(self):
        self.assertImplements()
        self.assertIn("prompts:recurrence", self.registry()["sequential-recurrence"]["provides"])

    def test_a_captured_chunk_has_the_bits_of_the_step_paths_scan(self):
        if self.device() != "cuda":
            self.skipTest("captured graphs are CUDA's")
        from crit_plugins.prompt_graphs import PromptGraphs
        torch = self.torch
        for shape in SHAPES:
            for kernels in (True, False):
                if not kernels and shape == SHAPES[0]:
                    continue   # the torch formula's graph of a wide bucket is a fallback's, not worth its capture time
                with self.subTest(shape=shape, kernels=kernels):
                    step, graph = self.models(shape, exact_rows=4096, kernels=kernels)   # every step row: the scan
                    # A sequence with a state of its own before the captures: their warm-ups (padding alone) leave it.
                    early = len(CHUNKS)
                    for model in (step, graph):
                        model.eager(early, 0, [4, 1, 4, 1, 5, 9, 2, 6, 5, 3])
                    prompts = PromptGraphs(graph.forward, device="cuda", buckets=(16, 64), pools=[graph.pool])
                    self.assertEqual(prompts.prepare(), [16, 64])
                    self.assertSame(step, graph, [early])
                    calls = graph.calls
                    for sequence, chunks in enumerate(CHUNKS):
                        frontier = 0
                        for chunk in chunks:
                            expected = step.eager(sequence, frontier, chunk)
                            found = prompts.run(sequence, frontier, chunk)
                            self.assertIsNotNone(found)
                            self.assertTrue(torch.equal(found, expected), (sequence, frontier, len(chunk)))
                            frontier += len(chunk)
                    self.assertEqual(graph.calls, calls, "every chunk replayed a graph")
                    self.assertSame(step, graph, range(len(CHUNKS) + 1))
                    # Decoding continues alike after a captured chunk and after the step path.
                    rows = [(s, sum(len(c) for c in CHUNKS[s]), 1) for s in range(len(CHUNKS))]
                    for model in (step, graph):
                        batch = StepBatch.build(rows, self.device())
                        batch.tokens_t = torch.tensor([7] * len(rows), dtype=torch.int64, device=self.device())
                        model.decoded = model.forward(batch)
                        for s, _, _ in rows:
                            model.pool.accept(s, 1)
                    self.assertTrue(torch.equal(step.decoded, graph.decoded))
                    self.assertSame(step, graph, range(len(CHUNKS) + 1))

    def test_a_bucket_past_the_crossover_runs_the_chunked_kernel_with_padding_that_keeps_the_state(self):
        if self.device() != "cuda":
            self.skipTest("captured graphs are CUDA's")
        from crit_plugins.prompt_graphs import PromptGraphs
        torch = self.torch
        # exact_rows below the chunk: the bucket past it runs the chunked kernel, and so does the step path's chunk.
        for shape, kernels, bucket, count in ((SHAPES[0], True, 512, 300), (SHAPES[1], False, 64, 40)):
            with self.subTest(shape=shape, kernels=kernels):
                step, graph = self.models(shape, kernels=kernels, exact_rows=bucket // 4)
                self.assertEqual((graph.pool.scan_rows, graph.pool.scan_rows_captured), (bucket // 4, bucket // 4))
                early = len(CHUNKS)
                for model in (step, graph):   # a sequence's state before the captures, which padding alone keeps
                    model.eager(early, 0, [4, 1, 4, 1, 5, 9, 2, 6, 5, 3])
                prompts = PromptGraphs(graph.forward, device="cuda", buckets=(bucket,), pools=[graph.pool])
                self.assertEqual(prompts.prepare(), [bucket])
                self.assertSame(step, graph, [early])
                chunk = [(7 * i) % VOCAB for i in range(count)]
                for model in (step, graph):   # a pending decode row first: the chunk applies it before its kernel
                    batch = StepBatch.build([(0, 0, 2)], self.device())
                    batch.tokens_t = torch.tensor([3, 5], dtype=torch.int64, device=self.device())
                    model.forward(batch)
                    model.pool.accept(0, 2)
                expected = step.eager(0, 2, chunk)
                found = prompts.run(0, 2, chunk)
                torch.testing.assert_close(found, expected, **PROMPT)
                self.assertSame(step, graph, [0], exact=False)
                self.assertSame(step, graph, [early])

    def test_a_padded_chunk_run_eagerly_gives_the_step_paths_values(self):
        # On any device: the padded PromptBatch run eagerly (what a captured chunk replays) against the step path.
        from crit_plugins.prompt_graphs import PromptGraphs
        torch = self.torch
        step, graph = self.models(SHAPES[1], exact_rows=4096)
        prompts = PromptGraphs(graph.forward, device=self.device(), buckets=(16, 64), pools=[graph.pool])
        for sequence, chunks in enumerate(CHUNKS):
            frontier = 0
            for chunk in chunks:
                expected = step.eager(sequence, frontier, chunk)
                found = prompts.run_eager(sequence, frontier, chunk, 16 if len(chunk) <= 16 else 64)
                if self.device() == "cuda":
                    self.assertTrue(torch.equal(found, expected), (sequence, frontier, len(chunk)))
                else:   # a CPU's vector math rounds a tensor's tail elements its own way: the convolution's silu
                    torch.testing.assert_close(found, expected, atol=1e-5, rtol=1e-5)
                frontier += len(chunk)
        bits = self.device() == "cuda"
        for sequence in range(len(CHUNKS)):
            for layer in range(LAYERS):
                torch.testing.assert_close(graph.pool.state(layer, sequence), step.pool.state(layer, sequence),
                                           atol=0 if bits else 1e-5, rtol=0 if bits else 1e-5)
                self.assertTrue(torch.equal(graph.pool.window(layer, sequence), step.pool.window(layer, sequence)))
