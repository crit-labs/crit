"""recurrent-state-recovery's decode option chunk (code/chunk_states.py): every row through recurrent-scans'
chunk-aligned exact form, so a prompt row has the bits of decoding its tokens one by one, verify windows keep their
accepted prefix with those bits (step accepts, captured rounds committed at any count, eager and in a CUDA graph),
captured prompt chunks continue a sequence as an eager prompt row does (padding alone changes nothing), the outputs
follow the float64 recurrence, and the convolution window is the scan form's."""
from crit_plugins import StepBatch
from crit_plugins.device import PromptBatch, RoundBatch
from crit_plugins.testing import PluginTestCase

PROMPT = dict(atol=2e-3, rtol=2e-2)


class View:
    def __init__(self, rows, width, n_emitted):
        self.rows, self.width, self.n_emitted = rows, width, n_emitted


class ChunkTests(PluginTestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.dev = self.device()
        if self.dev != "cuda":
            self.skipTest("the chunk form runs on recurrent-scans' CUDA kernel")
        g = torch.Generator().manual_seed(53)
        self.kh, self.h, self.dk, self.dv, T = 2, 6, 64, 64, 200
        self.inputs = {"q": torch.randn(T, self.kh, self.dk, generator=g), "k": torch.randn(T, self.kh, self.dk, generator=g),
                       "v": torch.randn(T, self.h, self.dv, generator=g), "g": -torch.rand(T, self.h, generator=g) * 0.5,
                       "b": torch.rand(T, self.h, generator=g), "x": torch.randn(T, 24, generator=g)}
        self.inputs = {k: v.to(self.dev) for k, v in self.inputs.items()}
        self.weight = torch.randn(24, 4, generator=g).to(self.dev)
        self.geometry = {"keys": self.dk, "values": self.dv, "epsilon": 1e-6, "query_scale": self.dk ** -0.5,
                         "normalize_keys": True, "normalize_queries": True}
        self.plugin = self.assemble(options={"decode": "chunk"})["recurrence"]
        if self.plugin.kernels(self.dev, self.dk) is None:
            self.skipTest("recurrent-scans did not build here")

    def pool(self, widest=0, plugin=None):
        pool = (plugin or self.plugin).pool([0], self.h, self.dk, self.dv, self.dev, 4, widest, (24, 4), self.kh, 1)
        pool.open("a")
        return pool

    def rows(self, pool, batch, first, count, accept=None, plugin=None):
        plugin = plugin or self.plugin
        span = slice(first, first + count)
        x = plugin.convolve(0, self.inputs["x"][span], self.weight, None, pool, batch)
        out = plugin.forward(0, *(self.inputs[n][span] for n in "qkvgb"), pool, batch, self.geometry)
        if not batch.round and not batch.prompt:
            pool.accept("a", count if accept is None else accept)
        return out.clone(), x.clone()

    def step(self, pool, first, count, accept=None, plugin=None):
        return self.rows(pool, StepBatch.build([("a", first, count)], self.dev), first, count, accept, plugin)

    def decoded(self, n):
        """Outputs, windows' outputs and the state of decoding tokens 0 .. n - 1 one by one."""
        pool = self.pool()
        outs, xs = zip(*(self.step(pool, t, 1) for t in range(n)))
        return self.torch.cat(outs), self.torch.cat(xs), pool.state(0, "a").clone()

    def test_prompt_rows_have_the_bits_of_decoding_their_tokens(self):
        torch = self.torch
        out, x, state = self.decoded(170)
        for cuts in ((170,), (40, 30, 100), (64, 1, 105), (63, 2, 105)):
            with self.subTest(cuts=cuts):
                pool, first, found = self.pool(), 0, []
                for n in cuts:
                    found.append(self.step(pool, first, n)[0])
                    first += n
                self.assertTrue(torch.equal(torch.cat(found), out))
                self.assertTrue(torch.equal(pool.state(0, "a"), state))

    def test_verify_windows_keep_the_accepted_prefix_with_the_decode_bits(self):
        torch = self.torch
        out, _, state = self.decoded(90)
        pool = self.pool(widest=3)
        got, first = [self.step(pool, 0, 10)[0]], 10
        for kept in (2, 4, 0, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1):   # windows across 64
            if first + 4 > 90:
                break
            window, _ = self.step(pool, first, 4, accept=kept)
            self.assertTrue(torch.equal(window[:kept], out[first:first + kept]), (first, kept))
            got.append(window[:kept])
            first += kept
        rest = [self.step(pool, t, 1)[0] for t in range(first, 90)]
        self.assertTrue(torch.equal(torch.cat(got + rest), out))
        self.assertTrue(torch.equal(pool.state(0, "a"), state))

    def test_captured_rounds_commit_any_count_eagerly_and_in_a_graph(self):
        torch = self.torch
        width = 3
        out, x, state = self.decoded(80)
        for graph in (False, True):
            for n in range(width + 2):
                with self.subTest(graph=graph, n=n):
                    pool = self.pool(widest=width)
                    self.step(pool, 0, 61)   # a round across the chunk's end
                    pool.bind(0, "a")
                    batch = RoundBatch(1, width, self.dev)
                    frontier = torch.tensor([61], device=self.dev)
                    columns = torch.zeros(1, width + 1, dtype=torch.int64, device=self.dev)
                    count = torch.tensor([n], device=self.dev)
                    q, k, v, g, b = (self.inputs[c][61:61 + width + 1] for c in "qkvgb")
                    xs = self.inputs["x"][61:61 + width + 1]

                    def round_():
                        batch.fill(1, width, frontier, columns)
                        xo = self.plugin.convolve(0, xs, self.weight, None, pool, batch)
                        o = self.plugin.forward(0, q, k, v, g, b, pool, batch, self.geometry)
                        pool.commit_round(View(1, width, count))
                        return o, xo
                    if graph:
                        side = torch.cuda.Stream()
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):   # the capture's warm-up commits nothing: unbound
                            pool.unbind(0)
                            round_()
                            pool.bind(0, "a")
                        torch.cuda.current_stream().wait_stream(side)
                        captured = torch.cuda.CUDAGraph()
                        pool.unbind(0)
                        with torch.cuda.graph(captured):
                            o, xo = round_()
                        pool.bind(0, "a")
                        captured.replay()
                    else:
                        o, xo = round_()
                    pool.unbind(0)
                    self.assertTrue(torch.equal(o[:n], out[61:61 + n]))
                    self.assertTrue(torch.equal(xo[:n], x[61:61 + n]))
                    rest = torch.cat([self.step(pool, t, 1)[0] for t in range(61 + n, 80)])
                    self.assertTrue(torch.equal(rest, out[61 + n:80]))
                    self.assertTrue(torch.equal(pool.state(0, "a"), state))

    def test_captured_prompt_chunks_continue_as_an_eager_prompt_row(self):
        torch = self.torch
        out, x, state = self.decoded(100)
        pool = self.pool()
        batch = PromptBatch(64, self.dev)
        pool.stage_prompt(batch.fill(32, None, 0, []))   # a capture's warm-up: padding alone changes nothing
        self.rows(pool, batch, 0, 32)
        pool.commit_prompt(batch)
        self.assertTrue(torch.equal(pool.state(0, "a"), torch.zeros_like(state)))
        found, first = [], 0
        for count, bucket in ((20, 32), (50, 64), (30, 32)):   # the second completes the first chunk
            pool.stage_prompt(batch.fill(bucket, "a", first, list(range(count))))
            span = slice(first, first + bucket)
            xo = self.plugin.convolve(0, self.inputs["x"][span], self.weight, None, pool, batch)
            o = self.plugin.forward(0, *(self.inputs[n][span] for n in "qkvgb"), pool, batch, self.geometry)
            pool.commit_prompt(batch)
            self.assertTrue(torch.equal(o[count:], torch.zeros_like(o[count:])))
            found.append((o[:count].clone(), xo[:count].clone()))
            first += count
        self.assertTrue(torch.equal(torch.cat([f[0] for f in found]), out))
        self.assertTrue(torch.equal(torch.cat([f[1] for f in found]), x))
        self.assertTrue(torch.equal(pool.state(0, "a"), state))

    def test_outputs_follow_the_float64_recurrence_and_the_facts_say_exact(self):
        torch = self.torch
        pool = self.pool()
        out, _ = self.step(pool, 0, 150)
        d = {n: t[:150].double() for n, t in self.inputs.items()}
        normal = lambda x: x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)  # noqa: E731
        S = torch.zeros(self.h, self.dk, self.dv, dtype=torch.float64, device=self.dev)
        ratio, ref = self.h // self.kh, []
        for t in range(150):
            k = normal(d["k"][t]).repeat_interleave(ratio, 0)
            q = normal(d["q"][t]).repeat_interleave(ratio, 0) * self.dk ** -0.5
            e = torch.exp(d["g"][t])
            delta = d["b"][t][:, None] * (d["v"][t] - e[:, None] * torch.einsum("hk,hkv->hv", k, S))
            S = e[:, None, None] * S + k[:, :, None] * delta[:, None, :]
            ref.append(torch.einsum("hk,hkv->hv", q, S))
        torch.testing.assert_close(out.double(), torch.stack(ref), **PROMPT)
        torch.testing.assert_close(pool.state(0, "a").double(), S, **PROMPT)
        self.assertEqual(self.plugin.exactness(), {"decode": "row-exact", "prompt": "exact"})
        facts = self.plugin.facts()["recurrence_prompts"]["pools"][-1]
        self.assertEqual((facts["decode"], facts["chunked"]), ("chunk", "chunk_rows"))

    def test_fork_and_the_window_follow_the_sequence(self):
        torch = self.torch
        pool = self.pool()
        self.step(pool, 0, 70)
        pool.fork("a", "b")
        self.assertTrue(torch.equal(pool.state(0, "b"), pool.state(0, "a")))
        self.assertTrue(torch.equal(pool.window(0, "b"), pool.window(0, "a")))
        a = self.step(pool, 70, 1)[0]
        batch = StepBatch.build([("b", 70, 1)], self.dev)
        b = self.plugin.forward(0, *(self.inputs[n][70:71] for n in "qkvgb"), pool, batch, self.geometry)
        self.assertTrue(torch.equal(a, b))
