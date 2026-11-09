"""sequential-recurrence's device form: a step's rows continue their sequences as the host form and a float64
recurrence do (strong decays included); a decode or verify row's bits do not depend on the rows beside it; a captured
round's verify window, rolled back to any accepted count, continues exactly as the serial decode (state and convolution
window), by every commit strategy; rounds capture and replay; prompt chunks (fla's kernel where installed, on its
stated tiles) agree within the stated tolerance. On CUDA the rows run recurrent-scans where the kernel registry has it
(LOCAL_KERNELS for a local build) and the torch formula otherwise; both keep every exact promise.

Tolerances: decode and verify rows (the FP32 column scan) against float64 within 1e-5 + 1e-4 |x|; prompt rows through a
chunked kernel within 2e-3 + 2e-2 |x| (fla in FP32 or the two-pass prefill: another summation order)."""
import random
import sys

from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch
from crit_plugins.testing import PluginTestCase

GEOMETRY = {"keys": 4, "values": 3, "epsilon": 1e-6, "query_scale": 0.5, "normalize_keys": True,
            "normalize_queries": True}
HEADS = 2
SCAN = dict(atol=1e-5, rtol=1e-4)
PROMPT = dict(atol=2e-3, rtol=2e-2)
COMMITS = ("deferred", "checkpoints", "recompute")


class View:
    """The part of a round's view a pool's commit reads."""

    def __init__(self, rows, width, n_emitted):
        self.rows, self.width, self.n_emitted = rows, width, n_emitted


class Rig:
    """Inputs of each sequence (one row per position), and the rows of a step or a round over them."""

    def __init__(self, torch, device, key_heads, heads, keys, values, channels=0, width=4, tokens=40, seed=0,
                 strong=False, dtype=None, epsilon=1e-6, normalize=(True, True), dilation=1):
        g = torch.Generator().manual_seed(seed)
        self.torch, self.device, self.dilation = torch, device, dilation
        self.key_heads, self.heads, self.keys, self.values, self.channels, self.width = (key_heads, heads, keys,
                                                                                        values, channels, width)
        dtype = dtype or torch.float32

        def inputs():
            log_decay = -(torch.rand(tokens, heads, generator=g) * 9 + 3) if strong else \
                -torch.rand(tokens, heads, generator=g) * 0.5
            found = {"q": torch.randn(tokens, key_heads, keys, generator=g).to(dtype),
                     "k": torch.randn(tokens, key_heads, keys, generator=g).to(dtype),
                     "v": torch.randn(tokens, heads, values, generator=g).to(dtype),
                     "g": log_decay, "b": torch.rand(tokens, heads, generator=g)}
            if channels:
                found["x"] = torch.randn(tokens, channels, generator=g).to(dtype)
            return {name: t.to(device) for name, t in found.items()}
        self.seq = {s: inputs() for s in "abcd"}
        self.geometry = {"keys": keys, "values": values, "epsilon": epsilon, "query_scale": keys ** -0.5,
                         "normalize_keys": normalize[0], "normalize_queries": normalize[1]}
        if not normalize[0]:   # keys of norm 1/2 keep the state bounded without the normalization
            for found in self.seq.values():
                k = found["k"].float()
                found["k"] = (k / (2 * k.norm(dim=-1, keepdim=True))).to(found["k"].dtype)
        if channels:
            self.weight = torch.randn(channels, width, generator=g).to(device)
            self.bias = torch.randn(channels, generator=g).to(device)

    def pool(self, recurrence, widest=0, sequences=6, rows=2):
        conv = (self.channels, self.width, self.dilation) if self.channels else None
        return recurrence.pool([0, 1], self.heads, self.keys, self.values, self.device, sequences, widest, conv,
                               self.key_heads, rows)

    def cat(self, parts, name):
        return self.torch.cat([self.seq[s][name][p:p + n] for s, p, n in parts])

    def step(self, recurrence, pool, parts, accept=None, layers=(0, 1)):
        """A StepBatch of rows (sequence, first position, tokens) through every layer; accept: the counts to commit
        (all of each row by default; None in the list skips that row's accept). Returns {layer: (outputs, conv)}."""
        batch = StepBatch.build(parts, self.device)
        found = {}
        for layer in layers:
            conv = None
            if self.channels:
                conv = recurrence.convolve(layer, self.cat(parts, "x"), self.weight, self.bias, pool, batch)
            out = recurrence.forward(layer, self.cat(parts, "q"), self.cat(parts, "k"), self.cat(parts, "v"),
                                     self.cat(parts, "g"), self.cat(parts, "b"), pool, batch, self.geometry)
            found[layer] = (out, conv)
        counts = accept if accept is not None else [n for _, _, n in parts]
        for (s, _, _), count in zip(parts, counts):
            if count is not None:
                pool.accept(s, count)
        return found

    def round_inputs(self, parts, width):
        """A round's inputs: rows (sequence, first position) of width + 1 columns."""
        return {name: self.cat([(s, p, width + 1) for s, p in parts], name) for name in self.seq["a"]}

    def round(self, recurrence, pool, batch, inputs, layers=(0, 1)):
        found = {}
        for layer in layers:
            conv = None
            if self.channels:
                conv = recurrence.convolve(layer, inputs["x"], self.weight, self.bias, pool, batch)
            out = recurrence.forward(layer, inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["b"], pool,
                                     batch, self.geometry)
            found[layer] = (out, conv)
        return found

    def round_batch(self, rows, width, frontier):
        torch = self.torch
        batch = RoundBatch(rows, width, self.device)
        return batch.fill(rows, width, torch.tensor(frontier, dtype=torch.int64, device=self.device),
                          torch.zeros(rows, width + 1, dtype=torch.int64, device=self.device))

    def reference(self, s, first, count, state=None):
        """float64 (outputs [count][heads][values], state [heads][keys][values]) from `state` (zeros)."""
        torch = self.torch
        d = {name: t[first:first + count].double() for name, t in self.seq[s].items()}
        ratio = self.heads // self.key_heads
        S = torch.zeros(self.heads, self.keys, self.values, dtype=torch.float64, device=self.device) \
            if state is None else state.double().clone()
        eps = self.geometry["epsilon"]
        normal = lambda x, on: x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps) if on else x  # noqa: E731
        out = []
        for t in range(count):
            k = normal(d["k"][t], self.geometry["normalize_keys"]).repeat_interleave(ratio, 0)
            q = normal(d["q"][t], self.geometry["normalize_queries"]).repeat_interleave(ratio, 0) * self.keys ** -0.5
            e = torch.exp(d["g"][t])
            delta = d["b"][t][:, None] * (d["v"][t] - e[:, None] * torch.einsum("hk,hkv->hv", k, S))
            S = e[:, None, None] * S + k[:, :, None] * delta[:, None, :]
            out.append(torch.einsum("hk,hkv->hv", q, S))
        return torch.stack(out), S

    def conv_reference(self, s, first, count, window):
        """FP64 (outputs [count][channels], the window after) of the rig's convolution at its dilation from `window`
        ([channels][(width - 1) * dilation], oldest first): tap i of token t reads x[t - (width - 1 - i) * dilation]."""
        torch = self.torch
        x = self.seq[s]["x"][first:first + count].double()
        padded = torch.cat([window.T.double(), x])
        w, d = self.weight.double(), self.dilation
        out = torch.stack([sum(padded[t + i * d] * w[:, i] for i in range(self.width)) + self.bias.double()
                           for t in range(count)])
        return torch.nn.functional.silu(out), padded[-(self.width - 1) * d:].T


class DeviceRecurrenceTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        self.recurrence = self.assemble(options={"exact_rows": 4})["recurrence"]
        rng = random.Random(29)
        # Per sequence and head, a list of host steps; the same values as tensors [tokens][heads][...].
        self.steps = {s: [[{"query": [rng.uniform(-1, 1) for _ in range(4)], "key": [rng.uniform(-1, 1) for _ in range(4)],
                             "value": [rng.uniform(-1, 1) for _ in range(3)], "log_decay": rng.uniform(-0.4, -0.01),
                             "beta": rng.uniform(0.1, 0.9)} for _ in range(HEADS)] for _ in range(80)] for s in "ab"}

    def plugins(self, **options):
        """The plugin, and on CUDA with the registry kernel also with the torch formula: [(name, plugin)]. Prompt rows
        past 16 tokens run the chunked kernel unless the options say otherwise (no crossover measured per test shape)."""
        options = {"exact_rows": 16, **options}
        found = [("default", self.assemble(options=options)["recurrence"])]
        plugin = found[0][1]
        if self.device() == "cuda" and plugin.kernels(self.device(), 128) is not None:
            found.append(("torch", self.assemble(options={**options, "kernels": False})["recurrence"]))
        return found

    # -- the host form's cases -------------------------------------------------------------------------------------------
    def tensors(self, sequence, first, count):
        rows = self.steps[sequence][first:first + count]
        t = lambda name: self.torch.tensor([[h[name] for h in row] for row in rows], dtype=self.torch.float32)  # noqa: E731
        return [t(name).to(self.device()) for name in ("query", "key", "value", "log_decay", "beta")]

    def run_rows(self, pool, rows):
        batch = StepBatch.build(rows, self.device())
        parts = [self.tensors(s, p, n) for s, p, n in rows]
        q, k, v, g, b = (self.torch.cat([part[i] for part in parts]) for i in range(5))
        return batch, self.recurrence.forward(0, q, k, v, g, b, pool, batch, GEOMETRY).cpu()

    def host(self, sequence, first, count, state=None):
        """The host form over the same tokens, per head: (outputs [count][heads][values], states)."""
        outs, states = [], []
        for head in range(HEADS):
            begin = state[head] if state else (0.0,) * 12
            o, s = self.recurrence.prefill(begin, [row[head] for row in self.steps[sequence][first:first + count]],
                                           GEOMETRY)
            outs.append(o)
            states.append(s)
        return [[outs[h][t] for h in range(HEADS)] for t in range(count)], states

    def test_rows_of_two_sequences_continue_their_states_as_the_host_form(self):
        pool = self.recurrence.pool([0], HEADS, 4, 3, self.device(), widest=3)
        pool.open("a")
        pool.open("b")
        batch, out = self.run_rows(pool, [("a", 0, 3), ("b", 0, 2)])
        pool.accept("a", 3)
        pool.accept("b", 2)
        expected_a, state_a = self.host("a", 0, 3)
        expected_b, _ = self.host("b", 0, 2)
        self.assertClose(out[0:3].tolist(), expected_a, **SCAN)
        self.assertClose(out[3:5].tolist(), expected_b, **SCAN)
        # The next step continues from the committed state.
        _, out = self.run_rows(pool, [("a", 3, 2)])
        expected, _ = self.host("a", 3, 2, state_a)
        self.assertClose(out.tolist(), expected, **SCAN)

    def test_accept_commits_a_verify_windows_accepted_tokens_only(self):
        pool = self.recurrence.pool([0], HEADS, 4, 3, self.device(), widest=3)
        pool.open("a")
        self.run_rows(pool, [("a", 0, 3)])
        pool.accept("a", 3)
        self.run_rows(pool, [("a", 3, 4)])   # a pending token and three drafts
        pool.accept("a", 2)
        _, state = self.host("a", 0, 5)
        found = pool.state(0, "a").cpu()
        for head in range(HEADS):
            self.assertClose(found[head].flatten().tolist(), list(state[head]), **SCAN)
        with self.assertRaisesRegex(ValueError, "held 4 tokens, not 5"):
            self.run_rows(pool, [("a", 5, 4)])
            pool.accept("a", 5)

    def test_a_prompt_chunk_agrees_with_the_host_form(self):
        pool = self.recurrence.pool([0], HEADS, 4, 3, self.device())
        pool.open("a")
        _, out = self.run_rows(pool, [("a", 0, 70)])   # past exact_rows: fla's chunked kernel where installed
        pool.accept("a", 70)
        expected, state = self.host("a", 0, 70)
        self.assertClose(out.tolist(), expected, **PROMPT)
        found = pool.state(0, "a").cpu()
        for head in range(HEADS):
            self.assertClose(found[head].flatten().tolist(), list(state[head]), **PROMPT)

    # -- the float64 recurrence, grouped keys, strong decays -----------------------------------------------------------
    def shapes(self):
        """(key heads, heads, keys, values): grouped keys at shapes the scan covers by masking, and on CUDA wider ones."""
        found = [(1, 2, 4, 3), (2, 4, 16, 8)]
        if self.device() == "cuda":
            found += [(4, 8, 128, 128), (1, 2, 100, 40)]
        return found

    def test_decode_and_verify_rows_follow_the_float64_recurrence(self):
        torch = self.torch
        for name, plugin in self.plugins():
            for shape in self.shapes():
                for strong in (False, True):
                    with self.subTest(kernel=name, shape=shape, strong=strong):
                        rig = Rig(torch, self.device(), *shape, channels=24, tokens=12, seed=3, strong=strong)
                        pool = rig.pool(plugin, widest=3)
                        pool.open("a")
                        pool.open("b")
                        found = rig.step(plugin, pool, [("a", 0, 1), ("b", 0, 4)])
                        found2 = rig.step(plugin, pool, [("a", 1, 4), ("b", 4, 1)], accept=[2, 1])
                        out_a = torch.cat([found[0][0][0:1], found2[0][0][0:2]])
                        ref_a, state_a = rig.reference("a", 0, 3)
                        torch.testing.assert_close(out_a.double(), ref_a, **SCAN)
                        torch.testing.assert_close(pool.state(0, "a").double(), state_a, **SCAN)
                        ref_b, state_b = rig.reference("b", 0, 5)
                        torch.testing.assert_close(torch.cat([found[0][0][1:5], found2[0][0][4:5]]).double(), ref_b, **SCAN)
                        torch.testing.assert_close(pool.state(1, "b").double(), state_b, **SCAN)
                        conv, window = rig.conv_reference("a", 0, 3, torch.zeros(24, 3, device=self.device()))
                        torch.testing.assert_close(torch.cat([found[1][1][0:1], found2[1][1][0:2]]).double(), conv,
                                                   atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(pool.window(1, "a").double(), window, atol=0, rtol=0)

    def test_prompt_rows_follow_the_float64_recurrence_within_the_stated_tolerance(self):
        torch = self.torch
        for name, plugin in self.plugins(exact_rows=8):
            for shape in self.shapes()[2 if self.device() == "cuda" else 1:][:1]:   # one shape: fla tunes per shape
                for strong in (False, True):
                    with self.subTest(kernel=name, shape=shape, strong=strong):
                        rig = Rig(torch, self.device(), *shape, channels=40, tokens=200, seed=5, strong=strong)
                        pool = rig.pool(plugin)
                        pool.open("a")
                        found = rig.step(plugin, pool, [("a", 0, 150)])
                        ref, state = rig.reference("a", 0, 150)
                        torch.testing.assert_close(found[0][0].double(), ref, **PROMPT)
                        torch.testing.assert_close(pool.state(0, "a").double(), state, **PROMPT)
                        conv, window = rig.conv_reference("a", 0, 150, torch.zeros(40, 3, device=self.device()))
                        torch.testing.assert_close(found[0][1].double(), conv, atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(pool.window(0, "a").double(), window, atol=0, rtol=0)
                        # and the decode rows after it continue from the prompt's state
                        after = rig.step(plugin, pool, [("a", 150, 1)])
                        ref, _ = rig.reference("a", 150, 1, state)
                        torch.testing.assert_close(after[0][0].double(), ref, **PROMPT)

    def test_a_prompt_row_through_the_scan_has_the_bits_of_the_serial_decode(self):
        torch = self.torch
        for name, plugin in self.plugins(exact_rows=64):
            with self.subTest(kernel=name):
                rig = Rig(torch, self.device(), *self.shapes()[-1], channels=16, tokens=30, seed=7)
                whole, serial = rig.pool(plugin), rig.pool(plugin)
                whole.open("a")
                serial.open("a")
                prompt = rig.step(plugin, whole, [("a", 0, 20)])
                tokens = [rig.step(plugin, serial, [("a", t, 1)]) for t in range(20)]
                for layer in (0, 1):
                    self.assertTrue(torch.equal(prompt[layer][0], torch.cat([t[layer][0] for t in tokens])))
                    conv = torch.cat([t[layer][1] for t in tokens])
                    if self.device() == "cuda":
                        self.assertTrue(torch.equal(prompt[layer][1], conv))
                    else:   # a CPU's vector math rounds a tensor's tail elements its own way: a prompt row's silu
                        torch.testing.assert_close(prompt[layer][1], conv, atol=1e-6, rtol=1e-6)
                    self.assertTrue(torch.equal(whole.state(layer, "a"), serial.state(layer, "a")))
                    self.assertTrue(torch.equal(whole.window(layer, "a"), serial.window(layer, "a")))

    # -- exact promises --------------------------------------------------------------------------------------------------
    def test_a_decode_or_verify_row_has_the_bits_it_has_alone(self):
        torch = self.torch
        for name, plugin in self.plugins():
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(kernel=name, dtype=dtype):
                    rig = Rig(torch, self.device(), *self.shapes()[-1], channels=48, tokens=12, seed=9, dtype=dtype)
                    together = rig.pool(plugin, widest=3)
                    for s in "abcd":
                        together.open(s)
                    parts = [("a", 0, 1), ("b", 0, 4), ("c", 0, 9), ("d", 0, 3)]   # c: a prompt row beside them
                    batched = rig.step(plugin, together, parts)
                    first = 0
                    for s, p, n in parts:
                        alone = rig.pool(plugin, widest=3)
                        alone.open(s)
                        found = rig.step(plugin, alone, [(s, p, n)])
                        for layer in (0, 1):
                            self.assertTrue(torch.equal(found[layer][0], batched[layer][0][first:first + n]), (s, layer))
                            self.assertTrue(torch.equal(found[layer][1], batched[layer][1][first:first + n]), (s, layer))
                        first += n

    def rolled_back(self, plugin, rig, width, n, sequences="ab"):
        """Two tokens, a verify window of width + 1 columns of which n are kept, then three tokens, per sequence, beside
        each other in one round; returns the pool and the outputs of the kept columns and the tokens after."""
        torch = self.torch
        pool = rig.pool(plugin, widest=width)
        for s in sequences:
            pool.open(s)
        for t in range(2):
            rig.step(plugin, pool, [(s, t, 1) for s in sequences])
        for r, s in enumerate(sequences):
            pool.bind(r, s)
        batch = rig.round_batch(len(sequences), width, [2] * len(sequences))
        verified = rig.round(plugin, pool, batch, rig.round_inputs([(s, 2) for s in sequences], width))
        counts = torch.tensor([n, max(n - 1, 0)][:len(sequences)], dtype=torch.int64, device=self.device())
        pool.commit_round(View(len(sequences), width, counts))
        for r in range(len(sequences)):
            pool.unbind(r)
        after = [rig.step(plugin, pool, [(s, 2 + int(c) + t, 1) for s, c in zip(sequences, counts.tolist())])
                 for t in range(3)]
        return pool, verified, after, counts.tolist()

    def test_every_accepted_count_rolls_back_to_the_serial_decode(self):
        torch = self.torch
        width = 3
        for commit in COMMITS:
            for name, plugin in self.plugins(commit=commit):
                for n in range(width + 2):
                    with self.subTest(commit=commit, kernel=name, n=n):
                        rig = Rig(torch, self.device(), *self.shapes()[-1], channels=40, tokens=12, seed=11)
                        pool, verified, after, counts = self.rolled_back(plugin, rig, width, n)
                        for r, (s, kept) in enumerate(zip("ab", counts)):
                            serial = rig.pool(plugin)
                            serial.open(s)
                            steps = [rig.step(plugin, serial, [(s, t, 1)]) for t in range(2 + kept + 3)]
                            for layer in (0, 1):
                                columns = slice(r * (width + 1), r * (width + 1) + kept)
                                expected = torch.cat([steps[2 + c][layer][0] for c in range(kept)]) if kept else None
                                if kept:
                                    self.assertTrue(torch.equal(verified[layer][0][columns], expected))
                                    conv = torch.cat([steps[2 + c][layer][1] for c in range(kept)])
                                    self.assertTrue(torch.equal(verified[layer][1][columns], conv))
                                for t in range(3):
                                    self.assertTrue(torch.equal(after[t][layer][0][r:r + 1], steps[2 + kept + t][layer][0]))
                                    self.assertTrue(torch.equal(after[t][layer][1][r:r + 1], steps[2 + kept + t][layer][1]))
                                self.assertTrue(torch.equal(pool.state(layer, s), serial.state(layer, s)))
                                self.assertTrue(torch.equal(pool.window(layer, s), serial.window(layer, s)))

    def test_a_round_commit_runs_again_with_a_smaller_count(self):
        torch = self.torch
        for commit in COMMITS:
            for name, plugin in self.plugins(commit=commit):
                with self.subTest(commit=commit, kernel=name):
                    rig = Rig(torch, self.device(), *self.shapes()[-1], channels=40, tokens=12, seed=13)
                    pools = []
                    for counts in ([[3]], [[4], [1]], [[1]]):
                        pool = rig.pool(plugin, widest=4)
                        pool.open("a")
                        pool.bind(0, "a")
                        batch = rig.round_batch(1, 4, [0])
                        rig.round(plugin, pool, batch, rig.round_inputs([("a", 0)], 4))
                        for count in counts:   # a truncation commits again before the next row
                            pool.commit_round(View(1, 4, torch.tensor(count, device=self.device())))
                        pools.append(pool)
                    for layer in (0, 1):
                        self.assertTrue(torch.equal(pools[1].state(layer, "a"), pools[2].state(layer, "a")))
                        self.assertTrue(torch.equal(pools[1].window(layer, "a"), pools[2].window(layer, "a")))
                        self.assertFalse(torch.equal(pools[0].state(layer, "a"), pools[2].state(layer, "a")))

    def test_an_unbound_round_row_changes_no_sequence(self):
        torch = self.torch
        for commit in COMMITS:
            for name, plugin in self.plugins(commit=commit):
                with self.subTest(commit=commit, kernel=name):
                    rig = Rig(torch, self.device(), *self.shapes()[-1], channels=40, tokens=12, seed=17)
                    pool = rig.pool(plugin, widest=2)
                    for s in "ab":
                        pool.open(s)
                    rig.step(plugin, pool, [("a", 0, 2), ("b", 0, 2)])
                    before = {s: [pool.state(layer, s).clone() for layer in (0, 1)] for s in "ab"}
                    pool.bind(1, "b")   # slot 0 holds no sequence
                    batch = rig.round_batch(2, 2, [2, 2])
                    found = rig.round(plugin, pool, batch, rig.round_inputs([("a", 2), ("b", 2)], 2))
                    pool.commit_round(View(2, 2, torch.tensor([3, 0], device=self.device())))
                    for layer in (0, 1):
                        self.assertTrue(torch.equal(pool.state(layer, "a"), before["a"][layer]))
                        self.assertTrue(torch.equal(pool.state(layer, "b"), before["b"][layer]))
                        out, conv = found[layer]   # the unbound row's columns are zero, the bound row's finite
                        self.assertTrue(torch.equal(out[:3], torch.zeros_like(out[:3])))
                        self.assertTrue(torch.equal(conv[:3], torch.zeros_like(conv[:3])))
                        self.assertTrue(bool(torch.isfinite(out[3:]).all()) and bool(torch.isfinite(conv[3:]).all()))

    def test_a_captured_round_replays_as_it_runs(self):
        torch = self.torch
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        width, rows = 3, 2
        for commit in COMMITS:
            for name, plugin in self.plugins(commit=commit):
                with self.subTest(commit=commit, kernel=name):
                    rig = Rig(torch, self.device(), *self.shapes()[-1], channels=40, tokens=40, seed=19)
                    graphed, eager = rig.pool(plugin, widest=width), rig.pool(plugin, widest=width)
                    for pool in (graphed, eager):
                        for r, s in enumerate("ab"):
                            pool.open(s)
                            pool.bind(r, s)
                    shape = rig.round_inputs([("a", 0), ("b", 0)], width)
                    inputs = {k: torch.zeros_like(v) for k, v in shape.items()}
                    counts = torch.zeros(rows, dtype=torch.int64, device=self.device())
                    frontier = torch.zeros(rows, dtype=torch.int64, device=self.device())
                    columns = torch.zeros(rows, width + 1, dtype=torch.int64, device=self.device())
                    batch = RoundBatch(rows, width, self.device())
                    found = {}

                    def round_(pool, outs):
                        batch.fill(rows, width, frontier, columns)
                        outs.update(rig.round(plugin, pool, batch, inputs))
                        pool.commit_round(View(rows, width, counts))
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        round_(graphed, {})   # warm-up: no rows commit (counts 0)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        round_(graphed, found)
                    position = {"a": 0, "b": 0}
                    for kept in ([2, 4], [0, 1], [4, 3]):
                        new = rig.round_inputs([("a", position["a"]), ("b", position["b"])], width)
                        for k in inputs:
                            inputs[k].copy_(new[k])
                        counts.copy_(torch.tensor(kept, device=self.device()))
                        graph.replay()
                        expected = {}
                        round_(eager, expected)
                        torch.cuda.synchronize()
                        for layer in (0, 1):
                            self.assertTrue(torch.equal(found[layer][0], expected[layer][0]))
                            self.assertTrue(torch.equal(found[layer][1], expected[layer][1]))
                        for s, n in zip("ab", kept):
                            position[s] += n
                    for layer in (0, 1):
                        for s in "ab":
                            self.assertTrue(torch.equal(graphed.state(layer, s), eager.state(layer, s)))
                            self.assertTrue(torch.equal(graphed.window(layer, s), eager.window(layer, s)))

    # -- sequences -------------------------------------------------------------------------------------------------------
    def test_a_fork_continues_as_its_parent_and_a_closed_sequence_starts_empty(self):
        torch = self.torch
        for name, plugin in self.plugins():
            with self.subTest(kernel=name):
                rig = Rig(torch, self.device(), *self.shapes()[-1], channels=16, tokens=12, seed=23)
                pool = rig.pool(plugin, widest=3)
                pool.open("a")
                rig.step(plugin, pool, [("a", 0, 4)], accept=[3])   # a pending deferred commit
                pool.fork("a", "b")
                for layer in (0, 1):
                    self.assertTrue(torch.equal(pool.state(layer, "a"), pool.state(layer, "b")))
                    self.assertTrue(torch.equal(pool.window(layer, "a"), pool.window(layer, "b")))
                rig.seq["b"] = rig.seq["a"]
                a = rig.step(plugin, pool, [("a", 3, 1)])
                b = rig.step(plugin, pool, [("b", 3, 1)])
                for layer in (0, 1):
                    self.assertTrue(torch.equal(a[layer][0], b[layer][0]))
                pool.close("b")
                pool.open("c")
                for layer in (0, 1):
                    self.assertEqual(int(torch.count_nonzero(pool.state(layer, "c"))), 0)
                    self.assertEqual(int(torch.count_nonzero(pool.window(layer, "c"))), 0)

    def test_accept_keeps_a_prompt_row_whole_or_drops_it(self):
        torch = self.torch
        rig = Rig(torch, self.device(), 1, 2, 4, 3, channels=8, tokens=20, seed=29)
        plugin = self.recurrence
        pool = rig.pool(plugin, widest=1)
        pool.open("a")
        rig.step(plugin, pool, [("a", 0, 6)], accept=[0])
        for layer in (0, 1):
            self.assertEqual(int(torch.count_nonzero(pool.state(layer, "a"))), 0)
            self.assertEqual(int(torch.count_nonzero(pool.window(layer, "a"))), 0)
        with self.assertRaisesRegex(ValueError, "keeps its final state only"):
            rig.step(plugin, pool, [("a", 0, 6)], accept=[4])
        with self.assertRaisesRegex(ValueError, "not open"):
            rig.step(plugin, pool, [("d", 0, 1)])
        with self.assertRaisesRegex(ValueError, "verify rows up to 2"):
            pool.bind(0, "a")
            rig.round(plugin, pool, rig.round_batch(1, 2, [0]), rig.round_inputs([("a", 0)], 2))


    def test_decode_rows_follow_the_float64_recurrence_by_every_geometry(self):
        torch = self.torch
        variants = [{"normalize": (False, True)}, {"normalize": (True, False)}, {"epsilon": 1e-3},
                    {"dtype": torch.float16}]
        for name, plugin in self.plugins():
            for variant in variants:
                with self.subTest(kernel=name, variant=variant):
                    rig = Rig(torch, self.device(), *self.shapes()[-1], channels=16, tokens=8, seed=37, **variant)
                    pool = rig.pool(plugin, widest=3)
                    pool.open("a")
                    found = rig.step(plugin, pool, [("a", 0, 4)])
                    rig.step(plugin, pool, [("a", 4, 2)])
                    ref, state = rig.reference("a", 0, 6)
                    torch.testing.assert_close(found[0][0].double(), ref[:4], **SCAN)
                    torch.testing.assert_close(pool.state(0, "a").double(), state, **SCAN)

    def test_a_flush_makes_a_deferred_commit_final(self):
        # commit_round reruns with a smaller count (a truncation) before a flush; flush(), state(), window() and fork()
        # apply a deferred commit, which is final from there; checkpoints and recompute stay rerunnable.
        torch = self.torch
        for commit in COMMITS:
            for name, plugin in self.plugins(commit=commit):
                with self.subTest(commit=commit, kernel=name):
                    rig = Rig(torch, self.device(), *self.shapes()[-1], channels=16, tokens=8, seed=41)
                    found, flushed = {}, None
                    for counts, flush in (([3], False), ([1], False), ([3, 1], False), ([3, 1], True)):
                        pool = rig.pool(plugin, widest=3)
                        pool.open("a")
                        pool.bind(0, "a")
                        rig.round(plugin, pool, rig.round_batch(1, 3, [0]), rig.round_inputs([("a", 0)], 3))
                        for i, n in enumerate(counts):
                            pool.commit_round(View(1, 3, torch.tensor([n], device=self.device())))
                            if flush and i == 0:
                                flushed = pool.state(0, "a").clone()   # a flush: the commit of 3 applied
                        found[(tuple(counts), flush)] = pool.state(0, "a").clone()
                    self.assertTrue(torch.equal(found[((3, 1), False)], found[((1,), False)]))
                    self.assertTrue(torch.equal(flushed, found[((3,), False)]))
                    if commit != "deferred":   # rerunnable after the flush too
                        self.assertTrue(torch.equal(found[((3, 1), True)], found[((1,), False)]))

    def test_a_round_of_more_rows_than_the_pool_has_slots_is_refused(self):
        torch = self.torch
        rig = Rig(torch, self.device(), 1, 2, 4, 3, channels=8, tokens=8, seed=43)
        pool = rig.pool(self.recurrence, widest=1, rows=1)
        pool.open("a")
        batch = rig.round_batch(2, 1, [0, 0])
        with self.assertRaisesRegex(ValueError, "a round of 2 rows in a pool of 1 round slots"):
            rig.round(self.recurrence, pool, batch, rig.round_inputs([("a", 0), ("a", 0)], 1))
        with self.assertRaisesRegex(ValueError, "a round of 2 rows in a pool of 1 round slots"):
            pool.commit_round(View(2, 1, torch.tensor([1, 1], device=self.device())))
        with self.assertRaisesRegex(ValueError, "keeps a state per round slot"):
            self.assemble(options={"commit": "checkpoints"})["recurrence"].pool([0], 2, 4, 3, self.device(), 2, 1)

    def test_keys_past_the_kernels_run_the_torch_formula_and_say_so(self):
        if self.device() != "cuda" or self.recurrence.kernels("cuda", 128) is None:
            self.skipTest("recurrent-scans on CUDA")
        from crit_plugins.kernels import loaded
        pool = self.recurrence.pool([0], 1, 300, 4, "cuda", 2)
        self.assertIsNone(pool.kernels)
        record = loaded()["crit-labs/recurrent-scans"]
        self.assertEqual(record["source"], "fallback")
        self.assertIn("300", record["reason"])

    def untimed(self, states, costs=None):
        """A context in which device_states' timing (scan_rows(), _costs()) fails the test when called; with `costs`,
        _costs() is that fake instead, a machine whose load moves a measured crossover, and still never called."""
        from contextlib import ExitStack
        from unittest import mock
        calls = []

        def refuse(*args, **kwargs):
            calls.append(args)
            raise AssertionError("a pool timed a path: which path a prompt row runs is the configuration's")
        stack = ExitStack()
        stack.enter_context(mock.patch.object(states, "scan_rows", refuse))
        stack.enter_context(mock.patch.object(states, "_costs", refuse if costs is None else
                                              (lambda *a, **k: (calls.append(a), costs(*a, **k))[1])))
        stack.calls = calls
        return stack

    def test_prompt_rows_run_the_scan_up_to_the_stated_exact_rows_and_nothing_times_them(self):
        """By default a pool's prompt rows run the scan up to 512 tokens (exact_rows' default), in an eager call and in
        a captured graph alike, and the chunked kernel past it; making the pool and running its rows time nothing.
        A prompt row through the scan has the state bits of the serial decode, and is within the stated tolerance of
        the chunked kernel's; the facts say where the pool switches, why, and how the paths agree."""
        torch = self.torch
        plugin = self.assemble(options={})["recurrence"]
        states = sys.modules[type(plugin).__module__].device_states
        self.assertEqual(states.DEFAULT_EXACT_ROWS, 512)
        # (key heads, heads, keys, values): the gated-delta layers of the 27B and of Flash-Next, and narrower ones.
        for shape in ((16, 48, 128, 128), (2, 6, 64, 64)):
            for kernels in ((True, False) if self.device() == "cuda" else (False,)):
                with self.subTest(shape=shape, kernels=kernels):
                    plugin = self.assemble(options={"kernels": kernels})["recurrence"]
                    chunked = self.assemble(options={"kernels": kernels, "exact_rows": 0})["recurrence"]
                    rig = Rig(torch, self.device(), *shape, channels=32, tokens=600, seed=47, dtype=torch.bfloat16)
                    with self.untimed(states) as timed:
                        pool = rig.pool(plugin)
                        report = plugin.facts()["recurrence_prompts"]
                        stated = report["pools"][-1]
                        kind = states.chunked_kind(pool, rig.geometry, plugin.prompt)
                        self.assertEqual((pool.scan_rows, pool.scan_rows_captured), (512, 512))
                        if kind is None:
                            self.assertEqual(stated["chunked"], None)
                            continue
                        self.assertEqual((stated["source"], stated["default"], stated["chunked"], stated["scan_rows"],
                                          stated["scan_rows_captured"]), ("exact_rows", 512, kind, 512, 512))
                        self.assertIn("2e-3 + 2e-2", report["chunked"])
                        tokens = 48
                        serial, other = rig.pool(plugin), rig.pool(chunked)
                        for p in (pool, serial, other):
                            p.open("a")
                        batch = StepBatch.build([("a", 0, tokens)], self.device())
                        plan = pool._plan(batch)
                        self.assertEqual(pool.delta_rows(plan, pool.scan_rows, kind)[1], [states.FINAL])
                        long = StepBatch.build([("a", 0, pool.scan_rows + 1)], self.device())
                        self.assertEqual(pool.delta_rows(pool._plan(long), pool.scan_rows, kind)[1], [states.FLUSH])
                        for p in (pool, serial, other):
                            p.last.clear()
                        found = rig.step(plugin, pool, [("a", 0, tokens)])
                        steps = [rig.step(plugin, serial, [("a", t, 1)]) for t in range(tokens)]
                        via = rig.step(chunked, other, [("a", 0, tokens)])
                    self.assertEqual(timed.calls, [])
                    for layer in (0, 1):
                        self.assertTrue(torch.equal(found[layer][0], torch.cat([s[layer][0] for s in steps])))
                        self.assertTrue(torch.equal(pool.state(layer, "a"), serial.state(layer, "a")))
                        self.assertTrue(torch.equal(pool.window(layer, "a"), serial.window(layer, "a")))
                        torch.testing.assert_close(pool.state(layer, "a"), other.state(layer, "a"), **PROMPT)
                        torch.testing.assert_close(found[layer][0], via[layer][0], **PROMPT)
                        self.assertTrue(torch.equal(pool.window(layer, "a"), other.window(layer, "a")))

    def test_the_same_prompt_gives_the_same_rows_whatever_the_machine_load(self):
        """Pools made while the machine's timings say the scan is ten times slower, then ten times faster, than it is
        (what a load-time measurement would read under other load) run the same prompt to the same bits: the stated
        exact_rows decides, and nothing reads the timings."""
        torch = self.torch
        plugin = self.assemble(options={})["recurrence"]
        states = sys.modules[type(plugin).__module__].device_states
        rig = Rig(torch, self.device(), 16, 48, 128, 128, channels=32, tokens=600, seed=53, dtype=torch.bfloat16)
        found = []
        for scale in (10.0, 0.1):
            def costs(torch_, fn, reps=3, scale=scale):   # (GPU ms, host ms): a slower or faster scan than measured
                return 1.0 * scale, 1.0 * scale
            with self.untimed(states, costs) as timed:
                pool = rig.pool(plugin)
                pool.open("a")
                # chunks of 400 and 200 tokens: under the default either side of a crossover measured at 300 or 900
                rows = [rig.step(plugin, pool, [("a", 0, 400)]), rig.step(plugin, pool, [("a", 400, 200)])]
                found.append(([r[layer][0] for r in rows for layer in (0, 1)],
                              [pool.state(layer, "a").clone() for layer in (0, 1)]))
            self.assertEqual(timed.calls, [])
        for a, b in zip(found[0][0] + found[0][1], found[1][0] + found[1][1]):
            self.assertTrue(torch.equal(a, b))

    def test_fla_runs_every_kernel_on_its_stated_tile_and_times_nothing(self):
        """fla autotunes its Triton kernels by timing, and some of their tiles give other bits. Every autotuned kernel
        fla's chunked path reaches holds its stated tile (device_states.FLA_TILES) as its one config, so nothing times
        a tile, and with every autotuner's picks cleared (as in a fresh process) a prompt row gets the same bits, in
        FP32 (chunk_fp32) and in BF16."""
        from unittest import mock
        torch = self.torch
        probe = self.assemble(options={})["recurrence"]
        states = sys.modules[type(probe).__module__].device_states
        rig = Rig(torch, self.device(), 2, 6, 128, 128, tokens=200, seed=61, dtype=torch.bfloat16)
        if states.chunked_kind(rig.pool(probe), rig.geometry, probe.prompt) != "fla":
            self.skipTest("fla's chunked kernel needs fla and CUDA")
        from triton.runtime.autotuner import Autotuner
        reached, run = {}, Autotuner.run

        def spy(tuner, *args, **kwargs):
            reached[id(tuner)] = tuner
            return run(tuner, *args, **kwargs)

        def refuse(*args, **kwargs):
            raise AssertionError("an fla kernel timed its tiles")

        for chunk_fp32 in (True, False):
            plugin = self.assemble(options={"exact_rows": 0, "chunk_fp32": chunk_fp32})["recurrence"]
            found = []
            for _ in range(2):
                for tuner in reached.values():
                    tuner.cache.clear()
                with mock.patch.object(Autotuner, "run", spy), mock.patch.object(Autotuner, "_bench", refuse):
                    pool = rig.pool(plugin)
                    pool.open("a")
                    rows = rig.step(plugin, pool, [("a", 0, 150)])
                found.append([rows[layer][0] for layer in (0, 1)] + [pool.state(layer, "a") for layer in (0, 1)])
            for a, b in zip(*found):
                self.assertTrue(torch.equal(a, b), chunk_fp32)
        stated = {kernel: tile for (_, kernel), tile in states.FLA_TILES.items()}
        self.assertEqual(sorted(t.base_fn.__name__ for t in reached.values()), sorted(stated))
        for tuner in reached.values():
            (config,) = tuner.configs
            self.assertEqual((config.kwargs, config.num_warps, config.num_stages), stated[tuner.base_fn.__name__])

    def test_a_prefix_has_the_whole_prompts_rows_at_the_configured_exact_rows(self):
        """A prompt cut into chunks of 256 tokens, and its prefix of 456 tokens (its last chunk 200): with exact_rows at
        the chunk (every prompt row through the scan) the prefix's rows are the whole prompt's, bit for bit. With
        exact_rows between the prefix's last chunk and the whole prompt's (the case a crossover measured at load fell
        into, crit smoke's causal_long), the prefix's last chunk runs the scan where the whole prompt's runs the chunked
        kernel: other bits, within the chunked kernel's stated tolerance."""
        torch = self.torch
        plugin = self.assemble(options={})["recurrence"]
        states = sys.modules[type(plugin).__module__].device_states
        rig = Rig(torch, self.device(), 16, 48, 128, 128, channels=32, tokens=600, seed=59, dtype=torch.bfloat16)
        cases = [(256, True)]
        probe = rig.pool(plugin)
        if states.chunked_kind(probe, rig.geometry, plugin.prompt) is not None:
            cases.append((230, False))
        for exact_rows, same in cases:
            with self.subTest(exact_rows=exact_rows):
                configured = self.assemble(options={"exact_rows": exact_rows})["recurrence"]
                rows = {}
                for name, tail in (("whole", 256), ("prefix", 200)):
                    pool = rig.pool(configured)
                    pool.open("a")
                    rig.step(configured, pool, [("a", 0, 256)])
                    rows[name] = rig.step(configured, pool, [("a", 256, tail)])
                for layer in (0, 1):
                    whole, prefix = rows["whole"][layer][0][:200], rows["prefix"][layer][0]
                    if same:
                        self.assertTrue(torch.equal(whole, prefix))
                    else:
                        self.assertFalse(torch.equal(whole, prefix))
                        torch.testing.assert_close(prefix, whole, **PROMPT)

    def test_a_dilated_convolution_follows_the_fp64_one_through_steps_rounds_and_prompt_chunks(self):
        """pool(conv=(channels, kernel, dilation)): Flash-Next's PLE convolution (4 x 2,560 channels, 4 taps at dilation
        3, on CUDA) and a small one, a pool of a convolution beside a one-value state. Step rows (a prompt row, a verify
        row with part of it accepted, decode rows), a captured round's rows committed at any count and a captured prompt
        chunk follow an FP64 dilated convolution, the windows exactly, on the kernel and the torch formula."""
        torch = self.torch
        from crit_plugins.device import PromptBatch
        shapes = [(24, 3, 2)] + ([(10240, 4, 3)] if self.device() == "cuda" else [])
        tol = dict(atol=1e-5, rtol=1e-5)
        for name, plugin in self.plugins(exact_rows=4096):   # prompt rows through the scan, whose bits are known
            for channels, width, dilation in shapes:
                with self.subTest(kernel=name, channels=channels, width=width, dilation=dilation):
                    rig = Rig(torch, self.device(), 1, 1, 1, 1, channels=channels, width=width, tokens=60, seed=61,
                              dilation=dilation)
                    span = (width - 1) * dilation
                    pool = rig.pool(plugin, widest=3)
                    self.assertEqual(tuple(pool.windows.shape[2:]), (channels, span))
                    for s in "abc":
                        pool.open(s)
                    zero = torch.zeros(channels, span, device=self.device())
                    # A prompt row (a: 30 inputs) beside a verify row (b: 4 inputs, 2 accepted).
                    found = rig.step(plugin, pool, [("a", 0, 30), ("b", 0, 4)], accept=[30, 2], layers=(0,))
                    ref, window_a = rig.conv_reference("a", 0, 30, zero)
                    torch.testing.assert_close(found[0][1][:30].double(), ref, **tol)
                    ref, _ = rig.conv_reference("b", 0, 4, zero)
                    torch.testing.assert_close(found[0][1][30:].double(), ref, **tol)
                    torch.testing.assert_close(pool.window(0, "a").double(), window_a, atol=0, rtol=0)
                    _, window_b = rig.conv_reference("b", 0, 2, zero)
                    torch.testing.assert_close(pool.window(0, "b").double(), window_b, atol=0, rtol=0)
                    # Decode rows continue each window.
                    found = rig.step(plugin, pool, [("a", 30, 1), ("b", 2, 1)], layers=(0,))
                    torch.testing.assert_close(found[0][1][0:1].double(), rig.conv_reference("a", 30, 1, window_a)[0],
                                               **tol)
                    ref, window_b = rig.conv_reference("b", 2, 1, window_b)
                    torch.testing.assert_close(found[0][1][1:2].double(), ref, **tol)
                    _, window_a = rig.conv_reference("a", 30, 1, window_a)
                    # A round of both rows (4 columns), committed at 3 and 1, the outputs from the windows.
                    pool.bind(0, "a")
                    pool.bind(1, "b")
                    batch = rig.round_batch(2, 3, [31, 3])
                    inputs = rig.round_inputs([("a", 31), ("b", 3)], 3)
                    out = plugin.convolve(0, inputs["x"], rig.weight, rig.bias, pool, batch)
                    torch.testing.assert_close(out[:4].double(), rig.conv_reference("a", 31, 4, window_a)[0], **tol)
                    torch.testing.assert_close(out[4:].double(), rig.conv_reference("b", 3, 4, window_b)[0], **tol)
                    pool.commit_round(View(2, 3, torch.tensor([3, 1], device=self.device())))
                    _, window_a = rig.conv_reference("a", 31, 3, window_a)
                    _, window_b = rig.conv_reference("b", 3, 1, window_b)
                    torch.testing.assert_close(pool.window(0, "a").double(), window_a, atol=0, rtol=0)
                    torch.testing.assert_close(pool.window(0, "b").double(), window_b, atol=0, rtol=0)
                    # A prompt chunk of 7 real inputs padded to 16 (c), as a captured chunk reads it.
                    chunk = PromptBatch(16, self.device()).fill(16, "c", 0, list(range(7)))
                    pool.stage_prompt(chunk)
                    x = torch.cat([rig.seq["c"]["x"][:7], torch.zeros_like(rig.seq["c"]["x"][:9])])
                    out = plugin.convolve(0, x, rig.weight, rig.bias, pool, chunk)
                    pool.commit_prompt(chunk)
                    ref, window_c = rig.conv_reference("c", 0, 7, zero)
                    torch.testing.assert_close(out[:7].double(), ref, **tol)
                    self.assertTrue(torch.equal(out[7:], torch.zeros_like(out[7:])))
                    torch.testing.assert_close(pool.window(0, "c").double(), window_c, atol=0, rtol=0)
