"""token-sampling's rounds:sampler: drafts and verify predictions on the device equal the host form's (sampling.py) for
the same rows, drafts, laws and Core's draws, rejection sampling keeps the target's law, and a captured round replays
with the rules staged after its capture. Each test runs on the CUDA sampler (native) and on the torch law."""
import math
import random

from crit_plugins.testing import PluginTestCase

RULES = [{"temperature": 0.0}, {"temperature": 0.8, "top_p": 0.9, "seed": 3}, {"temperature": 1.3, "top_k": 4, "seed": 5},
         {"temperature": 0.6, "min_p": 0.2, "seed": 9, "stream": 1}, {"temperature": 1.0, "top_p": 0.5, "top_k": 6, "seed": 11},
         {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": (1 << 63) + 12345}, {"temperature": 1.0, "seed": 2}]
KERNELS = ("native", "torch")


def view(torch, rows, width, mode, device, widest=None, logprobs=0):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView); with logprobs (a count of
    alternatives, or None for none kept) the driver's log-probability rows and the ranked alternatives' buffers."""
    from crit_plugins.rounds import SpeculativeRounds
    widest = width if widest is None else widest
    found = SpeculativeRounds(None, rows, widest, device, logprobs=logprobs if logprobs else logprobs is not None)
    return found.view(rows, width, mode)


class RoundSamplerTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        self.rng = random.Random(41)

    def sampler(self, kernel):
        return self.assemble(options={"rounds": kernel})["sampler"]

    def rows(self, count, vocab, scale=3.0):
        return [[round(self.rng.uniform(-scale, scale), 3) for _ in range(vocab)] for _ in range(count)]

    def peaked(self, count, vocab):
        """Rows shaped like a model's: a few likely tokens over a long unlikely tail."""
        found = [[round(self.rng.uniform(-6, -3), 3) for _ in range(vocab)] for _ in range(count)]
        for row in found:
            for t in self.rng.sample(range(vocab), min(8, vocab)):
                row[t] = round(self.rng.uniform(1, 6), 3)
        return found

    def exact(self, sampler, row, rule):
        """Whether the CUDA sampler settles the row's law in FP64 (greedy, or a kept set of at most 64 tokens);
        otherwise its full-vocabulary law weighs in FP32 and a draw near a boundary may differ from the FP64 host."""
        return not rule.get("temperature", 0.0) > 0 or len(sampler.law(row, rule)) <= 64

    def test_it_provides_the_rounds_extension(self):
        self.assertImplements()

    def test_the_sampler_cases_verify_columns_equal_the_host_form(self):
        """crit's sampler cases' verify rows (slots._sampler): greedy and sampled predictions, laws from law()."""
        torch = self.torch
        for kernel in KERNELS:
            s = self.sampler(kernel)
            rng = random.Random(7)
            rows = [[round(rng.uniform(-3, 3), 3) for _ in range(11)] for _ in range(6)]
            rows[2][4] = rows[2][7] = max(rows[2]) + 1
            rules = [dict(temperature=0.8, top_p=0.9, seed=3), dict(temperature=1.3, top_k=4, seed=5)]
            drafts = [s.select([rows[c]], dict(rules[0], position=c))[0] for c in range(3)]
            laws = [dict(s.law(rows[c], rules[1])) for c in range(3)]
            s.reserve(2, 3, 11, self.device())
            s.stage(0, {"temperature": 0.0})
            s.stage(1, rules[1])
            v = view(torch, 2, 3, "sampled", self.device(), logprobs=None)
            v.out_count[:] = torch.tensor([0, 2])
            v.drafts[:, :3] = torch.tensor([drafts, drafts])
            logits = torch.tensor([rows[:4], rows[:4]], dtype=torch.float32, device=self.device())
            v.draft_laws = s.laws(2, 3)
            v.draft_laws.write(2, 3, [laws, laws])
            found = s.predictions_device(v, logits, v.draft_laws).tolist()
            with self.subTest(kernel=kernel):
                self.assertEqual(found[0], s.predictions(rows[:4], drafts, {"temperature": 0.0}))
                self.assertEqual(found[1], s.predictions(rows[:4], drafts, dict(rules[1], position=2), laws))

    def test_mixed_rows_and_rules_predict_what_the_host_form_predicts(self):
        torch = self.torch
        for kernel in KERNELS:
            s = self.sampler(kernel)
            for vocab, width in ((11, 3), (300, 5), (5000, 2)):
                count = len(RULES)
                native = s.reserve(count, width, vocab, self.device()).native is not None
                for slot, rule in enumerate(RULES):
                    s.stage(slot, rule)
                for trial in range(4):
                    target = [self.peaked(width + 1, vocab) for _ in range(count)]
                    draft_rows = [self.peaked(width, vocab) for _ in range(count)]
                    laws = [[dict(s.law(draft_rows[r][c], dict(RULES[r], temperature=max(RULES[r].get("temperature", 0),
                                                                                         0.9)))) for c in range(width)]
                            for r in range(count)]
                    laws = [[dict(list(law.items())[:64]) for law in row] for row in laws]
                    drafts = [[max(law, key=law.get) if c % 2 else self.rng.choice(list(law)) for c, law in enumerate(row)]
                              for row in laws]
                    drafts[1][width - 1] = -1   # a cut draft: the target's own draw
                    out_count = [self.rng.randrange(0, 1000) for _ in range(count)]
                    for with_laws in (True, False):
                        v = view(torch, count, width, "sampled", self.device(), logprobs=None)
                        v.out_count.copy_(torch.tensor(out_count))
                        v.drafts[:, :width] = torch.tensor(drafts)
                        if with_laws:
                            v.draft_laws = s.laws(count, width)
                            v.draft_laws.write(count, width, laws)
                        logits = torch.tensor(target, dtype=torch.float32, device=self.device())
                        found = s.predictions_device(v, logits, v.draft_laws).tolist()
                        for r, rule in enumerate(RULES):
                            host = s.predictions(target[r], drafts[r], dict(rule, position=out_count[r]),
                                                 laws[r] if with_laws else None)
                            exact = not native or all(self.exact(s, row, rule) for row in target[r])
                            with self.subTest(kernel=kernel, vocab=vocab, trial=trial, row=r, laws=with_laws):
                                if exact:
                                    self.assertEqual(found[r][:width + 1], host)
                                else:
                                    kept = [{t for t, _ in s.law(row, rule)} for row in target[r]]
                                    self.assertTrue(all(t in k or t == d for t, k, d in
                                                        zip(found[r], kept, drafts[r] + [-1])), (found[r], host))

    def test_a_drafts_draw_and_law_are_the_host_laws(self):
        from crit_plugins.assemble import load_module
        torch = self.torch
        from_host = load_module(self.registry()["token-sampling"]).sampling
        for kernel in KERNELS:
            s = self.sampler(kernel)
            vocab, width = 300, 4
            count = len(RULES)
            native = s.reserve(count, width, vocab, self.device()).native is not None
            for slot, rule in enumerate(RULES):
                s.stage(slot, rule)
            v = view(torch, count, width, "sampled", self.device(), logprobs=None)
            v.draft_laws = s.laws(count, width)   # the round driver's view.draft_laws in sampled rounds
            out_count = [self.rng.randrange(0, 1000) for _ in range(count)]
            v.out_count.copy_(torch.tensor(out_count))
            for depth in range(width):
                rows = self.peaked(count, vocab)
                tokens = s.draft_device(v, torch.tensor(rows, dtype=torch.float32, device=self.device()), depth).tolist()
                laws = v.draft_laws.as_dicts(count, width)
                for r, rule in enumerate(RULES):
                    law = s.law(rows[r], rule)
                    if native and not self.exact(s, rows[r], rule):
                        with self.subTest(kernel=kernel, depth=depth, row=r, law="full"):
                            self.assertIn(tokens[r], {t for t, _ in law})
                            self.assertIsNone(laws[r][depth])   # no sparse form: verified by the target's own draw
                        continue
                    temperature = rule.get("temperature", 0.0)
                    if temperature > 0:
                        u = from_host.uniform(rule.get("seed", 0), out_count[r] + depth, rule.get("stream", 0),
                                              from_host.DRAFT)
                        expected = from_host.walk(law, u * math.fsum(p for _, p in law))
                    else:
                        expected = law[0][0]
                    with self.subTest(kernel=kernel, depth=depth, row=r):
                        self.assertEqual(tokens[r], expected)
                        got = laws[r][depth]
                        self.assertEqual(sorted(t for t in got if got[t] > 0), sorted(t for t, _ in law))
                        for t, p in law:
                            self.assertAlmostEqual(got[t], p, delta=1e-6)

    def test_a_draft_without_a_law_or_its_mass_takes_the_targets_own_draw(self):
        """A copied draft (its law cleared), a draft outside its sparse law and a draft whose dense law gives it no
        mass are verified by the target's own draw, as the host form verifies a draft with no law (None)."""
        torch = self.torch
        for kernel in KERNELS:
            s = self.sampler(kernel)
            vocab, width, rows = 40, 3, 4
            s.reserve(rows, width, vocab, self.device())
            rule = {"temperature": 1.0, "top_k": 10}
            for slot in range(rows):
                s.stage(slot, dict(rule, seed=31 + slot))
            target = [self.peaked(width + 1, vocab) for _ in range(rows)]
            laws = [[dict(s.law(row, rule)) for row in self.peaked(width, vocab)] for _ in range(rows)]
            # Drafts outside their laws' support (q(d) = 0), to be accepted by a q = 0 rule whenever p(d) > 0.
            drafts = [[max(set(range(vocab)) - set(law), key=lambda t: target[r][c][t]) for c, law in enumerate(row)]
                      for r, row in enumerate(laws)]
            v = view(torch, rows, width, "sampled", self.device(), logprobs=None)
            v.drafts[:, :width] = torch.tensor(drafts)
            v.draft_laws = s.laws(rows, width)
            v.draft_laws.write(rows, width, laws)
            cleared = torch.zeros(rows, width, dtype=torch.bool, device=self.device())
            cleared[:, 1] = True   # column 1 a copy: no law at all
            v.draft_laws.clear(rows, width, cleared)
            logits = torch.tensor(target, dtype=torch.float32, device=self.device())
            found = s.predictions_device(v, logits, v.draft_laws).tolist()
            for r in range(rows):
                host_laws = [laws[r][0], None, laws[r][2]]
                host = s.predictions(target[r], drafts[r], dict(rule, seed=31 + r), host_laws)
                with self.subTest(kernel=kernel, row=r):
                    self.assertEqual(found[r][:width + 1], host)
                    self.assertEqual(host, s.predictions(target[r], drafts[r], dict(rule, seed=31 + r)))

    def test_a_draft_whose_law_keeps_more_than_64_tokens_is_verified_by_the_targets_own_draw(self):
        """An open law (no top_k, a flat row) has no sparse form on CUDA: its draft is verified by the target's own
        draw, the host form's verify with no law for that column (the torch law keeps the dense law and its rule)."""
        torch = self.torch
        for kernel in KERNELS:
            s = self.sampler(kernel)
            vocab, width, rows = 300, 2, 2
            native = s.reserve(rows, width, vocab, self.device()).native is not None
            rule = {"temperature": 1.0, "top_k": 0}
            for slot in range(rows):
                s.stage(slot, dict(rule, seed=7 + slot))
            v = view(torch, rows, width, "sampled", self.device(), logprobs=None)
            v.draft_laws = s.laws(rows, width)
            flat = self.rows(rows, vocab, scale=0.5)
            for depth in range(width):
                v.drafts[:, depth] = s.draft_device(v, torch.tensor(flat, dtype=torch.float32, device=self.device()),
                                                    depth)
            laws = v.draft_laws.as_dicts(rows, width)
            target = [self.rows(width + 1, vocab, scale=0.5) for _ in range(rows)]
            found = s.predictions_device(v, torch.tensor(target, dtype=torch.float32, device=self.device()),
                                         v.draft_laws).tolist()
            for r in range(rows):
                drafts = v.drafts[r, :width].tolist()
                host = s.predictions(target[r], drafts, dict(rule, seed=7 + r), laws[r])
                with self.subTest(kernel=kernel, row=r):
                    if native:
                        self.assertEqual(laws[r], [None] * width)   # more than 64 kept: no sparse form
                        kept = [{t for t, _ in s.law(row, rule)} for row in target[r]]
                        self.assertTrue(all(t in k for t, k in zip(found[r], kept)))
                    else:
                        self.assertEqual(found[r][:width + 1], host)

    def test_a_second_reserve_keeps_the_rules_staged_and_the_facts_name_the_kernel(self):
        for kernel in KERNELS:
            s = self.sampler(kernel)
            s.reserve(2, 3, 50, self.device())
            s.stage(0, {"temperature": 0.7, "top_k": 5, "seed": 9})
            before = s._round_sampling()._round.h_rules[0].tolist()
            state = s.reserve(2, 3, 50, self.device(), logprobs=4)
            with self.subTest(kernel=kernel):
                self.assertEqual(state.h_rules[0].tolist(), before)
                self.assertEqual(state.top, 4)
                expected = "native" if state.native is not None else "torch"
                self.assertEqual(s.facts(), {"round_sampling": expected})
                if kernel == "torch":
                    self.assertEqual(expected, "torch")

    def test_rejection_sampling_keeps_the_target_law(self):
        """The first emitted token of a sampled round, its draft drawn from another law, follows the target's law
        (chi-square over 32,768 rounds of 6 tokens), and drafts are accepted at 1 - TV(p, q)."""
        torch = self.torch
        target_row = [1.2, 0.3, -0.4, 0.9, 0.0, -1.5]
        draft_row = [0.2, 1.1, -0.2, 0.9, 0.4, -0.5]
        rule = {"temperature": 1.0, "top_k": 5}
        for kernel in KERNELS:
            s = self.sampler(kernel)
            p = dict(s.law(target_row, rule))
            q = dict(s.law(draft_row, rule))
            count = 4096
            s.reserve(count, 1, 6, self.device())
            for slot in range(count):
                s.stage(slot, dict(rule, seed=slot * 7919 + 1))
            target = torch.tensor([target_row, target_row], dtype=torch.float32,
                                  device=self.device())[None].expand(count, 2, 6).contiguous()
            drafts_logits = torch.tensor(draft_row, dtype=torch.float32, device=self.device())[None].expand(count, 6)
            seen, accepted = [0] * 6, 0
            for batch in range(8):
                v = view(torch, count, 1, "sampled", self.device(), logprobs=None)
                v.draft_laws = s.laws(count, 1)
                v.out_count.fill_(batch)
                v.drafts[:, 0] = s.draft_device(v, drafts_logits.contiguous(), 0)
                predictions = s.predictions_device(v, target, v.draft_laws)
                first = predictions[:, 0].tolist()
                accepted += int((predictions[:, 0] == v.drafts[:, 0]).sum())
                for t in first:
                    seen[t] += 1
            n = 8 * count
            chi = sum((seen[t] - n * p.get(t, 0.0)) ** 2 / (n * p[t]) for t in p)
            with self.subTest(kernel=kernel):
                self.assertEqual(seen[5], 0)   # outside the top 5
                self.assertLess(chi, 20.5)     # 4 degrees of freedom: p < 0.0005
                tv = 0.5 * sum(abs(p.get(t, 0) - q.get(t, 0)) for t in range(6))
                self.assertAlmostEqual(accepted / n, 1 - tv, delta=4 * math.sqrt((1 - tv) * tv / n) + 1e-3)

    def test_greedy_rounds_take_the_argmax_and_logprobs_are_torchs_fp32_log_softmax(self):
        torch = self.torch
        for kernel in KERNELS:
            s = self.sampler(kernel)
            vocab, width, rows = 700, 3, 4
            s.reserve(rows, width, vocab, self.device(), logprobs=5)
            logits = torch.tensor([self.rows(width + 1, vocab) for _ in range(rows)], dtype=torch.float32,
                                  device=self.device())
            logits[0, 1, 10] = logits[0, 1, 20] = logits[0, 1].max() + 1   # a tie: the lower id
            for mode in ("greedy", "sampled"):
                v = view(torch, rows, width, mode, self.device(), logprobs=5)
                found = s.predictions_device(v, logits, None)
                s.logprobs_device(v, logits, found)   # the round driver's, after the accept
                expected = torch.log_softmax(logits, -1)
                with self.subTest(kernel=kernel, mode=mode):
                    self.assertEqual(found.tolist(), torch.argmax(logits, -1).tolist())
                    self.assertEqual(int(found[0, 1]), 10)
                    chosen = expected.gather(2, found[..., None])[..., 0]
                    self.assertTrue(torch.equal(v.logprobs[:, :width + 1], chosen))
                    ranked = torch.sort(expected, dim=-1, descending=True, stable=True).indices[..., :5]
                    self.assertEqual(v.top_ids[:, :width + 1].tolist(), ranked.tolist())
                    self.assertTrue(torch.equal(v.top_logprobs[:, :width + 1], expected.gather(2, ranked)))

    def test_a_captured_rounds_log_probabilities_are_the_step_and_score_paths_bit_for_bit(self):
        # A torch model's step and score paths take torch.log_softmax of FP32 logits rows (a step's row alone or a
        # stack of rows, a score's chunk of rows): a captured round scores its emitted tokens in the same arithmetic,
        # so each value, the chosen token's and every alternative's, is the same float. Vocabularies through each of
        # torch's row kernels (one warp, a block over registers, shared memory, a block over the row in memory).
        torch = self.torch
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        g = torch.Generator(device="cuda").manual_seed(29)
        for kernel in KERNELS:
            s = self.sampler(kernel)
            for vocab in (700, 4096, 12000, 151936, 248320):
                rows, width, top = 3, 2, 5
                s.reserve(rows, width, vocab, "cuda", logprobs=top)
                v = view(torch, rows, width, "greedy", "cuda", logprobs=top)
                logits = torch.zeros(rows, width + 1, vocab, device="cuda")
                tokens = torch.zeros(rows, width + 1, dtype=torch.int64, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):   # the driver's warm-ups
                        s.logprobs_device(v, logits, tokens)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    s.logprobs_device(v, logits, tokens)
                for trial in range(2):
                    # A model's rows: a long unlikely tail and a few likely tokens, two of them tied.
                    logits.copy_(torch.randn(rows, width + 1, vocab, device="cuda", generator=g) * 2 - 4)
                    likely = torch.randint(0, vocab, (rows, width + 1, 8), device="cuda", generator=g)
                    logits.scatter_(2, likely, torch.rand(rows, width + 1, 8, device="cuda", generator=g) * 6 + 2)
                    logits[0, 1, likely[0, 1, :2]] = 9.0
                    tokens.copy_(torch.randint(0, vocab, (rows, width + 1), device="cuda", generator=g))
                    tokens[1, 0] = likely[1, 0, 0]
                    graph.replay()
                    torch.cuda.synchronize()
                    flat = logits.reshape(-1, vocab)
                    alone = torch.stack([torch.log_softmax(flat[j].clone(), -1) for j in range(len(flat))])
                    stacked = torch.log_softmax(flat.flip(0), -1).flip(0)   # each row at another place
                    ranked = torch.sort(alone, dim=-1, descending=True, stable=True)
                    with self.subTest(kernel=kernel, vocab=vocab, trial=trial):
                        self.assertTrue(torch.equal(alone, stacked))
                        chosen = alone.gather(1, tokens.reshape(-1, 1))[:, 0].view(rows, width + 1)
                        self.assertTrue(torch.equal(v.logprobs[:, :width + 1], chosen))
                        self.assertEqual(v.top_ids[:, :width + 1].reshape(-1, top).tolist(),
                                         ranked.indices[:, :top].tolist())
                        self.assertTrue(torch.equal(v.top_logprobs[:, :width + 1].reshape(-1, top),
                                                    ranked.values[:, :top]))

    def test_a_captured_round_replays_with_rules_staged_after_it(self):
        torch = self.torch
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        for kernel in KERNELS:
            s = self.sampler(kernel)
            vocab, width, rows = 300, 3, len(RULES)
            s.reserve(rows, width, vocab, "cuda")
            for slot in range(rows):
                s.stage(slot, RULES[0])
            v = view(torch, rows, width, "sampled", "cuda", logprobs=None)
            v.draft_laws = s.laws(rows, width)
            draft_logits = torch.zeros(width, rows, vocab, device="cuda")
            target = torch.zeros(rows, width + 1, vocab, device="cuda")

            def round_():
                for depth in range(width):
                    v.drafts[:, depth] = s.draft_device(v, draft_logits[depth], depth)
                s.predictions_device(v, target, v.draft_laws)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                round_()
                round_()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                round_()
            for trial in range(3):
                for slot, rule in enumerate(RULES):
                    s.stage(slot, RULES[(slot + trial) % len(RULES)])
                draft_logits.copy_(torch.tensor([self.rows(rows, vocab) for _ in range(width)]))
                target.copy_(torch.tensor([self.rows(width + 1, vocab) for _ in range(rows)]))
                v.out_count.copy_(torch.tensor([self.rng.randrange(0, 500) for _ in range(rows)]))
                graph.replay()
                torch.cuda.synchronize()
                replayed = (v.drafts[:, :width].clone(), v.predictions[:, :width + 1].clone())
                round_()
                with self.subTest(kernel=kernel, trial=trial):
                    self.assertTrue(torch.equal(replayed[0], v.drafts[:, :width]))
                    self.assertTrue(torch.equal(replayed[1], v.predictions[:, :width + 1]))
