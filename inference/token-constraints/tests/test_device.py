"""token-constraints' device form: constrained rows masked before Core's law, the same tokens as the host form's
select(), in a step and staged for a CUDA graph (masks at a fixed address, refilled before each launch)."""
import random

from crit_plugins import Allowed
from crit_plugins.testing import PluginTestCase

VOCABULARY = 70


class ConstrainedDeviceTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch")
        self.dev = self.device()
        self.sampler = self.assemble()["sampler"]

    def case(self, seed, rows=6):
        rng = random.Random(seed)
        g = self.torch.Generator().manual_seed(seed)
        logits = (self.torch.randn(rows, VOCABULARY, generator=g) * 2).round(decimals=1)   # ties among the values
        params = []
        for r in range(rows):
            p = {"temperature": rng.choice([0.0, 0.7, 1.0, 1.3]), "top_k": rng.choice([0, 3, 20]),
                 "top_p": rng.choice([1.0, 0.9, 0.5]), "min_p": rng.choice([0.0, 0.05]), "seed": rng.randrange(1 << 40),
                 "position": rng.randrange(1 << 20), "stream": rng.randrange(4)}
            if r % 3 != 2:
                p["allowed"] = Allowed.of(sorted(rng.sample(range(VOCABULARY), rng.choice([1, 2, 9, 40]))), VOCABULARY)
            params.append(p)
        return logits, params

    def host(self, logits, params):
        return self.sampler.select(logits.tolist(), [dict(p, allowed=list(p["allowed"])) if "allowed" in p else p
                                                     for p in params])

    def test_it_has_the_sampler_interface_and_device_form(self):
        self.assertImplements()

    def test_sample_takes_the_host_forms_tokens_inside_each_mask(self):
        for seed in range(40):
            logits, params = self.case(seed)
            found = self.sampler.sample(logits.to(self.dev), params).tolist()
            self.assertEqual(found, self.host(logits, params), seed)
            for token, p in zip(found, params):
                if "allowed" in p:
                    self.assertIn(token, p["allowed"])

    def test_staged_masks_replay_from_a_cuda_graph_with_new_masks(self):
        if self.dev != "cuda":
            self.skipTest("CUDA graphs need CUDA")
        torch = self.torch
        staged = self.sampler.staged(6, "cuda", VOCABULARY)
        logits, params = self.case(100)
        buffer = logits.cuda()
        staged.set(params)
        self.assertEqual(staged.sample(buffer).tolist(), self.host(logits, params))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = staged.sample(buffer)
        for seed in range(101, 121):
            logits, params = self.case(seed)
            staged.set(params)                 # new rules and masks, outside the graph
            buffer.copy_(logits.cuda())
            graph.replay()
            self.assertEqual(out.tolist(), self.host(logits, params), seed)

    def test_a_padded_logits_row_never_takes_a_column_past_the_vocabulary(self):
        logits = self.torch.full((1, VOCABULARY + 26), -5.0)
        logits[0, VOCABULARY + 3] = 50.0                      # a padding column, the largest value
        allowed = Allowed.of([4, 9], VOCABULARY)
        self.assertIn(int(self.sampler.sample(logits.to(self.dev), [{"allowed": allowed}])[0]), (4, 9))
        if self.dev == "cuda":
            staged = self.sampler.staged(1, "cuda", VOCABULARY)
            staged.set([{"allowed": allowed}])
            self.assertIn(int(staged.sample(logits.cuda())[0]), (4, 9))

    def test_staged_masks_keep_unconstrained_rows_inside_the_vocabulary_and_clear_stale_rows(self):
        if self.dev != "cuda":
            self.skipTest("staged masks are a CUDA step's")
        torch = self.torch
        staged = self.sampler.staged(2, "cuda", VOCABULARY)
        logits = torch.full((2, VOCABULARY + 26), -5.0, device="cuda")
        logits[:, VOCABULARY + 3] = 50.0                      # a padding column, the largest value
        logits[:, 11] = 1.0
        allowed = Allowed.of([4, 9], VOCABULARY)
        staged.set([{"allowed": allowed}, {"allowed": allowed}])
        self.assertTrue(set(staged.sample(logits).tolist()) <= {4, 9})
        staged.set([{"temperature": 0.0}])                    # one row now: row 1's mask is gone
        mask = staged.mask(2, VOCABULARY + 26)
        self.assertEqual(int(mask[1].sum()), VOCABULARY)
        self.assertFalse(bool(mask[:, VOCABULARY:].any()))
        self.assertEqual(staged.sample(logits[:1]).tolist(), [11])


class ConstrainedRoundTests(PluginTestCase):
    """rounds:sampler: token-sampling's round form with each slot's mask staged beside its rules."""

    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("captured rounds run on CUDA")
        self.torch = __import__("torch")

    def view(self, rows, width, mode):
        from crit_plugins.rounds import SpeculativeRounds
        return SpeculativeRounds(None, rows, max(width, 1), "cuda").view(rows, width, mode)

    def test_it_provides_the_rounds_extension(self):
        self.assertImplements()

    def test_captured_round_predictions_mask_constrained_slots_as_the_host_form_does(self):
        torch = self.torch
        for kernel in ("native", "torch"):
            sampler = self.assemble(options={"rounds": kernel})["sampler"]
            sampler.reserve(2, 2, VOCABULARY, "cuda")
            rng = random.Random(5)
            g = torch.Generator().manual_seed(5)

            def stage():
                allowed = sorted(rng.sample(range(VOCABULARY), rng.choice([1, 3, 12])))
                rules = [{"temperature": rng.choice([0.0, 0.9]), "seed": rng.randrange(1 << 30),
                          "allowed": Allowed.of(allowed, VOCABULARY)},
                         {"temperature": rng.choice([0.0, 1.1]), "seed": rng.randrange(1 << 30)}]
                for slot, p in enumerate(rules):
                    sampler.stage(slot, p)
                return rules
            for width, mode in ((0, "sampled"), (2, "greedy")):
                view = self.view(2, width, mode)
                logits = torch.zeros(2, width + 1, VOCABULARY, device="cuda")
                drafts = torch.zeros(2, width, dtype=torch.int64, device="cuda")

                def run():
                    if width:
                        view.drafts[:, :width].copy_(drafts)
                    return sampler.predictions_device(view, logits)
                rules = stage()
                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = run()
                for trial in range(6):
                    rules = stage()                       # new rules and masks, outside the graph
                    rows = (torch.randn(2, width + 1, VOCABULARY, generator=g) * 2).round(decimals=1)
                    logits.copy_(rows.cuda())
                    view.out_count[:2] = torch.tensor([trial, 3 * trial])
                    drafts.copy_(rows[:, :width].argmax(-1).cuda() if width else drafts)
                    graph.replay()
                    found = out.tolist()
                    for r, p in enumerate(rules):
                        host = dict(p, position=trial if r == 0 else 3 * trial)
                        if mode == "greedy":
                            host["temperature"] = 0.0
                        if "allowed" in host:
                            host["allowed"] = list(host["allowed"])
                        self.assertEqual(found[r][0], sampler.select([rows[r, 0].tolist()], host)[0],
                                         (kernel, width, trial, r))
                    if width:
                        self.assertEqual(view.drafts[0, :width].tolist(), [-1] * width)   # cut: one token a round
                        want = [sampler.select([rows[1, c].tolist()], {"temperature": 0.0})[0]
                                for c in range(width + 1)]
                        self.assertEqual(found[1], want, (kernel, trial))
