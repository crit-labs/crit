"""topk-router's device form selects what its host form selects, with the same weights, under every rule; ties go to
the lower expert and group id; a row of NaN logits (a captured round's junk positions) still picks valid distinct
experts with weights 0 and leaves the other rows alone, NaN ranking below a real -inf; a bias off the device is moved
once; and a call captures in a CUDA graph."""
import random

from crit_plugins.testing import PluginTestCase

RULES = [{}, {"scoring": "sigmoid", "score_before_selection": True, "renormalize": True, "scaling": 2.5},
         {"groups": 4, "selected_groups": 2, "group_score_top": 2, "score_before_selection": True},
         {"scoring": "sigmoid"}, {"score_before_selection": True, "renormalize": True}]


class DeviceRouterTests(PluginTestCase):
    def routers(self, rule):
        """The router with crit's kernel topk-routing (where it builds) and with torch alone, on a GPU; torch
        on a CPU."""
        kernels = (True, False) if self.device() == "cuda" else (False,)
        return [self.assemble(options={**rule, "kernel": k})["router"] for k in kernels]

    def test_every_token_routes_as_the_host_form_routes_it(self):
        torch = __import__("torch") if self.device() else None
        rng = random.Random(4)
        logits = [[round(rng.uniform(-4, 4), 4) for _ in range(16)] for _ in range(9)]
        bias = [round(rng.uniform(-1, 1), 4) for _ in range(16)]
        for rule in RULES:
            for router in self.routers(rule):
                for b in (None, bias, torch.tensor(bias, device=self.device())):
                    with self.subTest(rule=rule, bias=b is not None, kernel=router.kernel is not None):
                        ids, weights = router.select(torch.tensor(logits, device=self.device()), 3, b)
                        host = router.route(logits, 3, None if b is None else bias)
                        self.assertEqual(ids.tolist(), [h[0] for h in host])
                        for row, (_, expected) in zip(weights.tolist(), host):
                            self.assertClose(row, expected, atol=1e-5)

    def test_ties_go_to_the_lower_expert_and_group(self):
        torch = __import__("torch") if self.device() else None
        logits = [[1.0, 2.0, 2.0, 0.5, 2.0, 0.5, 0.5, 0.5], [0.0] * 8]
        for rule in ({}, {"groups": 4, "selected_groups": 2, "group_score_top": 1, "score_before_selection": True}):
            for router in self.routers(rule):
                ids, _ = router.select(torch.tensor(logits, device=self.device()), 2)
                self.assertEqual(ids.tolist(), [h[0] for h in router.route(logits, 2)], rule)
        self.assertEqual(ids.tolist()[1], [0, 1])

    def test_a_row_of_nan_logits_still_routes_and_leaves_the_others_alone(self):
        torch = __import__("torch") if self.device() else None
        rng = random.Random(6)
        rows = [[rng.uniform(-3, 3) for _ in range(64)] for _ in range(4)]
        for rule in RULES:
            for router in self.routers(rule):
                with self.subTest(rule=rule, kernel=router.kernel is not None):
                    self.junk_rows_route(torch, router, rows)

    def junk_rows_route(self, torch, router, rows):
        clean = torch.tensor(rows, device=self.device())
        ids, weights = router.select(clean, 4)
        junk = clean.clone()
        junk[1] = float("nan")
        junk[2, 5] = float("nan")
        jids, jweights = router.select(junk, 4)
        self.assertEqual(jids[1].tolist(), [0, 1, 2, 3])
        self.assertEqual(jweights[1].tolist(), [0.0] * 4)
        for r in (0, 3):
            self.assertTrue(torch.equal(jids[r], ids[r]) and torch.equal(jweights[r], weights[r]))
        self.assertTrue(bool(((jids >= 0) & (jids < 64)).all()))
        self.assertFalse(bool(torch.isnan(jweights).any()))
        for r in range(4):
            self.assertEqual(len(set(jids[r].tolist())), 4)

    def test_nan_ranks_below_a_real_negative_infinity(self):
        torch = __import__("torch") if self.device() else None
        inf, nan = float("inf"), float("nan")
        logits = [[nan, -inf, 1.0, nan, -inf, nan, nan, nan], [-0.0, 0.0, nan, -inf, -1.0, nan, nan, nan]]
        for router in self.routers({"scoring": "sigmoid"}):
            with self.subTest(kernel=router.kernel is not None):
                ids, weights = router.select(torch.tensor(logits, device=self.device()), 4)
                self.assertEqual(ids.tolist(), [[2, 1, 4, 0], [0, 1, 4, 3]])   # -0 and +0 tie: the lower id first
                self.assertFalse(bool(torch.isnan(weights).any()))

    def test_a_bias_off_the_device_is_moved_once_and_a_captured_call_copies_nothing(self):
        torch = __import__("torch") if self.device() else None
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        g = torch.Generator().manual_seed(3)
        bias = torch.randn(512, generator=g) * 0.01   # on the CPU, as a checkpoint's tensor may be
        for router in self.routers({}):
            with self.subTest(kernel=router.kernel is not None):
                logits = torch.randn(6, 512, generator=g).cuda()
                router.select(logits, 10, bias)
                moved = router._device_bias(bias, logits.device)
                router.select(logits, 10, bias)
                self.assertIs(router._device_bias(bias, logits.device), moved)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    ids, weights = router.select(logits, 10, bias)
                graph.replay()
                eager_ids, eager_weights = router.select(logits.clone(), 10, bias.cuda())
                self.assertTrue(torch.equal(ids, eager_ids) and torch.equal(weights, eager_weights))

    def test_captured_prompt_buckets_route_their_real_rows_as_alone_and_their_padding_validly(self):
        torch = __import__("torch") if self.device() else None
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        from crit_plugins.prompt_graphs import BUCKETS, PromptGraphs
        g = torch.Generator().manual_seed(9)
        table = torch.randn(97, 64, generator=g).cuda() * 3
        bias = torch.randn(64, generator=g).cuda() * 0.01
        for router in self.routers({"score_before_selection": True, "renormalize": True}):
            with self.subTest(kernel=router.kernel is not None):
                def forward(batch):   # a prompt chunk's router logits, its padding NaN
                    t = batch.tokens_t % 97
                    logits = torch.where(batch.valid_t[:, None], table[t], torch.full_like(table[t], float("nan")))
                    ids, weights = router.select(logits, 6, bias)
                    return torch.cat([ids.float(), weights], 1)

                prompts = PromptGraphs(forward, device="cuda", buckets=BUCKETS)
                self.assertEqual(prompts.prepare(), list(BUCKETS))
                for count in (1, 16, 33, 128, 257, 512):
                    tokens = [(3 * i + count) % 97 for i in range(count)]
                    out = prompts.run(0, 0, tokens).clone()
                    ids, weights = router.select(table[torch.tensor(tokens, device="cuda")], 6, bias)
                    self.assertTrue(torch.equal(out[:count, :6].long(), ids) and torch.equal(out[:count, 6:], weights))
                    pad_ids, pad_weights = out[count:, :6].long(), out[count:, 6:]
                    self.assertTrue(bool(((pad_ids >= 0) & (pad_ids < 64)).all()) and bool((pad_weights == 0).all()))
                    for row in pad_ids.tolist():
                        self.assertEqual(len(set(row)), 6)

    def test_a_captured_selection_replays_new_logits(self):
        torch = __import__("torch") if self.device() else None
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        for router in self.routers({"score_before_selection": True, "renormalize": True}):
            self.replays(torch, router)

    def replays(self, torch, router):
        g = torch.Generator().manual_seed(2)
        logits = torch.randn(6, 512, generator=g).cuda()
        bias = torch.randn(512, generator=g).cuda() * 0.01
        router.select(logits, 10, bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ids, weights = router.select(logits, 10, bias)
        logits.copy_(torch.randn(6, 512, generator=g))
        graph.replay()
        eager_ids, eager_weights = router.select(logits.clone(), 10, bias)
        self.assertTrue(torch.equal(ids, eager_ids) and torch.equal(weights, eager_weights))
