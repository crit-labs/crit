"""experts-reference's device form computes the routed SwiGLU block its host form computes, a token's result the
same alone as beside other rows (its products run in chunks of one fixed size), whatever the global precision flags,
it captures in a CUDA graph, and a load after a forward writes its expert in place."""
from crit_plugins.slots import expert_weights
from crit_plugins.testing import PluginTestCase

USE = {"expert_source": "expert-tensors"}


class DeviceExpertsTests(PluginTestCase):
    def test_the_device_block_equals_the_host_block(self):
        torch = __import__("torch") if self.device() else None
        parts = self.assemble(USE)
        experts, device = parts["experts"], self.device()
        for e in range(4):
            gate_up, down = expert_weights(1, e)
            parts["expert_source"].add(1, e, gate_up, down)
            experts.load(1, e, torch.tensor(gate_up, device=device), torch.tensor(down, device=device))
        x = [[0.3, -0.2, 0.5, 0.1, -0.4, 0.7], [1.0, 0.0, -1.0, 0.5, 0.5, -0.5], [0.2, 0.2, -0.3, 0.9, 0.0, 0.1]]
        ids, weights = [[0, 2], [3, -1], [2, 1]], [[0.75, 0.25], [1.0, 0.6], [0.5, 0.5]]
        host = experts.run(1, x, ids, weights)
        tensor = lambda rows, dtype=torch.float32: torch.tensor(rows, dtype=dtype, device=device)  # noqa: E731
        out = experts.forward(1, tensor(x), tensor(ids, torch.int64), tensor(weights))
        for row, expected in zip(out.tolist(), host):
            self.assertClose(row, expected, atol=1e-5)
        alone = experts.forward(1, tensor(x[2:]), tensor(ids[2:], torch.int64), tensor(weights[2:]))
        self.assertEqual(alone.tolist()[0], out.tolist()[2])

    def wide(self, torch, experts_count=16, hidden=128, inter=64):
        parts = self.assemble(USE)
        experts, device = parts["experts"], self.device()
        g = torch.Generator().manual_seed(3)
        for e in range(experts_count):
            gate_up = (torch.randn(2 * inter, hidden, generator=g) * 0.05).bfloat16()
            down = (torch.randn(hidden, inter, generator=g) * 0.05).bfloat16()
            parts["expert_source"].add(2, e, gate_up.float().tolist(), down.float().tolist())
            experts.load(2, e, gate_up.to(device), down.to(device))
        return experts

    def routes(self, torch, tokens, top, experts_count, seed):
        g = torch.Generator().manual_seed(seed)
        ids = torch.stack([torch.randperm(experts_count, generator=g)[:top] for _ in range(tokens)])
        ids[torch.rand(tokens, top, generator=g) < 0.1] = -1
        x = torch.randn(tokens, 128, generator=g).bfloat16()
        device = self.device()
        return x.to(device), ids.to(device), torch.rand(tokens, top, generator=g).to(device)

    def test_many_routes_batched_equal_each_alone_and_the_host_block(self):
        torch = __import__("torch") if self.device() else None
        experts = self.wide(torch)
        x, ids, w = self.routes(torch, 70, 4, 16, 5)
        out = experts.forward(2, x, ids, w)
        for first, count in ((0, 1), (33, 1), (10, 9)):
            part = experts.forward(2, x[first:first + count].clone(), ids[first:first + count].clone(),
                                   w[first:first + count].clone())
            self.assertTrue(torch.equal(part, out[first:first + count]), (first, count))
        rows = [0, 33, 69]
        host = experts.run(2, x[rows].float().tolist(), ids[rows].tolist(), w[rows].tolist())
        for r, expected in zip(rows, host):
            self.assertClose(out[r].tolist(), expected, atol=1e-5, rtol=1e-4)

    def test_a_captured_call_replays_new_routes(self):
        torch = __import__("torch") if self.device() else None
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        experts = self.wide(torch)
        x, ids, w = self.routes(torch, 6, 4, 16, 7)
        out = experts.forward(2, x, ids, w)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = experts.forward(2, x, ids, w)
        nx, nids, nw = self.routes(torch, 6, 4, 16, 8)
        x.copy_(nx)
        ids.copy_(nids)
        w.copy_(nw)
        graph.replay()
        self.assertTrue(torch.equal(out, experts.forward(2, x.clone(), ids.clone(), w.clone())))

    def test_a_load_after_a_forward_writes_its_expert_in_place(self):
        torch = __import__("torch") if self.device() else None
        experts = self.wide(torch)
        x, ids, w = self.routes(torch, 9, 4, 16, 9)
        experts.forward(2, x, ids, w)
        g = torch.Generator().manual_seed(11)
        gate_up = (torch.randn(128, 128, generator=g) * 0.05).bfloat16().to(self.device())
        down = (torch.randn(128, 64, generator=g) * 0.05).bfloat16().to(self.device())
        experts.load(2, 5, gate_up, down)
        fresh = self.wide(torch)
        fresh.load(2, 5, gate_up, down)   # before its first forward
        self.assertTrue(torch.equal(experts.forward(2, x, ids, w), fresh.forward(2, x, ids, w)))

    def test_its_products_neither_read_nor_set_the_global_precision_flags(self):
        torch = __import__("torch") if self.device() else None
        if self.device() != "cuda":
            self.skipTest("TF32 is a CUDA flag")
        experts = self.wide(torch)
        x, ids, w = self.routes(torch, 5, 4, 16, 12)
        flags = torch.backends.cuda.matmul
        before = flags.allow_tf32
        flags.allow_tf32 = True
        try:
            out = experts.forward(2, x, ids, w)
            self.assertTrue(flags.allow_tf32)
        finally:
            flags.allow_tf32 = before
        host = experts.run(2, x.float().tolist(), ids.tolist(), w.tolist())
        for row, expected in zip(out.tolist(), host):
            self.assertClose(row, expected, atol=1e-5, rtol=1e-4)

    def test_captured_prompt_buckets_give_the_unpadded_rows_and_their_padding_routes_nowhere(self):
        torch = __import__("torch") if self.device() else None
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        from crit_plugins.prompt_graphs import BUCKETS, PromptGraphs
        experts = self.wide(torch)
        g = torch.Generator().manual_seed(13)
        hidden = torch.randn(97, 128, generator=g).bfloat16().cuda()
        routes = torch.stack([torch.randperm(16, generator=g)[:4] for _ in range(97)]).cuda()
        weights = torch.rand(97, 4, generator=g).cuda()

        def forward(batch):   # its padding NaN, its ids masked, as a model's prompt forward masks them
            t = batch.tokens_t % 97
            x = torch.where(batch.valid_t[:, None], hidden[t], torch.full_like(hidden[t], float("nan")))
            return experts.forward(2, x, routes[t].masked_fill(~batch.valid_t[:, None], -1), weights[t])

        prompts = PromptGraphs(forward, device="cuda", buckets=BUCKETS)
        self.assertEqual(prompts.prepare(), list(BUCKETS))
        for count in (3, 16, 70, 128, 400, 512):
            tokens = [(5 * i + count) % 97 for i in range(count)]
            out = prompts.run(0, 0, tokens).clone()
            t = torch.tensor(tokens, device="cuda")
            self.assertTrue(torch.equal(out[:count], experts.forward(2, hidden[t], routes[t], weights[t])), count)
            self.assertTrue(bool((out[count:] == 0).all()), count)

    def test_a_stored_format_is_refused(self):
        torch = __import__("torch") if self.device() else None
        experts = self.assemble(USE)["experts"]
        with self.assertRaisesRegex(ValueError, "dense"):
            experts.load(0, 0, {"weight": torch.zeros(4, 2)}, torch.zeros(2, 2), "nvfp4")
