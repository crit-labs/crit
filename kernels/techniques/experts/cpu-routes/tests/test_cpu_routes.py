"""crit's kernel cpu-routes on its own, over a slot in pinned host memory (the host's handoffs are Core's, whose own
tests and heterogeneous-expert-dispatch's run the full join): a post lists the CPU tier's routes (slot -2) in route
order with their expert keys, their tokens in first-use order and those tokens' rows rounded to BF16, raises ready
and counts the job; a post without CPU routes answers the slot itself (skipped) and its scatter writes nothing; a
scatter copies a served slot's rows into the routes' rows and NaN rows where the CPU failed. A captured post and
scatter replay new routes."""
import torch

from crit_kernels import build as kbuild

cr = kbuild.load("cpu-routes")
dev = torch.device("cuda")
HIDDEN, ROUTES, TOKENS = 96, 16, 8
SERVED, SKIPPED, FAILED = 1, 2, 3


class Slot:
    """A handoff slot's words (ready, done, status) and payload in pinned host memory, read by the device."""

    def __init__(self):
        self.ready, self.done, self.status = (torch.zeros(16, dtype=torch.int32).pin_memory() for _ in range(3))
        self.layout = cr.layout(HIDDEN, ROUTES, TOKENS)
        self.payload = torch.zeros(self.layout["bytes"], dtype=torch.uint8).pin_memory()
        self.routes = cr.Routes(HIDDEN, ROUTES, TOKENS, [(self.ready.data_ptr(), self.done.data_ptr(),
                                                          self.status.data_ptr(), self.payload.data_ptr())])

    def part(self, name, count, dtype, width=1):
        at = self.layout[name]
        size = torch.tensor([], dtype=dtype).element_size()
        return self.payload[at:at + count * width * size].view(dtype).view(count, width) if width > 1 else \
            self.payload[at:at + count * size].view(dtype)


def buffers():
    return (torch.zeros(ROUTES, dtype=torch.int32, device=dev), torch.zeros(2, dtype=torch.int32, device=dev),
            torch.zeros(2, dtype=torch.int64, device=dev), torch.zeros(1, dtype=torch.int32, device=dev))


def test_the_layout_is_64_byte_aligned_parts():
    found = cr.layout(HIDDEN, ROUTES, TOKENS)
    assert found["experts"] == 64 and all(found[k] % 64 == 0 for k in ("token", "x", "out"))
    assert found["bytes"] == found["out"] + ROUTES * HIDDEN * 4


def test_a_post_lists_the_cpu_routes_and_their_tokens_rows():
    s = Slot()
    slots = torch.tensor([0, -2, 1, -2, 2], dtype=torch.int32, device=dev)   # experts 1 and 3 are the CPU's
    ids = torch.tensor([[0, 1], [3, 2], [4, 0], [1, 3]], device=dev)
    x = torch.randn(4, HIDDEN, device=dev)
    cpu_routes, cpu_count, counters, status = buffers()
    s.routes.post(0, ids, slots, x, 100, cpu_routes, cpu_count, counters)
    torch.cuda.synchronize()
    assert cpu_count.tolist() == [4, 0] and cpu_routes[:4].tolist() == [1, 2, 6, 7]
    assert s.part("experts", 4, torch.int32).tolist() == [101, 103, 101, 103]
    assert s.part("token", 4, torch.int32).tolist() == [0, 1, 2, 2]   # tokens 0, 1 and 3, in first use
    rows = s.part("x", 3, torch.float32, HIDDEN)
    assert torch.equal(rows, x[[0, 1, 3]].bfloat16().float().cpu())
    assert int(s.ready[0]) == 1 and counters.tolist() == [1, 4]
    s.status[0] = SERVED   # the CPU's answer: route i's row i + 0.5
    out = s.part("out", 4, torch.float32, HIDDEN)
    out.copy_(torch.arange(4, dtype=torch.float32)[:, None] + 0.5)
    y = torch.zeros(8, HIDDEN, device=dev)
    s.routes.scatter(0, y, cpu_routes, cpu_count, status, ids.numel())
    torch.cuda.synchronize()
    assert int(status) == SERVED
    for i, r in enumerate([1, 2, 6, 7]):
        assert bool((y[r] == i + 0.5).all())
    assert not y[[0, 3, 4, 5]].any()
    s.status[0] = FAILED
    s.routes.scatter(0, y, cpu_routes, cpu_count, status, ids.numel())
    torch.cuda.synchronize()
    assert int(status) == FAILED and bool(y[[1, 2, 6, 7]].isnan().all()) and not y[[0, 3, 4, 5]].any()


def test_a_post_without_cpu_routes_answers_the_slot_itself():
    s = Slot()
    slots = torch.zeros(4, dtype=torch.int32, device=dev)
    ids = torch.tensor([[0, 1], [2, 3]], device=dev)
    cpu_routes, cpu_count, counters, status = buffers()
    s.routes.post(0, ids, slots, torch.randn(2, HIDDEN, device=dev), 0, cpu_routes, cpu_count, counters)
    y = torch.zeros(4, HIDDEN, device=dev)
    s.routes.scatter(0, y, cpu_routes, cpu_count, status, ids.numel())
    torch.cuda.synchronize()
    assert cpu_count.tolist() == [0, 0] and int(s.status[0]) == SKIPPED and int(s.ready[0]) == 0
    assert int(status) == SKIPPED and not y.any() and counters.tolist() == [0, 0]


def test_a_captured_post_and_scatter_replay_new_routes():
    """Two graphs, the CPU's answer between them (in an engine the handoff's join is in the stream between)."""
    s = Slot()
    slots = torch.tensor([-2, 0, 1, 2], dtype=torch.int32, device=dev)
    ids = torch.tensor([[1, 2], [3, 1]], device=dev)
    x = torch.randn(2, HIDDEN, device=dev)
    y = torch.zeros(4, HIDDEN, device=dev)
    cpu_routes, cpu_count, counters, status = buffers()
    post = lambda: s.routes.post(0, ids, slots, x, 0, cpu_routes, cpu_count, counters)  # noqa: E731
    scatter = lambda: s.routes.scatter(0, y, cpu_routes, cpu_count, status, ids.numel())  # noqa: E731
    post(), scatter()
    torch.cuda.synchronize()
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for graph, call in zip(graphs, (post, scatter)):
        with torch.cuda.graph(graph):
            call()
    ids.copy_(torch.tensor([[1, 0], [3, 1]], device=dev))   # route 1 is now expert 0, the CPU's
    graphs[0].replay()
    torch.cuda.synchronize()
    assert cpu_count.tolist() == [1, 0] and cpu_routes[0].item() == 1 and int(s.ready[0]) == 1
    assert torch.equal(s.part("x", 1, torch.float32, HIDDEN), x[:1].bfloat16().float().cpu())
    s.part("out", 1, torch.float32, HIDDEN).fill_(7.0)
    s.status[0] = SERVED
    graphs[1].replay()
    torch.cuda.synchronize()
    assert int(status) == SERVED and bool((y[1] == 7.0).all()) and not y[[0, 2, 3]].any()
