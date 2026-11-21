"""A routed layer's CPU share joined inside a CUDA stream (crit's kernel cpu-routes, which crit's
heterogeneous-expert-dispatch and expert-offload plugins load for their in-stream CPU tier, built on this GPU): one
kernel lists the routes the slot table sends to the CPU tier (slot -2) into a slot of one of the host's handoffs
(Core's crit_plugins.host) and posts it, or answers it itself when there is none; after the stream's join a second
kernel copies the CPU's rows into the device part's route rows (NaN rows and the status word where the CPU failed).
Capture-safe; the payload is host-expert-tier's InputLayout.

    routes = Routes(hidden, routes, tokens, [handoff.words(s) for s in range(slots)])
    routes.post(slot, ids, slots, x, key_base, cpu_routes, cpu_count, counters)
    routes.scatter(slot, y, cpu_routes, cpu_count, status, routes)
"""
from ._ops import ops

__all__ = ["Routes", "layout"]


def layout(hidden, routes, tokens):
    """The payload's offsets {experts, token, x, out, bytes} for `routes` routes and `tokens` tokens of `hidden`."""
    return dict(zip(("experts", "token", "x", "out", "bytes"), (int(v) for v in ops.layout(int(hidden), int(routes),
                                                                                         int(tokens)))))


class Routes:
    """The post and scatter kernels over one handoff's slots: words holds each slot's device addresses (ready, done,
    status, payload), as crit_plugins.host's Handoff.words(slot) gives them. The handoff and its join are the host's."""

    def __init__(self, hidden, routes, tokens, words):
        self.hidden, self.capacity, self.tokens = int(hidden), int(routes), int(tokens)
        self.words = [tuple(int(w) for w in slot) for slot in words]
        if not self.words or any(len(w) != 4 for w in self.words):
            raise ValueError("cpu routes: a handoff's slots, four device addresses each")

    def post(self, slot, ids, slots, x, key_base, cpu_routes, cpu_count, counters):
        ops.post(*self.words[int(slot)], self.hidden, self.capacity, self.tokens, ids, slots, x, int(key_base),
                 cpu_routes, cpu_count, counters)

    def scatter(self, slot, y, cpu_routes, cpu_count, status, routes):
        """routes: the call's route count (tokens x top), the most its post can list."""
        ops.scatter(*self.words[int(slot)], self.hidden, self.capacity, self.tokens, y, cpu_routes, cpu_count, status,
                    int(routes))

    def payload_bytes(self):
        return layout(self.hidden, self.capacity, self.tokens)["bytes"]

    def layout(self):
        """The payload's offsets (experts, token, x, out, bytes), to check against the tier's input_layout."""
        found = layout(self.hidden, self.capacity, self.tokens)
        return [found[k] for k in ("experts", "token", "x", "out", "bytes")]
