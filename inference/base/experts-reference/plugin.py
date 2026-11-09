"""The routed SwiGLU block in the experts slot: expert-streamed-prefill's FP64 reference (reference.py), the one every
experts plugin is compared with, and its device form, the same block in torch over the experts' matrices on the device:
every route's expert gathered by id and its gate/up, SiLU times up and down products in FP64 (an explicit precision of
each call, no global flag read or set) through batched matrix products over chunks of a fixed size, each token's routes
added in route order in FP32, so a token's result does not depend on the rows beside it; no host synchronization, so a
call at a fixed token count captures in a CUDA graph (rounds:experts, and prompts:experts: a captured prompt chunk's
padding, ids -1 as the model masks them, adds exactly nothing, whatever its rows hold)."""
from .prefill import reference


def capturing():
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def check_routes(x, ids, weights):
    """x [tokens][hidden], ids and weights [tokens][top] with one top for every token; returns top."""
    if len(ids) != len(x) or len(weights) != len(x):
        raise ValueError("one ids row and one weights row per token")
    top = len(ids[0]) if ids else 0
    if any(len(row) != top for row in ids) or any(len(row) != top for row in weights):
        raise ValueError("every token has the same number of routes and one weight per route")
    return top


class Matrices:
    """An expert_source's matrices for one call, read once per expert."""

    def __init__(self, source, layer):
        self.source, self.layer, self.cache = source, layer, {}

    def __call__(self, expert):
        if expert not in self.cache:
            self.cache[expert] = self.source.matrices(self.layer, expert)
        return self.cache[expert]

    def gate_up(self, expert):
        return self(expert)[0]

    def down(self, expert):
        return self(expert)[1]

    def inter(self, expert):
        return len(self(expert)[0]) // 2


class ReferenceExperts:
    def __init__(self, source):
        self.source, self.loaded, self.stacked, self.captured = source, {}, {}, set()

    def run(self, layer, x, ids, weights):
        check_routes(x, ids, weights)
        used = [e for row in ids for e in row if e >= 0]
        if not used:
            return [[0.0] * len(row) for row in x]
        matrices = Matrices(self.source, layer)
        return reference.moe_block(x, ids, weights, matrices.gate_up, matrices.down, matrices.inter(used[0]))

    def load(self, layer, expert, gate_up, down, format="bf16"):
        """The device form: an expert's dense matrices, kept as given (on the device, any float dtype). The reference
        takes dense matrices only: a stored format is refused. A load after the layer's first forward writes into its
        stacked matrices in place."""
        if format != "bf16" or isinstance(gate_up, dict) or isinstance(down, dict):
            raise ValueError(f"experts-reference takes dense matrices, not {format!r} stored tensors")
        if gate_up.shape[0] != 2 * down.shape[1] or gate_up.shape[1] != down.shape[0]:
            raise ValueError("gate_up is [2 * inter][hidden] and down [hidden][inter]")
        expert = int(expert)
        stacked = self.stacked.get(layer)
        if stacked is not None and expert < stacked[0].shape[0]:
            if (gate_up.shape, down.shape, gate_up.dtype, down.dtype) != (stacked[0].shape[1:], stacked[1].shape[1:],
                                                                          stacked[0].dtype, stacked[1].dtype):
                raise ValueError(f"layer {layer}'s experts are {stacked[0].dtype} [{list(stacked[0].shape[1:])}] and "
                                 f"[{list(stacked[1].shape[1:])}]")
            stacked[0][expert].copy_(gate_up)   # in place: a graph that captured the layer keeps its memory
            stacked[1][expert].copy_(down)
            return
        if layer in self.captured:
            raise ValueError(f"layer {layer} is in a captured CUDA graph: load its experts before the capture")
        self.loaded[(layer, expert)] = (gate_up, down)
        self.stacked.pop(layer, None)

    def _stacked(self, layer):
        """The layer's experts as [experts][2 * inter][hidden] and [experts][hidden][inter], expert e at index e (built
        once after its loads)."""
        import torch
        if layer not in self.stacked:
            found = {e: m for (l, e), m in self.loaded.items() if l == layer}
            if not found:
                raise KeyError(f"no experts of layer {layer} are loaded")
            count = max(found) + 1
            if len(found) != count:
                raise KeyError(f"layer {layer} lacks experts {sorted(set(range(count)) - set(found))[:8]}")
            gate_up = torch.stack([found[e][0] for e in range(count)])
            down = torch.stack([found[e][1] for e in range(count)])
            self.stacked[layer] = gate_up, down
            for e in range(count):   # the loaded matrices live on as views of the stack: no second copy
                self.loaded[(layer, e)] = (gate_up[e], down[e])
        if layer not in self.captured and capturing():
            self.captured.add(layer)
        return self.stacked[layer]

    def forward(self, layer, x, ids, weights):
        import torch
        tokens, top = ids.shape
        if x.shape[0] != tokens or weights.shape != ids.shape:
            raise ValueError("one ids row and one weights row per token")
        gate_up, down = self._stacked(layer)
        experts, inter, hidden = gate_up.shape[0], down.shape[2], x.shape[1]
        n = tokens * top
        flat = ids.reshape(-1).long()
        valid = (flat >= 0) & (flat < experts)
        expert = torch.where(valid, flat, torch.zeros_like(flat))
        token = torch.arange(n, device=x.device) // top
        chunk = max(1, min(64, (128 << 20) // (3 * hidden * inter * 8)))   # routes per FP64 product, fixed per shape
        padded = -(-n // chunk) * chunk
        if padded > n:
            expert = torch.cat([expert, torch.zeros(padded - n, dtype=expert.dtype, device=x.device)])
            token = torch.cat([token, torch.zeros(padded - n, dtype=token.dtype, device=x.device)])
        xd = x.double()
        rows = torch.zeros(padded, hidden, dtype=torch.float32, device=x.device)
        for c0 in range(0, padded, chunk):
            e, t = expert[c0:c0 + chunk], token[c0:c0 + chunk]
            both = torch.bmm(gate_up.index_select(0, e).double(), xd.index_select(0, t)[:, :, None])[..., 0]
            h = torch.nn.functional.silu(both[:, :inter]) * both[:, inter:]
            rows[c0:c0 + chunk] = torch.bmm(down.index_select(0, e).double(), h[:, :, None])[..., 0].float()
        w = weights.reshape(-1).float()
        contributions = torch.where(valid[:, None], rows[:n] * w[:, None], torch.zeros_like(rows[:n]))   # a skipped route
        contributions = contributions.view(tokens, top, hidden)                                         # adds exactly 0
        # Each token's routes summed in route order.
        total = torch.zeros(tokens, hidden, dtype=torch.float32, device=x.device)
        for j in range(top):
            total = total + contributions[:, j]
        return total


def create(context, options, parts):
    return ReferenceExperts(parts["expert_source"])
