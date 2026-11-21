"""A test stand-in for the experts slot's round form (rounds:experts) for plain BF16 experts, which experts-reference
also provides: the package's experts-reference in every respect, its device form the reference's arithmetic (each token's route
alone, SwiGLU in FP32 over the expert's matrices, the weighted outputs summed in route order) with the expert of each
route gathered on the device by its id instead of read back to the host, so it runs inside a captured round. A
token's bits do not depend on the tokens beside it."""
from crit_plugins.compose import package_plugin


class RoundExperts:
    def __init__(self, inner):
        self.inner, self.stacked = inner, {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def load(self, layer, expert, gate_up, down):
        self.inner.load(layer, expert, gate_up, down)
        self.stacked.pop(layer, None)

    def _layer(self, layer):
        """Every loaded expert's matrices of a layer, stacked by expert id (FP32, as the reference multiplies them)."""
        if layer not in self.stacked:
            import torch
            ids = sorted(e for (l, e) in self.inner.loaded if l == layer)
            if ids != list(range(len(ids))):
                raise KeyError(f"layer {layer}'s experts are not 0 .. {len(ids) - 1}")
            gate_up = torch.stack([self.inner.loaded[(layer, e)][0].float() for e in ids])
            down = torch.stack([self.inner.loaded[(layer, e)][1].float() for e in ids])
            self.stacked[layer] = (gate_up, down)
        return self.stacked[layer]

    def forward(self, layer, x, ids, weights):
        import torch
        tokens, top = ids.shape
        if x.shape[0] != tokens or weights.shape != ids.shape:
            raise ValueError("one ids row and one weights row per token")
        gate_up, down = self._layer(layer)
        total = torch.zeros(tokens, x.shape[1], dtype=torch.float32, device=x.device)
        rows = []
        for row in range(tokens):
            found = None
            for slot in range(top):
                expert = ids[row, slot:slot + 1]
                gate, up = (x[row:row + 1].float() @ gate_up.index_select(0, expert)[0].T).chunk(2, -1)
                out = (torch.nn.functional.silu(gate) * up) @ down.index_select(0, expert)[0].T
                part = out[0] * weights[row, slot].float()
                found = part if found is None else found + part
            rows.append(found)
        if rows:
            total = torch.stack(rows)
        return total


def create(context, options, parts):
    return RoundExperts(package_plugin(context, options["inner"], parts))
