"""expert-directory's sweep and gates bench: the fill, a plan's copies of whole expert bundles into cache slots, at the
model's NVFP4 bundle size (gate, up and down: 4-bit values and an FP8 scale per 16), from a store in device memory
(an engine's store is registered host memory read over the link, where the link bounds the fill; this bench times the
kernel itself). Bundle i is a function of i alone and copy i goes to slot i, listed last first, so a row's bytes do not
depend on the rows beside it. The plan's decisions are device_cache.LayerDirectory's, which device-expert-cache's tests
check plan for plan against its host twin."""
import torch


def bundle_bytes(hidden, intermediate):
    """An expert's NVFP4 bundle (gate, up, down: hidden x intermediate each, 4 bits a value and an FP8 scale per 16),
    padded to 16 bytes."""
    size = 3 * hidden * intermediate // 2 + 3 * hidden * intermediate // 16
    return (size + 15) // 16 * 16


class Fill:
    structure = "fill"

    def __init__(self):
        self._empty = None

    def shapes(self, model):
        hidden = model.get("hidden_size")
        intermediate = model.get("moe_intermediate_size", "expert_intermediate_size")
        if not hidden or not intermediate:
            return []
        return [{"bundle": bundle_bytes(int(hidden), int(intermediate))}]

    def inputs(self, shape, rows, device):
        bundle = int(shape["bundle"])
        keys = torch.arange(rows, device=device, dtype=torch.int64)
        source = ((keys[:, None] * 131 + torch.arange(bundle, device=device)[None, :] * 7) % 251).to(torch.uint8)
        jobs = torch.zeros(rows + 1, 2, dtype=torch.int32, device=device)
        order = torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int32)
        jobs[:rows, 0] = order
        jobs[:rows, 1] = order
        return {"source": source, "store": torch.zeros_like(source), "jobs": jobs,
                "count": torch.tensor([rows], dtype=torch.int32, device=device), "keys": keys,
                "first": torch.empty(0, dtype=torch.int32, device=device)}

    def run(self, module, s):
        module.fill(s["source"], s["store"], s["jobs"], s["count"], 0, 16, s["first"])
        return s["store"]

    def reference(self, s):
        return s["source"]   # every key copied into the slot of its number

    def check(self, got, want, s):
        if torch.equal(got, want):
            return None
        bad = int((got != want).any(dim=1).sum())
        return f"{bad} of {got.shape[0]} slots differ from their bundles"

    def first_row(self, out):
        return out[:1]

    def bytes(self, shape, rows):
        return 2 * rows * int(shape["bundle"])

    def yardstick(self, s):
        source, store, keys = s["source"], s["store"], s["keys"]
        return lambda: store.index_copy_(0, keys, source.index_select(0, keys))
