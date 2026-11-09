"""Expert weights in the expert_source slot, held as given: the store every experts plugin is compared on. Its bundle
store (the "experts:store" extension) is device-expert-cache's BundleStore: bundles in host memory the device reads."""
from .cache import expert_store


def check_expert(gate_up, down):
    """An expert's [2 * inter][hidden] gate/up and [hidden][inter] down rows, as lists."""
    if not gate_up or len(gate_up) % 2 or not down:
        raise ValueError("gate_up is [2 * inter][hidden] and down [hidden][inter]")
    hidden, inter = len(gate_up[0]), len(gate_up) // 2
    if len(down) != hidden or any(len(row) != hidden for row in gate_up) or any(len(row) != inter for row in down):
        raise ValueError("gate_up is [2 * inter][hidden] and down [hidden][inter]")
    return [[float(v) for v in row] for row in gate_up], [[float(v) for v in row] for row in down]


class ExpertTensors:
    def __init__(self, store_path="", csf_scales=""):
        self.experts, self.store_path, self.stores = {}, store_path, []
        self.csf = expert_store.csf_scales(csf_scales)

    def add(self, layer, expert, gate_up, down):
        if (layer, expert) in self.experts:
            raise ValueError(f"expert {expert} of layer {layer} is added already")
        self.experts[(layer, expert)] = check_expert(gate_up, down)

    def matrices(self, layer, expert):
        try:
            return self.experts[(layer, expert)]
        except KeyError:
            raise KeyError(f"no expert {expert} in layer {layer}") from None

    def open_store(self, layout, layers, experts, device, path=None, identity=None):
        """A BundleStore of layers x experts bundles of the slot layout `layout` for `device`: put() every bundle,
        then region() or address() (a complete store file at `path`, or the store_path option, for the checkpoint
        `identity`, opens with its bundles). With the csf_scales option the store keeps those scale components
        CSF-compressed (store.layout is csf_layout's, store.compressed names them) and the cache rebuilds the slot
        layout as it fills slots."""
        layout = expert_store.store_layout(layout, self.csf)
        store = expert_store.BundleStore(layout, layers, experts, device, path or self.store_path or None, identity)
        self.stores.append(store)
        return store

    def close(self):
        for store in self.stores:
            store.close()
        self.stores = []


def create(context, options, parts):
    return ExpertTensors(options["store_path"], options["csf_scales"])
