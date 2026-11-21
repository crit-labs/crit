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
    def __init__(self, store_path="", csf_scales="", context=None):
        self.experts, self.store_path, self.stores = {}, store_path, []
        self.csf = expert_store.csf_scales(csf_scales)
        self.context = context

    def add(self, layer, expert, gate_up, down):
        if (layer, expert) in self.experts:
            raise ValueError(f"expert {expert} of layer {layer} is added already")
        self.experts[(layer, expert)] = check_expert(gate_up, down)

    def matrices(self, layer, expert):
        try:
            return self.experts[(layer, expert)]
        except KeyError:
            raise KeyError(f"no expert {expert} in layer {layer}") from None

    def kept_bytes(self):
        """The host bytes of the store kept for the next engine and not opened yet (0 for none): memory a store
        opened with kept= has, whether it gets that store back or replaces it."""
        keeping = getattr(self.context, "keeping", None)
        return int(keeping("expert-tensors:store")) if callable(keeping) else 0

    def drop_kept(self):
        """Close the store kept for the next engine, which this one will not open (its experts go elsewhere)."""
        drop = getattr(self.context, "drop_kept", None)
        if callable(drop):
            drop("expert-tensors:store")

    def open_store(self, layout, layers, experts, device, path=None, identity=None, kept=None):
        """A BundleStore of layers x experts bundles of the slot layout `layout` for `device`: put() every bundle,
        then region() or address() (a complete store file at `path`, or the store_path option, for the checkpoint
        `identity`, opens with its bundles). With the csf_scales option the store keeps those scale components
        CSF-compressed (store.layout is csf_layout's, store.compressed names them) and the cache rebuilds the slot
        layout as it fills slots. kept: the key of what the bundles hold beyond these arguments (the checkpoint, the
        experts plugin's packing code); with it the store is kept for the next engine built for the model in this
        host (the context's keep(), crit_core.engine.LoadContext.keep): such an engine gets it back sealed, as the
        last engine left it (store.sealed: no put()), and its keeper closes it (store.kept), never the engine."""
        layout = expert_store.store_layout(layout, self.csf)
        path = path or self.store_path or None

        def build():
            return expert_store.BundleStore(layout, layers, experts, device, path, identity)
        keep = getattr(self.context, "keep", None)
        if kept is not None and callable(keep):
            key = {"kept": kept, "layout": expert_store._jsonable(layout), "layers": int(layers),
                   "experts": int(experts), "device": str(device), "path": None if path is None else str(path),
                   "identity": expert_store._jsonable(identity)}

            def made():
                store = build()
                store.kept = True
                return store
            return keep("expert-tensors:store", key, made, close=lambda store: store.close(),
                        nbytes=int(layers) * int(experts) * int(layout["bundle_bytes"]))
        store = build()
        self.stores.append(store)
        return store

    def close(self):
        for store in self.stores:
            store.close()
        self.stores = []


def create(context, options, parts):
    return ExpertTensors(options["store_path"], options["csf_scales"], context)
