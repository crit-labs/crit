"""Experts read from an immutable file in the expert_source slot: ssd-expert-materialization's FileExpertSource behind
its bounded HostExpertTier (implementation.py)."""
import hashlib
import os
import struct
import tempfile

from .cache import expert_store
from .rows import row_reader
from .ssd import bundle_file
from .ssd import implementation as ssd

ENCODING = "f64-rows"   # gate_up then down, row-major, little-endian FP64

def check_expert(gate_up, down):
    """An expert's [2 * inter][hidden] gate/up and [hidden][inter] down rows, as lists."""
    if not gate_up or len(gate_up) % 2 or not down:
        raise ValueError("gate_up is [2 * inter][hidden] and down [hidden][inter]")
    hidden, inter = len(gate_up[0]), len(gate_up) // 2
    if len(down) != hidden or any(len(row) != hidden for row in gate_up) or any(len(row) != inter for row in down):
        raise ValueError("gate_up is [2 * inter][hidden] and down [hidden][inter]")
    return [[float(v) for v in row] for row in gate_up], [[float(v) for v in row] for row in down]


def encode(gate_up, down):
    values = [v for row in gate_up for v in row] + [v for row in down for v in row]
    return struct.pack(f"<{len(values)}d", *values)


def decode(data, inter, hidden):
    values = struct.unpack(f"<{len(data) // 8}d", data)
    cut = 2 * inter * hidden
    gate_up = [list(values[r * hidden:(r + 1) * hidden]) for r in range(2 * inter)]
    down = [list(values[cut + r * inter:cut + (r + 1) * inter]) for r in range(hidden)]
    return gate_up, down


class SSDExperts:
    def __init__(self, host_bytes, directory, readers=16, csf_scales=""):
        self.csf = expert_store.csf_scales(csf_scales)
        handle, self.path = tempfile.mkstemp(prefix="crit-experts-", suffix=".bin", dir=directory or None)
        os.close(handle)
        self.host_bytes, self.directory, self.readers = host_bytes, directory or None, readers
        self.tier = ssd.HostExpertTier(host_bytes) if host_bytes else None
        self.regions, self.shapes, self.size = {}, {}, 0
        self.source = None
        self.stores = []

    def open_store(self, layout, layers, experts, device, path=None, identity=None):
        """The file tier of a bundle store (bundle_file.FileBundleStore) for layers x experts bundles of the slot
        layout `layout`: at `path` (kept, and reopened while its layout holds), or a working file in `directory` removed
        on close. Its own host tier of host_bytes holds recently read bundles. With the csf_scales option the file keeps
        those scale components CSF-compressed (store.compressed names them), rebuilt by the cache as it fills slots."""
        layout = expert_store.store_layout(layout, self.csf)
        owned = path is None
        if owned:
            handle, path = tempfile.mkstemp(prefix="crit-bundles-", suffix=".bin", dir=self.directory)
            os.close(handle)
        tier = ssd.HostExpertTier(self.host_bytes) if self.host_bytes else None
        store = bundle_file.FileBundleStore(layout, layers, experts, path, tier, row_reader, self.readers, owned,
                                            identity, expert_store.store_bundle)
        self.stores.append(store)
        return store

    def add(self, layer, expert, gate_up, down):
        if self.source is not None:
            raise ValueError("the expert file is immutable once read: add every expert before the first matrices()")
        key = (layer, expert)
        if key in self.regions:
            raise ValueError(f"expert {expert} of layer {layer} is added already")
        gate_up, down = check_expert(gate_up, down)
        data = encode(gate_up, down)
        with open(self.path, "ab") as file:
            file.write(data)
        self.regions[key] = ssd.ExpertRegion(self.size, len(data), ENCODING, hashlib.sha256(data).hexdigest())
        self.shapes[key] = (len(gate_up) // 2, len(down))
        self.size += len(data)

    def matrices(self, layer, expert):
        key = (layer, expert)
        if key not in self.regions:
            raise KeyError(f"no expert {expert} in layer {layer}")
        if self.source is None:
            self.source = ssd.FileExpertSource(self.path, self.regions, self.tier)
        return decode(self.source.read(key, protected=(key,)), *self.shapes[key])

    def close(self):
        for store in self.stores:
            store.close()
        self.stores = []
        try:
            if self.source is not None:
                self.source.quiesce()
        finally:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass


def memory(context, options):
    return {"host": options["host_bytes"]}


def create(context, options, parts):
    return SSDExperts(options["host_bytes"], options["directory"], options["readers"], options["csf_scales"])
