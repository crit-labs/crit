"""The host store a device expert cache reads: one contiguous bundle per (group, expert) key, in key order.

A checkpoint keeps an expert's matrices and scales as separate tensors, often spread over shards, or stacks every
expert of a layer in one tensor. build_store reads them once and writes one file of groups * experts bundles
(key = group * experts + expert, every bundle the same size, a multiple of `align` bytes) plus an index beside it
naming each component's offset in the bundle. The index carries the checkpoint's fingerprint (its shards' header
digests and sizes), so a store whose source is unchanged is reused instead of rebuilt.

open_store maps the file and register pins it for the device (mapped pinned memory, torch's cudart): registering a
mapped file takes seconds where allocating and filling tens of GB of pinned memory took minutes on one machine. With
unified addressing the device pointer of registered memory is its host pointer; a native engine maps the same file
and registers the mapping, which returns both.

A component names its tensor by a template over {group} and {expert} (for example
"model.layers.{group}.mlp.experts.{expert}.down_proj.weight"); a template without {expert} names a tensor that
stacks every expert along its first axis, and each expert's slice is its contiguous 1 / experts share.
"""
import ctypes
from dataclasses import dataclass
import json
import mmap
import os
from pathlib import Path


KIND = "device-expert-cache-store"
ALIGNMENT = 16  # bundle components start at multiples of 16 bytes (the cache copies 16-byte words)


@dataclass(frozen=True)
class Component:
    name: str        # the component's name inside a bundle (gate, up, down, gate_scale, ...)
    template: str    # tensor name template over {group} and {expert}


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def fingerprint(checkpoint):
    """The checkpoint's identity for the store: each shard's name, size and header digest."""
    return [{"shard": Path(s.path).name, "size": s.size, "header_sha256": s.header_sha256} for s in checkpoint.shards]


def _extent(checkpoint, component, group, expert, experts):
    """(shard path, file offset, nbytes, dtype, shape) of one expert's slice of a component."""
    stacked = "{expert}" not in component.template
    name = component.template.format(group=group, expert=expert)
    tensor = checkpoint.tensors.get(name)
    if tensor is None:
        raise ValueError(f"{component.name}: the checkpoint has no tensor {name}")
    if not stacked:
        return tensor.shard, tensor.data_begin, tensor.nbytes, tensor.dtype, tuple(tensor.shape)
    if not tensor.shape or tensor.shape[0] != experts or tensor.nbytes % experts:
        raise ValueError(f"{name} does not stack {experts} experts along its first axis")
    share = tensor.nbytes // experts
    return tensor.shard, tensor.data_begin + expert * share, share, tensor.dtype, tuple(tensor.shape[1:])


def layout(checkpoint, components, groups, experts, align=256):
    """The bundle layout: {"components": [{name, offset, nbytes, dtype, shape}], "bundle_bytes"}, from group 0,
    expert 0, checked to hold for every key."""
    if groups < 1 or experts < 1 or not components or align < ALIGNMENT or align % ALIGNMENT:
        raise ValueError("positive geometry, components and an alignment that is a multiple of 16")
    rows, offset = [], 0
    for c in components:
        _, _, nbytes, dtype, shape = _extent(checkpoint, c, 0, 0, experts)
        rows.append({"name": c.name, "offset": offset, "nbytes": nbytes, "dtype": dtype, "shape": list(shape)})
        offset = _round_up(offset + nbytes, ALIGNMENT)
    for g in range(groups):
        for e in range(experts):
            for c, row in zip(components, rows):
                _, _, nbytes, dtype, shape = _extent(checkpoint, c, g, e, experts)
                if (nbytes, dtype, list(shape)) != (row["nbytes"], row["dtype"], row["shape"]):
                    raise ValueError(f"{c.name} of group {g}, expert {e} differs from group 0, expert 0's")
    return {"components": rows, "bundle_bytes": _round_up(offset, align)}


def build_store(checkpoint, components, groups, experts, out, align=256, progress=None):
    """Writes out (the bundles) and out + ".json" (the index) from a crit_core.model.safetensors checkpoint
    (inspect_model). Returns the index. A store whose index matches this checkpoint, geometry and layout is kept."""
    out = Path(out)
    plan = layout(checkpoint, components, groups, experts, align)
    index = {"kind": KIND, "groups": groups, "experts": experts, **plan,
             "templates": {c.name: c.template for c in components}, "source": fingerprint(checkpoint)}
    index_path = Path(str(out) + ".json")
    if out.is_file() and index_path.is_file():
        try:
            if json.loads(index_path.read_text(encoding="utf-8")) == index and \
                    out.stat().st_size == groups * experts * plan["bundle_bytes"]:
                return index
        except (OSError, ValueError):
            pass
    partial = Path(str(out) + ".partial")
    handles = {}
    try:
        with open(partial, "wb") as dst:
            for g in range(groups):
                for e in range(experts):
                    bundle = bytearray(plan["bundle_bytes"])
                    for c, row in zip(components, plan["components"]):
                        shard, begin, nbytes, _, _ = _extent(checkpoint, c, g, e, experts)
                        src = handles.get(shard) or handles.setdefault(shard, open(shard, "rb"))
                        src.seek(begin)
                        data = src.read(nbytes)
                        if len(data) != nbytes:
                            raise ValueError(f"{shard} ends inside {c.template.format(group=g, expert=e)}")
                        bundle[row["offset"]:row["offset"] + nbytes] = data
                    dst.write(bundle)
                if progress is not None:
                    progress(g + 1, groups)
    finally:
        for handle in handles.values():
            handle.close()
    os.replace(partial, out)
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


class Store:
    """A store file mapped copy-on-write (pages stay the file's until written, which nothing does), with its index.
    bundle(key) is a memoryview of one bundle."""
    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads(Path(str(self.path) + ".json").read_text(encoding="utf-8"))
        if self.index.get("kind") != KIND:
            raise ValueError("not a device expert cache store")
        self.bundle_bytes = self.index["bundle_bytes"]
        self.keys = self.index["groups"] * self.index["experts"]
        size = self.path.stat().st_size
        if size != self.keys * self.bundle_bytes:
            raise ValueError(f"{self.path} holds {size} bytes, its index {self.keys * self.bundle_bytes}")
        self._file = open(self.path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)
        self._registered = None

    def bundle(self, key):
        if not 0 <= key < self.keys:
            raise ValueError("key outside the store")
        return memoryview(self._map)[key * self.bundle_bytes:(key + 1) * self.bundle_bytes]

    def component(self, key, name):
        row = next(c for c in self.index["components"] if c["name"] == name)
        return self.bundle(key)[row["offset"]:row["offset"] + row["nbytes"]]

    def address(self):
        """The mapping's host address."""
        import ctypes
        anchor = ctypes.c_char.from_buffer(self._map)
        address = ctypes.addressof(anchor)
        del anchor   # the mapping can be closed again only once no ctypes object exports its buffer
        return address

    def register(self, cudart=None):
        """Pins the mapping for the device (cudaHostRegister, mapped and read-only) and returns its device address;
        with unified addressing that is the host address. cudart: torch.cuda.cudart() unless given."""
        if self._registered is None:
            if cudart is None:
                import torch
                cudart = torch.cuda.cudart()
            address = self.address()
            flags = 0x02 | 0x08   # cudaHostRegisterMapped | cudaHostRegisterReadOnly
            error = cudart.cudaHostRegister(address, len(self._map), flags)
            if int(error) != 0:
                raise RuntimeError(f"cudaHostRegister of {self.path} failed ({error})")
            self._registered = (cudart, address)
        return self._registered[1]

    def close(self):
        if self._registered is not None:
            cudart, address = self._registered
            cudart.cudaHostUnregister(address)
            self._registered = None
        self._map.close()
        self._file.close()


def open_store(path):
    return Store(path)


BUNDLES = "crit-bundle-store"


# The decode of ModelOpt's CSF scales is Core's (crit_plugins.device): a model plugin that loads a checkpoint's
# compressed scales into plain matrices decodes by the same rule as a store builder here and a fill on the device.
from crit_plugins.device import csf_decode  # noqa: E402,F401


def csf_stack(parts):
    """The CSF of matrices stacked by rows (a checkpoint's gate and up scales into the bundle's gate_up scales): parts
    [(fixed, exceptions)] of [rows_i][groups] each (rows_i multiples of 16, one group count for all); the fixed slabs
    follow each other and each part's exception positions move by the rows before it."""
    import torch
    fixed, words, row = [], [], 0
    width = parts[0][0].shape[-1]
    groups = (width // 16 - 1) * 2
    for part_fixed, part_words in parts:
        if part_fixed.shape[-1] != width:
            raise ValueError("stacked CSF matrices have one group count")
        slabs = part_fixed.reshape(-1, width).to(torch.uint8)
        fixed.append(slabs)
        if part_words is not None and part_words.numel():
            w = part_words.reshape(-1).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            pos = (w & 0xFFFFFF) + row * groups
            if int(pos.max()) >= 1 << 24:
                raise ValueError("stacked CSF positions pass the 24 bits an exception word holds")
            moved = pos | (w & 0xFF000000)
            words.append((moved - (moved >= 1 << 31).to(torch.int64) * (1 << 32)).to(torch.int32))
        row += slabs.shape[0] * 16
    out = torch.cat(words) if words else torch.zeros(0, dtype=torch.int32)
    return torch.cat(fixed).contiguous(), out


def csf_encode(scales):
    """(fixed, exceptions) of E4M3 scale bytes [rows][groups] in the CSF layout csf_decode reads: each row's base its
    smallest byte, 4-bit codes above it, an exception word for every byte more than 15 above (an encoder for stores
    built from plain scales, and for tests; ModelOpt's encoder chooses its bases its own way)."""
    import torch
    scales = scales.to(torch.uint8).cpu()
    rows, groups = scales.shape
    if rows % 16 or groups % 2:
        raise ValueError("CSF encodes [rows][groups] with rows a multiple of 16 and groups even")
    base = scales.min(1).values
    codes = scales.to(torch.int32) - base.to(torch.int32)[:, None]
    far = codes > 15
    where = far.nonzero()
    words = (where[:, 0] * groups + where[:, 1]) | (scales[far].to(torch.int64) << 24)
    codes = torch.where(far, torch.zeros_like(codes), codes).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).reshape(rows // 16, 16 * (groups // 2))
    fixed = torch.cat([base.reshape(rows // 16, 16), packed], 1).contiguous()
    return fixed, (words - (words >= 1 << 31).to(torch.int64) * (1 << 32)).to(torch.int32)


COPY, CSF = 0, 1    # fill program ops


def _component_bytes(dtype, shape):
    import math
    import torch
    kind = dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype).replace("torch.", ""))
    return math.prod(shape) * torch.empty(0, dtype=kind).element_size()


def csf_layout(layout, scales):
    """The store layout of bundles that keep some scale components CSF-compressed (ModelOpt's encoding: about half
    the bytes of E4M3 scales) and the fill program that rebuilds the slot layout from them.

    layout: the slot layout (the experts plugin's layout(): {"bundle_bytes", "components": {name: (offset, dtype,
    shape)}}); scales: {component name: exception capacity}, each an E4M3 scale matrix [rows][groups] (rows a multiple
    of 16, groups even) stored as its CSF fixed bytes and up to `capacity` exception words (unused words -1). Returns
    {"bundle_bytes": the store bundle's bytes, "slot_bytes", "components": {name: (offset, nbytes)} of the store
    bundle, "program": [[op, src, dst, a, b, c, d, 0]]}: op COPY (src offset, dst offset, bytes) for every other
    component, op CSF (fixed offset, dst offset, rows, groups, exceptions offset, capacity) for each compressed one.
    The device expert cache runs the program in its fill (crit's kernel expert-directory's fill_program, or the torch ops)."""
    parts = sorted(layout["components"].items(), key=lambda item: item[1][0])
    unknown = set(scales) - {name for name, _ in parts}
    if unknown:
        raise ValueError(f"no components {sorted(unknown)} in the layout")
    at, components, program, names = 0, {}, [], []
    for name, (offset, dtype, shape) in parts:
        names.append(name)
        if name in scales:
            rows, groups = shape
            if rows % 16 or groups % 2 or _component_bytes(dtype, shape) != rows * groups:
                raise ValueError(f"{name}: CSF compresses byte scales [rows][groups], rows a multiple of 16")
            fixed = rows * (1 + groups // 2)
            capacity = int(scales[name])
            components[name + ".csf"] = (at, fixed)
            exceptions = at + -(-fixed // 16) * 16
            components[name + ".exceptions"] = (exceptions, 4 * capacity)
            program.append([CSF, at, offset, rows, groups, exceptions, capacity, 0])
            at = exceptions + -(-(4 * capacity) // 16) * 16
        else:
            nbytes = _component_bytes(dtype, shape)
            components[name] = (at, nbytes)
            program.append([COPY, at, offset, nbytes, 0, 0, 0, 0])
            at += -(-nbytes // 16) * 16
    return {"bundle_bytes": max(16, at), "slot_bytes": layout["bundle_bytes"], "components": components,
            "program": program, "names": names, "compressed": sorted(scales)}


def csf_scales(text):
    """An experts:store plugin's csf_scales option -> {component: exception capacity}, or None (no component kept
    compressed): "gate_up_scale:256,down_scale:128" names the slot layout's E4M3 scale components to keep CSF-compressed
    in the store and the exception words each bundle holds for each."""
    found = {}
    for part in (p.strip() for p in str(text).split(",")):
        if not part:
            continue
        name, colon, capacity = part.rpartition(":")
        if not colon or not name or not capacity.strip().isdigit():
            raise ValueError(f"csf_scales: component:capacity pairs, comma-separated, not {part!r}")
        found[name.strip()] = int(capacity)
    return found or None


def store_layout(layout, scales):
    """The layout a store keeps for the experts plugin's slot layout: the slot layout itself, or with scales (the
    csf_scales option's {component: capacity}) csf_layout's, whose fill program the cache runs to rebuild slots."""
    return csf_layout(layout, scales) if scales else dict(layout)


def store_bundle(layout, bundle, compressed=None):
    """The bytes a store keeps for a slot-layout bundle (the experts plugin's pack()): the bundle itself in a plain
    layout, else csf_bundle's. compressed: {component: (fixed, exceptions)}, or a list of such parts stacked by rows
    (a checkpoint's gate and up scales: csf_stack), for components the checkpoint holds CSF-compressed; the others of
    layout["compressed"] are encoded from the bundle's own scale bytes (csf_encode)."""
    import torch
    if "program" not in layout:
        if compressed:
            raise ValueError("the store keeps no compressed scales (its plugin's csf_scales option is empty): put the "
                             "slot-layout bundle alone")
        return bundle
    unknown = set(compressed or ()) - set(layout["compressed"])
    if unknown:
        raise ValueError(f"the store keeps {layout['compressed']} compressed, not {sorted(unknown)}")
    slot = torch.as_tensor(bundle).detach().reshape(-1).cpu() if hasattr(bundle, "detach") else \
        torch.frombuffer(bytearray(bundle), dtype=torch.uint8)
    if slot.numel() != layout["slot_bytes"]:
        raise ValueError(f"a slot-layout bundle is {layout['slot_bytes']} bytes, not {slot.numel()}")
    parts = {}
    for name, (op, _, dst, rows, groups, _, _, _) in zip(layout["names"], layout["program"]):
        if op != CSF:
            continue
        given = (compressed or {}).get(name)
        if given is None:
            parts[name] = csf_encode(slot[dst:dst + rows * groups].view(rows, groups))
        elif isinstance(given, list):
            parts[name] = csf_stack(given)
        else:
            parts[name] = given
    return csf_bundle(slot, layout, parts)


def csf_bundle(slot_bundle, store_layout, compressed):
    """A store bundle (uint8 [store bundle_bytes]) from a slot-layout bundle (the experts plugin's pack(), whose
    compressed scale components are not read: any bytes) and compressed {name: (fixed uint8, exceptions int32)}."""
    import torch
    out = torch.zeros(store_layout["bundle_bytes"], dtype=torch.uint8)
    slot = slot_bundle.detach().reshape(-1).cpu()
    for base, (op, src, dst, a, b, c, d, _) in zip(store_layout["names"], store_layout["program"]):
        if op == COPY:
            out[src:src + a] = slot[dst:dst + a]
            continue
        fixed, exceptions = compressed[base]
        fixed = fixed.reshape(-1).view(torch.uint8).cpu()
        if fixed.numel() != a * (1 + b // 2):
            raise ValueError(f"{base}: CSF fixed bytes of {a} rows x {b} groups expected")
        words = torch.as_tensor(exceptions).reshape(-1).view(torch.int32).cpu() if exceptions is not None else \
            torch.zeros(0, dtype=torch.int32)
        if words.numel() > d:
            raise ValueError(f"{base}: {words.numel()} exceptions, the store holds {d}")
        out[src:src + fixed.numel()] = fixed
        padded = torch.full((d,), -1, dtype=torch.int32)
        padded[:words.numel()] = words
        out[c:c + 4 * d] = padded.view(torch.uint8)
    return out


def _host_memory():
    try:
        from . import host_memory
    except ImportError:
        import host_memory
    return host_memory


READ_PIECE = 64 << 20    # a store file is read into memory in direct reads of this many bytes each
READ_BATCH = 32          # and the seal says how far it is after this many of them


class BundleStore:
    """Bundles of every (layer, expert) key in host memory a device reads (crit_plugins.device DEVICE_EXTENSIONS
    "experts:store"): key = layer * experts + expert, bundle_bytes each, in key order, as the experts plugin's layout()
    and pack() define them. Its bytes are anonymous host memory of its own, built by put(), its pages kept in host
    memory from the moment they are written (host_memory.lock). With `path` the seal writes the built store to a
    file, once and past the OS's page cache (Core's write_memory), and a complete file with the same layout is not
    built again: the seal reads it into the store's memory, by direct reads at a queue depth (Core's host reads).
    The first region() or address() seals the store: its memory is registered as mapped memory for `device`, and
    from then on put() is refused.

    The file is how the bundles outlast the process, never the memory the device reads. A file's mapping is the
    OS's page cache, which a registration faults in a page at a time and which the OS may drop meanwhile (one that
    returns unused cache to its host does): a 65 GiB store just written, or read in ahead of the registration, came
    off the disk again at 40 MB/s, 27 minutes of a load. Read past the page cache into memory of its own it takes
    20 seconds, at the drive's rate, and a store just built is registered where put() left it. Nor does the file
    pass through the page cache on its way out: a store that fills host memory, written through it bundle by
    bundle, left the host its swap to make room in.

    region() -> (host uint8 tensor [keys * bundle_bytes], CUDA uint8 tensor over its device address); address(layer,
    expert) -> a bundle's device address; read(keys, into) copies bundles into a host or device tensor [n][bundle_bytes].

    leave(keys), before the seal, names bundles the host is not to hold: a device cache keeps them for good (its
    pinned core), so each expert is in one place. A store on a file maps their pages onto a filler (host_memory
    leave_out) and the region stays one range with one device address; read(), bundle() and peek() give such a
    bundle from the file, and the region's bytes there are the filler's, which nothing may read.
    """

    mapped = True    # region() and address() are device-readable (a file tier's are not)
    kept = False     # kept for the next engine by its keeper (expert-tensors' open_store(kept=)), who closes it

    def __init__(self, layout, layers, experts, device, path=None, identity=None):
        """identity: the checkpoint's identity (a fingerprint: shard names, sizes, header digests) a store file is
        bound to; a file is reopened without puts only when its index names this identity and layout. A path needs
        one; a path another open store maps is refused."""
        if path and identity is None:
            raise ValueError("a bundle store file needs the identity of the checkpoint it holds")
        self.layout = dict(layout)
        self.layout["bundle_bytes"] = int(self.layout["bundle_bytes"])
        self.bundle_bytes = self.layout["bundle_bytes"]
        if self.bundle_bytes < 16 or self.bundle_bytes % 16 or layers < 1 or experts < 1:
            raise ValueError("bundles of a multiple of 16 bytes, layers and experts")
        self.layers, self.experts = int(layers), int(experts)
        self.keys = self.layers * self.experts
        self.size = self.keys * self.bundle_bytes
        self.compressed = tuple(self.layout["compressed"]) if "program" in self.layout else ()
        self.device = device
        self.path = Path(path) if path else None
        self._memory = _host_memory()
        self._mapping = None
        self._host = self._device = None
        self._registered = None
        self.working = None        # the owner's working(what, done, total): the seal says how far its read is
        self.left = frozenset()    # keys whose bundles the host does not hold (leave())
        self.left_bytes = 0        # the host bytes that saved
        self._file = None          # the store file's reads (left bundles; peek())
        self.complete = False
        self._index = {"kind": BUNDLES, "layers": self.layers, "experts": self.experts, **_jsonable(self.layout),
                       "identity": _jsonable(identity)}
        if self.path is not None:
            _claim(self.path)
        if self.path is not None and self._matches():
            self.complete = True    # its memory is made and read in at the seal (leave() says what stays out first)
            return
        if self.path is not None:
            _unlink(Path(str(self.path) + ".json"))   # an index names complete bytes only: gone before they change
        self._mapping = self._memory.map_anonymous(self.size)
        self._memory.lock(self._mapping)

    def _matches(self):
        index = Path(str(self.path) + ".json")
        try:
            return (json.loads(index.read_text(encoding="utf-8")) == self._index
                    and self.path.stat().st_size == self.size)
        except (OSError, ValueError):
            return False

    def key(self, layer, expert):
        if not (0 <= layer < self.layers and 0 <= expert < self.experts):
            raise ValueError(f"expert {expert} of layer {layer} is outside the store")
        return layer * self.experts + expert

    def put(self, layer, expert, bundle, compressed=None):
        """One expert's bundle in the slot layout (a uint8 tensor on any device, or bytes), kept as store_bundle()
        makes it (compressed: the checkpoint's CSF scales of the store's compressed components, for a store that keeps
        some)."""
        if self._registered is not None or self.complete:
            raise ValueError("the bundle store is sealed (registered for the device, or opened complete): put every "
                             "bundle before the first region() or address()")
        bundle = store_bundle(self.layout, bundle, compressed)
        data = bundle.detach().contiguous().cpu().numpy().tobytes() if hasattr(bundle, "numpy") else bytes(bundle)
        if len(data) != self.bundle_bytes:
            raise ValueError(f"a bundle is {self.bundle_bytes} bytes, not {len(data)}")
        at = self.key(layer, expert) * self.bundle_bytes
        ctypes.memmove(self._mapping.address + at, data, len(data))

    def leave(self, keys):
        """Bundles the host is not to hold (a cache's pinned core, layers kept whole on the device). Before the
        seal, for a store on a file; returns the keys left (none where the store has no file to read them from
        again, or is sealed already: every bundle stays on the host)."""
        if self.path is None or self._registered is not None:
            return self.left
        keys = frozenset(int(key) for key in keys)
        if any(not 0 <= key < self.keys for key in keys):
            raise ValueError("a key to leave is outside the store")
        self.left = self.left | keys
        return self.left

    def _leave_out(self):
        """The left bundles' pages out of the mapping, before it is registered."""
        if self.left:
            B = self.bundle_bytes
            self.left_bytes = self._memory.leave_out(self._mapping, [(key * B, (key + 1) * B) for key in self.left])

    def _reader(self):
        """The store file's reads (Core's host: positioned reads at a queue depth, past the OS's page cache where
        the file system allows: a file read into the store's memory is not held twice, and a bundle the host is not
        to hold does not stay in the cache either)."""
        if self._file is None:
            from crit_plugins.host import current
            self._file = current().reads("device-expert-cache", self.path, direct=True)
        return self._file

    def _from_file(self, key):
        """One bundle read from the store file (a host tensor of its own)."""
        import torch
        out = torch.empty(self.bundle_bytes, dtype=torch.uint8)
        self._reader().read([(key * self.bundle_bytes, memoryview(out.numpy()))])
        return out

    def peek(self, layer, expert):
        """One bundle's bytes read from the store file, without sealing the store (a copy); None for a store that
        is not a complete file."""
        if self.path is None or not self.complete:
            return None
        return self._from_file(self.key(layer, expert))

    def _read_file(self):
        """A complete store file into the store's memory, the left bundles' whole pages apart (the filler's): direct
        reads of READ_PIECE bytes at a queue depth, each landing where it goes."""
        B = self.bundle_bytes
        left = self._memory.whole_pages([(key * B, (key + 1) * B) for key in self.left], self.size)
        pieces, at = [], 0
        for first, end in left + [(self.size, self.size)]:
            while at < first:   # up to the next left range (or the end), a piece at a time
                n = min(READ_PIECE, first - at)
                pieces.append((at, n))
                at += n
            at = max(at, end)
        reads, address, total, done = self._reader(), self._mapping.address, sum(n for _, n in pieces), 0
        for first in range(0, len(pieces), READ_BATCH):
            batch = pieces[first:first + READ_BATCH]
            reads.read([(start, (ctypes.c_uint8 * n).from_address(address + start)) for start, n in batch])
            done += sum(n for _, n in batch)
            if self.working is not None:
                self.working("expert store: read into host memory, MiB", done >> 20, total >> 20)

    def _seal(self):
        if self._registered is not None:
            return
        memory = self._memory
        if self.path is not None and not self.complete:   # built here: the file a later load reads in
            from crit_plugins.host import write_memory
            report = self.working
            write_memory(self.path, self._mapping.address, self.size, None if report is None else (
                lambda done, total: report("expert store: written to its file, MiB", done >> 20, total >> 20)))
            Path(str(self.path) + ".json").write_text(json.dumps(self._index, indent=1), encoding="utf-8")
            self.complete = True
        stored = self._mapping is None   # a complete file opened without puts: read into memory now
        if stored:
            self._mapping = memory.map_anonymous(self.size)
            memory.lock(self._mapping)
        self._leave_out()                # before the registration, which holds the pages it finds
        if stored:
            self._read_file()
        cuda = str(self.device).startswith("cuda")
        address = memory.register(self._mapping.address, self.size) if cuda else None
        self._registered = address if address is not None else 0
        self._host = memory.host_tensor(self._mapping.address, self.size)
        self._device = memory.device_tensor(address, self.size, self.device) if cuda else None

    @property
    def sealed(self):
        """Registered for the device: its bundles are all there and put() is refused (a kept store an earlier engine
        sealed, say)."""
        return self._registered is not None

    def region(self):
        """(host tensor, device tensor) over the whole store; the device tensor is None on a host without CUDA."""
        self._seal()
        return self._host, self._device

    def address(self, layer, expert):
        """The device address of one bundle (the host address without CUDA)."""
        self._seal()
        base = self._registered or self._mapping.address
        return base + self.key(layer, expert) * self.bundle_bytes

    def bundle(self, layer, expert):
        """One bundle's bytes as a host tensor view (a left bundle: its bytes read from the file)."""
        self._seal()
        key = self.key(layer, expert)
        if key in self.left:
            return self._from_file(key)
        at = key * self.bundle_bytes
        return self._host[at:at + self.bundle_bytes]

    def read(self, keys, into):
        """Bundles of `keys` into `into` ([n][bundle_bytes] uint8, host or device), in order."""
        self._seal()
        host = self._host.view(self.keys, self.bundle_bytes)
        keys = [int(key) for key in keys]
        left = [i for i, key in enumerate(keys) if key in self.left]
        if left and not into.is_cuda:   # from the file, all at once, into the caller's rows
            self._reader().read([(keys[i] * self.bundle_bytes, memoryview(into[i].numpy())) for i in left])
        for i, key in enumerate(keys):
            if key not in self.left:
                into[i].copy_(host[key], non_blocking=into.is_cuda)
            elif into.is_cuda:
                into[i].copy_(self._from_file(key))

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._registered:
            self._memory.unregister(self._mapping.address)
        self._registered = None
        self._host = self._device = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self.path is not None:
            _release(self.path)
            self.path = None


_OPEN = set()   # store files open in this process


def _claim(path):
    key = str(Path(path).resolve())
    if key in _OPEN:
        raise ValueError(f"{path} is open as another bundle store")
    _OPEN.add(key)


def _release(path):
    _OPEN.discard(str(Path(path).resolve()))


def _unlink(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _jsonable(value):
    """The layout as JSON reads it back (tuples as lists, dtypes as their names)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def open_bundles(layout, layers, experts, device, path=None, identity=None):
    """A BundleStore (the "experts:store" extension's open_store)."""
    return BundleStore(layout, layers, experts, device, path, identity)


def row_slot_bytes(row_bytes):
    """The slot bytes a row of `row_bytes` takes in a cache: the row, padded to a multiple of 16 (the cache copies
    16-byte words)."""
    if int(row_bytes) < 1:
        raise ValueError("a row holds one byte or more")
    return _round_up(int(row_bytes), ALIGNMENT)


def open_rows(groups, keys, row_bytes, device, rows, path=None, identity=None):
    """A BundleStore of opaque rows in the caller's own layout (the "experts:rows" extension's row_store): `groups` x
    `keys` rows of `row_bytes` bytes each (any count, padded to row_slot_bytes in the store), key = group * keys + key,
    from `rows` ([groups * keys][row_bytes] uint8, a tensor or a sequence of bytes-like rows, in key order). The store
    knows nothing of what a row holds: an expert in a format of the model's own, a table row, any bytes the caller reads
    back from a cache slot (ExpertCache.rows()). Its row_bytes is the rows' own size."""
    slot = row_slot_bytes(row_bytes)
    store = BundleStore({"bundle_bytes": slot, "row_bytes": int(row_bytes)}, groups, keys, device, path, identity)
    store.row_bytes = int(row_bytes)
    if store.complete:
        return store
    if len(rows) != store.keys:
        store.close()
        raise ValueError(f"the store holds {store.keys} rows, not {len(rows)}")
    pad = bytes(slot - int(row_bytes))
    for key in range(store.keys):
        row = rows[key]
        data = row.detach().contiguous().cpu().numpy().tobytes() if hasattr(row, "numpy") else bytes(row)
        if len(data) != int(row_bytes):
            store.close()
            raise ValueError(f"row {key} is {len(data)} bytes, not {int(row_bytes)}")
        store.put(key // store.experts, key % store.experts, data + pad)
    return store
