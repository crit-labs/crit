"""The host store a device expert cache reads: one contiguous bundle per (group, expert) key, in key order.

A checkpoint keeps an expert's matrices and scales as separate tensors, often spread over shards, or stacks every
expert of a layer in one tensor. build_store reads them once and writes one file of groups * experts bundles
(key = group * experts + expert, every bundle the same size, a multiple of `align` bytes) plus an index beside it
naming each component's offset in the bundle. The index carries the checkpoint's fingerprint (its shards' header
digests and sizes), so a store whose source is unchanged is reused instead of rebuilt.

open_store maps the file and register pins it for the device (mapped pinned memory, torch's cudart): registering a
mapped file takes seconds where allocating and filling tens of GB of pinned memory took minutes on one machine. With
unified addressing the device pointer of registered memory is its host pointer; a native engine maps the same file
and calls Core's cuda_map_host, which returns both (expert_store.h).

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

from crit_plugins.files import pwrite

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


def csf_decode(fixed, exceptions=None):
    """E4M3 scale bytes [rows][groups] (uint8) from ModelOpt's CSF encoding (crit_core.model.schemes.csf_scale_shape):
    fixed uint8 [rows / 16][16 * (1 + groups / 2)], per slab of 16 rows one base byte per row, then each row's 4-bit
    codes (the low nibble first), scale = base + code (mod 256); exceptions (int32 or uint32 words: position row *
    groups + col in the low 24 bits, the scale byte in the high 8) replace single scales. On fixed's device (torch), so
    a store builder decodes on the host and a fill on the device by the same rule; leading dimensions (stacked
    matrices) decode each."""
    import torch
    lead, (slabs, width) = tuple(fixed.shape[:-2]), tuple(fixed.shape[-2:])
    if width % 16 or width < 32:
        raise ValueError("CSF fixed rows are 16 * (1 + groups / 2) bytes")
    groups = (width // 16 - 1) * 2
    f = fixed.reshape(-1, slabs, width).view(torch.uint8)
    m, size = f.shape[0], slabs * 16 * groups
    base = f[:, :, :16].to(torch.int32)
    packed = f[:, :, 16:].reshape(m, slabs, 16, groups // 2).to(torch.int32)
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(m, slabs, 16, groups)
    out = ((base[..., None] + codes) & 255).to(torch.uint8).reshape(m * size)
    if exceptions is not None and exceptions.numel():
        words = exceptions.reshape(m, -1).contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        pos = words & 0xFFFFFF
        valid = (words != 0xFFFFFFFF) & (pos < size)   # -1: an unused word of a fixed-capacity list
        index = torch.where(valid, pos + torch.arange(m, device=f.device)[:, None] * size, m * size)
        padded = torch.cat([out, out.new_zeros(1)])
        padded.index_copy_(0, index.reshape(-1), (words >> 24).to(torch.uint8).reshape(-1))
        out = padded[:-1]
    return out.reshape(*lead, slabs * 16, groups)


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
    The device expert cache runs the program in its fill (directory_ops.cu fill_program, or the torch ops)."""
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


class BundleStore:
    """Bundles of every (layer, expert) key in host memory a device reads (crit_plugins.device DEVICE_EXTENSIONS
    "experts:store"): key = layer * experts + expert, bundle_bytes each, in key order, as the experts plugin's layout()
    and pack() define them. Built by put() into anonymous memory, or into a file at `path` (written with positioned
    writes, then mapped read-only; a complete file with the same layout is opened as it is, without puts). The first
    region() or address() seals it: the memory is registered as mapped memory for `device` (a file read-only where the
    device allows it, else read into anonymous memory that is), and from then on put() is refused.

    region() -> (host uint8 tensor [keys * bundle_bytes], CUDA uint8 tensor over its device address); address(layer,
    expert) -> a bundle's device address; read(keys, into) copies bundles into a host or device tensor [n][bundle_bytes].
    """

    mapped = True    # region() and address() are device-readable (a file tier's are not)

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
        self._mapping = self._fd = None
        self._host = self._device = None
        self._registered = None
        self.complete = False
        self._index = {"kind": BUNDLES, "layers": self.layers, "experts": self.experts, **_jsonable(self.layout),
                       "identity": _jsonable(identity)}
        if self.path is not None:
            _claim(self.path)
        if self.path is not None and self._matches():
            self.complete = True
        elif self.path is not None:
            _unlink(Path(str(self.path) + ".json"))   # an index names complete bytes only: gone before they change
            self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o644)
            os.ftruncate(self._fd, self.size)
        else:
            self._mapping = self._memory.map_anonymous(self.size)

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
        if self._fd is not None:
            pwrite(self._fd, data, at)
        else:
            ctypes.memmove(self._mapping.address + at, data, len(data))

    def _seal(self):
        if self._registered is not None:
            return
        memory = self._memory
        if self._fd is not None:
            os.fsync(self._fd)
            os.close(self._fd)
            self._fd = None
            Path(str(self.path) + ".json").write_text(json.dumps(self._index, indent=1), encoding="utf-8")
            self.complete = True
        if self._mapping is None:
            self._mapping = memory.map_file(self.path, self.size)
        cuda = str(self.device).startswith("cuda")
        address = None
        if cuda:
            import torch
            index = torch.device(self.device).index or 0
            if not self._mapping.writable and memory.read_only_supported(index):
                try:
                    address = memory.register(self._mapping.address, self.size, read_only=True)
                except RuntimeError:
                    address = None
            if address is None:
                if not self._mapping.writable:   # read the file into memory the device may register
                    copy = memory.map_anonymous(self.size)
                    ctypes.memmove(copy.address, self._mapping.address, self.size)
                    self._mapping.close()
                    self._mapping = copy
                address = memory.register(self._mapping.address, self.size)
        self._registered = address if address is not None else 0
        self._host = memory.host_tensor(self._mapping.address, self.size)
        self._device = memory.device_tensor(address, self.size, self.device) if cuda else None

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
        """One bundle's bytes as a host tensor view."""
        self._seal()
        at = self.key(layer, expert) * self.bundle_bytes
        return self._host[at:at + self.bundle_bytes]

    def read(self, keys, into):
        """Bundles of `keys` into `into` ([n][bundle_bytes] uint8, host or device), in order."""
        self._seal()
        host = self._host.view(self.keys, self.bundle_bytes)
        for i, key in enumerate(keys):
            into[i].copy_(host[int(key)], non_blocking=into.is_cuda)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
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
