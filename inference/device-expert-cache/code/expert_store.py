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
