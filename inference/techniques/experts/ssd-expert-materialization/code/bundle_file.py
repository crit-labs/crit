"""The file tier of a bundle store: every (layer, expert) bundle in one file (key = layer * experts + expert,
bundle_bytes each, in key order), read on demand behind a bounded host tier (HostExpertTier: recently read bundles in
host memory, least recently used out). It has no device-readable region: a device expert cache fills from it with
read(keys, into) (into: pinned memory it then copies to the device), on the host, so a plan over it is not captured
(expert_cache.py says so); a mapped store serves captured rounds. Reads go through a row reader over the file (the
bundles are its rows: lookup-table-rows' native reader, or positioned reads in Python)."""
import json
import os

from crit_plugins.files import pread, pwrite
from pathlib import Path

KIND = "crit-bundle-file"


class FileBundleStore:
    mapped = False   # no device-readable region: a cache reads its bundles with read()

    def __init__(self, layout, layers, experts, path, tier=None, reader=None, readers=16, owned=False, identity=None,
                 pack=None):
        """path: the bundle file; a complete one (its index beside it, `path`.json, naming the same layout) opens with
        its bundles and no puts, any other is written anew. owned: remove the file when the store closes. identity: the
        checkpoint's identity the index binds the file to (needed for a file that is kept: not owned). pack(layout,
        bundle, compressed) -> the bytes kept for a slot-layout bundle (expert_store.store_bundle)."""
        if not owned and identity is None:
            raise ValueError("a kept bundle file needs the identity of the checkpoint it holds")
        self.layout = dict(layout)
        self.bundle_bytes = int(self.layout["bundle_bytes"])
        if self.bundle_bytes < 16 or self.bundle_bytes % 16 or layers < 1 or experts < 1:
            raise ValueError("bundles of a multiple of 16 bytes, layers and experts")
        self.layers, self.experts = int(layers), int(experts)
        self.keys = self.layers * self.experts
        self.compressed = tuple(self.layout["compressed"]) if "program" in self.layout else ()
        if self.compressed and pack is None:
            raise ValueError("a store that keeps compressed scales needs pack (expert_store.store_bundle)")
        self._pack = pack
        self.path = Path(path)
        self.tier, self._reader_module, self.readers, self.owned = tier, reader, int(readers), owned
        self._index = {"kind": KIND, "layers": self.layers, "experts": self.experts,
                       "layout": json.loads(json.dumps(self.layout, default=str)),
                       "identity": json.loads(json.dumps(identity, default=str))}
        self.complete = self._matches()
        if self.complete:
            self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        else:
            try:   # an index names complete bytes only: gone before they change
                os.remove(str(self.path) + ".json")
            except FileNotFoundError:
                pass
            self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o644)
            os.ftruncate(self._fd, self.keys * self.bundle_bytes)
        self._reader = None
        self.stats = {"tier_hits": 0, "file_reads": 0}

    def _matches(self):
        try:
            index = json.loads(Path(str(self.path) + ".json").read_text(encoding="utf-8"))
            return index == self._index and self.path.stat().st_size == self.keys * self.bundle_bytes
        except (OSError, ValueError):
            return False

    def key(self, layer, expert):
        if not (0 <= layer < self.layers and 0 <= expert < self.experts):
            raise ValueError(f"expert {expert} of layer {layer} is outside the store")
        return layer * self.experts + expert

    def put(self, layer, expert, bundle, compressed=None):
        """One expert's slot-layout bundle, kept as pack() makes it (compressed: the checkpoint's CSF scales)."""
        if self._reader is not None or self.complete:
            raise ValueError("the bundle file is immutable once read (or opened complete): put every bundle before "
                             "the first read()")
        if self._pack is not None:
            bundle = self._pack(self.layout, bundle, compressed)
        elif compressed:
            raise ValueError("the store keeps no compressed scales: put the slot-layout bundle alone")
        data = bundle.detach().contiguous().cpu().numpy().tobytes() if hasattr(bundle, "numpy") else bytes(bundle)
        if len(data) != self.bundle_bytes:
            raise ValueError(f"a bundle is {self.bundle_bytes} bytes, not {len(data)}")
        pwrite(self._fd, data, self.key(layer, expert) * self.bundle_bytes)

    def region(self):
        raise RuntimeError("a file tier has no device-readable region: a cache reads its bundles with read()")

    def address(self, layer, expert):
        raise RuntimeError("a file tier has no device addresses: a cache reads its bundles with read()")

    def _open(self):
        if self._reader is not None:
            return
        if not self.complete:
            os.fsync(self._fd)
            Path(str(self.path) + ".json").write_text(json.dumps(self._index), encoding="utf-8")
            self.complete = True
        native = self._reader_module.native() if self._reader_module is not None else None
        if native is not None:
            self._reader = native.Reader([str(self.path)], [0], [self.keys], self.bundle_bytes, self.readers, 0)
        else:
            self._reader = "python"
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)

    def read(self, keys, into):
        """Bundles of `keys` into `into` ([n][bundle_bytes] uint8, host or device), in order: from the host tier where
        it holds them, the rest read from the file (each once) and held there."""
        import torch
        self._open()
        keys = [int(k) for k in keys]
        if any(not 0 <= k < self.keys for k in keys):
            raise IndexError("a bundle outside the store")
        missing = []
        for i, k in enumerate(keys):
            held = self.tier.get(k) if self.tier is not None else None
            if held is not None:
                into[i].copy_(torch.frombuffer(bytearray(held), dtype=torch.uint8))
                self.stats["tier_hits"] += 1
            else:
                missing.append(i)
        if not missing:
            return
        wanted = list(dict.fromkeys(keys[i] for i in missing))
        staging = torch.empty(len(wanted), self.bundle_bytes, dtype=torch.uint8)
        if self._reader == "python":
            for j, k in enumerate(wanted):
                data = pread(self._fd, self.bundle_bytes, k * self.bundle_bytes)
                if len(data) != self.bundle_bytes:
                    raise OSError(f"short read of bundle {k}")
                staging[j].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        else:
            self._reader.gather(torch.tensor(wanted, dtype=torch.int64), staging)
        self.stats["file_reads"] += len(wanted)
        at = {k: j for j, k in enumerate(wanted)}
        for i in missing:
            into[i].copy_(staging[at[keys[i]]])
        if self.tier is not None:
            for k, j in at.items():
                self.tier.admit(k, staging[j].numpy().tobytes())

    def close(self):
        self._reader = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self.owned:
            for path in (self.path, Path(str(self.path) + ".json")):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
