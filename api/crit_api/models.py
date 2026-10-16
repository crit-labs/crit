"""Immutable resident model snapshots and generation-bound device allocation leases.

The host implementation maps verified private files. Device allocation, actual
byte accounting, synchronization and release are supplied by the engine backend.
"""
from dataclasses import dataclass
import hashlib
import mmap
import os
from pathlib import Path
import shutil
import time
import uuid

from crit.run.records import digest, keys, number, safe_path, sha, text, file_sha


def locked_read(path):
    """Keep a native model source open; Windows denies writes and replacement."""
    if os.name != 'nt':
        return path.open('rb')
    import ctypes
    import msvcrt
    from ctypes import wintypes
    create = ctypes.WinDLL('kernel32', use_last_error=True).CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 1, None, 3, 0x08000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), 'rb')


@dataclass
class DeviceAllocation:
    buffer: object
    nbytes: int
    wait: object
    free: object


class DeviceLease:
    def __init__(self, pool, key, entry):
        self.pool, self.key, self.entry = pool, key, entry
        self.generation = pool.generation
        self.closed = False
        entry["users"] += 1

    @property
    def buffer(self):
        if self.closed or self.generation != self.pool.generation or self.pool.blocked:
            raise ValueError("Device lease is closed, stale or quarantined")
        return self.entry["allocation"].buffer

    def close(self):
        if not self.closed:
            if self.pool.blocked or self.generation != self.pool.generation:
                raise ValueError("Device lease is stale or quarantined; retire its runtime")
            try:
                self.entry["allocation"].wait()
            except BaseException:
                self.pool.blocked = True
                raise
            self.entry["users"] -= 1
            self.closed = True


class ModelPool:
    def __init__(self, root, host_limit, device_limit):
        self.root = Path(root)
        self.host_limit = number(host_limit, "resident host bytes", 1, 2 ** 60, integer=True) if host_limit is not None else 2 ** 60
        self.device_limit = number(device_limit, "resident device bytes", 0, 2 ** 60, integer=True) if device_limit is not None else 2 ** 60
        self.generation = uuid.uuid4().hex
        self.model_id = None
        self.files = {}
        self.device = {}
        self.host_bytes = self.device_bytes = 0
        self.blocked = False
        self.sources = {}
        self.source_hashes = {}
        self.paths = {}

    @staticmethod
    def fingerprint(path):
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def verify_sources(self):
        if self.blocked:
            raise ValueError("Model residency is quarantined")
        for path, identity in self.sources.items():
            if (self.fingerprint(path) != identity or
                    (os.name == 'nt' and path in self.source_hashes and
                     file_sha(path) != self.source_hashes[path])):
                raise ValueError("Model source changed; unload before loading a new identity")

    def load(self, model_path, manifest, native_files=False, progress=None):
        if self.blocked:
            raise ValueError("Model residency is quarantined")
        model_id = digest(manifest)
        if self.model_id == model_id:
            self.verify_sources()
            return False
        if self.model_id is not None:
            raise ValueError("Unload the current model before changing model identity")
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("Model needs a nonempty content manifest")
        model_path = Path(model_path)
        total = sum(number(row["bytes"], "model file bytes", 0, 2 ** 60, integer=True) for row in manifest)
        if total > self.host_limit:
            raise ValueError("Model exceeds the resident RAM budget; increase --resident-ram-gib")
        target = self.root / self.generation
        target.mkdir(parents=True)
        completed, reported = 0, 0
        def report(index, name, size, force=False):
            nonlocal reported
            stamp = time.monotonic()
            if progress and (force or stamp - reported >= .25):
                progress({"completed_bytes": completed + size, "total_bytes": total,
                          "file_index": index, "file_count": len(manifest), "file_name": name})
                reported = stamp
        try:
            for index, row in enumerate(manifest, 1):
                keys(row, {"path", "bytes", "sha256"})
                sha(row["sha256"], "model file digest")
                destination = safe_path(target, row["path"])
                source = safe_path(model_path, row["path"]) if model_path.is_dir() else safe_path(model_path.parent, row["path"])
                if not model_path.is_dir() and source != model_path:
                    raise ValueError("Model file manifest names a different source")
                before = self.fingerprint(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                hasher, size = hashlib.sha256(), 0
                report(index, row['path'], size, force=True)
                input_file = locked_read(source) if native_files else source.open('rb')
                try:
                    output = None if native_files else destination.open('xb')
                    try:
                        for block in iter(lambda: input_file.read(8 * 1024 * 1024), b''):
                            size += len(block)
                            if size > row['bytes']:
                                raise ValueError('Model file grew during loading')
                            hasher.update(block)
                            if output:
                                output.write(block)
                            report(index, row['path'], size)
                    finally:
                        if output:
                            output.close()
                    if size != row['bytes'] or hasher.hexdigest() != row['sha256'] or before != self.fingerprint(source):
                        raise ValueError('Model content differs from its frozen manifest')
                    if native_files:
                        stream, input_file = input_file, None
                        self.paths[row['path']] = source
                    else:
                        destination.chmod(0o400)
                        stream = destination.open('rb')
                        self.paths[row['path']] = destination
                finally:
                    if input_file:
                        input_file.close()
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) if size else None
                self.files[row["path"]] = (stream, mapping)
                self.sources[source] = before
                if not native_files:
                    # Native Windows sources have a retained deny-write handle;
                    # copied sources need byte verification despite stale metadata.
                    self.source_hashes[source] = row['sha256']
                report(index, row['path'], size, force=True)
                completed += size
            self.model_id, self.host_bytes = model_id, total
            return True
        except BaseException:
            self.close()
            raise

    def host(self, name):
        self.verify_sources()
        mapping = self.files[name][1]
        return memoryview(mapping).toreadonly() if mapping is not None else memoryview(b"")

    def acquire_device(self, compatibility, nbytes, allocate, inputs=()):
        """inputs: files outside the model (for example a re-encoded draft-only weight file shipped with the package)
        whose bytes enter this allocation; their content identity joins the key, so changed bytes never reuse
        an allocation built from the old ones."""
        if self.blocked or self.model_id is None:
            raise ValueError("Load a healthy model before acquiring device weights")
        keys(compatibility, {"device", "representation", "dtype", "quantization", "layout", "shape", "abi"})
        for field in ("device", "representation", "dtype", "quantization", "layout", "abi"):
            text(compatibility[field], "device " + field)
        if not isinstance(compatibility["shape"], list) or not compatibility["shape"]:
            raise ValueError("Device shape requires a nonempty list of dimensions")
        for dimension in compatibility["shape"]:
            number(dimension, "device dimension", 1, 2 ** 60, integer=True)
        nbytes = number(nbytes, "device allocation bytes", 1, 2 ** 60, integer=True) if nbytes is not None else None
        sources = []
        for item in inputs:
            path = Path(item)
            if not path.is_file():
                raise ValueError("Device input is not a readable file: " + str(item))
            sources.append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)})
        key = digest({"model": self.model_id, "compatibility": compatibility, "inputs": sources})
        entry = self.device.get(key)
        if entry is None:
            if nbytes is not None and self.device_bytes + nbytes > self.device_limit:
                raise ValueError("Resident device budget exhausted; unload or increase --resident-vram-gib")
            try:
                allocation = allocate()
            except BaseException:
                self.blocked = True
                raise
            # Capture ownership before inspecting backend metadata. An invalid
            # extent or callback must not drop the only reference to live memory.
            entry = {"allocation": allocation, "users": 0, "nbytes": 0}
            self.device[key] = entry
            try:
                if not isinstance(allocation, DeviceAllocation):
                    raise ValueError("Device backend did not return an owned allocation")
                actual = number(allocation.nbytes, 'actual device allocation bytes', 1, 2 ** 60, integer=True)
                entry["nbytes"] = actual
                self.device_bytes += actual
                if nbytes is not None and actual != nbytes:
                    raise ValueError("Device backend allocation/accounting contract mismatch")
                if self.device_bytes > self.device_limit:
                    raise ValueError('Resident device budget exhausted by actual allocation; retire the runtime')
                if not callable(allocation.wait) or not callable(allocation.free):
                    raise ValueError("Device backend allocation/accounting contract mismatch")
            except BaseException:
                self.blocked = True
                raise
        elif nbytes is not None and entry["nbytes"] != nbytes:
            raise ValueError("Identical device representation has a different physical extent")
        return DeviceLease(self, key, entry)

    def close(self):
        if self.blocked:
            raise ValueError("Quarantined device ownership requires runtime process retirement")
        if any(entry["users"] for entry in self.device.values()):
            raise ValueError("Cannot unload weights while engine leases remain active")
        # Fence all allocations before freeing any; failed fences retain owners.
        try:
            for entry in self.device.values():
                entry["allocation"].wait()
            for key, entry in list(self.device.items()):
                entry["allocation"].free()
                self.device_bytes -= entry["nbytes"]
                del self.device[key]
        except BaseException:
            self.blocked = True
            raise
        for name, (stream, mapping) in list(self.files.items()):
            if mapping is not None:
                mapping.close()  # Exported views must have been released first.
            stream.close()
            del self.files[name]
        target = self.root / self.generation
        if target.exists():
            for path in target.rglob("*"):
                if path.is_file():
                    path.chmod(0o600)
            shutil.rmtree(target)
        self.model_id = None
        self.host_bytes = 0
        self.sources.clear()
        self.source_hashes.clear()
        self.paths.clear()
        self.generation = uuid.uuid4().hex


class ModelContext:
    """One engine's leases and arguments; mutable inference state stays in it."""
    def __init__(self, pool, configuration, arguments, progress=None):
        self.pool, self.configuration, self.arguments = pool, configuration, arguments
        self.leases, self.views = [], []
        self.progress = progress

    @property
    def model_id(self):
        return self.pool.model_id

    def host(self, name):
        view = self.pool.host(name)
        self.views.append(view)
        return view

    def model_file(self, name):
        """A verified file held for the pool lifetime, for native file loaders."""
        self.pool.verify_sources()
        return self.pool.paths[name]

    def device(self, compatibility, nbytes, allocate, inputs=()):
        def loading():
            if self.progress:
                self.progress("loading_weights")
            return allocate()
        value = self.pool.acquire_device(compatibility, nbytes, loading, inputs)
        self.leases.append(value)
        if self.progress:
            self.progress("loading_engine")
        return value

    def close(self):
        for value in self.leases:
            value.close()
        for value in self.views:
            value.release()
        self.leases.clear()
        self.views.clear()
