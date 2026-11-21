"""Lookup-table rows for a device step when the table stays in host memory or in files (crit_plugins.device
DEVICE_EXTENSIONS "embeddings:host", "rounds:embeddings"): open(source, width, dtype, device), prefetch(ids),
lookup(ids) -> rows on the device, capture-safe.

Three places a table lives:
- the device (store(rows)): one tensor indexed by the ids;
- host memory (open(tensor), or open(path, mapped=True)): the rows copied once into memory registered for the device
  (or the file mapped read-only and registered), and lookup gathers them over the link by torch's index_select on a
  device view of that memory: no copy per step, no device memory for the table;
- files (open(path or paths)): rows read by the native row reader (row_reader.cpp: a persistent pool of positioned
  reads, POSIX_FADV_RANDOM, a hot-row cache). lookup puts the gather on the stream as a host function: the step's ids
  go to pinned memory, the reader writes their rows into pinned memory there, and they are copied to the device, so
  a captured round reads its rows without Python. prefetch(ids) starts reading rows known ahead (the next prompt
  chunk, a round's pending tokens) into the hot-row cache, so the gather finds them in memory.

store() and open() return a Table, a handle with lookup, prefetch, drain, stats and close, so a model reads several
tables through one plugin (token rows and per-layer n-gram rows); the plugin's own lookup and prefetch read the last
table stored or opened. A table's rows are of one dtype, or stored NVFP4 (format "nvfp4": E2M1 codes, element 2k in
the low nibble of byte k, from `source`, one E4M3 scale per 16 values from `scales`, times `scale`, the table's global
scale as a multiplier) or FP8 (format "fp8": E4M3 codes, one byte a value, times `scale`, the table's one
multiplier), decoded on the device by torch into rows of `dtype`, or stored in a GGUF block format (format
"gguf:<TYPE>", Core's crit_core.model.gguf_blocks.TYPES: a row its blocks as the file stores them) decoded on the
device by crit's kernel gguf-rows (its dequantize: the format's own values, one rounding to `dtype`), or where that
kernel does not build by Core's host decoders, which a captured round cannot run (round_problems says so).

A lookup's rows stay valid until `ring` later lookups of the same size: each size has `ring` sets of pinned and device
buffers, allocated at its first lookup (outside a capture), used in turn.

Inside a captured round the ids of a file table are known only on the device (a round's drafts): the lookup's ids go to
pinned memory by a copy node, the reader's gather runs as a host node (cudaLaunchHostFunc) and the rows come back by a
copy node, so the graph itself asks the host for its rows. The stream waits for the gather (prefetch of what is known
ahead makes it a copy from memory); work of the graph that does not need the rows does not overlap it.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import os
import threading

from crit_plugins.files import pread

try:
    from . import row_reader
except ImportError:   # run from code/
    import row_reader


SITE_ROWS = 64   # the fewest rows a buffer set holds: sets are sized by powers of two


class _Site:
    """One buffer set of lookups: pinned ids and rows the host function fills, the rows' device copy. native: the
    native reader's site over the set's first n ids and rows, by n (a lookup reads exactly its rows)."""
    __slots__ = ("ids", "rows", "out", "native", "event")


class _PythonReader:
    """Python threads reading rows with pread, a bounded hot-row cache; the fallback of the native reader."""

    def __init__(self, files, row_bytes, threads, cache_rows):
        self.files, self.row_bytes = files, row_bytes
        self.fds = [os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)) for path, _, _ in files]
        for fd, (_, offset, rows) in zip(self.fds, files):
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, offset, rows * row_bytes, os.POSIX_FADV_RANDOM)
        self.first, total = [], 0
        for _, _, rows in files:
            self.first.append(total)
            total += rows
        self.total = total
        self.pool = ThreadPoolExecutor(threads)
        self.cache, self.cache_rows = OrderedDict(), cache_rows
        self.lock = threading.Lock()
        self.counts = [0, 0, 0, 0]
        self.pending = []

    def rows(self):
        return self.total

    def _read(self, row):
        f = max(i for i, first in enumerate(self.first) if first <= row)
        _, offset, _ = self.files[f]
        data = pread(self.fds[f], self.row_bytes, offset + (row - self.first[f]) * self.row_bytes)
        if len(data) != self.row_bytes:
            raise OSError(f"short read of row {row}")
        return data

    def gather(self, ids, out):
        self.drain()
        ids = ids.tolist()
        if any(not 0 <= r < self.total for r in ids):
            raise IndexError("row outside the table")
        found, missing = {}, []
        with self.lock:
            for r in dict.fromkeys(ids):
                if r in self.cache:
                    self.cache.move_to_end(r)
                    found[r] = self.cache[r]
                    self.counts[1] += 1
                else:
                    missing.append(r)
        for r, data in zip(missing, self.pool.map(self._read, missing)):
            found[r] = data
            self._remember(r, data)
        with self.lock:
            self.counts[0] += len(missing)
        view = out.view(-1).numpy()
        for i, r in enumerate(ids):
            view[i * self.row_bytes:(i + 1) * self.row_bytes] = memoryview(found[r])

    def _remember(self, r, data):
        if not self.cache_rows:
            return
        with self.lock:
            self.cache[r] = data
            self.cache.move_to_end(r)
            while len(self.cache) > self.cache_rows:
                self.cache.popitem(last=False)

    def prefetch(self, ids):
        if not self.cache_rows:
            return
        wanted = [r for r in dict.fromkeys(ids.tolist()) if 0 <= r < self.total and r not in self.cache]
        def fetch():
            for r in wanted:
                self._remember(r, self._read(r))
            with self.lock:
                self.counts[0] += len(wanted)
                self.counts[2] += len(wanted)
        self.pending.append(self.pool.submit(fetch))

    def drain(self):
        pending, self.pending = self.pending, []
        for future in pending:
            future.result()

    def stats(self):
        with self.lock:
            return list(self.counts)

    def failed(self):
        return False

    def close(self):
        self.pool.shutdown()
        for fd in self.fds:
            os.close(fd)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
FORMATS = (None, "nvfp4", "fp8")   # and "gguf:<TYPE>" (gguf_type)


def gguf_type(format):
    """The GGUF type id of format "gguf:<TYPE>", None for another format; ValueError for a type Core cannot decode."""
    if not isinstance(format, str) or not format.startswith("gguf:"):
        return None
    from crit_core.model.gguf_blocks import TYPES
    if format[5:] not in TYPES:
        raise ValueError(f"a GGUF table's type is one of {', '.join(TYPES)}, not {format[5:]!r}")
    return TYPES[format[5:]]


class _Part:
    """One source of rows of row_bytes: a device tensor, registered host memory, or files through a reader."""

    def __init__(self, owner, row_bytes, device):
        self.owner, self.row_bytes, self.device = owner, int(row_bytes), device
        self.mode = self.reader = self.rows_u8 = None
        self.native = False
        self.native_unavailable = None   # why the Python reader serves where the native one was wanted
        self._mapping = self._registered = None
        self._sites = {}
        self._captured = []   # sites a captured graph reads: out of the eager ring for the graph's life
        self.rows_count = 0

    def from_device(self, rows):
        import torch
        flat = rows.detach().contiguous().view(-1).view(torch.uint8)
        self.rows_count = flat.numel() // self.row_bytes
        self.rows_u8, self.mode = flat.view(self.rows_count, self.row_bytes), "device"

    def from_tensor(self, rows):
        import torch
        memory = self.owner._need_memory()
        data = rows.detach().contiguous().view(-1).view(torch.uint8).cpu()
        nbytes = data.numel()
        if nbytes % self.row_bytes or not nbytes:
            raise ValueError(f"the tensor holds {nbytes} bytes, not whole rows of {self.row_bytes}")
        self._mapping = memory.map_anonymous(nbytes)
        memory.host_tensor(self._mapping.address, nbytes).copy_(data)
        self._view(nbytes, read_only=False)

    def from_mapped(self, path, rows):
        self._mapping = self.owner._need_memory().map_file(path)
        self._view(rows * self.row_bytes, read_only=True)

    def from_resident(self, files):
        """The files' rows read once into registered host memory (no reader, no file kept open): a table held in
        memory, gathered over the link as a tensor's rows are."""
        nbytes = sum(rows * self.row_bytes for _, _, rows in files)
        self._mapping = self.owner._need_memory().map_anonymous(nbytes)
        out = memoryview(self.owner.memory.host_tensor(self._mapping.address, nbytes).numpy())
        at = 0
        for path, offset, rows in files:
            end = at + rows * self.row_bytes
            with open(path, "rb", buffering=0) as handle:
                handle.seek(offset)
                while at < end:
                    got = handle.readinto(out[at:end])
                    if not got:
                        raise ValueError(f"{path} ends before its {rows} rows of {self.row_bytes} bytes from {offset}")
                    at += got
        self._view(nbytes, read_only=False)

    def _view(self, nbytes, read_only):
        memory = self.owner.memory
        self.rows_count = nbytes // self.row_bytes
        if self.device.type == "cuda":
            index = self.device.index or 0
            self._registered = memory.register(self._mapping.address, self._mapping.size,
                                               read_only=read_only and memory.read_only_supported(index))
            view = memory.device_tensor(self._registered, nbytes, self.device)
        else:
            view = memory.host_tensor(self._mapping.address, nbytes)
        self.rows_u8, self.mode = view.view(self.rows_count, self.row_bytes), "mapped"

    def from_files(self, files):
        owner = self.owner
        native = row_reader.native() if owner.use_native else None
        paths, offsets, counts = [f[0] for f in files], [f[1] for f in files], [f[2] for f in files]
        if native is not None:
            self.reader = native.Reader(paths, offsets, counts, self.row_bytes, owner.readers, owner.cache_rows)
            self.native = True
        else:
            self.reader = _PythonReader(files, self.row_bytes, min(owner.readers, 32), owner.cache_rows)
            if owner.use_native:   # wanted, but it did not build here (no compiler or no Ninja)
                self.native_unavailable = row_reader.WHY[-1] if row_reader.WHY else "the native reader did not build"
        self.mode, self.rows_count = "file", sum(counts)

    def gather(self, ids, capturing):
        """uint8 [n][row_bytes] on the device (the host for a CPU table)."""
        import torch
        if self.mode in ("device", "mapped"):
            return self.rows_u8.index_select(0, ids.to(self.rows_u8.device))
        if not capturing and self.reader.failed():
            raise RuntimeError("a read of the table's rows failed in a host function: its rows were zeros; the table "
                               "serves no more lookups")
        if self.device.type != "cuda":
            out = torch.empty(ids.numel(), self.row_bytes, dtype=torch.uint8)
            self.reader.gather(ids.reshape(-1).to(torch.int64).contiguous(), out)
            return out
        n = ids.numel()
        site = self._site(n, capturing)
        if not capturing and site.event is not None:
            site.event.synchronize()   # the site's last lookup ran (its host function read site.ids)
        stream = torch.cuda.current_stream(self.device)
        site.ids[:n].copy_(ids.reshape(-1), non_blocking=True)
        if site.native is not None:
            if n not in site.native:   # the reader's site over this lookup's rows of the set
                site.native[n] = self.reader.site(site.ids[:n], site.rows[:n])
            self.reader.enqueue(site.native[n], stream.cuda_stream)
        else:
            if capturing:
                raise RuntimeError("the Python row reader cannot run inside a CUDA graph: build the native reader")
            stream.synchronize()
            self.reader.gather(site.ids[:n], site.rows[:n])
        site.out[:n].copy_(site.rows[:n], non_blocking=True)
        if not capturing:
            site.event = torch.cuda.Event()
            site.event.record(stream)
        return site.out[:n]

    def _set(self, rows):
        import torch
        site = _Site()
        site.ids = torch.empty(rows, dtype=torch.int64, pin_memory=True)
        site.rows = torch.empty(rows, self.row_bytes, dtype=torch.uint8, pin_memory=True)
        site.out = torch.empty(rows, self.row_bytes, dtype=torch.uint8, device=self.device)
        site.native = {} if self.native else None
        site.event = None
        return site

    def _site(self, n, capturing):
        """A buffer set for a lookup of `n` rows. A captured lookup has one of its own, made as it is captured and
        kept for its graph's life (the graph replays into it, and no eager lookup ever writes it). Eager lookups use
        `ring` sets in turn, sized by powers of two of rows (SITE_ROWS at least), a lookup using the first n: the
        prompt lengths a server meets share a few sets where each length would keep its own for good."""
        if capturing:
            self._captured.append(self._set(int(n)))
            return self._captured[-1]
        n = max(SITE_ROWS, 1 << (int(n) - 1).bit_length())
        sites = self._sites.get(n)
        if sites is None:
            sites = self._sites[n] = [[self._set(n) for _ in range(self.owner.ring)], 0]
        ring, at = sites
        sites[1] = (at + 1) % len(ring)
        return ring[at]

    def prefetch(self, ids):
        if self.mode == "file":
            import torch
            self.reader.prefetch(torch.as_tensor(ids, dtype=torch.int64).reshape(-1).contiguous())

    def drain(self):
        if self.mode == "file":
            self.reader.drain()

    def stats(self):
        return self.reader.stats() if self.mode == "file" else [0, 0, 0, 0]

    def failed(self):
        return self.mode == "file" and self.reader.failed()

    def close(self):
        if self.mode == "file" and not self.native and self.reader is not None:
            self.reader.close()
        self.reader, self._sites, self.rows_u8 = None, {}, None
        if self._registered is not None:
            self.owner.memory.unregister(self._mapping.address)
            self._registered = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self.mode = None


class Table:
    """One table's rows for a device step: lookup(ids) -> [tokens][width] of dtype on the device, prefetch(ids),
    drain(), stats(), close()."""

    def __init__(self, parts, width, dtype, device, format=None, scale=1.0, kernel=None):
        self.parts, self.width, self.dtype, self.device = parts, int(width), dtype, device
        self.format, self.scale = format, float(scale)
        self.rows_count = min(p.rows_count for p in parts)
        self._lut = None
        self.gguf = gguf_type(format)
        self.kernel = kernel if self.gguf is not None and getattr(device, "type", "cpu") == "cuda" else None
        if format == "nvfp4":   # made now: a captured lookup does no first-call work
            import torch
            self._lut = torch.tensor(E2M1, dtype=torch.float32, device=device)

    @property
    def mode(self):
        return self.parts[0].mode

    @property
    def native(self):
        return any(p.native for p in self.parts)

    @property
    def native_unavailable(self):
        """Why the Python reader serves where the native one was wanted (it did not build here), else None."""
        return next((p.native_unavailable for p in self.parts if p.native_unavailable), None)

    @property
    def device_table(self):
        """A device or host-memory table of one dtype as [rows][width] (what index_select reads), else None."""
        part = self.parts[0]
        if self.format is not None or part.rows_u8 is None:
            return None
        return part.rows_u8.view(self.dtype).view(part.rows_count, self.width)

    def lookup(self, ids):
        """The rows of ids ([tokens] int64, on the device or the host) on the device ([tokens][width] of dtype). Ids
        outside the table raise IndexError, checked with one read of the device outside a capture."""
        import torch
        if self.parts[0].mode is None:
            raise ValueError("the table is closed")
        capturing = self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        if ids.numel() and not capturing and bool(((ids < 0) | (ids >= self.rows_count)).any()):
            raise IndexError("a row outside the table")
        if self.format is None:
            return self.parts[0].gather(ids, capturing).view(self.dtype).view(-1, self.width)
        if self.gguf is not None:
            return self._gguf(self.parts[0].gather(ids, capturing))
        if self.format == "fp8":
            return self._fp8(self.parts[0].gather(ids, capturing))
        codes, scales = (p.gather(ids, capturing) for p in self.parts)
        return self._nvfp4(codes, scales)

    def _gguf(self, raw):
        """Stored rows [n][row bytes] as their values [n][width] of dtype: gguf-rows' dequantize on the device (the
        format's FP32 values, one rounding), else Core's host decoders."""
        import torch
        raw = raw.reshape(-1, self.parts[0].row_bytes)
        if not raw.shape[0]:
            return torch.empty(0, self.width, dtype=self.dtype, device=raw.device)
        if self.kernel is not None:
            return self.kernel.dequantize(self.kernel.GgufMatrix(raw.contiguous(), self.gguf, self.width), self.dtype)
        from crit_core.model.gguf_blocks import stored_values
        values = stored_values(raw.cpu().numpy(), self.gguf, self.width)
        return torch.from_numpy(values).to(device=raw.device, dtype=self.dtype)

    def _nvfp4(self, codes, scales):
        """E2M1 codes [n][width / 2] (element 2k the low nibble of byte k) times their E4M3 scale [n][width / 16] times
        the table's scale, in FP32 (exact products, one rounding by the scale), then rounded to dtype."""
        import torch
        if self._lut is None or self._lut.device != codes.device:
            self._lut = torch.tensor(E2M1, dtype=torch.float32, device=codes.device)
        n = codes.shape[0]
        nibbles = torch.stack((codes & 15, codes >> 4), -1).view(n, self.width).long()
        values = self._lut[nibbles].view(n, self.width // 16, 16)
        factors = scales.view(torch.float8_e4m3fn).float().view(n, self.width // 16, 1)
        out = (values * factors).view(n, self.width)
        if self.scale != 1.0:
            out = out * self.scale
        return out.to(self.dtype)

    def _fp8(self, codes):
        """E4M3 codes [n][width] times the table's scale, in FP32 (an E4M3 value times an FP32 scale rounds once),
        then rounded to dtype."""
        import torch
        out = codes.view(torch.uint8).view(-1, self.width).view(torch.float8_e4m3fn).float()
        if self.scale != 1.0:
            out = out * self.scale
        return out.to(self.dtype)

    def prefetch(self, ids):
        """Rows of host ids known ahead (a list, or a CPU tensor) read into the hot-row cache in the background."""
        for part in self.parts:
            part.prefetch(ids)

    def drain(self):
        for part in self.parts:
            part.drain()
        self.check()

    def check(self):
        """Raises when a read of the table's rows failed (in a host function, where it could not raise)."""
        if any(part.failed() for part in self.parts):
            raise RuntimeError("a read of the table's rows failed: the lookup that met it returned zeros")

    def round_problems(self):
        """Why a captured round or prompt chunk cannot look up this table, or []."""
        if any(p.mode == "file" and not p.native for p in self.parts):
            return ["the table is read by Python threads (the native row reader did not build, or native is off), "
                    "which a CUDA graph cannot run"]
        if self.gguf is not None and self.kernel is None:
            return [f"the table's {self.format[5:]} rows are decoded on the host (crit's kernel gguf-rows is not "
                    "built here), which a CUDA graph cannot run"]
        return []

    def stats(self):
        """{reads, hits, prefetched, failures} of its file readers (zeros for a table in memory)."""
        totals = [0, 0, 0, 0]
        for part in self.parts:
            totals = [a + b for a, b in zip(totals, part.stats())]
        return dict(zip(("reads", "hits", "prefetched", "failures"), totals))

    def close(self):
        for part in self.parts:
            part.close()


class HostRows:
    def __init__(self, readers=64, cache_rows=262144, ring=4, memory=None, native=True, gguf_kernel=None):
        self.readers, self.cache_rows, self.ring = int(readers), int(cache_rows), int(ring)
        self.memory, self.use_native = memory, native
        self.gguf_kernel = gguf_kernel   # () -> gguf-rows' package or None: loaded at the first GGUF table
        self.tables, self.last = [], None

    def _need_memory(self):
        if self.memory is None:
            raise ValueError("a table in host memory needs the host-memory module (host_memory.py)")
        return self.memory

    @staticmethod
    def _sizes(width, dtype, format):
        import torch
        type_id = gguf_type(format)
        if type_id is not None:
            from crit_core.model.gguf_blocks import row_bytes
            return [row_bytes(type_id, width)]
        if format not in FORMATS:
            raise ValueError(f"a table's format is one of {FORMATS} or gguf:<TYPE>, not {format!r}")
        if format == "nvfp4":
            if width % 16:
                raise ValueError("NVFP4 rows hold a multiple of 16 values")
            return [width // 2, width // 16]
        if format == "fp8":
            return [width]
        return [width * torch.empty(0, dtype=dtype).element_size()]

    def _table(self, parts, width, dtype, device, format, scale):
        kernel = None
        if gguf_type(format) is not None and self.gguf_kernel is not None:
            kernel = self.gguf_kernel()
        table = Table(parts, width, dtype, device, format, scale, kernel)
        self.tables.append(table)
        self.last = table
        return table

    def store(self, rows, format=None, scales=None, scale=1.0, dtype=None, device=None):
        """A table on the device: rows [ids][width] (of its dtype), or format "nvfp4": rows the codes [ids][width / 2]
        uint8, scales the E4M3 scales [ids][width / 16] (uint8 or float8), dtype the rows' decoded dtype (BF16 by
        default), scale the global scale (a multiplier); format "fp8": rows the E4M3 codes [ids][width] (uint8 or
        float8_e4m3fn), scale the table's multiplier, dtype the decoded rows' (BF16 by default); or format
        "gguf:<TYPE>": rows the stored blocks uint8 [ids][row bytes]. device: the device it serves, by default the one
        rows are on. Returns the Table."""
        import torch
        if rows.dim() != 2 or rows.shape[0] < 1:
            raise ValueError("a table of rows of one width")
        if device is not None and torch.device(device) != rows.device:
            rows = rows.to(device)
            scales = scales.to(device) if scales is not None else None
        if format is None:
            width, dtype, sources = rows.shape[1], rows.dtype, [rows]
        elif gguf_type(format) is not None:
            from crit_core.model.gguf_blocks import BLOCKS
            values, size = BLOCKS[gguf_type(format)]
            if rows.dtype != torch.uint8 or rows.shape[1] % size:
                raise ValueError(f"{format} rows are uint8, whole {size}-byte blocks")
            width, dtype, sources = rows.shape[1] // size * values, dtype or torch.bfloat16, [rows]
        elif format == "fp8":
            width, dtype, sources = rows.shape[1], dtype or torch.bfloat16, [rows.view(torch.uint8)]
        else:
            width, dtype, sources = rows.shape[1] * 2, dtype or torch.bfloat16, [rows, scales]
        sizes = self._sizes(width, dtype, format)
        if format == "nvfp4" and (scales is None or scales.shape[0] != rows.shape[0]
                                   or scales.numel() != rows.shape[0] * sizes[1]):
            raise ValueError("NVFP4 rows need their scales, one per 16 values of each row")
        parts = []
        for source, size in zip(sources, sizes):
            part = _Part(self, size, rows.device)
            part.from_device(source)
            parts.append(part)
        return self._table(parts, width, dtype, rows.device, format, scale)

    def open(self, source, width, dtype, device, mapped=False, format=None, scales=None, scale=1.0, resident=False):
        """The table in host memory or files: source a CPU tensor of rows, a file path, a list of paths (each holding
        the next rows in id order) or {"paths": [...], "offsets": [...], "rows": [...]}; width values of dtype a row.
        mapped: a file source mapped read-only and registered whole (it must fit in host memory) instead of read.
        resident: a file source's rows read once into registered host memory (it must fit there) instead of served
        by a reader from the files.
        format "nvfp4": source holds the code rows (width / 2 bytes each), scales the E4M3 scale rows (width / 16 bytes)
        in the same kinds of source, scale the global scale (a multiplier); format "fp8": source holds E4M3 code rows
        (width bytes each), scale the table's multiplier; format "gguf:<TYPE>": source holds the stored rows (whole
        blocks of the type for width values); rows decode to dtype. Returns the Table."""
        import torch
        device = torch.device(device)
        sizes = self._sizes(width, dtype, format)
        if format == "nvfp4" and scales is None:
            raise ValueError("NVFP4 rows need their scales")
        sources = [source, scales] if format == "nvfp4" else [source]
        parts = []
        try:
            for item, size in zip(sources, sizes):
                part = _Part(self, size, device)
                parts.append(part)
                if isinstance(item, torch.Tensor):
                    part.from_tensor(item)
                    continue
                files = self._files(item, size)
                if mapped:
                    if len(files) != 1 or files[0][1]:
                        raise ValueError("a mapped table is one file of rows from its first byte")
                    part.from_mapped(files[0][0], files[0][2])
                elif resident:
                    part.from_resident(files)
                else:
                    part.from_files(files)
        except BaseException:
            for part in parts:
                part.close()
            raise
        return self._table(parts, width, dtype, device, format, scale)

    @staticmethod
    def _files(source, row_bytes):
        if isinstance(source, dict):
            paths = [str(p) for p in source["paths"]]
            offsets = list(source.get("offsets") or [0] * len(paths))
            rows = list(source["rows"])
        else:
            paths = [str(p) for p in (source if isinstance(source, (list, tuple)) else [source])]
            offsets = [0] * len(paths)
            rows = [os.path.getsize(p) // row_bytes for p in paths]
        if not paths or len(offsets) != len(paths) or len(rows) != len(paths) or min(rows) < 1:
            raise ValueError("a table of files, each with an offset and its rows")
        return list(zip(paths, offsets, rows))

    # -- the last table stored or opened ----------------------------------------------------------------------------
    @property
    def mode(self):
        return self.last.mode if self.last is not None else None

    @property
    def native(self):
        return self.last is not None and self.last.native

    @property
    def native_unavailable(self):
        return self.last.native_unavailable if self.last is not None else None

    @property
    def device_table(self):
        return self.last.device_table if self.last is not None else None

    def lookup(self, ids):
        if self.last is None:
            raise ValueError("no table: store() or open() one")
        return self.last.lookup(ids)

    def prefetch(self, ids):
        if self.last is not None:
            self.last.prefetch(ids)

    def drain(self):
        if self.last is not None:
            self.last.drain()

    def round_problems(self):
        return [problem for table in self.tables for problem in table.round_problems()]

    def stats(self):
        totals = {"reads": 0, "hits": 0, "prefetched": 0, "failures": 0}
        for table in self.tables:
            for k, v in table.stats().items():
                totals[k] += v
        return totals

    def close(self):
        for table in self.tables:
            table.close()
        self.tables, self.last = [], None
