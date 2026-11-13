"""File-backed table rows in the embeddings slot. Host form: lookup-table-rows' RowTable (table_rows.py) over FP32 rows
in a file. Device form: device_rows.HostRows, the table on the device (store), in registered host memory gathered over
the link, or in files read by the native row reader (row_reader.cpp) as a host function on the stream, with prefetch
of rows known ahead (the "embeddings:host" and "rounds:embeddings" extensions)."""
import os
import struct
import tempfile

from .memory import host_memory
from .table import device_rows, table_rows


class FileTableRows:
    def __init__(self, options):
        self.workers, self.cache_rows = options["workers"], options["cache_rows"]
        self.directory = options["directory"] or None
        self.table, self.path, self.width = None, None, 0
        self.device_rows = device_rows.HostRows(options["readers"], options["prefetch_rows"], options["ring"],
                                                memory=host_memory, native=options["native"])

    # -- host form -----------------------------------------------------------------------------------------------
    def load(self, table):
        if not table or any(len(row) != len(table[0]) for row in table):
            raise ValueError("a table of rows of one width")
        self._close_host()
        self.width = len(table[0])
        handle, self.path = tempfile.mkstemp(prefix="crit-table-", suffix=".f32", dir=self.directory)
        with os.fdopen(handle, "wb") as file:
            for row in table:
                file.write(struct.pack(f"<{self.width}f", *row))
        region = table_rows.TableRegion(self.path, 0, len(table), 4 * self.width)
        self.table = table_rows.RowTable(region, self.workers, self.cache_rows)

    def rows(self, ids):
        if self.table is None:
            raise ValueError("no table is loaded")
        return [list(struct.unpack(f"<{self.width}f", data)) for data in self.table.gather(ids)]

    # -- device form ---------------------------------------------------------------------------------------------
    def store(self, rows, format=None, scales=None, scale=1.0, dtype=None, device=None):
        """A device table (on `device`, by default the one rows are on); returns its handle (device_rows.Table: lookup,
        prefetch, close)."""
        return self.device_rows.store(rows, format, scales, scale, dtype, device)

    def open(self, source, width, dtype, device, mapped=False, format=None, scales=None, scale=1.0):
        """A table in host memory or files; returns its handle (device_rows.Table)."""
        return self.device_rows.open(source, width, dtype, device, mapped, format, scales, scale)

    def prefetch(self, ids):
        self.device_rows.prefetch(ids)

    def lookup(self, ids):
        return self.device_rows.lookup(ids)

    def round_problems(self):
        """Why a captured round or prompt chunk cannot look up these tables, or [] (a table the Python reader
        serves: its gather is Python, which a CUDA graph cannot run). Asked by crit_plugins.rounds and
        crit_plugins.prompt_graphs before they capture, so open the tables first."""
        return self.device_rows.round_problems()

    def facts(self):
        rows = self.device_rows
        if rows.mode is None:
            return {}
        reader = ("native" if rows.native else "python") if rows.mode == "file" else None
        why = rows.native_unavailable if reader == "python" else None
        return {"embeddings": {"table": rows.mode, **({"reader": reader} if reader else {}),
                               **({"native_unavailable": why} if why else {})}}

    def telemetry(self):
        return {"table_rows": self.device_rows.stats()} if self.device_rows.mode == "file" else {}

    def _close_host(self):
        if self.table is not None:
            self.table.close()
            self.table = None
        if self.path is not None:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            self.path = None

    def close(self):
        self._close_host()
        self.device_rows.close()


def create(context, options, parts):
    return FileTableRows(options)
