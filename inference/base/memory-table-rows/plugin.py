"""A lookup table in memory in the embeddings slot: rows of numbers (host form), or tables whose rows are indexed
(device form): store(rows) returns a table handle (lookup-table-rows' Table: lookup(ids), close()), on the device, or
with the `host` option in registered host memory read over the link (lookup-table-rows' HostRows), which frees the
table's device memory. A plugin holds any number of tables; its own lookup(ids) reads the last one stored. Either
lookup captures in a CUDA graph."""
from .memory import host_memory
from .rows import device_rows


class MemoryTableRows:
    def __init__(self, host=False):
        self.table, self.host = [], host
        self.device_rows = device_rows.HostRows(memory=host_memory)

    def load(self, table):
        if not table or any(len(row) != len(table[0]) for row in table):
            raise ValueError("a table of rows of one width")
        self.table = [[float(v) for v in row] for row in table]

    def rows(self, ids):
        out = []
        for i in ids:
            if not 0 <= i < len(self.table):
                raise IndexError(f"row {i} outside the table")
            out.append(list(self.table[i]))
        return out

    def store(self, rows, format=None, scales=None, scale=1.0, dtype=None, device=None):
        """A table for lookup, returned as its handle (lookup(ids), close()): rows [ids][width] of their dtype, or format
        "nvfp4" (rows the E2M1 codes [ids][width / 2], scales the E4M3 scales [ids][width / 16], dtype the decoded
        dtype, scale the global scale), as lookup-table-rows' store() takes them. On the device, or (host) copied into
        registered host memory, `rows` then free to go; the table serves `device`, by default the one `rows` are on
        (with host, rows read on the CPU never pass through the device)."""
        import torch
        if rows.dim() != 2 or rows.shape[0] < 1:
            raise ValueError("a table of rows of one width")
        if not self.host:
            return self.device_rows.store(rows, format, scales, scale, dtype, device)
        device = rows.device if device is None else torch.device(device)
        if format is None:
            return self.device_rows.open(rows.cpu(), rows.shape[1], rows.dtype, device)
        if scales is None:
            raise ValueError("NVFP4 rows need their scales")
        return self.device_rows.open(rows.cpu(), rows.shape[1] * 2, dtype or torch.bfloat16, device,
                                     format=format, scales=scales.cpu(), scale=scale)

    def lookup(self, ids):
        """The rows of `ids` from the last table stored. Ids outside the table raise IndexError, checked with one read
        of the device; inside a CUDA graph capture the check is left to the step that built the ids (a capture cannot
        read the device), so a model can capture its lookups."""
        return self.device_rows.lookup(ids)

    def facts(self):
        if self.device_rows.last is None:
            return {}
        return {"embeddings": {"table": "host" if self.host else "device", "tables": len(self.device_rows.tables)}}

    def close(self):
        self.device_rows.close()


def create(context, options, parts):
    return MemoryTableRows(options["host"])
