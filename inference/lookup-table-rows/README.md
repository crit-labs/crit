# Read lookup-table rows from disk without read-ahead

Token embeddings and hashed n-gram embedding tables are read a few rows per token, never whole. They can stay on
disk even when they are larger than the rest of the model, and recon's placement puts them wherever room is left.
What makes them slow is the operating system: a row read through a file the kernel takes for a sequential stream,
or a page fault on a default mapping, pulls in the whole read-ahead window, often megabytes for a row of a few
kilobytes. This entry opens the table for random access, reads only the rows a token needs with positioned reads in
parallel, and keeps an optional cache of hot rows.

## When to use it

Use it when a gathered table (embedding or n-gram rows) stays in a file or a mapping instead of device or host
memory, as recon's placement scenario says for lookup tables on disk. It is unnecessary when the table fits host
memory and is loaded once, or when the device holds it.

## How it works

**Random-access advice.** On the file, `posix_fadvise(POSIX_FADV_RANDOM)` over the table's region; on a mapping,
`madvise(MADV_RANDOM)`. Either stops the kernel from reading ahead, so a cold row costs one small read. Windows has
no file advice: there the reads must be positioned reads of exactly one row, never a sequential stream.

**Exact positioned reads.** A row is `row_bytes` at `offset + row * row_bytes`. Each read is `pread` of exactly that
range (a locked seek and read where `pread` is missing), checked for its full length.

**Parallel cold rows.** One token's rows are independent, and an SSD serves many reads at once, so the rows a
gather misses are read by a small worker pool. Rows asked for twice are read once, and the result keeps request
order.

**Hot rows.** An optional least-recently-used cache holds rows that recur (frequent tokens and n-grams). Its size is
an engine choice charged against host memory.

## What the code supplies

| File | Supplies |
| --- | --- |
| [`table_rows.py`](code/table_rows.py) | `TableRegion`; `advise_random` (file) and `advise_mapping_random` (mapping); `RowTable` with `gather(rows)` in request order, a worker pool for cold rows, an LRU of hot rows, and `reads`, `bytes_read`, `hits` counters |

## What the engine supplies

- The table's layout: file, offset, row count and row size, including any per-row scales stored beside the codes.
- The row ids per token (token ids, or hashed n-gram ids from the model's own hashing).
- Where gathered rows go next (a pinned staging buffer, a device copy) and their arithmetic.
- The worker count and cache size, measured on the target disk.

## Worked example

Run from `code/`. A table of 64 rows of 16 bytes sits after a 32-byte header.

```python
import os, tempfile
from table_rows import RowTable, TableRegion

with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "table.bin")
    with open(path, "wb") as f:
        f.write(b"H" * 32 + b"".join(bytes([r]) * 16 for r in range(64)))
    table = RowTable(TableRegion(path, offset=32, rows=64, row_bytes=16), workers=4, cache_rows=8)
    rows = table.gather([5, 9, 5, 63])
    assert rows == [bytes([5]) * 16, bytes([9]) * 16, bytes([5]) * 16, bytes([63]) * 16]
    assert (table.reads, table.bytes_read) == (3, 48)        # row 5 read once, nothing but the rows
    table.gather([9, 63])
    assert (table.reads, table.hits) == (3, 2)               # both from the hot-row cache
    assert table.advice in ("fadvise_random", "none")        # "none" where the OS has no file advice
    table.close()
```

## Adapting it

Preserve: no read-ahead on the table (advice, or reads of exactly one row); reads checked for their full length;
rows returned in request order. Free: native reads (`io_uring`, overlapped I/O on Windows, Core's `ReadOnlyFile`
mapping with random advice), batching a whole prompt's rows, a host cache in pinned memory, and prefetching the next
token's rows where the engine knows them early.

## Costs and validation

A cold row costs one disk round trip; a warm one a memory copy. For scale only: one engine for a model with a 51 GB
n-gram table on WSL2 read 16 cold rows per token in 500 to 700 ms through a default mapping (8 MB of read-ahead per
row), and in 2 to 5 ms with random-access advice and parallel positioned reads. The page cache keeps rows that recur,
so a warm table can approach memory speed.

[Python fixtures](../../tests/test_techniques.py) check request order, duplicate rows, cache hits and eviction,
counters, rows outside the table, a file shorter than the table and short reads. None measure a disk.
