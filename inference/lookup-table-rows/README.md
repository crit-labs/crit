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

**Rows of draft tokens, while drafting.** In a speculative round, a draft token's rows are known the moment the token
exists, one draft step before the next and several before verify reads them. `RowPrefetch` starts them right then:
`start(rows)` queues a token's rows and returns a ticket, a service thread gathers every queued request in one batch
(consecutive draft tokens share the disk's parallelism), and `wait(ticket)` blocks only for rows still in flight. Read
after drafting instead, a round's rows (16 a token from an SSD) leave the GPU idle for the whole read. A rejected
draft's rows cost a read and stay in the hot-row cache. `stats()` reports how often a wait found its rows ready and
how long waits blocked, the idle time a round account charges to the table.

## What the code supplies

| File | Supplies |
| --- | --- |
| [`table_rows.py`](code/table_rows.py) | `TableRegion`; `advise_random` (file) and `advise_mapping_random` (mapping); `RowTable` with `gather(rows)` in request order, a worker pool for cold rows, an LRU of hot rows, and `reads`, `bytes_read`, `hits` counters; `RowPrefetch` (`start`, `ready`, `wait`) |
| [`table_rows.h`](code/table_rows.h) | The same in C++20 for native engines: `Region`, `RowGather` (random-access advice, positioned reads by a persistent worker pool with the calling thread, rows asked for twice read once, an LRU of hot rows, `Counters`) writing rows in request order straight into the caller's buffer, `StagingRing`, slots of pinned host memory a gather fills and the engine copies to the device, each reused only once its last copy has finished, and `RowPrefetch`: tickets started as tokens appear, queued requests gathered in one batch, `wait` per ticket, `stats` |

| [`device_rows.py`](code/device_rows.py) | `HostRows`, the plugin's device form ("embeddings:host", "rounds:embeddings"): `store(rows)` (a device table), `open(source, width, dtype, device, mapped=False)` (a CPU tensor's rows copied once into registered host memory, one file mapped and registered, or files read by the native reader), `prefetch(ids)`, `lookup(ids)`, `stats()` |
| [`row_reader.cpp`](code/row_reader.cpp) / [`row_reader.py`](code/row_reader.py) | The native reader the device form reads files with, built at load (host C++; no device code): files of rows in id order opened with POSIX_FADV_RANDOM, a persistent pool of `readers` threads taking rows off an atomic counter, rows asked for twice read once, one gather or prefetch batch at a time, a hot-row cache `prefetch` fills in the background, and `enqueue`: a gather as a CUDA host function on the stream |

**The device form.** A table on the device is indexed by `index_select`. A table in host memory (a tensor's rows,
or a file with `mapped=True`) lives in memory registered for the device, and `lookup` gathers the step's rows over
the link by `index_select` on a CUDA view of it: no device memory for the table, no copy per step, capture-safe
(memory-table-rows' `host` option is this). A table in files is read by the native reader: `lookup(ids)` copies the
ids into pinned memory, enqueues the reader's gather as a host function on the stream (a CUDA graph captures it as a
host node), and copies the rows to the device, so a captured round reads its rows with no Python on the way.
`prefetch(ids)` starts reading rows known ahead (the next prompt chunk's, a round's pending tokens') into the hot-row
cache (`prefetch_rows`), and the gather then copies them from memory. Each lookup size has `ring` sets of pinned and
device buffers used in turn, made at its first lookup outside a capture; a lookup's rows stay valid until `ring`
later lookups of that size. Without the native reader (it did not build, or `native: false`), Python threads read
the rows, which a graph cannot capture.

`store()` and `open()` return a `Table` handle (`lookup`, `prefetch`, `drain`, `stats`, `close`), so one plugin
serves several tables (token rows and per-layer n-gram rows); the plugin's own `lookup` and `prefetch` read the last
one. Rows are of one dtype, or stored NVFP4 (`format="nvfp4"`: the codes rows from `source`, element 2k in the low
nibble of byte k, the E4M3 scale rows from `scales`, the global scale `scale` as a multiplier), decoded on the device
by torch into rows of `dtype` (exact products in FP32, then one rounding by the scale and one to the dtype): a table
of codes and scales read from disk never exists decoded. Inside a captured round a file table's ids are known only on
the device; the copy, host node and copy above are the graph asking the host for its rows, so no mailbox is needed,
but the stream waits for the reads: rows the host knows ahead (the pending tokens, the next prompt chunk) belong in
`prefetch` so the host node only copies them.

**Into the device.** Gather each step's rows into a `StagingRing` slot over pinned host memory (a Core `HostBlock`
registered with `cuda_pin_host_block`), copy the slot to the device on the step's stream, and record an event after
the copy; the ring's `finished(slot)` is `cudaEventQuery` on that event, so a slot is overwritten only after its copy
has read it. Two or three slots let the next step's gather run while the current copy is in flight.

## What the engine supplies

- The table's layout: file, offset, row count and row size, including any per-row scales stored beside the codes.
- The row ids per token (token ids, or hashed n-gram ids from the model's own hashing).
- Where gathered rows go next and their arithmetic: the pinned memory behind the `StagingRing`, the device copy
  and the event its `finished` test reads.
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
rows returned in request order; a prefetched token's rows written before its wait returns. Free: native reads
(`io_uring`, overlapped I/O on Windows, Core's `ReadOnlyFile` mapping with random advice), batching a whole prompt's
rows, a host cache in pinned memory, and where a token's rows come from (a host drafter's token at once, a device
drafter's from mapped memory it writes, as [device-speculative-round](../device-speculative-round/README.md) shows).

## Costs and validation

A cold row costs one disk round trip; a warm one a memory copy. For scale only: one engine for a model with a 51 GB
n-gram table on WSL2 read 16 cold rows per token in 500 to 700 ms through a default mapping (8 MB of read-ahead per
row), and in 2 to 5 ms with random-access advice and parallel positioned reads. The page cache keeps rows that recur,
so a warm table can approach memory speed.

[Python fixtures](../../tests/test_techniques.py) check request order, duplicate rows, cache hits and eviction,
counters, rows outside the table, a file shorter than the table and short reads, and prefetched rows per token, each
ticket's rows handed out once, a failed read raised in its wait. The native fixture runs 40 draft tokens of 16 rows
through `RowPrefetch` (they share batches) and checks every row. None measure a disk.
