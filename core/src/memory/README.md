# Resource accounting

[resources.cpp](resources.cpp) implements ResourceBudget and reservation
accounting shared by the runtime and backends. Declarations are in
[core.h](../../include/crit/core.h); allocation services are in
[the CUDA backend](../backends/cuda/README.md).

## Host memory admission

A logical budget does not prove the operating system can back an allocation.
[host_memory.h](../../include/crit/host_memory.h) ([host_memory.cpp](host_memory.cpp))
reports what the host can still supply and checks a working set against it,
without allocating anything.

`observe_host_memory()` returns a `HostMemoryObservation` of three values in
bytes, each possibly unknown with an explaining note:

| Field | Linux | Windows |
| --- | --- | --- |
| `physical` | `MemAvailable` | `ullAvailPhys` |
| `commit` | `CommitLimit - Committed_AS`, only with `vm.overcommit_memory = 2` | `ullAvailPageFile` |
| `cgroup` | Tightest headroom over this process's cgroup and ancestors (v2 against `min(memory.max, memory.high)`, or the v1 memory controller), counting clean page cache as reclaimable: `active_file + inactive_file` (else `file - shmem`) minus dirty and writeback | unknown |

Malformed values never throw; they make that observation unknown and add a note.
`observe_host_memory_linux(root)` reads a synthetic `/proc` and `/sys/fs/cgroup`
tree for tests.

`admit_host(demand, observation, reserve = 0, require_physical = true)` checks a
`HostWorkingSet` of disjoint `anonymous`, `pinned` and `file` extents (`file` is
the additional clean file-backed working set, not the file size). Physical and
cgroup headroom pay all three; commit pays anonymous plus pinned. Zero demand is
admitted; with no known observation, or physical unknown under
`require_physical`, it is denied. Otherwise it is admitted when every known domain
fits demand plus `reserve`, and `binding`/`limit` name the tightest domain.

Observe right before a large host allocation, charge anonymous and pinned bytes
through `Resources::reserve(Memory::host, ...)`, and keep allocation failure
authoritative: admission is evidence, not a guarantee. Device memory is outside
this check.

## Large host blocks and the process pool

Engines whose weights live in host memory (routed experts, embeddings) allocate tens of GB that the device reads
through a mapping. [host_block.h](../../include/crit/host_block.h) ([host_block.cpp](host_block.cpp)) provides:

- `HostBlock(bytes)`: page-aligned anonymous pages (mmap / VirtualAlloc), not pinned. Fill it with the loader's
  threads, then pin it once with `cuda_pin_host_block(backend, block, touch_threads)` from
  [cuda_memory.h](../../include/crit/cuda_memory.h): page faults run in parallel during the fill and the
  registration is one call. Under WSL2 `cudaMallocHost` of 60 GB took about 90 s; fill-then-register took 35 s.
  `touch(threads)` reads every page once; after a registration under WSL2 the first CPU read of each page was slow
  (about 10 s over 60 GB on an engine's first request), so pin it with touch threads when the CPU reads the block.
  The registration is attached to the block and ends before the pages are freed; spans must not outlive the block.
- `HostBlockPool::process()`: blocks kept between engines opened one after another in one process (Eval rounds,
  serve switches). `acquire(key, bytes, keep, &reused, identity)` returns the cached block when key, size
  and identity match (pass what the bytes were filled from, such as the weight files' hashes, so another checkpoint
  of the same size never reuses them); otherwise it drops the cached block for that key before returning a new one, so an engine with a different
  layout (for example opened without its draft head) never holds a second copy. A block a live engine still holds
  is freed only when that engine releases it.

Charge the block through `Resources::reserve(Memory::host, ...)` and check it with `admit_host` first, as above.
