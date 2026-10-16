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
| `cgroup` | Tightest headroom over this process's cgroup and ancestors (v2, or the v1 memory controller), counting clean file pages as reclaimable | unknown |

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
