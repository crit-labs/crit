# Admit host memory against what the OS can actually back

Core's resource reservations are a logical budget: they prove the engine stays within
its own accounting, not that the operating system can back the bytes. This entry
observes physical, commit and cgroup headroom and admits a large host allocation only
when the tightest known limit allows it, so a budget-approved allocation does not push
the host into paging, hit the commit limit or trigger a cgroup OOM kill.

## When to use it

Use it before large host allocations (pinned staging, resident weights, a CPU KV cache)
when the process may share the host or run in a container. It is unnecessary on a
dedicated machine with measured headroom or for allocations small relative to free
memory.

## How it works

Three observations are independent, and an unknown one is reported with a reason, never
as zero:

| Observation | Source |
| --- | --- |
| Physical available | Linux `MemAvailable` (already counts reclaimable cache); Windows available physical memory |
| Commit headroom | Windows `ullAvailPageFile`; Linux `CommitLimit - Committed_AS` only with `vm.overcommit_memory = 2`, otherwise unknown |
| cgroup headroom (Linux) | Tightest level over the process's cgroup and ancestors: v2 `min(memory.max, memory.high) - memory.current + clean`, with `clean = active_file + inactive_file - file_dirty - file_writeback` (`file - shmem` when the file LRUs are not listed); v1 `hierarchical_memory_limit` with the same rule over `cache`, `dirty` and `writeback`, each in its `total_` form when listed, also used on hybrid hosts with no limited v2 level |

tmpfs and shared memory (`shmem`) are part of `file` and `cache` but are swap-backed, so
they are never counted as reclaimable. A level limited only by `memory.high` is limited:
past it the kernel throttles and reclaims the cgroup. A malformed counter counts nothing
as reclaimable and a malformed limit skips that level, each with a note.

Admission compares `request + reserve` with the tightest known observation and names it.
The reserve is the engine's room for the OS, other processes and unaccounted growth
(allocator slack, driver buffers). With no known observation the default is to deny.

The working-set variant takes disjoint incremental anonymous, pinned and clean
file-backed extents: physical and cgroup headroom pay all three, enforced commit pays
anonymous plus pinned, and the binding domain is the one with the least remaining
margin. Physical observation is required by default; zero demand is admitted without one.

## What the code supplies

Native engines call Core: `crit::core::observe_host_memory()`,
`observe_host_memory_linux(root)` and `admit_host(demand, observation, reserve,
require_physical)` in [host_memory.h](../../core/include/crit/host_memory.h)
([Core host memory admission](../../core/src/memory/README.md#host-memory-admission)),
with the working set as `HostWorkingSet{anonymous, pinned, file}`. This entry keeps the
Python reference and the guidance on what to count.

[memory.py](code/memory.py): `parse_meminfo`, `parse_stat`, `cgroup_headroom(root)`,
`observe_linux(root)` (any root, so fixtures can supply synthetic `/proc` and `/sys`),
`observe_windows()` via `GlobalMemoryStatusEx`, `observe()`, `Observation.limits()`,
`admit(request, observation, reserve, require_known)` returning
`Decision(admitted, binding, limit, reason)`, and `admit_working_set`, which follows
Core's `admit_host` rule and tie order.

## What the engine supplies

The reserve, the admission point and the response to denial (smaller chunks, fewer
resident items, or failing the request). Observe immediately before large allocations
and before raising a resident-cache capacity, and log the observation with the decision.
An observation is a snapshot, not a reservation: allocate, then charge Core resources,
and treat allocation failure as the authority. GPU memory is out of scope.

For mixed CPU/GPU plans, pass `admit_working_set` as the admission callback to
[measured dispatch](../heterogeneous-expert-dispatch/README.md). Count file pages touched
by CPU experts and upload sources, loader and repacking allocations, scratch and pinned
staging at their simultaneous peak; exclude already charged bytes, and never count
pinned extents as anonymous too. Large read-only mappings do not consume anonymous
commit.

## Worked example

Run from `code/`. A synthetic tree has 800 kB physical, enforced commit with 300 kB
headroom and a parent cgroup whose headroom counts only clean file pages.

```python
import os, tempfile
from memory import admit, observe_linux

def put(root, path, text):
    os.makedirs(os.path.dirname(os.path.join(root, path)), exist_ok=True)
    with open(os.path.join(root, path), "w") as f:
        f.write(text)

with tempfile.TemporaryDirectory() as root:
    put(root, "proc/meminfo", "MemAvailable: 800 kB\nCommitLimit: 900 kB\nCommitted_AS: 600 kB\n")
    put(root, "proc/sys/vm/overcommit_memory", "2\n")
    put(root, "proc/self/cgroup", "0::/app/worker\n")
    put(root, "sys/fs/cgroup/app/memory.max", "500000\n")
    put(root, "sys/fs/cgroup/app/memory.current", "450000\n")
    put(root, "sys/fs/cgroup/app/memory.stat", "file 60000\nfile_dirty 15000\nfile_writeback 5000\n")
    seen = observe_linux(root)

assert (seen.physical_available, seen.commit_available, seen.cgroup_available) == (819200, 307200, 90000)
assert admit(80000, seen, reserve=5000).admitted
denied = admit(80001, seen, reserve=10000)
assert not denied.admitted and denied.binding == "cgroup" and denied.limit == 80000

from memory import Observation, admit_working_set

# Clean file pages need RAM, but do not consume anonymous commit headroom.
seen = Observation(physical_available=1000, commit_available=100)
assert admit_working_set(anonymous_bytes=20, pinned_bytes=30, file_bytes=500,
                         observation=seen, reserve=10).admitted
assert not admit_working_set(anonymous_bytes=20, pinned_bytes=90, file_bytes=0,
                             observation=seen).admitted
```

## Adapting it

Preserve: unknowns stay unknown with a reason; only clean page cache (never shmem) is
reclaimable in cgroup headroom; commit headroom only where the OS enforces commit; admission against
the tightest known observation with an explicit reserve. Free: the reserve policy,
accepting unknowns, extra sources (Windows job objects, container runtimes, swap policy)
and how denials shrink the plan.

## Costs and validation

An observation reads a few small files or makes one system call. Another process can
take memory before the allocation, swapping pressure is not visible as a limit, and
unlimited nested cgroups or hidden `/sys/fs/cgroup` report cgroup headroom as unknown.

[Python fixtures](../../tests/test_techniques.py) build synthetic trees covering enforced
and unenforced commit, v2 ancestors with dirty, writeback and shmem pages, `memory.high`,
v1 limits, malformed counters and limits, missing files, exact-fit admission and the unknown policy, and run a live observation on the
build host. Core's [files and memory fixture](../../tests/core/io_tests.cpp) covers the
native path, and the [measured policy fixtures](../../tests/test_measured_expert_policies.py)
cover distinct binding limits and peak working-set accounting. None prove an admitted
allocation will succeed.
