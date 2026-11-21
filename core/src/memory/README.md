# Resource accounting

[resources.cpp](resources.cpp) implements ResourceBudget and reservation
accounting of the runtime. Declarations are in
[core/resources.h](../../include/crit/core/resources.h). A device's memory is Core's in Python:
`crit_plugins.memory` ([plugins](../../../docs/PLUGINS.md)).

A logical budget does not prove the operating system can back an allocation: Core's host memory admission
([crit_plugins.host.memory](../../python/crit_plugins/host/memory.py)) observes what the host can still supply
(physical memory, the commit headroom where the OS enforces commit, the cgroup's headroom) and checks a request or a
working set against it, and `crit_plugins.host.memory_bytes()`, which every host allocation is sized from, answers
with the tightest of them.
