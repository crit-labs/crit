# Resource accounting

[resources.cpp](resources.cpp) implements ResourceBudget and reservation
accounting of the runtime. Declarations are in
[core/resources.h](../../include/crit/core/resources.h). A device's memory is Core's in Python:
`crit_plugins.memory` ([plugins](../../../docs/PLUGINS.md)).

A logical budget does not prove the operating system can back an allocation: the
[host-memory-admission](../../../inference/host-memory-admission/README.md) entry observes what the host can still
supply and checks a working set against it.
