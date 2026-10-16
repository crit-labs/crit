#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace crit::core {
// Independent read-only observations in bytes; each may be unknown, and notes say why.
//   physical: obtainable without paging (Linux MemAvailable, Windows ullAvailPhys)
//   commit:   commit headroom where enforced (Windows ullAvailPageFile; Linux only with
//             vm.overcommit_memory = 2: CommitLimit - Committed_AS)
//   cgroup:   tightest ancestor cgroup headroom on Linux; only clean file pages
//             (file - file_dirty - file_writeback) count as reclaimable
// Malformed source values never throw: they become unknown (or zero reclaimable pages) plus a note.
struct HostMemoryObservation {
    std::optional<std::uint64_t> physical = {}, commit = {}, cgroup = {};
    std::vector<std::string> notes = {};
};
HostMemoryObservation observe_host_memory();
// Reads <root>/proc/meminfo, <root>/proc/sys/vm/overcommit_memory, <root>/proc/self/cgroup and the
// cgroup v2 tree (<root>/sys/fs/cgroup), or the v1 memory tree (<root>/sys/fs/cgroup/memory) when no
// v2 level is limited, as on hybrid hosts; "/" is the live system.
HostMemoryObservation observe_host_memory_linux(const std::filesystem::path& root);

// Disjoint incremental peak extents: anonymous excludes pinned; file is the additional clean
// file-backed working set (e.g. touched read-only mapping pages), not the file's whole size.
struct HostWorkingSet { std::uint64_t anonymous = 0, pinned = 0, file = 0; };
struct HostAdmission {
    bool admitted = false;
    std::string binding = {};                  // "cgroup", "commit", "physical" or empty
    std::optional<std::uint64_t> limit = {};   // binding headroom after the reserve (0 when the reserve exceeds it)
    std::string reason = {};
};
// Physical and cgroup headroom pay anonymous + pinned + file; commit pays anonymous + pinned. Admits
// when every known domain fits demand + reserve. Zero demand is admitted; otherwise no known domain,
// or physical unknown with require_physical, denies. binding is the smallest remaining margin;
// ties name the alphabetically first domain (cgroup, commit, physical). Exact for all inputs.
HostAdmission admit_host(const HostWorkingSet& demand, const HostMemoryObservation& observation,
                         std::uint64_t reserve = 0, bool require_physical = true);
}
