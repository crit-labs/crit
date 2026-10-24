#include "crit/host_memory.h"
#include <cstddef>
#include <fstream>
#include <limits>
#include <map>
#include <string_view>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace crit::core {
namespace {
namespace fs = std::filesystem;
using Note = std::vector<std::string>;
using Values = std::map<std::string, std::optional<std::uint64_t>, std::less<>>;   // nullopt: present but malformed
constexpr std::size_t max_source_bytes = std::size_t(1) << 20;
constexpr std::uint64_t u64_max = std::numeric_limits<std::uint64_t>::max();

std::uint64_t sub_floor(std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; }
std::uint64_t add_saturating(std::uint64_t a, std::uint64_t b) { return a > u64_max - b ? u64_max : a + b; }
bool space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
std::string_view trim(std::string_view text) {
    while(!text.empty() && space(text.front())) text.remove_prefix(1);
    while(!text.empty() && space(text.back())) text.remove_suffix(1);
    return text;
}
// Decimal digits only, without sign or overflow.
std::optional<std::uint64_t> parse_u64(std::string_view text) {
    if(text.empty()) return std::nullopt;
    std::uint64_t value = 0;
    for(const char c:text) {
        if(c < '0' || c > '9') return std::nullopt;
        const auto digit = std::uint64_t(c - '0');
        if(value > (u64_max - digit) / 10) return std::nullopt;
        value = value * 10 + digit;
    }
    return value;
}
std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    while(!text.empty()) {
        const auto end = text.find('\n');
        lines.push_back(text.substr(0, end));
        if(end == std::string_view::npos) break;
        text.remove_prefix(end + 1);
    }
    return lines;
}
std::vector<std::string_view> split_words(std::string_view text) {
    std::vector<std::string_view> words;
    std::size_t i = 0;
    while(i < text.size()) {
        while(i < text.size() && space(text[i])) ++i;
        const std::size_t start = i;
        while(i < text.size() && !space(text[i])) ++i;
        if(i > start) words.push_back(text.substr(start, i - start));
    }
    return words;
}
std::string shown(const fs::path& path) {
    try { const auto text = path.generic_u8string(); return std::string(text.begin(), text.end()); }
    catch(...) { return "<unrepresentable path>"; }
}
// A small text file, or nullopt when it cannot be read; oversized files are unreadable plus a note.
std::optional<std::string> read_text(const fs::path& path, Note& notes) {
    std::ifstream in(path, std::ios::binary);
    if(!in) return std::nullopt;
    std::string text;
    char buffer[4096];
    for(;;) {
        in.read(buffer, sizeof buffer);
        text.append(buffer, std::size_t(in.gcount()));
        if(text.size() > max_source_bytes) { notes.push_back(shown(path) + " exceeds 1 MiB"); return std::nullopt; }
        if(!in) break;
    }
    if(in.bad()) return std::nullopt;
    return text;
}
// /proc/meminfo "Key: value [kB]" lines; values in kB become bytes.
Values parse_meminfo(std::string_view text) {
    Values out;
    for(const auto line:split_lines(text)) {
        const auto colon = line.find(':');
        if(colon == std::string_view::npos) continue;
        const auto words = split_words(line.substr(colon + 1));
        auto value = words.empty() ? std::nullopt : parse_u64(words[0]);
        if(value && words.size() > 1 && words[1] == "kB")
            value = *value > u64_max / 1024 ? std::nullopt : std::optional<std::uint64_t>(*value * 1024);
        out[std::string(trim(line.substr(0, colon)))] = value;
    }
    return out;
}
// cgroup memory.stat "key value" lines.
Values parse_stat(std::string_view text) {
    Values out;
    for(const auto line:split_lines(text)) {
        const auto words = split_words(line);
        if(words.size() == 2) out[std::string(words[0])] = parse_u64(words[1]);
    }
    return out;
}
// Absent counts as zero; nullopt when present but malformed.
std::optional<std::uint64_t> stat_value(const Values& stat, std::string_view key) {
    const auto it = stat.find(key);
    return it == stat.end() ? std::optional<std::uint64_t>(0) : it->second;
}
// memory.stat key names per hierarchy; v1 prefers its hierarchical total_ form of each key when listed.
struct StatKeys { std::string_view cache, dirty, writeback; bool totals; };
constexpr StatKeys v2_keys{"file", "file_dirty", "file_writeback", false}, v1_keys{"cache", "dirty", "writeback", true};
std::string stat_key(const Values& stat, std::string_view name, bool totals) {
    std::string total = "total_" + std::string(name);
    return totals && stat.count(total) ? total : std::string(name);
}
// Clean page cache: the file LRUs (active_file + inactive_file, which exclude tmpfs/shmem pages) when
// listed, else cache - shmem; minus dirty and writeback. Malformed entries count nothing as reclaimable.
std::uint64_t clean_pages(const Values& stat, const StatKeys& keys, const fs::path& source, Note& notes) {
    const auto key = [&](std::string_view name) { return stat_key(stat, name, keys.totals); };
    const auto active = key("active_file"), inactive = key("inactive_file");
    const bool lru = stat.count(active) || stat.count(inactive);
    const auto a = stat_value(stat, lru ? active : key(keys.cache)), b = stat_value(stat, lru ? inactive : key("shmem"));
    const auto d = stat_value(stat, key(keys.dirty)), w = stat_value(stat, key(keys.writeback));
    if(!a || !b || !d || !w) {
        notes.push_back(shown(source) + " malformed; no file pages counted as reclaimable");
        return 0;
    }
    const auto cache = lru ? add_saturating(*a, *b) : sub_floor(*a, *b);
    return sub_floor(sub_floor(cache, *d), *w);
}
fs::path child(const fs::path& base, std::string_view absolute) {
    while(!absolute.empty() && absolute.front() == '/') absolute.remove_prefix(1);
    return absolute.empty() ? base : base / fs::path(std::string(absolute));
}
// The v1 memory controller's hierarchical limit at path, or unknown plus a note.
std::optional<std::uint64_t> cgroup_v1_headroom(const fs::path& root, const std::string& path, Note& notes) {
    const auto base = child(root / "sys/fs/cgroup/memory", path);
    const auto stat = parse_stat(read_text(base / "memory.stat", notes).value_or(""));
    const auto usage = read_text(base / "memory.usage_in_bytes", notes);
    const auto limit = stat.find("hierarchical_memory_limit");
    if(limit == stat.end() || !usage) { notes.push_back("no cgroup v1 memory limit"); return std::nullopt; }
    if(!limit->second) { notes.push_back("cgroup " + shown(base) + " hierarchical_memory_limit malformed"); return std::nullopt; }
    if(*limit->second >= (std::uint64_t(1) << 62)) { notes.push_back("no cgroup v1 memory limit"); return std::nullopt; }
    const auto used = parse_u64(trim(*usage));
    if(!used) { notes.push_back("cgroup " + shown(base) + " memory.usage_in_bytes malformed"); return std::nullopt; }
    const auto clean = clean_pages(stat, v1_keys, base / "memory.stat", notes);
    return sub_floor(add_saturating(*limit->second, clean), *used);
}
// Tightest headroom over this process's cgroup and its ancestors (v2). When no v2 level is limited
// and a v1 memory hierarchy is listed (systemd hybrid hosts keep the memory controller on v1), the
// v1 limit applies instead, so a hybrid host never reports an unlimited cgroup by mistake.
std::optional<std::uint64_t> cgroup_headroom(const fs::path& root, Note& notes) {
    const auto membership = read_text(root / "proc/self/cgroup", notes);
    if(!membership) { notes.push_back("cgroup membership unreadable"); return std::nullopt; }
    std::optional<std::string> v2, v1;
    for(const auto line:split_lines(*membership)) {
        const auto a = line.find(':');
        if(a == std::string_view::npos) continue;
        const auto b = line.find(':', a + 1);
        if(b == std::string_view::npos) continue;
        const auto hierarchy = line.substr(0, a), controllers = line.substr(a + 1, b - a - 1), path = line.substr(b + 1);
        if(hierarchy == "0" && controllers.empty()) { v2 = std::string(path); continue; }
        std::string_view rest = controllers;
        while(!rest.empty()) {
            const auto comma = rest.find(',');
            if(rest.substr(0, comma) == "memory") { v1 = std::string(path); break; }
            if(comma == std::string_view::npos) break;
            rest.remove_prefix(comma + 1);
        }
    }
    if(v2) {
        std::optional<std::uint64_t> best;
        std::string current = *v2;
        for(;;) {
            const auto base = child(root / "sys/fs/cgroup", current);
            // The effective limit is min(memory.max, memory.high): past memory.high the kernel throttles and
            // reclaims, so a level limited only by memory.high is still limited. Unreadable reads as "max".
            const auto max_file = read_text(base / "memory.max", notes), high_file = read_text(base / "memory.high", notes);
            const auto usage = read_text(base / "memory.current", notes);
            const auto max_text = max_file ? trim(*max_file) : "max", high_text = high_file ? trim(*high_file) : "max";
            if(usage && (max_text != "max" || high_text != "max")) {
                const auto max_bytes = max_text == "max" ? std::optional<std::uint64_t>(u64_max) : parse_u64(max_text);
                const auto high_bytes = high_text == "max" ? std::optional<std::uint64_t>(u64_max) : parse_u64(high_text);
                const auto used = parse_u64(trim(*usage));
                if(!max_bytes || !high_bytes || !used)
                    notes.push_back("cgroup " + shown(base) + " memory.max, memory.high or memory.current malformed");
                else {
                    const auto stat = parse_stat(read_text(base / "memory.stat", notes).value_or(""));
                    const auto clean = clean_pages(stat, v2_keys, base / "memory.stat", notes);
                    const auto headroom = sub_floor(add_saturating(std::min(*max_bytes, *high_bytes), clean), *used);
                    best = best ? std::min(*best, headroom) : headroom;
                }
            }
            if(current.empty() || current == "/") break;
            const auto slash = current.rfind('/');
            current = slash == 0 || slash == std::string::npos ? "/" : current.substr(0, slash);
        }
        if(best) return best;
    }
    if(v1) {
        const auto headroom = cgroup_v1_headroom(root, *v1, notes);
        if(!headroom && v2) notes.push_back("no cgroup v2 memory limit");
        return headroom;
    }
    notes.push_back(v2 ? "no cgroup v2 memory limit" : "no memory cgroup");
    return std::nullopt;
}
// Exact comparisons of sums up to four 64-bit terms.
struct Wide { std::uint64_t high = 0, low = 0; };
Wide plus(Wide a, std::uint64_t b) { a.low += b; if(a.low < b) ++a.high; return a; }
bool less(Wide a, Wide b) { return a.high != b.high ? a.high < b.high : a.low < b.low; }
std::string decimal(Wide value) { return value.high ? "more than 2^64" : std::to_string(value.low); }
}

HostMemoryObservation observe_host_memory_linux(const std::filesystem::path& root) {
    HostMemoryObservation o;
    const auto info = parse_meminfo(read_text(root / "proc/meminfo", o.notes).value_or(""));
    const auto available = info.find("MemAvailable");
    if(available == info.end()) o.notes.push_back("MemAvailable unavailable");
    else if(!available->second) o.notes.push_back("MemAvailable malformed");
    else o.physical = available->second;
    const auto text = read_text(root / "proc/sys/vm/overcommit_memory", o.notes);
    const std::string mode = text ? std::string(trim(*text)) : std::string("unknown");
    if(mode == "2") {
        const auto limit = info.find("CommitLimit"), committed = info.find("Committed_AS");
        if(limit != info.end() && limit->second && committed != info.end() && committed->second)
            o.commit = sub_floor(*limit->second, *committed->second);
        else o.notes.push_back("commit headroom unknown (CommitLimit or Committed_AS unavailable or malformed)");
    } else if(mode == "0" || mode == "1" || mode == "unknown") o.notes.push_back("commit not enforced (overcommit_memory=" + mode + ")");
    else o.notes.push_back("overcommit_memory malformed; commit unknown");
    o.cgroup = cgroup_headroom(root, o.notes);
    return o;
}

HostMemoryObservation observe_host_memory() {
#if defined(_WIN32)
    HostMemoryObservation o;
    MEMORYSTATUSEX status{};
    status.dwLength = DWORD(sizeof status);
    if(!GlobalMemoryStatusEx(&status)) { o.notes.push_back("GlobalMemoryStatusEx failed"); return o; }
    o.physical = status.ullAvailPhys;
    o.commit = status.ullAvailPageFile;   // commit limit minus committed bytes
    o.notes.push_back("no cgroups on Windows");
    return o;
#elif defined(__linux__)
    return observe_host_memory_linux("/");
#else
    HostMemoryObservation o;
    o.notes.push_back("unsupported platform");
    return o;
#endif
}

HostAdmission admit_host(const HostWorkingSet& demand, const HostMemoryObservation& observation, std::uint64_t reserve,
                         bool require_physical) {
    const Wide committed = plus(plus(Wide{}, demand.anonymous), demand.pinned);
    const Wide total = plus(committed, demand.file);
    if(!total.high && !total.low) return {true, "", std::nullopt, "no incremental host working set"};
    struct Domain { const char* name; std::uint64_t headroom; Wide cost; };   // cost = need + reserve
    Domain domains[3] = {};
    std::size_t known = 0;
    // Alphabetical, so the strict comparison below keeps the first name on ties.
    if(observation.cgroup) domains[known++] = {"cgroup", *observation.cgroup, plus(total, reserve)};
    if(observation.commit) domains[known++] = {"commit", *observation.commit, plus(committed, reserve)};
    if(observation.physical) domains[known++] = {"physical", *observation.physical, plus(total, reserve)};
    if(!known || (require_physical && !observation.physical)) {
        std::string why = "required host memory observation is unknown";
        for(const auto& note:observation.notes) why += "; " + note;
        return {false, "", std::nullopt, why};
    }
    // margin = headroom - cost; a's margin < b's margin exactly when a.headroom + b.cost < b.headroom + a.cost.
    std::size_t binding = 0;
    bool fits = true;
    for(std::size_t i = 0; i < known; ++i) {
        const auto& d = domains[i];
        if(less(Wide{0, d.headroom}, d.cost)) fits = false;
        const auto& b = domains[binding];
        if(i && less(plus(b.cost, d.headroom), plus(d.cost, b.headroom))) binding = i;
    }
    const auto& d = domains[binding];
    const auto limit = sub_floor(d.headroom, reserve);
    if(fits) return {true, d.name, limit, std::string("fits ") + d.name + " working-set headroom"};
    return {false, d.name, limit, std::string("exceeds ") + d.name + " working-set headroom: " +
            decimal(d.cost) + " bytes needed including a " + std::to_string(reserve) + "-byte reserve, " +
            std::to_string(d.headroom) + " available"};
}
}
