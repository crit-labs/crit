// StateCache fixtures. A fake storage keeps one value per KV row and per recurrent slot: the digest of the whole
// token prefix the state was computed for. A false prefix reuse, a write into a shared page or a lost commit then
// shows up as wrong content, not only as wrong counters.
#include <crit/state.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <new>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Counting global allocator: the hot-path case proves the cache allocates nothing after construction. Kept out of
// line so GCC's inlined new/free pairing heuristic does not misfire on the matching replacements.
#if defined(_MSC_VER)
#define STATE_TEST_NOINLINE __declspec(noinline)
#else
#define STATE_TEST_NOINLINE [[gnu::noinline]]
#endif
static std::atomic<long> allocations{0};
STATE_TEST_NOINLINE void* operator new(std::size_t size) {
    ++allocations;
    if(void* memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}
STATE_TEST_NOINLINE void operator delete(void* memory) noexcept { std::free(memory); }
STATE_TEST_NOINLINE void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
STATE_TEST_NOINLINE void* operator new[](std::size_t size) { return ::operator new(size); }
STATE_TEST_NOINLINE void operator delete[](void* memory) noexcept { ::operator delete(memory); }
STATE_TEST_NOINLINE void operator delete[](void* memory, std::size_t) noexcept { ::operator delete(memory); }
#undef STATE_TEST_NOINLINE

using namespace crit::core;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F action, std::string_view message) {
    try { action(); } catch(const std::exception& e) {
        if(std::string_view(e.what()).find(message) == std::string_view::npos)
            throw std::runtime_error("unexpected rejection: " + std::string(e.what()));
        return;
    }
    throw std::runtime_error("expected rejection missing: " + std::string(message));
}
template<class E, class F> void rejects_as(F action, std::string_view message) {
    try { action(); } catch(const E& e) {
        if(std::string_view(e.what()).find(message) == std::string_view::npos)
            throw std::runtime_error("unexpected rejection: " + std::string(e.what()));
        return;
    } catch(const std::exception& e) {
        throw std::runtime_error("rejection has the wrong type: " + std::string(e.what()));
    }
    throw std::runtime_error("expected rejection missing: " + std::string(message));
}

using Value = std::uint64_t;
using Tokens = std::vector<Token>;
// State after `token` given the state after the tokens before it.
Value digest(Value previous, Token token) {
    auto z = previous ^ (std::uint64_t(std::uint32_t(token))+0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 29))*0xbf58476d1ce4e5b9ull;
    return z ^ (z >> 32);
}
std::vector<Value> digests(const Tokens& tokens) {
    std::vector<Value> out;
    Value value = 0;
    for(auto token:tokens) out.push_back(value = digest(value,token));
    return out;
}
std::uint64_t constant_hash(std::uint64_t, std::uint64_t, std::span<const Token>) noexcept { return 7; }
std::uint64_t page_only_hash(std::uint64_t, std::uint64_t identity, std::span<const Token> tokens) noexcept {
    return chain_hash(0,identity,tokens);
}
// Every first page collides; later pages chain normally from that shared root hash.
std::uint64_t root_collision_hash(std::uint64_t previous, std::uint64_t identity, std::span<const Token> tokens) noexcept {
    return previous ? chain_hash(previous,identity,tokens) : 7;
}
// Distinct hashes whose home bucket in a 32-entry index (16 pages) is chosen by the page's first token: t / 10
// below 100, else the last bucket, so probe runs, holes and wrap-around are laid out by the test.
std::uint64_t homed_hash(std::uint64_t previous, std::uint64_t identity, std::span<const Token> tokens) noexcept {
    const auto home = tokens[0] < 100 ? std::uint64_t(tokens[0]/10) : 31u;
    return (chain_hash(previous,identity,tokens) << 20) | home;
}

struct Fake final : StateStorage {
    std::uint32_t page_tokens = 1;
    std::vector<Value> rows;                 // pages x page_tokens
    std::vector<std::vector<Value>> slots;   // components x slots
    std::vector<std::vector<Value>> windows; // window groups x pages x page_tokens
    std::vector<std::array<std::uint32_t,3>> row_copies, slot_copies;
    std::vector<std::array<std::uint32_t,4>> window_copies;
    long calls = 0, fail_at = -1;
    explicit Fake(const StateCacheConfig& c)
        : page_tokens(c.page_tokens ? c.page_tokens : 1), rows(std::size_t(c.pages)*page_tokens,0) {
        for(const auto& component:c.components) slots.emplace_back(component.slots,0);
        for(const auto& group:c.windows) windows.emplace_back(std::size_t(group.pages)*page_tokens,0);
        row_copies.reserve(1 << 14); slot_copies.reserve(1 << 14); window_copies.reserve(1 << 12);
    }
    void copy_window_rows(std::uint32_t window, PageId from, PageId to, std::uint32_t count) override {
        hook();
        require(from != to && count && count < page_tokens,"copy_window_rows needs a partial tail and a distinct page");
        auto& values = windows[window];
        for(std::uint32_t i = 0; i < count; ++i) values[std::size_t(to)*page_tokens+i] = values[std::size_t(from)*page_tokens+i];
        if(window_copies.size() < window_copies.capacity()) window_copies.push_back({window,from,to,count});
    }
    void hook() { if(calls++ == fail_at) throw std::runtime_error("injected storage failure"); }
    void copy_rows(PageId from, PageId to, std::uint32_t count) override {
        hook();
        require(from != to && count && count < page_tokens,"copy_rows needs a partial tail and a distinct page");
        for(std::uint32_t i = 0; i < count; ++i) rows[std::size_t(to)*page_tokens+i] = rows[std::size_t(from)*page_tokens+i];
        if(row_copies.size() < row_copies.capacity()) row_copies.push_back({from,to,count});
    }
    void copy_slot(std::uint32_t component, SlotId from, SlotId to) override {
        hook();
        require(from != to,"copy_slot needs distinct slots");
        slots[component][to] = slots[component][from];
        if(slot_copies.size() < slot_copies.capacity()) slot_copies.push_back({component,from,to});
    }
};

bool same_stats(const StateCacheStats& a, const StateCacheStats& b) {
    return a.free_pages == b.free_pages && a.cached_pages == b.cached_pages && a.retiring_pages == b.retiring_pages &&
           a.live_pages == b.live_pages && a.indexed_pages == b.indexed_pages && a.free_slots == b.free_slots &&
           a.retiring_slots == b.retiring_slots && a.live_slots == b.live_slots && a.sequences == b.sequences &&
           a.prefix_hits == b.prefix_hits && a.prefix_misses == b.prefix_misses && a.evictions == b.evictions;
}

// A simulated engine over one cache with an independent reference model of every open sequence.
struct Sim {
    struct Seq {
        SequenceId id;
        Tokens tokens;                    // committed
        std::vector<Value> values;        // values[r]: state after tokens[0..r]
        std::optional<Round> round;
        Tokens staged;                    // tokens of the pending round
        int doc = -1;                     // randomized walk: the document this sequence still follows
        std::uint64_t born = 0;           // evictions when its chain began (open, or the fork ancestor's open)
    };
    struct Snapshot {
        StateCacheStats stats;
        std::vector<std::vector<PageId>> tables, windows;
        std::vector<std::vector<SlotId>> lives;
        std::vector<std::uint32_t> counts;
        std::vector<std::uint64_t> layouts;
    };
    StateCacheConfig config;
    Resources resources;
    Fake storage;
    StateCache cache;
    std::list<Seq> seqs;
    std::uint32_t pt = 1, C = 0;
    // Retirement oracle: ids that left every sequence at an epoch not yet retired must not be written.
    std::map<PageId,std::uint64_t> page_retiring;
    std::vector<std::map<SlotId,std::uint64_t>> slot_retiring;
    std::set<PageId> last_pages;
    std::vector<std::set<SlotId>> last_slots;
    // The same for window pages, per group.
    std::uint32_t W = 0;
    std::vector<std::map<PageId,std::uint64_t>> window_retiring;
    std::vector<std::set<PageId>> last_windows;

    explicit Sim(const StateCacheConfig& c)
        : config(c), resources(std::size_t(1) << 30,0), storage(c), cache(c,storage,resources),
          pt(c.page_tokens), C(std::uint32_t(c.components.size())), slot_retiring(C), last_slots(C),
          W(std::uint32_t(c.windows.size())), window_retiring(W), last_windows(W) {}
    // First row the sequence's next row reads in window group w.
    std::size_t window_first(std::uint32_t w, std::size_t next) const {
        const std::size_t window = config.windows[w].window;
        return next+1 > window ? next+1-window : 0;
    }
    std::size_t window_holders(std::uint32_t w, PageId page) const {
        std::size_t count = 0;
        for(const auto& s:seqs) for(auto held:cache.window_table(s.id,w)) count += held == page;
        return count;
    }
    static Value last(const Seq& s) { return s.values.empty() ? 0 : s.values.back(); }
    RecurrentCommit mode(std::uint32_t c) const { return config.components[c].commit; }
    std::size_t rows_of(PageId page, std::uint32_t offset) const { return std::size_t(page)*pt+offset; }
    void fresh_slot(std::uint32_t c, SlotId slot) {
        require(slot != no_slot && !slot_retiring[c].count(slot),"slot reused before its epoch retired");
    }
    Seq& open() {
        Seq s;
        s.id = cache.open();
        s.born = cache.stats().evictions;
        for(std::uint32_t c = 0; c < C; ++c) {
            fresh_slot(c,cache.live_slot(s.id,c));
            storage.slots[c][cache.live_slot(s.id,c)] = 0;   // the engine initializes a new live slot
        }
        seqs.push_back(std::move(s));
        return seqs.back();
    }
    Seq& fork(Seq& parent) {
        Seq s;
        s.id = cache.fork(parent.id);
        for(std::uint32_t c = 0; c < C; ++c) fresh_slot(c,cache.live_slot(s.id,c));
        s.tokens = parent.tokens; s.values = parent.values; s.doc = parent.doc; s.born = parent.born;
        seqs.push_back(std::move(s));
        return seqs.back();
    }
    void close(Seq& s) {
        require(cache.close(s.id),"close rejected a live sequence");
        require(!cache.close(s.id),"close accepted a stale sequence");
        seqs.remove_if([&](const Seq& other) { return &other == &s; });
    }
    std::uint32_t attach(Seq& s, const Tokens& prompt, std::uint32_t max_reuse) {
        const auto expected = cache.reusable(prompt,max_reuse);
        std::uint32_t n = 0;
        try { n = cache.attach(s.id,prompt,max_reuse); }
        catch(...) {
            // A failed snapshot restore may have written the live slots: the engine initializes them again.
            for(std::uint32_t c = 0; c < C; ++c) storage.slots[c][cache.live_slot(s.id,c)] = 0;
            throw;
        }
        require(n == expected,"attach differs from reusable");
        require(n%pt == 0 && n <= max_reuse && n <= prompt.size(),"attach exceeded its cap or page granularity");
        s.tokens.assign(prompt.begin(),prompt.begin()+n);
        s.values = digests(s.tokens);
        return n;
    }
    // How many references the open sequences hold on a page.
    std::size_t holders(PageId page) const {
        std::size_t count = 0;
        for(const auto& s:seqs) for(auto held:cache.block_table(s.id)) count += held == page;
        return count;
    }
    std::optional<Round> begin(Seq& s, const Tokens& tokens, bool try_only = false) {
        const auto rows = std::uint32_t(tokens.size());
        auto round = try_only ? cache.try_begin(s.id,rows) : std::optional<Round>(cache.begin(s.id,rows));
        if(round) write(s,*round,tokens);
        return round;
    }
    // The engine writes every staged row and column checkpoint, after checking that no other owner can see it.
    void write(Seq& s, const Round& r, const Tokens& tokens) {
        const auto first = std::uint32_t(s.tokens.size()), rows = std::uint32_t(tokens.size());
        require(r.first_row == first && r.rows == rows && cache.staged(s.id) == rows,"round frontier differs");
        require(r.kv.size() == (config.pages ? rows : 0u) && r.start.size() == C && r.scratch.size() == std::size_t(C)*rows,
                "round span sizes differ");
        const auto table = cache.block_table(s.id);
        require(table.size() == (config.pages ? (first+rows+pt-1)/pt : 0u),"block table size differs");
        Value value = last(s);
        for(std::uint32_t i = 0; i < rows; ++i) {
            value = digest(value,tokens[i]);
            if(config.pages) {
                const auto slot = r.kv[i];
                require(slot.page == table[(first+i)/pt] && slot.offset == (first+i)%pt,"row slot differs from the block table");
                require(holders(slot.page) == 1,"a page with another owner was handed out for writing");
                require(!page_retiring.count(slot.page),"page written before its epoch retired");
                storage.rows[rows_of(slot.page,slot.offset)] = value;
            }
            for(std::uint32_t c = 0; c < C; ++c) if(mode(c) == RecurrentCommit::column_checkpoints) {
                const auto slot = r.scratch[std::size_t(c)*rows+i];
                fresh_slot(c,slot);
                storage.slots[c][slot] = value;
            }
            for(std::uint32_t w = 0; w < W; ++w) {
                const auto slot = r.window_kv[std::size_t(w)*rows+i];
                const auto window = cache.window_table(s.id,w);
                require(window.size() == table.size() && slot.page == window[(first+i)/pt] && slot.offset == (first+i)%pt,
                        "window row slot differs from the window table");
                require(window_holders(w,slot.page) == 1,"a window page with another owner was handed out for writing");
                require(!window_retiring[w].count(slot.page),"window page written before its epoch retired");
                storage.windows[w][rows_of(slot.page,slot.offset)] = value;
            }
        }
        require(r.window_kv.size() == std::size_t(W)*rows,"window row span size differs");
        for(std::uint32_t c = 0; c < C; ++c) {
            require(r.start[c] == cache.live_slot(s.id,c),"round start is not the live slot");
            const auto entries = r.scratch.subspan(std::size_t(c)*rows,rows);
            std::set<SlotId> distinct;
            for(std::uint32_t i = 0; i < rows; ++i) {
                const bool used = mode(c) == RecurrentCommit::column_checkpoints || (mode(c) == RecurrentCommit::recompute && i == 0);
                require(used == (entries[i] != no_slot),"scratch layout differs from the commit mode");
                if(used) { fresh_slot(c,entries[i]); require(distinct.insert(entries[i]).second && entries[i] != r.start[c],"scratch slot reused"); }
            }
        }
        s.round = r; s.staged = tokens;
    }
    void commit(Seq& s, std::size_t n) {
        const auto& r = *s.round;
        Value value = last(s);
        for(std::size_t i = 0; i < n; ++i) value = digest(value,s.staged[i]);
        std::vector<Value> kept(C);
        for(std::uint32_t c = 0; c < C && n; ++c) {
            if(mode(c) == RecurrentCommit::recompute) storage.slots[c][r.scratch[std::size_t(c)*r.rows]] = value;
            if(mode(c) == RecurrentCommit::in_place) {
                kept[c] = storage.slots[c][cache.live_slot(s.id,c)];
                storage.slots[c][cache.live_slot(s.id,c)] = value;
            }
        }
        try { cache.commit(s.id,r.id,std::span<const Token>(s.staged).first(n)); }
        catch(...) {
            // A failed snapshot copy leaves the round pending: the in-place update is the engine's to undo.
            for(std::uint32_t c = 0; c < C && n; ++c)
                if(mode(c) == RecurrentCommit::in_place) storage.slots[c][cache.live_slot(s.id,c)] = kept[c];
            throw;
        }
        for(std::size_t i = 0; i < n; ++i) { s.values.push_back(digest(last(s),s.staged[i])); s.tokens.push_back(s.staged[i]); }
        s.round.reset(); s.staged.clear();
    }
    void discard(Seq& s) {
        cache.discard(s.id,s.round->id);
        s.round.reset(); s.staged.clear();
    }
    void rewind(Seq& s, std::uint32_t tokens) {
        cache.rewind(s.id,tokens);
        s.tokens.resize(tokens); s.values.resize(tokens);
    }
    std::uint64_t advance_epoch() { track(); return cache.advance_epoch(); }
    void retire(std::uint64_t completed) {
        track();
        const auto closed = completed >= cache.epoch() ? cache.epoch() : completed;
        cache.retire(completed);
        std::erase_if(page_retiring,[&](const auto& entry) { return entry.second <= closed; });
        for(auto& pages:window_retiring) std::erase_if(pages,[&](const auto& entry) { return entry.second <= closed; });
        for(auto& slots:slot_retiring) std::erase_if(slots,[&](const auto& entry) { return entry.second <= closed; });
    }
    Snapshot snapshot() const {
        Snapshot shot;
        shot.stats = cache.stats();
        for(const auto& s:seqs) {
            const auto table = cache.block_table(s.id);
            shot.tables.emplace_back(table.begin(),table.end());
            for(std::uint32_t w = 0; w < W; ++w) {
                const auto window = cache.window_table(s.id,w);
                shot.windows.emplace_back(window.begin(),window.end());
            }
            std::vector<SlotId> lives;
            for(std::uint32_t c = 0; c < C; ++c) lives.push_back(cache.live_slot(s.id,c));
            shot.lives.push_back(lives);
            shot.counts.push_back(cache.committed(s.id)); shot.counts.push_back(cache.staged(s.id));
            shot.layouts.push_back(cache.layout_generation(s.id));
        }
        return shot;
    }
    // Exact equality, except that a failed copy-on-write which had to evict may cost that one cached page.
    static bool unchanged(const Snapshot& a, const Snapshot& b) {
        if(a.tables != b.tables || a.windows != b.windows || a.lives != b.lives || a.counts != b.counts || a.layouts != b.layouts) return false;
        if(same_stats(a.stats,b.stats)) return true;
        auto evicted = a.stats;
        evicted.cached_pages -= 1; evicted.free_pages += 1; evicted.indexed_pages -= 1; evicted.evictions += 1;
        return a.stats.free_pages == 0 && a.stats.cached_pages > 0 && same_stats(evicted,b.stats);
    }
    // Ids every open sequence holds: pages in block tables, live slots and pending scratch.
    void held(std::set<PageId>& pages, std::vector<std::set<SlotId>>& slots) const {
        for(const auto& s:seqs) {
            const auto table = cache.block_table(s.id);
            pages.insert(table.begin(),table.end());
            for(std::uint32_t c = 0; c < C; ++c) {
                slots[c].insert(cache.live_slot(s.id,c));
                if(s.round) for(auto slot:s.round->scratch.subspan(std::size_t(c)*s.round->rows,s.round->rows))
                    if(slot != no_slot) slots[c].insert(slot);
            }
        }
    }
    // Ids that left every sequence since the last call were released at the current epoch.
    void track() {
        std::set<PageId> pages;
        std::vector<std::set<SlotId>> slots(C);
        held(pages,slots);
        for(auto page:last_pages) if(!pages.count(page)) page_retiring[page] = cache.epoch();
        for(auto page:pages) page_retiring.erase(page);   // a re-attached page is read-only
        for(std::uint32_t w = 0; w < W; ++w) {
            std::set<PageId> windows;
            for(const auto& s:seqs) for(auto page:cache.window_table(s.id,w)) if(page != no_page) windows.insert(page);
            for(auto page:last_windows[w]) if(!windows.count(page)) window_retiring[w][page] = cache.epoch();
            for(auto page:windows) window_retiring[w].erase(page);
            last_windows[w] = windows;
        }
        for(std::uint32_t c = 0; c < C; ++c) {
            for(auto slot:last_slots[c]) if(!slots[c].count(slot)) slot_retiring[c][slot] = cache.epoch();
            last_slots[c] = slots[c];
        }
        last_pages = pages;
    }
    // Reference-model invariants after every step.
    void check() {
        std::set<PageId> pages;
        std::vector<std::set<SlotId>> slots(C);
        for(const auto& s:seqs) {
            const auto staged = s.round ? s.round->rows : 0u;
            require(cache.committed(s.id) == s.tokens.size() && cache.staged(s.id) == staged,"frontier differs from the model");
            const auto table = cache.block_table(s.id);
            const auto rows = s.tokens.size()+staged;
            require(table.size() == (config.pages ? (rows+pt-1)/pt : 0u),"block table size differs from the model");
            const std::set<PageId> own(table.begin(),table.end());
            require(own.size() == table.size(),"a block table holds a page twice");
            pages.insert(own.begin(),own.end());
            for(std::size_t r = 0; config.pages && r < s.tokens.size(); ++r)
                require(storage.rows[rows_of(table[r/pt],std::uint32_t(r%pt))] == s.values[r],"committed KV content differs");
            for(std::uint32_t c = 0; c < C; ++c) {
                const auto live = cache.live_slot(s.id,c);
                require(slots[c].insert(live).second,"a live slot is shared");
                require(storage.slots[c][live] == last(s),"live recurrent state differs");
                if(s.round) for(auto slot:s.round->scratch.subspan(std::size_t(c)*s.round->rows,s.round->rows))
                    if(slot != no_slot) require(slots[c].insert(slot).second,"a scratch slot is shared");
            }
        }
        for(std::uint32_t w = 0; w < W; ++w) {
            std::set<PageId> windows;
            for(const auto& s:seqs) {
                const auto table = cache.window_table(s.id,w);
                const auto block = cache.block_table(s.id);
                require(table.size() == block.size(),"window table size differs from the block table");
                std::size_t present = 0;
                for(auto page:table) if(page != no_page) { windows.insert(page); ++present; }
                require(std::set<PageId>(table.begin(),table.end()).size() == present+(present < table.size()),
                        "a window table holds a page twice");
                const auto values = s.values;
                for(auto r = window_first(w,s.tokens.size()); r < s.tokens.size(); ++r)
                    require(table[r/pt] != no_page && storage.windows[w][rows_of(table[r/pt],std::uint32_t(r%pt))] == values[r],
                            "window rows the next row reads differ");
            }
            const auto ws = cache.window_stats(w);
            require(ws.live_pages == windows.size(),"live window pages differ from the union of window tables");
            require(std::uint64_t(ws.free_pages)+ws.cached_pages+ws.retiring_pages+ws.live_pages == config.windows[w].pages,
                    "window page accounting");
            require(ws.cached_pages <= ws.indexed_pages,"cached window pages must be linked");
        }
        const auto st = cache.stats();
        require(st.live_pages == pages.size(),"live pages differ from the union of block tables");
        require(std::uint64_t(st.free_pages)+st.cached_pages+st.retiring_pages+st.live_pages == config.pages,"page accounting");
        require(st.cached_pages <= st.indexed_pages && st.indexed_pages <= config.pages,"index accounting");
        require(st.sequences == seqs.size(),"open sequence count differs");
        std::uint64_t live_slots = 0, total = 0;
        for(std::uint32_t c = 0; c < C; ++c) { live_slots += slots[c].size(); total += config.components[c].slots; }
        require(st.live_slots == live_slots+st.snapshots && st.free_slots+st.retiring_slots+st.live_slots == total,"slot accounting");
        track();
    }
    // Every committed full page of s is reusable now (default hash, no eviction since its chain began).
    void expect_complete(const Seq& s) {
        const auto n = std::uint32_t(s.tokens.size());
        require(cache.reusable(s.tokens,n) == n/pt*pt,"a committed chain is not reusable");
    }
    // Content check for a committed prefix read through another sequence's table.
    void expect_values(const Seq& s, const Tokens& tokens) {
        const auto values = digests(tokens);
        const auto table = cache.block_table(s.id);
        for(std::size_t r = 0; r < tokens.size(); ++r)
            require(storage.rows[rows_of(table[r/pt],std::uint32_t(r%pt))] == values[r],"reused KV content differs");
    }
};
StateCacheConfig attention(std::uint32_t pages, std::uint32_t page_tokens, std::uint32_t sequences,
                           std::uint32_t max_tokens, std::uint32_t round_rows, bool prefix = false) {
    StateCacheConfig c;
    c.pages = pages; c.page_tokens = page_tokens; c.sequences = sequences; c.max_tokens = max_tokens;
    c.max_round_rows = round_rows; c.prefix_cache = prefix; c.identity = prefix ? 0x51ce : 0;
    return c;
}
StateComponent component(RecurrentCommit commit, std::uint32_t slots) {
    StateComponent value;
    value.commit = commit; value.slots = slots;
    return value;
}
Tokens range(Token first, Token count) {
    Tokens out;
    for(Token i = 0; i < count; ++i) out.push_back(first+i);
    return out;
}

void config_rules() {
    Resources resources(std::size_t(1) << 24,0);
    const auto base = attention(4,2,1,8,2);
    Fake storage(base);
    const auto build = [&](auto edit) { auto c = base; edit(c); StateCache cache(c,storage,resources); };
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.page_tokens = 0; }); },"page_tokens");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.sequences = 0; }); },"sequences");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.max_tokens = 0; c.max_round_rows = 0; }); },"max_tokens");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.max_round_rows = 0; }); },"max_round_rows");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.max_round_rows = 9; }); },"max_round_rows");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.pages = no_page; }); },"page id range");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.pages = 0; }); },"pages or components");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.pages = 3; }); },"max_tokens exceeds pages");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) {
        c.prefix_cache = true; c.identity = 1; c.components = {component(RecurrentCommit::in_place,1)}; }); },"recurrent components");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) {
        c.prefix_cache = true; c.identity = 1; c.components = {component(RecurrentCommit::in_place,2)}; c.components[0].snapshots = 2; }); },
        "snapshots");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.pages = 0; c.components = {component(RecurrentCommit::in_place,1)};
        c.windows = {{4,2}}; }); },"windows require pages");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.windows = {{4,0}}; }); },"window must be positive");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.windows = {{2,2}}; }); },"window pages");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.windows = {{no_page,2}}; }); },"page id range");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.prefix_cache = true; }); },"identity");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) {
        c.components = {component(RecurrentCommit::column_checkpoints,2)}; }); },"cover every sequence");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.components = {component(RecurrentCommit::recompute,1)}; }); },
                                      "cover every sequence");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.components = {component(RecurrentCommit::in_place,no_slot)}; }); },
                                      "slot id range");
    rejects_as<std::invalid_argument>([&] { build([](auto& c) { c.components = {component(RecurrentCommit(7),4)}; }); },
                                      "commit mode");
    rejects_as<std::overflow_error>([&] { build([](auto& c) { c.pages = no_page-1; c.page_tokens = 0xffffffffu; }); },
                                    "accounting overflow");
    // Hundreds of gigabytes of metadata: refused by the ledger before anything is allocated.
    rejects_as<std::runtime_error>([&] { build([](auto& c) { c.sequences = 0xffffffffu; }); },"exceeds ceiling");
    require(resources.usage(Memory::host).live == 0,"rejected configurations kept a reservation");
    // Exact minimum slots are accepted; in_place needs no scratch.
    build([](auto& c) { c.components = {component(RecurrentCommit::column_checkpoints,3),component(RecurrentCommit::recompute,2),
                                         component(RecurrentCommit::in_place,1)}; });
    build([](auto& c) { c.pages = 0; c.components = {component(RecurrentCommit::in_place,1)}; });
    build([](auto& c) { c.prefix_cache = true; c.identity = 1; c.components = {component(RecurrentCommit::in_place,3)};
                        c.components[0].snapshots = 1; c.windows = {{4,3}}; });
    StateCache cache(base,storage,resources);
    require(cache.config().pages == 4 && cache.config().page_tokens == 2,"config differs");
    require(cache.epoch() == 1,"epochs start at one");
}

void transactions() {
    Sim sim(attention(5,2,2,6,4));
    auto& cache = sim.cache;
    auto& a = sim.open();
    auto r = *sim.begin(a,{1,2,3});
    require(cache.committed(a.id) == 0 && cache.staged(a.id) == 3 && cache.block_table(a.id).size() == 2,"staged frontier");
    require(r.kv[0].page == r.kv[1].page && r.kv[2].page != r.kv[0].page && r.kv[2].offset == 0,"row slots");
    sim.check();
    // A partial commit keeps the first rows and releases the staged rest at the current epoch.
    sim.commit(a,1);
    auto st = cache.stats();
    require(cache.committed(a.id) == 1 && cache.block_table(a.id).size() == 1,"partial commit frontier");
    require(st.live_pages == 1 && st.retiring_pages == 1 && st.free_pages == 3,"partial commit released pages");
    sim.retire(cache.epoch());
    require(cache.stats().free_pages == 4 && cache.stats().retiring_pages == 0,"retire returned pages");
    sim.check();
    // Zero accepted equals discard; discard drops every staged row.
    sim.begin(a,{4,5}); sim.commit(a,0); sim.check();
    require(cache.stats().retiring_pages == 1,"a zero commit released the staged page");
    r = *sim.begin(a,{4,5,6});
    sim.discard(a);
    sim.retire(cache.epoch());
    require(cache.committed(a.id) == 1 && cache.staged(a.id) == 0 && cache.block_table(a.id).size() == 1,"discard frontier");
    rejects_as<std::logic_error>([&] { cache.discard(a.id,r.id); },"stale state round");
    rejects_as<std::logic_error>([&] { cache.commit(a.id,r.id,{}); },"stale state round");
    sim.check();
    // A rejected commit consumes its round: the rows are discarded and the id goes stale.
    r = *sim.begin(a,{7,8});
    rejects_as<std::out_of_range>([&] { cache.commit(a.id,r.id,std::array<Token,3>{7,8,9}); },"exceeds staged rows");
    a.round.reset(); a.staged.clear();
    require(cache.staged(a.id) == 0 && cache.committed(a.id) == 1,"rejected commit left rows staged");
    rejects_as<std::logic_error>([&] { cache.commit(a.id,r.id,std::array<Token,1>{7}); },"stale state round");
    r = *sim.begin(a,{7,8});
    rejects_as<std::invalid_argument>([&] { cache.commit(a.id,r.id,std::array<Token,1>{-1}); },"nonnegative");
    a.round.reset(); a.staged.clear();
    require(cache.staged(a.id) == 0,"negative tokens left rows staged");
    sim.retire(cache.epoch());
    sim.check();
    // One pending round per sequence; a wrong round id consumes nothing.
    r = *sim.begin(a,{7});
    rejects_as<std::logic_error>([&] { cache.begin(a.id,1); },"already pending");
    rejects_as<std::logic_error>([&] { cache.try_begin(a.id,1); },"already pending");
    rejects_as<std::logic_error>([&] { cache.commit(a.id,r.id+1,std::array<Token,1>{7}); },"stale state round");
    rejects_as<std::logic_error>([&] { cache.pages_needed(a.id,1); },"already pending");
    rejects_as<std::logic_error>([&] { cache.rewind(a.id,0); },"pending round");
    require(cache.staged(a.id) == 1,"a stale round id consumed the round");
    sim.commit(a,1);
    sim.check();
    // Round limits: rows beyond max_round_rows are bad input, rows beyond max_tokens a logical overflow. pages_needed
    // never plans a round that begin would reject.
    rejects_as<std::invalid_argument>([&] { cache.begin(a.id,5); },"max_round_rows");
    rejects_as<std::invalid_argument>([&] { cache.try_begin(a.id,5); },"max_round_rows");
    rejects_as<std::invalid_argument>([&] { cache.pages_needed(a.id,5); },"max_round_rows");
    require(cache.pages_needed(a.id,4) == 2,"pages_needed within the round limit");
    sim.begin(a,{9,9,9,9}); sim.commit(a,4);
    require(cache.committed(a.id) == 6,"filled the sequence");
    rejects_as<std::length_error>([&] { cache.begin(a.id,1); },"exceeds max_tokens");
    rejects_as<std::length_error>([&] { cache.try_begin(a.id,1); },"exceeds max_tokens");
    rejects_as<std::length_error>([&] { cache.pages_needed(a.id,1); },"exceeds max_tokens");
    // A zero-row round acquires nothing and accepts only zero inputs.
    auto& b = sim.open();
    rejects_as<std::invalid_argument>([&] { cache.pages_needed(b.id,5); },"max_round_rows");
    auto zero = *sim.begin(b,{});
    require(zero.rows == 0 && zero.kv.empty() && cache.stats().live_pages == 3,"zero-row round acquired pages");
    sim.commit(b,0);
    zero = cache.begin(b.id,0);
    rejects_as<std::out_of_range>([&] { cache.commit(b.id,zero.id,std::array<Token,1>{1}); },"exceeds staged rows");
    sim.check();
    // Round ids are never reused, even across sequences.
    const auto r1 = cache.begin(b.id,1);
    cache.discard(b.id,r1.id);
    const auto r2 = cache.begin(b.id,1);
    require(r2.id > r1.id,"round ids repeat");
    cache.discard(b.id,r2.id);
    // Stale sequences.
    const auto stale = b.id;
    sim.close(b);
    rejects_as<std::logic_error>([&] { cache.committed(stale); },"stale state sequence");
    rejects_as<std::logic_error>([&] { cache.begin(stale,1); },"stale state sequence");
    rejects_as<std::logic_error>([&] { cache.block_table(stale); },"stale state sequence");
    rejects_as<std::logic_error>([&] { cache.attach(stale,Tokens{1},0); },"stale state sequence");
    auto& reopened = sim.open();
    require(reopened.id.index == stale.index && !(reopened.id == stale),"a reopened record kept its generation");
    rejects_as<std::out_of_range>([&] { cache.live_slot(reopened.id,0); },"component");
    sim.check();
}

void ownership() {
    Sim sim(attention(10,4,4,16,4));
    auto& cache = sim.cache;
    auto& a = sim.open();
    sim.begin(a,range(1,4)); sim.commit(a,4);
    sim.begin(a,range(5,2)); sim.commit(a,2);   // page 0 full, page 1 holds two rows
    const auto parent = std::vector<PageId>(cache.block_table(a.id).begin(),cache.block_table(a.id).end());
    // A fork shares committed pages without copying.
    auto& b = sim.fork(a);
    require(cache.stats().live_pages == 2 && sim.storage.row_copies.empty(),"fork copied eagerly");
    require(std::equal(parent.begin(),parent.end(),cache.block_table(b.id).begin()),"fork changed the table");
    require(cache.pages_needed(b.id,1) == 1 && cache.pages_needed(b.id,3) == 2,"copy-on-write counted in pages_needed");
    // The child's first round copies exactly the committed rows of the shared partial tail.
    sim.begin(b,{70});
    require(sim.storage.row_copies.size() == 1,"shared tail was not copied");
    const auto copy = sim.storage.row_copies[0];
    require(copy[0] == parent[1] && copy[2] == 2 && copy[1] == cache.block_table(b.id)[1],"copy-on-write rows");
    require(cache.block_table(b.id)[0] == parent[0],"a full shared page was copied");
    sim.check();
    sim.commit(b,1);
    // The parent's tail is exclusive again: no copy.
    const auto parent_round = *sim.begin(a,{80});
    require(sim.storage.row_copies.size() == 1 && parent_round.kv[0].page == parent[1],"exclusive tail copied");
    sim.commit(a,1);
    sim.begin(a,{81}); sim.commit(a,1);   // a: 8 rows, two full pages
    // A full shared tail is never copied; the child appends a fresh page.
    auto& c = sim.fork(a);
    sim.begin(c,{90});
    require(sim.storage.row_copies.size() == 1 && cache.block_table(c.id).size() == 3,"a full shared page was copied");
    sim.commit(c,1);
    // A discarded round keeps its private copy of the tail, with the same content.
    auto& d = sim.fork(b);   // b: 7 rows, three in its tail
    sim.begin(d,{11,12});
    require(sim.storage.row_copies.size() == 2 && sim.storage.row_copies[1][2] == 3,"second copy-on-write");
    sim.discard(d);
    require(cache.block_table(d.id)[1] != cache.block_table(b.id)[1],"discard dropped the private copy");
    sim.check();
    // Closing the parent leaves every child intact.
    sim.close(a);
    sim.check();
    sim.begin(b,{71}); sim.commit(b,1);
    sim.begin(d,{13}); sim.commit(d,1);
    sim.check();
    while(!sim.seqs.empty()) sim.close(sim.seqs.front());
    sim.retire(cache.epoch());
    require(cache.stats().free_pages == 10 && cache.stats().live_pages == 0,"teardown leaked pages");
}

void prefixes() {
    Sim sim(attention(16,4,5,32,12,true));
    auto& cache = sim.cache;
    const auto prompt = range(1,10);
    auto& a = sim.open();
    sim.begin(a,prompt); sim.commit(a,10);   // two full pages and a partial one
    require(cache.stats().indexed_pages == 2,"full committed pages were not indexed");
    const auto first = std::vector<PageId>(cache.block_table(a.id).begin(),cache.block_table(a.id).begin()+2);
    require(cache.reusable(prompt,9) == 8 && cache.reusable(prompt,7) == 4,"reuse cap");
    require(cache.reusable(range(1,8),7) == 4 && cache.reusable(range(1,8),8) == 8,"max_reuse below a full prompt");
    require(cache.reusable(Tokens{1,2,3,9,5,6,7,8,9},8) == 0 && cache.reusable(Tokens{1,2,3,4,9,9,9,9,9},8) == 4,
            "diverging prompts");
    require(cache.reusable(Tokens{},0) == 0 && cache.reusable(Tokens{1,2,3},3) == 0,"partial pages are never reused");
    rejects_as<std::invalid_argument>([&] { cache.reusable(Tokens{1,-2,3,4},4); },"nonnegative");
    // Live sharing: b attaches a's pages while a is open, then appends fresh pages without copying.
    auto& b = sim.open();
    require(sim.attach(b,prompt,9) == 8,"attach reused the full pages");
    require(std::equal(first.begin(),first.end(),cache.block_table(b.id).begin()),"attach used other pages");
    sim.expect_values(b,range(1,8));
    require(cache.stats().prefix_hits == 2 && cache.stats().prefix_misses == 0,"hit accounting");
    sim.begin(b,{20,21}); sim.commit(b,2);
    require(sim.storage.row_copies.empty(),"a full reused tail was copied");
    rejects_as<std::logic_error>([&] { cache.attach(b.id,prompt,9); },"empty sequence");
    sim.check();
    // First writer wins: an identical prefix computed independently stays unindexed.
    auto& c = sim.open();
    sim.begin(c,range(1,8)); sim.commit(c,8);
    require(cache.stats().indexed_pages == 2,"a duplicate page replaced the first writer");
    sim.close(a); sim.close(b);
    sim.retire(cache.epoch());
    require(cache.stats().cached_pages == 2,"released indexed pages are cached");
    auto& d = sim.open();
    require(sim.attach(d,prompt,9) == 8,"cached pages are reusable");
    require(std::equal(first.begin(),first.end(),cache.block_table(d.id).begin()),"reuse ignored the first writer");
    require(cache.stats().cached_pages == 0,"attached pages left the cached list");
    // Misses count the eligible pages that were not reused.
    auto& e = sim.open();
    require(sim.attach(e,Tokens{1,2,3,4,7,7,7,7,7},8) == 4,"partial reuse");
    require(cache.stats().prefix_hits == 5 && cache.stats().prefix_misses == 1,"miss accounting");
    sim.check();
    // Rewinding into an indexed page copies it before writing, so the indexed content stays intact.
    sim.rewind(d,6);
    sim.begin(d,{30});
    require(sim.storage.row_copies.size() == 1 && sim.storage.row_copies[0][0] == first[1] && sim.storage.row_copies[0][2] == 2,
            "an indexed tail was written in place");
    sim.commit(d,1);
    auto& f = sim.open();
    require(sim.attach(f,prompt,9) == 8,"indexed page lost after rewind");
    sim.expect_values(f,range(1,8));
    sim.check();
    // A duplicate takes a vacant index entry when released, so its content survives the first copy's eviction.
    while(!sim.seqs.empty()) sim.close(sim.seqs.front());
    sim.retire(cache.epoch());
    require(cache.evict(16) == 2,"evict took every cached page");
    require(cache.stats().cached_pages == 0 && cache.stats().indexed_pages == 0,"evict cleared the cache");
    auto& g = sim.open();
    sim.begin(g,range(1,4)); sim.commit(g,4);
    auto& h = sim.open();
    sim.begin(h,range(1,4)); sim.commit(h,4);      // h's page duplicates g's and stays unindexed
    const auto duplicate = cache.block_table(h.id)[0];
    sim.close(g);
    sim.retire(cache.epoch());
    require(cache.evict(1) == 1,"evicted the first writer");
    sim.close(h);                                  // the vacant entry now takes h's page
    sim.retire(cache.epoch());
    auto& k = sim.open();
    require(sim.attach(k,range(1,5),4) == 4 && cache.block_table(k.id)[0] == duplicate,"duplicate was not indexed on release");
    sim.expect_values(k,range(1,4));
    sim.check();
    // Without a prefix cache nothing is reused.
    Sim off(attention(4,2,1,8,2));
    auto& only = off.open();
    off.begin(only,{1,2}); off.commit(only,2);
    require(off.cache.reusable(Tokens{1,2,3},2) == 0 && off.cache.stats().indexed_pages == 0,"reuse without a prefix cache");
    auto& other = off.cache;
    off.close(only);
    auto& again = off.open();
    require(other.attach(again.id,Tokens{1,2,3},2) == 0,"attach without a prefix cache");
}

void collisions() {
    // Every hash equal: only the first page is indexed and every lookup is verified by tokens.
    auto c = attention(12,2,4,8,8,true);
    c.hash = constant_hash;
    Sim sim(c);
    auto& a = sim.open();
    sim.begin(a,{1,2,3,4}); sim.commit(a,4);
    require(sim.cache.stats().indexed_pages == 1,"colliding pages displaced the first writer");
    require(sim.cache.reusable(Tokens{1,2,3,4,5},4) == 2,"a colliding second page was reused");
    require(sim.cache.reusable(Tokens{5,6,1,2,5},4) == 0,"a colliding first page was reused");
    auto& b = sim.open();
    require(sim.attach(b,Tokens{5,6,7,8,9},4) == 0,"false reuse under a forced collision");
    sim.begin(b,{5,6,7,8}); sim.commit(b,4);
    sim.check();
    // A hash that ignores the chain: a page with the right tokens after a different prefix fails its parent link.
    auto d = attention(12,2,4,8,8,true);
    d.hash = page_only_hash;
    Sim chain(d);
    auto& x = chain.open();
    chain.begin(x,{1,2,3,4}); chain.commit(x,4);
    auto& y = chain.open();
    chain.begin(y,{5,6,7,8}); chain.commit(y,4);
    const auto x1 = chain.cache.block_table(x.id)[1];
    require(chain.cache.stats().indexed_pages == 4,"page-only hashes are distinct here");
    require(chain.cache.reusable(Tokens{5,6,3,4,9},4) == 2,"a page computed after another prefix was reused");
    auto& z = chain.open();
    require(chain.attach(z,Tokens{5,6,3,4,9},4) == 2,"verified prefix");
    chain.begin(z,{3,4}); chain.commit(z,2);
    require(chain.storage.rows[std::size_t(x1)*2+1] != digests(Tokens{5,6,3,4})[3],"the rejected page held other content");
    chain.check();
    require(chain.cache.reusable(Tokens{1,2,3,4,9},4) == 4,"the true chain is still reusable");
    // A duplicate's id is freed and reused for other tokens; its indexed child links to the canonical page instead.
    {
        Sim dup(d);
        auto& p = dup.open();
        dup.begin(p,{1,2}); dup.commit(p,2);
        auto& q = dup.open();
        dup.begin(q,{1,2,3,4}); dup.commit(q,4);   // q's first page duplicates p's; its second page is indexed
        const auto twin = dup.cache.block_table(q.id)[0];
        dup.close(q);
        dup.retire(dup.cache.epoch());
        auto& r = dup.open();
        dup.begin(r,{7,8}); dup.commit(r,2);
        require(dup.cache.block_table(r.id)[0] == twin,"the duplicate's id was reused");
        require(dup.cache.reusable(Tokens{7,8,3,4,9},4) == 2,"a page computed after [1,2] was reused after [7,8]");
        auto& t = dup.open();
        require(dup.attach(t,Tokens{7,8,3,4,9},4) == 2,"verified prefix after a reused id");
        dup.begin(t,{3,4}); dup.commit(t,2);
        dup.check();
    }
    // The parent page itself is evicted while its indexed child lives on, and its id comes back with other tokens.
    // The child's parent id matches again; only the generation tells the two incarnations apart.
    Sim gen(d);
    auto& g1 = gen.open();
    gen.begin(g1,{1,2}); gen.commit(g1,2);
    auto& g2 = gen.open();
    gen.begin(g2,{1,2,3,4}); gen.commit(g2,4);    // g2's second page links to g1's page and is indexed
    const auto canonical = gen.cache.block_table(g1.id)[0], child = gen.cache.block_table(g2.id)[1];
    require(gen.cache.reusable(Tokens{1,2,3,4,9},4) == 4,"a chain computed after a duplicate is reusable");
    gen.close(g1);
    gen.retire(gen.cache.epoch());
    require(gen.cache.evict(1) == 1 && gen.cache.reusable(Tokens{1,2,3,4,9},4) == 0,"the canonical parent was evicted");
    auto& g3 = gen.open();
    gen.begin(g3,{5,6}); gen.commit(g3,2);
    require(gen.cache.block_table(g3.id)[0] == canonical,"the evicted parent id was reused");
    require(gen.cache.stats().indexed_pages == 2,"the reused id and the orphaned child are indexed");
    require(gen.cache.reusable(Tokens{5,6,3,4,9},4) == 2,"a child of an earlier incarnation was reused");
    auto& g4 = gen.open();
    require(gen.attach(g4,Tokens{5,6,3,4,9},4) == 2,"verified prefix after a reused parent id");
    gen.expect_values(g4,Tokens{5,6});
    // Recomputing [3,4] after [5,6] takes the orphan's entry; the orphan's own chain is unreachable.
    gen.begin(g4,{3,4}); gen.commit(g4,2);
    require(gen.cache.block_table(g4.id)[1] != child && gen.cache.reusable(Tokens{5,6,3,4,9},4) == 4,
            "a linked page did not replace an orphaned entry");
    auto& g5 = gen.open();
    require(gen.attach(g5,Tokens{5,6,3,4,9},4) == 4,"the recomputed chain is reusable");
    gen.expect_values(g5,Tokens{5,6,3,4});
    gen.check();
    // When g2 leaves, its first page takes the vacant [1,2] entry; its second page loses the page-only collision.
    gen.close(g2);
    require(gen.cache.reusable(Tokens{1,2,3,4,9},4) == 2 && gen.cache.reusable(Tokens{5,6,3,4,9},4) == 4,
            "a released colliding page displaced a linked entry");
    gen.check();
    // Canonical links need the same tokens: [5,6] collides with the indexed [1,2], so the page after it must not
    // link to [1,2] and be reused there.
    {
        auto e = attention(12,2,4,8,8,true);
        e.hash = root_collision_hash;
        Sim root(e);
        auto& ra = root.open();
        root.begin(ra,{1,2}); root.commit(ra,2);
        auto& rb = root.open();
        root.begin(rb,{5,6,3,4}); root.commit(rb,4);
        require(root.cache.reusable(Tokens{1,2,3,4,9},4) == 2,"a page computed after a colliding root was reused");
        auto& t = root.open();
        require(root.attach(t,Tokens{1,2,3,4,9},4) == 2,"verified prefix after a colliding root");
        root.begin(t,{3,4}); root.commit(t,2);
        root.check();
        require(root.cache.reusable(Tokens{1,2,3,4,9},4) == 4 && root.cache.reusable(Tokens{5,6,3,4,9},4) == 0,
                "colliding roots");
    }
    // Canonical links need the same parent link: [3,4] after [5,6] has the same page-only hash and tokens as the
    // indexed [3,4] after [1,2], so the page after it must not link to the latter.
    Sim link(d);
    auto& la = link.open();
    link.begin(la,{1,2,3,4}); link.commit(la,4);
    auto& lb = link.open();
    link.begin(lb,{5,6,3,4,7,8}); link.commit(lb,6);
    require(link.cache.reusable(Tokens{1,2,3,4,7,8,9},6) == 4,"a page computed after [5,6,3,4] was reused after [1,2,3,4]");
    require(link.cache.reusable(Tokens{5,6,3,4,7,8,9},6) == 2,"a page after a colliding duplicate was linked");
    auto& lt = link.open();
    require(link.attach(lt,Tokens{1,2,3,4,7,8,9},6) == 4,"verified prefix after a colliding duplicate");
    link.begin(lt,{7,8}); link.commit(lt,2);
    link.check();
}

// Duplicate chains: two sequences compute the same content, so one copy of each page is indexed and pages computed
// after an unindexed copy must still be reachable through the index.
void reuse() {
    const auto config = attention(32,2,5,12,4,true);
    const Tokens prompt{1,2,3,4,5};
    const auto expect_reuse = [&](Sim& sim, const Tokens& tokens, std::uint32_t expected, const char* message) {
        require(sim.cache.reusable(tokens,std::uint32_t(tokens.size()-1)) == expected,message);
        auto& probe = sim.open();
        require(sim.attach(probe,tokens,std::uint32_t(tokens.size()-1)) == expected,message);
        sim.expect_values(probe,Tokens(tokens.begin(),tokens.begin()+expected));
        sim.check();
        sim.close(probe);
    };
    {
        // One sequence gets ahead at a page boundary: its continuation links to the indexed copy of [1,2].
        Sim sim(config);
        auto& a = sim.open();
        auto& b = sim.open();
        sim.begin(a,{1,2}); sim.commit(a,2);
        sim.begin(b,{1,2,3,4}); sim.commit(b,4);
        sim.begin(a,{3,4}); sim.commit(a,2);
        require(sim.cache.stats().indexed_pages == 2,"one copy of each page is indexed");
        expect_reuse(sim,prompt,4,"a chain computed after a duplicate is not reusable");
        sim.expect_complete(a); sim.expect_complete(b);
        sim.close(a); sim.close(b);
        sim.retire(sim.cache.epoch());
        require(sim.cache.stats().cached_pages == 2,"only the indexed copies are cached");
        expect_reuse(sim,prompt,4,"released duplicate chains lost reuse");
    }
    {
        // Two concurrent prefills of the same prompt continue differently; every later request reuses both pages.
        Sim sim(config);
        auto& s1 = sim.open();
        auto& s2 = sim.open();
        sim.begin(s1,{1,2}); sim.commit(s1,2);
        sim.begin(s2,{1,2}); sim.commit(s2,2);
        sim.begin(s1,{9,9}); sim.commit(s1,2);
        sim.begin(s2,{3,4}); sim.commit(s2,2);
        sim.close(s1); sim.close(s2);
        sim.retire(sim.cache.epoch());
        require(sim.cache.stats().indexed_pages == 3,"both continuations and one shared page are indexed");
        for(int request = 0; request < 3; ++request) {
            auto& s = sim.open();
            require(sim.attach(s,prompt,4) == 4,"a concurrent prefill's continuation is not reusable");
            sim.expect_values(s,Tokens{1,2,3,4});
            sim.close(s);
            sim.retire(sim.cache.epoch());
        }
        require(sim.cache.stats().prefix_misses == 0 && sim.cache.stats().indexed_pages == 3,"repeated requests missed");
    }
    {
        // Rewind and recommit of the same tokens makes an unindexed copy-on-write duplicate; the next page links to
        // the indexed original.
        Sim sim(config);
        auto& a = sim.open();
        sim.begin(a,{1,2,3,4}); sim.commit(a,4);
        sim.rewind(a,3);
        sim.begin(a,{4}); sim.commit(a,1);
        sim.begin(a,{5,6}); sim.commit(a,2);
        require(sim.storage.row_copies.size() == 1,"the rewound page was copied");
        sim.expect_complete(a);
        expect_reuse(sim,range(1,7),6,"a chain after a recommitted duplicate is not reusable");
    }
    for(const std::uint32_t evicted:{1u,2u}) {
        // The first copy's tail (or whole chain) is evicted; the duplicate chain takes the vacant entries on release,
        // head to tail, relinking to its own pages when the canonical parent is gone.
        Sim sim(config);
        auto& a = sim.open();
        auto& b = sim.open();
        sim.begin(a,{1,2,3,4}); sim.commit(a,4);
        sim.begin(b,{1,2,3,4}); sim.commit(b,4);
        sim.close(a);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(evicted) == evicted,"evicted the first copy");
        sim.close(b);
        sim.retire(sim.cache.epoch());
        require(sim.cache.stats().indexed_pages == 2 && sim.cache.stats().cached_pages == 2,"the duplicate chain was indexed");
        expect_reuse(sim,prompt,4,"a released duplicate chain is not reusable");
        auto& d = sim.open();
        sim.attach(d,prompt,4);
        sim.begin(d,{5,6}); sim.commit(d,2);
        sim.expect_complete(d);
    }
    {
        // An entry whose parent was evicted can never match. A linked recomputation replaces it, even while its owner
        // is still open.
        Sim sim(config);
        auto& s1 = sim.open();
        auto& s2 = sim.open();
        sim.begin(s1,{1,2}); sim.commit(s1,2);
        sim.begin(s2,{1,2,3,4}); sim.commit(s2,4);
        const auto orphan = sim.cache.block_table(s2.id)[1];
        sim.close(s1);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(1) == 1 && sim.cache.reusable(prompt,4) == 0,"the canonical parent was evicted");
        auto& s3 = sim.open();
        sim.begin(s3,{1,2,3,4}); sim.commit(s3,4);
        require(sim.cache.stats().indexed_pages == 2 && sim.cache.block_table(s3.id)[1] != orphan,"the orphan kept its entry");
        expect_reuse(sim,prompt,4,"a recomputed chain did not replace the orphan");
        sim.close(s2);   // its pages lose to the linked entries
        require(sim.cache.stats().indexed_pages == 2,"a released duplicate created a second entry");
        sim.check();
    }
    {
        // Only linked pages are indexed: after the original of a duplicate is evicted, the page computed after the
        // duplicate waits until the duplicate itself takes the vacant entry on release.
        Sim sim(config);
        auto& s1 = sim.open();
        auto& s2 = sim.open();
        sim.begin(s1,{1,2}); sim.commit(s1,2);
        sim.begin(s2,{1,2}); sim.commit(s2,2);
        sim.close(s1);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(1) == 1 && sim.cache.stats().indexed_pages == 0,"the original was evicted");
        sim.begin(s2,{3,4}); sim.commit(s2,2);
        require(sim.cache.stats().indexed_pages == 0,"a page after an unindexed duplicate was indexed");
        sim.close(s2);
        require(sim.cache.stats().indexed_pages == 2,"the released duplicate chain was not indexed");
        expect_reuse(sim,prompt,4,"the released duplicate chain is not reusable");
    }
    {
        // A copy-on-write release is a last release too: a rewound duplicate takes its vacant entry.
        Sim sim(config);
        auto& s1 = sim.open();
        auto& s2 = sim.open();
        sim.begin(s1,{1,2}); sim.commit(s1,2);
        sim.begin(s2,{1,2}); sim.commit(s2,2);
        sim.close(s1);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(1) == 1 && sim.cache.reusable(prompt,4) == 0,"the original was evicted");
        sim.rewind(s2,1);
        sim.begin(s2,{9});
        require(sim.storage.row_copies.size() == 1 && sim.cache.stats().indexed_pages == 1,"the copied duplicate was not indexed");
        sim.commit(s2,1);
        expect_reuse(sim,prompt,2,"the copied duplicate is not reusable");
    }
    {
        // The orphan is cached when its parent is evicted: the linked recomputation frees it immediately.
        Sim sim(config);
        auto& s1 = sim.open();
        auto& s2 = sim.open();
        sim.begin(s1,{1,2}); sim.commit(s1,2);
        sim.begin(s2,{1,2,3,4}); sim.commit(s2,4);
        sim.close(s1);
        sim.retire(sim.cache.epoch());
        sim.close(s2);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(1) == 1 && sim.cache.reusable(prompt,4) == 0,"the canonical parent was released first");
        const auto before = sim.cache.stats();
        require(before.cached_pages == 1 && before.indexed_pages == 1,"the orphan is cached");
        auto& s3 = sim.open();
        sim.begin(s3,{1,2,3,4}); sim.commit(s3,4);
        const auto after = sim.cache.stats();
        require(after.cached_pages == 0 && after.indexed_pages == 2 && after.evictions == before.evictions+1 &&
                after.free_pages == before.free_pages-1,"a replaced cached orphan was not freed");
        expect_reuse(sim,prompt,4,"the recomputed chain is not reusable");
    }
}

// Reuse completeness: with every staged token unique, every full committed page of a sequence is reusable right
// after its commit, whatever forks, attaches, rewinds, retirements and evictions came before. pages_needed matches
// what try_begin acquires.
void completeness() {
    int checks = 0;
    for(std::uint32_t trial = 0; trial < 60; ++trial) {
        std::mt19937 rng(7919+trial);
        StateCacheConfig c;
        c.page_tokens = 1+rng()%3; c.max_tokens = c.page_tokens*(2+rng()%6);
        c.max_round_rows = 1+rng()%std::min<std::uint32_t>(c.max_tokens,5);
        c.sequences = 1+rng()%5; c.pages = (c.max_tokens+c.page_tokens-1)/c.page_tokens+rng()%12;
        c.prefix_cache = true; c.identity = 5;
        Sim sim(c);
        auto& cache = sim.cache;
        std::vector<Tokens> docs;
        Token next = 1;
        const auto pick = [&](bool quiescent) -> Sim::Seq* {
            std::vector<Sim::Seq*> eligible;
            for(auto& s:sim.seqs) if(!quiescent || !s.round) eligible.push_back(&s);
            return eligible.empty() ? nullptr : eligible[rng()%eligible.size()];
        };
        for(int step = 0; step < 1200; ++step) {
            try {
                switch(rng()%10) {
                case 0: sim.open(); break;
                case 1: if(auto* s = pick(false)) { if(s->tokens.size() >= c.page_tokens) docs.push_back(s->tokens); sim.close(*s); } break;
                case 2: if(auto* s = pick(true)) sim.fork(*s); break;
                case 3: if(!docs.empty()) { auto& s = sim.open(); const auto& d = docs[rng()%docs.size()]; sim.attach(s,d,std::uint32_t(d.size())); } break;
                case 4: case 5: if(auto* s = pick(true)) {
                    const auto rows = std::min<std::uint32_t>(c.max_tokens-std::uint32_t(s->tokens.size()),1+rng()%c.max_round_rows);
                    const auto needed = cache.pages_needed(s->id,rows);
                    const auto before = cache.stats();
                    Tokens tokens;
                    for(std::uint32_t i = 0; i < rows; ++i) tokens.push_back(next++);
                    const auto round = sim.begin(*s,tokens,true);
                    const auto after = cache.stats();
                    const auto available = before.free_pages+before.cached_pages;
                    require(round ? available-(after.free_pages+after.cached_pages) == needed : needed > available,
                            "pages_needed differs from what try_begin acquired");
                }
                break;
                case 6: case 7: {
                    std::vector<Sim::Seq*> eligible;
                    for(auto& s:sim.seqs) if(s.round) eligible.push_back(&s);
                    if(eligible.empty()) break;
                    auto& s = *eligible[rng()%eligible.size()];
                    sim.commit(s,rng()%(s.round->rows+1));
                    sim.expect_complete(s);
                    ++checks;
                    break;
                }
                case 8: if(auto* s = pick(true); s && !s->tokens.empty()) sim.rewind(*s,std::uint32_t(rng()%(s->tokens.size()+1))); break;
                default: if(rng()%2) sim.retire(cache.epoch()); else cache.evict(std::uint32_t(rng()%3)); break;
                }
            } catch(const CapacityExhausted&) {}
            sim.check();
        }
    }
    require(checks > 4000,"the completeness walk committed too rarely");
}

// The prefix index is linear probing with backward-shift deletion. homed_hash lays out probe runs: removing an
// entry must keep every later entry of its run reachable, leave an entry already at or past its home in place, and
// handle runs that wrap past the last bucket. A released duplicate must find the shifted entry, not add a second.
void index_probing() {
    auto c = attention(16,2,8,4,2,true);
    c.hash = homed_hash;
    const auto one_page = [](Sim& sim, Token t) -> Sim::Seq& {
        auto& s = sim.open();
        sim.begin(s,{t,t}); sim.commit(s,2);
        return s;
    };
    const auto reusable = [](Sim& sim, Token t) { return sim.cache.reusable(Tokens{t,t,99},2); };
    {
        // Buckets: a(home 0) 0, b(home 0) 1, e(home 2) 2, x(home 0) 3, y(home 0) 4.
        Sim sim(c);
        auto& a = one_page(sim,1);
        auto& b = one_page(sim,2);
        auto& e = one_page(sim,20);
        auto& x = one_page(sim,3);
        auto& y = one_page(sim,4);
        require(sim.cache.stats().indexed_pages == 5,"five distinct pages are indexed");
        // Release order sets the eviction order: b (the middle of the run), then a (its head), then the rest.
        sim.close(b); sim.close(a); sim.close(e); sim.close(x); sim.close(y);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(1) == 1 && reusable(sim,2) == 0,"evicted the middle of the probe run");
        for(Token t:{1,20,3,4}) require(reusable(sim,t) == 2,"a probe-run entry was lost after removing its middle");
        require(sim.cache.evict(1) == 1 && reusable(sim,1) == 0,"evicted the head of the probe run");
        for(Token t:{20,3,4}) require(reusable(sim,t) == 2,"a probe-run entry was lost after removing its head");
        require(sim.cache.stats().indexed_pages == 3 && sim.cache.stats().cached_pages == 3,"index accounting after removals");
        // Duplicates of shifted entries, computed and released, find them instead of filling a hole.
        auto& dup = one_page(sim,4);
        auto& dup2 = one_page(sim,20);
        require(sim.cache.stats().indexed_pages == 3,"a duplicate of a shifted entry was indexed");
        sim.close(dup); sim.close(dup2);
        sim.retire(sim.cache.epoch());
        require(sim.cache.stats().indexed_pages == 3 && sim.cache.stats().cached_pages == 3,"a released duplicate added an entry");
        // Removed hashes are indexed again in vacated buckets, and every entry stays reachable.
        auto& again = one_page(sim,2);
        auto& again2 = one_page(sim,1);
        require(sim.cache.stats().indexed_pages == 5,"removed hashes were not indexed again");
        for(Token t:{1,2,20,3,4}) require(reusable(sim,t) == 2,"an entry was lost after reinsertion");
        sim.close(again); sim.close(again2);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(16) == 5 && sim.cache.stats().indexed_pages == 0,"evicting everything empties the index");
        for(Token t:{1,2,20,3,4}) require(reusable(sim,t) == 0,"an evicted page stayed reusable");
        sim.check();
    }
    {
        // A run wrapping past the last bucket: p(home 31) 31, q(home 0) 0, r(home 31) 1, t(home 31) 2.
        Sim sim(c);
        auto& p = one_page(sim,100);
        auto& q = one_page(sim,1);
        auto& r = one_page(sim,101);
        auto& t = one_page(sim,102);
        sim.close(p); sim.close(q); sim.close(r); sim.close(t);
        sim.retire(sim.cache.epoch());
        require(sim.cache.evict(1) == 1 && reusable(sim,100) == 0,"evicted the run's last-bucket entry");
        for(Token k:{1,101,102}) require(reusable(sim,k) == 2,"a wrapped probe-run entry was lost");
        auto& dup = one_page(sim,102);
        require(sim.cache.stats().indexed_pages == 3,"a duplicate of a wrapped entry was indexed");
        sim.close(dup);
        require(sim.cache.evict(1) == 1 && reusable(sim,1) == 0,"evicted the entry at bucket 0");
        for(Token k:{101,102}) require(reusable(sim,k) == 2,"a wrapped entry was lost after removing bucket 0");
        require(sim.cache.stats().indexed_pages == 2,"wrapped index accounting");
        sim.check();
    }
}

void lru() {
    Sim sim(attention(6,2,3,12,8,true));
    auto& cache = sim.cache;
    auto& a = sim.open();
    sim.begin(a,range(1,6)); sim.commit(a,6);
    const auto chain = std::vector<PageId>(cache.block_table(a.id).begin(),cache.block_table(a.id).end());
    sim.close(a);
    sim.retire(cache.epoch());
    require(cache.stats().cached_pages == 3 && cache.stats().free_pages == 3,"released chain is cached");
    // Allocation pressure takes free pages first, then evicts the least recently released page: the chain's tail.
    auto& b = sim.open();
    const auto round = *sim.begin(b,range(20,8));
    require(cache.stats().evictions == 1 && cache.stats().cached_pages == 2,"pressure evicted one cached page");
    require(round.kv[6].page == chain[2],"eviction did not take the chain's tail");
    require(cache.reusable(range(1,7),6) == 4,"eviction took a prefix page");
    sim.commit(b,8);
    // Referenced pages are never evicted: with every page referenced, allocation fails without change.
    auto& c = sim.open();
    require(sim.attach(c,range(1,5),4) == 4 && cache.stats().cached_pages == 0,"attach referenced cached pages");
    auto& d = sim.open();
    const auto before = sim.snapshot();
    require(!sim.begin(d,{1},true),"allocated a referenced page");
    rejects_as<CapacityExhausted>([&] { cache.begin(d.id,1); },"pages exhausted");
    require(Sim::unchanged(before,sim.snapshot()) && same_stats(before.stats,sim.snapshot().stats),"failed allocation changed state");
    require(cache.evict(10) == 0,"evicted referenced pages");
    sim.check();
    // Reattaching only the first page makes it the most recently released, so the second page goes first.
    sim.close(c);
    sim.retire(cache.epoch());
    require(sim.attach(d,range(1,3),2) == 2,"first page reusable");
    sim.close(d);
    sim.retire(cache.epoch());
    require(cache.evict(1) == 1 && cache.reusable(range(1,5),4) == 2,"LRU order ignored the reattached page");
    require(cache.stats().evictions == 2 && cache.stats().indexed_pages == 5 && cache.stats().free_pages == 1,
            "explicit eviction accounting");
    require(cache.evict(0) == 0,"evict(0) evicted");
    sim.check();
}

void epochs() {
    Sim sim(attention(3,2,2,6,4));
    auto& cache = sim.cache;
    auto& a = sim.open();
    sim.begin(a,{1,2,3}); sim.commit(a,1);   // one page retires at epoch 1
    sim.check();
    const auto closed = sim.advance_epoch();
    require(closed == 1 && cache.epoch() == 2,"advance_epoch returns the closed epoch");
    auto& b = sim.open();
    sim.begin(b,{5,6}); sim.commit(b,2);    // takes the last free page
    require(!sim.begin(b,{7},true),"a retiring page was allocated");
    sim.check();
    // Work submitted up to the closed epoch settled: its releases become allocatable, last freed first.
    sim.retire(closed);
    require(cache.stats().free_pages == 1 && cache.epoch() == 2,"retire of a closed epoch");
    const auto layout = cache.layout_generation(b.id);
    sim.begin(b,{7}); sim.commit(b,1);
    require(cache.layout_generation(b.id) != layout,"a reused page id kept the layout");
    sim.check();
    // Releases in a later epoch wait for that epoch; retire(epoch()) closes the current one.
    sim.close(a);
    sim.retire(closed);
    require(cache.stats().retiring_pages == 1,"an early retire freed a later release");
    sim.retire(cache.epoch());
    require(cache.stats().retiring_pages == 0 && cache.epoch() == 3,"retire(epoch()) closes the epoch");
    sim.retire(1000);
    require(cache.epoch() == 4,"a future epoch retires only the current one");
    sim.check();
    // A retiring indexed page can be referenced again before it retires (reads only).
    Sim cached(attention(4,2,2,8,2,true));
    auto& x = cached.open();
    cached.begin(x,{1,2}); cached.commit(x,2);
    cached.close(x);
    require(cached.cache.stats().retiring_pages == 1 && cached.cache.stats().indexed_pages == 1,"released indexed page retires");
    auto& y = cached.open();
    require(cached.attach(y,Tokens{1,2,3},2) == 2,"a retiring indexed page was not reusable");
    require(cached.cache.stats().retiring_pages == 0 && cached.cache.stats().live_pages == 1,"reattached page left retirement");
    cached.retire(cached.cache.epoch());
    cached.close(y);
    cached.retire(cached.cache.epoch());
    require(cached.cache.stats().cached_pages == 1,"page cached after retirement");
    cached.check();
    // Slots follow the same epochs.
    auto s = attention(0,1,1,8,2);
    s.components = {component(RecurrentCommit::column_checkpoints,3)};
    Sim slots(s);
    auto& q = slots.open();
    slots.begin(q,{1,2}); slots.commit(q,1);
    require(slots.cache.stats().retiring_slots == 2 && slots.cache.stats().free_slots == 0,"released slots retire");
    require(!slots.begin(q,{3},true),"a retiring slot was allocated");
    slots.retire(slots.cache.epoch());
    const auto r = *slots.begin(q,{3,4});
    require(r.scratch[0] == 2 && r.scratch[1] == 0,"slots return in release order to a LIFO stack");
    slots.commit(q,2);
    slots.check();
}

void forks() {
    auto c = attention(8,2,3,8,2);
    c.components = {component(RecurrentCommit::column_checkpoints,9),component(RecurrentCommit::recompute,6)};
    Sim sim(c);
    auto& cache = sim.cache;
    auto& a = sim.open();
    sim.begin(a,{1,2}); sim.commit(a,2);
    sim.begin(a,{3}); sim.commit(a,1);
    sim.retire(cache.epoch());
    sim.check();
    auto& b = sim.fork(a);
    require(sim.storage.slot_copies.size() == 2,"fork copies each component once");
    for(std::uint32_t k = 0; k < 2; ++k) {
        const auto copy = sim.storage.slot_copies[k];
        require(copy[0] == k && copy[1] == cache.live_slot(a.id,k) && copy[2] == cache.live_slot(b.id,k),"fork slot copy");
    }
    require(cache.committed(b.id) == 3 && cache.stats().live_pages == 2,"fork shares committed pages");
    sim.check();
    // Fork needs a quiescent parent.
    const auto pending = cache.begin(a.id,1);
    rejects_as<std::logic_error>([&] { cache.fork(a.id); },"no pending round");
    cache.discard(a.id,pending.id);
    sim.retire(cache.epoch());
    // Out of sequence records: no change.
    auto& c2 = sim.fork(b);
    const auto before = sim.snapshot();
    rejects_as<CapacityExhausted>([&] { cache.fork(a.id); },"sequences exhausted");
    rejects_as<CapacityExhausted>([&] { cache.open(); },"sequences exhausted");
    require(Sim::unchanged(before,sim.snapshot()) && same_stats(before.stats,sim.snapshot().stats),"failed fork changed state");
    // Children advance independently; closing the parent leaves them intact.
    sim.begin(b,{4,5}); sim.commit(b,2);
    sim.retire(cache.epoch());
    sim.begin(c2,{6}); sim.commit(c2,1);
    const auto stale = a.id;
    sim.close(a);
    sim.retire(cache.epoch());
    sim.begin(b,{7}); sim.commit(b,1);
    sim.check();
    rejects_as<std::logic_error>([&] { cache.fork(stale); },"stale state sequence");
}

void recurrent() {
    for(auto mode:{RecurrentCommit::column_checkpoints,RecurrentCommit::recompute,RecurrentCommit::in_place}) {
        const std::uint32_t scratch = mode == RecurrentCommit::column_checkpoints ? 4 : mode == RecurrentCommit::recompute ? 1 : 0;
        auto c = attention(0,1,2,200,4);
        c.components = {component(mode,2*(1+scratch))};
        Sim sim(c);
        auto& cache = sim.cache;
        auto& a = sim.open();
        const auto live = cache.live_slot(a.id,0);
        auto r = *sim.begin(a,{1,2,3,4});
        require(r.kv.empty() && r.start[0] == live,"recurrent round shape");
        sim.check();
        const auto layout = cache.layout_generation(a.id);
        sim.commit(a,3);
        if(mode == RecurrentCommit::column_checkpoints) require(cache.live_slot(a.id,0) == r.scratch[2],"live adopts scratch[n-1]");
        if(mode == RecurrentCommit::recompute) require(cache.live_slot(a.id,0) == r.scratch[0],"live adopts the recompute slot");
        if(mode == RecurrentCommit::in_place) require(cache.live_slot(a.id,0) == live && cache.layout_generation(a.id) == layout,
                                                      "in_place keeps the live slot");
        require(cache.stats().retiring_slots == (mode == RecurrentCommit::in_place ? 0u : scratch),"released slots retire");
        sim.check();
        // Zero accepted keeps the live slot and releases the scratch.
        sim.retire(cache.epoch());
        const auto kept = cache.live_slot(a.id,0);
        sim.begin(a,{5,6}); sim.commit(a,0);
        require(cache.live_slot(a.id,0) == kept && cache.stats().retiring_slots == std::min(scratch,2u),"zero commit");
        sim.begin(a,{5}); sim.discard(a);
        require(cache.live_slot(a.id,0) == kept,"discard kept the live slot");
        // A zero-step round.
        const auto zero = cache.begin(a.id,0);
        require(zero.scratch.empty() && zero.start[0] == kept,"zero-step round");
        cache.commit(a.id,zero.id,{});
        sim.retire(cache.epoch());
        sim.check();
        // Chunked continuation across rounds, then the 63/64/65 boundaries of a round starting after 62 tokens.
        while(a.tokens.size() < 62) {
            sim.begin(a,range(Token(a.tokens.size()),std::min<Token>(4,Token(62-a.tokens.size()))));
            sim.commit(a,a.staged.size());
            sim.retire(cache.epoch());
        }
        r = *sim.begin(a,{7,8,9,10});
        const auto expected = digests([&] { auto t = a.tokens; t.insert(t.end(),{7,8,9,10}); return t; }());
        const std::array<std::size_t,3> accepted{0,1,2};
        require(!checkpoint_node(62,accepted,62),"the round start has no node");
        for(std::uint64_t boundary:{63u,64u,65u}) {
            const auto node = *checkpoint_node(62,accepted,boundary);
            require(node == boundary-63,"checkpoint node");
            if(mode == RecurrentCommit::column_checkpoints)
                require(sim.storage.slots[0][r.scratch[node]] == expected[boundary-1],"column checkpoint holds the after-token state");
        }
        sim.commit(a,3);
        require(sim.storage.slots[0][cache.live_slot(a.id,0)] == expected[64],"committed state after 65 tokens");
        sim.retire(cache.epoch());
        sim.check();
        rejects_as<std::logic_error>([&] { cache.rewind(a.id,10); },"cannot rewind");
        cache.rewind(a.id,65);
        // A second sequence and a fork share nothing mutable.
        auto& b = sim.fork(a);
        sim.begin(b,{1}); sim.commit(b,1);
        sim.retire(cache.epoch());
        sim.begin(a,{2}); sim.commit(a,1);
        sim.check();
    }
}

void hybrid() {
    auto c = attention(10,4,3,32,3);
    c.components = {component(RecurrentCommit::column_checkpoints,12),component(RecurrentCommit::recompute,6)};
    Sim sim(c);
    auto& cache = sim.cache;
    auto& a = sim.open();
    const auto r = *sim.begin(a,{1,2,3});
    sim.commit(a,2);
    require(cache.live_slot(a.id,0) == r.scratch[1] && cache.live_slot(a.id,1) == r.scratch[3],"one commit covers every component");
    require(cache.block_table(a.id).size() == 1 && cache.committed(a.id) == 2,"one commit covers the KV rows");
    sim.check();
    auto& b = sim.fork(a);
    sim.begin(b,{4,5,6});
    require(sim.storage.row_copies.size() == 1 && sim.storage.slot_copies.size() == 2,"hybrid fork and copy-on-write");
    sim.commit(b,3);
    sim.begin(a,{7}); sim.commit(a,1);
    require(sim.storage.row_copies.size() == 1,"exclusive tail copied");
    sim.check();
    sim.begin(a,{8,9});
    sim.close(a);   // a pending round is discarded by close
    sim.check();
    sim.retire(cache.epoch());
    sim.check();
}

// A scripted workload whose storage throws at one call index: the failing call publishes nothing, the retry
// succeeds, and the final ids equal the failure-free run's (free lists were restored exactly).
void failures() {
    auto c = attention(12,4,4,32,3);
    c.components = {component(RecurrentCommit::column_checkpoints,16),component(RecurrentCommit::recompute,8)};
    const auto run = [&](long fail_at, int& failures_seen) {
        Sim sim(c);
        sim.storage.fail_at = fail_at;
        const auto step = [&](auto action) {
            const auto before = sim.snapshot();
            try { action(); }
            catch(const std::runtime_error& e) {
                require(std::string_view(e.what()) == "injected storage failure","unexpected failure");
                require(Sim::unchanged(before,sim.snapshot()) && same_stats(before.stats,sim.snapshot().stats),
                        "a failed storage hook published state");
                ++failures_seen;
                action();
            }
            sim.check();
        };
        Sim::Seq* a = nullptr; Sim::Seq* b = nullptr; Sim::Seq* d = nullptr;
        step([&] { a = &sim.open(); });
        step([&] { sim.begin(*a,{1,2,3}); });
        step([&] { sim.commit(*a,3); });
        step([&] { b = &sim.fork(*a); });             // two copy_slot calls
        step([&] { sim.begin(*b,{4,5}); });           // copy_rows
        step([&] { sim.commit(*b,2); });
        step([&] { d = &sim.fork(*b); });             // two copy_slot calls
        step([&] { sim.begin(*d,{5,6}); });           // copy_rows
        step([&] { sim.commit(*d,1); });
        step([&] { sim.begin(*a,{7}); });             // exclusive tail: no hook
        step([&] { sim.commit(*a,1); });
        std::vector<std::vector<std::uint64_t>> ids;
        for(const auto& s:sim.seqs) {
            std::vector<std::uint64_t> row(sim.cache.block_table(s.id).begin(),sim.cache.block_table(s.id).end());
            for(std::uint32_t k = 0; k < 2; ++k) row.push_back(sim.cache.live_slot(s.id,k));
            ids.push_back(row);
        }
        // Six hooks succeed in every run; a failed call (and any earlier hook of its operation) runs again.
        require(sim.storage.calls >= 6+(fail_at >= 0 && fail_at < 6 ? 1 : 0),"hook call count");
        return ids;
    };
    int none = 0;
    const auto reference = run(-1,none);
    require(none == 0,"failure-free run failed");
    for(long k = 0; k <= 6; ++k) {
        int seen = 0;
        const auto ids = run(k,seen);
        require(seen == (k < 6 ? 1 : 0),"each hook call index fails exactly once");
        require(ids == reference,"a failed hook left the free lists in another order");
    }
    // A copy-on-write that must evict may lose that cached page, but never publishes it or reuses it falsely.
    Sim sim(attention(3,2,3,6,2,true));
    auto& cache = sim.cache;
    auto& x = sim.open();
    sim.begin(x,{1,2}); sim.commit(x,2);
    sim.close(x);
    sim.retire(cache.epoch());
    auto& p = sim.open();
    sim.begin(p,{5}); sim.commit(p,1);
    auto& q = sim.fork(p);
    auto& r = sim.open();
    sim.begin(r,{9}); sim.commit(r,1);
    require(cache.stats().free_pages == 0 && cache.stats().cached_pages == 1,"setup leaves only a cached page");
    const auto before = sim.snapshot();
    sim.storage.fail_at = sim.storage.calls;
    rejects([&] { sim.begin(q,{6}); },"injected storage failure");
    sim.storage.fail_at = -1;
    const auto after = sim.snapshot();
    require(Sim::unchanged(before,after) && after.stats.free_pages == 1 && after.stats.indexed_pages == 0,
            "a failed copy-on-write returned its evicted page to the cache");
    require(cache.reusable(Tokens{1,2,3},2) == 0,"an evicted page stayed reusable");
    sim.begin(q,{6}); sim.commit(q,1);
    sim.check();
    // Storage hooks cannot mutate the cache: throwing calls are rejected, noexcept calls ignored.
    struct Reentrant final : StateStorage {
        StateCache* cache = nullptr;
        SequenceId other;
        bool closed = true;
        std::uint32_t evicted = 1;
        void copy_rows(PageId, PageId, std::uint32_t) override {
            closed = cache->close(other);
            evicted = cache->evict(1);
            cache->retire(cache->epoch());
            cache->open();
        }
        void copy_slot(std::uint32_t, SlotId, SlotId) override {}
    } hooks;
    Resources resources(std::size_t(1) << 20,0);
    StateCache nested(attention(4,2,3,8,2),hooks,resources);
    hooks.cache = &nested;
    const auto parent = nested.open();
    nested.commit(parent,nested.begin(parent,1).id,std::array<Token,1>{1});
    const auto child = nested.fork(parent);
    hooks.other = parent;
    const auto epoch = nested.epoch();
    rejects_as<std::logic_error>([&] { nested.begin(child,1); },"reentrant");
    require(!hooks.closed && hooks.evicted == 0 && nested.epoch() == epoch && nested.stats().sequences == 2,
            "a hook mutated the cache");
    require(nested.staged(child) == 0 && nested.stats().live_pages == 1,"a rejected reentrant begin published");
}

// Randomized walk against the reference model, with occasional storage failures. Checks every invariant after
// every step, including that no id is written before the epoch it was released in has retired. With the default
// hash, a sequence whose chain began after the last eviction must have every committed full page reusable, however
// many duplicates of its pages other sequences computed. A roomy walk is never short of pages.
void walk(StateCacheConfig config, int steps, std::uint32_t seed, bool roomy = false) {
    Sim sim(config);
    auto& cache = sim.cache;
    std::mt19937 rng(seed);
    const auto R = config.max_round_rows;
    const bool recurrent = !config.components.empty();
    std::vector<Tokens> docs(3);
    for(auto& doc:docs) for(std::uint32_t i = 0; i < config.max_tokens; ++i) doc.push_back(Token(rng()%3));
    const auto pick = [&](bool quiescent) -> Sim::Seq* {
        std::vector<Sim::Seq*> eligible;
        for(auto& s:sim.seqs) if(!quiescent || !s.round) eligible.push_back(&s);
        return eligible.empty() ? nullptr : eligible[rng()%eligible.size()];
    };
    const auto pending = [&]() -> Sim::Seq* {
        std::vector<Sim::Seq*> eligible;
        for(auto& s:sim.seqs) if(s.round) eligible.push_back(&s);
        return eligible.empty() ? nullptr : eligible[rng()%eligible.size()];
    };
    const auto follows = [&](const Sim::Seq& s) {
        return s.doc >= 0 && std::equal(s.tokens.begin(),s.tokens.end(),docs[std::size_t(s.doc)].begin());
    };
    int failures = 0, exhausted = 0, starved = 0, reused = 0, commits = 0, complete = 0;
    // Snapshots and windows end reuse early, so the completeness oracle covers attention-only caches.
    const bool oracle = config.prefix_cache && !config.hash && config.components.empty() && config.windows.empty();
    for(int step = 0; step < steps; ++step) {
        auto before = sim.snapshot();
        const bool inject = rng()%6 == 0;
        if(inject) sim.storage.fail_at = sim.storage.calls;
        try {
            switch(rng()%20) {
            case 0: sim.open(); break;
            case 1: case 2: case 3: if(auto* s = pick(false)) sim.close(*s); break;
            case 4: case 18: if(auto* s = pick(true)) sim.fork(*s); break;
            case 5: case 6: {
                auto& s = sim.open();
                before = sim.snapshot();   // a failed attach (snapshot restore) leaves the opened sequence empty
                const auto d = rng()%docs.size();
                Tokens prompt(docs[d].begin(),docs[d].begin()+1+std::ptrdiff_t(rng()%config.max_tokens));
                if(rng()%3 == 0) prompt.back() = Token(3+rng()%2);
                const auto cap = rng()%4 ? std::uint32_t(prompt.size()-1) : std::uint32_t(rng()%(prompt.size()+2));
                reused += sim.attach(s,prompt,cap) > 0;
                s.doc = int(d);
                if(!follows(s)) s.doc = -1;
                break;
            }
            case 7: case 8: case 9: case 10: {
                auto* s = pick(true);
                if(!s) break;
                const auto room = config.max_tokens-std::uint32_t(s->tokens.size());
                const auto rows = std::min<std::uint32_t>(room,rng()%8 ? std::uint32_t(1+rng()%R) : 0u);
                Tokens tokens;
                const bool doc = follows(*s) && rng()%4;
                for(std::uint32_t i = 0; i < rows; ++i)
                    tokens.push_back(doc ? docs[std::size_t(s->doc)][s->tokens.size()+i] : Token(rng()%3));
                const bool try_only = rng()%2;
                if(!sim.begin(*s,tokens,try_only)) {
                    ++exhausted; ++starved;
                    require(Sim::unchanged(before,sim.snapshot()) && same_stats(before.stats,sim.snapshot().stats),
                            "try_begin changed state on exhaustion");
                }
                break;
            }
            case 11: case 12: case 13: case 14:
                if(auto* s = pending()) {
                    const auto n = rng()%3 ? rng()%(s->round->rows+1) : s->round->rows;
                    const bool doc = follows(*s);
                    sim.commit(*s,n);
                    if(doc && !follows(*s)) s->doc = -1;
                    ++commits;
                }
                break;
            case 15:
                if(auto* s = pending(); s && rng()%2) sim.discard(*s);
                else if(auto* q = pick(true); q && !q->tokens.empty()) {
                    const auto keep = std::uint32_t(rng()%(q->tokens.size()+1));
                    // Dropped window pages are a window table's leading no_page entries.
                    bool dropped = false;
                    for(std::uint32_t w = 0; w < sim.W; ++w) {
                        const auto table = cache.window_table(q->id,w);
                        std::size_t lead = 0;
                        while(lead < table.size() && table[lead] == no_page) ++lead;
                        dropped = dropped || sim.window_first(w,keep)/sim.pt < lead;
                    }
                    if(recurrent && keep != q->tokens.size()) {
                        rejects_as<std::logic_error>([&] { cache.rewind(q->id,keep); },"cannot rewind");
                        require(Sim::unchanged(before,sim.snapshot()),"rejected rewind changed state");
                    } else if(dropped && keep != q->tokens.size()) {
                        rejects_as<std::logic_error>([&] { cache.rewind(q->id,keep); },"dropped window pages");
                        require(Sim::unchanged(before,sim.snapshot()),"rejected rewind changed state");
                    } else sim.rewind(*q,keep);
                }
                break;
            case 16:
                if(rng()%3) sim.retire(cache.epoch());
                else { const auto e = sim.advance_epoch(); if(rng()%2) sim.retire(e-std::min<std::uint64_t>(e,rng()%3)); }
                break;
            case 17: if(!roomy) cache.evict(std::uint32_t(rng()%3)); break;
            default: {
                // Finished sequences leave, as completed requests do.
                Sim::Seq* full = nullptr;
                for(auto& s:sim.seqs) if(s.tokens.size()+R > config.max_tokens) full = &s;
                if(full) sim.close(*full);
                break;
            }
            }
        } catch(const CapacityExhausted& e) {
            ++exhausted;
            starved += std::string_view(e.what()).find("sequences") == std::string_view::npos;
            require(Sim::unchanged(before,sim.snapshot()) && same_stats(before.stats,sim.snapshot().stats),
                    "capacity exhaustion changed state");
        } catch(const std::runtime_error& e) {
            if(!inject || std::string_view(e.what()) != "injected storage failure") throw;
            ++failures;
            require(Sim::unchanged(before,sim.snapshot()),"a failed storage hook published state");
        }
        sim.storage.fail_at = -1;
        sim.check();
        // Engines retire after each step's readiness; skipping half the steps keeps retiring ids outstanding.
        if(rng()%2) { sim.retire(cache.epoch()); sim.check(); }
        if(oracle) for(const auto& s:sim.seqs) if(s.born == cache.stats().evictions) { sim.expect_complete(s); ++complete; }
    }
    require(failures > 0 && (roomy || starved > 0) && exhausted > starved && commits > steps/10,
            "the walk did not reach storage failures, every kind of exhaustion and commits");
    require((recurrent && !config.prefix_cache) || !config.prefix_cache || reused > 0,"the walk never reused a prefix");
    require(!oracle || complete > steps,"the completeness oracle rarely applied");
    while(!sim.seqs.empty()) sim.close(sim.seqs.front());
    sim.retire(cache.epoch());
    sim.check();
    const auto st = cache.stats();
    require(st.live_pages == 0 && st.retiring_pages == 0 && st.live_slots == st.snapshots && st.retiring_slots == 0,"walk teardown leaked");
    for(std::uint32_t w = 0; w < config.windows.size(); ++w)
        require(cache.window_stats(w).live_pages == 0 && cache.window_stats(w).retiring_pages == 0,"walk teardown leaked window pages");
}
void randomized() {
    walk(attention(14,3,8,24,5,true),5000,715982);
    walk(attention(96,3,8,24,5,true),3000,2207,true);
    auto collide = attention(14,3,8,24,5,true);
    collide.hash = page_only_hash;
    walk(collide,1500,1468);
    collide.hash = constant_hash;
    walk(collide,1500,9182);
    auto mixed = attention(14,2,6,20,3);
    mixed.components = {component(RecurrentCommit::column_checkpoints,30),component(RecurrentCommit::recompute,14),
                        component(RecurrentCommit::in_place,8)};
    walk(mixed,3000,4242);
    // Sliding-window groups, with and without the prefix cache.
    auto windowed = attention(20,3,6,24,4,true);
    windowed.windows = {{12,7},{9,4}};
    walk(windowed,4000,5150);
    windowed.prefix_cache = false; windowed.identity = 0;
    walk(windowed,2000,6161);
    // Hybrid prefix reuse through snapshots, beside a window group.
    auto hybrid = attention(16,2,5,20,3,true);
    hybrid.components = {component(RecurrentCommit::column_checkpoints,16),component(RecurrentCommit::recompute,12),
                         component(RecurrentCommit::in_place,9)};
    hybrid.components[0].snapshots = 3; hybrid.components[1].snapshots = 2; hybrid.components[2].snapshots = 2;
    hybrid.windows = {{12,5}};
    walk(hybrid,4000,7272);
}

void capacity() {
    auto c = attention(4,2,2,8,4);
    c.components = {component(RecurrentCommit::column_checkpoints,6)};
    Sim sim(c);
    auto& cache = sim.cache;
    auto& a = sim.open();
    auto& b = sim.open();
    rejects_as<CapacityExhausted>([&] { cache.open(); },"sequences exhausted");
    sim.begin(a,{1,2,3,4});   // two pages, every free slot
    auto before = sim.snapshot();
    require(!sim.begin(b,{1},true),"a round started without scratch slots");
    rejects_as<CapacityExhausted>([&] { cache.begin(b.id,1); },"slots exhausted");
    require(Sim::unchanged(before,sim.snapshot()) && same_stats(before.stats,sim.snapshot().stats),"slot exhaustion changed state");
    sim.commit(a,4);
    require(!sim.begin(b,{1},true),"retiring slots were allocated");
    sim.retire(cache.epoch());
    sim.begin(b,{1,2}); sim.commit(b,2);
    before = sim.snapshot();
    require(cache.pages_needed(b.id,3) == 2 && !sim.begin(b,{3,4,5},true),"a round started without pages");
    rejects_as<CapacityExhausted>([&] { cache.begin(b.id,3); },"pages exhausted");
    static_assert(std::is_base_of_v<std::runtime_error,CapacityExhausted>,"capacity is a runtime error");
    require(Sim::unchanged(before,sim.snapshot()) && same_stats(before.stats,sim.snapshot().stats),"page exhaustion changed state");
    sim.check();
    // A fork with no free slot fails without change, then succeeds after retirement.
    auto f = attention(0,1,3,8,1);
    f.components = {component(RecurrentCommit::in_place,3)};
    Sim forks(f);
    auto& p = forks.open();
    auto& q = forks.fork(p);
    auto& extra = forks.fork(q);
    forks.close(extra);
    forks.close(q);
    before = forks.snapshot();
    rejects_as<CapacityExhausted>([&] { forks.cache.fork(p.id); },"slots exhausted");
    rejects_as<CapacityExhausted>([&] { forks.cache.open(); },"slots exhausted");
    require(Sim::unchanged(before,forks.snapshot()) && same_stats(before.stats,forks.snapshot().stats),"failed fork changed state");
    forks.retire(forks.cache.epoch());
    forks.fork(p);
    forks.check();
}

void layout() {
    Sim sim(attention(8,4,3,32,4));
    auto& cache = sim.cache;
    auto& a = sim.open();
    std::set<std::uint64_t> seen{cache.layout_generation(a.id)};
    const auto changed = [&](const Sim::Seq& s) { return seen.insert(cache.layout_generation(s.id)).second; };
    sim.begin(a,{1,2});
    require(changed(a),"a new page kept the layout");
    sim.commit(a,2);
    require(!changed(a),"a commit keeping every page changed the layout");
    sim.begin(a,{3});
    require(!changed(a),"a round inside the tail page changed the layout");
    sim.commit(a,1);
    sim.begin(a,{4,5,6});
    require(changed(a),"crossing a page boundary kept the layout");
    sim.commit(a,1);
    require(changed(a),"dropping a staged page kept the layout");
    sim.begin(a,{5});
    require(changed(a),"a fresh tail page kept the layout");
    sim.discard(a);
    require(changed(a),"discard dropping a page kept the layout");
    auto& b = sim.fork(a);
    require(changed(b),"a fork reused a layout generation");
    sim.begin(b,{6}); sim.commit(b,1);
    require(changed(b),"an appended page kept the layout");
    sim.rewind(b,4);
    require(changed(b),"rewind dropping a page kept the layout");
    sim.rewind(b,2);
    require(!changed(b),"rewind inside a page changed the layout");
    auto& c = sim.fork(b);
    sim.begin(c,{7});
    require(changed(c),"copy-on-write kept the layout");
    sim.commit(c,1);
    sim.check();
    // Recurrent: id swaps change the layout; in_place never does.
    auto r = attention(0,1,1,16,2);
    r.components = {component(RecurrentCommit::column_checkpoints,3)};
    Sim swap(r);
    auto& x = swap.open();
    auto g = swap.cache.layout_generation(x.id);
    swap.begin(x,{1,2});
    require(swap.cache.layout_generation(x.id) != g,"new scratch slots kept the layout");
    g = swap.cache.layout_generation(x.id);
    swap.commit(x,2);
    require(swap.cache.layout_generation(x.id) != g,"a live slot swap kept the layout");
    r.components = {component(RecurrentCommit::in_place,1)};
    Sim fixed(r);
    auto& y = fixed.open();
    g = fixed.cache.layout_generation(y.id);
    fixed.begin(y,{1,2}); fixed.commit(y,2);
    require(fixed.cache.layout_generation(y.id) == g,"in_place changed the layout");
    // Attach changes the layout of the sequence it fills.
    Sim reuse(attention(4,2,2,8,2,true));
    auto& s = reuse.open();
    reuse.begin(s,{1,2}); reuse.commit(s,2);
    auto& t = reuse.open();
    g = reuse.cache.layout_generation(t.id);
    require(reuse.attach(t,Tokens{1,2,3},2) == 2 && reuse.cache.layout_generation(t.id) != g,"attach kept the layout");
    // Recycled records: a sequence opened or forked into a closed sequence's record gets a fresh id generation and
    // a layout generation never seen before, even when it receives the same live slot and no pages.
    auto q = attention(8,2,2,8,2);
    q.components = {component(RecurrentCommit::in_place,4)};
    Sim rec(q);
    auto& keep = rec.open();
    auto& gone = rec.open();
    rec.begin(gone,{1}); rec.commit(gone,1);
    std::set<std::uint64_t> layouts{rec.cache.layout_generation(keep.id),rec.cache.layout_generation(gone.id)};
    const auto first = gone.id;
    const auto slot = rec.cache.live_slot(gone.id,0);
    rec.close(gone);
    rec.retire(rec.cache.epoch());
    auto& reopened = rec.open();
    require(reopened.id.index == first.index && !(reopened.id == first),"a reopened record kept its generation");
    require(rec.cache.live_slot(reopened.id,0) == slot,"the reopened record reuses the released live slot");
    require(rec.cache.layout_generation(reopened.id) != 0 && layouts.insert(rec.cache.layout_generation(reopened.id)).second,
            "a reopened record repeated a layout generation");
    const auto second = reopened.id;
    rec.close(reopened);
    rec.retire(rec.cache.epoch());
    auto& child = rec.fork(keep);
    require(child.id.index == first.index && !(child.id == first) && !(child.id == second),"a forked record kept its generation");
    require(rec.cache.layout_generation(child.id) != 0 && layouts.insert(rec.cache.layout_generation(child.id)).second,
            "a forked record repeated a layout generation");
    for(const auto stale:{first,second}) {
        require(!rec.cache.close(stale),"a stale id closed the forked child");
        rejects_as<std::logic_error>([&] { rec.cache.committed(stale); },"stale state sequence");
        rejects_as<std::logic_error>([&] { rec.cache.begin(stale,1); },"stale state sequence");
    }
    rec.begin(child,{2}); rec.commit(child,1);
    rec.check();
}

void metadata() {
    Resources resources(std::size_t(1) << 24,0);
    auto c = attention(64,16,8,1024,5,true);
    std::size_t bytes = 0;
    {
        Fake storage(c);
        StateCache cache(c,storage,resources);
        bytes = cache.metadata_bytes();
        require(resources.usage(Memory::host).live == bytes,"metadata reservation differs from metadata_bytes");
        // Every preallocated array, by element: per page, index entry, per sequence and per round row.
        const std::size_t P = 64, S = 8, T = 64, R = 5, index = 128;
        const auto expected = P*(4+1+1+8*4+4*4+16*sizeof(Token))+index*4+4+S*(8*3+1*3+4*4+T*4+R*sizeof(RowSlot));
        require(bytes == expected,"metadata bytes differ from the documented arrays");
        StateCache second(c,storage,resources);
        require(resources.usage(Memory::host).live == 2*bytes,"each cache reserves its own metadata");
    }
    require(resources.usage(Memory::host).live == 0 && resources.usage(Memory::host).peak == 2*bytes,"metadata was not released");
    Resources small(bytes-1,0);
    Fake storage(c);
    rejects_as<std::runtime_error>([&] { StateCache cache(c,storage,small); },"exceeds ceiling");
    require(small.usage(Memory::host).live == 0,"a rejected cache kept its reservation");
    // Components and slots are charged too.
    auto h = attention(4,2,2,8,2);
    h.components = {component(RecurrentCommit::column_checkpoints,6)};
    StateCache hybrid(h,storage,resources);
    const std::size_t slots = 6, S = 2, C = 1, R = 2, T = 4, P = 4;
    const auto expected = sizeof(StateComponent)*C+P*(4+1+1+8*4+4*4+2*sizeof(Token))+C*(1+4*4)+4+slots*(4+4+8)+
                          S*(8*3+1*3+4*4+T*4+C*4*2+C*R*4+R*sizeof(RowSlot));
    require(hybrid.metadata_bytes() == expected,"hybrid metadata bytes differ from the documented arrays");
    // Spare capacity moved in with the components vector is not retained uncharged.
    auto spare = h;
    spare.components.reserve(1 << 16);
    const auto live = resources.usage(Memory::host).live;
    StateCache moved(std::move(spare),storage,resources);
    require(moved.metadata_bytes() == expected && resources.usage(Memory::host).live == live+expected,
            "moved-in component capacity changed the reservation");
    require(moved.config().components.capacity() < (std::size_t(1) << 16) && moved.config().components.size() == 1,
            "moved-in component capacity was retained");
}

void hot_path() {
    auto c = attention(32,4,4,64,4,true);
    Fake storage(c);
    Resources resources(std::size_t(1) << 24,0);
    StateCache cache(c,storage,resources);
    const Tokens prompt = range(1,12), inputs = range(20,4);
    const std::span<const Token> in(inputs);
    auto h = attention(16,2,3,32,3);
    h.components = {component(RecurrentCommit::column_checkpoints,12),component(RecurrentCommit::recompute,6),
                    component(RecurrentCommit::in_place,3)};
    Fake hooks(h);
    StateCache mixed(h,hooks,resources);
    storage.row_copies.clear(); hooks.row_copies.clear(); hooks.slot_copies.clear();
    const auto before = allocations.load();
    for(int i = 0; i < 200; ++i) {
        storage.row_copies.clear(); hooks.row_copies.clear(); hooks.slot_copies.clear();
        const auto a = cache.open();
        cache.attach(a,prompt,std::uint32_t(prompt.size()-1));
        auto r = cache.try_begin(a,4);
        require(r.has_value(),"hot path exhausted");
        cache.commit(a,r->id,in.first(3));
        const auto b = cache.fork(a);
        const auto d = cache.begin(b,2);
        cache.discard(b,d.id);
        const auto e = cache.begin(b,2);   // copy-on-write of the shared tail
        cache.commit(b,e.id,in.first(2));
        cache.rewind(b,cache.committed(b)-1);
        static_cast<void>(cache.reusable(prompt,8)); static_cast<void>(cache.stats());
        static_cast<void>(cache.block_table(b)); static_cast<void>(cache.pages_needed(b,2));
        cache.close(a); cache.close(b);
        cache.retire(cache.epoch());
        cache.evict(1);
        const auto x = mixed.open();
        const auto y = mixed.begin(x,3);
        mixed.commit(x,y.id,in.first(2));
        const auto z = mixed.fork(x);
        const auto w = mixed.begin(z,1);
        mixed.commit(z,w.id,in.first(1));
        mixed.close(x); mixed.close(z);
        static_cast<void>(mixed.advance_epoch());
        mixed.retire(mixed.epoch());
    }
    require(allocations.load() == before,"the state cache allocated after construction");
}

// Hybrid prefix reuse: the recurrent state after a full page is snapshotted at commit when every component has it,
// attach restores it with the pages, a FIFO quota bounds snapshots per component, a failing snapshot copy leaves the
// round pending, and evicted pages take their snapshots along.
void snapshots() {
    auto c = attention(16,4,3,64,6,true);
    c.components = {component(RecurrentCommit::column_checkpoints,3+6+4),component(RecurrentCommit::recompute,3+1+4),
                    component(RecurrentCommit::in_place,3+4)};
    for(auto& k:c.components) k.snapshots = 4;
    Sim sim(c);
    auto& cache = sim.cache;
    auto& a = sim.open();
    sim.begin(a,range(1,6)); sim.commit(a,6);   // page 0 fills inside the round: only column checkpoints hold that state
    require(cache.stats().snapshots == 0,"a boundary without every component's state took a snapshot");
    sim.begin(a,range(7,2)); sim.commit(a,2);   // the round ends on page 1's boundary: every component has the state
    require(cache.stats().snapshots == 3,"each component snapshots page 1");
    sim.check();
    Tokens prompt = a.tokens;
    prompt.push_back(40); prompt.push_back(41);
    require(cache.reusable(prompt,std::uint32_t(prompt.size()-1)) == 8,"reuse ends at the page with snapshots");
    sim.retire(cache.epoch());
    auto& b = sim.open();
    require(sim.attach(b,prompt,std::uint32_t(prompt.size()-1)) == 8,"attach reuses two pages");
    require(cache.stats().prefix_hits == 2,"reused pages counted");
    sim.check();   // b's live slots hold the state after the 8 reused tokens
    sim.begin(b,{40,41}); sim.commit(b,2);
    sim.check();
    Tokens early = range(1,4);
    early.push_back(99); early.push_back(98);
    require(cache.reusable(early,5) == 0,"page 0 has no snapshot to end at");
    sim.close(a); sim.close(b);
    sim.retire(cache.epoch());
    cache.evict(16);
    require(cache.stats().snapshots == 3 && cache.reusable(prompt,9) == 0,"evicted pages took their snapshots; the slots stay pooled");
    sim.retire(cache.epoch());
    sim.check();

    // One round fills three pages; a quota of two keeps the two newest.
    auto q = attention(16,2,2,32,6,true);
    q.components = {component(RecurrentCommit::column_checkpoints,2+6+2)};
    q.components[0].snapshots = 2;
    Sim fifo(q);
    auto& d = fifo.open();
    fifo.begin(d,range(1,6)); fifo.commit(d,6);
    require(fifo.cache.stats().snapshots == 2,"the oldest snapshot is dropped at the quota");
    Tokens p6 = range(1,6), p4 = range(1,4), p2 = range(1,2);
    p6.push_back(50); p4.push_back(60); p4.push_back(61); p2.push_back(70); p2.push_back(71);
    require(fifo.cache.reusable(p6,6) == 6 && fifo.cache.reusable(p4,5) == 4 && fifo.cache.reusable(p2,3) == 0,
            "reuse ends at the newest snapshot each prompt reaches");
    fifo.check();
    // A failing snapshot copy publishes nothing and leaves the round pending; the retry commits.
    fifo.retire(fifo.cache.epoch());
    auto& e = fifo.open();
    fifo.begin(e,range(100,4));
    const auto before = fifo.cache.stats();
    fifo.storage.fail_at = fifo.storage.calls;
    rejects([&] { fifo.commit(e,4); },"injected storage failure");
    require(fifo.cache.staged(e.id) == 4 && fifo.cache.committed(e.id) == 0 && same_stats(before,fifo.cache.stats()),
            "a failed snapshot copy changed the cache");
    fifo.storage.fail_at = -1;
    fifo.commit(e,4);
    fifo.check();
    require(fifo.cache.stats().snapshots == 2,"the retry snapshots within the quota");
}

// Sliding-window groups: pages wholly older than the window leave the window table while full pages stay linked for
// reuse; attach needs the pages its window reads; shared window tails are copied before writing; rewind cannot reach
// dropped pages; window capacity is checked before anything changes, and nothing allocates after construction.
void windows() {
    auto c = attention(32,4,4,64,4,true);
    c.windows = {{8,6}};
    Fake storage(c);
    Resources resources(std::size_t(1) << 24,0);
    StateCache cache(c,storage,resources);
    const auto write = [&](const Round& r, const Tokens& all) {
        const auto values = digests(all);
        require(r.window_kv.size() == r.rows,"one window row per staged row");
        for(std::uint32_t i = 0; i < r.rows; ++i) {
            const auto row = r.first_row+i;
            storage.rows[std::size_t(r.kv[i].page)*4+r.kv[i].offset] = values[row];
            storage.windows[0][std::size_t(r.window_kv[i].page)*4+r.window_kv[i].offset] = values[row];
        }
    };
    // The rows the next row reads are present with their values; pages wholly before them are dropped.
    const auto expect_window = [&](SequenceId id, const Tokens& tokens, bool dropped = true) {
        const auto table = cache.window_table(id,0);
        const auto n = std::uint32_t(tokens.size());
        require(table.size() == (n+3)/4,"window table size differs from the block table");
        const auto values = digests(tokens);
        const auto first = n+1 > 6 ? n+1-6 : 0u;
        for(std::uint32_t i = 0; dropped && i < first/4; ++i) require(table[i] == no_page,"a page older than the window stayed");
        for(auto r = first; r < n; ++r)
            require(table[r/4] != no_page && storage.windows[0][std::size_t(table[r/4])*4+r%4] == values[r],"window rows differ");
    };
    const auto step = [&](SequenceId id, Tokens& tokens, const Tokens& more) {
        const auto r = cache.begin(id,std::uint32_t(more.size()));
        auto all = tokens;
        all.insert(all.end(),more.begin(),more.end());
        write(r,all);
        cache.commit(id,r.id,more);
        tokens = all;
        cache.retire(cache.epoch());
        expect_window(id,tokens);
    };
    const auto a = cache.open();
    Tokens ta;
    for(Token t = 1; t <= 24; t += 4) step(a,ta,range(t,4));
    require(cache.window_stats(0).live_pages == 2 && cache.window_stats(0).indexed_pages == 6 && cache.window_stats(0).cached_pages == 4,
            "the window holds two pages; every full page stays linked");
    cache.close(a);
    cache.retire(cache.epoch());
    // A new sequence reuses the 24 tokens with the two window pages row 24 reads.
    const auto b = cache.open();
    Tokens prompt = ta;
    prompt.push_back(90);
    require(cache.reusable(prompt,24) == 24 && cache.attach(b,prompt,24) == 24,"reuse with the window's pages");
    expect_window(b,ta);
    Tokens tb = ta;
    step(b,tb,{90,91});
    // A fork shares b's partial window tail: the first write copies it.
    const auto copies = storage.window_copies.size();
    const auto f = cache.fork(b);
    Tokens tf = tb;
    storage.fail_at = storage.calls;   // the copy fails once: nothing changes
    const auto stats = cache.window_stats(0);
    rejects([&] { cache.begin(f,1); },"injected storage failure");
    require(same_stats(stats,cache.window_stats(0)) && cache.staged(f) == 0,"a failed window copy changed the cache");
    storage.fail_at = -1;
    step(f,tf,{92});
    require(storage.window_copies.size() == copies+1,"a shared window tail is copied before writing");
    expect_window(b,tb);
    rejects_as<std::logic_error>([&] { cache.rewind(f,20); },"dropped window pages");
    cache.rewind(f,26);
    tf.resize(26);
    expect_window(f,tf);
    // Other prompts take the window pool: once a page row 24 reads is evicted, reuse stops short of it.
    const auto g = cache.open();
    Tokens tg;
    for(Token t = 500; cache.reusable(prompt,24) == 24 && t < 600; t += 4) step(g,tg,range(t,4));
    const auto reuse = cache.reusable(prompt,24);
    require(reuse < 24 && cache.stats().prefix_misses == 0,"evicted window pages cap reuse");
    const auto h = cache.open();
    require(cache.attach(h,prompt,24) == reuse,"attach agrees with reusable");
    expect_window(h,Tokens(prompt.begin(),prompt.begin()+reuse));
    for(const auto id:{b,f,g,h}) cache.close(id);
    cache.retire(cache.epoch());
    require(cache.window_stats(0).live_pages == 0 && cache.stats().live_pages == 0,"every page returned");

    // Window capacity is checked before anything changes.
    auto small = attention(32,4,2,64,4);
    small.windows = {{4,6}};
    Fake tight_storage(small);
    StateCache tight(small,tight_storage,resources);
    const auto x = tight.open(), y = tight.open();
    for(const auto id:{x,y}) { const auto r = tight.begin(id,4); tight.commit(id,r.id,range(1,4)); const auto r2 = tight.begin(id,4); tight.commit(id,r2.id,range(5,4)); }
    const auto full = tight.window_stats(0);
    require(full.free_pages == 0 && !tight.try_begin(x,4) && same_stats(full,tight.window_stats(0)),"window pages exhausted");
    rejects_as<CapacityExhausted>([&] { tight.begin(x,4); },"window pages exhausted");

    // Window groups and snapshots together allocate nothing after construction.
    auto hybrid = attention(32,4,2,64,4,true);
    hybrid.components = {component(RecurrentCommit::recompute,2+1+4)};
    hybrid.components[0].snapshots = 4;
    hybrid.windows = {{12,6}};
    Fake hooks(hybrid);
    StateCache mixed(hybrid,hooks,resources);
    const Tokens doc = range(1,16), inputs = range(1,4);
    const auto before = allocations.load();
    for(int i = 0; i < 100; ++i) {
        hooks.slot_copies.clear(); hooks.window_copies.clear(); hooks.row_copies.clear();
        const auto s = mixed.open();
        mixed.attach(s,doc,15);
        while(mixed.committed(s) < 16) {
            const auto start = mixed.committed(s);
            const auto r = mixed.begin(s,4-start%4);
            mixed.commit(s,r.id,std::span<const Token>(doc).subspan(start,4-start%4));
            mixed.retire(mixed.epoch());   // one step at a time, as a single-stream engine retires
        }
        const auto t = mixed.fork(s);
        const auto r = mixed.begin(t,2);
        mixed.commit(t,r.id,std::span<const Token>(inputs).first(2));
        static_cast<void>(mixed.window_table(t,0)); static_cast<void>(mixed.window_stats(0)); static_cast<void>(mixed.reusable(doc,15));
        mixed.close(s); mixed.close(t);
        mixed.retire(mixed.epoch());
        if(i % 10 == 9) mixed.evict(32);
    }
    require(allocations.load() == before,"window groups or snapshots allocated after construction");
}

void checkpoints() {
    const std::array<std::size_t,3> accepted{0,2,5};   // physical nodes; rejected siblings are skipped
    require(!checkpoint_node(64,accepted,64),"the round start has no verify node");
    require(*checkpoint_node(64,accepted,65) == 0 && *checkpoint_node(64,accepted,66) == 2 && *checkpoint_node(64,accepted,67) == 5,
            "checkpoint selects the accepted ordinal's physical node");
    rejects_as<std::out_of_range>([&] { checkpoint_node(64,accepted,63); },"checkpoint");
    rejects_as<std::out_of_range>([&] { checkpoint_node(64,accepted,68); },"checkpoint");
    require(!checkpoint_node(5,std::span<const std::size_t>{},5),"empty accepted prefix");
    rejects_as<std::out_of_range>([&] { checkpoint_node(5,std::span<const std::size_t>{},6); },"checkpoint");
    rejects_as<std::out_of_range>([&] { checkpoint_node(std::numeric_limits<std::uint64_t>::max(),accepted,0); },"checkpoint");
}
}

// A chunked prefill computes no state inside its round, only the state after the last row. As an every_row round of a
// column_checkpoints component, its commit snapshots page boundaries from scratch slots the engine never wrote, and a
// later prompt sharing only the first page attaches that state (on this fixture: 4 tokens reused, the live slot holds
// the never-written value). A final_row round takes one scratch slot, snapshots only its last row and reuses only that.
void round_states() {
    auto c = attention(16,4,4,32,8,true);
    c.components = {component(RecurrentCommit::column_checkpoints,4+8+4)};
    c.components[0].snapshots = 4;
    Fake storage(c);
    Resources resources{std::size_t{1} << 20,0};
    StateCache cache(c,storage,resources);
    const Tokens prompt = range(1,8);
    const auto states = digests(prompt);
    const Value poison = 0xdeadull;
    const auto a = cache.open();
    const auto free_before = cache.stats().free_slots;
    const auto r = cache.begin(a,8,RoundStates::final_row);
    require(r.states == RoundStates::final_row && r.scratch[0] != no_slot,"a final_row round names its state slot");
    for(std::uint32_t i = 1; i < 8; ++i) require(r.scratch[i] == no_slot,"a final_row round takes one scratch slot");
    require(cache.stats().free_slots == free_before-1,"one slot taken for the round");
    for(auto& value:storage.slots[0]) value = poison;   // nothing but the engine's write below holds a state
    storage.slots[0][r.scratch[0]] = states[7];          // the chunked prefill: the state after the last row
    cache.commit(a,r.id,prompt);
    require(storage.slots[0][cache.live_slot(a,0)] == states[7],"the round's state becomes live");
    // A prompt sharing the first page only: no snapshot exists at that boundary, so nothing is reused.
    const auto b = cache.open();
    Tokens other(prompt.begin(),prompt.begin()+5);
    other.push_back(99); other.push_back(100);
    require(cache.attach(b,other,std::uint32_t(other.size()-1)) == 0,"no reuse at a boundary without a written state");
    // The whole prompt reuses to the round's last row, with its state.
    cache.close(b);
    const auto d = cache.open();
    Tokens longer = prompt;
    longer.push_back(9);
    require(cache.attach(d,longer,std::uint32_t(longer.size()-1)) == 8,"reuse to the round's last row");
    require(storage.slots[0][cache.live_slot(d,0)] == states[7],"the reused state is the one the round wrote");
    // A final_row round may keep fewer rows (the engine then writes that state into scratch[0], as in recompute).
    const auto e = cache.open();
    const auto partial = cache.begin(e,6,RoundStates::final_row);
    storage.slots[0][partial.scratch[0]] = states[3];
    cache.commit(e,partial.id,Tokens(prompt.begin(),prompt.begin()+4));
    require(cache.committed(e) == 4 && storage.slots[0][cache.live_slot(e,0)] == states[3],"a shorter final_row commit");
    // Every-row rounds keep their per-row scratch.
    const auto next = cache.begin(e,3);
    require(next.states == RoundStates::every_row && next.scratch[2] != no_slot,"every_row rounds take a slot per row");
    cache.discard(e,next.id);
    rejects_as<std::invalid_argument>([&] { cache.begin(e,2,RoundStates(7)); },"unknown round states");
    // try_begin, discard and close of final_row rounds return their one slot; the next round is every_row again.
    cache.retire(cache.epoch());
    const auto before = cache.stats().free_slots;
    const auto tried = cache.try_begin(e,5,RoundStates::final_row);
    require(tried && tried->states == RoundStates::final_row && cache.stats().free_slots == before-1,"try_begin takes one slot");
    cache.discard(e,tried->id);
    cache.retire(cache.epoch());   // released slots become free once their epoch retires
    require(cache.stats().free_slots == before,"a discarded final_row round returns its slot");
    const auto again = cache.begin(e,2);
    require(again.states == RoundStates::every_row && again.scratch[1] != no_slot,"a later round is every_row by default");
    cache.discard(e,again.id);
    cache.begin(e,4,RoundStates::final_row);
    require(cache.close(e),"close discards a pending final_row round");
    cache.retire(cache.epoch());
}

int main(int argc, char** argv) {
    const std::array<std::pair<std::string_view,void(*)()>,23> cases{{
        {"state_config",config_rules},{"state_transactions",transactions},{"state_ownership",ownership},
        {"state_prefixes",prefixes},{"state_collisions",collisions},{"state_reuse",reuse},
        {"state_completeness",completeness},{"state_index",index_probing},{"state_lru",lru},{"state_epochs",epochs},
        {"state_forks",forks},{"state_recurrent",recurrent},{"state_hybrid",hybrid},{"state_failures",failures},
        {"state_randomized",randomized},{"state_capacity",capacity},{"state_layout",layout},{"state_metadata",metadata},
        {"state_hot_path",hot_path},{"state_checkpoints",checkpoints},{"state_snapshots",snapshots},{"state_windows",windows},{"state_round_states",round_states}}};
    try {
        const std::string_view name = argc > 1 ? argv[1] : "";
        bool found = false;
        for(const auto& [label,run]:cases) if(name == "all" || name == label) {
            run(); found = true;
            std::printf("crit core state passed: %.*s\n",int(label.size()),label.data());
        }
        require(found,"unknown state test case");
    } catch(const std::exception& e) {
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
    return 0;
}
