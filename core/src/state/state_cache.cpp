#include "crit/state.h"
#include <algorithm>
#include <limits>
#include <type_traits>

namespace crit::core {
namespace {
enum : std::uint8_t { page_free = 0, page_live = 1, page_retiring = 2, page_cached = 3 };
enum : std::uint8_t { page_hashed = 1, page_indexed = 2 };
constexpr auto last_epoch = std::numeric_limits<std::uint64_t>::max();
std::uint64_t mix(std::uint64_t z) noexcept {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
std::size_t product(std::size_t a, std::size_t b) {
    if(b && a > std::numeric_limits<std::size_t>::max()/b) throw std::overflow_error("state cache metadata accounting overflow");
    return a*b;
}
std::size_t sum(std::size_t a, std::size_t b) {
    if(b > std::numeric_limits<std::size_t>::max()-a) throw std::overflow_error("state cache metadata accounting overflow");
    return a+b;
}
std::uint32_t pages_for(std::uint64_t rows, std::uint32_t page_tokens) noexcept {
    return std::uint32_t((rows+page_tokens-1)/page_tokens);
}
// Scratch slots one round of `rows` rows takes from a component.
std::uint32_t scratch_for(RecurrentCommit mode, std::uint32_t rows) noexcept {
    if(mode == RecurrentCommit::column_checkpoints) return rows;
    if(mode == RecurrentCommit::recompute) return rows ? 1u : 0u;
    return 0;
}
void require_tokens(std::span<const Token> tokens) {
    for(auto token:tokens) if(token < 0) throw std::invalid_argument("state tokens must be nonnegative");
}
void validate(const StateCacheConfig& c) {
    if(!c.page_tokens) throw std::invalid_argument("page_tokens must be positive");
    if(!c.sequences) throw std::invalid_argument("sequences must be positive");
    if(!c.max_tokens) throw std::invalid_argument("max_tokens must be positive");
    if(!c.max_round_rows || c.max_round_rows > c.max_tokens)
        throw std::invalid_argument("max_round_rows must be in [1, max_tokens]");
    if(c.pages == no_page) throw std::invalid_argument("pages exceed the page id range");
    // A sequence that could never fit would report CapacityExhausted forever.
    if(c.pages && c.max_tokens > std::uint64_t(c.pages)*c.page_tokens)
        throw std::invalid_argument("max_tokens exceeds pages x page_tokens");
    if(!c.pages && c.components.empty()) throw std::invalid_argument("state cache requires pages or components");
    // Pages are implied: an attention-free cache has components, which the prefix cache rejects.
    if(c.prefix_cache && !c.components.empty())
        throw std::invalid_argument("prefix cache does not support recurrent components");
    if(c.prefix_cache && !c.identity) throw std::invalid_argument("prefix cache requires a nonzero identity");
    for(const auto& component:c.components) {
        if(component.commit != RecurrentCommit::column_checkpoints && component.commit != RecurrentCommit::recompute &&
           component.commit != RecurrentCommit::in_place)
            throw std::invalid_argument("unknown recurrent commit mode");
        if(component.slots == no_slot) throw std::invalid_argument("component slots exceed the slot id range");
        if(component.slots < std::uint64_t(c.sequences)+scratch_for(component.commit,c.max_round_rows))
            throw std::invalid_argument("component slots must cover every sequence and one round");
    }
}
// Throwing mutations reject reentry from a storage hook; noexcept ones ignore it.
class Busy {
    bool& flag_;
public:
    explicit Busy(bool& flag): flag_(flag) {
        if(flag_) throw std::logic_error("reentrant state cache mutation");
        flag_ = true;
    }
    ~Busy() { flag_ = false; }
    Busy(const Busy&) = delete;
    Busy& operator=(const Busy&) = delete;
};
}

std::uint64_t chain_hash(std::uint64_t previous, std::uint64_t identity, std::span<const Token> tokens) noexcept {
    auto h = mix(previous ^ 0x6a09e667f3bcc909ull);
    h = mix(h ^ identity);
    for(auto token:tokens) h = mix(h ^ std::uint64_t(std::uint32_t(token)));
    return mix(h ^ std::uint64_t(tokens.size()));
}

std::optional<std::size_t> checkpoint_node(std::uint64_t prefix_after, std::span<const std::size_t> accepted,
                                           std::uint64_t boundary_after) {
    if(boundary_after < prefix_after || boundary_after-prefix_after > accepted.size())
        throw std::out_of_range("checkpoint is outside the materialized accepted prefix");
    if(boundary_after == prefix_after) return std::nullopt;
    return accepted[std::size_t(boundary_after-prefix_after-1)];
}

struct StateCache::Impl {
    // Intrusive doubly linked page list over prev/next; a page is in at most one list.
    struct List { PageId head = no_page, tail = no_page; std::uint32_t size = 0; };
    struct Match { std::uint32_t pages = 0, eligible = 0; };
    StateCacheConfig config;
    StateStorage& storage;
    Reservation reservation;
    PageHash hash = chain_hash;
    // pages, page_tokens, sequences, max_round_rows, components, table capacity per sequence
    std::uint32_t P = 0, pt = 0, S = 0, R = 0, C = 0, T = 0;
    std::size_t index_capacity = 0, mask = 0, slot_total = 0;
    // Pages: explicit refcounts, list state, hash chain links and the committed tokens of every row.
    std::vector<std::uint32_t> refs;
    std::vector<std::uint8_t> state, flags;
    std::vector<std::uint64_t> generation, hashes, parent_generation, released;
    std::vector<PageId> parent, prev, next, free_pages, index;
    std::vector<Token> tokens;
    std::uint32_t free_count = 0, indexed = 0;
    List retiring, cached;
    // Slots, flattened per component from base[c]: a LIFO free stack and a FIFO retiring ring each.
    std::vector<std::uint8_t> mode;
    std::vector<std::uint32_t> base, slot_free_count, ring_head, ring_size;
    std::vector<SlotId> slot_free, ring;
    std::vector<std::uint64_t> slot_released;
    // Sequences.
    std::vector<std::uint64_t> seq_generation, round_ids, layouts;
    std::vector<std::uint8_t> opened, pending;
    std::vector<std::uint32_t> committed, staged, table_size, seq_free;
    std::vector<PageId> tables;
    std::vector<SlotId> live, starts, scratch;
    std::vector<RowSlot> kv;
    std::uint32_t seq_free_count = 0, open_count = 0;
    std::uint64_t epoch = 1, next_round = 1, layout_counter = 0, hits = 0, misses = 0, evictions = 0;
    bool busy = false;

    Impl(StateCacheConfig value, StateStorage& hooks): config(std::move(value)), storage(hooks) {
        // An exact-size copy: spare capacity moved in with the caller's vector would be retained uncharged.
        config.components = std::vector<StateComponent>(config.components.begin(),config.components.end());
        if(config.hash) hash = config.hash;
        P = config.pages; pt = config.page_tokens; S = config.sequences; R = config.max_round_rows;
        C = std::uint32_t(config.components.size());
        // Validation keeps ceil(max_tokens / page_tokens) <= pages.
        T = P ? pages_for(config.max_tokens,pt) : 0;
        for(const auto& component:config.components) slot_total = sum(slot_total,component.slots);
        if(config.prefix_cache) {
            // Power of two at least twice the pages: linear probing stays at most half full.
            const auto wanted = product(P,2);
            index_capacity = 2;
            while(index_capacity < wanted) index_capacity = product(index_capacity,2);
            mask = index_capacity-1;
        }
    }
    // Every array with its element count; measured before reserving, then allocated with the same counts.
    template<class F> void arrays(F&& f) {
        const std::size_t p = P, s = S, c = C;
        f(refs,p); f(state,p); f(flags,p); f(generation,p); f(hashes,p); f(parent_generation,p); f(released,p);
        f(parent,p); f(prev,p); f(next,p); f(free_pages,p); f(index,index_capacity); f(tokens,product(p,pt));
        f(mode,c); f(base,c+1); f(slot_free_count,c); f(ring_head,c); f(ring_size,c);
        f(slot_free,slot_total); f(ring,slot_total); f(slot_released,slot_total);
        f(seq_generation,s); f(round_ids,s); f(layouts,s); f(opened,s); f(pending,s);
        f(committed,s); f(staged,s); f(table_size,s); f(seq_free,s);
        f(tables,product(s,T)); f(live,product(s,c)); f(starts,product(s,c)); f(scratch,product(product(s,c),R));
        f(kv,P ? product(s,R) : 0);
    }
    std::size_t measure() {
        std::size_t bytes = product(config.components.capacity(),sizeof(StateComponent));
        arrays([&](auto& array, std::size_t count) {
            bytes = sum(bytes,product(count,sizeof(typename std::remove_reference_t<decltype(array)>::value_type)));
        });
        return bytes;
    }
    void allocate() {
        arrays([](auto& array, std::size_t count) { array.assign(count,{}); });
        std::fill(parent.begin(),parent.end(),no_page);
        std::fill(prev.begin(),prev.end(),no_page);
        std::fill(next.begin(),next.end(),no_page);
        std::fill(index.begin(),index.end(),no_page);
        // LIFO stacks hand out id 0 first.
        for(std::uint32_t i = 0; i < P; ++i) free_pages[i] = P-1-i;
        free_count = P;
        for(std::uint32_t c = 0; c < C; ++c) {
            const auto slots = config.components[c].slots;
            mode[c] = std::uint8_t(config.components[c].commit);
            base[c+1] = base[c]+slots;
            for(std::uint32_t i = 0; i < slots; ++i) slot_free[base[c]+i] = slots-1-i;
            slot_free_count[c] = slots;
        }
        for(std::uint32_t i = 0; i < S; ++i) seq_free[i] = S-1-i;
        seq_free_count = S;
        std::fill(tables.begin(),tables.end(),no_page);
        std::fill(live.begin(),live.end(),no_slot);
        std::fill(starts.begin(),starts.end(),no_slot);
        std::fill(scratch.begin(),scratch.end(),no_slot);
    }

    RecurrentCommit commit_mode(std::uint32_t c) const noexcept { return RecurrentCommit(mode[c]); }
    bool valid(SequenceId id) const noexcept {
        return id.index < S && opened[id.index] && seq_generation[id.index] == id.generation;
    }
    std::uint32_t check(SequenceId id) const {
        if(!valid(id)) throw std::logic_error("stale state sequence");
        return id.index;
    }
    PageId* table(std::uint32_t s) noexcept { return T ? tables.data()+std::size_t(s)*T : nullptr; }
    const PageId* table(std::uint32_t s) const noexcept { return T ? tables.data()+std::size_t(s)*T : nullptr; }
    SlotId* live_of(std::uint32_t s) noexcept { return C ? live.data()+std::size_t(s)*C : nullptr; }
    // Per sequence C x R entries; a round uses the first C x rows, row-major by component.
    SlotId* scratch_of(std::uint32_t s) noexcept { return scratch.data()+std::size_t(s)*C*R; }
    std::size_t row_of(PageId page) const noexcept { return std::size_t(page)*pt; }
    void bump(std::uint32_t s) noexcept { layouts[s] = ++layout_counter; }
    // The next round must copy the partial tail when another owner or the prefix index can still read it.
    bool copy_tail(std::uint32_t s) const noexcept {
        if(!P || !table_size[s] || committed[s]%pt == 0) return false;
        const auto page = table(s)[table_size[s]-1];
        return refs[page] > 1 || (flags[page] & page_hashed);
    }

    void link(List& list, PageId page) noexcept {
        prev[page] = list.tail; next[page] = no_page;
        if(list.tail != no_page) next[list.tail] = page; else list.head = page;
        list.tail = page; ++list.size;
    }
    void unlink(List& list, PageId page) noexcept {
        if(prev[page] != no_page) next[prev[page]] = next[page]; else list.head = next[page];
        if(next[page] != no_page) prev[next[page]] = prev[page]; else list.tail = prev[page];
        prev[page] = next[page] = no_page; --list.size;
    }
    PageId find(std::uint64_t h) const noexcept {
        for(auto i = std::size_t(h)&mask;; i = (i+1)&mask) {
            const auto page = index[i];
            if(page == no_page || hashes[page] == h) return page;
        }
    }
    const Token* stored(PageId page) const noexcept { return tokens.data()+row_of(page); }
    // A hashed page's parent link checks out: it is a root, or its parent is indexed in the same generation.
    bool linked(PageId page) const noexcept {
        const auto up = parent[page];
        return up == no_page || ((flags[up] & page_indexed) && generation[up] == parent_generation[page]);
    }
    void set_parent(PageId page, PageId up) noexcept {
        parent[page] = up;
        parent_generation[page] = up == no_page ? 0 : generation[up];
    }
    // The indexed page holding the same content as hashed page q (same tokens and the same parent link), else q.
    // Links to it make a page computed after a duplicate reachable through the index.
    PageId canonical(PageId q) const noexcept {
        if(flags[q] & page_indexed) return q;
        const auto entry = find(hashes[q]);
        if(entry != no_page && parent[entry] == parent[q] && parent_generation[entry] == parent_generation[q] &&
           std::equal(stored(q),stored(q)+pt,stored(entry))) return entry;
        return q;
    }
    void free_page(PageId page) noexcept {
        state[page] = page_free; flags[page] = 0;
        free_pages[free_count++] = page;
    }
    // Indexes a linked page. First writer wins while its own link checks out; an entry whose parent was evicted or
    // replaced can never match again, so a linked page takes its place (a cached loser goes to the free list).
    void index_page(PageId page) noexcept {
        if((flags[page] & page_indexed) || !linked(page)) return;
        const auto h = hashes[page];
        for(auto i = std::size_t(h)&mask;; i = (i+1)&mask) {
            const auto entry = index[i];
            if(entry != no_page && hashes[entry] != h) continue;
            if(entry == no_page) ++indexed;
            else {
                if(linked(entry)) return;
                flags[entry] = std::uint8_t(flags[entry] & ~page_indexed);
                if(state[entry] == page_cached) { unlink(cached,entry); free_page(entry); ++evictions; }
            }
            index[i] = page; flags[page] = std::uint8_t(flags[page] | page_indexed);
            return;
        }
    }
    // Linear-probing removal with backward shift, so lookups never need tombstones.
    void unindex(PageId page) noexcept {
        if(!(flags[page] & page_indexed)) return;
        auto i = std::size_t(hashes[page])&mask;
        while(index[i] != page) i = (i+1)&mask;
        index[i] = no_page;
        for(auto j = (i+1)&mask; index[j] != no_page; j = (j+1)&mask) {
            const auto home = std::size_t(hashes[index[j]])&mask;
            if(i <= j ? (i < home && home <= j) : (i < home || home <= j)) continue;
            index[i] = index[j]; index[j] = no_page; i = j;
        }
        flags[page] = std::uint8_t(flags[page] & ~page_indexed);
        --indexed;
    }
    // Callers check capacity first. Free pages go first, then the least recently released cached page.
    PageId take_page() noexcept {
        PageId page;
        if(free_count) page = free_pages[--free_count];
        else { page = cached.head; unlink(cached,page); unindex(page); ++evictions; }
        refs[page] = 1; state[page] = page_live; flags[page] = 0; ++generation[page];
        set_parent(page,no_page);
        return page;
    }
    // Undoes take_page after a failed hook; the page's bytes may be partly written, so never back to the cache.
    void give_back(PageId page) noexcept {
        refs[page] = 0;
        free_page(page);
    }
    void hold(PageId page) noexcept {
        if(!refs[page]) { unlink(state[page] == page_cached ? cached : retiring,page); state[page] = page_live; }
        ++refs[page];
    }
    // An unreferenced page retires at the current epoch.
    void release_page(PageId page) noexcept {
        if(--refs[page]) return;
        state[page] = page_retiring; released[page] = epoch;
        link(retiring,page);
    }
    // Before its last owner releases table entry i, a hashed page whose link no longer checks out is relinked to
    // the canonical copy of its predecessor (same content, so its rows stay valid after it), then indexed if its
    // hash has no live entry. Callers go head to tail, so a duplicate chain rejoins the index level by level.
    void adopt(const PageId* pages_of, std::uint32_t i) noexcept {
        const auto page = pages_of[i];
        if(refs[page] != 1 || !(flags[page] & page_hashed)) return;
        if(i && !linked(page)) set_parent(page,canonical(pages_of[i-1]));
        index_page(page);
    }
    // Shortens the table of s to `keep` pages, releasing tail first so a chain's last pages are evicted first.
    void truncate(std::uint32_t s, std::uint32_t keep) noexcept {
        auto* pages_of = table(s);
        if(config.prefix_cache) for(auto i = keep; i < table_size[s]; ++i) adopt(pages_of,i);
        while(table_size[s] > keep) release_page(pages_of[--table_size[s]]);
    }
    std::uint32_t slot_capacity(std::uint32_t c) const noexcept { return base[c+1]-base[c]; }
    SlotId take_slot(std::uint32_t c) noexcept { return slot_free[base[c]+ --slot_free_count[c]]; }
    void untake_slot(std::uint32_t c, SlotId slot) noexcept { slot_free[base[c]+slot_free_count[c]++] = slot; }
    void release_slot(std::uint32_t c, SlotId slot) noexcept {
        const auto at = (std::uint64_t(ring_head[c])+ring_size[c])%slot_capacity(c);
        ring[base[c]+std::size_t(at)] = slot; ++ring_size[c];
        slot_released[base[c]+slot] = epoch;
    }
    // Hashes page i of sequence s once it is full and committed: h_i = hash(h_{i-1}, identity, tokens). The parent
    // link names the canonical copy of the previous page, so a chain computed after a duplicate stays reachable.
    void seal(std::uint32_t s, std::uint32_t i) noexcept {
        const auto* pages_of = table(s);
        const auto page = pages_of[i];
        hashes[page] = hash(i ? hashes[pages_of[i-1]] : 0,config.identity,std::span<const Token>(stored(page),pt));
        set_parent(page,i ? canonical(pages_of[i-1]) : no_page);
        flags[page] = std::uint8_t(flags[page] | page_hashed);
        index_page(page);
    }
    // Verified full pages of a prompt: each hit must hold exactly the page's tokens and link to exactly the page
    // matched before it (id and generation), so a hash collision never yields a false reuse.
    Match match(std::span<const Token> prompt, std::uint32_t max_reuse, PageId* out) const noexcept {
        Match result;
        if(!config.prefix_cache) return result;
        const auto limit = std::min<std::uint64_t>({std::uint64_t(prompt.size()),max_reuse,config.max_tokens});
        result.eligible = std::uint32_t(limit/pt);
        PageId previous = no_page;
        std::uint64_t h = 0;
        for(; result.pages < result.eligible; ++result.pages) {
            const auto chunk = prompt.subspan(std::size_t(result.pages)*pt,pt);
            h = hash(h,config.identity,chunk);
            const auto page = find(h);
            if(page == no_page || parent[page] != previous ||
               (previous != no_page && parent_generation[page] != generation[previous]) ||
               !std::equal(chunk.begin(),chunk.end(),stored(page))) break;
            if(out) out[result.pages] = page;
            previous = page;
        }
        return result;
    }
    // Ends the pending round keeping inputs.size() rows (zero discards). Publication only: never throws.
    void finish_round(std::uint32_t s, std::span<const Token> inputs) noexcept {
        const auto first = committed[s], rows = staged[s], n = std::uint32_t(inputs.size()), last = first+n;
        bool changed = false;
        if(P) {
            auto* pages_of = table(s);
            for(std::uint32_t i = 0; i < n; ++i) {
                const auto row = first+i;
                tokens[row_of(pages_of[row/pt])+row%pt] = inputs[i];
            }
            if(config.prefix_cache)
                for(auto page = first/pt; (std::uint64_t(page)+1)*pt <= last; ++page) seal(s,page);
            const auto keep = pages_for(last,pt);
            if(table_size[s] > keep) { truncate(s,keep); changed = true; }
        }
        auto* lives = live_of(s);
        for(std::uint32_t c = 0; c < C; ++c) {
            const auto* entries = scratch_of(s)+std::size_t(c)*rows;
            if(commit_mode(c) == RecurrentCommit::column_checkpoints) {
                if(n) { release_slot(c,lives[c]); lives[c] = entries[n-1]; changed = true; }
                for(std::uint32_t i = 0; i < rows; ++i) if(!n || i != n-1) release_slot(c,entries[i]);
            } else if(commit_mode(c) == RecurrentCommit::recompute && rows) {
                if(n) { release_slot(c,lives[c]); lives[c] = entries[0]; changed = true; }
                else release_slot(c,entries[0]);
            }
        }
        committed[s] = last; staged[s] = 0; pending[s] = 0;
        if(changed) bump(s);
    }
    std::optional<Round> begin_round(SequenceId id, std::uint32_t rows, bool throws) {
        Busy guard(busy);
        const auto s = check(id);
        if(pending[s]) throw std::logic_error("state round already pending");
        if(rows > R) throw std::invalid_argument("round rows exceed max_round_rows");
        if(rows > config.max_tokens-committed[s]) throw std::length_error("state sequence exceeds max_tokens");
        if(next_round == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("state round ids exhausted");
        const auto first = committed[s];
        const bool copy = rows && copy_tail(s);
        const auto target = P ? pages_for(std::uint64_t(first)+rows,pt) : 0;
        if(target-table_size[s]+(copy ? 1u : 0u) > free_count+cached.size) {
            if(throws) throw CapacityExhausted("state cache pages exhausted");
            return std::nullopt;
        }
        for(std::uint32_t c = 0; c < C; ++c) if(scratch_for(commit_mode(c),rows) > slot_free_count[c]) {
            if(throws) throw CapacityExhausted("state cache slots exhausted");
            return std::nullopt;
        }
        auto* pages_of = table(s);
        const auto old_tail = copy ? pages_of[table_size[s]-1] : no_page;
        PageId replacement = no_page;
        if(copy) {
            replacement = take_page();
            try { storage.copy_rows(old_tail,replacement,first%pt); }
            catch(...) { give_back(replacement); throw; }
        }
        // Publication: nothing below allocates or throws.
        bool changed = false;
        if(copy) {
            std::copy_n(stored(old_tail),first%pt,tokens.begin()+std::ptrdiff_t(row_of(replacement)));
            if(config.prefix_cache) adopt(pages_of,table_size[s]-1);
            pages_of[table_size[s]-1] = replacement;
            release_page(old_tail);
            changed = true;
        }
        while(table_size[s] < target) { pages_of[table_size[s]++] = take_page(); changed = true; }
        RowSlot* rows_of = P ? kv.data()+std::size_t(s)*R : nullptr;
        for(std::uint32_t i = 0; P && i < rows; ++i) rows_of[i] = {pages_of[(first+i)/pt],(first+i)%pt};
        SlotId* start_of = C ? starts.data()+std::size_t(s)*C : nullptr;
        for(std::uint32_t c = 0; c < C; ++c) {
            start_of[c] = live_of(s)[c];
            const auto taken = scratch_for(commit_mode(c),rows);
            auto* entries = scratch_of(s)+std::size_t(c)*rows;
            for(std::uint32_t i = 0; i < rows; ++i) entries[i] = i < taken ? take_slot(c) : no_slot;
            if(taken) changed = true;
        }
        pending[s] = 1; staged[s] = rows; round_ids[s] = next_round++;
        if(changed) bump(s);
        Round round;
        round.id = round_ids[s]; round.first_row = first; round.rows = rows;
        round.kv = std::span<const RowSlot>(rows_of,P ? rows : 0);
        round.start = std::span<const SlotId>(start_of,C);
        round.scratch = std::span<const SlotId>(C ? scratch_of(s) : nullptr,std::size_t(C)*rows);
        return round;
    }
};

StateCache::StateCache(StateCacheConfig config, StateStorage& storage, Resources& resources) {
    validate(config);
    auto impl = std::make_unique<Impl>(std::move(config),storage);
    const auto bytes = impl->measure();
    impl->reservation = resources.reserve(Memory::host,bytes);
    impl->allocate();
    impl_ = std::move(impl);
}
StateCache::~StateCache() = default;
const StateCacheConfig& StateCache::config() const noexcept { return impl_->config; }

SequenceId StateCache::open() {
    auto& m = *impl_;
    Busy guard(m.busy);
    if(!m.seq_free_count) throw CapacityExhausted("state cache sequences exhausted");
    for(std::uint32_t c = 0; c < m.C; ++c) if(!m.slot_free_count[c]) throw CapacityExhausted("state cache slots exhausted");
    const auto s = m.seq_free[--m.seq_free_count];
    ++m.seq_generation[s]; m.opened[s] = 1; ++m.open_count;
    for(std::uint32_t c = 0; c < m.C; ++c) m.live_of(s)[c] = m.take_slot(c);
    m.bump(s);
    return {s,m.seq_generation[s]};
}
bool StateCache::close(SequenceId id) noexcept {
    auto& m = *impl_;
    if(m.busy || !m.valid(id)) return false;
    const auto s = id.index;
    if(m.pending[s]) m.finish_round(s,{});
    m.truncate(s,0);
    for(std::uint32_t c = 0; c < m.C; ++c) { m.release_slot(c,m.live_of(s)[c]); m.live_of(s)[c] = no_slot; }
    m.opened[s] = 0; m.committed[s] = 0; --m.open_count;
    m.seq_free[m.seq_free_count++] = s;
    return true;
}
SequenceId StateCache::fork(SequenceId parent) {
    auto& m = *impl_;
    Busy guard(m.busy);
    const auto p = m.check(parent);
    if(m.pending[p]) throw std::logic_error("fork requires no pending round");
    if(!m.seq_free_count) throw CapacityExhausted("state cache sequences exhausted");
    for(std::uint32_t c = 0; c < m.C; ++c) if(!m.slot_free_count[c]) throw CapacityExhausted("state cache slots exhausted");
    const auto s = m.seq_free[m.seq_free_count-1];
    auto* lives = m.live_of(s);
    for(std::uint32_t c = 0; c < m.C; ++c) lives[c] = m.take_slot(c);
    try { for(std::uint32_t c = 0; c < m.C; ++c) m.storage.copy_slot(c,m.live_of(p)[c],lives[c]); }
    catch(...) {
        for(auto c = m.C; c-- > 0;) { m.untake_slot(c,lives[c]); lives[c] = no_slot; }
        throw;
    }
    --m.seq_free_count;
    ++m.seq_generation[s]; m.opened[s] = 1; ++m.open_count;
    for(std::uint32_t i = 0; i < m.table_size[p]; ++i) {
        m.table(s)[i] = m.table(p)[i];
        ++m.refs[m.table(s)[i]];
    }
    m.table_size[s] = m.table_size[p]; m.committed[s] = m.committed[p];
    m.bump(s);
    return {s,m.seq_generation[s]};
}
std::uint32_t StateCache::reusable(std::span<const Token> prompt, std::uint32_t max_reuse) const {
    require_tokens(prompt);
    return impl_->match(prompt,max_reuse,nullptr).pages*impl_->pt;
}
std::uint32_t StateCache::attach(SequenceId id, std::span<const Token> prompt, std::uint32_t max_reuse) {
    auto& m = *impl_;
    Busy guard(m.busy);
    const auto s = m.check(id);
    if(m.pending[s] || m.committed[s] || m.table_size[s]) throw std::logic_error("attach requires an empty sequence");
    require_tokens(prompt);
    const auto found = m.match(prompt,max_reuse,m.table(s));
    for(std::uint32_t i = 0; i < found.pages; ++i) m.hold(m.table(s)[i]);
    m.hits += found.pages; m.misses += found.eligible-found.pages;
    m.table_size[s] = found.pages; m.committed[s] = found.pages*m.pt;
    if(found.pages) m.bump(s);
    return m.committed[s];
}
std::uint32_t StateCache::pages_needed(SequenceId id, std::uint32_t rows) const {
    const auto& m = *impl_;
    const auto s = m.check(id);
    if(m.pending[s]) throw std::logic_error("state round already pending");
    if(rows > m.R) throw std::invalid_argument("round rows exceed max_round_rows");
    if(rows > m.config.max_tokens-m.committed[s]) throw std::length_error("state sequence exceeds max_tokens");
    if(!m.P || !rows) return 0;
    return pages_for(std::uint64_t(m.committed[s])+rows,m.pt)-m.table_size[s]+(m.copy_tail(s) ? 1u : 0u);
}
std::optional<Round> StateCache::try_begin(SequenceId id, std::uint32_t rows) { return impl_->begin_round(id,rows,false); }
Round StateCache::begin(SequenceId id, std::uint32_t rows) { return *impl_->begin_round(id,rows,true); }
void StateCache::commit(SequenceId id, std::uint64_t round, std::span<const Token> inputs) {
    auto& m = *impl_;
    Busy guard(m.busy);
    const auto s = m.check(id);
    if(!m.pending[s] || m.round_ids[s] != round) throw std::logic_error("stale state round");
    // Consumed from here on: a rejected commit discards the round.
    if(inputs.size() > m.staged[s]) { m.finish_round(s,{}); throw std::out_of_range("commit exceeds staged rows"); }
    for(auto token:inputs) if(token < 0) {
        m.finish_round(s,{});
        throw std::invalid_argument("state tokens must be nonnegative");
    }
    m.finish_round(s,inputs);
}
void StateCache::discard(SequenceId id, std::uint64_t round) {
    auto& m = *impl_;
    Busy guard(m.busy);
    const auto s = m.check(id);
    if(!m.pending[s] || m.round_ids[s] != round) throw std::logic_error("stale state round");
    m.finish_round(s,{});
}
void StateCache::rewind(SequenceId id, std::uint32_t tokens) {
    auto& m = *impl_;
    Busy guard(m.busy);
    const auto s = m.check(id);
    if(m.pending[s]) throw std::logic_error("rewind requires no pending round");
    if(tokens > m.committed[s]) throw std::out_of_range("rewind exceeds committed rows");
    if(tokens == m.committed[s]) return;
    if(m.C) throw std::logic_error("recurrent state cannot rewind");
    const auto keep = pages_for(tokens,m.pt);
    const bool changed = m.table_size[s] > keep;
    m.truncate(s,keep);
    m.committed[s] = tokens;
    if(changed) m.bump(s);
}
std::span<const PageId> StateCache::block_table(SequenceId id) const {
    const auto s = impl_->check(id);
    return {impl_->table(s),impl_->table_size[s]};
}
std::uint32_t StateCache::committed(SequenceId id) const { return impl_->committed[impl_->check(id)]; }
std::uint32_t StateCache::staged(SequenceId id) const { return impl_->staged[impl_->check(id)]; }
SlotId StateCache::live_slot(SequenceId id, std::uint32_t component) const {
    const auto s = impl_->check(id);
    if(component >= impl_->C) throw std::out_of_range("state component out of range");
    return impl_->live[std::size_t(s)*impl_->C+component];
}
std::uint64_t StateCache::layout_generation(SequenceId id) const { return impl_->layouts[impl_->check(id)]; }
std::uint64_t StateCache::epoch() const noexcept { return impl_->epoch; }
std::uint64_t StateCache::advance_epoch() {
    auto& m = *impl_;
    Busy guard(m.busy);
    if(m.epoch == last_epoch) throw std::overflow_error("state cache epochs exhausted");
    return m.epoch++;
}
void StateCache::retire(std::uint64_t completed) noexcept {
    auto& m = *impl_;
    if(m.busy) return;
    if(completed >= m.epoch) { completed = m.epoch; if(m.epoch != last_epoch) ++m.epoch; }
    // Releases append at a nondecreasing epoch, so both retiring orders are sorted by epoch.
    while(m.retiring.head != no_page && m.released[m.retiring.head] <= completed) {
        const auto page = m.retiring.head;
        m.unlink(m.retiring,page);
        if(m.flags[page] & page_indexed) { m.state[page] = page_cached; m.link(m.cached,page); }
        else m.free_page(page);
    }
    for(std::uint32_t c = 0; c < m.C; ++c) {
        while(m.ring_size[c]) {
            const auto slot = m.ring[m.base[c]+m.ring_head[c]];
            if(m.slot_released[m.base[c]+slot] > completed) break;
            m.ring_head[c] = (m.ring_head[c]+1)%m.slot_capacity(c); --m.ring_size[c];
            m.untake_slot(c,slot);
        }
    }
}
std::uint32_t StateCache::evict(std::uint32_t pages) noexcept {
    auto& m = *impl_;
    if(m.busy) return 0;
    std::uint32_t evicted = 0;
    while(evicted < pages && m.cached.head != no_page) {
        const auto page = m.cached.head;
        m.unlink(m.cached,page); m.unindex(page);
        m.free_page(page);
        ++evicted; ++m.evictions;
    }
    return evicted;
}
StateCacheStats StateCache::stats() const noexcept {
    const auto& m = *impl_;
    StateCacheStats result;
    result.free_pages = m.free_count; result.cached_pages = m.cached.size; result.retiring_pages = m.retiring.size;
    result.live_pages = m.P-m.free_count-m.cached.size-m.retiring.size; result.indexed_pages = m.indexed;
    for(std::uint32_t c = 0; c < m.C; ++c) {
        result.free_slots += m.slot_free_count[c]; result.retiring_slots += m.ring_size[c];
        result.live_slots += m.slot_capacity(c)-m.slot_free_count[c]-m.ring_size[c];
    }
    result.sequences = m.open_count;
    result.prefix_hits = m.hits; result.prefix_misses = m.misses; result.evictions = m.evictions;
    return result;
}
std::size_t StateCache::metadata_bytes() const noexcept { return impl_->reservation.bytes(); }
}
