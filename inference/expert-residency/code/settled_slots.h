#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace crit::techniques {
// Serialized host control; the caller supplies physical completion. Keep this
// directory alive until close() succeeds (including after failed backend drain).
// One directory covers one immutable representation and physical slot extent.
// Keys must have nonthrowing hash/equality; loans also need nonthrowing moves.
template <class Key> class SettledSlots {
public:
  struct Use {
    std::size_t slot;
    std::uint64_t generation, id, owner;
  };
  struct Upload {
    std::size_t slot;
    std::uint64_t generation;
    bool evicted;
    std::uint64_t owner;
  };
  // A loan reserves whole adjacent slots in one contiguous physical bank.
  // Copies move complete slots (including scales, padding and guards).
  struct Region { std::size_t first, count; };
  struct Relocation { std::size_t source, destination; };
  struct Loan { std::uint64_t id, owner; };
  struct Copy { Loan loan; std::size_t index; };
  struct Borrow { Loan loan; std::uint64_t id; };
  struct Extent { std::size_t offset, nbytes; };
  struct Binding {
    std::uint64_t owner, generation;
    bool operator==(const Binding &) const = default;
  };

private:
  struct Slot {
    std::optional<Key> ready, pending;
    std::uint64_t generation = 0, age = 0;
    std::size_t uses = 0;
    std::uint64_t loan = 0;
  };
  enum class Phase { preparing, active, returning };
  struct Move {
    Relocation slots;
    bool started = false, settled = false, success = false;
    std::shared_ptr<void> work_owner;
  };
  struct LoanState {
    Region region;
    std::vector<Move> moves;
    Phase phase = Phase::preparing;
    bool published = false;
    std::size_t borrowers = 0;
  };
  struct BorrowState { Loan loan; std::shared_ptr<void> work_owner; };
  std::vector<Slot> slots_;
  std::unordered_map<Key, std::size_t> ready_;
  std::unordered_map<std::uint64_t, Use> uses_;
  std::uint64_t clock_ = 0, next_use_ = 0;
  std::unordered_map<std::uint64_t, LoanState> loans_;
  std::unordered_map<std::uint64_t, BorrowState> borrowers_;
  std::uint64_t next_loan_ = 0, next_borrow_ = 0, binding_generation_ = 1;
  std::size_t slot_bytes_ = 0;
  std::shared_ptr<void> backing_owner_;
  bool closed_ = false;
  static void available(std::uint64_t sequence) {
    if (sequence == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("slot sequence exhausted");
  }
  inline static std::atomic<std::uint64_t> owners_{0};
  static std::uint64_t next_owner() {
    auto value = owners_.load(std::memory_order_relaxed);
    do {
      available(value);
    } while (!owners_.compare_exchange_weak(value, value + 1,
                                            std::memory_order_relaxed));
    return value + 1;
  }
  const std::uint64_t owner_ = next_owner();
  Slot &checked(std::size_t i, std::uint64_t generation) {
    if (i >= slots_.size() || slots_[i].generation != generation)
      throw std::logic_error("stale slot generation");
    return slots_[i];
  }
  void open() const {
    if (closed_) throw std::logic_error("closed slot directory");
  }
  LoanState &checked(Loan loan) {
    open();
    auto it = loans_.find(loan.id);
    if (loan.owner != owner_ || it == loans_.end())
      throw std::logic_error("foreign or retired cache loan");
    return it->second;
  }
  static bool contains(Region region, std::size_t slot) {
    return slot >= region.first && slot - region.first < region.count;
  }
  void invalidate(std::size_t index) {
    auto &s = slots_[index];
    if (s.ready) ready_.erase(*s.ready);
    s.ready.reset();
  }

public:
  explicit SettledSlots(std::size_t capacity) : slots_(capacity) {
    if (!capacity)
      throw std::invalid_argument("empty slot capacity");
  }
  // The owner must jointly retain the allocation AND its physical reservation.
  // This aliases already charged bytes; lending never releases/recharges them.
  SettledSlots(std::size_t capacity, std::size_t slot_bytes,
               std::shared_ptr<void> backing_owner) : SettledSlots(capacity) {
    if (!slot_bytes || capacity > std::numeric_limits<std::size_t>::max() / slot_bytes || !backing_owner)
      throw std::invalid_argument("bounded slot geometry and reserved backing owner required");
    slot_bytes_ = slot_bytes;
    backing_owner_ = std::move(backing_owner);
  }
  SettledSlots(const SettledSlots &) = delete;
  SettledSlots &operator=(const SettledSlots &) = delete;
  std::optional<Use> acquire(const Key &key) {
    open();
    auto it = ready_.find(key);
    if (it == ready_.end() || slots_[it->second].loan)
      return {};
    available(clock_);
    available(next_use_);
    auto &s = slots_[it->second];
    Use use{it->second, s.generation, ++next_use_, owner_};
    uses_.emplace(use.id, use);
    ++s.uses;
    s.age = ++clock_;
    return use;
  }
  // The receipt is supplied by the caller only after every physical use of
  // this acquisition has completed. Logical graph submission is insufficient.
  void settle(Use use) {
    open();
    auto it = uses_.find(use.id);
    if (use.owner != owner_ || it == uses_.end() ||
        it->second.slot != use.slot || it->second.generation != use.generation)
      throw std::logic_error("unowned slot settlement");
    auto &s = checked(use.slot, use.generation);
    --s.uses;
    uses_.erase(it);
  }
  Upload begin_upload(const Key &key) {
    open();
    if (ready_.count(key))
      throw std::logic_error("key already resident");
    std::optional<std::size_t> victim;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      auto &s = slots_[i];
      if (s.pending && *s.pending == key)
        throw std::logic_error("duplicate pending upload");
      // Failed uploads leave empty slots with recent ages. Prefer any empty
      // slot before evicting valid contents; compare ages within each class.
      if (!s.uses && !s.pending && !s.loan &&
          (!victim || (!s.ready && slots_[*victim].ready) ||
           (s.ready.has_value() == slots_[*victim].ready.has_value() &&
            s.age < slots_[*victim].age)))
        victim = i;
    }
    if (!victim)
      throw std::runtime_error("all slots protected");
    auto &s = slots_[*victim];
    available(clock_);
    available(s.generation);
    available(binding_generation_);
    bool evicted = s.ready.has_value();
    if (s.ready)
      ready_.erase(*s.ready);
    s.ready.reset();
    s.pending = key;
    ++s.generation;
    s.age = ++clock_;
    ++binding_generation_;
    return Upload{*victim, s.generation, evicted, owner_};
  }
  // Success means the encoded bytes are ready for consumers. A failed upload
  // leaves an empty slot; the former bytes have no valid published identity.
  void finish_upload(Upload upload, bool success) {
    open();
    if (upload.owner != owner_)
      throw std::logic_error("foreign slot upload");
    auto &s = checked(upload.slot, upload.generation);
    if (!s.pending)
      throw std::logic_error("upload already finished");
    if (success) {
      s.ready = s.pending;
      ready_.emplace(*s.ready, upload.slot);
    }
    s.pending.reset();
  }
  // Explicit policy: relocate selected survivors; evict other unprotected
  // contents in the region and at destinations. No automatic victim scoring.
  // Validate and allocate all loan metadata before changing the directory.
  Loan begin_loan(Region region, std::span<const Relocation> relocations = {}) {
    static_assert(std::is_nothrow_move_constructible_v<Key> && std::is_nothrow_move_assignable_v<Key>);
    open();
    if (!backing_owner_ || !region.count || region.first >= slots_.size() ||
        region.count > slots_.size() - region.first || relocations.size() > region.count)
      throw std::invalid_argument("loan requires a nonempty in-bank slot region and owner");
    auto eligible = [&](std::size_t index) {
      if (index >= slots_.size()) throw std::out_of_range("loan slot outside bank");
      const auto &s = slots_[index];
      if (s.uses || s.pending || s.loan) throw std::runtime_error("loan intersects protected storage");
      available(s.generation);
    };
    for (std::size_t i = region.first; i < region.first + region.count; ++i) eligible(i);
    LoanState state{region, {}};
    state.moves.reserve(relocations.size());
    for (auto move : relocations) {
      if (!contains(region, move.source) || contains(region, move.destination))
        throw std::invalid_argument("relocations must leave the borrowed region");
      eligible(move.destination);
      if (!slots_[move.source].ready) throw std::invalid_argument("relocation source is empty");
      for (const auto &prior : state.moves)
        if (prior.slots.source == move.source || prior.slots.destination == move.destination)
          throw std::invalid_argument("aliased relocation");
      state.moves.push_back(Move{move, false, false, false, {}});
    }
    available(next_loan_);
    available(binding_generation_);
    const Loan loan{next_loan_ + 1, owner_};
    auto [it, inserted] = loans_.emplace(loan.id, std::move(state));
    (void)inserted;
    ++next_loan_;
    ++binding_generation_;
    for (std::size_t i = region.first; i < region.first + region.count; ++i) {
      auto &s = slots_[i];
      s.loan = loan.id;
      ++s.generation;
      bool survivor = false;
      for (const auto &move : it->second.moves) survivor |= move.slots.source == i;
      if (!survivor) invalidate(i);
    }
    for (const auto &move : it->second.moves) {
      auto &s = slots_[move.slots.destination];
      s.loan = loan.id;
      ++s.generation;
      invalidate(move.slots.destination);
    }
    return loan;
  }
  // Register BEFORE enqueue; a throwing/partial launch remains pending. The
  // private work owner retains any extra events, streams, staging or producers.
  Copy begin_relocation(Loan loan, std::size_t index, std::shared_ptr<void> work_owner) {
    auto &state = checked(loan);
    if (state.phase != Phase::preparing || index >= state.moves.size() || !work_owner)
      throw std::logic_error("invalid relocation submission");
    auto &move = state.moves[index];
    if (move.started) throw std::logic_error("relocation already submitted");
    move.work_owner = std::move(work_owner);
    move.started = true;
    return {loan, index};
  }
  // Attest only after ALL physical dependencies of this copy settle, including
  // producers and partial submissions. false means settled but unusable bytes.
  void settle_relocation(Copy copy, bool success) {
    auto &state = checked(copy.loan);
    if (copy.index >= state.moves.size()) throw std::logic_error("invalid relocation receipt");
    auto &move = state.moves[copy.index];
    if (!move.started || move.settled) throw std::logic_error("unowned relocation settlement");
    move.settled = true;
    move.success = success;
    move.work_owner.reset();
  }
  // All survivors become visible together. Their old replacement ages survive;
  // an unsuccessful transaction never publishes partially relocated contents.
  void activate_loan(Loan loan) {
    auto &state = checked(loan);
    if (state.phase != Phase::preparing) throw std::logic_error("loan is not preparing");
    for (const auto &move : state.moves)
      if (!move.settled || !move.success) throw std::runtime_error("relocations are not ready");
    available(binding_generation_);
    for (const auto &move : state.moves) {
      auto &source = slots_[move.slots.source];
      auto &destination = slots_[move.slots.destination];
      ready_.at(*source.ready) = move.slots.destination;
      destination.ready = std::move(source.ready);
      source.ready.reset();
      destination.age = source.age;
      destination.loan = 0;
    }
    state.published = true;
    state.phase = Phase::active;
    ++binding_generation_;
  }
  // Every queued or running borrower registers before submission. Dropping a
  // Borrow value or cancelling logical work does not settle its registered use.
  Borrow acquire_loan(Loan loan, std::shared_ptr<void> work_owner) {
    auto &state = checked(loan);
    if (state.phase != Phase::active || !work_owner) throw std::logic_error("loan is not available");
    available(next_borrow_);
    Borrow use{loan, next_borrow_ + 1};
    borrowers_.emplace(use.id, BorrowState{loan, std::move(work_owner)});
    ++next_borrow_;
    ++state.borrowers;
    return use;
  }
  Extent extent(Borrow use) {
    auto &state = checked(use.loan);
    auto it = borrowers_.find(use.id);
    if (it == borrowers_.end() || it->second.loan.id != use.loan.id)
      throw std::logic_error("unowned scratch use");
    return {state.region.first * slot_bytes_, state.region.count * slot_bytes_};
  }
  void settle_loan_use(Borrow use) {
    auto &state = checked(use.loan);
    auto it = borrowers_.find(use.id);
    if (it == borrowers_.end() || it->second.loan.id != use.loan.id)
      throw std::logic_error("unowned scratch settlement");
    --state.borrowers;
    borrowers_.erase(it);
  }
  // Cancellation uses the same two-phase return. It closes admission immediately
  // but does not assert that queued copies or scratch readers have completed.
  void begin_return(Loan loan) {
    auto &state = checked(loan);
    if (state.phase != Phase::returning) {
      available(binding_generation_);
      state.phase = Phase::returning;
      ++binding_generation_;
    }
  }
  void reclaim_loan(Loan loan) {
    auto &state = checked(loan);
    if (state.phase != Phase::returning || state.borrowers)
      throw std::runtime_error("loan has not retired");
    for (const auto &move : state.moves)
      if (move.started && !move.settled) throw std::runtime_error("relocation still in flight");
    available(binding_generation_);
    if (!state.published) {
      for (const auto &move : state.moves) {
        invalidate(move.slots.source);
        slots_[move.slots.destination].loan = 0;
      }
    }
    for (std::size_t i = state.region.first; i < state.region.first + state.region.count; ++i)
      slots_[i].loan = 0; // Bytes are EMPTY, never resurrect old scratch contents.
    loans_.erase(loan.id);
    ++binding_generation_;
  }
  Binding binding() const { open(); return {owner_, binding_generation_}; }
  bool valid(Binding value) const { return !closed_ && value == Binding{owner_, binding_generation_}; }
  std::size_t capacity() const { return slots_.size(); }
  std::size_t available_capacity() const {
    std::size_t count = 0;
    for (const auto &s : slots_) count += s.loan == 0;
    return count;
  }
  std::size_t resident() const {
    std::size_t count = 0;
    for (const auto &s : slots_) count += s.ready.has_value() && !s.loan;
    return count;
  }
  void close() {
    open();
    if (!uses_.empty() || !loans_.empty()) throw std::runtime_error("live cache uses or loans");
    for (const auto &s : slots_) if (s.pending) throw std::runtime_error("live cache upload");
    ready_.clear();
    slots_.clear();
    backing_owner_.reset();
    closed_ = true;
  }
};
} // namespace crit::techniques
