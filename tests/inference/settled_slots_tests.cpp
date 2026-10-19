#include "settled_slots.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <new>
#include <stdexcept>
#include <string>
using Slots = crit::techniques::SettledSlots<std::string>;
template <class F> void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception &) {
    rejected = true;
  }
  assert(rejected);
}
auto storage() { return std::make_shared<std::vector<unsigned char>>(128); }
void load(Slots &slots, const std::string &key) {
  auto upload = slots.begin_upload(key);
  slots.finish_upload(upload, true);
}
void returned(Slots &slots, Slots::Loan loan) {
  slots.begin_return(loan);
  slots.reclaim_loan(loan);
}
void loan_admission() {
  rejects([] { Slots x(2, 0, storage()); });
  rejects([] { Slots x(2, 8, {}); });
  rejects([] { Slots x(2, std::numeric_limits<std::size_t>::max(), storage()); });
  Slots metadata(4);
  rejects([&] { metadata.begin_loan({0, 1}); });
  Slots cache(4, 8, storage());
  load(cache, "a"); load(cache, "b");
  auto binding = cache.binding();
  rejects([&] { cache.begin_loan({0, 0}); });
  rejects([&] { cache.begin_loan({4, 1}); });
  rejects([&] { cache.begin_loan({1, std::numeric_limits<std::size_t>::max()}); });
  auto pin = *cache.acquire("a");
  rejects([&] { cache.begin_loan({0, 2}); }); // Queued and running readers share protection.
  assert(cache.valid(binding) && cache.resident() == 2);
  cache.settle(pin);
  auto uploading = cache.begin_upload("c");
  rejects([&] { cache.begin_loan({uploading.slot, 1}); });
  cache.finish_upload(uploading, false);
  const std::array<Slots::Relocation, 2> aliases{{{0, 3}, {1, 3}}};
  rejects([&] { cache.begin_loan({0, 2}, aliases); });
  const Slots::Relocation inside{0, 1}, empty{2, 3}, out{0, 4};
  rejects([&] { cache.begin_loan({0, 2}, std::span(&inside, 1)); });
  rejects([&] { cache.begin_loan({2, 1}, std::span(&empty, 1)); });
  rejects([&] { cache.begin_loan({0, 1}, std::span(&out, 1)); });
  assert(cache.resident() == 2 && cache.available_capacity() == 4);
  auto b = *cache.acquire("b");
  const Slots::Relocation pinned{0, b.slot};
  rejects([&] { cache.begin_loan({0, 1}, std::span(&pinned, 1)); });
  cache.settle(b);
  cache.close();
  rejects([&] { cache.begin_upload("closed"); });
  rejects([&] { cache.acquire("a"); });
  assert(!cache.valid(binding));
}
void relocation_failure() {
  Slots cache(5, 8, storage());
  load(cache, "a"); load(cache, "b"); load(cache, "c");
  const std::array<Slots::Relocation, 2> moves{{{0, 3}, {1, 4}}};
  auto loan = cache.begin_loan({0, 2}, moves);
  assert(cache.resident() == 1 && cache.available_capacity() == 1);
  assert(!cache.acquire("a") && !cache.acquire("b"));
  rejects([&] { cache.begin_upload("a"); }); // Reserved survivor identity.
  rejects([&] { cache.acquire_loan(loan, storage()); });
  rejects([&] { cache.activate_loan(loan); });
  auto owner = storage(); std::weak_ptr<void> retained = owner;
  auto copy = cache.begin_relocation(loan, 0, owner);
  owner.reset();
  assert(!retained.expired());
  rejects([&] { cache.begin_relocation(loan, 0, storage()); });
  rejects([&] { cache.begin_relocation(loan, 2, storage()); });
  cache.settle_relocation(copy, true);
  assert(retained.expired());
  rejects([&] { cache.settle_relocation(copy, true); });
  rejects([&] { cache.activate_loan(loan); });
  auto failed = cache.begin_relocation(loan, 1, storage());
  // Simulated partial copy/throw has no completion yet. Cancellation is not a fence.
  cache.begin_return(loan);
  rejects([&] { cache.reclaim_loan(loan); });
  rejects([&] { cache.begin_relocation(loan, 1, storage()); });
  rejects([&] { cache.close(); });
  cache.settle_relocation(failed, false); // Explicit backend drain finally succeeds.
  cache.reclaim_loan(loan);
  assert(cache.resident() == 1 && !cache.acquire("a") && !cache.acquire("b"));
  assert(cache.available_capacity() == 5);
  rejects([&] { cache.activate_loan(loan); });
  rejects([&] { cache.settle_relocation(failed, true); });
  load(cache, "a");
  assert(cache.resident() == 2);
  cache.close();

  Slots abandoned(2, 8, storage());
  load(abandoned, "x");
  const Slots::Relocation move{0, 1};
  auto unused = abandoned.begin_loan({0, 1}, std::span(&move, 1));
  returned(abandoned, unused); // Unsubmitted copies need no fence.
  assert(abandoned.resident() == 0);
  auto retry = abandoned.begin_loan({0, 2});
  abandoned.activate_loan(retry);
  rejects([&] { abandoned.begin_upload("x"); });
  returned(abandoned, retry);
  abandoned.close();
}
void independent_loans_and_owners() {
  auto bank = storage(); std::weak_ptr<void> allocation = bank;
  Slots cache(6, 8, bank), other(6, 8, storage());
  bank.reset();
  auto first = cache.begin_loan({1, 2}), second = cache.begin_loan({4, 1});
  auto foreign = other.begin_loan({1, 2});
  assert(!cache.valid(other.binding()));
  // Three free slots are fragmented; their total does not justify this range.
  rejects([&] { cache.begin_loan({0, 3}); });
  rejects([&] { other.activate_loan(first); });
  rejects([&] { cache.reclaim_loan(foreign); });
  cache.activate_loan(first); cache.activate_loan(second);
  auto owner = storage(); std::weak_ptr<void> work = owner;
  auto queued = cache.acquire_loan(first, owner);
  owner.reset();
  auto running = cache.acquire_loan(first, storage());
  auto independent = cache.acquire_loan(second, storage());
  auto extent = cache.extent(queued);
  assert(extent.offset == 8 && extent.nbytes == 16);
  rejects([&] { cache.extent({second, queued.id}); });
  rejects([&] { other.settle_loan_use(queued); });
  cache.begin_return(first);
  rejects([&] { cache.acquire_loan(first, storage()); });
  rejects([&] { cache.reclaim_loan(first); });
  rejects([&] { cache.close(); });
  assert(!allocation.expired() && !work.expired());
  cache.settle_loan_use(queued);
  assert(work.expired());
  rejects([&] { cache.settle_loan_use(queued); });
  rejects([&] { cache.extent(queued); });
  rejects([&] { cache.reclaim_loan(first); });
  cache.settle_loan_use(running);
  cache.reclaim_loan(first);
  assert(cache.available_capacity() == 5);
  cache.settle_loan_use(independent);
  returned(cache, second);
  auto fresh = cache.begin_loan({1, 2});
  assert(fresh.id != first.id);
  rejects([&] { cache.activate_loan(first); });
  returned(cache, fresh);
  cache.close();
  assert(allocation.expired());
  returned(other, foreign);
  other.close();
}
void failed_activation_and_reused_destination() {
  Slots cache(2, 8, storage());
  load(cache, "a");
  const Slots::Relocation move{0, 1};
  auto loan = cache.begin_loan({0, 1}, std::span(&move, 1));
  auto copy = cache.begin_relocation(loan, 0, storage());
  cache.settle_relocation(copy, false);
  rejects([&] { cache.activate_loan(loan); });
  assert(!cache.acquire("a") && cache.available_capacity() == 0);
  returned(cache, loan);
  load(cache, "a");
  auto source = *cache.acquire("a");
  cache.settle(source);
  const Slots::Relocation retry_move{source.slot, 1 - source.slot};
  loan = cache.begin_loan({source.slot, 1}, std::span(&retry_move, 1));
  copy = cache.begin_relocation(loan, 0, storage());
  cache.settle_relocation(copy, true);
  cache.activate_loan(loan);
  auto use = *cache.acquire("a");
  assert(use.slot == retry_move.destination);
  rejects([&] { cache.begin_loan({retry_move.destination, 1}); });
  cache.settle(use);
  // A relocated destination can serve another request while scratch is active.
  auto destination_loan = cache.begin_loan({retry_move.destination, 1});
  cache.activate_loan(destination_loan);
  returned(cache, loan); // Must not unlock the newer destination loan.
  assert(cache.available_capacity() == 1);
  rejects([&] { cache.begin_loan({retry_move.destination, 1}); });
  returned(cache, destination_loan);
  cache.close();
}
void bytes_and_repeated_recovery() {
  std::mt19937 random(512);
  for (std::size_t trial = 0; trial < 250; ++trial) {
    const std::size_t capacity = 4 + random() % 9, stride = 1 + random() % 23;
    auto bank = std::make_shared<std::vector<unsigned char>>(capacity * stride);
    Slots cache(capacity, stride, bank);
    for (std::size_t cycle = 0; cycle < 5; ++cycle) {
      // Refill all slots from independently generated exact source values.
      for (std::size_t key = 0; key < capacity; ++key) {
        auto name = std::to_string(key);
        auto hit = cache.acquire(name);
        if (hit) cache.settle(*hit);
        else {
          auto upload = cache.begin_upload(name);
          std::fill_n(bank->data() + upload.slot * stride, stride, static_cast<unsigned char>(key + 1));
          cache.finish_upload(upload, true);
        }
      }
      const std::size_t first = random() % capacity;
      const std::size_t count = 1 + random() % (capacity - first);
      // Choose survivors by physical location; policy is caller-owned.
      std::vector<Slots::Relocation> moves;
      std::vector<std::string> survivors;
      std::size_t destination = 0;
      for (std::size_t key = 0; key < capacity; ++key) {
        auto hit = cache.acquire(std::to_string(key));
        if (!hit) continue;
        auto slot = hit->slot; cache.settle(*hit);
        if (slot < first || slot >= first + count) continue;
        while (destination >= first && destination < first + count) ++destination;
        if (destination == capacity) break;
        moves.push_back({slot, destination++});
        survivors.push_back(std::to_string(key));
      }
      auto captured = cache.binding();
      auto loan = cache.begin_loan({first, count}, moves);
      assert(!cache.valid(captured));
      for (std::size_t i = 0; i < moves.size(); ++i) {
        auto copy = cache.begin_relocation(loan, i, bank);
        std::copy_n(bank->data() + moves[i].source * stride, stride,
                    bank->data() + moves[i].destination * stride);
        cache.settle_relocation(copy, true);
        assert(!cache.acquire(survivors[i])); // No partial READY publication.
      }
      auto preparing = cache.binding();
      cache.activate_loan(loan);
      assert(!cache.valid(preparing));
      auto scratch = cache.acquire_loan(loan, bank);
      auto extent = cache.extent(scratch);
      std::fill_n(bank->data() + extent.offset, extent.nbytes, static_cast<unsigned char>(255));
      auto active = cache.binding();
      cache.begin_return(loan);
      rejects([&] { cache.reclaim_loan(loan); });
      cache.settle_loan_use(scratch);
      cache.reclaim_loan(loan);
      assert(!cache.valid(active) && cache.available_capacity() == capacity);
      for (std::size_t key = 0; key < capacity; ++key) {
        auto hit = cache.acquire(std::to_string(key));
        if (!hit) continue; // Eviction only changes performance; misses use exact source.
        auto start = bank->data() + hit->slot * stride;
        assert(std::all_of(start, start + stride, [&](auto byte) { return byte == key + 1; }));
        cache.settle(*hit);
      }
      for (const auto &key : survivors) { auto hit = cache.acquire(key); assert(hit); cache.settle(*hit); }
    }
    cache.close();
  }
}
// Reverse loans: slots 4..7 alias memory an owner lent to the cache (KV positions
// a request cannot reach). Before a request writes positions behind slots 5 and 7
// the owner reclaims them with a slot loan; a live reader blocks slot 6 until it
// settles; a survivor moves to a free own slot; returned slots come back EMPTY.
void reverse_loans() {
  Slots cache(8); // address-agnostic: the owner's memory needs no backing owner here
  for (auto key : {"a", "b", "c", "d", "e", "f", "g", "h"}) load(cache, key);
  const std::array<std::size_t, 2> scattered{7, 5};
  auto binding = cache.binding();
  auto request = cache.begin_slot_loan(scattered);
  assert(!cache.valid(binding));
  assert(!cache.acquire("f") && !cache.acquire("h") && cache.resident() == 6);
  assert(cache.available_capacity() == 6);
  assert(cache.on_loan(5) && cache.on_loan(7) && !cache.on_loan(6));
  rejects([&] { (void)cache.on_loan(8); });
  cache.activate_loan(request);
  auto writer = cache.acquire_loan(request, storage()); // the request's KV writes
  rejects([&] { cache.extent(writer); });                 // the owner addresses its own memory

  auto reader = *cache.acquire("g"); // a pass still reads slot 6
  const std::size_t six = 6;
  rejects([&] { cache.begin_slot_loan(std::span(&six, 1)); });
  cache.settle(reader);
  rejects([&] { cache.begin_slot_loan(scattered); });           // already on loan
  const std::array<std::size_t, 2> duplicate{6, 6}, outside{6, 8};
  rejects([&] { cache.begin_slot_loan(duplicate); });
  rejects([&] { cache.begin_slot_loan(outside); });
  rejects([&] { cache.begin_slot_loan(std::span<const std::size_t>{}); });

  // A longer request also reaches slot 4: its expert survives in a free own slot.
  auto evicted = cache.begin_upload("spare"); // empties own slot 0's victim for the move
  cache.finish_upload(evicted, false);
  const std::size_t four = 4;
  const Slots::Relocation survive{4, evicted.slot};
  auto grow = cache.begin_slot_loan(std::span(&four, 1), std::span(&survive, 1));
  auto copy = cache.begin_relocation(grow, 0, storage());
  cache.settle_relocation(copy, true);
  cache.activate_loan(grow);
  auto moved = cache.acquire("e");
  assert(moved && moved->slot == evicted.slot);
  cache.settle(*moved);

  // The request ends: its writers settle, then the slots return to the cache empty.
  cache.begin_return(request);
  rejects([&] { cache.reclaim_loan(request); }); // the writer has not settled
  cache.settle_loan_use(writer);
  cache.reclaim_loan(request);
  returned(cache, grow);
  assert(!cache.acquire("f") && !cache.acquire("h")); // never resurrected
  assert(cache.available_capacity() == 8 && cache.resident() == 5 && !cache.on_loan(5));
  auto refill = cache.begin_upload("f"); // an empty returned slot is the first victim
  assert(!refill.evicted && (refill.slot == 4 || refill.slot == 5 || refill.slot == 7));
  cache.finish_upload(refill, true);
  for (int cycle = 0; cycle < 500; ++cycle) { // repeated requests keep exact accounting
    auto again = cache.begin_slot_loan(scattered);
    cache.activate_loan(again);
    returned(cache, again);
  }
  assert(cache.available_capacity() == 8);
  cache.close();
}
int main() {
  reverse_loans();
  loan_admission();
  relocation_failure();
  independent_loans_and_owners();
  failed_activation_and_reused_destination();
  bytes_and_repeated_recovery();
  rejects([] { Slots empty(0); });
  Slots slots(2);
  assert(!slots.acquire("a"));
  auto a = slots.begin_upload("a");
  assert(!slots.acquire("a"));
  rejects([&] { slots.begin_upload("a"); });
  slots.finish_upload(a, true);
  auto use = slots.acquire("a");
  assert(use && use->slot == a.slot);
  auto second = slots.acquire("a");
  slots.settle(*second);
  rejects([&] { slots.settle(*second); });
  auto b = slots.begin_upload("b");
  slots.finish_upload(b, true);
  auto use_b = slots.acquire("b");
  rejects([&] { slots.begin_upload("c"); });
  slots.settle(*use_b);
  auto c = slots.begin_upload("c");
  assert(c.slot == b.slot && c.evicted && c.generation > b.generation);
  rejects([&] { slots.settle(*use_b); });
  assert(!slots.acquire("b"));
  assert(!slots.acquire("c"));
  slots.finish_upload(c, false);
  assert(!slots.acquire("c"));
  assert(slots.resident() == 1);
  rejects([&] { slots.finish_upload(c, true); });
  slots.settle(*use);
  auto retry = slots.begin_upload("c");
  assert(retry.slot == c.slot && !retry.evicted);
  slots.finish_upload(retry, true);
  assert(slots.resident() == 2);
  auto retained_a = slots.acquire("a");
  assert(retained_a && retained_a->generation == a.generation);
  slots.settle(*retained_a);
  auto ready = slots.acquire("c");
  assert(ready);
  slots.settle(*ready);
  rejects([&] { slots.settle(*ready); });
  Slots first(1), second_directory(1);
  auto one = first.begin_upload("a"), two = second_directory.begin_upload("a");
  rejects([&] { second_directory.finish_upload(one, true); });
  first.finish_upload(one, true);
  second_directory.finish_upload(two, true);
  auto first_use = first.acquire("a"),
       second_use = second_directory.acquire("a");
  rejects([&] { second_directory.settle(*first_use); });
  first.settle(*first_use);
  second_directory.settle(*second_use);
  alignas(Slots) unsigned char storage[sizeof(Slots)];
  auto old = new (storage) Slots(1);
  auto old_upload = old->begin_upload("a");
  old->finish_upload(old_upload, true);
  auto stale = *old->acquire("a");
  old->settle(stale);
  old->~Slots();
  auto replacement = new (storage) Slots(1);
  auto replacement_upload = replacement->begin_upload("a");
  replacement->finish_upload(replacement_upload, true);
  auto replacement_use = *replacement->acquire("a");
  rejects([&] { replacement->settle(stale); });
  replacement->settle(replacement_use);
  replacement->~Slots();
  return 0;
}
