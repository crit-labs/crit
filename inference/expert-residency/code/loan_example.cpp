#include "settled_slots.h"
#include <crit/core.h>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Bank {
  crit::core::Reservation reservation;
  std::vector<unsigned char> bytes;
  Bank(crit::core::Resources &resources, std::size_t size)
      : reservation(resources.reserve(crit::core::Memory::host, size)), bytes(size) {}
};

// Real host bytes, synchronous copies and scratch work. Native adapters replace
// these operations with their own queues and settle only after physical fences.
int main() {
  using Slots = crit::techniques::SettledSlots<std::string>;
  constexpr std::size_t stride = 16;
  crit::core::Resources resources(4 * stride, 0);
  auto bank = std::make_shared<Bank>(resources, 4 * stride);
  Slots cache(4, stride, bank);
  auto upload = cache.begin_upload("weights-a");
  std::fill_n(bank->bytes.data() + upload.slot * stride, stride, static_cast<unsigned char>(42));
  cache.finish_upload(upload, true);
  auto old_binding = cache.binding();

  const Slots::Relocation move{upload.slot, 3};
  auto loan = cache.begin_loan({0, 2}, std::span(&move, 1));
  auto copy = cache.begin_relocation(loan, 0, bank);
  std::copy_n(bank->bytes.data() + move.source * stride, stride,
              bank->bytes.data() + move.destination * stride);
  cache.settle_relocation(copy, true); // The synchronous copy has returned.
  cache.activate_loan(loan);
  if (cache.valid(old_binding)) throw std::logic_error("stale captured binding survived");

  auto use = cache.acquire_loan(loan, bank);
  auto scratch = cache.extent(use);
  std::fill_n(bank->bytes.data() + scratch.offset, scratch.nbytes, static_cast<unsigned char>(99));
  if (resources.usage(crit::core::Memory::host).live != 4 * stride)
    throw std::logic_error("loan changed the physical reservation");
  cache.begin_return(loan); // Stop new borrowers before retiring this request.
  cache.settle_loan_use(use); // All synchronous scratch readers/writers finished.
  cache.reclaim_loan(loan);

  auto survivor = cache.acquire("weights-a");
  if (!survivor || survivor->slot != 3 ||
      !std::all_of(bank->bytes.begin() + 3 * stride, bank->bytes.end(), [](auto x) { return x == 42; }))
    throw std::logic_error("relocated cache content changed");
  cache.settle(*survivor);
  // Returned scratch bytes have no READY key. Ordinary exact-source uploads
  // repopulate them before decode can read them.
  auto refill = cache.begin_upload("weights-b");
  std::fill_n(bank->bytes.data() + refill.slot * stride, stride, static_cast<unsigned char>(7));
  cache.finish_upload(refill, true);
  auto decoded = cache.acquire("weights-b");
  if (!decoded || bank->bytes[decoded->slot * stride] != 7 || cache.available_capacity() != 4)
    throw std::logic_error("cache did not recover after scratch return");
  cache.settle(*decoded);
  bank.reset(); // The directory still retains both storage and its reservation.
  if (resources.usage(crit::core::Memory::host).live != 4 * stride)
    throw std::logic_error("cache released reserved backing early");
  cache.close();
  if (resources.usage(crit::core::Memory::host).live != 0)
    throw std::logic_error("settled cache did not release its reservation");
  std::cout << "Cache loan: complete relocation, scratch return and exact refill passed.\n";
}
