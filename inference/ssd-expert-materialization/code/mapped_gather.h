#pragma once
// Sparse expert-region gather over a read-only mapping. Core's ReadOnlyFile
// (crit/file.h) owns the file handle, the mapping and the read-ahead hint; this
// header adds coalesced hints, request-order output and bounded parallel copies.
#include <crit/file.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace crit::storage {
struct ByteRegion { std::size_t offset, bytes; };
struct GatherLimits {
  std::size_t output_bytes, regions;
  std::size_t workers = 1, parallel_threshold = 65536;
};
struct GatherResult {
  std::vector<std::byte> bytes; // Regions concatenated in request order, including duplicates.
  std::size_t advised_regions = 0, workers = 0;
};
// Holds an immutable file mapping. Windows denies write/delete sharing; POSIX
// users must independently prevent mutation/truncation while mapped (file()
// gives verify_unchanged()). Mapping is not a content checksum; verify the
// artifact identity before construction. Open failures throw as
// crit::core::ReadOnlyFile::open does; a file above maximum_bytes throws
// std::length_error.
class MappedFile {
  std::shared_ptr<const crit::core::ReadOnlyFile> file_;
  std::span<const std::byte> image_;
public:
  explicit MappedFile(const std::filesystem::path& path, std::size_t maximum_bytes)
      : file_(std::make_shared<const crit::core::ReadOnlyFile>(crit::core::ReadOnlyFile::open(path))) {
    if (file_->identity().size > maximum_bytes) throw std::length_error("Mapped source exceeds byte limit");
    image_ = file_->map();
  }
  const crit::core::ReadOnlyFile& file() const noexcept { return *file_; }
  std::size_t mapped_bytes() const noexcept { return image_.size(); }
  GatherResult gather(std::span<const ByteRegion> requested, GatherLimits limits,
                      const std::atomic_bool* cancelled = nullptr) const {
    if (!limits.workers || limits.workers > 64 || requested.size() > limits.regions)
      throw std::invalid_argument("Gather requires bounded regions and 1..64 workers");
    const auto stop = [&] {
      if (cancelled && cancelled->load(std::memory_order_relaxed)) throw std::runtime_error("Mapped gather cancelled");
    };
    stop();
    const auto file = file_; // Retains the mapping until every task has been joined.
    const auto image = image_;
    std::vector<std::size_t> offsets;
    offsets.reserve(requested.size());
    std::vector<ByteRegion> advice;
    advice.reserve(requested.size());
    std::size_t total = 0;
    for (auto region : requested) {
      if (region.offset > image.size() || region.bytes > image.size() - region.offset || region.bytes > limits.output_bytes - total)
        throw std::out_of_range("Mapped gather source/output bounds exceeded");
      offsets.push_back(total); total += region.bytes;
      if (region.bytes) advice.push_back(region);
    }
    GatherResult result{std::vector<std::byte>(total)};
    std::sort(advice.begin(), advice.end(), [](auto a, auto b) { return a.offset < b.offset; });
    std::size_t merged = 0;
    for (auto region : advice) {
      if (merged && region.offset <= advice[merged - 1].offset + advice[merged - 1].bytes) {
        auto& last = advice[merged - 1];
        last.bytes = std::max(last.offset + last.bytes, region.offset + region.bytes) - last.offset;
      } else advice[merged++] = region;
    }
    advice.resize(merged);
    // Every hint precedes the first copy. A hint is advice only: it never counts
    // as a completed read or as proof that pages are resident.
    for (auto region : advice) { stop(); ++result.advised_regions; file->advise_willneed(region.offset, region.bytes); }
    if (!total) return result;
    result.workers = total >= limits.parallel_threshold ? std::min(limits.workers, requested.size()) : 1;
    std::atomic_bool failed{false};
    const auto copy = [&](std::size_t worker) {
      try {
        for (std::size_t i = worker; i < requested.size(); i += result.workers) {
          stop();
          if (failed.load(std::memory_order_relaxed)) return;
          const auto region = requested[i];
          // Bound cancellation latency even for one very large region.
          for (std::size_t done = 0; done < region.bytes;) {
            stop();
            const auto count = std::min<std::size_t>(1024 * 1024, region.bytes - done);
            std::memcpy(result.bytes.data() + offsets[i] + done, image.data() + region.offset + done, count);
            done += count;
          }
        }
      } catch (...) { failed.store(true, std::memory_order_relaxed); throw; }
    };
    if (result.workers == 1) copy(0);
    else {
      std::vector<std::future<void>> pending;
      pending.reserve(result.workers);
      std::exception_ptr error;
      try {
        for (std::size_t worker = 0; worker < result.workers; ++worker)
          pending.push_back(std::async(std::launch::async, copy, worker));
      } catch (...) { error = std::current_exception(); failed.store(true, std::memory_order_relaxed); }
      for (auto& task : pending) {
        try { task.get(); } catch (...) { if (!error) error = std::current_exception(); }
      }
      if (error) std::rethrow_exception(error);
    }
    stop();
    return result;
  }
};
}
