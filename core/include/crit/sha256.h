#pragma once
#include <crit/file.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace crit::core {
// Portable FIPS 180-4 SHA-256. finish() returns the digest and resets for a new message.
class Sha256 {
public:
    void update(std::span<const std::byte> bytes) noexcept;
    std::array<std::uint8_t, 32> finish() noexcept;
private:
    std::array<std::uint32_t, 8> state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                           0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::array<std::uint8_t, 64> block_ = {};
    std::uint64_t bytes_ = 0;
    std::size_t used_ = 0;
};
std::string to_hex(std::span<const std::uint8_t, 32> digest);   // lowercase, equal to Python hexdigest()
std::string sha256_hex(std::span<const std::byte> bytes);
// Hashes the whole file through read_chunks, then verify_unchanged(); the row digest of crit run records.
std::string sha256_file(const ReadOnlyFile& file, const ReadOptions& options = {.direct = true});
}
