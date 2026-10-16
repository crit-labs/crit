#include "crit/sha256.h"
#include <algorithm>
#include <cstring>

namespace crit::core {
namespace {
constexpr std::uint32_t round_constants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
constexpr std::array<std::uint32_t, 8> initial_state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                                        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
void compress(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    std::uint32_t w[64];
    for(int i = 0; i < 16; ++i)
        w[i] = std::uint32_t(block[4 * i]) << 24 | std::uint32_t(block[4 * i + 1]) << 16 |
               std::uint32_t(block[4 * i + 2]) << 8 | std::uint32_t(block[4 * i + 3]);
    for(int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6], h = state[7];
    for(int i = 0; i < 64; ++i) {
        const std::uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + round_constants[i] + w[i];
        const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}
}

void Sha256::update(std::span<const std::byte> bytes) noexcept {
    const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
    std::size_t size = bytes.size();
    if(size == 0) return;   // an empty span may carry a null pointer, which memcpy must never see
    bytes_ += size;
    if(used_) {
        const std::size_t take = std::min(size, block_.size() - used_);
        std::memcpy(block_.data() + used_, data, take);
        used_ += take; data += take; size -= take;
        if(used_ < block_.size()) return;
        compress(state_, block_.data());
        used_ = 0;
    }
    for(; size >= block_.size(); data += block_.size(), size -= block_.size()) compress(state_, data);
    if(size) { std::memcpy(block_.data(), data, size); used_ = size; }
}
std::array<std::uint8_t, 32> Sha256::finish() noexcept {
    const std::uint64_t bits = bytes_ * 8;   // FIPS 180-4 length modulo 2^64
    block_[used_++] = 0x80;
    if(used_ > 56) {
        std::memset(block_.data() + used_, 0, block_.size() - used_);
        compress(state_, block_.data());
        used_ = 0;
    }
    std::memset(block_.data() + used_, 0, 56 - used_);
    for(int i = 0; i < 8; ++i) block_[std::size_t(56 + i)] = std::uint8_t(bits >> (56 - 8 * i));
    compress(state_, block_.data());
    std::array<std::uint8_t, 32> digest{};
    for(std::size_t i = 0; i < 8; ++i)
        for(std::size_t j = 0; j < 4; ++j) digest[4 * i + j] = std::uint8_t(state_[i] >> (24 - 8 * j));
    state_ = initial_state;
    block_ = {};
    bytes_ = 0;
    used_ = 0;
    return digest;
}
std::string to_hex(std::span<const std::uint8_t, 32> digest) {
    static const char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for(std::size_t i = 0; i < 32; ++i) { out[2 * i] = hex[digest[i] >> 4]; out[2 * i + 1] = hex[digest[i] & 15]; }
    return out;
}
std::string sha256_hex(std::span<const std::byte> bytes) {
    Sha256 hasher;
    hasher.update(bytes);
    const auto digest = hasher.finish();
    return to_hex(digest);
}
std::string sha256_file(const ReadOnlyFile& file, const ReadOptions& options) {
    Sha256 hasher;
    file.read_chunks(0, file.identity().size, [&](std::span<const std::byte> bytes, std::uint64_t) { hasher.update(bytes); }, options);
    file.verify_unchanged();
    const auto digest = hasher.finish();
    return to_hex(digest);
}
}
