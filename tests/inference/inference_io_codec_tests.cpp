#include "row_fp8.h"
#include "mapped_gather.h"
#include <array>
#include <fstream>
#include <iostream>

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
  try { function(); } catch (const std::exception&) { return; }
  throw std::runtime_error("Expected invalid input to fail");
}
void codec() {
  using namespace crit::fp8;
  for (unsigned code = 0; code < 256; ++code) {
    if ((code & 127U) == 127) { rejects([&] { decode(static_cast<std::uint8_t>(code)); }); continue; }
    const auto byte = static_cast<std::uint8_t>(code);
    const auto value = decode(byte);
    const unsigned e = (code >> 3) & 15, m = code & 7;
    const double expected = (e ? (1.0 + m / 8.0) * std::pow(2.0, static_cast<int>(e) - 7) : m / 512.0) * (code & 128 ? -1 : 1);
    require(value == expected && encode(value) == byte, "FP8 codebook round trip differs");
  }
  for (unsigned code = 0; code < 126; ++code) {
    const float halfway = (decode(static_cast<std::uint8_t>(code)) + decode(static_cast<std::uint8_t>(code + 1))) / 2;
    require(encode(halfway) == (code % 2 ? code + 1 : code), "FP8 tie is not even");
  }
  require(decode(126) == 448.f && encode(10000.f) == 126 && std::signbit(decode(128)), "FP8 boundaries differ");
  require(bf16_word(1.f + 1.f / 256) == 0x3f80 && bf16_word(1.f + 3.f / 256) == 0x3f82, "BF16 tie is not even");
  rejects([] { bf16_word(std::numeric_limits<float>::max()); });
  const std::array values{0.f, -0.f, 0.f, 1.f, -2.f, 4.f};
  auto weights = quantize(values, 2, 3); weights.validate();
  require(weights.scales[0] == 0 && weights.codes[1] == 128, "Zero row/sign differs");
  const std::array<std::uint16_t, 3> x{0x3f80, 0x4000, 0x4040};
  const auto output = project(weights, x, 1);
  double oracle = 0;
  for (std::size_t k = 0; k < 3; ++k) oracle += decode(weights.codes[3 + k]) * static_cast<double>(bf16(weights.scales[1])) * bf16(x[k]);
  require(output[0] == 0 && output[1] == oracle, "FP8 row scale semantics differ");
  auto bad = weights; bad.codes[0] = 1; rejects([&] { bad.validate(); });
  bad = weights; bad.scales[0] = 0x8000; rejects([&] { bad.validate(); });
  bad = weights; bad.codes.back() = 255; rejects([&] { bad.validate(); });
  const std::array tiny{std::numeric_limits<float>::denorm_min()};
  require(quantize(tiny, 1, 1).scales[0] == 1, "Underflow rescue missing");
  rejects([] { (void)extent(std::numeric_limits<std::size_t>::max(), 2); });
}
void gather(const std::filesystem::path& folder) {
  using namespace crit::storage;
  std::filesystem::create_directories(folder);
  const auto path = folder / "mapped-source.bin";
  std::vector<std::byte> source(65549);
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::byte>((i * 17 + i / 101) % 256);
  { std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char*>(source.data()), static_cast<std::streamsize>(source.size())); }
  rejects([&] { MappedFile invalid(path, source.size() - 1); });
  {
    MappedFile file(path, source.size());
    const std::array requests{ByteRegion{4090, 20}, ByteRegion{17, 11}, ByteRegion{4090, 20},
                             ByteRegion{65540, 9}, ByteRegion{source.size(), 0}, ByteRegion{4096, 19}};
    std::vector<std::byte> expected;
    for (auto r : requests) expected.insert(expected.end(), source.begin() + static_cast<std::ptrdiff_t>(r.offset), source.begin() + static_cast<std::ptrdiff_t>(r.offset + r.bytes));
    const auto sequential = file.gather(requests, {expected.size(), requests.size(), 1, 0});
    const auto parallel = file.gather(requests, {expected.size(), requests.size(), 4, 0});
    require(sequential.bytes == expected && parallel.bytes == expected, "Mapped gather order/content differs");
    require(parallel.advised_regions == 3 && parallel.workers == 4, "Gather did not coalesce hints or use bounded workers");
    rejects([&] { file.gather(requests, {expected.size() - 1, requests.size()}); });
    rejects([&] { file.gather(requests, {expected.size(), requests.size() - 1}); });
    const std::array outside{ByteRegion{source.size(), 1}};
    rejects([&] { file.gather(outside, {10, 1}); });
    const std::array overflow{ByteRegion{1, std::numeric_limits<std::size_t>::max()}};
    rejects([&] { file.gather(overflow, {10, 1}); });
    std::atomic_bool cancelled{true};
    rejects([&] { file.gather(requests, {expected.size(), requests.size(), 4, 0}, &cancelled); });
    require(file.gather({}, {0, 0}).bytes.empty(), "Empty gather differs");
    require(file.gather(requests, {expected.size(), requests.size(), 4, 0}).bytes == expected, "Source unusable after failure");
  }
  std::filesystem::remove(path);
  const auto empty = folder / "empty.bin";
  { std::ofstream file(empty, std::ios::binary); }
  { MappedFile file(empty, 0); require(file.gather({}, {0, 0}).bytes.empty(), "Empty mapped file differs"); }
  std::filesystem::remove(empty);
}
int main(int argc, char** argv) {
  try { codec(); if (argc != 2) throw std::runtime_error("Fixture output directory required"); gather(argv[1]);
    std::cout << "FP8 codebook/conversion and native mapped gather checks passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
