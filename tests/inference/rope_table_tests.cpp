// attention-state's rope_table.h: fixed tables and refusals ("rules"), and "print <dir> <head_dim> <length|->", the
// table of <dir>/config.json as "attention inv_freq..." in %.17g (or "error <message>") for rope_oracle.py, which
// compares it with the Python twin on the configs of tests/test_rope_tables.py.
#include "rope_table.h"
#include <crit/model_config.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace crit::techniques;
namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F action, std::string_view message) {
  try { action(); } catch (const std::exception& e) {
    require(std::string_view(e.what()).find(message) != std::string_view::npos,
            "unexpected rejection: " + std::string(e.what()) + " (wanted " + std::string(message) + ")");
    return;
  }
  throw std::runtime_error("expected rejection missing: " + std::string(message));
}
struct TempDir {
  fs::path path;
  TempDir() {
    std::random_device device;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path = fs::temp_directory_path() / ("crit-rope-table-" + std::to_string(stamp) + "-" + std::to_string(device()));
    fs::create_directories(path);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};

void rules() {
  TempDir dir;
  const auto config = [&](const std::string& json) {
    std::ofstream(dir.path / "config.json", std::ios::binary) << json;
    return crit::core::load_model_config(dir.path / "config.json");
  };
  const auto plain = rope_table(config(R"({"rope_theta":10000})").rope, 4);
  require(plain.inv_freq == std::vector<double>{1.0, 0.01} && plain.attention_factor == 1.0, "plain rotary table");
  // A declared attention_factor is the final scale, never multiplied by YaRN's own mscale.
  const auto yarn = config(R"({"rope_theta":1e6,"rope_scaling":{"rope_type":"yarn","factor":128,"attention_factor":1.4852,)"
                           R"("original_max_position_embeddings":4096}})");
  require(rope_table(yarn.rope, 128).attention_factor == 1.4852, "declared YaRN attention factor");
  const auto ratio = config(R"({"rope_theta":1e6,"rope_scaling":{"rope_type":"yarn","original_max_position_embeddings":4096}})");
  const auto explicit8 = config(R"({"rope_theta":1e6,"rope_scaling":{"rope_type":"yarn","factor":8,"original_max_position_embeddings":4096}})");
  require(rope_table(ratio.rope, 128, 32768).inv_freq == rope_table(explicit8.rope, 128, 32768).inv_freq,
          "YaRN without factor uses the length ratio");
  rejects([&] { rope_table(config(R"({"rope_scaling":{"rope_type":"su","factor":2}})").rope, 128); }, "has no table");
  rejects([&] { rope_table(config(R"({"rope_scaling":{"rope_type":"linear"}})").rope, 128); }, "needs factor");
  rejects([&] { rope_table(config(R"({"partial_rotary_factor":0.3})").rope, 10); }, "not a positive even number");
  rejects([&] { rope_table(config(R"({"rope_scaling":{"rope_type":"dynamic","factor":2}})").rope, 128); },
          "needs max_position_embeddings");
}

int print(const fs::path& dir, std::uint32_t head_dim, std::optional<std::uint64_t> length) {
  try {
    const auto config = crit::core::load_model_config(dir);
    const auto table = rope_table(config.rope, head_dim, config.max_positions, length);
    std::printf("%.17g", table.attention_factor);
    for (const double f : table.inv_freq) std::printf(" %.17g", f);
    std::printf("\n");
  } catch (const std::exception& e) { std::printf("error %s\n", e.what()); }
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "print" && argc == 5)
      return print(argv[2], std::uint32_t(std::stoul(argv[3])),
                   std::string(argv[4]) == "-" ? std::nullopt : std::optional<std::uint64_t>(std::stoull(argv[4])));
    if (mode == "rules" && argc == 2) {
      rules();
      std::cout << "rope table rules passed\n";
      return 0;
    }
    throw std::invalid_argument("rope_table_tests rules | print <dir> <head_dim> <length|->");
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
