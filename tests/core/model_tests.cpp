// Core model files: safetensors containers, checkpoint indexes, byte access and digests, config.json and
// generation_config.json. Every fixture is generated at runtime in a temporary directory.
#include <crit/model_config.h>
#include <crit/safetensors.h>
#include <crit/sha256.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#endif
using namespace crit::core;
namespace fs = std::filesystem;
namespace {
void require(bool value, const std::string& message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F action, std::string_view message) {
    try { action(); } catch(const std::exception& e) {
        require(std::string_view(e.what()).find(message) != std::string_view::npos,
                "unexpected rejection: " + std::string(e.what()) + " (wanted " + std::string(message) + ")");
        return;
    }
    throw std::runtime_error("expected rejection missing: " + std::string(message));
}
// Like rejects, and the exception must be an E (another type fails even when its message matches).
template<class E, class F> void rejects_as(F action, std::string_view message) {
    try { action(); } catch(const E& e) {
        require(std::string_view(e.what()).find(message) != std::string_view::npos,
                "unexpected rejection: " + std::string(e.what()) + " (wanted " + std::string(message) + ")");
        return;
    } catch(const std::exception& e) {
        throw std::runtime_error("wrong exception type: " + std::string(e.what()) + " (wanted " + std::string(message) + ")");
    }
    throw std::runtime_error("expected rejection missing: " + std::string(message));
}

struct TempDir {
    fs::path path;
    TempDir() {
        std::random_device device;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("crit-core-model-" + std::to_string(stamp) + "-" + std::to_string(device()));
        fs::create_directories(path);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};
void write_file(const fs::path& path, std::string_view bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
    require(bool(out), "fixture write failed");
}
std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
std::string text_of(std::span<const std::byte> bytes) { return {reinterpret_cast<const char*>(bytes.data()), bytes.size()}; }
bool symlinks_work(const fs::path& directory) {
    std::error_code error;
    fs::create_symlink(directory / "missing-target", directory / "probe-link", error);
    if(error) return false;
    fs::remove(directory / "probe-link", error);
    return true;
}

// Independent FIPS 180-4 SHA-256 (straightforward, unrelated to crit/sha256.h) for byte checks.
std::string reference_sha256(std::string_view data) {
    static const std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string message(data);
    message += '\x80';
    while(message.size() % 64 != 56) message += char(0);
    const std::uint64_t bits = std::uint64_t(data.size()) * 8;
    for(int i = 7; i >= 0; --i) message += char((bits >> (8 * i)) & 0xFF);
    auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for(std::size_t block = 0; block < message.size(); block += 64) {
        std::uint32_t w[64];
        for(std::size_t i = 0; i < 16; ++i)
            w[i] = std::uint32_t(static_cast<unsigned char>(message[block + 4 * i])) << 24 | std::uint32_t(static_cast<unsigned char>(message[block + 4 * i + 1])) << 16 |
                   std::uint32_t(static_cast<unsigned char>(message[block + 4 * i + 2])) << 8 | std::uint32_t(static_cast<unsigned char>(message[block + 4 * i + 3]));
        for(int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for(int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for(const auto word:h) for(int shift = 28; shift >= 0; shift -= 4) out += hex[(word >> shift) & 15];
    return out;
}

// Fixture containers. Bit widths are spelled out here rather than read from the library under test.
const std::map<std::string, std::uint64_t> widths{{"BOOL", 8}, {"U8", 8}, {"I8", 8}, {"U16", 16}, {"I16", 16}, {"U32", 32},
    {"I32", 32}, {"U64", 64}, {"I64", 64}, {"F8_E4M3", 8}, {"F8_E5M2", 8}, {"F8_E8M0", 8}, {"F16", 16}, {"BF16", 16},
    {"F32", 32}, {"F64", 64}, {"F4", 4}, {"F6_E2M3", 6}, {"F6_E3M2", 6}};
struct Spec { std::string name, dtype; std::vector<std::uint64_t> shape; };
std::uint64_t bytes_of(const Spec& spec) {
    std::uint64_t elements = 1;
    for(const auto dim:spec.shape) elements *= dim;
    return elements * widths.at(spec.dtype) / 8;
}
std::string shape_json(const std::vector<std::uint64_t>& shape) {
    std::string out = "[";
    for(std::size_t i = 0; i < shape.size(); ++i) out += (i ? "," : "") + std::to_string(shape[i]);
    return out + "]";
}
std::string le64(std::uint64_t value) {
    std::string out(8, '\0');
    for(int i = 0; i < 8; ++i) out[std::size_t(i)] = char((value >> (8 * i)) & 0xFF);
    return out;
}
std::string container(std::string_view header, std::string_view payload = {}, std::optional<std::uint64_t> declared = {}) {
    return le64(declared.value_or(header.size())) + std::string(header) + std::string(payload);
}
std::string pattern(std::size_t size, unsigned seed) {
    std::mt19937 random(seed);
    std::string out(size, '\0');
    for(auto& c:out) c = char(random() & 0xFF);
    return out;
}
// Records laid out contiguously in spec order; metadata is a JSON object text or empty.
struct Built { std::string header, payload; };
Built build(const std::vector<Spec>& specs, std::string_view metadata = {}, unsigned seed = 7) {
    Built out;
    out.header = "{";
    if(!metadata.empty()) out.header += "\"__metadata__\":" + std::string(metadata);
    std::uint64_t offset = 0;
    for(const auto& spec:specs) {
        const auto size = bytes_of(spec);
        if(out.header.size() > 1) out.header += ",";
        out.header += "\"" + spec.name + "\":{\"dtype\":\"" + spec.dtype + "\",\"shape\":" + shape_json(spec.shape) +
                      ",\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + size) + "]}";
        offset += size;
    }
    out.header += "}";
    out.payload = pattern(std::size_t(offset), seed);
    return out;
}
fs::path write_tensors(const fs::path& path, const std::vector<Spec>& specs, std::string_view metadata = {}, unsigned seed = 7) {
    const auto built = build(specs, metadata, seed);
    write_file(path, container(built.header, built.payload));
    return path;
}
fs::path write_raw(const fs::path& path, std::string_view header, std::string_view payload = {}, std::optional<std::uint64_t> declared = {}) {
    write_file(path, container(header, payload, declared));
    return path;
}
fs::path write_index(const fs::path& directory, std::string_view json) {
    const auto path = directory / "model.safetensors.index.json";
    write_file(path, json);
    return path;
}
CheckpointOptions strict_index() { CheckpointOptions options; options.index = IndexPolicy::strict; return options; }
CheckpointOptions header_only() { CheckpointOptions options; options.header_only = true; return options; }
std::string payload_of(const fs::path& file, const TensorInfo& tensor) { return read_file(file).substr(std::size_t(tensor.begin), std::size_t(tensor.nbytes())); }

void model_dtypes() {
    const std::vector<std::pair<std::string, std::uint32_t>> table{{"BOOL", 8}, {"U8", 8}, {"I8", 8}, {"U16", 16}, {"I16", 16},
        {"U32", 32}, {"I32", 32}, {"U64", 64}, {"I64", 64}, {"F8_E4M3", 8}, {"F8_E5M2", 8}, {"F8_E8M0", 8}, {"F16", 16},
        {"BF16", 16}, {"F32", 32}, {"F64", 64}, {"F4", 4}, {"F6_E2M3", 6}, {"F6_E3M2", 6}};
    for(std::size_t i = 0; i < table.size(); ++i) {
        const auto dtype = parse_dtype(table[i].first);
        require(dtype && std::size_t(*dtype) == i, "dtype order or spelling: " + table[i].first);
        require(dtype_name(*dtype) == table[i].first && dtype_bits(*dtype) == table[i].second, "dtype table: " + table[i].first);
    }
    for(const char* name:{"F6", "bf16", "F8_UNKNOWN", "", "C64", "BOOL "}) require(!parse_dtype(name), std::string("accepted dtype ") + name);
    require(dtype_name(DType(200)).empty() && dtype_bits(DType(200)) == 0, "invalid dtype has a name or width");
    TensorInfo t;
    t.dtype = DType::f32;
    t.shape = {2, 3};
    t.begin = 12; t.end = 36;
    require(t.nbytes() == 24 && t.elements() == 6 && t.element_aligned(), "tensor arithmetic");
    t.begin = 14;
    require(!t.element_aligned(), "misaligned F32 reported aligned");
    t.dtype = DType::f4; t.begin = 13;
    require(t.element_aligned(), "sub-byte dtypes are always element aligned");
    t.shape = {};
    require(t.elements() == 1, "scalar elements");
    t.shape = {std::uint64_t(1) << 40, std::uint64_t(1) << 40, 0};
    require(t.elements() == 0, "zero dimension elements");
    t.shape = {std::uint64_t(1) << 40, std::uint64_t(1) << 40};
    require(t.elements() == std::numeric_limits<std::uint64_t>::max(), "elements do not saturate");
    t.begin = 10; t.end = 4;
    require(t.nbytes() == 0, "inverted extent has bytes");
    t.dtype = DType(200);
    require(t.element_aligned(), "invalid dtype alignment");
}

void model_header_valid() {
    TempDir dir;
    // The audit's empty, scalar and out-of-order records.
    const auto path = write_raw(dir.path / "model.safetensors",
        R"({"later":{"dtype":"BF16","shape":[],"data_offsets":[1,3]},"zero":{"dtype":"F32","shape":[0,100],"data_offsets":[1,1]},)"
        R"("first":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", std::string("\x01\x02\x03", 3));
    const auto file = SafetensorsFile::open(path);
    require(file.tensors().size() == 3 && file.tensors()[0].name == "later" && file.tensors()[2].name == "first", "header order");
    const auto payload = file.payload_offset();
    require(payload == 8 + file.header_length() && file.header_bytes().front() == '{', "payload offset");
    require(file.tensors()[0].begin == payload + 1 && file.tensors()[0].nbytes() == 2 && file.tensors()[0].shape.empty(), "scalar extent");
    require(file.tensors()[1].nbytes() == 0 && file.tensors()[1].shape == std::vector<std::uint64_t>{0, 100}, "empty extent");
    require(file.metadata().empty() && !file.header_only(), "metadata or mode");
    const auto checkpoint = Checkpoint::open(path);
    require(checkpoint.tensors().size() == 3 && checkpoint.tensors()[0].name == "later" && checkpoint.tensors()[2].name == "first", "header order");
    require(checkpoint.at("later").dtype == DType::bf16 && checkpoint.find("missing") == nullptr, "lookup");
    require(checkpoint.shards().size() == 1 && checkpoint.index_file() == nullptr && checkpoint.notes().empty(), "single file checkpoint");
    rejects_as<std::out_of_range>([&] { (void)checkpoint.at("missing"); }, "no tensor 'missing'");
    // Metadata in document order, trailing space padding, and a header that is not first in the object.
    const auto padded = write_raw(dir.path / "padded.safetensors",
        R"({"w":{"dtype":"I16","shape":[2],"data_offsets":[0,4]},"__metadata__":{"format":"pt","b":"2","a":""}}     )", "abcd");
    const auto second = SafetensorsFile::open(padded);
    require(second.metadata().size() == 3 && second.metadata()[0].first == "format" && second.metadata()[1].second == "2" &&
            second.metadata()[2].first == "a" && second.metadata()[2].second.empty(), "metadata order");
    require(second.tensors().size() == 1 && second.header_bytes().back() == ' ', "padding");
    // The audit's multi-shard index: header names and dtypes, total_size, sorted shards.
    write_tensors(dir.path / "multi" / "second.safetensors", {{"b", "I32", {3}}});
    write_tensors(dir.path / "multi" / "first.safetensors", {{"a", "BF16", {2}}});
    const auto index = write_index(dir.path / "multi", R"({"weight_map":{"a":"first.safetensors","b":"second.safetensors"},"metadata":{"total_size":16}})");
    for(const auto& options:{CheckpointOptions{}, strict_index()}) {
        const auto multi = Checkpoint::open(index, options);
        require(multi.tensors().size() == 2 && multi.shards().size() == 2 && multi.notes().empty(), "multi-shard checkpoint");
        require(multi.at("b").dtype == DType::i32 && multi.at("b").shard == 1 && multi.at("a").shard == 0, "shard indices");
        require(multi.shards()[0].file().path().filename() == "first.safetensors", "shards sorted by name");
        require(multi.shards()[1].tensors()[0].shard == 1, "shard tensors carry their index");
        require(multi.index_file() && multi.source() == index, "index source");
    }
}

struct Bad { std::string header, payload; std::optional<std::uint64_t> declared; std::string message; };
void model_header_rejects() {
    TempDir dir;
    const auto u8 = [](std::string shape, std::string offsets) {
        return R"({"a":{"dtype":"U8","shape":)" + shape + R"(,"data_offsets":)" + offsets + "}}";
    };
    const std::string record = R"({"dtype":"U8","shape":[2],"data_offsets":[0,2]})";
    const std::string two(2, '\0');
    const std::vector<Bad> cases{
        // Truncated, huge or non-object headers.
        {"{}", "", 0, "header length is zero"},
        {"{}", "", 100, "header length 100 exceeds the 2 bytes after the length prefix"},
        {"{}", "", std::numeric_limits<std::uint64_t>::max(), "header length 18446744073709551615 exceeds the 100000000-byte limit"},
        {"[]", "", {}, "header must start with '{'"},
        {" {}", "", {}, "header must start with '{'"},
        {"{bad", "", {}, "header byte 1"},
        {"{\xff}", "", {}, "header byte 1"},
        {"{} x", "", {}, "unexpected data after the root value"},
        // Duplicate JSON fields at every level.
        {R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})", two, {}, "duplicate key 'a'"},
        {R"({"a":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}})", two, {}, "duplicate key 'dtype'"},
        {R"({"__metadata__":{"claim":"a","claim":"b"}})", two, {}, "duplicate key 'claim'"},
        // Non-integer, boolean, negative, non-finite and overflowing dimensions.
        {u8("[true]", "[0,0]"), "", {}, "dimension must be an integer, not true"},
        {u8("[-1]", "[0,0]"), "", {}, "dimension -1 is negative"},
        {u8("[1.0]", "[0,0]"), "", {}, "dimension must be an integer, not the real number 1"},
        {u8("[NaN]", "[0,0]"), "", {}, "non-finite number literal"},
        {u8("[Infinity]", "[0,0]"), "", {}, "non-finite number literal"},
        {u8("[18446744073709551616]", "[0,0]"), "", {}, "header byte 28: integer exceeds 64 bits"},
        {u8("[-9223372036854775809]", "[0,0]"), "", {}, "header byte 28: integer exceeds 64 bits"},
        {u8("[1]", "[0,123456789012345678901234567890]"), std::string(1, '\0'), {}, "header byte 49: integer exceeds 64 bits"},
        {R"({"a":{"dtype":"F64","shape":[1073741824,1073741824],"data_offsets":[0,0]}})", "", {}, "shape [1073741824, 1073741824] of F64 exceeds the 4398046511104-byte file limit"},
        {R"({"a":{"dtype":"F64","shape":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"data_offsets":[0,0]}})", "", {}, "rank 17 exceeds the 16 limit"},
        {u8("1", "[0,0]"), "", {}, "shape must be an array, not 1"},
        // Bad offsets, overlap, holes, unindexed tails and storage mismatches.
        {u8("[2]", "[true,2]"), two, {}, "data offset must be an integer, not true"},
        {u8("[2]", "[-1,1]"), two, {}, "data offset -1 is negative"},
        {u8("[2]", "[0,3]"), two, {}, "data offset 3 is past the 2-byte payload"},
        {u8("[2]", "[0,18446744073709551615]"), two, {}, "data offset 18446744073709551615 is past the 2-byte payload"},
        {u8("[2]", "[2,0]"), two, {}, "data_offsets end 0 precedes begin 2"},
        {u8("[2]", "[0]"), two, {}, "data_offsets must be [begin, end]"},
        {u8("[2]", "\"02\""), two, {}, "data_offsets must be [begin, end], not the string '02'"},
        {R"({"a":)" + record + R"(,"b":)" + record + "}", two, {}, "overlap at payload offset 0 between 'a' and 'b'"},
        {u8("[2]", "[1,3]"), std::string(3, '\0'), {}, "hole at payload offset 0 before 'a'"},
        {u8("[2]", "[0,2]"), std::string(3, '\0'), {}, "payload has 1 unindexed trailing bytes after 'a'"},
        {R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,2]}})", two, {}, "data_offsets [0, 2) hold 2 bytes but shape [2] of F32 needs 8"},
        {u8("[2]", "[0,2]"), std::string(1, '\0'), {}, "data offset 2 is past the 1-byte payload"},
        {R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}})", std::string(3, '\0'), {},
         "hole at payload offset 1 between 'a' and 'b'"},
        {R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"z":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}})", two, {},
         "overlap at payload offset 1 between 'a' and 'z'"},
        // Malformed metadata and unsupported dtypes.
        {R"({"a":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}})", std::string(1, '\0'), {}, "1 elements of F4 do not fill whole bytes"},
        {R"({"a":{"dtype":"F6","shape":[1],"data_offsets":[0,1]}})", std::string(1, '\0'), {}, "unsupported dtype the string 'F6'"},
        {R"({"a":{"dtype":"F8_UNKNOWN","shape":[1],"data_offsets":[0,1]}})", std::string(1, '\0'), {}, "unsupported dtype the string 'F8_UNKNOWN'"},
        {R"({"a":{"dtype":[],"shape":[1],"data_offsets":[0,1]}})", std::string(1, '\0'), {}, "unsupported dtype an array"},
        {R"({"a":{"dtype":"bad","shape":[1],"data_offsets":[0,1]}})", std::string(1, '\0'), {}, "unsupported dtype the string 'bad'"},
        {R"({"__metadata__":[]})", "", {}, "__metadata__ must map strings to strings, not an array"},
        {R"({"__metadata__":{"claim":true}})", "", {}, "'claim' holds true"},
        {R"({"__metadata__":{"claim":{"dtype":"BF16"}}})", "", {}, "'claim' holds an object"},
        {R"({"__metadata__":{"claim":"\ud800"}})", "", {}, "unpaired surrogate escape"},
        {R"({"__metadata__":null})", "", {}, "__metadata__ must map strings to strings, not null"},
        // Records with other key sets, and invalid names.
        {R"({"a":[]})", std::string(1, '\0'), {}, "tensor 'a': expected exactly the keys dtype, shape and data_offsets"},
        {R"({"a":{"dtype":"U8","shape":[1]}})", std::string(1, '\0'), {}, "expected exactly the keys dtype, shape and data_offsets"},
        {R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"other":0}})", std::string(1, '\0'), {}, "expected exactly the keys dtype, shape and data_offsets"},
        {R"({"":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", std::string(1, '\0'), {}, "tensor '': name is empty"},
        {R"({"a\u0000b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", std::string(1, '\0'), {}, "tensor 'a\\x00b': name contains NUL"},
    };
    const auto path = dir.path / "model.safetensors";
    for(const auto& bad:cases) {
        write_raw(path, bad.header, bad.payload, bad.declared);
        rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, bad.message);
        rejects_as<FormatError>([&] { (void)SafetensorsFile::open(path); }, bad.message);
    }
    try {
        write_raw(path, u8("[2]", "[0,3]"), two);
        (void)SafetensorsFile::open(path);
        throw std::runtime_error("bad offsets accepted");
    } catch(const FormatError& e) {
        require(e.source() == display_path(path) && e.where() == "tensor 'a'", "format error source or where: " + std::string(e.what()));
    }
    write_file(path, "tiny");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, "truncated header length: the file has 4 bytes");
    write_file(path, "");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, "truncated header length: the file has 0 bytes");
    write_raw(dir.path / "model.gguf", "{}");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(dir.path / "model.gguf"); }, "expected a .safetensors file, a .safetensors.index.json index or a model directory");
    rejects_as<FormatError>([&] { (void)SafetensorsFile::open(dir.path / "model.gguf"); }, "expected a .safetensors file");
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(dir.path / "absent.safetensors"); }, "absent.safetensors: open");
    fs::create_directories(dir.path / "folder.safetensors" / "inner");
    rejects_as<std::invalid_argument>([&] { (void)SafetensorsFile::open(dir.path / "folder.safetensors"); }, "not a regular file");
}

void model_limits() {
    TempDir dir;
    const auto path = write_tensors(dir.path / "model.safetensors", {{"a", "BF16", {2}}});
    const auto with = [](auto change) { CheckpointOptions options; change(options.limits); return options; };
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.header_bytes = 8; })); }, "exceeds the 8-byte limit");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.file_bytes = 8; })); }, "bytes exceed the 8-byte file limit");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.dimension = 1; })); }, "dimension 2 exceeds the 1 limit");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.total_bytes = 8; })); }, "checkpoint files exceed the 8-byte limit");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.retained_bytes = 8; })); }, "header and tensor records exceed the 8-byte retained_bytes limit");
    write_tensors(path, {{"a", "U8", {1}}, {"b", "U8", {1}}});
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.tensors = 1; })); }, "more than 1 tensors");
    write_tensors(path, {{"long", "U8", {1}}});
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.name_bytes = 3; })); }, "name of 4 bytes exceeds the 3-byte limit");
    write_tensors(path, {{"a", "U8", {1, 1, 1}}});
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, with([](auto& l) { l.rank = 2; })); }, "rank 3 exceeds the 2 limit");
    for(const auto& bad:{with([](auto& l) { l.rank = 0; }), with([](auto& l) { l.shards = 0; }), with([](auto& l) { l.tensors = 0; }),
                         with([](auto& l) { l.file_bytes = std::uint64_t(1) << 63; }), with([](auto& l) { l.header_bytes = 0; }),
                         with([](auto& l) { l.retained_bytes = std::numeric_limits<std::uint64_t>::max(); })}) {
        rejects_as<std::invalid_argument>([&] { (void)Checkpoint::open(path, bad); }, "safetensors limits must be in [1, 2^63 - 1]");
        rejects_as<std::invalid_argument>([&] { (void)SafetensorsFile::open(path, bad); }, "safetensors limits");
    }
    CheckpointOptions policy;
    policy.index = IndexPolicy(7);
    rejects_as<std::invalid_argument>([&] { (void)Checkpoint::open(path, policy); }, "unknown index policy");
    // Limits at their maximum still work, and the overflow bound holds at 2^63 - 1.
    const auto most = with([](auto& l) {
        l.header_bytes = l.index_bytes = l.name_bytes = l.dimension = l.file_bytes = l.total_bytes = l.retained_bytes = (std::uint64_t(1) << 63) - 1;
    });
    require(Checkpoint::open(path, most).tensors().size() == 1, "maximum limits");
    write_raw(path, R"({"a":{"dtype":"F4","shape":[4294967296,4294967296],"data_offsets":[0,0]}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, most); }, "exceeds the 9223372036854775807-byte file limit");
}

void model_subbyte() {
    TempDir dir;
    const auto path = write_tensors(dir.path / "model.safetensors", {{"scale", "F8_E8M0", {3}}, {"nib", "F4", {2, 3}}, {"pair", "F4", {2}},
        {"six", "F6_E2M3", {4}}, {"other", "F6_E3M2", {8}}, {"none", "F6_E3M2", {0}}, {"e4", "F8_E4M3", {2}}});
    {   // closed before the file is rewritten: Windows refuses writers while a handle holds it
        const auto checkpoint = Checkpoint::open(path);
        const std::map<std::string, std::uint64_t> expected{{"scale", 3}, {"nib", 3}, {"pair", 1}, {"six", 3}, {"other", 6}, {"none", 0}, {"e4", 2}};
        for(const auto& [name, bytes]:expected) {
            const auto& t = checkpoint.at(name);
            require(t.nbytes() == bytes, "sub-byte extent of " + name);
            require(t.element_aligned(), "byte or sub-byte tensor reported misaligned: " + name);
        }
        require(checkpoint.at("nib").elements() == 6 && checkpoint.at("nib").dtype == DType::f4 && checkpoint.at("six").dtype == DType::f6_e2m3, "sub-byte metadata");
        require(payload_of(path, checkpoint.at("other")) == text_of(checkpoint.map(checkpoint.at("other")).bytes), "sub-byte bytes");
    }
    for(const auto& [spec, message]:std::vector<std::pair<std::string, std::string>>{
            {R"({"a":{"dtype":"F4","shape":[3],"data_offsets":[0,1]}})", "3 elements of F4 do not fill whole bytes"},
            {R"({"a":{"dtype":"F6_E2M3","shape":[2],"data_offsets":[0,1]}})", "2 elements of F6_E2M3 do not fill whole bytes"},
            {R"({"a":{"dtype":"F6_E3M2","shape":[2,3],"data_offsets":[0,1]}})", "6 elements of F6_E3M2 do not fill whole bytes"},
            {R"({"a":{"dtype":"F4","shape":[4],"data_offsets":[0,1]}})", "data_offsets [0, 1) hold 1 bytes but shape [4] of F4 needs 2"}}) {
        write_raw(path, spec, std::string(1, '\0'));
        rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, message);
    }
    // Alignment is about absolute file offsets: pad the header so the payload starts on an 8-byte boundary.
    std::string header = R"({"u":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"f":{"dtype":"F32","shape":[1],"data_offsets":[1,5]},)"
                         R"("i":{"dtype":"I32","shape":[1],"data_offsets":[8,12]},"g":{"dtype":"U8","shape":[3],"data_offsets":[5,8]}})";
    while((8 + header.size()) % 8) header += ' ';
    write_raw(path, header, std::string(12, '\x5a'));
    const auto aligned = Checkpoint::open(path);
    require(!aligned.at("f").element_aligned() && aligned.at("i").element_aligned() && aligned.at("u").element_aligned(), "element alignment");
    require(aligned.at("i").begin % 4 == 0 && aligned.at("f").begin % 4 == 1, "alignment fixture");
}

void model_header_only() {
    TempDir dir;
    // crit eval's planning fixtures: a header of BF16 tensors and no payload.
    const std::vector<Spec> specs{{"model.embed_tokens.weight", "BF16", {1000, 64}}, {"model.layers.0.self_attn.q_proj.weight", "BF16", {64, 64}},
                                  {"lm_head.weight", "BF16", {1000, 64}}};
    const auto built = build(specs);
    const auto path = write_raw(dir.path / "model.safetensors", built.header);
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, "data offset 128000 is past the 0-byte payload");
    {   // closed before the file is rewritten: Windows refuses writers while a handle holds it
        const auto checkpoint = Checkpoint::open(path, header_only());
        require(checkpoint.header_only() && checkpoint.shards()[0].header_only() && checkpoint.tensors().size() == 3, "header_only checkpoint");
        require(checkpoint.at("lm_head.weight").nbytes() == 128000 && checkpoint.at("lm_head.weight").begin == 8 + built.header.size() + 136192, "header_only offsets");
        const auto& t = checkpoint.at("lm_head.weight");
        std::vector<std::byte> buffer(std::size_t(t.nbytes()));
        rejects_as<std::logic_error>([&] { (void)checkpoint.map(t); }, "header_only");
        rejects_as<std::logic_error>([&] { checkpoint.read(t, buffer); }, "header_only");
        rejects_as<std::logic_error>([&] { (void)checkpoint.read_chunks(t, [](std::span<const std::byte>, std::uint64_t) {}); }, "header_only");
        rejects_as<std::logic_error>([&] { (void)checkpoint.stream_shard(0, [](const TensorInfo&, std::span<const std::byte>, std::uint64_t) {}); }, "header_only");
        const auto rows = checkpoint.digest_files();
        require(rows.size() == 1 && rows[0].path == "model.safetensors" && rows[0].sha256 == reference_sha256(read_file(path)), "header_only digest");
        checkpoint.verify_digests(rows);
    }
    // A partial payload is fine; holes and offsets beyond the file limit still fail; trailing bytes are not checked.
    write_raw(path, built.header, std::string(100, '\0'));
    require(Checkpoint::open(path, header_only()).tensors().size() == 3, "partial payload in header_only mode");
    write_raw(path, built.header, std::string(400000, '\0'));
    require(Checkpoint::open(path, header_only()).tensors().size() == 3, "trailing bytes in header_only mode");
    write_raw(path, R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, header_only()); }, "hole at payload offset 0 before 'a'");
    write_raw(path, R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, header_only()); }, "overlap at payload offset 1 between 'a' and 'b'");
    auto small = header_only();
    small.limits.file_bytes = 200;
    write_raw(path, R"({"a":{"dtype":"U8","shape":[300],"data_offsets":[0,300]}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, small); }, "shape [300] of U8 exceeds the 200-byte file limit");
    write_raw(path, R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,300]}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(path, small); }, "data offset 300 exceeds the 200-byte file limit");
    // An index of header-only shards keeps every index rule, including a strict total_size.
    write_raw(dir.path / "idx" / "a.safetensors", build({{"x", "BF16", {4}}}).header);
    write_raw(dir.path / "idx" / "b.safetensors", build({{"y", "F32", {3}}}).header);
    const auto index = write_index(dir.path / "idx", R"({"metadata":{"total_size":20},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}})");
    auto strict = header_only();
    strict.index = IndexPolicy::strict;
    require(Checkpoint::open(index, strict).tensors().size() == 2, "header_only index");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, "is past the 0-byte payload");
    // total_bytes counts each file's size or, when larger, the end of its last declared tensor (as the Python port).
    write_raw(dir.path / "ext" / "a.safetensors", build({{"x", "U8", {1000}}}).header);
    write_raw(dir.path / "ext" / "b.safetensors", build({{"y", "U8", {3000}}}).header, std::string(10, '\0'));
    const auto ext_index = write_index(dir.path / "ext", R"({"weight_map":{"x":"a.safetensors","y":"b.safetensors"}})");
    const auto opened = Checkpoint::open(ext_index, header_only());
    const auto ends = opened.at("x").end + opened.at("y").end;
    require(opened.at("x").end > opened.shards()[0].file().identity().size && ends > 4000, "extents past the file sizes");
    const auto index_size = opened.index_file()->identity().size;
    auto bounded = header_only();
    bounded.limits.total_bytes = index_size + ends;
    require(Checkpoint::open(ext_index, bounded).tensors().size() == 2, "extents within total_bytes");
    bounded.limits.total_bytes = index_size + ends - 1;
    rejects_as<FormatError>([&] { (void)Checkpoint::open(ext_index, bounded); }, "checkpoint files exceed the " + std::to_string(index_size + ends - 1) + "-byte limit");
    const auto single = dir.path / "ext" / "a.safetensors";
    bounded.limits.total_bytes = opened.at("x").end;
    require(Checkpoint::open(single, bounded).tensors().size() == 1, "single file extent within total_bytes");
    bounded.limits.total_bytes = opened.at("x").end - 1;
    rejects_as<FormatError>([&] { (void)Checkpoint::open(single, bounded); }, "checkpoint files exceed the");
    bounded.extra_files = {"b.safetensors"};
    bounded.limits.total_bytes = ends;
    require(Checkpoint::open(single, bounded).tensors().size() == 2, "extra file extent within total_bytes");
    bounded.limits.total_bytes = ends - 1;
    rejects_as<FormatError>([&] { (void)Checkpoint::open(single, bounded); }, "checkpoint files exceed the");
    // A trailing tail past the last tensor counts by the file size, as without header_only.
    write_raw(dir.path / "ext" / "c.safetensors", build({{"z", "U8", {2}}}).header, std::string(50, '\0'));
    auto tail = header_only();
    tail.limits.total_bytes = fs::file_size(dir.path / "ext" / "c.safetensors");
    require(Checkpoint::open(dir.path / "ext" / "c.safetensors", tail).tensors().size() == 1, "file size larger than the extent");
    tail.limits.total_bytes -= 1;
    rejects_as<FormatError>([&] { (void)Checkpoint::open(dir.path / "ext" / "c.safetensors", tail); }, "checkpoint files exceed the");
}

void model_index() {
    TempDir dir;
    const auto& root = dir.path;
    write_tensors(root / "first.safetensors", {{"a", "BF16", {2}}});
    write_tensors(root / "second.safetensors", {{"b", "I32", {3}}});
    // Missing, undeclared, duplicate and wrong-shard tensors.
    const std::vector<std::pair<std::string, std::string>> maps{
        {R"({"weight_map":{"a":"first.safetensors","missing":"first.safetensors"}})", "1 weight_map tensors are absent from the shard headers, such as 'missing'"},
        {R"({"weight_map":{"a":"second.safetensors","b":"first.safetensors"}})", "tensor 'a': weight_map puts it in 'second.safetensors' but shard 'first.safetensors' holds it"},
        {R"({"weight_map":{"b":"first.safetensors"}})", "tensor 'a': shard 'first.safetensors' holds it but weight_map does not list it"},
    };
    for(const auto& [json, message]:maps) {
        const auto index = write_index(root, json);
        rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, message);
        rejects_as<FormatError>([&] { (void)Checkpoint::open(index, strict_index()); }, message);
    }
    write_tensors(root / "second.safetensors", {{"a", "I32", {1}}});
    rejects_as<FormatError>([&] { (void)Checkpoint::open(write_index(root, R"({"weight_map":{"a":"first.safetensors","b":"second.safetensors"}})")); },
                            "tensor 'a': appears in shards 'first.safetensors' and 'second.safetensors'");
    write_tensors(root / "second.safetensors", {{"b", "BF16", {2}}});
    // Path escapes and non-portable or reserved shard names.
    for(const std::string name:{"../outside.safetensors", "/tmp/a.safetensors", "C:\\\\a.safetensors", "sub/../a.safetensors", "sub//a.safetensors",
                                "sub/.hidden/a.safetensors", "a/a/a/a/a/a/a/a/a.safetensors", "a\\\\b.safetensors",
                                "file:stream.safetensors", "a.gguf", ".safetensors", "-a.safetensors", "a b.safetensors", "\\u00e9.safetensors"})
        rejects_as<FormatError>([&] { (void)Checkpoint::open(write_index(root, R"({"weight_map":{"a":")" + name + "\"}}")); },
                                "is not a portable relative .safetensors path");
    for(const std::string name:{"CON.safetensors", "nul.safetensors", "com1.safetensors", "LPT9.x.safetensors", "aux.safetensors", "con/a.safetensors"})
        rejects_as<FormatError>([&] { (void)Checkpoint::open(write_index(root, R"({"weight_map":{"a":")" + name + "\"}}")); }, "is a reserved device name");
    // Metadata bounds: strict rejects every deviation; compatible notes unknown keys and a stale total_size.
    const std::string one = R"("weight_map":{"a":"first.safetensors"})";
    for(const auto& [metadata, message]:std::vector<std::pair<std::string, std::string>>{
            {R"({"total_size":true})", "total_size must be an integer in [0, 70368744177664], not true"},
            {R"({"total_size":null})", "total_size must be an integer in [0, 70368744177664], not null"},
            {R"({"total_size":-1})", "total_size must be an integer in [0, 70368744177664], not -1"},
            {R"({"total_size":4.0})", "total_size must be an integer in [0, 70368744177664], not the real number 4"},
            {"[]", "index metadata must be an object, not an array"}}) {
        const auto index = write_index(root, "{" + one + ",\"metadata\":" + metadata + "}");
        rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, message);
        rejects_as<FormatError>([&] { (void)Checkpoint::open(index, strict_index()); }, message);
    }
    auto index = write_index(root, "{" + one + R"(,"metadata":{"total_size":99}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, strict_index()); }, "total_size 99 differs from the 4 tensor bytes in the shards");
    // Notes are read from a closed checkpoint: Windows refuses to rewrite the index while a handle holds it.
    const auto notes_of = [](const fs::path& path) { const auto c = Checkpoint::open(path); return std::vector<std::string>(c.notes().begin(), c.notes().end()); };
    auto notes = notes_of(index);
    require(notes.size() == 1 && notes[0] == "index metadata total_size 99 differs from the 4 tensor bytes in the shards", "stale total_size note");
    index = write_index(root, "{" + one + R"(,"metadata":{"unknown":1,"total_size":4,"another":"x"}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, strict_index()); }, "index metadata: unsupported index metadata keys ['another', 'unknown']");
    notes = notes_of(index);
    require(notes.size() == 1 && notes[0] == "index metadata keys ['another', 'unknown'] are not interpreted", "unknown metadata note");
    index = write_index(root, "{" + one + R"(,"metadata":{"total_parameters":2,"total_size":4}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, strict_index()); }, "unsupported index metadata keys ['total_parameters']");
    require(Checkpoint::open(index).notes().empty(), "total_parameters is understood in compatible mode");
    index = write_index(root, "{" + one + R"(,"metadata":{"total_parameters":true}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, "total_parameters must be an integer in [0, 9223372036854775807], not true");
    index = write_index(root, "{" + one + R"(,"config":{},"format":"pt"})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, strict_index()); }, "index: unsupported index keys ['config', 'format']");
    notes = notes_of(index);
    require(notes.size() == 1 && notes[0] == "index keys ['config', 'format'] are not interpreted", "unknown index key note");
    // One note per group, in a fixed order, whatever the number of keys: at most eight names, then the count.
    index = write_index(root, "{" + one + R"(,"version":1.0,"metadata":{"total_size":4,"producer":"x"}})");
    notes = notes_of(index);
    require(notes.size() == 2 && notes[0] == "index key 'version' is not interpreted" &&
            notes[1] == "index metadata key 'producer' is not interpreted", "single unknown keys");
    std::string many;
    for(int i = 11; i >= 0; --i) many += ",\"k" + std::string(i < 10 ? "0" : "") + std::to_string(i) + "\":" + std::to_string(i);
    index = write_index(root, "{" + one + many + "}");
    const std::string listed = "['k00', 'k01', 'k02', 'k03', 'k04', 'k05', 'k06', 'k07', ...] (12 in all)";
    notes = notes_of(index);
    require(notes.size() == 1 && notes[0] == "index keys " + listed + " are not interpreted", "bounded note: " + notes[0]);
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, strict_index()); }, "unsupported index keys " + listed);
    // Shown keys are cut like Python's: past 48 characters, the first 45 and "..."; never inside a UTF-8 sequence.
    const std::string k46(46, 'k'), k47(47, 'k');
    std::string accents;
    for(int i = 0; i < 50; ++i) accents += "\xC3\xA9";
    index = write_index(root, "{" + one + ",\"" + k47 + "\":1,\"" + k46 + "\":2,\"" + accents + "\":3,\"a'b\":4}");
    notes = notes_of(index);
    std::string accents44;
    for(int i = 0; i < 44; ++i) accents44 += "\xC3\xA9";
    require(notes.size() == 1 && notes[0] == "index keys ['a\\x27b', '" + k46 + "', '" + std::string(44, 'k') + "..., '" +
            accents44 + "...] are not interpreted", "shortened keys: " + notes[0]);
    // 5000 long keys still give one short note and a short strict error.
    std::string flood;
    for(int i = 0; i < 5000; ++i) flood += ",\"" + std::string(200, 'k') + std::to_string(10000 + i) + "\":1";
    index = write_index(root, "{" + one + flood + "}");
    notes = notes_of(index);
    require(notes.size() == 1 && notes[0].size() < 500 && notes[0].find("(5000 in all)") != std::string::npos, "flooded note");
    try { (void)Checkpoint::open(index, strict_index()); throw std::logic_error("flooded strict index accepted"); }
    catch(const FormatError& e) { require(std::string(e.what()).size() < 1000 + display_path(index).size(), "flooded strict error stays short"); }
    // Integers beyond 64 bits load in keys no rule reads, and fail where a count is read.
    index = write_index(root, "{" + one + R"(,"big":123456789012345678901234567890,"metadata":{"total_size":4,"note":[-99999999999999999999]}})");
    notes = notes_of(index);
    require(notes.size() == 2 && notes[0] == "index key 'big' is not interpreted", "wide integers in uninterpreted keys");
    // A count read beyond 64 bits says so, under both policies, worded like the Python port.
    index = write_index(root, "{" + one + R"(,"metadata":{"total_size":18446744073709551616}})");
    for(const auto& options:{CheckpointOptions{}, strict_index()})
        try { (void)Checkpoint::open(index, options); throw std::logic_error("wide total_size accepted"); }
        catch(const FormatError& e) {
            require(std::string(e.what()) == display_path(index) + ": index metadata total_size: integer exceeds 64 bits" &&
                    e.where() == "index metadata total_size", "wide total_size: " + std::string(e.what()));
        }
    index = write_index(root, "{" + one + R"(,"metadata":{"total_parameters":-18446744073709551616}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, ": index metadata total_parameters: integer exceeds 64 bits");
    index = write_index(root, "{" + one + R"(,"metadata":{"total_parameters":99999999999999999999}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, ": index metadata total_parameters: integer exceeds 64 bits");
    // In range but past the limit, or of another type, keeps the range message.
    index = write_index(root, "{" + one + R"(,"metadata":{"total_size":18446744073709551615}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, "index metadata: total_size must be an integer in [0, 70368744177664], not 18446744073709551615");
    index = write_index(root, "{" + one + R"(,"metadata":{"total_size":"18446744073709551616"}})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, "total_size must be an integer in [0, 70368744177664], not the string '18446744073709551616'");
    // The index stays strict JSON otherwise: no NaN or Infinity, even in an uninterpreted key.
    index = write_index(root, "{" + one + R"(,"ratio":Infinity})");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, "non-finite number literal");
    // Weight maps and index documents that are not usable.
    for(const auto& [json, message]:std::vector<std::pair<std::string, std::string>>{
            {R"({"weight_map":{}})", "weight_map must be a non-empty object, not an object"},
            {R"({"weight_map":[]})", "weight_map must be a non-empty object, not an array"},
            {R"({"weight_map":{"a":null}})", "shard for 'a' must be a string, not null"},
            {R"({"weight_map":{"":"first.safetensors"}})", "tensor '': name is empty"},
            {R"({"metadata":{}})", "index has no weight_map"},
            {R"([1])", "index must be a JSON object"},
            {R"({"weight_map":{"a":"one.safetensors","a":"two.safetensors"}})", "duplicate key 'a'"},
            {R"({"weight_map":{"a":"first.safetensors"},})", "trailing comma"}}) {
        index = write_index(root, json);
        rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, message);
    }
    write_index(root, "");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root / "model.safetensors.index.json"); }, "index size 0 is not in [1, 67108864] bytes");
    // Index limits.
    index = write_index(root, R"({"weight_map":{"a":"first.safetensors","b":"second.safetensors"}})");
    require(Checkpoint::open(index).tensors().size() == 2, "two-shard index");
    const auto with = [](auto change) { CheckpointOptions options; change(options.limits); return options; };
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, with([](auto& l) { l.shards = 1; })); }, "2 shards exceed the 1-shard limit");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, with([](auto& l) { l.tensors = 1; })); }, "weight_map lists 2 tensors, more than 1");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, with([](auto& l) { l.index_bytes = 8; })); }, "is not in [1, 8] bytes");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, with([](auto& l) { l.total_bytes = 80; })); }, "checkpoint files exceed the 80-byte limit");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, with([](auto& l) { l.name_bytes = 12; })); }, "shard 'first.safetensors' for 'a' name of 17 bytes exceeds the 12-byte limit");
    // A shard error names the shard file; a missing shard is an OS error naming it.
    write_raw(root / "second.safetensors", "{}", "x");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index); }, display_path(root / "second.safetensors") + ": payload: payload has 1 unindexed trailing bytes");
    fs::remove(root / "second.safetensors");
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(index); }, "second.safetensors: open");
    // Shards listed with a portable path in a subfolder (tensors/...).
    TempDir nested;
    fs::create_directories(nested.path / "tensors");
    write_tensors(nested.path / "tensors" / "first.safetensors", {{"a", "BF16", {2}}});
    write_tensors(nested.path / "tensors" / "second.safetensors", {{"b", "I32", {3}}});
    const auto nested_index = write_index(nested.path, R"({"weight_map":{"a":"tensors/first.safetensors","b":"tensors/second.safetensors"}})");
    for(const auto& options:{CheckpointOptions{}, strict_index()}) {
        const auto checkpoint = Checkpoint::open(nested_index, options);
        require(checkpoint.tensors().size() == 2 && checkpoint.shards().size() == 2 &&
                checkpoint.shards()[1].file().path().filename() == "second.safetensors", "index shards in a subfolder");
    }
}

void model_extra_files() {
    TempDir dir;
    const auto& root = dir.path;
    write_tensors(root / "model-00001-of-00002.safetensors", {{"a", "BF16", {2}}});
    write_tensors(root / "model-00002-of-00002.safetensors", {{"b", "U8", {5}}});
    write_tensors(root / "model_mtp.safetensors", {{"mtp.fc.weight", "F16", {3}}});
    const auto index = write_index(root, R"({"metadata":{"total_size":9},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}})");
    CheckpointOptions options = strict_index();
    options.extra_files = {"model_mtp.safetensors"};
    const auto checkpoint = Checkpoint::open(root, options);
    require(checkpoint.source() == index && checkpoint.shards().size() == 3 && checkpoint.tensors().size() == 3, "extra file joins the checkpoint");
    require(checkpoint.at("mtp.fc.weight").shard == 2 && checkpoint.shards()[2].file().path().filename() == "model_mtp.safetensors", "extras follow index shards");
    require(payload_of(root / "model_mtp.safetensors", checkpoint.at("mtp.fc.weight")) == text_of(checkpoint.map(checkpoint.at("mtp.fc.weight")).bytes), "extra bytes");
    const auto single_options = [&] { CheckpointOptions o; o.extra_files = {"model_mtp.safetensors"}; return o; }();
    require(Checkpoint::open(root / "model-00002-of-00002.safetensors", single_options).tensors().size() == 2, "extra file beside a single file");
    for(const auto& [extras, message]:std::vector<std::pair<std::vector<fs::path>, std::string>>{
            {{"../model_mtp.safetensors"}, "is not a portable flat .safetensors file name"},
            {{"sub/model_mtp.safetensors"}, "is not a portable flat .safetensors file name"},
            {{"model_mtp.bin"}, "is not a portable flat .safetensors file name"},
            {{"CON.safetensors"}, "is a reserved device name"},
            {{"model_mtp.safetensors", "model_mtp.safetensors"}, "extra file 'model_mtp.safetensors' is listed twice"},
            {{"model-00001-of-00002.safetensors"}, "extra file 'model-00001-of-00002.safetensors' is already part of the checkpoint"}}) {
        CheckpointOptions bad;
        bad.extra_files = extras;
        rejects_as<std::invalid_argument>([&] { (void)Checkpoint::open(index, bad); }, message);
    }
    write_tensors(root / "dup.safetensors", {{"b", "U8", {5}}});
    CheckpointOptions dup;
    dup.extra_files = {"dup.safetensors"};
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, dup); }, "tensor 'b': appears in 'model-00002-of-00002.safetensors' and extra file 'dup.safetensors'");
    dup.limits.shards = 2;
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, dup); }, "3 files exceed the 2-shard limit");
    CheckpointOptions absent;
    absent.extra_files = {"absent.safetensors"};
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(index, absent); }, "absent.safetensors: open");
    CheckpointOptions many;
    many.extra_files = {"model_mtp.safetensors"};
    many.limits.tensors = 2;
    rejects_as<FormatError>([&] { (void)Checkpoint::open(index, many); }, "more than 2 tensors across files");
}

void model_directory() {
    TempDir dir;
    const auto& root = dir.path;
    write_tensors(root / "all" / "model.safetensors", {{"single", "U8", {1}}});
    write_tensors(root / "all" / "other.safetensors", {{"other", "U8", {1}}});
    write_tensors(root / "all" / "shard.safetensors", {{"a", "U8", {1}}});
    write_index(root / "all", R"({"weight_map":{"a":"shard.safetensors"}})");
    require(Checkpoint::open(root / "all").source().filename() == "model.safetensors.index.json", "index preferred");
    fs::remove(root / "all" / "model.safetensors.index.json");
    require(Checkpoint::open(root / "all").source().filename() == "model.safetensors", "model.safetensors preferred");
    write_tensors(root / "one" / "weights.safetensors", {{"w", "U8", {1}}});
    fs::create_directories(root / "one" / "nested.safetensors");
    write_file(root / "one" / "notes.txt", "x");
    write_file(root / "one" / "weights.safetensors.index.json.bak", "x");
    require(Checkpoint::open(root / "one").source().filename() == "weights.safetensors", "the only .safetensors file");
    write_tensors(root / "two" / "b.safetensors", {{"b", "U8", {1}}});
    write_tensors(root / "two" / "a.safetensors", {{"a", "U8", {1}}});
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root / "two"); },
                            display_path(root / "two") + ": directory: 2 .safetensors files and no index; name one of 'a.safetensors', 'b.safetensors'");
    for(int i = 0; i < 10; ++i) write_tensors(root / "many" / ("part-" + std::to_string(i) + ".safetensors"), {{"t" + std::to_string(i), "U8", {1}}});
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root / "many"); },
                            "10 .safetensors files and no index; name one of 'part-0.safetensors', 'part-1.safetensors', 'part-2.safetensors', "
                            "'part-3.safetensors', 'part-4.safetensors', 'part-5.safetensors', 'part-6.safetensors', 'part-7.safetensors', ...");
    fs::create_directories(root / "empty");
    write_file(root / "empty" / "config.json", "{}");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root / "empty"); }, "no model.safetensors.index.json, model.safetensors or other .safetensors file");
    // A directory named like a candidate is skipped, as the Python port and Hugging Face (os.path.isfile) do.
    fs::create_directories(root / "odd" / "model.safetensors.index.json");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root / "odd"); }, "no model.safetensors.index.json, model.safetensors or other .safetensors file");
    fs::create_directories(root / "odd" / "model.safetensors");
    write_tensors(root / "odd" / "w.safetensors", {{"w", "U8", {1}}});
    require(Checkpoint::open(root / "odd").source().filename() == "w.safetensors", "directories named like the index and model.safetensors are skipped");
    write_tensors(root / "odd" / "v.safetensors", {{"v", "U8", {1}}});
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root / "odd"); }, "2 .safetensors files and no index; name one of 'v.safetensors', 'w.safetensors'");
#ifndef _WIN32
    // So is a FIFO (any entry neither a regular file nor a link), and the fallback does not count it either.
    fs::create_directories(root / "fifo");
    write_tensors(root / "fifo" / "w.safetensors", {{"w", "U8", {1}}});
    require(::mkfifo((root / "fifo" / "model.safetensors").c_str(), 0600) == 0, "mkfifo model.safetensors");
    require(Checkpoint::open(root / "fifo").source().filename() == "w.safetensors", "a FIFO named model.safetensors is skipped");
    require(::mkfifo((root / "fifo" / "model.safetensors.index.json").c_str(), 0600) == 0, "mkfifo index");
    require(Checkpoint::open(root / "fifo").source().filename() == "w.safetensors", "a FIFO named like the index is skipped");
#endif
    write_file(root / "model.bin", "x");
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root / "model.bin"); }, "expected a .safetensors file, a .safetensors.index.json index or a model directory");
}

void model_symlinks() {
    TempDir dir;
    const auto& root = dir.path;
    if(!symlinks_work(root)) { std::cout << "symbolic links unavailable; symlink cases skipped\n"; return; }
    // Hugging Face cache layout: snapshot entries are links into blobs.
    write_tensors(root / "blobs" / "1111", {{"a", "BF16", {2}}}, {}, 1);
    write_tensors(root / "blobs" / "2222", {{"b", "U8", {3}}}, {}, 2);
    write_file(root / "blobs" / "3333", R"({"weight_map":{"a":"one.safetensors","b":"two.safetensors"}})");
    fs::create_directories(root / "snapshot");
    fs::create_symlink(fs::path("..") / "blobs" / "1111", root / "snapshot" / "one.safetensors");
    fs::create_symlink(fs::path("..") / "blobs" / "2222", root / "snapshot" / "two.safetensors");
    fs::create_symlink(fs::path("..") / "blobs" / "3333", root / "snapshot" / "model.safetensors.index.json");
    const auto linked = Checkpoint::open(root / "snapshot");
    require(linked.tensors().size() == 2 && payload_of(root / "blobs" / "2222", linked.at("b")) == text_of(linked.map(linked.at("b")).bytes), "linked checkpoint");
    const auto rows = linked.digest_files();
    require(rows.size() == 3 && rows[0].path == "model.safetensors.index.json" && rows[1].path == "one.safetensors" &&
            rows[1].sha256 == reference_sha256(read_file(root / "blobs" / "1111")), "digest rows name the links and hash the targets");
    CheckpointOptions refuse;
    refuse.follow_symlinks = false;
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(root / "snapshot", refuse); }, "model.safetensors.index.json: open: symbolic link refused");
    // The audit's static case: a regular index naming a linked shard.
    write_tensors(root / "audit" / "actual.safetensors", {{"a", "U8", {1}}});
    fs::create_symlink("actual.safetensors", root / "audit" / "link.safetensors");
    const auto index = write_index(root / "audit", R"({"weight_map":{"a":"link.safetensors"}})");
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(index, refuse); }, "link.safetensors: open: symbolic link refused");
    require(Checkpoint::open(index).at("a").nbytes() == 1, "followed shard link");
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(root / "audit" / "link.safetensors", refuse); }, "symbolic link refused");
    rejects_as<std::system_error>([&] { (void)SafetensorsFile::open(root / "audit" / "link.safetensors", refuse); }, "symbolic link refused");
    CheckpointOptions extra = refuse;
    extra.extra_files = {"link.safetensors"};
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(root / "audit" / "actual.safetensors", extra); }, "symbolic link refused");
    // A dangling link is chosen by directory resolution and reported by the open.
    fs::create_directories(root / "dangling");
    fs::create_symlink("gone.safetensors", root / "dangling" / "model.safetensors");
    rejects_as<std::system_error>([&] { (void)Checkpoint::open(root / "dangling"); }, "model.safetensors: open");
    // Any link counts as a candidate, even one to a directory, which the open then refuses.
    fs::create_directories(root / "dirlink" / "real");
    write_tensors(root / "dirlink" / "w.safetensors", {{"w", "U8", {1}}});
    fs::create_directory_symlink("real", root / "dirlink" / "model.safetensors");
    rejects_as<std::invalid_argument>([&] { (void)Checkpoint::open(root / "dirlink"); }, "not a regular file");
}

void model_bytes() {
    TempDir dir;
    const std::vector<Spec> specs{{"big", "U8", {9000}}, {"mid", "F32", {1250}}, {"zero", "F16", {0, 7}}, {"odd", "U8", {3}},
                                  {"page", "BF16", {2048}}, {"tail", "I64", {5}}};
    const auto path = write_tensors(dir.path / "model.safetensors", specs, R"({"format":"pt"})", 11);
    const auto bytes = read_file(path);
    const auto digest = reference_sha256(bytes);
    std::optional<Checkpoint::View> survivor;
    {
        const auto checkpoint = Checkpoint::open(path);
        for(const auto& t:checkpoint.tensors()) {
            const auto expected = bytes.substr(std::size_t(t.begin), std::size_t(t.nbytes()));
            std::string copy(expected.size(), '\0');
            checkpoint.read(t, std::as_writable_bytes(std::span<char>(copy.data(), copy.size())));
            require(copy == expected, "read bytes of " + t.name);
            const auto view = checkpoint.map(t);
            require(text_of(view.bytes) == expected && (t.nbytes() == 0 || view.owner), "mapped bytes of " + t.name);
            std::string chunks;
            std::uint64_t next = t.begin;
            const auto stats = checkpoint.read_chunks(t, [&](std::span<const std::byte> part, std::uint64_t offset) {
                require(offset == next, "chunk offsets are absolute and contiguous");
                next += part.size();
                chunks += text_of(part);
            }, {.chunk_bytes = 4096, .alignment = 4096, .direct = false, .overlap = true});
            require(chunks == expected && stats.bytes == t.nbytes(), "chunked bytes of " + t.name);
        }
        survivor = checkpoint.map(checkpoint.at("big"));
        const auto& big = checkpoint.at("big");
        std::vector<std::byte> wrong(std::size_t(big.nbytes()) - 1);
        rejects_as<std::invalid_argument>([&] { checkpoint.read(big, wrong); }, "destination of 8999 bytes does not match tensor 'big' of 9000 bytes");
        TensorInfo foreign = big;
        foreign.begin += 1;
        rejects_as<std::invalid_argument>([&] { (void)checkpoint.map(foreign); }, "tensor 'big' does not belong to this checkpoint");
        foreign = big;
        foreign.name = "ghost";
        rejects_as<std::invalid_argument>([&] { checkpoint.read(foreign, wrong); }, "tensor 'ghost' does not belong to this checkpoint");
        foreign = big;
        foreign.shard = 3;
        rejects_as<std::invalid_argument>([&] { (void)checkpoint.read_chunks(foreign, [](std::span<const std::byte>, std::uint64_t) {}); }, "does not belong");
        // stream_shard over the options matrix: tensor bytes, offsets, digest and stats.
        for(const bool direct:{false, true}) for(const bool overlap:{false, true}) for(const std::size_t chunk:{std::size_t(4096), std::size_t(8192), std::size_t(1) << 20})
            for(const bool hash:{false, true}) {
                std::map<std::string, std::string> got;
                std::vector<std::string> order;
                std::string sha = "untouched";
                const auto stats = checkpoint.stream_shard(0, [&](const TensorInfo& t, std::span<const std::byte> part, std::uint64_t offset) {
                    require(!part.empty() && t.nbytes() > 0, "empty delivery");
                    require(offset == got[t.name].size(), "offsets within the tensor are contiguous");
                    if(order.empty() || order.back() != t.name) order.push_back(t.name);
                    got[t.name] += text_of(part);
                }, {.chunk_bytes = chunk, .alignment = 4096, .direct = direct, .overlap = overlap}, hash ? &sha : nullptr);
                for(const auto& t:checkpoint.tensors())
                    require(t.nbytes() == 0 ? !got.contains(t.name) : got[t.name] == bytes.substr(std::size_t(t.begin), std::size_t(t.nbytes())), "streamed bytes of " + t.name);
                require(order == std::vector<std::string>{"big", "mid", "odd", "page", "tail"}, "stream follows file order");
                const auto payload = checkpoint.shards()[0].payload_offset();
                require(hash ? sha == digest && stats.bytes == bytes.size() : sha == "untouched" && stats.bytes == bytes.size() - payload, "stream digest or stats");
            }
        rejects_as<std::out_of_range>([&] { (void)checkpoint.stream_shard(1, [](const TensorInfo&, std::span<const std::byte>, std::uint64_t) {}); }, "shard 1 out of range (1 shards)");
        rejects_as<std::invalid_argument>([&] { (void)checkpoint.stream_shard(0, {}); }, "tensor consumer is empty");
        rejects_as<std::invalid_argument>([&] {
            (void)checkpoint.stream_shard(0, [](const TensorInfo&, std::span<const std::byte>, std::uint64_t) {}, {.chunk_bytes = 1000, .alignment = 4096});
        }, "chunk_bytes must be a positive multiple");
        std::string sha = "untouched";
        rejects([&] {
            (void)checkpoint.stream_shard(0, [](const TensorInfo& t, std::span<const std::byte>, std::uint64_t) { if(t.name == "page") throw std::runtime_error("consumer failed"); },
                                          {.chunk_bytes = 4096, .alignment = 4096}, &sha);
        }, "consumer failed");
        require(sha == "untouched", "failed stream published a digest");
        const auto empty = checkpoint.map(checkpoint.at("zero"));
        require(empty.bytes.empty() && !empty.owner, "empty tensor view");
        checkpoint.read(checkpoint.at("zero"), {});
    }
    require(survivor && text_of(survivor->bytes) == bytes.substr(std::size_t(8 + SafetensorsFile::open(path).header_length()), 9000), "view outlives the checkpoint");
    survivor.reset();
    // A two-shard index: each shard streams and hashes to its own file digest.
    write_tensors(dir.path / "idx" / "a.safetensors", {{"x", "U8", {5000}}, {"y", "F16", {3}}}, {}, 21);
    write_tensors(dir.path / "idx" / "b.safetensors", {{"z", "U8", {70000}}}, {}, 22);
    const auto checkpoint = Checkpoint::open(write_index(dir.path / "idx", R"({"weight_map":{"x":"a.safetensors","y":"a.safetensors","z":"b.safetensors"}})"));
    for(std::uint32_t shard = 0; shard < 2; ++shard) {
        std::string sha, all;
        (void)checkpoint.stream_shard(shard, [&](const TensorInfo&, std::span<const std::byte> part, std::uint64_t) { all += text_of(part); },
                                      {.chunk_bytes = 8192, .alignment = 4096, .direct = true}, &sha);
        const auto file = read_file(checkpoint.shards()[shard].file().path());
        require(sha == reference_sha256(file) && all == file.substr(std::size_t(checkpoint.shards()[shard].payload_offset())), "shard stream digest");
    }
}

void model_digests() {
    TempDir dir;
    const auto& root = dir.path;
    write_tensors(root / "b.safetensors", {{"b", "U8", {300}}}, {}, 31);
    write_tensors(root / "a.safetensors", {{"a", "F32", {40}}}, {}, 32);
    write_tensors(root / "draft.safetensors", {{"mtp", "BF16", {6}}}, {}, 33);
    write_index(root, R"({"weight_map":{"a":"a.safetensors","b":"b.safetensors"}})");
    CheckpointOptions options;
    options.extra_files = {"draft.safetensors"};
    const auto checkpoint = Checkpoint::open(root, options);
    for(const auto& read:{ReadOptions{.direct = true}, ReadOptions{.chunk_bytes = 4096, .alignment = 4096, .direct = false, .overlap = false}}) {
        const auto rows = checkpoint.digest_files(read);
        std::vector<Checkpoint::FileDigest> expected;
        for(const char* name:{"a.safetensors", "b.safetensors", "draft.safetensors", "model.safetensors.index.json"}) {
            const auto content = read_file(root / name);
            expected.push_back({name, content.size(), reference_sha256(content)});
        }
        require(rows == expected, "digest rows equal the manifest rows of every checkpoint file");
    }
    auto rows = checkpoint.digest_files();
    checkpoint.verify_digests(rows);
    auto extended = rows;
    extended.push_back({"config.json", 2, std::string(64, '0')});
    std::reverse(extended.begin(), extended.end());
    checkpoint.verify_digests(extended);
    auto missing = rows;
    missing.erase(missing.begin() + 1);
    rejects_as<FormatError>([&] { checkpoint.verify_digests(missing); }, display_path(root / "b.safetensors") + ": digest: absent from the expected digests");
    auto sized = rows;
    sized[0].bytes += 1;
    rejects_as<FormatError>([&] { checkpoint.verify_digests(sized); }, "a.safetensors: digest: size " + std::to_string(rows[0].bytes) + " differs from the expected " + std::to_string(rows[0].bytes + 1) + " bytes");
    auto hashed = rows;
    hashed[2].sha256[5] = hashed[2].sha256[5] == '0' ? '1' : '0';
    rejects_as<FormatError>([&] { checkpoint.verify_digests(hashed); }, "draft.safetensors: digest: sha256 " + rows[2].sha256 + " differs from the expected " + hashed[2].sha256);
    // Sizes are all checked before any hashing: the size mismatch of a later file wins over a hash mismatch.
    hashed[3].bytes += 7;
    rejects_as<FormatError>([&] { checkpoint.verify_digests(hashed); }, "model.safetensors.index.json: digest: size");
    auto upper = rows;
    for(auto& c:upper[1].sha256) c = char(std::toupper(static_cast<unsigned char>(c)));
    rejects_as<std::invalid_argument>([&] { checkpoint.verify_digests(upper); }, "expected digest for 'b.safetensors' is not 64 lowercase hex digits");
    auto twice = rows;
    twice.push_back(rows[1]);
    rejects_as<std::invalid_argument>([&] { checkpoint.verify_digests(twice); }, "expected digests list 'b.safetensors' more than once");
    // Change detection: a touched shard fails verify_unchanged, streaming and hashing.
    checkpoint.verify_unchanged();
    const auto shard = root / "b.safetensors";
    fs::last_write_time(shard, fs::last_write_time(shard) + std::chrono::seconds(5));
    rejects_as<std::runtime_error>([&] { checkpoint.verify_unchanged(); }, "b.safetensors: file changed since open");
    rejects([&] { (void)checkpoint.digest_files(); }, "file changed since open");
    rejects([&] { (void)checkpoint.stream_shard(checkpoint.at("b").shard, [](const TensorInfo&, std::span<const std::byte>, std::uint64_t) {}); }, "file changed since open");
    // The whole file streams and hashes, then the final check fails: no digest is stored.
    std::string sha = "untouched";
    rejects_as<std::runtime_error>([&] {
        (void)checkpoint.stream_shard(checkpoint.at("b").shard, [](const TensorInfo&, std::span<const std::byte>, std::uint64_t) {}, {}, &sha);
    }, "b.safetensors: file changed since open");
    require(sha == "untouched", "a stream of a changed file stored its digest");
    // A change during the stream (the consumer moves the file's time) is caught the same way.
    const auto fresh = Checkpoint::open(root, options);
    std::size_t calls = 0;
    rejects_as<std::runtime_error>([&] {
        (void)fresh.stream_shard(fresh.at("b").shard, [&](const TensorInfo&, std::span<const std::byte>, std::uint64_t) {
            if(calls++ == 0) fs::last_write_time(shard, fs::last_write_time(shard) + std::chrono::seconds(5));
        }, {.chunk_bytes = 4096, .alignment = 4096}, &sha);
    }, "b.safetensors: file changed since open");
    require(calls > 0 && sha == "untouched", "a stream of a file changed mid-stream stored its digest");
    const auto index = root / "model.safetensors.index.json";
    const auto reopened = Checkpoint::open(root);
    fs::last_write_time(index, fs::last_write_time(index) + std::chrono::seconds(5));
    rejects_as<std::runtime_error>([&] { reopened.verify_unchanged(); }, "model.safetensors.index.json: file changed since open");
}

#ifdef __linux__
// Bytes this process read through the read(2) family since begin(), from /proc/self/io rchar. Reading that file
// adds its own length after the snapshot, so since() subtracts the length begin() read. nullopt when unavailable.
struct ReadCounter {
    std::optional<std::uint64_t> start;
    std::size_t text_bytes = 0;
    static std::optional<std::uint64_t> sample(std::size_t& text_bytes) {
        std::ifstream in("/proc/self/io", std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        text_bytes = text.size();
        const auto at = text.find("rchar: ");
        if(at == std::string::npos) return std::nullopt;
        return std::stoull(text.substr(at + 7));
    }
    void begin() { start = sample(text_bytes); }
    std::optional<std::uint64_t> since() const {
        std::size_t ignored = 0;
        const auto now = sample(ignored);
        if(!start || !now) return std::nullopt;
        return *now - *start - text_bytes;
    }
};
#endif

// Opens a file over and over while another thread keeps moving its modification time to new values. Every open
// must succeed or report "file changed since open", and at least one must report it: only the check that follows
// the header (or index) read can see a change made while that read was running.
void detects_concurrent_change(const fs::path& path, const std::function<void()>& open, const std::string& label) {
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> touches{0};
    const auto base = fs::last_write_time(path);
    std::thread toucher([&] {
        std::error_code error;
        for(std::int64_t i = 1; !stop.load(); ++i) {
            fs::last_write_time(path, base + std::chrono::milliseconds(i), error);
            if(!error) touches.fetch_add(1);
        }
    });
    struct Join {
        std::atomic<bool>& stop;
        std::thread& thread;
        ~Join() { stop = true; thread.join(); }
    } join{stop, toucher};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(40);
    while(touches.load() == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    require(touches.load() > 0, label + ": the modification time could not be changed");
    std::size_t opened = 0, detected = 0;
    while((detected == 0 || opened + detected < 8) && std::chrono::steady_clock::now() < deadline) {
        try {
            open();
            ++opened;
        } catch(const std::runtime_error& e) {
            require(std::string_view(e.what()).find("file changed since open") != std::string_view::npos, label + ": unexpected failure: " + e.what());
            ++detected;
        }
    }
    require(detected > 0, label + ": no change detected in " + std::to_string(opened) + " opens");
}

// The audit's metadata-read case: opening reads the 8-byte prefix and the header and never the payload, a
// header that cannot be retained is not read, and a file changed while its header or the index is read is refused.
void model_change_detection() {
    TempDir dir;
    const auto& root = dir.path;
#ifdef __linux__
    const auto big = write_tensors(root / "big.safetensors", {{"head", "U8", {7}}, {"payload", "U8", {std::uint64_t(1) << 20}}}, R"({"format":"pt"})", 51);
    write_tensors(root / "small.safetensors", {{"tail", "F32", {3}}}, {}, 52);
    const auto index = write_index(root, R"({"weight_map":{"head":"big.safetensors","payload":"big.safetensors","tail":"small.safetensors"}})");
    ReadCounter counter;
    counter.begin();
    if(counter.since() != std::optional<std::uint64_t>(0)) {
        std::cout << "/proc/self/io read accounting unavailable; read-count checks skipped\n";
    } else {
        counter.begin();
        const auto file = SafetensorsFile::open(big);
        const auto header_read = counter.since();
        require(header_read == 8 + file.header_length(), "header open read " + std::to_string(header_read.value_or(0)) + " bytes, not the prefix and header");
        counter.begin();
        const auto checkpoint = Checkpoint::open(index);
        const auto index_read = counter.since();
        std::uint64_t expected = fs::file_size(index);
        for(const auto& shard:checkpoint.shards()) expected += 8 + shard.header_length();
        require(index_read == expected, "checkpoint open read " + std::to_string(index_read.value_or(0)) + " bytes, not " + std::to_string(expected));
        CheckpointOptions tight;
        tight.limits.retained_bytes = file.header_length() - 1;
        counter.begin();
        rejects_as<FormatError>([&] { (void)SafetensorsFile::open(big, tight); }, "retained_bytes limit");
        require(counter.since() == std::optional<std::uint64_t>(8), "a header beyond the retained limit was read");
    }
#endif
    // A 4 MiB space-padded header and index keep the read window wide while parsing stays cheap.
    const std::string padding(std::size_t(4) << 20, ' ');
    const auto padded = write_raw(root / "padded.safetensors", R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})" + padding, "ab");
    detects_concurrent_change(padded, [&] { (void)SafetensorsFile::open(padded); }, "header read");
    write_file(root / "slow" / "model.safetensors.index.json", R"({"weight_map":{"a":"shard.safetensors"}})" + padding);
    write_tensors(root / "slow" / "shard.safetensors", {{"a", "U8", {2}}});
    const auto slow = root / "slow" / "model.safetensors.index.json";
    detects_concurrent_change(slow, [&] { (void)Checkpoint::open(slow); }, "index read");
}

// One record per tensor: shards hold slices of Checkpoint::tensors(), and retained_bytes() charges exactly the
// documented formula, which retained_bytes limits enforce at the byte.
std::uint64_t expected_retained(const SafetensorsFile& file) {
    std::uint64_t bytes = file.header_length();
    for(const auto& [key, value]:file.metadata()) bytes += key.size() + value.size();
    for(const auto& t:file.tensors()) bytes += sizeof(TensorInfo) + t.name.size() + 8 * t.shape.size();
    return bytes;
}
void model_retained() {
    TempDir dir;
    const auto& root = dir.path;
    write_tensors(root / "b.safetensors", {{"zeta", "U8", {3}}, {"alpha.with.a.long.tensor.name.weight", "F32", {2, 2}}}, R"({"format":"pt","key":"value"})", 41);
    write_tensors(root / "a.safetensors", {{"mid", "BF16", {4}}, {"beta", "I64", {1}}, {"omega", "U8", {0, 3, 1}}}, {}, 42);
    write_tensors(root / "draft.safetensors", {{"draft.w", "F16", {5}}}, {}, 43);
    const auto index = write_index(root, R"({"weight_map":{"zeta":"b.safetensors","alpha.with.a.long.tensor.name.weight":"b.safetensors",)"
                                         R"("mid":"a.safetensors","beta":"a.safetensors","omega":"a.safetensors"}})");
    CheckpointOptions options;
    options.extra_files = {"draft.safetensors"};
    auto checkpoint = Checkpoint::open(root, options);
    const std::vector<std::string> order{"mid", "beta", "omega", "zeta", "alpha.with.a.long.tensor.name.weight", "draft.w"};
    const auto all = checkpoint.tensors();
    require(all.size() == order.size(), "one record per tensor");
    for(std::size_t i = 0; i < order.size(); ++i) require(all[i].name == order[i], "shard-major header order at " + std::to_string(i) + ": " + all[i].name);
    std::size_t offset = 0;
    for(std::uint32_t i = 0; i < checkpoint.shards().size(); ++i) {
        const auto slice = checkpoint.shards()[i].tensors();
        require(slice.data() == all.data() + offset, "shard " + std::to_string(i) + " is a slice of the checkpoint records, not a copy");
        for(const auto& t:slice) require(t.shard == i, "slice records carry their shard");
        offset += slice.size();
    }
    require(offset == all.size(), "slices cover every record");
    for(const auto& t:all) require(checkpoint.find(t.name) == &t && &checkpoint.at(t.name) == &t, "lookup returns the single record of " + t.name);
    for(const char* absent:{"", "a", "alpha", "zz", "mid.", "betb"}) require(!checkpoint.find(absent), std::string("found absent tensor ") + absent);
    // Moving the checkpoint keeps the records, the slices and the byte access.
    const auto* records = all.data();
    const auto moved = std::move(checkpoint);
    require(moved.tensors().data() == records && moved.shards()[1].tensors().data() == records + 3 && moved.at("zeta").shard == 1, "moved checkpoint");
    require(text_of(moved.map(moved.at("zeta")).bytes) == payload_of(root / "b.safetensors", moved.at("zeta")), "moved checkpoint bytes");
    // Retained bytes: every file's header, metadata text and records; 4 more per tensor for the name index.
    std::uint64_t files = 0;
    for(const auto& shard:moved.shards()) {
        require(shard.retained_bytes() == expected_retained(shard), "retained bytes of " + display_path(shard.file().path()));
        files += shard.retained_bytes();
    }
    require(moved.retained_bytes() == files + 4 * order.size(), "checkpoint retained bytes");
    CheckpointOptions limited = options;
    limited.limits.retained_bytes = moved.retained_bytes();
    require(Checkpoint::open(root, limited).retained_bytes() == moved.retained_bytes(), "retained limit at the exact charge");
    limited.limits.retained_bytes -= 1;
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root, limited); },
                            display_path(index) + ": size: checkpoint headers and tensor records exceed the " + std::to_string(limited.limits.retained_bytes) + "-byte retained_bytes limit");
    // A budget that runs out inside a later file names that file.
    limited.limits.retained_bytes = moved.shards()[0].retained_bytes() + moved.shards()[1].header_length();
    rejects_as<FormatError>([&] { (void)Checkpoint::open(root, limited); }, display_path(root / "b.safetensors") + ": size: checkpoint headers and tensor records exceed");
    // A standalone file charges the same formula without the index.
    auto single = SafetensorsFile::open(root / "b.safetensors");
    const auto charge = expected_retained(single);
    require(single.retained_bytes() == charge && single.tensors().size() == 2, "standalone retained bytes");
    for(const std::uint64_t limit:{charge, charge - 1, single.header_length() + 13, single.header_length(), single.header_length() - 1}) {
        CheckpointOptions bounded;
        bounded.limits.retained_bytes = limit;
        if(limit >= charge) require(SafetensorsFile::open(root / "b.safetensors", bounded).retained_bytes() == charge, "standalone limit at the charge");
        else rejects_as<FormatError>([&] { (void)SafetensorsFile::open(root / "b.safetensors", bounded); },
                                     "size: header and tensor records exceed the " + std::to_string(limit) + "-byte retained_bytes limit");
    }
    // Moving a standalone file keeps its records; the moved-from file is empty.
    const auto* own = single.tensors().data();
    auto taken = std::move(single);
    require(taken.tensors().data() == own && taken.tensors()[1].name == "alpha.with.a.long.tensor.name.weight" && taken.retained_bytes() == charge, "moved file");
    require(single.tensors().empty() && single.retained_bytes() == 0, "moved-from file");
    auto assigned = SafetensorsFile::open(root / "a.safetensors");
    assigned = std::move(taken);
    require(assigned.tensors().data() == own && assigned.tensors()[0].name == "zeta" && assigned.metadata().size() == 2, "move-assigned file");
}

// Randomized round trips: random records in random member and payload orders must parse to exactly what was
// written; every strict truncation and every appended tail must be refused (header_only accepts both).
void model_random() {
    std::mt19937 random(20261015);
    const std::vector<std::string> dtypes{"BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64", "F8_E4M3", "F8_E5M2",
                                          "F8_E8M0", "F16", "BF16", "F32", "F64", "F4", "F6_E2M3", "F6_E3M2"};
    TempDir dir;
    const auto path = dir.path / "model.safetensors";
    for(int round = 0; round < 60; ++round) {
        const std::size_t count = random() % 9;
        std::vector<Spec> specs;
        for(std::size_t i = 0; i < count; ++i) {
            Spec spec{"t" + std::to_string(i) + "." + std::to_string(random() % 1000), dtypes[random() % dtypes.size()], {}};
            const std::size_t rank = random() % 4;
            for(std::size_t r = 0; r < rank; ++r) spec.shape.push_back(random() % 9 == 0 ? 0 : 1 + random() % 6);
            if(widths.at(spec.dtype) < 8) spec.shape.push_back(8);   // whole bytes for sub-byte dtypes
            specs.push_back(spec);
        }
        std::vector<std::size_t> layout(count), members(count);
        for(std::size_t i = 0; i < count; ++i) layout[i] = members[i] = i;
        std::shuffle(layout.begin(), layout.end(), random);
        std::shuffle(members.begin(), members.end(), random);
        std::vector<std::uint64_t> begin(count);
        std::uint64_t offset = 0;
        for(const auto i:layout) { begin[i] = offset; offset += bytes_of(specs[i]); }
        std::string header = "{";
        const bool metadata = random() % 2;
        for(std::size_t m = 0; m < count; ++m) {
            if(metadata && m == count / 2) header += std::string(header.size() > 1 ? "," : "") + R"("__metadata__":{"round":")" + std::to_string(round) + "\"}";
            const auto& spec = specs[members[m]];
            header += std::string(header.size() > 1 ? "," : "") + "\"" + spec.name + "\":{\"shape\":" + shape_json(spec.shape) + ",\"data_offsets\":[" +
                      std::to_string(begin[members[m]]) + "," + std::to_string(begin[members[m]] + bytes_of(spec)) + "],\"dtype\":\"" + spec.dtype + "\"}";
        }
        if(metadata && count == 0) header += R"("__metadata__":{"round":"empty"})";
        header += "}" + std::string(random() % 5, ' ');
        const auto payload = pattern(std::size_t(offset), unsigned(round));
        const auto file = container(header, payload);
        write_file(path, file);
        {   // closed before the file is rewritten: Windows refuses writers while a handle holds it
            const auto checkpoint = Checkpoint::open(path);
            const auto& shard = checkpoint.shards()[0];
            require(shard.tensors().size() == count && shard.metadata().size() == (metadata ? 1u : 0u), "random record count");
            for(std::size_t m = 0; m < count; ++m) {
                const auto& t = shard.tensors()[m];
                const auto& spec = specs[members[m]];
                require(t.name == spec.name && dtype_name(t.dtype) == spec.dtype && t.shape == spec.shape, "random record fields");
                require(t.begin == 8 + header.size() + begin[members[m]] && t.nbytes() == bytes_of(spec), "random record extent");
                require(text_of(checkpoint.map(t).bytes) == file.substr(std::size_t(t.begin), std::size_t(t.nbytes())), "random record bytes");
            }
            std::string streamed;
            std::string sha;
            const std::size_t chunk = std::size_t(4096) << (random() % 3);
            (void)checkpoint.stream_shard(0, [&](const TensorInfo&, std::span<const std::byte> part, std::uint64_t) { streamed += text_of(part); },
                                          {.chunk_bytes = chunk, .alignment = 4096, .direct = random() % 2 == 0, .overlap = random() % 2 == 0}, &sha);
            require(streamed == payload && sha == reference_sha256(file), "random stream");
        }
        for(int cut = 0; cut < 6; ++cut) {
            const std::size_t length = random() % file.size();
            write_file(path, std::string_view(file).substr(0, length));
            rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, length < 8 ? "truncated header length" : length < 8 + header.size() ? "header length" : "past the");
            if(length >= 8 + header.size()) require(Checkpoint::open(path, header_only()).tensors().size() == count, "header_only accepts a truncated payload");
        }
        write_file(path, file + pattern(1 + random() % 16, unsigned(round) + 1000));
        rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, "unindexed trailing bytes");
        require(Checkpoint::open(path, header_only()).tensors().size() == count, "header_only accepts trailing bytes");
        if(count >= 2) {   // moving one extent by a byte always opens a hole or an overlap
            const auto victim = members[random() % count];
            if(bytes_of(specs[victim]) == 0) continue;
            std::string shifted = header;
            const auto old_range = "[" + std::to_string(begin[victim]) + "," + std::to_string(begin[victim] + bytes_of(specs[victim])) + "]";
            const auto at = shifted.find("\"" + specs[victim].name + "\"");
            const auto pos = shifted.find(old_range, at);
            const bool up = begin[victim] == 0 || random() % 2;
            const auto moved = up ? begin[victim] + 1 : begin[victim] - 1;
            shifted.replace(pos, old_range.size(), "[" + std::to_string(moved) + "," + std::to_string(moved + bytes_of(specs[victim])) + "]");
            write_file(path, container(shifted, payload + std::string(1, '\0')));
            rejects_as<FormatError>([&] { (void)Checkpoint::open(path); }, "payload");
        }
    }
}

// Model config fixtures. Expected values come from crit eval's ceiling.attention_shape, context.model_limit
// and prepare.stop_ids on the same inputs (tests/test_eval.py and probes of those functions).
ModelConfig config_of(const fs::path& directory, std::string_view json) {
    write_file(directory / "config.json", json);
    return load_model_config(directory);
}
std::size_t full_layers(const std::vector<LayerKind>& kinds) { return std::size_t(std::count(kinds.begin(), kinds.end(), LayerKind::full_attention)); }

void model_config_eval() {
    TempDir dir;
    const auto& root = dir.path;
    const std::vector<std::tuple<std::string, std::optional<std::uint64_t>, std::string>> limits{
        {R"({"max_position_embeddings":262144})", 262144, "max_position_embeddings"},
        {R"({"max_position_embeddings":32768,"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}})", 131072,
         "yarn rope scaling: 32768 original positions x factor 4"},
        {"{}", std::nullopt, ""},
        {R"({"max_position_embeddings":4096,"rope_scaling":{"type":"yarn","factor":3,"original_max_position_embeddings":4097}})", 12291,
         "yarn rope scaling: 4097 original positions x factor 3"},
        {R"({"max_position_embeddings":4096,"rope_scaling":{"rope_type":"yarn","factor":2.5,"original_max_position_embeddings":4097}})", 10242,
         "yarn rope scaling: 4097 original positions x factor 2.5"},
        {R"({"max_position_embeddings":4096,"rope_scaling":{"rope_type":"yarn","factor":0,"original_max_position_embeddings":4097}})", 4096, "max_position_embeddings"},
        {R"({"seq_length":2048,"rope_scaling":{},"rope_parameters":{"rope_type":"yarn","factor":1.5,"original_max_position_embeddings":1000}})", 1500,
         "yarn rope scaling: 1000 original positions x factor 1.5"},
        {R"({"seq_length":2048,"rope_scaling":{"rope_type":"linear","factor":2.0},"rope_parameters":{"rope_type":"yarn","factor":8,"original_max_position_embeddings":1000}})",
         2048, "seq_length"},
        {R"({"max_position_embeddings":0,"n_positions":true,"max_seq_len":4096.0,"seq_length":"x","max_sequence_length":-5,"model_max_length":77})", 77, "model_max_length"},
        {R"({"rope_scaling":{"rope_type":"yarn","factor":0.000012345678,"original_max_position_embeddings":1000000000}})", 12345,
         "yarn rope scaling: 1000000000 original positions x factor 1.23457e-05"},
        {R"({"n_positions":10,"rope_scaling":{"rope_type":"","type":"yarn","factor":1000.0,"original_max_position_embeddings":3}})", 3000,
         "yarn rope scaling: 3 original positions x factor 1000"},
        {R"({"max_position_embeddings":4096,"text_config":{"rope_scaling":{"rope_type":"yarn","factor":2,"original_max_position_embeddings":8192}}})", 16384,
         "yarn rope scaling: 8192 original positions x factor 2"},
        {R"({"max_position_embeddings":4096,"rope_scaling":{"rope_type":"yarn","factor":2,"original_max_position_embeddings":8192},"text_config":{"rope_scaling":null}})",
         4096, "max_position_embeddings"},
        {R"({"max_position_embeddings":18446744073709551615})", std::numeric_limits<std::uint64_t>::max(), "max_position_embeddings"},
        // Python multiplies an int factor exactly; through a double this would be 18014398509481984.
        {R"({"rope_scaling":{"rope_type":"yarn","factor":2,"original_max_position_embeddings":9007199254740993}})", 18014398509481986u,
         "yarn rope scaling: 9007199254740993 original positions x factor 2"},
    };
    for(const auto& [json, value, source]:limits) {
        const auto config = config_of(root, json);
        std::string where = "unset";
        require(config.position_limit(&where) == value && where == source, "position limit of " + json + ": " + where);
        require(config.position_limit() == value, "position limit without a source");
    }
    rejects_as<FormatError>([&] {
        (void)config_of(root, R"({"rope_scaling":{"rope_type":"yarn","factor":1e20,"original_max_position_embeddings":3}})").position_limit();
    }, "key 'rope_scaling.factor': yarn position limit 3 x 1e+20 exceeds 2^64 - 1");
    // Typed rope fields are strict where eval would silently fall back to max_position_embeddings.
    rejects_as<FormatError>([&] {
        (void)config_of(root, R"({"max_position_embeddings":4096,"rope_scaling":{"rope_type":"yarn","factor":2,"original_max_position_embeddings":-3}})");
    }, "key 'rope_scaling.original_max_position_embeddings': expected an integer in [0, 18446744073709551615], got -3");
    rejects_as<FormatError>([&] {
        (void)config_of(root, R"({"rope_parameters":{"rope_type":"yarn","factor":8,"original_max_position_embeddings":4611686018427387904}})").position_limit();
    }, "key 'rope_parameters.factor': yarn position limit 4611686018427387904 x 8 exceeds 2^64 - 1");
    // ceiling.attention_shape fixtures: (full layers, heads, head dim, KV heads).
    auto ceiling = config_of(root, R"({"text_config":{"num_hidden_layers":4,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],)"
                                   R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"linear_num_value_heads":2,"linear_key_head_dim":4,"linear_value_head_dim":4}})");
    auto kinds = ceiling.layer_kinds();
    require(kinds == std::vector<LayerKind>{LayerKind::linear_attention, LayerKind::linear_attention, LayerKind::linear_attention, LayerKind::full_attention}, "ceiling layer kinds");
    require(full_layers(kinds) == 1 && ceiling.num_attention_heads == 2u && ceiling.effective_head_dim() == 8 && ceiling.effective_kv_heads() == 1, "ceiling shape");
    require(ceiling.linear && ceiling.linear->num_value_heads == 2u && ceiling.linear->key_head_dim == 4u && !ceiling.linear->num_key_heads, "linear attention sizes");
    const std::string dense = R"("num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"max_position_embeddings":32768)";
    auto config = config_of(root, "{" + dense + "}");
    require(full_layers(config.layer_kinds()) == 2 && config.effective_head_dim() == 8 && config.effective_kv_heads() == 1 && config.position_limit() == 32768u, "dense shape");
    require(!config.tie_word_embeddings, "absent tie stays absent");
    config = config_of(root, "{" + dense + R"(,"tie_word_embeddings":true})");
    require(config.tie_word_embeddings == true, "tied output head");
    config = config_of(root, R"({"num_hidden_layers":5,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","sliding_attention"],)"
                             R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"sliding_window":128,"linear_num_value_heads":2,"linear_key_head_dim":4,"linear_value_head_dim":4})");
    kinds = config.layer_kinds();
    require(full_layers(kinds) == 1 && kinds[4] == LayerKind::sliding_attention && config.sliding_window == 128u, "sliding and linear layers");
    config = config_of(root, R"({"num_hidden_layers":4,"kv_lora_rank":512,"qk_rope_head_dim":64})");
    require(config.mla && config.mla->kv_lora_rank == 512 && config.mla->qk_rope_head_dim == 64 && config.mla->v_head_dim == 0 && !config.mla->q_lora_rank, "latent attention");
    require(full_layers(config.layer_kinds()) == 4, "latent layers");
    rejects_as<FormatError>([&] { (void)config.effective_head_dim(); }, "key 'num_attention_heads': missing (no head_dim)");
    config = config_of(root, R"({"num_hidden_layers":10,"full_attention_interval":4,"num_attention_heads":16,"hidden_size":2048})");
    kinds = config.layer_kinds();
    require(full_layers(kinds) == 2 && kinds[3] == LayerKind::full_attention && kinds[7] == LayerKind::full_attention && kinds[0] == LayerKind::linear_attention &&
            std::count(kinds.begin(), kinds.end(), LayerKind::linear_attention) == 8, "interval layers");
    require(config.effective_head_dim() == 128 && config.effective_kv_heads() == 16, "derived head dim and KV heads");
    config = config_of(root, R"({"num_hidden_layers":3,"layer_types":[],"full_attention_interval":0})");
    require(full_layers(config.layer_kinds()) == 3, "empty layer_types and zero interval mean all full");
    config = config_of(root, R"({"num_hidden_layers":3,"layer_types":["attention","chunked_attention","mamba"]})");
    require(config.layer_kinds() == std::vector<LayerKind>{LayerKind::full_attention, LayerKind::other, LayerKind::other}, "attention alias and unknown kinds");
    // Qwen3-Next-80B-A3B: transformers derives 36 linear_attention and 12 full_attention layer types.
    config = config_of(root, R"({"model_type":"qwen3_next","num_hidden_layers":48,"full_attention_interval":4,"hidden_size":2048,"num_attention_heads":16,)"
                             R"("num_key_value_heads":2,"head_dim":256,"max_position_embeddings":262144,"linear_num_key_heads":16,"linear_num_value_heads":32,)"
                             R"("linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4})");
    kinds = config.layer_kinds();
    require(kinds.size() == 48 && full_layers(kinds) == 12 && std::count(kinds.begin(), kinds.end(), LayerKind::linear_attention) == 36 &&
            kinds[3] == LayerKind::full_attention && kinds[47] == LayerKind::full_attention && kinds[46] == LayerKind::linear_attention, "Qwen3-Next layer kinds");
    // Gemma 3n gives intermediate_size per layer: it fills intermediate_sizes and leaves intermediate_size absent.
    std::string sizes;
    for(int i = 0; i < 35; ++i) sizes += (i ? "," : "") + std::string("16384");
    config = config_of(root, R"({"architectures":["Gemma3nForConditionalGeneration"],"model_type":"gemma3n","text_config":{"model_type":"gemma3n_text",)"
                             R"("hidden_size":2048,"intermediate_size":[)" + sizes + R"(],"num_hidden_layers":35,"num_attention_heads":8,"num_key_value_heads":2,)"
                             R"("head_dim":256,"vocab_size":262400}})");
    require(!config.intermediate_size && config.intermediate_sizes == std::vector<std::uint32_t>(35, 16384) && config.hidden_size == 2048u, "Gemma 3n per-layer sizes");
    config = config_of(root, R"({"intermediate_size":11008})");
    require(config.intermediate_size == 11008u && config.intermediate_sizes.empty(), "integer intermediate size");
    config = config_of(root, R"({"intermediate_size":[]})");
    require(!config.intermediate_size && config.intermediate_sizes.empty(), "empty per-layer sizes");
    require(!config_of(root, R"({"intermediate_size":null})").intermediate_size, "null intermediate size");
}

void model_config_types() {
    TempDir dir;
    const auto& root = dir.path;
    // Strict types: booleans never count as integers, reals are not integers, negative or oversized values fail.
    for(const auto& [json, message]:std::vector<std::pair<std::string, std::string>>{
            {R"({"hidden_size":true})", "key 'hidden_size': expected an integer in [0, 4294967295], got true"},
            {R"({"hidden_size":4096.0})", "key 'hidden_size': expected an integer in [0, 4294967295], got 4096"},
            {R"({"vocab_size":-1})", "key 'vocab_size': expected an integer in [0, 4294967295], got -1"},
            {R"({"num_hidden_layers":4294967296})", "key 'num_hidden_layers': expected an integer in [0, 4294967295], got 4294967296"},
            {R"({"head_dim":"128"})", "key 'head_dim': expected an integer in [0, 4294967295], got '128'"},
            {R"({"rms_norm_eps":"1e-6"})", "key 'rms_norm_eps': expected a finite number, got '1e-6'"},
            {R"({"layer_norm_eps":false})", "key 'layer_norm_eps': expected a finite number, got false"},
            {R"({"tie_word_embeddings":1})", "key 'tie_word_embeddings': expected true or false, got 1"},
            {R"({"architectures":"LlamaForCausalLM"})", "key 'architectures': expected an array, got 'LlamaForCausalLM'"},
            {R"({"architectures":["A",1]})", "key 'architectures': entry 1 is 1, not a string"},
            {R"({"layer_types":["full_attention",null]})", "key 'layer_types': entry 1 is null, not a string"},
            {R"({"model_type":7})", "key 'model_type': expected a string, got 7"},
            {R"({"torch_dtype":["bfloat16"]})", "key 'torch_dtype': expected a string, got an array"},
            {R"({"text_config":[]})", "key 'text_config': expected an object, got an array"},
            {R"({"rope_scaling":"yarn"})", "key 'rope_scaling': expected an object, got 'yarn'"},
            {R"({"rope_parameters":false})", "key 'rope_parameters': expected an object, got false"},
            {R"({"rope_scaling":{"factor":true}})", "key 'rope_scaling.factor': expected a finite number, got true"},
            {R"({"rope_scaling":{"rope_type":5}})", "key 'rope_scaling.rope_type': expected a string, got 5"},
            {R"({"rope_scaling":{"original_max_position_embeddings":true}})", "key 'rope_scaling.original_max_position_embeddings': expected an integer"},
            {R"({"rope_scaling":{"short_factor":[1.0,"x"]}})", "key 'rope_scaling.short_factor[1]': expected a finite number, got 'x'"},
            {R"({"rope_scaling":{"mrope_section":[16,-1]}})", "key 'rope_scaling.mrope_section[1]': expected an integer in [0, 4294967295], got -1"},
            {R"({"rope_theta":"10000"})", "key 'rope_theta': expected a finite number, got '10000'"},
            {R"({"rope_parameters":{"rope_theta":null},"rope_scaling":{"rope_theta":"x","rope_type":"default"}})", "key 'rope_scaling.rope_theta': expected a finite number"},
            {R"({"num_experts":true})", "key 'num_experts': expected an integer in [0, 4294967295], got true"},
            {R"({"n_routed_experts":8,"num_experts_per_tok":2.5})", "key 'num_experts_per_tok': expected an integer"},
            {R"({"num_local_experts":8,"norm_topk_prob":"yes"})", "key 'norm_topk_prob': expected true or false, got 'yes'"},
            {R"({"kv_lora_rank":512,"q_lora_rank":-2})", "key 'q_lora_rank': expected an integer"},
            {R"({"linear_conv_kernel_dim":1.5})", "key 'linear_conv_kernel_dim': expected an integer"},
            {R"({"eos_token_id":"2"})", "key 'eos_token_id': expected an integer in [-2147483648, 2147483647], got '2'"},
            {R"({"eos_token_id":[1,true]})", "key 'eos_token_id[1]': expected an integer in [-2147483648, 2147483647], got true"},
            {R"({"eos_token_id":[1,null]})", "key 'eos_token_id[1]': expected an integer, got null"},
            {R"({"eos_token_id":2147483648})", "key 'eos_token_id': expected an integer in [-2147483648, 2147483647], got 2147483648"},
            {R"({"bos_token_id":[1]})", "key 'bos_token_id': expected an integer in [-2147483648, 2147483647], got an array"},
            {R"({"quantization_config":"fp8"})", "key 'quantization_config': expected an object, got 'fp8'"},
            {R"({"quantization_config":{"quant_method":1}})", "key 'quantization_config.quant_method': expected a string, got 1"},
            {R"({"text_config":{"hidden_size":-3}})", "key 'hidden_size': expected an integer in [0, 4294967295], got -3"},
            {R"([1,2])", "root: the config must be a JSON object, not an array"},
            {R"({"a":1,"a":2})", "duplicate key 'a'"},
            {R"({"intermediate_size":"4096"})", "key 'intermediate_size': expected an integer or an array of integers, got '4096'"},
            {R"({"intermediate_size":true})", "key 'intermediate_size': expected an integer or an array of integers, got true"},
            {R"({"intermediate_size":{"0":1}})", "key 'intermediate_size': expected an integer or an array of integers, got an object"},
            {R"({"intermediate_size":4294967296})", "key 'intermediate_size': expected an integer in [0, 4294967295], got 4294967296"},
            {R"({"intermediate_size":[1,-2]})", "key 'intermediate_size[1]': expected an integer in [0, 4294967295], got -2"},
            {R"({"intermediate_size":[1,null]})", "key 'intermediate_size[1]': expected an integer in [0, 4294967295], got null"},
            {R"({"intermediate_size":[1,2.5]})", "key 'intermediate_size[1]': expected an integer in [0, 4294967295], got 2.5"},
            // NaN and Infinity load as values, and every typed real field refuses them.
            {R"({"rope_theta":NaN})", "key 'rope_theta': expected a finite number, got nan"},
            {R"({"rms_norm_eps":Infinity})", "key 'rms_norm_eps': expected a finite number, got inf"},
            {R"({"rope_scaling":{"factor":-Infinity}})", "key 'rope_scaling.factor': expected a finite number, got -inf"},
            {R"({"rope_scaling":{"long_factor":[1.0,NaN]}})", "key 'rope_scaling.long_factor[1]': expected a finite number, got nan"},
            {R"({"n_routed_experts":8,"routed_scaling_factor":Infinity})", "key 'routed_scaling_factor': expected a finite number, got inf"},
            {R"({"hidden_size":Infinity})", "key 'hidden_size': expected an integer in [0, 4294967295], got inf"},
            {R"({"rope_theta":1e400})", "real number out of range"},
            // Integers outside [-2^63, 2^64) load as values; every typed field refuses them the same way.
            {R"({"vocab_size":18446744073709551616})", "key 'vocab_size': integer exceeds 64 bits"},
            {R"({"hidden_size":-9223372036854775809})", "key 'hidden_size': integer exceeds 64 bits"},
            {R"({"intermediate_size":[1,123456789012345678901]})", "key 'intermediate_size[1]': integer exceeds 64 bits"},
            {R"({"eos_token_id":[1,99999999999999999999]})", "key 'eos_token_id[1]': integer exceeds 64 bits"},
            {R"({"pad_token_id":-99999999999999999999})", "key 'pad_token_id': integer exceeds 64 bits"},
            {R"({"rope_theta":100000000000000000000})", "key 'rope_theta': integer exceeds 64 bits"},
            {R"({"rope_scaling":{"original_max_position_embeddings":18446744073709551616}})", "key 'rope_scaling.original_max_position_embeddings': integer exceeds 64 bits"},
            {R"({"rope_scaling":{"rope_type":"yarn","factor":18446744073709551616,"original_max_position_embeddings":4}})", "key 'rope_scaling.factor': integer exceeds 64 bits"},
            {R"({"model_max_length":1000000000000000019884624838656})", "key 'model_max_length': integer exceeds 64 bits"},
            {R"({"max_position_embeddings":-18446744073709551616})", "key 'max_position_embeddings': integer exceeds 64 bits"},
            {R"({"max_position_embeddings":0,"seq_length":18446744073709551616,"model_max_length":5})", "key 'seq_length': integer exceeds 64 bits"},
            {R"({"architectures":[123456789012345678901]})", "key 'architectures': entry 0 is a very large integer, not a string"},
            {R"({"text_config":{"num_hidden_layers":18446744073709551616}})", "key 'num_hidden_layers': integer exceeds 64 bits"}}) {
        rejects_as<FormatError>([&] { (void)config_of(root, json); }, message);
    }
    // Untyped keys keep such values: Mamba2's time_step_limit with Infinity, and integers beyond 64 bits.
    {
        const auto mamba = config_of(root, R"({"model_type":"mamba2","num_hidden_layers":2,"eos_token_id":1,"time_step_limit":[0.0,Infinity],"x":NaN,)"
                                           R"("big":123456789012345678901,"nested":{"neg":-99999999999999999999,"inf":-Infinity},"rope_theta":10000})");
        require(mamba.model_type == "mamba2" && mamba.rope.theta == 10000.0 && mamba.eos_token_ids == std::vector<Token>{1}, "non-finite and wide untyped values load");
        require(std::isinf(mamba.get("time_step_limit")->array()[1].real()) && std::isnan(mamba.get("x")->real()), "non-finite values stay reachable");
        require(mamba.get("big")->exceeds_64_bits() && mamba.get("nested")->find("neg")->exceeds_64_bits(), "wide integers stay reachable");
        // The length keys stop at the first positive integer, so a later wide one is never read.
        require(config_of(root, R"({"max_position_embeddings":4096,"model_max_length":1000000000000000019884624838656})").max_positions == 4096u, "length keys stop early");
    }
    // Null reads as absent; unknown keys stay reachable; the merge rule matches eval's text_config.
    auto config = config_of(root, R"({"architectures":["Gemma3ForConditionalGeneration"],"model_type":"gemma3","vocab_size":262208,)"
        R"("hidden_size":null,"tie_word_embeddings":true,"quantization_config":{"quant_method":"fp8","weight_block_size":[128,128]},"custom":{"k":[1,2]},)"
        R"("eos_token_id":[1,106],"text_config":{"model_type":"gemma3_text","hidden_size":2560,"vocab_size":null,"head_dim":256,)"
        R"("num_attention_heads":8,"num_key_value_heads":4,"num_hidden_layers":2,"rms_norm_eps":1e-06,"torch_dtype":"bfloat16","dtype":null,)"
        R"("rope_theta":1000000.0,"sliding_window":1024,"eos_token_id":null,"pad_token_id":-1,"bos_token_id":2}})");
    require(config.text && config.text != &config.document.root() && config.model_type == "gemma3" && config.text_model_type == "gemma3_text", "model types");
    require(config.architectures == std::vector<std::string>{"Gemma3ForConditionalGeneration"}, "root architectures stay visible");
    require(config.hidden_size == 2560u && !config.vocab_size && config.head_dim == 256u, "text section overrides the root, null included");
    require(config.rms_norm_eps == 1e-6 && config.norm_eps_key == "rms_norm_eps" && config.dtype == "bfloat16", "eps and dtype fallbacks");
    require(config.tie_word_embeddings == true && config.quantization && config.quantization->method == "fp8" && config.quantization->raw &&
            config.quantization->raw->find("weight_block_size"), "root-only keys");
    require(config.eos_token_ids.empty() && config.pad_token_id == -1 && config.bos_token_id == 2, "token ids follow the merge");
    require(config.get("custom") && config.get("custom")->find("k") && !config.get("absent"), "unknown keys through get()");
    require(config.get("vocab_size") && config.get("vocab_size")->is_null(), "get returns the text section's null");
    require(config.rope.theta == 1e6 && config.rope.type.empty() && !config.rope.raw && config.rope.key.empty(), "theta without a rope object");
    require(config.require_u32("num_attention_heads") == 8, "require_u32");
    rejects_as<FormatError>([&] { (void)config.require_u32("vocab_size"); }, "key 'vocab_size': missing");
    rejects_as<FormatError>([&] { (void)config.require_u32("intermediate_size"); }, "key 'intermediate_size': missing");
    rejects_as<FormatError>([&] { (void)config.require_u32("custom"); }, "key 'custom': expected an integer");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"hidden_size":0})").require_u32("hidden_size"); }, "expected a positive integer, got 0");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"text_config":null,"vocab_size":0})").require_u32("vocab_size"); }, "expected a positive integer, got 0");
    // dtype before torch_dtype; norm eps keys in order.
    config = config_of(root, R"({"dtype":"float16","torch_dtype":"bfloat16","layer_norm_epsilon":0.00001,"layer_norm_eps":null})");
    require(config.dtype == "float16" && config.norm_eps_key == "layer_norm_epsilon" && config.rms_norm_eps == 1e-5, "first non-null fallback keys");
    // Rope: rope_scaling first (eval's order), theta and partial factor fallbacks, rope_type before type.
    config = config_of(root, R"({"rope_scaling":{"type":"longrope","short_factor":[1,1.5],"long_factor":[2.0],"factor":32.0,"attention_factor":1.19},)"
                             R"("rope_parameters":{"rope_theta":500000,"partial_rotary_factor":0.5},"original_max_position_embeddings":4096})");
    require(config.rope.key == "rope_scaling" && config.rope.type == "longrope" && config.rope.factor == 32.0 && config.rope.attention_factor == 1.19, "rope_scaling read");
    require(config.rope.short_factor == std::vector<double>{1.0, 1.5} && config.rope.long_factor == std::vector<double>{2.0}, "rope factors");
    require(config.rope.theta == 500000.0 && !config.rope.partial_rotary_factor && config.rope.original_max_position_embeddings == 4096u, "rope fallbacks");
    config = config_of(root, R"({"rope_scaling":null,"rope_parameters":{"rope_type":"llama3","rope_theta":500000.0,"factor":8.0,"low_freq_factor":1.0,)"
                             R"("high_freq_factor":4.0,"original_max_position_embeddings":8192,"partial_rotary_factor":0.25,"mrope_section":[16,24,24]}})");
    require(config.rope.key == "rope_parameters" && config.rope.type == "llama3" && config.rope.theta == 500000.0 && config.rope.low_freq_factor == 1.0 &&
            config.rope.high_freq_factor == 4.0 && config.rope.original_max_position_embeddings == 8192u && config.rope.partial_rotary_factor == 0.25 &&
            config.rope.mrope_section == std::vector<std::uint32_t>{16, 24, 24} && config.rope.raw && config.rope_by_layer_type.empty(), "rope_parameters read");
    config = config_of(root, R"({"rope_theta":10000,"partial_rotary_factor":1.0,"rope_scaling":{"rope_type":"yarn","type":"ignored","factor":4,"beta_fast":32,"beta_slow":1,)"
                             R"("mscale":0.707,"mscale_all_dim":0.707,"partial_rotary_factor":0.5,"rope_theta":1}})");
    require(config.rope.theta == 10000.0 && config.rope.partial_rotary_factor == 1.0 && config.rope.type == "yarn" && config.rope.beta_fast == 32.0 &&
            config.rope.beta_slow == 1.0 && config.rope.mscale == 0.707 && config.rope.mscale_all_dim == 0.707, "section values win over the rope object");
    // transformers v5 rope_parameters keyed by layer type.
    config = config_of(root, R"({"rope_parameters":{"sliding_attention":{"rope_type":"default","rope_theta":10000.0},)"
                             R"("full_attention":{"rope_type":"linear","factor":8.0,"rope_theta":1000000.0}}})");
    require(config.rope.type.empty() && !config.rope.theta && config.rope.key == "rope_parameters" && config.rope_by_layer_type.size() == 2, "per layer type rope");
    require(config.rope_by_layer_type[0].layer_type == "sliding_attention" && config.rope_by_layer_type[0].rope.theta == 10000.0 &&
            config.rope_by_layer_type[1].rope.type == "linear" && config.rope_by_layer_type[1].rope.factor == 8.0 && config.rope_by_layer_type[1].rope.raw, "per layer entries");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"rope_parameters":{"full_attention":{"factor":"x"}}})"); }, "key 'rope_parameters.full_attention.factor'");
    // Experts from the first non-null key, recorded; zero experts is dense.
    config = config_of(root, R"({"num_experts":null,"num_local_experts":8,"n_routed_experts":64,"num_experts_per_tok":2,"n_shared_experts":null,)"
                             R"("num_shared_experts":1,"moe_intermediate_size":768,"first_k_dense_replace":1,"decoder_sparse_step":1,"n_group":8,"topk_group":4,)"
                             R"("mlp_only_layers":[0,5],"norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sigmoid","topk_method":"noaux_tc","shared_expert_intermediate_size":512})");
    require(config.moe && config.moe->experts == 8 && config.moe->experts_key == "num_local_experts" && config.moe->experts_per_token == 2 &&
            config.moe->shared_experts == 1u && config.moe->moe_intermediate_size == 768u && config.moe->first_k_dense_replace == 1u &&
            config.moe->decoder_sparse_step == 1u && config.moe->n_group == 8u && config.moe->topk_group == 4u &&
            config.moe->mlp_only_layers == std::vector<std::uint32_t>{0, 5} && config.moe->norm_topk_prob == true && config.moe->routed_scaling_factor == 2.5 &&
            config.moe->scoring_func == "sigmoid" && config.moe->topk_method == "noaux_tc" && config.moe->shared_expert_intermediate_size == 512u, "moe fields");
    require(!config_of(root, R"({"num_experts":0,"n_routed_experts":64})").moe, "zero experts is dense");
    config = config_of(root, R"({"n_routed_experts":256,"num_nextn_predict_layers":1})");
    require(config.moe && config.moe->experts_key == "n_routed_experts" && config.moe->experts_per_token == 0 && config.num_nextn_predict_layers == 1u, "routed experts");
    // Derived values reject inconsistent configs instead of rounding.
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"num_attention_heads":3,"hidden_size":100})").effective_head_dim(); },
                            "key 'hidden_size': 100 is not divisible by num_attention_heads 3");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"num_attention_heads":8})").effective_head_dim(); }, "key 'hidden_size': missing (no head_dim)");
    require(config_of(root, R"({"head_dim":0,"num_attention_heads":8,"hidden_size":512})").effective_head_dim() == 64, "zero head_dim falls back");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"num_attention_heads":8,"num_key_value_heads":3})").effective_kv_heads(); },
                            "key 'num_key_value_heads': 3 does not divide num_attention_heads 8");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"num_key_value_heads":2})").effective_kv_heads(); }, "key 'num_attention_heads': missing");
    require(config_of(root, R"({"num_attention_heads":8,"num_key_value_heads":0})").effective_kv_heads() == 8, "zero KV heads falls back");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"num_hidden_layers":3,"layer_types":["full_attention"]})").layer_kinds(); }, "key 'layer_types': 1 entries for 3 layers");
    rejects_as<FormatError>([&] { (void)config_of(root, "{}").layer_kinds(); }, "key 'num_hidden_layers': missing");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"num_hidden_layers":0})").layer_kinds(); }, "key 'num_hidden_layers': expected a positive integer, got 0");
    rejects_as<FormatError>([&] { (void)config_of(root, R"({"num_hidden_layers":65537})").layer_kinds(); }, "65537 layers exceed the 65536-layer limit");
    require(config_of(root, R"({"num_hidden_layers":65536})").layer_kinds().size() == 65536, "layer limit");
    // Files: a directory or the file itself; missing files and size limits.
    write_file(root / "named.json", R"({"vocab_size":5})");
    require(load_model_config(root / "named.json").vocab_size == 5u, "config file path");
    rejects_as<std::system_error>([&] { (void)load_model_config(root / "missing"); }, "missing: open");
    fs::create_directories(root / "nothing");
    rejects_as<std::system_error>([&] { (void)load_model_config(root / "nothing"); }, "config.json: open");
    rejects_as<FormatError>([&] { (void)load_model_config(root / "named.json", {.max_bytes = 4}); }, "bytes exceeds the 4-byte limit");
    write_file(root / "inf.json", R"({"vocab_size":5,"time_step_limit":[0.0,Infinity]})");
    require(load_model_config(root / "inf.json", {.max_bytes = 4096, .allow_nonfinite = false}).vocab_size == 5u, "config.json always allows NaN and Infinity");
    // Copies share the document, so their pointers stay valid after the original is gone.
    std::optional<ModelConfig> original = config_of(root, R"({"text_config":{"hidden_size":64},"rope_scaling":{"rope_type":"yarn","factor":2,"original_max_position_embeddings":10}})");
    const ModelConfig copy = *original;
    original.reset();
    require(copy.get("hidden_size") && copy.get("hidden_size")->u64() == 64 && copy.rope.raw && copy.position_limit() == 20u, "copied config pointers");
}

void model_generation() {
    TempDir dir;
    const auto& root = dir.path;
    require(!load_generation_config(root), "absent generation config");
    write_file(root / "generation_config.json", R"({"bos_token_id":1,"eos_token_id":[2,106],"pad_token_id":0,"do_sample":true,"temperature":0.6,)"
        R"("top_p":0.95,"top_k":20,"min_p":0,"repetition_penalty":1.05,"max_new_tokens":512,"max_length":32768,"stop_strings":["</s>","###"],)"
        R"("transformers_version":"4.51.0","cache_implementation":"hybrid"})");
    const auto generation = load_generation_config(root);
    require(generation && generation->eos_token_ids == std::vector<Token>{2, 106} && generation->bos_token_id == 1 && generation->pad_token_id == 0, "generation ids");
    require(generation->do_sample == true && generation->temperature == 0.6 && generation->top_p == 0.95 && generation->top_k == 20u &&
            generation->min_p == 0.0 && generation->repetition_penalty == 1.05 && generation->max_new_tokens == 512u && generation->max_length == 32768u, "sampling defaults");
    require(generation->stop_strings == std::vector<std::string>{"</s>", "###"} && generation->document.root().find("cache_implementation"), "stop strings and raw keys");
    write_file(root / "generation_config.json", R"({"eos_token_id":7,"stop_strings":"<|end|>","temperature":null})");
    const auto minimal = load_generation_config(root);
    require(minimal && minimal->eos_token_ids == std::vector<Token>{7} && minimal->stop_strings == std::vector<std::string>{"<|end|>"} && !minimal->temperature &&
            !minimal->top_k && !minimal->do_sample, "single id and stop string");
    for(const auto& [json, message]:std::vector<std::pair<std::string, std::string>>{
            {R"({"temperature":-0.1})", "key 'temperature': expected a number at least 0, got -0.1"},
            {R"({"top_p":1.5})", "key 'top_p': expected a number between 0 and 1, got 1.5"},
            {R"({"min_p":-1})", "key 'min_p': expected a number between 0 and 1, got -1"},
            {R"({"repetition_penalty":0})", "key 'repetition_penalty': expected a number greater than 0, got 0"},
            {R"({"temperature":true})", "key 'temperature': expected a finite number, got true"},
            {R"({"top_k":-1})", "key 'top_k': expected an integer in [0, 4294967295], got -1"},
            {R"({"max_new_tokens":1.0})", "key 'max_new_tokens': expected an integer"},
            {R"({"do_sample":"true"})", "key 'do_sample': expected true or false, got 'true'"},
            {R"({"eos_token_id":-1})", "key 'eos_token_id': expected an integer in [0, 2147483647], got -1"},
            {R"({"pad_token_id":[0]})", "key 'pad_token_id': expected an integer in [-2147483648, 2147483647], got an array"},
            {R"({"bos_token_id":2147483648})", "key 'bos_token_id': expected an integer in [-2147483648, 2147483647], got 2147483648"},
            {R"({"eos_token_id":[1,-1]})", "key 'eos_token_id[1]': expected an integer in [0, 2147483647], got -1"},
            {R"({"stop_strings":["ok",""]})", "key 'stop_strings': expected non-empty strings, got ''"},
            {R"({"stop_strings":""})", "key 'stop_strings': expected non-empty strings"},
            {R"({"stop_strings":[1]})", "key 'stop_strings': entry 0 is 1, not a string"},
            {R"({"text_config":3})", "key 'text_config': expected an object, got 3"},
            {R"("x")", "root: the generation config must be a JSON object"},
            {R"({"top_k":1,"top_k":2})", "duplicate key 'top_k'"},
            {R"({"temperature":Infinity})", "key 'temperature': expected a finite number, got inf"},
            {R"({"top_p":NaN})", "key 'top_p': expected a finite number, got nan"},
            {R"({"repetition_penalty":-Infinity})", "key 'repetition_penalty': expected a finite number, got -inf"},
            {R"({"top_k":18446744073709551616})", "key 'top_k': integer exceeds 64 bits"},
            {R"({"eos_token_id":[2,-99999999999999999999]})", "key 'eos_token_id[1]': integer exceeds 64 bits"},
            {R"({"max_length":1e400})", "real number out of range"}}) {
        write_file(root / "generation_config.json", json);
        rejects_as<FormatError>([&] { (void)load_generation_config(root); }, message);
    }
    // transformers copies a config's pad -1 into generation_config.json; it loads like config.json does.
    write_file(root / "generation_config.json", R"({"_from_model_config":true,"bos_token_id":-2147483648,"eos_token_id":1,"pad_token_id":-1})");
    const auto padded = load_generation_config(root);
    require(padded && padded->pad_token_id == -1 && padded->bos_token_id == std::numeric_limits<Token>::min() && padded->eos_token_ids == std::vector<Token>{1},
            "negative bos and pad ids");
    write_file(root / "custom.json", R"({"top_k":3})");
    require(load_generation_config(root / "custom.json")->top_k == 3u, "generation config file path");
    // Keys no field reads may hold NaN, Infinity or integers beyond 64 bits, as transformers can write them.
    write_file(root / "generation_config.json", R"({"eos_token_id":2,"guidance_scale":Infinity,"epsilon_cutoff":NaN,"seed":123456789012345678901,"temperature":0.7})");
    const auto loose = load_generation_config(root);
    require(loose && loose->eos_token_ids == std::vector<Token>{2} && loose->temperature == 0.7 &&
            std::isinf(loose->document.root().find("guidance_scale")->real()) && loose->document.root().find("seed")->exceeds_64_bits(), "untyped generation values load");
    // Any limits passed still read config files with allow_nonfinite.
    require(load_generation_config(root, {.max_bytes = 4096})->eos_token_ids == std::vector<Token>{2}, "limits keep allow_nonfinite on");
    rejects_as<std::system_error>([&] { (void)load_generation_config(root / "absent.json"); }, "absent.json: open");
}

void model_stop_ids() {
    TempDir dir;
    const auto& root = dir.path;
    // prepare.stop_ids gives [0, 3, 7, 9, 11] here, [3, 7, 9] without the generation config, [] for nulls.
    const auto config = config_of(root, R"({"eos_token_id":[7,3],"text_config":{"eos_token_id":9}})");
    write_file(root / "generation_config.json", R"({"eos_token_id":3,"text_config":{"eos_token_id":[11,0]}})");
    const auto generation = load_generation_config(root);
    require(stop_token_ids(config, &*generation) == std::vector<Token>{0, 3, 7, 9, 11}, "stop id union");
    require(stop_token_ids(config, nullptr) == std::vector<Token>{3, 7, 9}, "stop ids without a generation config");
    require(config.eos_token_ids == std::vector<Token>{9}, "merged eos follows text_config");
    require(stop_token_ids(config_of(root, R"({"eos_token_id":null,"text_config":{}})"), nullptr).empty(), "null eos");
    require(stop_token_ids(config_of(root, R"({"eos_token_id":5,"text_config":null})"), nullptr) == std::vector<Token>{5}, "null text_config");
    require(stop_token_ids(ModelConfig{}, nullptr).empty(), "default config has no stop ids");
    // prepare.stop_ids gives [1] for a pad -1 model whose generation config copied it.
    write_file(root / "pad" / "generation_config.json", R"({"_from_model_config":true,"bos_token_id":0,"eos_token_id":1,"pad_token_id":-1})");
    const auto pad_config = config_of(root / "pad", R"({"bos_token_id":0,"eos_token_id":1,"pad_token_id":-1})");
    const auto pad_generation = load_generation_config(root / "pad");
    require(pad_generation && stop_token_ids(pad_config, &*pad_generation) == std::vector<Token>{1}, "stop ids beside a pad id of -1");
    const auto negative = config_of(root, R"({"text_config":{"eos_token_id":4},"eos_token_id":-1})");
    require(negative.eos_token_ids == std::vector<Token>{4}, "negative root eos hidden by the text section");
    rejects_as<FormatError>([&] { (void)stop_token_ids(negative, nullptr); }, "key 'eos_token_id': expected an integer in [0, 2147483647], got -1");
    const auto nested = config_of(root, R"({"eos_token_id":1,"text_config":{"eos_token_id":[2,-7]}})");
    rejects_as<FormatError>([&] { (void)stop_token_ids(nested, nullptr); }, "key 'text_config.eos_token_id[1]': expected an integer in [0, 2147483647], got -7");
    // Files with Infinity and wide integers in untyped keys still give their stop ids; a wide eos id is refused.
    write_file(root / "loose" / "generation_config.json", R"({"eos_token_id":[5,2],"guidance_scale":Infinity,"seed":99999999999999999999})");
    const auto loose_config = config_of(root / "loose", R"({"eos_token_id":7,"time_step_limit":[0.0,Infinity],"big":123456789012345678901})");
    const auto loose_generation = load_generation_config(root / "loose");
    require(loose_generation && stop_token_ids(loose_config, &*loose_generation) == std::vector<Token>{2, 5, 7}, "stop ids beside non-finite and wide values");
    write_file(root / "wide" / "generation_config.json", R"({"text_config":{"eos_token_id":18446744073709551616}})");
    const auto wide_generation = load_generation_config(root / "wide");
    rejects_as<FormatError>([&] { (void)stop_token_ids(loose_config, &*wide_generation); }, "key 'text_config.eos_token_id': integer exceeds 64 bits");
    // Randomized: the union equals a set built independently from the four sources.
    std::mt19937 random(77);
    for(int round = 0; round < 50; ++round) {
        std::set<Token> expected;
        auto ids = [&](bool list) {
            if(random() % 4 == 0) return std::string("null");
            if(!list) { const Token id = Token(random() % 50); expected.insert(id); return std::to_string(id); }
            std::string out = "[";
            const auto count = random() % 4;
            for(std::uint32_t i = 0; i < count; ++i) { const Token id = Token(random() % 50); expected.insert(id); out += (i ? "," : "") + std::to_string(id); }
            return out + "]";
        };
        const auto model = config_of(root, "{\"eos_token_id\":" + ids(random() % 2) + ",\"text_config\":{\"eos_token_id\":" + ids(random() % 2) + "}}");
        write_file(root / "generation_config.json", "{\"eos_token_id\":" + ids(random() % 2) + ",\"text_config\":{\"eos_token_id\":" + ids(random() % 2) + "}}");
        const auto gen = load_generation_config(root);
        require(stop_token_ids(model, &*gen) == std::vector<Token>(expected.begin(), expected.end()), "randomized stop id union");
    }
}

void model_rope() {
    TempDir dir;
    const auto config = [&](const std::string& json) {
        write_file(dir.path / "config.json", json);
        return load_model_config(dir.path / "config.json");
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

// rope_table of <dir>/config.json for the rope_oracle.py comparison: "attention inv_freq..." in %.17g, or
// "error <message>".
int print_rope(const fs::path& dir, std::uint32_t head_dim, std::optional<std::uint64_t> length) {
    try {
        const auto config = load_model_config(dir);
        const auto table = rope_table(config.rope, head_dim, config.max_positions, length);
        std::printf("%.17g", table.attention_factor);
        for(const double f : table.inv_freq) std::printf(" %.17g", f);
        std::printf("\n");
    } catch(const std::exception& e) { std::printf("error %s\n", e.what()); }
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "all";
        if(test == "model_rope_print" && argc == 5)
            return print_rope(argv[2], std::uint32_t(std::stoul(argv[3])),
                              std::string(argv[4]) == "-" ? std::nullopt : std::optional<std::uint64_t>(std::stoull(argv[4])));
        const std::vector<std::pair<std::string, std::function<void()>>> tests{
            {"model_dtypes", model_dtypes}, {"model_header_valid", model_header_valid}, {"model_header_rejects", model_header_rejects},
            {"model_limits", model_limits}, {"model_subbyte", model_subbyte}, {"model_header_only", model_header_only},
            {"model_index", model_index}, {"model_extra_files", model_extra_files}, {"model_directory", model_directory},
            {"model_symlinks", model_symlinks}, {"model_bytes", model_bytes}, {"model_digests", model_digests},
            {"model_change_detection", model_change_detection}, {"model_retained", model_retained}, {"model_random", model_random}, {"model_config_eval", model_config_eval}, {"model_config_types", model_config_types},
            {"model_generation", model_generation}, {"model_stop_ids", model_stop_ids}, {"model_rope", model_rope}};
        bool found = false;
        for(const auto& [name, run]:tests) if(test == "all" || test == name) {
            run();
            found = true;
            std::cout << "crit core model passed: " << name << '\n';
        }
        require(found, "unknown test");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
