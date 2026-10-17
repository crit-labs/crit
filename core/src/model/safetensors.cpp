#include "crit/safetensors.h"
#include "crit/sha256.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>

namespace crit::core {
namespace {
namespace fs = std::filesystem;
constexpr std::uint64_t limit_max = (std::uint64_t(1) << 63) - 1;
constexpr std::string_view file_suffix = ".safetensors", index_suffix = ".safetensors.index.json";
constexpr const char* index_name = "model.safetensors.index.json";
constexpr const char* single_name = "model.safetensors";
constexpr std::size_t listed_candidates = 8;
constexpr std::size_t unknown_shown = 8;   // unknown index keys named in one error or note

struct DTypeRow { std::string_view name; std::uint32_t bits; };
constexpr std::array<DTypeRow, 19> dtype_rows{{   // in DType order
    {"BOOL", 8}, {"U8", 8}, {"I8", 8}, {"U16", 16}, {"I16", 16}, {"U32", 32}, {"I32", 32}, {"U64", 64}, {"I64", 64},
    {"F8_E4M3", 8}, {"F8_E5M2", 8}, {"F8_E8M0", 8}, {"F16", 16}, {"BF16", 16}, {"F32", 32}, {"F64", 64},
    {"F4", 4}, {"F6_E2M3", 6}, {"F6_E3M2", 6}}};

std::string str(std::uint64_t value) { return std::to_string(value); }
bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}
// Short printable rendering of a name for diagnostics; never splits a UTF-8 sequence.
std::string shown(std::string_view text) {
    std::size_t cut = std::min<std::size_t>(text.size(), 96);
    while(cut < text.size() && cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
    std::string out = "'";
    for(char c:text.substr(0, cut)) {
        const auto b = static_cast<unsigned char>(c);
        if(b < 0x20 || b == 0x7F || c == '\'' || c == '\\') {
            static const char hex[] = "0123456789abcdef";
            out += "\\x"; out += hex[b >> 4]; out += hex[b & 15];
        } else out += c;
    }
    return out + (cut < text.size() ? "'..." : "'");
}
std::string integer_text(const json::Value& value) {
    if(value.exceeds_64_bits()) return "an integer beyond 64 bits";
    try { return str(value.u64()); } catch(const FormatError&) {}
    return std::to_string(value.i64());
}
std::string describe(const json::Value& value) {
    switch(value.kind()) {
    case json::Kind::null: return "null";
    case json::Kind::boolean: return value.boolean() ? "true" : "false";
    case json::Kind::integer: return integer_text(value);
    case json::Kind::real: {
        char buffer[32];
        const auto done = std::to_chars(buffer, buffer + sizeof buffer, value.real());
        return "the real number " + std::string(buffer, done.ptr);
    }
    case json::Kind::string: return "the string " + shown(value.string());
    case json::Kind::array: return "an array";
    case json::Kind::object: return "an object";
    }
    return "a value";
}
// The value of a non-negative JSON integer; nullopt for a negative one or one beyond 64 bits (the caller checked
// the kind).
std::optional<std::uint64_t> non_negative(const json::Value& value) {
    try { return value.u64(); } catch(const FormatError&) { return std::nullopt; }
}
// A header integer beyond 64 bits fails where a JSON error in the header would, at its byte.
void within_64_bits(const json::Value& value, const std::string& source) {
    if(value.exceeds_64_bits()) throw FormatError(source, "header byte " + str(value.offset()), "integer exceeds 64 bits");
}
std::string shape_text(const std::vector<std::uint64_t>& shape) {
    std::string out = "[";
    for(std::size_t i = 0; i < shape.size(); ++i) out += (i ? ", " : "") + str(shape[i]);
    return out + "]";
}
void validate_options(const CheckpointOptions& options) {
    const auto& l = options.limits;
    for(const std::uint64_t value:{l.header_bytes, l.index_bytes, l.name_bytes, l.dimension, std::uint64_t(l.tensors),
                                    std::uint64_t(l.shards), std::uint64_t(l.rank), l.file_bytes, l.total_bytes, l.retained_bytes})
        if(value == 0 || value > limit_max) throw std::invalid_argument("safetensors limits must be in [1, 2^63 - 1]");
    if(options.index != IndexPolicy::strict && options.index != IndexPolicy::compatible)
        throw std::invalid_argument("unknown index policy");
}
// Why a tensor name is invalid (empty, too long, NUL), or nullopt. JSON already guarantees UTF-8.
std::optional<std::string> name_problem(std::string_view name, std::uint64_t limit) {
    if(name.empty()) return std::string("name is empty");
    if(name.size() > limit) return "name of " + str(name.size()) + " bytes exceeds the " + str(limit) + "-byte limit";
    if(name.find('\0') != std::string_view::npos) return std::string("name contains NUL");
    return std::nullopt;
}
bool alnum(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); }
// The audit's flat portable shard rule: [A-Za-z0-9][A-Za-z0-9._-]*\.safetensors, no Windows device name.
std::optional<std::string> shard_name_problem(std::string_view name, std::uint64_t limit) {
    if(auto problem = name_problem(name, limit)) return problem;
    const bool portable = name.size() > file_suffix.size() && ends_with(name, file_suffix) && alnum(name.front()) &&
                          std::all_of(name.begin(), name.end(), [](char c) { return alnum(c) || c == '.' || c == '_' || c == '-'; });
    if(!portable) return std::string("is not a portable flat .safetensors file name");
    std::string device(name.substr(0, name.find('.')));
    for(auto& c:device) if(c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
    const bool numbered = device.size() == 4 && (device.starts_with("COM") || device.starts_with("LPT")) && device[3] >= '0' && device[3] <= '9';
    if(device == "CON" || device == "PRN" || device == "AUX" || device == "NUL" || numbered) return std::string("is a reserved device name");
    return std::nullopt;
}
// An index shard: a relative "/"-separated path of at most shard_depth portable components ending in a flat shard
// name, as checkpoints that keep their shards in a subfolder (tensors/...) list them. Every component starts with a
// letter or digit, so none is "." or "..", and nothing leaves the index's directory but through a link.
constexpr std::size_t shard_depth = 8;
std::optional<std::string> shard_path_problem(std::string_view name, std::uint64_t limit) {
    if(auto problem = name_problem(name, limit)) return problem;
    std::vector<std::string_view> parts;
    for(std::size_t start = 0;;) {
        const auto end = name.find('/', start);
        parts.push_back(name.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
        if(end == std::string_view::npos) break;
        start = end + 1;
    }
    const auto component = [](std::string_view part) {
        return !part.empty() && alnum(part.front()) &&
               std::all_of(part.begin(), part.end(), [](char c) { return alnum(c) || c == '.' || c == '_' || c == '-'; });
    };
    bool portable = parts.size() <= shard_depth && std::all_of(parts.begin(), parts.end() - 1, component);
    if(portable) {
        if(auto problem = shard_name_problem(parts.back(), limit)) {
            if(*problem != "is a reserved device name") portable = false;
            else return problem;
        }
    }
    if(!portable) return std::string("is not a portable relative .safetensors path");
    for(const auto part:parts) {
        std::string device(part.substr(0, part.find('.')));
        for(auto& c:device) if(c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
        const bool numbered = device.size() == 4 && (device.starts_with("COM") || device.starts_with("LPT")) && device[3] >= '0' && device[3] <= '9';
        if(device == "CON" || device == "PRN" || device == "AUX" || device == "NUL" || numbered) return std::string("is a reserved device name");
    }
    return std::nullopt;
}
// model.safetensors.index.json, else model.safetensors, else the directory's only *.safetensors entry. Each
// candidate must be a regular file or a link (a dangling one included); a directory by that name is skipped.
fs::path resolve_directory(const fs::path& directory) {
    for(const char* name:{index_name, single_name}) {
        const auto candidate = directory / name;
        std::error_code ignored;
        const auto type = fs::symlink_status(candidate, ignored).type();
        if(!ignored && (type == fs::file_type::regular || type == fs::file_type::symlink)) return candidate;
    }
    std::set<std::string> listed;
    std::size_t count = 0;
    fs::path only;
    for(const auto& entry:fs::directory_iterator(directory)) {
        const auto& path = entry.path();
        if(path.extension() != fs::path(file_suffix)) continue;
        std::error_code ignored;
        if(!entry.is_regular_file(ignored) && !entry.is_symlink(ignored)) continue;
        ++count;
        only = path;
        listed.insert(display_path(path.filename()));
        if(listed.size() > listed_candidates) listed.erase(std::prev(listed.end()));
    }
    const auto source = display_path(directory);
    if(count == 1) return only;
    if(count == 0) throw FormatError(source, "directory", "no model.safetensors.index.json, model.safetensors or other .safetensors file");
    std::string names;
    for(const auto& name:listed) names += (names.empty() ? "" : ", ") + shown(name);
    if(count > listed.size()) names += ", ...";
    throw FormatError(source, "directory", str(count) + " .safetensors files and no index; name one of " + names);
}
// Reads a whole small file that was already opened and size-checked, then verifies it did not change.
std::string read_all(const ReadOnlyFile& file) {
    if(file.identity().size > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error(display_path(file.path()) + ": index too large for this platform");
    std::string text(std::size_t(file.identity().size), '\0');
    file.read_exact(0, std::as_writable_bytes(std::span<char>(text.data(), text.size())));
    file.verify_unchanged();
    return text;
}
// Bytes of the first `chars` code points of valid UTF-8 text (all of it when shorter).
std::size_t prefix_bytes(std::string_view text, std::size_t chars) {
    std::size_t i = 0;
    for(std::size_t seen = 0; i < text.size(); ++i)
        if((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80 && seen++ == chars) break;
    return i;
}
// A key as the Python port shows it: quoted (escaped like shown()), and when that takes more than 48
// characters, its first 45 characters and "...". Only a bounded prefix of the key is rendered.
std::string short_key(std::string_view key) {
    std::string out = "'";
    std::size_t chars = 1, i = 0;
    while(i < key.size() && chars <= 48) {
        const auto b = static_cast<unsigned char>(key[i]);
        std::size_t length = 1;
        if(b < 0x20 || b == 0x7F || key[i] == '\'' || key[i] == '\\') {
            static const char hex[] = "0123456789abcdef";
            out += "\\x"; out += hex[b >> 4]; out += hex[b & 15];
            chars += 4;
        } else {
            if(b >= 0x80) length = b < 0xE0 ? 2u : b < 0xF0 ? 3u : 4u;   // JSON keys are valid UTF-8
            out.append(key.substr(i, length));
            ++chars;
        }
        i += length;
    }
    if(i == key.size()) { out += '\''; ++chars; }
    return chars <= 48 ? out : out.substr(0, prefix_bytes(out, 45)) + "...";
}
// Sorted keys for one error or note: at most unknown_shown of them, then "...] (N in all)".
std::string key_list(const std::vector<std::string>& keys) {
    std::string out = "[";
    for(std::size_t i = 0; i < keys.size() && i < unknown_shown; ++i) out += (i ? ", " : "") + short_key(keys[i]);
    return keys.size() <= unknown_shown ? out + "]" : out + ", ...] (" + str(keys.size()) + " in all)";
}
// Index keys no rule interprets: strict fails naming them; compatible adds one note for the whole group.
void unknown_keys(const std::vector<std::string>& keys, const std::string& what, const std::string& source, bool strict,
                  std::vector<std::string>& notes) {
    if(keys.empty()) return;
    if(strict) throw FormatError(source, what, "unsupported " + what + " keys " + key_list(keys));
    notes.push_back(keys.size() == 1 ? what + " key " + short_key(keys.front()) + " is not interpreted"
                                     : what + " keys " + key_list(keys) + " are not interpreted");
}
// What a file counts against total_bytes: its size or, with header_only, its last tensor's end when larger.
std::uint64_t extent(const SafetensorsFile& file) {
    std::uint64_t end = file.file().identity().size;
    for(const auto& tensor:file.tensors()) end = std::max(end, tensor.end);
    return end;
}
bool lowercase_hex64(std::string_view text) {
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
}

std::string_view dtype_name(DType dtype) noexcept {
    const auto i = std::size_t(dtype);
    return i < dtype_rows.size() ? dtype_rows[i].name : std::string_view();
}
std::uint32_t dtype_bits(DType dtype) noexcept {
    const auto i = std::size_t(dtype);
    return i < dtype_rows.size() ? dtype_rows[i].bits : 0;
}
std::optional<DType> parse_dtype(std::string_view name) noexcept {
    for(std::size_t i = 0; i < dtype_rows.size(); ++i) if(dtype_rows[i].name == name) return DType(i);
    return std::nullopt;
}

std::uint64_t TensorInfo::nbytes() const noexcept { return end > begin ? end - begin : 0; }
std::uint64_t TensorInfo::elements() const noexcept {
    if(std::find(shape.begin(), shape.end(), std::uint64_t(0)) != shape.end()) return 0;
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t count = 1;
    for(const auto dim:shape) count = count > max / dim ? max : count * dim;
    return count;
}
bool TensorInfo::element_aligned() const noexcept { return begin % std::max<std::uint32_t>(1, dtype_bits(dtype) / 8) == 0; }

SafetensorsFile::SafetensorsFile(ReadOnlyFile file) : file_(std::move(file)) {}
// A moved vector keeps its buffer, so the span (into owned_ or a Checkpoint's records) stays valid.
SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : file_(std::move(other.file_)), header_(std::move(other.header_)), metadata_(std::move(other.metadata_)), owned_(std::move(other.owned_)),
      tensors_(std::exchange(other.tensors_, {})), retained_(std::exchange(other.retained_, 0)), header_only_(other.header_only_) {}
SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
    if(this != &other) {
        file_ = std::move(other.file_);
        header_ = std::move(other.header_);
        metadata_ = std::move(other.metadata_);
        owned_ = std::move(other.owned_);
        tensors_ = std::exchange(other.tensors_, {});
        retained_ = std::exchange(other.retained_, 0);
        header_only_ = other.header_only_;
    }
    return *this;
}
SafetensorsFile SafetensorsFile::open(const std::filesystem::path& path, const CheckpointOptions& options) {
    validate_options(options);
    return open_shard(path, options, 0, 0);
}
SafetensorsFile SafetensorsFile::open_shard(const std::filesystem::path& path, const CheckpointOptions& options, std::uint32_t shard,
                                            std::uint64_t retained_before) {
    const auto& limits = options.limits;
    const auto source = display_path(path);
    const auto file_name = display_path(path.filename());
    if(file_name.size() <= file_suffix.size() || !ends_with(file_name, file_suffix))
        throw FormatError(source, "file", "expected a .safetensors file");
    auto file = ReadOnlyFile::open(path, options.follow_symlinks);
    const std::uint64_t size = file.identity().size;
    if(size > limits.file_bytes) throw FormatError(source, "size", str(size) + " bytes exceed the " + str(limits.file_bytes) + "-byte file limit");
    if(size < 8) throw FormatError(source, "header", "truncated header length: the file has " + str(size) + " bytes");
    std::array<std::byte, 8> prefix{};
    file.read_exact(0, prefix);
    std::uint64_t length = 0;
    for(std::size_t i = prefix.size(); i-- > 0;) length = (length << 8) | std::to_integer<std::uint64_t>(prefix[i]);
    if(length == 0) throw FormatError(source, "header", "header length is zero");
    if(length > limits.header_bytes)
        throw FormatError(source, "header", "header length " + str(length) + " exceeds the " + str(limits.header_bytes) + "-byte limit");
    if(length > size - 8)
        throw FormatError(source, "header", "header length " + str(length) + " exceeds the " + str(size - 8) + " bytes after the length prefix");
    if(length > std::numeric_limits<std::size_t>::max()) throw std::runtime_error(source + ": header too large for this platform");
    // Everything kept is charged before it is reserved, so a header that cannot be retained is never read.
    std::uint64_t retained = 0;
    const auto charge = [&](std::uint64_t bytes) {
        const std::uint64_t used = retained_before + retained;   // both <= the limit, so neither sum wraps
        if(bytes > limits.retained_bytes - used)
            throw FormatError(source, "size", std::string(retained_before ? "checkpoint headers and tensor records" : "header and tensor records") +
                              " exceed the " + str(limits.retained_bytes) + "-byte retained_bytes limit");
        retained += bytes;
    };
    charge(length);
    SafetensorsFile out(std::move(file));
    out.header_.resize(std::size_t(length));
    out.file_.read_exact(8, std::as_writable_bytes(std::span<char>(out.header_.data(), out.header_.size())));
    out.file_.verify_unchanged();
    if(out.header_.front() != '{') throw FormatError(source, "header", "header must start with '{'");
    json::Document document;
    try {
        document = json::Document::parse(out.header_, source, {.max_bytes = std::size_t(length), .max_depth = 64, .max_nodes = std::size_t(length)});
    } catch(const FormatError& e) {
        const std::string_view what = e.what();
        const std::size_t skip = e.source().size() + e.where().size() + 4;
        throw FormatError(source, "header " + e.where(), std::string(what.size() > skip ? what.substr(skip) : what));
    }
    const auto& root = document.root();
    if(root.kind() != json::Kind::object) throw FormatError(source, "header", "header must be a JSON object");
    if(const auto* metadata = root.find("__metadata__")) {
        if(metadata->kind() != json::Kind::object)
            throw FormatError(source, "__metadata__", "__metadata__ must map strings to strings, not " + describe(*metadata));
        for(const auto& member:metadata->object())
            if(member.value.kind() != json::Kind::string)
                throw FormatError(source, "__metadata__", "__metadata__ must map strings to strings; " + shown(member.key) + " holds " + describe(member.value));
        std::uint64_t text = 0;   // bounded by the header length
        for(const auto& member:metadata->object()) text += member.key.size() + member.value.string().size();
        charge(text);
        out.metadata_.reserve(metadata->object().size());
        for(const auto& member:metadata->object()) out.metadata_.emplace_back(member.key, std::string(member.value.string()));
    }
    const std::uint64_t payload_offset = 8 + length, payload = size - payload_offset;
    const std::uint64_t offset_limit = options.header_only ? limits.file_bytes : payload;
    const auto members = root.object();
    // The fixed part of every record that can be kept is charged up front; names and shapes as they come.
    const std::size_t records = std::min<std::size_t>(members.size() - (root.find("__metadata__") ? 1 : 0), limits.tensors);
    charge(sizeof(TensorInfo) * std::uint64_t(records));
    out.owned_.reserve(records);
    for(const auto& member:members) {
        if(member.key == "__metadata__") continue;
        const std::string where = "tensor " + shown(member.key);
        if(auto problem = name_problem(member.key, limits.name_bytes)) throw FormatError(source, where, *problem);
        if(out.owned_.size() >= limits.tensors) throw FormatError(source, "header", "more than " + str(limits.tensors) + " tensors");
        const auto& record = member.value;
        const json::Value* dtype_value = record.kind() == json::Kind::object ? record.find("dtype") : nullptr;
        const json::Value* shape_value = dtype_value ? record.find("shape") : nullptr;
        const json::Value* offsets = shape_value ? record.find("data_offsets") : nullptr;
        if(!offsets || record.object().size() != 3) throw FormatError(source, where, "expected exactly the keys dtype, shape and data_offsets");
        const auto dtype = dtype_value->kind() == json::Kind::string ? parse_dtype(dtype_value->string()) : std::nullopt;
        if(!dtype) throw FormatError(source, where, "unsupported dtype " + describe(*dtype_value));
        if(shape_value->kind() != json::Kind::array) throw FormatError(source, where, "shape must be an array, not " + describe(*shape_value));
        const auto dims = shape_value->array();
        if(dims.size() > limits.rank) throw FormatError(source, where, "rank " + str(dims.size()) + " exceeds the " + str(limits.rank) + " limit");
        TensorInfo info;
        info.name = member.key;
        info.dtype = *dtype;
        info.shard = shard;
        info.shape.reserve(dims.size());
        for(const auto& dim:dims) {
            if(dim.kind() != json::Kind::integer) throw FormatError(source, where, "dimension must be an integer, not " + describe(dim));
            within_64_bits(dim, source);
            const auto value = non_negative(dim);
            if(!value) throw FormatError(source, where, "dimension " + integer_text(dim) + " is negative");
            if(*value > limits.dimension) throw FormatError(source, where, "dimension " + str(*value) + " exceeds the " + str(limits.dimension) + " limit");
            info.shape.push_back(*value);
        }
        // Elements are bounded by the file limit in bits, so neither the product nor the byte count overflows.
        const std::uint64_t bits = dtype_bits(*dtype);
        const std::uint64_t max_elements = limits.file_bytes / bits * 8 + limits.file_bytes % bits * 8 / bits;
        std::uint64_t elements = std::find(info.shape.begin(), info.shape.end(), std::uint64_t(0)) == info.shape.end() ? 1 : 0;
        for(const auto dim:info.shape) {
            if(elements && dim > max_elements / elements)
                throw FormatError(source, where, "shape " + shape_text(info.shape) + " of " + std::string(dtype_name(*dtype)) +
                                  " exceeds the " + str(limits.file_bytes) + "-byte file limit");
            elements *= dim;
        }
        if(elements % 8 * bits % 8)
            throw FormatError(source, where, str(elements) + " elements of " + std::string(dtype_name(*dtype)) + " do not fill whole bytes");
        const std::uint64_t nbytes = elements / 8 * bits + elements % 8 * bits / 8;
        if(offsets->kind() != json::Kind::array || offsets->array().size() != 2)
            throw FormatError(source, where, "data_offsets must be [begin, end], not " + describe(*offsets));
        std::uint64_t bounds[2] = {0, 0};
        for(std::size_t i = 0; i < 2; ++i) {
            const auto& item = offsets->array()[i];
            if(item.kind() != json::Kind::integer) throw FormatError(source, where, "data offset must be an integer, not " + describe(item));
            within_64_bits(item, source);
            const auto value = non_negative(item);
            if(!value) throw FormatError(source, where, "data offset " + integer_text(item) + " is negative");
            if(*value > offset_limit)
                throw FormatError(source, where, options.header_only
                    ? "data offset " + str(*value) + " exceeds the " + str(offset_limit) + "-byte file limit"
                    : "data offset " + str(*value) + " is past the " + str(offset_limit) + "-byte payload");
            bounds[i] = *value;
        }
        if(bounds[1] < bounds[0]) throw FormatError(source, where, "data_offsets end " + str(bounds[1]) + " precedes begin " + str(bounds[0]));
        if(bounds[1] - bounds[0] != nbytes)
            throw FormatError(source, where, "data_offsets [" + str(bounds[0]) + ", " + str(bounds[1]) + ") hold " + str(bounds[1] - bounds[0]) +
                              " bytes but shape " + shape_text(info.shape) + " of " + std::string(dtype_name(*dtype)) + " needs " + str(nbytes));
        info.begin = payload_offset + bounds[0];
        info.end = payload_offset + bounds[1];
        charge(info.name.size() + sizeof(std::uint64_t) * std::uint64_t(info.shape.size()));   // its name and shape
        out.owned_.push_back(std::move(info));
    }
    // Coverage: sorted by (begin, end, name), the extents must tile the payload from its start without gaps.
    std::vector<const TensorInfo*> order;
    order.reserve(out.owned_.size());
    for(const auto& tensor:out.owned_) order.push_back(&tensor);
    std::sort(order.begin(), order.end(), [](const TensorInfo* a, const TensorInfo* b) {
        return a->begin != b->begin ? a->begin < b->begin : a->end != b->end ? a->end < b->end : a->name < b->name;
    });
    std::uint64_t cursor = payload_offset;
    const TensorInfo* previous = nullptr;
    for(const auto* tensor:order) {
        if(tensor->begin != cursor) {
            const bool hole = tensor->begin > cursor;
            const auto at = (hole ? cursor : tensor->begin) - payload_offset;
            const auto between = previous ? "between " + shown(previous->name) + " and " + shown(tensor->name) : "before " + shown(tensor->name);
            throw FormatError(source, "payload", std::string(hole ? "hole" : "overlap") + " at payload offset " + str(at) + " " + between);
        }
        cursor = tensor->end;
        previous = tensor;
    }
    if(!options.header_only && cursor != size)
        throw FormatError(source, "payload", "payload has " + str(size - cursor) + " unindexed trailing bytes" +
                          (previous ? " after " + shown(previous->name) : std::string()));
    out.tensors_ = out.owned_;
    out.retained_ = retained;
    out.header_only_ = options.header_only;
    return out;
}

Checkpoint Checkpoint::open(const std::filesystem::path& path, const CheckpointOptions& options) {
    validate_options(options);
    const auto& limits = options.limits;
    std::vector<std::string> extras;
    extras.reserve(options.extra_files.size());
    for(const auto& extra:options.extra_files) {
        const auto name = display_path(extra);
        if(auto problem = shard_name_problem(name, limits.name_bytes)) throw std::invalid_argument("extra file " + shown(name) + ": " + *problem);
        if(std::find(extras.begin(), extras.end(), name) != extras.end()) throw std::invalid_argument("extra file " + shown(name) + " is listed twice");
        extras.push_back(name);
    }
    std::error_code ignored;
    const fs::path target = fs::is_directory(path, ignored) ? resolve_directory(path) : path;
    const auto source = display_path(target);
    const auto name = display_path(target.filename());
    const auto directory = target.parent_path();
    Checkpoint out;
    out.source_ = target;
    out.header_only_ = options.header_only;
    std::uint64_t total = 0;
    std::set<std::string> members;
    std::unordered_map<std::string_view, std::uint32_t> holder;   // tensor name -> shard; views die with the final move
    // File extents against total_bytes; retained memory was charged by open_shard against what came before.
    auto account = [&](std::uint64_t bytes, std::uint64_t retained) {
        if(bytes > limits.total_bytes - total) throw FormatError(source, "size", "checkpoint files exceed the " + str(limits.total_bytes) + "-byte limit");
        total += bytes;
        out.retained_ += retained;
    };
    auto shard_label = [&](std::uint32_t shard) { return shown(display_path(out.shards_[shard].file().path().filename())); };
    if(ends_with(name, index_suffix)) {
        out.index_.emplace(ReadOnlyFile::open(target, options.follow_symlinks));
        const auto size = out.index_->identity().size;
        if(size == 0 || size > limits.index_bytes)
            throw FormatError(source, "size", "index size " + str(size) + " is not in [1, " + str(limits.index_bytes) + "] bytes");
        account(size, 0);
        const auto text = read_all(*out.index_);   // refuses an index longer than size_t on narrow platforms
        const auto document = json::Document::parse(text, source, {.max_bytes = std::size_t(size), .max_depth = 64, .max_nodes = std::size_t(size)});
        const auto& root = document.root();
        if(root.kind() != json::Kind::object) throw FormatError(source, "index", "index must be a JSON object");
        const bool strict = options.index == IndexPolicy::strict;
        const auto* weight_map = root.find("weight_map");
        if(!weight_map) throw FormatError(source, "index", "index has no weight_map");
        std::vector<std::string> unknown;
        for(const auto& member:root.object()) if(member.key != "metadata" && member.key != "weight_map") unknown.push_back(member.key);
        std::sort(unknown.begin(), unknown.end());
        unknown_keys(unknown, "index", source, strict, out.notes_);
        if(weight_map->kind() != json::Kind::object || weight_map->object().empty())
            throw FormatError(source, "index weight_map", "weight_map must be a non-empty object, not " + describe(*weight_map));
        const auto entries = weight_map->object();
        if(entries.size() > limits.tensors)
            throw FormatError(source, "index weight_map", "weight_map lists " + str(entries.size()) + " tensors, more than " + str(limits.tensors));
        std::set<std::string_view> names;
        for(const auto& entry:entries) {
            if(auto problem = name_problem(entry.key, limits.name_bytes)) throw FormatError(source, "index weight_map", "tensor " + shown(entry.key) + ": " + *problem);
            if(entry.value.kind() != json::Kind::string)
                throw FormatError(source, "index weight_map", "shard for " + shown(entry.key) + " must be a string, not " + describe(entry.value));
            if(auto problem = shard_path_problem(entry.value.string(), limits.name_bytes))
                throw FormatError(source, "index weight_map", "shard " + shown(entry.value.string()) + " for " + shown(entry.key) + " " + *problem);
            names.insert(entry.value.string());
        }
        if(names.size() > limits.shards)
            throw FormatError(source, "index weight_map", str(names.size()) + " shards exceed the " + str(limits.shards) + "-shard limit");
        std::optional<std::uint64_t> total_size;
        if(const auto* metadata = root.find("metadata")) {
            if(metadata->kind() != json::Kind::object) throw FormatError(source, "index metadata", "index metadata must be an object, not " + describe(*metadata));
            std::vector<std::string> extra_keys;
            for(const auto& member:metadata->object())
                if(member.key != "total_size" && (strict || member.key != "total_parameters")) extra_keys.push_back(member.key);
            std::sort(extra_keys.begin(), extra_keys.end());
            unknown_keys(extra_keys, "index metadata", source, strict, out.notes_);
            auto counted = [&](const char* key, std::uint64_t maximum) -> std::optional<std::uint64_t> {
                const auto* value = metadata->find(key);
                if(!value) return std::nullopt;
                // Worded like the Python port: "<source>: index metadata total_size: integer exceeds 64 bits".
                if(value->kind() == json::Kind::integer && value->exceeds_64_bits())
                    throw FormatError(source, "index metadata " + std::string(key), "integer exceeds 64 bits");
                const auto number = value->kind() == json::Kind::integer ? non_negative(*value) : std::nullopt;
                if(!number || *number > maximum)
                    throw FormatError(source, "index metadata", std::string(key) + " must be an integer in [0, " + str(maximum) + "], not " + describe(*value));
                return number;
            };
            total_size = counted("total_size", limits.total_bytes);
            if(!strict) counted("total_parameters", limit_max);
        }
        if(names.size() + extras.size() > limits.shards)
            throw FormatError(source, "index", str(names.size() + extras.size()) + " files exceed the " + str(limits.shards) + "-shard limit");
        out.shards_.reserve(names.size() + extras.size());
        for(const auto shard_name:names) {
            const auto shard = std::uint32_t(out.shards_.size());
            out.shards_.push_back(SafetensorsFile::open_shard(directory / fs::path(std::string(shard_name)), options, shard, out.retained_));
            const auto& file = out.shards_.back();
            account(extent(file), file.retained_bytes());
            for(const auto& tensor:file.tensors()) {
                const auto where = "tensor " + shown(tensor.name);
                if(const auto seen = holder.find(tensor.name); seen != holder.end())
                    throw FormatError(source, where, "appears in shards " + shard_label(seen->second) + " and " + shown(shard_name));
                const auto* declared = weight_map->find(tensor.name);
                if(!declared) throw FormatError(source, where, "shard " + shown(shard_name) + " holds it but weight_map does not list it");
                if(declared->string() != shard_name)
                    throw FormatError(source, where, "weight_map puts it in " + shown(declared->string()) + " but shard " + shown(shard_name) + " holds it");
                holder.emplace(tensor.name, shard);
            }
        }
        std::size_t missing = 0;
        std::string_view first_missing;
        for(const auto& entry:entries) {
            if(holder.contains(entry.key)) continue;
            if(missing++ == 0 || entry.key < first_missing) first_missing = entry.key;
        }
        if(missing)
            throw FormatError(source, "index weight_map", str(missing) + " weight_map tensors are absent from the shard headers, such as " + shown(first_missing));
        std::uint64_t tensor_bytes = 0;
        for(const auto& file:out.shards_)
            for(const auto& tensor:file.tensors()) {
                if(tensor.nbytes() > std::numeric_limits<std::uint64_t>::max() - tensor_bytes)
                    throw std::overflow_error(source + ": tensor byte total overflows 2^64 - 1");   // header_only with extreme limits
                tensor_bytes += tensor.nbytes();
            }
        if(total_size && *total_size != tensor_bytes) {
            const auto message = "total_size " + str(*total_size) + " differs from the " + str(tensor_bytes) + " tensor bytes in the shards";
            if(strict) throw FormatError(source, "index metadata", message);
            out.notes_.push_back("index metadata " + message);
        }
        for(const auto shard_name:names) members.emplace(shard_name);
    } else if(name.size() > file_suffix.size() && ends_with(name, file_suffix)) {
        if(1 + extras.size() > limits.shards)
            throw FormatError(source, "index", str(1 + extras.size()) + " files exceed the " + str(limits.shards) + "-shard limit");
        out.shards_.reserve(1 + extras.size());
        out.shards_.push_back(SafetensorsFile::open_shard(target, options, 0, 0));
        account(extent(out.shards_.back()), out.shards_.back().retained_bytes());
        for(const auto& tensor:out.shards_.back().tensors()) holder.emplace(tensor.name, 0);
        members.insert(name);
    } else {
        throw FormatError(source, "file", "expected a .safetensors file, a .safetensors.index.json index or a model directory");
    }
    for(const auto& extra:extras)
        if(members.contains(extra)) throw std::invalid_argument(source + ": extra file " + shown(extra) + " is already part of the checkpoint");
    for(const auto& extra:extras) {
        const auto shard = std::uint32_t(out.shards_.size());
        out.shards_.push_back(SafetensorsFile::open_shard(directory / fs::path(extra), options, shard, out.retained_));
        const auto& file = out.shards_.back();
        account(extent(file), file.retained_bytes());
        for(const auto& tensor:file.tensors()) {
            if(const auto seen = holder.find(tensor.name); seen != holder.end())
                throw FormatError(source, "tensor " + shown(tensor.name), "appears in " + shard_label(seen->second) + " and extra file " + shown(extra));
            holder.emplace(tensor.name, shard);
            if(holder.size() > limits.tensors) throw FormatError(source, "header", "more than " + str(limits.tensors) + " tensors across files");
        }
    }
    // One record per tensor: move every shard's records into shard-major storage and leave each shard a slice.
    const std::size_t count = holder.size();
    holder.clear();
    if(sizeof(std::uint32_t) * std::uint64_t(count) > limits.retained_bytes - out.retained_)
        throw FormatError(source, "size", "checkpoint headers and tensor records exceed the " + str(limits.retained_bytes) + "-byte retained_bytes limit");
    out.retained_ += sizeof(std::uint32_t) * std::uint64_t(count);
    out.tensors_.reserve(count);   // no reallocation below, so the slices stay valid
    for(auto& file:out.shards_) {
        const std::size_t first = out.tensors_.size();
        std::move(file.owned_.begin(), file.owned_.end(), std::back_inserter(out.tensors_));
        std::vector<TensorInfo>().swap(file.owned_);
        file.tensors_ = std::span<const TensorInfo>(out.tensors_).subspan(first);
    }
    out.by_name_.resize(count);
    std::iota(out.by_name_.begin(), out.by_name_.end(), std::uint32_t(0));
    std::sort(out.by_name_.begin(), out.by_name_.end(), [&](std::uint32_t a, std::uint32_t b) { return out.tensors_[a].name < out.tensors_[b].name; });
    return out;
}
const TensorInfo* Checkpoint::find(std::string_view name) const noexcept {
    const auto it = std::lower_bound(by_name_.begin(), by_name_.end(), name, [&](std::uint32_t i, std::string_view key) { return tensors_[i].name < key; });
    return it != by_name_.end() && tensors_[*it].name == name ? &tensors_[*it] : nullptr;
}
const TensorInfo& Checkpoint::at(std::string_view name) const {
    if(const auto* tensor = find(name)) return *tensor;
    throw std::out_of_range(display_path(source_) + ": no tensor " + shown(name));
}
const TensorInfo& Checkpoint::owned(const TensorInfo& tensor) const {
    if(header_only_) throw std::logic_error(display_path(source_) + ": checkpoint was opened header_only; tensor bytes are unavailable");
    const auto* found = find(tensor.name);
    if(!found || found->shard != tensor.shard || found->begin != tensor.begin || found->end != tensor.end)
        throw std::invalid_argument(display_path(source_) + ": tensor " + shown(tensor.name) + " does not belong to this checkpoint");
    return *found;
}
Checkpoint::View Checkpoint::map(const TensorInfo& tensor) const {
    const auto& t = owned(tensor);
    if(t.begin == t.end) return {};
    const auto& file = shards_[t.shard].file();
    auto owner = file.mapping_owner();
    const auto bytes = file.map();
    return {bytes.subspan(std::size_t(t.begin), std::size_t(t.nbytes())), std::move(owner)};
}
void Checkpoint::read(const TensorInfo& tensor, std::span<std::byte> dst) const {
    const auto& t = owned(tensor);
    if(dst.size() != t.nbytes())
        throw std::invalid_argument("destination of " + str(dst.size()) + " bytes does not match tensor " + shown(t.name) + " of " + str(t.nbytes()) + " bytes");
    if(!dst.empty()) shards_[t.shard].file().read_exact(t.begin, dst);
}
ReadStats Checkpoint::read_chunks(const TensorInfo& tensor, const ChunkConsumer& consume, const ReadOptions& options) const {
    const auto& t = owned(tensor);
    return shards_[t.shard].file().read_chunks(t.begin, t.nbytes(), consume, options);
}
ReadStats Checkpoint::stream_shard(std::uint32_t shard, const TensorConsumer& consume, const ReadOptions& options, std::string* sha256_hex) const {
    if(header_only_) throw std::logic_error(display_path(source_) + ": checkpoint was opened header_only; tensor bytes are unavailable");
    if(shard >= shards_.size()) throw std::out_of_range("shard " + str(shard) + " out of range (" + str(shards_.size()) + " shards)");
    if(!consume) throw std::invalid_argument("tensor consumer is empty");
    const auto& file = shards_[shard];
    std::vector<const TensorInfo*> order;
    order.reserve(file.tensors().size());
    for(const auto& tensor:file.tensors()) if(tensor.end > tensor.begin) order.push_back(&tensor);
    std::sort(order.begin(), order.end(), [](const TensorInfo* a, const TensorInfo* b) { return a->begin < b->begin; });
    Sha256 hasher;
    const std::uint64_t start = sha256_hex ? 0 : file.payload_offset(), size = file.file().identity().size;
    std::size_t next = 0;   // first tensor not yet fully delivered
    const auto stats = file.file().read_chunks(start, size - start, [&](std::span<const std::byte> bytes, std::uint64_t offset) {
        if(sha256_hex) hasher.update(bytes);
        const std::uint64_t end = offset + bytes.size();
        for(std::size_t i = next; i < order.size() && order[i]->begin < end; ++i) {
            const auto& tensor = *order[i];
            const std::uint64_t lo = std::max(offset, tensor.begin), hi = std::min(end, tensor.end);
            if(lo < hi) consume(tensor, bytes.subspan(std::size_t(lo - offset), std::size_t(hi - lo)), lo - tensor.begin);
            if(tensor.end > end) break;
            next = i + 1;
        }
    }, options);
    file.file().verify_unchanged();
    if(sha256_hex) {
        const auto digest = hasher.finish();
        *sha256_hex = to_hex(digest);
    }
    return stats;
}
void Checkpoint::verify_unchanged() const {
    if(index_) index_->verify_unchanged();
    for(const auto& shard:shards_) shard.file().verify_unchanged();
}
std::vector<Checkpoint::FileDigest> Checkpoint::digest_files(const ReadOptions& options) const {
    std::vector<FileDigest> rows;
    rows.reserve(shards_.size() + 1);
    auto add = [&](const ReadOnlyFile& file) {
        rows.push_back({display_path(file.path().filename()), file.identity().size, sha256_file(file, options)});
    };
    if(index_) add(*index_);
    for(const auto& shard:shards_) add(shard.file());
    std::sort(rows.begin(), rows.end(), [](const FileDigest& a, const FileDigest& b) { return a.path < b.path; });
    return rows;
}
void Checkpoint::verify_digests(std::span<const FileDigest> expected, const ReadOptions& options) const {
    for(const auto& row:expected)
        if(!lowercase_hex64(row.sha256)) throw std::invalid_argument("expected digest for " + shown(row.path) + " is not 64 lowercase hex digits");
    std::vector<const ReadOnlyFile*> files;
    files.reserve(shards_.size() + 1);
    if(index_) files.push_back(&*index_);
    for(const auto& shard:shards_) files.push_back(&shard.file());
    struct ByPath {
        bool operator()(const FileDigest* a, const std::string& b) const { return a->path < b; }
        bool operator()(const std::string& a, const FileDigest* b) const { return a < b->path; }
        bool operator()(const FileDigest* a, const FileDigest* b) const { return a->path < b->path; }
    };
    std::vector<const FileDigest*> sorted;
    sorted.reserve(expected.size());
    for(const auto& row:expected) sorted.push_back(&row);
    std::sort(sorted.begin(), sorted.end(), ByPath{});
    std::vector<const FileDigest*> rows(files.size(), nullptr);
    for(std::size_t i = 0; i < files.size(); ++i) {
        const auto path = display_path(files[i]->path().filename());
        const auto range = std::equal_range(sorted.begin(), sorted.end(), path, ByPath{});
        if(range.second - range.first > 1) throw std::invalid_argument("expected digests list " + shown(path) + " more than once");
        const auto display = display_path(files[i]->path());
        if(range.first == range.second) throw FormatError(display, "digest", "absent from the expected digests");
        rows[i] = *range.first;
        if(rows[i]->bytes != files[i]->identity().size)
            throw FormatError(display, "digest", "size " + str(files[i]->identity().size) + " differs from the expected " + str(rows[i]->bytes) + " bytes");
    }
    for(std::size_t i = 0; i < files.size(); ++i) {
        const auto actual = sha256_file(*files[i], options);
        if(actual != rows[i]->sha256)
            throw FormatError(display_path(files[i]->path()), "digest", "sha256 " + actual + " differs from the expected " + rows[i]->sha256);
    }
}
}
