#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace crit::core {
// Malformed or unsupported data (not caller misuse, not an OS failure). what() is
// "<source>: <where>: <reason>"; where is e.g. "byte 1234", "tensor 'x'" or "key 'rope_scaling.factor'".
class FormatError : public std::runtime_error {
public:
    FormatError(std::string source, std::string where, std::string reason);
    const std::string& source() const noexcept;
    const std::string& where() const noexcept;
private:
    struct Detail;
    std::shared_ptr<const Detail> detail_;   // shared so copying the exception never throws
};

namespace json {
// Enforced while parsing; every limit must be positive (std::invalid_argument otherwise).
// Nodes count every JSON value (scalars, arrays, objects); depth counts open containers.
struct Limits {
    std::size_t max_bytes = std::size_t(128) << 20;
    std::uint32_t max_depth = 128;
    std::size_t max_nodes = std::size_t(1) << 24;
    // Opt-in, for metadata such as the config.json files transformers writes with Python's json: the literals
    // NaN, Infinity and -Infinity parse as non-finite reals. A real that overflows (1e400) stays an error.
    bool allow_nonfinite = false;
};
enum class Kind : std::uint8_t { null, boolean, integer, real, string, array, object };
struct Member;
namespace detail { struct DocumentData; struct Parser; }

// View into a Document's immutable node arena; valid while any copy of that Document lives.
// Typed accessors throw FormatError("<source>: byte <offset>: expected ...") on a kind mismatch.
class Value {
public:
    Kind kind() const noexcept { return kind_; }
    std::size_t offset() const noexcept { return offset_; }   // source byte offset of the value's first byte
    bool is_null() const noexcept { return kind_ == Kind::null; }
    bool boolean() const;
    // Integer tokens only (no '.', 'e'); "integer out of range for u64/i64" when the exact value does not fit,
    // "integer exceeds 64 bits" for an exceeds_64_bits() integer. -0 is 0.
    std::uint64_t u64() const;
    std::int64_t i64() const;
    // Integer or real token; finite unless the document was parsed with Limits::allow_nonfinite. Throws
    // "integer exceeds 64 bits" for an exceeds_64_bits() integer.
    double real() const;
    std::string_view string() const;                // unescaped, valid UTF-8 (may contain U+0000)
    std::span<const Value> array() const;
    std::span<const Member> object() const;         // document order; keys are unique
    const Value* find(std::string_view key) const;  // object only; nullptr when absent; O(log n)
    // An integer token outside [-2^63, 2^64): kept so the document loads, but it has no numeric value here.
    bool exceeds_64_bits() const noexcept { return wide_; }
private:
    friend struct detail::Parser;
    const detail::DocumentData* doc_ = nullptr;
    const void* data_ = nullptr;   // children, members or string bytes
    std::uint64_t bits_ = 0;       // integer magnitude, real bits, or element/byte count
    std::size_t offset_ = 0;
    Kind kind_ = Kind::null;
    bool negative_ = false;        // integer sign; boolean value
    bool wide_ = false;            // integer outside [-2^63, 2^64); bits_ is then 0
    [[noreturn]] void mismatch(const char* expected) const;
};
struct Member { std::string key; Value value; };

// A strictly parsed RFC 8259 document: UTF-8 only (no BOM, overlongs, encoded surrogates or code
// points above U+10FFFF), no comments, trailing commas, NaN/Infinity (unless Limits::allow_nonfinite)
// or raw control characters; \u escapes pair surrogates; duplicate (unescaped) keys are rejected at
// every depth; integers in [-2^63, 2^64) are exact and any other is kept as an integer whose numeric
// accessors throw (Value::exceeds_64_bits); reals come from std::from_chars and must be finite
// (underflow becomes a signed zero); only JSON whitespace may surround the root. Iterative, so deep
// input cannot exhaust the stack. Copies share the arena; default-constructed holds a null root.
class Document {
public:
    Document();
    static Document parse(std::string_view text, std::string source_name, const Limits& limits = {});
    // Opens with ReadOnlyFile, rejects files above limits.max_bytes before reading, and verifies
    // the file did not change while it was read. The source name is the path.
    static Document read_file(const std::filesystem::path& path, const Limits& limits = {});
    const Value& root() const noexcept;
    const std::string& source() const noexcept;
private:
    std::shared_ptr<const detail::DocumentData> data_;
    explicit Document(std::shared_ptr<const detail::DocumentData>);
};
}
}
