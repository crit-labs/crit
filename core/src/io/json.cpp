#include "crit/json.h"
#include "crit/file.h"
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace crit::core {
struct FormatError::Detail { std::string source, where; };
FormatError::FormatError(std::string source, std::string where, std::string reason)
    : std::runtime_error(source + ": " + where + ": " + reason),
      detail_(std::make_shared<const Detail>(Detail{std::move(source), std::move(where)})) {}
const std::string& FormatError::source() const noexcept { return detail_->source; }
const std::string& FormatError::where() const noexcept { return detail_->where; }

namespace json {
namespace detail {
struct DocumentData {
    std::string source;
    std::string strings;              // unescaped string values, back to back
    std::vector<Value> values;        // array elements, contiguous per array
    std::vector<Member> members;      // object members, contiguous per object
    std::vector<std::size_t> order;   // per object: member positions sorted by key (parallel to members)
    Value root;
};
}
namespace {
const std::string empty_source;
const Value null_value;
std::string at_byte(std::size_t offset) { return "byte " + std::to_string(offset); }
// Short, printable rendering of a key for diagnostics; never splits a UTF-8 sequence.
std::string shown_key(std::string_view key) {
    std::size_t cut = std::min<std::size_t>(key.size(), 64);
    while(cut < key.size() && cut > 0 && (static_cast<unsigned char>(key[cut]) & 0xC0) == 0x80) --cut;
    std::string out = "'";
    for(char c:key.substr(0, cut)) {
        const auto b = static_cast<unsigned char>(c);
        if(b < 0x20 || b == 0x7F || c == '\'' || c == '\\') {
            static const char hex[] = "0123456789abcdef";
            out += "\\x"; out += hex[b >> 4]; out += hex[b & 15];
        } else out += c;
    }
    return out + (cut < key.size() ? "'..." : "'");
}
void validate_limits(const Limits& limits) {
    if(limits.max_bytes == 0 || limits.max_depth == 0 || limits.max_nodes == 0)
        throw std::invalid_argument("json limits must be positive");
}
bool is_digit(char c) { return c >= '0' && c <= '9'; }
int hex_value(char c) {
    if(c >= '0' && c <= '9') return c - '0';
    if(c >= 'a' && c <= 'f') return c - 'a' + 10;
    if(c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
void append_utf8(std::string& out, std::uint32_t cp) {
    if(cp < 0x80) out += char(cp);
    else if(cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
    else if(cp < 0x10000) {
        out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F));
    } else {
        out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F));
        out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F));
    }
}
}

namespace detail {
struct Parser {
    struct Pending { Value value; std::size_t index = 0; };   // index into strings, values or members
    struct PendingMember { std::string key; std::size_t key_offset = 0; Pending item; };
    struct Frame { Kind kind; std::size_t offset, first; };
    enum class State { value, value_or_close, key_or_close, key, colon, after_value, done };

    std::string_view text;
    const Limits& limits;
    DocumentData& doc;
    std::size_t pos = 0, nodes = 0;
    std::vector<Frame> frames;
    std::vector<Pending> items;          // elements of open arrays
    std::vector<PendingMember> fields;   // members of open objects; the last one's value may still be parsing
    std::vector<std::size_t> value_index, member_index;   // parallel to doc.values / doc.members
    std::size_t root_index = 0;
    bool after_comma = false;

    Parser(std::string_view source_text, const Limits& parse_limits, DocumentData& data)
        : text(source_text), limits(parse_limits), doc(data) {}

    [[noreturn]] void fail(std::size_t at, const std::string& reason) const { throw FormatError(doc.source, at_byte(at), reason); }
    unsigned char byte(std::size_t at) const { return static_cast<unsigned char>(text[at]); }
    void skip_whitespace() {
        while(pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r')) ++pos;
    }
    [[noreturn]] void unexpected() const {
        if(pos >= text.size()) fail(pos, "unexpected end of input");
        const auto b = byte(pos);
        if(b == '/') fail(pos, "comments are not allowed");
        if(text.substr(pos, 3) == "NaN" || text.substr(pos, 8) == "Infinity") fail(pos, "non-finite number literal");
        if(b >= 0x20 && b < 0x7F) fail(pos, std::string("unexpected character '") + char(b) + "'");
        static const char hex[] = "0123456789abcdef";
        fail(pos, std::string("unexpected byte 0x") + hex[b >> 4] + hex[b & 15]);
    }
    void count_node() {
        if(++nodes > limits.max_nodes) fail(pos, "document exceeds the " + std::to_string(limits.max_nodes) + "-node limit");
    }

    // Length of the valid UTF-8 sequence at `at` (lead byte >= 0x80).
    std::size_t utf8(std::size_t at) const {
        const auto b0 = byte(at);
        auto cont = [&](std::size_t i) { return at + i < text.size() && (byte(at + i) & 0xC0) == 0x80; };
        if(b0 < 0xC0) fail(at, "unexpected utf-8 continuation byte");
        if(b0 < 0xC2) fail(at, "overlong utf-8 encoding");
        std::size_t length = b0 < 0xE0 ? 2 : b0 < 0xF0 ? 3 : 4;
        if(b0 >= 0xF8) fail(at, "invalid utf-8 lead byte");
        if(b0 >= 0xF5) fail(at, "utf-8 code point above U+10FFFF");
        for(std::size_t i = 1; i < length; ++i) if(!cont(i)) fail(at, "truncated utf-8 sequence");
        const auto b1 = byte(at + 1);
        if(b0 == 0xE0 && b1 < 0xA0) fail(at, "overlong utf-8 encoding");
        if(b0 == 0xED && b1 >= 0xA0) fail(at, "utf-8 encoded surrogate");
        if(b0 == 0xF0 && b1 < 0x90) fail(at, "overlong utf-8 encoding");
        if(b0 == 0xF4 && b1 >= 0x90) fail(at, "utf-8 code point above U+10FFFF");
        return length;
    }
    std::uint32_t hex4(std::size_t at, std::size_t escape) const {
        if(at + 4 > text.size()) fail(escape, "invalid \\u escape");
        std::uint32_t cp = 0;
        for(std::size_t i = 0; i < 4; ++i) {
            const int h = hex_value(text[at + i]);
            if(h < 0) fail(escape, "invalid \\u escape");
            cp = cp << 4 | std::uint32_t(h);
        }
        return cp;
    }
    // Decodes the string starting at the opening quote and appends its bytes to out.
    void decode_string(std::string& out) {
        const std::size_t start = pos++;
        for(;;) {
            const std::size_t run = pos;
            while(pos < text.size()) {
                const auto b = byte(pos);
                if(b == '"' || b == '\\' || b < 0x20 || b >= 0x80) break;
                ++pos;
            }
            out.append(text.data() + run, pos - run);
            if(pos >= text.size()) fail(start, "unterminated string");
            const auto b = byte(pos);
            if(b == '"') { ++pos; return; }
            if(b < 0x20) fail(pos, "raw control character in string");
            if(b >= 0x80) {
                const auto length = utf8(pos);
                out.append(text.data() + pos, length);
                pos += length;
                continue;
            }
            const std::size_t escape = pos++;
            if(pos >= text.size()) fail(start, "unterminated string");
            switch(text[pos++]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                std::uint32_t cp = hex4(pos, escape);
                pos += 4;
                if(cp >= 0xDC00 && cp <= 0xDFFF) fail(escape, "unpaired surrogate escape");
                if(cp >= 0xD800 && cp <= 0xDBFF) {
                    if(pos + 1 >= text.size() || text[pos] != '\\' || text[pos + 1] != 'u') fail(escape, "unpaired surrogate escape");
                    const std::uint32_t low = hex4(pos + 2, pos);
                    if(low < 0xDC00 || low > 0xDFFF) fail(escape, "unpaired surrogate escape");
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    pos += 6;
                }
                append_utf8(out, cp);
                break;
            }
            default: fail(escape, "invalid escape");
            }
        }
    }
    Pending number() {
        const std::size_t start = pos;
        const bool negative = text[pos] == '-';
        if(negative) ++pos;
        if(pos >= text.size() || !is_digit(text[pos])) {
            if(text.substr(pos, 8) == "Infinity") {
                if(limits.allow_nonfinite) return nonfinite(start);
                fail(start, "non-finite number literal");
            }
            fail(start, "invalid number");
        }
        const std::size_t digits = pos;
        if(text[pos] == '0') { ++pos; if(pos < text.size() && is_digit(text[pos])) fail(start, "leading zero in number"); }
        else while(pos < text.size() && is_digit(text[pos])) ++pos;
        const std::size_t integer_end = pos;
        bool integer = true;
        if(pos < text.size() && text[pos] == '.') {
            ++pos; integer = false;
            if(pos >= text.size() || !is_digit(text[pos])) fail(start, "invalid number");
            while(pos < text.size() && is_digit(text[pos])) ++pos;
        }
        const std::size_t fraction_end = pos;
        if(pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            ++pos; integer = false;
            if(pos < text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
            if(pos >= text.size() || !is_digit(text[pos])) fail(start, "invalid number");
            while(pos < text.size() && is_digit(text[pos])) ++pos;
        }
        Pending p;
        p.value.offset_ = start;
        if(integer) {
            // [-2^63, 2^64) is exact; any other integer stays an integer without a value (exceeds_64_bits).
            std::uint64_t magnitude = 0;
            bool wide = false;
            for(std::size_t i = digits; i < integer_end && !wide; ++i) {
                const auto d = std::uint64_t(text[i] - '0');
                if(magnitude > (std::numeric_limits<std::uint64_t>::max() - d) / 10) wide = true;
                else magnitude = magnitude * 10 + d;
            }
            if(negative && magnitude > std::uint64_t(1) << 63) wide = true;
            p.value.kind_ = Kind::integer;
            p.value.negative_ = negative;
            p.value.wide_ = wide;
            p.value.bits_ = wide ? 0 : magnitude;
            return p;
        }
        double value = 0;
        const auto [end, error] = std::from_chars(text.data() + start, text.data() + pos, value, std::chars_format::general);
        if(end != text.data() + pos || (error != std::errc() && error != std::errc::result_out_of_range)) fail(start, "invalid number");
        if(error == std::errc::result_out_of_range) {
            // Decimal exponent of the leading significant digit decides underflow (to signed zero) vs overflow.
            std::int64_t exponent = 0;
            if(fraction_end < pos) {
                std::size_t i = fraction_end + 1;
                const bool minus = text[i] == '-';
                if(text[i] == '+' || text[i] == '-') ++i;
                for(; i < pos; ++i) exponent = std::min<std::int64_t>(exponent * 10 + (text[i] - '0'), std::int64_t(1) << 40);
                if(minus) exponent = -exponent;
            }
            std::int64_t lead = std::int64_t(integer_end - digits) - 1;
            for(std::size_t i = digits; i < fraction_end; ++i) {
                if(text[i] == '.') continue;
                if(text[i] != '0') break;
                --lead;
            }
            if(lead + exponent >= 0) fail(start, "real number out of range");
            value = negative ? -0.0 : 0.0;
        }
        if(!std::isfinite(value)) fail(start, "real number out of range");
        p.value.kind_ = Kind::real;
        p.value.bits_ = std::bit_cast<std::uint64_t>(value);
        return p;
    }
    // NaN, Infinity or -Infinity (pos at 'N' or 'I'; start at the sign) under Limits::allow_nonfinite.
    Pending nonfinite(std::size_t start) {
        Pending p;
        p.value.offset_ = start;
        p.value.kind_ = Kind::real;
        double value = std::numeric_limits<double>::quiet_NaN();
        if(text.substr(pos, 8) == "Infinity") {
            value = text[start] == '-' ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
            pos += 8;
        } else if(start == pos && text.substr(pos, 3) == "NaN") pos += 3;
        else unexpected();
        p.value.bits_ = std::bit_cast<std::uint64_t>(value);
        return p;
    }
    Pending literal() {
        Pending p;
        p.value.offset_ = pos;
        if(text.substr(pos, 4) == "true") { p.value.kind_ = Kind::boolean; p.value.negative_ = true; pos += 4; }
        else if(text.substr(pos, 5) == "false") { p.value.kind_ = Kind::boolean; pos += 5; }
        else if(text.substr(pos, 4) == "null") pos += 4;
        else fail(pos, "invalid literal");
        return p;
    }
    void attach(Pending p, State& state) {
        state = State::after_value;
        if(frames.empty()) { doc.root = p.value; root_index = p.index; return; }
        if(frames.back().kind == Kind::array) items.push_back(p);
        else fields.back().item = p;
    }
    Pending close_array() {
        const Frame frame = frames.back();
        frames.pop_back();
        Pending p;
        p.value.kind_ = Kind::array;
        p.value.offset_ = frame.offset;
        p.value.bits_ = items.size() - frame.first;
        p.index = doc.values.size();
        for(std::size_t i = frame.first; i < items.size(); ++i) {
            doc.values.push_back(items[i].value);
            value_index.push_back(items[i].index);
        }
        items.resize(frame.first);
        return p;
    }
    Pending close_object() {
        const Frame frame = frames.back();
        const std::size_t count = fields.size() - frame.first, begin = doc.members.size();
        // Sort positions by unescaped key; equal neighbours are duplicates. Report the first in document order.
        doc.order.resize(begin + count);
        const auto order = doc.order.begin() + std::ptrdiff_t(begin);
        for(std::size_t i = 0; i < count; ++i) order[std::ptrdiff_t(i)] = i;
        const auto* base = fields.data() + frame.first;
        std::sort(order, order + std::ptrdiff_t(count), [&](std::size_t a, std::size_t b) {
            const int c = base[a].key.compare(base[b].key);
            return c != 0 ? c < 0 : a < b;
        });
        std::size_t duplicate = count;
        for(std::size_t i = 1; i < count; ++i)
            if(base[order[std::ptrdiff_t(i)]].key == base[order[std::ptrdiff_t(i - 1)]].key)
                duplicate = std::min(duplicate, order[std::ptrdiff_t(i)]);
        if(duplicate < count) fail(base[duplicate].key_offset, "duplicate key " + shown_key(base[duplicate].key));
        frames.pop_back();
        for(std::size_t i = frame.first; i < fields.size(); ++i) {
            doc.members.push_back({std::move(fields[i].key), fields[i].item.value});
            member_index.push_back(fields[i].item.index);
        }
        fields.resize(frame.first);
        Pending p;
        p.value.kind_ = Kind::object;
        p.value.offset_ = frame.offset;
        p.value.bits_ = count;
        p.index = begin;
        return p;
    }
    void open(Kind kind) {
        if(frames.size() >= limits.max_depth) fail(pos, "nesting exceeds the depth limit of " + std::to_string(limits.max_depth));
        frames.push_back({kind, pos, kind == Kind::array ? items.size() : fields.size()});
        ++pos;
    }
    void run() {
        State state = State::value;
        while(state != State::done) {
            skip_whitespace();
            if(state == State::after_value) {
                if(frames.empty()) { state = State::done; break; }
                if(pos >= text.size()) unexpected();
                const char c = text[pos];
                const bool array = frames.back().kind == Kind::array;
                if(c == ',') { ++pos; after_comma = true; state = array ? State::value : State::key; }
                else if(array && c == ']') { ++pos; attach(close_array(), state); }
                else if(!array && c == '}') { ++pos; attach(close_object(), state); }
                else if(array) fail(pos, "expected ',' or ']'");
                else fail(pos, "expected ',' or '}'");
                continue;
            }
            if(pos >= text.size()) unexpected();
            const char c = text[pos];
            if(state == State::value_or_close && c == ']') { ++pos; attach(close_array(), state); continue; }
            if(state == State::key_or_close && c == '}') { ++pos; attach(close_object(), state); continue; }
            if(state == State::key_or_close || state == State::key) {
                if(c == '"') {
                    fields.push_back({std::string(), pos, {}});
                    decode_string(fields.back().key);
                    after_comma = false;
                    state = State::colon;
                } else if(after_comma && c == '}') fail(pos, "trailing comma");
                else fail(pos, "expected a string key");
                continue;
            }
            if(state == State::colon) {
                if(c != ':') fail(pos, "expected ':'");
                ++pos;
                state = State::value;
                continue;
            }
            // A value (root, array element or member value).
            if(after_comma && c == ']') fail(pos, "trailing comma");
            after_comma = false;
            count_node();
            if(c == '[') { open(Kind::array); state = State::value_or_close; continue; }
            if(c == '{') { open(Kind::object); state = State::key_or_close; continue; }
            if(c == '"') {
                Pending p;
                p.value.kind_ = Kind::string;
                p.value.offset_ = pos;
                p.index = doc.strings.size();
                decode_string(doc.strings);
                p.value.bits_ = doc.strings.size() - p.index;
                attach(std::move(p), state);
            } else if(c == '-' || is_digit(c)) attach(number(), state);
            else if(c == 't' || c == 'f' || c == 'n') attach(literal(), state);
            else if(limits.allow_nonfinite && (c == 'N' || c == 'I')) attach(nonfinite(pos), state);
            else unexpected();
        }
        if(pos != text.size()) fail(pos, "unexpected data after the root value");
    }
    void link(Value& value, std::size_t index) const {
        value.doc_ = &doc;
        if(value.kind_ == Kind::array) value.data_ = doc.values.data() + index;
        else if(value.kind_ == Kind::object) value.data_ = doc.members.data() + index;
        else if(value.kind_ == Kind::string) value.data_ = doc.strings.data() + index;
    }
    void finish() {
        // Release growth slack before views are linked; nothing reallocates afterwards.
        doc.strings.shrink_to_fit();
        doc.values.shrink_to_fit();
        doc.members.shrink_to_fit();
        doc.order.shrink_to_fit();
        for(std::size_t i = 0; i < doc.values.size(); ++i) link(doc.values[i], value_index[i]);
        for(std::size_t i = 0; i < doc.members.size(); ++i) link(doc.members[i].value, member_index[i]);
        link(doc.root, root_index);
    }
};
}

namespace {
[[noreturn]] void value_error(const detail::DocumentData* doc, std::size_t offset, const std::string& reason) {
    throw FormatError(doc ? doc->source : std::string("json"), at_byte(offset), reason);
}
}
void Value::mismatch(const char* expected) const { value_error(doc_, offset_, std::string("expected ") + expected); }
bool Value::boolean() const {
    if(kind_ != Kind::boolean) mismatch("boolean");
    return negative_;
}
std::uint64_t Value::u64() const {
    if(kind_ != Kind::integer) mismatch("integer");
    if(wide_) value_error(doc_, offset_, "integer exceeds 64 bits");
    if(negative_ && bits_ != 0) value_error(doc_, offset_, "integer out of range for u64");
    return bits_;
}
std::int64_t Value::i64() const {
    if(kind_ != Kind::integer) mismatch("integer");
    if(wide_) value_error(doc_, offset_, "integer exceeds 64 bits");
    constexpr auto max = std::uint64_t(std::numeric_limits<std::int64_t>::max());
    if(bits_ > max + (negative_ ? 1 : 0)) value_error(doc_, offset_, "integer out of range for i64");
    if(!negative_) return std::int64_t(bits_);
    return bits_ == max + 1 ? std::numeric_limits<std::int64_t>::min() : -std::int64_t(bits_);
}
double Value::real() const {
    if(kind_ == Kind::integer) {
        if(wide_) value_error(doc_, offset_, "integer exceeds 64 bits");
        return negative_ ? -double(bits_) : double(bits_);
    }
    if(kind_ != Kind::real) mismatch("number");
    return std::bit_cast<double>(bits_);
}
std::string_view Value::string() const {
    if(kind_ != Kind::string) mismatch("string");
    return {static_cast<const char*>(data_), std::size_t(bits_)};
}
std::span<const Value> Value::array() const {
    if(kind_ != Kind::array) mismatch("array");
    return {static_cast<const Value*>(data_), std::size_t(bits_)};
}
std::span<const Member> Value::object() const {
    if(kind_ != Kind::object) mismatch("object");
    return {static_cast<const Member*>(data_), std::size_t(bits_)};
}
const Value* Value::find(std::string_view key) const {
    const auto members = object();
    if(members.size() <= 8) {
        for(const auto& m:members) if(m.key == key) return &m.value;
        return nullptr;
    }
    const auto* order = doc_->order.data() + (members.data() - doc_->members.data());
    const auto* end = order + members.size();
    const auto* it = std::lower_bound(order, end, key, [&](std::size_t i, std::string_view k) { return members[i].key < k; });
    return it != end && members[*it].key == key ? &members[*it].value : nullptr;
}

Document::Document() = default;
Document::Document(std::shared_ptr<const detail::DocumentData> data) : data_(std::move(data)) {}
const Value& Document::root() const noexcept { return data_ ? data_->root : null_value; }
const std::string& Document::source() const noexcept { return data_ ? data_->source : empty_source; }
Document Document::parse(std::string_view text, std::string source_name, const Limits& limits) {
    validate_limits(limits);
    auto data = std::make_shared<detail::DocumentData>();
    data->source = std::move(source_name);
    if(text.size() > limits.max_bytes)
        throw FormatError(data->source, "size", std::to_string(text.size()) + " bytes exceeds the " + std::to_string(limits.max_bytes) + "-byte limit");
    if(text.substr(0, 3) == "\xEF\xBB\xBF") throw FormatError(data->source, at_byte(0), "byte order mark is not allowed");
    detail::Parser parser(text, limits, *data);
    parser.run();
    parser.finish();
    return Document(std::move(data));
}
Document Document::read_file(const std::filesystem::path& path, const Limits& limits) {
    validate_limits(limits);
    const auto file = ReadOnlyFile::open(path);
    const auto size = file.identity().size;
    if(size > limits.max_bytes)
        throw FormatError(display_path(path), "size", std::to_string(size) + " bytes exceeds the " + std::to_string(limits.max_bytes) + "-byte limit");
    std::string text(std::size_t(size), '\0');
    file.read_exact(0, std::as_writable_bytes(std::span<char>(text.data(), text.size())));
    file.verify_unchanged();
    return parse(text, display_path(path), limits);
}
}
}
