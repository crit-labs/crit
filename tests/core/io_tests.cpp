// Core files: strict JSON, SHA-256, read-only files (identity, exact and chunked reads, mapping)
// and host memory admission. Every fixture is generated at runtime in a temporary directory.
#include <crit/file.h>
#include <crit/host_memory.h>
#include <crit/json.h>
#include <crit/sha256.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <sys/vfs.h>
#endif
using namespace crit::core;
namespace fs = std::filesystem;
#if defined(__GNUC__) || defined(__clang__)
// Sanitizer builds: an undefined-behaviour report fails the case instead of only printing.
extern "C" const char* __ubsan_default_options() { return "halt_on_error=1:print_stacktrace=1"; }
#endif
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
// Like rejects, and the exception must be an E.
template<class E, class F> void rejects_as(F action, std::string_view message) {
    rejects([&] {
        try { action(); } catch(const E&) { throw; } catch(const std::exception& e) {
            throw std::runtime_error("wrong exception type: " + std::string(e.what()));
        }
    }, message);
}
std::span<const std::byte> bytes_of(std::string_view text) { return std::as_bytes(std::span<const char>(text.data(), text.size())); }
std::string text_of(std::span<const std::byte> bytes) { return {reinterpret_cast<const char*>(bytes.data()), bytes.size()}; }

struct TempDir {
    fs::path path;
    TempDir() {
        std::random_device device;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("crit-core-io-" + std::to_string(stamp) + "-" + std::to_string(device()));
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
std::string random_bytes(std::size_t size, unsigned seed) {
    std::mt19937 random(seed);
    std::string out(size, '\0');
    for(auto& c:out) c = char(random() & 0xFF);
    return out;
}

// ---------------------------------------------------------------------------------------------- JSON
json::Document parse(std::string_view text, const json::Limits& limits = {}) { return json::Document::parse(text, "fixture.json", limits); }

void json_valid() {
    const auto doc = parse("  {\"a\" : [1, 2.5, \"x\", true, false, null], \"b\": {}, \"c\": []}  \n\t\r");
    const auto& root = doc.root();
    require(root.kind() == json::Kind::object && root.offset() == 2 && doc.source() == "fixture.json", "root object");
    const auto members = root.object();
    require(members.size() == 3 && members[0].key == "a" && members[1].key == "b" && members[2].key == "c", "document order");
    const auto a = root.find("a")->array();
    require(a.size() == 6 && a[0].u64() == 1 && a[1].real() == 2.5 && a[2].string() == "x", "array values");
    require(a[3].boolean() && !a[4].boolean() && a[5].is_null() && a[5].kind() == json::Kind::null, "literals");
    require(a[0].kind() == json::Kind::integer && a[1].kind() == json::Kind::real && a[0].real() == 1.0, "number kinds");
    require(root.find("b")->object().empty() && root.find("c")->array().empty() && !root.find("d"), "empty containers");
    require(a[2].offset() == std::string_view("  {\"a\" : [1, 2.5, ").size(), "value offsets");
    for(const char* text:{"0", "-0", "\"\"", "[]", "{}", "true", "false", "null", " 7 ", "{}        ", "[[[]]]", "\"\\u0000\""})
        (void)parse(text);
    require(parse("\"\\u0000\"").root().string() == std::string_view("\0", 1), "escaped NUL kept");
    // Copies share the arena and keep views valid after the original is gone.
    json::Value kept;
    std::string_view view;
    {
        auto original = parse("{\"k\":\"value\"}");
        const json::Document copy = original;
        original = json::Document();
        kept = copy.root();
        view = copy.root().find("k")->string();
        require(view == "value" && kept.find("k") != nullptr, "copied document");
        json::Document moved = std::move(original);
        require(moved.root().is_null() && moved.source().empty(), "default document is a null root");
    }
    const json::Document fresh;
    require(fresh.root().is_null() && fresh.source().empty(), "default document");
}

void json_rejects() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"", "unexpected end of input"}, {"   ", "unexpected end of input"}, {"\xEF\xBB\xBF{}", "byte order mark"},
        {"{\"a\":1,\"a\":2}", "duplicate key 'a'"}, {"{\"x\":[{\"y\":{\"k\":1,\"z\":0,\"k\":2}}]}", "duplicate key 'k'"},
        {"{\"a\":1,\"\\u0061\":2}", "duplicate key 'a'"}, {"[{\"b\":1},{\"b\":1,\"b\":1}]", "duplicate key 'b'"},
        {"NaN", "non-finite"}, {"Infinity", "non-finite"}, {"-Infinity", "non-finite"}, {"[NaN]", "non-finite"},
        {"{\"a\":Infinity}", "non-finite"}, {"nan", "invalid literal"}, {"// c\n{}", "comments are not allowed"},
        {"[1,/*x*/2]", "comments are not allowed"}, {"{}/**/", "unexpected data after the root value"},
        {"[1,]", "trailing comma"}, {"{\"a\":1,}", "trailing comma"}, {"[,]", "unexpected character ','"},
        {"[1 2]", "expected ',' or ']'"}, {"{\"a\":1 \"b\":2}", "expected ',' or '}'"}, {"{\"a\" 1}", "expected ':'"},
        {"{1:2}", "expected a string key"}, {"{'a':1}", "expected a string key"}, {"['a']", "unexpected character '''"},
        {"\"abc", "unterminated string"}, {"\"a\\", "unterminated string"}, {std::string("\"a\x01" "b\""), "raw control character"},
        {"\"a\tb\"", "raw control character"}, {"\"a\nb\"", "raw control character"}, {"\"\\x41\"", "invalid escape"},
        {"\"\\U0041\"", "invalid escape"}, {"\"\\u12G4\"", "invalid \\u escape"}, {"\"\\u12\"", "invalid \\u escape"},
        {"\"\\uD800\"", "unpaired surrogate escape"}, {"\"\\uDC00\"", "unpaired surrogate escape"},
        {"\"\\uD800\\u0041\"", "unpaired surrogate escape"}, {"\"\\uD800x\"", "unpaired surrogate escape"},
        {"\"\\uD800\\uD800\"", "unpaired surrogate escape"}, {"\"\\uDBFF\\uE000\"", "unpaired surrogate escape"},
        {"1e400", "real number out of range"},
        {"-1e400", "real number out of range"}, {"1.5e99999999999999999999", "real number out of range"},
        {"01", "leading zero"}, {"-01", "leading zero"}, {"00", "leading zero"}, {"1.", "invalid number"},
        {"1.e5", "invalid number"}, {".5", "unexpected character '.'"}, {"+1", "unexpected character '+'"},
        {"1e", "invalid number"}, {"1e+", "invalid number"}, {"-", "invalid number"}, {"- 1", "invalid number"},
        {"0x10", "unexpected data after the root value"}, {"1.5.3", "unexpected data after the root value"},
        {"tru", "invalid literal"}, {"True", "unexpected character 'T'"}, {"nul", "invalid literal"},
        {"falsey", "unexpected data after the root value"}, {"{} {}", "unexpected data after the root value"},
        {"1 2", "unexpected data after the root value"}, {std::string("\0", 1), "unexpected byte 0x00"},
        {std::string("[1]\0", 4), "unexpected data after the root value"}, {"\f[]", "unexpected byte 0x0c"},
        {"[1,\v2]", "unexpected byte 0x0b"}, {"\xC2\xA0[]", "unexpected byte 0xc2"}, {"\xC3\xA9", "unexpected byte 0xc3"},
        {"{\"a\":1", "unexpected end of input"}, {"[", "unexpected end of input"}, {"[1", "unexpected end of input"},
        {"{\"a\"", "unexpected end of input"}, {"{\"a\":", "unexpected end of input"}, {"]", "unexpected character ']'"},
        {"}", "unexpected character '}'"}, {"{", "unexpected end of input"}, {"[}", "unexpected character '}'"},
        {"{]", "expected a string key"}, {"[1}", "expected ',' or ']'"}, {"{\"a\":1]", "expected ',' or '}'"},
    };
    for(const auto& [text, reason]:cases) rejects_as<FormatError>([&] { (void)parse(text); }, reason);
    // Diagnostics name the source and the byte where the problem starts.
    try { (void)parse("{\"a\":1,\n \"a\":2}"); throw std::logic_error("duplicate accepted"); }
    catch(const FormatError& e) {
        require(e.source() == "fixture.json" && e.where() == "byte 9", "duplicate key location");
        require(std::string(e.what()) == "fixture.json: byte 9: duplicate key 'a'", "formatted diagnostic");
        const FormatError copy = e;
        require(copy.where() == e.where() && std::string(copy.what()) == e.what(), "copied format error");
    }
    try { (void)parse("[1, 2, oops]"); } catch(const FormatError& e) { require(e.where() == "byte 7", "syntax location"); }
    // A long key is shortened in diagnostics without splitting a UTF-8 sequence; control bytes are escaped.
    const std::string key = std::string(63, 'k') + "\xC3\xA9" + std::string(40, 'z');
    rejects_as<FormatError>([&] { (void)parse("{\"" + key + "\":1,\"" + key + "\":2}"); }, "duplicate key '" + std::string(63, 'k') + "'...");
    rejects_as<FormatError>([&] { (void)parse("{\"a\\u0001'\":1,\"a\\u0001'\":2}"); }, "duplicate key 'a\\x01\\x27'");
}

// Independent UTF-8 validity: generic structure first, then the code point range rules.
bool reference_utf8(std::string_view s) {
    for(std::size_t i = 0; i < s.size();) {
        const auto b = static_cast<unsigned char>(s[i]);
        if(b < 0x80) { ++i; continue; }
        std::size_t n = 0;
        std::uint32_t cp = 0, minimum = 0;
        if((b & 0xE0) == 0xC0) { n = 1; cp = b & 0x1Fu; minimum = 0x80; }
        else if((b & 0xF0) == 0xE0) { n = 2; cp = b & 0x0Fu; minimum = 0x800; }
        else if((b & 0xF8) == 0xF0) { n = 3; cp = b & 0x07u; minimum = 0x10000; }
        else return false;
        for(std::size_t k = 1; k <= n; ++k) {
            if(i + k >= s.size() || (static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
            cp = cp << 6 | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        }
        if(cp < minimum || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += n + 1;
    }
    return true;
}
bool accepts(std::string_view text) {
    try { (void)parse(text); return true; } catch(const FormatError&) { return false; }
}
void json_utf8() {
    const std::vector<std::pair<std::string, std::string>> rejected = {
        {"\xC0\xAF", "overlong utf-8 encoding"}, {"\xC1\xBF", "overlong utf-8 encoding"}, {"\xE0\x80\xAF", "overlong utf-8 encoding"},
        {"\xE0\x9F\xBF", "overlong utf-8 encoding"}, {"\xF0\x80\x80\xAF", "overlong utf-8 encoding"},
        {"\xF0\x8F\xBF\xBF", "overlong utf-8 encoding"}, {"\xED\xA0\x80", "utf-8 encoded surrogate"},
        {"\xED\xBF\xBF", "utf-8 encoded surrogate"}, {"\xF4\x90\x80\x80", "above U+10FFFF"}, {"\xF5\x80\x80\x80", "above U+10FFFF"},
        {"\xF7\xBF\xBF\xBF", "above U+10FFFF"}, {"\xF8\x88\x80\x80\x80", "invalid utf-8 lead byte"}, {"\xFF", "invalid utf-8 lead byte"},
        {"\x80", "unexpected utf-8 continuation byte"}, {"\xBF", "unexpected utf-8 continuation byte"}, {"\xC3", "truncated utf-8 sequence"},
        {"\xE2\x82", "truncated utf-8 sequence"}, {"\xF0\x9F\x98", "truncated utf-8 sequence"}, {"\xC3\x41", "truncated utf-8 sequence"},
    };
    for(const auto& [content, reason]:rejected) {
        rejects_as<FormatError>([&] { (void)parse("\"" + content + "\""); }, reason);
        rejects_as<FormatError>([&] { (void)parse("{\"" + content + "\":1}"); }, reason);   // keys too
    }
    for(const char* valid:{"\xC2\x80", "\xDF\xBF", "\xE0\xA0\x80", "\xED\x9F\xBF", "\xEE\x80\x80", "\xEF\xBF\xBF", "\xF0\x90\x80\x80", "\xF4\x8F\xBF\xBF"})
        require(parse(std::string("\"") + valid + "\"").root().string() == valid, "valid utf-8 kept byte for byte");
    // Exhaustive two-byte sequences and broad three- and four-byte sweeps against the reference decoder.
    std::size_t checked = 0;
    auto check = [&](std::string content) {
        require(accepts("\"" + content + "\"") == reference_utf8(content), "utf-8 verdict differs from the reference");
        ++checked;
    };
    for(unsigned b0 = 0x80; b0 <= 0xFF; ++b0)
        for(unsigned b1 = 0; b1 <= 0xFF; ++b1) check({char(b0), char(b1)});
    // Three- and four-byte leads: every second byte (where the overlong, surrogate and U+10FFFF rules
    // live) with the continuation-range edges for the remaining bytes.
    const unsigned edges[] = {0x22, 0x7F, 0x80, 0xBF, 0xC0};
    for(unsigned b0 = 0xE0; b0 <= 0xF7; ++b0)
        for(unsigned b1 = 0; b1 <= 0xFF; ++b1)
            for(const unsigned b2:edges) {
                if(b0 < 0xF0) check({char(b0), char(b1), char(b2)});
                else for(const unsigned b3:{0x7Fu, 0x80u, 0xBFu}) check({char(b0), char(b1), char(b2), char(b3)});
            }
    require(checked > 80000, "utf-8 sweep ran");
}

void json_strings() {
    const auto doc = parse("[\"\\\"\\\\\\/\\b\\f\\n\\r\\t\", \"\\u0041\\u00e9\\u20AC\\uFFFF\", \"\\uD83D\\uDE00\", \"\\udbff\\udfff\","
                           " \"plain \xC3\xA9 \xF0\x9F\x98\x80\", \"\\u0000x\", \"\"]");
    const auto a = doc.root().array();
    require(a[0].string() == "\"\\/\b\f\n\r\t", "simple escapes");
    require(a[1].string() == "A\xC3\xA9\xE2\x82\xAC\xEF\xBF\xBF", "BMP escapes");
    require(a[2].string() == "\xF0\x9F\x98\x80", "surrogate pair escape");
    require(a[3].string() == "\xF4\x8F\xBF\xBF", "highest code point escape");
    require(a[4].string() == "plain \xC3\xA9 \xF0\x9F\x98\x80", "raw UTF-8");
    require(a[5].string() == std::string_view("\0x", 2) && a[6].string().empty(), "NUL and empty strings");
    // Keys compare unescaped, both for duplicates and for find().
    const auto keys = parse("{\"\\u006bey\":1, \"t\\tab\":2, \"\\ud83d\\ude00\":3}");
    require(keys.root().find("key")->u64() == 1 && keys.root().find("t\tab")->u64() == 2, "unescaped keys");
    require(keys.root().find("\xF0\x9F\x98\x80")->u64() == 3 && keys.root().object()[0].key == "key", "astral keys");
    rejects_as<FormatError>([] { (void)parse("{\"\\ud83d\\ude00\":1, \"\xF0\x9F\x98\x80\":2}"); }, "duplicate key");
}

void json_numbers() {
    const auto doc = parse("[0, -0, 18446744073709551615, 9223372036854775807, -9223372036854775808, -9223372036854775809,"
                           " 1e2, 1E+2, 1e-2, -0.0, 0.1, 1e-400, -1e-400, 4.9e-324, 1.7976931348623157e308, -18446744073709551615,"
                           " 9007199254740993, 2.5E-3, 0e0, 0.0e-0]");
    const auto a = doc.root().array();
    require(a[0].u64() == 0 && a[0].i64() == 0 && a[1].u64() == 0 && a[1].i64() == 0, "zero and negative zero integers");
    require(a[1].kind() == json::Kind::integer && std::signbit(a[1].real()), "-0 stays an integer token with a sign");
    require(a[2].u64() == std::numeric_limits<std::uint64_t>::max(), "u64 maximum");
    rejects_as<FormatError>([&] { (void)a[2].i64(); }, "integer out of range for i64");
    require(a[3].i64() == std::numeric_limits<std::int64_t>::max() && a[3].u64() == 9223372036854775807ull, "i64 maximum");
    require(a[4].i64() == std::numeric_limits<std::int64_t>::min(), "i64 minimum");
    rejects_as<FormatError>([&] { (void)a[4].u64(); }, "integer out of range for u64");
    require(!a[4].exceeds_64_bits() && a[5].exceeds_64_bits(), "-2^63 fits, -2^63 - 1 does not");
    rejects_as<FormatError>([&] { (void)a[5].i64(); }, "integer exceeds 64 bits");
    require(a[6].kind() == json::Kind::real && a[6].real() == 100.0 && a[7].real() == 100.0 && a[8].real() == 0.01, "exponent forms are reals");
    rejects_as<FormatError>([&] { (void)a[6].u64(); }, "expected integer");
    rejects_as<FormatError>([&] { (void)a[6].i64(); }, "expected integer");
    require(a[9].real() == 0.0 && std::signbit(a[9].real()) && a[10].real() == 0.1, "real zero and fraction");
    require(a[11].real() == 0.0 && !std::signbit(a[11].real()) && std::signbit(a[12].real()), "underflow becomes a signed zero");
    require(a[13].real() > 0 && a[13].real() < 1e-323, "smallest subnormal");
    require(a[14].real() == std::numeric_limits<double>::max(), "largest finite real");
    require(a[15].kind() == json::Kind::integer && a[15].exceeds_64_bits(), "a negative 64-bit magnitude below -2^63 is kept without a value");
    rejects_as<FormatError>([&] { (void)a[15].i64(); }, "integer exceeds 64 bits");
    rejects_as<FormatError>([&] { (void)a[15].real(); }, "integer exceeds 64 bits");
    require(!a[2].exceeds_64_bits() && !a[0].exceeds_64_bits() && !a[6].exceeds_64_bits(), "in-range integers and reals fit");
    require(a[16].u64() == 9007199254740993ull, "integers above 2^53 stay exact");
    require(a[17].real() == 2.5e-3 && a[18].real() == 0.0 && a[19].real() == 0.0, "more real forms");
    rejects_as<FormatError>([&] { (void)doc.root().array()[0].string(); }, "expected string");
    rejects_as<FormatError>([] { (void)parse("\"1\"").root().real(); }, "expected number");
    // Integers outside [-2^63, 2^64) load anywhere, keep their place, and refuse every numeric accessor.
    const std::string digits(5000, '7');
    const auto wide = parse("{\"u\":18446744073709551616,\"n\":-18446744073709551616,\"x\":[123456789012345678901234567890,1],"
                            "\"z\":-9223372036854775809,\"long\":" + digits + ",\"minus\":-" + digits + ",\"max\":18446744073709551615}");
    for(const char* key:{"u", "n", "z", "long", "minus"}) {
        const auto& value = *wide.root().find(key);
        require(value.kind() == json::Kind::integer && value.exceeds_64_bits(), std::string("wide integer ") + key);
        rejects_as<FormatError>([&] { (void)value.u64(); }, "integer exceeds 64 bits");
        rejects_as<FormatError>([&] { (void)value.i64(); }, "integer exceeds 64 bits");
        rejects_as<FormatError>([&] { (void)value.real(); }, "integer exceeds 64 bits");
    }
    const auto items = wide.root().find("x")->array();
    require(items.size() == 2 && items[0].exceeds_64_bits() && items[1].u64() == 1 && !items[1].exceeds_64_bits(), "a wide integer keeps its neighbours");
    require(wide.root().find("max")->u64() == std::numeric_limits<std::uint64_t>::max(), "2^64 - 1 still fits");
    try { (void)wide.root().find("u")->u64(); throw std::logic_error("wide integer read"); }
    catch(const FormatError& e) {
        require(std::string(e.what()) == "fixture.json: byte 5: integer exceeds 64 bits" && e.where() == "byte 5", "wide integer diagnostic names its byte");
    }
    for(const char* text:{"18446744073709551616", "-18446744073709551616", "-9223372036854775809", "99999999999999999999999"})
        require(parse(text).root().exceeds_64_bits(), std::string("wide root ") + text);
    for(const char* text:{"18446744073709551615", "-9223372036854775808", "-0", "0", "1.5", "true", "\"1\""})
        require(!parse(text).root().exceeds_64_bits(), std::string("fits ") + text);
}

void json_nonfinite() {
    // Off by default: NaN and Infinity are rejected (json_rejects). With allow_nonfinite they are reals.
    json::Limits allow;
    allow.allow_nonfinite = true;
    const auto doc = parse("[NaN, Infinity, -Infinity, 1.5, {\"k\":NaN}]", allow);
    const auto a = doc.root().array();
    require(a[0].kind() == json::Kind::real && std::isnan(a[0].real()) && a[0].offset() == 1, "NaN");
    require(a[1].kind() == json::Kind::real && a[1].real() == std::numeric_limits<double>::infinity() && a[1].offset() == 6, "Infinity");
    require(a[2].kind() == json::Kind::real && a[2].real() == -std::numeric_limits<double>::infinity() && a[2].offset() == 16, "-Infinity");
    require(a[3].real() == 1.5 && std::isnan(a[4].find("k")->real()), "neighbours and nested values");
    rejects_as<FormatError>([&] { (void)a[1].u64(); }, "expected integer");
    for(const char* text:{"NaN", "Infinity", "-Infinity", " NaN ", "{\"a\":[Infinity]}"}) (void)parse(text, allow);
    // Only the three literals: other spellings, signs and overflowing reals still fail.
    for(const auto& [text, reason]:std::vector<std::pair<std::string, std::string>>{
            {"nan", "invalid literal"}, {"inf", "unexpected character 'i'"}, {"Inf", "unexpected character 'I'"},
            {"Infinit", "unexpected character 'I'"}, {"-NaN", "invalid number"}, {"+Infinity", "unexpected character '+'"},
            {"- Infinity", "invalid number"}, {"NaNa", "unexpected data after the root value"}, {"[Infinity1]", "expected ',' or ']'"},
            {"1e400", "real number out of range"}, {"-1e400", "real number out of range"}, {"{NaN:1}", "expected a string key"},
            {"{\"a\":1,\"a\":NaN}", "duplicate key 'a'"}, {"[NaN,]", "trailing comma"}})
        rejects_as<FormatError>([&] { (void)parse(text, allow); }, reason);
    rejects_as<FormatError>([&] { (void)parse("[1, NaN]"); }, "fixture.json: byte 4: non-finite number literal");
    // Each literal is one node.
    json::Limits nodes = allow;
    nodes.max_nodes = 3;
    (void)parse("[NaN,-Infinity]", nodes);
    rejects_as<FormatError>([&] { (void)parse("[NaN,-Infinity,Infinity]", nodes); }, "document exceeds the 3-node limit");
    // read_file honours the option.
    TempDir dir;
    write_file(dir.path / "config.json", "{\"time_step_limit\": [0.0, Infinity]}");
    rejects_as<FormatError>([&] { (void)json::Document::read_file(dir.path / "config.json"); }, "non-finite number literal");
    require(std::isinf(json::Document::read_file(dir.path / "config.json", allow).root().find("time_step_limit")->array()[1].real()), "read_file with allow_nonfinite");
}

void json_limits() {
    // Depth: the limit counts open containers; deep input fails fast without recursion.
    json::Limits shallow;
    shallow.max_depth = 4;
    (void)parse("[[[[1]]]]", shallow);
    rejects_as<FormatError>([&] { (void)parse("[[[[[1]]]]]", shallow); }, "nesting exceeds the depth limit of 4");
    rejects_as<FormatError>([&] { (void)parse("{\"a\":{\"b\":{\"c\":{\"d\":{}}}}}", shallow); }, "depth limit");
    (void)parse(std::string(128, '[') + std::string(128, ']'));
    rejects_as<FormatError>([] { (void)parse(std::string(129, '[') + std::string(129, ']')); }, "nesting exceeds the depth limit of 128");
    rejects_as<FormatError>([] { (void)parse(std::string(100000, '[')); }, "depth limit");
    json::Limits deep;
    deep.max_depth = 200000;
    rejects_as<FormatError>([&] { (void)parse(std::string(100000, '['), deep); }, "unexpected end of input");
    const auto nested = parse(std::string(100000, '[') + std::string(100000, ']'), deep);   // iterative, no stack overflow
    const json::Value* cursor = &nested.root();
    for(int i = 1; i < 100000; ++i) cursor = &cursor->array()[0];
    require(cursor->array().empty(), "deep nesting parsed");
    std::string objects;
    for(int i = 0; i < 50000; ++i) objects += "{\"k\":";
    objects += "0";
    for(int i = 0; i < 50000; ++i) objects += "}";
    (void)parse(objects, deep);
    // Nodes: every value counts, keys do not.
    json::Limits few;
    few.max_nodes = 4;
    (void)parse("[1,2,3]", few);
    (void)parse("{\"a\":1,\"b\":[]}", few);
    rejects_as<FormatError>([&] { (void)parse("[1,2,3,4]", few); }, "document exceeds the 4-node limit");
    rejects_as<FormatError>([&] { (void)parse("{\"a\":[1,{},2]}", few); }, "node limit");
    // Bytes: the whole text, padding included.
    json::Limits small;
    small.max_bytes = 8;
    (void)parse("[1, 2]  ", small);
    rejects_as<FormatError>([&] { (void)parse("[1, 2]   ", small); }, "9 bytes exceeds the 8-byte limit");
    try { (void)parse("[1, 2]   ", small); } catch(const FormatError& e) { require(e.where() == "size", "size diagnostic"); }
    for(int field = 0; field < 3; ++field) {
        json::Limits zero;
        if(field == 0) zero.max_bytes = 0;
        if(field == 1) zero.max_depth = 0;
        if(field == 2) zero.max_nodes = 0;
        rejects_as<std::invalid_argument>([&] { (void)parse("1", zero); }, "json limits must be positive");
    }
}

void json_access() {
    std::string text = "{";
    for(int i = 0; i < 200; ++i) text += (i ? "," : "") + std::string("\"key") + std::to_string((i * 37) % 200) + "\":" + std::to_string(i);
    text += ",\"nested\":{\"deeper\":[{\"x\":true}]}}";
    const auto doc = parse(text);
    const auto& root = doc.root();
    for(int i = 0; i < 200; ++i) {
        const auto* value = root.find("key" + std::to_string((i * 37) % 200));
        require(value && value->u64() == std::uint64_t(i), "binary search find");
    }
    require(!root.find("key200") && !root.find("") && !root.find("nestec") && !root.find("zzz"), "absent keys");
    require(root.find("nested")->find("deeper")->array()[0].find("x")->boolean(), "chained find");
    require(root.object()[0].key == "key0" && root.object()[1].key == "key37", "document order kept");
    const auto& integer = *root.find("key0");
    rejects_as<FormatError>([&] { (void)integer.boolean(); }, "fixture.json: byte ");
    rejects_as<FormatError>([&] { (void)integer.boolean(); }, "expected boolean");
    rejects_as<FormatError>([&] { (void)integer.array(); }, "expected array");
    rejects_as<FormatError>([&] { (void)integer.object(); }, "expected object");
    rejects_as<FormatError>([&] { (void)integer.find("x"); }, "expected object");
    rejects_as<FormatError>([&] { (void)root.u64(); }, "expected integer");
    try { (void)root.find("nested")->string(); }
    catch(const FormatError& e) { require(e.where() == "byte " + std::to_string(root.find("nested")->offset()), "accessor location"); }
    const json::Value orphan;
    require(orphan.is_null() && orphan.offset() == 0, "default value");
    rejects_as<FormatError>([&] { (void)orphan.string(); }, "json: byte 0: expected string");
}

void json_file() {
    TempDir dir;
    write_file(dir.path / "config.json", "{\"hidden_size\": 4096, \"eos_token_id\": [1, 2]}\n");
    const auto doc = json::Document::read_file(dir.path / "config.json");
    require(doc.root().find("hidden_size")->u64() == 4096 && doc.source() == display_path(dir.path / "config.json"), "read_file");
    json::Limits small;
    small.max_bytes = 16;
    rejects_as<FormatError>([&] { (void)json::Document::read_file(dir.path / "config.json", small); }, "-byte limit");
    try { (void)json::Document::read_file(dir.path / "config.json", small); }
    catch(const FormatError& e) { require(e.where() == "size" && e.source() == display_path(dir.path / "config.json"), "size checked first"); }
    write_file(dir.path / "bad.json", "{\"a\":1,\"a\":2}");
    rejects_as<FormatError>([&] { (void)json::Document::read_file(dir.path / "bad.json"); }, "bad.json: byte 7: duplicate key 'a'");
    rejects_as<std::system_error>([&] { (void)json::Document::read_file(dir.path / "missing.json"); }, "missing.json: open");
    rejects_as<std::invalid_argument>([&] { (void)json::Document::read_file(dir.path); }, "not a regular file");
    write_file(dir.path / "empty.json", "");
    rejects_as<FormatError>([&] { (void)json::Document::read_file(dir.path / "empty.json"); }, "unexpected end of input");
#ifndef _WIN32
    // The size is checked before reading: a sparse 1 TiB file is refused without allocating its bytes.
    write_file(dir.path / "huge.json", "{}");
    fs::resize_file(dir.path / "huge.json", std::uintmax_t(1) << 40);
    try { (void)json::Document::read_file(dir.path / "huge.json"); throw std::logic_error("1 TiB JSON file was read"); }
    catch(const FormatError& e) {
        require(e.where() == "size" && std::string(e.what()).find("1099511627776 bytes exceeds the 134217728-byte limit") != std::string::npos,
                "size checked before reading");
    }
#endif
}

// Randomized oracle: random trees serialized with random whitespace and escapes must parse back
// exactly; random mutations must get the same verdict as an independent recursive validator.
struct Tree {
    json::Kind kind = json::Kind::null;
    bool flag = false;   // boolean value, integer sign
    std::uint64_t magnitude = 0;
    double real = 0;
    std::string text;
    std::vector<std::string> keys;   // object keys, parallel to items
    std::vector<Tree> items;         // array elements or object values
};
struct Generator {
    std::mt19937_64 random;
    explicit Generator(std::uint64_t seed) : random(seed) {}
    std::uint64_t below(std::uint64_t n) { return random() % n; }
    std::string text() {
        std::string out;
        const auto length = below(6);
        for(std::uint64_t i = 0; i < length; ++i) {
            static const std::uint32_t picks[] = {'a', 'Z', '0', '"', '\\', '/', 0x01, 0x1F, 0x7F, 0xE9, 0x20AC, 0xFFFF, 0x1F600, 0x10FFFF, 0xD7FF, 0xE000};
            const std::uint32_t cp = below(3) ? picks[below(std::size(picks))] : std::uint32_t('a' + below(26));
            if(cp < 0x80) out += char(cp);
            else if(cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
            else if(cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
            else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
        }
        return out;
    }
    Tree tree(int depth) {
        Tree t;
        switch(below(depth > 4 ? 5 : 7)) {
        case 0: break;
        case 1: t.kind = json::Kind::boolean; t.flag = below(2) != 0; break;
        case 2:
            t.kind = json::Kind::integer; t.flag = below(2) != 0;
            t.magnitude = below(3) == 0 ? random() : below(3) == 0 ? std::numeric_limits<std::uint64_t>::max() - below(3) : below(1000);
            if(below(8) == 0) {   // beyond 64 bits: 2^64 and up, as decimal digits
                t.text = below(2) ? "18446744073709551616" : std::to_string(1 + below(9));
                for(auto n = below(30); n; --n) t.text += char('0' + below(10));
                if(t.text.size() < 21) t.text = "18446744073709551616";
            }
            break;
        case 3: {
            t.kind = json::Kind::real;
            const auto bits = random();
            double value = 0;
            std::memcpy(&value, &bits, sizeof value);
            t.real = std::isfinite(value) ? value : double(std::int64_t(bits % 2000000) - 1000000) / 1024.0;
            break;
        }
        case 4: t.kind = json::Kind::string; t.text = text(); break;
        case 5: { t.kind = json::Kind::array; const auto n = below(5); for(std::uint64_t i = 0; i < n; ++i) t.items.push_back(tree(depth + 1)); break; }
        default: {
            t.kind = json::Kind::object;
            std::set<std::string> seen;
            const auto n = below(5);
            for(std::uint64_t i = 0; i < n; ++i) {
                auto key = text();
                if(seen.insert(key).second) { t.keys.push_back(key); t.items.push_back(tree(depth + 1)); }
            }
        }
        }
        return t;
    }
    std::string space() {
        static const char ws[] = {' ', '\t', '\n', '\r'};
        std::string out;
        for(auto n = below(4) ? 0 : below(3); n; --n) out += ws[below(4)];
        return out;
    }
    void quote(std::string& out, const std::string& s) {
        out += '"';
        for(std::size_t i = 0; i < s.size();) {
            const auto b = static_cast<unsigned char>(s[i]);
            std::size_t n = b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
            std::uint32_t cp = n == 1 ? b : n == 2 ? (b & 0x1Fu) : n == 3 ? (b & 0x0Fu) : (b & 0x07u);
            for(std::size_t k = 1; k < n; ++k) cp = cp << 6 | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
            auto u = [&](std::uint32_t v) { char buf[8]; std::snprintf(buf, sizeof buf, below(2) ? "\\u%04x" : "\\u%04X", unsigned(v)); out += buf; };
            if(cp == '"') out += "\\\"";
            else if(cp == '\\') out += below(2) ? "\\\\" : "\\u005c";
            else if(cp < 0x20) { if(cp == '\n' && below(2)) out += "\\n"; else u(cp); }
            else if(cp >= 0x10000 && below(2)) { u(0xD800 + ((cp - 0x10000) >> 10)); u(0xDC00 + ((cp - 0x10000) & 0x3FF)); }
            else if(cp >= 0x80 && cp < 0x10000 && below(2)) u(cp);
            else if(cp == '/' && below(2)) out += "\\/";
            else out.append(s, i, n);
            i += n;
        }
        out += '"';
    }
    void write(std::string& out, const Tree& t) {
        out += space();
        switch(t.kind) {
        case json::Kind::null: out += "null"; break;
        case json::Kind::boolean: out += t.flag ? "true" : "false"; break;
        case json::Kind::integer: out += (t.flag ? "-" : "") + (t.text.empty() ? std::to_string(t.magnitude) : t.text); break;
        case json::Kind::real: {
            char buf[64];
            const auto r = std::to_chars(buf, buf + sizeof buf, t.real);
            std::string s(buf, r.ptr);
            if(s.find_first_of(".eE") == std::string::npos) s += ".0";
            out += s;
            break;
        }
        case json::Kind::string: quote(out, t.text); break;
        case json::Kind::array:
            out += '[';
            for(std::size_t i = 0; i < t.items.size(); ++i) { if(i) out += ','; write(out, t.items[i]); }
            out += space() + ']';
            break;
        case json::Kind::object:
            out += '{';
            for(std::size_t i = 0; i < t.keys.size(); ++i) {
                if(i) out += ',';
                out += space();
                quote(out, t.keys[i]);
                out += space() + ':';
                write(out, t.items[i]);
            }
            out += space() + '}';
            break;
        }
        out += space();
    }
};
void same(const Tree& t, const json::Value& v) {
    require(v.kind() == t.kind, "random tree kind");
    switch(t.kind) {
    case json::Kind::null: break;
    case json::Kind::boolean: require(v.boolean() == t.flag, "random boolean"); break;
    case json::Kind::integer:
        if(!t.text.empty() || (t.flag && t.magnitude > (std::uint64_t(1) << 63))) {   // outside [-2^63, 2^64)
            require(v.exceeds_64_bits(), "random wide integer");
            rejects_as<FormatError>([&] { (void)v.i64(); }, "integer exceeds 64 bits");
            rejects_as<FormatError>([&] { (void)v.real(); }, "integer exceeds 64 bits");
            break;
        }
        require(!v.exceeds_64_bits(), "random integer fits");
        if(!t.flag) require(v.u64() == t.magnitude, "random integer");
        else require(v.i64() == (t.magnitude == (std::uint64_t(1) << 63) ? std::numeric_limits<std::int64_t>::min() : -std::int64_t(t.magnitude)), "random negative integer");
        require(v.real() == (t.flag ? -double(t.magnitude) : double(t.magnitude)), "random integer as real");
        break;
    case json::Kind::real: require(std::bit_cast<std::uint64_t>(v.real()) == std::bit_cast<std::uint64_t>(t.real), "random real round trip"); break;
    case json::Kind::string: require(v.string() == t.text, "random string"); break;
    case json::Kind::array:
        require(v.array().size() == t.items.size(), "random array size");
        for(std::size_t i = 0; i < t.items.size(); ++i) same(t.items[i], v.array()[i]);
        break;
    case json::Kind::object:
        require(v.object().size() == t.keys.size(), "random object size");
        for(std::size_t i = 0; i < t.keys.size(); ++i) {
            require(v.object()[i].key == t.keys[i] && v.find(t.keys[i]) == &v.object()[i].value, "random member");
            same(t.items[i], v.object()[i].value);
        }
        break;
    }
}
// Straightforward recursive RFC 8259 validator with the same strictness rules (small inputs only).
struct Reference {
    std::string_view s;
    std::size_t i = 0;
    int depth = 0;
    void ws() { while(i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; }
    bool literal(std::string_view word) { if(s.substr(i, word.size()) != word) return false; i += word.size(); return true; }
    bool string(std::string* out) {
        if(i >= s.size() || s[i] != '"') return false;
        ++i;
        std::string value;
        auto hex = [&](std::uint32_t& v) {
            if(i + 4 > s.size()) return false;
            v = 0;
            for(int k = 0; k < 4; ++k) {
                const char c = s[i++];
                const int d = std::isxdigit(static_cast<unsigned char>(c)) ? (c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10) : -1;
                if(d < 0) return false;
                v = v * 16 + std::uint32_t(d);
            }
            return true;
        };
        for(;;) {
            if(i >= s.size()) return false;
            const auto b = static_cast<unsigned char>(s[i]);
            if(b == '"') { ++i; break; }
            if(b < 0x20) return false;
            if(b == '\\') {
                if(++i >= s.size()) return false;
                const char e = s[i++];
                const auto simple = std::string_view("\"\\/bfnrt").find(e);
                if(simple != std::string_view::npos) { value += "\"\\/\b\f\n\r\t"[simple]; continue; }
                if(e != 'u') return false;
                std::uint32_t cp = 0, low = 0;
                if(!hex(cp)) return false;
                if(cp >= 0xDC00 && cp <= 0xDFFF) return false;
                if(cp >= 0xD800 && cp <= 0xDBFF) {
                    if(!literal("\\u") || !hex(low) || low < 0xDC00 || low > 0xDFFF) return false;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                }
                if(cp < 0x80) value += char(cp);
                else if(cp < 0x800) { value += char(0xC0 | (cp >> 6)); value += char(0x80 | (cp & 0x3F)); }
                else if(cp < 0x10000) { value += char(0xE0 | (cp >> 12)); value += char(0x80 | ((cp >> 6) & 0x3F)); value += char(0x80 | (cp & 0x3F)); }
                else { value += char(0xF0 | (cp >> 18)); value += char(0x80 | ((cp >> 12) & 0x3F)); value += char(0x80 | ((cp >> 6) & 0x3F)); value += char(0x80 | (cp & 0x3F)); }
                continue;
            }
            std::size_t n = 1;
            if(b >= 0x80) { n = (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 1; if(!reference_utf8(s.substr(i, n))) return false; }
            if(b >= 0x80 && n == 1) return false;
            value.append(s.substr(i, n));
            i += n;
        }
        if(out) *out = value;
        return true;
    }
    bool number() {
        const std::size_t start = i;
        const bool negative = i < s.size() && s[i] == '-';
        if(negative) ++i;
        if(i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
        if(s[i] == '0') ++i; else while(i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        bool integer = true;
        if(i < s.size() && s[i] == '.') { ++i; integer = false; if(i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false; while(i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i; }
        if(i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            ++i; integer = false;
            if(i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
            if(i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
            while(i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
        }
        if(integer) return true;   // any length: integers beyond 64 bits load too
        const std::string token(s.substr(start, i - start));
        return !std::isinf(std::strtod(token.c_str(), nullptr));
    }
    bool value() {
        ws();
        if(i >= s.size()) return false;
        bool ok = false;
        if(s[i] == '[') {
            if(++depth > 128) return false;
            ++i; ws();
            if(i < s.size() && s[i] == ']') { ++i; ok = true; }
            else for(;;) {
                if(!value()) return false;
                ws();
                if(i < s.size() && s[i] == ',') { ++i; continue; }
                if(i < s.size() && s[i] == ']') { ++i; ok = true; }
                break;
            }
            --depth;
        } else if(s[i] == '{') {
            if(++depth > 128) return false;
            ++i; ws();
            std::set<std::string> keys;
            if(i < s.size() && s[i] == '}') { ++i; ok = true; }
            else for(;;) {
                ws();
                std::string key;
                if(!string(&key) || !keys.insert(key).second) return false;
                ws();
                if(i >= s.size() || s[i] != ':') return false;
                ++i;
                if(!value()) return false;
                ws();
                if(i < s.size() && s[i] == ',') { ++i; continue; }
                if(i < s.size() && s[i] == '}') { ++i; ok = true; }
                break;
            }
            --depth;
        } else if(s[i] == '"') ok = string(nullptr);
        else if(s[i] == '-' || std::isdigit(static_cast<unsigned char>(s[i]))) ok = number();
        else ok = literal("true") || literal("false") || literal("null");
        return ok;
    }
    bool document() { if(!value()) return false; ws(); return i == s.size(); }
};
void json_random() {
    Generator g(20261015);
    for(int round = 0; round < 3000; ++round) {
        const Tree tree = g.tree(0);
        std::string text;
        g.write(text, tree);
        const auto doc = parse(text);
        same(tree, doc.root());
        require(Reference{text}.document(), "reference rejects a generated document");
    }
    static const char interesting[] = {',', ']', '}', '[', '{', '"', '\\', ':', 'u', '0', '-', 'e', '.', ' ', '\x01', '/', 'D', '8'};
    static const unsigned char high[] = {0x80, 0xBF, 0xC0, 0xC2, 0xE0, 0xED, 0xF0, 0xF4, 0xF5, 0xFF};
    std::size_t accepted = 0, rejected = 0;
    for(int round = 0; round < 20000; ++round) {
        std::string text;
        g.write(text, g.tree(0));
        for(auto edits = 1 + g.below(3); edits; --edits) {
            const std::size_t at = text.empty() ? 0 : std::size_t(g.below(text.size()));
            switch(g.below(5)) {
            case 0: if(!text.empty()) text[at] = char(g.below(256)); break;
            case 1: text.insert(text.begin() + std::ptrdiff_t(at), interesting[g.below(sizeof interesting)]); break;
            case 2: text.insert(text.begin() + std::ptrdiff_t(at), char(high[g.below(sizeof high)])); break;
            case 3: if(!text.empty()) text.erase(at, 1); break;
            default: text.resize(at); break;
            }
        }
        const bool expected = Reference{text}.document();
        bool got = false;
        try { (void)parse(text); got = true; } catch(const FormatError&) {}
        if(got != expected) {
            std::string hex;
            for(const unsigned char c:text) { char b[4]; std::snprintf(b, sizeof b, "%02x", unsigned(c)); hex += b; }
            throw std::runtime_error("mutated document verdict differs from the reference: " + hex);
        }
        (got ? accepted : rejected)++;
    }
    require(accepted > 1000 && rejected > 1000, "mutation fuzz covered both verdicts");
}

// ---------------------------------------------------------------------------------------------- SHA-256
void sha256_vectors() {
    require(sha256_hex({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty message");
    require(sha256_hex(bytes_of("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    require(sha256_hex(bytes_of("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "448-bit message");
    require(sha256_hex(bytes_of("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu")) ==
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1", "896-bit message");
    const std::string million(1000000, 'a');
    require(sha256_hex(bytes_of(million)) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "one million a");
    Sha256 hasher;
    for(int i = 0; i < 1000; ++i) hasher.update(bytes_of(std::string_view(million).substr(0, 1000)));
    const auto digest = hasher.finish();
    require(to_hex(digest) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "one million a in pieces");
    // Padding boundaries (Python hashlib digests of bytes(i % 251 for i in range(n))).
    const std::vector<std::pair<std::size_t, const char*>> lengths = {
        {55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"}, {56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        {57, "2fe741af801cc238602ac0ec6a7b0c3a8a87c7fc7d7f02a3fe03d1c12eac4d8f"}, {63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
        {64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"}, {65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
        {119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"}, {120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"},
        {128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"}, {1000, "4e4c294b331f7a2099a379bec34b9f9fc03dc46ab465d998f4d683da53487e6d"}};
    for(const auto& [n, hex]:lengths) {
        std::string data(n, '\0');
        for(std::size_t i = 0; i < n; ++i) data[i] = char(i % 251);
        require(sha256_hex(bytes_of(data)) == hex, "padding boundary digest " + std::to_string(n));
    }
}

void sha256_chunked() {
    const auto data = random_bytes(70000, 5);
    const auto whole = sha256_hex(bytes_of(data));
    std::mt19937 random(9);
    Sha256 hasher;
    for(int round = 0; round < 200; ++round) {
        for(std::size_t at = 0; at < data.size();) {
            const std::size_t n = std::min<std::size_t>(data.size() - at, random() % 4 == 0 ? random() % 200 : random() % 7000);
            hasher.update(bytes_of(std::string_view(data).substr(at, n)));
            at += n;
        }
        const auto digest = hasher.finish();   // also resets the hasher for the next round
        require(to_hex(digest) == whole, "chunked update equals one shot");
    }
    hasher.update({});
    const auto empty = hasher.finish();
    require(to_hex(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "reset after finish");
    // Empty spans (null data) between partial blocks are no-ops.
    hasher.update(bytes_of("ab"));
    hasher.update({});
    hasher.update(std::span<const std::byte>(static_cast<const std::byte*>(nullptr), 0));
    hasher.update(bytes_of("c"));
    hasher.update({});
    const auto abc = hasher.finish();
    require(to_hex(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "empty updates after a partial block");
    TempDir dir;
    write_file(dir.path / "weights.bin", data);
    const auto file = ReadOnlyFile::open(dir.path / "weights.bin");
    require(sha256_file(file) == whole, "sha256_file direct");
    ReadOptions buffered;
    buffered.direct = false;
    buffered.chunk_bytes = 4096;
    require(sha256_file(file, buffered) == whole, "sha256_file buffered");
    write_file(dir.path / "empty.bin", "");
    require(sha256_file(ReadOnlyFile::open(dir.path / "empty.bin")) == sha256_hex({}), "empty file digest");
#ifndef _WIN32
    // Same size, new bytes and a later time: every chunk reads fine, so only verify_unchanged() can
    // refuse to return the digest of the modified content.
    write_file(dir.path / "same.bin", data);
    const auto held = ReadOnlyFile::open(dir.path / "same.bin");
    auto changed = data;
    changed[100] = char(changed[100] ^ 1);
    write_file(dir.path / "same.bin", changed);
    fs::last_write_time(dir.path / "same.bin", fs::last_write_time(dir.path / "same.bin") + std::chrono::seconds(3));
    for(bool direct:{true, false}) {
        ReadOptions o;
        o.direct = direct;
        rejects_as<std::runtime_error>([&] { (void)sha256_file(held, o); }, "same.bin: file changed since open (modified time");
    }
    write_file(dir.path / "weights.bin", data.substr(0, 1000));   // shrunk after open
    rejects_as<std::runtime_error>([&] { (void)sha256_file(file, buffered); }, "weights.bin: unexpected end of file at offset 1000");
#else
    std::ofstream writer(dir.path / "weights.bin", std::ios::binary | std::ios::trunc);
    require(!writer, "a held file refuses writers, so it cannot change while hashed");
#endif
}

// ---------------------------------------------------------------------------------------------- Files
void file_identity() {
    TempDir dir;
    write_file(dir.path / "a.bin", "hello");
    write_file(dir.path / "b.bin", "other");
    const auto a = ReadOnlyFile::open(dir.path / "a.bin");
    require(a.path() == dir.path / "a.bin" && a.identity().size == 5, "path and size");
    require(a.handle_identity() == a.identity() && a.path_identity() == a.identity(), "handle and path agree");
    a.verify_unchanged();
    const auto b = ReadOnlyFile::open(dir.path / "b.bin");
    require(b.identity() != a.identity() && (b.identity().file_id != a.identity().file_id || b.identity().device != a.identity().device), "distinct files");
#ifndef _WIN32
    struct stat status{};
    require(::stat((dir.path / "a.bin").c_str(), &status) == 0, "stat fixture");
    require(a.identity().device == std::uint64_t(status.st_dev) && a.identity().file_id == std::uint64_t(status.st_ino), "POSIX identity fields");
    require(a.identity().modified_ns / 1000000000 == std::int64_t(status.st_mtime) && a.identity().changed_ns / 1000000000 == std::int64_t(status.st_ctime), "POSIX times");
#endif
    std::error_code linked;
    fs::create_hard_link(dir.path / "a.bin", dir.path / "hard.bin", linked);
    if(!linked) {
        const auto hard = ReadOnlyFile::open(dir.path / "hard.bin");
        require(hard.identity().file_id == a.identity().file_id && hard.identity().device == a.identity().device, "hard link names the same file");
    }
    std::error_code symlinked;
    fs::create_symlink(dir.path / "b.bin", dir.path / "link.bin", symlinked);
    if(!symlinked) {
        const auto followed = ReadOnlyFile::open(dir.path / "link.bin");
        require(followed.identity() == b.identity(), "followed symlink has the target identity");
        followed.verify_unchanged();
        rejects_as<std::system_error>([&] { (void)ReadOnlyFile::open(dir.path / "link.bin", false); }, "link.bin: open: symbolic link refused");
        const auto direct = ReadOnlyFile::open(dir.path / "b.bin", false);
        direct.verify_unchanged();
    }
    // Move-only ownership; a moved-from file refuses use.
    auto first = ReadOnlyFile::open(dir.path / "a.bin");
    auto second = std::move(first);
    require(second.identity().size == 5 && first.path().empty() && first.identity() == FileIdentity{}, "moved file");
    rejects_as<std::logic_error>([&] { first.verify_unchanged(); }, "moved from");
    rejects_as<std::logic_error>([&] { (void)first.map(); }, "moved from");
    first.advise_willneed(0, 1);
    first = std::move(second);
    first.verify_unchanged();
    write_file(dir.path / "empty.bin", "");
    require(ReadOnlyFile::open(dir.path / "empty.bin").identity().size == 0, "empty file identity");
}

void file_change() {
    TempDir dir;
    const auto path = dir.path / "model.bin";
    write_file(path, "0123456789");
    {
        const auto file = ReadOnlyFile::open(path);
        file.verify_unchanged();
#ifdef _WIN32
        std::ofstream writer(path, std::ios::binary | std::ios::app);
        require(!writer, "writers are refused while the file is held");
#else
        write_file(path, "0123456789abc");   // same inode, new size
        rejects_as<std::runtime_error>([&] { file.verify_unchanged(); }, "model.bin: file changed since open (size");
        rejects_as<std::runtime_error>([&] { (void)file.map(); }, "file changed since open");
#endif
    }
#ifndef _WIN32
    {
        write_file(path, "0123456789");
        const auto file = ReadOnlyFile::open(path);
        write_file(path, "9876543210");   // same size; bump the time so coarse clocks cannot hide it
        fs::last_write_time(path, fs::last_write_time(path) + std::chrono::seconds(3));
        rejects_as<std::runtime_error>([&] { file.verify_unchanged(); }, "modified time");
    }
    {
        write_file(path, "0123456789");
        const auto file = ReadOnlyFile::open(path);
        write_file(dir.path / "replacement.bin", "0123456789");
        fs::rename(dir.path / "replacement.bin", path);   // replaces the name; unlinking the old file changes its ctime
        rejects_as<std::runtime_error>([&] { file.verify_unchanged(); }, "file changed since open");
    }
    // Through a symlink the held file stays untouched while the name moves on or disappears.
    write_file(dir.path / "a.bin", "aaaa");
    write_file(dir.path / "b.bin", "bbbb");
    std::error_code linked;
    fs::create_symlink(dir.path / "a.bin", dir.path / "current", linked);
    if(!linked) {
        const auto file = ReadOnlyFile::open(dir.path / "current");
        file.verify_unchanged();
        fs::remove(dir.path / "current");
        fs::create_symlink(dir.path / "b.bin", dir.path / "current");
        require(file.handle_identity() == file.identity(), "held file untouched");
        rejects_as<std::runtime_error>([&] { file.verify_unchanged(); }, "current: file changed since open (path: file id");
        fs::remove(dir.path / "current");
        rejects_as<std::runtime_error>([&] { file.verify_unchanged(); }, "file changed since open (path no longer resolves");
        const auto nofollow = ReadOnlyFile::open(dir.path / "a.bin", false);
        fs::create_symlink(dir.path / "b.bin", dir.path / "a-link");
        nofollow.verify_unchanged();
    }
#endif
    {
        // A relative path is resolved once at open; a later working-directory change is not a file change.
        struct Restore {
            fs::path cwd = fs::current_path();
            ~Restore() { std::error_code ignored; fs::current_path(cwd, ignored); }
        } restore;
        write_file(dir.path / "a" / "relative.bin", "relative");
        fs::create_directories(dir.path / "b");
        fs::current_path(dir.path / "a");
        const auto file = ReadOnlyFile::open("relative.bin");
        require(file.path() == fs::path("relative.bin") && file.path_identity() == file.identity(), "relative path kept as given");
        fs::current_path(dir.path / "b");
        file.verify_unchanged();
        require(file.path_identity() == file.identity(), "path identity after a directory change");
        require(sha256_file(file) == sha256_hex(bytes_of("relative")), "relative path hashes after a directory change");
    }
    {
        write_file(path, "0123456789");
        auto file = std::make_unique<ReadOnlyFile>(ReadOnlyFile::open(path));
#ifndef _WIN32
        fs::remove(path);
        rejects_as<std::runtime_error>([&] { file->verify_unchanged(); }, "file changed since open");
        std::array<std::byte, 4> head{};
        file->read_exact(0, head);   // the held handle still reads the original bytes
        require(text_of(head) == "0123", "held handle survives unlink");
#else
        std::error_code refused;
        fs::remove(path, refused);
        require(bool(refused), "delete is refused while the file is held");
#endif
    }
}

void file_read_exact() {
    TempDir dir;
    const auto data = random_bytes(100000, 3);
    write_file(dir.path / "data.bin", data);
    const auto file = ReadOnlyFile::open(dir.path / "data.bin");
    std::vector<std::byte> all(data.size());
    file.read_exact(0, all);
    require(text_of(all) == data, "whole file");
    std::vector<std::byte> middle(777);
    file.read_exact(12345, middle);
    require(text_of(middle) == data.substr(12345, 777), "middle region");
    file.read_exact(data.size(), {});
    std::array<std::byte, 1> one{};
    file.read_exact(data.size() - 1, one);
    require(char(one[0]) == data.back(), "last byte");
    rejects_as<std::out_of_range>([&] { file.read_exact(data.size(), one); }, "extends past end of file (size 100000)");
    rejects_as<std::out_of_range>([&] { file.read_exact(data.size() + 1, {}); }, "past end of file");
    rejects_as<std::out_of_range>([&] { file.read_exact(std::numeric_limits<std::uint64_t>::max(), one); }, "past end of file");
    std::vector<std::byte> over(10);
    rejects_as<std::out_of_range>([&] { file.read_exact(data.size() - 9, over); }, "read of 10 bytes at offset 99991");
#ifndef _WIN32
    fs::resize_file(dir.path / "data.bin", 50000);   // shrinks underneath the held handle
    rejects_as<std::runtime_error>([&] { file.read_exact(49995, over); }, "unexpected end of file at offset 50000");
#endif
}

// Reader slots: every slot reads the held file's bytes (a pattern whose period exceeds a sector, at a size that is not
// a sector multiple), ids past the slots use the held handle, eight threads reading through their own slots at
// once all see their own regions, and the slot count opens exactly readers - 1 private handles on Windows (none on
// POSIX, where pread is concurrent on one descriptor) and closes the previous ones when reset.
void file_read_slots() {
    TempDir dir;
    std::string data(1048576 + 1234, '\0');
    for(std::size_t i = 0; i < data.size(); ++i) data[i] = char(std::uint8_t((i * 31 + 7 + (i >> 12) * 17) & 0xFF));
    write_file(dir.path / "rows.bin", data);
    auto file = ReadOnlyFile::open(dir.path / "rows.bin");
    require(file.readers() == 1 && file.private_readers() == 0, "one reader slot by default");
    rejects_as<std::invalid_argument>([&] { file.set_readers(0); }, "at least one reader slot");
    for(const std::size_t slots : {std::size_t(8), std::size_t(3)}) {
        file.set_readers(slots);
#ifdef _WIN32
        require(file.readers() == slots && file.private_readers() == slots - 1, "private handles for slots 1 .. readers - 1");
#else
        require(file.readers() == slots && file.private_readers() == 0, "POSIX slots share the held descriptor");
#endif
        for(std::size_t worker = 0; worker < slots + 2; ++worker) {   // the last two ids are past the slots
            std::vector<std::byte> got(64);
            file.read_exact(1000, got, worker);
            require(text_of(got) == data.substr(1000, 64), "a reader slot reads the file's bytes");
            require(text_of(got) != data.substr(2000, 64), "the pattern distinguishes offsets");
            std::vector<std::byte> tail(1234);
            file.read_exact(data.size() - 1234, tail, worker);
            require(text_of(tail) == data.substr(data.size() - 1234), "a reader slot reads the unaligned tail");
        }
    }
    file.set_readers(8);
    std::vector<std::thread> threads;
    std::atomic<int> wrong{0};
    for(std::size_t t = 0; t < 8; ++t)
        threads.emplace_back([&, t] {
            for(int rep = 0; rep < 16; ++rep) {
                const std::size_t offset = (t * 131071 + std::size_t(rep) * 4099) % (data.size() - 777);
                std::vector<std::byte> got(777);
                file.read_exact(offset, got, t);
                if(text_of(got) != data.substr(offset, 777)) ++wrong;
            }
        });
    for(auto& thread : threads) thread.join();
    require(wrong == 0, "concurrent reads through reader slots");
    std::vector<std::byte> over(10);
    rejects_as<std::out_of_range>([&] { file.read_exact(data.size() - 9, over, 3); }, "extends past end of file");
}

// Whether the file system holding path rejects misaligned direct reads (EINVAL, ERROR_INVALID_PARAMETER).
bool enforces_direct_alignment(const fs::path& path) {
#if defined(_WIN32)
    (void)path;
    return true;   // FILE_FLAG_NO_BUFFERING needs sector-aligned offsets, sizes and buffers
#elif defined(__linux__)
    struct statfs info{};
    return ::statfs(path.c_str(), &info) == 0 && (info.f_type == 0xEF53 || info.f_type == 0x58465342);   // ext2/3/4, XFS
#else
    (void)path;
    return false;
#endif
}

void file_read_chunks() {
    TempDir dir;
    const auto data = random_bytes(5 * 1048576 + 1237, 11);
    write_file(dir.path / "artifact.bin", data);
    const auto file = ReadOnlyFile::open(dir.path / "artifact.bin");
    bool any_direct = false;
    for(bool direct:{true, false})
        for(bool overlap:{true, false})
            for(std::size_t chunk:{std::size_t(65536), std::size_t(1) << 20})
                for(auto [offset, length]:{std::pair<std::uint64_t, std::uint64_t>{0, data.size()}, {4095, 3}, {12345, 2 * 1048576 + 7},
                                           {data.size() - 10, 10}, {4096, 65536}, {1, data.size() - 1}}) {
                    std::string got;
                    std::uint64_t expect_at = offset;
                    std::size_t calls = 0;
                    ReadOptions o;
                    o.direct = direct;
                    o.overlap = overlap;
                    o.chunk_bytes = chunk;
                    const auto stats = file.read_chunks(offset, length, [&](std::span<const std::byte> bytes, std::uint64_t at) {
                        require(at == expect_at && !bytes.empty() && bytes.size() <= chunk, "chunks arrive in order and nonempty");
                        expect_at += bytes.size();
                        got += text_of(bytes);
                        ++calls;
                    }, o);
                    require(stats.bytes == length && got.size() == length && stats.chunks == calls, "byte and chunk counts");
                    require(got == data.substr(std::size_t(offset), std::size_t(length)), "exact region bytes");
                    require(direct || (!stats.direct_used && stats.fallback_reason.empty()), "buffered reads never claim direct I/O");
                    require(stats.direct_used || !direct || !stats.fallback_reason.empty(), "a refused direct read says why");
                    require(!stats.direct_used || stats.fallback_reason.empty(), "direct reads carry no fallback reason");
                    any_direct = any_direct || stats.direct_used;
                }
    // A smaller alignment and odd offsets still return exact bytes.
    ReadOptions small;
    small.direct = true;
    small.alignment = 512;
    small.chunk_bytes = 1536;
    std::string got;
    const auto stats = file.read_chunks(777, 10000, [&](std::span<const std::byte> bytes, std::uint64_t) { got += text_of(bytes); }, small);
    require(got == data.substr(777, 10000) && stats.bytes == 10000, "512-byte alignment");
    std::cout << "direct I/O " << (any_direct ? "used" : "refused, buffered fallback") << " on this file system\n";
    // Alignment 1 is valid, but file systems that enforce direct I/O alignment refuse the first read at
    // an odd offset or size before any byte is delivered; the call then retries buffered and says why.
    const bool strict = any_direct && enforces_direct_alignment(dir.path);
    for(bool overlap:{true, false})
        for(std::uint64_t offset:{std::uint64_t(777), std::uint64_t(0)}) {
            ReadOptions odd;
            odd.direct = true;
            odd.overlap = overlap;
            odd.alignment = 1;
            odd.chunk_bytes = 65537;
            const std::uint64_t length = 1048576 + 3;
            std::string region;
            std::uint64_t expect_at = offset;
            std::size_t calls = 0;
            const auto result = file.read_chunks(offset, length, [&](std::span<const std::byte> bytes, std::uint64_t at) {
                require(at == expect_at && bytes.size() <= odd.chunk_bytes, "each byte is delivered once, in order");
                expect_at += bytes.size();
                region += text_of(bytes);
                ++calls;
            }, odd);
            require(region == data.substr(std::size_t(offset), std::size_t(length)) && result.bytes == length, "exact bytes after a retry");
            require(calls == 16 && result.chunks == 16, "the retry delivers every chunk once");
            if(result.direct_used) require(!strict && result.fallback_reason.empty(), "misaligned direct reads must be refused here");
            else require(result.fallback_reason.find(any_direct ? "direct read refused: " : "direct ") == 0,
                         "fallback reason: " + result.fallback_reason);
        }
#ifdef __linux__
    // kernfs refuses O_DIRECT opens, so the second handle cannot be made and the call reads buffered.
    bool refused_open = false;
    for(const char* name:{"/sys/devices/system/cpu/possible", "/sys/devices/system/cpu/online", "/sys/kernel/mm/transparent_hugepage/enabled"}) {
        struct statfs info{};
        if(::statfs(name, &info) != 0 || info.f_type != 0x62656572) continue;   // SYSFS_MAGIC
        std::ifstream in(name, std::ios::binary);
        const std::string expected{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        const auto sys = ReadOnlyFile::open(name);
        if(expected.empty() || expected.size() > sys.identity().size) continue;
        for(bool overlap:{true, false}) {
            ReadOptions o;
            o.direct = true;
            o.overlap = overlap;
            std::string read;
            const auto result = sys.read_chunks(0, expected.size(),
                                                [&](std::span<const std::byte> bytes, std::uint64_t) { read += text_of(bytes); }, o);
            require(read == expected && result.bytes == expected.size() && !result.direct_used, "buffered read after a refused direct open");
            require(result.fallback_reason.find("direct open refused: ") == 0, "open refusal reason: " + result.fallback_reason);
        }
        refused_open = true;
        break;
    }
    std::cout << "direct open refusal " << (refused_open ? "exercised on sysfs" : "not exercised (no sysfs file)") << "\n";
#endif
}

void file_read_errors() {
    TempDir dir;
    const auto data = random_bytes(300000, 2);
    write_file(dir.path / "artifact.bin", data);
    const auto file = ReadOnlyFile::open(dir.path / "artifact.bin");
    int calls = 0;
    const auto count = [&](std::span<const std::byte>, std::uint64_t) { ++calls; };
    const auto none = file.read_chunks(7, 0, count);
    require(none.chunks == 0 && none.bytes == 0 && calls == 0, "empty region reads nothing");
    rejects_as<std::out_of_range>([&] { (void)file.read_chunks(data.size() - 5, 6, count); }, "region [299995, 300001) extends past end of file");
    rejects_as<std::out_of_range>([&] { (void)file.read_chunks(data.size() + 1, 0, count); }, "past end of file");
    rejects_as<std::invalid_argument>([&] { (void)file.read_chunks(std::numeric_limits<std::uint64_t>::max(), 2, count); }, "region overflows");
    for(auto [chunk, alignment]:{std::pair<std::size_t, std::size_t>{5000, 4096}, {0, 4096}, {4096, 0}, {4096, 3}, {6144, 4096}, {4097, 4097}}) {
        ReadOptions bad;
        bad.chunk_bytes = chunk;
        bad.alignment = alignment;
        rejects_as<std::invalid_argument>([&] { (void)file.read_chunks(0, 1, count, bad); }, "chunk_bytes must be a positive multiple of a power-of-two alignment");
    }
    rejects_as<std::invalid_argument>([&] { (void)file.read_chunks(0, 1, ChunkConsumer{}); }, "chunk consumer is empty");
    require(calls == 0, "invalid requests never call the consumer");
    for(bool direct:{true, false})
        for(bool overlap:{true, false}) {
            ReadOptions chunked;
            chunked.chunk_bytes = 65536;
            chunked.direct = direct;
            chunked.overlap = overlap;
            calls = 0;
            rejects_as<std::runtime_error>([&] {
                (void)file.read_chunks(0, data.size(), [&](std::span<const std::byte>, std::uint64_t) { if(++calls == 3) throw std::runtime_error("consumer stop"); }, chunked);
            }, "consumer stop");
            require(calls == 3, "the consumer exception stops the stream");
        }
#ifndef _WIN32
    fs::resize_file(dir.path / "artifact.bin", 100000);
    for(bool overlap:{true, false}) {
        ReadOptions o;
        o.chunk_bytes = 65536;
        o.overlap = overlap;
        rejects_as<std::runtime_error>([&] { (void)file.read_chunks(0, data.size(), count, o); }, "unexpected end of file at offset 100000");
    }
#endif
}

void file_map() {
    TempDir dir;
    const auto data = random_bytes(200000, 4);
    write_file(dir.path / "weights.bin", data);
    std::shared_ptr<const void> owner;
    std::span<const std::byte> view;
    {
        const auto file = ReadOnlyFile::open(dir.path / "weights.bin");
        file.advise_willneed(0, 4096);   // before mapping: a read-ahead hint on the handle
        std::vector<const std::byte*> seen(8);
        std::vector<std::thread> threads;
        for(std::size_t i = 0; i < seen.size(); ++i) threads.emplace_back([&, i] { seen[i] = file.map().data(); });
        for(auto& t:threads) t.join();
        require(std::all_of(seen.begin(), seen.end(), [&](const std::byte* p) { return p && p == seen[0]; }), "one mapping for all threads");
        view = file.map();
        require(view.size() == data.size() && text_of(view) == data, "mapped bytes");
        file.advise_willneed(1000, 100000);
        file.advise_willneed(data.size() - 1, 1000000);
        file.advise_willneed(data.size(), 10);
        owner = file.mapping_owner();
        require(owner != nullptr, "mapping owner");
    }
#ifndef _WIN32
    fs::remove(dir.path / "weights.bin");
#endif
    require(text_of(view) == data, "mapping outlives the file object");
    owner.reset();
    write_file(dir.path / "empty.bin", "");
    const auto empty = ReadOnlyFile::open(dir.path / "empty.bin");
    require(empty.map().empty() && empty.mapping_owner() != nullptr, "empty file maps to an empty view");
    empty.advise_willneed(0, 10);
    write_file(dir.path / "other.bin", "abcdef");
    const auto owned = ReadOnlyFile::open(dir.path / "other.bin").mapping_owner();
    require(owned != nullptr, "mapping owner from a temporary file object");
}

void file_open_errors() {
    TempDir dir;
    try { (void)ReadOnlyFile::open(dir.path / "missing.bin"); throw std::logic_error("missing file opened"); }
    catch(const std::system_error& e) {
        require(e.code() == std::errc::no_such_file_or_directory, "missing file errno");
        require(std::string(e.what()).find(display_path(dir.path / "missing.bin") + ": open") != std::string::npos, "message names the path");
    }
    rejects_as<std::invalid_argument>([&] { (void)ReadOnlyFile::open(dir.path); }, display_path(dir.path) + ": not a regular file");
    rejects_as<std::invalid_argument>([&] { (void)ReadOnlyFile::open(dir.path, false); }, "not a regular file");
    // An embedded NUL would make the OS open the prefix, a different file than the one named.
    write_file(dir.path / "a.bin", "hello");
    auto native = (dir.path / "a.bin").native();
    native.push_back(fs::path::value_type());
    native += fs::path("-suffix").native();
    const fs::path truncated(native);
    require(truncated.native().size() == (dir.path / "a.bin").native().size() + 8, "fixture path keeps the NUL");
    rejects_as<std::invalid_argument>([&] { (void)ReadOnlyFile::open(truncated); }, "a.bin\\0-suffix: path contains a NUL character");
    rejects_as<std::invalid_argument>([&] { (void)json::Document::read_file(truncated); }, "path contains a NUL character");
    require(display_path(truncated) == display_path(dir.path / "a.bin") + "\\0-suffix", "display escapes NUL");
#ifndef _WIN32
    const auto fifo = dir.path / "pipe";
    require(::mkfifo(fifo.c_str(), 0600) == 0, "mkfifo");
    rejects_as<std::invalid_argument>([&] { (void)ReadOnlyFile::open(fifo); }, "pipe: not a regular file");   // must not block
    write_file(dir.path / "secret.bin", "x");
    fs::permissions(dir.path / "secret.bin", fs::perms::none);
    if(::geteuid() != 0) {
        try { (void)ReadOnlyFile::open(dir.path / "secret.bin"); throw std::logic_error("unreadable file opened"); }
        catch(const std::system_error& e) { require(e.code() == std::errc::permission_denied, "permission errno"); }
    }
    fs::permissions(dir.path / "secret.bin", fs::perms::owner_all);
#endif
}

// ---------------------------------------------------------------------------------------------- Host memory
void put(const fs::path& path, const std::string& text) { write_file(path, text); }
bool has_note(const HostMemoryObservation& o, std::string_view text) {
    return std::any_of(o.notes.begin(), o.notes.end(), [&](const std::string& n) { return n.find(text) != std::string::npos; });
}
HostWorkingSet anonymous(std::uint64_t bytes) { return {bytes, 0, 0}; }

void host_memory_v2() {
    TempDir dir;
    const auto& root = dir.path;
    put(root / "proc/meminfo", "MemTotal: 1000 kB\nMemAvailable:      800 kB\nCommitLimit: 900 kB\nCommitted_AS: 600 kB\n");
    put(root / "proc/sys/vm/overcommit_memory", "2\n");
    put(root / "proc/self/cgroup", "0::/app/worker\n");
    // The parent limit is tighter after counting only clean file pages as reclaimable.
    put(root / "sys/fs/cgroup/app/worker/memory.max", "max\n");
    put(root / "sys/fs/cgroup/app/worker/memory.current", "100\n");
    put(root / "sys/fs/cgroup/app/memory.max", "500000\n");
    put(root / "sys/fs/cgroup/app/memory.current", "450000\n");
    put(root / "sys/fs/cgroup/app/memory.stat", "anon 300000\nfile 60000\nfile_dirty 15000\nfile_writeback 5000\n");
    auto o = observe_host_memory_linux(root);
    require(o.physical == 800u * 1024 && o.commit == 300u * 1024 && o.cgroup == 90000u && o.notes.empty(), "v2 observations");
    // Nested limits: the tightest level wins wherever it sits.
    put(root / "sys/fs/cgroup/app/worker/memory.max", "80100\n");
    put(root / "sys/fs/cgroup/app/worker/memory.stat", "file 10\nfile_dirty 20\n");
    o = observe_host_memory_linux(root);
    require(o.cgroup == 80000u, "child limit tighter than parent");
    put(root / "sys/fs/cgroup/memory.max", "450050\n");
    put(root / "sys/fs/cgroup/memory.current", "450000\n");
    o = observe_host_memory_linux(root);
    require(o.cgroup == 50u, "root level considered");
    put(root / "sys/fs/cgroup/app/worker/memory.max", "max\n");
    put(root / "sys/fs/cgroup/memory.max", "max\n");
    put(root / "proc/self/cgroup", "1:name=systemd:/x\n0::/app/worker\n");
    require(observe_host_memory_linux(root).cgroup == 90000u, "v2 line found among others");
    // Usage above the limit floors at zero.
    put(root / "sys/fs/cgroup/app/memory.current", "999999\n");
    require(observe_host_memory_linux(root).cgroup == 0u, "headroom floors at zero");
    // Overcommit not enforced: commit unknown; no cgroup limit: cgroup unknown.
    put(root / "proc/sys/vm/overcommit_memory", "0\n");
    put(root / "sys/fs/cgroup/app/memory.max", "max\n");
    o = observe_host_memory_linux(root);
    require(!o.commit && !o.cgroup && o.physical && o.notes.size() == 2, "unknown commit and cgroup");
    require(has_note(o, "commit not enforced (overcommit_memory=0)") && has_note(o, "no cgroup v2 memory limit"), "unknown reasons");
    // tmpfs/shmem pages sit in file but are swap-backed: only the file LRUs, else file - shmem, are reclaimable.
    put(root / "sys/fs/cgroup/app/memory.max", "500000\n");
    put(root / "sys/fs/cgroup/app/memory.current", "450000\n");
    put(root / "sys/fs/cgroup/app/memory.stat", "file 60000\nshmem 50000\nfile_dirty 1000\nfile_writeback 0\n");
    require(observe_host_memory_linux(root).cgroup == 59000u, "file minus shmem without the file LRUs");
    put(root / "sys/fs/cgroup/app/memory.stat",
        "file 60000\nshmem 50000\nactive_file 4000\ninactive_file 6000\nfile_dirty 1000\nfile_writeback 500\n");
    require(observe_host_memory_linux(root).cgroup == 58500u, "file LRUs preferred");
    // A cgroup at its limit holding mostly shmem has no reclaimable headroom for a large request.
    constexpr std::uint64_t gib = std::uint64_t(1) << 30;
    put(root / "sys/fs/cgroup/app/memory.max", std::to_string(8 * gib) + "\n");
    put(root / "sys/fs/cgroup/app/memory.current", std::to_string(8 * gib) + "\n");
    put(root / "sys/fs/cgroup/app/memory.stat", "anon " + std::to_string(2 * gib) + "\nfile " + std::to_string(6 * gib) +
        "\nshmem " + std::to_string(6 * gib) + "\nactive_file 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n");
    o = observe_host_memory_linux(root);
    const auto shmem = admit_host(anonymous(5 * gib), o);
    require(o.cgroup == 0u && !shmem.admitted && shmem.binding == "cgroup", "shmem is not reclaimable");
    // memory.high limits too: the effective limit is min(memory.max, memory.high).
    put(root / "sys/fs/cgroup/app/memory.max", "max\n");
    put(root / "sys/fs/cgroup/app/memory.high", "460000\n");
    put(root / "sys/fs/cgroup/app/memory.current", "450000\n");
    put(root / "sys/fs/cgroup/app/memory.stat", "active_file 4000\ninactive_file 6000\nfile_dirty 1000\nfile_writeback 500\n");
    require(observe_host_memory_linux(root).cgroup == 18500u, "memory.high alone limits");
    put(root / "sys/fs/cgroup/app/memory.max", "455000\n");
    require(observe_host_memory_linux(root).cgroup == 13500u, "memory.max below memory.high");
    put(root / "sys/fs/cgroup/app/memory.max", "max\n");
    put(root / "sys/fs/cgroup/app/memory.high", "max\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "no cgroup v2 memory limit"), "memory.max and memory.high both max");
}

void host_memory_v1() {
    TempDir dir;
    const auto& root = dir.path;
    put(root / "proc/meminfo", "MemAvailable: 4 kB\n");
    put(root / "proc/self/cgroup", "4:memory:/job\n1:cpu:/job\n");
    put(root / "sys/fs/cgroup/memory/job/memory.usage_in_bytes", "700\n");
    put(root / "sys/fs/cgroup/memory/job/memory.stat", "cache 100\ndirty 30\nwriteback 10\nhierarchical_memory_limit 1000\n");
    auto o = observe_host_memory_linux(root);
    require(o.cgroup == 1000u - 700u + 60u && o.physical == 4096u, "v1 hierarchical limit");
    require(has_note(o, "commit not enforced (overcommit_memory=unknown)"), "missing overcommit file");
    put(root / "sys/fs/cgroup/memory/job/memory.stat",
        "cache 100\ndirty 30\nwriteback 10\ntotal_cache 500\ntotal_dirty 100\ntotal_writeback 50\nhierarchical_memory_limit 1000\n");
    require(observe_host_memory_linux(root).cgroup == 1000u - 700u + 350u, "hierarchical totals preferred");
    put(root / "proc/self/cgroup", "7:cpuacct,memory,cpu:/job\n");
    require(observe_host_memory_linux(root).cgroup == 650u, "memory among several controllers");
    put(root / "sys/fs/cgroup/memory/job/memory.stat", "cache 1\nhierarchical_memory_limit 9223372036854771712\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "no cgroup v1 memory limit"), "unlimited v1 cgroup");
    put(root / "proc/self/cgroup", "3:cpu:/job\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "no memory cgroup"), "no memory controller");
    // systemd hybrid hosts list the unified hierarchy (0::) while the memory controller stays on v1:
    // with no limited v2 level the v1 limit still applies, so admission cannot overshoot it.
    put(root / "proc/meminfo", "MemAvailable: 1000000 kB\n");
    put(root / "proc/self/cgroup", "12:memory:/job\n1:name=systemd:/job\n0::/job\n");
    put(root / "sys/fs/cgroup/memory/job/memory.stat", "hierarchical_memory_limit 1000\n");
    put(root / "sys/fs/cgroup/memory/job/memory.usage_in_bytes", "700\n");
    o = observe_host_memory_linux(root);
    require(o.cgroup == 300u && !has_note(o, "cgroup v2") && !has_note(o, "no cgroup"), "hybrid host uses the v1 limit");
    const auto hybrid = admit_host(anonymous(500000), o);
    require(!hybrid.admitted && hybrid.binding == "cgroup" && hybrid.limit == 300u, "hybrid admission bound by the v1 limit");
    put(root / "sys/fs/cgroup/job/memory.max", "max\n");   // unified levels without a limit change nothing
    put(root / "sys/fs/cgroup/job/memory.current", "5\n");
    require(observe_host_memory_linux(root).cgroup == 300u, "unlimited unified level falls through to v1");
    put(root / "sys/fs/cgroup/job/memory.max", "105\n");   // a limited v2 level is the memory controller
    require(observe_host_memory_linux(root).cgroup == 100u, "limited v2 level wins");
    put(root / "sys/fs/cgroup/job/memory.max", "max\n");
    put(root / "sys/fs/cgroup/memory/job/memory.stat", "hierarchical_memory_limit 9223372036854771712\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "no cgroup v1 memory limit") && has_note(o, "no cgroup v2 memory limit"), "hybrid without limits");
    put(root / "proc/self/cgroup", "0::/job\n12:memory:/job\n");   // line order does not matter
    put(root / "sys/fs/cgroup/memory/job/memory.stat", "hierarchical_memory_limit 1000\n");
    require(observe_host_memory_linux(root).cgroup == 300u, "hybrid membership in either order");
    // v1 shmem is in cache but not reclaimable: the file LRUs, else cache - shmem, each preferring total_.
    put(root / "sys/fs/cgroup/memory/job/memory.stat", "cache 500\nshmem 300\ndirty 50\nhierarchical_memory_limit 1000\n");
    require(observe_host_memory_linux(root).cgroup == 450u, "v1 cache minus shmem");
    put(root / "sys/fs/cgroup/memory/job/memory.stat",
        "cache 500\nshmem 300\ntotal_cache 900\ntotal_shmem 800\ndirty 50\nhierarchical_memory_limit 1000\n");
    require(observe_host_memory_linux(root).cgroup == 350u, "v1 hierarchical shmem preferred");
    put(root / "sys/fs/cgroup/memory/job/memory.stat", "cache 500\nshmem 300\ntotal_active_file 100\n"
        "total_inactive_file 60\ninactive_file 1\ntotal_dirty 20\nhierarchical_memory_limit 1000\n");
    require(observe_host_memory_linux(root).cgroup == 440u, "v1 file LRUs preferred");
}

void host_memory_malformed() {
    TempDir dir;
    const auto& root = dir.path;
    // Nothing readable: unknown everywhere, never an exception.
    auto o = observe_host_memory_linux(root);
    require(!o.physical && !o.commit && !o.cgroup, "empty tree is unknown");
    require(has_note(o, "MemAvailable unavailable") && has_note(o, "cgroup membership unreadable"), "empty tree notes");
    put(root / "proc/meminfo", "MemAvailable: 12x kB\nCommitLimit: -5 kB\nCommitted_AS: 1 kB\nBad line\n");
    put(root / "proc/sys/vm/overcommit_memory", "2\n");
    put(root / "proc/self/cgroup", "garbage\n0::/a/b\n");
    put(root / "sys/fs/cgroup/a/b/memory.max", "12z\n");
    put(root / "sys/fs/cgroup/a/b/memory.current", "5\n");
    put(root / "sys/fs/cgroup/a/memory.max", "1000\n");
    put(root / "sys/fs/cgroup/a/memory.current", "400\n");
    put(root / "sys/fs/cgroup/a/memory.stat", "file 100\nfile_dirty lots\n");
    o = observe_host_memory_linux(root);
    require(!o.physical && has_note(o, "MemAvailable malformed"), "malformed MemAvailable");
    require(!o.commit && has_note(o, "commit headroom unknown"), "malformed commit counters");
    require(o.cgroup == 600u, "malformed level skipped; malformed stat counts no reclaimable pages");
    require(has_note(o, "memory.max, memory.high or memory.current malformed") && has_note(o, "memory.stat malformed"), "cgroup notes");
    for(const char* counters:{"file 100\nshmem x\n", "active_file 100\ninactive_file -1\n", "active_file 100\nfile_writeback 1.5\n"}) {
        put(root / "sys/fs/cgroup/a/memory.stat", counters);
        o = observe_host_memory_linux(root);
        require(o.cgroup == 600u && has_note(o, "memory.stat malformed"), "malformed reclaimable counters count nothing");
    }
    put(root / "sys/fs/cgroup/a/memory.stat", "file 100\n");
    put(root / "sys/fs/cgroup/a/memory.high", "lots\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "memory.max, memory.high or memory.current malformed"), "malformed memory.high skips the level");
    fs::remove(root / "sys/fs/cgroup/a/memory.high");
    put(root / "proc/meminfo", "MemAvailable: 18446744073709551615 kB\nCommitLimit: 99999999999999999999999\nCommitted_AS: 1\n");
    put(root / "proc/sys/vm/overcommit_memory", "two\n");
    o = observe_host_memory_linux(root);
    require(!o.physical && has_note(o, "MemAvailable malformed") && has_note(o, "overcommit_memory malformed"), "overflowing values");
    put(root / "proc/meminfo", "MemAvailable: 18446744073709551615\n");
    require(observe_host_memory_linux(root).physical == std::numeric_limits<std::uint64_t>::max(), "largest value without a unit");
    put(root / "proc/self/cgroup", "5:memory:/v1\n");
    put(root / "sys/fs/cgroup/memory/v1/memory.usage_in_bytes", "10\n");
    put(root / "sys/fs/cgroup/memory/v1/memory.stat", "hierarchical_memory_limit 1e9\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "hierarchical_memory_limit malformed"), "malformed v1 limit");
    put(root / "sys/fs/cgroup/memory/v1/memory.stat", "hierarchical_memory_limit 100\ncache x\n");
    put(root / "sys/fs/cgroup/memory/v1/memory.usage_in_bytes", "10\n");
    o = observe_host_memory_linux(root);
    require(o.cgroup == 90u && has_note(o, "memory.stat malformed"), "malformed v1 cache counts nothing reclaimable");
    put(root / "sys/fs/cgroup/memory/v1/memory.usage_in_bytes", "ten\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "memory.usage_in_bytes malformed"), "malformed v1 usage");
    put(root / "sys/fs/cgroup/memory/v1/memory.stat", std::string(std::size_t(1) << 20, 'x') + "\n");
    o = observe_host_memory_linux(root);
    require(!o.cgroup && has_note(o, "exceeds 1 MiB"), "oversized source file");
}

void host_memory_admission() {
    // The single-extent fixture of the native header: anonymous demand pays every domain.
    HostMemoryObservation o{800u * 1024, 300u * 1024, 90000u, {}};
    auto d = admit_host(anonymous(95000), o);
    require(!d.admitted && d.binding == "cgroup" && d.limit == 90000u, "cgroup binds");
    require(d.reason.find("exceeds cgroup working-set headroom") != std::string::npos, "denial reason");
    d = admit_host(anonymous(80000), o, 5000);
    require(d.admitted && d.limit == 85000u && d.reason == "fits cgroup working-set headroom", "fits after reserve");
    d = admit_host(anonymous(80001), o, 10000);
    require(!d.admitted && d.limit == 80000u, "one byte over");
    require(admit_host(anonymous(80000), o, 10000).admitted, "exactly fits");
    d = admit_host(anonymous(1), o, 90001);
    require(!d.admitted && d.limit == 0u, "reserve above headroom");
    // Working sets (memory.py admit_working_set): file pages never pay commit.
    const HostWorkingSet mixed{21, 3, 40};
    for(const auto& [observation, binding]:{std::pair<HostMemoryObservation, std::string>{{1000u, 23u, 1000u, {}}, "commit"},
                                            {{1000u, 1000u, 63u, {}}, "cgroup"}, {{63u, 1000u, 1000u, {}}, "physical"}}) {
        d = admit_host(mixed, observation);
        require(!d.admitted && d.binding == binding, "every domain can bind: " + binding);
    }
    d = admit_host({0, 0, 40}, HostMemoryObservation{100u, 0u, std::nullopt, {}});
    require(d.admitted && d.binding == "commit" && d.limit == 0u, "file working set does not consume commit");
    require(admit_host(mixed, HostMemoryObservation{64u, 24u, 64u, {}}).admitted, "exact fit in every domain");
    require(admit_host({30, 0, 0}, HostMemoryObservation{30u, std::nullopt, std::nullopt, {}}).admitted, "anonymous fits physical");
    require(!admit_host({11, 6, 40}, HostMemoryObservation{30u, 1000u, std::nullopt, {}}).admitted, "physical rejects the split");
    // Unknown observations fail closed unless demand is zero.
    d = admit_host({}, HostMemoryObservation{});
    require(d.admitted && d.binding.empty() && !d.limit && d.reason == "no incremental host working set", "zero demand");
    HostMemoryObservation unknown;
    unknown.notes = {"MemAvailable unavailable"};
    d = admit_host(anonymous(1), unknown);
    require(!d.admitted && d.binding.empty() && !d.limit && d.reason.find("unknown; MemAvailable unavailable") != std::string::npos, "unknown denies");
    require(!admit_host(anonymous(1), unknown, 0, false).admitted, "nothing known denies even without the physical requirement");
    const HostMemoryObservation commit_only{std::nullopt, 1000u, std::nullopt, {}};
    require(!admit_host(anonymous(1), commit_only).admitted, "physical required by default");
    d = admit_host(anonymous(1), commit_only, 0, false);
    require(d.admitted && d.binding == "commit", "commit alone when physical is optional");
    // Ties name the alphabetically first domain; the binding is the smallest margin, not the smallest headroom.
    require(admit_host(anonymous(10), HostMemoryObservation{100u, 100u, 100u, {}}).binding == "cgroup", "tie order");
    require(admit_host(anonymous(10), HostMemoryObservation{100u, 100u, std::nullopt, {}}).binding == "commit", "tie order without cgroup");
    require(admit_host({0, 0, 50}, HostMemoryObservation{100u, 60u, std::nullopt, {}}).binding == "physical", "margin, not headroom");
    // Exact for extreme values: sums beyond 64 bits never wrap into admission.
    constexpr auto max = std::numeric_limits<std::uint64_t>::max();
    const HostMemoryObservation huge{max, max, max, {}};
    require(!admit_host({max, max, max}, huge).admitted && !admit_host({max, 0, 0}, huge, 1).admitted, "no wraparound");
    require(admit_host({max, 0, 0}, huge).admitted && admit_host({max - 5, 0, 0}, huge, 5).admitted, "largest exact fit");
    d = admit_host({max, max, 0}, huge);
    require(!d.admitted && d.reason.find("more than 2^64") != std::string::npos, "oversized demand reported");
    // Randomized oracle with values small enough for exact signed arithmetic.
    std::mt19937_64 random(77);
    for(int round = 0; round < 20000; ++round) {
        auto pick = [&] { return std::uint64_t(random() % 4 == 0 ? random() % 4 : random() % 2000); };
        const HostWorkingSet demand{pick(), pick(), pick()};
        HostMemoryObservation obs;
        if(random() % 4) obs.physical = pick();
        if(random() % 3) obs.commit = pick();
        if(random() % 3) obs.cgroup = pick();
        const auto reserve = random() % 3 ? std::uint64_t(0) : pick();
        const bool require_physical = random() % 2 != 0;
        const auto got = admit_host(demand, obs, reserve, require_physical);
        const std::int64_t total = std::int64_t(demand.anonymous + demand.pinned + demand.file);
        std::vector<std::pair<std::string, std::pair<std::int64_t, std::int64_t>>> known;   // name -> (headroom, need)
        if(obs.cgroup) known.push_back({"cgroup", {std::int64_t(*obs.cgroup), total}});
        if(obs.commit) known.push_back({"commit", {std::int64_t(*obs.commit), std::int64_t(demand.anonymous + demand.pinned)}});
        if(obs.physical) known.push_back({"physical", {std::int64_t(*obs.physical), total}});
        if(total == 0) { require(got.admitted && got.binding.empty(), "oracle: zero demand"); continue; }
        if(known.empty() || (require_physical && !obs.physical)) { require(!got.admitted && got.binding.empty(), "oracle: unknown"); continue; }
        std::size_t best = 0;
        bool fits = true;
        for(std::size_t i = 0; i < known.size(); ++i) {
            const auto margin = known[i].second.first - std::int64_t(reserve) - known[i].second.second;
            if(margin < known[best].second.first - std::int64_t(reserve) - known[best].second.second) best = i;
            fits = fits && known[i].second.second + std::int64_t(reserve) <= known[i].second.first;
        }
        const auto limit = std::max<std::int64_t>(0, known[best].second.first - std::int64_t(reserve));
        require(got.admitted == fits && got.binding == known[best].first && got.limit == std::uint64_t(limit), "oracle: admission");
    }
}

void host_memory_live() {
    const auto live = observe_host_memory();
#if defined(__linux__) || defined(_WIN32)
    require(live.physical && *live.physical > 0, "live physical memory");
#else
    require(!live.notes.empty(), "unsupported platform explains itself");
#endif
}
}

int main(int argc, char** argv) {
    try {
        const std::string test = argc > 1 ? argv[1] : "all";
        const std::vector<std::pair<std::string, std::function<void()>>> tests{
            {"json_valid", json_valid}, {"json_rejects", json_rejects}, {"json_utf8", json_utf8}, {"json_strings", json_strings},
            {"json_numbers", json_numbers}, {"json_limits", json_limits}, {"json_access", json_access}, {"json_file", json_file},
            {"json_nonfinite", json_nonfinite},
            {"json_random", json_random}, {"sha256_vectors", sha256_vectors}, {"sha256_chunked", sha256_chunked},
            {"file_identity", file_identity}, {"file_change", file_change}, {"file_read_exact", file_read_exact}, {"file_read_slots", file_read_slots},
            {"file_read_chunks", file_read_chunks}, {"file_read_errors", file_read_errors}, {"file_map", file_map},
            {"file_open_errors", file_open_errors}, {"host_memory_v2", host_memory_v2}, {"host_memory_v1", host_memory_v1},
            {"host_memory_malformed", host_memory_malformed}, {"host_memory_admission", host_memory_admission},
            {"host_memory_live", host_memory_live}};
        bool found = false;
        for(const auto& [name, run]:tests) if(test == "all" || test == name) {
            run();
            found = true;
            std::cout << "crit core io passed: " << name << '\n';
        }
        require(found, "unknown test");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
