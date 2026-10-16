// Stop matching, text streams and the offset decoder against brute-force oracles and synthetic frontends. The offset
// decoder's oracles use only each frontend's full decode and the bytes of each token, never the decoder's rules.
#include <crit/text.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace crit::core;
namespace {
void require(bool value, const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F action, std::string_view expected) {
    try { action(); } catch(const std::exception& e) {
        require(std::string_view(e.what()).find(expected) != std::string_view::npos,"unexpected diagnostic");
        return;
    }
    throw std::runtime_error("expected rejection missing");
}
const std::string replacement = "\xEF\xBF\xBD";
std::string fffd(std::size_t count) {
    std::string out;
    for(std::size_t i = 0; i < count; ++i) out += replacement;
    return out;
}

// Brute-force reference: the stop whose match ends first in the whole text (ties: longest, then lowest index).
struct Expected { std::string released; bool stopped = false; std::size_t index = 0; };
Expected first_completed(const std::string& text, const std::vector<std::string>& stops) {
    for(std::size_t end = 1; end <= text.size(); ++end) {
        std::size_t best = stops.size();
        for(std::size_t i = 0; i < stops.size(); ++i) {
            const auto& s = stops[i];
            if(s.size() <= end && text.compare(end-s.size(),s.size(),s) == 0 && (best == stops.size() || s.size() > stops[best].size()))
                best = i;
        }
        if(best != stops.size()) return {text.substr(0,end-stops[best].size()),true,best};
    }
    return {text,false,0};
}
// Longest suffix of `text` that is a proper prefix of some stop.
std::size_t minimal_hold(const std::string& text, const std::vector<std::string>& stops) {
    std::size_t hold = 0;
    for(const auto& s:stops)
        for(std::size_t k = std::min(text.size(),s.size()-1); k > hold; --k)
            if(text.compare(text.size()-k,k,s,0,k) == 0) { hold = k; break; }
    return hold;
}

void stop_rules() {
    rejects([]{StopMatcher m({"a",""});},"nonempty");
    rejects([]{StopMatcher m(std::vector<std::string>(17,"x"));},"16");
    rejects([]{StopMatcher m({std::string(257,'x')});},"256");
    StopMatcher limits(std::vector<std::string>(16,std::string(256,'x')));
    require(limits.push("abc").released == "abc","sixteen 256-byte stops rejected or held unrelated text");

    StopMatcher none({});
    require(none.push("hello").released == "hello" && none.finish().empty(),"no stops changed text");
    rejects([&]{none.push("x");},"closed");

    // Minimal holdback: only a proper prefix of a stop is held.
    StopMatcher hold({"STOP"});
    auto step = hold.push("hello S");
    require(step.released == "hello " && !step.stopped,"holdback was not minimal");
    step = hold.push("TO");
    require(step.released.empty() && !step.stopped,"growing prefix released early");
    step = hold.push("x");
    require(step.released == "STOx","broken prefix not released");
    require(hold.finish().empty(),"finish released a stale holdback");
    StopMatcher tail({"STOP"});
    require(tail.push("abST").released == "ab" && tail.finish() == "ST","finish lost held text");

    // First completed stop wins, independent of chunking (the old scanner picked the earliest start).
    for(const auto& chunks:std::vector<std::vector<std::string>>{{"abcd"},{"a","b","c","d"},{"ab","cd"},{"abc","d"}}) {
        StopMatcher m({"abcd","bc"});
        std::string released; bool stopped = false; std::size_t index = 9;
        for(const auto& chunk:chunks) {
            const auto s = m.push(chunk);
            released += s.released;
            if(s.stopped) { stopped = true; index = s.stop_index; break; }
        }
        require(stopped && index == 1 && released == "a","overlapping stops depended on chunking");
    }
    // Completions at the same byte: the earliest start (longest stop), then the lowest index.
    StopMatcher tie({"bc","abc"});
    step = tie.push("xabcz");
    require(step.stopped && step.stop_index == 1 && step.released == "x","tie did not prefer the earliest start");
    StopMatcher same({"q","q"});
    step = same.push("aq");
    require(step.stopped && step.stop_index == 0 && step.released == "a","duplicate stops did not keep the lowest index");
    rejects([&]{same.push("x");},"closed");
    rejects([&]{same.finish();},"closed");
    // Self-overlapping stops keep their partial match across chunks (KMP borders).
    StopMatcher border({"aab"});
    require(border.push("aa").released.empty() && border.push("a").released == "a","border fallback lost a prefix");
    step = border.push("b");
    require(step.stopped && step.released.empty(),"self-overlapping stop missed");
    // Multibyte UTF-8 stops match byte-exactly, even when chunks split characters.
    StopMatcher utf({"\xE6\x97\xA5\xE6\x9C\xAC"});
    require(utf.push("x\xE6\x97").released == "x","multibyte prefix not held");
    step = utf.push("\xA5\xE6\x9C\xAC tail");
    require(step.stopped && step.released.empty(),"multibyte stop split across chunks missed");
}

void stop_oracle() {
    std::mt19937_64 rng(20261015);
    const std::vector<std::string> alphabet{"a","b","c","\xC3\xA9","\xE6\x97\xA5","\xF0\x9F\x99\x82"};
    const auto pick = [&](std::size_t n){return std::uniform_int_distribution<std::size_t>(0,n-1)(rng);};
    for(int trial = 0; trial < 4000; ++trial) {
        const bool wide = trial % 4 == 0;
        std::vector<std::string> stops(1+pick(4));
        for(auto& stop:stops) {
            const auto length = 1+pick(wide ? 4 : 3);
            for(std::size_t i = 0; i < length; ++i) stop += alphabet[pick(wide ? alphabet.size() : 3)];
        }
        std::string text;
        for(std::size_t i = 0, length = pick(40); i < length; ++i) text += alphabet[pick(wide ? alphabet.size() : 3)];
        const auto expected = first_completed(text,stops);
        StopMatcher matcher(stops);
        std::string released, seen;
        bool stopped = false; std::size_t index = 0;
        for(std::size_t at = 0; at < text.size() && !stopped;) {
            // Chunks at arbitrary byte boundaries, including inside UTF-8 sequences and empty chunks.
            const auto length = std::min(text.size()-at,pick(6));
            const auto step = matcher.push(std::string_view(text).substr(at,length));
            seen.append(text,at,length); at += length;
            released += step.released;
            if(step.stopped) { stopped = true; index = step.stop_index; break; }
            require(released.size() == seen.size()-minimal_hold(seen,stops),"holdback was not minimal");
            require(seen.compare(0,released.size(),released) == 0,"released text diverged from input");
        }
        if(!stopped) released += matcher.finish();
        require(stopped == expected.stopped && released == expected.released,"stop matching depended on chunking");
        require(!stopped || index == expected.index,"wrong stop index");
    }
}

void stop_linear() {
    // Every byte extends a 255-byte partial match on all sixteen stops: rescanning the holdback per push would
    // cost about 10^10 byte comparisons here, while the incremental matcher does a few per stop per byte.
    std::vector<std::string> stops;
    for(int i = 0; i < 16; ++i) stops.push_back(std::string(255,'a')+char('b'+i));
    StopMatcher matcher(stops);
    std::size_t released = 0;
    const auto started = std::chrono::steady_clock::now();
    for(int i = 0; i < 200000; ++i) released += matcher.push("a").released.size();
    const auto elapsed = std::chrono::steady_clock::now()-started;
    require(released == 200000-255,"holdback exceeded the longest proper stop prefix");
    require(elapsed < std::chrono::seconds(20),"stop matching was not linear in the text length");
    const auto step = matcher.push("z");
    require(step.released.size() == 256 && !step.stopped,"mismatch did not release the holdback");
    const auto stopped = matcher.push(std::string(300,'a')+"q");
    require(stopped.stopped && stopped.stop_index == 15 && stopped.released == std::string(45,'a'),"last stop missed");
}

// Each token decodes to a fixed string; incomplete text is held until finish. Counts decoder calls.
struct ScriptDecoder final:TextDecoder {
    std::map<Token,std::string> pieces;
    std::shared_ptr<int> pushes = std::make_shared<int>(0);
    std::string held;
    bool hold_all = false;
    std::string push(std::span<const Token> tokens) override {
        std::string out;
        for(auto t:tokens) {
            ++*pushes;
            if(t == 13) throw std::runtime_error("script decoder failure");
            const auto found = pieces.find(t);
            out += found == pieces.end() ? std::string(1,char('a'+t%26)) : found->second;
        }
        if(hold_all) { held += out; return {}; }
        return out;
    }
    std::string finish() override { return std::exchange(held,{}); }
};
std::unique_ptr<TextStream> script(std::vector<std::string> stops, std::shared_ptr<int>* pushes = nullptr, bool hold_all = false) {
    auto decoder = std::make_unique<ScriptDecoder>();
    decoder->pieces = {{0,"S"},{1,"T"},{2,"O"},{3,"P"},{4,"xy"},{5,"STOP"}};
    decoder->hold_all = hold_all;
    if(pushes) *pushes = decoder->pushes;
    return std::make_unique<TextStream>(std::move(decoder),std::move(stops));
}
void text_stream() {
    rejects([]{TextStream s(nullptr,{});},"decoder");
    rejects([]{script({""});},"nonempty");
    // A stop across token boundaries: tokens_used names the completing token; later tokens stay undecoded.
    std::shared_ptr<int> pushes;
    auto stream = script({"TOP"},&pushes);
    auto piece = stream->push(std::vector<Token>{4,0,1});
    require(piece.text == "xyS" && !piece.stopped && piece.tokens_used == 3,"held prefix escaped");
    piece = stream->push(std::vector<Token>{2,3,4,4});
    require(piece.stopped && piece.tokens_used == 2 && piece.text.empty(),"stop across tokens not trimmed");
    require(*pushes == 5,"tokens after the stop were decoded");
    rejects([&]{stream->push(std::vector<Token>{4});},"text stream is closed");
    rejects([&]{stream->finish();},"text stream is closed");
    // A stop inside one token's text, and text before it in the same token.
    stream = script({"OP"});
    piece = stream->push(std::vector<Token>{4,5,4});
    require(piece.stopped && piece.tokens_used == 2 && piece.text == "xyST","stop inside one token's text");
    // finish() flushes held text; without stops everything passes.
    stream = script({"STOP"});
    piece = stream->push(std::vector<Token>{4,0,1});
    require(piece.text == "xy","held text released early");
    piece = stream->finish();
    require(piece.text == "ST" && !piece.stopped && piece.tokens_used == 0,"finish lost held text");
    // A stop may complete in the decoder's flushed tail.
    stream = script({"yS"},nullptr,true);
    piece = stream->push(std::vector<Token>{4,0});
    require(piece.text.empty() && !piece.stopped,"holding decoder released text");
    piece = stream->finish();
    require(piece.stopped && piece.text == "x","stop in the flushed tail missed");
    // A decoder exception closes the stream.
    stream = script({});
    rejects([&]{stream->push(std::vector<Token>{4,13});},"script decoder failure");
    rejects([&]{stream->push(std::vector<Token>{4});},"text stream is closed");
}

// Lossy UTF-8 decoding: each maximal invalid subpart becomes one U+FFFD (the usual tokenizer behavior).
std::string lossy(std::string_view bytes) {
    std::string out;
    for(std::size_t i = 0; i < bytes.size();) {
        const auto b = static_cast<unsigned char>(bytes[i]);
        if(b < 0x80) { out.push_back(char(b)); ++i; continue; }
        std::size_t need = 0; unsigned char lo = 0x80, hi = 0xBF;
        if(b >= 0xC2 && b <= 0xDF) need = 1;
        else if(b >= 0xE0 && b <= 0xEF) { need = 2; if(b == 0xE0) lo = 0xA0; if(b == 0xED) hi = 0x9F; }
        else if(b >= 0xF0 && b <= 0xF4) { need = 3; if(b == 0xF0) lo = 0x90; if(b == 0xF4) hi = 0x8F; }
        else { out += replacement; ++i; continue; }
        std::size_t j = 1;
        for(; j <= need && i+j < bytes.size(); ++j) {
            const auto c = static_cast<unsigned char>(bytes[i+j]);
            if(c < lo || c > hi) break;
            lo = 0x80; hi = 0xBF;
        }
        if(j == need+1) out.append(bytes.substr(i,j)); else out += replacement;
        i += j;
    }
    return out;
}
bool continuation(char byte) { return (static_cast<unsigned char>(byte) & 0xC0) == 0x80; }
// Byte-level frontend: ids 0-255 are bytes, later ids are merged byte strings that may straddle characters, and
// `empty` ids decode to nothing (like skipped special tokens). With `fallback` it is a SentencePiece frontend with byte
// fallback, like Hugging Face's ByteFallback decoder: ids 0-255 are <0xNN> tokens, and a run of them (skipped special
// tokens do not end it) decodes as its bytes when they form valid UTF-8 and as one U+FFFD per byte otherwise.
// `strip_space` drops one leading space of the decoded text (SentencePiece), `ignore_invalid` drops every U+FFFD (a
// frontend that shows nothing for an incomplete character) and `fail` makes detokenize throw.
struct ByteFrontend final:TextFrontend {
    std::vector<std::string> pieces;
    std::map<std::string,Token> merges;
    Token empty = -1, fail = -1;
    bool fallback = false, strip_space = false, ignore_invalid = false;
    mutable std::size_t widest = 0;
    std::vector<Token> tokenize(std::string_view) const override { throw std::logic_error("unused"); }
    std::string detokenize(std::span<const Token> tokens) const override {
        widest = std::max(widest,tokens.size());
        std::string text, run;
        const auto flush = [&]{ text += lossy(run) == run ? run : fffd(run.size()); run.clear(); };
        for(auto t:tokens) {
            if(t < 0 || std::size_t(t) >= pieces.size()) throw std::invalid_argument("token outside fixture vocabulary");
            if(t == fail) throw std::runtime_error("fixture detokenize failure");
            if(!fallback || t == empty) text += pieces[std::size_t(t)];
            else if(t < 256) run += pieces[std::size_t(t)];
            else { flush(); text += pieces[std::size_t(t)]; }
        }
        if(fallback) flush(); else text = lossy(text);
        if(ignore_invalid) for(std::size_t at; (at = text.find(replacement)) != std::string::npos;) text.erase(at,replacement.size());
        if(strip_space && !text.empty() && text.front() == ' ') text.erase(0,1);
        return text;
    }
    Token add(const std::string& piece) {
        const auto found = merges.find(piece);
        if(found != merges.end()) return found->second;
        pieces.push_back(piece);
        return merges[piece] = Token(pieces.size()-1);
    }
    std::string bytes(std::span<const Token> tokens) const {
        std::string out;
        for(auto t:tokens) out += pieces[std::size_t(t)];
        return out;
    }
};
std::shared_ptr<ByteFrontend> byte_frontend(const std::vector<std::string>& corpus) {
    auto f = std::make_shared<ByteFrontend>();
    for(int b = 0; b < 256; ++b) { f->pieces.push_back(std::string(1,char(b))); f->merges[f->pieces.back()] = Token(b); }
    for(const auto& text:corpus)
        for(std::size_t i = 0; i < text.size(); ++i)
            for(std::size_t n = 2; n <= 4 && i+n <= text.size(); ++n) f->add(text.substr(i,n));
    f->pieces.push_back({});
    f->empty = Token(f->pieces.size()-1);
    return f;
}
std::vector<Token> joined(const std::vector<Token>& first, const std::vector<Token>& second) {
    auto all = first;
    all.insert(all.end(),second.begin(),second.end());
    return all;
}

// The oracles below use only the frontend's full decode and the bytes of each token, never the decoder's rules.
// ByteText: the text `output` adds after `context`, from the bytes alone: the output's bytes from the start of the
// character the context ends in (the context's complete characters stay context text; with no context bytes, a
// SentencePiece frontend's stripped leading space goes too), and per output token the end of its bytes in that text.
// Nullopt when that text is not valid UTF-8.
struct ByteText { std::string text; std::vector<std::size_t> ends; };
std::optional<ByteText> byte_text(const ByteFrontend& f, const std::vector<Token>& context, const std::vector<Token>& output) {
    const auto prefix = f.bytes(context), whole = prefix+f.bytes(output);
    auto start = prefix.size();
    while(start && start < whole.size() && continuation(whole[start])) --start;
    if(prefix.empty() && f.strip_space && !whole.empty() && whole.front() == ' ') start = 1;
    ByteText out{whole.substr(std::min(start,whole.size())),{}};
    if(lossy(out.text) != out.text) return std::nullopt;
    auto end = prefix.size();
    for(const auto t:output) {
        end += f.pieces[std::size_t(t)].size();
        out.ends.push_back(end > start ? end-start : 0);
    }
    return out;
}
// Whether the context's bytes end with the first one or two bytes of a literal U+FFFD that the output's bytes complete,
// and, for a SentencePiece frontend, its final byte run holds no other byte.
bool cut_literal(const ByteFrontend& f, const std::vector<Token>& context, const std::vector<Token>& output) {
    const auto prefix = f.bytes(context), whole = prefix+f.bytes(output);
    std::size_t run = 0;
    for(auto at = context.size(); at-- && (context[at] == f.empty || context[at] < 256);) run += context[at] != f.empty;
    for(const std::size_t k:{std::size_t(1),std::size_t(2)})
        if(prefix.ends_with(replacement.substr(0,k)) && whole.compare(prefix.size()-k,3,replacement) == 0) return !f.fallback || run == k;
    return false;
}
// ByteText for TextStream: a literal U+FFFD that cut_literal finds stays context text when the context's decode is a
// prefix of the full decode, like the full-decode oracle says.
std::optional<ByteText> stream_bytes(const ByteFrontend& f, const std::vector<Token>& context, const std::vector<Token>& output) {
    auto bytes = byte_text(f,context,output);
    if(bytes && cut_literal(f,context,output) && f.detokenize(joined(context,output)).starts_with(f.detokenize(context))) {
        bytes->text.erase(0,replacement.size());
        for(auto& end:bytes->ends) end = end > replacement.size() ? end-replacement.size() : 0;
    }
    return bytes;
}
// The text streaming `output` after `context` must release: detokenize(context + output) minus detokenize(context)
// when the latter is a prefix of the former (U+FFFD that the output leaves in place stays context text, and so does a
// literal U+FFFD whose first bytes end the context: the decode cannot tell the output's bytes that complete it from an
// incomplete character left in place); otherwise the context ends inside a character that the output completes, and
// the text is ByteText's when the full decode ends with it. A SentencePiece byte run that ends inside a character
// decodes as U+FFFD per byte, so for that frontend ByteText's text comes first, unless the run is only those first
// bytes of a literal U+FFFD. Nullopt when nothing applies (a byte run that the output makes invalid as a whole).
struct OracleCounts { std::size_t prefix = 0, completed = 0, undefined = 0, seeded = 0; };
std::optional<std::string> expected_text(const ByteFrontend& f, const std::vector<Token>& context, const std::vector<Token>& output,
                                         OracleCounts& counts) {
    const auto head = context.empty() ? std::string{} : f.detokenize(context);
    const auto full = f.detokenize(joined(context,output));
    const auto bytes = byte_text(f,context,output);
    const bool completed = bytes && full.ends_with(bytes->text);
    if(full.starts_with(head) && (!f.fallback || !completed || full.substr(head.size()) == bytes->text || cut_literal(f,context,output))) {
        ++counts.prefix;
        return full.substr(head.size());
    }
    if(completed) { ++counts.completed; return bytes->text; }
    ++counts.undefined;
    return std::nullopt;
}
// What a TextStream over the decoder must return, from ByteText: the text before the first completed stop, and the
// number of tokens up to the one whose bytes complete the stop's last character. A trailing U+FFFD may be a character
// whose remaining bytes have not arrived, so a stop that ends with U+FFFD completes once the next character that is
// not U+FFFD is complete, or in the flushed tail (the fixtures keep such runs far below offset_decoder_hold).
struct StreamResult { std::string text; bool stopped = false; std::size_t used = 0; };
StreamResult expected_stream(const ByteText& bytes, const std::vector<std::string>& stops) {
    const auto first = first_completed(bytes.text,stops);
    if(!first.stopped) return {bytes.text,false,bytes.ends.size()};
    auto end = first.released.size()+stops[first.index].size();
    if(stops[first.index].ends_with(replacement)) {
        while(bytes.text.compare(end,replacement.size(),replacement) == 0) end += replacement.size();
        if(end == bytes.text.size()) return {first.released,true,bytes.ends.size()};
        do ++end; while(end < bytes.text.size() && continuation(bytes.text[end]));
    }
    std::size_t used = 0;
    while(bytes.ends.at(used) < end) ++used;
    return {first.released,true,used+1};
}
std::string show(const std::string& text) {
    std::string out;
    for(const char c:text) {
        const auto b = static_cast<unsigned char>(c);
        if(b >= 0x20 && b < 0x7F) out += c;
        else { static const char* digits = "0123456789abcdef"; out += "\\x"; out += digits[b>>4]; out += digits[b&15]; }
    }
    return out;
}
std::string show(const std::vector<Token>& tokens) {
    std::string out;
    for(const auto t:tokens) out += (out.empty() ? "" : " ")+std::to_string(t);
    return out;
}
std::size_t chunk(std::mt19937_64& rng) { return std::uniform_int_distribution<std::size_t>(0,4)(rng); }
// Streams `output` through offset_decoder in random chunks and compares the concatenation with the oracle.
void check_stream(const std::shared_ptr<ByteFrontend>& f, const std::vector<Token>& context, const std::vector<Token>& output,
                  std::mt19937_64& rng, OracleCounts& counts) {
    auto decoder = offset_decoder(f,context);
    std::string streamed;
    for(std::size_t at = 0; at < output.size();) {
        const auto length = std::min(output.size()-at,chunk(rng));
        const auto piece = decoder->push(std::span<const Token>(output).subspan(at,length));
        require(piece.empty() || !continuation(piece.front()),"piece starts inside a character");
        streamed += piece; at += length;
    }
    streamed += decoder->finish();
    const auto expected = expected_text(*f,context,output,counts);
    if(expected && streamed != *expected) {
        std::cerr << "context [" << show(context) << "] output [" << show(output) << "]\nexpected [" << show(*expected)
                  << "] streamed [" << show(streamed) << "]\n";
        throw std::runtime_error("offset decoder output differs from the full decode");
    }
}
// Streams `output` through a TextStream over offset_decoder and compares text, stop and tokens_used with the oracle.
void check_text_stream(const std::shared_ptr<ByteFrontend>& f, const std::vector<Token>& context, const std::vector<Token>& output,
                       const std::vector<std::string>& stops, std::mt19937_64& rng) {
    const auto bytes = stream_bytes(*f,context,output);
    if(!bytes) return;
    const auto expected = expected_stream(*bytes,stops);
    TextStream stream(offset_decoder(f,context),stops);
    StreamResult got;
    for(std::size_t at = 0; at < output.size() && !got.stopped;) {
        const auto length = std::min(output.size()-at,chunk(rng));
        const auto piece = stream.push(std::span<const Token>(output).subspan(at,length));
        got.text += piece.text;
        got.stopped = piece.stopped;
        got.used = at+piece.tokens_used;
        at += length;
    }
    if(!got.stopped) {
        const auto tail = stream.finish();
        got.text += tail.text;
        got.stopped = tail.stopped;
    }
    if(got.text != expected.text || got.stopped != expected.stopped || (expected.stopped && got.used != expected.used)) {
        for(const auto& stop:stops) std::cerr << "stop [" << show(stop) << "] ";
        std::cerr << "text [" << show(bytes->text) << "]\n";
        std::cerr << "context [" << show(context) << "] output [" << show(output) << "]\nexpected [" << show(expected.text) << "] "
                  << expected.stopped << " " << expected.used << " got [" << show(got.text) << "] " << got.stopped << " " << got.used << "\n";
        throw std::runtime_error("text stream differs from the byte-offset oracle");
    }
}
// Up to three stops from the oracle's text (whole characters), sometimes one that never occurs.
std::vector<std::string> pick_stops(const std::string& text, std::mt19937_64& rng) {
    std::vector<std::size_t> starts;
    for(std::size_t i = 0; i < text.size(); ++i) if(!continuation(text[i])) starts.push_back(i);
    std::vector<std::string> stops;
    const auto count = std::uniform_int_distribution<int>(0,3)(rng);
    for(int i = 0; i < count; ++i) {
        if(starts.empty() || std::uniform_int_distribution<int>(0,4)(rng) == 0) { stops.push_back("\x01none"); continue; }
        const auto first = std::uniform_int_distribution<std::size_t>(0,starts.size()-1)(rng);
        const auto last = std::min(starts.size(),first+std::uniform_int_distribution<std::size_t>(1,4)(rng));
        stops.push_back(text.substr(starts[first],(last < starts.size() ? starts[last] : text.size())-starts[first]));
    }
    return stops;
}
// Streams `output` after `context`, one token per push.
std::string stream_all(const std::shared_ptr<const TextFrontend>& f, const std::vector<Token>& context, const std::vector<Token>& output) {
    auto decoder = offset_decoder(f,context);
    std::string out;
    for(const auto t:output) out += decoder->push(std::span<const Token>(&t,1));
    return out+decoder->finish();
}
// Decoder output equals the oracle, which must be defined.
void require_full(const std::shared_ptr<ByteFrontend>& f, const std::vector<Token>& context, const std::vector<Token>& output,
                  const char* message) {
    OracleCounts counts;
    const auto expected = expected_text(*f,context,output,counts);
    require(expected.has_value(),"fixture case has no full-decode oracle");
    f->widest = 0;   // measure the decoder's own detokenize() calls
    const auto streamed = stream_all(f,context,output);
    if(streamed != *expected) {
        std::cerr << "expected [" << show(*expected) << "] streamed [" << show(streamed) << "]\n";
        throw std::runtime_error(message);
    }
}
// TextStream over offset_decoder, one token per push: (text, stopped, tokens kept).
StreamResult stream_text(const std::shared_ptr<const TextFrontend>& f, const std::vector<Token>& context,
                         const std::vector<Token>& output, const std::vector<std::string>& stops) {
    TextStream stream(offset_decoder(f,context),stops);
    StreamResult got;
    for(std::size_t i = 0; i < output.size(); ++i) {
        const auto piece = stream.push(std::span<const Token>(output).subspan(i,1));
        got.text += piece.text;
        if(piece.stopped) { got.stopped = true; got.used = i+1; return got; }
    }
    const auto tail = stream.finish();
    got.text += tail.text;
    got.stopped = tail.stopped;
    got.used = output.size();
    return got;
}
// TextStream result equals the byte-offset oracle, which must be defined.
void require_stream(const std::shared_ptr<ByteFrontend>& f, const std::vector<Token>& context, const std::vector<Token>& output,
                    const std::vector<std::string>& stops, const char* message) {
    const auto bytes = stream_bytes(*f,context,output);
    require(bytes.has_value(),"fixture case has no byte-offset oracle");
    const auto expected = expected_stream(*bytes,stops);
    const auto got = stream_text(f,context,output,stops);
    if(got.text != expected.text || got.stopped != expected.stopped || got.used != expected.used) {
        std::cerr << "expected [" << show(expected.text) << "] " << expected.stopped << " " << expected.used << " got ["
                  << show(got.text) << "] " << got.stopped << " " << got.used << "\n";
        throw std::runtime_error(message);
    }
}
// Tokens for `text` whose boundaries all fall inside characters of `width` bytes: its first byte, then merged pieces
// of `width` bytes that end one character and start the next (byte-level merges that straddle characters), then the
// rest.
std::vector<Token> straddled(ByteFrontend& f, const std::string& text, std::size_t width) {
    std::vector<Token> tokens{Token(static_cast<unsigned char>(text.front()))};
    std::size_t at = 1;
    for(; at+width < text.size(); at += width) tokens.push_back(f.add(text.substr(at,width)));
    tokens.push_back(f.add(text.substr(at)));
    return tokens;
}
std::string repeat(const std::string& text, std::size_t count) {
    std::string out;
    for(std::size_t i = 0; i < count; ++i) out += text;
    return out;
}
std::vector<Token> byte_ids(const std::string& bytes) {
    std::vector<Token> tokens;
    for(const char c:bytes) tokens.push_back(Token(static_cast<unsigned char>(c)));
    return tokens;
}
// SentencePiece pieces with byte fallback: word pieces (U+2581, stored as a space, marks a word start), <0xNN> ids for
// everything else, a piece that is a literal U+FFFD, and `empty` for skipped special tokens.
struct Pieces { std::shared_ptr<ByteFrontend> f; std::vector<Token> words; Token hello = 0, world = 0, ok = 0, replacement_piece = 0; };
Pieces sentencepiece() {
    Pieces p{byte_frontend({}),{}};
    p.f->fallback = true;
    p.f->strip_space = true;
    for(const auto* word:{"hello","world","the","caf\xC3\xA9","ing","lo",",","s","ok","d","mor","\xC3\xBC" "ber"}) {
        const std::string piece = word;
        p.words.push_back(p.f->add(" "+piece));
        p.words.push_back(p.f->add(piece));
    }
    p.replacement_piece = p.f->add(" "+replacement);
    p.hello = p.f->merges.at(" hello"); p.world = p.f->merges.at(" world"); p.ok = p.f->merges.at(" ok");
    return p;
}
// Decodes a longer window differently: an 'x' followed by more text becomes U+00FC, so the decode of the released
// prefix is not a prefix of the next decode.
struct Rewriting final:TextFrontend {
    std::vector<Token> tokenize(std::string_view) const override { throw std::logic_error("unused"); }
    std::string detokenize(std::span<const Token> tokens) const override {
        std::string out;
        for(std::size_t i = 0; i < tokens.size(); ++i)
            out += tokens[i] == 'x' && i+1 < tokens.size() ? std::string("\xC3\xBC") : std::string(1,char(tokens[i]));
        return out;
    }
};
const std::string lambda = "\xCE\xBB", gamma_ = "\xCE\xB3", de = "\xD0\xB4", sun = "\xE6\x97\xA5", book = "\xE6\x9C\xAC",
                  smile = "\xF0\x9F\x98\x80", a_umlaut = "\xC3\xA4", clef = "\xF0\x9D\x84\x9E";

void offset_rules() {
    rejects([]{offset_decoder(nullptr);},"frontend");
    auto f = byte_frontend({});
    auto decoder = offset_decoder(f);
    require(decoder->push(std::vector<Token>{'h','i'}) == "hi" && decoder->finish().empty(),"plain bytes");
    rejects([&]{decoder->push(std::vector<Token>{'x'});},"text decoder is closed");
    rejects([&]{decoder->finish();},"text decoder is closed");
    // Exceptions close the decoder.
    f->fail = 'z';
    decoder = offset_decoder(f);
    rejects([&]{decoder->push(std::vector<Token>{'a','z'});},"fixture detokenize failure");
    rejects([&]{decoder->push(std::vector<Token>{'a'});},"text decoder is closed");
    f->fail = -1;
    // A split character is held until complete; finish flushes an incomplete tail.
    decoder = offset_decoder(f);
    require(decoder->push(std::vector<Token>{0xE6,0x97}).empty(),"partial character escaped");
    require(decoder->push(std::vector<Token>{0xA5}) == sun,"completed character not released");
    require(decoder->push(std::vector<Token>{0xE6}).empty() && decoder->finish() == replacement,"finish did not flush the tail");
    // A context that ends inside a character: the output starts with the character it completes, for 2-, 3- and 4-byte
    // characters, a merged token that completes it, and complete characters before it in the same token.
    require(stream_all(f,{'a','b',0xE6},{0x97,0xA5,'c'}) == sun+"c","context character completed by the output");
    require(stream_all(f,{'a',0xC3},{0xA9}) == "\xC3\xA9" && stream_all(f,{0xF0,0x9F},{0x98,0x80,'x'}) == smile+"x","short contexts");
    require(stream_all(f,{'a',0xC3},{f->add("\xA9xy")}) == "\xC3\xA9xy","merged token completing a context character");
    require(stream_all(f,{f->add("g\xC3")},{f->add("\xA4g")}) == a_umlaut+"g","complete character before the split one");
    require(stream_all(f,{'a',0xFF,0xE6},{0x97,0xA5}) == sun,"invalid byte before the split character");
    for(const auto& [context,output]:std::vector<std::pair<std::vector<Token>,std::vector<Token>>>{
            {{'a','b',0xE6},{0x97,0xA5,'c'}},{{0xF0,0x9F},{0x98,0x80,'x'}},{{f->add("g\xC3")},{f->add("\xA4g")}},{{'a',0xFF,0xE6},{0x97,0xA5}}})
        require_full(f,context,output,"context character differs from the full decode");
    // U+FFFD that the output leaves in place stays context text: an incomplete character that the output does not
    // complete, and an invalid byte.
    require(stream_all(f,{'a','b',0xE6},{0x97,'c'}) == "c" && stream_all(f,{'a',0xE6},{'b'}) == "b","incomplete context tail");
    require(stream_all(f,{'a',0xFF},{0x97,'c'}) == fffd(1)+"c","invalid context byte");
    // A context that ends inside a literal U+FFFD that the output completes: the decode cannot tell it from an
    // incomplete character left in place, so the U+FFFD the context decode shows stays context text.
    for(const auto& [context,output]:std::vector<std::pair<std::vector<Token>,std::vector<Token>>>{
            {{'x','a',0xEF},{0xBF,0xBD,'o','k'}},{{'x',0xEF,0xBF},{0xBD,'o','k'}},{{'x',0xEF},{f->add("\xBF\xBDok")}}}) {
        require(stream_all(f,context,output) == "ok","completed literal U+FFFD left the context");
        require_full(f,context,output,"completed literal U+FFFD differs from the full decode");
    }
    // A complete literal U+FFFD whose last bytes a merge joins to the first bytes of the character the context ends in
    // stays context text, also for a stop: the output starts with the character it completes.
    for(const auto& [context,output,text]:std::vector<std::tuple<std::vector<Token>,std::vector<Token>,std::string>>{
            {{'x',0xEF,0xBF,f->add("\xBD\xC3")},{0xA9,'o','k'},"\xC3\xA9ok"},
            {{'x',f->add("\xEF\xBF\xBD\xE6")},{0x97,0xA5,'!'},sun+"!"},
            {{'x',0xEF,f->add("\xBF\xBD\xF0\x9F")},{0x98,0x80},smile}}) {
        require(stream_all(f,context,output) == text,"literal U+FFFD merged into the split character left the context");
        require_full(f,context,output,"literal U+FFFD merged into the split character differs from the full decode");
        require_stream(f,context,output,{fffd(1)},"literal U+FFFD merged into the split character matched a stop");
    }
    // A stop over the completed character ends at the token that completes it.
    const auto split = std::vector<Token>{f->add(lambda+lambda),f->add(lambda+"\xCE")};
    const auto stopped = stream_text(f,split,{0xBB,'x'},{lambda});
    require(stopped.stopped && stopped.text.empty() && stopped.used == 1,"stop over a context character the output completes");
    // Only a short tail of the context is decoded.
    f->widest = 0;
    decoder = offset_decoder(f,std::vector<Token>(50,'q'));
    require(decoder->push(std::vector<Token>{'r'}) == "r" && f->widest <= offset_decoder_context+4,"context window was not short");
    // The context provides SentencePiece spacing, also when its decode ends with U+FFFD (a literal replacement
    // character, an invalid byte, an incomplete character) and when special tokens follow it.
    auto sp = sentencepiece();
    require(stream_all(sp.f,{sp.hello},{sp.world}) == " world" && stream_all(sp.f,{},{sp.world}) == "world","context spacing");
    for(const auto& context:std::vector<std::vector<Token>>{{sp.hello,sp.replacement_piece},{sp.hello,0xFF},{0xFF},{sp.hello,0xE6},
                                                             {sp.replacement_piece},{sp.hello,sp.f->empty,sp.f->empty,sp.f->empty,
                                                             sp.f->empty,sp.f->empty},{sp.hello,sp.f->empty}}) {
        require(stream_all(sp.f,context,{sp.world}) == " world","context lost its spacing");
        require_full(sp.f,context,{sp.world},"context spacing differs from the full decode");
    }
    // A decode that rewrites released text: released text stays, and output resumes after it.
    auto rewriting = std::make_shared<Rewriting>();
    decoder = offset_decoder(rewriting);
    std::vector<std::string> pieces;
    for(const Token t:{'a','x','b','c'}) pieces.push_back(decoder->push(std::vector<Token>{t}));
    pieces.push_back(decoder->finish());
    require(pieces == std::vector<std::string>({"a","x","b","c",""}),"rewritten prefix was retracted or repeated");
    require(stream_all(rewriting,{'x'},{'y','z'}) == "yz","rewritten context leaked into output");
}
// Long runs: a run of straddling merges, invalid bytes and tokens without text reaching offset_decoder_hold and
// beyond. The output equals the full decode and the window stays short: these runs have no clean boundary, so each
// cut leaves offset_decoder_context + 3 tokens, and offset_decoder_hold more arrive before the next one.
constexpr std::size_t short_window = offset_decoder_context+3+offset_decoder_hold;
void offset_bounds() {
    auto f = byte_frontend({});
    // Every token of these runs ends inside a character, so no prefix of the run decodes cleanly.
    for(const auto& [character,count]:std::vector<std::pair<std::string,std::size_t>>{{lambda,40},{gamma_,70},{de,17},
                                                                                        {sun,17},{smile,20},{a_umlaut,33}}) {
        const auto text = repeat(character,count);
        const auto run = straddled(*f,text,character.size());
        require_full(f,{},run,"straddling run differs from the full decode");
        require(f->widest <= short_window,"straddling run grew the window");
        require_full(f,{'g'},run,"straddling run after a prompt");
        require(f->widest <= short_window,"straddling run after a prompt grew the window");
        // The prompt ends with the run's first byte, which the output's first token completes.
        require_full(f,{'g',run.front()},std::vector<Token>(run.begin()+1,run.end()),"straddling run completing a prompt character");
        // The same run with a stop over all its characters, and over its first one.
        require_stream(f,{},joined({'g'},straddled(*f,text,character.size())),{text},"stop over a straddling run");
        require_stream(f,{},straddled(*f,text,character.size()),{character},"stop at the first straddled character");
    }
    // Runs of invalid bytes: the full decode, U+FFFD by U+FFFD.
    require_full(f,{},std::vector<Token>(200,0x80),"run of invalid bytes");
    require(f->widest <= short_window,"invalid run grew the window");
    for(std::size_t before = 10; before <= 40; ++before) for(const bool merged:{false,true}) {
        std::vector<Token> output(before,0x80);
        output.push_back(merged ? f->add(" \xE6") : Token(0xE6));
        output.insert(output.end(),{0x97,0xA5,'x'});
        require_full(f,{},output,"bounded release broke a character whose bytes are adjacent");
        require_full(f,{'a',0xFF},output,"bounded release after an invalid context byte");
    }
    // Tokens without text (skipped special tokens) do not count: a character split by a long run of them still
    // completes, text after them is unaffected, and the window stays short.
    for(const std::size_t empties:{1u,3u,4u,15u,16u,17u,31u,32u,33u,64u,200u}) {
        std::vector<Token> output{0xE6};
        output.insert(output.end(),empties,f->empty);
        output.insert(output.end(),{0x97,0xA5,'y'});
        require_full(f,{},output,"character split by tokens without text");
        require(f->widest <= offset_decoder_context+8,"tokens without text grew the window");
        output = {0xC3};
        output.insert(output.end(),empties,f->empty);
        output.insert(output.end(),{f->add("\xA4g"),'h','e','l'});
        require_full(f,{},output,"text after a split character was lost");
        require_full(f,{'a','b'},output,"text after a split character was lost after a prompt");
        std::vector<Token> context{'a'};
        context.insert(context.end(),empties,f->empty);
        require_full(f,context,{0xC3,0xA9},"context ending with tokens without text");
    }
    f->widest = 0;
    auto decoder = offset_decoder(f,std::vector<Token>{'a'});
    for(int i = 0; i < 200; ++i) require(decoder->push(std::vector<Token>{f->empty}).empty(),"empty token produced text");
    require(decoder->push(std::vector<Token>{'b'}) == "b" && f->widest <= offset_decoder_context+3,"zero-text run grew the window");
    // SentencePiece spacing survives runs without text, at any length.
    auto sp = sentencepiece();
    for(const std::size_t empties:{5u,15u,16u,17u,31u,32u,33u,64u,100u}) {
        const std::vector<Token> specials(empties,sp.f->empty);
        require(stream_all(sp.f,{sp.hello},joined(specials,{sp.world})) == " world","empty run after the context lost the next space");
        require(stream_all(sp.f,{},joined(joined({sp.hello},specials),{sp.world})) == "hello world","empty run lost the next space");
        require(stream_all(sp.f,{},joined(joined({' '},specials),{' ',sp.world})) == "  world","empty run at the start lost a space");
        require_full(sp.f,{sp.hello},joined(specials,{sp.world}),"spacing after an empty run differs from the full decode");
    }
    // A frontend that shows nothing for an incomplete character: its bytes still complete it.
    auto quiet = byte_frontend({});
    quiet->ignore_invalid = true;
    std::vector<Token> hidden{'a'};
    hidden.insert(hidden.end(),offset_decoder_hold+2,quiet->empty);
    hidden.insert(hidden.end(),{0xE6,0x97,0xA5,'x'});
    require(stream_all(quiet,{},hidden) == "a"+sun+"x","hidden incomplete character dropped");
    require(stream_all(quiet,{'a',0xE6,0x97},{0xA5,'x'}) == sun+"x","hidden incomplete context character dropped");
    // Tokens without text inside its characters: the bytes still complete them (2-, 3- and 4-byte characters, a context
    // that ends inside one, before or after tokens without text), and a stop over one ends at the token that completes it.
    for(const auto& character:{a_umlaut,sun,smile}) for(const std::size_t empties:{1u,2u,3u,4u,17u}) {
        std::vector<Token> output{'a'};
        for(std::size_t i = 0; i < character.size(); ++i) {
            output.push_back(Token(static_cast<unsigned char>(character[i])));
            if(i+1 < character.size()) output.insert(output.end(),empties,quiet->empty);
        }
        output.push_back('x');
        for(std::size_t cut = 0; cut+1 < output.size(); ++cut) {
            const std::vector<Token> context(output.begin(),output.begin()+std::ptrdiff_t(cut));
            const std::vector<Token> rest(output.begin()+std::ptrdiff_t(cut),output.end());
            require_full(quiet,context,rest,"hidden character split by tokens without text");
            require_stream(quiet,context,rest,{character},"stop over a hidden character split by tokens without text");
        }
    }
    // Distinct tokens without text: up to offset_decoder_hold of them inside a hidden character, and a long run of
    // them keeps the window short.
    std::vector<Token> specials;
    for(int i = 0; i < 40; ++i) { quiet->pieces.push_back({}); specials.push_back(Token(quiet->pieces.size()-1)); }
    std::vector<Token> distinct{'a',0xF0};
    distinct.insert(distinct.end(),specials.begin(),specials.begin()+6);
    distinct.push_back(0x9F);
    distinct.insert(distinct.end(),specials.begin()+6,specials.begin()+12);
    distinct.insert(distinct.end(),{0x98,specials[12],0x80,'y'});
    require_full(quiet,{},distinct,"distinct tokens without text dropped a hidden character");
    require_stream(quiet,{'H'},distinct,{smile},"stop over a character split by distinct tokens without text");
    quiet->widest = 0;
    decoder = offset_decoder(quiet,std::vector<Token>{'a'});
    for(int i = 0; i < 400; ++i) require(decoder->push(std::vector<Token>{specials[std::size_t(i)%specials.size()]}).empty(),"empty token produced text");
    require(decoder->push(std::vector<Token>{'b'}) == "b" && quiet->widest <= offset_decoder_context+offset_decoder_hold+2,
            "distinct tokens without text grew the window");
}
std::vector<Token> tokenize_random(ByteFrontend& f, const std::string& text, std::mt19937_64& rng) {
    std::vector<Token> tokens;
    for(std::size_t i = 0; i < text.size();) {
        const auto n = std::min(text.size()-i,std::uniform_int_distribution<std::size_t>(1,4)(rng));
        const auto found = f.merges.find(text.substr(i,n));
        if(n > 1 && found != f.merges.end()) tokens.push_back(found->second);
        else for(std::size_t k = 0; k < n; ++k) tokens.push_back(Token(static_cast<unsigned char>(text[i+k])));
        if(std::uniform_int_distribution<int>(0,9)(rng) == 0) tokens.push_back(f.empty);
        i += n;
    }
    return tokens;
}
// Long runs at character boundaries: tokens without text, or invalid bytes followed by a multibyte character.
void insert_runs(ByteFrontend& f, std::vector<Token>& tokens, std::size_t at, std::mt19937_64& rng) {
    const auto length = std::uniform_int_distribution<std::size_t>(offset_decoder_hold-3,2*offset_decoder_hold+8)(rng);
    std::vector<Token> run;
    if(std::uniform_int_distribution<int>(0,1)(rng)) run.assign(length,f.empty);
    else { run.assign(length,Token(0x80)); run.insert(run.end(),{0xE6,0x97,0xA5}); }
    tokens.insert(tokens.begin()+std::ptrdiff_t(at),run.begin(),run.end());
}
void offset_bytes() {
    const std::vector<std::string> words{"hello"," world"," caf\xC3\xA9"," cr\xC3\xA8me"," \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",
        " \xF0\x9F\x99\x82"," na\xC3\xAFve",",","!"," \xC3\x9C" "ber"," x"," \xE2\x82\xAC" "5"," "+repeat(lambda,4),
        " "+repeat(gamma_,3)," "+repeat(de,3)," "+repeat(smile,2)," "+a_umlaut+"g"+a_umlaut};
    auto f = byte_frontend(words);
    const std::vector<std::string> runs{lambda,gamma_,de,sun,smile,a_umlaut};
    std::mt19937_64 rng(7);
    OracleCounts counts;
    for(int trial = 0; trial < 1500; ++trial) {
        std::vector<Token> tokens;
        for(int i = 0, n = std::uniform_int_distribution<int>(1,10)(rng); i < n; ++i) {
            auto text = words[std::uniform_int_distribution<std::size_t>(0,words.size()-1)(rng)];
            // An occasional invalid byte: decoded as U+FFFD by both the stream and the full decode.
            if(trial % 7 == 0 && std::uniform_int_distribution<int>(0,2)(rng) == 0) text.insert(std::uniform_int_distribution<std::size_t>(0,text.size())(rng),1,'\xFF');
            if(trial % 11 == 0 && i == 1) insert_runs(*f,tokens,tokens.size(),rng);
            if(trial % 5 == 0 && i == 0) {
                // A run of merges that straddle characters, past offset_decoder_hold.
                const auto& character = runs[std::uniform_int_distribution<std::size_t>(0,runs.size()-1)(rng)];
                const auto run = straddled(*f,repeat(character,std::uniform_int_distribution<std::size_t>(3,40)(rng)),character.size());
                tokens.insert(tokens.end(),run.begin(),run.end());
            }
            const auto word = tokenize_random(*f,text,rng);
            tokens.insert(tokens.end(),word.begin(),word.end());
        }
        const auto cut = std::uniform_int_distribution<std::size_t>(0,tokens.size())(rng);
        const std::vector<Token> context(tokens.begin(),tokens.begin()+std::ptrdiff_t(cut)), output(tokens.begin()+std::ptrdiff_t(cut),tokens.end());
        check_stream(f,context,output,rng,counts);
        if(const auto bytes = byte_text(*f,context,output)) check_text_stream(f,context,output,pick_stops(bytes->text,rng),rng);
    }
    require(counts.prefix > 800 && counts.completed > 200,"random streams missed a context kind");
}
void offset_sentencepiece() {
    auto sp = sentencepiece();
    auto& f = sp.f;
    require(f->detokenize(std::vector<Token>{sp.hello,sp.world}) == "hello world","fixture strips more than the first space");
    require(f->detokenize(byte_ids(de+"\xCE")) == fffd(3) && f->detokenize(joined(byte_ids("\xE6\x97"),{f->empty,0xA5})) == sun,
            "fixture is not a byte-fallback decoder");
    // A prompt whose last offset_decoder_context tokens start inside a character: the output still decodes as in the
    // full decode, and a stop over its first character ends at the token that completes it.
    const auto love = joined({sp.hello},byte_ids(smile+smile));
    require_full(f,love,joined(byte_ids(smile),{sp.ok}),"window starting inside a character changed the output");
    require_stream(f,love,joined(byte_ids(smile),{sp.ok}),{smile},"stop after a window starting inside a character");
    require_stream(f,joined({sp.hello},byte_ids(sun+book)),byte_ids(sun+book),{book},"stop after a byte-fallback prompt");
    require_full(f,byte_ids(lambda+smile),byte_ids(smile+smile+smile),"byte-fallback prompt and output");
    // A prompt ending inside a byte-fallback character: the output completes it.
    require_full(f,joined({sp.hello},byte_ids("\xF0\x9F")),joined(byte_ids("\x98\x80"),{sp.ok}),"prompt ending inside a character");
    require_stream(f,joined({sp.hello},byte_ids("\xF0\x9F")),joined(byte_ids("\x98\x80"),{sp.ok}),{smile},"stop over a completed prompt character");
    // A prompt whose byte run ends in literal U+FFFD (bytes EF BF BD, which decode as U+FFFD like a start inside a
    // character): the window starts at a character boundary of the run, so the output's byte-fallback characters decode
    // as in the full decode and a stop over the first one ends at the token that completes it.
    const std::vector<std::tuple<std::vector<Token>,std::vector<Token>,std::string>> literal_runs{
        {joined({sp.hello},byte_ids(sun+fffd(2))),joined(byte_ids(smile),{sp.ok}),smile},
        {joined({f->merges.at(" d")},byte_ids(fffd(3))),byte_ids(sun),sun},
        {joined({sp.hello},byte_ids(fffd(4))),joined(byte_ids(sun+book),{sp.ok}),book},
        {joined({sp.hello},byte_ids(replacement+smile)),joined(byte_ids(a_umlaut),{sp.ok}),a_umlaut},
        {joined({sp.hello},byte_ids(sun+replacement+book)),byte_ids(clef+clef),clef}};
    for(const auto& [context,output,stop]:literal_runs) {
        require_full(f,context,output,"prompt ending in literal U+FFFD bytes changed the output");
        require_stream(f,context,output,{stop},"stop after a prompt ending in literal U+FFFD bytes");
    }
    // A prompt cut inside a literal U+FFFD whose byte run holds other characters: the run decodes as U+FFFD per byte
    // until the output's byte completes it. The output starts with that U+FFFD; the prompt's characters are not
    // released or matched by a stop, also when its U+FFFD per byte match literal U+FFFD of the output by chance.
    const auto split = joined({sp.hello},byte_ids(de+replacement+"\xEF\xBF"));
    require(f->detokenize(split) == "hello"+fffd(7),"fixture byte run is not invalid");
    require_full(f,split,joined(byte_ids("\xBD"),{sp.ok}),"prompt cut inside a literal U+FFFD");
    require_stream(f,split,joined(byte_ids("\xBD"),{sp.ok}),{de},"stop over prompt text");
    const auto chance = joined({sp.hello},byte_ids(replacement+"\xEF"));
    const auto literals = joined(byte_ids("\xBF\xBD"+fffd(2)),{sp.ok});
    require(f->detokenize(joined(chance,literals)).starts_with(f->detokenize(chance)),"fixture decodes do not match by chance");
    require(stream_all(f,chance,literals) == fffd(3)+" ok","U+FFFD per byte taken for prompt text");
    require_full(f,chance,literals,"U+FFFD per byte matching by chance differs from the full decode");
    // A byte that makes released text invalid as a whole: released text stays, and the tokens since the last release
    // decode on their own, one U+FFFD for the lone byte.
    require(stream_all(f,{},byte_ids(de+"\xCE")) == de+replacement,"lone byte after a released run");
    require(stream_all(f,{},byte_ids(de+de+de+"\xCE")) == de+de+de+replacement,"lone byte after a long released run");
    require(stream_all(f,{},joined(byte_ids(gamma_+gamma_+"\xCE"),{sp.ok})) == gamma_+gamma_+replacement+" ok","lone byte before a piece");
    // A character split by a long run of skipped special tokens completes, and a stop over it is found.
    const std::vector<Token> specials(2*offset_decoder_hold,f->empty);
    const auto mor = joined(joined({f->merges.at("d"),f->merges.at(" mor"),0xCE},specials),byte_ids("\xBB\xCE\xBB"));
    require_stream(f,{},mor,{"mor"+lambda},"stop over a character split by special tokens");
    const auto open = stream_text(f,{},joined(mor,{0xCE}),{"mor"+lambda});
    require(open.stopped && open.text == "d " && open.used == 3+specials.size()+1,"stop before an invalid byte run");
    // Random streams: word pieces, byte-fallback characters (runs of them), ASCII byte tokens, literal U+FFFD, skipped
    // special tokens and long runs of them, cut anywhere.
    std::mt19937_64 rng(11);
    const std::vector<std::string> multibyte{"\xC3\xA9",sun,book,smile,lambda,de,"\xF0\x9F\x99\x82",replacement};
    OracleCounts counts;
    for(int trial = 0; trial < 1500; ++trial) {
        std::vector<Token> tokens;
        for(int i = 0, n = std::uniform_int_distribution<int>(1,14)(rng); i < n; ++i) {
            const auto kind = std::uniform_int_distribution<int>(0,9)(rng);
            if(kind <= 2) {
                for(int k = 0, chars = std::uniform_int_distribution<int>(1,trial % 4 == 0 ? 12 : 3)(rng); k < chars; ++k) {
                    const auto bytes = byte_ids(multibyte[std::uniform_int_distribution<std::size_t>(0,multibyte.size()-1)(rng)]);
                    tokens.insert(tokens.end(),bytes.begin(),bytes.end());
                    if(std::uniform_int_distribution<int>(0,9)(rng) == 0) tokens.insert(tokens.end()-1,f->empty);
                }
            } else if(kind == 3) tokens.push_back(Token(std::uniform_int_distribution<int>('a','z')(rng)));
            else if(kind == 4) tokens.push_back(sp.replacement_piece);
            else if(kind == 5) tokens.insert(tokens.end(),trial % 9 == 0 ? std::uniform_int_distribution<std::size_t>(1,40)(rng) : std::size_t(1),f->empty);
            else tokens.push_back(sp.words[std::uniform_int_distribution<std::size_t>(0,sp.words.size()-1)(rng)]);
        }
        const auto cut = std::uniform_int_distribution<std::size_t>(0,tokens.size())(rng);
        const std::vector<Token> context(tokens.begin(),tokens.begin()+std::ptrdiff_t(cut)), output(tokens.begin()+std::ptrdiff_t(cut),tokens.end());
        // The prompt ends on a character boundary but its last offset_decoder_context tokens start inside one.
        const auto bytes = f->bytes(context);
        if(cut >= offset_decoder_context && lossy(bytes) == bytes) {
            const auto tail = f->bytes(std::span<const Token>(context).last(offset_decoder_context));
            if(!tail.empty() && continuation(tail.front())) ++counts.seeded;
        }
        check_stream(f,context,output,rng,counts);
        if(const auto text = byte_text(*f,context,output)) check_text_stream(f,context,output,pick_stops(text->text,rng),rng);
    }
    require(counts.prefix > 600 && counts.completed > 300 && counts.seeded > 100,"random streams missed a context kind");
}
// TextStream over offset_decoder: tokens_used names the token whose bytes complete the stop's last character, also
// when that token starts the next character, and U+FFFD of invalid bytes is released after offset_decoder_hold
// tokens without a clean boundary, all but the last 3.
void stream_offsets() {
    // One rule and one hold constant with crit_core.text (IncrementalDetokenizer HOLD_LIMIT and context_tokens).
    require(offset_decoder_hold == 16 && offset_decoder_context == 5,"offset decoder limits differ from crit_core.text");
    auto f = byte_frontend({});
    // Each straddled token completes one character and starts the next.
    require_stream(f,{},straddled(*f,repeat(smile,4),4),{smile},"stop completed by a straddling token");
    require_stream(f,{},straddled(*f,repeat(lambda,10),2),{lambda},"stop at the first of a straddled run");
    const std::vector<Token> greek{0xCF,f->add("\x83\xCE"),f->add("\xB1\xCE"),f->add("\xBB\xCE"),0xBB};
    require_stream(f,{'H','i'},greek,{lambda},"stop completed by a token that starts the next character");
    const auto stopped = stream_text(f,{'H','i'},greek,{lambda});
    require(stopped.stopped && stopped.used == 4 && stopped.text == "\xCF\x83\xCE\xB1","tokens_used named a later token");
    // Invalid bytes: the first U+FFFD is released at the token that brings the run to offset_decoder_hold.
    auto result = stream_text(f,{},joined(joined({'a'},std::vector<Token>(20,0x80)),{'b'}),{replacement});
    require(result.stopped && result.text == "a" && result.used == 1+offset_decoder_hold,"U+FFFD stop after a held run");
    result = stream_text(f,{},std::vector<Token>(40,0x80),{fffd(5)});
    require(result.stopped && result.text.empty() && result.used == offset_decoder_hold,"long U+FFFD stop");
    result = stream_text(f,{},std::vector<Token>(offset_decoder_hold-1,0x80),{replacement});
    require(result.stopped && result.text.empty() && result.used == offset_decoder_hold-1,"U+FFFD stop in the flushed tail");
    // A stop over held U+FFFD may still complete in the flushed tail.
    result = stream_text(f,{},{'a',0x80},{"a"+replacement});
    require(result.stopped && result.text.empty(),"stop in the flushed tail");
}
void utf8() {
    for(const auto& text:std::vector<std::string>{"","plain","caf\xC3\xA9","\xE6\x97\xA5","\xF0\x9F\x99\x82","\xEF\xBF\xBD","\xF4\x8F\xBF\xBF","\xED\x9F\xBF",
                                "\xE0\xA0\x80","\xF0\x90\x80\x80",std::string(1,'\0')})
        require(valid_utf8(text),"valid UTF-8 rejected");
    for(const auto& text:std::vector<std::string>{"\x80","\xBF","\xC0\x80","\xC1\xBF","\xC3","\xC3x","\xE6\x97","\xE0\x9F\xBF","\xED\xA0\x80","\xF0\x8F\xBF\xBF",
                                "\xF4\x90\x80\x80","\xF5\x80\x80\x80","\xFF","a\xE6\x97\xA5\x97"})
        require(!valid_utf8(text),"invalid UTF-8 accepted");
    // A truncated sequence is invalid even when the bytes after the view would complete it.
    const std::string complete = "x\xC3\xA9\xE6\x97\xA5\xF0\x9F\x99\x82";
    for(const std::size_t end:{2,4,5,7,8,9}) require(!valid_utf8(std::string_view(complete).substr(0,end)),"truncated view accepted");
    require(valid_utf8(complete),"complete characters rejected");
    // A stop that is not valid UTF-8 could match inside a character and cut it.
    rejects([]{StopMatcher m({"ok","\x97"});},"not valid utf-8");
    rejects([]{StopMatcher m({"\xE6\x97"});},"not valid utf-8");
    rejects([]{TextStream s(offset_decoder(byte_frontend({})),{"\xC3"});},"not valid utf-8");
}
}
int main(int argc, char** argv) {
    try {
        const std::string name = argc > 1 ? argv[1] : "all";
        const std::vector<std::pair<std::string,std::function<void()>>> cases{
            {"text_stop_rules",stop_rules},{"text_stop_oracle",stop_oracle},{"text_stop_linear",stop_linear},
            {"text_stream",text_stream},{"text_offset_rules",offset_rules},{"text_offset_bytes",offset_bytes},
            {"text_offset_sentencepiece",offset_sentencepiece},{"text_offset_bounds",offset_bounds},
            {"text_stream_offsets",stream_offsets},{"text_utf8",utf8}};
        bool found = false;
        for(const auto& [case_name,run]:cases) if(name == "all" || name == case_name) {
            run(); found = true;
            std::cout << "crit core text passed: " << case_name << '\n';
        }
        require(found,"unknown test");
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
