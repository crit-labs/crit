#include "crit/text.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace crit::core {
namespace {
constexpr std::string_view replacement = "\xEF\xBF\xBD";
// Ids that can carry the bytes of one incomplete UTF-8 character (at most 3 bytes).
constexpr std::size_t max_partial = 3;
// Ids the window keeps as context once text is released, at least.
constexpr std::size_t keep_context = std::max<std::size_t>(offset_decoder_context,2);
bool continuation(char byte) { return (static_cast<unsigned char>(byte) & 0xC0) == 0x80; }
// U+FFFD characters in the text.
std::size_t replacements(std::string_view text) {
    std::size_t count = 0;
    for(auto at = text.find(replacement); at != std::string_view::npos; at = text.find(replacement,at+replacement.size())) ++count;
    return count;
}
// The text without its trailing run of U+FFFD.
std::string_view without_replacements(std::string_view text) {
    while(text.ends_with(replacement)) text.remove_suffix(replacement.size());
    return text;
}
// Characters in UTF-8 text (bytes that do not continue a character).
std::size_t characters(std::string_view text) {
    return std::size_t(std::count_if(text.begin(),text.end(),[](char byte){return !continuation(byte);}));
}
// Bytes of the first `count` characters of the text.
std::size_t character_bytes(std::string_view text, std::size_t count) {
    std::size_t at = 0;
    for(; at < text.size(); ++at) if(!continuation(text[at]) && count-- == 0) break;
    return at;
}
// Bytes of the longest common prefix of two texts that ends on a character boundary.
std::size_t shared(std::string_view first, std::string_view second) {
    const auto limit = std::min(first.size(),second.size());
    std::size_t at = 0;
    while(at < limit && first[at] == second[at]) ++at;
    while(at && ((at < first.size() && continuation(first[at])) || (at < second.size() && continuation(second[at])))) --at;
    return at;
}
// The algorithm of crit_core.text's IncrementalDetokenizer (detokenize.py) over a TextFrontend, with the same rules
// and limits; character counts and slices there are code points, here the bytes of whole characters. TextFrontend has
// no special-token list, so the ids Python skips are recognized by their decode. An id whose decode is empty even when
// doubled and that leaves the window's decode unchanged is unsettled: it may have no text (a skipped special token), or
// carry bytes of a character that the frontend shows nothing for while it is incomplete. Unsettled ids wait at the end
// of the window, uncounted, until an id that changes the decode arrives. Then, oldest first, each one the decode stays
// the same without is dropped (it has no text, like an id Python skips), and the others are counted, in order, as
// Python counts them. A run keeps at most max_partial copies of an id (no character holds more) and offset_decoder_hold
// ids: beyond that its oldest id is dropped when the decode stays the same without it and counted otherwise, so with a
// frontend that hides incomplete characters a character whose bytes and the ids without text inside it exceed
// offset_decoder_hold ids can be lost (Python skips special ids by id and keeps it).
class OffsetDecoder final : public TextDecoder {
    std::shared_ptr<const TextFrontend> frontend_;
    std::vector<Token> window_;             // the counted ids, then the unsettled run; all counts below are counted ids
    std::size_t unsettled_ = 0;             // ids at the end of window_ that may have no text
    std::size_t read_ = 0;                  // counted ids [0, read_) are context, the rest arrived since the last anchor
    std::vector<std::size_t> clean_;        // window lengths where the decode ended on a clean boundary
    std::string head_;                      // leading text of the window decode already released or prompt text
    std::string base_;                      // head_ when the ids since the last anchor began
    std::string prompt_;                    // the prompt decode while its final U+FFFD run is open
    bool open_ = false;
    // Decodes of the prompt without its last 1 to max_partial ids, with that count, longest first.
    std::vector<std::pair<std::string,std::size_t>> prompt_heads_;
    std::string last_;                      // decode of window_
    std::size_t outputs_ = 0;
    bool closed_ = false;

    std::size_t counted() const { return window_.size()-unsettled_; }
    // Decode of counted ids [from, to); the unsettled run belongs to the last counted id.
    std::string decode(std::size_t from, std::size_t to) const {
        if(from >= to) return {};
        const auto last = to == counted() ? window_.size() : to;
        return frontend_->detokenize(std::span<const Token>(window_).subspan(from,last-from));
    }
    std::string decode_all() const { return window_.empty() ? std::string{} : frontend_->detokenize(window_); }
    // Decode of window_ without the id at `index`.
    std::string decode_without(std::size_t index) const {
        std::vector<Token> ids(window_);
        ids.erase(ids.begin()+std::ptrdiff_t(index));
        return ids.empty() ? std::string{} : frontend_->detokenize(ids);
    }
    bool without_text(Token token) const {
        const std::array<Token,2> twice{token,token};
        return frontend_->detokenize(twice).empty();
    }
    // Drops counted ids [0, count).
    void cut(std::size_t count) { window_.erase(window_.begin(),window_.begin()+std::ptrdiff_t(count)); }
    // The unsettled run before the id just added, whose decode is `full`: drops, oldest first, each id the decode stays
    // `full` without, and returns how many it keeps (they have text and are counted).
    std::size_t settle(const std::string& full) {
        auto at = window_.size()-1-unsettled_;
        std::size_t kept = 0;
        for(std::size_t i = 0; i < unsettled_; ++i) {
            if(decode_without(at) == full) window_.erase(window_.begin()+std::ptrdiff_t(at));
            else { ++at; ++kept; }
        }
        unsettled_ = 0;
        return kept;
    }
    // The prompt's last offset_decoder_context ids, moved back by up to max_partial ids to one that starts a character
    // (the decode does not start with U+FFFD once up to max_partial final ids are set aside). When none does (byte-level
    // merges that straddle characters, a byte run that ends in a literal U+FFFD), the start whose decode holds the
    // fewest U+FFFD plus ids set aside, the latest of equals: a start inside a byte-fallback character turns the run
    // into one U+FFFD per byte.
    void seed(std::span<const Token> context) {
        std::unordered_map<Token,bool> known;
        const auto lacks_text = [&](Token id) {
            const auto [entry,added] = known.try_emplace(id,false);
            if(added) entry->second = without_text(id);
            return entry->second;
        };
        // The last offset_decoder_context + max_partial ids with text and the ids without it around them (each run
        // capped like the output's, the newest kept), and whether the prompt holds more.
        const auto wanted = offset_decoder_context+max_partial;
        std::vector<Token> tail;
        std::size_t texts = 0, run = 0;
        bool more = false;
        for(auto at = context.size(); at--;) {
            const auto token = context[at];
            if(lacks_text(token)) {
                if(run < offset_decoder_hold && std::count(tail.end()-std::ptrdiff_t(run),tail.end(),token) < std::ptrdiff_t(max_partial)) {
                    tail.push_back(token);
                    ++run;
                }
                continue;
            }
            if(texts == wanted) { more = true; break; }
            tail.push_back(token);
            ++texts; run = 0;
        }
        std::reverse(tail.begin(),tail.end());
        // Settle the tail as the output's ids are settled; the final run stays unsettled.
        std::optional<std::string> decoded;    // decode of window_, when known
        for(const auto token:tail) {
            if(lacks_text(token)) {
                if(!decoded) decoded = decode_all();
                window_.push_back(token);
                auto text = decode_all();
                if(text == *decoded) { ++unsettled_; continue; }
                decoded = std::move(text);
            } else {
                window_.push_back(token);
                decoded.reset();
            }
            if(unsettled_) {
                settle(decoded ? *decoded : decode_all());
                decoded.reset();
            }
        }
        const auto end = counted();
        if(!end) return;
        const auto plain = end > offset_decoder_context ? end-offset_decoder_context : 0;
        const auto lowest = plain > max_partial ? plain-max_partial : 0;
        auto best = plain;
        auto best_score = std::numeric_limits<std::size_t>::max();
        for(auto start = plain;; --start) {
            if(start == 0 && !more) return;
            for(std::size_t partial = 0; partial <= std::min(max_partial,end-start-1); ++partial) {
                const auto text = decode(start,end-partial);
                if(text.empty()) continue;
                if(!text.starts_with(replacement)) { cut(start); return; }
                if(const auto score = replacements(text)+partial; score < best_score) { best_score = score; best = start; }
            }
            if(start == lowest) break;
        }
        cut(best);
    }
    // Once the output shows how the prompt's final U+FFFD run continues (text follows it, or the decode ends cleanly),
    // or when `force` is set, account for the longest prompt text the window decode starts with: the part of the prompt
    // decode it keeps (U+FFFD left in place stays prompt text), or a longer decode of the prompt's complete characters.
    // A decode of the prompt without its last ids is exact when the rest of the window decodes on its own to the rest
    // of `full`; one that ends with U+FFFD (a literal U+FFFD before an incomplete character) counts only when exact.
    // The prompt decode's final U+FFFD run is a byte run turned into U+FFFD as a whole by the prompt's incomplete
    // character when an exact decode shows that the prompt decode is not that decode plus the decode of the ids after
    // it, or, when the window decode does not keep all of the prompt decode, when those ids decode to one U+FFFD each,
    // alone and together (one per byte). Then the U+FFFD the kept part ends with are not prompt text, except those an
    // exact decode shows. Otherwise they are complete characters of the prompt (a literal U+FFFD or invalid bytes before
    // its incomplete character), also when no exact decode shows them (a merge that straddles a literal U+FFFD and the
    // next character).
    bool resolve(const std::string& full, bool force) {
        if(!force && characters(without_replacements(full)) <= characters(head_) && full.ends_with(replacement)) return false;
        const auto exact = [&](const std::string& text, std::size_t partial) {
            return full.starts_with(text) && text+decode(read_-partial,counted()) == full;
        };
        auto keep = std::string_view(prompt_).substr(0,shared(full,prompt_));
        const bool whole = keep.size() == prompt_.size();
        const auto per_byte = [&](const std::string& tail, std::size_t partial) {
            if(tail.size() != partial*replacement.size() || replacements(tail) != partial) return false;
            for(auto at = read_-partial; at < read_; ++at)
                if(frontend_->detokenize(std::span<const Token>(window_).subspan(at,1)) != replacement) return false;
            return true;
        };
        bool turned = false;
        for(const auto& [text,partial]:prompt_heads_) {
            if(!exact(text,partial)) continue;
            const auto tail = decode(read_-partial,read_);
            if(text+tail != prompt_ || (!whole && per_byte(tail,partial))) { turned = true; break; }
        }
        if(turned) keep = without_replacements(keep);
        std::string head = characters(keep) >= characters(head_) ? std::string(keep) : head_;
        for(const auto& [text,partial]:prompt_heads_)
            if(characters(text) > characters(head) && full.starts_with(text) && (!text.ends_with(replacement) || exact(text,partial))) {
                head = text;
                break;
            }
        head_ = base_ = std::move(head);
        open_ = false;
        prompt_.clear(); prompt_heads_.clear();
        return true;
    }
    // The window's last id is counted (no run is unsettled) and its decode is `full`: what it releases.
    std::string count(const std::string& full) {
        ++outputs_;
        last_ = full;
        return step(full);
    }
    std::string step(const std::string& full) {
        const auto pending = counted()-read_;
        if(open_ && !resolve(full,pending >= offset_decoder_hold)) return {};
        if(full.starts_with(head_)) {
            const auto tail = std::string_view(full).substr(head_.size());
            const auto stable = without_replacements(tail);
            head_ += stable;
            if(stable.size() == tail.size()) { anchor(); return std::string(stable); }
            if(pending >= offset_decoder_hold) return std::string(stable)+bound(full);
            return std::string(stable);
        }
        if(full.ends_with(replacement) && pending < offset_decoder_hold) return {}; // a byte run is still open
        return diverge(full);
    }
    // The unsettled run just grew: keep at most max_partial copies of an id and offset_decoder_hold ids. The oldest
    // goes when the decode stays the same without it; an oldest id the decode needs has text after all and is counted.
    void trim(std::string& out) {
        const auto first = window_.size()-unsettled_;
        const auto run = window_.begin()+std::ptrdiff_t(first);
        if(std::count(run,window_.end(),window_.back()) > std::ptrdiff_t(max_partial)) {
            const auto copy = std::size_t(std::find(run,window_.end(),window_.back())-window_.begin());
            if(decode_without(copy) == last_) { window_.erase(window_.begin()+std::ptrdiff_t(copy)); --unsettled_; return; }
        }
        if(unsettled_ <= offset_decoder_hold) return;
        if(decode_without(first) == last_) { window_.erase(run); --unsettled_; return; }
        const std::vector<Token> rest(run+1,window_.end());
        window_.resize(first+1);
        unsettled_ = 0;
        const auto text = decode_all();
        out += count(text);
        window_.insert(window_.end(),rest.begin(),rest.end());
        unsettled_ = rest.size();
        last_ = decode_all();
    }
    // The decode ends cleanly and is fully accounted for: the ids become context, cut back to the latest clean
    // boundary that leaves at least keep_context ids.
    void anchor() {
        const auto length = counted();
        read_ = length;
        base_ = head_;
        clean_.push_back(length);
        std::size_t boundary = 0;
        for(const auto offset:clean_) {
            if(offset+keep_context > length) break;
            boundary = offset;
        }
        if(boundary < keep_context) return;
        cut(boundary);
        std::erase_if(clean_,[&](std::size_t offset){return offset <= boundary;});
        for(auto& offset:clean_) offset -= boundary;
        read_ -= boundary;
        head_ = base_ = last_ = decode_all();
    }
    // offset_decoder_hold ids without a clean boundary: release the trailing U+FFFD run except its last max_partial
    // characters, which may still be one incomplete character, and cut the window to its last ids, keeping the ids
    // that may carry that character.
    std::string bound(const std::string& full) {
        auto held = full.substr(head_.size());
        std::string released;
        if(const auto characters_held = characters(held); characters_held > max_partial) {
            released = held.substr(0,character_bytes(held,characters_held-max_partial));
            held.erase(0,released.size());
            head_ += released;
        }
        const auto length = counted();
        bool found = false;
        for(auto keep = keep_context+max_partial; keep < length && !found; ++keep) {
            auto text = decode(length-keep,length);
            if(!text.ends_with(held)) continue;
            cut(length-keep);
            head_ = text.substr(0,text.size()-held.size());
            last_ = std::move(text);
            found = true;
        }
        if(!found && length > keep_context+max_partial) { // no shorter window keeps the held run: release it
            released += held;
            cut(length-keep_context);
            head_ = last_ = decode_all();
        }
        clean_.clear();
        read_ = counted();
        base_ = head_;
        return released;
    }
    // Text already released or accounted for changed: what the ids since the last anchor decode to on their own,
    // without the text already released from them.
    std::string alone() const {
        const auto text = decode(read_,counted());
        const auto released = base_.size() <= head_.size() ? std::string_view(head_).substr(base_.size()) : std::string_view{};
        return text.substr(shared(text,released));
    }
    std::string diverge(const std::string& full) {
        auto text = alone();
        head_ = full;
        anchor();
        return text;
    }
public:
    OffsetDecoder(std::shared_ptr<const TextFrontend> frontend, std::span<const Token> context):frontend_(std::move(frontend)) {
        if(!frontend_) throw std::invalid_argument("offset decoder requires a text frontend");
        seed(context);
        read_ = counted();
        last_ = decode_all();
        head_ = base_ = std::string(without_replacements(last_));
        open_ = head_.size() != last_.size();
        if(!open_) return;
        prompt_ = last_;
        const auto length = counted();
        for(std::size_t partial = 1; partial <= std::min(max_partial,length); ++partial) prompt_heads_.emplace_back(decode(0,length-partial),partial);
        std::sort(prompt_heads_.begin(),prompt_heads_.end(),[](const auto& a, const auto& b) {
            const auto ca = characters(a.first), cb = characters(b.first);
            return ca != cb ? ca > cb : a < b;
        });
    }
    std::string push(std::span<const Token> tokens) override {
        if(closed_) throw std::logic_error("text decoder is closed");
        try {
            std::string out;
            for(const auto token:tokens) {
                window_.push_back(token);
                auto full = decode_all();
                if(full == last_ && without_text(token)) {
                    ++unsettled_;
                    trim(out);
                    continue;
                }
                if(unsettled_) {
                    // Count the run's ids that have text, in order, before this one.
                    if(const auto kept = settle(full)) {
                        const std::vector<Token> ids(window_.end()-std::ptrdiff_t(kept+1),window_.end()-1);
                        window_.resize(window_.size()-kept-1);
                        for(const auto id:ids) {
                            window_.push_back(id);
                            const auto text = decode_all();
                            out += count(text);
                        }
                        window_.push_back(token);
                        full = decode_all();
                    }
                }
                out += count(full);
            }
            return out;
        } catch(...) { closed_ = true; throw; }
    }
    std::string finish() override {
        if(closed_) throw std::logic_error("text decoder is closed");
        closed_ = true;
        if(!outputs_) return {};
        const auto full = decode_all();
        if(open_) resolve(full,true);
        if(full.starts_with(head_)) return full.substr(head_.size());
        return alone();
    }
};
}

bool valid_utf8(std::string_view text) noexcept {
    for(std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        if(lead < 0x80) { ++i; continue; }
        std::size_t need = 0;
        unsigned char low = 0x80, high = 0xBF;
        if(lead >= 0xC2 && lead <= 0xDF) need = 1;
        else if(lead >= 0xE0 && lead <= 0xEF) { need = 2; if(lead == 0xE0) low = 0xA0; else if(lead == 0xED) high = 0x9F; }
        else if(lead >= 0xF0 && lead <= 0xF4) { need = 3; if(lead == 0xF0) low = 0x90; else if(lead == 0xF4) high = 0x8F; }
        else return false;
        if(text.size()-i <= need) return false;
        for(std::size_t j = 1; j <= need; ++j) {
            const auto byte = static_cast<unsigned char>(text[i+j]);
            if(byte < low || byte > high) return false;
            low = 0x80; high = 0xBF;
        }
        i += need+1;
    }
    return true;
}

StopMatcher::StopMatcher(std::vector<std::string> stops) {
    if(stops.size() > stop_strings_limit) throw std::invalid_argument("stop strings exceed 16 entries");
    std::size_t total = 0;
    for(const auto& stop:stops) {
        if(stop.empty()) throw std::invalid_argument("stop strings must be nonempty");
        if(stop.size() > stop_string_bytes_limit) throw std::invalid_argument("stop string exceeds 256 bytes");
        if(!valid_utf8(stop)) throw std::invalid_argument("stop string is not valid utf-8");
        total += stop.size();
    }
    bytes_.reserve(total); borders_.reserve(total);
    starts_.reserve(stops.size()+1); matched_.assign(stops.size(),0);
    for(const auto& stop:stops) {
        starts_.push_back(std::uint32_t(bytes_.size()));
        bytes_ += stop;
        // Knuth-Morris-Pratt borders: borders_[start+i] is the longest proper border of stop[0, i].
        const auto start = borders_.size();
        borders_.push_back(0);
        for(std::size_t i = 1, k = 0; i < stop.size(); ++i) {
            while(k && stop[i] != stop[k]) k = borders_[start+k-1];
            if(stop[i] == stop[k]) ++k;
            borders_.push_back(std::uint8_t(k));
        }
    }
    starts_.push_back(std::uint32_t(bytes_.size()));
}
StopMatcher::Step StopMatcher::push(std::string_view appended) {
    if(closed_) throw std::logic_error("stop matcher is closed");
    const auto stops = matched_.size();
    for(std::size_t at = 0; at < appended.size(); ++at) {
        const char byte = appended[at];
        std::size_t best = stops;
        for(std::size_t i = 0; i < stops; ++i) {
            const auto start = starts_[i], length = starts_[i+1]-start;
            std::size_t k = matched_[i];
            while(k && bytes_[start+k] != byte) k = borders_[start+k-1];
            if(bytes_[start+k] == byte) ++k;
            matched_[i] = std::uint16_t(k);
            // Completions at the same byte: the longest starts earliest; equal lengths keep the lowest index.
            if(k == length && (best == stops || length > starts_[best+1]-starts_[best])) best = i;
        }
        if(best != stops) {
            const auto end = held_.size()+at+1, length = std::size_t(starts_[best+1]-starts_[best]);
            Step step{held_,true,best};
            step.released.append(appended.substr(0,at+1));
            step.released.resize(end-length);
            closed_ = true; held_.clear();
            return step;
        }
    }
    std::size_t hold = 0;
    for(const auto k:matched_) hold = std::max<std::size_t>(hold,k);
    const auto total = held_.size()+appended.size();
    Step step;
    if(hold >= appended.size()) {
        // The new holdback starts inside the previous one.
        const auto keep = hold-appended.size(), release = held_.size()-keep;
        step.released = held_.substr(0,release);
        held_.erase(0,release);
        held_.append(appended);
    } else {
        step.released.reserve(total-hold);
        step.released.append(held_).append(appended.substr(0,appended.size()-hold));
        held_.assign(appended.substr(appended.size()-hold));
    }
    return step;
}
std::string StopMatcher::finish() {
    if(closed_) throw std::logic_error("stop matcher is closed");
    closed_ = true;
    return std::exchange(held_,{});
}

TextStream::TextStream(std::unique_ptr<TextDecoder> decoder, std::vector<std::string> stops)
    :decoder_(std::move(decoder)),matcher_(std::move(stops)) {
    if(!decoder_) throw std::invalid_argument("text stream requires a decoder");
}
TextStream::Piece TextStream::push(std::span<const Token> tokens) {
    if(closed_) throw std::logic_error("text stream is closed");
    if(tokens.size() > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("text stream push exceeds 2^32 tokens");
    try {
        Piece piece;
        for(std::size_t i = 0; i < tokens.size(); ++i) {
            auto step = matcher_.push(decoder_->push(tokens.subspan(i,1)));
            piece.text += step.released;
            if(step.stopped) {
                closed_ = true; piece.stopped = true; piece.tokens_used = std::uint32_t(i+1);
                return piece;
            }
        }
        piece.tokens_used = std::uint32_t(tokens.size());
        return piece;
    } catch(...) { closed_ = true; throw; }
}
TextStream::Piece TextStream::finish() {
    if(closed_) throw std::logic_error("text stream is closed");
    closed_ = true;
    auto step = matcher_.push(decoder_->finish());
    Piece piece{std::move(step.released),step.stopped,0};
    if(!step.stopped) piece.text += matcher_.finish();
    return piece;
}
std::unique_ptr<TextDecoder> offset_decoder(std::shared_ptr<const TextFrontend> frontend, std::span<const Token> context) {
    return std::make_unique<OffsetDecoder>(std::move(frontend),context);
}
}
