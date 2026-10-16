#pragma once
#include <crit/core.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crit::core {
// Most stop strings per matcher or request, and most bytes per stop string.
inline constexpr std::size_t stop_strings_limit = 16;
inline constexpr std::size_t stop_string_bytes_limit = 256;
// offset_decoder seeds its window with this many trailing context tokens (moved back by up to 3 to a token that
// starts a character) and keeps at least this many as context once text is released. A run of offset_decoder_hold
// tokens without a clean character boundary cuts the window back to its last tokens (bounded memory for streams
// that never resolve). Both match crit_core.text's IncrementalDetokenizer (context_tokens=5, HOLD_LIMIT).
inline constexpr std::size_t offset_decoder_context = 5;
inline constexpr std::size_t offset_decoder_hold = 16;

// Well-formed UTF-8: shortest forms only, no surrogates, nothing above U+10FFFF.
bool valid_utf8(std::string_view text) noexcept;

// Byte-exact stop strings over complete text. The first stop to complete wins (ties: earliest start, then lowest
// index); only the longest suffix that is a proper prefix of some stop is held back. Results do not depend on how
// text is chunked, and work is linear in its length. Closed after a stop or finish(); then logic_error.
class StopMatcher {
public:
    // No stops passes text through. invalid_argument for an empty stop, more than stop_strings_limit stops, a stop
    // over stop_string_bytes_limit bytes or one that is not valid UTF-8 (a valid stop only matches whole characters).
    explicit StopMatcher(std::vector<std::string> stops);
    // released: text that can no longer belong to a stop; on a stop it ends where the stop begins.
    struct Step { std::string released; bool stopped = false; std::size_t stop_index = 0; };
    Step push(std::string_view appended);
    // Releases the held text when no stop matched.
    std::string finish();
private:
    std::string bytes_;                   // stops, concatenated
    std::vector<std::uint32_t> starts_;   // stop i is bytes_[starts_[i], starts_[i+1])
    std::vector<std::uint8_t> borders_;   // per stop byte: longest proper border of the prefix ending there
    std::vector<std::uint16_t> matched_;  // per stop: longest suffix of the text that is a prefix of it
    std::string held_;
    bool closed_ = false;
};

// One reply's text: accepted tokens go through the decoder one at a time and its text through a StopMatcher.
// Closed after a stop, finish() or any exception; then logic_error("text stream is closed").
class TextStream {
public:
    // invalid_argument for a null decoder or invalid stops (see StopMatcher).
    TextStream(std::unique_ptr<TextDecoder>, std::vector<std::string> stops);
    struct Piece { std::string text; bool stopped = false; std::uint32_t tokens_used = 0; };
    // On a stop, tokens_used is the index of the completing token + 1 and later tokens are not decoded;
    // otherwise it is the span size.
    Piece push(std::span<const Token>);
    // Flushes the decoder and the held text; a stop may still complete in the flushed tail (tokens_used 0).
    Piece finish();
private:
    std::unique_ptr<TextDecoder> decoder_;
    StopMatcher matcher_;
    bool closed_ = false;
};

// Incremental decoder over a frontend's detokenize(), the algorithm of crit_core.text's IncrementalDetokenizer: each
// token extends a short window and releases what its decode adds beyond the text already released, except a trailing
// run of U+FFFD (the bytes EF BF BD), which may be a character whose remaining bytes have not arrived; that run is held
// until later text follows it or finish() flushes it. The window starts with the last offset_decoder_context tokens of
// `context`, moved back to a token that starts a character (or, when none does, to the start whose decode holds the
// fewest U+FFFD). Context text is never released: when the context ends inside a character, the output starts with the
// character it completes, and U+FFFD that the output leaves in place stays context text, including a literal U+FFFD
// whose first bytes end the context when the full decode keeps the U+FFFD the context's decode shows there (the decode
// cannot tell it from an incomplete character left in place). For the frontend's own encodings, the released text
// equals detokenize(context + output) without the context's complete characters. Tokens whose decode is empty even when
// doubled and that leave the window's decode unchanged (such as skipped special tokens) wait at the window's end
// uncounted; a later token that changes the decode drops each one the decode does not need and counts the others (a
// frontend that hides incomplete characters). Such a run keeps at most 3 copies of a token and offset_decoder_hold
// tokens. After offset_decoder_hold tokens without a clean boundary the window is cut to its last tokens, keeping those
// that may carry an incomplete character, and a long U+FFFD run is released except its last 3 characters. Released text
// is never retracted: when a later token changes text already released (a byte run that becomes invalid as a whole),
// the tokens since the last release are decoded on their own, without repeating what was released from them. Closed
// after finish() or an exception; then logic_error("text decoder is closed"). invalid_argument for a null frontend.
std::unique_ptr<TextDecoder> offset_decoder(std::shared_ptr<const TextFrontend>, std::span<const Token> context = {});
}
