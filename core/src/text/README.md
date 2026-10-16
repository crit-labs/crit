# Text streams

[text.cpp](text.cpp) implements [crit/text.h](../../include/crit/text.h):

| Name | Purpose |
| --- | --- |
| `StopMatcher` | Byte-exact stop strings over text that arrives in chunks |
| `TextStream` | One reply's decoded text with its stop strings applied |
| `offset_decoder` | An incremental `TextDecoder` built on a frontend's `detokenize()` |
| `valid_utf8` | The well-formedness check stop strings must pass |

The [request runtime](../runtime/README.md#request-text-and-stop-strings) uses
them for `RequestOptions::text` and `RequestOptions::stop_strings`. They depend
only on [core.h](../../include/crit/core.h) and the standard library and also work
on their own.

## StopMatcher

```cpp
crit::core::StopMatcher matcher({"</answer>", "\n\n"});
auto step = matcher.push(piece);        // step.released can no longer belong to a stop
if(step.stopped) { /* step.stop_index names the stop; released ends where it begins */ }
std::string tail = matcher.finish();    // the held text, when no stop matched
```

The first stop to complete wins; at the same byte the earliest start (the longest
stop) wins, then the lowest index. Only the longest suffix that is a proper prefix
of some stop is held back, so results do not depend on chunking, even chunks that
split UTF-8 characters. Matching compares bytes without normalization. Every stop
must be well-formed UTF-8, so a match never leaves a cut character in released
text. Work is linear in the text length times the number of stops (one
Knuth-Morris-Pratt state per stop). An empty stop list passes text through. After
a stop or `finish()` the matcher is closed.

Limits: `stop_strings_limit` (16 stops) and `stop_string_bytes_limit` (256 bytes
per stop). Invalid stops throw `std::invalid_argument`; use after closing throws
`std::logic_error`. `valid_utf8(text)` accepts shortest-form UTF-8 only (U+0000
is valid).

## TextStream

```cpp
crit::core::TextStream stream(std::move(decoder), {"STOP"});
auto piece = stream.push(tokens);       // piece.text: newly released text
if(piece.stopped) { /* tokens[0, piece.tokens_used) are kept */ }
else piece = stream.finish();           // flushes the decoder tail and held text
```

`push()` feeds tokens to the decoder one at a time and its text through a
`StopMatcher`. On a stop, `tokens_used` is the index of the completing token plus
one and later tokens are not decoded; otherwise it is the span size. Over
`offset_decoder`, the completing token is the one whose bytes complete the stop's
last character. `finish()` flushes the decoder, then releases held text; a stop
may still complete in that tail (`stopped` with `tokens_used` zero). The stream
closes after a stop, `finish()` or any exception. A null decoder, invalid stops
or a push of more than 2^32 tokens throw `std::invalid_argument`; use after
closing throws `std::logic_error("text stream is closed")`.

## offset_decoder

```cpp
auto decoder = crit::core::offset_decoder(frontend, prompt);   // context is optional
std::string text = decoder->push(tokens);
text += decoder->finish();
```

`offset_decoder` turns a frontend's `detokenize()` into a streaming decoder with
prefix and read offsets over a short window, the scheme common serving stacks use
for tokenizers without incremental decoding. It runs the algorithm of the Python
`IncrementalDetokenizer` ([crit_core.text](../../python/crit_core/text/README.md#streaming-text))
with the same rules and limits, so both release the same text at the same tokens.
It assumes the frontend decodes an incomplete UTF-8 sequence as U+FFFD, as
tokenizers usually do.

- Release: each token extends the window, and the decoder releases what the
  window's decode adds beyond released text, except a trailing run of U+FFFD that
  may be an incomplete character. That run is held until text follows or
  `finish()`. A character is released by the token whose bytes complete it.
- Context: the window is seeded with the last `offset_decoder_context` (5) tokens
  of `context`, moved back by up to 3 tokens to a character start. The context
  provides spacing (a SentencePiece leading space) and is never released; when it
  ends inside a character, the output starts with the character it completes.
- Tokens without text (a skipped special token, or a byte a frontend hides while
  incomplete) wait at the window's end and do not count toward the context, the
  window or the hold limit, like the ids the Python mirror skips.
- Result: for the frontend's own encodings, the concatenated results equal
  `detokenize(context + output)` without the context's complete characters,
  apart from the limits below.
- Bound: after `offset_decoder_hold` (16) tokens without a clean character
  boundary the window is cut, keeping at most 3 tokens that may carry an
  incomplete character, and a held U+FFFD run is released except its last 3
  characters. Every `detokenize()` call sees a short window.
- Released text is never retracted. When a later token changes released text (a
  SentencePiece byte run turned into U+FFFD, or a frontend that rewrites text),
  the tokens since the last clean boundary are decoded on their own without
  repeating released text.

Limits:

- A frontend that emits U+FFFD itself delays that text until later text or
  `finish()`, at most 16 tokens.
- When generation ends inside a SentencePiece byte run that began before released
  text, the tail is flushed as one U+FFFD per byte, while the full decode turns
  the whole run into U+FFFD.
- `TextFrontend` has no list of special tokens, so tokens without text are
  recognized by their decode alone. With a frontend that shows nothing for an
  incomplete character, a character spanning more than `offset_decoder_hold`
  tokens can be lost. A frontend with byte fallback that also hides incomplete
  characters is not supported.

The decoder retains its frontend and is used serially; the frontend must allow
concurrent calls (see [text services](../runtime/README.md#text-services)). A null
frontend throws `std::invalid_argument`; frontend exceptions propagate and close
the decoder, after which use throws `std::logic_error("text decoder is closed")`.

## Tests

The [text fixture](../../../tests/core/text_tests.cpp) checks `StopMatcher`
against a brute-force oracle under random chunking, and `offset_decoder` with
`TextStream` against oracles built only from each frontend's full decode, for
byte-level and SentencePiece byte-fallback frontends.
