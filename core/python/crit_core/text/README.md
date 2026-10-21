# Text layer (Python)

`crit_core.text` turns chat messages into prompt token ids and generated ids back
into reply text. It renders chat templates the way Hugging Face transformers 5.19
does, tokenizes with the `tokenizers` library, streams text without retraction,
and applies stop ids and stop strings with Core's stop rules. It needs
`tokenizers` and `jinja2` (pinned in CI to 0.23.3 and 3.1.6). Names are
re-exported lazily ([`__init__.py`](__init__.py)): `from crit_core.text import
StopMatcher` loads only the standard library.

```python
from crit_core.text import load_model_text

model = load_model_text("path/to/model", clock=None)       # or a crit_core.files.ModelFiles source
ids = model.encode_chat([{"role": "user", "content": "Hi"}], add_generation_prompt=True, enable_thinking=False)
stream = model.text_stream(ids, stop_strings=["\n\n"])
for chunk in engine_output_chunks:                           # ids produced by an engine
    delta = stream.push(chunk)                               # delta.text, delta.consumed, delta.finish_reason
    if delta.finish_reason:
        break
else:
    delta = stream.finish()                                  # flushes held text; finish_reason "length"
```

## Modules

| Module | Contents |
| --- | --- |
| [errors.py](errors.py) | `TextError(ValueError)` and its subclasses `ModelTextError`, `TemplateError`, `TokenError`; read caps `MAX_FILE_BYTES` and `MAX_CONFIG_BYTES` |
| [tokenizer.py](tokenizer.py) | `Tokenizer`, `AddedToken`, `SPECIAL_TOKEN_NAMES`, `special_tokens_from` |
| [template.py](template.py) | `environment`, `ChatTemplate`, `ChatTemplates`, `template_sources`, `GenerationTag` |
| [messages.py](messages.py) | `openai_messages` (OpenAI chat messages to template messages) |
| [generation.py](generation.py) | `GenerationConfig`, `legacy_eos_ids`, `model_generation_view` |
| [detokenize.py](detokenize.py) | `IncrementalDetokenizer` |
| [markers.py](markers.py) | `MarkerDecoder`, `TokenText`; delimiter provenance for reasoning and tool parsing |
| [stops.py](stops.py) | `StopMatcher` (standard library only) |
| [stream.py](stream.py) | `TextStream`, `TextDelta` |
| [model.py](model.py) | `load_model_text`, `ModelText`, `TextIdentity` |

## Loading a model

`load_model_text(model, *, clock=datetime.now, split_special_tokens=False)` takes
a directory path or any `ModelFiles` from [crit_core.files](../files.py); any
other object raises `TypeError`. Every read passes a cap
(`read(name, max_bytes=...)`), so a subclass that overrides `read` must accept
`max_bytes`; one that implements the `_read` and `_names` hooks needs nothing
more. `tokenizer.json` is required; `tokenizer_config.json`,
`special_tokens_map.json`, `chat_template.jinja`, `additional_chat_templates/*.jinja`,
`chat_template.json`, `generation_config.json` and `config.json` are read when
present. A model without a chat template loads, but `render` raises
`ModelTextError`.

`ModelText.identity` records each file read (`FileRecord`), the `crit_core`,
`jinja2` and `tokenizers` versions and a sha256 over both; callers record their
own render variables (a fixed clock, `enable_thinking`).

Caps are 16 MiB (`MAX_CONFIG_BYTES`) for `config.json` and
`generation_config.json` and 128 MiB (`MAX_FILE_BYTES`) for the others. JSON
files other than `tokenizer.json` follow the strict reader
([jsonfile.py](../jsonfile.py)) and raise `ModelTextError` naming the file.

## Tokenizer

- `Tokenizer.from_bytes`, `from_file` and `from_files(files)`; `backend` is the
  shared read-only `tokenizers.Tokenizer` with truncation and padding off.
  `vocab_size` is max id + 1 including added tokens (ids must be contiguous).
- `added_tokens`, `special_ids`, `special_tokens` and
  `bos_token_id`/`eos_token_id` follow transformers.
- `encode(text, *, add_special_tokens, split_special_tokens=None)`: the keyword
  is required; True applies the post-processor (for example a BOS). Control-token
  text such as `<|im_end|>` is parsed as the control token, as in transformers;
  `split_special_tokens=True` splits it into ordinary pieces. `encode_chat`
  always parses control tokens.
- `decode(ids, *, skip_special_tokens=False)` keeps special tokens by default, as
  transformers does. `clean_up_tokenization_spaces` is not applied.
- `validate`, `decode`, `id_to_token` and the detokenizer raise `TokenError` for
  anything but an integer in `[0, vocab_size)`. `token_to_id` returns None for
  unknown text.

## Chat templates

`environment(clock)` builds transformers' sandboxed environment, including the
`{% generation %}` tag, `raise_exception` and `strftime_now` over `clock()`. With
`clock=None`, `strftime_now` is undefined and templates use their own fallback
date, which keeps renders reproducible.

`ChatTemplate(source, *, name, special_tokens, clock)` compiles at construction.
`render(messages, *, tools, documents, add_generation_prompt,
continue_final_message, assistant_spans, **variables)`:

- Variables are the special-token strings overridden by the caller's
  (`enable_thinking`, `date_string`, ...), plus `messages`, `tools`, `documents`
  and `add_generation_prompt`. An empty conversation is refused.
- `continue_final_message` (True, or the field to continue) renders the final
  message open-ended with transformers' tag algorithm. It cannot be combined
  with `add_generation_prompt` or `assistant_spans`.
- A list passed as `assistant_spans` receives the `(start, end)` character spans
  of `{% generation %}` blocks.
- Every failure raises `TemplateError` with the template name and line.
- Output above `MAX_RENDER_CHARS` (2^26 characters) is refused.

`template_sources(files)` discovers templates as transformers does
(`chat_template.jinja`, `additional_chat_templates/`, `tokenizer_config.json`,
legacy `chat_template.json`). `ChatTemplates.select(name=None, *, tools=None)` picks
the named template, else `tool_use` when tools are given and it exists, else
`default`, else raises `ModelTextError`.

## Messages

`openai_messages(messages)` returns new dicts for templates: `developer` becomes
`system`; null content becomes `""` and text parts are joined;
`tool_calls[*].function.arguments` given as a JSON string becomes a dict
(`TextError` unless it is an object); `tool_call_id`, `name` and
`reasoning_content` are kept and other keys dropped.

## Generation config

`GenerationConfig.from_files(files, tokenizer=None)` reads
`generation_config.json`, else `config.json` the way transformers'
`from_model_config` does. A field (ids, sampling values, `max_new_tokens`,
`stop_strings`, `raw`) is None unless a file sets it; `sampling_defaults()`
returns only explicit values, never transformers' implicit `top_k=50`.
`stop_token_ids` is the ordered union of `eos_token_ids`, the `eos_token_id` of
both files and their `text_config`, and the tokenizer's eos token. Values are
checked against the native `GenerationConfig`'s ranges (eos ids in `[0, 2^31)`,
a `pad_token_id` of -1 allowed) and against the tokenizer; violations raise
`ModelTextError`.

`legacy_eos_ids(files)` is crit Eval's `stop_ids` rule, used by `crit eval
prepare`: the sorted union of `eos_token_id` at the top level and in
`text_config` of both files.

## Streaming text

`IncrementalDetokenizer(tokenizer, prompt_ids, *, skip_special_tokens=True, context_tokens=5)`
mirrors Core's `TextDecoder`: `push(ids)` returns newly stable text, `finish()`
flushes the rest, and the decoder is closed after `finish()` or any error. Each
step decodes a short window starting a few tokens back, so byte-level splits,
SentencePiece byte fallback and leading spaces come out exactly, and text is
never retracted.

- Release: a step releases what the window decodes to beyond the text already
  released, except a trailing run of U+FFFD, which may be an incomplete
  character; that run is held until text follows or `finish()`.
- Prompt: the window is seeded with the prompt's last `context_tokens` ids,
  moved back to a character start. Prompt text is never released; when the
  prompt ends inside a character, the output starts with the character it
  completes.
- Result: for the tokenizer's own encodings, the concatenated results equal
  `decode(prompt + ids)` without the prompt's complete characters, apart from
  the limits below.
- Bounded memory: the window keeps `max(context_tokens, 2)` released ids. After
  16 ids without a clean boundary (`HOLD_LIMIT`) it is cut, and a held U+FFFD
  run is released except its last three characters.
- `anomalies` counts the times already released text changed (a SentencePiece
  byte run that a later byte turns into U+FFFD); decoding then restarts from the
  last clean boundary without repeating released text.

`tokenizers.decoders.DecodeStream` is not used: it re-emits a prefill that ends
mid-character and cannot flush.

`StopMatcher(stop_strings)` (at most 16, each 1 to 256 UTF-8 bytes) applies
Core's rule: the first stop to *complete* wins, ties go to the earliest start,
and only the longest suffix that is a proper prefix of a stop is held back, so
results never depend on chunking. `push(delta)` and `finish()` return releasable
text; `stopped`, `stop_index` and `stop_offset` (characters) describe the match.
`advance(cumulative, final=False)` takes the cumulative text of crit serve's
streaming callback: with stops `["abcd", "bc"]` the text `"abcd"` stops at
`"bc"` whether it arrives at once or in pieces.

`TextStream(tokenizer, prompt_ids, *, stop_strings, stop_token_ids, skip_special_tokens=True, include_stop_str=False)`
processes ids one at a time. A stop id is consumed and never decoded. A stop
string may span tokens and is matched on released text; the token that completes
it is the last consumed token. `TextDelta.consumed` counts the ids of that
`push` belonging to the reply, `finish_reason` is `"stop"` or, from `finish()`,
`"length"`, and `stop` is the matched string or id. `include_stop_str` appends a
matched stop string to the text.

## Reasoning and tool delimiters

`ReasoningSplitter(format, tokenizer=...)`, `ToolCallParser(format, tools,
tokenizer=...)` and `ToolReply(format, request, tokenizer=...)` resolve the format's
opening and closing markers independently. With the default `delimiters="auto"`,
a marker encoded as one special token requires that token id; ordinary tokens
spelling the same text cannot open or close a block. Other markers keep text
matching across chunks. `ReasoningFormat` and `ToolFormat` accept
`delimiters="text"` for formats that explicitly use spelling regardless of ids.
Without a tokenizer, standalone parsers retain their existing text API.

Feed token-aware parsers from `MarkerDecoder(tokenizer, prompt_ids,
markers={id: spelling}, skip_special_tokens=False)`. Its `push(ids)` and `finish()`
return strings and `TokenText(text, token_id)` events. Use the parsers' `token_ids`
maps (`{spelling: id}`) to select the delimiters. One incremental decoder retains
UTF-8 and whitespace context across those events. A decoder that does not release
a selected marker's spelling at its token raises `TextError`. Optional
`stop_token_ids` use `TextStream`'s stop handling. Alternatively, `stream=...` wraps
an existing text stream; its configured stop and decoding options stay in force.
With no selected token delimiters, chunks pass directly through that stream.
`ResidentChat` wraps its model's text stream for both buffered and streaming replies
and checks prompt-opened reasoning by the same token identity rule.

Reasoning text events preserve `TokenText` values for downstream tool parsing;
reasoning content, parsed call bodies and final tool text are plain strings.
Tool body parsers still define their own text syntax and escaping rules.

## Limits

- Templates are trusted like the rest of the checkpoint: the render cap bounds
  neither memory nor time.
- A model that emits U+FFFD itself delays that text until later text or
  `finish()`, at most 16 ids.
- `encode_chat` uses `add_special_tokens=False` like transformers, so a BOS comes
  only from the template (two leading BOS tokens raise `TextError`). Control-token
  text inside messages is parsed as control tokens; filter untrusted messages.
