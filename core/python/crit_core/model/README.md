# Model files: config and safetensors

`crit_core.model` reads the two model files an engine needs before it loads
weights: `config.json` ([config.py](config.py)) and safetensors headers and
indexes ([safetensors.py](safetensors.py)). Both use the standard library only
and never read tensor bytes. Names are re-exported from [`__init__.py`](__init__.py);
strict JSON comes from [jsonfile.py](../jsonfile.py) and `ModelFiles` sources
from [files.py](../files.py).

## config.json

`load_model_config(source, *, max_bytes)` takes a model directory, a
`config.json` path or a `ModelFiles` source and returns a frozen `ModelConfig`;
any other source raises `TypeError`. `parse_model_config(dict, source)` does the
same for an already parsed object. The file is strict JSON with the native
reader's rules, except that `NaN`, `Infinity` and `-Infinity` load, as
transformers writes them (Mamba2's `time_step_limit` is `[0.0, Infinity]`).
Duplicate keys, an unpaired surrogate escape, nesting beyond 128 levels
(`MAX_CONFIG_DEPTH`), a number beyond a double's range and files over 16 MiB
(`max_bytes`, default `MAX_CONFIG_BYTES`) are errors. A non-directory
`ModelFiles` source is checked on what its reader returns, so `max_bytes` bounds
what is parsed, not what the source holds in memory. A missing `config.json`, or
one that cannot be opened as a regular file, is a `ConfigError`.

Non-finite numbers and integers outside `[-2^63, 2^64)` are kept where no typed
field reads them; typed fields refuse them. Booleans are never numbers.

**Text section.** `text_section(config)` returns the config with `text_config`'s
keys over the root's, so root-only keys such as `architectures` stay visible.
Every typed field and rule reads this section. `ModelConfig.get(key)` reads it;
`raw` is the root, `text` the `text_config` and `section` the merged view, all
read-only; `to_dict()` returns a plain copy of the root.

**Typed fields** mirror the native `ModelConfig` (sizes, head counts, positions,
norm epsilon, `dtype`, `rope`, `rope_by_layer_type`, `layer_types`, `moe`, `mla`,
`linear`, token ids and `quantization`; see [config.py](config.py)). The
quantization subtree is kept raw, never interpreted; unknown keys stay reachable
through `get`. No architecture defaults are invented.

`RopeConfig` reads `rope_scaling`, else `rope_parameters`, with `rope_type`
before `type`; `theta` is the section's `rope_theta`, else the rope object's. A
rope object whose members are all objects is keyed by layer type (transformers
v5 Gemma 3): `rope_by_layer_type` then holds one `RopeLayer(layer_type, rope)`
per entry in file order, each read on its own with no fallback to the section.

**Rules shared with crit eval.** These take a config mapping (usually
`text_section(...)`), return what crit eval's protocol `crit-eval-v2` rules
returned for every well-typed config, and are what Eval calls. `ModelConfig` has
a method for each.

| Function | Result |
| --- | --- |
| `position_limit(config)` | `(limit, source)`: the first positive integer among `max_position_embeddings`, `n_positions`, `max_seq_len`, `seq_length`, `max_sequence_length`, `model_max_length`; a YaRN `rope_scaling` (else `rope_parameters`) with positive `original_max_position_embeddings` and `factor` overrides it with their product. `(None, None)` when nothing is declared. |
| `attention_shape(config)` | `(full layers, heads, head dim, KV heads)` or `None`. Full layers count `layer_types` entries `full_attention`/`attention`, else `layers // full_attention_interval`, else all layers. Head dim is `head_dim`, else `hidden_size // heads`. |
| `kv_elements_per_token(config)` | K and V elements per token over full-attention layers; with `kv_lora_rank`, `layers * (kv_lora_rank + qk_rope_head_dim)`. |
| `fixed_state_bytes(config, kv_bytes_per_element)` | Linear-attention recurrence (FP32 state, BF16 window of `linear_conv_kernel_dim - 1`, default kernel 4) plus the KV cache of `sliding_attention` layers. |

A rule reads values in crit eval's order and only where crit eval does, and
validates each value it reads: a wrongly typed, negative or out-of-range count,
or a malformed `layer_types`, raises `ConfigError` where crit eval would compute
with it. A YaRN product of 2^64 or more, or one that is not finite, raises
`ConfigError` naming the factor's key.

**Strict helpers.** `layer_kinds(config)` returns `FULL_ATTENTION`,
`SLIDING_ATTENTION`, `LINEAR_ATTENTION` or `OTHER` per layer, from `layer_types`,
else `full_attention_interval` (as Hugging Face derives for Qwen3-Next and
Qwen3.5), else all full. `effective_head_dim`, `effective_kv_heads` and
`require_int(key)` raise instead of rounding or defaulting.

**Limits of the rules.** A `sliding_window` without `layer_types` (Mistral style)
counts as full attention, so KV estimates are upper bounds. `fixed_state_bytes`
counts recurrent state only for layers `layer_types` lists as `linear_attention`,
so a hybrid declared only through `full_attention_interval` reports none. Only
`text_config` nests.

## safetensors

`inspect_safetensors(path)` validates one `.safetensors` file and returns a
`ShardInfo`; `inspect_model(path)` accepts a file, a `*.safetensors.index.json`
or a model directory and returns a `Checkpoint`. Only the 8-byte length prefix,
the header and the index are read, unbuffered, so no read reaches tensor bytes.

A directory resolves to `model.safetensors.index.json`, else
`model.safetensors`, else its only `*.safetensors` file; several candidates and
no index is an error. Files beside an index are never picked up silently:
`extra_files=["model_mtp.safetensors"]` adds named flat files, whose tensors may
not repeat another file's and which count toward the limits.

**Header rules.** The header is integer-only strict JSON (`JSON_DEPTH` 64) with
`__metadata__` mapping strings to strings and every other entry holding exactly
`dtype`, `shape` and `data_offsets`. Extents must match the offsets, and tensors
must cover the payload contiguously to the end of the file. The file must keep
its identity from open until the header is read.

**Dtypes** (`DTYPE_BITS`): `BOOL`, `U8`, `I8`, `F8_E4M3`, `F8_E5M2`, `F8_E8M0` 8;
`U16`, `I16`, `F16`, `BF16` 16; `U32`, `I32`, `F32` 32; `U64`, `I64`, `F64` 64;
`F4` 4; `F6_E2M3`, `F6_E3M2` 6. A sub-byte tensor must fill whole bytes.
`TensorInfo.element_aligned` reports whether the file offset is a multiple of
the element size.

**Index rules.** `weight_map` is a non-empty object of tensor names to portable
flat shard names; each tensor sits in exactly the shard the map names, and the
map lists exactly the tensors in the headers. `NaN` and `Infinity` are refused
anywhere in an index.

| `IndexPolicy` | Differences |
| --- | --- |
| `STRICT` | Top-level keys only `metadata` and `weight_map`; metadata only `total_size`, which must equal the tensor bytes. |
| `COMPATIBLE` (default) | Also `metadata.total_parameters`. Unknown keys and a `total_size` that disagrees with the headers become `Checkpoint.notes`. |

**Options.** `header_only=True` reads header-only files (planning fixtures,
stripped checkpoints) whose offsets may lie beyond the file.
`follow_symlinks=True` (default) reads symlinked files such as Hugging Face cache
snapshots; `False` refuses them.

**Limits** (`Limits.production()`, the defaults): 100,000,000-byte headers,
64 MiB indexes, 2**20 tensors, 4096 files, rank 16, 4 TiB per file and 64 TiB per
checkpoint. The header size bounds the reader's memory.

**Results.** `TensorInfo(name, dtype, shape, shard, data_begin, data_end)` has
absolute file offsets; `ShardInfo` adds the header SHA-256 and file identity;
`Checkpoint` holds tensors sorted by name, shards, the index SHA-256, notes and
`tensor_bytes`. All results are read-only.

## Errors

`ConfigError` and `SafetensorsError` are `ValueError` subclasses whose messages
start with the file, for example `model.safetensors: tensor 'w': data_offsets
[0, 2) hold 2 bytes but shape [2] of F32 needs 8`. A failing `ModelFiles` source
is a `ConfigError`; filesystem failures on safetensors files propagate as
`OSError`; invalid arguments raise `ValueError`.

## Differences from the C++ port

The C++ loaders in [core/src/model](../../../src/model/README.md) (`load_model_config`,
`Checkpoint::open`, `load_generation_config` and `stop_token_ids`) apply the
same rules as this package and `crit_core.text`'s `GenerationConfig` and
`legacy_eos_ids`. What differs on purpose:

- Tensor order: `Checkpoint.tensors` is sorted by name; the C++
  `Checkpoint::tensors()` is shard-major (header order within each shard).
- Errors: Python raises `ConfigError`, `SafetensorsError` or `ModelTextError`
  (all `ValueError`) where C++ throws `FormatError`, and `OSError` where C++
  throws `std::system_error`. Python messages give line and column; C++ gives a
  byte offset.
- Limits: the C++ `retained_bytes` limit and JSON node limit
  (`json::Limits::max_nodes`, 2^24 by default) have no Python counterpart;
  Python bounds a document by bytes and depth. An integer longer than
  `sys.get_int_max_str_digits()` (4300 by default) is a JSON error in Python
  even under an unread key; C++ keeps it.
- Stop ids: `legacy_eos_ids`, like crit eval, reads only the eos ids; the C++
  `stop_token_ids` takes a validated `GenerationConfig`, so a malformed sampling
  field fails before any stop id is read.
- Sources: `load_model_config` also accepts a `ModelFiles` source, and
  `GenerationConfig.from_files` follows transformers (falling back to
  `config.json` and checking ids against a tokenizer); the C++
  `GenerationConfig` is `generation_config.json` as written.
