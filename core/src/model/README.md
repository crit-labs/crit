# Model checkpoints and configs

Validated access to Hugging Face style model directories: safetensors checkpoints
(single files, sharded indexes and extra files such as a draft layer), their tensor
bytes and file digests, and typed views of `config.json` and
`generation_config.json`. Tensor values are never decoded; only names, dtypes,
shapes and byte extents are interpreted. Built on the strict JSON, read-only file
and SHA-256 pieces in [io](../io/README.md).

| Header | Source | Ships |
| --- | --- | --- |
| [safetensors.h](../../include/crit/safetensors.h) | [safetensors.cpp](safetensors.cpp) | `DType`, `TensorInfo`, `SafetensorsLimits`, `IndexPolicy`, `CheckpointOptions`, `SafetensorsFile`, `Checkpoint` |
| [model_config.h](../../include/crit/model_config.h) | [model_config.cpp](model_config.cpp) | `ModelConfig`, `load_model_config`, `GenerationConfig`, `load_generation_config`, `stop_token_ids` |

## Safetensors files

A file is an 8-byte little-endian header length `N`, `N` bytes of JSON, then the
payload. `TensorInfo::begin` and `end` are absolute file offsets
(`8 + N + data_offset`). Dtypes: `BOOL`, `U8`, `I8`, `U16`, `I16`, `F16`, `BF16`,
`U32`, `I32`, `F32`, `U64`, `I64`, `F64`, `F8_E4M3`, `F8_E5M2`, `F8_E8M0` (8 bits),
`F6_E2M3`, `F6_E3M2` (6) and `F4` (4); spellings are exact. A sub-byte tensor must
fill whole bytes. `element_aligned()` reports (but does not require) alignment.

`SafetensorsFile::open(path, options)` reads only the prefix and header, never the
payload, and nothing larger than a checked limit is reserved or read. It checks:
the `.safetensors` name; file and header sizes; that the file is unchanged after
the header read; strict JSON (no duplicate keys, UTF-8, finite numbers, depth 64;
errors name `header byte K`); `__metadata__` as string to string; per tensor a
valid name, exactly `dtype`, `shape` and `data_offsets`, a known dtype, bounded
rank and dimensions, and an extent equal to the shape's byte size; and finally
that extents tile the payload from offset 0 with no hole or overlap up to the end
of file. `tensors()` keeps header order.

### Limits

`SafetensorsLimits` values must be in `[1, 2^63 - 1]`:

| Limit | Default | Bounds |
| --- | --- | --- |
| `header_bytes` | 100,000,000 | one header |
| `index_bytes` | 64 MiB | the index file |
| `name_bytes` | 4096 | tensor and shard names |
| `dimension` | 2^48 | one shape entry |
| `tensors` | 2^20 | records per file and across files |
| `shards` | 4096 | index shards plus extra files |
| `rank` | 16 | shape length |
| `file_bytes` | 2^42 (4 TiB) | one file |
| `total_bytes` | 2^46 (64 TiB) | the index plus every file |
| `retained_bytes` | 2^30 (1 GiB) | memory kept after open, summed over a `Checkpoint` |

A file is charged its raw header, metadata text and per tensor
`sizeof(TensorInfo)` plus name bytes plus 8 bytes per dimension; a `Checkpoint`
adds 4 bytes per tensor for its name index. Parsed JSON is released before `open`
returns.

## Checkpoints

`Checkpoint::open(path, options)` accepts a `.safetensors` file, a
`*.safetensors.index.json` index or a directory, which resolves to
`model.safetensors.index.json`, else `model.safetensors`, else its only
`*.safetensors` file (several without an index is an error listing them).

### Index rules

The index is strict JSON with a `weight_map` from tensor name to shard file name,
and optional `metadata`. Shard names are portable relative paths. They have at most
8 `/`-separated components of `[A-Za-z0-9][A-Za-z0-9._-]*`, and the last one ends in
`.safetensors`, as in `tensors/model-00001-of-00004.safetensors`. No component is a
Windows device name, and none can be `.` or `..`, so no path leaves the index's
directory except through a link. A tensor in two shards, in a shard but not listed,
listed for another shard, or listed but absent are errors.

| Aspect | `IndexPolicy::strict` | `IndexPolicy::compatible` (default) |
| --- | --- | --- |
| Top-level keys besides `metadata` and `weight_map` | error | one note |
| `metadata` keys besides `total_size` | error | `total_parameters` accepted; one note for others |
| `total_size` differing from summed tensor bytes | error | note |

Notes are returned by `notes()`. Types are checked under both policies, and
`NaN`/`Infinity` are refused everywhere in an index.

### Extra files

`CheckpointOptions::extra_files` adds flat `.safetensors` names from the source
directory, such as a multi-token-prediction layer outside the index. They are
validated like shards, follow them in `shards()`, count against the limits, and
must not repeat a file or hold a tensor another file holds.

### Symbolic links

`follow_symlinks = true` (default) opens links, as Hugging Face cache snapshots
need; digest rows name the link and hash the target. `false` refuses a link as the
final component of every opened file with `std::system_error`.

### Header-only mode

`header_only = true` validates headers and index rules without the payload
(offsets bounded by `file_bytes`; holes and overlaps still fail). `map`, `read`,
`read_chunks` and `stream_shard` throw `std::logic_error`; digests and
`verify_unchanged` work.

## Tensor bytes

`tensors()` holds every tensor once, shard-major in `shards()` order and header
order within a shard; `shards()[i].tensors()` is a slice of the same records.
`find` is a binary search; `at` throws `std::out_of_range`. Byte access takes a
`TensorInfo` from this checkpoint:

| Member | Behavior |
| --- | --- |
| `map(t)` | Zero-copy view into the shard's read-only mapping plus an owner that outlives the checkpoint |
| `read(t, dst)` | Exactly `t.nbytes()` bytes |
| `read_chunks(t, consume, options)` | `ReadOnlyFile::read_chunks` over the extent (absolute offsets) |
| `stream_shard(i, consume, options, sha256_hex)` | One pass over shard `i` in file order, then `verify_unchanged()`; with `sha256_hex`, stores the file digest only after success |
| `verify_unchanged()` | Checks every file against its identity at open |

Const members may run concurrently. On POSIX a file truncated under a mapping
raises `SIGBUS`; pair `map` with `verify_unchanged()` and controlled storage.

## Digests

`digest_files(options)` returns run-record manifest rows `{path, bytes, sha256}`
for the index and every shard and extra file, sorted by path.
`verify_digests(expected, options)` requires one row per checkpoint file (others
are ignored), checks presence and sizes before hashing, and throws `FormatError`
naming the file on any mismatch.

## Model config

`load_model_config(dir_or_file, limits)` reads `config.json` as strict JSON except
that `NaN`/`Infinity` load, as transformers writes them. Fields read the text
section: a `text_config` key wins over the root's, and `null` hides the root
value. Without `text_config`, an `llm_config` object (multimodal wrappers such as
NemotronH_Omni) is the text section. Which keys spell a field, in the order they are
tried, comes from one table both readers share,
[config_keys.json](../../python/crit_core/model/config_keys.json): CMake generates the C++
arrays from it and `crit_core.model.config` loads it. A model family that spells a field
its own way gets a key there, not a branch in either reader. Known keys with the wrong type throw `FormatError` "key 'x': ...";
booleans never count as numbers and reals never as integers. Typed fields refuse
non-finite numbers and integers beyond 64 bits; unread keys may hold anything.

| Field | Keys |
| --- | --- |
| `architectures`, `model_type`, `text_model_type` | `architectures`, root and section `model_type` |
| Sizes and counts | `vocab_size`, `hidden_size`, `intermediate_size` (or per-layer `intermediate_sizes`), `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` |
| `max_positions` | first positive of `max_position_embeddings`, `n_positions`, `max_seq_len`, `seq_length`, `max_sequence_length`, `model_max_length` |
| `rms_norm_eps` | first of `rms_norm_eps`, `layer_norm_eps`, `layer_norm_epsilon` |
| `dtype` | `dtype`, else `torch_dtype` |
| `rope` | `rope_scaling`, else `rope_parameters`; `rope_theta`; per-layer-type entries in `rope_by_layer_type` |
| `sliding_window`, `full_attention_interval`, `tie_word_embeddings`, `num_nextn_predict_layers` | same names |
| `layer_types` | `layer_types`, else `layers_block_type` (`mamba` and `moe` layers read as other kinds; without `num_hidden_layers` its length is the layer count, `layer_count_from`) |
| `moe` | `num_experts`/`num_local_experts`/`n_routed_experts` and the routing keys |
| `mla` | positive `kv_lora_rank` with its head dims |
| `linear` | `linear_*` head and kernel keys |
| `eos_token_ids`, `bos_token_id`, `pad_token_id` | same keys; `eos_token_id` may be a list |
| `quantization` | `quantization_config` (`method` is `quant_method`) |

Derived values: `effective_head_dim()` (`head_dim`, else `hidden_size /
num_attention_heads`, which must divide); `effective_kv_heads()`; `layer_kinds()`
from `layer_types`, else every `full_attention_interval`-th layer full and the
rest linear, else all full; and `position_limit(&source)`, crit eval's
`model_limit`, which applies YaRN scaling (`original x factor`) when present.

`rope_table(rope, head_dim, max_position_embeddings, sequence_length)` turns a
`RopeConfig` into the rotary inverse frequencies (one per rotated pair of
`rotary_dim(rope, head_dim)`) and the attention factor that scales cos and sin,
as Hugging Face's `ROPE_INIT_FUNCTIONS` derive them: default and `mrope`,
`linear`, `dynamic` NTK, `yarn` (`mscale`/`mscale_all_dim`, a declared
`attention_factor` used as is, `truncate`), `llama3` and `longrope`. Engines read
their table here instead of deriving it; the derivation is where serving stacks
lose accuracy silently (a YaRN scale applied twice, `mscale` ignored).
`crit_core.model.rope_table` is the Python twin; `tests/test_core_rope.py` checks it
against Transformers and `tests/core/rope_oracle.py` checks the two agree.

### Differences from crit eval

The typed reader is stricter where eval would guess: wrong-typed known keys fail
at load; a `null` `text_config` is absent; `layer_types` must have exactly
`num_hidden_layers` entries; `hidden_size` must divide by the head count when
`head_dim` is absent (eval floors); a YaRN limit of 2^64 or more is an error.

## Generation config and stop ids

`load_generation_config(dir_or_file, limits)` returns nullopt when a directory has
no `generation_config.json`. It types `eos_token_ids`, `bos_token_id`,
`pad_token_id`, `do_sample`, `temperature`, `top_p`, `min_p`,
`repetition_penalty`, `top_k`, `max_new_tokens`, `max_length` and `stop_strings`;
other keys stay in `document`. It is not a `SamplingParams`: an adapter
translates it, and `do_sample = false` or `top_p = 0` means greedy.

`stop_token_ids(config, generation)` is crit eval's `prepare.stop_ids`: the sorted
union of `eos_token_id` in the root and `text_config` of both configs. Negative
stop ids throw `FormatError`.

## Errors

| Failure | Exception |
| --- | --- |
| Malformed checkpoint data | `FormatError` `"<file>: <where>: <reason>"` |
| Malformed config | `FormatError` `"<file>: key 'x': <reason>"` |
| OS failure, refused symbolic link | `std::system_error` naming the path |
| Bad limits, options, destination size or expected digests; a foreign tensor; not a regular file | `std::invalid_argument` |
| Byte access on a `header_only` checkpoint | `std::logic_error` |
| `at` of an absent name, `stream_shard` of a missing shard | `std::out_of_range` |
| A file changed since open | `std::runtime_error` "file changed since open" |

`open` returns a complete object or throws.

## Differences from the Python port

The Python port (`crit_core.model`'s `load_model_config` and `inspect_model`, and
`crit_core.text`'s `GenerationConfig` and `legacy_eos_ids`) applies the same
rules. Intentional differences:

- Tensor order: C++ is shard-major; Python's `tensors` mapping is sorted by name.
- Errors: C++ `FormatError` is `ConfigError`, `SafetensorsError` or
  `ModelTextError` (all `ValueError`) in Python; other OS failures are `OSError`.
- Messages: C++ locates JSON errors by byte offset, Python by line and column.
- Limits: `retained_bytes` and the JSON node limit exist only in C++. Python
  refuses integers longer than `sys.get_int_max_str_digits()` in any JSON file.
- Stop ids: `stop_token_ids` validates the whole generation config first;
  `legacy_eos_ids` reads only the eos ids.
- Sources: Python's `load_model_config` also accepts `ModelFiles`, and
  `GenerationConfig.from_files` follows transformers' fallbacks.

The [native fixture](../../../tests/core/model_tests.cpp) builds every file at run
time, including adversarial headers and indexes and crit eval config fixtures.
