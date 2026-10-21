# Model files: sources, config, safetensors, GGUF, PyTorch and ONNX

`crit_core.model` reads the model files an engine needs before it loads
weights: one source per model whatever its weight format ([source.py](source.py)),
`config.json` ([config.py](config.py)), safetensors headers and
indexes ([safetensors.py](safetensors.py)), GGUF headers ([gguf.py](gguf.py))
and the description files a GGUF derives ([gguf_files.py](gguf_files.py)),
PyTorch state dicts ([torch_pickle.py](torch_pickle.py)) and ONNX graphs
([onnx.py](onnx.py)).
All use the standard library only and never decode tensor bytes. Names are re-exported from [`__init__.py`](__init__.py);
strict JSON comes from [jsonfile.py](../jsonfile.py) and `ModelFiles` sources
from [files.py](../files.py).

## Model sources

`open_model(path)` takes what the user points at, a model folder or a weight
file, and returns a `ModelSource`. Everything that is not a weight (the config
rules, `crit_core.text`, Eval's context and ceiling, `crit run recon`,
`crit serve`) reads the model through it:

| Field | What it is |
| --- | --- |
| `format` | `"safetensors"`, `"gguf"`, `"torch"`, `"onnx"`, or `None` for a folder without weight files (or a file in an engine's own format) |
| `files` | the description files as a `ModelFiles` source: the folder's own, plus those the format derives from its weights' metadata (a fresh source per access, so its records cover one reader) |
| `derived_names` | which of them were derived |
| `weights` | the format's header object (`Checkpoint`, `GgufModel`, `TorchCheckpoint`, `OnnxModel`), validated on first use |
| `weight_files(drafts=False)` | the files holding the language model's tensors, each with `.path` and `.tensors`; with `drafts`, draft (`mtp`) files beside the checkpoint join them |
| `draft_blocks` | draft layers stored among the model's own blocks (a GGUF's `nextn` layers) |
| `config()` | `config.json` read with the rules below; an empty config when there is none |

A weight format is one `ModelFormat(name, suffixes, open, holds, fallback)` in
`FORMATS`: the suffixes of the files that hold its weights, `open(path) ->
ModelSource` for a folder or one of those files, `holds(path)` for a format
whose suffix other files share (a `.bin` is torch's only when torch.save wrote
it), and `fallback` for a format whose files a folder may carry beside another
format's (Hugging Face repositories ship `pytorch_model.bin` or `model.onnx`
next to `model.safetensors`, and the safetensors are the weights). A folder
holding two formats that are not fallbacks is a `ModelSourceError`. A lone
`.safetensors` or `.bin` file has no description files; a lone `.gguf` has all
of them, derived; a lone `.onnx` reads its folder's, or the repository's above
an `onnx/` folder. Description files come in groups (the
config; the tokenizer with its templates; the generation config): a folder
that holds any file of a group supplies that whole group, and only the groups
it lacks are derived. Results are cached while the folder's entries are
unchanged. Eval's reference also needs the weights as the Hugging Face
checkpoint they came from; that part needs torch and lives in Eval
(`crit_eval.reference.open_checkpoint`).

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

## GGUF

`inspect_gguf(path)` validates one `.gguf` file (versions 2
and 3, little-endian) and returns a `GgufFile`; `inspect_gguf_model(path)`
accepts a file, the first part of a split set or a model directory and returns
a `GgufModel`. The file is mapped read-only and only the header pages are
touched; tensor bytes are never decoded. A vocabulary-only file (no tensors,
as `--vocab-only` writes it) may end with its header, unpadded.

**Header rules.** Magic `GGUF`; a byte-swapped version is refused as
big-endian. Metadata keys are unique non-empty UTF-8 strings of known value
types, arrays nest at most `ARRAY_DEPTH` (2) levels and BOOLs are 0 or 1.
`general.alignment` is a UINT32 power of two (default 32). Tensor names are
unique, with 1 to 4 dimensions of at least 1, a known ggml type (`GGML_TYPES`:
F32, F16, BF16, the integer types, Q4_0 to Q8_1, the K-quants, the i-quants,
TQ1_0/TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0) and `ne[0]` a multiple of the type's
block. Offsets sit on the alignment, and the tensors tile the data section in
offset order with nothing between them but padding, up to the end of the file.
`Limits.max_header_bytes` bounds metadata plus tensor table.

**Results.** `TensorInfo(name, dtype, shape, shard, data_begin, data_end)`
has the ggml type name as `dtype`, the row-major shape (`ne` reversed, as torch
orders it) and absolute file offsets; `ne`, `elements`, `nbytes` and
`ggml_type` (block, type size) derive from them. `GgufFile` holds the metadata
(arrays as tuples), each key's value type, the tensors in table order, the
alignment, the data offset and the SHA-256 of every byte before it.
`GgufModel` joins the parts in split order (tensors sorted by name, the first
part's metadata); `get(".block_count")` reads an architecture key.

**Split sets and folders.** `find_gguf(directory)` picks the folder's only
`.gguf` or the first part of its only `NAME-00001-of-0000N.gguf` set, ignores
multimodal projectors (`mmproj` in the name) and refuses two models. Each part
must carry the `split.no` and `split.count` its name says, and
`split.tensors.count` must equal the tensors found.

**Description files.** GGUF's converter writes the model's description
into the GGUF's metadata, and [gguf_files.py](gguf_files.py) gives it back as
the files the rest of crit reads: `config.json` (`derive_config`, by
`CONFIG_RULES` per architecture: `llama`, `qwen2`, `qwen3`, `qwen3moe`,
`qwen35`, `qwen35moe`, each the inverse of the converter's parameters for it,
with what the reference runtime fixes for the architecture stated; another architecture
gets the generic shape keys when its metadata needs nothing more, and is
refused by name otherwise), `tokenizer.json` (`derive_tokenizer`: byte-level
BPE with the pre-tokenizer regex, normalizer and `ignore_merges` that
`PRE_TOKENIZERS` gives `tokenizer.ggml.pre`), `tokenizer_config.json`, the
chat templates (`chat_template.jinja` and `additional_chat_templates/`) and
`generation_config.json` (BOS, padding, the end-of-generation ids and
`general.sampling.*`). FLOAT32 values become the shortest decimal that rounds
to them. A Llama 3 GGUF's `rope_freqs` factors become the released
`llama3` RoPE parameters that reproduce every factor.

**A GGUF beside the base model's files.** A folder's own `config.json` or
tokenizer files are used instead of derived ones, and must agree with the
GGUF. `check_gguf_folder(model, config, directory)` compares them and returns a
`GgufCheck`: `errors` for
layers (the GGUF's `block_count` less its `nextn_predict_layers`), hidden size,
attention and key/value heads (per-layer arrays, zero for recurrent layers),
head size, MLP or expert widths, expert counts, vocabulary rows, RMS norm
epsilon and RoPE base (as FLOAT32), every token string (the converter's `[PAD<i>]`
fillers aside) and BPE merge against `tokenizer.json`; `notes` for a different
declared context length and a chat template that differs from the GGUF's
`tokenizer.chat_template`, where the folder's files are used.
`raise_errors(directory)` turns the errors into a `GgufError`.

## PyTorch state dicts

`inspect_torch(path)` validates one file `torch.save` wrote (`pytorch_model.bin`,
`*.pt`, `*.pth`: a zip archive of STORED entries) and returns a `TorchFile`;
`inspect_torch_model(path)` accepts a file, a `.bin.index.json` or a folder
(its `pytorch_model.bin.index.json`, else `pytorch_model.bin`, else its only
state dict) and returns a `TorchCheckpoint`. The pickle is read without running
it: the unpickler resolves only the globals a state dict uses (`OrderedDict`,
torch's tensor and parameter rebuild functions, storage types, dtypes,
`torch.Size`) to inert stand-ins and refuses any other global, so a file that
would run code under `torch.load` is refused instead. The state dict must hold
only tensors, each a contiguous view of its storage (tied weights share one;
float8 comes through `UntypedStorage`), little-endian, inside its storage
entry. The legacy format (before torch 1.6) is refused by name. Tensors are
`TensorInfo` with safetensors dtype names and absolute file offsets. Trainer
files beside a model (`training_args.bin`, `optimizer.pt`, ...) are not
weights.

## ONNX

`inspect_onnx(path)` walks one `.onnx` file's protobuf (memory-mapped, without
the onnx package) and returns an `OnnxModel` whose tensors are `OnnxTensor`s
named as the Hugging Face checkpoint the graph was exported from: a named
initializer keeps its name; a generated MatMul operand (`onnx::MatMul_123`, a
linear layer's weight stored `[in, out]`, `transposed`) and a Gemm's take the
consuming node's module scope (`/model/layers.0/mlp/up_proj/MatMul` gives
`model.layers.0.mlp.up_proj.weight`); a MatMulNBits node's inputs become
`<module>.nbits_weight`, `.nbits_scales`, `.nbits_zero_points` and
`.nbits_g_idx` with the node's `K`, `N`, `bits` and `block_size` as
`attributes`. Other generated initializers are constants, not weights.
External data must sit in a file beside the model, inside its bounds, with the
length the shape needs. Refused: graphs from onnxruntime-genai's model builder
(a `genai_config.json` beside them: fused, renamed layers), weights of other
quantized ops (`MatMulInteger`, `MatMulBnb4`, `QLinearMatMul`,
`DequantizeLinear`, ...), sparse or segmented initializers, weights stored as
varint fields and 4-bit element types. `find_onnx(folder)` takes a folder's
only `.onnx`.

## Errors

`ConfigError` and `SafetensorsError` are `ValueError` subclasses whose messages
start with the file, for example `model.safetensors: tensor 'w': data_offsets
[0, 2) hold 2 bytes but shape [2] of F32 needs 8`. A failing `ModelFiles` source
is a `ConfigError`; filesystem failures on safetensors files propagate as
`OSError`; invalid arguments raise `ValueError`. `GgufError` (a `ValueError`)
covers GGUF files, split sets, derivations and folder checks, `TorchFileError`
PyTorch files and `OnnxError` ONNX graphs; `ModelSourceError` a path that does
not exist or a folder of two formats.

## Differences from the C++ port

The model source, the GGUF reader and its derived files have no C++
counterpart yet: `Checkpoint::open` reads safetensors only.

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
