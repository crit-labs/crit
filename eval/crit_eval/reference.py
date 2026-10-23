"""The quality reference: `crit eval reference --model DIR --data DIR`.

The reference is the checkpoint the data was prepared from, the one the user chose. It runs the model's official
Hugging Face `transformers` modeling code with FP32 activations and the stored weights dequantized exactly to FP32.
Decoder layers are streamed onto the GPU one at a time, so a model larger than device memory still runs; each layer's
stored bytes are read in large reads while the previous layer runs and reach the GPU in one transfer, where they are
dequantized. It scores every position of the prepared quality texts (protocol quality: each conversation's prompt
followed by its written answer) teacher-forced, once per model, suite and reference version, and stores each
position's target log-probability and top-k under the data (score_data, crit_eval.fixed); every engine's scores of
the same texts are judged against them. The conversations use eager attention with the masks the modeling code
builds; the long texts the same FP32 arithmetic in blocks of query rows (chunked_attention), with boolean masks only
where the modeling code requires one (sdpa's mask rules), since a 32K-token float mask and score matrix would not
fit. It needs the device to itself: stop the engine's server first.

The model's own forward drives every pass, so whatever its text model does before, between and after the decoder
layers (embedding scales, repeated residual streams, per-layer inputs, position ids, masks, a final norm or mixer)
is its own code. A pass first records the arguments the forward hands each decoder layer, without running it; then
each layer is loaded once and run for every sequence; finally the forward runs again with the last layer's output,
so its own tail produces the hidden states the head scores. The forward must hand each layer the previous layer's
output unchanged (checked). Mixture-of-experts layers use transformers' eager experts loop.

A text runs in chunks of PROMPT_CHUNK tokens that resume the transformers Cache the forward builds, kept per decoder
layer in FP32 scratch files (PromptState) until the text is scored. All chunks run layer-major (_layer_major): each
layer, once loaded, runs every chunk it can, a chunk once the layers whose cache state its forward reads have run
the chunks before it; reads of cache state the chunk-major order (each chunk a round through every layer) would hold
otherwise make the texts run that way, and a model whose cache cannot be resumed runs each text whole.

Checkpoint keys reach the modeling code's parameter names through transformers' own conversion mapping for the
model type (transformers.conversion_mapping): renamings, prefixes and converters such as per-expert matrices merged
into fused expert tensors or a sharded table concatenated. Each source matrix is decoded on its own first, unless the
checkpoint stores the scales of the converted tensor itself (one scale for a table stored in shards), in which case
the stored parts are joined before decoding. Embedding tables are never materialized: a lookup reads only the bytes
of the rows it needs, by parallel reads (Checkpoint.rows_many), and decodes those (plain, FP8 and NVFP4 storage; other
schemes decode the table once on the CPU). Tensors outside the text model (vision towers, draft layers) are never read.

Supported stored formats: safetensors, PyTorch state dicts and ONNX graphs (crit_core.model.open_model), their
weights unquantized or in any scheme of crit_eval.schemes (FP8, NVFP4, compressed-tensors int, AWQ, GPTQ, MLX,
bitsandbytes, ONNX Runtime's MatMulNBits), each decoded to exactly the FP32 values its own library produces.
A GGUF (a lone .gguf, a split set or a folder holding one) is read through crit_eval.gguf_layout:
every block type gguf-py decodes, with GGUF's reference converter layout undone per architecture, and the
config.json its metadata derives (crit_core.model.gguf_files) unless its folder has the base model's.
Requires torch, transformers and safetensors, gguf for a GGUF and numpy for ONNX. This is the slow, trusted path,
run once per prepared data.
"""
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from . import EVAL_DIR, load_json, write_json
from .prepare import PREPARED, model_identity
from .schemes import PRIMARY, SchemeError, quantization_of, refuse_unread

# torch dtype names -> safetensors dtype names
TORCH_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32", "float64": "F64", "float8_e4m3fn": "F8_E4M3",
                "float8_e5m2": "F8_E5M2", "float8_e8m0fnu": "F8_E8M0", "uint8": "U8", "int8": "I8", "int16": "I16",
                "int32": "I32", "uint32": "U32", "int64": "I64", "bool": "BOOL"}
CHUNKED = "crit_chunked"
QUERY_BLOCK = 256
# Tokens a text runs per chunk. REFERENCE_VERSION changes whenever what the reference computes does, so nothing
# stored by another version is read.
PROMPT_CHUNK = 8192
REFERENCE_VERSION = 3
# Schemes whose stored rows are the table's rows, with scales per tensor or per row (crit_eval.schemes; None is
# plain): an embedding stored so is gathered row by row.
GATHERABLE = (None, "fp8", "nvfp4")
# Readers that gather rows at once: enough to keep an NVMe queue busy with small reads.
GATHER_THREADS = 16
# A decoder layer's stored tensors are read ahead of it in large reads (Checkpoint.stage): neighbours in the file
# closer than STAGE_GAP bytes are read with the bytes between them, in pieces of STAGE_READ bytes by STAGE_THREADS
# readers, into one host buffer of at most STAGE_LIMIT bytes (a larger layer is read tensor by tensor).
STAGE_GAP = 1 << 20
STAGE_READ = 64 << 20
STAGE_THREADS = 4
STAGE_LIMIT = 8 << 30
# Positions a fixed text scores through the head at once (a full row of logits per position, in FP64).
SCORE_ROWS = 512
# What a forward that cannot resume a stored cache raises.
RESUME_ERRORS = (RuntimeError, NotImplementedError, TypeError, ValueError, AttributeError)
LAYER_ZERO = re.compile(r"(?:^|(?<=\.))layers\.0\.")


def chunked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None,
                      **kwargs):
    """Eager causal attention computed QUERY_BLOCK query rows at a time: FP32 scores, softmax over each whole
    row, never a full length x length matrix. Registered with transformers' AttentionInterface as CHUNKED (and
    sdpa's mask rules for its masks); causality and any sliding window come from positions, and a supplied mask
    (boolean, or additive as eager's) further restricts each block of rows. Query heads share their key/value head
    by broadcasting (transformers' repeat_kv without the copy), and a sliding-window layer reads only the columns
    its block can see."""
    import torch
    batch, heads, length, _ = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    query = query.view(batch, kv_heads, groups, length, query.shape[-1])
    key, value = key[:, :, None], value[:, :, None]
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    offset = key.shape[3] - length
    columns = torch.arange(key.shape[3], device=query.device)
    out = torch.empty(batch, kv_heads, groups, length, value.shape[-1], dtype=query.dtype, device=query.device)
    for start in range(0, length, QUERY_BLOCK):
        stop = min(start + QUERY_BLOCK, length)
        rows = torch.arange(start + offset, stop + offset, device=query.device)[:, None]
        width = stop + offset
        first = max(0, start + offset - sliding_window + 1) if sliding_window else 0
        scores = torch.matmul(query[:, :, :, start:stop], key[:, :, :, first:width].transpose(-2, -1)).mul_(scaling)
        allowed = columns[None, first:width] <= rows
        if sliding_window:
            allowed &= columns[None, first:width] > rows - sliding_window
        if attention_mask is not None:
            block = attention_mask[..., start:stop, first:width]
            block = (block.view(batch, kv_heads, groups, *block.shape[-2:]) if block.shape[1] == heads
                     else block[:, :, None])
            if block.dtype == torch.bool:
                allowed = allowed & block
            else:
                scores += block
        weights = torch.softmax(scores.masked_fill_(~allowed, float("-inf")), dim=-1, dtype=torch.float32)
        del scores
        out[:, :, :, start:stop] = torch.matmul(weights.to(query.dtype), value[:, :, :, first:width])
    return out.view(batch, heads, length, -1).transpose(1, 2).contiguous(), None


def _read_spans(reads, threads=None, random=True):
    """Fill each (path, offset, out) of `reads` (out a contiguous uint8 numpy array) with the file's bytes at offset,
    by `threads` readers (GATHER_THREADS), each reading its share in file order. random: the files are opened for
    random reads, so the kernel reads no pages around them (Linux; elsewhere the reads are plain)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    threads = threads or GATHER_THREADS
    by_file = {}
    for path, offset, out in reads:
        by_file.setdefault(path, []).append((offset, out.reshape(-1)))
    jobs = []
    for path, spans in by_file.items():
        spans.sort(key=lambda span: span[0])
        share = max(1, math.ceil(len(spans) / (4 * threads)))
        jobs += [(path, spans[i:i + share]) for i in range(0, len(spans), share)]

    def read(job):
        path, spans = job
        with open(path, "rb", buffering=0) as f:
            if random and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            for offset, out in spans:
                f.seek(offset)
                view, done = memoryview(out), 0
                while done < len(view):
                    count = f.readinto(view[done:])
                    if not count:
                        raise OSError(f"{path}: ends before byte {offset + len(view)}")
                    done += count
    if len(jobs) == 1:
        read(jobs[0])
    elif jobs:
        with ThreadPoolExecutor(min(threads, len(jobs))) as pool:
            list(pool.map(read, jobs))


def _view(flat, start, nbytes, dtype, shape):
    """Bytes [start, start + nbytes) of uint8 `flat` as a tensor of `dtype` and `shape`; copied first when they do
    not start at a multiple of the dtype's size, as a view of another dtype requires."""
    piece = flat[start:start + nbytes]
    if start % dtype.itemsize:
        piece = piece.clone()
    return piece.view(dtype).reshape(shape)


def _stored_tensor(checkpoint, key):
    """Whether `key` names a stored tensor, or (for a .weight name) a matrix stored in a quantized layout."""
    if checkpoint.has(key):
        return True
    return key.endswith(".weight") and any(checkpoint.has(key[: -len(".weight")] + s) for s in PRIMARY)


class TensorCheckpoint:
    """A checkpoint keyed by Hugging Face tensor names (has, raw, rows), which dense() decodes to FP32 with the scheme
    its tensors are stored in (crit_eval.schemes). quantization: the config's quantization_config (or MLX's
    quantization) as a dict."""
    where = {}
    quantization = {}

    def has(self, key):
        return key in self.where

    def rows(self, key, index):
        """Rows `index` (a 1-D int64 CPU tensor) of stored tensor `key`, as stored."""
        return self.raw(key)[index]

    def rows_many(self, requests):
        """rows(key, index) for each (key, index) of `requests`."""
        return [self.rows(key, index) for key, index in requests]

    def fit(self, key, tensor, shape):
        """The tensor as the module stores it: Hugging Face checkpoints keep the modeling code's shapes."""
        return tensor

    def prefix(self, names=()):
        """The prefix of the text model's tensors: of the candidates the checkpoint's decoder layer 0 tensors give
        (vision towers and draft layers have their own), the one under which most of `names` (the modeling code's
        parameter names, such as layers.0.mlp.gate.weight) are stored."""
        candidates = {key[: match.start()] for key in self.where for match in [LAYER_ZERO.search(key)] if match}
        names = list(names) or ["layers.0.input_layernorm.weight"]
        scored = sorted(((sum(_stored_tensor(self, p + n) for n in names), p) for p in candidates), reverse=True)
        if not scored or scored[0][0] == 0:
            raise KeyError("cannot find the decoder layer prefix in the checkpoint")
        return scored[0][1]

    def dense(self, base, device):
        """FP32 weight for module path `base` (without the .weight suffix)."""
        from .schemes import dense
        return dense(self, base, device)

    def embedding(self, base):
        """The embedding table for gathering rows: as stored when it is plain, else decoded to FP32 on the CPU."""
        from .schemes import scheme
        if scheme(self, base) is None:
            return self.raw(base + ".weight")
        return self.dense(base, "cpu")


class _Substituted(TensorCheckpoint):
    """A view of `checkpoint` in which the tensors of `replace` ({key: tensor}) stand for stored ones (gathered rows,
    or parts joined), so crit_eval.schemes decodes them with the checkpoint's other tensors."""

    def __init__(self, checkpoint, replace):
        self.checkpoint, self.replace, self.quantization = checkpoint, replace, checkpoint.quantization

    def has(self, key):
        return key in self.replace or self.checkpoint.has(key)

    def raw(self, key):
        return self.replace[key] if key in self.replace else self.checkpoint.raw(key)

    def header(self, key):
        if key not in self.replace:
            return self.checkpoint.header(key)
        tensor = self.replace[key]
        return SimpleNamespace(dtype=TORCH_DTYPES[str(tensor.dtype).removeprefix("torch.")], shape=tuple(tensor.shape))


class Checkpoint(TensorCheckpoint):
    """A safetensors checkpoint."""

    def __init__(self, model, quantization=None):
        from crit_core.model import inspect_model
        from safetensors import safe_open
        self.model = Path(model)
        self.quantization = dict(quantization or {})
        self.files = {}
        self.where = {}
        self.info = {}
        # The files and headers Core validates (crit_core.model.inspect_model) before torch reads any tensor.
        for shard in inspect_model(self.model).shards:
            handle = safe_open(str(shard.path), "pt", device="cpu")
            self.files[shard.path.name] = handle
            for key in handle.keys():
                self.where[key] = handle
            for tensor in shard.tensors:
                self.info[tensor.name] = tensor

    def _mapped(self, key, read):
        """read(bytes [rows, row bytes]) over a mapping of tensor `key`'s bytes, viewed as its dtype; None when the
        tensor is empty or of a dtype torch lacks. The file is mapped per call: the pages read stay in the page
        cache, not in this process, and no open mapping keeps the file from being moved or deleted (Windows)."""
        import numpy as np
        import torch
        tensor = self.info.get(key)
        name = next((name for name, short in TORCH_DTYPES.items() if short == getattr(tensor, "dtype", None)), None)
        if tensor is None or name is None or not tensor.nbytes:
            return None
        rows = tensor.shape[0] if tensor.shape else 1
        mapped = np.memmap(tensor.shard, dtype=np.uint8, mode="r", offset=tensor.data_begin,
                           shape=(rows, tensor.nbytes // rows))
        picked = torch.from_numpy(np.array(read(mapped)))
        del mapped
        return picked.view(getattr(torch, name))

    def raw(self, key):
        if key in self.placed:
            return self.placed[key]
        tensor = self._mapped(key, lambda data: data)
        if tensor is None:
            return self.where[key].get_tensor(key)
        return tensor.reshape(self.info[key].shape)

    placed = {}

    def stage(self, keys, buffer=None, pin=False):
        """Read stored tensors `keys` into one uint8 host buffer (`buffer` when it is large enough, else a new one,
        pinned when `pin`), in large reads in file order: (buffer, {key: (start, nbytes, dtype, shape)}). Each run
        of tensors keeps its offset modulo 64 from the file, so the tensors stay as aligned as they are stored.
        place() then puts them on a device for raw()."""
        import torch
        known = set(TORCH_DTYPES.values())
        keys = [key for key in keys if key in self.info and self.info[key].nbytes and self.info[key].dtype in known]
        runs = []
        for key in sorted(keys, key=lambda k: (str(self.info[k].shard), self.info[k].data_begin)):
            tensor = self.info[key]
            if runs and runs[-1][0] == tensor.shard and tensor.data_begin - runs[-1][2] <= STAGE_GAP:
                runs[-1][2] = max(runs[-1][2], tensor.data_begin + tensor.nbytes)
                runs[-1][3].append(key)
            else:
                runs.append([tensor.shard, tensor.data_begin, tensor.data_begin + tensor.nbytes, [key]])
        layout, starts, size = {}, [], 0
        for _, begin, end, members in runs:
            size += (begin - size) % 64
            starts.append(size)
            for key in members:
                tensor = self.info[key]
                name = next(name for name, short in TORCH_DTYPES.items() if short == tensor.dtype)
                layout[key] = (size + tensor.data_begin - begin, tensor.nbytes, getattr(torch, name), tuple(tensor.shape))
            size += end - begin
        if buffer is None or buffer.numel() < size:
            buffer = None
            try:
                buffer = torch.empty(size, dtype=torch.uint8, pin_memory=pin)
            except RuntimeError:   # no page-locked memory to be had: pageable
                buffer = torch.empty(size, dtype=torch.uint8)
        host = buffer.numpy()
        reads = [(shard, begin + offset, host[start + offset:start + min(end - begin, offset + STAGE_READ)])
                 for (shard, begin, end, _), start in zip(runs, starts)
                 for offset in range(0, end - begin, STAGE_READ)]
        _read_spans(reads, STAGE_THREADS, random=False)
        return buffer, layout

    def place(self, buffer, layout, device):
        """Make raw() read the staged tensors from `device`: the buffer copied there in one transfer (a copy on the
        CPU too, so the buffer can be read into again). Returns the placed copy."""
        size = max((start + nbytes for start, nbytes, _, _ in layout.values()), default=0)
        placed = buffer[:size].to(device, copy=True)
        self.placed = {key: _view(placed, *where) for key, where in layout.items()}
        return placed

    def rows(self, key, index):
        return self.rows_many([(key, index)])[0]

    def rows_many(self, requests):
        """Rows read from the files: each distinct row once, with a read of exactly its bytes (consecutive rows in
        one read), the reads of every request issued together by GATHER_THREADS readers. A lookup touches rows
        scattered over a table far larger than memory; faulting them in through a mapping reads one page at a time
        and around it, the readers keep the device's queue full and the file is read only where asked."""
        import numpy as np
        import torch
        plans, reads = [], []
        for key, index in requests:
            tensor = self.info.get(key)
            name = next((name for name, short in TORCH_DTYPES.items() if short == getattr(tensor, "dtype", None)), None)
            if tensor is None or name is None or not tensor.nbytes or not tensor.shape:
                plans.append(None)
                continue
            unique, inverse = np.unique(index.numpy(), return_inverse=True)
            width = tensor.nbytes // tensor.shape[0]
            out = np.empty((len(unique), width), dtype=np.uint8)
            cuts = np.flatnonzero(np.diff(unique) != 1) + 1
            for start, stop in zip([0, *cuts.tolist()], [*cuts.tolist(), len(unique)]):
                if stop > start:
                    reads.append((tensor.shard, tensor.data_begin + int(unique[start]) * width, out[start:stop]))
            plans.append((out, inverse, name, tensor.shape))
        _read_spans(reads)
        found = []
        for (key, index), plan in zip(requests, plans):
            if plan is None:
                found.append(super().rows(key, index))
                continue
            out, inverse, name, shape = plan
            if not len(index):
                found.append(torch.empty((0, *shape[1:]), dtype=getattr(torch, name)))
                continue
            picked = torch.from_numpy(out).view(getattr(torch, name))
            found.append(picked[torch.from_numpy(inverse.reshape(-1))].reshape(len(index), *shape[1:]))
        return found

    def header(self, key):
        part = self.where[key].get_slice(key)
        return SimpleNamespace(dtype=part.get_dtype(), shape=tuple(part.get_shape()))


class StateDictCheckpoint(TensorCheckpoint):
    """PyTorch state dicts (crit_core.model.TorchCheckpoint), memory-mapped by torch.load once Core has read
    their pickles without running them; weights_only keeps torch's own loader to tensors as well."""

    def __init__(self, weights, quantization=None):
        import torch
        self.quantization = dict(quantization or {})
        self.where = {}
        for shard in weights.shards:
            state = torch.load(shard.path, map_location="cpu", weights_only=True, mmap=True)
            for tensor in shard.tensors:
                self.where[tensor.name] = state[tensor.name]

    def raw(self, key):
        return self.where[key]

    def header(self, key):
        tensor = self.where[key]
        return SimpleNamespace(dtype=TORCH_DTYPES[str(tensor.dtype).removeprefix("torch.")], shape=tuple(tensor.shape))


class OnnxCheckpoint(TensorCheckpoint):
    """An ONNX graph's weights (crit_core.model.OnnxModel) under Hugging Face names, read from the bytes Core
    located; a weight stored as its transpose (a MatMul's [in, out] operand) is transposed back."""

    def __init__(self, weights, quantization=None):
        self.quantization = dict(quantization or {})
        self.where = {tensor.name: tensor for tensor in weights.tensors}

    def raw(self, key):
        import numpy as np
        import torch
        tensor = self.where[key]
        values = np.fromfile(tensor.shard, dtype=np.uint8, count=tensor.nbytes, offset=tensor.data_begin)
        dtype = getattr(torch, next(name for name, short in TORCH_DTYPES.items() if short == tensor.dtype))
        out = torch.from_numpy(values.copy()).view(dtype).reshape(tensor.shape)
        return out.T.contiguous() if tensor.transposed else out

    def header(self, key):
        tensor = self.where[key]
        shape = tuple(reversed(tensor.shape)) if tensor.transposed else tuple(tensor.shape)
        return SimpleNamespace(dtype=tensor.dtype, shape=shape, attributes=tensor.attributes)


def open_checkpoint(model, config):
    """The reference's checkpoint for the model (crit_core.model.open_model): its safetensors (Checkpoint), its
    PyTorch state dicts (StateDictCheckpoint) or its GGUF (GgufCheckpoint). config: transformers' text config."""
    from crit_core.model import open_model
    source = open_model(model)
    if source.format == "gguf":
        from .gguf_layout import GgufCheckpoint
        return GgufCheckpoint(source.weights, config)
    quantization = quantization_of(source.config())
    refuse_unread(quantization)
    if source.format == "torch":
        return StateDictCheckpoint(source.weights, quantization)
    if source.format == "onnx":
        return OnnxCheckpoint(source.weights, quantization)
    if source.format != "safetensors":
        raise SystemExit(f"{model}: no weight files the reference reads (safetensors, GGUF, PyTorch or ONNX)")
    return Checkpoint(source.path, quantization)


def _text_config(model):
    """transformers' own config object, which its modeling classes need, from the model's config.json (the
    folder's own, or the one a GGUF derives); it must first pass Core's rules (crit_core.model.ModelSource.config),
    as it did when the suite was prepared."""
    import json
    from crit_core.model import CONFIG_NAME, open_model
    from transformers import AutoConfig, CONFIG_MAPPING
    source = open_model(model)
    source.config()
    if CONFIG_NAME in source.derived:
        data = json.loads(source.files.read(CONFIG_NAME))
        if data.get("model_type") not in CONFIG_MAPPING:
            raise SystemExit(f"{model}: the config derived from the GGUF names no transformers model type "
                             f"({data.get('model_type')!r}); the reference needs config rules for its architecture "
                             "(crit_core.model.gguf_files) or the base model's config.json beside it")
        config = CONFIG_MAPPING[data["model_type"]].from_dict(data)
    else:
        config = AutoConfig.from_pretrained(str(source.folder or model))
    text = getattr(config, "text_config", None) or config
    text._attn_implementation = "eager"
    # The modeling code's own per-expert loop (FP32 matmuls), not a grouped GEMM kernel.
    text._experts_implementation = "eager"
    return text


@contextmanager
def _meta_parameters():
    """Build modules with their parameters on the meta device and their buffers real: the checkpoint supplies
    every parameter, while buffers a module computes (rotary frequencies, an embedding's scale) keep their values.
    torch.empty (how modules allocate parameters before initialising them) allocates nothing either, so a table
    larger than host memory can be declared; a buffer filled from torch.empty stays meta and fails loudly if used."""
    import torch
    from torch.overrides import TorchFunctionMode
    register = torch.nn.Module.register_parameter

    def meta(module, name, param):
        register(module, name, None if param is None else
                 torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad))

    class EmptyOnMeta(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is torch.empty and kwargs.get("device") is None:
                kwargs["device"] = "meta"
            return func(*args, **kwargs)
    torch.nn.Module.register_parameter = meta
    try:
        with EmptyOnMeta():
            yield
    finally:
        torch.nn.Module.register_parameter = register


_GATHERED = {}


def _gathering(module, rows):
    """Turn nn.Embedding `module` into one whose lookups call rows(index) -> FP32 rows instead of reading its
    (meta) weight. A subclass's own forward (an embedding scale) still runs: the gathering class sits between it and
    nn.Embedding in the method order."""
    import torch
    if "base" not in _GATHERED:
        class GatheredRows(torch.nn.Embedding):
            def forward(self, input):
                if self.max_norm is not None:
                    raise SchemeError("the reference does not gather rows of a max_norm embedding")
                found = self._crit_rows(input.reshape(-1).to("cpu", torch.int64))
                return found.reshape(*input.shape, found.shape[-1]).to(input.device)
        _GATHERED["base"] = GatheredRows
    base = _GATHERED["base"]
    cls = type(module)
    if not issubclass(cls, base):
        if cls is not torch.nn.Embedding and cls not in _GATHERED:
            _GATHERED[cls] = type(cls.__name__, (cls, base), {})
        module.__class__ = base if cls is torch.nn.Embedding else _GATHERED[cls]
    module._crit_rows = rows


class _Captured(Exception):
    """Raised by the last decoder layer's stand-in once a forward has handed every layer its arguments."""


class CacheUnsupported(Exception):
    """The model's forward or its cache cannot be resumed after a stored prompt; the reference scores full passes."""


def _encode(value, path, tensors):
    """A JSON description of cache state `value` (a cache or cache layer attribute at `path`), its tensors put in
    `tensors` under their paths. Tensors, plain values, dtypes, devices, and dicts, lists and tuples of them are
    stored; a class is kept as the fresh cache's own (checked by name). Anything else cannot be resumed."""
    import torch
    if isinstance(value, torch.Tensor):
        tensors[path] = value.detach().to("cpu").clone().contiguous()
        return {"tensor": path}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"value": value}
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"device": str(value)}
    if isinstance(value, dict) and all(isinstance(k, (int, str)) for k in value):
        return {"dict": [[k, _encode(v, f"{path}.{k}", tensors)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [_encode(v, f"{path}.{i}", tensors) for i, v in enumerate(value)]}
    if isinstance(value, type):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    raise CacheUnsupported(f"cache state {path} is a {type(value).__name__}, which the reference cannot store")


def _decode(spec, tensor, current=None):
    """The value `spec` describes, its tensors made by tensor(path); current: the attribute's present value."""
    import torch
    (kind, body), = spec.items()
    if kind == "tensor":
        return tensor(body)
    if kind == "value":
        return body
    if kind == "dtype":
        return getattr(torch, body)
    if kind == "device":
        return torch.device(body)
    if kind == "dict":
        return {k: _decode(v, tensor) for k, v in body}
    if kind in ("list", "tuple"):
        return (list if kind == "list" else tuple)(_decode(v, tensor) for v in body)
    if not isinstance(current, type) or f"{current.__module__}.{current.__qualname__}" != body:
        raise CacheUnsupported(f"the fresh cache holds another class than the stored {body}")
    return current


def _state(obj, path, skip=()):
    """(description, tensors) of every attribute of cache or cache layer `obj` but `skip`."""
    tensors = {}
    return {name: _encode(value, f"{path}{name}", tensors) for name, value in vars(obj).items()
            if name not in skip}, tensors


def _apply(obj, spec, tensor):
    for name, value in spec.items():
        setattr(obj, name, _decode(value, tensor, getattr(obj, name, None)))


class PromptState:
    """The reference's cache after a prompt (the transformers Cache its own forward created, one entry per decoder
    layer), stored in `folder` as one safetensors file per decoder layer and one for the cache's own attributes, in
    FP32 as the reference computed them, with index.json (identity, files and their sha256) written last. Between
    uses a layer's tensors are meta placeholders of their shapes: the forward still reads lengths from them, only the
    layer being run holds its tensors on the device."""

    def __init__(self, folder, identity, ids):
        self.folder, self.identity, self.ids = Path(folder), identity, ids
        self.cache, self.done, self.files = None, 0, {}

    def ready(self):
        index = self.folder / "index.json"
        if not index.is_file():
            return False
        stored = load_json(index)
        if stored.get("identity") != self.identity:
            return False
        self.files = stored["files"]
        return True

    def _file(self, name):
        return self.folder / name

    def _read(self, name):
        """(description, tensors) of a stored file, its bytes checked against the recorded sha256."""
        import hashlib
        import json
        from safetensors.torch import load
        data = self._file(name).read_bytes()
        if self.files.get(name) not in (None, hashlib.sha256(data).hexdigest()):
            raise SystemExit(f"{self._file(name)}: its bytes differ from the sha256 recorded in index.json")
        size = int.from_bytes(data[:8], "little")
        spec = json.loads(json.loads(data[8:8 + size])["__metadata__"]["state"])
        return spec, load(data)

    def _write(self, name, spec, tensors):
        import hashlib
        import json
        from safetensors.torch import save
        data = save(tensors, metadata={"state": json.dumps(spec)})
        self.folder.mkdir(parents=True, exist_ok=True)
        self._file(name).write_bytes(data)
        self.files[name] = hashlib.sha256(data).hexdigest()

    @staticmethod
    def _layer(index):
        return f"layer-{index:03d}.safetensors"

    def _placeholders(self, layer, spec, tensors):
        import torch
        _apply(layer, spec, lambda path: torch.empty(tensors[path].shape, dtype=tensors[path].dtype, device="meta"))

    def adopt(self, cache):
        """Take the cache a forward created (None when it created none)."""
        from transformers.cache_utils import Cache
        if not isinstance(cache, Cache):
            raise CacheUnsupported("the model's forward creates no transformers Cache")
        self.cache = cache

    def restore(self, cache, device):
        """Make `cache`, freshly created by the forward, this prompt's: its own attributes as stored, every layer's
        tensors placeholders until the layer runs."""
        self.adopt(cache)
        spec, tensors = self._read("cache.safetensors")
        _apply(cache, spec, lambda path: tensors[path].to(device))
        self._grow(len([n for n in self.files if n.startswith("layer-")]))
        for index in range(len(cache.layers)):
            name = self._layer(index)
            if name in self.files:
                header = self._header(name)
                self._placeholders(cache.layers[index], *header)

    def _header(self, name):
        """(description, {path: meta tensor}) of a stored layer from its header only."""
        import json
        import torch
        with self._file(name).open("rb") as f:
            size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(size))
        spec = json.loads(header.pop("__metadata__")["state"])
        names = {short: name for name, short in TORCH_DTYPES.items()}
        return spec, {path: torch.empty(info["shape"], dtype=getattr(torch, names[info["dtype"]]), device="meta")
                      for path, info in header.items()}

    def _grow(self, count):
        """Layers a cache without a config creates as they are first updated."""
        layers, make = self.cache.layers, getattr(self.cache, "layer_class_to_replicate", None)
        while len(layers) < count and make is not None:
            layers.append(make())
        if len(layers) < count:
            raise CacheUnsupported("the cache holds fewer layers than the model has decoder layers")

    def enter(self, index, device):
        """Before decoder layer `index` runs: its stored tensors (if any) on the device."""
        name = self._layer(index)
        if name in self.files:
            self._grow(index + 1)
            spec, tensors = self._read(name)
            _apply(self.cache.layers[index], spec, lambda path: tensors[path].to(device))

    def leave(self, index, keep):
        """After decoder layer `index` ran: stored back (keep) or dropped, placeholders in its place."""
        self._grow(index + 1)
        layer = self.cache.layers[index]
        if keep:
            spec, tensors = _state(layer, f"{index}.")
            self._write(self._layer(index), spec, tensors)
        else:
            spec, tensors = self._header(self._layer(index))
        self._placeholders(layer, spec, tensors)

    def seal(self):
        """Store the cache's own attributes and the index: the prompt's state is complete."""
        spec, tensors = _state(self.cache, "cache.", skip=("layers",))
        self._write("cache.safetensors", spec, tensors)
        write_json(self.folder / "index.json", {"identity": self.identity, "files": self.files})


class _Pass:
    """One forward over `ids` streamed through the decoder layers. state: the PromptState it resumes or extends
    (None: no cache); keep: whether the layers' new state is stored (a prompt chunk) or dropped (a reply)."""

    def __init__(self, ids, long, state=None, keep=False):
        self.ids, self.long, self.state, self.keep = ids, long, state, keep


def _hidden(args, kwargs):
    """(slot, hidden states) of a decoder layer call: the first positional argument, else hidden_states."""
    return (0, args[0]) if args else ("hidden_states", kwargs["hidden_states"])


class _OrderUnsafe(Exception):
    """The model's forward or a decoder layer read cache state that the layer-major order holds at another point
    than the chunk-major order does: the prompts run chunk by chunk instead."""


def _fingerprint(value):
    """What identifies a cache attribute and its contents: tensors by identity and version (an in-place change
    bumps it), containers by their items, plain values by value, anything else by identity."""
    import torch
    if isinstance(value, torch.Tensor):
        return ("tensor", id(value), value._version)
    if isinstance(value, dict):
        return ("dict", tuple((key, _fingerprint(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_fingerprint(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    return ("object", id(value))


class _Own:
    """A cache's own attributes (all but its layers) at one point: what each pass after it reads."""

    def __init__(self, cache):
        self.order = list(vars(cache))
        self.values = {key: value for key, value in vars(cache).items() if key != "layers"}
        self.mark = _fingerprint(self.values)

    def held_by(self, cache):
        return _fingerprint({key: value for key, value in vars(cache).items() if key != "layers"}) == self.mark

    def apply(self, cache):
        """Make them the cache's again, in their order; their objects must be as they were."""
        if _fingerprint(self.values) != self.mark:
            raise _OrderUnsafe("the cache's own state was changed in place")
        attributes = vars(cache)
        layers = attributes["layers"]
        attributes.clear()
        attributes.update((key, layers if key == "layers" else self.values[key]) for key in self.order)


@contextmanager
def _watched(cache):
    """The indices of the cache's layers whose state (their own attributes, read by any code, their methods' included)
    is read while the block runs; class attributes (is_sliding) are no state. The layers' classes read through a
    recording __getattribute__ for the block's duration (not a subclass: transformers registers cache layer
    subclasses by type)."""
    ids = {id(layer): index for index, layer in enumerate(cache.layers)}
    seen = set()
    classes = {type(layer) for layer in cache.layers}
    saved = {cls: cls.__dict__.get("__getattribute__") for cls in classes}
    reads = {cls: cls.__getattribute__ for cls in classes}

    def recording(read):
        def __getattribute__(self, name):
            index = ids.get(id(self))
            if index is not None and not name.startswith("__"):
                try:
                    own = name in object.__getattribute__(self, "__dict__")
                except AttributeError:   # a layer without a __dict__: every read counts
                    own = True
                if own:
                    seen.add(index)
            return read(self, name)
        return __getattribute__
    for cls in classes:
        cls.__getattribute__ = recording(reads[cls])
    try:
        yield seen
    finally:
        for cls, original in saved.items():
            if original is None:
                del cls.__getattribute__
            else:
                cls.__getattribute__ = original


class _Prompt:
    """A text's state and its chunks in the layer-major order: per decoder layer the chunks it ran (done), and
    the cache's own attributes after the forward handed each chunk to the layers (own)."""

    def __init__(self, state, long, count):
        self.state, self.long = state, long
        self.chunks = [state.ids[c:c + PROMPT_CHUNK] for c in range(0, len(state.ids), PROMPT_CHUNK)]
        self.done = [0] * count
        self.own = []


class _Job:
    """One pass of the layer-major order: chunk `chunk` (its ids) of `prompt`. calls: what the forward hands each
    layer; hidden: its hidden states before layer `at` (after the last, its output), on the host; waiting: the
    layers whose state its last capture read too early."""

    def __init__(self, prompt, ids, chunk):
        self.prompt, self.ids, self.chunk, self.long = prompt, ids, chunk, prompt.long
        self.calls = self.hidden = None
        self.at, self.waiting = 0, set()


class StreamingReference:
    def __init__(self, model, device="cuda"):
        import os
        # Chunked attention allocates a score block of another width for every block of query rows; torch's default
        # segments then fragment until the device is full (a 32K-token text on Gemma-4-31B), while expandable
        # segments reuse the memory. Read when the process first allocates on the device; arithmetic is unchanged.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModel
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.device = torch, device
        self.config = _text_config(model)
        self.checkpoint = open_checkpoint(model, self.config)
        with _meta_parameters():
            self.skeleton = AutoModel.from_config(self.config)
        self.names = dict(self.skeleton.state_dict(keep_vars=True))
        self.prefix = self.checkpoint.prefix(self.names)
        self.configs = list({id(c): c for m in self.skeleton.modules() for c in [getattr(m, "config", None)]
                             if c is not None and hasattr(c, "_attn_implementation")}.values())
        self.converted, self.members = self._conversions()
        self.tables = {}
        # Decoder layers' stored tensors by layer, and the read of the next layer under way (Checkpoint.stage).
        self.layer_keys, self.ahead, self.reader, self.host = None, None, None, None

    def _conversions(self):
        """({parameter name: (converter, {source pattern: [(base, stored key), ...]})}, {base: [stored keys]}): the
        checkpoint's matrices that transformers' conversion mapping for the model type turns into a parameter of
        another name or shape (only for checkpoints keyed by Hugging Face names; a GGUF is presented in the
        modeling code's layout already)."""
        if not isinstance(self.checkpoint, TensorCheckpoint):
            return {}, {}
        members = {}
        for key in self.checkpoint.where:
            members.setdefault(key.rsplit(".", 1)[0], []).append(key)
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming, dot_natural_key, rename_source_key
        transforms = get_model_conversion_mapping(self.skeleton)
        renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
        converters = [t for t in transforms if isinstance(t, WeightConverter)]
        found = {}
        if not converters:
            return found, members
        for key in self.checkpoint.where:
            suffix = next((s for s in PRIMARY if key.endswith(s)), None)
            if suffix is None:
                continue
            base = key[: -len(suffix)]
            renamed, pattern = rename_source_key(base + ".weight", renamings, converters,
                                                 self.skeleton.base_model_prefix, self.names)
            if pattern is None or renamed not in self.names:
                continue
            converter = next(c for c in converters if pattern in c.source_patterns)
            found.setdefault(renamed, (converter, {}))[1].setdefault(pattern, []).append((base, key))
        for _, parts in found.values():
            for pattern in parts:
                parts[pattern].sort(key=lambda part: dot_natural_key(part[1]))
        return found, members

    def _target(self, name):
        """The checkpoint base the converted parameter `name` would have if it were stored whole."""
        return self.prefix + (name[: -len(".weight")] if name.endswith(".weight") else name)

    def _convert(self, name, device):
        """Parameter `name` in FP32 from its sources through its converter's operations. Sources are decoded one by
        one, unless the checkpoint stores scales for the converted tensor itself: then they are its stored parts,
        joined before decoding with those scales."""
        from .schemes import dense
        converter, parts = self.converted[name]
        target = self._target(name)
        joined = target in self.members
        inputs = {pattern: [self.checkpoint.raw(key).to(device) if joined else self.checkpoint.dense(base, device)
                            for base, key in sources] for pattern, sources in parts.items()}
        for operation in converter.operations:
            inputs = operation.convert(inputs, source_patterns=converter.source_patterns,
                                       target_patterns=converter.target_patterns, full_layer_name=name,
                                       config=self.config)
        (tensor,) = inputs.values()
        if joined:
            suffix = next(key[len(base):] for base, key in next(iter(parts.values())))
            tensor = dense(_Substituted(self.checkpoint, {target + suffix: tensor}), target, device)
        return tensor.float()

    def _tensor(self, name, value):
        """Parameter or persistent buffer `name` (a modeling-code name) on the device: floating point in FP32, other
        buffers in their own dtype; None for a buffer the checkpoint does not store."""
        if name in self.converted:
            tensor = self._convert(name, self.device)
        else:
            key = self.prefix + name
            if name.rsplit(".", 1)[-1] == "weight":
                tensor = self.checkpoint.dense(key[: -len(".weight")], self.device)
            elif self.checkpoint.has(key) or isinstance(value, self.torch.nn.Parameter):
                tensor = self.checkpoint.raw(key).to(self.device)
                tensor = tensor.float() if tensor.is_floating_point() else tensor.to(value.dtype)
            else:
                return None
            tensor = self.checkpoint.fit(key, tensor, value.shape)
        return tensor

    def _row_keys(self, base, primary, full_rows):
        """The stored keys whose rows make rows of matrix `base` (stored key `primary`): it and its per-row scales
        (scales per tensor are read whole)."""
        keys = [primary]
        for key in self.members.get(base, ()):
            shape = tuple(self.checkpoint.header(key).shape)
            if key == primary or math.prod(shape) <= 1:
                continue
            if shape[0] != full_rows:
                raise SchemeError(f"{key}: scales not per row or per tensor; cannot gather rows of {base}")
            keys.append(key)
        return keys

    def _decode_rows(self, base, gathered):
        """FP32 rows of matrix `base` on the device from its gathered stored rows ({key: rows})."""
        from .schemes import dense
        return dense(_Substituted(self.checkpoint, gathered), base, self.device)

    def _gather_rows(self, base, primary, index, full_rows, rows=None):
        """FP32 rows `index` of matrix `base` (stored key `primary`), on the device: its stored rows (or `rows`,
        already gathered) decoded with its scales, per-row scales gathered alike."""
        keys = self._row_keys(base, primary, full_rows)
        wanted = keys if rows is None else keys[1:]
        gathered = dict(zip(wanted, self.checkpoint.rows_many([(key, index) for key in wanted])))
        if rows is not None:
            gathered[primary] = rows
        return self._decode_rows(base, gathered)

    def _primary(self, base):
        return next((base + s for s in PRIMARY if self.checkpoint.has(base + s)), None)

    def _gatherable(self, base, primary, rows=None):
        """Whether matrix `base` is stored as rows of the table (GATHERABLE schemes)."""
        from .schemes import scheme
        if primary is None:
            return False
        view = self.checkpoint if rows is None else _Substituted(self.checkpoint, {primary: rows})
        return scheme(view, base) in GATHERABLE

    def _row_source(self, name):
        """rows(index) -> FP32 rows of the embedding table parameter `name`, read without materializing the table
        where its storage allows (see the module docstring), else from the table decoded once on the CPU."""
        key = self.prefix + name
        base = key[: -len(".weight")]
        if isinstance(self.checkpoint, TensorCheckpoint):
            if name in self.converted:
                source = self._converted_rows(name)
                if source is not None:
                    return source
            else:
                primary = self._primary(base)
                if self._gatherable(base, primary):
                    full = self.checkpoint.header(primary).shape[0]
                    return lambda index: self._gather_rows(base, primary, index, full)

        def table(index):
            if name not in self.tables:
                self.tables[name] = (self._convert(name, "cpu") if name in self.converted
                                     else self.checkpoint.embedding(base))
            return self.tables[name][index].to(self.device).float()
        return table

    def _converted_rows(self, name):
        """A row source for a table its converter concatenates from parts along its rows, or None."""
        import torch
        from transformers.core_model_loading import Concatenate
        converter, parts = self.converted[name]
        operations = converter.operations
        if len(parts) != 1 or len(operations) != 1 or not isinstance(operations[0], Concatenate) or operations[0].dim:
            return None
        sources = next(iter(parts.values()))
        sizes = [self.checkpoint.header(key).shape[0] for _, key in sources]
        ends = torch.tensor(sizes).cumsum(0)
        target = self._target(name)
        if target in self.members:
            # Parts of one stored tensor, decoded together with the target's own scales.
            first = sources[0][1]
            suffix = first[len(sources[0][0]):]
            header = self.checkpoint.header(first)
            dtype = getattr(torch, next(name for name, short in TORCH_DTYPES.items() if short == header.dtype))
            meta = torch.empty((sum(sizes), *header.shape[1:]), device="meta", dtype=dtype)
            if not self._gatherable(target, target + suffix, meta):
                return None

            def joined(index):
                part = torch.searchsorted(ends, index, right=True)
                picks = [(part == j, j) for j in part.unique().tolist()]
                rows = None
                for (where, j), found in zip(picks, self.checkpoint.rows_many(
                        [(sources[j][1], index[where] - (ends[j] - sizes[j])) for where, j in picks])):
                    if rows is None:
                        rows = torch.empty((len(index), *found.shape[1:]), dtype=found.dtype)
                    rows[where] = found
                return self._gather_rows(target, target + suffix, index, sum(sizes), rows)
            return joined
        if not all(self._gatherable(base, key) for base, key in sources):
            return None

        def each(index):
            part = torch.searchsorted(ends, index, right=True)
            pieces = []
            for j in part.unique().tolist():
                where = part == j
                base, key = sources[j]
                pieces.append((where, base, self._row_keys(base, key, sizes[j]), index[where] - (ends[j] - sizes[j])))
            found = iter(self.checkpoint.rows_many([(key, local) for _, _, keys, local in pieces for key in keys]))
            out = None
            for where, base, keys, _ in pieces:
                rows = self._decode_rows(base, {key: next(found) for key in keys})
                if out is None:
                    out = torch.empty((len(index), *rows.shape[1:]), dtype=rows.dtype, device=rows.device)
                out[where.to(rows.device)] = rows
            return out
        return each

    def _gather_embeddings(self, module, path):
        """Make every embedding in `module` (at modeling-code path `path`) gather rows; returns their weight names
        relative to `module`, which stay meta."""
        names = set()
        for sub, child in module.named_modules():
            if isinstance(child, self.torch.nn.Embedding):
                weight = (sub + "." if sub else "") + "weight"
                _gathering(child, self._row_source(path + weight))
                names.add(weight)
        return names

    def _load(self, module, path):
        """Load `module` (modeling-code path `path`) from the checkpoint onto the device: parameters and the
        persistent buffers the checkpoint stores, weights dequantized to FP32; embeddings gather rows."""
        gathered = self._gather_embeddings(module, path)
        state = {}
        for name, value in module.state_dict(keep_vars=True).items():
            tensor = None if name in gathered else self._tensor(path + name, value)
            state[name] = value if tensor is None else tensor
        module.load_state_dict(state, strict=True, assign=True)
        self._place(module)
        self._config_dt_limit(module)
        return module

    def _place(self, module):
        """Move what `module` holds off the device (buffers it computed) onto it; meta tensors stay meta."""
        for child in module.modules():
            for group in (child._parameters, child._buffers):
                for key, tensor in group.items():
                    if tensor is not None and not tensor.is_meta and tensor.device != self.torch.device(self.device):
                        moved = tensor.to(self.device)
                        group[key] = (self.torch.nn.Parameter(moved, requires_grad=tensor.requires_grad)
                                      if isinstance(tensor, self.torch.nn.Parameter) else moved)

    def _load_outer(self):
        """Everything of the text model outside its decoder layers (embeddings, final norm or mixer, rotary
        frequencies)."""
        for name, child in self.skeleton.named_children():
            if child is not self.skeleton.layers:
                self._load(child, name + ".")
        for name, value in self.skeleton.state_dict(keep_vars=True).items():
            tensor = None if "." in name else self._tensor(name, value)
            if tensor is not None:
                setattr(self.skeleton, name, self.torch.nn.Parameter(tensor, requires_grad=False)
                        if isinstance(value, self.torch.nn.Parameter) else tensor)
        self._place(self.skeleton)

    def layer(self, index, upcoming=None):
        """Decoder layer `index` with its parameters and the persistent buffers the checkpoint stores (such as a
        layer scalar), weights dequantized to FP32, on the device. Its stored tensors arrive in one transfer, read
        ahead while the previous layer ran where the checkpoint stages reads (Checkpoint.stage); `upcoming`, the
        layer to load next, is read while this one runs."""
        placed = None
        if hasattr(self.checkpoint, "stage"):
            ahead, self.ahead = self.ahead, None
            staged = ahead[1].result() if ahead is not None else None   # the host buffer is shared: finish it
            if ahead is None or ahead[0] != index:
                staged = self._stage(index)
            if staged is not None:
                placed = self.checkpoint.place(*staged, self.device)
            if upcoming is not None:
                if self.reader is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self.reader = ThreadPoolExecutor(1)
                self.ahead = (upcoming, self.reader.submit(self._stage, upcoming))
        try:
            module = self._load(self.skeleton.layers[index], f"layers.{index}.")
        finally:
            if placed is not None:
                self.checkpoint.placed = {}
        if placed is not None:
            self._own(module, placed)
        return module

    def _stage(self, index):
        """(buffer, layout) of decoder layer `index`'s stored tensors read into the host buffer, or None when it has
        none to stage or more than STAGE_LIMIT bytes. Its embedding tables' tensors are left out: they are gathered
        by rows, never read whole."""
        if self.layer_keys is None:
            self.layer_keys = {}
            pattern = re.compile(re.escape(self.prefix) + r"layers\.(\d+)\.")
            for key in self.checkpoint.where:
                found = pattern.match(key)
                if found:
                    self.layer_keys.setdefault(int(found.group(1)), []).append(key)
        path = f"layers.{index}."
        tables = [path + sub for sub, child in self.skeleton.layers[index].named_modules()
                  if isinstance(child, self.torch.nn.Embedding)]
        skip = {key for name in tables if name + ".weight" in self.converted
                for sources in self.converted[name + ".weight"][1].values() for _, key in sources}
        skip.update(key for name in tables for key in self.members.get(self._target(name + ".weight"), ()))
        stems = tuple(self.prefix + name + "." for name in tables)
        keys = [key for key in self.layer_keys.get(index, ()) if key not in skip and not key.startswith(stems)]
        if not keys or sum(self.checkpoint.info[key].nbytes for key in keys) > STAGE_LIMIT:
            return None
        self.host, layout = self.checkpoint.stage(keys, self.host, pin=str(self.device).startswith("cuda"))
        return self.host, layout

    def _own(self, module, placed):
        """Give every tensor of `module` that is a view of the placed copy (a stored FP32 tensor as it is) its own
        memory, so the copy is freed once the layer is loaded."""
        storage = placed.untyped_storage().data_ptr()
        for child in module.modules():
            for group in (child._parameters, child._buffers):
                for key, tensor in group.items():
                    if tensor is not None and not tensor.is_meta and tensor.untyped_storage().data_ptr() == storage:
                        group[key] = (self.torch.nn.Parameter(tensor.clone(), requires_grad=tensor.requires_grad)
                                      if isinstance(tensor, self.torch.nn.Parameter) else tensor.clone())

    def _config_dt_limit(self, layer):
        """Mamba-2 mixers take the model's own dt limit. transformers sets some mixers' limit to
        (time_step_min, inf) although config.json declares time_step_limit (usually [0, inf]) and the model's own
        kernels (mamba_chunk_scan_combined) apply that; the clamp makes long-memory heads forget, which shows
        as retrieval failing beyond ~64K tokens and as drift against engines that follow the config."""
        limit = getattr(self.config, "time_step_limit", None)
        if not isinstance(limit, (list, tuple)) or len(limit) != 2:
            return
        lo, hi = (float(v) for v in limit)
        for module in layer.modules():
            current = getattr(module, "time_step_limit", None)
            if isinstance(current, tuple) and len(current) == 2 and tuple(map(float, current)) != (lo, hi):
                module.time_step_limit = (lo, hi)

    def _implementation(self, implementation):
        """Point the model's attention (and the masks its forward builds) at `implementation` ("eager" or CHUNKED)."""
        for config in self.configs:
            config._attn_implementation = implementation

    def _layer_count(self):
        return min(len(self.skeleton.layers), self.config.num_hidden_layers)

    def _forward(self, ids, stand_in, cache=None, use_cache=False):
        """Run the model's own forward on `ids` with decoder layer i replaced by stand_in(i) (its weights never
        read), after the prompt `cache` holds (use_cache: let the forward create one); returns the forward's output."""
        layers = self.skeleton.layers
        for i, layer in enumerate(layers):
            layer.forward = stand_in(i)
        try:
            return self.skeleton(input_ids=self.torch.tensor([ids], device=self.device), past_key_values=cache,
                                 use_cache=use_cache or cache is not None)
        finally:
            for layer in layers:
                layer.__dict__.pop("forward", None)

    def _capture(self, ids, cache=None, use_cache=False):
        """[(args, kwargs, slot)] per decoder layer: what the model's forward hands each layer for `ids`."""
        count = self._layer_count()
        calls, handed = [None] * count, []

        def stand_in(i):
            def forward(*args, **kwargs):
                slot, hidden = _hidden(args, kwargs)
                if i >= count or (i and (calls[i - 1] is None or hidden is not handed[-1])):
                    raise SchemeError("the model's forward does not hand each decoder layer the previous layer's "
                                      "output; the layer-streamed reference cannot drive it")
                calls[i] = (args, kwargs, slot)
                handed.append(hidden)
                if i == count - 1:
                    raise _Captured
                return hidden
            return forward
        try:
            self._forward(ids, stand_in, cache, use_cache)
        except _Captured:
            return calls
        raise SchemeError("the model's forward did not call every decoder layer")

    def _tail(self, ids, final):
        """The model's own forward after its decoder layers (final norm, mixer) on the last layer's output. It runs
        without a cache: the tail acts on each position's hidden state, and a cache must not be advanced twice."""
        last = self._layer_count() - 1
        out = self._forward(ids, lambda i: lambda *args, **kwargs: final if i == last else _hidden(args, kwargs)[1])
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def identity(self, model_sha):
        """What the prompt states and scores the reference stores depend on besides their ids."""
        import transformers
        device = self.torch.cuda.get_device_name(self.device) if str(self.device).startswith("cuda") else "cpu"
        return {"reference": REFERENCE_VERSION, "transformers": transformers.__version__,
                "torch": self.torch.__version__, "device": device, "model_sha256": model_sha}

    @staticmethod
    def _register():
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        AttentionInterface.register(CHUNKED, chunked_attention)
        AttentionMaskInterface.register(CHUNKED, sdpa_mask)

    @staticmethod
    def _cache_of(calls):
        """The cache the forward hands its decoder layers (None without one)."""
        from transformers.cache_utils import Cache
        args, kwargs, _ = calls[0]
        return next((value for value in (*args, *kwargs.values()) if isinstance(value, Cache)), None)

    def _resume(self, ids, state):
        """_capture for `ids` after the prompt `state` holds."""
        if state.cache is None and not state.files:
            calls = self._capture(ids, use_cache=True)   # a prompt's first chunk: the forward makes the cache
            state.adopt(self._cache_of(calls))
            return calls
        if state.cache is None:   # a stored prompt: a fresh cache from the forward, made the prompt's
            state.restore(self._cache_of(self._capture(ids[:1], use_cache=True)), self.device)
        return self._capture(ids, state.cache)

    def _stream(self, passes, log, check, label=""):
        """Run `passes` through the decoder layers, each layer loaded once: first capture what the forward hands
        every layer for each pass, then layer by layer run every pass, its cache layer (a pass with a PromptState)
        on the device only while it runs. Returns each pass's last layer output."""
        torch = self.torch
        count = self._layer_count()
        records, states = [], []
        for p in passes:
            check()
            self._implementation(CHUNKED if p.long else "eager")
            state = p.state
            if state is None:
                calls = self._capture(p.ids)
            else:
                try:
                    calls = self._resume(p.ids, state)
                except RESUME_ERRORS as exc:
                    raise CacheUnsupported(f"the model's forward cannot resume its cache ({exc})") from exc
            states.append(_hidden(*calls[0][:2])[1])
            records.append(calls)
        for index in range(count):
            started = time.time()
            layer = self.layer(index, index + 1 if index + 1 < count else None)
            loaded = time.time() - started
            for s, (p, calls) in enumerate(zip(passes, records)):
                check()
                self._implementation(CHUNKED if p.long else "eager")
                args, kwargs, slot = calls[index]
                if slot == 0:
                    args = (states[s],) + tuple(args[1:])
                else:
                    kwargs = dict(kwargs, hidden_states=states[s])
                if p.state is not None:
                    p.state.enter(index, self.device)
                states[s] = layer(*args, **kwargs)
                if p.state is not None:
                    p.state.leave(index, p.keep)
                calls[index] = None
            layer.to("meta")
            del layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"{label}layer {index + 1}/{count} {time.time() - started:.1f}s (load {loaded:.1f}s)")
        return states

    def _capture_job(self, job):
        """Capture what the forward hands each layer for prompt chunk `job` when the cache state its forward reads
        is the state the chunk-major order hands it: every layer it reads having run exactly the chunks before it.
        Returns whether it did; if not, job.waiting names the layers read too early, and nothing is changed."""
        prompt = job.prompt
        self._implementation(CHUNKED if job.long else "eager")
        if job.chunk == 0:
            try:
                calls = self._capture(job.ids, use_cache=True)   # the forward makes the cache
                prompt.state.adopt(self._cache_of(calls))
            except RESUME_ERRORS as exc:
                raise CacheUnsupported(f"the model's forward cannot resume its cache ({exc})") from exc
        else:
            cache, before = prompt.state.cache, prompt.own[job.chunk - 1]
            before.apply(cache)
            calls, seen = None, set()
            try:
                with _watched(cache) as seen:
                    calls = self._capture(job.ids, cache)
            except RESUME_ERRORS as exc:
                if all(prompt.done[j] == job.chunk for j in seen):
                    raise CacheUnsupported(f"the model's forward cannot resume its cache ({exc})") from exc
            job.waiting = {j for j in seen if prompt.done[j] != job.chunk}
            if calls is None or job.waiting:
                before.apply(cache)
                return False
        prompt.own.append(_Own(prompt.state.cache))
        # The job carries its hidden states on the host; the calls (every layer's handed the same tensor) do not.
        job.hidden = _hidden(*calls[0][:2])[1].to("cpu")
        job.calls = [((None,) + tuple(args[1:]), kwargs, slot) if slot == 0 else
                     (args, dict(kwargs, hidden_states=None), slot) for args, kwargs, slot in calls]
        return True

    @staticmethod
    def _runnable(job):
        """Whether prompt chunk `job` can run its next layer: its prompt has run that layer on the chunks before it."""
        return job.calls is not None and job.at < len(job.calls) and job.prompt.done[job.at] == job.chunk

    def _run_job(self, job, layer):
        """Run decoder layer job.at on prompt chunk `job`, with its cache as the chunk-major order holds it then: the
        cache's own attributes as the forward left them for this chunk, and every other layer whose state the layer
        reads as far as that order has run it (raises _OrderUnsafe otherwise)."""
        prompt, index = job.prompt, job.at
        self._implementation(CHUNKED if job.long else "eager")
        args, kwargs, slot = job.calls[index]
        hidden = job.hidden.to(self.device)
        if slot == 0:
            args = (hidden,) + tuple(args[1:])
        else:
            kwargs = dict(kwargs, hidden_states=hidden)
        state, own = prompt.state, prompt.own[job.chunk]
        own.apply(state.cache)
        state.enter(index, self.device)
        with _watched(state.cache) as seen:
            out = layer(*args, **kwargs)
        if not own.held_by(state.cache):
            raise _OrderUnsafe("a decoder layer changes the cache's own state")
        early = sorted(j for j in seen if j != index and prompt.done[j] != job.chunk + (j < index))
        if early:
            raise _OrderUnsafe(f"decoder layer {index} reads the cache of layer {early[0]}")
        state.leave(index, True)
        prompt.done[index] += 1
        job.calls[index] = None
        job.at += 1
        job.hidden = out.to("cpu")

    def _layer_major(self, prompts, jobs, log, check):
        """Run every prompt chunk (jobs, in order per prompt) through the decoder layers, loading each layer when its
        turn comes and running every chunk it can on it before the next: always the lowest layer that has one. A
        chunk is captured once the layers whose state its forward reads have run the chunks before it (usually the
        first layer of each kind, so only the layers up to those run chunk by chunk), and retried as those layers
        advance. Returns the number of layer loads."""
        torch = self.torch
        count = self._layer_count()

        def capture(prompt):
            """Capture the prompt's next chunk once the layers its last attempt read too early have caught up."""
            for job in jobs:
                if job.prompt is not prompt or job.calls is not None or job.chunk != len(prompt.own):
                    continue
                if any(prompt.done[j] != job.chunk for j in job.waiting):
                    return
                check()
                if not self._capture_job(job):
                    return
        for prompt in prompts:
            capture(prompt)
        loads = 0
        while True:
            waiting = [job.at for job in jobs if self._runnable(job)]
            if not waiting:
                if all(job.calls is not None and job.at == count for job in jobs):
                    return loads
                raise _OrderUnsafe("no chunk can run: the forward reads the cache of layers that cannot run yet")
            index, started = min(waiting), time.time()
            layer = self.layer(index, index + 1 if index + 1 < count else None)
            loaded, ran, loads = time.time() - started, 0, loads + 1
            while True:
                ready = [job for job in jobs if job.at == index and self._runnable(job)]
                if not ready:
                    break
                for job in ready:
                    if job.at == index and self._runnable(job):
                        check()
                        self._run_job(job, layer)
                        ran += 1
                        capture(job.prompt)
            layer.to("meta")
            del layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"layer-major: layer {index + 1}/{count} {time.time() - started:.1f}s (load {loaded:.1f}s), "
                f"{ran} passes")

    def score_texts(self, texts, top_k, valid_vocab, scratch, log=print, long=(), check=lambda: None):
        """Every position of each text (a list of ids) teacher-forced: per text (target log-probabilities of ids[1:]
        [n - 1] float64, the top_k ids [n - 1, top_k] int64 and their log-probabilities [n - 1, top_k] float64).
        Texts run in PROMPT_CHUNK-token chunks that resume the cache, all of them layer-major (_layer_major); the
        chunks' cache states are scratch files in `scratch`, removed after. Where the layer-major order is unsafe
        the texts run chunk-major, and a model whose cache cannot be resumed runs each text whole. The texts whose
        indices are in `long` use chunked_attention."""
        import shutil
        torch = self.torch
        self._register()
        scratch, count = Path(scratch), self._layer_count()
        chunks = [[ids[c:c + PROMPT_CHUNK] for c in range(0, len(ids), PROMPT_CHUNK)] for ids in texts]
        outputs, unsafe = None, None
        with torch.no_grad():
            self._load_outer()
            try:
                prompts, jobs = [], []
                for t, ids in enumerate(texts):
                    prompts.append(_Prompt(PromptState(scratch / f"text-{t:03d}", {}, list(ids)), t in long, count))
                    jobs += [_Job(prompts[-1], piece, c) for c, piece in enumerate(prompts[-1].chunks)]
                try:
                    self._layer_major(prompts, jobs, log, check)
                    outputs = [[job.hidden for job in jobs if job.prompt is prompt] for prompt in prompts]
                except _OrderUnsafe as exc:
                    unsafe = str(exc)   # (outside the handler, so the traceback frees the passes' tensors)
                del prompts, jobs
                if unsafe is not None:
                    log(f"reference: {unsafe}; the texts run chunk by chunk")
                    shutil.rmtree(scratch, ignore_errors=True)
                    outputs = self._text_rounds(texts, chunks, long, scratch, log, check)
            except CacheUnsupported as exc:
                log(f"reference: the cache cannot be resumed ({exc}); each text runs whole")
                outputs = self._stream([_Pass(ids, t in long) for t, ids in enumerate(texts)], log, check, "texts: ")
                outputs, chunks = [[out] for out in outputs], [[ids] for ids in texts]
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            head, cap = self._head(valid_vocab)
            return [self._text_rows(ids, pieces, hidden, head, cap, top_k, t in long)
                    for t, (ids, pieces, hidden) in enumerate(zip(texts, chunks, outputs))]

    def _text_rounds(self, texts, chunks, long, scratch, log, check):
        """score_texts' chunk-major order: round c runs every text's chunk c through all layers. Returns per text its
        chunks' last layer outputs."""
        states = [PromptState(scratch / f"text-{t:03d}", {}, list(ids)) for t, ids in enumerate(texts)]
        outputs = [[] for _ in texts]
        for c in range(max(map(len, chunks), default=0)):
            todo = [t for t in range(len(texts)) if c < len(chunks[t])]
            passes = [_Pass(chunks[t][c], t in long, states[t], keep=True) for t in todo]
            for t, out in zip(todo, self._stream(passes, log, check, f"texts {c + 1}/{max(map(len, chunks))}: ")):
                outputs[t].append(out.to("cpu"))
        return outputs

    def _head(self, valid_vocab):
        """(the output head [valid_vocab, hidden] in FP32 on the device, the final logit soft-cap or None)."""
        self.tables.clear()
        tied = not self.checkpoint.has("lm_head.weight") and getattr(self.config, "tie_word_embeddings", False)
        embedding = next(name for name, module in self.skeleton.named_modules()
                         if module is self.skeleton.get_input_embeddings())
        head = self.checkpoint.dense(self.prefix + embedding if tied else "lm_head", self.device)[:valid_vocab]
        return head, getattr(self.config, "final_logit_softcapping", None)

    def _text_rows(self, ids, pieces, outputs, head, cap, top_k, long):
        """score_texts' rows of one text from its pieces' (consecutive chunks of ids) last layer outputs, through the
        model's tail and head, SCORE_ROWS positions at a time."""
        torch = self.torch
        targets, tops, values = [], [], []
        start = 0
        for piece, out in zip(pieces, outputs):
            self._implementation(CHUNKED if long else "eager")
            hidden = self._tail(piece, out.to(self.device))[0]
            following = torch.tensor(ids[start + 1:start + len(piece) + 1], device=self.device)
            for row in range(0, len(following), SCORE_ROWS):
                logits = hidden[row:row + SCORE_ROWS][:len(following) - row] @ head.T
                if cap:
                    logits = torch.tanh(logits / cap) * cap
                logprobs = torch.log_softmax(logits.double(), dim=-1)
                block = following[row:row + SCORE_ROWS]
                targets.append(logprobs.gather(1, block[:, None])[:, 0].cpu())
                top = logprobs.topk(top_k, dim=-1)
                tops.append(top.indices.cpu())
                values.append(top.values.cpu())
            start += len(piece)
        return torch.cat(targets), torch.cat(tops), torch.cat(values)


def store_texts(reference, texts, top_k, valid_vocab, data, model_sha, log=print, check=lambda: None):
    """Score every fixed text once per identity and store the scores under prepared data folder `data`
    (crit_eval.fixed: each position's target log-probability, top_k ids and their log-probabilities, with the identity:
    the reference's, top_k, valid_vocab and every text's ids). texts: {text id: (ids, long)}. Returns how they were
    obtained: "stored" when the data holds them under the same identity, else "scored" with its seconds and bytes."""
    import shutil
    from . import fixed
    identity = dict(reference.identity(model_sha), **fixed.identity(texts, top_k, valid_vocab))
    positions = sum(len(ids) - 1 for ids, _ in texts.values())
    if fixed.load(data, identity) is not None:
        return {"method": "stored", "positions": positions}
    shutil.rmtree(Path(data) / fixed.STORE, ignore_errors=True)
    names = list(texts)
    started = time.time()
    rows = reference.score_texts([texts[name][0] for name in names], top_k, valid_vocab,
                                 Path(data) / fixed.STORE / "scratch", log=log,
                                 long={i for i, name in enumerate(names) if texts[name][1]}, check=check)
    size = fixed.write(data, identity, {name: (target.tolist(), top.tolist(), values.tolist())
                                        for name, (target, top, values) in zip(names, rows)})
    return {"method": "scored", "seconds": round(time.time() - started, 1), "positions": positions, "bytes": size}


def score_data(model, data, log=print, check=lambda: None):
    """Score every quality text of the prepared data at every position on the checkpoint it was prepared from and
    store the scores under the data (crit_eval.fixed), unless they are stored already under the same identity.
    check() runs between layers and texts: it raises to stop. Returns how they were obtained."""
    from . import fixed
    protocol = load_json(EVAL_DIR / "protocol.json")
    prepared = load_json(Path(data) / PREPARED)
    if model_identity(model)["sha256"] != prepared["model"]["sha256"]:
        raise SystemExit(f"{model} is not the checkpoint {data} was prepared from: the reference is that checkpoint")
    return store_texts(StreamingReference(model), fixed.texts(prepared), protocol["quality"]["top_k"],
                       prepared["valid_vocab"], data, prepared["model"]["sha256"], log=log, check=check)
