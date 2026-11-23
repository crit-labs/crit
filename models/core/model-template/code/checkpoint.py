"""The checkpoint a composed model reads: its safetensors headers, its tensors read one at a time onto a device, and
its matrices as the projections and experts slots load them (stored as they are: bf16, fp8 per channel or tensor,
mxfp8, nvfp4 with its scales decoded where the checkpoint compresses them), several matrices that read the same input
stacked by output rows into one. A matrix only a drafter reads may be stored in a format the slots do not take (FP8 in
blocks, as an NVFP4 checkpoint keeps its MTP layer's experts): it is re-encoded at load (crit_plugins.encodings), and
the stored target verifies every draft."""
import json
import os
import re
from pathlib import Path


BLOCK = 4096        # hold() reads whole blocks, as a direct read asks
PIECE = 32 << 20    # in pieces of this many bytes, several at once
# safetensors' dtype names as torch's
DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn",
          "F8_E5M2": "float8_e5m2", "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
          "U16": "uint16", "U32": "uint32", "U64": "uint64", "BOOL": "bool"}


def _uncache(path):
    """Asks the OS to drop its page cache's copy of the file at `path` (the pages no mapping uses), where it takes
    such advice: 63 GiB of a checkpoint's cached shards went in 7 s, and memory filled beside them four times
    faster after."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


class Checkpoint:
    """The checkpoint's safetensors headers ({name: TensorInfo}) and its tensors read one at a time onto a device:
    the shards its index names (model.safetensors.index.json: wherever it keeps them, a tensors/ subfolder too) and
    the safetensors files beside it that the index does not list (a draft layer stored apart); without an index,
    every safetensors file of the folder. `names`: the layout's rules for a checkpoint that names its tensors its own
    way, [[pattern, replacement], ...] regular expressions applied in order to every name (re.sub), so the blocks
    read it under the names they know; each tensor keeps its bytes where they are. A rule's third entry shifts the
    number its first group matched (a layer's tensor the checkpoint keeps under the number before)."""

    def __init__(self, model, names=None):
        from safetensors import safe_open
        from crit_core.model.safetensors import INDEX_NAME, inspect_model, inspect_safetensors
        self.folder = model = Path(model)
        self.headers, self.files = {}, {}
        self._held, self._buffer, self._reads = {}, None, {}   # hold(): {name: its offset in the buffer}, shard reads
        listed = set()
        if (model / INDEX_NAME).is_file():
            found = inspect_model(model / INDEX_NAME, header_only=True)
            self.headers.update(found.tensors)
            listed = {shard.path for shard in found.shards}
        for path in sorted(model.absolute().glob("*.safetensors")):
            if path in listed:
                continue
            for tensor in inspect_safetensors(path, header_only=True).tensors:
                if tensor.name in self.headers:
                    raise ValueError(f"model-template: {tensor.name} is in {self.headers[tensor.name].shard.name} "
                                     f"and in {path.name}")
                self.headers[tensor.name] = tensor
        if not self.headers:
            raise ValueError(f"model-template: no safetensors files in {model}")
        self.stored_names = {}   # the names the rules changed: {the blocks' name: the checkpoint's}
        if names:
            def rule(entry):
                pattern, replacement = re.compile(entry[0]), entry[1]
                if len(entry) < 3:
                    return pattern, replacement
                shift = int(entry[2])
                return pattern, lambda m: m.expand(replacement.replace("\\1", str(int(m.group(1)) + shift)))
            rules = [rule(entry) for entry in names]
            renamed = {}
            for name, info in self.headers.items():
                new = name
                for pattern, replacement in rules:
                    new = pattern.sub(replacement, new)
                if new in renamed:
                    raise ValueError(f"model-template: the layout's names give {self.stored_names.get(new, new)} and "
                                     f"{name} one name, {new}")
                renamed[new] = info
                if new != name:
                    self.stored_names[new] = name
            self.headers = renamed
        for path in sorted({str(info.shard) for info in self.headers.values()}):
            self.files[path] = safe_open(path, "pt", device="cpu")

    def __contains__(self, name):
        return name in self.headers

    def names(self):
        return self.headers.keys()

    def nbytes(self, name):
        return self.headers[name].nbytes

    def shape(self, name):
        return tuple(self.headers[name].shape)

    def get(self, name, device):
        info = self.headers[name]
        at = self._held.get(name)
        if at is not None:   # held (hold()): its bytes copied out of the checkpoint's memory, which the next hold reuses
            import torch
            dtype = getattr(torch, DTYPES[info.dtype])
            flat = torch.frombuffer(self._buffer, dtype=torch.uint8, count=info.nbytes, offset=at)
            return flat.clone().view(dtype).reshape(tuple(info.shape)).to(device)
        return self.files[str(info.shard)].get_tensor(self.stored_names.get(name, name)).to(device)

    def hold(self, prefix):
        """Reads the bytes of every tensor named `prefix`... into memory of the checkpoint's own, past the OS's page
        cache (Core's host reads: direct, at a queue depth), and serves get() of those names from it until the next
        hold() or release(). For tensors read once on their way elsewhere (a layer's routed experts, packed into an
        experts plugin's store): through the shards' mappings a cold layer comes off the drive a page a fault, and
        its pages stay in the page cache, where a store that fills host memory then has to push them out page by
        page (65 GiB of experts filled their store at 130 MB/s beside them). For the same reason the page cache's
        copy of a shard is dropped at the shard's first hold (whatever read it before: a hash of the model's files):
        the bytes are read past it from now on. Returns the bytes held; 0, and the mappings serve as before, where
        no tensor has the prefix."""
        import mmap
        from crit_plugins.host import current
        self.release()
        spans, held = {}, 0
        for name, info in self.headers.items():
            if name.startswith(prefix) and info.nbytes:
                spans.setdefault(str(info.shard), []).append((info.data_begin, info.data_end, name))
                held += info.nbytes
        if not spans:
            return 0
        plan, size = [], 0   # (shard, file offset, bytes, buffer offset) of each merged range, whole blocks
        for path, found in sorted(spans.items()):
            limit, merged = Path(path).stat().st_size, []
            for first, end, name in sorted(found):
                first, end = first // BLOCK * BLOCK, min(limit, -(-end // BLOCK) * BLOCK)
                if merged and first <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([first, end])
            at = 0
            for first, end, name in sorted(found):
                while merged[at][1] < end:
                    at += 1
                # the buffer offset of merged range `at` is known once the plan has it: its place in this shard's
                self._held[name] = (path, at, first - merged[at][0])
            for index, (first, end) in enumerate(merged):
                plan.append((path, index, first, end - first, size))
                size += -(-(end - first) // BLOCK) * BLOCK
        if self._buffer is None or len(self._buffer) < size:
            self._buffer = mmap.mmap(-1, size)
        view, where = memoryview(self._buffer), {}
        for path, index, first, nbytes, at in plan:
            where[path, index] = at
            if path not in self._reads:
                self._reads[path] = current().reads("model-template", path, depth=8, direct=True)
                _uncache(path)
        for path in sorted({path for path, *_ in plan}):   # a shard's ranges together, in pieces a reader takes whole
            self._reads[path].read([(first + done, view[at + done:at + done + min(PIECE, nbytes - done)])
                                    for shard, _, first, nbytes, at in plan if shard == path
                                    for done in range(0, nbytes, PIECE)])
        view.release()
        self._held = {name: where[path, index] + inside for name, (path, index, inside) in self._held.items()}
        return held

    def release(self):
        """Ends hold(): get() reads the shards' mappings again. The memory stays for the next hold()."""
        self._held = {}

    def close(self):
        self.release()
        for reads in self._reads.values():
            reads.close()
        self._reads, self._buffer = {}, None
        self.files.clear()


def open_checkpoint(model, config, layout=None):
    """(checkpoint, its Matrices class): the safetensors files (renamed by the layout's `names`), or a GGUF read under
    the same names by the entry the layout's `gguf` gives its architecture (gguf.py)."""
    from .gguf import GgufCheckpoint, GgufMatrices, is_gguf
    layout = layout or {}
    if is_gguf(model):
        return GgufCheckpoint(model, config, layout), GgufMatrices
    return Checkpoint(model, layout.get("names")), Matrices


def stop_tokens(model, config):
    """The checkpoint's end tokens: generation_config.json's eos_token_id (a GGUF's as Core derives it), else
    config.json's."""
    from .gguf import generation_ends, is_gguf
    model = Path(model)
    generation = model / "generation_config.json"
    eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id") if generation.is_file() else None
    if eos is None and is_gguf(model):
        eos = generation_ends(model)
    if eos is None:
        eos = (config.get("text_config") or {}).get("eos_token_id", config.get("eos_token_id"))
    return sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []


class Matrices:
    """A checkpoint's matrices as the projections slot's load() and the experts slot's load() take them."""

    def __init__(self, checkpoint, projections, quantization, device):
        from crit_core.model.schemes import matrix_parts
        import torch
        self.ck, self.proj, self.quant, self.device, self.torch = checkpoint, projections, quantization or {}, device, torch
        self.parts = matrix_parts

    def has(self, base):
        return self.parts(self.ck.headers, base, self.quant) is not None

    def _mx(self, p):
        """Whether an FP8 matrix with block scales is MXFP8: one E8M0 exponent byte per 32 inputs of each row."""
        return (p.scale_granularity == "block" and tuple(p.block or ()) == (1, 32)
                and self.ck.headers[p.scales].dtype == "U8")

    def format(self, base):
        """The format name `base` loads as (bf16, fp8:channel, fp8:tensor, mxfp8, nvfp4), read from the headers
        alone."""
        p = self.parts(self.ck.headers, base, self.quant)
        if p is None:
            raise KeyError(f"model-template: no matrix {base} in the checkpoint")
        if p.scheme == "plain":
            return "bf16"
        if p.scheme == "fp8":
            return {"row": "fp8:channel", "tensor": "fp8:tensor"}.get(p.scale_granularity,
                                                                         "mxfp8" if self._mx(p) else "fp8:block")
        return p.scheme

    def stored(self, base, device=None, index=None):
        """(format, tensors) of matrix `base` as the projections slot's load takes them, on `device` (the model's
        by default); `index`: of a stack of matrices stored as one tensor ([experts][out][in]), matrix `index`."""
        torch, ck = self.torch, self.ck
        p = self.parts(ck.headers, base, self.quant)
        if p is None:
            raise KeyError(f"model-template: no matrix {base} in the checkpoint")
        device = self.device if device is None else device
        if index is None:
            get = lambda name: ck.get(name, device)  # noqa: E731
        else:
            stack = p.shape[:-2]
            if len(stack) != 1 or not 0 <= index < stack[0]:
                raise ValueError(f"model-template: {base} is no stack of matrices with a matrix {index} "
                                 f"(shape {list(p.shape)})")

            def get(name):   # the stack's tensor of matrix `index`, or a tensor the stack shares
                cache = self.__dict__.setdefault("_stacks", {})   # a layer's stacks, read once for all its matrices
                if name not in cache:
                    if len(cache) >= 8:
                        cache.clear()
                    cache[name] = ck.get(name, "cpu")
                found = cache[name]
                return (found[index] if found.dim() and found.shape[0] == stack[0] else found).contiguous().to(device)
        if p.scheme == "plain":
            return "bf16", {"weight": get(p.codes).to(torch.bfloat16)}
        if p.scheme == "fp8":
            if p.scale_granularity == "row":
                return "fp8:channel", {"weight": get(p.codes), "scale": get(p.scales).float().reshape(-1)}
            if p.scale_granularity == "tensor":
                return "fp8:tensor", {"weight": get(p.codes), "scale": get(p.scales).float().reshape(())}
            if self._mx(p):
                return "mxfp8", {"weight": get(p.codes), "scale": get(p.scales)}
            # FP8 in blocks (a multiplier per b0 x b1 block): the projections slot loads it as stored (fp8-block-rows)
            return "fp8:block", {"weight": get(p.codes), "scale": get(p.scales).float(), "block": tuple(p.block)}
        g = get(p.global_scale).float().reshape(())
        if p.global_denominator is not None:   # weight = code * scale * amax / denominator: the slot's divisor
            g = get(p.global_denominator).float().reshape(()) / g
        elif not p.global_divides:
            g = 1.0 / g
        if p.scale_encoding == "csf":   # ModelOpt's compressed scales: E4M3 bytes again, as an uncompressed one stores
            from crit_core.model.schemes import CSF_EXCEPTIONS
            from crit_plugins.device import csf_decode
            words = base + CSF_EXCEPTIONS
            scales = csf_decode(get(p.scales), get(words) if words in ck else None).view(torch.float8_e4m3fn)
        else:
            scales = get(p.scales)
        found = {"weight": get(p.codes), "scale": scales, "global_scale": g}
        if p.input_scale is not None:
            i = get(p.input_scale).float().reshape(())
            found["input_scale"] = i if p.input_divides else 1.0 / i
        return "nvfp4", found

    def reformat(self, base, want):
        """(format, tensors) of matrix `base` in format `want`: as stored where it is stored so, else its exact
        values re-encoded (bf16, nvfp4 without input scales, fp8:channel), for a matrix only a drafter reads."""
        from crit_plugins import encodings
        fmt, tensors = self.stored(base)
        if fmt == want:
            return fmt, tensors
        if fmt not in encodings.SOURCES:
            raise ValueError(f"model-template: {base} is stored {fmt}, which does not re-encode to {want}")
        if want == "bf16":
            rows = tensors["weight"].shape[0]
            return "bf16", {"weight": encodings.values(tensors, fmt, self.device, 0, rows).to(self.torch.bfloat16)}
        if want in ("nvfp4", "fp8:channel"):
            encoded, fmt = encodings.encode(tensors, fmt, "fp8" if want == "fp8:channel" else want, self.device)
            return fmt, encoded
        raise ValueError(f"model-template: {base} is stored {fmt}; a draft layer's matrices re-encode to bf16, "
                         f"nvfp4 or fp8:channel, not {want}")

    def stacked(self, bases, device=None):
        """(format, tensors) of matrices `bases` (one input) stacked by output rows, on `device` (the model's by
        default)."""
        return self._stack([self.stored(b, device) for b in bases], bases)

    def _stack(self, found, bases):
        torch = self.torch
        fmt = found[0][0]
        if any(f != fmt for f, _ in found):
            raise ValueError(f"model-template: {bases} are stored in different formats; stack only one")
        if len(found) == 1:
            return fmt, found[0][1]
        parts = [t for _, t in found]
        tensors = {"weight": torch.cat([t["weight"] for t in parts])}
        if fmt in ("fp8:channel", "mxfp8"):
            tensors["scale"] = torch.cat([t["scale"] for t in parts])
        elif fmt == "fp8:block":
            if any(t["weight"].shape[0] % t["block"][0] for t in parts[:-1]) or len({t["block"] for t in parts}) > 1:
                raise ValueError(f"model-template: {bases} are FP8 in blocks that do not stack by rows")
            tensors["scale"], tensors["block"] = torch.cat([t["scale"] for t in parts]), parts[0]["block"]
        elif fmt == "fp8:tensor":
            if len({float(t["scale"]) for t in parts}) != 1:
                raise ValueError(f"model-template: {bases} have different FP8 tensor scales")
            tensors["scale"] = parts[0]["scale"]
        elif fmt == "nvfp4":
            tensors["scale"] = torch.cat([t["scale"] for t in parts])
            tensors["global_scale"] = torch.cat([t["global_scale"].expand(t["weight"].shape[0]) for t in parts])
            inputs = [t.get("input_scale") for t in parts]
            if all(i is not None for i in inputs) and len({float(i) for i in inputs}) == 1:
                tensors["input_scale"] = inputs[0]
        return fmt, tensors

    def load(self, bases, role, encode=None, rows=None):
        """The matrices `bases` stacked into one, loaded in the projections slot (re-encoded to `encode` when only a
        drafter reads it: by the slot, or here where the slot does not take the stored format). Loaded once: the
        layers that run one checkpoint layer (a looped decoder's passes) get the same loaded matrix. rows: the
        first rows alone (a head stored with padded rows past the vocabulary)."""
        key = (id(self.proj), tuple(bases), role, encode, rows)   # what one projections slot loaded
        cache = self.__dict__.setdefault("_loaded", {})
        if key not in cache:
            cache[key] = self._load(bases, role, encode, rows)
        return cache[key]

    def _load(self, bases, role, encode=None, rows=None):
        if encode is not None and any(self.format(b) == "fp8:block" for b in bases):
            # The slot loads no FP8 blocks: the matrices re-encode here, to what is asked where its kernels take the
            # shape, else to their BF16 values, which the slot keeps as stored and reports (encoding_problem).
            want = encode if encode != "fp8" else "fp8:channel"
            problem = getattr(self.proj, "encoding_problem", None)
            rows = sum(self.ck.shape(b + ".weight")[0] for b in bases) if hasattr(self.ck, "shape") else None
            columns = self.ck.shape(bases[0] + ".weight")[1] if hasattr(self.ck, "shape") else None
            if problem is not None and rows is not None and problem(want, rows, columns):
                fmt, tensors = self._stack([self.reformat(b, "bf16") for b in bases], bases)
                return self.proj.load(tensors, fmt, role=role, encode=encode)
            fmt, tensors = self._stack([self.reformat(b, want) for b in bases], bases)
            return self.proj.load(tensors, fmt, role=role)
        fmt, tensors = self.stacked(bases)
        if rows is not None and tensors["weight"].shape[0] > rows:   # each row's tensors: its codes and scales
            tensors = {k: (v[:rows].contiguous() if hasattr(v, "dim") and v.dim() >= 1 and v.shape[0] ==
                           tensors["weight"].shape[0] else v) for k, v in tensors.items()}
        if fmt == "fp8:block" and "fp8:block" not in getattr(self.proj, "formats", ()):   # a slot's stated formats
            raise ValueError(f"model-template: {bases[0]} is FP8 in blocks of {tensors['block']}, which this "
                             "projections slot does not load; a draft layer's matrix re-encodes (option mtp_encode)")
        if encode is not None:
            return self.proj.load(tensors, fmt, role=role, encode=encode)
        return self.proj.load(tensors, fmt, role=role)

    def together(self, bases, role, encode=None):
        """The matrices `bases` (one input) as the fewest loaded matrices whose rows side by side are theirs in order:
        one stacked matrix where they are stored in one format, else each alone."""
        if len({self.format(b) for b in bases}) == 1:
            return [self.load(list(bases), role, encode)]
        return [self.load([b], role, encode) for b in bases]

    def bias(self, bases):
        """The stacked biases of `bases` as FP32 on the device, or None when none of them stores one."""
        names = [b + ".bias" for b in bases]
        if not any(n in self.ck for n in names):
            return None
        if not all(n in self.ck for n in names):
            raise ValueError(f"model-template: only some of {bases} store a bias")
        return self.torch.cat([self.ck.get(n, self.device).float() for n in names])

    def rows(self, base):
        """The output rows of matrix `base`."""
        p = self.parts(self.ck.headers, base, self.quant)
        if p is None:
            raise KeyError(f"model-template: no matrix {base} in the checkpoint")
        return p.shape[0]

    def stacked_expert(self, gate_up, down, index, interleaved=False, device=None):
        """(format, gate_up, down) of expert `index` of experts stored as two stacks, gate_up [experts][2 * inter][in]
        and down [experts][hidden][inter] (any format stored(): BF16, MXFP8, NVFP4), as the experts slot's load() takes
        them: the gate's rows, then the up projection's. `interleaved`: the stack keeps each gate row beside its up
        row (gate rows at even, up rows at odd places), moved apart here as whole rows with their scales."""
        fmt, gu = self.stored(gate_up, device, index)
        _, dn = self.stored(down, device, index)
        if interleaved:
            torch = self.torch
            for key in ("weight", "scale"):
                if key in gu and gu[key].dim() >= 2:
                    rows = gu[key]
                    gu[key] = torch.cat([rows[0::2], rows[1::2]]).contiguous()
        if fmt == "bf16":
            return fmt, gu["weight"], dn["weight"]
        return fmt, gu, dn

    def expert(self, gate, up, down, want=None, device=None):
        """(format, gate_up, down) of one routed expert as the experts slot's load() takes it: bfloat16 matrices, or the
        stored tensors as the projections slot's load takes them. want: the format the experts slot was begun with,
        which a draft layer's experts stored otherwise are re-encoded to (on the model's device). device: where an
        expert read as stored is handed over (the model's by default; the host for an experts plugin that packs its
        bundles there, so no expert passes through the device on its way to host memory)."""
        if want is not None and self.format(gate) != want:
            fmt, gate_up = self._stack([self.reformat(gate, want), self.reformat(up, want)], [gate, up])
            _, down_t = self.reformat(down, want)
        else:
            fmt, gate_up = self.stacked([gate, up], device)
            _, down_t = self.stacked([down], device)
        if fmt == "bf16":
            return fmt, gate_up["weight"], down_t["weight"]
        return fmt, gate_up, down_t
