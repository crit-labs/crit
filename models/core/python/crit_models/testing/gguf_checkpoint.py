"""A small Hugging Face checkpoint written as GGUF's reference converter writes it, for the template's GGUF tests.

The converter's steps, done here in its own direction from the data the template's code/gguf.py undoes them by: the
reader's convention (COMMON, OUTSIDE, the routed experts' stacks) and the model plugin's layout's gguf entry for the
architecture: its parts, outside, draft and constants tables, and its transforms (norms stored as 1 + w, -exp(A_log),
query and key rows permuted for a rotary layout, heads tiled, unit axes squeezed, a tensor split into parts by rows, a
matrix split by heads into its key and value parts, tensors joined along their inputs). Tensors the reader would read
apart (joined-rows) are written apart, as a converter that does not join them writes them; routed experts are stacked
[experts][rows][columns]. Matrices are stored Q8_0 (32 values a block, one FP16 scale), each one's values handed back
so the reference computes what the file stores; the rest F32.

    path, values = write_gguf(hf_folder, out_folder, "qwen35moe", layout)   # values: {hf name: the stored values, FP32}

The folder gets the checkpoint's config.json beside the GGUF, as a model downloaded with its base model's files has it.
A draft layer (under the layout's draft prefix) is written as the block after the text layers, its own tensors as the
draft table's parts. One step of a converter the reader does not need told is the fixture's own: which architectures'
converters keep a routed layer's gate and up projections as one stack (FUSED_GATE_UP)."""
import functools
import importlib.util
import json
from pathlib import Path
import re
import shutil
import struct

ALIGN = 32
F32, Q8_0 = 0, 8
# The converters that keep the routed experts' gate and up projections as one stack (ffn_gate_up_exps), each expert's
# [2 x inter][hidden] as the Hugging Face checkpoint stacks it.
FUSED_GATE_UP = ("gemma4",)
SHARD = re.compile(r"\.shard_(\d+)\.weight$")   # a table transformers saves in shards, joined by rows


@functools.lru_cache(maxsize=None)
def reader():
    """The template's code/gguf.py (its convention's tables, rules_of, sizes), loaded as its tests load it."""
    path = Path(__file__).resolve().parents[3] / "model-template" / "code" / "gguf.py"
    spec = importlib.util.spec_from_file_location("crit_models_testing_gguf_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plugin_layout(arch):
    """The layout of the models/cache plugin whose gguf reads GGUF architecture `arch`."""
    for path in sorted((Path(__file__).resolve().parents[4] / "cache").glob("*/layout.json")):
        layout = json.loads(path.read_text(encoding="utf-8"))
        if arch in layout.get("gguf", {}):
            return layout
    raise ValueError(f"no models/cache layout reads GGUF architecture {arch}")


def q8_0(rows):
    """FP32 [N][K] -> (Q8_0 blocks uint8 [N][K / 32 * 34], their values FP32 [N][K]), as the format quantizes."""
    import numpy as np
    w = np.asarray(rows, dtype=np.float32)
    blocks = w.reshape(w.shape[0], -1, 32)
    d = (np.abs(blocks).max(axis=-1, keepdims=True) / 127).astype(np.float16)
    scale = d.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(scale == 0, 0, np.round(blocks / np.where(scale == 0, 1, scale)))
    q = np.clip(q, -127, 127).astype(np.int8)
    raw = np.concatenate([d.view(np.uint8), q.view(np.uint8)], axis=-1).reshape(w.shape[0], -1)
    return np.ascontiguousarray(raw), (scale * q.astype(np.float32)).astype(np.float32).reshape(w.shape)


def _rotary_permuted(n, heads):
    """The source row of each stored row: pairs of rotary rows interleaved per head (permuted-rotary)."""
    import numpy as np
    return np.arange(n).reshape(heads, 2, n // heads // 2).swapaxes(1, 2).reshape(n)


def _tiled(count, groups, unit, before=0):
    """The source index of each stored index: heads grouped (g R + r) stored tiled (r G + g) (tiled-heads)."""
    import numpy as np
    ratio = count // groups
    heads = np.array([g * ratio + r for r in range(ratio) for g in range(groups)])
    return np.concatenate([np.arange(before), before + (heads[:, None] * unit + np.arange(unit)).reshape(-1)])


def _gguf_part(pairs, name):
    """The GGUF part, with its suffix, that the reader reads as `name` (a name under a layer or the draft prefix): the
    first of the (part, name) pairs that gives it, or None."""
    for part, given in pairs:
        if given == name:   # stored with its own name (ssm_a, ssm_dt.bias, layer_output_scale.weight)
            return part
        if not part.endswith((".weight", ".bias")):
            for suffix in (".weight", ".bias"):
                if name == given + suffix:
                    return part + suffix
    return None


def _suffixed(part):
    """(the name without its .weight or .bias, the suffix)."""
    for suffix in (".weight", ".bias"):
        if part.endswith(suffix):
            return part[:-len(suffix)], suffix
    return part, ""


class _Tensor:
    def __init__(self, name, kind, shape, data):
        self.name, self.kind, self.shape, self.data = name, kind, tuple(shape), data


def _string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _metadata(key, value):
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", 8) + _string(value)
    if isinstance(value, (list, tuple)):
        big = any(v >= 1 << 31 for v in value)
        kind, fmt = (11, "<q") if big else (5, "<i")
        return (_string(key) + struct.pack("<IIQ", 9, kind, len(value))
                + b"".join(struct.pack(fmt, int(v)) for v in value))
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", int(value))


def _write(path, metadata, tensors):
    entries = [_metadata(k, v) for k, v in metadata.items()]
    table, payload, cursor = [], [], 0
    for t in tensors:
        cursor = -(-cursor // ALIGN) * ALIGN
        ne = tuple(reversed(t.shape))
        table.append(_string(t.name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
                     + struct.pack("<IQ", t.kind, cursor))
        payload.append((cursor, t.data))
        cursor += len(t.data)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(table), len(entries)) + b"".join(entries) + b"".join(table)
    head += bytes(-len(head) % ALIGN)
    body = bytearray()
    for start, data in payload:
        body += bytes(start - len(body)) + data
    body += bytes(-len(body) % ALIGN)
    path.write_bytes(head + bytes(body))


def write_gguf(folder, out, arch, quantize=True, dense=(), layout=None):
    """The safetensors checkpoint in `folder` written to out/model.gguf as GGUF's converter writes `arch`, by the
    layout's gguf entry for it (`layout`: the model plugin's; by default the models/cache plugin's that reads `arch`);
    returns (the GGUF's path, {hf name: the values the file stores, FP32 in the Hugging Face layout} of every quantized
    tensor). dense: Hugging Face name endings of matrices kept F32 as the file's other tensors (a GGUF stores some
    matrices so: routers, small gates)."""
    import numpy as np
    import torch
    from safetensors.torch import load_file
    g = reader()
    layout = plugin_layout(arch) if layout is None else layout
    rules = g.rules_of(layout["gguf"][arch])
    steps, draft = rules.transforms, (layout.get("draft") or {}).get("prefix")
    folder, out = Path(folder), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    c = config.get("text_config") or config
    size = functools.partial(g.config_size, c)
    hf = {}
    for shard in sorted(folder.glob("*.safetensors")):
        hf.update(load_file(str(shard)))
    prefix = next(n[:-len("embed_tokens.weight")] for n in hf if n.endswith("embed_tokens.weight"))
    layers = c["num_hidden_layers"]
    drafts = 1 if draft and any(n.startswith(draft) for n in hf) else 0
    if drafts and rules.draft is None:
        raise ValueError(f"the checkpoint has a draft layer; the layout's gguf.{arch} reads no draft block")
    values, tensors, metadata = {}, [], {"general.architecture": arch, f"{arch}.block_count": layers + drafts}
    if drafts:
        metadata[f"{arch}.nextn_predict_layers"] = drafts

    def over(own, common):   # the entry's (part, name) pairs, then the convention's it does not mean otherwise
        return list(own.items()) + [(k, v) for k, v in common.items() if k not in own]

    def placed(pairs):   # {name: GGUF name}, "{key}" placed; a layer the config does not list: no such tensor
        found = {}
        for part, name in pairs:
            try:
                found.setdefault(g.placed_name(c, name, part), part)
            except ValueError:
                continue
        return found

    layer_pairs, draft_pairs = over(rules.parts, g.COMMON), list((rules.draft or {}).items())
    outside, constants = placed(over(rules.outside, g.OUTSIDE)), placed(rules.constants.items())
    zero, negated = steps.get("zero-centred"), tuple(steps.get("negated-exp", ()))
    tiled, joined_columns = steps.get("tiled-heads"), steps.get("joined-columns", {})
    split_rows, split_heads = steps.get("split-rows", {}), steps.get("split-heads", {})
    # a joined-columns part's names: {(name, a draft layer's own): (part, its place)}
    columns = {(name, part in (rules.draft or {})): (part, i) for part, names in joined_columns.items()
               for i, name in enumerate(names)}

    def first(keys):
        keys = [keys] if isinstance(keys, str) else keys
        found = [int(c[k]) for k in keys if c.get(k) is not None]
        if not found:
            raise ValueError(f"the config has none of {keys}")
        return found[0]

    def order(name):
        """(rows, inputs, unit) the converter moves of tensor `name`: permuted-rotary's rows, tiled-heads'."""
        for tensor, keys in steps.get("permuted-rotary", {}).items():
            if name.endswith((tensor + ".weight", tensor + ".bias")):
                return lambda n: _rotary_permuted(n, first(keys)), None, None
        for ending, how in (tiled or {}).get("tensors", {}).items():
            if name.endswith(ending):
                heads, groups = size(tiled["heads"], 0), size(tiled["groups"], 0)
                if not heads or not groups or heads == groups:
                    return None, None, None
                unit = size(how.get("unit", 1))
                index = _tiled(heads, groups, unit, size(how.get("before", 0)))
                if how.get("along", "rows") == "rows":
                    return lambda n: index, None, unit
                return None, index, unit
        return None, None, None

    def stored(hf_name, w, rows=None, inputs=None, unit=None):
        """A matrix's stored tensor data and kind: Q8_0 (its values to the reference) or F32, orders applied."""
        w = w.float().numpy()
        matrix = quantize and w.ndim == 2 and w.shape[1] % 32 == 0 and w.shape[0] > 1 and not hf_name.endswith(dense)
        if matrix:
            raw, back = q8_0(w)
            values[hf_name] = back
            data = raw
        else:
            data = w.astype(np.float32)
        if rows is not None:
            data = data[rows]
        if inputs is not None:
            if matrix:   # whole heads move: their 34-byte blocks
                per = unit // 32 * 34
                blocks = data.reshape(data.shape[0], -1, per)
                data = blocks[:, inputs.reshape(-1, unit)[:, 0] // unit].reshape(data.shape[0], -1)
            else:
                data = data[:, inputs]
        return (Q8_0 if matrix else F32), np.ascontiguousarray(data)

    def add(gguf_name, kind, data, shape):
        tensors.append(_Tensor(gguf_name, kind, shape, data.tobytes()))

    def write(name, w, gguf_name):
        """Tensor `name` stored as `gguf_name`, the entry's transforms that name it done."""
        rows, inputs, unit = order(name)
        rows = rows(w.shape[0]) if rows is not None else None
        for ending, spec in steps.get("squeezed", {}).items():
            if name.endswith(ending) and w.dim() == spec["rank"]:
                w = w.squeeze(spec["axis"])
        if w.dim() == 0:
            w = w.reshape(1)
        kind, data = stored(name, w, rows, inputs, unit)
        if zero and name.endswith(tuple(zero["ends"])) and not name.endswith(tuple(zero.get("except", ()))):
            data = data + 1
        if name.endswith(negated):
            data = -np.exp(data)
        add(gguf_name, kind, data.astype(np.float32) if kind == F32 else data, tuple(w.shape))

    experts, joins, tables = {}, {}, {}
    for name, w in hf.items():
        if draft and name.startswith(draft):
            rest = name[len(draft):]
            own = not rest.startswith(g.DRAFT_LAYER)
            block, part = layers, rest if own else rest[len(g.DRAFT_LAYER):]
        else:
            canonical = g.PREFIX + name[len(prefix):] if name.startswith(prefix) else name
            if canonical in outside:
                write(name, w, outside[canonical])
                continue
            sharded = SHARD.search(canonical)
            if sharded and canonical[:sharded.start()] + ".weight" in outside:
                tables.setdefault(canonical[:sharded.start()] + ".weight", {})[int(sharded[1])] = w
                continue
            if canonical in constants:
                key = constants[canonical]
                metadata[arch + key if key.startswith(".") else key] = [int(v) for v in w.tolist()]
                continue
            found = re.fullmatch(re.escape(g.PREFIX) + r"layers\.(\d+)\.(.+)", canonical)
            if found is None:
                raise ValueError(f"the fixture writes no GGUF tensor for {name}")
            own, block, part = False, int(found[1]), found[2]
        at = f"blk.{block}."
        if not own:
            expert = re.fullmatch(re.escape(rules.experts) + r"(\d+)\.(gate|up|down)_proj\.weight", part)
            if expert:
                experts.setdefault((block, expert[2]), {})[int(expert[1])] = (name, w)
                continue
            if part in (rules.experts + "gate_up_proj", rules.experts + "down_proj"):
                if arch in FUSED_GATE_UP:   # stored [experts][2 * inter][hidden] and [experts][hidden][inter]
                    stack = w
                else:
                    stack = w if part.endswith("down_proj") or w.shape[1] != c["hidden_size"] else w.transpose(1, 2)
                    if part.endswith("down_proj") and stack.shape[1] != c["hidden_size"]:
                        stack = stack.transpose(1, 2)
                if part.endswith("gate_up_proj"):
                    inter = stack.shape[1] // 2
                    for e in range(stack.shape[0]):
                        experts.setdefault((block, "gate"), {})[e] = (f"{name}:{e}:gate", stack[e, :inter])
                        experts.setdefault((block, "up"), {})[e] = (f"{name}:{e}:up", stack[e, inter:])
                else:
                    for e in range(stack.shape[0]):
                        experts.setdefault((block, "down"), {})[e] = (f"{name}:{e}", stack[e])
                continue
        base, suffix = _suffixed(part)
        if (base, own) in columns:
            joined, i = columns[base, own]
            joins.setdefault((block, joined, suffix), {})[i] = (name, w)
            continue
        if not own and base in split_rows:
            pieces = split_rows[base]["parts"]
            if len(pieces) != 2:
                raise ValueError(f"the fixture splits {base} into two parts, not {pieces}")
            rows = size(split_rows[base]["rows"])
            kind, data = stored(name, w)
            add(at + pieces[0] + suffix, kind, data[:rows], (rows, w.shape[1]))
            add(at + pieces[1] + suffix, kind, data[rows:], (w.shape[0] - rows, w.shape[1]))
            continue
        if not own and base in split_heads:
            spec = split_heads[base]
            heads, key_rows = size(spec["heads"]), size(spec["rows"])
            kv = w.float().reshape(heads, -1, w.shape[-1])
            key, value = kv[:, :key_rows].transpose(1, 2).contiguous(), kv[:, key_rows:].contiguous()
            back = []
            for gguf_part, t in zip(spec["parts"], (key, value)):
                raw, got = q8_0(t.reshape(-1, t.shape[-1]).numpy())
                add(at + gguf_part + suffix, Q8_0, raw, t.shape)
                back.append(torch.from_numpy(got).reshape(t.shape))
            values[name] = torch.cat([back[0].transpose(1, 2), back[1]], 1).reshape(w.shape).numpy()
            continue
        gguf_part = _gguf_part(draft_pairs if own else layer_pairs, part)
        if gguf_part is None:
            raise ValueError(f"the fixture writes no GGUF tensor for {name}")
        write(name, w, at + gguf_part)
    for (block, part, suffix), found in sorted(joins.items()):   # joined along their inputs, as the converter writes
        names = joined_columns[part]
        if sorted(found) != list(range(len(names))):
            raise ValueError(f"the checkpoint has only some of {names}")
        parts = [found[i] for i in range(len(names))]
        key = f"{part}:joined"
        kind, data = stored(key, torch.cat([w for _, w in parts], 1))
        if key in values:
            back, start = values.pop(key), 0
            for hf_name, w in parts:
                values[hf_name], start = back[:, start:start + w.shape[1]], start + w.shape[1]
        add(f"blk.{block}.{part}{suffix}", kind, data, (parts[0][1].shape[0], sum(w.shape[1] for _, w in parts)))
    stacks = {}
    for (layer, which), found in sorted(experts.items()):
        parts = [stored(*found[e]) for e in sorted(found)]
        stacks[layer, which] = (parts[0][0], np.stack([p[1] for p in parts]), tuple(found[0][1].shape))
    for (layer, which), (kind, data, (rows, columns_)) in sorted(stacks.items()):
        if arch in FUSED_GATE_UP and which in ("gate", "up"):   # one gate-up stack, the gate's rows first
            if which == "up":
                continue
            data, rows, which = np.concatenate([data, stacks[layer, "up"][1]], axis=1), 2 * rows, "gate_up"
        add(f"blk.{layer}.ffn_{which}_exps.weight", kind, data, (data.shape[0], rows, columns_))
    for name, shards in tables.items():   # a table saved in shards: its rows joined, F32
        rows = torch.cat([shards[i].float() for i in sorted(shards)])
        add(outside[name], F32, rows.numpy().astype(np.float32), tuple(rows.shape))
    path = out / "model.gguf"
    _write(path, metadata, tensors)
    shutil.copy(folder / "config.json", out / "config.json")
    for name in ("generation_config.json",):
        if (folder / name).is_file():
            shutil.copy(folder / name, out / name)
    return path, values


def stored_reference(model, values):
    """The reference model's tensors set to the values the GGUF stores (its quantized matrices), so it computes
    what the file holds. A fused experts tensor gets each expert's part back."""
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            if ":" in name:   # a fused stack's expert part: name:expert[:gate|up]
                base, *where = name.split(":")
                p = params[base]
                e = int(where[0])
                v = torch.from_numpy(value).to(p.dtype)
                hidden_last = p.shape[-1] != v.shape[-1]
                target = p[e].t() if hidden_last else p[e]
                if len(where) == 2:
                    inter = target.shape[0] // 2
                    target = target[:inter] if where[1] == "gate" else target[inter:]
                target.copy_(v)
                continue
            if name in params:
                params[name].copy_(torch.from_numpy(value).to(params[name].dtype))
