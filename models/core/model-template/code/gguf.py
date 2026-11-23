"""A GGUF checkpoint read as the composed model reads a safetensors one: its tensors under the Hugging Face names the
blocks read, its quantized matrices as stored (format gguf:<TYPE>, the rows the file holds, for gguf-projections and an
experts plugin that takes GGUF experts), its other tensors decoded on the host by Core's decoders.

GGUF's reference converter renames every tensor of the Hugging Face checkpoint by one convention (blk.<layer>.<part>,
token_embd, output_norm, output, a routed layer's experts as stacks) and, for some architectures, names other parts and
reorders or transforms a tensor on the way. This reader knows the convention alone (COMMON, OUTSIDE, EXPERT_STACKS)
and generic steps (TRANSFORMS); what an architecture adds is data in the model plugin's layout, whose `gguf` maps each
general.architecture the plugin reads to an entry:

    parts       {GGUF part: name under the layer}: the architecture's parts beside the convention's (over them where a
                part means another tensor). A part is named without its .weight or .bias, which carries over; one the
                converter stores without such a suffix, or with another (.scale), with it
    outside     {GGUF tensor: name}: tensors outside the layers beside the convention's. "{key}" in a name is the one
                layer the config key `key` lists (the layers' numbers from 1, as a layout's `before` lists them)
    draft       {GGUF part of a draft (nextn) block: name under the layout's draft prefix}: the first block after the
                text layers is read as the draft layer, its decoder layer's parts as any layer's (under the draft
                prefix's layers.0.) and the parts named here as the draft layer's own; without it no draft block is
                read. A draft block's own copies of the embedding and the head (DRAFT_COPIES) are never read: the
                template drafts with the model's own
    experts     where a layer's routed experts are, under the layer (mlp.experts. by default)
    constants   {metadata key: name}: integer arrays a converter moves to metadata (a leading "." the architecture's
                own key), read back as int64 tensors; "{key}" as in outside
    transforms  {step: its parameters}: the converter's steps, each undone here on the tensors its parameters name

A size in the parameters (rows, heads, a unit) is a number, a config key, or a list of terms summed, each a number, a
config key or a list of them multiplied: [["num_key_heads", "key_head_dim", 2], ["num_value_heads", ...]].

    zero-centred      {ends: [name endings], except: [name endings]}: stored as 1 + w, read as w (in FP32, one
                      rounding each way)
    negated-exp       [name endings]: stored as -exp(w), read as log(-a)
    squeezed          {name ending: {axis, rank}}: a rank-`rank` tensor stored without its unit axis `axis`, given it
                      back
    permuted-rotary   {tensor: heads}: query or key rows the converter interleaved per head for its rotary layout
                      (the tensor's .weight and .bias), moved back as whole rows; heads a config key, or a list of
                      them of which the first the config has
    tiled-heads       {heads, groups, tensors: {name ending: {unit, before, along}}}: heads the converter tiled (head
                      r G + g) where the Hugging Face layout groups them (g R + r) as there are fewer groups than heads:
                      a tensor's `unit`-wide heads after `before` other entries, along its rows or (along: inputs) its
                      inputs, moved as whole rows or as whole blocks of the stored format
    joined-rows       {GGUF part: {names, rows}}: tensors the converter wrote as one, their rows in order, read as each
                      (the first one's `rows` rows, the second the rest)
    joined-columns    {GGUF part: [names]}: tensors the converter joined along their inputs, read as each (equal
                      widths, whole blocks of the stored format)
    split-rows        {name: {parts, rows}}: one tensor the converter wrote as several parts by rows, read joined
                      (`rows`: the first part's, checked)
    split-heads       {name: {parts: [key, value], heads, rows}}: a matrix the converter split by heads into a key part
                      [heads][inputs][rows] (each head's key rows transposed) and a value part [heads][rows][inputs],
                      read joined back and decoded (a plain matrix); heads and rows (the key rows a head) checked

The names a step gives are in its part's place: under the part's layer, or under the draft prefix for a part the draft
table names. Moves of whole rows or blocks apply to quantized matrices without decoding them.

A tensor of the file that maps to no name is refused (so a converter's new part fails here, not later as a wrong
answer), the file's rotary frequency table (read by the config) excepted.

    ck = GgufCheckpoint(path, config, layout)  # path: the .gguf (a split set's first part) or its folder
    ck.get("model.layers.0.input_layernorm.weight", "cuda")
    matrices = GgufMatrices(ck, projections, None, "cuda")
    matrices.format("model.layers.0.self_attn.q_proj")            # "gguf:Q6_K", or "bf16" for a dense tensor
    matrices.load(["model.layers.0.self_attn.q_proj", ...], "qkv")
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import re

PREFIX = "model."
DENSE = ("F32", "F16", "BF16")
BLOCK = re.compile(r"blk\.(\d+)\.(.+?)(\.weight|\.bias)?$")
PLACE = re.compile(r"\{(\w+)\}")   # "{key}" in a name: the one layer a config key lists

# The convention's parts of a layer -> the Hugging Face name under model.layers.<i>. (a ".weight" or ".bias" suffix
# carries over; a part stored without one keeps its own name).
COMMON = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    "ffn_gate_inp": "mlp.gate",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj", "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj", "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
}
# Tensors outside the layers -> their Hugging Face names (lm_head outside the model's prefix).
OUTSIDE = {"token_embd.weight": PREFIX + "embed_tokens.weight", "output_norm.weight": PREFIX + "norm.weight",
           "output.weight": "lm_head.weight"}
# Read by the config rule (a rope scaling's frequency table), not by the model.
IGNORED = ("rope_freqs.weight",)
# A routed layer's experts: [experts][rows][row bytes] stacks, read as each expert's matrices; a gate-up stack holds the
# gate's rows, then the up projection's.
EXPERT_STACKS = {"ffn_gate_exps": "gate_proj", "ffn_up_exps": "up_proj", "ffn_down_exps": "down_proj"}
GATE_UP_STACK = "ffn_gate_up_exps"
# A draft block's own copies of the embedding and the head: the template drafts with the model's own.
DRAFT_COPIES = ("nextn.embed_tokens", "nextn.shared_head_head")
DRAFT_LAYER = "layers.0."   # the draft layer's decoder layer, under the layout's draft prefix

ENTRY_KEYS = ("parts", "outside", "draft", "experts", "constants", "transforms")
TRANSFORMS = ("zero-centred", "negated-exp", "squeezed", "permuted-rotary", "tiled-heads", "joined-rows",
              "joined-columns", "split-rows", "split-heads")


@dataclass(frozen=True)
class Rules:
    """A layout's gguf entry for one architecture (rules_of)."""
    parts: dict = field(default_factory=dict)        # the layers' parts beside COMMON
    outside: dict = field(default_factory=dict)      # the tensors outside the layers beside OUTSIDE
    draft: dict = None                               # a draft block's own parts; None: no draft block is read
    experts: str = "mlp.experts."                    # where a layer's routed experts are, under the layer
    constants: dict = field(default_factory=dict)    # metadata arrays read as int64 tensors
    transforms: dict = field(default_factory=dict)   # {step: its parameters}


def _is_size(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, str)):
        return True
    return isinstance(value, list) and bool(value) and all(
        isinstance(term, (int, str)) and not isinstance(term, bool) or isinstance(term, list) and term and all(
            isinstance(f, (int, str)) and not isinstance(f, bool) for f in term) for term in value)


def _names(value, least=1):
    return isinstance(value, list) and len(value) >= least and all(isinstance(n, str) and n for n in value)


def _table(value):
    return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())


def _transform_problems(step, spec):
    """What is wrong with one step's parameters ([] when nothing)."""
    def each(check, what):
        if not isinstance(spec, dict) or not spec or not all(isinstance(k, str) and check(v) for k, v in spec.items()):
            return [f"{step} is {what}"]
        return []
    if step == "zero-centred":
        ok = isinstance(spec, dict) and _names(spec.get("ends")) and set(spec) <= {"ends", "except"} and (
            "except" not in spec or _names(spec["except"]))
        return [] if ok else [f"{step} is {{ends: [name endings], except: [name endings]}}"]
    if step == "negated-exp":
        return [] if _names(spec) else [f"{step} is [name endings]"]
    if step == "squeezed":
        return each(lambda v: isinstance(v, dict) and set(v) == {"axis", "rank"} and all(
            isinstance(v[k], int) and not isinstance(v[k], bool) for k in v) and 0 <= v["axis"] < v["rank"],
            "{name ending: {axis, rank}}")
    if step == "permuted-rotary":
        return each(lambda v: isinstance(v, str) and v or _names(v), "{tensor: heads (a config key, or config keys "
                    "of which the first the config has)}")
    if step == "tiled-heads":
        tensors = spec.get("tensors") if isinstance(spec, dict) else None
        ok = isinstance(spec, dict) and set(spec) == {"heads", "groups", "tensors"} and _is_size(spec["heads"]) and \
            _is_size(spec["groups"]) and isinstance(tensors, dict) and tensors and all(
                isinstance(v, dict) and set(v) <= {"unit", "before", "along"} and all(
                    _is_size(v[k]) for k in ("unit", "before") if k in v) and v.get("along", "rows") in (
                    "rows", "inputs") for v in tensors.values())
        return [] if ok else [f"{step} is {{heads, groups, tensors: {{name ending: {{unit, before, along: rows | "
                              "inputs}}}}"]
    if step == "joined-rows":
        return each(lambda v: isinstance(v, dict) and set(v) == {"names", "rows"} and _names(v["names"], 2) and len(
            v["names"]) == 2 and _is_size(v["rows"]), "{GGUF part: {names: [first, second], rows: the first's}}")
    if step == "joined-columns":
        return each(lambda v: _names(v, 2), "{GGUF part: [names, equal widths]}")
    if step == "split-rows":
        return each(lambda v: isinstance(v, dict) and set(v) <= {"parts", "rows"} and _names(v.get("parts"), 2) and (
            "rows" not in v or _is_size(v["rows"])), "{name: {parts: [GGUF parts in row order], rows: the first's}}")
    if step == "split-heads":
        return each(lambda v: isinstance(v, dict) and set(v) <= {"parts", "heads", "rows"} and _names(
            v.get("parts"), 2) and len(v["parts"]) == 2 and all(_is_size(v[k]) for k in ("heads", "rows") if k in v),
            "{name: {parts: [key, value], heads, rows: the key rows a head}}")
    return [f"{step} is no step this reader takes ({', '.join(TRANSFORMS)})"]


def rules_of(spec):
    """The Rules a layout's gguf entry for one architecture gives, or ValueError naming what is wrong with it."""
    spec = dict(spec or {})
    problems = [f"{key} is no key of an entry ({', '.join(ENTRY_KEYS)})" for key in sorted(set(spec) - set(ENTRY_KEYS))]
    problems += [f"{key} is {{GGUF name: name}}" for key in ("parts", "outside", "draft", "constants")
                 if key in spec and not _table(spec[key])]
    if not isinstance(spec.get("experts", ""), str):
        problems.append("experts is the routed experts' prefix under the layer")
    transforms = spec.get("transforms", {})
    if not isinstance(transforms, dict):
        problems.append("transforms is {step: its parameters}")
    else:
        for step, parameters in transforms.items():
            problems += _transform_problems(step, parameters)
    if problems:
        raise ValueError("model-template: a layout's gguf entry: " + "; ".join(problems))
    return Rules(dict(spec.get("parts", {})), dict(spec.get("outside", {})),
                 dict(spec["draft"]) if "draft" in spec else None, spec.get("experts", "mlp.experts."),
                 dict(spec.get("constants", {})), dict(transforms))


@dataclass
class Entry:
    """One Hugging Face tensor: where its values are and what turns them into it."""
    shape: tuple
    sources: tuple                  # GGUF TensorInfos (several for a joined tensor), () for a constant
    dtype: str                      # the source's ggml type ("I64" for a constant)
    rows: object = None             # a slice of the source's rows (a joined tensor's part), or an expert index
    expert: int = None
    order: tuple = None             # (index of output rows from source rows, or of input heads, "rows" | "inputs",
                                    # the heads' width along the inputs)
    transform: str = None           # "minus_one", "log_neg"
    values: tuple = None            # a constant's values
    columns: slice = None           # a slice of the source's input columns (a joined tensor's part), whole blocks
    heads: tuple = None             # (heads, key rows a head): a matrix joined from its key and value parts, decoded

    @property
    def stored(self):
        return self.dtype not in DENSE + ("I64",) and self.heads is None

    @property
    def nbytes(self):
        if not self.sources:
            return 8 * len(self.values)
        total = sum(s.nbytes for s in self.sources)
        if self.expert is not None:
            total //= self.sources[0].shape[0]
        if isinstance(self.rows, slice):
            n = self.sources[0].shape[-2] if len(self.sources[0].shape) > 1 else self.sources[0].shape[0]
            total = total * (self.rows.stop - self.rows.start) // n
        if self.columns is not None:
            total = total * (self.columns.stop - self.columns.start) // self.sources[0].shape[-1]
        return total

    # The safetensors header's fields the template reads of a table (its rows read in place from the file).
    @property
    def shard(self):
        return self.sources[0].shard

    @property
    def data_begin(self):
        return self.sources[0].data_begin


def _cfg(config):
    return (config.get("text_config") or config) if isinstance(config, dict) else {}


def _permuted_rows(n, heads):
    """The source row of each Hugging Face query or key row: the converter interleaved each head's rotary pairs."""
    import numpy as np
    per = n // heads
    return np.arange(n).reshape(heads, per // 2, 2).transpose(0, 2, 1).reshape(n)


def _grouped_heads(count, groups, unit, before=0):
    """The source index of each Hugging Face index along a dimension whose last `count` x `unit` entries (after
    `before` others) are heads the converter tiled (head r G + g) and the Hugging Face layout groups (g R + r)."""
    import numpy as np
    ratio = count // groups
    heads = np.arange(count).reshape(ratio, groups).T.reshape(count)   # grouped position -> tiled head
    within = (heads[:, None] * unit + np.arange(unit)).reshape(-1)
    return np.concatenate([np.arange(before), before + within])


def config_size(config, expr, missing=None):
    """The value of a size (a number, a config key, or a sum of products of them: rules_of) in `config`; a key the
    config lacks raises ValueError, or gives `missing` where that is not None."""
    if isinstance(expr, int):
        return expr
    if isinstance(expr, str):
        value = config.get(expr)
        if value is None:
            if missing is not None:
                return missing
            raise ValueError(f"model-template: reading this GGUF takes the config's {expr}")
        return int(value)
    total = 0
    for term in expr:
        found = [config_size(config, f, missing) for f in (term if isinstance(term, list) else [term])]
        if missing is not None and missing in found:
            return missing
        total += math.prod(found)
    return total


def placed_name(config, template, what):
    """`template` with each "{key}" replaced by the one layer config key `key` lists (numbers from 1)."""
    def layer(match):
        found = config.get(match[1])
        listed = list(found) if isinstance(found, (list, tuple)) else [] if found is None else [found]
        if len(listed) != 1:
            raise ValueError(f"model-template: the GGUF's {what} serves one layer, the one the config's {match[1]} "
                             f"lists; it lists {listed or 'none'}")
        return str(int(listed[0]) - 1)
    return PLACE.sub(layer, template)


class GgufCheckpoint:
    """A GGUF model's tensors under Hugging Face names (the template's Checkpoint interface): `name in ck`, names(),
    nbytes(), shape(), get(name, device), headers[name] (shape, dtype, nbytes, shard, data_begin), close()."""

    def __init__(self, path, config, layout=None):
        """layout: the model plugin's layout: its gguf ({general.architecture: entry}) and its draft prefix."""
        from crit_core.model.gguf import inspect_gguf_model
        from crit_core.model.gguf_blocks import GgufTensors
        self.folder = Path(path)
        self.model = inspect_gguf_model(path)
        self.arch = self.model.architecture
        layout = dict(layout or {})
        readers = dict(layout.get("gguf") or {})
        if self.arch not in readers:
            raise ValueError(f"model-template: the model plugin's layout reads GGUF architectures "
                             f"{', '.join(sorted(readers)) or 'none'} (its gguf); this file's general.architecture is "
                             f"{self.arch!r}")
        self.rules = rules_of(readers[self.arch])
        self.draft = (layout.get("draft") or {}).get("prefix")
        if self.rules.draft is not None and not self.draft:
            raise ValueError(f"model-template: the layout's gguf.{self.arch} names a draft block's parts; the layout "
                             "has no draft prefix")
        self.config = _cfg(config)
        self.reader = GgufTensors(self.model)
        self.headers = {}
        self._map()

    # -- the names -------------------------------------------------------------------------------------------------
    def _size(self, expr, missing=None):
        return config_size(self.config, expr, missing)

    def _add(self, name, entry):
        if name in self.headers:
            raise ValueError(f"model-template: two tensors of the GGUF map to {name}")
        self.headers[name] = entry

    def _map(self):
        rules, tensors, steps = self.rules, self.model.tensors, self.rules.transforms
        parts, outside = {**COMMON, **rules.parts}, {**OUTSIDE, **rules.outside}
        joined_rows, joined_columns = steps.get("joined-rows", {}), steps.get("joined-columns", {})
        # the parts of one tensor the converter split (split-rows, split-heads): {part: (step, name, its parts)}
        pieces = {part: (step, name, spec["parts"]) for step in ("split-rows", "split-heads")
                  for name, spec in steps.get(step, {}).items() for part in spec["parts"]}
        layers = int(self.config.get("num_hidden_layers") or self.model.get(".block_count") or 0)
        unmapped = []
        for gguf_name, info in tensors.items():
            if gguf_name in IGNORED:
                continue
            if gguf_name in outside:
                name = placed_name(self.config, outside[gguf_name], gguf_name)
                self._add(name, self._entry(name, info))
                continue
            found = BLOCK.fullmatch(gguf_name)
            if found is None:
                unmapped.append(gguf_name)
                continue
            layer, part, suffix = int(found[1]), found[2], found[3] or ""
            at, table = f"{PREFIX}layers.{layer}.", parts
            if layer >= layers:   # a draft (nextn) block: the draft layer's names (the first draft block's alone)
                if layer > layers or rules.draft is None or part in DRAFT_COPIES:
                    continue
                at = self.draft + DRAFT_LAYER
                if part in rules.draft or part + suffix in rules.draft:   # the draft layer's own
                    at, table = self.draft, rules.draft
            if part in EXPERT_STACKS:
                self._experts(at, EXPERT_STACKS[part], info)
            elif part == GATE_UP_STACK:
                half = info.shape[1] // 2
                self._experts(at, "gate_proj", info, slice(0, half))
                self._experts(at, "up_proj", info, slice(half, 2 * half))
            elif part in joined_rows:
                spec = joined_rows[part]
                rows = self._size(spec["rows"])
                for name, cut in zip(spec["names"], (slice(0, rows), slice(rows, info.shape[0]))):
                    self._add(at + name + suffix, self._entry(at + name + suffix, info, rows=cut))
            elif part in joined_columns:
                self._columns(at, joined_columns[part], info, suffix)
            elif part in pieces:
                self._pieces(at, layer, part, suffix, info, *pieces[part])
            elif part + suffix in table:   # a part the table names with its suffix (a bias kept as its own tensor)
                self._add(at + table[part + suffix], self._entry(at + table[part + suffix], info))
            elif part in table:
                name = at + table[part] + suffix
                self._add(name, self._entry(name, info))
            else:
                unmapped.append(gguf_name)
        for key, template in rules.constants.items():
            values = self.model.get(key)
            if isinstance(values, tuple):
                self._add(placed_name(self.config, template, key),
                          Entry((len(values),), (), "I64", values=tuple(int(v) for v in values)))
        if unmapped:
            raise ValueError(f"model-template: {len(unmapped)} tensors of this {self.arch} GGUF map to no name the "
                             f"template reads: {', '.join(sorted(unmapped)[:8])}")

    def _experts(self, at, part, info, rows=None):
        for e in range(info.shape[0]):
            shape = info.shape[1:] if rows is None else (rows.stop - rows.start, info.shape[2])
            self._add(f"{at}{self.rules.experts}{e}.{part}.weight",
                      Entry(tuple(shape), (info,), info.dtype, rows=rows, expert=e))

    def _columns(self, at, names, info, suffix):
        """A tensor the converter joined along its inputs (joined-columns): each name's equal share of its input
        columns, whole blocks of its stored format."""
        width = info.shape[-1] // len(names)
        for i, name in enumerate(names):
            entry = Entry((info.shape[0], width), (info,), info.dtype, columns=slice(i * width, (i + 1) * width))
            if entry.stored:
                from crit_core.model.gguf_blocks import BLOCKS, TYPES
                if width % BLOCKS[TYPES[info.dtype]][0]:
                    raise ValueError(f"model-template: {info.name} is {info.dtype}, whose blocks do not split at its "
                                     f"{width}-wide parts")
            self._add(at + name + suffix, entry)

    def _pieces(self, at, layer, part, suffix, info, step, name, members):
        """One part of a tensor the converter split (split-rows, split-heads): read with the others at the first."""
        tensors = self.model.tensors
        found = [tensors.get(f"blk.{layer}.{member}{suffix}") for member in members]
        if any(f is None for f in found):
            have = [m for m, f in zip(members, found) if f is not None]
            raise ValueError(f"model-template: blk.{layer} has {', '.join(have)} without "
                             f"{', '.join(m for m in members if m not in have)}")
        if part != members[0]:
            return   # read with the first
        spec = self.rules.transforms[step][name]
        if step == "split-rows":
            if "rows" in spec and info.shape[0] != self._size(spec["rows"]):
                raise ValueError(f"model-template: {info.name} holds {info.shape[0]} rows, the config gives "
                                 f"{name}'s first part {self._size(spec['rows'])}")
            self._add(at + name + suffix, self._entry(at + name + suffix, info, joined=tuple(found[1:])))
            return
        key, value = found
        if len(key.shape) != 3 or len(value.shape) != 3 or tuple(value.shape[::2]) != tuple(key.shape[:2]):
            raise ValueError(f"model-template: {key.name} {list(key.shape)} and {value.name} {list(value.shape)} are "
                             "not one matrix's key and value rows by heads")
        heads, rank, rows = key.shape
        for what, stored in (("heads", heads), ("rows", rows)):
            if what in spec and stored != self._size(spec[what]):
                raise ValueError(f"model-template: {key.name} holds {stored} {what}, the config "
                                 f"{self._size(spec[what])}")
        self._add(at + name + suffix,
                  Entry((heads * (rows + value.shape[1]), rank), (key, value), key.dtype, heads=(heads, rows)))

    def _entry(self, name, info, rows=None, joined=()):
        """The entry of HF tensor `name` from GGUF `info` (and `joined`, rows after it), the transforms that name it
        applied."""
        shape = list(info.shape)
        if rows is not None:
            shape[0] = rows.stop - rows.start
        shape[0] += sum(j.shape[0] for j in joined)
        entry = Entry(tuple(shape), (info, *joined), info.dtype, rows=rows)
        steps = self.rules.transforms
        for ending, spec in steps.get("squeezed", {}).items():
            if name.endswith(ending) and len(entry.shape) == spec["rank"] - 1:
                entry.shape = entry.shape[:spec["axis"]] + (1,) + entry.shape[spec["axis"]:]
        zero = steps.get("zero-centred")
        if zero and name.endswith(tuple(zero["ends"])) and not name.endswith(tuple(zero.get("except", ()))):
            entry.transform = "minus_one"
        if name.endswith(tuple(steps.get("negated-exp", ()))):
            entry.transform = "log_neg"
        for tensor, heads in steps.get("permuted-rotary", {}).items():
            if name.endswith((tensor + ".weight", tensor + ".bias")):
                entry.order = (_permuted_rows(entry.shape[0], self._heads(heads)), "rows", None)
        if "tiled-heads" in steps:
            order = self._tiled(name, steps["tiled-heads"])
            if order is not None:
                entry.order = order
        return entry

    def _heads(self, keys):
        """The value of a config key, or of a list of them the first the config has."""
        for key in [keys] if isinstance(keys, str) else keys:
            if self.config.get(key) is not None:
                return int(self.config[key])
        raise ValueError(f"model-template: reading this GGUF takes the config's "
                         f"{keys if isinstance(keys, str) else ' or '.join(keys)}")

    def _tiled(self, name, spec):
        """The tiled heads' order of tensor `name` where there are fewer groups than heads, else None."""
        for ending, how in spec["tensors"].items():
            if not name.endswith(ending):
                continue
            heads, groups = self._size(spec["heads"], 0), self._size(spec["groups"], 0)
            if not heads or not groups or heads == groups:
                return None
            unit, before = self._size(how.get("unit", 1)), self._size(how.get("before", 0))
            return _grouped_heads(heads, groups, unit, before), how.get("along", "rows"), unit
        return None

    # -- the Checkpoint interface ---------------------------------------------------------------------------------
    def __contains__(self, name):
        return name in self.headers

    def names(self):
        return self.headers.keys()

    def nbytes(self, name):
        return self.headers[name].nbytes

    def shape(self, name):
        return tuple(self.headers[name].shape)

    def rows(self, name):
        """The stored rows of a quantized tensor, uint8 numpy [N][row bytes] (an expert's, a joined tensor's part,
        the orders applied): what gguf-projections loads."""
        import numpy as np
        e = self.headers[name]
        parts = []
        for info in e.sources:
            raw = self.reader.rows(info.name)
            if e.expert is not None:
                raw = raw[e.expert]
            if isinstance(e.rows, slice):
                raw = raw[e.rows]
            if e.columns is not None:   # whole blocks of the stored format
                from crit_core.model.gguf_blocks import BLOCKS, TYPES
                values, size = BLOCKS[TYPES[e.dtype]]
                raw = raw[:, e.columns.start // values * size:e.columns.stop // values * size]
            parts.append(raw)
        raw = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if e.order is not None:
            index, along, unit = e.order
            if along == "rows":
                raw = raw[index]
            else:
                raw = self._move_inputs(name, raw, index, unit)
        raw = np.ascontiguousarray(raw)
        return raw if raw.flags.writeable else raw.copy()   # the file's mapping is read-only; a load takes a copy

    def _move_inputs(self, name, raw, index, unit):
        """A matrix's input columns reordered (whole `unit`-wide heads) by moving whole blocks of its stored format."""
        from crit_core.model.gguf_blocks import BLOCKS, TYPES
        e = self.headers[name]
        values, size = BLOCKS[TYPES[e.dtype]] if e.stored else (1, {"F32": 4, "F16": 2, "BF16": 2}[e.dtype])
        if unit % values:
            raise ValueError(f"model-template: {name} is {e.dtype}, whose {values}-value blocks do not align with the "
                             f"{unit}-wide heads the converter reordered; the inputs cannot be moved as stored")
        heads = index.reshape(-1, unit)[:, 0] // unit
        per = unit // values * size
        blocks = raw.reshape(raw.shape[0], -1, per)
        return blocks[:, heads].reshape(raw.shape[0], -1)

    def values(self, name):
        """FP32 (int64 for a constant) torch tensor of HF tensor `name` on the CPU, its Hugging Face shape and layout."""
        import numpy as np
        import torch
        from crit_core.model.gguf_blocks import TYPES, stored_values
        e = self.headers[name]
        if not e.sources:
            return torch.tensor(e.values, dtype=torch.int64)
        if e.heads is not None:
            key, value = (self._decoded(info) for info in e.sources)
            return torch.cat([key.transpose(1, 2), value], dim=1).reshape(e.shape)
        if e.stored:
            raw = self.rows(name)
            flat = raw.reshape(-1, raw.shape[-1])
            k = e.shape[-1] if len(e.shape) > 1 else e.shape[0]
            out = np.empty((flat.shape[0], k), dtype=np.float32)
            for start in range(0, flat.shape[0], 8192):   # bounded temporaries
                out[start:start + 8192] = stored_values(flat[start:start + 8192], TYPES[e.dtype], k)
            tensor = torch.from_numpy(out)
        else:
            parts = []
            for info in e.sources:
                kind = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[info.dtype]
                t = torch.from_numpy(np.array(self.reader.raw(info.name))).view(kind).reshape(info.shape)
                if e.expert is not None:
                    t = t[e.expert]
                if isinstance(e.rows, slice):
                    t = t[e.rows]
                if e.columns is not None:
                    t = t[:, e.columns]
                parts.append(t.float())
            tensor = parts[0] if len(parts) == 1 else torch.cat(parts)
            if e.order is not None:
                index, along, _ = e.order
                tensor = tensor[torch.from_numpy(index)] if along == "rows" else tensor[:, torch.from_numpy(index)]
        if e.transform == "minus_one":
            tensor = tensor - 1
        elif e.transform == "log_neg":
            tensor = torch.log(-tensor)
        return tensor.reshape(e.shape)

    def _decoded(self, info):
        """FP32 torch tensor of a whole GGUF tensor, its stored shape (quantized rows decoded)."""
        import numpy as np
        import torch
        from crit_core.model.gguf_blocks import TYPES, stored_values
        if info.dtype in DENSE:
            kind = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[info.dtype]
            return torch.from_numpy(np.array(self.reader.raw(info.name))).view(kind).reshape(info.shape).float()
        raw = self.reader.rows(info.name)
        flat = raw.reshape(-1, raw.shape[-1])
        return torch.from_numpy(stored_values(flat, TYPES[info.dtype], info.shape[-1])).reshape(info.shape)

    def get(self, name, device):
        return self.values(name).to(device)

    def close(self):
        self.reader.close()


class GgufMatrices:
    """A GGUF checkpoint's matrices as the projections slot's load() and the experts slot's load() take them (the
    template's Matrices interface): quantized ones as gguf:<TYPE> with their stored rows, dense ones as bf16."""

    def __init__(self, checkpoint, projections, quantization, device):
        import torch
        self.ck, self.proj, self.device, self.torch = checkpoint, projections, device, torch
        self.stored_experts = True   # the model sets it: whether the experts plugin takes GGUF experts as stored

    def _entry(self, base):
        e = self.ck.headers.get(base + ".weight")
        if e is None:
            raise KeyError(f"model-template: no matrix {base} in the checkpoint")
        return e

    def has(self, base):
        return base + ".weight" in self.ck

    def format(self, base):
        e = self._entry(base)
        return f"gguf:{e.dtype}" if e.stored else "bf16"

    def stored(self, base, device=None):
        torch = self.torch
        e = self._entry(base)
        if e.stored:
            return f"gguf:{e.dtype}", {"weight": torch.from_numpy(self.ck.rows(base + ".weight"))}
        where = self.device if device is None else device
        return "bf16", {"weight": self.ck.get(base + ".weight", where).to(torch.bfloat16)}

    def stacked(self, bases, device=None):
        torch = self.torch
        found = [self.stored(b, device) for b in bases]
        fmt = found[0][0]
        if any(f != fmt for f, _ in found):
            raise ValueError(f"model-template: {bases} are stored in different formats; stack only one")
        if len(found) == 1:
            return fmt, found[0][1]
        return fmt, {"weight": torch.cat([t["weight"] for _, t in found])}

    def load(self, bases, role, encode=None, rows=None):
        fmt, tensors = self.stacked(bases)
        if rows is not None and tensors["weight"].shape[0] > rows:
            raise ValueError(f"model-template: {bases[0]} stores {tensors['weight'].shape[0]} rows, the head predicts "
                             f"{rows}: a GGUF's rows are taken whole")
        if encode is not None:
            return self.proj.load(tensors, fmt, role=role, encode=encode)
        return self.proj.load(tensors, fmt, role=role)

    def together(self, bases, role, encode=None):
        if len({self.format(b) for b in bases}) == 1 and len({self.ck.shape(b + ".weight")[-1] for b in bases}) == 1:
            return [self.load(list(bases), role, encode)]
        return [self.load([b], role, encode) for b in bases]

    def bias(self, bases):
        names = [b + ".bias" for b in bases]
        if not any(n in self.ck for n in names):
            return None
        if not all(n in self.ck for n in names):
            raise ValueError(f"model-template: only some of {bases} store a bias")
        return self.torch.cat([self.ck.get(n, self.device).float() for n in names])

    def rows(self, base):
        return self._entry(base).shape[0]

    def expert(self, gate, up, down, want=None, device=None):
        """(format, gate_up, down) of one routed expert: bf16 matrices where the file stores them dense or the
        experts plugin takes no GGUF experts (stored_experts: their stored values, rounded to BF16), else "gguf" with
        each part's stored rows and type ({"weight": uint8 [rows][row bytes], "type": "IQ2_S"}): the gate's and the up
        projection's rows stacked (one type), the down projection's as stored. `want` (the format a draft layer's
        experts are re-encoded to) is never given for a GGUF: each of its layers keeps the formats it is stored in.
        device: where decoded values are handed over (the model's by default); stored rows are host bytes."""
        if want is not None:
            raise ValueError(f"model-template: a GGUF's experts keep their stored formats, not {want}")
        torch = self.torch
        if not self.stored_experts:
            where = self.device if device is None else device
            values = [self.ck.get(b + ".weight", where).to(torch.bfloat16) for b in (gate, up, down)]
            return "bf16", torch.cat(values[:2]), values[2]
        fmt, gate_up = self.stacked([gate, up], device)
        down_fmt, down_t = self.stacked([down], device)
        if fmt == "bf16" and down_fmt == "bf16":
            return "bf16", gate_up["weight"], down_t["weight"]
        if fmt == "bf16" or down_fmt == "bf16":
            raise ValueError(f"model-template: {gate} and {down} are stored one dense and one quantized")
        return "gguf", {"weight": gate_up["weight"], "type": fmt[5:]}, {"weight": down_t["weight"], "type": down_fmt[5:]}

    def expert_formats(self, base):
        """(gate_up type, down type) of the routed experts under `base` (a layer's experts prefix), from expert 0."""
        return (self._entry(base + "0.gate_proj").dtype, self._entry(base + "0.down_proj").dtype)


def is_gguf(path):
    """Whether `path` is a GGUF model (a .gguf file, or a folder with one and no safetensors)."""
    from crit_core.model.gguf import SUFFIX, find_gguf
    path = Path(path)
    if path.is_file():
        return path.suffix == SUFFIX
    return path.is_dir() and not any(path.glob("*.safetensors")) and find_gguf(path) is not None


def generation_ends(path):
    """The end tokens of a GGUF's derived generation_config.json (Core's derivation from its metadata), or None."""
    import json
    from crit_core.model import open_model
    try:
        found = json.loads(open_model(path).files.read("generation_config.json")).get("eos_token_id")
    except (OSError, ValueError, KeyError):
        return None
    if found is None:
        return None
    return sorted(set(found if isinstance(found, list) else [found]))


__all__ = ["COMMON", "GgufCheckpoint", "GgufMatrices", "OUTSIDE", "TRANSFORMS", "config_size", "generation_ends",
           "is_gguf", "placed_name", "rules_of"]
