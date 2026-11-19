"""Layer-cut checkpoints: a real checkpoint with only a few of its decoder layers, every kept tensor byte for byte.

A cut keeps the decoder layers that cover every kind of layer the model has, and everything outside the decoder
layers a text model reads: embeddings, head, final norms, the draft (MTP) layer, lookup tables (hashed n-gram rows)
and the residual's own mixers. Vision and audio towers are left out. Its config.json (and its text_config), its
quantization configs (config.json's quantization_config, hf_quant_config.json) and its index are rewritten to the
kept layers: num_hidden_layers, every per-layer list (layer_types and any list as long as the layers), the layer ids
lists (ple_layer_ids, numbered from 1; mlp_only_layers, from 0) and every module name or pattern naming a decoder
layer (renumbered, or dropped with its layer).

A shard all of whose tensors are kept under their own names is hard-linked into the cut (no bytes copied, the same
file); any other shard is written again with its kept tensors only, their bytes copied as stored. So a cut of a
checkpoint whose large lookup table sits in shards of its own costs the disk only its kept layers.

    python tests/support/layer_cut.py SOURCE TARGET [--layers 0,1,2,3]

Without --layers the cut keeps the decoder's first layers up to the first of every layer kind (layer_types), of
every layer a `before` table joins (ple_layer_ids) and of a dense and a routed layer where both exist, so periodic
patterns (one full-attention layer in four) keep their period.
"""
import argparse
import json
import os
import re
import shutil
import struct
from pathlib import Path

LAYER = re.compile(r"(?<![a-z_])layers\.(\d+)(?=\.|\*|$)")
SKIP_TENSORS = re.compile(r"(^|\.)(visual|vision_tower|vision_model|audio_tower|multi_modal_projector|embed_vision|"
                          r"embed_audio)\.")
DRAFT = ("mtp.",)
INDEX = "model.safetensors.index.json"


def _header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    return 8 + size, header


def text_config(config):
    """The config's text model's part: its text_config where it has one."""
    return config.get("text_config") or config


def default_layers(config):
    """The decoder's first layers up to the first of every kind of layer (see the module's docstring)."""
    text = text_config(config)
    count = int(text["num_hidden_layers"])
    wanted = set()
    kinds = text.get("layer_types") or []
    for kind in dict.fromkeys(kinds):
        wanted.add(kinds.index(kind))
    for one_based in text.get("ple_layer_ids") or []:
        wanted.add(int(one_based) - 1)
    dense = set(text.get("mlp_only_layers") or [])
    if dense and len(dense) < count:
        wanted.add(min(dense))
        wanted.add(min(set(range(count)) - dense))
    if text.get("first_k_dense_replace"):
        wanted.add(int(text["first_k_dense_replace"]))   # the first routed layer
    last = max(wanted) if wanted else 0
    return list(range(last + 1))


class Cut:
    """The renumbering of a cut: kept layer ids (in order) -> their new ids."""

    def __init__(self, layers, count, draft_prefixes=DRAFT):
        self.layers = sorted(set(int(i) for i in layers))
        if not self.layers or self.layers[-1] >= count:
            raise ValueError(f"layers {self.layers} of a decoder of {count}")
        self.new = {old: new for new, old in enumerate(self.layers)}
        self.count, self.draft = count, tuple(draft_prefixes)

    def name(self, name):
        """A tensor or module name under the cut: renamed, or None where it names a layer that goes."""
        if name.startswith(self.draft) or name.startswith(tuple("." + d for d in self.draft)):
            return name   # the draft layer's own layers.0 is not the decoder's
        found = LAYER.search(name)
        if found is None:
            return name
        old = int(found.group(1))
        if old not in self.new:
            return None
        return name[:found.start(1)] + str(self.new[old]) + name[found.end(1):]

    def strings(self, value):
        """Every module name or pattern inside a quantization config renamed, those of cut layers dropped."""
        if isinstance(value, str):
            return self.name(value)
        if isinstance(value, list):
            out = [self.strings(v) for v in value]
            return [v for v, raw in zip(out, value) if not (isinstance(raw, str) and v is None)]
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                renamed = self.name(key) if LAYER.search(key) else key
                if renamed is not None:
                    out[renamed] = self.strings(item)
            return out
        return value

    def config(self, config):
        """config.json under the cut."""
        config = json.loads(json.dumps(config))
        text = text_config(config)
        count = int(text["num_hidden_layers"])
        for key, value in list(text.items()):
            if isinstance(value, list) and len(value) == count and key not in ("ple_layer_ids", "mlp_only_layers"):
                text[key] = [value[i] for i in self.layers]
        text["num_hidden_layers"] = len(self.layers)
        if "ple_layer_ids" in text:
            text["ple_layer_ids"] = [self.new[i - 1] + 1 for i in text["ple_layer_ids"] if i - 1 in self.new]
        if "mlp_only_layers" in text:
            text["mlp_only_layers"] = [self.new[i] for i in text["mlp_only_layers"] if i in self.new]
        for holder in ((config, text) if text is not config else (config,)):
            if isinstance(holder.get("quantization_config"), dict):
                holder["quantization_config"] = self.strings(holder["quantization_config"])
        return config


def _write_shard(source, target, renames):
    """A shard of the kept tensors of `source` ({old name: new name}), their bytes copied as stored."""
    begin, header = _header(source)
    metadata = header.pop("__metadata__", None)
    kept = sorted(renames, key=lambda n: header[n]["data_offsets"][0])
    out, at = {}, 0
    for name in kept:
        first, last = header[name]["data_offsets"]
        out[renames[name]] = {"dtype": header[name]["dtype"], "shape": header[name]["shape"],
                              "data_offsets": [at, at + last - first]}
        at += last - first
    if metadata is not None:
        out["__metadata__"] = metadata
    data = json.dumps(out, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    with open(source, "rb") as src, open(target, "wb") as dst:
        dst.write(struct.pack("<Q", len(data)) + data)
        for name in kept:
            first, last = header[name]["data_offsets"]
            src.seek(begin + first)
            left = last - first
            while left:
                chunk = src.read(min(left, 64 << 20))
                dst.write(chunk)
                left -= len(chunk)


def cut(source, target, layers=None, link=True):
    """Write the cut of checkpoint `source` into `target` (a new folder); returns a summary dict."""
    source, target = Path(source), Path(target)
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"{target} is not empty")
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    count = int(text_config(config)["num_hidden_layers"])
    plan = Cut(default_layers(config) if layers is None else layers, count)
    index = source / INDEX
    shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())) if index.is_file() \
        else sorted(p.name for p in source.glob("*.safetensors"))
    shards += sorted(p.relative_to(source).as_posix() for p in source.glob("*.safetensors")
                     if p.relative_to(source).as_posix() not in shards)
    weight_map, linked, written, copied = {}, [], [], 0
    for shard in shards:
        _, header = _header(source / shard)
        header.pop("__metadata__", None)
        renames = {}
        for name in header:
            new = None if SKIP_TENSORS.search(name) else plan.name(name)
            if new is not None:
                renames[name] = new
        if not renames:
            continue
        (target / shard).parent.mkdir(parents=True, exist_ok=True)
        if len(renames) == len(header) and all(k == v for k, v in renames.items()):
            if link:
                try:
                    os.link(source / shard, target / shard)
                    linked.append(shard)
                except OSError:
                    shutil.copyfile(source / shard, target / shard)
                    written.append(shard)
            else:
                shutil.copyfile(source / shard, target / shard)
                written.append(shard)
        else:
            _write_shard(source / shard, target / shard, renames)
            written.append(shard)
        copied += (target / shard).stat().st_size if shard in written else 0
        weight_map.update({new: shard for new in renames.values()})
    total = sum((target / s).stat().st_size for s in set(weight_map.values()))
    (target / INDEX).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(
        weight_map.items()))}, indent=2), encoding="utf-8")
    (target / "config.json").write_text(json.dumps(plan.config(config), indent=2), encoding="utf-8")
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in (INDEX, "config.json"):
            if path.name == "hf_quant_config.json":
                found = json.loads(path.read_text(encoding="utf-8"))
                (target / path.name).write_text(json.dumps(plan.strings(found), indent=2), encoding="utf-8")
            elif path.stat().st_size < (256 << 20):
                shutil.copyfile(path, target / path.name)
    summary = {"source": str(source), "layers": plan.layers, "tensors": len(weight_map), "linked": linked,
               "written": written, "bytes": total, "bytes_written": copied}
    (target / "layer_cut.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a layer-cut copy of a safetensors checkpoint")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--layers", help="decoder layer ids to keep, comma-separated (default: see the docstring)")
    parser.add_argument("--copy", action="store_true", help="copy whole shards instead of hard-linking them")
    args = parser.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    summary = cut(args.source, args.target, layers, link=not args.copy)
    print(json.dumps({**summary, "linked": len(summary["linked"]), "written": len(summary["written"]),
                      "gib": round(summary["bytes"] / 2 ** 30, 1),
                      "gib_written": round(summary["bytes_written"] / 2 ** 30, 1)}))


if __name__ == "__main__":
    raise SystemExit(main())
