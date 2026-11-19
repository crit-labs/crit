"""Checkpoint manifests: what a real checkpoint stores, without its weights, small enough to keep in crit's tests.

A manifest holds a checkpoint's config.json, its quantization config (hf_quant_config.json where it keeps one), its
generation config, and every tensor's name, dtype and shape from the safetensors headers alone, so a test can plan a
model over the real checkpoint's tensors on a machine that has none of them (tests/test_checkpoint_manifests.py).

Names are kept compact, and the shards' layout is not kept: every run of digits in a name is a field, and the tensors
that share the rest of their name and their dtype are one group, whose fields are stored as ranges where they form a
full grid of one shape (a layer's 512 experts, 48 layers) and as a list otherwise (with each member's shape where
they differ: a compressed scale's exceptions). Small tensors that the model reads values of (a hash's multipliers
and offsets, a table's scale) keep their bytes: those of at most VALUE_BYTES in a group of at most VALUE_GROUP.

    python tests/support/manifests.py /llm/models/crit/nvidia-Qwen3.8-Flash-Next-NVFP4 --out tests/manifests

writes tests/manifests/<folder name>.json.gz from the headers (it reads 8 bytes and the header of each shard, never
a weight).
"""
import argparse
import base64
import gzip
import itertools
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "crit-checkpoint-manifest-v1"
VALUE_BYTES = 1024
VALUE_GROUP = 64
CONFIGS = ("config.json", "hf_quant_config.json", "generation_config.json", "quantization_config.json")
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I16": 2, "U16": 2, "F16": 2,
               "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
FIELD = re.compile(r"\d+")
SHARD = "model.safetensors"   # a manifest keeps no file layout: its tensors read as one file's
HERE = Path(__file__).resolve().parents[1] / "manifests"


def checkpoint_folder(folder):
    """The folder holding the checkpoint's config.json: `folder` itself, or the one revision folder below it (a hub
    snapshot kept as <name>/<revision>/)."""
    folder = Path(folder)
    if (folder / "config.json").is_file():
        return folder
    found = [p.parent for p in folder.glob("*/config.json")]
    if len(found) != 1:
        raise ValueError(f"{folder}: no config.json here or in exactly one folder below it")
    return found[0]


def _header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return 8 + size, header


def _shards(folder):
    """The checkpoint's safetensors files: those its index names and those beside it the index does not list."""
    index = folder / "model.safetensors.index.json"
    listed = []
    if index.is_file():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        listed = [folder / n for n in names]
    extra = sorted(p for p in folder.glob("*.safetensors") if p not in listed)
    return listed + extra


def _ranges(values):
    """Sorted ints as [[first, last], ...] runs."""
    runs = []
    for v in sorted(values):
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return runs


def _expand(runs):
    return [v for first, last in runs for v in range(first, last + 1)]


def read(folder, name=None):
    """The manifest of the checkpoint in `folder` (a dict, as written)."""
    folder = checkpoint_folder(folder)
    configs = {c: json.loads((folder / c).read_text(encoding="utf-8")) for c in CONFIGS if (folder / c).is_file()}
    groups, values = {}, {}
    for shard in _shards(folder):
        begin, header = _header(shard)
        for tensor, info in header.items():
            fields = tuple(int(v) for v in FIELD.findall(tensor))
            key = (FIELD.sub("{}", tensor), info["dtype"])
            groups.setdefault(key, []).append((fields, tuple(info["shape"])))
            start, end = info["data_offsets"]
            values[tensor] = (shard, begin + start, end - start)
    out = []
    for (pattern, dtype), members in sorted(groups.items()):
        members.sort()
        group = {"name": pattern, "dtype": dtype}
        shapes = {shape for _, shape in members}
        fields = [f for f, _ in members]
        if len(shapes) == 1:
            group["shape"] = list(shapes.pop())
            if fields[0]:
                axes = [sorted({f[i] for f in fields}) for i in range(len(fields[0]))]
                if math.prod(len(a) for a in axes) == len(fields):
                    group["grid"] = [_ranges(a) for a in axes]
                else:
                    group["fields"] = fields
        else:   # one shape a member (a compressed scale's exceptions, a table's last shard)
            group["fields"], group["shapes"] = fields, [list(shape) for _, shape in members]
        if len(members) <= VALUE_GROUP:
            stored = {}
            for f, shape in members:
                if math.prod(shape) * DTYPE_BYTES[dtype] > VALUE_BYTES:
                    continue
                tensor = pattern.format(*f)
                path, at, count = values[tensor]
                with open(path, "rb") as file:
                    file.seek(at)
                    stored[tensor] = base64.b64encode(file.read(count)).decode()
            if stored:
                group["values"] = stored
        out.append(group)
    return {"schema": SCHEMA, "name": name or _name(folder), "configs": configs, "groups": out}


def _name(folder):
    """A checkpoint's name: its folder's, or the folder above a revision folder (a 40-hex-digit name)."""
    return folder.parent.name if re.fullmatch(r"[0-9a-f]{40}", folder.name) else folder.name


def write(manifest, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    with gzip.GzipFile(out, "wb", mtime=0) as f:   # mtime 0: the same checkpoint writes the same bytes
        f.write(data)
    return out


def load(path):
    with gzip.open(path, "rb") as f:
        manifest = json.loads(f.read())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA}")
    return manifest


def available():
    """The manifests kept in tests/manifests, by name."""
    return {p.name[:-len(".json.gz")]: p for p in sorted(HERE.glob("*.json.gz"))}


@dataclass(frozen=True)
class Tensor:
    """A tensor as a manifest describes it: what crit_core.model.safetensors.TensorInfo says of a header entry."""
    name: str
    dtype: str
    shape: tuple
    shard: Path
    data_begin: int
    data_end: int
    value: bytes = None

    @property
    def nbytes(self):
        return self.data_end - self.data_begin


def tensors(manifest, root=Path("/manifest")):
    """{name: Tensor} of every tensor the manifest lists, as one file's (SHARD under `root`) with offsets laid out in
    name order: a manifest keeps no file layout, and offsets only have to be distinct and consistent."""
    found = []
    for group in manifest["groups"]:
        if "grid" in group:
            members = itertools.product(*(_expand(a) for a in group["grid"]))
        elif "fields" in group:
            members = group["fields"]
        else:
            members = [()]
        shapes = group.get("shapes") or itertools.repeat(group.get("shape"))
        stored = group.get("values") or {}
        for fields, shape in zip(members, shapes):
            name = group["name"].format(*fields)
            value = stored.get(name)
            found.append((SHARD, name, group["dtype"], tuple(shape), math.prod(shape) * DTYPE_BYTES[group["dtype"]],
                          base64.b64decode(value) if value is not None else None))
    out, offsets = {}, {}
    for shard, name, dtype, shape, nbytes, value in sorted(found):
        at = offsets.get(shard, 0)
        offsets[shard] = at + nbytes
        out[name] = Tensor(name, dtype, shape, root / shard, at, at + nbytes, value)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write checkpoint manifests from safetensors headers")
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args(argv)
    for folder in args.folders:
        manifest = read(folder)
        path = write(manifest, Path(args.out) / f"{manifest['name']}.json.gz")
        count = sum(1 for _ in tensors(manifest))
        print(f"{path}: {count} tensors in {len(manifest['groups'])} groups, {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    raise SystemExit(main())
