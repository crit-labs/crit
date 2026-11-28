"""A model plugin's load over a checkpoint manifest: every tensor the composed model reads, planned and loaded the way
a real load does it, without the weights or a GPU.

plan(manifest, placement) runs the ComposedModel of the finished model plugin that serves the manifest's model_type
(models/cache; the models core's compose() over the plugin's layout) itself over a ManifestCheckpoint (the manifest's
headers: tensors_dtype, shape and size) on the CPU, with the slots filled by stand-ins that take what the real plugins
take and refuse the rest as they would:

- projections: quantized-projections' load formats (its device_matrices.FORMATS) and its re-encodings (ENCODINGS);
- experts: expert-offload's rule (one format for every layer begun, W4A4 and W4A16 not mixed in a layer) over the
  formats grouped-experts packs (its bundles' _KINDS);
- embeddings: lookup-table-rows' formats (BF16/F16/F32 rows or NVFP4 rows with their scales), on the device or read
  where the checkpoint stores them;
- placement: device-placement's rule, tables off the device where `placement` is "tables-off".

A tensor comes back with the manifest's bytes where it keeps them (a hash's multipliers), a scalar (a global scale)
as ones, and any other as a meta tensor of its dtype and shape (no bytes). The report lists each tensor the model read, the formats each
slot was given, every tensor of the checkpoint no part of the model read (outside the parts a text model never reads,
UNREAD_OK), and the error that stopped the load, if any.
"""
import importlib
import importlib.util
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for _folder in ("inference/core/python", "cli"):
    if str(ROOT / _folder) not in sys.path:
        sys.path.insert(0, str(ROOT / _folder))

from support import manifests  # noqa: E402

# Parts of a checkpoint a text decoder never reads: vision and audio towers and their projectors.
UNREAD_OK = re.compile(r"^(model\.)?(visual|vision_tower|vision_model|audio_tower|audio_model|multi_modal_projector|"
                       r"embed_vision|embed_audio|mm_projector|vision_embed)\.|^(model\.)?language_model\.(visual|"
                       r"vision_tower)\.")
SCALAR = 16              # bytes: a tensor up to this size comes back with values (the manifest's, else ones)
TORCH = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
         "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool", "F8_E4M3": "float8_e4m3fn",
         "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu", "U32": "uint32", "U16": "uint16", "U64": "uint64"}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def family_compose(folder):
    """The compose module of the model plugin in `folder` as its package imports it, the core beside it and the
    plugin's own techniques mounted (blocks.own_sources: a family's blocks and drafters its layout may name), as
    Core's loader composes it."""
    from crit_plugins.assemble import plugin_package
    from crit_plugins.manifest import load_manifest
    return importlib.import_module(f"{plugin_package(load_manifest(Path(folder))).__name__}.core.template.compose")


def _torch_norms():
    """The norms slot's torch form (fused-norms' torch_norms): a block that turns a probe row at load (a family's
    shared-KV attention counts its rotated dimensions so) runs it on the CPU."""
    name = "dry_torch_norms"
    return sys.modules.get(name) or _module(name, ROOT / "inference/techniques/norms/fused-norms/code/torch_norms.py")


def _constants(path, *names):
    """Module-level constants of a plugin's source, read without importing it (its imports need a GPU build)."""
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found


PROJECTION = _constants(ROOT / "inference/techniques/projections/quantized-projections/code/device_matrices.py", "FORMATS", "ENCODINGS")
EXPERT_FORMATS = tuple(_constants(ROOT / "inference/techniques/experts/grouped-experts/code/bundles.py", "_KINDS")["_KINDS"]) + ("bf16",)
TABLE_FORMATS = _constants(ROOT / "inference/techniques/memory/lookup-table-rows/code/device_rows.py", "FORMATS")["FORMATS"]


class ManifestCheckpoint:
    """The models core's Checkpoint over a manifest: its headers, and tensors without bytes."""

    def __init__(self, manifest, folder):
        import torch
        self.torch, self.folder = torch, Path(folder)
        self.headers = manifests.tensors(manifest, root=self.folder)
        self.read, self.opened = set(), set()
        self.at = {(str(t.shard), t.data_begin): name for name, t in self.headers.items()}

    def __contains__(self, name):
        return name in self.headers

    def names(self):
        return self.headers.keys()

    def nbytes(self, name):
        return self.headers[name].nbytes

    def shape(self, name):
        return tuple(self.headers[name].shape)

    def get(self, name, device):
        torch = self.torch
        info = self.headers[name]
        self.read.add(name)
        dtype = getattr(torch, TORCH[info.dtype])
        if info.value is not None:
            return torch.frombuffer(bytearray(info.value), dtype=dtype).reshape(info.shape).clone()
        if info.nbytes <= SCALAR:
            if dtype.is_floating_point and dtype.itemsize == 1:
                return torch.zeros(info.shape, dtype=torch.uint8).view(dtype)
            return torch.ones(info.shape, dtype=dtype) if dtype.is_floating_point else torch.zeros(info.shape,
                                                                                                   dtype=dtype)
        return torch.empty(info.shape, dtype=dtype, device="meta")

    def files(self, source):
        """Mark the tensors a {paths, offsets} file source names as read (a table read where it is stored)."""
        for path, offset in zip(source["paths"], source["offsets"]):
            name = self.at.get((str(path), int(offset)))
            if name is not None:
                self.opened.add(name)

    def close(self):
        pass


class Recorder:
    """What a stand-in was given: {slot: {format: count}}, and the problems a real plugin would have raised."""

    def __init__(self):
        self.formats, self.problems = {}, []

    def took(self, slot, fmt):
        found = self.formats.setdefault(slot, {})
        found[str(fmt)] = found.get(str(fmt), 0) + 1

    def refuse(self, message):
        raise ValueError(message)


class _Loaded:
    def __init__(self, fmt, tensors):
        self.format, self.rows = fmt, int(tensors["weight"].shape[0]) if "weight" in tensors else 0


class Projections:
    def __init__(self, rec):
        self.rec = rec

    def load(self, tensors, fmt, role=None, encode=None):
        if fmt not in PROJECTION["FORMATS"]:
            self.rec.refuse(f"quantized-projections loads {', '.join(PROJECTION['FORMATS'])}, not {fmt!r} ({role})")
        if encode is not None and fmt not in PROJECTION["ENCODINGS"].get(encode, ()):
            self.rec.refuse(f"quantized-projections does not re-encode {fmt} to {encode}")
        self.rec.took("projections", fmt if encode is None else f"{fmt}->{encode}")
        return _Loaded(fmt, tensors)

    def facts(self):
        return {}


class Embeddings:
    def __init__(self, rec, ck):
        self.rec, self.ck = rec, ck

    def store(self, rows, format=None, scales=None, scale=1.0, dtype=None, device=None):
        torch = self.ck.torch
        if format not in TABLE_FORMATS:
            self.rec.refuse(f"lookup-table-rows stores rows of {TABLE_FORMATS}, not {format!r}")
        if format is None and rows.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            self.rec.refuse(f"lookup-table-rows stores 16- or 32-bit rows, not {rows.dtype}")
        self.rec.took("embeddings", format or str(rows.dtype).replace("torch.", ""))
        return SimpleNamespace(lookup=None)

    def open(self, source, width, dtype, device, mapped=False, format=None, scales=None, scale=1.0):
        if format not in TABLE_FORMATS:
            self.rec.refuse(f"lookup-table-rows opens rows of {TABLE_FORMATS}, not {format!r}")
        for item in (source, scales):
            if isinstance(item, dict):
                self.ck.files(item)
        self.rec.took("embeddings", f"open:{format or str(dtype).replace('torch.', '')}")
        return SimpleNamespace(lookup=None)


class Experts:
    """expert-offload's load rule over grouped-experts' bundle formats."""
    complete = False
    round_counters = ()

    def __init__(self, rec):
        self.rec, self.shape, self.a4, self.begun = rec, None, {}, {}

    def activate(self, name):
        pass

    def begin(self, layers, experts, format, hidden, inter, device_bytes=0, identity=None, device="cuda", **more):
        if format not in EXPERT_FORMATS:
            self.rec.refuse(f"expert-offload packs {', '.join(EXPERT_FORMATS)} experts, not {format!r}")
        self.shape = (layers, experts, format, hidden, inter)
        self.begun = {"layers": layers, "experts": experts, "format": format,
                      **{k: v for k, v in more.items() if k in ("decode_rows", "top")}}

    def load(self, layer, expert, gate_up, down, format="bf16"):
        if self.shape is None:
            self.rec.refuse("experts load before begin()")
        if format != self.shape[2]:
            self.rec.refuse(f"expert-offload: the experts were begun as {self.shape[2]}, not {format} (layer "
                            f"{layer})")
        if format == "nvfp4":
            carries = all(isinstance(m, dict) and m.get("input_scale") is not None for m in (gate_up, down))
            if self.a4.setdefault(int(layer), carries) != carries:
                self.rec.refuse(f"layer {layer}'s nvfp4 experts mix W4A4 and W4A16")
        self.rec.took("experts", format)

    def open(self, lend=None):
        pass

    def spare_bytes(self, pool_bytes):
        return 0

    def stages_prompts(self):
        return True


class _Pool:
    def __init__(self, layers, kv_heads=1, head_dim=1, value_dim=None):
        self.layers, self.kv_heads, self.head_dim = list(layers), kv_heads, head_dim
        self.value_dim = head_dim if value_dim is None else value_dim

    def set_scales(self, *args):
        pass


class KvCache:
    def pool(self, layers, kv_heads, head_dim, tokens, device, dtype, page=None, value_dim=None, **extra):
        return _Pool(layers, kv_heads, head_dim, value_dim)


class Recurrence:
    def pool(self, layers, *args, **kwargs):
        return _Pool(layers)


class Placement:
    def __init__(self, tables_off):
        self.tables_off = tables_off

    def plan(self, tensors, room):
        return {name: ("host" if self.tables_off and name.startswith("tables.") else "device") for name in tensors}


class Router:
    routing = None


class Drafter:
    uses_head = True


class Parts:
    """The slots as a model plugin's factory sees them, filled by the stand-ins."""

    PLUGINS = {"projections": "quantized-projections", "attention": "partitioned-attention",
               "kv_cache": "attention-state", "sampler": "token-sampling", "norms": "fused-norms",
               "embeddings": "lookup-table-rows", "recurrence": "sequential-recurrence", "router": "topk-router",
               "experts": "expert-offload", "drafter": "greedy-head-drafter", "placement": "device-placement"}

    def __init__(self, instances, facts, context, plugins):
        self.instances, self._facts, self.context, self.plugins = instances, facts, context, plugins

    def slots(self):
        return list(self.instances)

    def plugin(self, slot):
        return self.plugins.get(slot) if slot in self.instances else None

    def facts(self, slot):
        return tuple(self._facts.get(slot, ())) if slot in self.instances else ()

    def __getitem__(self, slot):
        return self.instances[slot]

    def get(self, slot, default=None):
        return self.instances.get(slot, default)


def _facts(plugin):
    path = next(iter(ROOT.glob(f"inference/**/{plugin}/plugin.json")), None)
    return json.loads(path.read_text(encoding="utf-8")).get("provides", []) if path else []


# The library's plugins a slot may take, in the order the planner tries them: the first whose facts the layout's
# blocks accept (partitioned-attention has no sliding windows, softmax-attention computes them).
ATTENTION = ("partitioned-attention", "softmax-attention")


def _shapes_only(real):
    """crit_plugins.encodings.encode over meta tensors: the encoded tensors' dtypes and shapes, no values (its
    arithmetic is tested on real values, tests/test_encodings.py)."""
    def encode(tensors, format, encode, device=None):
        import torch
        weight = tensors["weight"]
        if weight.device.type != "meta":
            return real(tensors, format, encode, device)
        rows, columns = weight.shape
        meta = dict(device="meta")
        if encode == "fp8":
            return {"weight": torch.empty(rows, columns, dtype=torch.float8_e4m3fn, **meta),
                    "scale": torch.empty(rows, **meta)}, "fp8:channel"
        return {"weight": torch.empty(rows, columns // 2, dtype=torch.uint8, **meta),
                "scale": torch.empty(rows, columns // 16, dtype=torch.uint8, **meta),
                "global_scale": torch.tensor(1.0)}, "nvfp4"
    return encode


def _csf_shapes_only(real):
    """crit_plugins.device.csf_decode over meta tensors: the decoded scales' dtype and shape, no values (its arithmetic
    is tested on real values, device-expert-cache's tests/test_csf.py). The decode's dozen tensor operations cost
    3 ms a call on meta tensors, and a checkpoint with CSF scales decodes one per expert matrix: 74,000 calls, 110 s
    of the 112 s its plan took."""
    def decode(fixed, exceptions=None):
        import torch
        if fixed.device.type != "meta":
            return real(fixed, exceptions)
        slabs, width = tuple(fixed.shape[-2:])
        if width % 16 or width < 32:
            raise ValueError("CSF fixed rows are 16 * (1 + groups / 2) bytes")
        return torch.empty(*fixed.shape[:-2], slabs * 16, (width // 16 - 1) * 2, dtype=torch.uint8, device="meta")
    return decode


def plan(manifest, placement="tables-on", options=None):
    """{model_type, plugin, read, unread, formats, error, experts} of the load of the model plugin that serves
    `manifest` (plugin: its id, None where no finished model plugin serves the model_type; experts:
    what expert-offload's begin() was given and each layer's W4A4 or not), with the first
    attention plugin of ATTENTION whose facts the layout's blocks accept."""
    from crit_plugins import device, encodings
    real, real_csf = encodings.encode, device.csf_decode
    encodings.encode, device.csf_decode = _shapes_only(real), _csf_shapes_only(real_csf)
    try:
        for attention in ATTENTION:
            report = _plan(manifest, placement, options, attention)
            if "the attention plugin does not provide" not in (report["error"] or ""):
                break
    finally:
        encodings.encode, device.csf_decode = real, real_csf
    return report


def _plan(manifest, placement, options, attention):
    import torch
    torch.set_default_device(None)
    config = manifest["configs"].get("config.json") or {}
    report = {"name": manifest["name"], "model_type": config.get("model_type"), "placement": placement,
              "plugin": None, "error": None, "formats": {}, "read": 0, "unread": []}
    from crit.models import plugin_for
    types = [t for t in ((config.get("text_config") or {}).get("model_type"), config.get("model_type")) if t]
    found = plugin_for(types)
    if found is None:
        report["error"] = f"no finished model plugin serves model_type {config.get('model_type')!r}"
        report["covered"] = False
        return report
    compose = family_compose(found["folder"])   # the family's own techniques are blocks its layout may name
    layout = compose.read(found["folder"] / compose.LAYOUT)
    report["covered"], report["plugin"] = True, found["id"]
    rec = Recorder()
    with tempfile.TemporaryDirectory() as folder:
        for name, data in manifest["configs"].items():
            Path(folder, name).write_text(json.dumps(data), encoding="utf-8")
        ck = ManifestCheckpoint(manifest, folder)
        # a draft layer in the checkpoint (a companion drafter's folder lies beside it, which a manifest has not)
        has_draft = bool((layout.get("draft") or {}).get("prefix")) and any(
            n.startswith(layout["draft"]["prefix"]) for n in ck.names())
        context = SimpleNamespace(config=config, model=Path(folder), context_tokens=4096, device="cpu",
                                  checkpoint=ck, memory=None, cache=None, package=None)
        instances = {"projections": Projections(rec), "attention": SimpleNamespace(),
                     "kv_cache": KvCache(), "sampler": SimpleNamespace(), "norms": _torch_norms(),
                     "embeddings": Embeddings(rec, ck), "recurrence": Recurrence(), "router": Router(),
                     "experts": Experts(rec), "placement": Placement(placement == "tables-off")}
        if has_draft:
            instances["drafter"] = Drafter()
        from crit_models.config import architecture
        if not architecture(config).routed_layers():   # a dense model's selection has no router or experts
            del instances["router"], instances["experts"]
        plugins = {**Parts.PLUGINS, "model": found["id"], "attention": attention}
        facts = {slot: _facts(plugins[slot]) for slot in instances}
        parts = Parts(instances, facts, context, plugins)
        report["plugins"] = {slot: plugins[slot] for slot in instances}
        chosen = {"rounds": False, "prompt_buckets": "", "span_seconds": 0.0, "prompt_cache": 0, "passes": "off",
                  "mtp": has_draft, **(options or {})}
        try:
            compose.compose(context, chosen, parts, folder=found["folder"])
        except Exception as error:   # the load's own refusal: the report names it and where it was raised
            frames = traceback.extract_tb(error.__traceback__)
            where = next((f"{Path(f.filename).name}:{f.lineno}" for f in reversed(frames)
                          if "model-template" in f.filename or "dry_template" in f.filename), "")
            report["error"] = f"{type(error).__name__}: {error} ({where})"
        report["formats"] = rec.formats
        experts = instances.get("experts")
        if experts is not None:   # what begin() was given, and which layers' nvfp4 experts carry input scales
            report["experts"] = {**experts.begun, "a4": dict(experts.a4)}
        seen = ck.read | ck.opened
        report["read"] = len(seen)
        if report["error"] is None:
            report["unread"] = sorted(n for n in ck.names() if n not in seen and not UNREAD_OK.match(n))
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Plan the model plugins' loads over checkpoint manifests")
    parser.add_argument("names", nargs="*", help="manifest names (default: every manifest in tests/manifests)")
    parser.add_argument("--placement", default="tables-on", choices=("tables-on", "tables-off"))
    args = parser.parse_args(argv)
    found = manifests.available()
    for name in args.names or found:
        report = plan(manifests.load(found[name]), args.placement)
        unread = report["unread"]
        print(json.dumps({**report, "unread": unread[:8], "unread_count": len(unread)}, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
