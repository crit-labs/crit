"""plugin.json: what a plugin fills, needs, provides and takes, checked before any of its code runs."""
import json
import math
import re
from pathlib import Path

SCHEMA = "crit-plugin-v1"
MANIFEST = "plugin.json"
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
SLOT = re.compile(r"[a-z][a-z0-9_]{0,63}")
FACT = re.compile(r"[a-z0-9][a-z0-9_.:+-]{0,127}")
ENTRY = re.compile(r"([A-Za-z_][A-Za-z0-9_/]*\.py):([A-Za-z_][A-Za-z0-9_]*)")
SOURCE = re.compile(r"[a-z][a-z0-9_]{0,63}")
OPTION_TYPES = ("int", "float", "bool", "str", "choice")

# The slots crit names, so that plugins written for different engines mean the same thing by a slot. An engine may
# name others; a slot is only a name both its plugins and its engine agree on.
SLOTS = {
    "model": "the architecture: its layer plan and weight names, and the batched step over the other slots that Core's "
             "concurrent runtime runs",
    "placement": "where each tensor lives (the device, host memory, files) at the context the engine serves",
    "attention": "the attention kernels a layer runs over its KV state, in prefill and in decode",
    "kv_cache": "how KV state is stored and paged: its format, page size, prefix reuse",
    "projections": "dense weight formats and their GEMM/GEMV kernels, with fused epilogues",
    "experts": "routed experts: their execution and dispatch",
    "expert_residency": "which routed experts are resident on the device and how a miss is admitted",
    "expert_source": "where a routed expert's weights come from when they are not resident: host memory, files",
    "router": "expert routing: scores, top-k and the routing description experts read",
    "recurrence": "linear-attention or SSM state and its scan kernels",
    "norms": "normalization, rope and other elementwise work around the matmuls",
    "drafter": "draft proposals for speculative decoding",
    "verifier": "the speculative round: verify, accept and repair",
    "sampler": "token sampling and its constraints",
    "graphs": "graph capture and replay of the passes",
    "embeddings": "token embedding and lookup-table rows",
}

# The slots every engine fills. Core knows no model and no device: an engine is its plugins, and without an
# architecture, a placement, its matrices, attention over its KV state and a sampler there is no engine.
REQUIRED_SLOTS = ("model", "placement", "projections", "attention", "kv_cache", "sampler")
# Slots whose plugins decide rather than compute (plans, capture choices, residency, drafts and their acceptance):
# their code may run on the host beside any device, "devices": ["host"]. A plugin in any other slot computes on the
# device the engine serves on. Scheduling is no slot: Core's concurrent runtime schedules every engine's steps (mixed
# prefill and decode, admission, chunks from the model's info), since every engine batches.
POLICY_SLOTS = ("placement", "graphs", "expert_residency", "expert_source", "drafter", "verifier")
# The devices a plugin can run on; "host" is a policy plugin's.
DEVICES = ("cuda", "rocm", "metal", "vulkan", "cpu")
HOST = "host"


class PluginError(ValueError):
    """A plugin, or a selection of plugins, that crit refuses: the message names what to change."""


def _strings(value, what, pattern, where, empty=True):
    if not isinstance(value, list) or (not empty and not value):
        raise PluginError(f"{where}: {what} is a {'' if empty else 'nonempty '}list")
    for item in value:
        if not isinstance(item, str) or not pattern.fullmatch(item):
            raise PluginError(f"{where}: {what} holds {item!r}, not a name matching {pattern.pattern}")
    if len(set(value)) != len(value):
        raise PluginError(f"{where}: {what} names an entry twice")
    return list(value)


def _needs(value, where):
    """needs: facts the assembly must provide, each a fact or a list of facts any one of which will do."""
    if not isinstance(value, list):
        raise PluginError(f"{where}: needs is a list")
    out = []
    for item in value:
        alternatives = item if isinstance(item, list) else [item]
        _strings(alternatives, "a need", FACT, where, empty=False)
        out.append(alternatives)
    return out


def _option(name, spec, where):
    if not SLOT.fullmatch(name) or not isinstance(spec, dict):
        raise PluginError(f"{where}: option {name!r} needs a lowercase name and an object")
    unknown = set(spec) - {"type", "default", "min", "max", "choices", "description"}
    if unknown or "type" not in spec or "default" not in spec or spec["type"] not in OPTION_TYPES:
        raise PluginError(f"{where}: option {name} has a type ({', '.join(OPTION_TYPES)}) and a default, and nothing "
                          f"but min, max, choices and description beside them")
    if spec["type"] == "choice":
        if not isinstance(spec.get("choices"), list) or not spec["choices"]:
            raise PluginError(f"{where}: option {name} of type choice lists its choices")
    elif "choices" in spec:
        raise PluginError(f"{where}: only a choice option has choices")
    if ("min" in spec or "max" in spec) and spec["type"] not in ("int", "float"):
        raise PluginError(f"{where}: only a number option has min and max")
    check_option(name, spec, spec["default"], where)
    return dict(spec)


def check_option(name, spec, value, where):
    """A value for option `name`, as its spec allows."""
    kind = spec["type"]
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    ok = {"int": isinstance(value, int) and not isinstance(value, bool),
          "float": number and math.isfinite(value),
          "bool": isinstance(value, bool),
          "str": isinstance(value, str),
          "choice": value in spec.get("choices", [])}[kind]
    if ok and kind in ("int", "float"):
        ok = spec.get("min", -math.inf) <= value <= spec.get("max", math.inf)
    if not ok:
        allowed = (f"one of {spec['choices']}" if kind == "choice" else
                   kind + (f" in [{spec.get('min', '-inf')}, {spec.get('max', 'inf')}]" if "min" in spec or "max" in spec
                           else ""))
        raise PluginError(f"{where}: option {name} is {allowed}, not {value!r}")
    return value


def _needs_when(value, where):
    """needs_when: fact -> needs (as `needs`) that hold only where the engine or another plugin provides the fact: a
    stored format the plugin decodes only with a device feature ({"stored:nvfp4": ["cuda:fp4"]})."""
    if not isinstance(value, dict):
        raise PluginError(f"{where}: needs_when maps a fact to the needs it brings")
    out = {}
    for fact, needs in value.items():
        if not isinstance(fact, str) or not FACT.fullmatch(fact):
            raise PluginError(f"{where}: needs_when's key {fact!r} is a fact")
        out[fact] = _needs(needs, where)
    return out


def _sources(value, folder, where):
    """sources: name -> folder of code the plugin imports as the subpackage `.name`: its own code folder, or another
    plugin's or technique's, which crit plugins add copies into the engine with it."""
    if not isinstance(value, dict):
        raise PluginError(f"{where}: sources maps subpackage names to folders of code")
    out = {}
    for name, relative in value.items():
        if not isinstance(name, str) or not SOURCE.fullmatch(name) or (folder / f"{name}.py").exists():
            raise PluginError(f"{where}: source {name!r} needs a lowercase identifier that names no file of the plugin")
        if not isinstance(relative, str) or not relative or "\\" in relative or Path(relative).is_absolute():
            raise PluginError(f"{where}: source {name} is a relative folder path with forward slashes")
        path = (folder / relative).resolve()
        if not path.is_dir():
            raise PluginError(f"{where}: source {name} names {relative}, which is no folder")
        out[name] = {"path": relative, "folder": path}
    return out


KERNEL = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


def _kernels(value, where):
    """kernels: crit's kernels the plugin loads (crit_plugins.kernels), each {name: the kernel's id in crit's kernel
    library (kernels/techniques/<family>/<id>), fallback: what the plugin runs when the kernel cannot be built here (a line of
    text), backend: the build it loads where that is not torch's own device backend (cpu for a CPU kernel beside a
    CUDA torch), optional}."""
    if not isinstance(value, list):
        raise PluginError(f"{where}: kernels is a list of {{name, fallback}}")
    out = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"name", "fallback", "backend"} or "name" not in item:
            raise PluginError(f"{where}: a kernel is {{name, fallback}} (and optionally backend)")
        name, fallback = item["name"], item.get("fallback")
        if not isinstance(name, str) or not KERNEL.fullmatch(name):
            raise PluginError(f"{where}: kernel {name!r} is the id of one of crit's kernels (kernels/techniques/<family>/<id>)")
        if not isinstance(fallback, str) or not fallback.strip():
            raise PluginError(f"{where}: kernel {name!r} says what the plugin runs without it (fallback)")
        backend = item.get("backend")
        if backend is not None and backend not in ("cuda", "rocm", "metal", "xpu", "cpu"):
            raise PluginError(f"{where}: kernel {name}'s backend is cuda, rocm, metal, xpu or cpu")
        if any(o["name"] == name for o in out):
            raise PluginError(f"{where}: kernel {name} is named twice")
        out.append({"name": name, "fallback": fallback, "backend": backend})
    return out


def load_manifest(folder):
    """The checked manifest of the plugin in `folder`, with `folder` added. Its id is the folder's name."""
    folder = Path(folder)
    path = folder / MANIFEST
    where = f"{folder.name}/{MANIFEST}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PluginError(f"{folder} holds no {MANIFEST}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PluginError(f"{where} is not readable JSON: {exc}") from None
    if not isinstance(data, dict):
        raise PluginError(f"{where} is a JSON object")
    required = {"schema", "id", "title", "fills", "entry"}
    optional = {"provides", "needs", "needs_when", "conflicts", "options", "reference", "test_with", "sources",
                "tests", "description", "requires", "devices", "kernels"}
    missing, unknown = required - set(data), set(data) - required - optional
    if missing or unknown:
        raise PluginError(f"{where}: " + "; ".join(
            ([f"missing {', '.join(sorted(missing))}"] if missing else []) +
            ([f"unknown {', '.join(sorted(unknown))}"] if unknown else [])))
    if data["schema"] != SCHEMA:
        raise PluginError(f"{where}: schema is {SCHEMA}")
    if not isinstance(data["id"], str) or not ID.fullmatch(data["id"]) or data["id"] != folder.name:
        raise PluginError(f"{where}: id is the folder's name, lowercase letters, digits and dashes")
    if data["id"] == "base":
        raise PluginError(f"{where}: base is the folder of a folder's base plugins, no plugin's id")
    if not isinstance(data["title"], str) or not data["title"].strip() or len(data["title"]) > 200:
        raise PluginError(f"{where}: title is a line of text")
    manifest = {
        "id": data["id"], "title": data["title"], "description": data.get("description", ""),
        "fills": _strings(data["fills"], "fills", SLOT, where, empty=False),
        "provides": _strings(data.get("provides", []), "provides", FACT, where),
        "needs": _needs(data.get("needs", []), where),
        "needs_when": _needs_when(data.get("needs_when", {}), where),
        "conflicts": _strings(data.get("conflicts", []), "conflicts", FACT, where),
        "requires": _strings(data.get("requires", []), "requires", SLOT, where),
        "devices": _strings(data.get("devices", []), "devices", SLOT, where),
        "folder": folder,
    }
    if "model" in manifest["fills"] and manifest["fills"] != ["model"]:
        raise PluginError(f"{where}: a model plugin fills only model; the slots its step calls are plugins of their own")
    if set(manifest["requires"]) & set(manifest["fills"]):
        raise PluginError(f"{where}: requires names the other slots the plugin calls, not its own")
    unknown = [d for d in manifest["devices"] if d not in DEVICES + (HOST,)]
    if unknown:
        raise PluginError(f"{where}: devices holds {', '.join(unknown)}; a plugin runs on {', '.join(DEVICES)} or, "
                          f"in {', '.join(POLICY_SLOTS)} only, on the host")
    computing = [slot for slot in manifest["fills"] if slot not in POLICY_SLOTS]
    if HOST in manifest["devices"] and computing:
        raise PluginError(f"{where}: {', '.join(computing)} computes on the device the engine serves on; only a plugin of "
                          f"{', '.join(POLICY_SLOTS)} runs on the host")
    if not isinstance(manifest["description"], str):
        raise PluginError(f"{where}: description is text")
    if not isinstance(data["entry"], str) or not ENTRY.fullmatch(data["entry"]):
        raise PluginError(f"{where}: entry is FILE.py:FUNCTION, the factory that creates the plugin")
    file, function = data["entry"].rsplit(":", 1)
    if ".." in Path(file).parts or not (folder / file).is_file():
        raise PluginError(f"{where}: entry names {file}, which is no file of the plugin")
    manifest["entry"] = (file, function)
    options = data.get("options", {})
    if not isinstance(options, dict):
        raise PluginError(f"{where}: options is an object of option specs")
    manifest["options"] = {name: _option(name, spec, where) for name, spec in options.items()}
    reference = data.get("reference")
    if reference is not None:
        _strings(reference, "reference", ID, where, empty=False)
        if data["id"] in reference:
            raise PluginError(f"{where}: a plugin is not its own reference")
    manifest["reference"] = reference
    manifest["test_with"] = _strings(data.get("test_with", []), "test_with", ID, where)
    if data["id"] in manifest["test_with"]:
        raise PluginError(f"{where}: test_with names the other plugins its tests use")
    manifest["sources"] = _sources(data.get("sources", {}), folder, where)
    manifest["kernels"] = _kernels(data.get("kernels", []), where)
    tests = data.get("tests", "tests")
    if not isinstance(tests, str) or ".." in Path(tests).parts:
        raise PluginError(f"{where}: tests names the plugin's test folder")
    manifest["tests"] = folder / tests
    for fact in manifest["provides"]:
        if fact in manifest["conflicts"]:
            raise PluginError(f"{where}: {fact} is both provided and in conflicts")
    return manifest


def options_for(manifest, given, where=None):
    """The plugin's options: its defaults with `given` over them, each checked."""
    where = where or manifest["id"]
    given = given or {}
    if not isinstance(given, dict):
        raise PluginError(f"{where}: options are an object")
    unknown = sorted(set(given) - set(manifest["options"]))
    if unknown:
        raise PluginError(f"{where}: {manifest['id']} has no option {', '.join(unknown)}"
                          + (f" (it has {', '.join(sorted(manifest['options']))})" if manifest["options"] else ""))
    out = {}
    for name, spec in manifest["options"].items():
        out[name] = check_option(name, spec, given.get(name, spec["default"]), where)
    return out
