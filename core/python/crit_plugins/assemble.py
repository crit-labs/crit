"""An engine's plugins put together when it loads: which plugin fills each slot, whether their needs, conflicts and
memory hold, and the order they are created in. Everything here runs once per load, never per token."""
from collections import OrderedDict
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

from .manifest import HOST, ID, MANIFEST, REQUIRED_SLOTS, SLOT, SLOTS, PluginError, load_manifest, options_for
from .device import DEVICE_EXTENSIONS, POOL_EXTENSIONS, device_missing, extension_missing
from .observe import PLUGIN_MODULES, Observed, unobserve

# The DEVICE_EXTENSIONS facts whose methods live on an object the device form returns (a pool), which the assembly cannot
# see before the model plugin makes one: PluginTestCase.assertImplements checks those.
POOL_FACTS = tuple(POOL_EXTENSIONS)

SELECTION_KEYS = {"use", "base", "options"}


BASE = "base"   # a folder of plugins keeps its base plugins, the plain plugin of each slot, in base/


def plugin_folders(root):
    """The plugin folders of a folder of plugins: each subfolder holding a plugin.json, then those of base/."""
    root = Path(root)
    found = sorted(p for p in root.iterdir() if (p / MANIFEST).is_file())
    base = root / BASE
    if base.is_dir() and not (base / MANIFEST).is_file():
        found += sorted(p for p in base.iterdir() if (p / MANIFEST).is_file())
    return found


def plugins_root(folder):
    """The folder of plugins a plugin's folder lies in: its parent, or the parent's parent for a base plugin."""
    parent = Path(folder).resolve().parent
    return parent.parent if parent.name == BASE and not (parent / MANIFEST).is_file() else parent


class Registry:
    """The plugins an engine ships, by id: every folder holding a plugin.json in the given folders of plugins or in
    their base/ folders."""

    def __init__(self, manifests):
        self.manifests = OrderedDict((m["id"], m) for m in manifests)
        self._check_references()

    @classmethod
    def discover(cls, *roots):
        found = {}
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                raise PluginError(f"{root} is no folder of plugins")
            for folder in plugin_folders(root):
                manifest = load_manifest(folder)
                if manifest["id"] in found:
                    raise PluginError(f"two plugins are named {manifest['id']}: {found[manifest['id']]['folder']} and "
                                      f"{folder}")
                found[manifest["id"]] = manifest
        return cls(found.values())

    def __contains__(self, plugin_id):
        return plugin_id in self.manifests

    def __getitem__(self, plugin_id):
        try:
            return self.manifests[plugin_id]
        except KeyError:
            raise PluginError(f"no plugin {plugin_id}" + (f" (the engine has {', '.join(self.manifests)})"
                                                         if self.manifests else " (the engine has none)")) from None

    def __iter__(self):
        return iter(self.manifests.values())

    def __len__(self):
        return len(self.manifests)

    def _check_references(self):
        """A plugin's reference: plugins of the registry that fill its slots between them, one slot each (for a fused
        plugin the separate plugins it equals, for another the plugin its tests compare it with); and the plugins its
        tests use (test_with) are there."""
        for manifest in self.manifests.values():
            for plugin_id in manifest["test_with"]:
                if plugin_id not in self.manifests:
                    raise PluginError(f"{manifest['id']}: its tests use {plugin_id}, which is no plugin of the engine "
                                      "(crit plugins add copies a plugin with the plugins its tests use)")
            if manifest["reference"] is None:
                continue
            filled = []
            for plugin_id in manifest["reference"]:
                if plugin_id not in self.manifests:
                    raise PluginError(f"{manifest['id']}: its reference names {plugin_id}, which is no plugin of the "
                                      "engine (crit plugins add copies a plugin with its reference)")
                other = self.manifests[plugin_id]
                if len(manifest["fills"]) > 1 and len(other["fills"]) >= len(manifest["fills"]):
                    raise PluginError(f"{manifest['id']}: its reference {plugin_id} is fused as well; name the "
                                      "separate plugins")
                filled += other["fills"]
            if sorted(filled) != sorted(manifest["fills"]):
                raise PluginError(f"{manifest['id']} fills {', '.join(manifest['fills'])}; its reference plugins fill "
                                  f"{', '.join(filled)}: they must fill the same slots, each once")

    def reference_use(self, plugin_id):
        """The `use` map of a plugin's reference plugins in its slots."""
        manifest = self[plugin_id]
        if manifest["reference"] is None:
            raise PluginError(f"{plugin_id} names no reference")
        return {slot: other for other in manifest["reference"] for slot in self[other]["fills"]}


def normalize(selection, registry=None):
    """configuration.plugins in its full form: use (slot -> plugin id or None), base (each used slot's plain choice,
    the one an ablation falls back to: None for an optional slot, and by default the plugin in use for a slot every
    engine fills, which an engine never runs without) and options (plugin id -> option values)."""
    if not isinstance(selection, dict):
        raise PluginError("configuration.plugins is an object with use, base and options")
    unknown = set(selection) - SELECTION_KEYS
    if unknown or "use" not in selection:
        raise PluginError("configuration.plugins has use (slot -> plugin), and only base and options beside it"
                          + (f"; unknown {', '.join(sorted(unknown))}" if unknown else ""))
    out = {}
    for key in ("use", "base"):
        value = selection.get(key, {})
        if not isinstance(value, dict):
            raise PluginError(f"plugins.{key} maps slots to plugin ids")
        for slot, plugin_id in value.items():
            if not isinstance(slot, str) or not SLOT.fullmatch(slot):
                raise PluginError(f"plugins.{key}: {slot!r} is no slot name")
            if plugin_id is not None and (not isinstance(plugin_id, str) or not ID.fullmatch(plugin_id)):
                raise PluginError(f"plugins.{key}.{slot}: {plugin_id!r} is no plugin id")
            if registry is not None and plugin_id is not None:
                registry[plugin_id]
        out[key] = dict(value)
    extra = sorted(set(out["base"]) - set(out["use"]))
    if extra:
        raise PluginError(f"plugins.base names {', '.join(extra)}, which plugins.use does not: base gives each used "
                          "slot its plain choice")
    out["base"] = {slot: out["base"].get(slot, out["use"][slot] if slot in REQUIRED_SLOTS else None)
                   for slot in out["use"]}
    options = selection.get("options", {})
    if not isinstance(options, dict) or not all(isinstance(v, dict) for v in options.values()):
        raise PluginError("plugins.options maps plugin ids to their option values")
    out["options"] = {k: dict(v) for k, v in options.items()}
    return out


def check(selection, registry, facts=(), strict_options=True, device=None):
    """Check a selection without running any plugin code. Returns (use, order, options): the slot map, the plugin ids
    in the order they are created (each after the plugins whose facts it needs, the model plugin last, after every
    slot its step calls) and each one's options. facts: what the checkpoint and the machine provide (the model's
    architecture, the GPU), which a plugin's needs may name. device: the device an engine serves on ("cuda"): the
    selection is then a whole engine, which fills every required slot (REQUIRED_SLOTS) with plugins that run there;
    without one it is a part of an engine (a plugin's tests, an ablation's comparison)."""
    selection = normalize(selection, registry)
    use = selection["use"]
    used = OrderedDict()
    for slot, plugin_id in use.items():
        if plugin_id is not None:
            used.setdefault(plugin_id, []).append(slot)
    for plugin_id, slots in used.items():
        fills = registry[plugin_id]["fills"]
        if sorted(slots) != sorted(fills):
            missing = sorted(set(fills) - set(slots))
            wrong = sorted(set(slots) - set(fills))
            raise PluginError(f"{plugin_id} fills {', '.join(fills)}" + (
                f": name it in {', '.join(missing)} too, a fused plugin fills all its slots" if missing else "") + (
                f"; it does not fill {', '.join(wrong)}" if wrong else ""))
    for slot, plugin_id in selection["base"].items():
        if plugin_id is not None and slot not in registry[plugin_id]["fills"]:
            raise PluginError(f"plugins.base.{slot}: {plugin_id} does not fill {slot}")
    for plugin_id in used:
        empty = [slot for slot in registry[plugin_id]["requires"] if use.get(slot) is None]
        if empty:
            raise PluginError(f"{plugin_id} requires {', '.join(empty)}, which the selection leaves empty: its step "
                              "calls a plugin there")
    if device is not None:
        _check_engine(use, used, registry, device)
    model = use.get("model")
    provided = {}
    for fact in facts:
        provided.setdefault(fact, "the engine")
    for plugin_id in used:
        for fact in registry[plugin_id]["provides"]:
            if fact in provided and provided[fact] != "the engine":
                raise PluginError(f"{plugin_id} and {provided[fact]} both provide {fact}: one engine has one of each")
            provided.setdefault(fact, plugin_id)
    after = {plugin_id: set() for plugin_id in used}
    for plugin_id in used:
        manifest = registry[plugin_id]
        for fact in manifest["conflicts"]:
            if fact in provided:
                raise PluginError(f"{plugin_id} conflicts with {fact}, which {provided[fact]} provides")
        when = [(fact, needs) for fact, needs in manifest["needs_when"].items() if fact in provided]
        for alternatives, because in ([(a, None) for a in manifest["needs"]]
                                      + [(a, fact) for fact, needs in when for a in needs]):
            sources = [provided[fact] for fact in alternatives if fact in provided and provided[fact] != plugin_id]
            if not sources:
                raise PluginError(f"{plugin_id} needs {' or '.join(alternatives)}"
                                  + (f" where {provided[because]} provides {because}" if because else "")
                                  + ", which no other plugin of the selection and no engine fact provides")
            # The model's facts are the engine's (its draft head, its layout): they order nothing, since the model is
            # created last.
            after[plugin_id] |= {source for source in sources if source in used and source != model}
    order = _order(after, [plugin_id for plugin_id in used if plugin_id != model] + ([model] if model else []))
    options = {}
    for plugin_id, given in selection["options"].items():
        if plugin_id not in used and (strict_options and plugin_id not in selection["base"].values()):
            raise PluginError(f"plugins.options names {plugin_id}, which the selection does not use")
    for plugin_id in order:
        options[plugin_id] = options_for(registry[plugin_id], selection["options"].get(plugin_id),
                                         f"plugins.options.{plugin_id}")
    return use, order, options


def fit(manifest, facts, device, registry=None):
    """Why the plugin of `manifest` cannot serve in an engine on `device` for a model whose facts are `facts` (the
    checkpoint's and the device's, crit_core.engine's architecture_facts and machine_facts), or []: it runs elsewhere
    (or is a host reference), conflicts with a fact, or needs a fact neither the facts nor any plugin of `registry`
    provides. A need another plugin provides is no reason: a selection may add that plugin."""
    devices = manifest["devices"]
    if not devices:
        return ["a host reference, the oracle device plugins are tested against"]
    if device not in devices and HOST not in devices:
        return [f"runs on {', '.join(devices)}"]
    facts = set(facts)
    reasons = [f"conflicts with {fact}" for fact in manifest["conflicts"] if fact in facts]
    offered = {fact for m in registry or () if m["id"] != manifest["id"] for fact in m["provides"]}
    needs = list(manifest["needs"]) + [a for fact, needs in manifest["needs_when"].items() if fact in facts for a in needs]
    for alternatives in needs:
        if not any(fact in facts or fact in offered for fact in alternatives):
            reasons.append(f"needs {' or '.join(alternatives)}")
    return reasons


def fitting(registry, facts, device):
    """slot -> [(plugin id, reasons)] for every plugin of `registry` filling the slot, those that fit (no reasons)
    first; every slot crit names appears, an empty list where nothing fills it."""
    out = {slot: [] for slot in SLOTS}
    for manifest in registry:
        reasons = fit(manifest, facts, device, registry)
        for slot in manifest["fills"]:
            out.setdefault(slot, []).append((manifest["id"], reasons))
    for slot in out:
        out[slot].sort(key=lambda item: (bool(item[1]), item[0]))
    return out


def _check_engine(use, used, registry, device):
    """A whole engine: every required slot filled, every plugin running on `device` (or, in a policy slot, on the
    host). A plugin without devices is a host reference, the oracle device plugins are tested against."""
    empty = [slot for slot in REQUIRED_SLOTS if use.get(slot) is None]
    if empty:
        raise PluginError(f"an engine fills {', '.join(REQUIRED_SLOTS)}; the selection leaves {', '.join(empty)} empty")
    for plugin_id, slots in used.items():
        devices = registry[plugin_id]["devices"]
        if not devices:
            raise PluginError(f"{plugin_id} is a host reference, the oracle device plugins are tested against: serve a "
                              f"plugin that runs on {device} in {', '.join(slots)}")
        if device not in devices and HOST not in devices:
            raise PluginError(f"{plugin_id} runs on {', '.join(devices)}, not on {device}")


def _order(after, ids):
    """The ids with each after the ones it needs (stable otherwise); a cycle is refused."""
    order, state = [], {}

    def visit(plugin_id, path):
        if state.get(plugin_id) == "done":
            return
        if state.get(plugin_id) == "visiting":
            cycle = path[path.index(plugin_id):] + [plugin_id]
            raise PluginError("plugins need each other in a cycle: " + " -> ".join(cycle))
        state[plugin_id] = "visiting"
        for other in sorted(after[plugin_id], key=ids.index):
            visit(other, path + [plugin_id])
        state[plugin_id] = "done"
        order.append(plugin_id)

    for plugin_id in ids:
        visit(plugin_id, [])
    return order


class _SourceLoader(importlib.machinery.SourceFileLoader):
    """A plugin module's loader: it compiles the module's source every time, never reading bytecode (a __pycache__
    beside the source, or under PYTHONPYCACHEPREFIX, could hold code other than the package's) and never writing it, so
    loading a package's plugins runs the bytes its manifest lists and leaves them as they were. A __pycache__ written
    into a sealed package by loading it (a debug script without -B) would make the next serve refuse it."""

    def get_code(self, fullname):
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)

    def set_data(self, path, data, *, _mode=0o666):
        pass


class _PluginFinder:
    """Finds the modules of plugin packages (crit_plugin_*, load_module's) as Python's path finder does, with a loader
    that writes no bytecode: a module a plugin imports later (inside a method, from its sources) keeps the package's
    bytes too."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if not fullname.startswith(PLUGIN_MODULES) or path is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or type(spec.loader) is not importlib.machinery.SourceFileLoader:
            return spec
        return importlib.util.spec_from_file_location(fullname, spec.origin, loader=_SourceLoader(fullname, spec.origin),
                                                      submodule_search_locations=spec.submodule_search_locations)

    @staticmethod
    def invalidate_caches():
        pass


def _find_plugins():
    if not any(finder is _PluginFinder for finder in sys.meta_path):
        sys.meta_path.insert(0, _PluginFinder)


def load_module(manifest):
    """The plugin's entry module, imported as a submodule of a package for its folder, so its relative imports
    resolve inside the plugin. A module already imported (the same folder, an engine opened again) is reused. No
    module of a plugin package writes bytecode (_SourceLoader)."""
    _find_plugins()
    folder = Path(manifest["folder"]).resolve()
    tag = hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    package = f"crit_plugin_{manifest['id'].replace('-', '_')}_{tag}"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        sys.modules[package] = module
        for name, source in manifest.get("sources", {}).items():   # code the plugin imports as .name
            sub = types.ModuleType(f"{package}.{name}")
            sub.__path__ = [str(source["folder"])]
            sub.__package__ = sub.__name__
            sys.modules[sub.__name__] = sub
            setattr(module, name, sub)
    file = manifest["entry"][0]
    name = package + "." + file[:-3].replace("/", ".")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, folder / file, loader=_SourceLoader(name, str(folder / file)))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return sys.modules[name]


class Assembly:
    """The engine's plugins, created: assembly[slot] is the plugin filling it (None for an empty slot). memory: each
    plugin's declared memory (memory_of); measured: the device bytes each plugin's factory was measured to take beyond
    the placement plans made while it ran, None where the device's memory cannot be read."""

    def __init__(self, use, order, options, instances, memory):
        self.use, self.order, self.options = dict(use), list(order), options
        self.instances, self.memory = instances, memory
        self.measured = {}
        self.parts = None   # what the factories were handed (assemble's _Parts)

    def declared_device(self):
        """The device bytes the plugins' own memory counts as planned: each plugin's declaration, at most what its
        factory was measured to take (a workspace declared but never allocated covers nothing), the declaration of a
        plugin whose memory could not be measured."""
        return sum(m["device"] if self.measured.get(p) is None else min(m["device"], self.measured[p])
                   for p, m in self.memory.items())

    def unmeasured(self):
        """The plugins that declare device memory the loader could not measure (counted at their declaration)."""
        return [p for p in self.order if self.memory[p]["device"] and self.measured.get(p) is None]

    def __getitem__(self, slot):
        if slot not in self.use:
            raise KeyError(f"the selection has no slot {slot}")
        plugin_id = self.use[slot]
        return None if plugin_id is None else self.instances[plugin_id]

    def __contains__(self, slot):
        return slot in self.use

    def get(self, slot, default=None):
        return self[slot] if slot in self.use and self.use[slot] is not None else default

    def facts(self):
        """What the engine reports of its plugins in its facts: the plugins it runs, so crit serve shows which ones,
        and the engine facts a plugin knows of its own work (a plugin object's optional facts(), such as the
        projections plugin's prompt_activations), the first plugin in creation order winning a key."""
        from .kernels import loaded
        found = {}
        for plugin_id in self.order:
            report = getattr(self.instances.get(plugin_id), "facts", None)
            if callable(report):
                for key, value in dict(report() or {}).items():
                    found.setdefault(key, value)
        found["plugins"] = dict(self.use)
        # crit's kernels this engine's plugins asked for, and how each was built (crit_plugins.kernels).
        kernels = {name: record for name, record in loaded().items() if set(record.get("plugins") or ()) & set(self.order)}
        if kernels:
            found["kernels"] = kernels
        return found

    def telemetry(self):
        """What the engine's plugins report of their work so far, for crit serve's /health and Eval: each plugin
        object's optional telemetry() (a dict of sections, such as a device expert cache's {"expert_cache": {slots,
        resident, lent, reads, hits, fills, fill_bytes}} or a drafter's {"drafts": {accepted_by_depth, rounds,
        drafted}}), in creation order with the model's last; a section two plugins report holds the keys of both, the
        first plugin winning a key."""
        found = {}
        for plugin_id in self.order:
            report = getattr(self.instances.get(plugin_id), "telemetry", None)
            if not callable(report):
                continue
            for key, value in dict(report() or {}).items():
                if isinstance(value, dict) and isinstance(found.get(key), dict):
                    for inner, item in value.items():
                        found[key].setdefault(inner, item)
                else:
                    found.setdefault(key, dict(value) if isinstance(value, dict) else value)
        return found

    def unobserve(self):
        """Take the counting proxies off the hot path once the loader's checks have passed: the counting stops (a
        proxy left where Core cannot reach it passes calls through), the factories' parts hand out the plugins
        themselves, and every proxy the plugins keep is replaced by its target (crit_plugins.observe.unobserve).
        Returns how many it replaced."""
        parts = self.parts
        if parts is None or parts._calls is None:
            return 0
        parts._calls.stop()
        parts._calls = None
        return unobserve([self.instances[plugin_id] for plugin_id in self.order if plugin_id in self.instances])

    def close(self):
        """Close the plugins in the reverse of their creation; the first error is raised after every close ran."""
        failure = None
        for plugin_id in reversed(self.order):
            close = getattr(self.instances.get(plugin_id), "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:  # noqa: BLE001 - every plugin still closes
                    failure = failure or exc
        self.instances, self.parts = {}, None
        if failure is not None:
            raise failure


def _measure(assembly, plugin_id, held, now, plans):
    """The device memory plugin_id's factory took (now - held) beyond the placement plans made while it ran, into
    assembly.measured; a plugin that took more than its memory() declares, DECLARED_SLACK aside, is refused."""
    if now is None:
        assembly.measured[plugin_id] = None   # unmeasured: no reading of this device's memory
        return
    grown = max(0, now - (held or 0) - _device_planned(plans))
    assembly.measured[plugin_id] = grown
    declared = assembly.memory[plugin_id]["device"]
    if grown > declared * (1 + DECLARED_SLACK[0]) + DECLARED_SLACK[1]:
        raise PluginError(f"{plugin_id}: its factory took {grown / 2**20:.1f} MiB of device memory beyond the placement "
                          f"plans made while it ran, and its memory() declares {declared / 2**20:.1f} MiB: a plugin "
                          "declares the workspaces it allocates when it is made, and its tensors (weights, KV, "
                          "state) go through the placement plugin's plan")


class _Parts:
    """The plugins created so far, by slot, as a plugin's factory sees them: those it needs are always there. With
    `calls` (crit_plugins.observe.Calls), each behind a proxy that counts the calls made to its slot, until the loader
    has run its checks (Assembly.unobserve). slots(), plugin(slot) and facts(slot) tell what fills each slot and the
    facts its plugin provides, which a model plugin's round driver checks (crit_plugins.rounds.round_problems)."""

    def __init__(self, use, instances, calls=None, registry=None):
        self._use, self._instances, self._calls, self._registry = use, instances, calls, registry
        # The loader's call counter (None without one): a driver that captures graphs over these parts counts its
        # replays there as the calls its captures made (crit_plugins.observe.Calls.replayed).
        self.observer = calls

    def slots(self):
        """The slots of the selection, filled or empty."""
        return list(self._use)

    def plugin(self, slot):
        """The id of the plugin filling `slot`, or None."""
        return self._use.get(slot)

    def facts(self, slot):
        """The facts the plugin filling `slot` provides (its manifest's provides), () for an empty slot."""
        plugin_id = self._use.get(slot)
        if plugin_id is None or self._registry is None or plugin_id not in self._registry:
            return ()
        return tuple(self._registry[plugin_id]["provides"])

    def __getitem__(self, slot):
        plugin_id = self._use.get(slot)
        if plugin_id is None or plugin_id not in self._instances:
            raise KeyError(f"no plugin fills {slot} yet: a plugin reads only the slots whose facts it needs, and the "
                           "model plugin, created last, every slot")
        found = self._instances[plugin_id]
        return found if self._calls is None else Observed(found, slot, self._calls)

    def get(self, slot, default=None):
        try:
            return self[slot]
        except KeyError:
            return default


# What a plugin's factory may take on the device beyond its memory() declaration and the placement plans made while
# it ran: small tables, the allocator's rounding and a math library's workspace (a first matrix product on a CUDA
# stream takes 32 MiB), as a share of the declaration plus a fixed amount.
DECLARED_SLACK = (0.05, 64 << 20)


def device_bytes(device="cuda"):
    """The bytes torch holds allocated on every CUDA device of the process, or None where Core cannot read them (another
    device, no CUDA, torch not loaded by a plugin). It never initializes CUDA."""
    torch = sys.modules.get("torch")
    if device != "cuda" or torch is None or not torch.cuda.is_available():
        return None
    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))


def device_memory(device="cuda"):
    """{"allocated": bytes, "reserved": bytes} torch holds on every CUDA device of the process (reserved: what its
    caching allocator took from the device, allocated within it), or None where they cannot be read, as device_bytes.
    It never initializes CUDA."""
    torch = sys.modules.get("torch")
    if device != "cuda" or torch is None or not torch.cuda.is_available():
        return None
    devices = range(torch.cuda.device_count())
    return {"allocated": sum(torch.cuda.memory_allocated(i) for i in devices),
            "reserved": sum(torch.cuda.memory_reserved(i) for i in devices)}


def release_cached(device="cuda"):
    """Hand the device back what torch's caching allocator holds and nothing uses (a load's temporaries: checkpoint
    conversions, warm-ups), where torch is loaded and the device is CUDA: returns the bytes released, or None where the
    device's memory cannot be read. Live tensors stay where they are."""
    before = device_memory(device)
    if before is None:
        return None
    torch = sys.modules["torch"]
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return max(0, before["reserved"] - device_memory(device)["reserved"])


def _device_planned(plans):
    """The bytes placement plans put on the device: plans as crit_plugins.observe.Calls records them."""
    return sum(size for tensors, places in plans for name, size in tensors.items() if places.get(name) == "device")


def memory_of(registry, order, options, context):
    """Each plugin's declared memory, {"device": bytes, "host": bytes}, from its module's memory(context, options)."""
    out = {}
    for plugin_id in order:
        function = getattr(load_module(registry[plugin_id]), "memory", None)
        value = function(context, options[plugin_id]) if callable(function) else {}
        if not isinstance(value, dict) or set(value) - {"device", "host"} or not all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value.values()):
            raise PluginError(f"{plugin_id}: memory() returns {{\"device\": bytes, \"host\": bytes}}")
        out[plugin_id] = {"device": value.get("device", 0), "host": value.get("host", 0)}
    return out


def assemble(selection, registry, context=None, facts=(), budget=None, device=None, observe=None, measure=None):
    """Create the selected plugins: check the selection (check; a whole engine on `device` when one is given), their
    declared memory against `budget` ({"device": bytes, "host": bytes}, either optional) before any is created, then
    call each plugin's factory, entry(context, options, parts), in order; on `device` each plugin must have its
    slots' device form (crit_plugins.device), checked as it is created, before a later plugin calls it. observe: a
    crit_plugins.observe.Calls that counts every call a plugin makes to another slot's plugin (and records the
    placement plans). measure: the device bytes the process holds now, or None where they cannot be read (default:
    device_bytes, torch's on CUDA): each factory's growth beyond the device bytes of the placement plans made while it
    ran (with observe) is the plugin's measured memory (Assembly.measured), and a plugin that takes more than its
    memory() declares (DECLARED_SLACK aside) is refused, named with both. Returns the Assembly; a failure closes the
    plugins already created. Core's loader (crit_core.engine) assembles every engine; this is its plugin half."""
    if not isinstance(registry, Registry):
        registry = Registry.discover(registry)
    use, order, options = check(selection, registry, facts, device=device)
    memory = memory_of(registry, order, options, context)
    for kind in ("device", "host"):
        limit = (budget or {}).get(kind)
        total = sum(m[kind] for m in memory.values())
        if limit is not None and total > limit:
            shares = ", ".join(f"{p} {m[kind] / 2**30:.2f} GiB" for p, m in memory.items() if m[kind])
            raise PluginError(f"the plugins need {total / 2**30:.2f} GiB of {kind} memory, the budget is "
                              f"{limit / 2**30:.2f} GiB ({shares})")
    instances = {}
    assembly = Assembly(use, order, options, instances, memory)
    parts = assembly.parts = _Parts(use, instances, observe, registry)
    measure = measure or device_bytes
    try:
        for plugin_id in order:
            held, seen = measure(), len(observe.plans) if observe is not None else 0
            module = load_module(registry[plugin_id])
            factory = getattr(module, registry[plugin_id]["entry"][1], None)
            if not callable(factory):
                raise PluginError(f"{plugin_id}: {'.'.join(registry[plugin_id]['entry'])} is no function")
            instances[plugin_id] = factory(context, options[plugin_id], parts)
            if device is not None and device != HOST:
                for slot in registry[plugin_id]["fills"]:
                    lacking = device_missing(slot, instances[plugin_id])
                    if lacking:
                        raise PluginError(f"{plugin_id} in {slot} lacks {', '.join(lacking)} of the {slot} device form: "
                                          f"an engine on {device} serves only plugins that compute there, and a host "
                                          "form is an oracle no engine serves")
            for fact in registry[plugin_id]["provides"]:
                if fact in DEVICE_EXTENSIONS and fact not in POOL_FACTS:
                    lacking = extension_missing(fact, instances[plugin_id])
                    if lacking:
                        raise PluginError(f"{plugin_id} provides {fact} but lacks {', '.join(lacking)}: a plugin that "
                                          "provides a device extension has its methods (crit_plugins.device."
                                          "DEVICE_EXTENSIONS)")
            _measure(assembly, plugin_id, held, measure(), observe.plans[seen:] if observe is not None else ())
    except BaseException:
        try:
            assembly.close()
        except BaseException:  # noqa: BLE001 - the creation failure is the one to report
            pass
        raise
    return assembly
