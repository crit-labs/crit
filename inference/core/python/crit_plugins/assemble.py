"""An engine's plugins put together when it loads: which plugin fills each slot, whether their needs, conflicts and
memory hold, and the order they are created in. Everything here runs once per load, never per token."""
from collections import OrderedDict
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

from .manifest import (HOST, ID, MANIFEST, REQUIRED_SLOTS, SLOT, PluginError, load_manifest, options_for,
                       source_plugin)
from .contracts import Contracts
from .memory import device_bytes, device_memory, release_cached   # noqa: F401 - the loader reads them from here too
from .observe import PLUGIN_MODULES, Observed, unobserve



SELECTION_KEYS = {"use", "base", "options"}


# A folder of plugins holds plugins (a folder with a plugin.json) and groups of them (any other folder: the library's
# categories, attention/, experts/, ...; an engine keeps the layout crit plugins add copies). It may hold plugins.json:
# the mark of a folder of plugins, which names the other folders of plugins it is used with (a library split over
# several folders), {"with": ["relative/folder", ...]}.
TOGETHER = "plugins.json"


def together(root):
    """The folders of plugins used together with `root`: root, then those its plugins.json names, transitively (each
    once, those that exist)."""
    import json
    found, queue = [], [Path(root).resolve()]
    while queue:
        folder = queue.pop(0)
        if folder in found or not folder.is_dir():
            continue
        found.append(folder)
        listing = folder / TOGETHER
        if listing.is_file():
            try:
                named = json.loads(listing.read_text(encoding="utf-8")).get("with", [])
            except (OSError, ValueError, AttributeError) as error:
                raise PluginError(f"{listing}: {{\"with\": [folders of plugins]}} ({error})") from None
            queue += [(folder / other).resolve() for other in named]
    return found


def plugin_folders(root):
    """The plugin folders of a folder of plugins: each subfolder holding a plugin.json, then those of its groups (a
    subfolder without one, by name; names starting with _ or . are no groups: _sources, caches)."""
    root = Path(root)
    found = sorted(p for p in root.iterdir() if (p / MANIFEST).is_file())
    for group in sorted(p for p in root.iterdir() if p.is_dir() and not (p / MANIFEST).is_file()
                        and p.name[:1] not in "_."):
        found += sorted(p for p in group.iterdir() if (p / MANIFEST).is_file())
    return found


def plugins_root(folder):
    """The folder of plugins a plugin's folder lies in: its parent, or for a plugin in a group the parent's parent,
    the folder that holds plugins.json (the library's, and an engine's that crit plugins add made)."""
    parent = Path(folder).resolve().parent
    if not (parent / TOGETHER).is_file() and (parent.parent / TOGETHER).is_file():
        return parent.parent
    return parent


class Registry:
    """The plugins an engine ships, by id: every folder holding a plugin.json in the given folders of plugins or in
    their groups (plugin_folders)."""

    def __init__(self, manifests):
        self.manifests = OrderedDict((m["id"], m) for m in manifests)
        self._check_references()
        self._check_hosts()

    @property
    def contracts(self):
        """Core's slots and facts with those the registry's plugins define (crit_plugins.contracts)."""
        return Contracts.of(self)

    def _check_hosts(self):
        """A host plugin (devices "host") fills policy slots only: slots that decide rather than compute."""
        for manifest in self.manifests.values():
            if HOST not in manifest["devices"]:
                continue
            computing = [slot for slot in manifest["fills"] if not self.contracts.is_policy(slot)]
            if computing:
                raise PluginError(f"{manifest['id']}: {', '.join(computing)} computes on the device the engine serves "
                                  f"on; only a plugin of a policy slot ({', '.join(sorted(self.contracts.policy))}) "
                                  "runs on the host")

    @classmethod
    def discover(cls, *roots):
        found = {}
        for given in roots:
            if not Path(given).is_dir():
                raise PluginError(f"{given} is no folder of plugins")
        for root in dict.fromkeys(folder for given in roots for folder in together(given)):
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
    engine fills, which an engine never runs without, and with a registry for a slot a plugin in use requires) and
    options (plugin id -> option values)."""
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
    # A slot a plugin in use requires (its manifest's requires and requires_when) is no more optional than a slot
    # every engine fills: a base that left it empty would be an engine its own plugins refuse.
    filled = set(REQUIRED_SLOTS)
    if registry is not None:
        for plugin_id in {p for p in out["use"].values() if p is not None}:
            manifest = registry[plugin_id]
            filled.update(manifest.get("requires", ()))
            for slots in (manifest.get("requires_when") or {}).values():
                filled.update(slots)
    out["base"] = {slot: out["base"].get(slot, out["use"][slot] if slot in filled else None)
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
        for fact, slots in manifest["requires_when"].items():
            empty = [slot for slot in slots if use.get(slot) is None]
            if fact in provided and empty:
                raise PluginError(f"{plugin_id} requires {', '.join(empty)} where {provided[fact]} provides {fact}, "
                                  "which the selection leaves empty: its step calls a plugin there")
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


def approximations(registry, options):
    """The approximations a selection's resolved options set: [{plugin, option, value, exact, what}] for each option
    whose manifest declares `approximates` and whose value is none of its exact ones."""
    found = []
    for plugin_id, given in options.items():
        for name, spec in (registry[plugin_id].get("options") or {}).items():
            said = spec.get("approximates")
            if said and name in given and given[name] not in said["exact"]:
                found.append({"plugin": plugin_id, "option": name, "value": given[name], "exact": said["exact"],
                              "what": said["what"]})
    return found


class Dry:
    """The selection as a plugin's resolve() sees it, before any plugin exists: which plugin fills each slot
    (plugin), the facts it provides (facts) and its module (module: the functions a slot's plugin offers for a dry
    answer, such as the kv_cache plugin's price of a pool position)."""

    def __init__(self, use, registry):
        self._use, self._registry = dict(use), registry

    def slots(self):
        return list(self._use)

    def plugin(self, slot):
        return self._use.get(slot)

    def facts(self, slot):
        plugin_id = self._use.get(slot)
        return tuple(self._registry[plugin_id]["provides"]) if plugin_id in self._registry else ()

    def module(self, slot):
        """The module of the plugin filling `slot`, or None for an empty slot."""
        plugin_id = self._use.get(slot)
        return load_module(self._registry[plugin_id]) if plugin_id in self._registry else None


# Engine facts that add up over the plugins that hold their part (each plugin's pools a token of context): a plugin's
# resolve() states its own share, and the resolution is their sum.
ADDITIVE_FACTS = ("kv_bytes_per_token",)
# Prices a resolve() states for sizing an engine before it exists (crit_core.spec; crit Eval's context and placement
# read them through the spec): the KV elements a token of context takes over the pools a sequence grows (Eval prices
# them at its protocol's precision), the device bytes of the KV pools that do not grow with the context (a sliding
# window's rings, one a sequence the engine runs at once), and the bytes the projections slot holds the checkpoint's
# target matrices in against their stored bytes. A loaded engine is not held to a price (resolution_problems skips
# them): the plan that sizes the load is (crit_core.engine.plan_problems, against the bytes the load measured).
PRICES = ("kv_elements_per_token", "kv_ring_bytes", "held_matrix_bytes", "stored_matrix_bytes")


def resolve(registry, use, order, options, context):
    """What the engine of a selection will state of itself, answered before any plugin is created: each plugin's
    module's optional resolve(context, options, dry) returns engine facts it can tell from the checkpoint's headers,
    its options and the selection alone (Dry), with no weights read and no device touched (a model plugin's
    kv_bytes_per_token: its pools by its layers, priced by the kv_cache plugin). A fact of ADDITIVE_FACTS is each
    plugin's share, summed; two plugins that state another fact otherwise are refused. Returns {"facts": {key:
    value}, "by": {key: "plugin" or "plugin + plugin"}}. Core's loader runs it before it creates the plugins and
    holds the loaded engine to it (resolution_problems); crit run checks a plan against it before any load
    (crit_core.engine.resolve_package)."""
    dry, facts, by = Dry(use, registry), {}, {}
    for plugin_id in order:
        function = getattr(load_module(registry[plugin_id]), "resolve", None)
        if not callable(function):
            continue
        found = function(context, options[plugin_id], dry)
        if found is None:
            continue
        if not isinstance(found, dict) or not all(isinstance(key, str) for key in found):
            raise PluginError(f"{plugin_id}: resolve() returns a dict of engine facts")
        for key, value in found.items():
            if key in ADDITIVE_FACTS or key in PRICES:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise PluginError(f"{plugin_id}: resolve() states {key} as "
                                      f"{'bytes' if key in ADDITIVE_FACTS else 'a count'} (an int of 0 or more)")
                facts[key] = facts.get(key, 0) + value
                by[key] = f"{by[key]} + {plugin_id}" if key in by else plugin_id
                continue
            if key in facts and facts[key] != value:
                raise PluginError(f"{plugin_id}: resolve() states {key} {value!r}, which {by[key]} states as "
                                  f"{facts[key]!r}")
            facts.setdefault(key, value)
            by.setdefault(key, plugin_id)
    return {"facts": facts, "by": by}


def resolution_problems(resolved, stated):
    """Why a loaded engine is not what its plugins resolved it to be ([] when it is): each resolved fact against the
    engine's own (its info() and its plugins' facts, `stated`). Numbers agree to the byte, everything else as equal."""
    problems = []
    for key, value in sorted((resolved or {}).get("facts", {}).items()):
        if key in PRICES:   # a price for sizing, which the plan that sized the load is held to instead
            continue
        found = stated.get(key)
        if found != value:
            problems.append(f"{resolved['by'][key]}'s resolve() states {key} {value!r}, and the loaded engine "
                            f"{found!r}")
    return problems


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
    out = {slot: [] for slot in registry.contracts.slots}
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


def _subpackages(module, sources, through):
    """Each source as the subpackage module.name over its folder. A source that is another plugin's folder (it holds
    plugin.json: a plugin the plugin extends or composes) carries that plugin's manifest (__crit_manifest__) and its
    own sources nested under it, so its modules import as in their own plugin (from .name.plugin import Class)."""
    for name, source in sources.items():   # code the plugin imports as .name
        sub = types.ModuleType(f"{module.__name__}.{name}")
        sub.__path__ = [str(source["folder"])]
        sub.__package__ = sub.__name__
        sys.modules[sub.__name__] = sub
        setattr(module, name, sub)
        folder = Path(source["folder"]).resolve()
        if source_plugin(source) and folder not in through:
            nested = load_manifest(folder)
            sub.__crit_manifest__ = nested
            _subpackages(sub, nested["sources"], (*through, folder))


def plugin_package(manifest):
    """The package of a plugin's folder: its modules import as PACKAGE.FILE, its sources as PACKAGE.NAME (a source
    that is another plugin's folder with that plugin's sources nested under it). Made once a folder."""
    _find_plugins()
    folder = Path(manifest["folder"]).resolve()
    tag = hashlib.sha256(str(folder).encode()).hexdigest()[:12]
    package = f"crit_plugin_{manifest['id'].replace('-', '_')}_{tag}"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(folder)]
        module.__crit_manifest__ = manifest
        sys.modules[package] = module
        _subpackages(module, manifest.get("sources", {}), (folder,))
    return sys.modules[package]


def load_module(manifest, file=None):
    """The plugin's entry module (or its module `file`, FILE.py in its folder), imported as a submodule of the
    plugin's package (plugin_package), so its relative imports resolve inside the plugin. A module already imported
    (the same folder, an engine opened again) is reused. No module of a plugin package writes bytecode
    (_SourceLoader)."""
    folder = Path(manifest["folder"]).resolve()
    package = plugin_package(manifest).__name__
    if file is None and manifest["entry"] is None:
        raise PluginError(f"{manifest['id']} fills no slot: it is a core other plugins extend and has no factory")
    file = file or manifest["entry"][0]
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


def _engine_policy(method):
    """An Assembly method run with its engine's policy active (crit_plugins.policy), where it has one: the plugins'
    facts and telemetry answer for their own engine whichever thread asks."""
    import functools

    @functools.wraps(method)
    def run(self, *args, **kwargs):
        if self.policy is None:
            return method(self, *args, **kwargs)
        with self.policy.active():
            return method(self, *args, **kwargs)
    return run


class Assembly:
    """The engine's plugins, created: assembly[slot] is the plugin filling it (None for an empty slot). memory: each
    plugin's declared memory (memory_of); measured: the device bytes each plugin's factory was measured to take beyond
    the placement plans made while it ran, None where the device's memory cannot be read."""

    def __init__(self, use, order, options, instances, memory):
        self.use, self.order, self.options = dict(use), list(order), options
        self.approximations = []   # [{plugin, option, value, exact, what}] the options set (assemble: approximations)
        self.instances, self.memory = instances, memory
        self.measured = {}
        self.parts = None   # what the factories were handed (assemble's _Parts)
        self.policy = None  # the engine's policy (crit_plugins.policy), its context's: what its facts report

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

    @_engine_policy
    def facts(self):
        """What the engine reports of its plugins in its facts: the plugins it runs, so crit serve shows which ones,
        and the engine facts a plugin knows of its own work (a plugin object's optional facts(), such as the
        projections plugin's prompt_activations), the first plugin in creation order winning a key; and the
        approximations its options set (approximations(): an option whose manifest says it approximates the model,
        set to a value that does), which Eval's drift and an ablation judge by quality, not speed alone."""
        from .kernels import loaded
        found = {}
        for plugin_id in self.order:
            report = getattr(self.instances.get(plugin_id), "facts", None)
            if callable(report):
                for key, value in dict(report() or {}).items():
                    found.setdefault(key, value)
        found["plugins"] = dict(self.use)
        if self.approximations:
            found["approximations"] = [dict(row) for row in self.approximations]
        # crit's kernels this engine's plugins asked for, and how each was built (crit_plugins.kernels).
        kernels = {name: record for name, record in loaded().items() if set(record.get("plugins") or ()) & set(self.order)}
        if kernels:
            found["kernels"] = kernels
        # The exactness the engine asks and what its plugins' rows keep (exactness()).
        kept = self.exactness()
        if kept is not None:
            found["exactness"] = kept
        return found

    @_engine_policy
    def exactness(self):
        """The exactness of the engine's rows (crit_plugins.exactness's levels): {"asked": the levels the
        configuration asks per phase (crit_plugins.kernels.exactness()) or None, "decode" and "prompt": the lowest
        level any plugin keeps there, "plugins": {plugin id: {phase: level}}}, from each plugin object's optional
        exactness() (the level its rows keep as it is loaded, a phase it computes nothing in left out). None where
        the configuration asks nothing and no plugin states a level; a phase no plugin states is left out. A level
        that does not exist is the plugin's error (PluginError)."""
        from .kernels import exactness as asked_levels
        asked, plugins = asked_levels(), {}
        for plugin_id in self.order:
            report = getattr(self.instances.get(plugin_id), "exactness", None)
            stated = dict(report() or {}) if callable(report) else {}
            if stated:
                plugins[plugin_id] = stated
        if asked is None and not plugins:
            return None
        found = {"asked": asked}
        if plugins:
            from .exactness import LEVELS, PHASES
            for plugin_id, stated in plugins.items():
                wrong = [f"{phase}: {level}" for phase, level in stated.items()
                         if phase not in PHASES or level not in LEVELS]
                if wrong:
                    raise PluginError(f"{plugin_id}: exactness() states {', '.join(wrong)}: a level ({', '.join(LEVELS)}) "
                                      f"per phase ({', '.join(PHASES)})")
            for phase in PHASES:
                levels = [stated[phase] for stated in plugins.values() if phase in stated]
                if levels:
                    found[phase] = min(levels, key=LEVELS.index)
            found["plugins"] = plugins
        return found

    @_engine_policy
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


def exactness_shortfalls(report):
    """Where an engine's plugins keep less than its configuration asks, as sentences ([] where they keep it, or
    nothing is asked): report is Assembly.exactness(). For the decode rows exact and row-exact are one level."""
    asked = (report or {}).get("asked")
    if not asked:
        return []
    from .exactness import keeps
    return [f"{plugin_id} keeps its {phase} rows {stated[phase]} where the configuration's exactness asks {asked[phase]}"
            for plugin_id, stated in (report.get("plugins") or {}).items() for phase in asked
            if phase in stated and not keeps(phase, stated[phase], asked[phase])]


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

    def __init__(self, use, instances, calls=None, registry=None, context=None):
        self._use, self._instances, self._calls, self._registry = use, instances, calls, registry
        # The engine's context, as every factory got it: Core's drivers find the engine's own services there (its
        # graph pool: crit_plugins.rounds, crit_plugins.prompt_graphs).
        self.context = context
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
    assembly.policy = getattr(context, "policy", None)
    assembly.approximations = approximations(registry, options)
    parts = assembly.parts = _Parts(use, instances, observe, registry, context)
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
                    lacking = registry.contracts.device_missing(slot, instances[plugin_id])
                    if lacking:
                        raise PluginError(f"{plugin_id} in {slot} lacks {', '.join(lacking)} of the {slot} device form: "
                                          f"an engine on {device} serves only plugins that compute there, and a host "
                                          "form is an oracle no engine serves")
            contracts = registry.contracts
            for fact in registry[plugin_id]["provides"]:
                if contracts.known(fact) and fact not in contracts.pools:
                    lacking = contracts.extension_missing(fact, instances[plugin_id])
                    if lacking:
                        raise PluginError(f"{plugin_id} provides {fact} but lacks {', '.join(lacking)}: a plugin that "
                                          "provides a device extension has its methods (crit_plugins.contracts: "
                                          f"{contracts.by.get(('the fact', fact), 'Core')} defines it)")
            _measure(assembly, plugin_id, held, measure(), observe.plans[seen:] if observe is not None else ())
    except BaseException:
        try:
            assembly.close()
        except BaseException:  # noqa: BLE001 - the creation failure is the one to report
            pass
        raise
    return assembly
