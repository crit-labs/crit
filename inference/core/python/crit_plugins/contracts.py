"""The slots and facts plugins agree on: Core's own, and those the library's plugins define.

Core names the slots and facts of the engine itself: the model, placement, the KV pool, sampling, drafting and
verifying, and the step, round and prompt forms its drivers run (manifest.SLOTS, slots.INTERFACES,
device.DEVICE_INTERFACES, device.DEVICE_EXTENSIONS). A layer part's slot (attention, projections, experts, a router,
a recurrence, embeddings, norms, ...) and its facts (attention:sinks, experts:activations, ...) are defined by the
plugin that defines the slot, its base plugin, in its manifest:

    "defines": {
      "slots": {"SLOT": {"about": "...", "host": {method: what}, "device": {method: what}, "cases": "FILE.py:FUNC",
                         "policy": false}},
      "facts": {"FACT": {"about": "...", "methods": {method: what}, "pool": "SLOT"}}
    }

`host` is the slot's interface over plain rows, `device` its device form, `cases` the function the slot's plugins are
compared with their reference on (crit_plugins.testing), `policy` a slot whose plugins decide on the host rather than
compute; a fact's `pool` is the slot whose device form returns the object that holds its methods. So a new kind of
layer part, or a new variant of one, is a definition its technique adds, not a change to Core. Two plugins may define
the same thing only alike (an engine's copy of a base plugin and the library's).
"""
from pathlib import Path

from .manifest import MANIFEST, POLICY_SLOTS, SLOTS, PluginError


def _core():
    from . import device, slots
    return {"slots": dict(SLOTS), "host": {k: dict(v) for k, v in slots.INTERFACES.items()},
            "device": {k: dict(v) for k, v in device.DEVICE_INTERFACES.items()},
            "facts": {k: dict(v) for k, v in device.DEVICE_EXTENSIONS.items()},
            "pools": dict(device.POOL_EXTENSIONS), "cases": dict(slots.CASES), "policy": set(POLICY_SLOTS),
            "by": {}}


def _check_defines(value, where):
    """The `defines` of a manifest, checked as data: {slots: {...}, facts: {...}}."""
    if not isinstance(value, dict) or set(value) - {"slots", "facts"}:
        raise PluginError(f"{where}: defines is {{slots, facts}}")
    for slot, spec in value.get("slots", {}).items():
        if not isinstance(spec, dict) or set(spec) - {"about", "host", "device", "cases", "policy", "host_about",
                                                         "device_about"}:
            raise PluginError(f"{where}: defines.slots.{slot} is {{about, host, device, cases, policy}}")
        for part in ("host", "device"):
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in spec.get(part, {}).items()):
                raise PluginError(f"{where}: defines.slots.{slot}.{part} maps each method to what it takes and returns")
    for fact, spec in value.get("facts", {}).items():
        if not isinstance(spec, dict) or set(spec) - {"about", "methods", "pool"} or not isinstance(
                spec.get("methods", {}), dict):
            raise PluginError(f"{where}: defines.facts.{fact} is {{about, methods, pool}}")
    return value


class Contracts:
    """Core's slots and facts with those the manifests define (Contracts.of(registry) for an engine's or the
    library's)."""

    def __init__(self, manifests=()):
        self.__dict__.update(_core())
        self._modules = {}
        for manifest in manifests:
            defines = manifest.get("defines") or {}
            if defines:
                self._merge(manifest, defines)

    @classmethod
    def of(cls, registry):
        found = getattr(registry, "_contracts", None)
        if found is None:
            found = cls(list(registry))
            try:
                registry._contracts = found
            except AttributeError:
                pass
        return found

    def _merge(self, manifest, defines):
        who = manifest["id"]

        def put(table, key, value, what):
            if key in table and table[key] != value:
                other = self.by.get((what, key), "Core")
                raise PluginError(f"{who} defines {what} {key} otherwise than {other} does")
            table[key] = value
            self.by[(what, key)] = who

        for slot, spec in defines.get("slots", {}).items():
            put(self.slots, slot, spec.get("about", ""), "the slot")
            if "host" in spec:
                put(self.host, slot, dict(spec["host"]), "the host interface of")
            if "device" in spec:
                put(self.device, slot, dict(spec["device"]), "the device form of")
            if spec.get("cases"):
                self.cases.setdefault(slot, (manifest, spec["cases"]))
                self.by.setdefault(("the cases of", slot), (manifest, spec["cases"].split(":", 1)[0]))
            if spec.get("policy"):
                self.policy.add(slot)
        for fact, spec in defines.get("facts", {}).items():
            put(self.facts, fact, dict(spec.get("methods", {})), "the fact")
            if spec.get("pool"):
                put(self.pools, fact, spec["pool"], "the pool of")

    # -- what a plugin must have ------------------------------------------------------------------------------
    def host_missing(self, slot, plugin):
        """The host interface methods of `slot` that `plugin` lacks (none for a slot nothing defines)."""
        return [name for name in self.host.get(slot, {}) if not callable(getattr(plugin, name, None))]

    def device_missing(self, slot, plugin):
        """The device-form methods of `slot` that `plugin` lacks (none for a slot without a device form)."""
        return [name for name in self.device.get(slot, {}) if not callable(getattr(plugin, name, None))]

    def extension_missing(self, fact, plugin, made=()):
        """The methods of `fact` that neither `plugin` nor any object of `made` (what its device form returned: a
        pool) has."""
        holders = [plugin, *made]
        return [name for name in self.facts.get(fact, {})
                if not any(callable(getattr(holder, name, None)) for holder in holders)]

    def known(self, fact):
        return fact in self.facts

    def is_policy(self, slot):
        return slot in self.policy

    def pool_slots(self):
        """The slots whose device form returns a pool that a fact's methods live on (the KV pool, a state pool)."""
        return set(self.pools.values())

    def case(self, slot):
        """The function that makes `slot`'s comparison cases, or None: Core's, or the one a manifest names."""
        found = self.cases.get(slot)
        if found is None or callable(found):
            return found
        manifest, name = found
        made = getattr(self.case_module(slot), name.split(":", 1)[1])
        self.cases[slot] = made
        return made

    def case_module(self, slot):
        """The module of a defined slot's cases (FILE.py of its definition), with the helpers they share (the
        synthetic weights an experts case adds, for one), or None for a slot whose cases are Core's or none."""
        found = self.by.get(("the cases of", slot))
        if found is None:
            return None
        manifest, file = found
        key = (manifest["id"], file)
        if key not in self._modules:
            import importlib.util
            path = Path(manifest["folder"]) / file
            spec = importlib.util.spec_from_file_location(
                f"crit_cases_{manifest['id'].replace('-', '_')}_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._modules[key] = module
        return self._modules[key]


def policy_slot(parts, slot):
    """Whether `slot` decides rather than computes, by the contracts of the engine whose parts `parts` are (Core's
    own policy slots where they carry none)."""
    registry = getattr(parts, "_registry", None)
    return Contracts.of(registry).is_policy(slot) if registry is not None else slot in POLICY_SLOTS


__all__ = ["Contracts", "MANIFEST", "policy_slot"]
