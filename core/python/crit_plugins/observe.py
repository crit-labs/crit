"""Which slots an engine's plugins actually call: the proof that an engine is its plugins, not a model plugin that
carries the whole engine and fills the other slots with plugins it never uses.

assemble(..., observe=Calls()) hands each plugin's factory the other slots' plugins behind a counting proxy: every
call of one of their methods counts for that slot, and so does every call on an object such a method returned that
plugin code made (a KV pool, a loaded matrix, a state pool), which comes back behind the same slot's proxy. A proxy
passed back into a plugin method is unwrapped first, so a plugin always sees its own objects. Tensors, numbers and
other values pass through unwrapped. Core's loader runs one request on a new engine and refuses it when a slot every
engine fills (or one its plugins require) took no part in it.

Once the loader's checks have passed, the proxies leave the hot path (Assembly.unobserve): every proxy and counted
method a plugin keeps, anywhere it can be reached from the plugin objects (attributes, containers, torch modules,
closures, partials), is replaced by its target (unobserve), and one left where Core cannot reach it (inside a library's
C object) passes calls straight through, counting nothing.
"""
from collections import Counter
import sys

# The package names crit_plugins.assemble.load_module gives a plugin's modules: objects of their classes are plugin
# code's, and are counted for the slot whose plugin returned them.
PLUGIN_MODULES = "crit_plugin_"


class Calls(Counter):
    """Calls per slot since the last reset(), and every placement plan a plugin was given: (tensors, places) in
    `plans`, which Core's loader holds the model's device memory to. stop() ends the counting: the proxies made from
    then on, and those left, pass calls through.

    A captured graph (a decode round, a prompt chunk) replays the work its capture enqueued through these proxies, with
    no call of its own: its driver notes the slots called while it captured (called_since) and counts each replay as
    one call of each of them (replayed), so a request served by replays still shows which plugins did its work."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plans = []
        self.active = True

    def reset(self):
        self.clear()

    def stop(self):
        self.active = False

    def snapshot(self):
        """The counts now, for called_since."""
        return dict(self)

    def called_since(self, snapshot):
        """The slots called since `snapshot` (a capture's), sorted."""
        return sorted(slot for slot, count in self.items() if count > snapshot.get(slot, 0))

    def replayed(self, slots):
        """A replay of work captured through the slots `slots`: one call of each, while the counting runs."""
        if self.active:
            for slot in slots:
                self[slot] += 1


def unwrap(value):
    if isinstance(value, Observed):
        return object.__getattribute__(value, "_target")
    if isinstance(value, Counted):
        return value.method
    return value


def _plugin_made(value):
    module = type(value).__module__ or ""
    return module.startswith(PLUGIN_MODULES) and not isinstance(value, (type, Observed))


class Counted:
    """A method of an observed plugin or plugin-made object: each call counts for its slot while the counting runs."""

    __slots__ = ("method", "slot", "calls", "name")

    def __init__(self, method, slot, calls, name):
        self.method, self.slot, self.calls, self.name = method, slot, calls, name

    def __call__(self, *args, **kwargs):
        calls = self.calls
        if not calls.active:
            return self.method(*args, **kwargs)
        calls[self.slot] += 1
        args, kwargs = [unwrap(a) for a in args], {k: unwrap(v) for k, v in kwargs.items()}
        result = self.method(*args, **kwargs)
        if self.slot == "placement" and self.name == "plan" and isinstance(result, dict):
            tensors = args[0] if args else kwargs.get("tensors")
            calls.plans.append((dict(tensors or {}), dict(result)))
        return Observed(result, self.slot, calls) if _plugin_made(result) else result

    def __repr__(self):
        return f"Counted({self.slot}: {self.method!r})"


class Observed:
    """A slot's plugin, or an object its methods made, as another plugin sees it: its methods count their calls."""

    __slots__ = ("_target", "_slot", "_calls")

    def __init__(self, target, slot, calls):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_slot", slot)
        object.__setattr__(self, "_calls", calls)

    def __getattr__(self, name):
        found = getattr(object.__getattribute__(self, "_target"), name)
        calls = object.__getattribute__(self, "_calls")
        if not calls.active or not callable(found) or isinstance(found, type):
            return found
        return Counted(found, object.__getattribute__(self, "_slot"), calls, name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_target"), name, value)

    def __repr__(self):
        return f"Observed({object.__getattribute__(self, '_slot')}: {object.__getattribute__(self, '_target')!r})"


_SCALARS = (type(None), bool, int, float, complex, str, bytes, bytearray, range, memoryview)


def _skipped(value):
    """Values unobserve never looks into: scalars, classes, modules, tensors (a tensor's attributes hold no plugin
    object) and the proxies themselves (replaced, not walked)."""
    import types
    if isinstance(value, _SCALARS + (type, types.ModuleType, Observed, Counted, types.BuiltinFunctionType)):
        return True
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _plain(value):
    """`value` with the proxies in it replaced, and whether anything was: a proxy by its target, a counted method by
    the method, a tuple, named tuple, set or functools.partial holding them by a new one holding the targets."""
    import functools
    if isinstance(value, (Observed, Counted)):
        return unwrap(value), True
    if isinstance(value, (tuple, set, frozenset)) and any(isinstance(v, (Observed, Counted, tuple, functools.partial))
                                                          for v in value):
        items = [_plain(v) for v in value]
        if any(changed for _, changed in items):
            plain = [v for v, _ in items]
            kind = type(value)
            return (kind(*plain) if hasattr(value, "_fields") else kind(plain)), True
    if isinstance(value, functools.partial):
        function, changed = _plain(value.func)
        arguments, moved = _plain(tuple(value.args))
        keywords = {k: _plain(v) for k, v in value.keywords.items()}
        if changed or moved or any(c for _, c in keywords.values()):
            return functools.partial(function, *arguments, **{k: v for k, (v, _) in keywords.items()}), True
    return value, False


def _entries(holder):
    """(key, value, put) for every value `holder` holds that unobserve looks at, `put(new)` storing a replacement."""
    import collections
    import functools
    import types
    if isinstance(holder, dict):
        if any(isinstance(key, (Observed, Counted)) for key in holder):   # proxies as keys: the dict rebuilt in order
            items = [(unwrap(key), value) for key, value in holder.items()]
            holder.clear()
            holder.update(items)
        return [(key, value, lambda new, key=key: holder.__setitem__(key, new)) for key, value in list(holder.items())]
    if isinstance(holder, (list, collections.deque)):
        return [(index, value, lambda new, index=index: holder.__setitem__(index, new))
                for index, value in enumerate(list(holder))]
    if isinstance(holder, (tuple, set, frozenset)):
        return [(None, value, None) for value in holder]   # replaced through the holder's own holder (_plain)
    if isinstance(holder, functools.partial):
        return [(None, value, None) for value in (holder.func, *holder.args)] + [(None, holder.keywords, None)]
    found = []
    if isinstance(holder, types.FunctionType):
        for cell in holder.__closure__ or ():
            try:
                found.append((None, cell.cell_contents, lambda new, cell=cell: setattr(cell, "cell_contents", new)))
            except ValueError:   # an empty cell
                continue
        return found
    if isinstance(holder, types.MethodType):
        return [(None, holder.__self__, None), (None, holder.__func__, None)]
    try:
        attributes = getattr(holder, "__dict__", None)
    except Exception:  # noqa: BLE001 - an object whose attribute access fails holds nothing unobserve can reach
        attributes = None
    if isinstance(attributes, dict):
        found.append((None, attributes, None))
    for kind in type(holder).__mro__:
        names = kind.__dict__.get("__slots__", ())
        for name in (names,) if isinstance(names, str) else names:
            if name in ("__dict__", "__weakref__"):
                continue
            try:
                value = object.__getattribute__(holder, name)
            except AttributeError:
                continue
            found.append((name, value, lambda new, name=name: object.__setattr__(holder, name, new)))
    return found


def unobserve(roots):
    """Replace every proxy (Observed) and counted method reachable from `roots` (an engine's plugin objects) by its
    target, in place: in every object's attributes (its __dict__ and __slots__: a torch module's submodules, a
    SimpleNamespace, any library object a plugin keeps), in lists, deques, dicts (their keys too), tuples, named tuples,
    sets and functools.partial objects (rebuilt), and in the closures of functions and the objects bound methods hold.
    Tensors, classes and modules are not looked into. Returns how many it replaced."""
    seen, stack, replaced = set(), list(roots), 0
    while stack:
        holder = stack.pop()
        if id(holder) in seen or _skipped(holder):
            continue
        seen.add(id(holder))
        for _, value, put in _entries(holder):
            plain, changed = _plain(value)
            if changed and put is not None:
                put(plain)
                replaced += 1
            if not _skipped(plain):
                stack.append(plain)
    return replaced
