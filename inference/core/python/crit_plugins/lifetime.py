"""An engine's device memory from its load to its close: a closed engine proves it gave back what it took.

    opened = lifetime.open(memory)    # Core's loader, before any plugin is made (refused beside a leaking close)
    opened.watch({plugin id: plugin object})   # the engine's plugins, followed by weak references
    ...
    report = opened.close()           # once the engine dropped every reference it held: collected, torch's cache
                                      # emptied, the device measured against its level before the load

Core's allocator owns the device's memory during a step; at a close, what is left is whatever something still
refers to. Python frees an engine's memory only once nothing refers to its objects, so one stray reference (a
round driver kept after its close, a drafter's head holding the model in a cycle, a plugin's scratch kept in a
module) keeps gigabytes, and the next engine opened in the process (crit serve's reload and model switch, Eval's
lower-context round, an ablation's variant) runs out of memory. So a close measures, after a collection and with
torch's cache emptied, what of the load's is still there:

    ranges   every elastic range made since the load (paged pools, growing workspaces, the engine's scratch) is closed
    givers   every giver listed since the load is gone (a giver's release refers to its plugin)
    graphs   no graph captured since the load is alive (the device follows each by a weak reference,
             crit_plugins.memory's captured(); a live graph holds its pool)
    torch    torch's allocated and reserved bytes are back at their level before the load

What is the process's and outlives every engine is named and measured apart, never covered by a tolerance:

    the CUDA context, loaded modules and compiled kernels     outside torch's allocator: not measured at all
    the capture stream's cuBLAS workspaces                    a pool of the process's (memory.pool(process=True))
    cuBLAS and cuBLASLt workspaces of torch's handles          freed by the check (they outlive the threads that made
                                                              them; no graph Core captures uses them)
    tensors the process keeps for every engine                listed by name (memory.process(owner, name, tensor):
                                                              a plugin's constant tables, buffers shared by engines)
    ballast ranges (a test's share of the device taken away)  owner "ballast"
    host memory (kept host stores, pinned buffers)            not device memory: not measured

The torch leg compares whole bytes: torch's allocated bytes are the sums of its blocks (each a multiple of 512
bytes), the same for the same tensors, so no slack is allowed. A close is judged only when no other engine is open
on the device: the device's readings do not tell two engines' memory apart, and crit opens one engine at a time
(every reopen above closes the engine first). A close beside another engine says so (`judged` False) and hands
the objects it watched to the engine opened before it, whose judged close then names them.

The report says what still holds memory: the ranges, givers and graphs by name, torch's bytes, and the plugin
objects still alive with the references that keep them (held_by: the chain from a module global or an object
the collector cannot see past, to the object). A close that did not return its memory is recorded on the device
(memory.leaking) and, with CRIT_CLOSE_CHECK=raise (tests and checks), raises LeakingClose; otherwise it writes one
line to stderr and returns (a close often runs on a failure's way out, where an exception would hide the first
one). Either way the next load on the device measures the leaking close again (a holder may have let go since)
and refuses to start beside it, with the report as its message (open()), instead of running out of memory later.
"""
import gc
import os
import sys
import types
import weakref

CHECK = "CRIT_CLOSE_CHECK"   # "raise": a close that does not return its memory raises (tests, checks); else it warns
PROCESS_OWNERS = ("ballast",)   # elastic ranges of these owners are the process's (crit_plugins.memory's ballast)
EXCLUDED = (
    "the CUDA context, loaded modules and compiled kernels (outside torch's allocator: not measured)",
    "the capture stream's cuBLAS workspaces (a pool of the process's)",
    "cuBLAS and cuBLASLt workspaces of torch's handles (freed by the check)",
    "tensors the process keeps for every engine (memory.process, by name)",
    "ballast ranges (a test's share of the device)",
    "host memory (kept host stores, pinned buffers)",
)
BLOCK = 512          # torch's allocator rounds every block to this many bytes
DEPTH = 12           # how far held_by follows references back from an object
VISITS = 4000        # the objects held_by looks at, at most
TENSORS = 8          # the tensors a report names one by one


class LeakingClose(RuntimeError):
    """A closed engine did not return its memory (its report: .report), or a load refused beside such a close."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def strict():
    """Whether a close that does not return its memory raises (CRIT_CLOSE_CHECK=raise) instead of warning."""
    return os.environ.get(CHECK, "").strip().lower() == "raise"


def open(memory):
    """An engine starts to load on `memory`'s device (crit_plugins.memory's DeviceMemory): refused (LeakingClose)
    while the device's last close did not return its memory, measured again first; else the device's level now
    (Opened), which the engine's close is measured against."""
    refuse(memory)
    opened = Opened(memory)
    memory.opened.append(opened)
    return opened


def refuse(memory):
    """Raise LeakingClose while the device's last close that did not return its memory still has not: measured
    again (a caller that kept the engine object may have dropped it since). An engine dropped without its close
    (Opened.dropped) is measured here first, as its close would have been, so it neither stays open on the device
    (every later close beside it unjudged) nor leaks unseen."""
    for dropped in list(getattr(memory, "abandoned", ())):
        memory.abandoned.remove(dropped)
        dropped.close(quiet=True)
    leaking = memory.leaking
    if leaking is None:
        return
    report = leaking.measure()
    if report["returned"] is not False:
        memory.leaking = None
        return
    leaking.report = report
    raise LeakingClose(f"no engine loads on {memory.device} beside the last engine's close, which did not return "
                       f"its memory: {report['message']}", report)


def _block(nbytes):
    return max(BLOCK, -(-int(nbytes) // BLOCK) * BLOCK)


class Opened:
    """One engine's device memory from its load (open()) to its close (close()); the module's description."""

    def __init__(self, memory):
        self.memory = memory
        self.watched = []       # [(name, weak reference)] of the engine's plugin objects
        self.packages = {}      # the plugins' packages: {package: plugin id}, whose modules' globals a report walks
        self.report = None
        memory.readings.release()   # the level before the load: torch's cache emptied, as a close measures it
        self.ranges = {id(found) for found in memory.ranges.values()}
        self.givers = {(name, id(giver[0])) for name, giver in memory.givers.items()}
        self.captures = memory.captures
        self.level = self.torch()

    # -- what is watched ------------------------------------------------------------------------------------------
    def watch(self, objects):
        """Follow the engine's objects ({name: object}, its plugins by id) by weak references: one still alive at the
        close is named in the report with what holds it. Objects without weak references are left out."""
        from .observe import PLUGIN_MODULES, unwrap
        for name, found in objects.items():
            found = unwrap(found)
            if found is None:
                continue
            package = (type(found).__module__ or "").split(".")[0]
            if package.startswith(PLUGIN_MODULES):
                self.packages.setdefault(package, str(name))
            try:
                self.watched.append((str(name), weakref.ref(found)))
            except TypeError:
                pass

    # -- readings ---------------------------------------------------------------------------------------------------
    def torch(self):
        """torch's bytes on the device beside the process's own: {"allocated", "reserved", "excluded": {"allocated",
        "reserved", "by": {name: bytes}}}, allocated and reserved without the excluded; None where the device's memory
        cannot be read."""
        memory = self.memory
        state = memory.readings.torch()
        if state is None:
            return None
        segments = [(found[0], found[1], found[2], len(found) > 3 and bool(found[3]),
                     found[4] if len(found) > 4 else None) for found in memory.readings.segments()]
        allocated = reserved = 0
        by = {}
        held = set()   # the segments already set aside
        for first, end, used, private, pool in segments:
            if pool is not None and tuple(pool) in memory.process_pools:
                name = memory.pools.get(tuple(pool), "process")
                allocated, reserved = allocated + used, reserved + (end - first)
                by[name] = by.get(name, 0) + (end - first)
                held.add(first)
        storages = set()
        for name, tensor in memory.process_tensors.items():
            try:
                storage = tensor.untyped_storage()
                at, nbytes = storage.data_ptr(), storage.nbytes()
            except Exception:  # noqa: BLE001 - a tensor that cannot be read (freed, another kind) sets nothing aside
                continue
            if not at or at in storages:
                continue
            storages.add(at)
            segment = next((s for s in segments if s[0] <= at < s[1]), None)
            if segment is None:
                continue   # not torch's memory on this device
            if segment[0] in held and segment[4] is not None and tuple(segment[4]) in memory.process_pools:
                continue   # inside a pool already set aside
            allocated += _block(nbytes)
            by[name] = _block(nbytes)
            if segment[0] not in held:   # its segment is kept by the tensor: the process's while the tensor lives
                held.add(segment[0])
                reserved += segment[1] - segment[0]
        for name, nbytes in getattr(memory, "leased", {}).items():
            # a host's device lease (memory.lease): its allocation outlives the engine by design, set aside by the
            # bytes the host accounts for it
            allocated, reserved = allocated + nbytes, reserved + nbytes
            by[name] = nbytes
        return {"allocated": int(state["allocated"]) - allocated, "reserved": int(state["reserved"]) - reserved,
                "excluded": {"allocated": allocated, "reserved": reserved, "by": by}}

    def others(self):
        """The other engines open on the device now (their Opened)."""
        return [found for found in self.memory.opened if found is not self]

    def measure(self):
        """The device against its level at the load, after a collection and torch's cache emptied: the report."""
        memory = self.memory
        others = self.others()
        gc.collect()
        if not others:   # no other engine's thread uses a workspace now
            memory.readings.clear_workspaces()
        memory.readings.release()
        report = {"device": memory.device, "returned": None, "judged": not others, "excluded": list(EXCLUDED)}
        # Each object is taken from its weak reference alone, so no list of ours is among what holds it.
        report["objects"] = []
        for name, ref in self.watched:
            found = ref()
            if found is not None:
                report["objects"].append({"name": name, "type": type(found).__name__, "held_by": held_by(found)})
            found = None
        if others:
            report["message"] = (f"not judged: {len(others)} other engine(s) are open on {memory.device}, and its "
                                 "readings do not tell engines' memory apart")
            return report
        report["ranges"] = [{"name": key, "backed": int(found.mapped), "reserved": int(found.size)}
                            for key, found in memory.ranges.items()
                            if id(found) not in self.ranges and not key.startswith(tuple(f"{o}/" for o in PROCESS_OWNERS))]
        report["givers"] = sorted(name for name, giver in memory.givers.items()
                                  if (name, id(giver[0])) not in self.givers)
        report["graphs"] = []
        for ref, name, number in list(memory.graphs):
            graph = ref() if number > self.captures else None
            if graph is not None:
                report["graphs"].append({"pool": name, "capture": number, "held_by": held_by(graph)})
            graph = None
        level = self.torch()
        if level is None or self.level is None:
            report["torch"] = None
        else:
            grown = {key: max(0, level[key] - self.level[key]) for key in ("allocated", "reserved")}
            report["torch"] = {"before": {key: self.level[key] for key in ("allocated", "reserved")},
                               "after": {key: level[key] for key in ("allocated", "reserved")},
                               "excluded": level["excluded"], "grown": grown}
            if any(grown.values()):
                report["tensors"] = self.tensors()
        leaks = bool(report["ranges"] or report["givers"] or report["graphs"]
                     or (report["torch"] and any(report["torch"]["grown"].values())))
        report["returned"] = not leaks
        report["message"] = describe(report)
        return report

    def tensors(self):
        """The device tensors the closed plugins' modules still hold in their globals, largest first (TENSORS of
        them): {"plugin", "name", "bytes", "shape", "dtype"}. A plugin object still alive is walked as well."""
        torch = sys.modules.get("torch")
        if torch is None:
            return []
        alive = {name: found for name, found in ((n, r()) for n, r in self.watched) if found is not None}
        kept = set()
        for tensor in self.memory.process_tensors.values():
            try:
                kept.add(tensor.untyped_storage().data_ptr())
            except Exception:  # noqa: BLE001
                pass
        found, seen = [], set()
        try:
            for plugin, name, tensor in self.memory.holdings(alive, self.packages):
                storage = tensor.untyped_storage()
                at = storage.data_ptr()
                if at in seen or at in kept:
                    continue
                seen.add(at)
                found.append({"plugin": plugin, "name": name, "bytes": int(storage.nbytes()),
                              "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")})
        except Exception:  # noqa: BLE001 - the walk only names holders; the measurement stands without it
            pass
        return sorted(found, key=lambda row: -row["bytes"])[:TENSORS]

    # -- the close --------------------------------------------------------------------------------------------------
    def dropped(self):
        """The engine object was collected without its close (Core's loader finalizes it with this): it is no longer
        open on the device, and the next load measures it (refuse()). Nothing where it was closed."""
        memory = self.memory
        if self in memory.opened:
            memory.opened.remove(self)
            memory.abandoned.append(self)

    def close(self, quiet=False):
        """The engine is closed and dropped every reference it held: measure (measure()), record a close that did
        not return its memory on the device (memory.leaking; the next load refuses beside it) and raise LeakingClose
        under CRIT_CLOSE_CHECK=raise, else warn on stderr. quiet: only record it (a failed load's close, whose
        traceback still holds what the loader made: the next load measures it again). Returns the report."""
        memory = self.memory
        if self in memory.opened:
            memory.opened.remove(self)
        report = self.report = self.measure()
        if not report["judged"]:
            others = self.others()
            if others:   # the engine opened before this one is judged with this one's objects too
                others[0].watched.extend(self.watched)
                others[0].packages.update({k: v for k, v in self.packages.items() if k not in others[0].packages})
            return report
        if report["returned"] is False:
            memory.leaking = self
            message = f"the closed engine did not return its memory: {report['message']}"
            if quiet:
                return report
            if strict():
                raise LeakingClose(message, report)
            sys.stderr.write(f"crit_core: {message}; the next engine refuses to load beside it\n")
        return report


def gib(nbytes):
    return f"{nbytes / 2**30:.2f} GiB" if nbytes >= 1 << 28 else f"{nbytes / 2**20:.1f} MiB"


def describe(report):
    """A report in one line: what still holds the closed engine's memory, or that it returned."""
    parts = []
    for found in report.get("ranges") or ():
        parts.append(f"elastic range {found['name']} is open ({gib(found['backed'])} backed of "
                     f"{gib(found['reserved'])} reserved)")
    if report.get("givers"):
        parts.append("giver(s) still listed on the device: " + ", ".join(report["givers"]))
    graphs = report.get("graphs") or ()
    if graphs:
        pools = sorted({found["pool"] for found in graphs})
        held = next((found["held_by"][0] for found in graphs if found["held_by"]), None)
        parts.append(f"{len(graphs)} captured graph(s) alive ({', '.join(pools)})"
                     + (f", held by {held}" if held else ""))
    level = report.get("torch")
    if level and any(level["grown"].values()):
        parts.append(f"torch holds {gib(level['grown']['allocated'])} allocated and {gib(level['grown']['reserved'])} "
                     "reserved beyond its level before the load")
        named = [f"{t['plugin']} {t['name']} ({gib(t['bytes'])}, {t['dtype']} {t['shape']})"
                 for t in report.get("tensors") or ()]
        if named:
            parts.append("held in the plugins' modules or objects: " + "; ".join(named))
    if parts:
        for found in report.get("objects") or ():
            parts.append(f"plugin object {found['name']} ({found['type']}) alive"
                         + (f", held by {found['held_by'][0]}" if found["held_by"] else ""))
        return "; ".join(parts)
    if level is None:
        return "returned (its ranges, givers and graphs; torch's bytes cannot be read here)"
    return "returned"


# -- what holds an object -----------------------------------------------------------------------------------------
def _module_of(namespace):
    for name, module in list(sys.modules.items()):
        if module is not None and getattr(module, "__dict__", None) is namespace:
            return name
    return None


def _owner_of(namespace, own):
    """The object whose __dict__ `namespace` is, or None."""
    for found in gc.get_referrers(namespace):
        if id(found) not in own and getattr(found, "__dict__", None) is namespace and not isinstance(found, type):
            return found
    return None


def _edges(holder, target, own):
    """How `holder` refers to `target`: [(label, the object to go on from)]; a namespace (an object's __dict__) is
    stepped over to its object, a cell to its function."""
    if isinstance(holder, dict):
        keys = [key for key, value in list(holder.items()) if value is target] or ["?"]
        module = _module_of(holder)
        if module is not None:
            return [(f"module {module} global {keys[0]}", None)]
        owner = _owner_of(holder, own)
        if owner is not None:
            return [(f"{type(owner).__name__}.{keys[0]}", owner)]
        return [(f"dict[{keys[0]!r}]", holder)]
    if isinstance(holder, (list, tuple)):
        index = next((i for i, value in enumerate(holder) if value is target), "?")
        return [(f"{type(holder).__name__}[{index}]", holder)]
    if isinstance(holder, types.CellType):
        functions = [f for f in gc.get_referrers(holder) if id(f) not in own and isinstance(f, tuple)]
        found = [g for t in functions for g in gc.get_referrers(t)
                 if isinstance(g, types.FunctionType) and g.__closure__ is t]
        if found:
            return [(f"closure of {found[0].__qualname__}", found[0])]
        return [("a closure's cell", holder)]
    if isinstance(holder, types.MethodType):
        return [(f"bound method {holder.__qualname__}", holder)]
    if isinstance(holder, types.FunctionType):
        return [(f"function {holder.__qualname__}", holder)]
    names = list(getattr(type(holder), "__slots__", ()) or ())
    try:
        names += list(vars(holder))   # an object whose attributes live inline refers to them itself
    except TypeError:
        pass
    for name in names:
        try:
            if getattr(holder, name, None) is target:
                return [(f"{type(holder).__name__}.{name}", holder)]
        except Exception:  # noqa: BLE001 - an attribute that cannot be read names nothing
            continue
    return [(type(holder).__name__, holder)]


def held_by(target, most=2):
    """What keeps `target` alive: up to `most` chains of references, each from where it starts (a module's global,
    or an object nothing the collector sees refers to: native code, a thread's frame) to the target, as text
    ("module m global _KEPT -> list[0] -> Rounds.model -> Model"). [] where nothing the collector sees refers to it."""
    own = set()
    first = (target, [])
    paths, queue, seen = [], [first], {id(target)}
    own.update({id(paths), id(queue), id(seen), id(own), id(first)})
    end = type(target).__name__
    visits = 0
    frame = sys._getframe()
    while queue and len(paths) < most and visits < VISITS:
        current, path = queue.pop(0)
        own.add(id(path))
        referrers = [r for r in gc.get_referrers(current)
                     if id(r) not in own and r is not frame and not isinstance(r, types.FrameType)
                     and r is not queue and r is not paths]
        visits += 1 + len(referrers)
        if not referrers and path:
            paths.append(" -> ".join(["(referred to from outside Python's collector)"] + path[::-1] + [end]))
            continue
        for holder in referrers:
            for label, onward in _edges(holder, current, own):
                step = path + [label]
                if onward is None:   # a module's global: a root
                    paths.append(" -> ".join(step[::-1] + [end]))
                elif id(onward) not in seen and len(step) < DEPTH:
                    seen.add(id(onward))
                    entry = (onward, step)
                    own.update((id(entry), id(step)))
                    queue.append(entry)
            if len(paths) >= most:
                break
        del referrers
    if not paths and queue:   # nothing reached a root within the walk: the longest chain seen
        current, path = queue[-1]
        if path:
            paths.append(" -> ".join(["..."] + path[::-1] + [end]))
    return paths


__all__ = ["CHECK", "EXCLUDED", "LeakingClose", "Opened", "describe", "held_by", "open", "refuse", "strict"]
