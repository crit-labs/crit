"""Captured graphs of any pass a model plugin runs over fixed buffers: Core's general capture helper, for passes the
round driver (crit_plugins.rounds) and the prompt graphs (crit_plugins.prompt_graphs) do not shape. A model plugin
that decodes its own way (no drafts and no verify, state in a layout of its own with no bind, unbind or commit_round),
a layer group it replays on its own, a step of fixed shape: each is a capture-safe `forward(key)` over buffers at
fixed addresses, and Captures captures one graph per key (a row count, a token bucket, any hashable shape the plugin
names) and replays it.

    captures = Captures(forward, device=device, keys=(1, 2, 4), parts=parts, uses=("projections", "norms"),
                        extra_parts=[my_state])
    captures.prepare()             # at load: capture every key where the selection can (else nothing)
    out = captures.run(rows)       # the key's graph, replayed; the eager forward(key) where it is not captured

Mechanism, as the round driver's: before any capture the selection is checked (capture_problems: every slot of `uses`
provides the fact the capture needs, `rounds:<slot>` by default, and none gives a round_problems() reason of its own);
each key is warmed up eagerly (`warmups`, on the capture stream) with every part's reset_capture() before each warm-up
and after the capture, so what the warm-ups and the capture changed is put back; the capture runs with no garbage
collection inside it and with cuBLAS workspaces made in Core's pool for them (crit_plugins.rounds.capture_workspaces);
the graphs share the engine's graph memory pool where Core's loader made one (share_pool, as the rounds and prompt
graphs), else one of their own; their memory is planned through the placement plugin (an entry "<name>:captures"), and
while Core's loader counts each slot's calls a replay counts as a call of every slot its capture called. What the
helper does not do is the plugin's: putting a step's inputs into the fixed buffers before run(), reading the output
(a tensor of the graph's own, overwritten by the next run) after it.

The device work sits behind a capture backend, so the helper's logic is tested on a host without CUDA: CudaCapture
(the default) captures CUDA graphs; EagerCapture records the pass and replays it by running it again into the captured
outputs, for tests of a plugin's capture logic (keys, warm-ups, resets, refusals, counts) on the CPU. A replay of a
CUDA graph gives what the eager forward gives over the same buffers; that only holds on a GPU, where a plugin's tests
check it.
"""
import gc

from .manifest import POLICY_SLOTS, PluginError
from .memory import of as memory_of
from .rounds import _method, _own_problems, capture_stream, capture_workspaces, round_fact


class CapturesRefused(PluginError):
    """A selection whose plugins cannot run inside a captured graph: the plugin runs its pass eagerly."""


def capture_problems(parts, uses=None, fact=round_fact, calls="the captured pass", form="device form"):
    """Why a captured pass over the plugins of `parts` (a factory's parts) cannot be captured, or []: each slot of
    `uses` (the slots the pass calls; by default every filled computing slot) provides fact(slot) (round_fact: the
    rounds:<slot> a plugin's device form provides when it runs inside a captured graph), and each of those plugins that
    has round_problems() (the reasons its configuration keeps it out of a captured graph) gives none. `calls` and
    `form` name the pass and the plugins' form in the reasons."""
    if uses is None:
        uses = [slot for slot in parts.slots() if parts.plugin(slot) and slot != "model" and slot not in POLICY_SLOTS]
    problems = []
    for slot in dict.fromkeys(uses):
        plugin = parts.plugin(slot)
        if plugin is None:
            problems.append(f"no plugin fills {slot}, which {calls} calls")
        elif fact(slot) not in parts.facts(slot):
            problems.append(f"{plugin} in {slot} does not provide {fact(slot)}: its {form} is not known to run inside "
                            "a captured graph")
        else:
            problems += [f"{plugin} in {slot} cannot run inside a captured graph as configured: {reason}"
                         for reason in _own_problems(parts, slot, plugin)]
    return problems


def capture_graph(device, stream, pool, run, observer=None):
    """run() captured into a new CUDA graph on `stream` (the process's capture stream, crit_plugins.rounds.
    capture_stream) in the graph memory `pool`, with fresh cuBLAS workspaces (capture_workspaces) and no garbage
    collection inside the capture (a collection there frees what dead cycles held, pinned host memory among it, which
    breaks the next capture's flush of torch's host cache): collected first, so nothing dead outlives the capture.
    Returns (graph, what run() returned, the slots the capture called by Core's loader's count, [] without one)."""
    import torch
    graph = torch.cuda.CUDAGraph()
    memory = memory_of(device)
    collecting = gc.isenabled()
    gc.collect()
    gc.disable()
    observed = observer.snapshot() if observer is not None else None
    try:
        capture_workspaces(stream)
        with memory.capturing(pool), torch.cuda.graph(graph, pool=pool, stream=stream):
            output = run()
    finally:
        if collecting:
            gc.enable()
    return graph, output, observer.called_since(observed) if observed is not None else []


class CudaCapture:
    """The capture backend of CUDA devices: graphs captured on the process's capture stream (capture_graph), their
    warm-ups on it too, the device memory each capture took measured."""

    def capturable(self, device):
        return device.type == "cuda"

    def pool(self, device, label):
        return memory_of(device).pool("captures", label)

    def capture(self, device, warm, run, pool, observer=None):
        """warm() (the eager warm-ups) then run() captured; returns (graph, output, slots called, device bytes)."""
        import torch
        torch.cuda.synchronize(device)
        allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
        stream = capture_stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            warm()
            stream.synchronize()
            graph, output, called = capture_graph(device, stream, pool, run, observer)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        grown = max(torch.cuda.memory_allocated(device) - allocated, torch.cuda.memory_reserved(device) - reserved, 0)
        return graph, output, called, grown

    def replay(self, graph):
        graph.replay()

    def free(self, device):
        import torch
        return int(torch.cuda.mem_get_info(device)[0])


class EagerCapture:
    """A capture backend for tests on any device: a "capture" runs the pass once (its warm-ups first, as a capture's)
    and keeps its output tensors, and a replay runs the pass again and copies what it gives into them, so the output
    lives at the captured addresses as a graph's does. The slots the capture called are counted as on CUDA, and a
    replay's own calls are not (a graph's replay makes none). It checks
    a plugin's capture logic (keys, warm-ups, resets, refusals, counts, plans), not that its pass is capture-safe."""

    def __init__(self, free=1 << 40):
        self.free_bytes, self.captured, self.replays = int(free), 0, 0

    def capturable(self, device):
        return True

    def pool(self, device, label):
        return label

    def capture(self, device, warm, run, pool, observer=None):
        warm()
        observed = observer.snapshot() if observer is not None else None
        output = run()
        called = observer.called_since(observed) if observed is not None else []
        self.captured += 1
        return _Recorded(run, output, observer), output, called, 0

    def replay(self, graph):
        self.replays += 1
        graph.replay()

    def free(self, device):
        return self.free_bytes


class _Recorded:
    """A pass EagerCapture recorded. Its replay runs the pass with the loader's counting paused, as a graph's replay
    makes no call: the driver counts it (replayed)."""

    def __init__(self, run, output, observer=None):
        self.run, self.output, self.observer = run, output, observer

    def replay(self):
        counting = self.observer is not None and self.observer.active
        if counting:
            self.observer.active = False
        try:
            _copy_into(self.output, self.run())
        finally:
            if counting:
                self.observer.active = True


def _copy_into(kept, fresh):
    if isinstance(kept, (list, tuple)):
        for a, b in zip(kept, fresh):
            _copy_into(a, b)
    elif isinstance(kept, dict):
        for name in kept:
            _copy_into(kept[name], fresh[name])
    elif kept is not None:
        kept.copy_(fresh)


class Captures:
    """Captured graphs of forward(key), one per key (the module's docstring). forward: a capture-safe pass over
    buffers at fixed addresses, its output tensors (a tensor, or a tuple, list or dict of them) its own; keys: the
    keys prepare() captures; parts (the model factory's) and uses (the slots forward calls) check the selection's
    facts (capture_problems with `fact`, rounds:<slot> by default), give the placement plugin the captures' memory and
    the loader's call counter the replays; extra_parts: objects whose reset_capture() puts back what a warm-up or the
    capture changed (the plugin's own state: its pools, its positions); warmups: eager runs of each key before its
    capture; name: the captures' name in memory pools, plans ("<name>:captures") and telemetry; share_pool as
    crit_plugins.rounds' (None: the engine's pool where it has one; False: each key its own); backend: CudaCapture
    by default, EagerCapture to test the logic without CUDA."""

    def __init__(self, forward, *, device, keys=(), parts=None, uses=None, fact=round_fact, extra_parts=(),
                 warmups=2, name="captures", share_pool=None, backend=None):
        import torch
        if warmups < 0:
            raise ValueError("warm-ups are a count of zero or more")
        if not isinstance(name, str) or not name or ":" in name:
            raise ValueError("a captures' name is a word, with no colon")
        self.torch, self.forward, self.device = torch, forward, torch.device(device)
        self.keys, self.warmups, self.name = list(dict.fromkeys(keys)), int(warmups), name
        self.backend = backend if backend is not None else CudaCapture()
        found = {}
        for part in extra_parts:
            if part is not None:
                found.setdefault(id(part), part)
        self.parts = list(found.values())
        self._checked = (parts, uses, fact)
        get = getattr(parts, "get", None) if parts is not None else None
        self._placement = get("placement") if callable(get) else None
        self._observer = getattr(parts, "observer", None) if parts is not None else None
        self._engine_pool = getattr(getattr(parts, "context", None), "graphs", None)
        self.share_pool = bool(share_pool) or (share_pool is None and self._engine_pool is not None)
        self._pool, self._pools = None, {}
        self.problems = self._problems()
        self.graphs = {}       # key -> (graph, its output)
        self.slot_calls = {}   # key -> the slots its capture called (each replay counts as a call of each)
        self.captured_bytes, self._planned = 0, 0
        self.counts = {"replays": 0, "eager": 0}

    def _problems(self):
        parts, uses, fact = self._checked
        return [] if parts is None else capture_problems(parts, uses, fact)

    # Running ----------------------------------------------------------------------------------------------------------

    def captured(self, key):
        return key in self.graphs

    def run(self, key):
        """The pass of `key`: its graph replayed (its output tensors, the graph's own: read or copy them before the
        next run), or the eager forward(key) where the key is not captured. A replay counts as a call of every slot
        its capture called while Core's loader counts them."""
        if key not in self.graphs:
            self.counts["eager"] += 1
            return self.forward(key)
        graph, output = self.graphs[key]
        self.backend.replay(graph)
        if self._observer is not None:
            self._observer.replayed(self.slot_calls[key])
        self.counts["replays"] += 1
        return output

    def telemetry(self):
        """Replays and eager runs so far, the keys captured and the device bytes their captures took."""
        return {"captures": {self.name: {**self.counts, "keys": [repr(k) for k in self.graphs],
                                         "captured_bytes": self.captured_bytes}}}

    # Capturing --------------------------------------------------------------------------------------------------------

    def prepare(self, keys=None):
        """Capture each key (default the driver's keys) where the selection can (a capturable device, no problems);
        returns the keys captured."""
        self.problems = self._problems()
        if self.problems or not self.backend.capturable(self.device):
            return []
        for key in (self.keys if keys is None else keys):
            if key not in self.graphs:
                self.capture(key)
        return [key for key in self.graphs]

    def capture(self, key):
        """Capture the pass of `key`: `warmups` eager runs, each after every part's reset_capture(), then the capture
        after one more, then a last reset, so the parts' state is what it was. Refuses (CapturesRefused) a selection
        with problems. The device memory it took is planned through the placement plugin."""
        self.problems = self._problems()
        if self.problems:
            raise CapturesRefused(f"the {self.name} cannot be captured: " + "; ".join(self.problems))
        if not self.backend.capturable(self.device):
            raise ValueError(f"{self.device} has no captured graphs")
        if key in self.graphs:
            return self.graphs[key][0]

        def warm():
            for _ in range(self.warmups):
                self._reset()
                self.forward(key)
            self._reset()
        graph, output, called, grown = self.backend.capture(self.device, warm, lambda: self.forward(key),
                                                            self._pool_for(key), self._observer)
        self._reset()
        self.graphs[key], self.slot_calls[key] = (graph, output), list(called)
        self.captured_bytes += grown
        self._plan()
        return graph

    def _pool_for(self, key):
        if self.share_pool:
            if self._pool is None:
                self._pool = self._engine_pool or self.backend.pool(self.device, f"{self.name}, every key")
            return self._pool
        if key not in self._pools:
            self._pools[key] = self.backend.pool(self.device, f"{self.name}, {key!r}")
        return self._pools[key]

    def _reset(self):
        for part in self.parts:
            reset = _method(part, "reset_capture")
            if reset is not None:
                reset()

    def _plan(self):
        """The device memory the captures took so far, planned through the placement plugin (the entry
        "<name>:captures"), so Core's loader counts it among the engine's planned bytes."""
        if self._placement is None or self.captured_bytes <= self._planned:
            return
        grown, entry = self.captured_bytes - self._planned, f"{self.name}:captures"
        free = self.backend.free(self.device)
        placed = self._placement.plan({entry: grown}, {"device": free + grown, "host": 0})
        if placed.get(entry) != "device":
            raise CapturesRefused(f"the placement plugin put the memory of the {self.name} captures off the device")
        self._planned = self.captured_bytes


__all__ = ["Captures", "CapturesRefused", "CudaCapture", "EagerCapture", "capture_graph", "capture_problems"]
