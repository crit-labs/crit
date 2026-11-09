"""Short prompt chunks as captured graphs: Core's driver a model plugin runs a sequence's prompt chunk through, so that
the chunk's whole forward (every layer's plugin calls) is one graph replay instead of each kernel's launch and the
Python around it.

A chunk of N tokens runs in the smallest captured bucket of at least N tokens (BUCKETS by default, the model's to
choose): its real tokens first, then padding that changes no state, in a crit_plugins.device.PromptBatch at fixed
addresses. The model gives the chunk's forward(batch) -> its output (the last real token's logits, say), the same
prompt arithmetic its eager path runs over its plugins, and every plugin it calls provides prompts:<slot>
(DEVICE_EXTENSIONS): its prompt form reads the batch's device tensors only and its padding writes nothing a sequence
keeps. Pools stage the chunk's sequence into their fixed buffers before each replay (stage_prompt) and settle it
after (commit_prompt). Each bucket is captured once at load (two warm-ups of padding alone, which change no state,
then the capture), in a graph memory pool of its own, and the memory the captures take is planned through the
placement plugin (the entry "prompts:captures").

The graphs slot's plugin chooses, chunk by chunk, between the model's eager path and a padded bucket (its choose():
the offers carry each bucket's measured replay time, the eager time measured at the bucket and the arithmetic each
computes, "bits": a timing may choose only among offers of the same bits); without one, the smallest bucket runs. A chunk past the largest bucket, a selection whose plugins cannot run inside a captured graph
(prompt_problems: a slot's plugin lacks prompts:<slot>, or its round_problems() gives a reason) and a host device run
the eager path. Mechanism and policy split as for decode rounds: Core owns the batch, the captures, their memory and
the facts check; the graphs slot (a policy slot, on the host) only chooses among the offers.

The output equals the eager prompt path's over the same padded batch exactly (the same plugins' calls, replayed); it
equals the unpadded eager path's where every plugin's prompt form is row-invariant (a row's result independent of the
rows beside it and of the token count), which plugins promise per slot.
"""
import gc
import time

from .device import PromptBatch
from .manifest import POLICY_SLOTS, PluginError
from .rounds import _method, _own_problems, capture_stream

BUCKETS = (16, 64, 128, 512)
# Replays of a captured bucket before its time is measured (the first ones upload the graph and warm its caches), and
# replays timed: the offer carries their median.
WARM_REPLAYS, TIMED_REPLAYS = 8, 25
PHASE = "prefill"
IDENTITY = ("prompt-graphs",)
# The slots only decoding calls: a prompt chunk's forward never does.
DECODE_SLOTS = ("sampler", "drafter", "verifier")


class PromptsRefused(PluginError):
    """A selection whose plugins cannot run a prompt chunk inside a captured graph: the model runs it eagerly."""


def prompt_fact(slot):
    """The fact a slot's plugin provides when its prompt form runs inside a captured prompt chunk."""
    return "prompts:" + slot


def prompt_problems(parts, uses=None):
    """Why a prompt chunk over the plugins of `parts` (a factory's parts) cannot be captured, or []: each slot of `uses`
    (the slots the model's prompt forward calls; by default every filled computing slot but the decoding ones) provides
    prompts:<slot>, and each of those plugins that has round_problems() (the reasons its configuration keeps it out of
    a captured graph) gives none."""
    filled = [slot for slot in parts.slots() if parts.plugin(slot)]
    if uses is None:
        uses = [slot for slot in filled if slot != "model" and slot not in POLICY_SLOTS and slot not in DECODE_SLOTS]
    problems = []
    for slot in dict.fromkeys(uses):
        plugin = parts.plugin(slot)
        if plugin is None:
            problems.append(f"no plugin fills {slot}, which the prompt chunk calls")
        elif prompt_fact(slot) not in parts.facts(slot):
            problems.append(f"{plugin} in {slot} does not provide {prompt_fact(slot)}: its prompt form is not known to "
                            "run inside a captured graph")
        else:
            problems += [f"{plugin} in {slot} cannot run inside a captured graph as configured: {reason}"
                         for reason in _own_problems(parts, slot, plugin)]
    return problems


class PromptGraphs:
    """Captured prompt chunks for a model plugin (the module's docstring). forward(batch) -> tensor: the chunk's
    forward over a PromptBatch, capture-safe, its output at a fixed address (the last real token's row, batch.last_t);
    buckets: the token counts captured; pools and extra_parts: the objects that stage and settle a chunk's sequence
    (stage_prompt(batch) before, commit_prompt(batch) after each run) and put back what the warm-ups changed
    (reset_capture()); parts (the model factory's) and uses (the slots forward calls) check the selection's prompt facts
    (prompt_problems), give the graphs slot's plugin (parts.get("graphs")) its choice and the placement plugin the
    captures' memory. share_pool: the buckets share one graph memory pool."""

    def __init__(self, forward, *, device, buckets=BUCKETS, parts=None, uses=None, pools=(), extra_parts=(),
                 share_pool=False):
        import torch
        self.torch, self.forward, self.device = torch, forward, torch.device(device)
        buckets = sorted({int(b) for b in buckets})
        if not buckets or buckets[0] < 1:
            raise ValueError("prompt buckets are token counts of one or more")
        self.buckets = buckets
        self.batch = PromptBatch(buckets[-1], self.device)
        found = {}
        for part in (*pools, *extra_parts):
            if part is not None:
                found.setdefault(id(part), part)
        self.parts = list(found.values())
        self._checked = (parts, uses)
        get = getattr(parts, "get", None) if parts is not None else None
        self._placement = get("placement") if callable(get) else None
        self._chooser = get("graphs") if callable(get) else None
        self._observer = getattr(parts, "observer", None) if parts is not None else None
        self.slot_calls = {}  # bucket -> the slots its capture called (each replay counts as a call of each)
        self.problems = self._problems()
        self.graphs = {}      # bucket -> (graph, its output)
        self.costs = {}       # bucket -> {"eager_ns", "graph_ns", "bytes"}
        self.captured_bytes, self._planned = 0, 0
        self.share_pool, self._pool = bool(share_pool), None
        self.counts = {"graph_chunks": 0, "eager_chunks": 0, "tokens": 0, "padding": 0}
        self._chosen = {}     # chunk size -> its bucket or None, while the offers stay as they are

    def _problems(self):
        parts, uses = self._checked
        return [] if parts is None else prompt_problems(parts, uses)

    # Choosing ---------------------------------------------------------------------------------------------------------

    def bucket_for(self, count):
        """The smallest captured bucket of at least `count` tokens, or None."""
        return next((b for b in self.buckets if b >= count and b in self.graphs), None)

    def offers(self, count):
        """The graphs slot's offers for a chunk of `count` tokens: the model's eager path (its time measured at the
        smallest bucket that holds the chunk) and each captured bucket that holds it, padded (its replay measured).
        Each offer's "bits" names the arithmetic it computes: a bucket's replay runs the slots' prompt forms over the
        padded chunk (a PromptBatch), which need not give a row the bits the model's eager path (a StepBatch) gives it,
        nor the bits of another bucket's padding, so every bucket is its own ("padded:<bucket>"), one of exactly the
        chunk's tokens included, and the eager path is "eager". A graphs plugin chooses by time only among offers of
        the same bits."""
        shape = {"phase": PHASE, "tokens": count, "rows": 1, "identity": IDENTITY}
        fitting = [b for b in self.buckets if b >= count and b in self.graphs]
        eager_ns = self.costs[fitting[0]]["eager_ns"] if fitting else 0
        offers = [{"name": "eager", "mode": "eager", "shape": shape, "run_ns": eager_ns, "workspace_bytes": 0,
                   "resident_bytes": 0, "setup_ns": 0, "setup_peak_bytes": 0, "ready": True, "bits": "eager"}]
        for bucket in fitting:
            offers.append({"name": f"graph:{bucket}", "mode": "full", "shape": {**shape, "tokens": bucket},
                           "run_ns": self.costs[bucket]["graph_ns"], "workspace_bytes": 0,
                           "resident_bytes": self.costs[bucket]["bytes"], "setup_ns": 0, "setup_peak_bytes": 0,
                           "ready": True, "bits": f"padded:{bucket}"})
        return shape, offers

    def choose(self, count):
        """The bucket a chunk of `count` tokens runs in, or None for the model's eager path: the graphs slot's choice
        among the offers when the selection fills it, else the smallest captured bucket that holds the chunk. Asked
        once per chunk size; a capture changes the offers and asks again."""
        if count in self._chosen:
            return self._chosen[count]
        bucket = self._choose(count)
        self._chosen[count] = bucket
        return bucket

    def _choose(self, count):
        if self.problems or not 1 <= count or self.bucket_for(count) is None:
            return None
        if self._chooser is None:
            return self.bucket_for(count)
        shape, offers = self.offers(count)
        free = self.torch.cuda.mem_get_info(self.device)[0]
        name = self._chooser.choose(shape, offers, int(free))
        if name == "eager":
            return None
        if not isinstance(name, str) or not name.startswith("graph:") or int(name[6:]) not in self.graphs:
            raise PluginError(f"the graphs plugin chose {name!r}, which is none of the offers")
        return int(name[6:])

    # Running ----------------------------------------------------------------------------------------------------------

    def run(self, sequence, frontier, tokens):
        """The chunk `tokens` of `sequence` from position `frontier` as a captured graph: its output tensor (the graph's
        own, overwritten by the next run: read or copy it first), or None when it runs on the model's eager path
        instead (choose). Every part stages the sequence before the replay and settles it after. While Core's loader
        counts the calls each slot gets (parts.observer), the replay counts as a call of every slot its capture
        called."""
        bucket = self.choose(len(tokens))
        if bucket is None:
            self.counts["eager_chunks"] += 1
            return None
        graph, output = self.graphs[bucket]
        self._stage(bucket, sequence, frontier, tokens, derive=False)   # the graph derives positions and validity
        graph.replay()
        if self._observer is not None:
            self._observer.replayed(self.slot_calls[bucket])
        self._settle()
        self.counts["graph_chunks"] += 1
        self.counts["tokens"] += len(tokens)
        self.counts["padding"] += bucket - len(tokens)
        return output

    def run_eager(self, sequence, frontier, tokens, bucket):
        """The same padded chunk run eagerly (the plugins' calls the graph replays, made one by one): what run() gives
        for it, for tests and for a device without graphs."""
        self._stage(bucket, sequence, frontier, tokens)
        output = self.forward(self.batch)
        self._settle()
        return output

    def _stage(self, bucket, sequence, frontier, tokens, derive=True):
        self.batch.fill(bucket, sequence, frontier, tokens, derive=derive)
        for part in self.parts:
            stage = _method(part, "stage_prompt")
            if stage is not None:
                stage(self.batch)

    def _settle(self):
        for part in self.parts:
            commit = _method(part, "commit_prompt")
            if commit is not None:
                commit(self.batch)

    def telemetry(self):
        """Chunks run as graphs and eagerly, and the real and padding tokens the graphs ran (host counters)."""
        return {"prompt_graphs": {**self.counts, "buckets": sorted(self.graphs)}}

    # Capturing --------------------------------------------------------------------------------------------------------

    def prepare(self):
        """Capture every bucket where the selection can (a GPU, no prompt_problems); returns the buckets captured."""
        self.problems = self._problems()
        if self.problems or self.device.type != "cuda":
            return []
        for bucket in self.buckets:
            if bucket not in self.graphs:
                self.capture(bucket)
        return sorted(self.graphs)

    def capture(self, bucket):
        """Capture the chunk of `bucket` tokens: two eager warm-ups of padding alone (no sequence staged: they change
        no state), the second timed, then the capture on the process's capture stream with no garbage collection
        inside it (the batch's derive() and the forward), then WARM_REPLAYS replays of padding and TIMED_REPLAYS
        timed, their median the offer's time. The device memory it took is planned through the placement plugin."""
        torch = self.torch
        self.problems = self._problems()
        if self.problems:
            raise PromptsRefused("the prompt chunk cannot be captured: " + "; ".join(self.problems))
        if self.device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        if bucket not in self.buckets:
            raise ValueError(f"{bucket} is none of the buckets {self.buckets}")
        torch.cuda.synchronize(self.device)
        allocated, reserved = torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device)
        stream = capture_stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._reset()
            eager_ns = None
            for _ in range(2):
                self._stage(bucket, None, 0, [])
                started = time.perf_counter_ns()
                self.forward(self.batch)
                stream.synchronize()
                eager_ns = time.perf_counter_ns() - started
            self._reset()
            graph = torch.cuda.CUDAGraph()
            if self.share_pool and self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            self._stage(bucket, None, 0, [], derive=False)
            collecting = gc.isenabled()
            gc.collect()   # nothing dead is freed inside the capture (pinned host memory freed there breaks the next)
            gc.disable()
            observed = self._observer.snapshot() if self._observer is not None else None
            try:
                with torch.cuda.graph(graph, pool=self._pool if self.share_pool else None, stream=stream):
                    self.batch.derive()
                    output = self.forward(self.batch)
            finally:
                if collecting:
                    gc.enable()
            self.slot_calls[bucket] = self._observer.called_since(observed) if observed is not None else []
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self._reset()
        self._stage(bucket, None, 0, [], derive=False)
        for _ in range(WARM_REPLAYS):
            graph.replay()
        torch.cuda.synchronize(self.device)
        timings = []
        for _ in range(TIMED_REPLAYS):
            started = time.perf_counter_ns()
            graph.replay()
            torch.cuda.synchronize(self.device)
            timings.append(time.perf_counter_ns() - started)
        grown = max(torch.cuda.memory_allocated(self.device) - allocated,
                    torch.cuda.memory_reserved(self.device) - reserved, 0)
        self.captured_bytes += grown
        self.graphs[bucket] = (graph, output)
        self.costs[bucket] = {"eager_ns": eager_ns, "graph_ns": sorted(timings)[len(timings) // 2], "bytes": grown}
        self._chosen.clear()
        self._plan()
        return graph

    def _reset(self):
        for part in self.parts:
            reset = _method(part, "reset_capture")
            if reset is not None:
                reset()

    def _plan(self):
        """The device memory the captures took so far, planned through the placement plugin (the entry
        "prompts:captures"), so Core's loader counts it among the engine's planned bytes."""
        if self._placement is None or self.captured_bytes <= self._planned:
            return
        grown = self.captured_bytes - self._planned
        free = self.torch.cuda.mem_get_info(self.device)[0]
        placed = self._placement.plan({"prompts:captures": grown}, {"device": free + grown, "host": 0})
        if placed.get("prompts:captures") != "device":
            raise PromptsRefused("the placement plugin put the captured prompt chunks' memory off the device")
        self._planned = self.captured_bytes


__all__ = ["BUCKETS", "PromptGraphs", "PromptsRefused", "prompt_fact", "prompt_problems"]
