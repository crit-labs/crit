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

Core chooses the bucket by the chunk's length alone, never by a time: the smallest captured bucket that holds it. A
replay computes the padded chunk, whose bits need not be the unpadded eager path's (below), so the length must decide
which arithmetic a chunk runs, in every process alike. A chunk past the largest bucket, a selection whose plugins
cannot run inside a captured graph (prompt_problems: a slot's plugin lacks prompts:<slot>, or its round_problems()
gives a reason) and a host device run the eager path; a model that wants its prompt chunks eager captures no buckets.

The output equals the eager prompt path's over the same padded batch exactly (the same plugins' calls, replayed); it
equals the unpadded eager path's where every plugin's prompt form is row-invariant (a row's result independent of the
rows beside it and of the token count), which plugins promise per slot.
"""
from .captures import capture_graph, capture_problems
from .device import PromptBatch
from .contracts import policy_slot
from .manifest import PluginError
from .memory import of as memory_of
from .rounds import _method, capture_stream

BUCKETS = (16, 64, 128, 512)
# Replays of a captured bucket before its time is measured (the first ones upload the graph and warm its caches), and
# replays timed: the offer carries their median.
WARM_REPLAYS = 8
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
    if uses is None:
        uses = [slot for slot in parts.slots() if parts.plugin(slot) and slot != "model"
                and not policy_slot(parts, slot) and slot not in DECODE_SLOTS]
    return capture_problems(parts, uses, prompt_fact, "the prompt chunk", "prompt form")


class PromptGraphs:
    """Captured prompt chunks for a model plugin (the module's docstring). forward(batch) -> tensor: the chunk's
    forward over a PromptBatch, capture-safe, its output at a fixed address (the last real token's row, batch.last_t);
    buckets: the token counts captured; pools and extra_parts: the objects that stage and settle a chunk's sequence
    (stage_prompt(batch) before, commit_prompt(batch) after each run) and put back what the warm-ups changed
    (reset_capture()); parts (the model factory's) and uses (the slots forward calls) check the selection's prompt facts
    (prompt_problems) and give the placement plugin the captures' memory. share_pool: the buckets share one graph memory pool, the engine's where Core's loader
    assembled the engine (its rounds capture into it too: crit_plugins.rounds' rule for the plugins holds), else
    one of the driver's own; None (the default) shares where the engine has a pool, False gives each bucket its
    own."""

    def __init__(self, forward, *, device, buckets=BUCKETS, parts=None, uses=None, pools=(), extra_parts=(),
                 share_pool=None):
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
        self._observer = getattr(parts, "observer", None) if parts is not None else None
        self.slot_calls = {}  # bucket -> the slots its capture called (each replay counts as a call of each)
        self.problems = self._problems()
        self.graphs = {}      # bucket -> (graph, its output)
        self.captured_bytes, self._planned = 0, 0
        self._engine_pool = getattr(getattr(parts, "context", None), "graphs", None)
        self.share_pool = bool(share_pool) or (share_pool is None and self._engine_pool is not None)
        self._pool = None
        self.counts = {"graph_chunks": 0, "eager_chunks": 0, "tokens": 0, "padding": 0}

    def _problems(self):
        parts, uses = self._checked
        return [] if parts is None else prompt_problems(parts, uses)

    # Choosing ---------------------------------------------------------------------------------------------------------

    def bucket_for(self, count):
        """The smallest captured bucket of at least `count` tokens, or None."""
        return next((b for b in self.buckets if b >= count and b in self.graphs), None)

    def choose(self, count):
        """The bucket a chunk of `count` tokens runs in, or None for the model's eager path: the smallest captured
        bucket that holds the chunk, where the selection's plugins run inside a captured graph."""
        if self.problems or not 1 <= count:
            return None
        return self.bucket_for(count)

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
        no state), then the capture on the process's capture stream with no garbage collection inside it (the batch's
        derive() and the forward), then WARM_REPLAYS replays of padding. The device memory it took is planned through
        the placement plugin."""
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
            for _ in range(2):   # warm-ups of padding alone, which change no state
                self._stage(bucket, None, 0, [])
                self.forward(self.batch)
                stream.synchronize()
            self._reset()
            memory = memory_of(self.device)
            if self.share_pool and self._pool is None:
                self._pool = self._engine_pool or memory.pool("prompts", "every bucket")
            pool = self._pool if self.share_pool else memory.pool("prompts", f"{bucket} tokens")
            self._stage(bucket, None, 0, [], derive=False)

            def chunk():
                self.batch.derive()
                return self.forward(self.batch)
            graph, output, self.slot_calls[bucket] = capture_graph(self.device, stream, pool, chunk, self._observer)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self._reset()
        self._stage(bucket, None, 0, [], derive=False)
        for _ in range(WARM_REPLAYS):
            graph.replay()
        torch.cuda.synchronize(self.device)
        grown = max(torch.cuda.memory_allocated(self.device) - allocated,
                    torch.cuda.memory_reserved(self.device) - reserved, 0)
        self.captured_bytes += grown
        self.graphs[bucket] = (graph, output)
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
