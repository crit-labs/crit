"""Prompt spans run layer by layer: a pass.

A prompt chunk runs through every layer, then the next chunk does. Where a layer's weights are not all on the device
that order is the costly one: an experts plugin that keeps routed experts in host memory brings a layer's experts to
the device for every chunk that reads them (a long chunk routes nearly all of them), so a prompt of n chunks moves
every expert of every layer over the link n times, gigabytes a chunk whatever its size. A pass turns the loops
around: every chunk of a span runs through layer 0, then every chunk through layer 1, and so on, so each layer's
experts cross the link once for the whole span (the experts plugin's `stages_prompts()` says it stages so, and its
cache serves a layer's later chunks from what the first one staged).

Nothing else changes. A chunk through a layer is the same call with the same inputs in either order: the layer's
keys, values, index keys and recurrent state of the earlier chunks are in its pools when a chunk reads them, because
the layer ran those chunks first, and what a layer leaves the next is the residual's state of each chunk
(residual.py: begin, advance, end), which a pass keeps between layers. So a prompt's tokens are the bits of the
chunks run one after another, chunk boundaries included, and a pass is no decision an Eval has to see.

A pass takes as many of Core's steps as its work takes time (Core's prefill passes: crit/core/model.h): a step runs
chunks through layers until its time is up, reports the span as the pass's tokens and consumes nothing, and the step
in which the last layer ends consumes the span. A cancel settles between those steps as between any others. That is
also what keeps a single chunk's step inside the cancel grace when the chunk's experts alone take longer to bring
to the device than the grace allows: its layers spread over the steps they need.

States keeps the chunks' residual states between layers in device memory the device's givers make room for
(crit_plugins.memory: a paged range, so an elastic expert cache gives the bytes and takes them back after the pass)
and outside torch's allocator, whose blocks a prompt step's memory is measured by. Where the device has no elastic
ranges (the CPU, a driver without the virtual memory API) the states are the tensors themselves.

A pass ends where the prompt cache marks a place, as a chunk does: a mark keeps every layer's state at one
position. A KV pool that keeps a sliding window's ring of pages runs passes as any other: a ring page is a
position's, the same for every layer, and each layer has its own keys and values there, so when the next layer
comes back to the span's first chunk it finds its own window before it, written when that layer last ran those
positions.
"""
import time

from crit_plugins import StepBatch

from .blocks import Context
from . import residual

ALIGN = 64      # a stored tensor starts at a multiple of this many bytes
UNIT = 1 << 20  # the states' paged range is backed in units of this many bytes


class States:
    """The residual states of a pass's chunks between its layers. `sizes`: the chunks' token counts, `budget`: the
    most bytes the states may take; `fit` is how many of the chunks, from the first, the budget and the device's
    memory hold (the givers making room: an elastic expert cache gives its coldest slots), one at least. put(chunk,
    state) keeps a state (a tuple of tensors, None where the residual has none yet), get(chunk) gives it back as
    the tensors the next layer advances, in place where it writes in place; close() gives the memory back."""

    def __init__(self, model, sizes, budget):
        self.m, self.paged = model, None
        self.kept = {}      # chunk -> its state: tensors of the range, or the tensors themselves
        self.bases = []     # each chunk's first byte and its bytes in the range
        end = 0
        for tokens in sizes:
            need = self.chunk_bytes(tokens)
            if self.bases and end + need > int(budget):
                break
            self.bases.append((end, need))
            end += -(-need // ALIGN) * ALIGN
        self.fit = len(self.bases)
        memory = self._memory()
        if memory is None or not memory.elastic_ranges:
            self.bases = []   # no range here: every state stays where torch made it
            return
        units = -(-end // UNIT)
        try:
            self.paged = memory.paged("model-template", "pass", units * UNIT, [(0, UNIT, units)])
        except MemoryError:   # not a granule left: the states stay in torch's allocator, a chunk at a time
            self.bases, self.fit = [], 1
            return
        while self.fit > 1 and not self.paged.back(self._units(self.fit)):
            self.fit -= 1     # the device has no memory for that many chunks' states: a shorter pass
        if self.fit == 1 and not self.paged.back(self._units(1)):
            self.bases = []
        self.bases = self.bases[:self.fit]

    def _memory(self):
        if self.m.device.type != "cuda":
            return None
        from crit_plugins.memory import of
        return of(self.m.device)

    def _units(self, chunks):
        first, size = self.bases[chunks - 1]
        return -(-(first + size) // UNIT)

    def chunk_bytes(self, tokens):
        """The bytes a chunk of `tokens` tokens is given: the residual's state at the embedding rows' width, and a
        start for each of its tensors."""
        return int(tokens) * residual.carry(self.m, 2) + 4 * ALIGN

    @property
    def bytes(self):
        """The device bytes the states' range takes now."""
        return self.paged.bytes if self.paged is not None else 0

    def put(self, chunk, state):
        """Keep a chunk's state: each tensor in the chunk's bytes of the range (a tensor that is there already, the
        streams a layer advanced in place, stays as it is), or the tensors themselves where the range has no room
        for them (no range here, a state wider than the residual planned)."""
        if chunk >= len(self.bases):
            self.kept[chunk] = tuple(state)
            return
        first, size = self.bases[chunk]
        region, at, found, moves = self.paged.tensor[first:first + size], 0, [], []
        low, high = region.data_ptr(), region.data_ptr() + size
        for tensor in state:
            if tensor is None:
                found.append(None)
                continue
            nbytes = tensor.numel() * tensor.element_size()
            if at + nbytes > size:   # wider than planned: this state stays in torch's allocator
                self.kept[chunk] = tuple(state)
                return
            view = region[at:at + nbytes].view(tensor.dtype).view(tensor.shape)
            if tensor.data_ptr() != view.data_ptr():
                # A tensor already in the range elsewhere (the state's tensors moved: one grew, or the residual
                # keeps a tensor of an earlier layer's state) is read before any is written: no write overlaps it.
                inside = low <= tensor.data_ptr() < high
                moves.append((view, tensor.clone() if inside else tensor))
            found.append(view)
            at = -(-(at + nbytes) // ALIGN) * ALIGN
        for view, tensor in moves:
            view.copy_(tensor)
        self.kept[chunk] = tuple(found)

    def get(self, chunk):
        return self.kept[chunk]

    def drop(self, chunk):
        self.kept.pop(chunk, None)

    def close(self):
        """The pass is over (or dropped): its states go, and the range's memory is the device's again (its givers
        take it back when the prompt's last step ends: the experts plugin's step frame)."""
        self.kept, self.bases = {}, []
        paged, self.paged = self.paged, None
        if paged is not None:
            paged.close()


class Chunk:
    """One chunk of a pass: its first position, its ids, its windows of ids and what the blocks of its forward
    compute once (Context.shared)."""
    __slots__ = ("index", "first", "count", "ids", "windows", "shared")

    def __init__(self, index, first, count):
        self.index, self.first, self.count = index, first, count
        self.ids = self.windows = None
        self.shared = {}


class Pass:
    """A span of a sequence's prompt, its chunks run layer by layer.

        made = Pass(model, sequence, tokens, first, counts)   # tokens: the sequence's ids from position 0
        done = made.run(deadline, finish)                    # some chunks through some layers; True once all are

    counts: the chunks' token counts from position `first` on; the pass holds as many of them, from the first, as the
    model's pass budget and the device's memory keep states for (`tokens`: what it holds). finish(chunk, h, feature)
    is called for each chunk when its last layer ran, in chunk order: the head's rows and the draft feature, as the
    residual's forward returns them. A pass that is dropped before its end (a cancel) has advanced its sequence's
    pools past the position it started at: close() it and close the sequence."""

    def __init__(self, model, sequence, tokens, first, counts):
        m = self.m = model
        self.sequence, self.first = sequence, int(first)
        self.states = States(m, counts, m.pass_bytes)
        self.chunks, at = [], self.first
        for count in list(counts)[:self.states.fit]:
            self.chunks.append(Chunk(len(self.chunks), at, int(count)))
            at += int(count)
        self.tokens = at - self.first
        self.layer = self.chunk = 0        # the next layer and chunk to run
        self.steps = self.units = 0
        self.unit_seconds = 0.0            # the longest a chunk took through a layer so far
        self.slowest = (0.0, 0, 0)         # that time with its layer and chunk, and the longest run() took
        self.step_seconds = 0.0
        self.reported = False              # a step told Core of the pass: its tokens may count past a step's grant
        self.ended = None                  # the last chunk's last head row and draft feature, once its layers ran
        torch = m.torch
        for chunk in self.chunks:
            ids = tokens[chunk.first:chunk.first + chunk.count]
            chunk.ids = torch.tensor(ids, dtype=torch.int64, device=m.device)
            if m.window_tokens:
                chunk.windows = m._windows([(tokens[max(0, chunk.first - m.window_tokens):chunk.first], ids)])

    @property
    def done(self):
        return self.layer >= len(self.m.decoder)

    def _context(self, chunk):
        """The chunk's context for the layer under way: a batch of its own, as a step's is (a pool plans a batch
        once: its readers at a reserve, its states' rows at the batch's first use), the slots of the pools that
        layer's blocks write, and what the chunk's blocks share."""
        m = self.m
        batch = StepBatch.build([(self.sequence, chunk.first, chunk.count)], m.device)
        ctx = Context(m, batch, m.rows.of(batch), m._reserve(batch, m.layer_pools[self.layer]),
                      ids=chunk.ids, windows=chunk.windows)
        ctx.shared = chunk.shared
        return ctx

    def run(self, deadline, finish):
        """Chunks through layers from where the pass stands, until every chunk ran through every layer (True) or the
        next chunk would end past `deadline` (time.perf_counter(); None: no limit), a chunk through a layer at
        least. A layer's state pools accept each chunk as it ran: the next chunk continues them."""
        m, torch = self.m, self.m.torch
        layers, states = m.decoder, self.states
        cuda = m.device.type == "cuda"
        self.steps += 1
        ran, entered = 0, time.perf_counter()
        while self.layer < len(layers):
            last = self.layer + 1 == len(layers)
            while self.chunk < len(self.chunks):
                started = time.perf_counter()
                if deadline is not None and ran and started + self.unit_seconds > deadline:
                    self.step_seconds = max(self.step_seconds, started - entered)
                    return False
                chunk = self.chunks[self.chunk]
                ctx = self._context(chunk)
                state = (states.get(chunk.index) if self.layer
                         else residual.begin(m, layers, m.embedded(chunk.ids), ctx))
                state = residual.advance(m, layers, self.layer, state, ctx)
                for pool in m.layer_states[self.layer]:
                    pool.accept(self.sequence, chunk.count)
                if last:
                    h, feature = residual.end(m, layers, state, ctx)
                    states.drop(chunk.index)
                    finish(chunk, h, feature)
                    del h, feature
                else:
                    states.put(chunk.index, state)
                del state, ctx
                if cuda and deadline is not None:   # the unit's device work is done: its time is the step's
                    torch.cuda.current_stream(m.device).synchronize()
                took = time.perf_counter() - started
                if took > self.unit_seconds:
                    self.unit_seconds, self.slowest = took, (round(took, 4), self.layer, self.chunk)
                self.chunk += 1
                self.units += 1
                ran += 1
            self.layer, self.chunk = self.layer + 1, 0
        states.close()
        self.step_seconds = max(self.step_seconds, time.perf_counter() - entered)
        return True

    def close(self):
        self.states.close()
