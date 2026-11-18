"""What an engine's device buffers really hold: the memory audit behind `crit plugins memory`.

The census (crit_plugins.memory) says who holds each byte of a device. The audit says what the engine does with it:
which buffers it never writes, which it writes a small part of, what its captured graphs keep for their replays and
what each step allocates. It needs nothing from the engine: it reads the device.

    audit = Audit("cuda")            # before the engine loads: the device records which code makes each allocation
    engine = crit_core.engine.load(package, model)
    audit.attach(engine)
    audit.read("load")               # a reading: every page's fingerprint ...
    audit.mark()                     # ... and then a pattern into the bytes Core knows to be scratch,
    ...a request...                  # both as this request ends
    ...requests...                   # a workload, the caller's
    audit.read("first")
    ...the workload's last request...
    ...other requests...
    audit.read("second")
    ...their last request...
    report = audit.report()
    audit.close()

With an engine attached, a reading and the mark are taken by the engine's own thread as its next request ends,
before the model retires it: the request's sequence is still as it left it (its state, its KV pages, the scratch
its last step wrote), where a moment later the model closes it and its pages go back to the device. They are not
taken from the caller's thread: generate() returns while the engine's thread still settles the request, and a page
read while it is unmapped is an illegal access. So read() comes before the request whose end it reads. Without an
engine (a test of buffers of its own) a reading is taken at once.

A reading fingerprints every page of the device memory the process holds: torch's allocator (each block is a
tensor's storage, or free) in pages of PAGE bytes, the unit its blocks are multiples of, and every backed granule of
the elastic ranges in pages of RANGE_PAGE. A page's fingerprint is its 64-bit words' weighted sum (odd weights, so
any one changed word changes it) and whether it is all zero. Nothing is written and nothing but the audit's own
buffer is allocated, so the engine's state stays as it was, and so do its allocator's counters.

What the readings say of a buffer, in bytes of whole pages:

    written    the pages whose fingerprint changed between two readings: the workload wrote them
    reach      where the last written page ends: writes confined to a buffer's start show a workspace sized past its use
    zero       the pages all zero at every reading and never changed
    constant   the rest: pages that hold something and never changed

and the verdicts drawn from them:

    dead           every byte zero at every reading (marked scratch: never written since the mark). Certain: the
                   workload stored nothing there, and whatever read it read zeros.
    oversized      a quarter of it or more (and OVERSIZED bytes) zero at every reading, beside bytes in use. Certain
                   for the part: the workload never reached it.
    partly written a quarter or more constant beside written pages. Not certain: room never used, or state set
                   while the engine loaded.
    written        the workload wrote most of it: scratch or state in use.
    constant       nothing changed. Not certain: weights or a table, or a buffer only the load wrote.

Two readings apart tell written from constant; a second workload of other prompts tells scratch from a buffer that
one prompt set happened to fill with the same values. So the caller reads after each of two different prompt sets.

Where Core knows bytes to be scratch it does better than compare: mark() fills them with a pattern, and what still
holds the pattern at the last reading was never written since. Those are the free blocks of captured graphs' own
pools (a capture's temporaries: its replays write there, and no replay may read one before writing it) and the
growing buffers (crit_plugins.memory.Growing: a workspace of every call shape). There the verdicts are certain.

A graph's pool is reported by the name its driver gave it (rounds/..., prompts/...), with the allocations its
captures made there by the code that made them (the device records them from the audit's start), so a pool's bytes
show as the gather, the product or the workspace they are. A paged range reports, per plane, the units backed and
the units the workload wrote. Each step the engine ran is counted by kind (a prompt step by its token count, an
eager decode step, Core's native rounds): the graphs it replayed, its allocations, the most it held above its start
(its transient peak) and what it left allocated.

The verdicts hold for the workload that ran. The caller reports it with them.
"""
import re
from bisect import bisect_right
from collections import Counter, defaultdict

from .memory import of as memory_of, where

PAGE = 512               # bytes: torch's allocator makes blocks in multiples of it, so a block's pages are its own
RANGE_PAGE = 4096        # an elastic range's page (its granules are megabytes)
CHUNK = 32 << 20         # bytes fingerprinted at a time: the audit's own buffer
MARK = 0x5A5A5A5A5A5A5A5A
OVERSIZED = 1 << 20      # the least unwritten part worth the verdict
SHARE = 4                # ... and at least a 1/SHARE of the buffer
ALLOCATIONS = 12         # a pool's largest capture allocations listed
INDEX = re.compile(r"\[[^\]]*\]")


class _Words:
    def __init__(self, at, words):
        self.__cuda_array_interface__ = {"shape": (int(words),), "typestr": "<i8", "data": (int(at), False),
                                         "version": 3}


class Reader:
    """Device memory by address: pages' fingerprints, and the mark. Everything it computes goes into one buffer of
    CHUNK bytes made here (an elastic range where the device has them, so torch's allocator is not asked)."""

    def __init__(self, memory):
        import torch
        self.torch, self.memory, self.device = torch, memory, torch.device(memory.device)
        self.range = memory.elastic("audit", "pages", CHUNK, initial=CHUNK) if memory.elastic_ranges else None
        if self.range is not None and self.range.prefix < CHUNK:
            self.range.close()
            raise MemoryError("the memory audit needs %d MiB of device memory for itself" % (CHUNK >> 20))
        store = self.range.tensor if self.range is not None else torch.empty(CHUNK, dtype=torch.uint8,
                                                                               device=self.device)
        self.words = store.view(torch.int64)
        generator = torch.Generator().manual_seed(0x61756469)
        self.weights, self.marked = {}, {}
        for page in (PAGE, RANGE_PAGE):
            weights = torch.randint(-(1 << 62), 1 << 62, (page // 8,), generator=generator, dtype=torch.int64) | 1
            self.weights[page] = weights.to(self.device)
            self.marked[page] = int((weights * _signed(MARK)).sum())   # a marked page's fingerprint
        self.out = torch.empty(3, CHUNK // PAGE, dtype=torch.int64, device=self.device)

    @property
    def bytes(self):
        return CHUNK + self.out.numel() * 8

    def _view(self, at, words):
        return self.torch.as_tensor(_Words(at, words), device=self.device)

    def pages(self, first, end, page):
        """(fingerprints int64, zero bool), host tensors, of the pages of [first, end), whole pages of mapped memory."""
        torch, words, weights = self.torch, page // 8, self.weights[page]
        prints, zero = [], []
        for at in range(first, end, CHUNK):
            rows = (min(end, at + CHUNK) - at) // page
            rows_view = self._view(at, rows * words).view(rows, words)
            product = self.words[:rows * words].view(rows, words)
            torch.mul(rows_view, weights, out=product)
            torch.sum(product, dim=1, out=self.out[0, :rows])
            torch.amax(rows_view, dim=1, out=self.out[1, :rows])
            torch.amin(rows_view, dim=1, out=self.out[2, :rows])
            host = self.out[:, :rows].cpu()
            prints.append(host[0].clone())
            zero.append((host[1] == 0) & (host[2] == 0))
        if not prints:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.bool)
        return torch.cat(prints), torch.cat(zero)

    def mark(self, first, end):
        for at in range(first, end, CHUNK):
            self._view(at, (min(end, at + CHUNK) - at) // 8).fill_(_signed(MARK))

    def close(self):
        if self.range is not None:
            self.range.close()
        self.range = self.words = self.out = None


def _signed(value):
    return value - (1 << 64) if value >= 1 << 63 else value


class Reading:
    """One reading: the pages of every span of mapped memory, torch's blocks, and the elastic ranges' backed runs."""

    def __init__(self, label, spans, blocks, ranges, names=None, memory=None):
        self.label, self.names, self.memory = label, names or {}, memory
        self.spans = sorted(spans, key=lambda span: span[0])   # (first, end, page, fingerprints, zero)
        self.firsts = [span[0] for span in self.spans]
        self.blocks = blocks                                   # (first, end, state, pool, frames), by address
        self.active = {first: end for first, end, state, _, _ in blocks if state == "active_allocated"}
        self.ranges = ranges                                   # key -> [(first, end)] backed runs

    def pages(self, first, end):
        """(page, fingerprints, zero, first page's address) of the pages holding [first, end), or None where the
        reading did not hold all of them."""
        i = bisect_right(self.firsts, first) - 1
        if i < 0 or end > self.spans[i][1]:
            return None
        start, _, page, prints, zero = self.spans[i]
        a, b = (first - start) // page, -(-(end - start) // page)
        return page, prints[a:b], zero[a:b], start + a * page


class Audit:
    """The memory audit of one device (the module's description). Made before the engine loads; `record` False
    leaves the allocator's history off (pools and blocks are then not named by code)."""

    def __init__(self, device="cuda", record=True):
        self.memory = memory_of(device)
        if self.memory.total is None:
            raise ValueError(f"the memory audit reads a CUDA device's memory, and {device} is none")
        import torch
        self.torch = torch
        self.reader = Reader(self.memory)
        if record:
            self.memory.record()
        self.readings, self.marks, self.marked_at = [], [], None
        self.engine, self.plugins = None, {}
        self.steps, self._replays, self._replay = [], [0], None
        self._due = []   # with an engine: ("read", label) and ("mark", None) for its thread, as a request ends

    # -- the engine -------------------------------------------------------------------------------------------------
    def attach(self, engine):
        """Audit `engine` (crit_core.engine.load's): its plugin objects name the buffers, and every step it runs
        from now on is counted."""
        torch = self.torch
        self.engine, self.plugins = engine, dict(getattr(engine, "plugins", None) or {})
        self._replay = torch.cuda.CUDAGraph.replay
        replay, count = self._replay, self._replays

        def counted(graph):
            count[0] += 1
            return replay(graph)
        torch.cuda.CUDAGraph.replay = counted
        engine.audit = self

    def retire(self):
        """The engine's thread, as a request ends: the readings and the mark that wait for it."""
        while self._due:
            kind, label = self._due.pop(0)
            self._read(label) if kind == "read" else self._mark()

    def step(self, rows):
        """The engine's thread, before a step: what to call after it, which counts the step."""
        torch, index = self.torch, self.reader.device.index or 0
        prompt = sum(row.prefill_tokens for row in rows)
        before, replays = torch.cuda.memory_stats(index), self._replays[0]
        torch.cuda.reset_peak_memory_stats(index)

        def done():
            after = torch.cuda.memory_stats(index)
            held = before.get("allocated_bytes.all.current", 0)
            self.steps.append((prompt, len(rows), self._replays[0] - replays,
                               after.get("allocation.all.allocated", 0) - before.get("allocation.all.allocated", 0),
                               after.get("allocated_bytes.all.peak", 0) - held,
                               after.get("allocated_bytes.all.current", 0) - held))
        return done

    def close(self):
        if self._replay is not None:
            self.torch.cuda.CUDAGraph.replay, self._replay = self._replay, None
        if self.engine is not None and getattr(self.engine, "audit", None) is self:
            self.engine.audit = None
        if self.memory.recorded is not None:
            self.memory.record(0)
        self.engine, self.plugins, self.readings = None, {}, []
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    # -- readings ---------------------------------------------------------------------------------------------------
    def _runs(self, elastic):
        """An elastic range's granules backed on the device as runs of addresses [(first, end)]. Granules backed
        from host memory (a paged range's host pages) are not the device's bytes: they are counted, never read."""
        runs = []
        for granule in sorted(set(elastic.handles) - set(getattr(elastic, "host", ()))):
            first = elastic.base + granule * elastic.gran
            if runs and runs[-1][1] == first:
                runs[-1][1] = first + elastic.gran
            else:
                runs.append([first, first + elastic.gran])
        return [tuple(run) for run in runs]

    def _ranges(self):
        return {key: made for key, made in self.memory.ranges.items() if made is not self.reader.range and made.base}

    def read(self, label):
        """Take a reading and keep it under `label`: at once, or with an engine attached as its next request ends
        (the module's description)."""
        if self.engine is not None:
            self._due.append(("read", label))
        else:
            self._read(label)

    def mark(self):
        """Fill the bytes Core knows to be scratch with the mark, as of the reading before: the free blocks of
        captured graphs' own pools and the growing buffers. At once, or with an engine attached as its next request
        ends, after the readings asked for before it."""
        if self.engine is not None:
            self._due.append(("mark", None))
        else:
            self._mark()

    def _read(self, label):
        self.torch.cuda.synchronize(self.reader.device)
        blocks = self.memory.blocks()
        spans, at = [], None
        for first, end, _, _, _ in blocks:        # torch's segments: blocks that touch are one span of mapped memory
            if at is not None and at[1] == first:
                at[1] = end
            else:
                at = [first, end]
                spans.append(at)
        spans = [(first, end, PAGE) + self.reader.pages(first, end, PAGE) for first, end in spans]
        ranges = {}
        for key, made in self._ranges().items():
            ranges[key] = self._runs(made)
            spans += [(first, end, RANGE_PAGE) + self.reader.pages(first, end, RANGE_PAGE) for first, end in ranges[key]]
        names = {}   # what holds each storage and the device as it stands, read on this thread too
        for plugin, name, tensor in self.memory.holdings(self.plugins):
            storage = tensor.untyped_storage()
            names.setdefault(storage.data_ptr(), (plugin, name, list(tensor.shape),
                                                  str(tensor.dtype).removeprefix("torch."), storage.nbytes()))
        self.readings.append(Reading(label, spans, blocks, ranges, names, self.memory.report()))
        return self.readings[-1]

    def _mark(self):
        if not self.readings:
            raise ValueError("mark() follows a reading")
        last = self.readings[-1]
        marks = [(first, end) for first, end, state, pool, _ in last.blocks if pool != (0, 0) and state == "inactive"]
        for key, made in self._ranges().items():
            if made.workspace:
                marks += [(first, end) for first, end in last.ranges.get(key, ())]
        for first, end in marks:
            self.reader.mark(first, end)
        self.torch.cuda.synchronize(self.reader.device)
        self.marks, self.marked_at = sorted(marks), len(self.readings)
        return sum(end - first for first, end in marks)

    def _marks(self, first, end):
        """The marked spans that overlap [first, end)."""
        i = max(0, bisect_right(self.marks, (first, float("inf"))) - 1)
        while i < len(self.marks) and self.marks[i][0] < end:
            if self.marks[i][1] > first:
                yield self.marks[i]
            i += 1

    def judge(self, first, end, since=0):
        """What the readings from `since` on say of the bytes [first, end): {"bytes", "written", "reach", "zero",
        "constant", "verdict", "certain"}, or None where they do not all hold them or are too few to compare. A
        marked page is judged by the mark alone, from the first reading after it: written once it no longer holds
        the mark, and counted as zero while it does."""
        views = [reading.pages(first, end) for reading in self.readings[since:]]
        if not views or any(view is None for view in views):
            return None
        torch, page, at = self.torch, views[0][0], views[0][3]
        marked = torch.zeros(len(views[0][1]), dtype=torch.bool)
        for a, b in self._marks(first, end) if self.marked_at is not None else ():
            marked[(max(a, at) - at) // page:-(-(min(b, end) - at) // page)] = True
        after = views[max(0, (self.marked_at or 0) - since):]
        if (len(views) < 2 and not bool(marked.all())) or (not after and bool(marked.any())):
            return None
        changed, zero, unmarked = torch.zeros_like(marked), torch.ones_like(marked), torch.zeros_like(marked)
        for before, later in zip(views, views[1:]):
            changed |= before[1] != later[1]
        for view in views:
            zero &= view[2]
        for view in after:
            unmarked |= view[1] != self.reader.marked[page]
        written = torch.where(marked, unmarked, changed)
        zero = torch.where(marked, ~unmarked, zero & ~changed)
        nbytes = end - first
        count = lambda pages: min(nbytes, int(pages.sum()) * page)
        last = written.nonzero()
        found = {"bytes": nbytes, "written": count(written), "zero": count(zero),
                 "reach": min(nbytes, at + (int(last[-1]) + 1) * page - first) if len(last) else 0}
        found["constant"] = max(0, nbytes - found["written"] - found["zero"])
        found["verdict"], found["certain"] = _verdict(found)
        return found

    # -- the report -------------------------------------------------------------------------------------------------
    def report(self, least=1 << 20):
        """The audit so far (the module's description), JSON-ready:

        {"device", "memory": the device at the last reading (crit_plugins.memory's report()), "readings": their
         labels,
         "audit_bytes": what the audit itself holds, "marked_bytes",
         "buffers": every buffer of torch's allocator of `least` bytes or more, the largest unused part first:
             {"plugin" (None: no plugin object holds it), "name" (where it is held, or the code that made it),
              "shape", "dtype", "pool" (a graph's pool it was made in), "since" (the first reading that held it),
              "bytes", "written", "reach", "zero", "constant", "verdict", "certain"}; one first held at the last
              reading has no verdict ("new"),
         "plugins": {plugin: {"buffers", "bytes", "written", "zero", "constant"}} over all of them, small ones too,
         "pools": each captured graph's pool {"name", "bytes", "kept" (allocated: tensors a plugin keeps),
             "temporaries" (free: what replays write), "written" (of the temporaries, since the mark),
             "allocations": the largest its captures made [{"bytes", "count", "where"}]},
         "elastic": each range {"name", "reserved", "backed" (at the last reading), "new" (first backed at it:
             not judged), "written", "zero", "constant", "verdict", "certain"}, a paged one with "planes":
             [{"unit_bytes", "units", "backed", "written"}] (units),
         "steps": [{"kind", "steps", "replays", "allocations", "most_allocations", "peak", "kept"}] (per step:
             mean graph replays and allocations, the most allocations, the largest transient peak, mean bytes left
             allocated),
         "totals": {"dead", "oversized" (the unwritten parts), "new"} in bytes}"""
        if self._due:
            raise ValueError("a reading waits for a request to end: read() comes before the request it reads")
        if len(self.readings) < 2:
            raise ValueError("a report compares at least two readings")
        last, labels = self.readings[-1], [reading.label for reading in self.readings]
        names = last.names
        buffers, plugins, pools, totals = [], defaultdict(Counter), {}, Counter()
        for first, end, state, pool, frames in last.blocks:
            if pool != (0, 0):
                held = pools.setdefault(pool, {"name": self.memory.pools.get(pool, "unnamed"), "bytes": 0, "kept": 0,
                                               "temporaries": 0, "written": 0})
                held["bytes"] += end - first
                held["kept" if state == "active_allocated" else "temporaries"] += end - first
                if state != "active_allocated":
                    found = self.judge(first, end)
                    held["written"] += found["written"] if found else 0
                    continue
            if state != "active_allocated":
                continue
            plugin, name, shape, dtype, nbytes = names.get(first, (None, where(frames) or "made by no recorded code",
                                                                   None, None, end - first))
            nbytes = min(nbytes, end - first)
            since = next(k for k in range(len(self.readings))
                         if all(reading.active.get(first) == end for reading in self.readings[k:]))
            found = self.judge(first, first + nbytes, since) or {"bytes": nbytes, "verdict": "new", "certain": True}
            found.update(plugin=plugin, name=name, shape=shape, dtype=dtype, since=labels[since],
                         pool=self.memory.pools.get(pool, "unnamed") if pool != (0, 0) else None)
            count = plugins[plugin or ""]
            count["buffers"] += 1
            for key in ("bytes", "written", "zero", "constant"):
                count[key] += found.get(key, 0)
            totals[found["verdict"]] += found["zero"] if found["verdict"] == "oversized" else nbytes
            if nbytes >= least:
                buffers.append(found)
        buffers.sort(key=lambda found: (-found.get("zero", 0), -found["bytes"]))
        for pool, held in pools.items():
            made = (self.memory.recorded or {}).get(pool) or {}
            made = sorted(((key, count) for key, count in made.items() if key != "read"), key=lambda item: -item[0][0])
            held["allocations"] = [{"bytes": nbytes, "count": count, "where": code}
                                   for (nbytes, code), count in made[:ALLOCATIONS]]
        return {"device": self.memory.device, "memory": last.memory, "readings": labels,
                "audit_bytes": self.reader.bytes, "marked_bytes": sum(end - first for first, end in self.marks),
                "buffers": buffers, "plugins": {plugin: dict(count) for plugin, count in sorted(plugins.items())},
                "pools": sorted(pools.values(), key=lambda held: -held["bytes"]),
                "elastic": [self._elastic(key, made) for key, made in sorted(self._ranges().items())],
                "steps": self._steps(),
                "totals": {key: int(totals[key]) for key in ("dead", "oversized", "new")}}

    def _since(self, key, made):
        """{granule's address: the first reading from which every reading holds it} for the granules of an elastic
        range that the last reading holds: a range's backing comes and goes (a cache shrinks for a prompt and grows
        back), and a granule is judged over the readings that hold it without a gap."""
        held = [{at for first, end in reading.ranges.get(key, ()) for at in range(first, end, made.gran)}
                for reading in self.readings]
        since = {}
        for at in held[-1]:
            k = len(held) - 1
            while k > 0 and at in held[k - 1]:
                k -= 1
            since[at] = k
        return since

    def _elastic(self, key, made):
        since, found = self._since(key, made), Counter()
        pieces = []                                   # runs of granules judged over the same readings
        for at in sorted(since):
            if pieces and pieces[-1][1] == at and pieces[-1][2] == since[at]:
                pieces[-1][1] = at + made.gran
            else:
                pieces.append([at, at + made.gran, since[at]])
        for first, end, k in pieces:
            judged = self.judge(first, end, k)
            if judged is None:                        # held by the last reading alone
                found["new"] += end - first
            for name in ("bytes", "written", "zero", "constant") if judged else ():
                found[name] += judged[name]
        row = {"name": key, "reserved": made.size, "backed": len(since) * made.gran, "new": found["new"],
               "written": found["written"], "zero": found["zero"], "constant": found["constant"]}
        if getattr(made, "host_mapped", 0):
            row["host"] = made.host_mapped
        row["verdict"], row["certain"] = _verdict({**row, "bytes": found["bytes"]}) if found["bytes"] else ("new", True)
        if made.planes:
            row["planes"] = [self._plane(made, since, *plane) for plane in made.planes]
        return row

    def _plane(self, made, since, first, per, units):
        """A plane's units: how many the last reading holds and how many the workload wrote (a unit's own pages:
        one a neighbour shares counts for neither, unless the unit has no page of its own)."""
        backed = written = 0
        for unit in range(units if per else 0):
            a, b = made.base + first + unit * per, made.base + first + (unit + 1) * per
            granules = range(a // made.gran * made.gran, b, made.gran)
            if any(at not in since for at in granules):
                continue
            backed += 1
            inner = (-(-a // RANGE_PAGE) * RANGE_PAGE, b // RANGE_PAGE * RANGE_PAGE)
            judged = self.judge(*(inner if inner[0] < inner[1] else (a, b)), max(since[at] for at in granules))
            written += bool(judged and judged["written"])
        return {"unit_bytes": per, "units": units, "backed": backed, "written": written}

    def _steps(self):
        kinds = defaultdict(list)
        for prompt, rows, replays, allocations, peak, kept in self.steps:
            if prompt:
                top = 1 << max(4, (prompt - 1).bit_length())
                kind = f"prompt step, {top // 2 + 1 if top > 16 else 1} to {top} tokens"
            else:
                kind = "decode step (the model's step)"
            kinds[kind].append((replays, allocations, peak, kept))
        found = []
        for kind, rows in sorted(kinds.items(), key=lambda item: (item[0].startswith("decode"), len(item[0]), item[0])):
            n = len(rows)
            found.append({"kind": kind, "steps": n, "replays": round(sum(r[0] for r in rows) / n, 2),
                          "allocations": round(sum(r[1] for r in rows) / n), "most_allocations": max(r[1] for r in rows),
                          "peak": max(r[2] for r in rows), "kept": round(sum(r[3] for r in rows) / n)})
        native = getattr(self.engine, "rounds", None)
        if native is not None:
            found.append({"kind": "native round (no Python)", "steps": int(native.runs[0]), "replays": 1.0,
                          "allocations": 0, "most_allocations": 0, "peak": 0, "kept": 0})
        return found


def _verdict(found):
    """(verdict, certain) of a judged buffer (the module's description)."""
    nbytes, written, zero, constant = found["bytes"], found["written"], found["zero"], found["constant"]
    if not written and not constant:
        return "dead", True
    if zero >= max(OVERSIZED, nbytes // SHARE):
        return "oversized", True
    if written and constant >= max(OVERSIZED, nbytes // SHARE):
        return "partly written", False
    if written:
        return "written", True
    return "constant", False


def pattern(name):
    """A buffer's name without its indices: buffers held alike (a layer's matrix each) share it."""
    return INDEX.sub("[]", name or "")


__all__ = ["Audit", "PAGE", "RANGE_PAGE", "Reader", "Reading", "pattern"]
