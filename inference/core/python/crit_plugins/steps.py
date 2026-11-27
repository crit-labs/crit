"""A prompt step's budget: the rows a prompt chunk may take so that a step ends within the run's bound, learned from
the steps the engine runs, not fixed by a fit at load.

Core cancels a request between steps, and crit serve retires an engine whose cancel does not settle within its grace
(GRACE: one second): every prompt step must end well inside it, at every depth the engine serves. A model plugin that
chunks its prompts asks the engine's budget for a chunk's rows instead of sizing chunks itself:

    budget = context.steps.budget(chunk, prior=costs)   # in the model's factory: Core's LoadContext holds the budgets
    rows = budget.rows(position)        # the rows a prompt chunk that starts at `position` may take
    budget.ran(rows, position)          # in step(): this step ran such a chunk (each prompt chunk of the step)

Core's step engine times every step that ran prompt rows and tells the budgets (Steps.observe). The model's
calibration at load (`prior`: {c, a, b}, three timed steps on zeroed KV) gives the form

    seconds = c + a * rows + b * rows * (position + rows)

and the budget corrects it by depth: per BUCKET positions it keeps the ratio of what the engine's steps there took to
what the form predicted (slower steps taken at once, faster ones averaged in, FOLLOW of each), and a chunk is sized by
the form times its depth's ratio (the nearest shallower depth's where none ran yet). So a step at a depth, a cache
state or a load the calibration never saw is measured as it runs, the next chunk at that depth is sized by what such
steps took, and a deep step that ran past the bound shrinks the deep chunks, never the shallow ones.

Where a prompt chunk's boundaries change its rows' bits (the engine asks prompt rows below row-exact: a GEMM whose
order follows the row count, a recurrence's two prompt paths), the spans must not depend on what ran before:
`adaptive` False keeps the prior's spans, and the budget still counts the steps that ran past the bound (facts'
`over_bound`), which the serving checks read. A model whose chunk runs over several of Core's steps (`units`: a pass
through its layers, a step a layer) says how many units a step ran (ran(..., units=)).

Each chunk's span is a multiple of QUANTUM rows (few distinct shapes), at most the model's prefill chunk, and within
`room` device bytes at `per_row` bytes a row where those are given.
"""
# The step contract a model plugin states in its info ("step_contract"): 3, Core decides the engine's lanes, its
# waiting places, the prompt spans' bound, the activation reserve planned before Core measures its step arena and
# when cached device memory goes back (Steps; docs/PLUGINS.md#step-settings); the model plugin declares what it can
# take in its manifest (`steps`: the most lanes and waiting places, Steps.capable), and a configuration's option
# bounds them only where it names one (CORE, the options' default, leaves each to Core). 2: the model's options sent
# Core's values as bounds of their own; 1 (none stated): the model chose them itself, and Core still applies its
# runtime's cap over its info.
CONTRACT = 3
CORE = -1              # a model option's value that leaves the setting to Core (the options' default, contract 3)
LANES = 4              # the requests stepped at once where the model plugin bounds them no lower
PENDING = 2            # the requests that wait for a lane beside them, where the model bounds them no lower
ACTIVATIONS = 768 << 20   # the device bytes planned for a step's activations before Core measures its step arena
QUANTUM = 16           # spans are multiples of this many rows: few distinct shapes, the attention reader's tiles
GRACE = 1.0            # the cancel grace in seconds, which crit serve's cancellation uses: a prompt step ends inside it
SECONDS = 0.75 * GRACE   # the default bound of a prompt step: a quarter of the grace for what the fit does not see
BUCKET = 8192          # the depth a ratio holds for: positions [k * BUCKET, (k + 1) * BUCKET)
FOLLOW = 0.5           # a faster step's share of its depth's ratio: half the latest step, half what was learned

class StepBudget:
    """The rows of a prompt chunk by the position it starts at (the module's description)."""

    def __init__(self, chunk, prior=None, seconds=SECONDS, room=None, per_row=0.0, deepest=0, units=1,
                 adaptive=True):
        self.chunk, self.seconds, self.room = int(chunk), float(seconds), room
        self.per_row, self.deepest, self.units = float(per_row or 0.0), int(deepest), max(1, int(units))
        self.adaptive = bool(adaptive)
        self.prior = None if prior is None else {k: float(prior[k]) for k in ("c", "a", "b")}
        self.costs = None if prior is None else dict(self.prior)
        self.ratios = {}       # {depth bucket: what steps there took over what the form predicted}
        self._step = []        # the chunks the step in flight ran: (rows, position, units)
        self.observed = 0      # the steps that ran prompt chunks and were timed
        self.over_bound = 0    # of them, those that took longer than the bound
        self.slowest = 0.0     # the most seconds one took
        self._rows = {}

    # -- the spans ------------------------------------------------------------------------------------------------------
    def predicted(self, rows, position):
        """The seconds a step that runs a chunk of `rows` rows at `position` takes, a unit of it: the form times what
        the engine's steps at that depth took over it (ratio)."""
        return self._form(rows, position) * self.ratio(position) / self.units

    def _form(self, rows, position):
        c = self.costs
        return c["c"] + c["a"] * rows + c["b"] * rows * (position + rows)

    def ratio(self, position):
        """What the engine's steps at `position`'s depth took over what the form predicted: the depth's own, else the
        nearest shallower depth's that ran, else 1."""
        bucket = int(position) // BUCKET
        if bucket in self.ratios:
            return self.ratios[bucket]
        shallower = [b for b in self.ratios if b < bucket]
        return self.ratios[max(shallower)] if shallower else 1.0

    def rows(self, position):
        """The span at `position`: the prefill chunk without a fit; else the longest span (the chunk less a multiple
        of QUANTUM, QUANTUM at least) within the room whose predicted time is within the bound. Where none is, the
        span that prefills fastest by the fit, among those predicted to end inside the cancel grace where any does."""
        if self.costs is None:
            return self.chunk
        key = position // QUANTUM
        found = self._rows.get(key)
        if found is None:
            found = self._rows[key] = self._span(position)
        return found

    def _span(self, position):
        spans = list(range(self.chunk, QUANTUM, -QUANTUM)) + [min(self.chunk, QUANTUM)]   # longest first
        roomy = [r for r in spans if self.room is None or self.per_row * r <= self.room] or spans[-1:]
        within = [r for r in roomy if self.predicted(r, position) <= self.seconds]
        if within:
            return within[0]
        if len(roomy) == 1:
            return roomy[0]
        graceful = [r for r in roomy if self.predicted(r, position) < GRACE] or roomy
        return max(graceful, key=lambda r: (r / max(self.predicted(r, position), 1e-12), r))

    def stop(self, position):
        """The position a prompt chunk starting at `position` does not pass (crit_core.batch.step_item's stop)."""
        return position + self.rows(position)

    # -- the loop: what the steps took ----------------------------------------------------------------------------------
    def ran(self, rows, position, units=None):
        """The model ran a prompt chunk of `rows` rows at `position` in the step in flight (`units`: of its units,
        all by default)."""
        if int(rows) > 0:
            self._step.append((int(rows), int(position), self.units if units is None else max(1, int(units))))

    def observe(self, seconds):
        """Core: the step in flight took `seconds`. Where the budget adapts, the ratio of its deepest chunk's depth
        moves toward what the step took over what the form predicted for its chunks, and the spans are sized again;
        a step past the bound is counted either way."""
        chunks, self._step = self._step, []
        if not chunks:
            return
        self.observed += 1
        self.slowest = max(self.slowest, float(seconds))
        if seconds > self.seconds:
            self.over_bound += 1
        if self.costs is None or not self.adaptive:
            return
        units = max(u for _, _, u in chunks)
        whole = float(seconds) * self.units / units   # the step's chunks through every unit
        c = self.costs
        form = c["c"] + sum(c["a"] * r + c["b"] * r * (p + r) for r, p, _ in chunks)
        if form <= 0:
            return
        bucket = max(p for _, p, _ in chunks) // BUCKET
        known = self.ratios.get(bucket)
        found = whole / form
        # A step slower than its depth's ratio said sets the ratio: an average remembering the faster steps would size
        # the next chunk past the bound again. A faster one is averaged in, so one quick step does not grow them.
        self.ratios[bucket] = found if known is None or found > known else known + FOLLOW * (found - known)
        self._rows.clear()

    def facts(self):
        """The spans for the engine's facts: the load's calibration (costs) and the ratio per depth the steps taught,
        the steps timed and those past the bound, the slowest, and the span and its predicted seconds at position 0
        and at the deepest position."""
        if self.costs is None:
            return {"rows": self.chunk, "calibrated": False, "observed": self.observed, "over_bound": self.over_bound}
        found = {"rows": self.chunk, "calibrated": True, "seconds": self.seconds, "costs": self.costs,
                 "adaptive": self.adaptive, "observed": self.observed,
                 "ratios": {b * BUCKET: round(r, 3) for b, r in sorted(self.ratios.items())},
                 "over_bound": self.over_bound, "slowest_seconds": round(self.slowest, 4),
                 **({"room_bytes": int(self.room)} if self.room is not None else {}),
                 **({"row_bytes": int(self.per_row)} if self.per_row else {}),
                 **({} if self.units == 1 else {"units": self.units})}
        steps = {position: (self.rows(position), self.predicted(self.rows(position), position))
                 for position in sorted({0, self.deepest})}
        found["steps"] = [{"position": p, "rows": r, "seconds": float(f"{s:.4g}"),
                           "tokens_per_second": float(f"{r / max(s * self.units, 1e-12):.4g}")}
                          for p, (r, s) in steps.items()]
        if any(s > self.seconds for _, s in steps.values()):
            found["over_budget"] = (f"a prompt step is predicted to take {max(s for _, s in steps.values()):.2f} s, "
                                    f"{self.costs['c']:.2f} s of it whatever its span: no span meets the bound "
                                    f"{self.seconds:g} s, so spans are the ones that prefill fastest")
        return found


def fit(timings, peaks):
    """The cost model of three timed prompt steps {(rows, position): seconds} (rows r1 < r2 at position 0, r2 at
    position p > 0) and their peaks {rows: bytes}: {c, a, b, m} (StepBudget's prior: seconds = c + a * rows + b *
    rows * (position + rows); m the most bytes a row took), each rounded to four significant figures so a recorded
    decision reads back as it was written."""
    (r1, _), (r2, _), (_, p) = sorted(timings, key=lambda k: (k[1], k[0]))
    t1, t2, t3 = timings[(r1, 0)], timings[(r2, 0)], timings[(r2, p)]
    b = max(0.0, (t3 - t2) / (r2 * p))
    a = max(0.0, (t2 - t1) / (r2 - r1) - b * (r1 + r2))
    c = max(0.0, t1 - a * r1 - b * r1 * r1)
    m = max(v / r for r, v in peaks.items()) if peaks else 0.0
    return {k: float(f"{v:.4g}") for k, v in {"c": c, "a": a, "b": b, "m": m}.items()}


class Steps:
    """An engine's step plan (LoadContext.steps): what Core decides of the engine's steps, which its model plugin
    follows (CONTRACT; docs/PLUGINS.md#step-settings), and its prompt-step budgets, which the model's factory makes and
    Core's step engine tells each prompt step's time.

    The model plugin states its limits (the most lanes it takes, what its steps' activations need) and asks: lanes()
    and pending() are the requests it sizes its states and captured rounds for, under the caller's cap on what Core's
    runtime holds (max_concurrent_requests: crit serve's --max-concurrency); seconds() bounds a prompt step;
    calibrated() times its prompt steps through Core (a decision recorded like any other) and gives the budget;
    activations() is what it plans for activations before Core measures the step arena; release() hands back
    torch's cached memory where free memory is below what the next step needs (threshold())."""

    def __init__(self, max_concurrent_requests=None, release_below=None, memory=None):
        self.budgets = []
        self.most = {}   # the model plugin's manifest's `steps`: {"lanes": n, "pending": n}, the most it takes
        self.cap = None if max_concurrent_requests is None else int(max_concurrent_requests)
        # The configuration's release_below_mib in bytes (0: never release), None: Core's threshold (threshold()).
        self.release_below = None if release_below is None else int(release_below)
        self.memory = memory   # the device's memory (crit_plugins.memory), where the engine has one

    def capable(self, steps=None):
        """What the model plugin declares it can take (its manifest's `steps`): {"lanes": the most requests it steps
        at once, "pending": the most that wait}; lanes() and pending() stay within it."""
        self.most = {key: int(value) for key, value in (steps or {}).items()}
        return self

    def lanes(self, bound=None):
        """The requests the engine steps at once, which the model sizes what each lane holds for (its states, a
        captured round's rows): LANES, or the configuration's bound where it names one (None or CORE: Core's), at
        most what the model plugin declares (capable) and at most the caller's cap on the requests Core holds (but 2:
        every engine batches; Core's runtime applies the cap itself)."""
        lanes = LANES if bound is None or int(bound) == CORE else max(1, int(bound))
        if "lanes" in self.most:
            lanes = min(lanes, max(1, self.most["lanes"]))
        return lanes if self.cap is None else min(lanes, max(2, self.cap))

    def pending(self, bound=None, lanes=None):
        """The requests that wait for a lane in Core's runtime beside the lanes: PENDING, or the configuration's
        bound (None or CORE: Core's), at most what the model plugin declares, and within the caller's cap less the
        lanes."""
        pending = PENDING if bound is None or int(bound) == CORE else max(0, int(bound))
        if "pending" in self.most:
            pending = min(pending, max(0, self.most["pending"]))
        if self.cap is not None:
            pending = min(pending, max(0, self.cap - (self.lanes() if lanes is None else int(lanes))))
        return pending

    def seconds(self, bound=None):
        """The bound a prompt step ends inside: SECONDS (a share of the cancel grace), the model's bound where it is
        tighter; 0, None or CORE: SECONDS."""
        return SECONDS if not bound or float(bound) <= 0 else min(float(bound), SECONDS)

    def activations(self, need=None):
        """The device bytes the model plans for its steps' activations before Core measures them: its own need where
        it states one, else ACTIVATIONS. Once the warm requests ran, Core keeps its measured step arena free
        (crit_plugins.memory's size_arena) and the device's givers take the rest, whatever was planned."""
        return ACTIVATIONS if need is None or int(need) < 0 else int(need)

    def threshold(self):
        """The free device memory below which a prompt's or a request's end hands back what torch's allocator caches
        and nothing uses: the configuration's release_below_mib where it names one (0: never), else what the next
        step needs free, the device's headroom and Core's step arena; None where Core has no device memory."""
        if self.release_below is not None:
            return self.release_below or None
        if self.memory is None:
            return None
        return int(self.memory.headroom) + int(getattr(self.memory, "arena", 0) or 0)

    def release(self, device="cuda"):
        """Hand back torch's cached memory nothing uses where the device has less free than threshold()."""
        below = self.threshold()
        if below:
            from .device import release_cached_under
            return release_cached_under(below, device)
        return None

    def calibrated(self, plugin, chunk, *, key, measure, seconds=None, room=None, deepest=0, units=1,
                   adaptive=True):
        """The prompt-step budget of a model plugin's chunks, calibrated by Core: measure() runs its three timed
        prompt steps ({"timings": [[rows, position, seconds], ...], "peaks": [[rows, bytes], ...]}, a short and a full
        chunk at position 0 and a full chunk at `deepest`), settled as the plugin's decision "spans" (recorded like
        any other, crit_plugins.decisions: key names what the steps depend on), fitted (fit) and the budget's prior,
        its bound seconds(seconds), its rows a row's measured bytes within `room`."""
        from . import decisions
        costs = decisions.settle(plugin, "spans", key=key, measure=measure,
                                 decide=lambda found: fit({(r, p): s for r, p, s in found["timings"]},
                                                          {r: b for r, b in found["peaks"]}))
        return self.budget(chunk, prior=costs, seconds=self.seconds(seconds), room=room, per_row=costs["m"],
                           deepest=deepest, units=units, adaptive=adaptive)

    def budget(self, chunk, **options):
        made = StepBudget(chunk, **options)
        self.budgets.append(made)
        return made

    def observe(self, seconds):
        for budget in self.budgets:
            budget.observe(seconds)

    def largest(self):
        """The most rows a prompt step of the engine runs: its budgets' longest chunk (0 without a budget). Core's
        loader serves a prompt and a score that long, so the step arena holds what the largest steps take."""
        return max((int(b.chunk) for b in self.budgets), default=0)
