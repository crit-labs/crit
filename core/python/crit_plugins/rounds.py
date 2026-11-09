"""Captured decode rounds across plugins: Core's round driver, which a model plugin runs its decode and verify rows
through so that Core launches them natively, with no Python between rounds (crit_core.native.NativeRounds).

A round decodes `rows` rows (round slots 0 .. rows - 1, the decoding sequences packed at the front) at `width` drafts:
each row's pending token and its drafts are verified in width + 1 columns at positions frontier .. frontier + width, the
drafts the target agrees with are kept with the correction or bonus token after them, each row's state keeps what it
emitted, and one block of results reaches pinned host memory. Every tensor a round touches is allocated once, so a
round of (rows, width, mode) is one CUDA graph, captured at load.

Two drivers share that round:

- SpeculativeRounds runs it over three hooks an engine supplies (draft, verify, commit), for code that composes its own
  round (device-speculative-round's python_round adds rejection sampling and copied drafts for it).
- Rounds composes the library's device forms into it, each named by the fact its plugin provides
  (crit_plugins.device.DEVICE_EXTENSIONS):

    start    start_round(view) on every part that has it (below)
    draft    drafter.draft(view, head, sampler)                    draft:device (every width, 0 included: catch-up)
    verify   model.verify(view, batch) -> logits [rows][width + 1][vocabulary] FP32 over a RoundBatch
    predict  sampler.predictions_device(view, logits, draft_laws)  rounds:sampler (greedy argmax or rejection)
    accept   verifier.accept_device(view)                          verify:device; else the chain accept below
             (writes n_emitted, n_matched, emitted, and a tree's view.tree_path)
    score    in the log-probability form (Forms, below), each emitted token's in emitted order:
             sampler.logprobs_device(view, logits, tokens) over the logits rows that predicted them (a tree's gathered
             by view.tree_path), else the driver's own
    advance  the driver's own state from view.n_emitted and view.emitted (pending, frontier, allowance, out_count,
             catch-up)
    commit   model.commit_round(view), each pool's commit_round(view), drafter.commit_round(view),
             head.commit_round(view)
    count    count_round(view) on every part that has it: each adds its per-row counts into view.counters
    pack     the result block, rows of [n_emitted, n_matched, frontier, tokens (widest + 1), accepted, the engine's
             counters], to pinned host memory

A round's parts are the model, the sampler, the drafter, the head, the verifier, every pool and the extra parts the
model registers (extra_parts: a device expert cache). Core's native rounds run no Python between rounds, so whatever
bookkeeping a part does each round runs inside the captured round, in its two optional capture-safe hooks:
start_round(view) at the round's start (before the draft) and count_round(view) after the commits (before the pack).

Rounds refuses to capture (RoundsRefused, naming the plugin) when a plugin a round calls lacks its fact: every slot the
model's verify calls (uses, by default every computing slot of the selection) provides rounds:<slot>, the sampler
rounds:sampler, the drafter draft:device; or when such a plugin, configured as it is, says it cannot run inside a
captured round (its optional round_problems(): an expert cache over a file tier, a table read through Python). The
model then decodes through step(), which is slower but just as correct.

Pools follow the round slots (bind, unbind, move, seed) and keep each row's accepted prefix (commit_round, idempotent
from the round's start, which start_round(view) may snapshot: a truncation commits a second time, over a view that
carries what the round's plugins set on its own, a tree's view.tree_path among them). A row a step
does not schedule sits the round out (view.hold): it emits nothing, and every commit_round keeps a row with
n_emitted == 0 as it was. A model computes inputs that depend on the tokens before each column (hashed n-gram ids)
inside verify from a TokenRing among its pools.

Widths: a driver captures rounds at several widths (widths=, each in 0 .. widest) and a width policy (width_policy=)
chooses each round's width from the rounds before it, inside the captured round: it writes the next round's width per
row count into the driver's width table, which each round's pack copies to host memory with its block, so Core's
native rounds read it before the next launch (crit_core.native.NativeRounds' widths) and a model's own launches ask
width(rows). FixedWidth (the default) never changes it; StepWidth steps one captured width up after a round that kept
every draft and one down after a round that kept fewer; RateWidth picks the width with the most tokens per unit of
round time from each draft depth's acceptance rate and the rounds' measured times (calibrate()). The width changes
which drafts are verified, never what a row emits: greedy rounds emit the target's own greedy tokens at any width, and
a sampled round's rejection sampling keeps the target's law at any width chosen from the rounds before it (each
position's draws are keyed by the position, and no draw a choice read is read again).

Rows of a greedy or sampled round may follow a continuation of their own (start_row's follow, with a driver sized for
continuations): "replay" (the continuation is the row's prediction: its drafts are accepted while they equal it, a
request replaying a fixed output, as Eval's replay) or "forced" (the continuation is its drafts too, so the round keeps
width + 1 of its tokens: scoring a fixed text). Such a row rides the same captured graph as the others; a forced row's
acceptance is by construction, so it counts in neither the acceptance by depth nor the width policy.

Modes: "greedy" and "sampled" (the sampler predicts), "replay" (a fixed continuation is the prediction: Eval's replay,
which accepts a draft while it equals the continuation) and "forced" (the continuation is the draft too: scoring a fixed
text width + 1 positions per target pass).

Forms: a driver built with log-probabilities (logprobs=) or continuations (continuation=) captures each (rows, width,
mode) round twice, sharing one graph memory pool: the log-probability form scores every emitted token (the score step
above, into view.logprobs and the alternatives) and runs the rows that follow a continuation of their own; the plain form
does neither (its view's logprobs, top_ids and top_logprobs are None) and costs less. A round runs its log-probability
form when one of its rows (held rows aside) asks for log-probabilities (start_row's or bind's logprobs) or follows a
continuation, else its plain form; Core's native rounds choose the same way from their requests (a request asking for
log-probabilities, a replay). Both forms emit the same tokens for the same rows, and the telemetry counts the rounds of
each form; the device memory both captures take is measured (captured_bytes, by form in captured_bytes_by_form).
"""
import gc
import inspect

from .device import RoundBatch
from .manifest import POLICY_SLOTS, PluginError

MODES = ("greedy", "sampled", "replay", "forced")
FOLLOW = {None: 0, "replay": 1, "forced": 2}   # a row's own continuation inside a greedy or sampled round
FORMS = ("plain", "logprobs")                  # a round's forms, by its key's last entry (False, True)
# Each row of the result block: tokens emitted, drafts matched, the frontier after the round, then the emitted tokens
# (widest + 1 entries, -1 past the count), then the drafts it accepted and the engine's own counters of the round
# (counters=, named in the model's info as round_counters: crit_core.native.NativeRounds sums them into each request's
# facts under those names).
HEAD = ("n_emitted", "n_matched", "frontier")

_CAPTURE_STREAMS = {}   # device -> the one stream every round driver of the process captures on


def capture_stream(device):
    """The stream round drivers capture their rounds on, one per device for the process: a stream's first matrix
    multiply gives it cuBLAS workspaces of its own, which a new stream per capture would hold once per captured round."""
    import torch
    device = torch.device(device)
    if device not in _CAPTURE_STREAMS:
        _CAPTURE_STREAMS[device] = torch.cuda.Stream(device)
    return _CAPTURE_STREAMS[device]


class RoundsRefused(PluginError):
    """A selection whose plugins cannot run inside a captured round: the model decodes through step()."""


def round_fact(slot):
    """The fact a slot's plugin provides when its device form runs inside a captured round."""
    return {"drafter": "draft:device", "verifier": "verify:device"}.get(slot, "rounds:" + slot)


def round_problems(parts, uses=None, drafts=True):
    """Why a round over the plugins of `parts` (a factory's parts, crit_plugins.assemble) cannot be captured, or []:
    each slot of `uses` (the slots the model's verify calls; by default every filled computing slot) provides
    rounds:<slot>, the sampler rounds:sampler, and when the round drafts the drafter draft:device; and each of those
    plugins that has round_problems() (no arguments: the reasons its configuration keeps it out of a captured round, []
    when none) gives none. A tree drafter needs its verifier's accept_device (verify:device) and the attention
    plugin's ancestor masks (attention:tree); a chain needs neither, Core's chain accept stands in."""
    filled = [slot for slot in parts.slots() if parts.plugin(slot)]
    if uses is None:
        uses = [slot for slot in filled if slot != "model" and slot not in POLICY_SLOTS]
    wanted = list(dict.fromkeys(list(uses) + ["sampler"] + (["drafter"] if drafts and parts.plugin("drafter") else [])))
    problems = []
    for slot in wanted:
        plugin = parts.plugin(slot)
        if plugin is None:
            problems.append(f"no plugin fills {slot}, which the round calls")
        elif round_fact(slot) not in parts.facts(slot):
            problems.append(f"{plugin} in {slot} does not provide {round_fact(slot)}: its device form is not known to "
                            "run inside a captured round")
        else:
            problems += [f"{plugin} in {slot} cannot run inside a captured round as configured: {reason}"
                         for reason in _own_problems(parts, slot, plugin)]
    if drafts and parts.plugin("drafter") and "draft:tree" in parts.facts("drafter") \
            and "verify:device" not in parts.facts("verifier"):
        problems.append(f"{parts.plugin('drafter')} drafts trees, and {parts.plugin('verifier') or 'no verifier'} "
                        "provides no verify:device to accept them on the device")
    if drafts and parts.plugin("drafter") and "draft:tree" in parts.facts("drafter") \
            and "attention:tree" not in parts.facts("attention"):
        problems.append(f"{parts.plugin('drafter')} drafts trees, and {parts.plugin('attention') or 'no attention'} "
                        "provides no attention:tree to verify them under ancestor masks")
    return problems


def _own_problems(parts, slot, plugin):
    """The reasons the plugin filling `slot` gives (its optional round_problems()), asked of the plugin itself, past the
    loader's counting proxy."""
    from .observe import unwrap
    found = parts.get(slot) if callable(getattr(parts, "get", None)) else None
    ask = _method(unwrap(found) if found is not None else None, "round_problems")
    if ask is None:
        return []
    reasons = ask()
    if not isinstance(reasons, (list, tuple)) or not all(isinstance(r, str) and r for r in reasons):
        raise PluginError(f"{plugin}'s round_problems() returns a list of reasons, not {reasons!r}")
    return list(reasons)


def _arguments(function):
    """How many positional arguments a callable takes (None: any number), read past the loader's counting proxy."""
    from .observe import unwrap
    try:
        parameters = inspect.signature(unwrap(function)).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_POSITIONAL for p in parameters):
        return None
    return len([p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])


def _method(part, name):
    found = getattr(part, name, None) if part is not None else None
    return found if callable(found) else None


class RoundView:
    """One round's tensors, the driver's buffers cut to its rows and width, which the hooks and plugins read and
    write. `batch` (Rounds: the round's RoundBatch) is set before verify. A plain-form round's view (logprobs False:
    the module's docstring, Forms) has no log-probability buffers: logprobs, top_ids and top_logprobs are None."""

    def __init__(self, rounds, rows, width, mode, logprobs=True):
        self.rows, self.width, self.mode = rows, width, mode
        self.pending = rounds.pending[:rows]                # each row's input at column 0, its last emitted token
        self.frontier = rounds.frontier[:rows]              # the position of column 0
        self.allowance = rounds.allowance[:rows]            # output tokens the row may still emit (0: inactive)
        self.out_count = rounds.out_count[:rows]            # output tokens so far: column c predicts output out_count + c
        self.drafts = rounds.drafts[:rows]                  # [rows, widest]: the drafter writes [:, :width]
        laws = rounds.draft_laws                            # sampled rounds: the law each draft was drawn from
        self.draft_laws = laws[:rows] if getattr(laws, "dim", None) else laws
        self.predictions = rounds.predictions[:rows]        # [rows, widest + 1]: written [:, :width + 1]
        scored = logprobs and rounds.logprobs is not None
        self.logprobs = rounds.logprobs[:rows] if scored else None
        self.top_ids = rounds.top_ids[:rows] if scored and rounds.top_ids is not None else None   # [rows, cols, k]
        self.top_logprobs = rounds.top_logprobs[:rows] if scored and rounds.top_logprobs is not None else None
        self.catch_up_tokens = rounds.catch_up[:rows]       # [rows, widest + 1]: the previous round's tokens, 0 past
        self.catch_up_count = rounds.catch_up_count[:rows]  # how many of them it emitted
        self.n_emitted = rounds.n_emitted[:rows]            # this round's count, set by the accept
        self.n_matched = rounds.n_matched[:rows]            # drafts accepted, set by the accept
        self.emitted = rounds.emitted[:rows]                # this round's tokens, -1 past n_emitted, set by the accept
        self.stops = rounds.stops[:rows]                    # each row's stop token ids, -1 padded
        self.hold = rounds.hold[:rows]                      # 1: the row sits this round out (emits nothing)
        self.follow = rounds.follow[:rows]                  # FOLLOW: 1 the row's text is its prediction, 2 its drafts too
        self.counters = rounds.counters[:rows]              # [rows, counters]: zero at the round's start
        self.counter_names = rounds.counter_names
        self.slots = rounds.slot_index[:rows]               # == arange(rows): each row's round slot
        self.batch = None
        self.tree_path = None   # a tree verifier's accepted nodes ([rows, width + 1]): emitted j is node path[j]'s

    def counter(self, name):
        """The [rows] column of view.counters that counts `name` (one of the engine's round counters), to add into."""
        return self.counters[:, self.counter_names.index(name)]

    def columns(self):
        """The verify columns' input tokens: the pending token, then the drafts ([rows, width + 1])."""
        import torch
        return torch.cat([self.pending[:, None], self.drafts[:, :self.width]], dim=1)


class FixedWidth:
    """Every round at the width it is launched at: Core's native rounds at the widths native_rounds() was given, a
    model's own launches at its own width (width(rows) gives `width`, by default the widest of the driver's widths).
    The default width policy."""
    name, adaptive = "fixed", False

    def __init__(self, width=None):
        self.width = width

    def start(self, widths):
        if self.width is not None and self.width not in widths:
            raise ValueError(f"the fixed width {self.width} is none of the round widths {list(widths)}")
        return widths[-1] if self.width is None else int(self.width)

    def reserve(self, rounds):
        pass

    def state(self):
        return {}

    def observe(self, rounds, view, judged):
        pass


class StepWidth:
    """One captured width up after a round whose judged rows (decoding rows whose allowance held the whole round) kept
    at least `up` of their drafts, one down after a round that kept less than `down` of them, else the same width,
    from the width the round ran at; one width for every row count, `start` (by default the narrowest) at first. The
    defaults (up 1, down 1) step up after a round that kept every draft and down after one that kept fewer; a batch of
    rows judges its drafts together (`down` below `up` leaves a band where the width stays)."""
    name, adaptive = "step", True

    def __init__(self, start=None, up=1.0, down=1.0):
        if not 0.0 <= down <= up <= 1.0:
            raise ValueError("0 <= down <= up <= 1: the kept fractions a round steps down below and up at")
        self.first, self.up, self.down = start, float(up), float(down)

    def start(self, widths):
        if self.first is not None and self.first not in widths:
            raise ValueError(f"the start width {self.first} is none of the round widths {list(widths)}")
        return widths[0] if self.first is None else int(self.first)

    def reserve(self, rounds):
        self.widths = rounds.round_widths
        self.values = rounds.torch.tensor(self.widths, dtype=rounds.torch.int32, device=rounds.device)

    def state(self):
        return {}

    def observe(self, rounds, view, judged):
        torch, width, rows = rounds.torch, view.width, view.rows
        if width not in self.widths:
            return   # a round at a width this policy never chooses: the next width stays as it was
        here = self.widths.index(width)
        count = judged.sum()
        kept = (rounds.n_matched[:rows].clamp(max=width) * judged).sum().double()
        offered = (count * width).double()
        up = kept >= self.up * offered
        down = (kept < self.down * offered) & ~up
        index = (up.long() - down.long() + here).clamp(0, len(self.widths) - 1)
        chosen = self.values.index_select(0, index.view(1))
        table = rounds.width_table
        table.copy_(torch.where((table >= 0) & (count > 0), chosen.expand_as(table), table))


class RateWidth:
    """For each row count, the captured width with the most tokens per unit of round time. Per draft depth d, c_d is
    the rate at which a draft there is accepted when the drafts before it were: decayed counts (`decay` a round) over
    the judged rows of every round (decoding rows whose allowance held the whole round), smoothed toward the depth
    before it with `weight` rows' worth of counts (c_1 toward `prior`). A row then emits 1 + sum over d <= w of
    c_1 ... c_d tokens a round at width w, and a round of r rows at width w costs costs[r - 1][w's index]: the captured
    rounds' measured times (Rounds.calibrate() calls set_costs), else 1 + column_cost * r * (w + 1). Ties go to the
    narrower width; `start` (by default the narrowest) at first. The default decay follows the text closely (a few
    rounds' memory): on a toy whose drafts are good and poor in stretches of 48 positions it beat 0.9 and 0.98, which
    settle on the width best on average."""
    name, adaptive = "rate", True

    def __init__(self, start=None, decay=0.7, weight=4.0, prior=0.5, column_cost=0.05):
        if not 0.0 < decay <= 1.0 or weight <= 0 or not 0.0 <= prior <= 1.0 or column_cost < 0:
            raise ValueError("decay in (0, 1], a positive weight, a prior in [0, 1] and a non-negative column cost")
        self.first, self.decay, self.weight, self.prior, self.column_cost = start, decay, weight, prior, column_cost

    def start(self, widths):
        if self.first is not None and self.first not in widths:
            raise ValueError(f"the start width {self.first} is none of the round widths {list(widths)}")
        return widths[0] if self.first is None else int(self.first)

    def reserve(self, rounds):
        torch = self.torch = rounds.torch
        double = dict(dtype=torch.float64, device=rounds.device)
        self.depths = max(rounds.widest, 1)
        self.reached = torch.zeros(self.depths, **double)
        self.accepted = torch.zeros(self.depths, **double)
        self.prior_rate = torch.full((), self.prior, **double)   # on the device: a capture copies nothing from the host
        self.values = torch.tensor(rounds.round_widths, dtype=torch.int32, device=rounds.device)
        self.index = self.values.long()
        rows = torch.arange(1, rounds.rows + 1, **double)
        self.costs = 1.0 + self.column_cost * rows[:, None] * (self.values.double()[None, :] + 1.0)
        self.depth = torch.arange(1, self.depths + 1, dtype=torch.long, device=rounds.device)

    def set_costs(self, costs):
        """costs [rows][widths]: each captured round's time (any unit; inf where a round of that row count and width
        is never run)."""
        costs = self.costs.new_tensor(costs)
        if tuple(costs.shape) != tuple(self.costs.shape) or not bool((costs > 0).all()):
            raise ValueError(f"a positive cost per row count and width, {list(self.costs.shape)}")
        self.costs.copy_(costs)

    def state(self):
        return {"reached": self.reached, "accepted": self.accepted}

    def rates(self):
        """c_d per depth ([depths] FP64): the smoothed acceptance rates the choice reads."""
        found, before = [], self.prior_rate
        for d in range(self.depths):
            before = (self.accepted[d] + self.weight * before) / (self.reached[d] + self.weight)
            found.append(before)
        return self.torch.stack(found)

    def observe(self, rounds, view, judged):
        torch, width, rows = rounds.torch, view.width, view.rows
        count = judged.sum()
        self.reached.mul_(torch.where(count > 0, self.decay, 1.0))
        self.accepted.mul_(torch.where(count > 0, self.decay, 1.0))
        if width:
            matched = rounds.n_matched[:rows].clamp(max=width)[:, None]
            depth = self.depth[None, :width]
            rows_judged = judged[:, None] > 0
            self.reached[:width] += ((matched >= depth - 1) & rows_judged).sum(0).double()
            self.accepted[:width] += ((matched >= depth) & rows_judged).sum(0).double()
        expected = torch.cat([self.reached.new_ones(1), 1.0 + torch.cumsum(torch.cumprod(self.rates(), 0), 0)])
        score = expected.index_select(0, self.index)[None, :] / self.costs
        chosen = self.values.index_select(0, torch.argmax(score, dim=1))   # the first best: the narrower on a tie
        table = rounds.width_table
        table.copy_(torch.where((table >= 0) & (count > 0), chosen, table))


WIDTH_POLICIES = {"fixed": FixedWidth, "step": StepWidth, "rate": RateWidth}


def width_policy(name, **options):
    """The width policy `name` (fixed, step or rate) with its options: what a model plugin's option selects."""
    if name not in WIDTH_POLICIES:
        raise ValueError(f"width policies are {', '.join(WIDTH_POLICIES)}, not {name!r}")
    return WIDTH_POLICIES[name](**options)


class SpeculativeRounds:
    """Device state and rounds for up to `rows` rows with drafts up to `widest` wide on `device`, over `hooks`
    (draft(view), verify(view), commit(view), optionally reset_capture(), start_round(view) and count_round(view));
    `stops` stop ids per row at most, `continuation` tokens per row for replay and forced rounds, `logprobs` to pack
    each emitted column's log-probability (True: the verify hook writes view.logprobs; a count k: k alternatives too,
    ranked as Core ranks them, the lower id first among equals) and `counters` the names of the engine's per-row round
    counters (its info's round_counters). share_pool: the captured rounds share one CUDA graph memory pool (less memory; only for plugins
    that keep no tensor made inside one capture for another), else each (rows, width, mode) round has its own, which
    its two forms share (the module's docstring, Forms: a driver with logprobs or continuation captures both). widths:
    the widths rounds run at (each captured where rounds are captured; by default widest alone) and width_policy the
    policy choosing each round's (FixedWidth by default; the module's docstring)."""

    def __init__(self, hooks, rows, widest, device, stops=4, continuation=0, logprobs=False, share_pool=False,
                 counters=(), widths=None, width_policy=None):
        import torch
        if rows < 1 or widest < 0 or stops < 0 or continuation < 0:
            raise ValueError("at least one row, and non-negative widths, stop and continuation counts")
        found = sorted({int(w) for w in (widths if widths is not None else (widest,))})
        if not found or found[0] < 0 or found[-1] > widest:
            raise ValueError(f"round widths lie in 0 .. widest ({widest}), not {list(widths)}")
        self.round_widths = tuple(found)
        self.torch = torch
        self.hooks, self.rows, self.widest, self.device = hooks, rows, widest, torch.device(device)
        cols = self.cols = widest + 1
        long = dict(dtype=torch.long, device=self.device)
        self.pending, self.frontier, self.allowance, self.out_count = (torch.zeros(rows, **long) for _ in range(4))
        self.n_emitted, self.n_matched, self.catch_up_count, self.hold = (torch.zeros(rows, **long) for _ in range(4))
        self.counter_names = tuple(counters)
        if len(set(self.counter_names)) != len(self.counter_names) or not all(
                isinstance(name, str) and name for name in self.counter_names):
            raise ValueError("round counters are distinct names")
        self.counters = torch.zeros(rows, max(len(self.counter_names), 1), **long)
        self.slot_index = torch.arange(rows, **long)
        # The round's start, which a truncation goes back to.
        self.start = {name: torch.zeros(rows, **long) for name in ("pending", "frontier", "allowance", "out_count")}
        self.drafts = torch.zeros(rows, max(widest, 1), **long)
        self.predictions = torch.zeros(rows, cols, **long)
        self.emitted = torch.full((rows, cols), -1, **long)
        self.catch_up = torch.zeros(rows, cols, **long)
        self.stops = torch.full((rows, max(stops, 1)), -1, **long)
        self.continuation = torch.zeros(rows, max(continuation, 1), **long)
        self.following = continuation > 0   # rows may follow a continuation of their own (start_row's follow)
        self.follow = torch.zeros(rows, **long)
        # The forms rounds are captured in (the module's docstring, Forms): the plain one, and the log-probability one
        # when the driver scores tokens or rows may follow a continuation. Each slot's need of the latter (its request
        # asks for log-probabilities, or it follows a continuation), which a round of the model's own reads.
        self.forms = (False, True) if logprobs or self.following else (False,)
        self.scoring = [False] * rows
        self.top = 0 if isinstance(logprobs, bool) else int(logprobs)
        self.logprobs = torch.zeros(rows, cols, dtype=torch.float32, device=self.device) if logprobs else None
        self.top_ids = torch.zeros(rows, cols, max(self.top, 1), **long) if self.top else None
        self.top_logprobs = torch.zeros(rows, cols, max(self.top, 1), dtype=torch.float32, device=self.device) \
            if self.top else None
        self.draft_laws = None
        self.columns = torch.arange(cols, **long)
        # Acceptance by draft depth over every row of every round (held and finished rows not counted), on the
        # device: row rounds, then per depth the rows drafted there and those whose draft there was accepted, then
        # per width the rounds run there (with a decoding row), then per form (plain, log-probabilities) the rounds
        # run in it (with a decoding row). Each round's pack copies them to host memory, which telemetry() reads.
        depths = max(widest, 1)
        self.draft_counts = torch.zeros(1 + 2 * depths + widest + 1 + len(FORMS), **long)
        self.row_rounds = self.draft_counts[:1]
        self.offered_by_depth = self.draft_counts[1:1 + depths]
        self.accepted_by_depth = self.draft_counts[1 + depths:1 + 2 * depths]
        self.rounds_by_width = self.draft_counts[1 + 2 * depths:1 + 2 * depths + widest + 1]
        self.rounds_by_form = self.draft_counts[1 + 2 * depths + widest + 1:]
        # The readback block: n_emitted, n_matched, frontier after the round, the emitted tokens, accepted, counters.
        self.words_at = len(HEAD) + cols
        self.block = torch.zeros(rows, self.words_at + 1 + len(self.counter_names), **long)
        pinned = self.device.type == "cuda"
        self.host_counts = torch.zeros(self.draft_counts.shape, dtype=torch.long, pin_memory=True) if pinned else None
        self.host = torch.zeros(rows, self.block.shape[1], dtype=torch.long, pin_memory=pinned)
        self.host_logprobs = (torch.zeros(rows, cols, dtype=torch.float32, pin_memory=pinned) if logprobs else None)
        self.host_top_ids = torch.zeros(rows, cols, max(self.top, 1), dtype=torch.long, pin_memory=pinned) \
            if self.top else None
        self.host_top_logprobs = torch.zeros(rows, cols, max(self.top, 1), dtype=torch.float32, pin_memory=pinned) \
            if self.top else None
        # The form of the last round packed (1: log-probabilities), which each round's pack copies to host memory with
        # its block from a constant of its own: adopt() reads it after a native round, whichever form Core launched.
        self.form_values = torch.arange(len(FORMS), **long)
        self.host_form = torch.zeros(1, dtype=torch.long, pin_memory=pinned)
        # Each row count's width for the next round (-1: not run in rounds), which the width policy rewrites inside
        # each round and the pack copies to host memory (Core's native rounds read it there before each launch).
        self.width_policy = width_policy if width_policy is not None else FixedWidth()
        start = self.width_policy.start(self.round_widths)
        self.width_table = torch.full((rows,), start, dtype=torch.int32, device=self.device)
        self.host_widths = torch.full((rows,), start, dtype=torch.int32, pin_memory=pinned)
        self.width_policy.reserve(self)
        # A round's key: (rows, width, mode, logprobs), logprobs its form (the module's docstring, Forms).
        self.graphs = {}
        self.share_pool, self._pool, self._pools = bool(share_pool), None, {}
        self.captured_bytes, self.captured_bytes_by_form = 0, dict.fromkeys(FORMS, 0)
        self.event = torch.cuda.Event() if pinned else None
        self.kept = {}         # key -> what the round's plugins set on its view (a truncation's view)
        self.launched = None   # the key of the round in flight
        self.last = None       # the key of the last round read or adopted, which a truncation applies to
        self._held, self._last_held = False, None   # a hold staged for the round in flight; the last round's
        self._observer = None   # Rounds: the loader's call counter, which counts each replay (graph_calls)
        self.graph_calls = {}   # key -> the slots its capture called

    # Rows -----------------------------------------------------------------------------------------------------------

    def start_row(self, slot, frontier, token, allowance, stops=(), continuation=(), outputs=1, follow=None,
                  logprobs=False):
        """Row `slot` decodes from here: `token` is its pending token (the last output token it has, the prefill's
        first one after a prompt) at position `frontier`, `outputs` the output tokens it has so far (token included:
        Core's draw position of the next one), `allowance` the tokens it may still emit. The drafter's first catch-up
        ingests the pending token (a one-column seed). `stops` are its stop token ids; `continuation`, for replay and
        forced rounds, the fixed text its output follows, from the first output token on; `follow` ("replay" or
        "forced", the module's docstring) has the row follow it inside greedy and sampled rounds too. `logprobs`: its
        request asks for log-probabilities, so a round of it runs the log-probability form (as does one of a row that
        follows its continuation)."""
        torch = self.torch
        stops, continuation = list(stops), list(continuation)
        if len(stops) > self.stops.shape[1] or len(continuation) > self.continuation.shape[1]:
            raise ValueError("more stop ids or continuation tokens than the driver was sized for")
        if follow not in FOLLOW:
            raise ValueError(f"a row follows its continuation as one of replay, forced, not {follow!r}")
        if follow is not None and not (self.following and continuation):
            raise ValueError("a row that follows a continuation needs one, in a driver sized for continuations")
        if logprobs and self.logprobs is None:
            raise ValueError("a row asking for log-probabilities needs a driver built with logprobs")
        if outputs < 1:
            raise ValueError("a decoding row has its pending token among its outputs")
        values = {"pending": token, "frontier": frontier, "allowance": allowance, "out_count": outputs,
                  "n_emitted": 1, "catch_up_count": 1, "n_matched": 0, "hold": 0}
        for name, value in values.items():
            getattr(self, name)[slot] = int(value)
        self.catch_up[slot].zero_()
        self.catch_up[slot, 0] = int(token)
        self.emitted[slot].fill_(-1)
        self.stops[slot].fill_(-1)
        if stops:
            self.stops[slot, :len(stops)] = torch.tensor(stops, dtype=torch.long)
        self.continuation[slot].zero_()
        if continuation:
            self.continuation[slot, :len(continuation)] = torch.tensor(continuation, dtype=torch.long)
        self.follow[slot] = FOLLOW[follow]
        self.scoring[slot] = bool(logprobs) or follow is not None

    def release(self, slot):
        """Row `slot` stops decoding: it emits nothing in later rounds until it starts again."""
        self.allowance[slot] = 0
        self.follow[slot] = 0
        self.scoring[slot] = False

    def move(self, source, target):
        """Row `source`'s state moves to slot `target` (to keep the decoding rows at the front)."""
        for tensor in (self.pending, self.frontier, self.allowance, self.out_count, self.n_emitted, self.n_matched,
                       self.catch_up_count, self.catch_up, self.emitted, self.stops, self.continuation, self.predictions,
                       self.drafts, self.hold, self.follow, *self.start.values(), *[t for t in (self.logprobs,) if t is not None]):
            tensor[target] = tensor[source]
        self.allowance[source] = 0
        self.scoring[target], self.scoring[source] = self.scoring[source], False

    def logprob_form(self, rows, hold=()):
        """Whether a round of the first `rows` rows runs its log-probability form: one of them not in `hold` asks for
        log-probabilities or follows a continuation (start_row), in a driver that has the form."""
        held = set(hold)
        return self.forms[-1] and any(self.scoring[row] for row in range(rows) if row not in held)

    # Rounds ---------------------------------------------------------------------------------------------------------

    def view(self, rows, width, mode, logprobs=True):
        """The round's view; logprobs False: its plain form's, without log-probability buffers."""
        return RoundView(self, rows, width, mode, logprobs)

    def _from_continuation(self, rows, out, count):
        """Each row's continuation from its output count on, `count` tokens, into `out` (clamped at its end)."""
        index = (self.out_count[:rows, None] + self.columns[None, :count]).clamp(max=self.continuation.shape[1] - 1)
        out.copy_(self.continuation[:rows].gather(1, index))

    def _follow(self, rows, out, count, level):
        """The rows that follow their continuation at `level` or past it (FOLLOW) take it into `out` (`count`
        columns): a greedy or sampled round's per-row replay or forced text (capture-safe)."""
        if not self.following or not count:
            return
        index = (self.out_count[:rows, None] + self.columns[None, :count]).clamp(max=self.continuation.shape[1] - 1)
        out.copy_(self.torch.where(self.follow[:rows, None] >= level, self.continuation[:rows].gather(1, index), out))

    def _round(self, rows, width, mode, logprobs):
        """One round of (rows, width, mode) in its log-probability form (logprobs True: the emitted tokens scored,
        rows following their continuations) or its plain form (neither)."""
        view = self.view(rows, width, mode, logprobs)
        fresh = dict(vars(view))
        for name, tensor in self.start.items():
            tensor[:rows].copy_(getattr(self, name)[:rows])
        self.counters[:rows].zero_()
        self._start(view)
        if mode == "forced":
            self._from_continuation(rows, view.drafts[:, :width], width)
        else:
            self._draft(view)
            if logprobs:   # forced rows: their text is the draft
                self._follow(rows, view.drafts[:, :width], width, FOLLOW["forced"])
        self._verify(view)
        if mode in ("replay", "forced"):
            self._from_continuation(rows, view.predictions[:, :width + 1], width + 1)
        elif logprobs:   # rows that follow their text: it is their prediction
            self._follow(rows, view.predictions[:, :width + 1], width + 1, FOLLOW["replay"])
        self._accept_view(view)
        if logprobs:
            self._scored(view)
        self._advance(rows, width)
        self._count(view, logprobs)
        self._choose(view)
        self._commit(view)
        self._counted(view)
        self._pack(rows, logprobs)
        # What the plugins set on the view (a tree verifier's tree_path, a tree drafter's parents): a truncation's
        # commit reads them again. Tensors a captured round made live in its graph's pool, rewritten by each replay.
        self.kept[(rows, width, mode, logprobs)] = {name: value for name, value in vars(view).items()
                                                    if name != "batch" and fresh.get(name, fresh) is not value}

    def _kept_view(self, rows, width, mode, logprobs):
        """The view of the last round of (rows, width, mode) in its form again, with what its plugins set on it."""
        view = self.view(rows, width, mode, logprobs)
        view.__dict__.update(self.kept.get((rows, width, mode, logprobs), {}))
        return view

    # The parts of a round a subclass composes otherwise: SpeculativeRounds calls its hooks.
    def _start(self, view):
        start = _method(self.hooks, "start_round")
        if start is not None:
            start(view)

    def _draft(self, view):
        self.hooks.draft(view)   # every width, 0 included: a stateful drafter catches up on every round

    def _verify(self, view):
        self.hooks.verify(view)

    def _scored(self, view):
        """Log-probabilities of the emitted tokens, in emitted order, after the accept (the hooks' verify writes them
        for its chain's columns, which are in emitted order)."""

    def _accept_view(self, view):
        self._accept(view.rows, view.width)

    def _commit(self, view):
        self.hooks.commit(view)

    def _counted(self, view):
        count = _method(self.hooks, "count_round")
        if count is not None:
            count(view)

    def _reset(self):
        reset = _method(self.hooks, "reset_capture")
        if reset is not None:
            reset()

    def _count(self, view, logprobs):
        """The round's rows in the acceptance counters (capture-safe): rows that decoded (not held, not finished), and
        the round in its width's and its form's counts when one did."""
        rows, width = view.rows, view.width
        active = ((self.hold[:rows] == 0) & (self.start["allowance"][:rows] > 0)).long()
        self.row_rounds += active.sum()
        decoded = (active.sum() > 0).long()
        self.rounds_by_width[width] += decoded
        self.rounds_by_form[int(bool(logprobs))] += decoded
        if width:
            drafted = active * (self.follow[:rows] < FOLLOW["forced"]).long()   # a forced row's drafts are its text
            kept = (self.n_matched[:rows, None] > self.columns[None, :width]).long() * drafted[:, None]
            self.accepted_by_depth[:width] += kept.sum(0)
            self.offered_by_depth[:width] += drafted.sum()

    def _choose(self, view):
        """The width policy's choice of the next round's widths (capture-safe), from this round's judged rows: those
        that decoded (not held, not finished) with allowance for every column, so a row's last tokens do not count as
        rejected drafts."""
        rows, width = view.rows, view.width
        judged = ((self.hold[:rows] == 0) & (self.start["allowance"][:rows] > width)
                  & (self.follow[:rows] < FOLLOW["forced"])).long()
        self.width_policy.observe(self, view, judged)

    def telemetry(self):
        """The rounds so far, for the engine's telemetry ({"drafts": ..., "rounds": ...}, docs/PLUGINS.md, Facts and
        telemetry): row rounds run, drafts made, and per draft depth the rows drafted there and those whose draft there
        was accepted; the rounds run in each form (plain, logprobs) and the device memory each form's captures took. It
        never waits on the device, so it may run on another thread while rounds run: on a GPU it reads the copy each
        round's pack leaves in host memory (as of the last round whose copy landed)."""
        counts = (self.host_counts if self.host_counts is not None else self.draft_counts).tolist()
        depths = max(self.widest, 1)
        offered = counts[1:1 + self.widest]
        accepted = counts[1 + depths:1 + depths + self.widest]
        by_width = counts[1 + 2 * depths:1 + 2 * depths + self.widest + 1]
        by_form = counts[1 + 2 * depths + self.widest + 1:]
        return {"drafts": {"rounds": counts[0], "drafted": sum(offered), "offered_by_depth": offered,
                           "accepted_by_depth": accepted, "width_policy": self.width_policy.name,
                           "widths": list(self.round_widths),
                           "rounds_by_width": {str(w): by_width[w] for w in range(len(by_width)) if by_width[w]}},
                "rounds": {"by_form": dict(zip(FORMS, by_form)), "forms": [FORMS[f] for f in self.forms],
                           "captured_bytes": dict(self.captured_bytes_by_form)}}

    def width(self, rows):
        """The width the policy chose for the next round of `rows` rows (as of the last round read; before any round
        its start width; -1 for a row count native_rounds() left to step())."""
        if not 1 <= rows <= self.rows:
            raise ValueError(f"1 .. {self.rows} rows")
        return int(self.host_widths[rows - 1])

    def width_facts(self):
        """The widths and policy for the engine's info (draft_widths, width_policy)."""
        return {"draft_widths": list(self.round_widths), "width_policy": self.width_policy.name}

    def _accept(self, rows, width):
        """Chain acceptance for every row on the device: the drafts that match the predictions, then the correction
        or bonus token, ending after the first predicted stop token and bounded by the allowance; a held row emits
        nothing. Writes n_emitted, n_matched and emitted, as a verifier's accept_device does."""
        torch = self.torch
        predictions = self.predictions[:rows, :width + 1]
        if width:
            matched = (self.drafts[:rows, :width] == predictions[:, :width]).long().cumprod(1).sum(1)
        else:
            matched = torch.zeros_like(self.n_emitted[:rows])
        stop = (predictions[:, :, None] == self.stops[:rows, None, :]).any(2)
        first_stop = torch.where(stop, self.columns[None, :width + 1], width + 1).min(1).values
        count = torch.minimum(torch.minimum(matched + 1, first_stop + 1), self.allowance[:rows]).clamp(min=0)
        count = torch.where(self.hold[:rows] > 0, torch.zeros_like(count), count)
        self.n_matched[:rows] = torch.minimum(matched, (count - 1).clamp(min=0))
        self.n_emitted[:rows] = count
        kept = self.columns[None, :width + 1] < count[:, None]
        self.emitted[:rows].fill_(-1)
        self.emitted[:rows, :width + 1] = torch.where(kept, predictions, -1)

    def _advance(self, rows, width):
        """Each row's state after the round from its accept (view.n_emitted and view.emitted), from the round's start:
        the pending token is its last emitted one. A held row keeps the catch-up it had, which the drafter ingests
        again next round (catch-up is by position, so ingesting it twice is the same)."""
        torch = self.torch
        count = self.n_emitted[:rows]
        held = self.hold[:rows] > 0
        last = self.emitted[:rows].gather(1, (count - 1).clamp(min=0)[:, None]).squeeze(1)
        self.pending[:rows] = torch.where(count > 0, last, self.start["pending"][:rows])
        self.frontier[:rows] = self.start["frontier"][:rows] + count
        self.allowance[:rows] = self.start["allowance"][:rows] - count
        self.out_count[:rows] = self.start["out_count"][:rows] + count
        self.catch_up[:rows] = torch.where(held[:, None], self.catch_up[:rows], self.emitted[:rows].clamp(min=0))
        self.catch_up_count[:rows] = torch.where(held, self.catch_up_count[:rows], count)

    def _pack(self, rows, logprobs):
        block = self.block[:rows]
        block[:, 0] = self.n_emitted[:rows]
        block[:, 1] = self.n_matched[:rows]
        block[:, 2] = self.frontier[:rows]
        block[:, len(HEAD):self.words_at] = self.emitted[:rows]
        block[:, self.words_at] = self.n_matched[:rows]
        if self.counter_names:
            block[:, self.words_at + 1:] = self.counters[:rows, :len(self.counter_names)]
        self.host[:rows].copy_(block, non_blocking=True)
        self.host_widths.copy_(self.width_table, non_blocking=True)
        if self.host_counts is not None:
            self.host_counts.copy_(self.draft_counts, non_blocking=True)
        self.host_form.copy_(self.form_values[int(bool(logprobs)):int(bool(logprobs)) + 1], non_blocking=True)
        if logprobs and self.logprobs is not None:
            self.host_logprobs[:rows].copy_(self.logprobs[:rows], non_blocking=True)
            if self.top:
                self.host_top_ids[:rows].copy_(self.top_ids[:rows], non_blocking=True)
                self.host_top_logprobs[:rows].copy_(self.top_logprobs[:rows], non_blocking=True)

    def capture(self, rows, width, mode):
        """Capture the round of (rows, width, mode) as a CUDA graph in each of the driver's forms (the module's
        docstring, Forms; two warm-ups first, every row held, so a bound sequence's state stays as it was), the
        driver's buffers put back around each and the hooks' own by their reset_capture(). Each captured round gets
        its own memory pool, which its forms share, unless share_pool. The device memory the captures took (their
        warm-ups' lasting allocations, the pool, the capture stream's workspaces) is in self.captured_bytes, by form in
        self.captured_bytes_by_form. Returns the graphs, {logprobs: graph} by form."""
        if self.device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        self._check(rows, width, mode)
        self._refuse()
        return {logprobs: self._capture(rows, width, mode, logprobs) for logprobs in self.forms}

    def _capture(self, rows, width, mode, logprobs):
        torch = self.torch
        saved = {name: tensor.clone() for name, tensor in self._state().items()}

        def restore():
            for name, tensor in self._state().items():
                tensor.copy_(saved[name])
            self._reset()
        torch.cuda.synchronize(self.device)
        allocated, reserved = torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device)
        stream = capture_stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                restore()
                self.hold.fill_(1)
                self._round(rows, width, mode, logprobs)
            stream.synchronize()
            restore()
            graph = torch.cuda.CUDAGraph()
            # One pool for every round (share_pool), else one per (rows, width, mode) round for its two forms: they
            # never run at once, and what one form's capture makes is read only right after its own replays (its kept
            # view), never by the other's.
            if self.share_pool:
                if self._pool is None:
                    self._pool = torch.cuda.graph_pool_handle()
                pool = self._pool
            else:
                pool = self._pools.setdefault((rows, width, mode), torch.cuda.graph_pool_handle())
            # No garbage collection while the round is captured: a collection there frees what dead cycles held (an
            # engine closed before this one: its pinned result buffers), and pinned host memory freed while its
            # stream captures leaves torch's host allocator a block it cannot settle, so the next capture's flush of
            # the host cache fails with CUDA's invalid argument. Collected first, so nothing dead outlives the capture.
            collecting = gc.isenabled()
            gc.collect()
            gc.disable()
            observed = self._observer.snapshot() if self._observer is not None else None
            try:
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._round(rows, width, mode, logprobs)
            finally:
                if collecting:
                    gc.enable()
            key = (rows, width, mode, logprobs)
            self.graph_calls[key] = self._observer.called_since(observed) if observed is not None else []
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        restore()
        grown = max(torch.cuda.memory_allocated(self.device) - allocated,
                    torch.cuda.memory_reserved(self.device) - reserved, 0)
        self.captured_bytes += grown
        self.captured_bytes_by_form[FORMS[logprobs]] += grown
        self.graphs[key] = graph
        return graph

    def warm(self, mode, widths=None):
        """Compile at load what a round without a graph runs: one eager round, every row held (the capture's
        warm-ups), of each (rows, width) of `mode` the driver may launch but has not captured, in each of its forms
        (the module's docstring, Forms), rows 1 .. self.rows at `widths` (default the driver's round widths). A model's step() that launches the bound rows' round itself
        (a request Core's native rounds do not serve) then compiles nothing on a request. The driver's state is put
        back after them, so a bound sequence's stays as it was. Returns the rounds run."""
        torch = self.torch
        if self.device.type != "cuda":
            return 0
        chosen = sorted(set(self.round_widths if widths is None else widths))
        todo = [(rows, width, logprobs) for rows in range(1, self.rows + 1) for width in chosen
                for logprobs in self.forms if (rows, width, mode, logprobs) not in self.graphs]
        if not todo:
            return 0
        saved = {name: tensor.clone() for name, tensor in self._state().items()}

        def restore():
            for name, tensor in self._state().items():
                tensor.copy_(saved[name])
            self._reset()
        for rows, width, logprobs in todo:
            self._check(rows, width, mode)
            restore()
            self.hold.fill_(1)
            self._round(rows, width, mode, logprobs)
        torch.cuda.synchronize(self.device)
        restore()
        # Nothing of theirs stays: the views their plugins set (a truncation reads only a real round's) and the last
        # round's logits, then their intermediates back to the device, as the loader measures what the engine holds.
        for rows, width, logprobs in todo:
            self.kept.pop((rows, width, mode, logprobs), None)
        if getattr(self, "_logits", None) is not None:
            self._logits = None
        torch.cuda.empty_cache()
        return len(todo)

    def _refuse(self):
        """Raises RoundsRefused when the round cannot be captured (Rounds: a plugin lacks its round fact)."""

    def _state(self):
        names = ("pending", "frontier", "allowance", "out_count", "n_emitted", "n_matched", "catch_up_count", "catch_up",
                 "emitted", "drafts", "predictions", "hold", "counters", "width_table", "draft_counts")
        return {**{name: getattr(self, name) for name in names}, **{"start_" + k: v for k, v in self.start.items()},
                **{"width_policy_" + k: v for k, v in self.width_policy.state().items()}}

    def _check(self, rows, width, mode):
        if not 1 <= rows <= self.rows or not 0 <= width <= self.widest or mode not in MODES:
            raise ValueError(f"a round runs 1..{self.rows} rows at width 0..{self.widest} in one of {', '.join(MODES)}")

    def launch(self, rows, width, mode="greedy", hold=()):
        """Run one round of the first `rows` rows at `width` drafts: its graph when captured, else eagerly, in its
        log-probability form when a row not held asks for log-probabilities or follows a continuation (logprob_form),
        else in its plain form. `hold`: rows of those that sit this round out (emit nothing, keep their state), for a
        step that schedules only some of the bound rows."""
        self._check(rows, width, mode)
        if self.launched is not None:
            raise ValueError("read the round in flight before launching another")
        if hold:
            mask = [0] * rows
            for row in hold:
                mask[row] = 1
            self.hold[:rows].copy_(self.torch.tensor(mask, dtype=self.torch.long))
            self._held = True
        key = (rows, width, mode, self.logprob_form(rows, hold))
        graph = self.graphs.get(key)
        if graph is not None:
            graph.replay()
            if self._observer is not None:
                self._observer.replayed(self.graph_calls[key])
        else:
            self._round(*key)
        if self.event is not None:
            self.event.record()
        self.launched = key

    def read(self):
        """The round in flight's results, after one wait: per row {"tokens", "matched", "frontier", "accepted"}, its
        counters by name and, from a round in its log-probability form, "logprobs" (one per token) and with
        alternatives "top" (per token [(token, logprob), ...])."""
        if self.launched is None:
            raise ValueError("no round in flight")
        rows, scored = self.launched[0], self.launched[3]
        self.last, self.launched = self.launched, None
        if self.event is not None:
            self.event.synchronize()
        self._last_held = None
        if self._held:   # Core's native rounds never hold a row; a truncation of this round holds them again
            self._last_held = self.hold.clone()
            self.hold.zero_()
            self._held = False
        block = self.host[:rows].tolist()
        logprobs = self.host_logprobs[:rows].tolist() if scored and self.host_logprobs is not None else None
        ids = self.host_top_ids[:rows, :, :self.top].tolist() if scored and self.top else None
        values = self.host_top_logprobs[:rows, :, :self.top].tolist() if scored and self.top else None
        out = []
        for row, entries in enumerate(block):
            count = entries[0]
            found = {"tokens": entries[len(HEAD):len(HEAD) + count], "matched": entries[1], "frontier": entries[2],
                     "accepted": entries[self.words_at],
                     **{name: entries[self.words_at + 1 + i] for i, name in enumerate(self.counter_names)}}
            if logprobs is not None:
                found["logprobs"] = logprobs[row][:count]
            if ids is not None:
                found["top"] = [list(zip(ids[row][c], values[row][c])) for c in range(count)]
            out.append(found)
        return out

    def native_rounds(self, library, widths, mode="greedy"):
        """Core's native decode rounds over this driver's captured graphs of `mode` (crit_core.native.NativeRounds):
        a StepEngine model returns this from native_rounds(library), and Core then launches a decode step's round
        itself, reads self.host and commits the tokens, with no Python in between; the block's accepted drafts and
        counters go into the request's facts. widths[r - 1] is the width of the first round of r rows (None: r rows
        go to the model's step()), which an adaptive width policy then chooses round by round among the driver's
        widths (a start width must be one of them); every width a round may run needs its graph captured first
        (_native_graphs). Launch rounds on the stream that was current here only. After a native round Core calls the
        model's commit() only for rows it trims, ends, cancels or discards; adopt(rows, width) before truncate()
        then. Core launches each round in its log-probability form when a row's request asks for log-probabilities
        or replays a continuation, else in its plain form (the module's docstring, Forms), each form's graph
        captured."""
        from crit_core.native import NativeRounds
        graphs = {logprobs: {(rows, width): graph.raw_cuda_graph_exec()
                             for (rows, width, graph_mode, form), graph in self.graphs.items()
                             if graph_mode == mode and form == logprobs} for logprobs in self.forms}
        needed = self._native_graphs(widths)
        for logprobs in self.forms:
            for rows, width in needed:
                if (rows, width) not in graphs[logprobs]:
                    raise ValueError(f"no {mode} graph captured for {rows} rows at width {width} in the "
                                     f"{FORMS[logprobs]} form")
        table = [-1 if w is None else int(w) for w in widths]
        self.width_table.copy_(self.torch.tensor(table, dtype=self.torch.int32))
        self.host_widths.copy_(self.width_table.cpu())
        self._native_mode = mode
        stream = self.torch.cuda.current_stream(self.device).cuda_stream
        # The rounds' log-probabilities of their emitted tokens (logprobs=), which Core gives the requests that ask.
        ranked = dict(logprobs=self.host_logprobs.data_ptr(), top=self.top,
                      top_ids=self.host_top_ids.data_ptr() if self.top else 0,
                      top_logprobs=self.host_top_logprobs.data_ptr() if self.top else 0) \
            if self.logprobs is not None else {}
        return NativeRounds.cuda(library, stream, graphs[False], self.host.data_ptr(), self.host.shape[1],
                                 self.host.element_size(), self.rows, self.widest, self.host_widths.data_ptr(),
                                 max([w for _, w in needed] or [0]), telemetry=self.words_at,
                                 counters=len(self.counter_names), logprob_graphs=graphs.get(True), **ranked)

    def _native_graphs(self, widths):
        """The (rows, width) rounds native rounds over `widths` may run: each row count given a width at it, and with
        an adaptive width policy at every width of the driver's."""
        if len(widths) != self.rows:
            raise ValueError(f"one width per row count, {self.rows}")
        needed = []
        for rows, width in enumerate(widths, 1):
            if width is None:
                continue
            if self.width_policy.adaptive and width not in self.round_widths:
                raise ValueError(f"the start width {width} of {rows} rows is none of the round widths "
                                 f"{list(self.round_widths)}, which the {self.width_policy.name} policy chooses among")
            self._check(rows, width, "greedy")
            for w in (self.round_widths if self.width_policy.adaptive else (width,)):
                needed.append((rows, w))
        return needed

    def calibrate(self, mode="greedy", repeats=3):
        """The time of every captured round of `mode` in its plain form, the one most rounds run (CUDA events around
        `repeats` replays, every row held and the driver's state put back after), [rows][widths] in ms (inf where no
        round is captured), given to the width policy when it takes costs (set_costs). Returns the table."""
        torch = self.torch
        if self.device.type != "cuda":
            raise ValueError("calibrating times captured rounds: a CUDA device")
        saved = {name: tensor.clone() for name, tensor in self._state().items()}
        table = [[float("inf")] * len(self.round_widths) for _ in range(self.rows)]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for (rows, width, graph_mode, logprobs), graph in sorted(self.graphs.items()):
            if graph_mode != mode or logprobs or width not in self.round_widths:
                continue
            self.hold.fill_(1)
            graph.replay()
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            table[rows - 1][self.round_widths.index(width)] = start.elapsed_time(end) / repeats
        for name, tensor in self._state().items():
            tensor.copy_(saved[name])
        self._reset()
        torch.cuda.synchronize(self.device)
        costs = _method(self.width_policy, "set_costs")
        if costs is not None:
            costs(table)
        return table

    def adopt(self, rows, width):
        """A native round of (rows, width) ran (NativeRounds.last): it is the round a truncate() applies to, in the
        form Core launched it in (the round's pack left its form in host memory). Native rounds hold no row."""
        mode = getattr(self, "_native_mode", "greedy")
        self._check(rows, width, mode)
        self.last, self._last_held = (rows, width, mode, bool(self.host_form[0])), None

    def truncate(self, keep):
        """Core kept fewer tokens of the last round than it emitted (a stop it did not know, an output limit): each
        row keeps at most keep[row] of its emitted tokens (a prefix: a verifier's accepted path too), and every state
        is selected again from the round's start."""
        if self.last is None:
            raise ValueError("no round read to truncate")
        rows, width, mode, logprobs = self.last
        torch = self.torch
        keep = torch.as_tensor(list(keep), dtype=torch.long, device=self.device)
        count = torch.minimum(self.n_emitted[:rows], keep.clamp(min=0))
        self.n_matched[:rows] = torch.minimum(self.n_matched[:rows], (count - 1).clamp(min=0))
        self.n_emitted[:rows] = count
        self.emitted[:rows].masked_fill_(self.columns[None, :] >= count[:, None], -1)
        held = self._last_held
        if held is not None:
            self.hold.copy_(held)
        self._advance(rows, width)
        self._recommit(rows, width, mode, logprobs)
        if held is not None:
            self.hold.zero_()

    def _recommit(self, rows, width, mode, logprobs):
        self._commit(self._kept_view(rows, width, mode, logprobs))


class Rounds(SpeculativeRounds):
    """The round of a model plugin over the library's device forms (the module's docstring): `model` has
    verify(view, batch) -> [rows][width + 1][vocabulary] FP32 and optionally commit_round(view), count_round(view) and
    reset_capture(); `sampler` the sampler slot's plugin (predictions_device, and where it has them stage, reserve,
    laws, logprobs_device); `drafter` and `head` (the model's draft head, draft:head-device) when the selection drafts,
    `verifier` when it accepts other than by a chain; `pools` a list of the per-sequence state the round's rows read (a
    KV pool, a recurrent state pool, a TokenRing), which follow the round slots through bind/unbind/move/seed and keep
    each row's accepted prefix in commit_round; `extra_parts` other objects the round calls only through the two
    hooks every part may have, start_round(view) and count_round(view) (a device expert cache: its per-round
    bookkeeping and per-row reads and hits). `parts` (the model factory's) and `uses` (the slots its
    verify calls) check the selection's round facts before any capture (round_problems); with parts, the driver plans
    the device memory its captures take through the placement plugin (an entry "rounds:captures"), so Core's loader
    counts it as planned. vocabulary (the logits' width; needed when the sampler or drafter has reserve()) sizes
    their round buffers, reserved here before the first capture. rows, widest, device, logprobs (the alternatives packed per token, 0: none), stops,
    continuation, counters, widths and width_policy as SpeculativeRounds'; sampled: rounds that draw (the sampler's
    laws() hold the drafts' laws).

    A model plugin serving its rows through it declares max_row_tokens and decode_token_work of widest + 1 in its info
    (a round emits and computes up to widest + 1 columns a row; Core counts a native round's work as its width + 1,
    and refuses an engine whose grants cannot hold its widths), and round_counters naming `counters`."""

    def __init__(self, model, sampler, *, rows, widest, device, drafter=None, verifier=None, head=None, pools=(),
                 extra_parts=(), parts=None, uses=None, logprobs=0, stops=4, continuation=0, vocabulary=None,
                 sampled=False, share_pool=False, counters=(), widths=None, width_policy=None):
        super().__init__(None, rows, widest, device, stops=stops, continuation=continuation, logprobs=logprobs,
                         share_pool=share_pool, counters=counters, widths=widths, width_policy=width_policy)
        if drafter is not None and head is None:
            raise ValueError("a drafter drafts from the model's draft head (draft:head-device): give the head")
        if isinstance(pools, dict):
            raise ValueError("pools is a list of the round's pools")
        self.model, self.sampler, self.drafter, self.verifier, self.head = model, sampler, drafter, verifier, head
        self.pools = [(pool, _arguments(pool.bind)) for pool in pools]
        for pool, _ in self.pools:
            # A pool with a fixed number of round rows (DEVICE_EXTENSIONS rounds:<slot>: a KV pool's round tables, a
            # state pool's rows) holds rounds of at most that many rows.
            slots = getattr(pool, "round_rows", None)
            if slots is not None:
                if isinstance(slots, bool) or not isinstance(slots, int):
                    raise ValueError(f"a pool's round_rows is the number of round rows it holds (an int); "
                                     f"{type(pool).__name__}'s is {type(slots).__name__}")
                if slots < rows:
                    raise ValueError(f"rounds of {rows} rows bind round slots 0 .. {rows - 1}, and a pool of the round "
                                     f"({type(pool).__name__}) has {slots} round rows: give it {rows} or more")
        self.extra_parts = list(extra_parts)
        self.batch = RoundBatch(rows, widest, device)
        self.inputs = self.torch.zeros(rows, self.cols, dtype=self.torch.long, device=self.device)
        self._checked = (parts, uses, drafter is not None)
        self.problems = self._problems()
        self._placement = parts.get("placement") if parts is not None and callable(getattr(parts, "get", None)) else None
        self._observer = getattr(parts, "observer", None) if parts is not None else None
        self._accepts = _method(verifier, "accept_device") is not None
        self._logits = None
        self.vocabulary = vocabulary
        self.bound = {}         # slot -> (sequence, capacity, session, params)
        self.native = None      # the NativeRounds this driver handed Core
        self._planned = 0       # the captured bytes planned through the placement plugin so far
        for part, arguments in ((sampler, (rows, widest, vocabulary, self.device, self.top)),
                                (drafter, (rows, widest, vocabulary, self.device))):
            reserve = _method(part, "reserve")
            if reserve is None:
                continue
            if not vocabulary:
                raise ValueError("give the vocabulary (the logits' width): the sampler and drafter reserve their round "
                                 "buffers from it before the first capture")
            reserve(*arguments)
        laws = _method(sampler, "laws")
        if sampled:
            if laws is None:
                raise ValueError("sampled rounds keep each draft's law: the sampler gives them (laws(rows, width))")
            self.draft_laws = laws(rows, widest)
        self.sampled = bool(sampled)

    def _parts(self):
        """Every part of the round once (a part in two roles is called once per hook)."""
        found = {}
        for part in (self.model, self.sampler, self.drafter, self.head, self.verifier, *(p for p, _ in self.pools),
                     *self.extra_parts):
            if part is not None:
                found.setdefault(id(part), part)
        return list(found.values())

    # Slots ----------------------------------------------------------------------------------------------------------

    def bind(self, slot, sequence, frontier, token, allowance, capacity, *, stops=(), continuation=(), params=None,
             session=None, tokens=None, outputs=1, follow=None, logprobs=False):
        """Round slot `slot` decodes `sequence` from here (start_row's frontier, token, allowance, outputs, stops,
        continuation, follow: a replay or forced row, which needs the chain accept, and logprobs: its request asks for
        log-probabilities): every pool binds it (a KV pool reserves its pages up to `capacity` positions: the prompt,
        the output limit and the widest round), `tokens` (the sequence's tokens at positions 0 .. frontier: the prompt,
        the outputs so far, the pending token last) seed every pool and the drafter that have seed(slot, tokens), the
        sampler stages its rules (`params`, select()'s), the drafter binds it where it keeps state of its own, and
        Core's native rounds run `session` in the slot."""
        if follow is not None and self._accepts:
            raise ValueError("a row that follows its continuation is accepted by the chain: the selection's verifier "
                             "accepts on the device (a tree)")
        self.start_row(slot, frontier, token, allowance, stops, continuation, outputs, follow, logprobs)
        for pool, arguments in self.pools:
            pool.bind(slot, sequence, capacity) if arguments is None or arguments >= 3 else pool.bind(slot, sequence)
        bind = _method(self.drafter, "bind")
        if bind is not None:
            bind(slot, sequence)
        if tokens is not None:
            tokens = [int(t) for t in tokens]
            if len(tokens) != frontier + 1 or tokens[-1] != int(token):
                raise ValueError("tokens are the sequence's positions 0 .. frontier, the pending token last")
            for part in (*(p for p, _ in self.pools), self.drafter):
                seed = _method(part, "seed")
                if seed is not None:
                    seed(slot, tokens)
        stage = _method(self.sampler, "stage")
        if params is not None and stage is not None:
            stage(slot, params)
        self.bound[slot] = (sequence, capacity, session, params)
        if self.native is not None and session is not None:
            self.native.bind(slot, session)

    def unbind(self, slot):
        """Round slot `slot` holds no sequence: it emits nothing, and every pool lets it go."""
        self.release(slot)
        for pool, _ in self.pools:
            pool.unbind(slot)
        unbind = _method(self.drafter, "unbind")
        if unbind is not None:
            unbind(slot)
        self.bound.pop(slot, None)
        if self.native is not None:
            self.native.unbind(slot)

    def move(self, source, target):
        """The sequence in slot `source` moves to slot `target` (empty), so the decoding rows stay at the front."""
        sequence, capacity, session, params = self.bound[source]
        super().move(source, target)
        for pool, arguments in self.pools:
            if _method(pool, "move") is not None:
                pool.move(source, target)
            else:
                pool.unbind(source)
                pool.bind(target, sequence, capacity) if arguments is None or arguments >= 3 else \
                    pool.bind(target, sequence)
        stage = _method(self.sampler, "stage")
        if params is not None and stage is not None:
            stage(target, params)
        move = _method(self.drafter, "move")
        if move is not None:
            move(source, target)
        del self.bound[source]
        self.bound[target] = (sequence, capacity, session, params)
        if self.native is not None:
            self.native.unbind(source)
            if session is not None:
                self.native.bind(target, session)

    # The round ------------------------------------------------------------------------------------------------------

    def _draft(self, view):
        if self.drafter is not None:
            self.drafter.draft(view, self.head, self.sampler)

    def _verify(self, view):
        rows, width = view.rows, view.width
        self.inputs[:rows, 0] = view.pending
        if width:
            self.inputs[:rows, 1:width + 1] = view.drafts[:, :width]
        view.batch = self.batch.fill(rows, width, view.frontier, self.inputs)
        logits = self._logits = self.model.verify(view, view.batch)
        if tuple(logits.shape[:2]) != (rows, width + 1) or logits.dim() != 3:
            raise ValueError(f"verify returns the logits of every column, [{rows}][{width + 1}][vocabulary], "
                             f"not {list(logits.shape)}")
        if view.mode in ("greedy", "sampled"):
            laws = view.draft_laws if view.mode == "sampled" else None
            found = self.sampler.predictions_device(view, logits, laws)
            written = view.predictions[:, :width + 1]
            # A sampler that wrote view.predictions in place returns that slice (no copy: at one row narrower than the
            # widest, a view of it has other strides over the same elements, which copy_ refuses).
            if found is not None and found.data_ptr() != written.data_ptr():
                written.copy_(found.reshape(rows, width + 1))

    def _scored(self, view):
        """Each emitted token's log-probability (and ranked alternatives) in emitted order, after the accept: emitted
        token j was predicted by column j of a chain, by node view.tree_path[j] of a tree (its verifier sets the path),
        whose logits rows are gathered into emitted order. The sampler's logprobs_device computes them where it has it
        (Core's TokenLogprob and tie order); otherwise the driver does, in Core's tie order. Past n_emitted the values
        are of token 0 and Core reads none of them. A plain-form round (view.logprobs None) scores nothing."""
        if view.logprobs is None or self._logits is None:
            return
        width, logits = view.width, self._logits
        chosen = view.emitted[:, :width + 1].clamp(min=0)
        if view.tree_path is not None:
            index = view.tree_path[:, :width + 1].clamp(min=0)
            logits = logits.gather(1, index[..., None].expand(-1, -1, logits.shape[-1]))
        given = _method(self.sampler, "logprobs_device")
        if given is not None:
            given(view, logits, chosen)
            return
        values = logits.float().log_softmax(-1)
        view.logprobs[:, :width + 1] = values.gather(2, chosen[..., None]).squeeze(2)
        if self.top:
            # Core's ranking: higher first, the lower id first among equals (a stable sort, not topk's tie order).
            ranked, order = values.sort(dim=-1, descending=True, stable=True)
            view.top_ids[:, :width + 1] = order[..., :self.top]
            view.top_logprobs[:, :width + 1] = ranked[..., :self.top]

    def _accept_view(self, view):
        if self._accepts:
            self.verifier.accept_device(view)   # writes n_emitted, n_matched and emitted; the driver advances
        else:
            self._accept(view.rows, view.width)

    def _start(self, view):
        for part in self._parts():
            start = _method(part, "start_round")
            if start is not None:
                start(view)

    def _commit(self, view):
        commit = _method(self.model, "commit_round")
        if commit is not None:
            commit(view)
        for pool, _ in self.pools:
            pool.commit_round(view)
        for part in (self.drafter, self.head):
            commit = _method(part, "commit_round")
            if commit is not None:
                commit(view)

    def _counted(self, view):
        for part in self._parts():
            count = _method(part, "count_round")
            if count is not None:
                count(view)

    def _recommit(self, rows, width, mode, logprobs):
        # A replayed graph's round left the batch's host attributes at the last eager fill: refill it as the round
        # had it (its inputs and frontier are the round's start) before the pools read it again.
        view = self._kept_view(rows, width, mode, logprobs)
        view.batch = self.batch.fill(rows, width, self.start["frontier"], self.inputs)
        self._commit(view)

    def _reset(self):
        for part in self._parts():
            reset = _method(part, "reset_capture")
            if reset is not None:
                reset()

    def _problems(self):
        """Why the selection cannot run inside a captured round, asked again (a plugin's round_problems() reads its
        configuration as it is now)."""
        parts, uses, drafts = self._checked
        return [] if parts is None else round_problems(parts, uses, drafts=drafts)

    def _refuse(self):
        self.problems = self._problems()
        if self.problems:
            raise RoundsRefused("the round cannot be captured: " + "; ".join(self.problems))

    def capture(self, rows, width, mode):
        graphs = super().capture(rows, width, mode)
        self._plan()
        return graphs

    def _plan(self):
        """The device memory the captures took so far, planned through the placement plugin (the entry
        "rounds:captures"), so Core's loader counts it among the engine's planned bytes."""
        if self._placement is None or self.captured_bytes <= self._planned:
            return
        grown = self.captured_bytes - self._planned
        free = self.torch.cuda.mem_get_info(self.device)[0]
        placed = self._placement.plan({"rounds:captures": grown}, {"device": free + grown, "host": 0})
        if placed.get("rounds:captures") != "device":
            raise RoundsRefused("the placement plugin put the captured rounds' memory off the device")
        self._planned = self.captured_bytes

    def _check(self, rows, width, mode):
        super()._check(rows, width, mode)
        if mode == "sampled" and not self.sampled and width:
            raise ValueError("sampled rounds keep each draft's law: build the driver with sampled=True")

    # Core's native rounds -------------------------------------------------------------------------------------------

    def native_rounds(self, library, widths, mode="greedy"):
        """Core's native rounds (SpeculativeRounds.native_rounds) over graphs captured here for every width named,
        or None when the selection cannot run inside a captured round (self.problems) or the device is no GPU: the
        model then decodes every row through step(). With an adaptive width policy every width of the driver's is
        captured for each row count given one, and a policy that weighs round times (RateWidth) gets the captured
        rounds' times (calibrate()). Each round is captured in each of the driver's forms. Core opens them before any
        request, so no slot is bound yet; bind() binds each session for Core from then on."""
        self.problems = self._problems()
        if self.problems or self.device.type != "cuda":
            return None
        for rows, width in self._native_graphs(widths):
            if any((rows, width, mode, logprobs) not in self.graphs for logprobs in self.forms):
                self.capture(rows, width, mode)
        if _method(self.width_policy, "set_costs") is not None:
            self.calibrate(mode)   # a policy weighing the rounds' times gets them before the first round
        self.warmed = self.warm(mode)   # the rounds a model's step() may launch without a graph, compiled now
        self.native = super().native_rounds(library, widths, mode)
        if self.native is not None:   # Core launches them: the loader counts their runs as the captures' calls
            self.native.captured_slots = sorted({slot for key, slots in self.graph_calls.items() if key[2] == mode
                                                 for slot in slots})
        return self.native


class TokenRing:
    """The last `size` tokens of each round slot's sequence before its pending token, on the device: for inputs a model
    computes inside a captured round from the tokens before each column (hashed n-gram ids, a token history), which its
    verify reads with windows(). A pool of the round (Rounds' pools): bind() empties a slot (every entry `fill`), seed()
    gives it the sequence's tokens (Rounds.bind's tokens: positions 0 .. frontier, the pending token last, which the
    ring leaves out), and each round keeps the inputs of the columns it accepted (commit_round, from the round's start:
    a truncation's second commit gives the same window). Every buffer is allocated once; windows() and commit_round are
    capture-safe."""

    def __init__(self, rows, size, device, fill=-1):
        import torch
        if rows < 1 or size < 1:
            raise ValueError("a token ring holds one row or more of one token or more")
        self.torch, self.size, self.fill = torch, int(size), int(fill)
        self.ring = torch.full((rows, self.size), self.fill, dtype=torch.long, device=device)
        self.start = self.ring.clone()
        self.offsets = torch.arange(self.size, dtype=torch.long, device=device)

    def bind(self, slot, sequence, capacity=None):
        self.ring[slot].fill_(self.fill)

    def seed(self, slot, tokens):
        """The sequence's tokens at positions 0 .. frontier (host ints, the pending token last): the ring holds the
        last `size` before the pending one."""
        tokens = list(tokens)[:-1][-self.size:]
        self.ring[slot].fill_(self.fill)
        if tokens:
            self.ring[slot, self.size - len(tokens):] = self.torch.tensor(tokens, dtype=self.torch.long)

    def unbind(self, slot):
        self.ring[slot].fill_(self.fill)

    def move(self, source, target):
        self.ring[target] = self.ring[source]
        self.ring[source].fill_(self.fill)

    def start_round(self, view):
        self.start[:view.rows] = self.ring[:view.rows]

    def commit_round(self, view):
        rows, cols = view.rows, view.width + 1
        inputs = view.batch.tokens_t.view(rows, cols)
        joined = self.torch.cat([self.start[:rows], inputs], dim=1)
        self.ring[:rows] = joined.gather(1, self.offsets[None, :] + view.n_emitted[:, None])

    def windows(self, tokens, count):
        """[rows, k, count]: for each of `tokens` ([rows, k], the inputs that follow each row's ring: a round's columns,
        a draft step's token), the `count` tokens ending at it (count - 1 of history, `fill` before the sequence's
        start); count at most size + 1."""
        if not 1 <= count <= self.size + 1:
            raise ValueError(f"windows of 1 to {self.size + 1} tokens")
        rows = tokens.shape[0]
        joined = self.torch.cat([self.ring[:rows, self.size - (count - 1):], tokens], dim=1)
        return joined.unfold(1, count, 1)


__all__ = ["FORMS", "HEAD", "MODES", "WIDTH_POLICIES", "FixedWidth", "RateWidth", "RoundView", "Rounds", "RoundsRefused",
           "SpeculativeRounds", "StepWidth", "TokenRing", "capture_stream", "round_fact", "round_problems",
           "width_policy"]
