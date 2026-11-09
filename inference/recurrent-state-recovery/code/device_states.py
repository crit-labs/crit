"""The gated-delta recurrence of many sequences on a device, with the causal convolution before it and the device
commit of a speculative round's accepted prefix: the recurrence slot's device form (crit_plugins.device), in torch, over
crit-labs' recurrent-scans kernels where they are installed.

Each token decays the state (exp(log_decay) per value head), takes the beta-weighted delta of its value against the
state's prediction for its key, and reads the result with its query: reference.py's formula over [heads][keys][values]
FP32 states, keys and queries L2-normalized (x / sqrt(sum x^2 + epsilon)) when the geometry says so. Value head h reads
key head h // (heads / key_heads), so a model passes grouped keys and queries as they come out of its projections.

A StatePool keeps, per layer and sequence, the live state, and for rows of up to `widest + 1` tokens (decode tokens,
verify windows) a stash of each token's normalized key, decay and delta; with `conv` the window of the last kernel - 1
inputs of a causal depthwise convolution and a stash of the row's inputs. A row reads its sequence's live state and
window and writes neither: its stash says what it would add, and a commit picks how much of it to keep.

- step() rows (host-driven): accept(sequence, count) records the count, which the sequence's next row (or a flush)
  applies from the stash on the device (deferred commit, recurrent-state-recovery's Strategy C); a row of more tokens
  (a prompt chunk) keeps its final state and window, which accept(sequence, its tokens) makes live (0 drops them).
- captured rounds (crit_plugins.device.RoundBatch): bind(slot, sequence) points round slot `slot` at the sequence in
  a device table the round reads, and commit_round(view) keeps each row's first view.n_emitted columns on the device,
  by the pool's commit strategy: "deferred" records the count (the next row applies it), "checkpoints" has the row write
  the state before and after every column and selects one, "recompute" has the row keep its start state and replays the
  stashed prefix into the live state. The window is always deferred. A commit can run again with a smaller count before
  the sequence's next row (a host truncation), and before a flush: flush(), state(), window() and fork() apply a
  deferred commit, which makes it final (checkpoints and recompute stay rerunnable until the next round). A round row
  of an unbound slot reads and writes no sequence and outputs zeros.
- captured prompt chunks (crit_plugins.device.PromptBatch, prompts:recurrence): stage_prompt(batch) puts the chunk's
  sequence and length in fixed device rows before each replay; the chunk continues the live state and window in place
  (its real tokens only; the padding takes no step and outputs zeros) and commit_prompt(batch) settles it after, with
  no accept. Padding alone (a capture's warm-ups) changes no sequence.

The arithmetic (the column scan with its stash, checkpoints and replay; the convolution) is recurrent-scans' pool rows
when the plugin found the kernel (crit_plugins.kernels.registry_kernel), and this module's torch formula otherwise; the
commit's bookkeeping (counts, the checkpoint select) is torch's here in both cases. Prompt rows up to the pool's
scan_rows tokens run the scan with the step's decode rows (one launch a layer); longer ones fla's chunked kernel or
recurrent-scans' two-pass chunked prefill, by the plugin's choice. The scan and a chunked kernel give other bits, so
which one a prompt row runs follows from its length and the configuration only, never from a timing: the pool's
scan_rows (an eager call) and scan_rows_captured (inside a captured graph) are the plugin's exact_rows option, the same
value for both (DEFAULT_EXACT_ROWS when a pool is made without a plugin). scan_rows() measures where a chunked kernel
overtakes the scan on a shape and device, to recommend a value for the configuration; nothing calls it at load. fla's
chunked kernel runs on stated tiles too (FLA_TILES): fla autotunes its Triton kernels by timing, and some of their tiles
give other bits, so every kernel on its path is pinned to one. Every
operation on rows and commits takes fixed shapes from the batch and reads sequences through device tensors, so a
captured round replays it; a step's row tables reach the device by asynchronous copies from pinned host memory.
"""

import time

from crit_plugins.device import StepBatch, capturing

STASH, CHECKPOINTS, RECOMPUTE, FINAL, FLUSH = 0, 1, 2, 3, 4   # recurrent-scans' row kinds
REPLAY, FLUSH_PENDING = 2, 3                                  # and its replay modes
COMMITS = ("deferred", "checkpoints", "recompute")
ROUND_KINDS = {"deferred": STASH, "checkpoints": CHECKPOINTS, "recompute": RECOMPUTE}
META = 5                                                      # a row table's columns: first, tokens, sequence, kind, slot
UPLOADS = 4                                                   # pinned host tables in flight
# Prompt rows of up to this many tokens run the scan where neither the plugin nor the caller states a value: 512, the
# largest of crit_plugins.prompt_graphs' default buckets, so every captured bucket and every eager chunk of up to 512
# tokens has the bits of decoding its tokens one by one, and a chunk of a longer prompt cut at 1,024 tokens or more runs
# the chunked kernel like the full chunks before it whenever it is longer than 512 tokens. On an RTX 5090 the scan
# stays the cheaper path in an eager call up to 860-1,190 tokens over the gated-delta shapes measured (host launches
# included) and in GPU time up to 106-246 (scan_rows()).
DEFAULT_EXACT_ROWS = 512


def conv_shape(conv):
    """(channels, kernel, dilation) of a pool's conv argument: (channels, kernel) or (channels, kernel, dilation)."""
    found = tuple(int(x) for x in conv)
    if len(found) not in (2, 3):
        raise ValueError("conv is (channels, kernel) or (channels, kernel, dilation)")
    channels, width, dilation = (*found, 1) if len(found) == 2 else found
    if channels < 1 or not 2 <= width <= 8 or dilation < 1:
        raise ValueError("conv is (channels, kernel[, dilation]): one channel or more, a kernel of 2 to 8 inputs, a "
                         "dilation of 1 or more")
    return channels, width, dilation


class StatePool:
    """Every sequence's recurrent state per layer (module docstring). layers: the layer ids; heads: value heads, each a
    [keys][values] state; key_heads: heads of the queries and keys (heads by default); sequences: open at once; widest:
    the widest verify window's drafts (rows of up to widest + 1 tokens commit any prefix); conv: (channels, kernel) or
    (channels, kernel, dilation) for the causal convolution (its window: the last (kernel - 1) * dilation inputs of each
    channel), or None; rows: round slots (sequences by default; given for the checkpoints and recompute
    strategies, which keep a state per slot); commit: a round's strategy; kernels: recurrent-scans' module, or None for
    the torch formula."""

    def __init__(self, layers, heads, keys, values, device, sequences=None, widest=0, conv=None, key_heads=None,
                 rows=None, commit="deferred", kernels=None):
        import torch
        if commit not in COMMITS:
            raise ValueError(f"commit is one of {', '.join(COMMITS)}, not {commit!r}")
        if commit != "deferred" and rows is None:
            raise ValueError(f"the {commit} commit keeps a state per round slot: give the round's rows")
        key_heads = heads if key_heads is None else int(key_heads)
        if heads < 1 or keys < 1 or values < 1 or key_heads < 1 or heads % key_heads:
            raise ValueError("a state of keys x values per value head, value heads a multiple of key heads")
        if widest < 0:
            raise ValueError("widest is the drafts of the widest verify window: zero or more")
        self.torch = torch
        self.layers = {layer: i for i, layer in enumerate(layers)}
        self.heads, self.key_heads, self.keys, self.values = heads, key_heads, keys, values
        self.device = torch.device(device)
        self.sequences = 64 if sequences is None else int(sequences)
        self.widest, self.capacity, self.commit = int(widest), int(widest) + 1, commit
        self.kernels = kernels if self.device.type == "cuda" else None
        self.round_rows = self.sequences if rows is None else int(rows)
        count, n = len(self.layers), self.sequences
        # The spare sequence (index n) where the round rows of unbound slots write: the counts' always (the commit
        # scatters them), the states' where the commit scatters them too (checkpoints) or the torch formula runs a row.
        spare = int(self.kernels is None)
        self.spare = n
        f32 = dict(dtype=torch.float32, device=self.device)
        i32 = dict(dtype=torch.int32, device=self.device)
        if self.kernels is not None:   # the kernel's layout: per-block counts and parities, a two-half stash
            self.blocks = self.kernels.pool_blocks(heads, values)
            self.half = self.kernels.pool_half_floats(key_heads, heads, keys, values, self.capacity)
            halves, conv_blocks = 2, self.kernels.pool_conv_blocks(int(conv[0])) if conv else 0
        else:                          # the torch formula's: one count, one half
            self.blocks = 1
            self.half = self.capacity * (key_heads * keys + heads + heads * values)
            halves, conv_blocks = 1, 1
        self.states = torch.zeros(count, n + max(spare, int(commit == "checkpoints")), heads, keys, values, **f32)
        self.stash = torch.zeros(count, n + spare, halves * self.half, **f32)
        self.control = torch.zeros(count, n + 1, 2 * self.blocks, **i32)
        self.channels = self.width = self.span = 0
        self.dilation = 1
        if conv is not None:
            channels, width, dilation = conv_shape(conv)
            self.channels, self.width, self.dilation = channels, width, dilation
            self.span = (width - 1) * dilation   # the window: the last span inputs of each channel, oldest first
            self.windows = torch.zeros(count, n + spare, channels, self.span, **f32)
            self.conv_stash = torch.zeros(count, n + spare, self.capacity, channels, **f32)
            self.conv_pending = torch.zeros(count, n + 1, conv_blocks, **i32)
            self._conv_none = (torch.zeros(0, channels, **f32), torch.zeros(0, channels, **f32),
                               torch.zeros(channels, width, **f32))
        self.table = torch.full((self.round_rows,), -1, **i32)
        self.slot_ids = torch.arange(self.round_rows, dtype=torch.int64, device=self.device)
        size = heads * keys * values
        self.checkpoints = (torch.zeros(count, self.round_rows, self.capacity + 1, size, **f32)
                            if commit == "checkpoints" else None)
        self.starts = torch.zeros(count, self.round_rows, size, **f32) if commit == "recompute" else None
        self.free = list(range(n - 1, -1, -1))
        self.index, self.bound = {}, {}
        self.last = {}            # sequence -> (tokens of its last row, prompt row) since its last accept
        self.finals, self.final_windows = {}, {}   # sequence -> {layer index: state / window after its prompt row}
        self._plans = None        # the last StepBatch's plan (the same for every layer of a step)
        # Prompt rows up to this many tokens run the scan (forward): in an eager call, and inside a captured graph. The
        # plugin that makes the pool sets both from its exact_rows option; never measured here.
        self.scan_rows = self.scan_rows_captured = DEFAULT_EXACT_ROWS
        # The prompt chunk stage_prompt put in the pool's fixed buffers (a captured chunk reads them): its row table
        # (the scan's, kind FINAL, writing the final state and window over the live ones), its flush row (the pending
        # count applied, before a chunked kernel), its sequence's index (padding alone: the spare sequence where the
        # pool keeps one, else sequence 0, which is then written its own values) and whether one is staged.
        self.prompt_meta = torch.tensor([[0, 0, -1, FINAL, -1]], dtype=torch.int32, device=self.device)
        self.prompt_flush = torch.tensor([[0, 0, -1, FLUSH, -1]], dtype=torch.int32, device=self.device)
        self.prompt_index = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.prompt_real = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.prompt_sequence = None
        self._uploads, self._upload = [None] * UPLOADS, 0   # pinned host tables and the copy event of each

    # -- sequences ------------------------------------------------------------------------------------------------------
    def open(self, sequence):
        if sequence in self.index:
            raise ValueError(f"sequence {sequence!r} is open already")
        if not self.free:
            raise MemoryError(f"the state pool's {self.sequences} sequences are all open")
        i = self.index[sequence] = self.free.pop()
        self.states[:, i].zero_()
        self.control[:, i].zero_()
        if self.channels:
            self.windows[:, i].zero_()
            self.conv_pending[:, i].zero_()

    def close(self, sequence):
        i = self.index.pop(sequence, None)
        if i is None:
            return
        for slot in [s for s, held in self.bound.items() if held == sequence]:
            self.unbind(slot)
        self.free.append(i)
        for kept in (self.last, self.finals, self.final_windows):
            kept.pop(sequence, None)

    def fork(self, sequence, new):
        """A new sequence with the sequence's committed state and window (a flush first)."""
        self.flush(sequence)
        source = self._index(sequence)
        self.open(new)
        target = self.index[new]
        self.states[:, target].copy_(self.states[:, source])
        if self.channels:
            self.windows[:, target].copy_(self.windows[:, source])

    def state(self, layer, sequence):
        """The sequence's committed state of a layer: [heads][keys][values] (a flush first)."""
        self.flush(sequence)
        return self.states[self.layers[layer], self._index(sequence)]

    def window(self, layer, sequence):
        """The sequence's committed convolution window of a layer: [channels][kernel - 1], oldest input first (a flush
        first)."""
        self.flush(sequence)
        return self.windows[self.layers[layer], self._index(sequence)]

    def accept(self, sequence, count):
        """Commit the first `count` tokens of the sequence's row in the last step, every layer (and window): any prefix
        of a row of up to widest + 1 tokens, all or none of a longer one."""
        tokens, prompt = self.last.pop(sequence, (0, False))
        finals, windows = self.finals.pop(sequence, {}), self.final_windows.pop(sequence, {})
        if count == 0:
            return
        i = self._index(sequence)
        if prompt:
            if count != tokens:
                raise ValueError(f"a prompt row of {tokens} tokens keeps its final state only, not {count}")
            for li, final in finals.items():
                self.states[li, i].copy_(final)
            for li, window in windows.items():
                self.windows[li, i].copy_(window)
            return
        if not 0 < count <= tokens:
            raise ValueError(f"the sequence's last row held {tokens} tokens, not {count}")
        self.control[:, i, :self.blocks].fill_(count)
        if self.channels:
            self.conv_pending[:, i].fill_(count)

    def flush(self, sequence=None):
        """Apply the pending commit of the sequence (every open one by default) to its live state and window. A
        deferred commit is final from here: commit_round may run again with a smaller count only before a flush."""
        indices = [self._index(sequence)] if sequence is not None else sorted(self.index.values())
        if not indices:
            return
        meta = self._meta([[0, 0, i, FLUSH, -1] for i in indices])
        if self.kernels is not None:
            self.kernels.pool_delta_replay(self.states, self.stash, self.control, None, None, meta=meta, rows=len(indices),
                                           capacity=self.capacity, key_heads=self.key_heads, mode=FLUSH_PENDING)
            if self.channels:
                out, x, weight = self._conv_none
                self.kernels.pool_conv(out, self.windows, self.conv_stash, self.conv_pending, None, x, weight, None,
                                       meta=meta, rows=len(indices), kind=FLUSH, layer=0, layers=len(self.layers),
                                       capacity=self.capacity, silu=False)
            return
        index = meta[:, 2].long()
        for r in range(len(indices)):
            for li in range(len(self.layers)):
                _torch_state(self, li, index[r:r + 1], FLUSH)
                if self.channels:
                    _torch_window(self, li, index[r:r + 1])

    # -- captured prompt chunks (crit_plugins.prompt_graphs) ------------------------------------------------------------
    def stage_prompt(self, batch):
        """The PromptBatch's sequence (batch.sequence; None: padding alone, which changes no state) in the pool's fixed
        prompt buffers, so the chunk's forward and convolve read and write its state and window on the device: its
        batch.count real tokens continue the live state and window, the padding takes no step. Before each replay."""
        sequence = batch.sequence
        i = -1 if sequence is None else self._index(sequence)
        rows = self._meta([[0, batch.count, i, FINAL, i], [0, 0, i, FLUSH, -1]])
        self.prompt_meta.copy_(rows[0:1])
        self.prompt_flush.copy_(rows[1:2])
        self.prompt_index.fill_(i if i >= 0 else self.spare if self.states.shape[1] > self.spare else 0)
        self.prompt_real.fill_(i >= 0)
        self.prompt_sequence = sequence
        if sequence is not None:   # the chunk follows whatever the sequence's last step left
            self.last.pop(sequence, None)
            self.finals.pop(sequence, None)
            self.final_windows.pop(sequence, None)

    def commit_prompt(self, batch):
        """The sequence's state and window now cover batch.frontier + batch.count positions: the chunk wrote them in
        place (no accept follows a captured chunk). After the replay."""
        if batch.sequence != self.prompt_sequence:
            raise ValueError(f"the pool staged {self.prompt_sequence!r}'s prompt chunk, not {batch.sequence!r}'s")
        self.prompt_sequence = None

    # -- captured rounds ------------------------------------------------------------------------------------------------
    def bind(self, slot, sequence):
        """Round slot `slot` holds `sequence`'s state and window."""
        if not 0 <= slot < self.round_rows:
            raise ValueError(f"round slot {slot} is outside the pool's {self.round_rows}")
        self.table[slot:slot + 1].fill_(self._index(sequence))
        self.bound[slot] = sequence

    def unbind(self, slot):
        if 0 <= slot < self.round_rows:
            self.table[slot:slot + 1].fill_(-1)
        self.bound.pop(slot, None)

    def move(self, source, target):
        """The sequence of round slot `source` moves to slot `target`."""
        sequence = self.bound.get(source)
        self.unbind(source)
        if sequence is not None:
            self.bind(target, sequence)

    def commit_round(self, view):
        """Keep each round row's first view.n_emitted columns (state and window) on the device; capture-safe, and again
        with smaller counts before the rows' next round (a truncation) and, for the deferred commit, before a flush."""
        torch = self.torch
        rows, columns = view.rows, view.width + 1
        self._round_rows(rows)
        count = view.n_emitted[:rows].clamp(0, columns)
        sequences = self.table[:rows].long()
        rows_at = torch.where(sequences < 0, self.spare, sequences)   # unbound rows write the spare sequence
        slots = self.slot_ids[:rows]
        counts = count.to(torch.int32)[None, :, None]
        if self.commit == "deferred":
            self.control[:, rows_at, :self.blocks] = counts
        elif self.commit == "checkpoints":
            chosen = self.checkpoints[:, slots, count]
            self.states[:, rows_at] = chosen.view(len(self.layers), rows, self.heads, self.keys, self.values)
        elif self.kernels is not None:
            accepted = view.n_emitted[:rows]
            if accepted.dtype != torch.int64 or accepted.stride(0) != 1:
                accepted = accepted.to(torch.int64).contiguous()
            self.kernels.pool_delta_replay(self.states, self.stash, self.control, self.starts, accepted, slots=slots,
                                           table=self.table, rows=rows, columns=columns, capacity=self.capacity,
                                           key_heads=self.key_heads, mode=REPLAY)
        else:
            for r in range(rows):
                _torch_recompute(self, rows_at[r:r + 1], r, count[r])
        if self.channels:
            self.conv_pending[:, rows_at] = counts

    # -- internals ------------------------------------------------------------------------------------------------------
    def _index(self, sequence):
        try:
            return self.index[sequence]
        except KeyError:
            raise ValueError(f"sequence {sequence!r} is not open") from None

    def _round_rows(self, rows):
        if not 1 <= rows <= self.round_rows:
            raise ValueError(f"a round of {rows} rows in a pool of {self.round_rows} round slots (rows)")

    def _meta(self, table):
        """A row table [rows][META] int32 on the device: copied asynchronously from pinned host memory on CUDA (a ring
        of UPLOADS host buffers, each reused once its last copy is done)."""
        torch = self.torch
        if self.device.type != "cuda":
            return torch.tensor(table, dtype=torch.int32, device=self.device).view(-1, META)
        values = torch.tensor(table, dtype=torch.int32).view(-1)
        k = self._upload
        self._upload = (k + 1) % UPLOADS
        found = self._uploads[k]
        if found is None or found[0].numel() < values.numel():
            host = torch.empty(max(values.numel(), 64 * META), dtype=torch.int32, pin_memory=True)
        else:
            host, event = found
            event.synchronize()   # the copy that last read this buffer is done (long ago, a step back)
        host[:values.numel()].copy_(values)
        meta = torch.empty(values.numel(), dtype=torch.int32, device=self.device)
        meta.copy_(host[:values.numel()], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._uploads[k] = (host, event)
        return meta.view(-1, META)

    def _plan(self, batch):
        """A StepBatch's rows, the same for every layer of a step: {"rows": (sequence, index, first, tokens) each,
        "prompts": the rows of more than widest + 1 tokens, "conv": the convolution's row table, "index": each row's
        sequence index on the device}, and the state's row tables by how prompt rows run (delta_rows)."""
        if self._plans is not None and self._plans[0] is batch:
            return self._plans[1]
        rows, conv, prompts = [], [], []
        for r in range(batch.rows):
            sequence, first, count, _ = batch.row(r)
            i = self._index(sequence)
            prompt = count > self.capacity
            rows.append((sequence, i, first, count))
            conv.append([first, count, i, FLUSH if prompt else STASH, -1])
            if prompt:
                prompts.append(r)
            self.last[sequence] = (count, prompt)
        meta = self._meta(conv)
        plan = {"rows": rows, "prompts": prompts, "conv": meta, "index": meta[:, 2].long(), "delta": {}}
        self._plans = (batch, plan)
        return plan

    def delta_rows(self, plan, exact_rows, chunked):
        """(row table, kinds, scratch slots) of the state's rows: a prompt row runs the scan keeping its last state
        (FINAL, in a scratch slot of its own) when it is at most exact_rows tokens or no chunked kernel runs, else the
        chunked kernel after a flush (FLUSH)."""
        key = (exact_rows, chunked)
        if key not in plan["delta"]:
            kinds = [STASH if count <= self.capacity else FINAL if count <= exact_rows or chunked is None else FLUSH
                     for _, _, _, count in plan["rows"]]
            slots, table = [], []
            for (_, i, first, count), kind in zip(plan["rows"], kinds):
                slot = len([s for s in slots if s >= 0]) if kind == FINAL else -1
                slots.append(slot)
                table.append([first, count, i, kind, slot])
            plan["delta"][key] = (self._meta(table), kinds, slots)
        return plan["delta"][key]


# -- the device form -------------------------------------------------------------------------------------------------
def _normal(torch, x, epsilon, enabled):
    if not enabled:
        return x
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + epsilon)


def _grouped(pool, x, name):
    """x [tokens][key heads][keys], its rows contiguous."""
    if x.dim() != 3 or x.shape[1] != pool.key_heads or x.shape[2] != pool.keys:
        raise ValueError(f"{name} is [tokens][{pool.key_heads} key heads][{pool.keys}], not {list(x.shape)}")
    return x if x.stride(2) == 1 else x.contiguous()


# Every Triton kernel fla's chunk_gated_delta_rule runs forward, (module, kernel) -> its stated tile (meta-parameters,
# warps, stages). fla autotunes each by timing its configs at the first call of a shape (and keeps the pick in Triton's
# cache across processes), and some tiles give other bits: measured on an RTX 5090 at a gated-delta shape of 16 key
# heads and 48 value heads of 128, l2norm_fwd_kernel's 8-row blocks on 8 or 16 warps and 16-row blocks on 16 warps against its
# other tiles, and chunk_fwd_kernel_o's 64-key block against its 32- or 128-key ones. Pinned, a prompt row gets these
# tiles' bits under any load and on every run. Each is one of fla's own configs for the kernel on every NVIDIA GPU, and
# one its autotuner picked for FP32 rows (the plugin's chunk_fp32 default) at that shape; with BF16 rows they take about
# 3% more kernel time than its picks at 2,048 tokens.
FLA_TILES = {
    ("fla.modules.norm.l2norm", "l2norm_fwd_kernel"): ({"BT": 16}, 4, 3),
    ("fla.ops.utils.cumsum", "chunk_local_cumsum_scalar_kernel"): ({}, 2, 3),
    ("fla.ops.gated_delta_rule.chunk_fwd", "chunk_gated_delta_rule_fwd_kkt_solve_kernel"): ({"BK": 64}, 1, 3),
    ("fla.ops.gated_delta_rule.wy_fast", "recompute_w_u_fwd_kernel"): ({}, 8, 2),
    ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"): ({"BV": 32}, 2, 2),
    ("fla.ops.common.chunk_o", "chunk_fwd_kernel_o"): ({"BK": 32, "BV": 32}, 2, 3),
}
_FLA = {}   # "pinned": whether fla's kernels hold FLA_TILES, decided once a process


def fla_pinned():
    """Whether fla is installed and its chunked kernels run on FLA_TILES: each kernel's autotuner holds its stated
    config as its only one, which Triton's autotuner runs without timing anything. Pinned once a process, for every
    caller of those fla kernels in it. False without fla, or with an fla whose kernels are not where FLA_TILES names
    them (the tiles were stated for another fla): its chunked kernel then does not run."""
    if "pinned" not in _FLA:
        _FLA["pinned"] = _pin_fla()
    return _FLA["pinned"]


def _pin_fla():
    import importlib
    try:
        import triton
        from triton.runtime.autotuner import Autotuner
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        tuners = []
        for module, kernel in FLA_TILES:
            tuner = getattr(importlib.import_module(module), kernel, None)
            while tuner is not None and not isinstance(tuner, Autotuner):   # through triton.heuristics' wrapper
                tuner = getattr(tuner, "fn", None)
            if tuner is None:
                return False
            tuners.append(tuner)
    except ImportError:
        return False
    for tuner, (meta, warps, stages) in zip(tuners, FLA_TILES.values()):
        tuner.configs = [triton.Config(dict(meta), num_warps=warps, num_stages=stages)]
    return True


def chunked_kind(pool, geometry, prompt):
    """The chunked kernel prompt rows run through: "fla" (prompt "fla", fla installed and pinned to FLA_TILES, its
    normalization), "two_pass" (recurrent-scans' prefill, its shapes) or None (the scan)."""
    if pool.device.type != "cuda":
        return None
    if prompt == "fla" and geometry["normalize_keys"] == geometry["normalize_queries"] and geometry["epsilon"] == 1e-6:
        if fla_pinned():
            return "fla"
    if pool.kernels is not None and pool.keys in (64, 128) and pool.values % 64 == 0:
        return "two_pass"
    return None


STANDARD = {"epsilon": 1e-6, "normalize_keys": True, "normalize_queries": True}   # the models' normalization
SCAN_AGREEMENT = "the state after the prompt, its window and its outputs: the bits of decoding its tokens one by one"
CHUNKED_AGREEMENT = ("the state after the prompt and its outputs within 2e-3 + 2e-2 |x| of the scan's, the window the "
                     "same bits (fla in FP32 on its stated tiles, the two-pass prefill: another summation order)")
CAPTURED = ("a captured prompt chunk (prompts:recurrence, a padded bucket): in a bucket of up to scan_rows_captured "
            "tokens the "
            "scan over its real tokens, the bits of the step's scan rows; in a larger one the chunked kernel over the "
            "bucket, the padding's decay 1 and beta 0, within the chunked tolerance; padding changes no state or window")
PROBE_TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)   # prompt lengths the crossovers are looked for at
_CROSSOVERS = {}   # (shape, device, kernels, chunked kernel, chunk_fp32) -> the measured crossovers
WAYS = ("eager", "captured")


def scan_rows(pool, prompt, chunk_fp32=True):
    """{"eager": rows, "captured": rows}: the longest prompt row the scan runs (one launch a layer, the bits of decode
    rows) before the chunked kernel (chunked_kind) costs less, measured on the pool's shape and device at each of
    PROBE_TOKENS until the chunked kernel is the cheaper both ways, each crossover interpolated between the two lengths
    around it. An eager call costs the longer of its host time (Python and launches, which the stream waits for when the
    host is behind, as in a short prompt) and its GPU time; a call inside a captured graph costs its GPU time. Without
    a chunked kernel (or off CUDA) every prompt row runs the scan. Measured once a process for a shape, a GPU, the
    scan's kernel and the chunked kernel.

    A recommendation for the plugins' exact_rows option, which a configuration states: the two paths give other bits
    and the measurement moves with the machine's load (on one RTX 5090 at one shape, eager 861 to 1,044 tokens from one
    process to the next), so no pool calls this when it is made or runs."""
    import torch
    geometry = {"keys": pool.keys, "values": pool.values, "query_scale": pool.keys ** -0.5, **STANDARD}
    chunked = chunked_kind(pool, geometry, prompt)
    if chunked is None:
        return dict.fromkeys(WAYS, 1 << 30)
    key = (pool.key_heads, pool.heads, pool.keys, pool.values, str(pool.device),
           torch.cuda.get_device_name(pool.device), pool.kernels is not None, chunked, chunk_fp32)
    if key in _CROSSOVERS:
        return _CROSSOVERS[key]
    probe = StatePool([0], pool.heads, pool.keys, pool.values, pool.device, 1, 0, None, pool.key_heads, None,
                      "deferred", pool.kernels)
    probe.open(0)
    g = torch.Generator(device="cpu").manual_seed(0)
    last, found = None, {}
    for tokens in PROBE_TOKENS:
        q, k = (torch.randn(tokens, pool.key_heads, pool.keys, generator=g).to(pool.device, torch.bfloat16)
                for _ in range(2))
        v = torch.randn(tokens, pool.heads, pool.values, generator=g).to(pool.device, torch.bfloat16)
        log_decay = (-torch.rand(tokens, pool.heads, generator=g) * 0.5).to(pool.device)
        beta = torch.rand(tokens, pool.heads, generator=g).to(pool.device)
        batch = StepBatch.build([(0, 0, tokens)], pool.device)

        def run(exact):
            forward(0, q, k, v, log_decay, beta, probe, batch, geometry, exact, chunk_fp32, prompt)
            probe.last.clear()
            probe.finals.clear()
        scan, chunk = _costs(torch, lambda: run(1 << 30)), _costs(torch, lambda: run(0))
        times = {"eager": (max(scan), max(chunk)), "captured": (scan[0], chunk[0])}
        for way in WAYS:
            s1, c1 = times[way]
            if way in found or c1 >= s1:
                continue
            if last is None:
                found[way] = tokens // 2
            else:   # where the two lines cross between the last length the scan won and this one
                t0, before = last
                s0, c0 = before[way]
                ahead, behind = max(c0 - s0, 0.0), s1 - c1
                found[way] = int(t0 + (tokens - t0) * ahead / (ahead + behind))
        if len(found) == len(WAYS):
            break
        last = (tokens, times)
    probe.close(0)
    _CROSSOVERS[key] = {way: max(found.get(way, PROBE_TOKENS[-1]), 1) for way in WAYS}
    return dict(_CROSSOVERS[key])


def _costs(torch, fn, reps=3):
    """(GPU ms, host ms) of fn, the least of reps: queued behind a spin kernel, so CUDA events time the GPU alone and
    the host's clock its Python and launches alone."""
    fn()
    torch.cuda.synchronize()
    gpu = host = None
    for _ in range(reps):
        torch.cuda._sleep(20_000_000)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        began = time.perf_counter()
        fn()
        launched = (time.perf_counter() - began) * 1e3
        end.record()
        end.synchronize()
        found = start.elapsed_time(end)
        gpu = found if gpu is None else min(gpu, found)
        host = launched if host is None else min(host, launched)
    return gpu, host


def _unbound_zero(torch, out, sequences, rows, cols):
    """A round's outputs with the rows of unbound slots (sequence -1) zero, capture-safe."""
    view = out.view(rows, cols, -1)
    view.copy_(torch.where((sequences >= 0)[:, None, None], view, torch.zeros((), dtype=out.dtype, device=out.device)))


def forward(layer, query, key, value, log_decay, beta, pool, batch, geometry, exact_rows=None, chunk_fp32=True,
            prompt="fla", out=None):
    """Outputs [tokens][heads][values] FP32 of a StepBatch's or RoundBatch's rows, each from its sequence's committed
    state. A prompt row (more than widest + 1 tokens) of at most exact_rows tokens (None: the pool's scan_rows, or
    scan_rows_captured while the stream captures a graph, both the plugin's exact_rows) runs the scan, one launch a
    layer with the step's decode rows, its state the
    bits of decoding its tokens one by one; a longer one the chunked kernel (chunked_kind), in FP32 with chunk_fp32
    where fla runs it: in BF16 its outputs part from the exact recurrence by up to tens of percent where a strong decay
    leaves them small against their neighbours, in FP32 by a few percent, at twice the kernel's time."""
    torch = pool.torch
    li = pool.layers[layer]
    q, k = _grouped(pool, query, "query"), _grouped(pool, key, "key")
    v = value if value.stride(2) == 1 else value.contiguous()
    if not q.dtype == k.dtype == v.dtype or q.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        q, k, v = q.float(), k.float(), v.float()
    g = log_decay if log_decay.dtype == torch.float32 and log_decay.stride(-1) == 1 else log_decay.float().contiguous()
    b = beta if beta.dtype == torch.float32 and beta.stride(-1) == 1 else beta.float().contiguous()
    tokens = v.shape[0]
    if out is None:
        out = torch.empty(tokens, pool.heads, pool.values, dtype=torch.float32, device=v.device)
    args = (geometry["epsilon"], geometry["query_scale"], geometry["normalize_keys"], geometry["normalize_queries"])
    if batch.prompt:
        return _prompt_state(pool, li, q, k, v, g, b, batch, geometry, chunk_fp32, prompt, out)
    if batch.round:
        cols = batch.width + 1
        pool._round_rows(batch.rows)
        if cols > pool.capacity:
            raise ValueError(f"a round of {cols} columns in a pool of verify rows up to {pool.capacity} (widest)")
        kind = ROUND_KINDS[pool.commit]
        scratch = pool.checkpoints[li] if kind == CHECKPOINTS else pool.starts[li] if kind == RECOMPUTE else None
        if pool.kernels is not None:
            pool.kernels.pool_delta(out, pool.states, pool.stash, pool.control, scratch, q, k, v, g, b,
                                    starts=batch.starts_t, slots=batch.slots_t, table=pool.table, rows=batch.rows,
                                    kind=kind, layer=li, capacity=pool.capacity, epsilon=args[0], scale=args[1],
                                    normalize_keys=args[2], normalize_queries=args[3])
            return out
        sequences = pool.table[:batch.rows].long()
        rows_at = torch.where(sequences < 0, pool.spare, sequences)
        for r in range(batch.rows):
            _torch_state(pool, li, rows_at[r:r + 1], kind, r * cols, cols, (q, k, v, g, b), args, out, scratch, r)
        _unbound_zero(torch, out, sequences, batch.rows, cols)
        return out
    chunked = chunked_kind(pool, geometry, prompt)
    plan = pool._plan(batch)
    rows, prompts, index = plan["rows"], plan["prompts"], plan["index"]
    if exact_rows is None:   # the pool's stated crossover (the plugin's exact_rows)
        exact_rows = pool.scan_rows_captured if capturing() else pool.scan_rows
    meta, kinds, slots = pool.delta_rows(plan, exact_rows, chunked)
    finals = [r for r, kind in enumerate(kinds) if kind == FINAL]
    size = pool.heads * pool.keys * pool.values
    scratch = torch.empty(len(finals), size, dtype=torch.float32, device=pool.device) if finals else None
    if pool.kernels is not None:
        pool.kernels.pool_delta(out, pool.states, pool.stash, pool.control, scratch, q, k, v, g, b, meta=meta,
                                rows=batch.rows, kind=STASH, layer=li, capacity=pool.capacity, epsilon=args[0],
                                scale=args[1], normalize_keys=args[2], normalize_queries=args[3])
    else:
        for r, ((_, i, first, count), kind) in enumerate(zip(rows, kinds)):
            _torch_state(pool, li, index[r:r + 1], kind, first, count, (q, k, v, g, b), args, out, scratch, slots[r])
    shape = (pool.heads, pool.keys, pool.values)
    for r in finals:   # the scratch holds the FINAL rows' states only, each kept until accept
        pool.finals.setdefault(rows[r][0], {})[li] = scratch[slots[r]].view(shape)
    for r in prompts:
        sequence, i, first, count = rows[r]
        if kinds[r] != FLUSH:
            continue
        span = slice(first, first + count)
        state = pool.states[li, i]
        if chunked == "fla":
            final = _fla(torch, q[span], k[span], v[span], g[span], b[span], state, geometry, chunk_fp32, out[span])
        else:
            final = _two_pass(pool, q[span], k[span], v[span], g[span], b[span], state, geometry, out[span])
        pool.finals.setdefault(sequence, {})[li] = final
    return out


def _prompt_state(pool, li, q, k, v, g, b, batch, geometry, chunk_fp32, prompt, out):
    """A captured prompt chunk's rows (a PromptBatch staged by stage_prompt), from the live state after its pending
    count, the final state written over the live one; the padding takes no step and its outputs are zeros. A bucket of
    up to the pool's scan_rows_captured tokens (the crossover in GPU time: the chunk runs as a graph) runs the scan over
    the real tokens (its count the staged row table's, so the launch is the same for any chunk); a larger one the
    chunked kernel over the whole bucket, the padding's decay exp(0) = 1 and beta 0 (a step that leaves the state's
    bits)."""
    torch = pool.torch
    if v.shape[0] != batch.bucket:
        raise ValueError(f"a prompt chunk of {batch.bucket} tokens (its bucket), not {v.shape[0]}")
    args = (geometry["epsilon"], geometry["query_scale"], geometry["normalize_keys"], geometry["normalize_queries"])
    chunked = chunked_kind(pool, geometry, prompt)
    if chunked is not None and pool.scan_rows_captured is not None and batch.bucket > pool.scan_rows_captured:
        return _prompt_chunked(pool, li, q, k, v, g, b, batch, geometry, chunk_fp32, chunked, args, out)
    out.zero_()
    if pool.kernels is not None:
        live = pool.states[li].view(pool.states.shape[1], -1)   # FINAL's scratch slot i: sequence i's live state
        pool.kernels.pool_delta(out, pool.states, pool.stash, pool.control, live, q, k, v, g, b, meta=pool.prompt_meta,
                                rows=1, kind=STASH, layer=li, capacity=pool.capacity, epsilon=args[0], scale=args[1],
                                normalize_keys=args[2], normalize_queries=args[3])
        return out
    # The torch formula steps through the bucket: a padding token's decay exp(0) = 1 and beta 0 leave the state's bits.
    valid = batch.valid_t[:, None]
    g, b = torch.where(valid, g, 0.0), torch.where(valid, b, 0.0)
    final = torch.empty(1, pool.heads * pool.keys * pool.values, dtype=torch.float32, device=pool.device)
    _torch_state(pool, li, pool.prompt_index, FINAL, 0, batch.bucket, (q, k, v, g, b), args, out, final, 0)
    pool.states[li].index_copy_(0, pool.prompt_index, final.view(1, pool.heads, pool.keys, pool.values))
    out.copy_(torch.where(valid[:, :, None], out, 0.0))
    return out


def _prompt_chunked(pool, li, q, k, v, g, b, batch, geometry, chunk_fp32, chunked, args, out):
    """A captured prompt chunk through the chunked kernel: the staged sequence's pending count applied (its flush row),
    its live state gathered, the bucket run with the padding's decay and beta masked to a step that changes nothing, the
    final state scattered back (padding alone writes back what it read)."""
    torch = pool.torch
    valid = batch.valid_t[:, None]
    g, b = torch.where(valid, g, 0.0), torch.where(valid, b, 0.0)
    if pool.kernels is not None:
        pool.kernels.pool_delta(out, pool.states, pool.stash, pool.control, None, q, k, v, g, b, meta=pool.prompt_flush,
                                rows=1, kind=STASH, layer=li, capacity=pool.capacity, epsilon=args[0], scale=args[1],
                                normalize_keys=args[2], normalize_queries=args[3])
    else:
        _torch_state(pool, li, pool.prompt_index, FLUSH)
    state = pool.states[li].index_select(0, pool.prompt_index)[0]
    if chunked == "fla":
        final = _fla(torch, q, k, v, g, b, state, geometry, chunk_fp32, out)
    else:
        final = _two_pass(pool, q, k, v, g, b, state, geometry, out)
    final = torch.where(pool.prompt_real[:, None, None], final, state)
    pool.states[li].index_copy_(0, pool.prompt_index, final[None])
    out.copy_(torch.where(valid[:, :, None], out, 0.0))
    return out


def _fla(torch, q, k, v, g, b, state, geometry, chunk_fp32, out):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    q, k, v = (x[None].float() if chunk_fp32 else x[None] for x in (q, k, v))
    o, final = chunk_gated_delta_rule(q, k, v, g[None], b[None], scale=geometry["query_scale"],
                                      initial_state=state[None].contiguous(), output_final_state=True,
                                      use_qk_l2norm_in_kernel=geometry["normalize_keys"])
    out.copy_(o[0])
    return final[0].float()


def _two_pass(pool, q, k, v, g, b, state, geometry, out):
    torch, eps = pool.torch, geometry["epsilon"]
    qn = _normal(torch, q.float(), eps, geometry["normalize_queries"]).contiguous()
    kn = _normal(torch, k.float(), eps, geometry["normalize_keys"]).contiguous()
    final = state.clone()
    found = torch.empty(out.shape, dtype=torch.float32, device=out.device)
    pool.kernels.gated_delta_prefill(qn, kn, v.float().contiguous(), g.contiguous(), b.contiguous(), final,
                                     scale=geometry["query_scale"], out=found)
    out.copy_(found)
    return final


def convolve(layer, x, weight, bias, pool, batch, activation="silu", out=None):
    """The causal depthwise convolution of each row of x ([tokens][channels]) over its sequence's window, then the
    activation (silu, or None): y[t][c] = act(bias[c] + sum_i weight[c][i] x[t - (kernel - 1 - i) * dilation][c]) at
    the pool's dilation (1 unless its conv says), inputs before the row from the window; y in x's dtype (out, when
    given, is not x). The window keeps the row's inputs pending, as the
    state's rows: accept and commit_round make the accepted prefix's inputs part of it (a prompt row's all of them)."""
    torch = pool.torch
    if not pool.channels:
        raise ValueError("the pool keeps no convolution window: pool(..., conv=(channels, kernel[, dilation]))")
    li = pool.layers[layer]
    weight = weight.reshape(pool.channels, pool.width)
    if weight.dtype != torch.float32 or not weight.is_contiguous():
        weight = weight.float().contiguous()
    if bias is not None and (bias.dtype != torch.float32 or not bias.is_contiguous()):
        bias = bias.float().contiguous()
    if activation not in ("silu", None):
        raise ValueError(f"the convolution's activation is silu or None, not {activation!r}")
    silu = activation == "silu"
    if x.dim() != 2 or x.shape[1] != pool.channels:
        raise ValueError(f"x is [tokens][{pool.channels}]")
    if x.stride(1) != 1 or x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        x = x.contiguous() if x.dtype in (torch.float32, torch.bfloat16, torch.float16) else x.float().contiguous()
    if out is None:
        out = torch.empty_like(x)
    elif x.numel() and out.data_ptr() == x.data_ptr():
        raise ValueError("the convolution writes its outputs beside its inputs: out must not be x")
    if batch.prompt:   # a captured prompt chunk: its real inputs from the live window, the window after them in place
        if x.shape[0] != batch.bucket:
            raise ValueError(f"a prompt chunk of {batch.bucket} tokens (its bucket), not {x.shape[0]}")
        if pool.kernels is not None:   # zeros for the padding, the window after the real inputs in place
            pool.kernels.pool_conv_prompt(out, pool.windows, pool.conv_stash, pool.conv_pending, x, weight, bias,
                                          meta=pool.prompt_meta, layer=li, capacity=pool.capacity, silu=silu)
            return out
        final = _torch_window(pool, li, pool.prompt_index, FINAL, 0, batch.bucket, x, weight, bias, silu, out,
                              batch.count_t)
        pool.windows[li].index_copy_(0, pool.prompt_index, final[None])
        out.copy_(torch.where(batch.valid_t[:, None], out, torch.zeros((), dtype=out.dtype, device=out.device)))
        return out
    if batch.round:
        cols = batch.width + 1
        pool._round_rows(batch.rows)
        if cols > pool.capacity:
            raise ValueError(f"a round of {cols} columns in a pool of verify rows up to {pool.capacity} (widest)")
        if pool.kernels is not None:
            pool.kernels.pool_conv(out, pool.windows, pool.conv_stash, pool.conv_pending, None, x, weight, bias,
                                   starts=batch.starts_t, slots=batch.slots_t, table=pool.table, rows=batch.rows,
                                   kind=STASH, layer=li, layers=1, capacity=pool.capacity, silu=silu)
            return out
        sequences = pool.table[:batch.rows].long()
        rows_at = torch.where(sequences < 0, pool.spare, sequences)
        for r in range(batch.rows):
            _torch_window(pool, li, rows_at[r:r + 1], STASH, r * cols, cols, x, weight, bias, silu, out)
        _unbound_zero(torch, out, sequences, batch.rows, cols)
        return out
    plan = pool._plan(batch)
    rows, meta, prompts, index = plan["rows"], plan["conv"], plan["prompts"], plan["index"]
    if pool.kernels is not None:
        pool.kernels.pool_conv(out, pool.windows, pool.conv_stash, pool.conv_pending, None, x, weight, bias, meta=meta,
                               rows=batch.rows, kind=STASH, layer=li, layers=1, capacity=pool.capacity, silu=silu)
        for r in prompts:
            sequence, i, first, count = rows[r]
            span = slice(first, first + count)
            final = torch.empty(pool.channels, pool.span, dtype=torch.float32, device=pool.device)
            pool.kernels.conv_segments(out[span], pool.windows[li, i], final, x[span], weight, bias, silu=silu)
            pool.final_windows.setdefault(sequence, {})[li] = final
        return out
    for r, (sequence, i, first, count) in enumerate(rows):
        prompt = count > pool.capacity
        final = _torch_window(pool, li, index[r:r + 1], FINAL if prompt else STASH, first, count, x, weight, bias, silu,
                              out)
        if prompt:
            pool.final_windows.setdefault(sequence, {})[li] = final
    return out


# -- the torch formula (the fallback, and the CPU's) -------------------------------------------------------------------
def _views(pool, half):
    """key [capacity][key heads][keys], decay [capacity][heads], delta [capacity][heads][values] of a stash half."""
    c, kh, h = pool.capacity, pool.key_heads, pool.heads
    a, b = c * kh * pool.keys, c * h
    return (half[:a].view(c, kh, pool.keys), half[a:a + b].view(c, h), half[a + b:a + b + c * h * pool.values].view(c, h, pool.values))


def _replay(pool, torch, S, key, decay, delta, n):
    """S [.., heads, keys, values] after the first n (a device scalar) stashed steps: every step computed, kept where
    its column is below n, so the shapes are fixed."""
    ratio = pool.heads // pool.key_heads
    for c in range(pool.capacity):
        kc = key[..., c, :, :].repeat_interleave(ratio, dim=-2)
        e, d = decay[..., c, :], delta[..., c, :, :]
        step = S * e[..., :, None, None] + kc[..., :, :, None] * d[..., :, None, :]
        S = torch.where(n > c, step, S)
    return S


def _torch_state(pool, li, index, kind, first=0, count=0, inputs=None, args=None, out=None, scratch=None, slot=0):
    """One row of the torch formula (the kernel's semantics: pending applied, then the row's kind); index [1] int64 of
    its sequence, slot the row's place in the scratch."""
    torch = pool.torch
    states, stash, control = pool.states[li], pool.stash[li], pool.control[li]
    S = states.index_select(0, index)[0]
    row = stash.index_select(0, index)[0]
    key, decay, delta = _views(pool, row[:pool.half])
    pending = control.index_select(0, index)[0, 0].clamp(0, pool.capacity)
    S = _replay(pool, torch, S, key, decay, delta, pending)
    states.index_copy_(0, index, S[None])
    control.index_fill_(0, index, 0)
    if kind == FLUSH or count <= 0:
        return
    if kind == CHECKPOINTS:
        scratch[slot, 0] = S.reshape(-1)
    elif kind == RECOMPUTE:
        scratch[slot] = S.reshape(-1)
    q, k, v, g, b = inputs
    epsilon, scale, normalize_keys, normalize_queries = args
    ratio = pool.heads // pool.key_heads
    for t in range(count):
        at = first + t
        kv, qv = k[at].float(), q[at].float()
        ik = 1.0 / torch.sqrt((kv * kv).sum(-1) + epsilon) if normalize_keys else torch.ones_like(kv[:, 0])
        iq = (1.0 / torch.sqrt((qv * qv).sum(-1) + epsilon) if normalize_queries else torch.ones_like(qv[:, 0])) * scale
        kn = kv * ik[:, None]                                       # [key heads][keys]
        kh, qh = kn.repeat_interleave(ratio, 0), (qv * iq[:, None]).repeat_interleave(ratio, 0)
        e = torch.exp(g[at].double()).float()                       # [heads]
        x = (kh[:, :, None] * S).sum(1)                             # k . S, [heads][values]
        y = (qh[:, :, None] * S).sum(1)
        d = b[at][:, None] * (v[at].float() - e[:, None] * x)
        out[at] = e[:, None] * y + (qh * kh).sum(-1)[:, None] * d
        S = S * e[:, None, None] + kh[:, :, None] * d[:, None, :]
        if kind in (STASH, RECOMPUTE) and t < pool.capacity:
            key[t], decay[t], delta[t] = kn, e, d
        if kind == CHECKPOINTS and t < pool.capacity:
            scratch[slot, t + 1] = S.reshape(-1)
    if kind in (STASH, RECOMPUTE):
        stash.index_copy_(0, index, row[None])
    if kind == FINAL:
        scratch[slot] = S.reshape(-1)


def _torch_recompute(pool, index, slot, n):
    """The recompute commit of one round row: its start state advanced by n stashed steps, every layer."""
    torch = pool.torch
    S = pool.starts[:, slot].view(len(pool.layers), pool.heads, pool.keys, pool.values)
    halves = pool.stash.index_select(1, index)[:, 0, :pool.half]
    views = [_views(pool, halves[li]) for li in range(len(pool.layers))]
    key, decay, delta = (torch.stack([v[j] for v in views]) for j in range(3))
    S = _replay(pool, torch, S, key, decay, delta, n)
    pool.states.index_copy_(1, index, S[:, None])


def _torch_window(pool, li, index, kind=FLUSH, first=0, count=0, x=None, weight=None, bias=None, silu=True, out=None,
                  real=None):
    """One row of the convolution in torch: pending inputs shifted in, then the row (STASH: its inputs stashed; FINAL:
    the window after it returned, after its first `real` inputs where real ([1] int64 on the device) is given)."""
    torch = pool.torch
    windows, stash, pending = pool.windows[li], pool.conv_stash[li], pool.conv_pending[li]
    window = windows.index_select(0, index)[0]
    inputs = stash.index_select(0, index)[0]
    n = pending.index_select(0, index)[0, 0].clamp(0, pool.capacity)
    for c in range(pool.capacity):
        window = torch.where(n > c, torch.cat([window[:, 1:], inputs[c][:, None]], 1), window)
    windows.index_copy_(0, index, window[None])
    pending.index_fill_(0, index, 0)
    if kind == FLUSH or count <= 0:
        return None
    xs = x[first:first + count].float()
    padded = torch.cat([window.T, xs])
    base = bias if bias is not None else torch.zeros_like(weight[:, 0])
    # A decode or verify row a token at a time: a token's bits are then those it has in any row (a CPU's vector math
    # rounds a tensor's tail elements its own way); a prompt row at once.
    spans = [(t, 1) for t in range(count)] if kind == STASH else [(0, count)]
    d = pool.dilation   # tap i of token t reads padded[t + i d]: x[t - (kernel - 1 - i) d]
    for t, tokens in spans:
        acc = base[None, :].expand(tokens, -1)
        for i in range(pool.width):
            acc = acc + weight[:, i] * padded[t + i * d:t + i * d + tokens]
        out[first + t:first + t + tokens] = (torch.nn.functional.silu(acc) if silu else acc).to(out.dtype)
    if kind == STASH:
        inputs[:count] = xs
        stash.index_copy_(0, index, inputs[None])
        return None
    if real is not None:   # the window after the row's real inputs: padded[real .. real + span - 1]
        at = real + torch.arange(pool.span, dtype=torch.int64, device=padded.device)
        return padded.index_select(0, at).T.contiguous()
    return padded[-pool.span:].T.contiguous()
