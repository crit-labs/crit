"""The attention slot's device form over attention-state's KV pool (pages.py), which the library's attention plugins
share: each token attends its sequence's positions up to its own, read where the pool keeps them.

Arithmetic comes from crit's paged-kv-attention kernels (crit_plugins.kernels: built on this GPU with the engine's
tuned knobs or the defaults), then from torch where they do not load; FlashInfer serves nothing unless the plugins'
option flashinfer is "auto" (a yardstick to compare with: its paged prefill for eager prompt rows of 16-bit and FP8
pages, and with decode "flashinfer" a step's window rows). A row of up to the pool's `window` tokens is a window row
(decode tokens, verify windows), a longer one a prompt row:

- prompt rows: with prompt_reader "decode", as window rows of up to 64 columns through crit's split-KV reader, each
  program folding its partitions as the merge does (fused, no partials): a prompt token's bits are those of the same
  position decoded or in a round; else crit's prompt reader over every page kind (16-bit, FP8, fp8:row, INT8; int8_qk:
  INT8 queries on integer tensor cores), one row a launch, with log-sum-exps where asked; FlashInfer's paged prefill
  over 16-bit and FP8 pages only with option flashinfer "auto" (planned once per step, no KV split, rows planned in
  groups of one query-tile class); else the row's KV gathered and torch's scaled dot product attention;
- window rows: crit's split-KV reader, in a step (decode "split", the default) as in a captured round (a
  RoundBatch) and in a pool with step buffers, which reads every page type with no host work, its partitions walking
  each row's own length, its partials packed by the rows' tokens; else a capture-safe torch fallback over the same
  page tables. So a row a step decodes (plain decoding, log-probabilities, a replay) has the bits of the same row in a
  round. decode "flashinfer": a step's window rows through FlashInfer's decode and prefill over 16-bit and FP8 pages
  instead (another reduction than a round's: for engines that decode in steps only);
- tree masks (`parents`), block-sparse lists (`blocks`) and log-sum-exp outputs (`lse`): the split-KV reader, else
  torch's masked softmax over the gathered rows;
- reader "flash" (the plugins' option): every launch above that goes to crit's split-KV reader (block lists aside)
  takes the kernel's flash structures instead (contiguous splits of one chunk per pool, one tile shape per pool: one
  reduction for decode rows, windows, rounds and fused prompt pieces), and crit's prompt reader is the flash prompt
  chunk (not row-invariant: its splits follow the chunk's length); reader "cuda": the same launches through the
  kernel's CUDA flash tile instead (cuda-window, cuda-fused, cuda-prefill: queries in shared memory, 16 rows a warp,
  each warp's softmax in registers; one chunk and block per pool, its own reduction).

Every split-KV launch over a pool takes one partition count and one piece size per pool (block-sparse lists
PIECE_PARTITIONS in every launch: prompt pieces, steps and rounds), and the kernel's tile for its columns (tiles every
GPU multiplies with mma.sync, one summation order whatever the tile): so a column's bits are the same in a step and in
a round, alone and in a verify window, on every GPU, and a block-sparse row's the same as a prompt's. The torch fallbacks are the plain arithmetic the kernels are tested against (FP32 masked softmax), and serve
correctly where crit's kernel is not there; in a capture the fallback reads every position its
page tables hold (its cost follows the pool's capacity), in an eager step only the positions its rows reach.

What the device form allocates it allocates when it is made, as memory() declares it (device_memory): the split-KV
reader's partials where crit's kernel loads, and where FlashInfer serves its scheduling workspace and its wrappers
(each with FlashInfer's own int workspace and as much pinned host memory). Nothing it plans or copies from the host
runs under a capture: a capture reaching a FlashInfer plan or a host row's metadata is an error.

SPLIT_TOKENS, CHUNK, PIECE_TOKENS, PIECE_ROWS and PIECE_PARTITIONS are defaults measured on one sm_120 GPU (an RTX
5090), as the kernel's own partitions, piece and tiles are where the engine has not tuned it (crit kernels sweep).
"""
from crit_plugins.device import capturing
import math

from . import pages as kv_pages

SPLIT_TOKENS = 1024     # positions per KV split of FlashInfer's decode, whatever the batch
CHUNK = 1024            # positions per pass of the torch fallback over page tables
PIECE_TOKENS = 64       # block-sparse prompt rows: tokens per launch row,
PIECE_ROWS = 8          # launch rows per launch,
PIECE_PARTITIONS = 4    # and KV partitions (each token is a program of its own), for every block-sparse launch
# FlashInfer 0.7.3's wrappers: each allocates an int workspace of this many bytes on the device, and as many pinned on
# the host, when it is made, and a prefill wrapper a KV-length buffer too. Its prefill picks one of these query tiles
# per plan; one prefill wrapper serves each.
FLASHINFER_INT_BYTES = 8 << 20
FLASHINFER_PREFILL_BYTES = 32768 * 4
PREFILL_TILES = (16, 64, 128)


def engine_device(context):
    """The device the engine runs on: the load context's, or CUDA when torch has it (a plugin's own tests)."""
    device = getattr(context, "device", None)
    if device is None:
        try:
            import torch
        except ImportError:
            return "cpu"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def device_memory(device, mode, kernels_loaded, kernels_mib, flashinfer, flashinfer_mib, decode,
                  prompt_reader="prefill"):
    """(device bytes, host bytes) the device form allocates when it is made: the split-KV reader's partials
    (kernels_mib, where crit's kernel loaded), and where FlashInfer serves (mode auto, option flashinfer auto,
    FlashInfer installed) its scheduling workspace (flashinfer_mib) and its wrappers (one per prefill tile class, and
    a decode wrapper with decode "flashinfer"), each FLASHINFER_INT_BYTES on the device and on the host (pinned), a
    prefill wrapper FLASHINFER_PREFILL_BYTES more on the device; none with prompt_reader "decode" (every row the
    split-KV reader's). A plugin's memory() declares them."""
    if mode == "torch" or not str(device).startswith("cuda"):
        return 0, 0
    found = (int(kernels_mib) << 20) if kernels_loaded and kernels_mib else 0
    host = 0
    if _flashinfer_serves(mode, flashinfer) and prompt_reader != "decode":
        wrappers = len(PREFILL_TILES) + (decode == "flashinfer")
        found += (int(flashinfer_mib) << 20) + wrappers * FLASHINFER_INT_BYTES
        found += len(PREFILL_TILES) * FLASHINFER_PREFILL_BYTES
        host += wrappers * FLASHINFER_INT_BYTES
    return found, host


def _flashinfer_serves(mode, flashinfer):
    return mode != "torch" and flashinfer == "auto" and _flashinfer_installed()


class Windows:
    """Window rows of one launch, as device tensors: each row's table row (`rows`, into `tables`), first token
    (`starts`), column count (`counts`, None: `columns` each) and column 0's position (`firsts`); `columns` the widest
    row. With `offsets` (each row's first partial row, its tokens packed after the rows before it) the partials follow
    the tokens: `bounds` (host) holds the most partial rows each row takes and `total` (host) the most all of them take
    (a capture's rows may take any count up to the window, and no more than the step's tokens together)."""

    def __init__(self, tables, rows, starts, counts, firsts, columns, offsets=None, bounds=None, total=None):
        self.tables, self.rows, self.starts, self.counts, self.firsts = tables, rows, starts, counts, firsts
        self.columns, self.offsets, self.bounds, self.total = int(columns), offsets, bounds, total

    def launches(self, capacity):
        """[(rows, partial rows)]: the rows of each launch (a slice: consecutive rows) whose partials fit `capacity`
        partial rows (None: any), and the partial rows it takes (None without offsets: rows x columns)."""
        count = int(self.starts.shape[0])
        if self.offsets is None:
            step = count if capacity is None else capacity // self.columns
            if step < 1:
                raise _too_small(self.columns)
            return [(slice(a, min(count, a + step)), None) for a in range(0, count, step)]
        found, first, used = [], 0, 0
        for r, bound in enumerate(self.bounds):
            if capacity is not None and min(used + bound, self.total) > capacity:
                if r == first:
                    raise _too_small(bound)
                found.append((slice(first, r), min(used, self.total)))
                first, used = r, 0
            used += bound
        found.append((slice(first, count), min(used, self.total)))
        return found


def _too_small(tokens):
    return ValueError(f"a row of {tokens} tokens needs more partials than the split-KV reader's workspace holds: raise "
                      "the attention plugin's workspace_mib")


class DeviceAttention:
    def __init__(self, kernels=None, mode="auto", int8_qk=False, decode="split", partitions=0, piece=0, device=None,
                 kernels_mib=128, flashinfer="off", flashinfer_mib=128, prompt_reader="prefill", reader="split"):
        """decode: what reads a step's window rows: "split" (the split-KV reader, as captured rounds read them: a
        decode row's bits the same in a step and in a round) or "flashinfer" (FlashInfer's decode and prefill over
        16-bit and FP8 pages where it serves, another reduction than a round's). prompt_reader: "prefill" (prompt rows
        through a prompt reader, FlashInfer's or crit's: fastest, another reduction than decode rows') or
        "decode" (prompt rows in the decode's reduction: a scored token's bits are the same token's served; window rows
        through the split-KV reader too, no FlashInfer). reader: which of crit's kernel structures read the pages:
        "split" (the split-KV launch, piece i of a row's positions to partition i mod P, and crit's prompt reader) or
        "flash" (the flash structures: a row's positions in contiguous splits of one chunk per pool, its columns times
        a KV head's query heads packed into tensor-core tiles, and the flash prompt chunk; partitions and piece do not
        apply; block lists stay on the split launch) or "cuda" (the same through the CUDA flash tile: cuda-window,
        cuda-fused and cuda-prefill, one chunk and block per pool, cuda_chunk(); int8_qk prompt chunks stay on the
        Triton flash prompt chunk)."""
        if decode not in ("split", "flashinfer"):
            raise ValueError(f"decode is split or flashinfer, not {decode!r}")
        if prompt_reader not in ("prefill", "decode"):
            raise ValueError(f"prompt_reader is prefill or decode, not {prompt_reader!r}")
        if reader not in ("split", "flash", "cuda"):
            raise ValueError(f"reader is split, flash or cuda, not {reader!r}")
        if prompt_reader == "decode" and int8_qk:
            raise ValueError("int8_qk is the prompt reader's (INT8 query-key products): prompt_reader decode reads "
                             "prompt rows as decode rows")
        self.kernels = kernels          # crit's paged-kv-attention module, or None
        # flash: False (reader split), True (the Triton flash structures) or "cuda" (the CUDA flash tile), as the
        # kernel's attend and prefill take it.
        self.prompt_reader, self.flash = prompt_reader, {"split": False, "flash": True, "cuda": "cuda"}[reader]
        if prompt_reader == "decode":
            decode, flashinfer = "split", "off"
        self.mode, self.int8_qk, self.decode = mode, int8_qk, decode
        self.partitions, self.piece, self.device = int(partitions), int(piece), device
        if self.piece and not (_power_of_two(self.piece) and 16 <= self.piece <= 128):
            raise ValueError(f"a piece is a power of two of 16 to 128 positions (0: the kernel's own), not {piece}")
        self._steps = {}                # what a step plans at its first layer, by kind: kind -> (key, plan)
        self._parts, self._prompt_rows, self._widths, self._warmed = {}, {}, {}, set()
        # Everything allocated now, as memory() declares it (device_memory): the split-KV reader's partials, and
        # FlashInfer's scheduling workspace and wrappers.
        self.split_workspace = self.flashinfer_workspace = None
        self._wrappers = {}
        if mode != "torch" and device is not None and str(device).startswith("cuda"):
            torch = _torch()
            if kernels is not None and kernels_mib:
                self.split_workspace = torch.empty((int(kernels_mib) << 20) // 4, dtype=torch.float32, device=device)
            if _flashinfer_serves(mode, flashinfer):
                import flashinfer as fi
                self.flashinfer_workspace = torch.empty(int(flashinfer_mib) << 20, dtype=torch.uint8, device=device)
                # One wrapper per kind (decode, each prefill tile class), planned again each step: they share the
                # scheduling workspace (they run one after another on one stream), each keeps its own int workspace.
                if decode == "flashinfer":
                    self._wrappers[True] = fi.BatchDecodeWithPagedKVCacheWrapper(self.flashinfer_workspace, "NHD",
                                                                                 use_tensor_cores=True)
                for tile in PREFILL_TILES:
                    self._wrappers[tile] = fi.BatchPrefillWithPagedKVCacheWrapper(self.flashinfer_workspace, "NHD")

    def facts(self):
        """The engine facts of the split-KV reader on its GPU (where crit's kernel loads): the tile of a window
        past 32 query rows. On GPUs whose tensor cores take four warps over 64 rows as warpgroup MMA (sm_90's wgmma,
        sm_100's and sm_110's tcgen05), another summation order than the decode tiles' mma.sync, such windows are read in
        32-row two-warp tiles, each reading the row's pages again: wide windows keep that cost there, so a column's bits
        stay the decode's."""
        if self.prompt_reader == "decode":
            found = {"prompt_reads": "in the decode's reduction (prompt_reader decode): a step's prompt rows and a "
                                     "captured chunk as window rows of up to 64 columns through the split-KV reader with "
                                     "the pool's partitions and piece, each program folding its partitions as the merge "
                                     "does (its torch fallback over the page tables the same way): a scored token's "
                                     "bits equal the same token's served (served == scored), over every page type; a "
                                     "prompt chunk's own fresh keys and values aside (attention-state's fresh_mib)",
                     "served_vs_scored": "exact"}
        else:
            found = {"prompt_reads": "through a prompt reader (prompt_reader prefill): FlashInfer's prefill or "
                                     "crit's, another reduction than decode rows' (FP32 row sums in one pass over the "
                                     "positions, where the split-KV reader sums on the tensor cores per partition and "
                                     "merges): attention outputs apart by up to 1.6e-2 (BF16 outputs; 2.4e-4 at 12K "
                                     "positions, the 27B and Nanbeige shapes), so a scored token's log-probability may "
                                     "differ from the served one's by more than 0.001 nats: prompt_reader decode for "
                                     "served == scored",
                     "served_vs_scored": "within the readers' rounding"}
        if self.flash == "cuda":
            found["reader"] = ("cuda (reader cuda): window rows (decode tokens, verify windows, trees, captured rounds, "
                               "prompt_reader decode's prompt pieces) through crit's CUDA flash tile (16 query rows a "
                               "warp, each warp's softmax in registers), a row's positions in contiguous splits of one "
                               "chunk per pool merged in split order: a column's bits the same alone, in any window, in "
                               "a step and in a round (another reduction than reader split's and flash's); block lists "
                               "through the split launch")
            if self.prompt_reader != "decode":
                found["prompt_rows"] = ("not row-invariant (reader cuda): crit's prompt reader is the CUDA flash "
                                        "prompt chunk, which splits a chunk's positions by its length, so a prompt "
                                        "token's bits follow the chunk it is read in (as FlashInfer's prefill, another "
                                        "reduction than decode rows'); reader split keeps crit's earlier prompt "
                                        "reader")
        elif self.flash:
            found["reader"] = ("flash (reader flash): window rows (decode tokens, verify windows, trees, captured rounds, "
                               "prompt_reader decode's prompt pieces) through crit's flash structures, a row's positions "
                               "in contiguous splits of one chunk per pool merged in split order: a column's bits the "
                               "same alone, in any window, in a step and in a round (another reduction than reader "
                               "split's); block lists through the split launch")
            if self.prompt_reader != "decode":
                found["prompt_rows"] = ("not row-invariant (reader flash): crit's prompt reader is the flash prompt "
                                        "chunk, which splits a chunk's positions by its length, so a prompt token's "
                                        "bits follow the chunk it is read in (as FlashInfer's prefill, another "
                                        "reduction than decode rows'); reader split keeps crit's earlier prompt "
                                        "reader")
        if self.kernels is None or self.device is None or not str(self.device).startswith("cuda"):
            return found
        torch = _torch()
        capability = torch.cuda.get_device_capability(torch.device(self.device))
        if self.flash:
            return found     # one flash tile per pool (the kernel's flash_tile), whatever the window
        rows, warps, _ = self.kernels.tile_for(64, 1, capability)
        wide = f"{rows}-row {warps}-warp tiles"
        if rows < 64:
            wide += (" (this GPU takes four warps over 64 rows as warpgroup MMA: windows past 32 query rows read their "
                     "pages once a 32-row tile)")
        found["split_kv_wide_windows"] = wide
        return found

    def close(self):
        self.split_workspace = self.flashinfer_workspace = None
        self._wrappers.clear()
        self._steps.clear()
        self._warmed.clear()

    # -- the device form -------------------------------------------------------------------------------------------------
    def forward(self, layer, queries, pool, batch, scale=None, blocks=None, block_size=None, parents=None, lse=False):
        """queries [tokens][heads][dim] of a StepBatch or RoundBatch over the pool's KV (written for this step
        already); each token sees its sequence's positions up to its own. blocks [tokens][k] KV block ids (block_size
        positions each, -1 for none): a token sees only those blocks' positions; parents [tokens]: each token's parent
        column in its row (-1 for none), a verify window's draft tree, whose columns see their ancestors and
        themselves. lse: also return each token's log-sum-exp ([tokens][heads] FP32, natural log over the scaled
        scores). Returns out [tokens][heads][value dim] in the queries' dtype (and lse)."""
        torch = _torch()
        scale = scale if scale is not None else queries.shape[-1] ** -0.5
        if blocks is not None and not block_size:
            raise ValueError("block-sparse attention takes the blocks' size in positions (block_size)")
        paged = (queries.is_cuda and self.mode != "torch" and callable(getattr(pool, "layer_pages", None))
                 and getattr(pool, "dtype", None) in kv_pages.KERNEL_KINDS)
        # prompt_reader "decode": every row over the pool's page tables in one reduction (the split-KV reader, or its
        # torch fallback over the same tables where it does not serve).
        tabled = paged or (self.prompt_reader == "decode" and callable(getattr(pool, "rows_at", None)))
        if paged and not capturing():
            self._warm(layer, queries, pool, lse)
        if batch.prompt:
            return self._prompt(layer, queries, pool, batch, scale, blocks, block_size, parents, lse, paged)
        if batch.round:
            return self._round(layer, queries, pool, batch, scale, blocks, block_size, parents, lse, paged)
        tokens, heads = queries.shape[0], queries.shape[1]
        out = queries.new_empty(tokens, heads, pool.value_dim)
        found = queries.new_empty(tokens, heads, dtype=torch.float32) if lse else None
        if not tabled:
            self._eager(layer, queries, pool, batch, range(batch.rows), scale, blocks, block_size, parents, out, found,
                        _fresh(pool, layer, tokens, blocks, parents))
            return (out, found) if lse else out
        q = self._queries(queries, pool) if paged else queries
        special = blocks is not None or parents is not None or lse
        reader = self.reader(pool, q, special) if paged else "torch"
        if getattr(pool, "step_rows", 0):
            windows = pool.windows
            if windows:
                self._step_buffer_windows(layer, q, pool, batch, windows, scale, blocks, block_size, parents, out,
                                          found, reader == "split")
            chosen = set(windows)
            rest = [r for r in range(batch.rows) if r not in chosen]
        else:
            short = [r for r in range(batch.rows) if batch.row(r)[2] <= pool.window]
            # decode "split" (the default): a step's window rows as a round reads them (the split-KV reader, else its
            # torch fallback), so a decode row's bits are the same in a step and in a round; else torch's gathered
            # SDPA.
            if short and (reader == "split" or self.decode == "split"):
                meta, extent = self._step_windows(pool, batch, short, q.device)
                self._windows(layer, q, pool, meta, scale, blocks, block_size, parents, out, found,
                              reader == "split", extent)
                rest = [r for r in range(batch.rows) if batch.row(r)[2] > pool.window]
            else:
                rest = list(range(batch.rows))
        if rest:
            self._prompts(layer, q, pool, batch, rest, scale, blocks, block_size, parents, out, found, reader)
        return (out, found) if lse else out

    def reader(self, pool, q, special=False):
        """What reads a step's window rows over this pool: "flashinfer" (its decode and prefill over 16-bit and FP8
        pages, decode "flashinfer", a pool without step buffers, no trees, block lists or log-sum-exps), "split" (crit's
        split-KV reader) or "torch" (the capture-safe fallback over the page tables)."""
        if not special and self.decode == "flashinfer" and not getattr(pool, "step_rows", 0) and \
                self._flashinfer_reads(pool, q):
            return "flashinfer"
        return "split" if self._split_reads(pool, q) else "torch"

    # -- captured rounds -------------------------------------------------------------------------------------------------
    def _round(self, layer, queries, pool, batch, scale, blocks, block_size, parents, lse, paged):
        torch = _torch()
        rows, cols = batch.rows, batch.width + 1
        tables = getattr(pool, "round_tables", None)
        if tables is None:
            raise ValueError("a round reads the KV pool's round tables (rounds:kv_cache): this pool has none")
        q = self._queries(queries, pool) if paged else queries
        # In the queries' dtype, as a step's rows are: a round's column equals the step's row bit for bit.
        out = queries.new_empty(queries.shape[0], queries.shape[1], pool.value_dim)
        found = queries.new_empty(queries.shape[0], queries.shape[1], dtype=torch.float32) if lse else None
        meta = Windows(tables, batch.slots_t, batch.starts_t[:rows], None, batch.frontier_t, cols)
        self._windows(layer, q, pool, meta, scale, blocks, block_size, parents, out, found,
                      paged and self._split_reads(pool, q))
        return (out, found) if lse else out

    # -- captured prompt chunks -----------------------------------------------------------------------------------------
    def _prompt(self, layer, queries, pool, batch, scale, blocks, block_size, parents, lse, paged):
        """A PromptBatch (prompts:attention): the bucket's tokens over the pages stage_prompt put in the pool's prompt
        table, causal from batch.frontier_t, read on the device (capture-safe, no FlashInfer plan): crit's prompt
        reader (its position read on the device), or, with block lists or without crit's kernel, the bucket as window
        rows of up to 64 columns through the split-KV reader or its torch fallback. The padding's rows see positions past the chunk, which hold nothing a sequence keeps (their own KV went to the dump page)."""
        torch = _torch()
        table = getattr(pool, "prompt_table", None)
        if table is None:
            raise ValueError("a prompt batch reads the KV pool's prompt table (prompts:kv_cache): this pool has none")
        if parents is not None:
            raise ValueError("a prompt chunk is causal: it has no draft tree")
        q = self._queries(queries, pool) if paged else queries
        out = queries.new_empty(queries.shape[0], queries.shape[1], pool.value_dim)
        found = queries.new_empty(queries.shape[0], queries.shape[1], dtype=torch.float32) if lse else None
        fresh = _fresh(pool, layer, q.shape[0], blocks)
        if fresh is not None:   # from the pool's first fresh position (the chunk past the window), on the device
            fresh = (fresh[0], fresh[1], pool.prompt_fresh_first)
        given = (fresh[0], fresh[1], batch.frontier_t, fresh[2]) if fresh is not None else None
        if blocks is None and self.prompt_reader == "decode":
            # In the decode's reduction: the bucket as window rows of up to 64 columns, through a fused split-KV launch
            # (or its torch fallback over the same tables); fresh rows from the chunk's frontier (their shift) and its
            # first fresh position.
            meta = self._prompt_windows(table, batch, q.device)
            if paged and self._split_reads(pool, q):
                self._fused(layer, q, pool, meta, scale, out, found, given)
            else:
                windows_torch(layer, q, pool, meta, scale, None, None, None, out, found,
                              None if capturing() else batch.frontier + batch.bucket, given)
            return (out, found) if lse else out
        if paged and blocks is None and self._split_reads(pool, q) and pool.value_dim == pool.head_dim:
            pages = pool.layer_pages(layer)
            scaled = len(pages) == 4
            multipliers = pool.scale_values[pool.layers[layer]] if pool.dtype == "fp8" else (1.0, 1.0)
            self.kernels.prefill(q, pages[0], pages[1], pool.dtype, table[0], batch.frontier_t, pool.page,
                                 key_scales=pages[2] if scaled else None, value_scales=pages[3] if scaled else None,
                                 multipliers=multipliers, scale=scale, out=out, lse=found,
                                 int8_qk=self.int8_qk and pool.dtype == "int8", fresh=fresh, flash=self.flash)
            return (out, found) if lse else out
        meta = self._prompt_windows(table, batch, q.device)
        extent = None if capturing() else batch.frontier + batch.bucket
        split = paged and self._split_reads(pool, q) and fresh is None
        self._windows(layer, q, pool, meta, scale, blocks, block_size, None, out, found, split, extent,
                      partitions=PIECE_PARTITIONS, fresh=given)
        return (out, found) if lse else out

    def _prompt_windows(self, table, batch, device):
        """The bucket as window rows of up to 64 columns over the prompt table, their first positions from
        batch.frontier_t on the device (made once a bucket, outside a capture)."""
        torch = _torch()
        key = (batch.bucket, str(device))
        found = self._prompt_rows.get(key)
        if found is None:
            if capturing():
                raise RuntimeError("a prompt bucket's window rows are made outside a capture: run it eagerly first")
            columns = min(64, batch.bucket)
            count = -(-batch.bucket // columns)
            starts = torch.arange(count, dtype=torch.int32, device=device) * columns
            counts = torch.tensor([min(columns, batch.bucket - r * columns) for r in range(count)], dtype=torch.int32,
                                  device=device)
            found = self._prompt_rows[key] = (torch.zeros(count, dtype=torch.int32, device=device), starts, counts,
                                              starts.long(), columns)
        rows, starts, counts, offsets, columns = found
        return Windows(table, rows, starts, counts, batch.frontier_t + offsets, columns)

    # -- window rows -----------------------------------------------------------------------------------------------------
    def _step_buffer_windows(self, layer, q, pool, batch, windows, scale, blocks, block_size, parents, out, found,
                             split):
        """A step's window rows through the pool's step buffers: in a capture every row may take up to the pool's
        window and every position its table holds (a replay's rows are not known yet), else what the rows take."""
        n = len(windows)
        if capturing():
            bounds, total, extent = [pool.window] * n, q.shape[0], None
        else:
            rows = [batch.row(r) for r in windows]
            bounds = [count for _, _, count, _ in rows]
            total, extent = sum(bounds), max(position + count for _, _, count, position in rows)
        meta = Windows(pool.d_tables, pool.d_rows[:n], pool.d_starts[:n], pool.d_counts[:n], pool.d_firsts[:n],
                       pool.window, offsets=pool.d_offsets[:n], bounds=bounds, total=total)
        self._windows(layer, q, pool, meta, scale, blocks, block_size, parents, out, found, split, extent)

    def _windows(self, layer, q, pool, meta, scale, blocks, block_size, parents, out, found, split, extent=None,
                 partitions=None, fresh=None):
        """Window rows through the split-KV reader, or the torch fallback over the same tables (extent: the positions
        the rows reach, None: every position the tables hold; fresh: a prompt chunk's own keys and values, the torch
        fallback's)."""
        if not split:
            windows_torch(layer, q, pool, meta, scale, blocks, block_size, parents, out, found, extent, fresh)
            return
        pages = pool.layer_pages(layer)
        scaled = len(pages) == 4
        multipliers = pool.scale_values[pool.layers[layer]] if pool.dtype == "fp8" else (1.0, 1.0)
        # Block-sparse rows read a few blocks each: the prompt pieces' partitions in steps and rounds too, so a decoded
        # row has the bits of the same row in a prompt. The flash structures (reader flash): the pool's chunk, its
        # splits over the launch's page tables.
        flash = self.flash if blocks is None else False
        chunk = self._flash_chunk(pool, q) if flash else None
        if flash:
            # cuda: the pool's chunk and ramp (cuda_parts); flash: the chunk's splits
            parts = (self.kernels.cuda_parts(pool.head_dim, pool.value_dim, q.shape[1], pool.kv_heads, pool.page,
                                             pool.dtype, meta.tables.shape[1] * pool.page, chunk)
                     if flash == "cuda" else self.kernels.flash_parts(meta.tables.shape[1] * pool.page, chunk))
        else:
            parts = partitions or (PIECE_PARTITIONS if blocks is not None else self._partitions(pool, q.device))
        capacity = None
        if self.split_workspace is not None:
            capacity = self.split_workspace.numel() // self.kernels.workspace_floats(1, q.shape[1], parts,
                                                                                     pool.value_dim)
        columns = self._launch_columns(q, pool, meta, flash=flash)
        for cut, tokens in meta.launches(capacity):
            # A later launch's slices start inside the tensors: copied, they keep the alignment the kernel was
            # compiled for at load (Triton specializes a pointer by its 16-byte alignment).
            part = (lambda t: t[cut]) if not cut.start else (lambda t: t[cut].clone())
            self.kernels.attend(q, pages[0], pages[1], pool.dtype, meta.tables, part(meta.rows), part(meta.starts),
                                part(meta.firsts), columns, pool.page,
                                counts=part(meta.counts) if meta.counts is not None else None,
                                offsets=part(meta.offsets) if meta.offsets is not None else None, tokens=tokens,
                                key_scales=pages[2] if scaled else None, value_scales=pages[3] if scaled else None,
                                multipliers=multipliers, scale=scale, partitions=parts, piece=self.piece or None,
                                parents=parents, blocks=blocks, block_size=block_size, out=out, lse=found,
                                workspace=self.split_workspace, flash=flash, chunk=chunk)

    def _step_windows(self, pool, batch, chosen, device):
        """The chosen short rows' metadata and extent for one step (made at its first layer, kept for the others)."""
        key = (id(pool), pool.steps, tuple(chosen))
        cached = self._steps.get("windows")
        if cached is not None and cached[0] == key:
            return cached[1]
        rows = [batch.row(r) for r in chosen]
        found = (_windows_of(pool, rows, device), max(position + count for _, _, count, position in rows))
        self._steps["windows"] = (key, found)
        return found

    # -- prompt rows -----------------------------------------------------------------------------------------------------
    def _prompts(self, layer, q, pool, batch, rows, scale, blocks, block_size, parents, out, found, reader):
        fresh = _fresh(pool, layer, q.shape[0], blocks, parents)
        if blocks is None and parents is None and self.prompt_reader == "decode":
            self._decode_rows(layer, q, pool, batch, rows, scale, out, found, fresh, reader == "split")
            return
        if fresh is not None:
            # Prompt rows (past the window): their own chunk's keys and values as the projections gave them, the
            # earlier positions from the pages. The other rows read the pages.
            own = [r for r in rows if batch.row(r)[2] > pool.window]
            if self._split_reads(pool, q) and q.shape[-1] == pool.value_dim:
                self._prefill_rows(layer, q, pool, batch, own, scale, out, found, fresh)
            else:
                self._eager(layer, q, pool, batch, own, scale, None, None, None, out, found, fresh)
            rows = [r for r in rows if batch.row(r)[2] <= pool.window]
            if not rows:
                return
        # crit's prompt reader reads every page kind its split-KV reader does (16-bit, FP8, fp8:row, INT8), with or
        # without log-sum-exps.
        prefill = self._split_reads(pool, q) and q.shape[-1] == pool.value_dim
        if blocks is not None or parents is not None or (found is not None and not prefill):
            if blocks is not None and self._split_reads(pool, q) and parents is None:
                self._sparse_pieces(layer, q, pool, batch, rows, scale, blocks, block_size, out, found)
                return
            self._eager(layer, q, pool, batch, rows, scale, blocks, block_size, parents, out, found)
            return
        if self._flashinfer_reads(pool, q) and found is None:   # option flashinfer "auto" only (a yardstick)
            self._flashinfer_rows(layer, q, pool, batch, rows, scale, out)
            return
        if prefill:
            self._prefill_rows(layer, q, pool, batch, rows, scale, out, found, None)
            return
        self._eager(layer, q, pool, batch, rows, scale, None, None, None, out, found)

    def _decode_rows(self, layer, q, pool, batch, rows, scale, out, found, fresh, split):
        """Prompt rows in the decode's reduction (prompt_reader "decode"): pieces of up to 64 tokens as window rows,
        through a fused split-KV launch with the pool's partitions and piece (each program folding its partitions as
        the merge does: no partials) where it serves, else the torch fallback over the same page tables; a prompt
        token's bits are those of the same position decoded. With `fresh` a prompt row's (past the window) own
        positions from the step's keys and values as write() got them."""
        torch = _torch()
        pieces, shifts, firsts = [], [], []
        for r in rows:
            sequence, first, count, position = batch.row(r)
            own = fresh is not None and count > pool.window
            for at in range(0, count, 64):
                pieces.append((sequence, first + at, min(64, count - at), position + at))
                shifts.append(position - first)              # a fresh row's index: its position - shift
                firsts.append(position if own else 1 << 62)  # its first position read fresh (none: past every one)
        meta = _windows_of(pool, pieces, q.device)
        given = None
        if fresh is not None:
            # Host values cross with the metadata (a step's rows are planned on the host).
            marks = torch.tensor([shifts, firsts], dtype=torch.int64).to(q.device)
            given = (fresh[0], fresh[1], marks[0], marks[1])
        if split:
            self._fused(layer, q, pool, meta, scale, out, found, given)
        else:
            windows_torch(layer, q, pool, meta, scale, None, None, None, out, found,
                          max(position + count for _, _, count, position in pieces), given)

    def _warm(self, layer, queries, pool, lse):
        """Compile at load what a step's window rows launch. The first eager call over a pool and query geometry (a
        prompt chunk's or a round's warm-up, or Core's first request: an engine makes them all at load) launches the
        split-KV reader once for each query tile a window row of 1 .. pool.window columns gets (_launch_columns),
        and with prompt_reader decode each tile a prompt piece of 1 .. 64 tokens gets in its fused launch, one row
        over the dump page. A request's window rows of any width and length then compile nothing (draft trees
        and block lists aside: their launches compile at their first step)."""
        q = self._queries(queries, pool)
        key = (str(q.device), tuple(q.shape[1:]), q.dtype, id(pool), bool(lse))
        if key in self._warmed or not self._split_reads(pool, q):
            return
        self._warmed.add(key)
        torch = _torch()
        device = q.device
        one = lambda value, dtype: torch.tensor([value], dtype=dtype, device=device)
        q = q[:1]
        out = q.new_empty(1, q.shape[1], pool.value_dim)
        found = q.new_empty(1, q.shape[1], dtype=torch.float32) if lse else None
        scale = q.shape[-1] ** -0.5
        # The tables a step's rows launch over: _windows_of's (TABLE_WIDTH pages a row, or a multiple), and the
        # pool's step buffers' where it keeps them (their row 0's positions read, nothing written).
        tables = [torch.full((1, TABLE_WIDTH), pool.dump, dtype=torch.int32, device=device),
                  torch.full((1, 2 * TABLE_WIDTH), pool.dump, dtype=torch.int32, device=device)]
        if getattr(pool, "step_rows", 0) and getattr(pool, "d_tables", None) is not None:
            tables.append(pool.d_tables)

        def rows(table, columns):
            return Windows(table, one(0, torch.int32), one(0, torch.int32), one(1, torch.int32), one(0, torch.int64),
                           columns, offsets=one(0, torch.int32), bounds=[1], total=1)

        widest = 1 << (max(1, min(64, int(pool.window))) - 1).bit_length()
        for table in tables:
            launched = set()
            for columns in (1 << i for i in range(widest.bit_length())):
                meta = rows(table, columns)
                width = self._launch_columns(q, pool, meta, flash=self.flash)
                if width not in launched:
                    launched.add(width)
                    self._windows(layer, q, pool, meta, scale, None, None, None, out, found, True)
            if self.prompt_reader == "decode":
                launched = set()
                for columns in (1 << i for i in range(7)):
                    meta = rows(table, columns)
                    width = self._launch_columns(q, pool, meta, fused=True, flash=self.flash)
                    if width not in launched:
                        launched.add(width)
                        self._fused(layer, q, pool, meta, scale, out, found, None)

    def _launch_columns(self, q, pool, meta, fused=False, flash=False):
        """The width a launch of window rows takes: rows packed by their tokens (meta.offsets) take their widest row's
        power of two raised to the columns of the query tile it gets (tile_for), so every width up to those shares
        one kernel variant and a request whose rows are wider or narrower than any before compiles nothing after
        load. The tile, the programs and a column's bits are those of the narrower launch (the columns past a row's
        count do nothing). Rows laid out by columns (no offsets) keep their own width, and so do flash launches (one
        tile per pool whatever the width)."""
        if meta.offsets is None or flash:
            return meta.columns
        key = (str(q.device), q.shape[1], pool.kv_heads, meta.columns, fused)
        found = self._widths.get(key)
        if found is None:
            torch = _torch()
            group = q.shape[1] // pool.kv_heads
            rows = self.kernels.tile_for(meta.columns, group, torch.cuda.get_device_capability(q.device), fused)[0]
            found = self._widths[key] = min(self.kernels.MAX_COLUMNS, max(meta.columns, rows // group))
        return found

    def _fused(self, layer, q, pool, meta, scale, out, found, fresh):
        """Window rows (Windows) through one fused split-KV launch with the pool's partitions and piece."""
        pages = pool.layer_pages(layer)
        scaled = len(pages) == 4
        multipliers = pool.scale_values[pool.layers[layer]] if pool.dtype == "fp8" else (1.0, 1.0)
        flash = {"flash": self.flash, "chunk": self._flash_chunk(pool, q)} if self.flash else {}
        self.kernels.attend(q, pages[0], pages[1], pool.dtype, meta.tables, meta.rows, meta.starts, meta.firsts,
                            self._launch_columns(q, pool, meta, fused=True, flash=self.flash), pool.page,
                            counts=meta.counts, key_scales=pages[2] if scaled else None,
                            value_scales=pages[3] if scaled else None, multipliers=multipliers, scale=scale,
                            partitions=self._partitions(pool, q.device), piece=self.piece or None, out=out, lse=found,
                            fused=True, fresh=fresh, **flash)

    def _prefill_rows(self, layer, q, pool, batch, rows, scale, out, found, fresh):
        """Prompt rows through crit's prompt reader, one row a launch: with `fresh` (the step's keys and values
        as write() got them) each row's own positions from them, the earlier ones from the pages."""
        pages = pool.layer_pages(layer)
        scaled = len(pages) == 4
        multipliers = pool.scale_values[pool.layers[layer]] if pool.dtype == "fp8" else (1.0, 1.0)
        for r in rows:
            sequence, first, count, position = batch.row(r)
            span = slice(first, first + count)
            self.kernels.prefill(q[span], pages[0], pages[1], pool.dtype, pool.device_tables.get(sequence), position,
                                 pool.page, key_scales=pages[2] if scaled else None,
                                 value_scales=pages[3] if scaled else None, multipliers=multipliers, scale=scale,
                                 out=out[span], lse=found[span] if found is not None else None,
                                 int8_qk=self.int8_qk and pool.dtype == "int8",
                                 fresh=(fresh[0][span], fresh[1][span]) if fresh is not None else None,
                                 flash=self.flash)

    def _sparse_pieces(self, layer, q, pool, batch, rows, scale, blocks, block_size, out, found):
        """Prompt rows with block lists through the sparse reader: pieces of up to PIECE_TOKENS tokens, PIECE_ROWS
        pieces a launch, few partitions (each token is a program of its own already)."""
        pieces = []
        for r in rows:
            sequence, first, count, position = batch.row(r)
            for at in range(0, count, PIECE_TOKENS):
                pieces.append((sequence, first + at, min(PIECE_TOKENS, count - at), position + at))
        for at in range(0, len(pieces), PIECE_ROWS):
            meta = _windows_of(pool, pieces[at:at + PIECE_ROWS], q.device)
            self._windows(layer, q, pool, meta, scale, blocks, block_size, None, out, found, True,
                          partitions=PIECE_PARTITIONS)

    def _eager(self, layer, q, pool, batch, rows, scale, blocks, block_size, parents, out, found, fresh=None):
        """Rows in torch from the pool's gathered KV: SDPA, or the masked softmax for trees, blocks and lse; with
        `fresh` (the step's keys and values as write() got them) each prompt row's own positions from them."""
        if blocks is None and parents is None and found is None:
            _gathered(layer, q, pool, batch, rows, scale, out, fresh)
            return
        for r in rows:
            sequence, first, count, position = batch.row(r)
            keys, values = _with_fresh(pool, pool.gather(layer, sequence, position + count), fresh, first, count)
            span = slice(first, first + count)
            windows_dense(q[span], keys, values, position, scale, blocks[span] if blocks is not None else None,
                          block_size, parents[span] if parents is not None else None, out[span],
                          found[span] if found is not None else None)

    # -- FlashInfer ------------------------------------------------------------------------------------------------------
    def _flashinfer_rows(self, layer, q, pool, batch, rows, scale, out):
        plans = self._flashinfer_plan(pool, batch, rows, q, scale)
        pages = pool.layer_pages(layer)
        k_scale, v_scale = pool.scale_values[pool.layers[layer]] if pool.dtype == "fp8" else (None, None)
        for index, wrapper in plans:
            got = wrapper.run(_take(q, index).contiguous(), pages[:2], k_scale=k_scale, v_scale=v_scale)
            _place(out, got, index)

    def _flashinfer_plan(self, pool, batch, chosen, q, scale):
        """The step's FlashInfer plans for the chosen rows, made at its first layer (a step is one reserve())."""
        torch = _torch()
        key = (id(pool), pool.steps, q.shape[1], tuple(chosen))
        cached = self._steps.get("flashinfer")
        if cached is not None and cached[0] == key:
            return cached[1]
        if capturing():
            raise RuntimeError("FlashInfer plans a step on the host: a captured step reads its window rows through "
                               "the split-KV reader (a pool with step buffers, or decode \"split\") and keeps prompt "
                               "rows out of the capture")
        rows = {r: batch.row(r) for r in chosen}
        one = [r for r in chosen if rows[r][2] == 1] if True in self._wrappers else []
        # FlashInfer's prefill picks its query tile from the average (query rows x query heads per KV head) of the
        # rows it plans together: rows planned in groups that would each get the same tile alone keep their own bits.
        group_size = q.shape[1] // pool.kv_heads
        classes = {}
        for r in chosen:
            if r not in one:
                classes.setdefault(_prefill_tile(rows[r][2] * group_size, pool.head_dim), []).append(r)
        plans = []
        for group, kind in [(one, True)] + [(classes[c], c) for c in sorted(classes)]:
            if not group:
                continue
            index = [t for r in group for t in range(rows[r][1], rows[r][1] + rows[r][2])]
            index = (index[0], index[0] + len(index)) if index == list(range(index[0], index[0] + len(index))) else \
                torch.tensor(index, dtype=torch.int64, device=q.device)
            tables = [(rows[r][3] + rows[r][2], pool.table(rows[r][0])[:math.ceil((rows[r][3] + rows[r][2]) / pool.page)])
                      for r in group]
            plans.append((index, self._flashinfer(pool, group, tables, rows, q.shape[1], q.dtype, scale, q.device,
                                                  kind)))
        self._steps["flashinfer"] = (key, plans)
        return plans

    def _flashinfer(self, pool, group, found, rows, heads, dtype, scale, device, kind):
        """The wrapper of `kind` (True: decode, else a prefill tile class) planned for the group's rows."""
        torch = _torch()
        wrapper = self._wrappers[kind]
        indptr, indices, last = [0], [], []
        for length, table in found:
            indices += table
            indptr.append(len(indices))
            last.append(length - (len(table) - 1) * pool.page)
        indptr = torch.tensor(indptr, dtype=torch.int32)
        last = torch.tensor(last, dtype=torch.int32)
        indices = torch.tensor(indices, dtype=torch.int32).to(device)
        storage = pool.k.dtype
        if kind is True:
            wrapper.plan(indptr, indices, last, heads, pool.kv_heads, pool.head_dim, pool.page,
                         pos_encoding_mode="NONE", q_data_type=dtype, kv_data_type=storage, sm_scale=scale,
                         fixed_split_size=max(1, SPLIT_TOKENS // pool.page))
        else:
            qo = [0]
            for r in group:
                qo.append(qo[-1] + rows[r][2])
            # Prompt rows bring their own parallelism: no KV split (batch invariant, and no partial-result buffers
            # that outgrow the workspace on long chunks).
            wrapper.plan(torch.tensor(qo, dtype=torch.int32), indptr, indices, last, heads, pool.kv_heads,
                         pool.head_dim, pool.page, causal=True, pos_encoding_mode="NONE", sm_scale=scale,
                         q_data_type=dtype, kv_data_type=storage, disable_split_kv=True)
        return wrapper

    # -- what reads what -------------------------------------------------------------------------------------------------
    def _queries(self, queries, pool):
        torch = _torch()
        dtype = torch.float16 if pool.dtype == "fp16" else torch.bfloat16
        return queries if queries.dtype == dtype and queries.stride(-1) == 1 else queries.to(dtype).contiguous()

    def _split_reads(self, pool, q):
        """crit's split-KV reader reads this pool."""
        return (self.kernels is not None and q.is_cuda and pool.dtype in kv_pages.KERNEL_KINDS
                and _power_of_two(pool.head_dim) and _power_of_two(pool.value_dim)
                and 16 <= pool.head_dim <= 256 and 16 <= pool.value_dim <= 256 and q.shape[1] % pool.kv_heads == 0)

    def _flashinfer_reads(self, pool, q):
        """FlashInfer serves (its wrappers were made) and reads this pool's pages."""
        if not self._wrappers or pool.dtype not in ("bf16", "fp16", "fp8") or pool.head_dim not in (64, 128, 256) \
                or pool.value_dim != pool.head_dim:
            return False
        return True

    def _flash_chunk(self, pool, q):
        """Positions a split of every flash launch over a pool (reader flash or cuda): the kernel's for the pool's
        geometry (its tuning's, else its rule: flash_chunk(), cuda_chunk()), fixed per pool, never by the batch."""
        key = (self.flash, pool.kv_heads, pool.dtype, pool.head_dim, pool.value_dim, q.shape[1], pool.page)
        if key not in self._parts:
            chunks = self.kernels.cuda_chunk if self.flash == "cuda" else self.kernels.flash_chunk
            self._parts[key] = chunks(pool.head_dim, pool.value_dim, q.shape[1], pool.kv_heads, pool.page, pool.dtype)
        return self._parts[key]

    def _partitions(self, pool, device):
        """The KV partitions of every split-KV launch over a pool: fixed per pool and device, never by the batch (the
        option's, else the kernel's for the pool's geometry: its tuning's, else its rule)."""
        key = (str(device), pool.kv_heads, pool.dtype, pool.head_dim, pool.value_dim)
        if key not in self._parts:
            self._parts[key] = self.partitions or self.kernels.partitions(device, pool.kv_heads, pool.dtype,
                                                                          pool.head_dim, pool.value_dim)
        return self._parts[key]


# -- torch -------------------------------------------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _power_of_two(n):
    return n > 0 and n & (n - 1) == 0


_FLASHINFER = []


def _flashinfer_installed():
    if not _FLASHINFER:
        try:
            import flashinfer  # noqa: F401
            _FLASHINFER.append(True)
        except ImportError:
            _FLASHINFER.append(False)
    return _FLASHINFER[0]


# The page tables of a step's window rows are this many pages wide or a multiple of it: a launch compiled at load
# reads them whatever their rows' lengths (the kernel specializes a stride of 1 and multiples of 16 apart).
TABLE_WIDTH = 16


def _windows_of(pool, rows, device):
    """Windows of host rows (sequence, first token, count, first position), their tables and metadata in one copy to
    the device each, their partials packed by their tokens. Not under a capture (host values)."""
    torch = _torch()
    if capturing():
        raise RuntimeError("window rows' metadata crosses from the host: a captured step reads its window rows "
                           "through the pool's step buffers, a captured round through its round tables")
    tables = [pool.table(sequence)[:math.ceil((position + count) / pool.page)] for sequence, _, count, position in rows]
    # The tables' width in multiples of TABLE_WIDTH (the dump page past each row's own): every launch's table stride
    # falls in one of the kernel's specializations, whatever the rows' lengths (DeviceAttention._warm compiles it).
    grid = torch.full((len(rows), -(-max(len(t) for t in tables) // TABLE_WIDTH) * TABLE_WIDTH), pool.dump,
                      dtype=torch.int32)
    for i, t in enumerate(tables):
        grid[i, :len(t)] = torch.tensor(t, dtype=torch.int32)
    counts = [count for _, _, count, _ in rows]
    offsets = [sum(counts[:i]) for i in range(len(rows))]
    meta = torch.tensor([[first, count, position, offset] for (_, first, count, position), offset in zip(rows, offsets)],
                        dtype=torch.int64)
    grid, meta = grid.to(device), meta.to(device)
    widest = max(counts)
    # Launch widths in powers of two (fewer kernel variants); a column's bits do not depend on its launch's width.
    return Windows(grid, torch.arange(len(rows), dtype=torch.int32, device=device), meta[:, 0].to(torch.int32),
                   meta[:, 1].to(torch.int32), meta[:, 2].contiguous(), 1 << (widest - 1).bit_length(),
                   offsets=meta[:, 3].to(torch.int32), bounds=counts, total=sum(counts))


def _prefill_tile(packed, head_dim):
    """The query tile FlashInfer's prefill plan gives rows of `packed` query rows (tokens x query heads per KV head)
    on average: 128 past 64 (head dims under 256), 64 past 16, else 16."""
    if packed > 64 and head_dim < 256:
        return 128
    return 64 if packed > 16 else 16


def _take(q, index):
    return q[index[0]:index[1]] if isinstance(index, tuple) else q.index_select(0, index)


def _place(out, found, index):
    if isinstance(index, tuple):
        out[index[0]:index[1]] = found
    else:
        out.index_copy_(0, index, found)


def _fresh(pool, layer, tokens, blocks=None, parents=None):
    """The step's keys and values as write() got them, where the pool keeps them (attention-state's FP8 and INT8
    pages, a prompt step) and the step's prompt rows read them (not with block lists or a draft tree), else None."""
    if blocks is not None or parents is not None:
        return None
    read = getattr(pool, "fresh_rows", None)
    return read(layer, tokens) if callable(read) else None


def _with_fresh(pool, gathered, fresh, first, count):
    """A row's gathered keys and values, a prompt row's (past the pool's window) with its own last `count` positions
    from `fresh` (tokens first ..)."""
    if fresh is None or count <= pool.window:
        return gathered
    keys, values = gathered
    keys, values = keys.clone(), values.clone()
    keys[-count:] = fresh[0][first:first + count].to(keys.dtype)
    values[-count:] = fresh[1][first:first + count].to(values.dtype)
    return keys, values


def _gathered(layer, queries, pool, batch, rows, scale, out, fresh=None):
    """Each row's KV gathered from the pool (a prompt row's own positions from `fresh` where given), multiplied by
    torch's SDPA."""
    torch = _torch()
    heads = queries.shape[1]
    for r in rows:
        sequence, first, count, position = batch.row(r)
        keys, values = _with_fresh(pool, pool.gather(layer, sequence, position + count), fresh, first, count)
        q = queries[first:first + count].transpose(0, 1).unsqueeze(0).to(keys.dtype)
        k, v = keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0)
        # One token sees every position held (no mask); a chunk from position 0 is plain causal; only a later chunk
        # needs the offset mask. Without a mask torch's fused kernels serve the row.
        mask, causal = None, count > 1 and position == 0
        if count > 1 and position > 0:
            seen = torch.arange(position + count, device=queries.device)
            mask = seen[None, :] <= (position + torch.arange(count, device=queries.device))[:, None]
        found = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                                                 scale=scale, enable_gqa=heads != keys.shape[1])
        out[first:first + count] = found[0].transpose(0, 1).to(out.dtype)


def ancestors(parents, live):
    """[rows][columns] parent columns (-1 for none) -> each column's ancestors and itself as a 64-bit mask (bit j:
    column j), by pointer jumping on the device (capture-safe)."""
    torch = _torch()
    rows, cols = parents.shape
    column = torch.arange(cols, device=parents.device)
    one = torch.ones((), dtype=torch.int64, device=parents.device)
    anc = torch.where(live, one << column[None, :], 0)
    par = torch.where(live, parents.long(), -1)
    for _ in range(cols):
        has = par >= 0
        anc = anc | torch.where(has, one << par.clamp(min=0), 0)
        par = torch.where(has, parents.long().gather(1, par.clamp(min=0)), -1)
    return anc


def _visible(n, first, counts, columns, anc, member):
    """[rows][columns][positions]: what each column sees of positions n ([rows][positions] or [positions])."""
    torch = _torch()
    n = n if n.dim() == 2 else n[None, :]
    col = torch.arange(columns, device=n.device)
    held = n < (first + counts)[:, None]                                       # [R, N]
    if anc is None:
        seen = n[:, None, :] <= (first[:, None] + col[None, :])[:, :, None]   # [R, C, N]
    else:
        w = n - first[:, None]
        inside = (w >= 0) & (w < counts[:, None])
        bit = (anc[:, :, None] >> w.clamp(0, 63)[:, None, :]) & 1
        seen = (n < first[:, None])[:, None, :] | (inside[:, None, :] & (bit != 0))
    seen = seen & held[:, None, :]
    if member is not None:
        seen = seen & member
    return seen


def windows_torch(layer, q, pool, meta, scale, blocks, block_size, parents, out, found, extent=None, fresh=None):
    """Window rows (Windows) in FP32 torch over the pool's page tables, a masked online softmax over passes of CHUNK
    positions up to `extent` (the positions the rows reach, a host number; None: every position the tables hold, as a
    capture needs): no host work, the plain arithmetic the split-KV reader is tested against. fresh: (keys, values,
    shift, first) of prompt chunks: a row's positions from first on read from keys[position - shift] instead of the
    pages (shift and first int64 on the device, one a row or one for every row)."""
    torch = _torch()
    device = q.device
    R, C = int(meta.starts.shape[0]), meta.columns
    hq, d = q.shape[1], q.shape[2]
    hkv, dv = pool.kv_heads, pool.value_dim
    g = hq // hkv
    col = torch.arange(C, device=device)
    counts = meta.counts.long().clamp(max=C) if meta.counts is not None else \
        torch.full((R,), C, dtype=torch.int64, device=device)
    first = meta.firsts.long()
    live = col[None, :] < counts[:, None]
    tokens = q.shape[0]
    tok = torch.where(live, meta.starts.long()[:, None] + col[None, :], tokens)            # dead -> a scratch row
    qq = torch.cat([q, q.new_zeros(1, hq, d)]).index_select(0, tok.reshape(-1)).float().view(R, C, hkv, g, d)
    anc = None
    if parents is not None:
        flat = torch.cat([parents.reshape(-1).long(), parents.new_full((1,), -1).long()])
        anc = ancestors(flat.index_select(0, tok.reshape(-1)).view(R, C), live)
    tables = meta.tables.index_select(0, meta.rows.long())
    width = tables.shape[1]
    if extent is None or capturing():
        extent = width * pool.page
    extent = max(CHUNK, math.ceil(min(extent, width * pool.page) / CHUNK) * CHUNK)
    bitmap = None
    if blocks is not None:
        nblk = math.ceil(extent / block_size)
        listed = torch.cat([blocks, blocks.new_full((1, blocks.shape[1]), -1)]).index_select(0, tok.reshape(-1)).long()
        bitmap = torch.zeros(R * C, nblk + 1, dtype=torch.bool, device=device).view(R, C, nblk + 1)
        bitmap.view(R * C, nblk + 1).scatter_(1, torch.where((listed >= 0) & (listed < nblk), listed + 1, 0), True)
    outs, sums = [], []
    # Row by row and column by column, each pass the same shapes whatever the rows and columns beside it: a column's
    # bits are its own. A pass past a column's positions sees nothing and leaves its sums as they were.
    for r in range(R):
        m = [torch.full((hkv, g), float("-inf"), device=device) for _ in range(C)]
        l = [torch.zeros(hkv, g, device=device) for _ in range(C)]
        acc = [torch.zeros(hkv, g, dv, device=device) for _ in range(C)]
        for s0 in range(0, extent, CHUNK):
            n = torch.arange(s0, s0 + CHUNK, device=device)
            page = tables[r, (n // pool.page).clamp(max=width - 1)]
            keys, values = pool.rows_at(layer, page.long() * pool.page + n % pool.page)        # [N, hkv, d]
            if fresh is not None:
                own = r if fresh[2].numel() > 1 else 0
                at = n - fresh[2][own]
                inside = ((n >= fresh[3][own]) & (at >= 0) & (at < fresh[0].shape[0]))[:, None, None]
                at = at.clamp(0, fresh[0].shape[0] - 1)
                keys = torch.where(inside, fresh[0].index_select(0, at).float(), keys)
                values = torch.where(inside, fresh[1].index_select(0, at).float(), values)
            member = bitmap[r][:, 1 + n // block_size][None] if bitmap is not None else None
            seen = _visible(n, first[r:r + 1], counts[r:r + 1], C, anc[r:r + 1] if anc is not None else None,
                            member)[0] & live[r][:, None]
            for c in range(C):
                s = torch.einsum("kgd,nkd->kgn", qq[r, c], keys) * scale
                s = s.masked_fill(~seen[c][None, None, :], float("-inf"))
                top = torch.maximum(m[c], s.amax(-1))
                sub = torch.where(top == float("-inf"), 0.0, top)
                p = torch.exp(s - sub[..., None])
                alpha = torch.exp(m[c] - sub)
                l[c] = l[c] * alpha + p.sum(-1)
                acc[c] = acc[c] * alpha[..., None] + torch.einsum("kgn,nkd->kgd", p, values)
                m[c] = top
        for c in range(C):
            outs.append(torch.where(l[c][..., None] > 0, acc[c] / l[c].clamp(min=1e-30)[..., None], 0.0))
            sums.append(torch.where(l[c] > 0, m[c] + torch.log(l[c].clamp(min=1e-30)), float("-inf")))
    o = torch.stack(outs).view(R * C, hq, dv)
    index = tok.reshape(-1)
    target = torch.cat([out, out.new_zeros(1, hq, dv)])
    target.index_copy_(0, index, o.to(out.dtype))
    out.copy_(target[:tokens])
    if found is not None:
        target = torch.cat([found, found.new_zeros(1, hq)])
        target.index_copy_(0, index, torch.stack(sums).view(R * C, hq))
        found.copy_(target[:tokens])


def windows_dense(q, keys, values, position, scale, blocks, block_size, parents, out, found):
    """One row of tokens at positions position.. over its gathered keys [n][kv_heads][dim] and values, in FP32: causal,
    or within the row's draft tree (parents), and within each token's listed blocks."""
    torch = _torch()
    count, hq, d = q.shape
    hkv = keys.shape[1]
    g = hq // hkv
    n = torch.arange(keys.shape[0], device=q.device)
    first = torch.tensor([position], device=q.device)
    counts = torch.tensor([count], device=q.device)
    anc = None
    if parents is not None:
        anc = ancestors(parents.reshape(1, count), torch.ones(1, count, dtype=torch.bool, device=q.device))
    member = None
    if blocks is not None:
        listed = blocks.long()
        member = ((n // block_size)[None, None, :] == listed[:, :, None]).any(1)[None]     # [1, count, n]
    seen = _visible(n, first, counts, count, anc, member)[0]                               # [count, n]
    qq = q.float().view(count, hkv, g, d)
    s = torch.einsum("ckgd,nkd->ckgn", qq, keys.float()) * scale
    s = s.masked_fill(~seen[:, None, None, :], float("-inf"))
    top = s.amax(-1)
    sub = torch.where(top == float("-inf"), 0.0, top)
    p = torch.exp(s - sub[..., None])
    total = p.sum(-1)
    o = torch.einsum("ckgn,nkd->ckgd", p, values.float()) / total.clamp(min=1e-30)[..., None]
    out.copy_(torch.where(total[..., None] > 0, o, 0.0).view(count, hq, -1).to(out.dtype))
    if found is not None:
        found.copy_(torch.where(total > 0, top + torch.log(total.clamp(min=1e-30)), float("-inf")).view(count, hq))
