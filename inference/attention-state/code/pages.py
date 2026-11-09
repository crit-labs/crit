"""Paged KV of many sequences on a device, in torch: the kv_cache slot's device form (crit_plugins.device).

A pool keeps every layer's keys and values in pages of `page` tokens ([layers][pages][page][kv_heads][dim]) and each
sequence's page table. reserve(batch) gives each of a step's tokens its slot (page * page_size + offset) at its
position, taking free pages as a sequence grows and copying a page a fork still shares before it is written
(copy on write), so a forked prefix stays shared until one of them diverges. write() stores a layer's keys and values
at their slots, keep() sets a sequence's valid length after a step (a verify window's accepted rows kept, the rest
rolled back) and frees the pages past it, and gather() returns a sequence's first n positions as contiguous rows for an
attention kernel that reads them so.

Page types: BF16, FP16, FP32; FP8 E4M3 with the layer's key and value scales ("fp8", stored = value / scale); FP8 E4M3
with one FP32 scale per position and KV head ("fp8:row", its absolute maximum over 448, the quotient rounded to the
nearest E4M3 value: fp8_rows); INT8 with one FP32 scale per position and KV head ("int8", its absolute maximum over 127,
rounded to the nearest code, ties to even: int8_rows). A pool given the registry's paged-kv-attention kernels writes
through them (one launch for keys and values, the same codes and scales as the torch rules here), else in torch.

Captured decode rounds (crit_plugins.device DEVICE_EXTENSIONS "rounds:kv_cache"): bind(slot, sequence, capacity) takes
the sequence's pages up to `capacity` positions and puts its page table in round_tables[slot], a device tensor at a
fixed address; reserve(RoundBatch) then computes every token's slot on the device from those tables and the batch's
positions, write() stores there, and commit_round(view) sets each slot's valid length to the round's frontier on the
device (round_lengths; a held row, n_emitted 0, keeps the length start_round(view) found). A token no bound sequence
holds (an unbound slot, a position past the capacity) and a write's slot below 0 land on the dump page, one page past
the sequences' (index `pages`), which no table holds for a sequence and no borrower is lent; every unused entry of a
round's or a step's page table points there. unbind(slot) reads the length back and frees the pages past it.

Every sequence's arithmetic depends only on its own pages, so a step's rows decode the same batched as alone.

Lending: the pool's page tensors and `spare` bytes after them are one allocation (`memory`), pages are taken lowest
first, and every page at or past the floor (one past the highest page in use) holds no KV. A device expert cache
borrows that memory (lendable(): the allocation, each page tensor's place in it, the spare bytes, the floor) and hears of
the floor through on_reserve(callback): before a page at or past the bound it last returned is handed out, and after the
floor falls to its low bound, so it reclaims a slot before the pages under it are written.
"""
from crit_plugins.device import capturing
import heapq
import math

DTYPES = ("bf16", "fp16", "fp32", "fp8", "fp8:row", "int8")
# Page types whose codes carry an FP32 scale per position and KV head beside them.
ROW_SCALED = ("fp8:row", "int8")
# Page types the registry's paged-kv-attention kernels read and write (FP32 pages are torch's).
KERNEL_KINDS = ("bf16", "fp16", "fp8", "fp8:row", "int8")
# Page types that store keys and values at a lower precision than the projections gave them: a prompt chunk's own
# keys and values are kept fresh beside them for its attention (fresh_rows).
QUANTIZED = ("fp8", "fp8:row", "int8")


class FreshBuffer:
    """A prompt chunk's keys and values as the projections gave them, one buffer for a plugin's pools: `data` the
    uint8 device buffer, `tag` (pool, layer, tokens, reserve) of the write that holds it now, so a pool reads it only
    while its own write of that layer and step is the last one."""

    def __init__(self, data):
        self.data, self.tag = data, None


class KVPool:
    """window: the most tokens of a row read as a window row (decode tokens, verify windows, short prompt pieces: the
    attention plugins' split-KV reader); a longer row is a prompt row. step_rows > 0 keeps a step's buffers at fixed
    addresses (step buffers): reserve() writes the slots of up to step_tokens tokens, and the first token, column
    count, first position, first partial row and page table of up to step_rows window rows (`windows`, in row order),
    into the same device tensors every step, so a model can capture a decode step in a CUDA graph and replay it after
    the next reserve(). round_rows: the round slots bind() may use (DEVICE_EXTENSIONS "rounds:kv_cache"). kernels:
    the registry's paged-kv-attention module, or None (torch)."""

    def __init__(self, layers, kv_heads, head_dim, tokens, device, dtype="bf16", *, page=16, value_dim=None,
                 step_rows=0, step_tokens=0, window=16, round_rows=8, kernels=None, spare=0, fresh=None):
        """spare: bytes after the pages, in the same allocation, that no page uses: a borrower's own (lendable()).
        fresh: a FreshBuffer (the kv_cache plugin's, declared in its memory()) a pool of FP8 or INT8 pages keeps a
        prompt chunk's keys and values in, in the compute dtype, as write() was given them (fresh_rows)."""
        import torch
        if dtype not in DTYPES:
            raise ValueError(f"a KV pool stores {', '.join(DTYPES)}, not {dtype!r}")
        if kv_heads < 1 or head_dim < 1 or tokens < 1 or page < 1 or spare < 0 or window < 1:
            raise ValueError("a KV pool has heads, a head dimension, tokens, a page size and a window of a token or more")
        self.torch = torch
        self.layers = {layer: i for i, layer in enumerate(layers)}
        # value_dim 0: a pool of keys only (an indexer's keys, a per-block row cache).
        self.kv_heads, self.head_dim = kv_heads, head_dim
        self.value_dim = head_dim if value_dim is None else int(value_dim)
        self.page, self.device, self.dtype = page, torch.device(device), dtype
        self.storage = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
                        "fp8": torch.float8_e4m3fn, "fp8:row": torch.float8_e4m3fn, "int8": torch.int8}[dtype]
        self.compute = torch.float32 if dtype == "fp32" else (torch.float16 if dtype == "fp16" else torch.bfloat16)
        self.pages = math.ceil(tokens / page)
        # One allocation: the page tensors (keys, values, a row-scaled pool's scales), then `spare` bytes for a borrower.
        # Each tensor holds the sequences' pages and the dump page after them (index `pages`), which lies past every
        # region lendable() gives.
        n, stored = len(self.layers), self.pages + 1
        self.dump, self.stored_slots = self.pages, stored * page    # the dump page; the slots each layer stores
        shapes = [((n, stored, page, kv_heads, head_dim), self.storage),
                  ((n, stored, page, kv_heads, self.value_dim), self.storage)]
        if dtype in ROW_SCALED:
            shapes += [((n, stored, page, kv_heads), torch.float32)] * 2
        offsets, end, tensors = [], 0, []
        for shape, kind in shapes:
            offsets.append(end)
            end = -(-(end + math.prod(shape) * torch.empty(0, dtype=kind).element_size()) // 256) * 256
        self.spare = (end, end + int(spare))
        self.memory = torch.zeros(end + int(spare), dtype=torch.uint8, device=self.device)
        # Each page tensor of each layer in memory: (first byte, bytes per page, pages), page p at first + p * bytes.
        self.regions = []
        for first, (shape, kind) in zip(offsets, shapes):
            size = math.prod(shape) * torch.empty(0, dtype=kind).element_size()
            tensors.append(self.memory[first:first + size].view(kind).view(shape))
            per_page = size // (n * stored)
            self.regions += [(first + i * stored * per_page, per_page, self.pages) for i in range(n)]
        self.k, self.v = tensors[0], tensors[1]
        # FP8 pools: per layer key and value scales (stored = value / scale), on the device and as numbers.
        self.scales = torch.ones(len(self.layers), 2, dtype=torch.float32, device=self.device)
        self.scale_values = [(1.0, 1.0)] * len(self.layers)
        # reserve() calls so far: an attention kernel that plans once per step keys its plan on it.
        self.steps = 0
        # fp8:row and INT8 pools: a key and a value scale per position and KV head ([layers][pages][page][kv_heads]).
        self.k_scales = self.v_scales = None
        if dtype in ROW_SCALED:
            self.k_scales, self.v_scales = tensors[2], tensors[3]
        self.free = list(range(self.pages))   # a heap: the lowest free page first
        self.refs = [0] * self.pages
        self.floor = 0                        # pages [floor, pages) hold no KV
        self._lender, self._bounds = None, (-1, None)
        self.tables, self.lengths = {}, {}
        self.kernels = kernels if self.device.type == "cuda" and dtype in KERNEL_KINDS else None
        # A prompt chunk's keys and values as write() got them (pools of FP8 or INT8 pages with a fresh buffer): the
        # attention reads its chunk's own positions from them, the earlier ones from the pages.
        self.fresh_k = self.fresh_v = None
        self.fresh_tokens = 0
        self._staging, self._reserves = False, 0
        self._fresh_buffer = None
        width = kv_heads * (head_dim + self.value_dim) * torch.empty(0, dtype=self.compute).element_size()
        if fresh is not None and dtype in QUANTIZED and self.value_dim and fresh.data.device.type == self.device.type:
            self.fresh_tokens = fresh.data.numel() // width
        if self.fresh_tokens:
            self._fresh_buffer, data = fresh, fresh.data
            cap, size = self.fresh_tokens, torch.empty(0, dtype=self.compute).element_size()
            keys = cap * kv_heads * head_dim * size
            self.fresh_k = data[:keys].view(self.compute).view(cap, kv_heads, head_dim)
            self.fresh_v = data[keys:keys + cap * kv_heads * self.value_dim * size].view(self.compute).view(
                cap, kv_heads, self.value_dim)
        # Each sequence's page table on the device as of its last reserve(), which gather() reads (no copy per layer),
        # and the pinned staging buffer the step's ids cross in one asynchronous copy, free again once _copied passed.
        self.device_tables, self._host, self._copied = {}, None, None
        # Step buffers (step_rows > 0): fixed addresses for the step's slots and its window rows' metadata.
        self.window, self.step_rows, self.step_tokens = int(window), int(step_rows), int(step_tokens)
        self.windows, self.window_tokens = [], 0   # the step's window rows (batch rows) and their tokens
        if self.step_rows:
            if self.device.type != "cuda" or self.step_tokens < self.step_rows:
                raise ValueError("step buffers are a CUDA pool's, for at least one token per row")
            long, int32 = dict(dtype=torch.int64), dict(dtype=torch.int32)
            self.d_slots = torch.zeros(self.step_tokens, device=self.device, **long)
            self.d_starts = torch.zeros(self.step_rows, device=self.device, **int32)
            self.d_counts = torch.zeros(self.step_rows, device=self.device, **int32)
            self.d_firsts = torch.zeros(self.step_rows, device=self.device, **long)
            # Each window row's first partial row in the split-KV reader's workspace: its tokens' rank among the
            # step's window tokens (so the partials a launch needs follow its tokens).
            self.d_offsets = torch.zeros(self.step_rows, device=self.device, **int32)
            self.d_rows = torch.arange(self.step_rows, device=self.device, **int32)
            self.d_tables = torch.full((self.step_rows, self.pages), self.dump, device=self.device, **int32)
            self._h_slots = torch.zeros(self.step_tokens, pin_memory=True, **long)
            self._h_meta = torch.zeros(self.step_rows, 4, pin_memory=True, **long)
            self._h_tables = torch.full((self.step_rows, self.pages), self.dump, pin_memory=True, **int32)
        # Captured rounds: each round slot's page table, capacity and valid length at fixed device addresses, and the
        # lengths start_round found (a held row keeps its own).
        self.round_rows = int(round_rows)
        self.bound, self.slot_of = {}, {}
        slots = max(1, self.round_rows)
        self.round_tables = torch.full((slots, self.pages), self.dump, dtype=torch.int32, device=self.device)
        self.round_capacity = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self.round_lengths = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self._start = torch.zeros(slots, dtype=torch.int64, device=self.device)
        self._moved = torch.zeros(slots, dtype=torch.int64, device=self.device)   # a round's tree path moved already
        # Captured prompt chunks (prompts:kv_cache): the staged chunk's KV slots (made for the largest bucket at the
        # first stage_prompt) and its sequence's page table, at fixed addresses.
        self.prompt_slots = None
        self.prompt_table = torch.full((1, self.pages), self.dump, dtype=torch.int32, device=self.device)
        # With a fresh buffer: the staged chunk's first position its attention reads fresh (its frontier, a chunk past
        # the window, a prompt row; else past the bucket: a chunk of up to `window` tokens reads its pages, as the same
        # chunk in a StepBatch does), [1] int64.
        self.prompt_fresh_first = torch.zeros(1, dtype=torch.int64, device=self.device) if self.fresh_tokens else None

    # -- sequences -----------------------------------------------------------------------------------------------------
    def open(self, sequence):
        if sequence in self.tables:
            raise ValueError(f"sequence {sequence!r} is open already")
        self.tables[sequence], self.lengths[sequence] = [], 0

    def close(self, sequence):
        if sequence in self.slot_of:
            self._release(self.slot_of[sequence])
        for page in self.tables.pop(sequence, []):
            self._drop(page)
        self.lengths.pop(sequence, None)
        self.device_tables.pop(sequence, None)

    def fork(self, sequence, new):
        """A new sequence sharing `sequence`'s pages up to its valid length (copied when either writes into one). A
        sequence a round slot holds writes its tail page without the host: the new sequence gets its own copy of it."""
        if new in self.tables:
            raise ValueError(f"sequence {new!r} is open already")
        length = self.length(sequence)
        table = self._table(sequence)[:math.ceil(length / self.page)]
        for page in table:
            self.refs[page] += 1
        self.tables[new], self.lengths[new] = list(table), length
        if sequence in self.slot_of and length % self.page:
            self.tables[new][-1] = self._copy(table[-1])

    def length(self, sequence):
        self._table(sequence)
        if sequence in self.slot_of:   # rounds keep it on the device
            self.lengths[sequence] = int(self.round_lengths[self.slot_of[sequence]])
        return self.lengths[sequence]

    def keep(self, sequence, length):
        """The sequence's valid KV is its first `length` positions; pages past them are freed (a sequence a round slot
        holds keeps its pages up to its capacity)."""
        table = self._table(sequence)
        if not 0 <= length <= len(table) * self.page:
            raise ValueError(f"sequence {sequence!r} holds {len(table) * self.page} reserved positions, not {length}")
        self.lengths[sequence] = length
        if sequence in self.slot_of:   # between rounds: the next round starts from it (start_round)
            self.round_lengths[self.slot_of[sequence]] = length
            return
        needed = math.ceil(length / self.page)
        for page in table[needed:]:
            self._drop(page)
        del table[needed:]

    def set_scales(self, layer, key_scale, value_scale):
        """An fp8 pool's scales for a layer (stored = value / scale)."""
        self.scales[self.layers[layer]] = self.torch.tensor([key_scale, value_scale], dtype=self.torch.float32)
        self.scale_values[self.layers[layer]] = (float(key_scale), float(value_scale))

    def fresh_rows(self, layer, tokens):
        """The keys and values the last write() of `layer` stored for this step's `tokens` tokens, as it got them
        ([tokens][kv_heads][dim] each, the compute dtype), where the pool keeps them (FP8 and INT8 pages, a step with
        prompt rows, a chunk the fresh buffer holds), else None: a prompt row (a StepBatch row past the window, a
        PromptBatch's chunk from prompt_fresh_first on) attends its own chunk from them and the earlier positions
        from the pages."""
        if self.fresh_k is None or self._fresh_buffer.tag != (id(self), layer, tokens, self._reserves):
            return None
        return self.fresh_k[:tokens], self.fresh_v[:tokens]

    def layer_pages(self, layer):
        """A layer's pages as they are stored, for a kernel that reads them in place: (keys [pages][page][kv_heads]
        [head_dim], values [pages][page][kv_heads][value_dim]); an fp8:row or int8 pool adds their scales ([pages][page]
        [kv_heads] each)."""
        i = self.layers[layer]
        if self.dtype in ROW_SCALED:
            return self.k[i], self.v[i], self.k_scales[i], self.v_scales[i]
        return self.k[i], self.v[i]

    # -- a step ----------------------------------------------------------------------------------------------------------
    def reserve(self, batch):
        """Each token's slot at its position, pages taken as the sequences grow ([tokens] int64 on the device). The
        slots and the rows' page tables reach the device in one asynchronous copy, so the step's kernels queue behind
        it without the host waiting. A RoundBatch's slots are computed on the device from the bound page tables; a
        PromptBatch's are the ones stage_prompt staged (device work only)."""
        self._reserves += 1
        # Prompt tokens' keys and values are kept fresh for the step's attention: a PromptBatch's, a StepBatch's
        # with a row past the window (a prompt row); never a round's.
        self._staging = bool(getattr(batch, "prompt", False)) or (
            not batch.round and any(batch.row(r)[2] > self.window for r in range(batch.rows)))
        if getattr(batch, "prompt", False):
            if self.prompt_slots is None:
                raise ValueError("a prompt batch reads the slots stage_prompt(batch) staged: stage it first")
            return self.prompt_slots[:batch.bucket]
        if batch.round:
            return self._reserve_round(batch)
        self.steps += 1
        self._lend_ahead(batch)
        slots, tables = [], []
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            table = self._table(sequence)
            last = position + count - 1
            while len(table) * self.page <= last:
                table.append(self._take())
            for p in range(position // self.page, last // self.page + 1):
                if self.refs[table[p]] > 1:   # shared with a fork: copy before writing
                    table[p] = self._copy(table[p])
            slots += [table[q // self.page] * self.page + q % self.page for q in range(position, last + 1)]
            tables.append((sequence, list(table)))
        self._refresh_bound(tables)
        if self.step_rows:
            return self._step_buffers(batch, slots, tables)
        found = self._to_device(slots + [page for _, table in tables for page in table])
        offset = len(slots)
        for sequence, table in tables:
            self.device_tables[sequence] = found[offset:offset + len(table)]
            offset += len(table)
        return found[:len(slots)]

    def _step_buffers(self, batch, slots, tables):
        """The step's slots, and its window rows' first tokens, counts, first positions, first partial rows and page
        tables (unused entries at the dump page), into the step buffers in one asynchronous copy each; every row's table
        also as device_tables (for gather())."""
        torch = self.torch
        windows = [r for r in range(batch.rows) if batch.row(r)[2] <= self.window]
        if len(slots) > self.step_tokens or len(windows) > self.step_rows:
            raise ValueError(f"a step of {len(slots)} tokens and {len(windows)} rows of up to {self.window} "
                             f"tokens is larger than the pool's step buffers ({self.step_tokens} tokens, "
                             f"{self.step_rows} rows)")
        if self._copied is not None:
            self._copied.synchronize()
        k, n = len(slots), len(windows)
        self._h_slots[:k] = torch.tensor(slots, dtype=torch.int64)
        offset = 0
        for i, r in enumerate(windows):
            table = tables[r][1]
            _, first, count, position = batch.row(r)
            self._h_meta[i] = torch.tensor([first, count, position, offset], dtype=torch.int64)
            self._h_tables[i, :len(table)] = torch.tensor(table, dtype=torch.int32)
            self._h_tables[i, len(table):] = self.dump
            offset += count
        self.d_slots[:k].copy_(self._h_slots[:k], non_blocking=True)
        if n:
            meta = self._h_meta[:n]
            self.d_starts[:n].copy_(meta[:, 0], non_blocking=True)
            self.d_counts[:n].copy_(meta[:, 1], non_blocking=True)
            self.d_firsts[:n].copy_(meta[:, 2], non_blocking=True)
            self.d_offsets[:n].copy_(meta[:, 3], non_blocking=True)
            self.d_tables[:n].copy_(self._h_tables[:n], non_blocking=True)
        self._copied = torch.cuda.Event()
        self._copied.record()
        found = self._to_device([page for _, table in tables for page in table]) if tables else None
        offset = 0
        for sequence, table in tables:
            self.device_tables[sequence] = found[offset:offset + len(table)]
            offset += len(table)
        self.windows, self.window_tokens = windows, sum(batch.row(r)[2] for r in windows)
        return self.d_slots[:k]

    def _to_device(self, values):
        """values (ints) as an int64 tensor on the pool's device: on CUDA through a pinned buffer, asynchronously."""
        torch = self.torch
        if self.device.type != "cuda":
            return torch.tensor(values, dtype=torch.int64, device=self.device)
        if self._copied is not None:
            self._copied.synchronize()
        n = len(values)
        if self._host is None or self._host.numel() < n:
            self._host = torch.empty(max(n, 4096), dtype=torch.int64, pin_memory=True)
        self._host[:n] = torch.tensor(values, dtype=torch.int64)
        out = torch.empty(n, dtype=torch.int64, device=self.device)
        out.copy_(self._host[:n], non_blocking=True)
        self._copied = torch.cuda.Event()
        self._copied.record()
        return out

    def _rows(self, store, i, dim):
        """Layer i of `store` as [slots][kv_heads][dim] (dim 0 too: a pool of keys only), the dump page's slots last."""
        return store[i].view(self.stored_slots, self.kv_heads, dim)

    def write(self, layer, slots, keys, values):
        """Store a layer's keys [tokens][kv_heads][head_dim] and values at their slots (values None in a pool of keys
        only, value_dim 0). A slot below 0 is skipped: nothing a sequence reads changes (the kernel writes nowhere,
        torch writes the dump page)."""
        i = self.layers[layer]
        if self.fresh_k is not None and self._staging and keys.shape[0] <= self.fresh_tokens:
            tokens = keys.shape[0]
            self.fresh_k[:tokens].copy_(keys)
            self.fresh_v[:tokens].copy_(values)
            self._fresh_buffer.tag = (id(self), layer, tokens, self._reserves)
        parts = [(keys, self.k, self.k_scales, self.head_dim, 0)]
        if self.value_dim:
            parts.append((values, self.v, self.v_scales, self.value_dim, 1))
        if self.kernels is not None:
            row = self.dtype in ROW_SCALED
            self.kernels.write(keys, values if self.value_dim else None, slots, self.k[i],
                               self.v[i] if self.value_dim else None, self.dtype, self.k_scales[i] if row else None,
                               self.v_scales[i] if row and self.value_dim else None, multipliers=self.scale_values[i])
            return
        slots = self.torch.where(slots >= 0, slots, self.dump * self.page)
        for rows, store, scales, dim, which in parts:
            target = self._rows(store, i, dim)
            if self.dtype == "fp8":
                # E4M3 has no index_copy kernel: its bytes are copied as uint8.
                codes = (rows.float() / self.scales[i, which]).clamp(-448, 448).to(self.storage)
                target.view(self.torch.uint8).index_copy_(0, slots, codes.view(self.torch.uint8))
            elif self.dtype in ROW_SCALED:
                codes, scale = (fp8_rows if self.dtype == "fp8:row" else int8_rows)(rows)
                if codes.dtype == self.torch.float8_e4m3fn:
                    target, codes = target.view(self.torch.uint8), codes.view(self.torch.uint8)
                target.index_copy_(0, slots, codes)
                scales[i].view(self.stored_slots, self.kv_heads).index_copy_(0, slots, scale)
            else:
                target.index_copy_(0, slots, rows.to(self.storage))

    def gather(self, layer, sequence, count):
        """The sequence's first `count` positions of a layer: keys [count][kv_heads][head_dim] and values, in the
        compute dtype (dequantized from an fp8, fp8:row or int8 pool)."""
        table = self._table(sequence)
        if count > len(table) * self.page:
            raise ValueError(f"sequence {sequence!r} holds {len(table) * self.page} positions, not {count}")
        i = self.layers[layer]
        needed = math.ceil(count / self.page)
        index = self.device_tables.get(sequence)
        if index is None or index.numel() < needed:   # not reserved since it was opened, forked or bound
            index = self.device_tables[sequence] = self._to_device(table)
        index = index[:needed]
        n = needed * self.page
        k = self.k[i].index_select(0, index).reshape(n, self.kv_heads, self.head_dim)[:count]
        v = self.v[i].index_select(0, index).reshape(n, self.kv_heads, self.value_dim)[:count]
        if self.dtype == "fp8":
            return (k.float() * self.scales[i, 0]).to(self.compute), (v.float() * self.scales[i, 1]).to(self.compute)
        if self.dtype in ROW_SCALED:
            ks = self.k_scales[i].index_select(0, index).reshape(n, self.kv_heads)[:count]
            vs = self.v_scales[i].index_select(0, index).reshape(n, self.kv_heads)[:count]
            return (k.float() * ks[..., None]).to(self.compute), (v.float() * vs[..., None]).to(self.compute)
        return k.to(self.compute), v.to(self.compute)

    def rows_at(self, layer, slots):
        """The keys and values at `slots` (any shape of int64 slots) of a layer, dequantized to FP32: [..., kv_heads,
        dim] each (values [..., kv_heads, 0] in a pool of keys only). Device work only, so a captured round reads
        rows through it (kv:keys)."""
        i = self.layers[layer]
        flat = slots.reshape(-1)
        k = self._rows(self.k, i, self.head_dim).index_select(0, flat).float()
        v = self._rows(self.v, i, self.value_dim).index_select(0, flat).float()
        if self.dtype == "fp8":
            k, v = k * self.scales[i, 0], v * self.scales[i, 1]
        elif self.dtype in ROW_SCALED:
            k = k * self.k_scales[i].view(self.stored_slots, self.kv_heads).index_select(0, flat)[..., None]
            v = v * self.v_scales[i].view(self.stored_slots, self.kv_heads).index_select(0, flat)[..., None]
        shape = tuple(slots.shape)
        return k.view(*shape, self.kv_heads, self.head_dim), v.view(*shape, self.kv_heads, self.value_dim)

    def slots_at(self, round_slots, positions):
        """The KV slots of `positions` of the sequences bound to `round_slots` (both int64 tensors of one shape): a
        captured round's own reservation for rows it places itself (a per-block row cache's block numbers); a
        position outside its slot's capacity gets the dump page's slot, which no sequence reads. Device work only
        (kv:keys)."""
        shape = tuple(positions.shape)
        return self._slots_at(round_slots.reshape(-1).long(), positions.reshape(-1).long()).view(shape)

    def table(self, sequence):
        """The sequence's page ids, in order."""
        return list(self._table(sequence))

    def free_pages(self):
        return len(self.free)

    # -- captured prompt chunks (DEVICE_EXTENSIONS "prompts:kv_cache") -------------------------------------------------
    def stage_prompt(self, batch):
        """A captured prompt chunk's sequence (batch.sequence; None: padding only, a capture's warm-ups) in the pool's
        prompt buffers, outside the graph: its pages taken up to batch.frontier + batch.count (a page a fork shares in
        that range copied first), its real tokens' KV slots in prompt_slots and the padding's at the dump page, its page
        table in prompt_table (unused entries at the dump page). reserve(batch) inside the graph returns those slots."""
        torch = self.torch
        if self.prompt_slots is None or self.prompt_slots.numel() < batch.tokens_max:
            if capturing():
                raise RuntimeError("stage_prompt runs outside the graph, before each replay")
            self.prompt_slots = torch.full((batch.tokens_max,), self.dump * self.page, dtype=torch.int64,
                                           device=self.device)
        frontier, count = batch.frontier, batch.count
        slots, table = [], []
        if batch.sequence is not None:
            sequence = batch.sequence
            if sequence in self.slot_of:
                raise ValueError(f"sequence {sequence!r} decodes in round slot {self.slot_of[sequence]}: its prompt "
                                 "chunks run before it is bound")
            table = self._table(sequence)
            if count:
                last = frontier + count - 1
                self._lend_rows([(sequence, count, frontier)])
                while len(table) * self.page <= last:
                    table.append(self._take())
                for p in range(frontier // self.page, last // self.page + 1):
                    if self.refs[table[p]] > 1:   # shared with a fork: copy before writing
                        table[p] = self._copy(table[p])
            slots = [table[q // self.page] * self.page + q % self.page for q in range(frontier, frontier + count)]
            self.device_tables.pop(sequence, None)
        host = torch.full((batch.bucket,), self.dump * self.page, dtype=torch.int64)
        if slots:
            host[:len(slots)] = torch.tensor(slots, dtype=torch.int64)
        self.prompt_slots[:batch.bucket].copy_(host)
        row = torch.full((self.pages,), self.dump, dtype=torch.int32)
        if table:
            row[:len(table)] = torch.tensor(table, dtype=torch.int32)
        self.prompt_table[0].copy_(row)
        if self.prompt_fresh_first is not None:
            self.prompt_fresh_first.fill_(frontier if count > self.window else frontier + batch.bucket)

    def commit_prompt(self, batch):
        """After a captured prompt chunk: its sequence's valid length becomes batch.frontier + batch.count."""
        if batch.sequence is not None:
            self.lengths[batch.sequence] = batch.frontier + batch.count

    # -- captured rounds (DEVICE_EXTENSIONS "rounds:kv_cache") ---------------------------------------------------------
    def bind(self, slot, sequence, capacity):
        """Round slot `slot` holds `sequence`, whose pages are taken now up to `capacity` positions (its prompt, its
        output limit and the widest verify window), so a captured round reserves and writes without the host; a page
        the round could write that a fork shares is copied first. Outside a capture."""
        torch = self.torch
        if not 0 <= slot < self.round_rows:
            raise ValueError(f"round slot {slot} is outside the pool's {self.round_rows} round slots")
        if slot in self.bound:
            raise ValueError(f"round slot {slot} holds {self.bound[slot]!r} already")
        if sequence in self.slot_of:
            raise ValueError(f"sequence {sequence!r} is in round slot {self.slot_of[sequence]} already")
        table, length = self._table(sequence), self.lengths[sequence]
        capacity = int(capacity)
        if capacity < max(length, 1):
            raise ValueError(f"sequence {sequence!r} holds {length} positions: a capacity of {capacity} is too small")
        wanted = max(0, math.ceil(capacity / self.page) - len(table))
        if wanted > len(self.free):
            raise MemoryError(f"binding {sequence!r} up to {capacity} positions takes {wanted} pages; "
                              f"{len(self.free)} of the KV pool's {self.pages} are free")
        while len(table) * self.page < capacity:
            table.append(self._take())
        for p in range(length // self.page, len(table)):
            if self.refs[table[p]] > 1:
                table[p] = self._copy(table[p])
        row = torch.full((self.pages,), self.dump, dtype=torch.int32)
        row[:len(table)] = torch.tensor(table, dtype=torch.int32)
        self.round_tables[slot].copy_(row)
        self.round_capacity[slot] = capacity
        self.round_lengths[slot] = length
        self._start[slot] = length
        self.bound[slot], self.slot_of[sequence] = sequence, slot
        self.device_tables.pop(sequence, None)

    def unbind(self, slot):
        """Round slot `slot` holds no sequence; its sequence's valid length is read back from the device (one wait) and
        the pages past it are freed. Outside a capture."""
        if slot not in self.bound:
            raise ValueError(f"round slot {slot} holds no sequence")
        sequence = self.bound[slot]
        length = int(self.round_lengths[slot])
        self._release(slot)
        self.keep(sequence, length)

    def move(self, source, target):
        """The sequence in round slot `source` moves to slot `target` (the round driver keeps its rows at the front)."""
        if source not in self.bound or target in self.bound or not 0 <= target < self.round_rows:
            raise ValueError(f"round slot {source} must hold a sequence and slot {target} none")
        for tensor in (self.round_tables, self.round_capacity, self.round_lengths, self._start):
            tensor[target] = tensor[source]
        sequence = self.bound.pop(source)
        self.round_capacity[source] = 0
        self.round_tables[source].fill_(self.dump)
        self.bound[target], self.slot_of[sequence] = sequence, target

    def start_round(self, view):
        """A round begins: each row's valid length as the round finds it (a held row keeps it), and no row's draft-tree
        path moved yet (commit_round moves it once). Capture-safe."""
        rows = view.rows
        self._start[:rows].copy_(self.round_lengths[:rows])
        self._moved[:rows].zero_()

    def commit_round(self, view):
        """After a round's acceptance, from what start_round(view) found: each row's valid length becomes its frontier
        (view.frontier, the first position without KV) where it emitted (view.n_emitted > 0) and stays as the round
        found it where it emitted nothing (a held row), on the device. With a draft tree (view.tree_path [rows]
        [width + 1]: node 0, then the accepted nodes in order, -1 past them), the accepted nodes' keys and values move
        from their nodes' positions to consecutive ones first, once a round. Idempotent from the round's start: a
        truncation's second commit finds the path moved and sets the lengths from the same start. Capture-safe."""
        torch = self.torch
        rows = view.rows
        emitted = view.n_emitted[:rows]
        path = getattr(view, "tree_path", None)
        if path is not None:
            self._compact(rows, emitted, path[:rows])
        self.round_lengths[:rows].copy_(torch.where(emitted > 0, view.frontier[:rows], self._start[:rows]))

    def _reserve_round(self, batch):
        if batch.rows > self.round_rows:
            raise ValueError(f"a round of {batch.rows} rows reads round slots 0 .. {batch.rows - 1}; this KV pool has "
                             f"{self.round_rows} (attention-state's round_rows)")
        slot = batch.slots_t.index_select(0, batch.rows_t)
        return self._slots_at(slot, batch.positions_t)

    def _slots_at(self, slot, positions):
        """The KV slots of `positions` of the sequences in round slots `slot` (both [n] int64), the dump page's first
        slot where a position lies outside its slot's capacity."""
        torch = self.torch
        capacity = self.round_capacity.index_select(0, slot)
        page = self.round_tables[slot, (positions // self.page).clamp(0, self.pages - 1)]
        inside = (positions >= 0) & (positions < capacity)
        return torch.where(inside, page.long() * self.page + positions % self.page, self.dump * self.page)

    def _compact(self, rows, emitted, path):
        """The accepted tree path's KV (node path[j] at start + path[j]) to positions start + j, j < n_emitted."""
        torch = self.torch
        width = path.shape[1]
        columns = torch.arange(width, device=self.device)
        start = self._start[:rows]
        keep = (columns[None, :] < emitted[:, None]) & (columns[None, :] > 0) & (path >= 0) & \
            (self._moved[:rows, None] == 0)
        slot = torch.arange(rows, device=self.device)[:, None].expand(rows, width).reshape(-1)
        # Columns not kept go from the dump page to itself (position -1 lies outside every slot).
        source = self._slots_at(slot, torch.where(keep, start[:, None] + path.long(), -1).reshape(-1))
        target = self._slots_at(slot, torch.where(keep, start[:, None] + columns[None, :], -1).reshape(-1))
        n, held = len(self.layers), self.stored_slots
        stores = [store.view(n, held, self.kv_heads, self.head_dim) for store in (self.k,)]
        if self.value_dim:
            stores.append(self.v.view(n, held, self.kv_heads, self.value_dim))
        if self.k_scales is not None:
            stores += [scales.view(n, held, self.kv_heads) for scales in (self.k_scales, self.v_scales)]
        for flat in stores:
            if flat.dtype == torch.float8_e4m3fn:
                flat = flat.view(torch.uint8)
            flat.index_copy_(1, target, flat.index_select(1, source))
        self._moved[:rows] = 1

    def _release(self, slot):
        sequence = self.bound.pop(slot)
        del self.slot_of[sequence]
        self.round_capacity[slot] = 0
        self.round_tables[slot].fill_(self.dump)

    def _refresh_bound(self, tables):
        """A bound sequence's page table after a step reserved through it on the host (a copied page)."""
        for sequence, table in tables:
            slot = self.slot_of.get(sequence)
            if slot is not None:
                self.round_tables[slot, :len(table)].copy_(self.torch.tensor(table, dtype=self.torch.int32))

    # -- pages -----------------------------------------------------------------------------------------------------------
    def _table(self, sequence):
        try:
            return self.tables[sequence]
        except KeyError:
            raise ValueError(f"sequence {sequence!r} is not open") from None

    def _take(self):
        if not self.free:
            raise MemoryError(f"the KV pool's {self.pages} pages of {self.page} tokens are all in use")
        page = self.free[0]
        if page >= self.floor:
            high = self._bounds[1]
            if self._lender is not None and (high is None or page >= high):
                self._bounds = self._lender(page + 1)   # the borrower gives the memory back before KV lands there
            self.floor = page + 1
        heapq.heappop(self.free)
        self.refs[page] = 1
        return page

    def _drop(self, page):
        self.refs[page] -= 1
        if self.refs[page] == 0:
            heapq.heappush(self.free, page)
            if page == self.floor - 1:
                while self.floor and self.refs[self.floor - 1] == 0:
                    self.floor -= 1
                if self._lender is not None and self.floor <= self._bounds[0]:
                    self._bounds = self._lender(self.floor)

    def _lend_ahead(self, batch):
        """The borrower hears once of the floor a reserve will reach (the pages it takes are the lowest free ones):
        grown tables and copied shared pages, before any is handed out."""
        self._lend_rows([(batch.row(r)[0], batch.row(r)[2], batch.row(r)[3]) for r in range(batch.rows)])

    def _lend_rows(self, rows):
        """_lend_ahead for rows of (sequence, token count, first position)."""
        if self._lender is None:
            return
        need = 0
        for sequence, count, position in rows:
            table = self._table(sequence)
            last = position + count - 1
            grow = max(0, last // self.page + 1 - len(table))
            shared = sum(1 for p in range(position // self.page, min(len(table), last // self.page + 1))
                         if self.refs[table[p]] > 1)
            need += grow + shared
        if need and self.free:
            top = max(heapq.nsmallest(min(need, len(self.free)), self.free))
            high = self._bounds[1]
            if top >= self.floor and (high is None or top >= high):
                self._bounds = self._lender(top + 1)

    # -- lending -------------------------------------------------------------------------------------------------------
    def lendable(self):
        """What a borrower may use: {"memory": the pool's one uint8 allocation, "regions": [(first byte, bytes per page,
        pages)] of each page tensor of each layer, "spare": (first, end) bytes no page ever uses, "floor": pages
        [floor, pages) hold no KV}."""
        return {"memory": self.memory, "regions": list(self.regions), "spare": self.spare, "floor": self.floor}

    def on_reserve(self, callback):
        """callback(floor) -> (low, high): called now, before a page at or past `high` is handed out (the new floor:
        reclaim what lies under it before returning), and after the floor falls to `low` or below (pages freed). None
        removes it."""
        self._lender = callback
        self._bounds = callback(self.floor) if callback is not None else (-1, None)

    def _copy(self, page):
        new = self._take()
        self.k[:, new].copy_(self.k[:, page])
        self.v[:, new].copy_(self.v[:, page])
        if self.k_scales is not None:
            self.k_scales[:, new].copy_(self.k_scales[:, page])
            self.v_scales[:, new].copy_(self.v_scales[:, page])
        self._drop(page)
        return new


def _row_scales(x, top):
    """The FP32 scale of each row: its absolute maximum over `top` (an IEEE division, never a multiply by the
    reciprocal, which torch does for a Python number on CUDA), at least 1e-30."""
    amax = x.abs().amax(dim=-1)
    return (amax / amax.new_full(amax.shape, top)).clamp(min=1e-30)


def int8_rows(rows):
    """Rows [tokens][kv_heads][dim] as INT8 codes and one FP32 scale per token and head: the scale is the row's
    absolute maximum over 127 (at least 1e-30), each value its quotient rounded to the nearest code, ties to even."""
    import torch
    x = rows.float()
    scale = _row_scales(x, 127.0)
    return torch.round(x / scale[..., None]).clamp(-127, 127).to(torch.int8), scale


def fp8_rows(rows):
    """Rows [tokens][kv_heads][dim] as E4M3 codes and one FP32 scale per token and head: the scale is the row's
    absolute maximum over 448 (at least 1e-30), each value its quotient rounded to the nearest E4M3 value, ties to
    even."""
    import torch
    x = rows.float()
    scale = _row_scales(x, 448.0)
    return (x / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn), scale
