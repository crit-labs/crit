"""paged-kv-attention's sweep benches (crit_kernels.sweep): one layer's pages of the model's attention shape (head
widths, query and KV heads) for each page kind, written with torch's encoding of the kernel's rule (the write bench's
reference too), each sequence and query row seeded by its own index, a float64 reference over the stored values.

Rows: write, tokens; split, decode rows (one column each, 16K positions); window, merge and fused, the columns of four
rows over 4K positions (flash-window, flash-fused, cuda-window and cuda-fused too); sparse, decode tokens over 16 listed
blocks of 64 positions; prefill, flash-prefill and cuda-prefill, a chunk's tokens after 1,024 positions. A shape may
set the context (N), the batch rows (R) and the chunk's position (P).

Each call is timed as an engine's captured rounds run it, a CUDA graph replay (captured once per configuration, shape
and row count, after an eager call that compiles it), with the partials in the bench's own workspace."""
import json
import math

import torch

KINDS = {"bf16": 0, "fp16": 0, "fp8": 1, "fp8:row": 2, "int8": 3}
STORE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.float8_e4m3fn, "fp8:row": torch.float8_e4m3fn,
         "int8": torch.int8}
MODEL_KINDS = ("bf16", "fp8:row", "int8")   # the page kinds a model's shapes are swept with (--shapes for others)
PAGE = 16                                    # attention-state's default page
MULTIPLIERS = (0.02, 0.03)                   # an "fp8" layer's key and value multipliers


def _power_of_two(n):
    return n > 0 and n & (n - 1) == 0


def model_shapes(model, value_dim=True):
    """The model's attention shape for each page kind: head width, value width, query and KV heads."""
    hq = model.get("num_attention_heads")
    if not hq:
        return []
    d = model.get("head_dim") or model.get("hidden_size", 0) // hq
    dv = model.get("v_head_dim") or d
    hkv = model.get("num_key_value_heads") or hq
    if not (_power_of_two(d) and _power_of_two(dv) and 16 <= min(d, dv) and max(d, dv) <= 256 and hq % hkv == 0):
        return []
    out = []
    for kind in MODEL_KINDS:
        shape = {"D": d, "HQ": hq, "HKV": hkv, "PAGE": PAGE, "KIND": KINDS[kind], "kind": kind}
        if value_dim:
            shape["DV"] = dv
        out.append(shape)
    return out


def _kind(shape):
    if "kind" in shape:
        return shape["kind"]
    return {0: "bf16", 1: "fp8", 2: "fp8:row", 3: "int8"}[int(shape["KIND"])]


def _dtype(shape):
    return torch.float16 if _kind(shape) == "fp16" else torch.bfloat16


def encode(x, kind, multiplier=1.0):
    """x [..., width] FP32 -> (codes in the page dtype, FP32 scales [...] or None): the kernel's write rule in torch
    (IEEE divisions)."""
    if kind in ("bf16", "fp16"):
        return x.to(STORE[kind]), None
    if kind == "fp8":
        return (x / torch.full_like(x, multiplier)).clamp(-448, 448).to(torch.float8_e4m3fn), None
    top = 448.0 if kind == "fp8:row" else 127.0
    amax = x.abs().amax(-1)
    scale = (amax / torch.full_like(amax, top)).clamp(min=1e-30)
    y = (x / scale[..., None]).clamp(-top, top)
    return (y.to(torch.float8_e4m3fn) if kind == "fp8:row" else torch.round(y).to(torch.int8)), scale


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _shared_memory():
    """Shared memory a block may take on this GPU (its opt-in limit)."""
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "shared_memory_per_block_optin", 0) or 101376


def _code_bytes(shape):
    return 2 if _kind(shape) in ("bf16", "fp16") else 1


def _kv_bytes(shape, positions):
    """Bytes of `positions` positions of a shape's keys and values (and their scales) in its page kind."""
    d = int(shape["D"])
    dv = int(shape.get("DV", d))
    kind = _kind(shape)
    per = torch.empty(0, dtype=STORE[kind]).element_size() * (d + dv) + (8 if kind in ("fp8:row", "int8") else 0)
    return positions * int(shape["HKV"]) * per


class Pages:
    """One layer's pages for `sequences` sequences of `length` positions, sequence s over its own shuffled pages and
    its keys and values seeded by s alone."""

    def __init__(self, kind, hkv, d, dv, page, sequences, length, device):
        self.kind, self.hkv, self.d, self.dv, self.page = kind, hkv, d, dv, page
        self.multipliers = MULTIPLIERS if kind == "fp8" else (1.0, 1.0)
        self.width = math.ceil(length / page)
        n = self.width * sequences
        self.k = torch.zeros(n, page, hkv, d, dtype=STORE[kind], device=device)
        self.v = torch.zeros(n, page, hkv, dv, dtype=STORE[kind], device=device)
        self.ks = self.vs = None
        if kind in ("fp8:row", "int8"):
            self.ks = torch.zeros(n, page, hkv, dtype=torch.float32, device=device)
            self.vs = torch.zeros_like(self.ks)
        tables = []
        for s in range(sequences):
            g = torch.Generator(device=device).manual_seed(7919 * s + 1)
            order = (torch.randperm(self.width, generator=g, device=device) + s * self.width).to(torch.int32)
            tables.append(order)
            slots = self.slots(order, torch.arange(self.width * page, device=device))
            for at in range(0, slots.numel(), 8192):
                part = slots[at:at + 8192]
                for pages, scales, m, w in ((self.k, self.ks, self.multipliers[0], d),
                                            (self.v, self.vs, self.multipliers[1], dv)):
                    codes, scale = encode(torch.randn(part.numel(), hkv, w, generator=g, device=device), kind, m)
                    pages.view(-1, hkv, w)[part] = codes
                    if scales is not None:
                        scales.view(-1, hkv)[part] = scale
        self.tables = torch.stack(tables)

    def slots(self, table, positions):
        return table[positions // self.page].long() * self.page + positions % self.page

    def rows(self, s, n):
        """The stored keys and values of positions 0..n-1 of sequence s in float64."""
        slots = self.slots(self.tables[s], torch.arange(n, device=self.k.device))
        k = self.k.view(-1, self.hkv, self.d)[slots].double()
        v = self.v.view(-1, self.hkv, self.dv)[slots].double()
        if self.kind == "fp8":
            k, v = k * self.multipliers[0], v * self.multipliers[1]
        if self.ks is not None:
            k = k * self.ks.view(-1, self.hkv)[slots].double()[..., None]
            v = v * self.vs.view(-1, self.hkv)[slots].double()[..., None]
        return k, v


def attention(q, keys, values, seen, scale):
    """q [T][hq][d] float64 over keys [n][hkv][d] and values [n][hkv][dv], seen [T][n] -> out [T][hq][dv]."""
    hq, hkv = q.shape[1], keys.shape[1]
    k = keys.repeat_interleave(hq // hkv, dim=1)
    v = values.repeat_interleave(hq // hkv, dim=1)
    s = torch.einsum("thd,nhd->thn", q, k) * scale
    s = s.masked_fill(~seen[:, None, :], float("-inf"))
    return torch.einsum("thn,nhd->thd", torch.softmax(s, -1), v)


def _queries(seeds, hq, d, dtype, device):
    """One query row [hq][d] per seed, each from its own generator."""
    return torch.stack([torch.randn(hq, d, generator=torch.Generator().manual_seed(int(s))) for s in seeds]).to(
        device, dtype)


def _close(got, want, atol=8e-3, rtol=2e-2):
    error = (got.double() - want).abs()
    bad = int((error > atol + rtol * want.abs()).sum())
    return None if bad == 0 else f"{bad} outputs beyond {atol} + {rtol} x |float64 attention|"


class _Base:
    def __init__(self):
        self._pages, self._states, self._references = {}, {}, {}
        self._shape = None
        self._graphs = {}   # (build, its selected configurations, state key) -> (graph, output)
        self._workspaces = {}
        self._library = {}      # the yardstick's pages in the library's format and its planned wrappers
        self._library_buffer = None

    def _library_pages(self, p):
        """The pages as FlashInfer reads them: (keys, values, key scale, value scale). 16-bit pages as they are; FP8
        pages with a multiplier per layer as they are, the multipliers as FlashInfer's scalar scales; FP8 and INT8 pages
        with a scale per position and head (no library reads those) widened once to BF16 (the same values, two bytes a
        value instead of one)."""
        key = ("pages", id(p))
        if key not in self._library:
            if p.kind in ("bf16", "fp16"):
                found = (p.k, p.v, None, None)
            elif p.kind == "fp8":
                found = (p.k, p.v, p.multipliers[0], p.multipliers[1])
            else:
                found = tuple((codes.float() * scales[..., None]).to(torch.bfloat16)
                              for codes, scales in ((p.k, p.ks), (p.v, p.vs))) + (None, None)
            self._library[key] = found
        return self._library[key]

    def _flashinfer(self, s, p, lists, columns):
        """FlashInfer's paged attention over the same pages, a zero-argument call: row i's queries (`columns` of them,
        each seeing the positions up to its own, causally) over the pages lists[i] = (page ids, positions seen by its
        last column). A batch decode (tensor-core kernels, which take any query heads per KV head) where every row is
        one column of equal widths, else a batch prefill; planned once outside the timed call. None where FlashInfer is
        not installed."""
        try:
            import flashinfer
        except ImportError:
            return None
        k, v, k_scale, v_scale = self._library_pages(p)
        device, q = p.k.device, s["q"]
        if self._library_buffer is None:
            self._library_buffer = torch.empty(256 << 20, dtype=torch.uint8, device=device)
        counts = [int(pages.numel()) for pages, _ in lists]
        indptr = torch.tensor([0] + [sum(counts[:i + 1]) for i in range(len(counts))], dtype=torch.int32,
                              device=device)
        indices = torch.cat([pages.to(torch.int32) for pages, _ in lists])
        last = torch.tensor([n - (c - 1) * p.page for (_, n), c in zip(lists, counts)], dtype=torch.int32,
                            device=device)
        hq = q.shape[1]
        out = torch.empty(q.shape[0], hq, p.dv, dtype=q.dtype, device=device)
        if columns == 1 and p.d == p.dv:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._library_buffer, "NHD", use_tensor_cores=True)
            wrapper.plan(indptr, indices, last, hq, p.hkv, p.d, p.page, q_data_type=q.dtype, kv_data_type=k.dtype)
        else:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._library_buffer, "NHD")
            starts = torch.arange(0, columns * (len(lists) + 1), columns, dtype=torch.int32, device=device)
            wrapper.plan(starts, indptr, indices, last, hq, p.hkv, p.d, p.page, head_dim_vo=p.dv, causal=True,
                         q_data_type=q.dtype, kv_data_type=k.dtype)
        self._library[("wrapper", s["key"])] = wrapper   # kept with the state's pages
        return lambda: wrapper.run(q, (k, v), k_scale=k_scale, v_scale=v_scale, out=out)

    def _workspace(self, s, floats):
        """The partials' workspace of a state's launch, the bench's own (as attention-state gives its own): a module
        workspace that grows would leave graphs captured before over freed memory."""
        key = (s["key"], floats)
        if key not in self._workspaces:
            self._workspaces[key] = torch.empty(floats, dtype=torch.float32, device=s["q"].device)
        return self._workspaces[key]

    def _replay(self, module, s, call):
        """call() replayed from a CUDA graph captured for the build's current configuration and this state."""
        selected = json.dumps(getattr(getattr(module, "_crit_tuning", None), "SELECTED", {}), sort_keys=True)
        key = (id(module), selected, s["key"])
        found = self._graphs.get(key)
        if found is None:
            if any(k[:2] != key[:2] for k in self._graphs):
                self._graphs.clear()   # another configuration: its graphs are done with
            call()                     # compiles, grows the module's workspace outside the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = call()
            found = self._graphs[key] = (graph, out)
        found[0].replay()
        return found[1]

    def shapes(self, model):
        return model_shapes(model)

    def _key(self, shape):
        """The shape's key; a new shape drops the previous one's pages, inputs and references."""
        key = tuple(sorted(shape.items()))
        if key != self._shape:
            self._pages.clear()
            self._states.clear()
            self._references.clear()
            self._graphs.clear()
            self._workspaces.clear()
            self._library.clear()
            self._shape = key
        return key

    def pages(self, shape, sequences, length, device):
        key = (sequences, length)
        if key not in self._pages:
            d = int(shape["D"])
            self._pages[key] = Pages(_kind(shape), int(shape["HKV"]), d, int(shape.get("DV", d)),
                                     int(shape.get("PAGE", PAGE)), sequences, length, device)
        return self._pages[key]

    def reference(self, state):
        if state["key"] not in self._references:
            self._references[state["key"]] = self._reference(state)
        return self._references[state["key"]]

    def first_row(self, out):
        return out[0]


class Write(_Base):
    """A step's tokens (BF16, the projections') into shuffled slots of a pool of 8,192 slots."""
    structure = "write"
    SLOTS = 8192

    def inputs(self, shape, rows, device):
        kind, hkv, d = _kind(shape), int(shape["HKV"]), int(shape["D"])
        dv = int(shape.get("DV", d))
        key = self._key(shape)
        if "pool" not in self._states:
            g = torch.Generator(device=device).manual_seed(5)
            keys = torch.randn(self.SLOTS // 2, hkv, d, generator=g, device=device).to(torch.bfloat16)
            values = torch.randn(self.SLOTS // 2, hkv, dv, generator=g, device=device).to(torch.bfloat16)
            slots = torch.randperm(self.SLOTS, generator=g, device=device)[:self.SLOTS // 2]
            kp = torch.zeros(self.SLOTS, hkv, d, dtype=STORE[kind], device=device)
            vp = torch.zeros(self.SLOTS, hkv, dv, dtype=STORE[kind], device=device)
            ks = vs = None
            if kind in ("fp8:row", "int8"):
                ks, vs = torch.zeros(self.SLOTS, hkv, device=device), torch.zeros(self.SLOTS, hkv, device=device)
            self._states["pool"] = dict(keys=keys, values=values, slots=slots, kp=kp, vp=vp, ks=ks, vs=vs, kind=kind)
        s = self._states["pool"]
        return dict(s, keys=s["keys"][:rows], values=s["values"][:rows], slots=s["slots"][:rows], key=(key, rows),
                    multipliers=MULTIPLIERS if kind == "fp8" else (1.0, 1.0))

    def run(self, module, s):
        self._replay(module, s, lambda: module.write(s["keys"], s["values"], s["slots"], s["kp"], s["vp"], s["kind"],
                                                     s["ks"], s["vs"], multipliers=s["multipliers"]))
        return s   # the written slots are read by check() and first_row(), outside the timed call

    def _written(self, s):
        T = s["slots"].shape[0]
        out = [_bits(s["kp"][s["slots"]]).reshape(T, -1), _bits(s["vp"][s["slots"]]).reshape(T, -1)]
        if s["ks"] is not None:
            out += [_bits(s["ks"][s["slots"]]).reshape(T, -1), _bits(s["vs"][s["slots"]]).reshape(T, -1)]
        return torch.cat(out, 1)

    def first_row(self, s):
        return self._written(s)[0]

    def _reference(self, s):
        T = s["slots"].shape[0]
        coded = [encode(x.float(), s["kind"], m) for x, m in ((s["keys"], s["multipliers"][0]),
                                                                (s["values"], s["multipliers"][1]))]
        out = [_bits(codes).reshape(T, -1) for codes, _ in coded]
        if s["ks"] is not None:
            out += [_bits(scale).reshape(T, -1) for _, scale in coded]
        return torch.cat(out, 1)

    def check(self, got, want, s):
        bad = int((self._written(got) != want).any(1).sum())
        return None if bad == 0 else f"{bad} tokens' codes or scales differ from the encoding rule"

    def yardstick(self, s):
        """FlashInfer's append_paged_kv_cache into the same slots, the pool seen as pages of one position (each token
        its own one-page sequence): a copy into 16-bit pages. FlashInfer appends codes in the cache's dtype and encodes
        nothing, so the FP8 and INT8 kinds have no library write (reported as such)."""
        try:
            import flashinfer
        except ImportError:
            return None
        if s["kind"] not in ("bf16", "fp16"):
            raise NotImplementedError("no library encodes FP8 or INT8 pages: FlashInfer appends codes already in "
                                      "the cache's dtype")
        T, device = s["slots"].shape[0], s["slots"].device
        kc, vc = s["kp"].unsqueeze(1), s["vp"].unsqueeze(1)
        keys, values = s["keys"].to(kc.dtype), s["values"].to(vc.dtype)
        batch = torch.arange(T, dtype=torch.int32, device=device)
        positions = torch.zeros(T, dtype=torch.int32, device=device)
        indices = s["slots"].to(torch.int32)
        indptr = torch.arange(T + 1, dtype=torch.int32, device=device)
        last = torch.ones(T, dtype=torch.int32, device=device)
        return lambda: flashinfer.append_paged_kv_cache(keys, values, batch, positions, (kc, vc), indices, indptr,
                                                        last)

    def bytes(self, shape, rows):
        d = int(shape["D"])
        return _kv_bytes(shape, rows) + rows * int(shape["HKV"]) * (d + int(shape.get("DV", d))) * 2


class _Windows(_Base):
    """R rows (4) of `rows` columns over their sequences' pages (row r's column 0 at position N - 7r, N 4,096), query
    rows seeded by (row, column)."""
    fused = False
    flash = False
    N, R = 4096, 4

    def inputs(self, shape, rows, device):
        columns = rows
        n, R = int(shape.get("N", self.N)), int(shape.get("R", self.R))
        d, hq = int(shape["D"]), int(shape["HQ"])
        key = (self._key(shape), columns)
        pages = self.pages(shape, R, n + 64, device)
        if key not in self._states:
            q = _queries([1000 * r + c for r in range(R) for c in range(columns)], hq, d, _dtype(shape), device)
            firsts = torch.tensor([n - 7 * r for r in range(R)], dtype=torch.int64, device=device)
            starts = torch.arange(0, R * columns, columns, dtype=torch.int32, device=device)
            rows_t = torch.arange(R, dtype=torch.int32, device=device)
            out = torch.empty(R * columns, hq, pages.dv, dtype=_dtype(shape), device=device)
            self._states[key] = dict(q=q, firsts=firsts, starts=starts, rows=rows_t, out=out, pages=pages,
                                     columns=columns, key=key, R=R)
        return self._states[key]

    def run(self, module, s):
        p = s["pages"]
        workspace = None
        flash = {"flash": self.flash} if self.flash else {}
        if not self.fused:
            if self.flash:
                chunks = module.cuda_chunk if self.flash == "cuda" else module.flash_chunk
                chunk = chunks(p.d, p.dv, s["q"].shape[1], p.hkv, p.page, p.kind)
                positions = p.tables.shape[1] * p.page
                parts = (module.cuda_parts(p.d, p.dv, s["q"].shape[1], p.hkv, p.page, p.kind, positions)
                         if self.flash == "cuda" else module.flash_parts(positions, chunk))
            else:
                parts = module.partitions(p.k.device, p.hkv, p.kind, p.d, p.dv)
            workspace = self._workspace(s, module.workspace_floats(s["R"] * s["columns"], s["q"].shape[1], parts,
                                                                   p.dv))
        return self._replay(module, s, lambda: module.attend(
            s["q"], p.k, p.v, p.kind, p.tables, s["rows"], s["starts"], s["firsts"], s["columns"], p.page,
            key_scales=p.ks, value_scales=p.vs, multipliers=p.multipliers, out=s["out"], fused=self.fused,
            workspace=workspace, **flash))

    def _reference(self, s):
        p, cols = s["pages"], s["columns"]
        out = []
        for r in range(s["R"]):
            first = int(s["firsts"][r])
            keys, values = p.rows(r, first + cols)
            seen = torch.arange(first + cols, device=keys.device)[None, :] <= first + torch.arange(
                cols, device=keys.device)[:, None]
            out.append(attention(s["q"][r * cols:(r + 1) * cols].double(), keys, values, seen, p.d ** -0.5))
        return torch.cat(out)

    def check(self, got, want, s):
        return _close(got, want)

    def yardstick(self, s):
        """FlashInfer over the same rows: R rows of `columns` causal columns over their sequences' pages (a batch
        decode for one column, a batch prefill for more)."""
        p, cols = s["pages"], s["columns"]
        lists = []
        for r in range(s["R"]):
            n = int(s["firsts"][r]) + cols
            lists.append((p.tables[r][:-(-n // p.page)], n))
        return self._flashinfer(s, p, lists, cols)

    def bytes(self, shape, rows):
        return _kv_bytes(shape, int(shape.get("R", self.R)) * (int(shape.get("N", self.N)) + rows))


class Window(_Windows):
    structure = "window"


class Merge(_Windows):
    structure = "merge"


class Fused(_Windows):
    structure = "fused"
    fused = True


class FlashWindow(_Windows):
    """The flash window launch (attend(..., flash=True)) over the same rows; its splits' workspace the bench's own."""
    structure = "flash-window"
    flash = True

    def valid_for(self, knobs, shape):
        """A split a whole number of blocks, a tile and its stages within the GPU's shared memory."""
        d = int(shape["D"])
        dv = int(shape.get("DV", d))
        block = knobs.get("BLOCK_N") or 32
        return (not knobs.get("CHUNK") or knobs["CHUNK"] % block == 0) and _flash_fits(
            knobs.get("ROWS") or 32, d, dv, block, knobs.get("STAGES") or (1 if max(d, dv) > 128 else 2))


class FlashFused(_Windows):
    structure = "flash-fused"
    flash = True
    fused = True


def _flash_fits(rows, d, dv, block, stages):
    """A flash tile of `rows` query rows and `stages` blocks of keys and values within the GPU's shared memory (16-bit
    in shared memory: an estimate of what Triton allocates)."""
    return rows * d * 2 + stages * block * (d + dv) * 2 + 1024 <= _shared_memory()


def _cuda_fits(shape, warps, block, stages, dvp=None):
    """A CUDA flash tile (16 query rows a warp in shared memory, `stages` blocks of keys and values, 8-bit pages' codes
    and their 16-bit conversion) within the GPU's shared memory with at least one stage (the launch takes fewer stages
    where more do not fit), and its widths ones it takes."""
    d = int(shape["D"])
    dv = int(shape.get("DV", d))
    dvp = dvp or dv
    byte = _kind(shape) not in ("bf16", "fp16")
    stage = block * (d + dvp) * (1 if byte else 2) + (block * 8 if byte else 0)
    conv = block * (d + dvp) * 2 + block * 8 if byte else 0
    return d in (64, 128, 256) and dv in (64, 128, 256) and 16 * warps * d * 2 + stage + conv <= _shared_memory()


def _converts(knobs, shape):
    """A cuda tile's CONVERT knob set only where it changes the tile: over 8-bit pages."""
    return not knobs.get("CONVERT") or _kind(shape) not in ("bf16", "fp16")


def _cuda_variants(shape, prefill=False):
    """The CUDA variant (kernel.json "cuda" "variant") a cuda structure's bench calls on a shape: the sweep compiles it
    before it holds the GPU."""
    d = int(shape["D"])
    dv = d if prefill else int(shape.get("DV", d))
    return [{"T": int(_kind(shape) == "fp16"), "KIND": int(shape["KIND"]), "D": d, "DV": dv, "W": 0, "S": 0, "VS": 0}]


class CudaWindow(_Windows):
    """The CUDA flash tile's window launch (attend(..., flash="cuda")) over the same rows; its splits' workspace the
    bench's own."""
    structure = "cuda-window"
    flash = "cuda"

    def variants(self, shape):
        return _cuda_variants(shape)

    def valid_for(self, knobs, shape):
        """A split a whole number of blocks, a tile within the GPU's shared memory; CONVERT only over 8-bit pages
        (16-bit pages convert nothing: the same tile)."""
        block = knobs.get("BLOCK_N") or (16 if _kind(shape) in ("bf16", "fp16") else 32)
        return (not knobs.get("CHUNK") or knobs["CHUNK"] % (block << knobs.get("RAMP", 0)) == 0) and _cuda_fits(
            shape, knobs.get("WARPS") or 4, block, 1) and _converts(knobs, shape)


class CudaFused(_Windows):
    structure = "cuda-fused"
    flash = "cuda"
    fused = True

    def variants(self, shape):
        return _cuda_variants(shape)

    def valid_for(self, knobs, shape):
        d = int(shape["D"])
        dv = int(shape.get("DV", d))
        split = knobs.get("VSPLIT") or 1
        return dv % split == 0 and dv // split >= 64 and _cuda_fits(shape, knobs.get("WARPS") or 4, 32, 1,
                                                                    dv // split) and _converts(knobs, shape)


class Split(_Windows):
    """`rows` decode rows of one column each (row r at position N - 7r, N 16,384, over its own sequence): the piece
    size and partition count the sweep tries reach attend through partitions() and piece_size()."""
    structure = "split"
    N, SEQUENCES = 16384, 8

    def inputs(self, shape, rows, device):
        n = int(shape.get("N", self.N))
        d, hq = int(shape["D"]), int(shape["HQ"])
        key = (self._key(shape), rows)
        pages = self.pages(shape, max(rows, self.SEQUENCES), n + 1, device)
        if key not in self._states:
            q = _queries(range(rows), hq, d, _dtype(shape), device)
            firsts = torch.tensor([n - 7 * r for r in range(rows)], dtype=torch.int64, device=device)
            starts = torch.arange(rows, dtype=torch.int32, device=device)
            out = torch.empty(rows, hq, pages.dv, dtype=_dtype(shape), device=device)
            self._states[key] = dict(q=q, firsts=firsts, starts=starts, rows=starts, out=out, pages=pages, columns=1,
                                     key=key, R=rows)
        return self._states[key]

    def valid_for(self, knobs, shape):
        """A piece's keys within a third of the GPU's shared memory (an estimate of what its pipelined loads take):
        larger pieces do not compile there, and would drop the configuration for every shape."""
        d = int(shape["D"])
        piece = knobs["PIECE"]
        return not piece or piece * max(d, int(shape.get("DV", d))) * _code_bytes(shape) <= _shared_memory() // 3

    def bytes(self, shape, rows):
        return _kv_bytes(shape, rows * int(shape.get("N", self.N)))


class Sparse(_Base):
    """`rows` decode tokens (rows of one column, row r over sequence r mod 4 at position N - 1 - 7r, N 8,192), each
    seeing 16 listed blocks of 64 positions (its own block among them), over four partitions as attention-state's
    block-sparse launches take."""
    structure = "sparse"
    N, BLOCKS, BLOCK, SEQUENCES = 8192, 16, 64, 4

    def inputs(self, shape, rows, device):
        n = int(shape.get("N", self.N))
        d, hq = int(shape["D"]), int(shape["HQ"])
        key = (self._key(shape), rows)
        pages = self.pages(shape, self.SEQUENCES, n, device)
        if key not in self._states:
            q = _queries(range(rows), hq, d, _dtype(shape), device)
            firsts = torch.tensor([n - 1 - 7 * r for r in range(rows)], dtype=torch.int64, device=device)
            blocks = []
            for r in range(rows):
                own = int(firsts[r]) // self.BLOCK
                chosen = torch.randperm(own, generator=torch.Generator().manual_seed(31 + r))[:self.BLOCKS - 1]
                blocks.append(sorted(chosen.tolist() + [own]))
            blocks = torch.tensor(blocks, dtype=torch.int32, device=device)
            starts = torch.arange(rows, dtype=torch.int32, device=device)
            out = torch.empty(rows, hq, pages.dv, dtype=_dtype(shape), device=device)
            self._states[key] = dict(q=q, firsts=firsts, starts=starts, rows=starts % self.SEQUENCES, blocks=blocks,
                                     out=out, pages=pages, key=key, R=rows)
        return self._states[key]

    def run(self, module, s):
        p = s["pages"]
        workspace = self._workspace(s, module.workspace_floats(s["R"], s["q"].shape[1], 4, p.dv))
        return self._replay(module, s, lambda: module.attend(
            s["q"], p.k, p.v, p.kind, p.tables, s["rows"], s["starts"], s["firsts"], 1, p.page, key_scales=p.ks,
            value_scales=p.vs, multipliers=p.multipliers, partitions=4, blocks=s["blocks"], block_size=self.BLOCK,
            out=s["out"], workspace=workspace))

    def _reference(self, s):
        p = s["pages"]
        out = []
        for r in range(s["R"]):
            first = int(s["firsts"][r])
            keys, values = p.rows(r % self.SEQUENCES, first + 1)
            n = torch.arange(first + 1, device=keys.device)
            seen = (n // self.BLOCK)[:, None].eq(s["blocks"][r].long()[None, :]).any(1)
            out.append(attention(s["q"][r:r + 1].double(), keys, values, seen[None, :], p.d ** -0.5))
        return torch.cat(out)

    def check(self, got, want, s):
        return _close(got, want)

    def yardstick(self, s):
        """FlashInfer's batch decode over exactly the positions a token sees: each token's page list is the pages of
        its listed blocks (its own block's up to its position), so the softmax runs over the same keys."""
        p = s["pages"]
        per = self.BLOCK // p.page
        lists = []
        for r in range(s["R"]):
            table, first = p.tables[r % self.SEQUENCES], int(s["firsts"][r])
            ids = []
            for b in s["blocks"][r].tolist():
                ids.append(table[b * per:min((b + 1) * per, first // p.page + 1)])
            ids = torch.cat(ids)
            lists.append((ids, (ids.numel() - 1) * p.page + first % p.page + 1))
        return self._flashinfer(s, p, lists, 1)

    def bytes(self, shape, rows):
        return _kv_bytes(shape, rows * self.BLOCKS * self.BLOCK)


class Prefill(_Base):
    """A chunk of `rows` tokens of one sequence at position P (1,024) on, its query rows seeded once (row t the same
    for every chunk length); the reference checks three of its rows."""
    structure = "prefill"
    P, MAX = 1024, 2048

    def shapes(self, model):
        return [dict(s, INT8=0) for s in model_shapes(model, value_dim=False)]

    def inputs(self, shape, rows, device):
        p0 = int(shape.get("P", self.P))
        d, hq = int(shape["D"]), int(shape["HQ"])
        top = max(rows, self.MAX)
        key = self._key(shape)
        pages = self.pages(shape, 1, p0 + top, device)
        if "q" not in self._states or self._states["q"].shape[0] < rows:
            g = torch.Generator(device=device).manual_seed(3)
            self._states["q"] = torch.randn(top, hq, d, generator=g, device=device).to(_dtype(shape))
            self._states["out"] = torch.empty(top, hq, d, dtype=_dtype(shape), device=device)
        return dict(q=self._states["q"][:rows], out=self._states["out"][:rows], pages=pages, p0=p0,
                    int8=bool(int(shape.get("INT8", 0))), key=(key, rows), rows=rows)

    flash = False

    def run(self, module, s):
        p = s["pages"]
        flash = {"flash": self.flash} if self.flash else {}
        return self._replay(module, s, lambda: module.prefill(
            s["q"], p.k, p.v, p.kind, p.tables[0], s["p0"], p.page, key_scales=p.ks, value_scales=p.vs,
            multipliers=p.multipliers, out=s["out"], int8_qk=s["int8"], **flash))

    def valid_for(self, knobs, shape):
        """A block of queries and two of keys and values within the GPU's shared memory (the estimate Triton's
        allocation follows at these shapes): larger blocks do not compile there."""
        d = int(shape["D"])
        block_n = knobs["BLOCK_N"] or (64 if d <= 128 else 32)
        return knobs["BLOCK_M"] * d * 2 + 2 * block_n * d * _code_bytes(shape) + 8192 <= _shared_memory()

    def yardstick(self, s):
        """FlashInfer's batch prefill of the one causal chunk over its sequence's pages (BF16 queries: FlashInfer has
        no INT8 query path, so an int8_qk shape is timed against the 16-bit one)."""
        p = s["pages"]
        n = s["p0"] + s["rows"]
        return self._flashinfer(s, p, [(p.tables[0][:-(-n // p.page)], n)], s["rows"])

    def _picked(self, rows):
        return sorted({0, rows // 2, rows - 1})

    def _reference(self, s):
        p = s["pages"]
        end = s["p0"] + s["rows"]
        keys, values = p.rows(0, end)
        positions = torch.tensor([s["p0"] + t for t in self._picked(s["rows"])], device=keys.device)
        seen = torch.arange(end, device=keys.device)[None, :] <= positions[:, None]
        return attention(s["q"][self._picked(s["rows"])].double(), keys, values, seen, p.d ** -0.5)

    def check(self, got, want, s):
        # INT8 queries (int8_qk) approximate the queries: within 0.07 of the float64 attention of the BF16 ones
        return _close(got[self._picked(s["rows"])], want, atol=7e-2 if s["int8"] else 8e-3)


class FlashPrefill(Prefill):
    """The flash prompt chunk (prefill(..., flash=True)) over the same chunk and pages."""
    structure = "flash-prefill"
    flash = True

    def valid_for(self, knobs, shape):
        d = int(shape["D"])
        return _flash_fits(knobs["BLOCK_M"] or 32, d, d, knobs["BLOCK_N"] or 32, knobs["STAGES"] or (1 if d > 128 else 2))


class CudaPrefill(Prefill):
    """The CUDA flash tile's prompt chunk (prefill(..., flash="cuda")) over the same chunk and pages (BF16 queries)."""
    structure = "cuda-prefill"
    flash = "cuda"

    def variants(self, shape):
        return _cuda_variants(shape, prefill=True)

    def shapes(self, model):
        return [dict(s, INT8=0) for s in model_shapes(model, value_dim=False)]

    def valid_for(self, knobs, shape):
        return not int(shape.get("INT8", 0)) and _cuda_fits(shape, knobs["WARPS"] or 4, knobs["BLOCK_N"] or 32, 1)
