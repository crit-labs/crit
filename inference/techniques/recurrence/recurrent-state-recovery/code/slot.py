"""What the recurrence slot's gated-delta plugins share (sequential-recurrence and recurrent-state-recovery): the host
form's arguments as chunked_prefill takes them, crit's kernel recurrent-scans for a CUDA pool, every sequence's state
and convolution window in a device_states.StatePool, its prompt rows' crossover from the exact_rows option, the facts
and exactness they report, and the pools given back at close. Each plugin adds its own host prefill() and options."""
from . import chunk_states
from . import chunked_prefill as recurrence
from . import device_states

KERNEL = "recurrent-scans"
MAX_KEYS = 256   # the pool rows' widest key dimension


def steps_of(steps):
    return [recurrence.Step(tuple(s["query"]), tuple(s["key"]), tuple(s["value"]), s["log_decay"], s["beta"])
            for s in steps]


def geometry_of(geometry):
    return recurrence.Geometry(**geometry)


def check_state(state, geometry):
    if len(state) != geometry.keys * geometry.values:
        raise ValueError("the state is keys * values values")
    return tuple(float(v) for v in state)


class RecurrenceSlot:
    """The device form of a gated-delta recurrence plugin. owner: the plugin's id (its pools' name in the device's
    readings); folder: the plugin's folder (where load_kernel finds its kernel tuning); chunk_fp32: the chunked
    kernel's FP32 accumulation (device_states.forward); decode: scan (decode rows through the column scan) or chunk
    (every row through recurrent-scans' chunk-aligned exact form, chunk_states)."""
    prompt = "two_pass"   # prompt rows: recurrent-scans' chunked prefill (the prefill option's), else the scan

    def __init__(self, owner, folder, exact_rows=device_states.DEFAULT_EXACT_ROWS, commit="deferred", kernels=True,
                 prefill="tensor_cores", scan="pool-delta-pipe", chunk_fp32=True, decode="scan"):
        if decode not in ("scan", "chunk"):
            raise ValueError(f"decode is scan or chunk, not {decode!r}")
        self.owner, self.folder, self.decode = owner, folder, decode
        self.exact_rows, self.commit, self.chunk_fp32 = exact_rows, commit, chunk_fp32
        self.use_kernels, self.arithmetic, self.scan = kernels, prefill, scan
        self.prompts = []   # each pool's prompt rows: where the scan hands them to the chunked kernel
        self.elastic = False   # the pools' states backed for the sequences that are open (the elastic option)
        self.made = []   # the device pools made, which close() discards

    def close(self):
        """The engine closes: every pool made gives its device memory back (device_states.StatePool.discard)."""
        for made in self.made:
            made.discard()
        self.made = []

    def kernels(self, device, keys, conv=None):
        """crit's kernel recurrent-scans (built for this GPU, tuned or default knobs) for a CUDA pool of `keys` key
        dimensions and convolution `conv`, or None (the torch formula, recorded in the engine's facts when the
        kernel builds but does not cover the shape)."""
        import torch
        if not self.use_kernels or torch.device(device).type != "cuda":
            return None
        from crit_plugins.kernels import record_fallback, load_kernel
        module = load_kernel(KERNEL, folder=self.folder)
        if module is not None and keys > MAX_KEYS:
            record_fallback(KERNEL, folder=self.folder,
                            reason=f"key dimensions {keys} past the pool rows' {MAX_KEYS}: the torch formula runs them")
            return None
        if module is not None and conv is not None:
            _, width, dilation = device_states.conv_shape(conv)
            if not module.pool_conv_supported(width, dilation):
                record_fallback(KERNEL, folder=self.folder,
                                reason=f"a convolution of {width} taps at dilation {dilation} the pool rows do not run: "
                                       "the torch formula runs the pool")
                return None
        return module

    def pool(self, layers, heads, keys, values, device, sequences=None, widest=0, conv=None, key_heads=None, rows=None):
        """The device form: every sequence's state per layer (device_states.StatePool), committed by `commit`."""
        pool = device_states.StatePool(layers, heads, keys, values, device, sequences, widest, conv, key_heads, rows,
                                       self.commit, self.kernels(device, keys, conv), elastic=self.elastic,
                                       owner=self.owner)
        if self.decode == "chunk":   # every row through the chunk-aligned exact form (chunk_states)
            pool.chunk = chunk_states.ChunkState(pool)
        self.prompted(pool)
        self.made.append(pool)
        return pool

    def prompted(self, pool):
        """Sets the pool's crossovers from the exact_rows option, the same value in an eager call and inside a captured
        graph: prompt rows of up to that many tokens run the scan, longer ones the chunked kernel (0: every prompt
        row). The two give other bits, so the configuration chooses, never a timing; device_states.scan_rows measures a
        value to recommend. The chunked kernel's arithmetic is the prefill option's (pool.prefill)."""
        pool.prefill = self.arithmetic
        pool.pipelined = self.scan == "pool-delta-pipe"
        chunked = device_states.chunked_kind(pool, device_states.STANDARD, self.prompt)
        pool.scan_rows = pool.scan_rows_captured = int(self.exact_rows)
        if pool.chunk is not None:
            chunked = "chunk_rows"
        self.prompts.append({"decode": "chunk" if pool.chunk is not None else "scan",
                             "heads": pool.heads, "key_heads": pool.key_heads, "keys": pool.keys, "values": pool.values,
                             "layers": len(pool.layers), "chunked": chunked,
                             "chunked_arithmetic": device_states.CHUNKED_ARITHMETIC.get(chunked),
                             "chunked_row_invariant": None if chunked is None else False,
                             "scan_rows": None if chunked is None else pool.scan_rows,
                             "scan_rows_captured": None if chunked is None else pool.scan_rows_captured,
                             "source": "exact_rows", "default": device_states.DEFAULT_EXACT_ROWS,
                             "scan": self.scan if pool.kernels is not None else None})

    def facts(self):
        """Where prompt rows run (the scan up to scan_rows tokens eagerly, scan_rows_captured in a captured graph, then
        the chunked kernel) and how far each path's state after a prompt is from the exact recurrence's (scan: the bits
        of decode rows of the same tokens; chunked: within the plugin's stated tolerance)."""
        if not self.prompts:
            return {}
        return {"recurrence_prompts": {"pools": list(self.prompts), "scan": device_states.SCAN_AGREEMENT,
                                       "chunked": device_states.CHUNKED_AGREEMENT, "captured": device_states.CAPTURED}}

    def exactness(self):
        """The level its rows keep (device_states.prompt_exactness): prompt rows exact where every one runs the scan
        (prompt_rows scan, or no chunked kernel for the pools' shapes), repeatable where long ones reach a chunked
        kernel."""
        return device_states.prompt_exactness(self.prompts)

    def forward(self, layer, query, key, value, log_decay, beta, pool, batch, geometry, out=None):
        return device_states.forward(layer, query, key, value, log_decay, beta, pool, batch, geometry, None,
                                     self.chunk_fp32, self.prompt, out)

    def convolve(self, layer, x, weight, bias, pool, batch, activation="silu", out=None):
        return device_states.convolve(layer, x, weight, bias, pool, batch, activation, out)
