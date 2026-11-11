"""The gated-delta recurrence in the recurrence slot, token by token: recurrent-state-recovery's reference
(chunked_prefill.sequential over reference.py) as the host form; on the device, every sequence's state and convolution
window in a pool (device_states.py), decode tokens and verify windows through the column scan with a device commit of
their accepted prefix (crit's kernel recurrent-scans' pool rows where it builds, the torch formula otherwise), prompt
chunks through the same scan up to exact_rows tokens, the configuration's (default 512), and through recurrent-scans'
chunked prefill past it (on tensor cores or in FP32, the prefill option). Option fla (default off, a yardstick): fla's
chunked kernel instead where fla is installed, each of its Triton kernels on a stated tile (device_states.FLA_TILES)
rather than the one fla's autotune times best."""
from pathlib import Path

from .recurrence import chunked_prefill as recurrence
from .recurrence import device_states

HERE = Path(__file__).resolve().parent
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


class SequentialRecurrence:
    prompt = "two_pass"   # prompt rows: recurrent-scans' chunked prefill (fla's with option fla), else the scan

    def __init__(self, exact_rows=device_states.DEFAULT_EXACT_ROWS, chunk_fp32=True, commit="deferred", kernels=True,
                 folder=HERE, prefill="tensor_cores", scan="pool-delta-pipe"):
        self.exact_rows, self.chunk_fp32, self.commit = exact_rows, chunk_fp32, commit
        self.use_kernels, self.folder, self.arithmetic, self.scan = kernels, folder, prefill, scan
        self.prompts = []   # each pool's prompt rows: where the scan hands them to the chunked kernel

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
        """The device form: every sequence's state per layer (device_states.StatePool)."""
        pool = device_states.StatePool(layers, heads, keys, values, device, sequences, widest, conv, key_heads, rows,
                                       self.commit, self.kernels(device, keys, conv))
        self.prompted(pool)
        return pool

    def prompted(self, pool):
        """Sets the pool's crossovers from the exact_rows option, the same value in an eager call and inside a captured
        graph: prompt rows of up to that many tokens run the scan, longer ones the chunked kernel (0: every prompt
        row). The two give other bits, so the configuration chooses, never a timing; device_states.scan_rows measures a
        value to recommend. Without fla, recurrent-scans' chunked prefill in the prefill option's arithmetic."""
        pool.prefill = self.arithmetic
        pool.pipelined = self.scan == "pool-delta-pipe"
        chunked = device_states.chunked_kind(pool, device_states.STANDARD, self.prompt)
        pool.scan_rows = pool.scan_rows_captured = int(self.exact_rows)
        self.prompts.append({"heads": pool.heads, "key_heads": pool.key_heads, "keys": pool.keys, "values": pool.values,
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

    def forward(self, layer, query, key, value, log_decay, beta, pool, batch, geometry, out=None):
        return device_states.forward(layer, query, key, value, log_decay, beta, pool, batch, geometry, None,
                                     self.chunk_fp32, self.prompt, out)

    def convolve(self, layer, x, weight, bias, pool, batch, activation="silu", out=None):
        return device_states.convolve(layer, x, weight, bias, pool, batch, activation, out)

    def prefill(self, state, steps, geometry):
        geometry = geometry_of(geometry)
        outputs, state = recurrence.sequential(check_state(state, geometry), steps_of(steps), geometry)
        return [list(o) for o in outputs], tuple(state)


def create(context, options, parts):
    found = SequentialRecurrence(options["exact_rows"], options["chunk_fp32"], options["commit"], options["kernels"],
                                 prefill=options["prefill"], scan=options["scan"])
    if options.get("fla"):
        found.prompt = "fla"   # the yardstick: fla's chunked kernel where it is installed
    return found
