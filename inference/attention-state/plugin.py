"""KV state in the kv_cache slot: attention-state's transactional causal reference (implementation.py) as the host
form, and its paged pool on the device (pages.py) as the device form, with captured rounds' slot binding
(rounds:kv_cache). Its pages are written by crit's paged-kv-attention kernels when they load (one launch per
layer for keys and values), else in torch."""
from pathlib import Path

from .state import device_attention
from .state import implementation as reference
from .state import pages

KERNELS = "paged-kv-attention"


class KVStateCache:
    def __init__(self, page=16, step_rows=0, step_tokens=0, window=16, round_rows=8, kernels=None, fresh=None):
        self.states, self.page = {}, page
        self.step_rows, self.step_tokens, self.window = step_rows, step_tokens, window
        self.round_rows, self.kernels, self.fresh = round_rows, kernels, fresh
        self.pools = []   # the device pools made, whose bytes per token and prompt reads facts() reports

    def _state(self, sequence):
        try:
            return self.states[sequence]
        except KeyError:
            raise ValueError(f"sequence {sequence!r} is not open") from None

    def open(self, sequence, key_width, value_width, capacity):
        if sequence in self.states:
            raise ValueError(f"sequence {sequence!r} is open already")
        self.states[sequence] = reference.KVState(capacity, key_width, value_width)

    def stage(self, sequence, keys, values):
        if len(keys) != len(values):
            raise ValueError("one value row per key row")
        state = self._state(sequence)
        for key, value in zip(keys, values):
            state.stage(key, value)

    def commit(self, sequence, count=None):
        self._state(sequence).commit(count)

    def rollback(self, sequence):
        self._state(sequence).rollback()

    def rows(self, sequence, pending=False):
        state = self._state(sequence)
        rows = state.committed + (state.pending if pending else [])
        return [(position, list(key), list(value)) for position, (key, value) in enumerate(rows)]

    def fork(self, sequence, new):
        state = self._state(sequence)
        self.open(new, state.key_width, state.value_width, state.capacity)
        self.states[new].committed = list(state.committed)

    def release(self, sequence):
        self.states.pop(sequence, None)

    def pool(self, layers, kv_heads, head_dim, tokens, device, dtype="bf16", *, page=None, value_dim=None, spare=0):
        """The device form: a paged KV pool of every sequence (pages.KVPool), `page` tokens a page (the plugin's
        option unless given), value_dim the values' width (the keys' unless given; 0: a pool of keys only), its window
        rows up to the plugin's window, with step buffers at fixed addresses when the plugin's step_rows asks for them
        (a CUDA pool: a model captures its decode steps in CUDA graphs over them), and round slots for captured rounds.
        spare: bytes after the pages in the same allocation for a borrower's own use (a device expert cache's slots),
        which then also borrows the pages no sequence holds (lendable(), on_reserve()). A pool of FP8 or INT8 pages
        keeps each prompt chunk's keys and values fresh in the plugin's fresh buffer for the chunk's attention."""
        rows = self.step_rows if str(device).startswith("cuda") else 0
        made = pages.KVPool(layers, kv_heads, head_dim, tokens, device, dtype, page=page or self.page,
                            value_dim=value_dim, step_rows=rows, step_tokens=self.step_tokens if rows else 0,
                            window=self.window, round_rows=self.round_rows, kernels=self.kernels, spare=spare,
                            fresh=self.fresh)
        self.pools.append(made)
        return made

    def facts(self):
        """The engine facts of the pools made so far: kv_bytes_per_token, the device bytes one more token of context
        takes across them, its slot in each page tensor ([layers][pages][page][...]: every layer's keys and values at
        its pool's precision, an int8 pool's scales too), which crit-eval sizes the KV cache at a context by; and over
        FP8 or INT8 pools which keys and values a prompt row reads (a StepBatch row of more than `window` tokens, a
        PromptBatch's chunk of more than `window` real tokens): with the fresh buffer its own chunk's as the
        projections gave them (16-bit: steps of up to fresh_tokens tokens) and the earlier positions from the pages,
        else everything from the pages. None before a pool is made."""
        if not self.pools:
            return {}
        found = {"kv_bytes_per_token": sum(tensor[:, 0, 0].numel() * tensor.element_size() for pool in self.pools
                                           for tensor in (pool.k, pool.v, pool.k_scales, pool.v_scales)
                                           if tensor is not None)}
        quantized = [p for p in self.pools if p.dtype in pages.QUANTIZED and p.value_dim]
        if not quantized:
            return found
        tokens = min(p.fresh_tokens for p in quantized)
        if not tokens:
            found["prompt_kv"] = ("a prompt row reads its own chunk's keys and values from the FP8 or INT8 pages, as "
                                  "decode rows read them (no fresh buffer: option fresh_mib 0)")
            return found
        found["prompt_kv"] = (f"a prompt row (a step's row, or a prompt batch's chunk, of more than {self.window} "
                              "tokens) reads its own chunk's keys and values as the projections gave them (16-bit), "
                              "the earlier positions from the FP8 or INT8 pages; window rows (decode tokens, verify "
                              "windows, pieces of up to the window), rounds, block lists and draft trees read the "
                              "pages: a scored continuation reads its own tokens otherwise than the decode that served "
                              "them (outside a 0.001-nat honesty tolerance; fresh_mib 0 for served == scored)")
        found["prompt_kv_fresh_tokens"] = tokens
        return found


def memory(context, options):
    """The fresh buffer the plugin makes on a CUDA engine (fresh_mib), exactly: a prompt chunk's 16-bit keys and
    values, which a pool of FP8 or INT8 pages keeps for the chunk's attention (shared by its pools, one layer at a
    time)."""
    on_cuda = device_attention.engine_device(context).startswith("cuda")
    return {"device": (int(options["fresh_mib"]) << 20) if on_cuda else 0, "host": 0}


def create(context, options, parts):
    kernels = None
    if options["kernels"] == "auto":
        from crit_plugins.kernels import load_kernel
        kernels = load_kernel(KERNELS, folder=Path(__file__).parent)
    fresh = None
    device = device_attention.engine_device(context)
    if device.startswith("cuda") and options["fresh_mib"]:
        import torch
        fresh = pages.FreshBuffer(torch.empty(int(options["fresh_mib"]) << 20, dtype=torch.uint8, device=device))
    return KVStateCache(options["page"], options["step_rows"], options["step_tokens"], options["window"],
                        options["round_rows"], kernels, fresh)
