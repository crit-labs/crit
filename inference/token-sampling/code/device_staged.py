"""Core's sampling law for a step whose rules are staged in device buffers beforehand: no host read and no host value
inside, so a model can capture it in a CUDA graph with its decode step (the sampler slot's optional staged form).

    staged = Staged(rows, device)        # buffers for up to `rows` rows, at fixed addresses
    staged.set(params)                   # the step's rules (outside a capture): one asynchronous copy
    tokens = staged.sample(logits)       # [rows] int64, device work only

The law is device_law's, with every rule applied to every row (a neutral rule leaves the row as it is), so one captured
sample serves any mix of greedy and sampled rows: candidates rank by (value descending, index ascending), greedy takes
the first, a sampled row keeps exp((value - best) / temperature) through top_k, top_p and min_p, and Core's draw
u = uniform(seed, position, stream) (Philox4x32-10, computed on the device bit for bit as sampling.py's) picks the
first kept token whose cumulative weight exceeds u times the kept weight. A constrained row (an allowed set) is
device_law's: it is never staged."""
import torch
import triton
import triton.language as tl

WORD = 0xFFFFFFFF


@triton.jit
def _uniform_kernel(key_ptr, counter_ptr, out_ptr, n):
    """Core's uniform(seed, position, stream) per row: key = seed (lo, hi), counter = (position lo, hi, stream, 0)."""
    r = tl.program_id(0)
    if r < n:
        k0 = tl.load(key_ptr + 2 * r).to(tl.uint32)
        k1 = tl.load(key_ptr + 2 * r + 1).to(tl.uint32)
        c0 = tl.load(counter_ptr + 3 * r).to(tl.uint32)
        c1 = tl.load(counter_ptr + 3 * r + 1).to(tl.uint32)
        c2 = tl.load(counter_ptr + 3 * r + 2).to(tl.uint32)
        c3 = (c0 * 0).to(tl.uint32)
        m0 = (c0 * 0 + 0xD2511F53).to(tl.uint32)
        m1 = (c0 * 0 + 0xCD9E8D57).to(tl.uint32)
        w0 = (c0 * 0 + 0x9E3779B9).to(tl.uint32)
        w1 = (c0 * 0 + 0xBB67AE85).to(tl.uint32)
        for i in tl.static_range(10):
            if i > 0:
                k0 = k0 + w0
                k1 = k1 + w1
            hi0 = tl.umulhi(m0, c0)
            lo0 = m0 * c0
            hi1 = tl.umulhi(m1, c2)
            lo1 = m1 * c2
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        bits = (c0.to(tl.uint64) << 32 | c1.to(tl.uint64)) >> 11
        tl.store(out_ptr + r, bits.to(tl.float64) * 1.1102230246251565e-16)


class Staged:
    """Static buffers of a step's sampling rules for up to `rows` rows on `device`."""

    def __init__(self, rows, device):
        self.rows, self.device, self.count = int(rows), torch.device(device), 0
        self.h_rules = torch.zeros(self.rows, 4, dtype=torch.float64, pin_memory=True)   # temperature, top_k, top_p, min_p
        self.h_ints = torch.zeros(self.rows, 5, dtype=torch.int64, pin_memory=True)      # seed lo, hi, position lo, hi, stream
        self.rules = torch.zeros(self.rows, 4, dtype=torch.float64, device=self.device)
        self.ints = torch.zeros(self.rows, 5, dtype=torch.int64, device=self.device)
        self.u = torch.zeros(self.rows, dtype=torch.float64, device=self.device)
        self._copied = None

    def set(self, params):
        """The rules of the step's rows, in row order (dicts as device_law.sample takes them, no allowed set)."""
        if len(params) > self.rows:
            raise ValueError(f"{len(params)} rows for staged buffers of {self.rows}")
        if self._copied is not None:
            self._copied.synchronize()
        for r, p in enumerate(params):
            if p.get("allowed") is not None:
                raise ValueError("a constrained row is sampled by sample(), not staged")
            temperature = float(p.get("temperature", 0.0))
            top_p, top_k, min_p = float(p.get("top_p", 1.0)), int(p.get("top_k", 0)), float(p.get("min_p", 0.0))
            if temperature > 0 and (not 0 < top_p <= 1 or top_k < 0 or not 0 <= min_p <= 1):
                raise ValueError("top_p in (0, 1], top_k non-negative and min_p in [0, 1]")
            seed, position, stream = int(p.get("seed", 0)), int(p.get("position", 0)), int(p.get("stream", 0))
            self.h_rules[r] = torch.tensor([temperature, top_k, top_p, min_p], dtype=torch.float64)
            self.h_ints[r] = torch.tensor([seed & WORD, seed >> 32, position & WORD, position >> 32, stream],
                                          dtype=torch.int64)
        self.count = len(params)
        self.rules.copy_(self.h_rules, non_blocking=True)
        self.ints.copy_(self.h_ints, non_blocking=True)
        self._copied = torch.cuda.Event()
        self._copied.record()

    def sample(self, logits):
        """One token per row of logits ([rows][vocabulary], rows <= the staged buffers), by the staged rules."""
        rows = logits.shape[0]
        rules = self.rules[:rows]
        temperature, top_k, top_p, min_p = (rules[:, c] for c in range(4))
        key = self.ints[:rows, 0:2].contiguous()
        counter = self.ints[:rows, 2:5].contiguous()
        _uniform_kernel[(rows,)](key, counter, self.u, rows)
        u = self.u[:rows, None]
        v = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
        ranked, order = torch.sort(v, dim=-1, descending=True, stable=True)
        greedy = order[:, 0]
        ranked = ranked.double()
        width = ranked.shape[-1]
        position = torch.arange(width, device=logits.device)[None, :]
        finite = torch.isfinite(ranked).sum(-1)
        sampled = temperature > 0
        t = torch.where(sampled, temperature, torch.ones_like(temperature))[:, None]
        weights = torch.exp((ranked - ranked[:, :1]) / t)
        k = top_k.to(torch.int64)
        length = torch.where((k > 0) & (k < finite), k, finite)
        weights = torch.where(position < length[:, None], weights, torch.zeros_like(weights))
        cumulative = torch.cumsum(weights, -1)
        total = cumulative.gather(1, (length - 1).clamp(min=0)[:, None])
        cut = torch.minimum(torch.searchsorted(cumulative, top_p[:, None] * total)[:, 0], length - 1) + 1
        length = torch.where(top_p < 1, cut, length)
        weights = torch.where(position < length[:, None], weights, torch.zeros_like(weights))
        kept = ((weights >= min_p[:, None]) & (position < length[:, None])).sum(-1)
        length = torch.where(min_p > 0, kept, length)
        weights = torch.where(position < length[:, None], weights, torch.zeros_like(weights))
        cumulative = torch.cumsum(weights, -1)
        total = cumulative.gather(1, (length - 1).clamp(min=0)[:, None])
        pick = torch.minimum(torch.searchsorted(cumulative, u * total, right=True),
                             (length - 1).clamp(min=0)[:, None])
        return torch.where(sampled, order.gather(1, pick)[:, 0], greedy)
