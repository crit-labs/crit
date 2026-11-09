"""Core's sampling law inside a captured decode round: the sampler slot's "rounds:sampler" extension
(crit_plugins.device.DEVICE_EXTENSIONS), drafting and verifying on the device with no host value inside.

    sampler.reserve(rows, widest, vocab, device, logprobs=0)   # every buffer, once, for the widest round
    sampler.stage(slot, params)                                # a request's rules (outside a capture)
    tokens = sampler.draft_device(view, logits, depth)         # a draft per row, its law kept (sampled rounds)
    sampler.predictions_device(view, logits, view.draft_laws)  # each verify column's prediction

A round's rows are its slots 0 .. rows - 1. Row r's column c draws at output position view.out_count[r] + c (the
output tokens before column 0, Core's position), with Core's Philox uniform keyed on (seed, position, stream, purpose):
`sample` for the verify columns, `draft` for a drafter's draws, `accept` and `residual` for the rejection rule. So a
round's tokens are those of sampling.verify (the host form's predictions) for the same rows, drafts and laws:

- greedy rows (temperature 0): every column's argmax, the lower id on a tie;
- sampled rows, drafts without laws (draft_laws None: greedy or copied drafts): every column the target's own draw;
- sampled rows, drafts with laws: column c < width accepts draft d when u_accept * q(d) < p(d), else draws from
  max(p - q, 0) with u_residual (the draft itself when nothing is left); the last column draws the bonus from p; a cut
  column (draft -1), a column whose law is empty (a copied draft, DraftLaws.clear) and a draft its law gives no mass
  (q(d) = 0) take the target's own draw.

The tokens equal the host form's wherever the law is settled exactly: greedy rows, top_k <= 64, and open rows whose kept
set or draw stays within the top 64 (the fast path, FP64). A row the one-block full-vocabulary law takes weighs in FP32,
so a draw within ~1e-5 of a cumulative boundary may differ from the FP64 host form; and a draft whose law keeps more than
64 tokens has no sparse form, so it is verified by the target's own draw where the host form (with the law) would use
the rejection rule: the same law, other tokens.

On CUDA the arithmetic is token-sampling's sampling law (sampling.cu, with the binding sampling_torch.cpp), built at
load through torch's extension builder (native()): greedy rows and 1 <= top_k <= 64 on its fast path (stage blocks of
4,096 logits keep their best top_k + 1 processed candidates, one law block per row merges them and runs Core's steps 7
to 12 in FP64, the verify rule included), open rows settled there when top_p or min_p closes the kept set within the
top 64 or the draw lands among them, every other row on a one-block full-vocabulary law (FP32 values and weights: a draw
within ~1e-5 of a boundary may differ), and Core's log-probabilities fused into the same launches. A drafter's laws are
then kept sparse (at most 64 kept (token, probability) pairs per draft; a law keeping more has no sparse form, and its
draft is verified by the target's own draw, which keeps the law at a lower acceptance). Where it does not build (no CUDA
compiler), with the `rounds` option torch, and on the CPU, the same rules run in torch over sorted rows in FP64 with
dense laws: capture-safe and exact against the host form up to cumulative sums in a parallel order (a draw within
rounding of a boundary may differ), and slower (a sort of every row).

Greedy rounds (view.mode greedy, replay or forced) take torch's argmax, whose first maximum is the lower id.
"""
import hashlib
import os
from pathlib import Path
import warnings

import torch

HERE = Path(__file__).resolve().parent
SAMPLE, DRAFT, ACCEPT, RESIDUAL = 0, 1, 2, 3     # crit::core::DrawPurpose
MAX_TOP_K = 64
WORD = 0xFFFFFFFF
ONE = 0x3F800000                                 # the bits of FP32 1.0
KERNELS = ("native", "torch")                    # sampling.cu built at load, or the torch law

_NATIVE = {}


class Native:
    """sampling.cu's launches for torch tensors (the extension module built at load), None for an absent buffer."""

    def __init__(self, module):
        self.module = module
        self._empty = {}

    def _or(self, t, device):
        if t is not None:
            return t
        key = str(device)
        if key not in self._empty:
            self._empty[key] = torch.empty(0, device=device)
        return self._empty[key]

    def layout(self):
        return self.module.layout()

    def pack_request(self, temperature=0.0, top_p=1.0, min_p=0.0, top_k=0, seed=0, stream=0, position=0):
        return self.module.pack_request(float(temperature), float(top_p), float(min_p), int(top_k),
                                        int(seed) & ((1 << 64) - 1), int(stream), int(position))

    def scratch_bytes(self, vocab, rows):
        return self.module.scratch_bytes(int(vocab), int(rows))

    def logprob_scratch_bytes(self, vocab, rows, n):
        return self.module.logprob_scratch_bytes(int(vocab), int(rows), int(n))

    def verify(self, logits, vocab, requests, count, columns, drafts, sparse, dense, predictions, scratch, lp_n=0,
               chosen=None, top_ids=None, top_values=None):
        d = logits.device
        self.module.verify(logits, int(vocab), requests, int(count), int(columns), self._or(drafts, d),
                           self._or(sparse, d), self._or(dense, d), predictions, scratch, int(lp_n),
                           self._or(chosen, d), self._or(top_ids, d), self._or(top_values, d))

    def sample(self, logits, vocab, requests, count, columns, position_base, purpose, out, sparse=None, dense=None,
               scratch=None, lp_n=0, chosen=None, top_ids=None, top_values=None, tree=None, tree_tokens=None):
        d = logits.device
        self.module.sample(logits, int(vocab), requests, int(count), int(columns), int(position_base), int(purpose), out,
                           self._or(sparse, d), self._or(dense, d), scratch, int(lp_n), self._or(chosen, d),
                           self._or(top_ids, d), self._or(top_values, d), self._or(tree, d), self._or(tree_tokens, d))

    def logprobs(self, logits, vocab, chosen, n, chosen_values, top_ids=None, top_values=None, scratch=None):
        d = logits.device
        self.module.logprobs(logits, int(vocab), chosen, int(n), chosen_values, self._or(top_ids, d),
                             self._or(top_values, d), scratch)


def native(device):
    """token-sampling's sampling law (sampling.cu with sampling_torch.cpp) built for `device`'s architecture through
    torch's extension builder, once a process and cached across processes under a name hashed from the sources' paths
    and contents (each engine's copy of the plugin builds its own); None where it does not build (no CUDA compiler, no
    Core headers) or on a CPU device, and the torch law serves."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"{major}.{minor}"   # no architecture-specific instructions: the plain target
    if arch not in _NATIVE:
        from torch.utils.cpp_extension import load
        sources = [HERE / "sampling.cu", HERE / "sampling_torch.cpp"]
        digest = hashlib.sha256()
        for source in sources + [HERE / "sampling.h"]:
            digest.update(str(source).encode() + source.read_bytes())
        # The target as an explicit nvcc flag: torch's builder then adds none of its own (no TORCH_CUDA_ARCH_LIST read
        # or written).
        target = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
        try:
            core = _core_include()
            if core is None:
                raise FileNotFoundError("Core's headers (crit/sampling.h) were not found; set CRIT_CORE_INCLUDE")
            module = load(name=f"crit_sampling_{arch.replace('.', '_')}_{digest.hexdigest()[:12]}",
                          sources=[str(s) for s in sources], extra_include_paths=[str(HERE), str(core)],
                          extra_cflags=["-O3", "-std=c++20"],
                          extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", target], verbose=False)
            _NATIVE[arch] = Native(module)
        except Exception as error:   # no compiler, or headers that do not match: the torch law serves
            warnings.warn(f"token-sampling: the sampling law did not build for {arch} ({error}); rounds sample with "
                          "the torch law instead")
            _NATIVE[arch] = None
    return _NATIVE[arch]


def _core_include():
    """Core's public headers: CRIT_CORE_INCLUDE, else the crit checkout this plugin lies in (inference/... beside
    core/include), else the crit_core package's checkout."""
    found = os.environ.get("CRIT_CORE_INCLUDE")
    candidates = [Path(found)] if found else []
    candidates += [parent / "core" / "include" for parent in HERE.parents]
    try:
        import crit_core
        candidates.append(Path(crit_core.__file__).resolve().parents[2] / "include")
    except ImportError:
        pass
    return next((c for c in candidates if (c / "crit" / "sampling.h").is_file()), None)


# --- Core's Philox draw in torch (int64 arithmetic, exact): the torch law's uniforms -------------------------------

def _mulhilo(a, b):
    """(hi, lo) 32-bit words of the 64-bit product of the constant a and the tensor b (values in [0, 2^32))."""
    ah, al = a >> 16, a & 0xFFFF
    bh, bl = b >> 16, b & 0xFFFF
    mid = ah * bl + al * bh
    low = al * bl + ((mid & 0xFFFF) << 16)
    return ((ah * bh + (mid >> 16) + (low >> 32)) & WORD), low & WORD


def uniform_words(k0, k1, c0, c1, stream, purpose):
    """Core's uniform from the 32-bit words of the seed (k0 low, k1 high) and the position (c0, c1), int64 tensors
    holding values in [0, 2^32), broadcast together with stream; FP64, bit-identical to crit::core::uniform."""
    c2 = torch.as_tensor(stream) & WORD
    c3 = torch.zeros_like(torch.as_tensor(c0)) + (int(purpose) & WORD)
    k0, k1, c0, c1, c2, c3 = torch.broadcast_tensors(torch.as_tensor(k0), torch.as_tensor(k1), torch.as_tensor(c0),
                                                     torch.as_tensor(c1), c2, c3)
    for round_ in range(10):
        if round_:
            k0, k1 = (k0 + 0x9E3779B9) & WORD, (k1 + 0xBB67AE85) & WORD
        hi0, lo0 = _mulhilo(0xD2511F53, c0)
        hi1, lo1 = _mulhilo(0xCD9E8D57, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    bits = (c0 << 21) | (c1 >> 11)
    return bits.double() * 2.0 ** -53


def uniforms(seed, position, stream, purpose):
    """Core's uniform(seed, position, stream, purpose) for tensors of int64 seeds (any 64-bit pattern), positions and
    streams (broadcast together), FP64; bit-identical to crit::core::uniform and sampling.uniform."""
    seed, position = torch.as_tensor(seed), torch.as_tensor(position)
    return uniform_words(seed & WORD, (seed >> 32) & WORD, position & WORD, (position >> 32) & WORD, stream, purpose)


def _signed(seed):
    """A 64-bit unsigned seed as the int64 with its bits."""
    seed = int(seed) & ((1 << 64) - 1)
    return seed - (1 << 64) if seed >= 1 << 63 else seed


class DraftLaws:
    """The laws a round's drafts were drawn from, at fixed addresses for up to rows x widest drafts: sparse
    (SparseLaw, `words` int32 words per draft: count, 64 ids, 64 FP32 probabilities) for the CUDA sampler, dense
    ([vocab] FP32 per draft) for the torch law. A round of (rows, width) uses the first rows * width entries, row
    major, so its laws are contiguous at any width."""

    def __init__(self, rows, widest, vocab, device, sparse_words=None):
        self.rows, self.widest, self.vocab = rows, max(widest, 1), vocab
        self.sparse = sparse_words is not None
        if self.sparse:
            self.words = sparse_words
            self.buffer = torch.zeros(rows * self.widest, sparse_words, dtype=torch.int32, device=device)
        else:
            self.buffer = torch.zeros(rows * self.widest, vocab, dtype=torch.float32, device=device)

    def view(self, rows, width):
        """[rows][width][words] int32 (sparse) or [rows][width][vocab] FP32 (dense)."""
        return self.buffer[:rows * width].view(rows, width, self.buffer.shape[1])   # width 0 too

    def clear(self, rows, width, use):
        """Where use [rows][width] holds, the draft has no law (a copied draft, drawn from none): its column is verified
        by the target's own draw; capture-safe."""
        laws = self.view(rows, width)
        if self.sparse:
            laws[..., 0].masked_fill_(use, 0)
        else:
            laws.masked_fill_(use[..., None], 0.0)

    def write(self, rows, width, laws):
        """Host laws ([rows][width] dicts token -> probability, None for no law) into the round's laws (outside a
        capture: a host drafter's laws, tests). A sparse law keeps at most 64 tokens, in the order given."""
        found = torch.zeros(rows * width, self.buffer.shape[1], dtype=self.buffer.dtype)
        for r in range(rows):
            for c in range(width):
                law = laws[r][c]
                entry = found[r * width + c]
                if self.sparse:
                    if law is None:
                        entry[0] = -1
                        continue
                    if len(law) > MAX_TOP_K:
                        raise ValueError(f"a sparse law keeps at most {MAX_TOP_K} tokens, not {len(law)}")
                    entry[0] = len(law)
                    entry[1:1 + len(law)] = torch.tensor(list(law), dtype=torch.int32)
                    probabilities = torch.tensor(list(law.values()), dtype=torch.float32).view(torch.int32)
                    entry[1 + MAX_TOP_K:1 + MAX_TOP_K + len(law)] = probabilities
                elif law is not None:
                    for t, p in law.items():
                        entry[t] = p
        self.buffer[:rows * width].copy_(found)

    def as_dicts(self, rows, width):
        """The laws as host dicts token -> probability, [rows][width] (tests and diagnostics; synchronizes)."""
        laws = self.view(rows, width).cpu()
        out = []
        for r in range(rows):
            row = []
            for c in range(width):
                if self.sparse:
                    count = int(laws[r, c, 0])
                    ids = laws[r, c, 1:1 + max(count, 0)].tolist()
                    p = laws[r, c, 1 + MAX_TOP_K:1 + MAX_TOP_K + max(count, 0)].view(torch.float32).tolist()
                    row.append(dict(zip(ids, p)) if count >= 0 else None)
                else:
                    nz = laws[r, c].nonzero().flatten().tolist()
                    row.append({t: float(laws[r, c, t]) for t in nz})
            out.append(row)
        return out


class RoundState:
    """The sampler's buffers for rounds of up to `rows` rows at up to `widest` drafts over `vocab` logits: each slot's
    staged rules and every scratch a round's launches read, allocated once."""

    def __init__(self, rows, widest, vocab, device, logprobs=0, kernel="native"):
        self.rows, self.widest, self.vocab, self.device = rows, widest, vocab, torch.device(device)
        self.top, self.kernel = int(logprobs), kernel
        if not 0 <= self.top <= MAX_TOP_K:
            raise ValueError(f"rounds report up to {MAX_TOP_K} alternatives, not {logprobs}")
        if kernel not in KERNELS:
            raise ValueError(f"rounds sample with {', '.join(KERNELS)}, not {kernel!r}")
        self.native = native(self.device) if kernel == "native" else None
        cols = widest + 1
        dev = dict(device=self.device)
        # The rules of each slot for the torch law: temperature, top_p, min_p, top_k (FP64); seed, stream (int64).
        self.h_rules = torch.zeros(rows, 4, dtype=torch.float64)
        self.h_ints = torch.zeros(rows, 2, dtype=torch.int64)
        self.rules = torch.zeros(rows, 4, dtype=torch.float64, **dev)
        self.ints = torch.zeros(rows, 2, dtype=torch.int64, **dev)
        self.drafts = torch.zeros(rows * max(widest, 1), dtype=torch.int32, **dev)
        self.predictions = torch.zeros(rows * cols, dtype=torch.int32, **dev)
        self.tokens = torch.zeros(rows, dtype=torch.int32, **dev)
        self.chosen = torch.zeros(rows * cols, dtype=torch.float32, **dev)
        self.top_ids = torch.zeros(rows * cols, max(self.top, 1), dtype=torch.int32, **dev)
        self.top_values = torch.zeros(rows * cols, max(self.top, 1), dtype=torch.float32, **dev)
        if self.native is not None:
            layout = self.native.layout()
            self.request_bytes, self.position_word = layout["request"], layout["position"] // 8
            self.requests = torch.zeros(rows * self.request_bytes, dtype=torch.uint8, **dev)
            self.scratch = torch.zeros(self.native.scratch_bytes(vocab, rows * cols) // 4 + 2, dtype=torch.float32, **dev)
            self.lp_scratch = torch.zeros(self.native.logprob_scratch_bytes(vocab, rows * cols, self.top) // 8 + 1,
                                          dtype=torch.float64, **dev)
            self.laws = DraftLaws(rows, widest, vocab, self.device, layout["sparse"] // 4)
            self.sparse_column = torch.zeros(rows, layout["sparse"] // 4, dtype=torch.int32, **dev)
            for slot in range(rows):
                self._request(slot, {})
        else:
            self.laws = DraftLaws(rows, widest, vocab, self.device)

    def _request(self, slot, rule):
        packed = self.native.pack_request(rule.get("temperature", 0.0), rule.get("top_p", 1.0), rule.get("min_p", 0.0),
                                          rule.get("top_k", 0), rule.get("seed", 0), rule.get("stream", 0), 0)
        b = self.request_bytes
        self.requests[slot * b:(slot + 1) * b].copy_(packed)

    def stage(self, slot, params):
        if not 0 <= slot < self.rows:
            raise ValueError(f"round slot {slot} outside the {self.rows} reserved")
        if params.get("allowed") is not None:
            raise ValueError("a constrained request decodes plainly, outside rounds: it is never staged")
        temperature = float(params.get("temperature", 0.0))
        top_p, top_k, min_p = float(params.get("top_p", 1.0)), int(params.get("top_k", 0)), float(params.get("min_p", 0.0))
        if temperature > 0 and (not 0 < top_p <= 1 or top_k < 0 or not 0 <= min_p <= 1):
            raise ValueError("top_p in (0, 1], top_k non-negative and min_p in [0, 1]")
        seed, stream = int(params.get("seed", 0)), int(params.get("stream", 0))
        if not 0 <= seed < 1 << 64 or not 0 <= stream <= WORD:
            raise ValueError("seed is unsigned 64-bit and stream unsigned 32-bit")
        rule = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": min_p, "seed": seed,
                "stream": stream}
        self.h_rules[slot] = torch.tensor([temperature, top_p, min_p, float(top_k)], dtype=torch.float64)
        self.h_ints[slot] = torch.tensor([_signed(seed), stream], dtype=torch.int64)
        self.rules[slot].copy_(self.h_rules[slot])
        self.ints[slot].copy_(self.h_ints[slot])
        if self.native is not None:
            self._request(slot, rule)

    def positions(self, view):
        """The draw position of each row's column 0 (view.out_count) into the staged requests, on the device."""
        if self.native is not None:
            words = self.requests.view(torch.int64).view(self.rows, self.request_bytes // 8)
            words[:view.rows, self.position_word].copy_(_out_count(view)[:view.rows])

    def check(self, view):
        if view.rows > self.rows or view.width > self.widest:
            raise ValueError(f"a round of {view.rows} rows at width {view.width} exceeds the sampler's reserve of "
                             f"{self.rows} rows at width {self.widest}")


def _out_count(view):
    """Each row's output count before column 0 (the round view's out_count): the draw position of column 0."""
    return view.out_count


class RoundSampling:
    """The sampler's rounds:sampler extension over a RoundState; CoreSampler mixes it in."""

    _round, kernel = None, "native"

    def __init__(self):
        self._staged = {}   # slot -> the params staged there, kept across a reserve() of other buffers

    def reserve(self, rows, widest, vocab, device, logprobs=0):
        """Every buffer of rounds of up to `rows` rows at up to `widest` drafts over `vocab` logits on `device`, with
        `logprobs` ranked alternatives per emitted column (0: the chosen token's only); call it once before capturing
        rounds. Returns the round state."""
        found = self._round
        if found is None or (found.rows, found.widest, found.vocab, found.device, found.top, found.kernel) != (
                rows, widest, vocab, torch.device(device), int(logprobs), self.kernel):
            self._round = RoundState(rows, widest, vocab, device, logprobs, self.kernel)
            for slot, params in sorted(self._staged.items()):   # the rules staged before, in the new buffers
                if slot < rows:
                    self._round.stage(slot, params)
        return self._round

    def _state(self, view, vocab, drafting=False):
        if self._round is None:
            raise ValueError("reserve(rows, widest, vocab, device) the sampler's round buffers before a round")
        self._round.check(view)
        if vocab != self._round.vocab and not (drafting and vocab < self._round.vocab):
            raise ValueError(f"logits of {vocab} columns for round buffers over {self._round.vocab}")
        return self._round

    def stage(self, slot, params):
        """The sampling rules of the request in round slot `slot` (select()'s params: temperature, top_p, top_k,
        min_p, seed, stream; its draw positions come from each round's view.out_count), kept in device buffers the
        round reads. Outside a capture, after reserve()."""
        if self._round is None:
            raise ValueError("reserve(rows, widest, vocab, device) the sampler's round buffers before staging")
        self._round.stage(slot, params)
        self._staged[int(slot)] = dict(params)

    def laws(self, rows, width):
        """The round's draft laws (DraftLaws, reserved with the round): what a drafter writes and view.draft_laws is."""
        if self._round is None:
            raise ValueError("reserve the sampler's round buffers first")
        return self._round.laws

    def draft_device(self, view, logits, depth, targets=None):
        """The draft of each row at `depth` (drafts[:, depth], output position view.out_count + depth) from its
        logits row ([rows][vocab] FP32): the argmax for greedy rows and in greedy rounds; in sampled rounds a draw
        from the row's processed law with Core's `draft` uniform, the law kept in view.draft_laws (set to the
        sampler's DraftLaws). targets ([vocab] int64), for a head over a draft vocabulary or a subset of the target's:
        the target id of each logits column, which the draft and its law are mapped to. Returns [rows] int64 target
        ids; capture-safe."""
        rows = view.rows
        if view.mode != "sampled":
            found = torch.argmax(logits, dim=-1)
            return found if targets is None else targets[found]
        if view.draft_laws is None:
            raise ValueError("a sampled round keeps each draft's law: the round driver sets view.draft_laws to the "
                             "sampler's laws(rows, widest)")
        x = logits if logits.dtype == torch.float32 and logits.stride(-1) == 1 else logits.float().contiguous()
        state = self._state(view, x.shape[-1], drafting=targets is not None)
        laws = view.draft_laws.view(rows, view.width)
        if state.native is not None:
            state.positions(view)
            state.native.sample(x, x.shape[-1], state.requests, rows, 1, depth, DRAFT, state.tokens[:rows],
                                sparse=state.sparse_column[:rows], scratch=state.scratch)
            column = state.sparse_column[:rows]
            tokens = state.tokens[:rows].long()
            if targets is not None:
                ids = column[:, 1:1 + MAX_TOP_K].long().clamp(0, x.shape[-1] - 1)
                column[:, 1:1 + MAX_TOP_K] = targets[ids].to(torch.int32)
                tokens = torch.where(tokens >= 0, targets[tokens.clamp(min=0)], tokens)   # -1: nothing selectable
            laws[:, depth].copy_(column)
            return tokens
        tokens, law = _torch_sample(x, state, rows, _out_count(view)[:rows] + depth, DRAFT)
        if targets is not None:
            mapped = torch.zeros(rows, state.vocab, dtype=law.dtype, device=law.device)
            law = mapped.scatter_add_(1, targets[None, :].expand(rows, -1), law)
            tokens = torch.where(tokens >= 0, targets[tokens.clamp(min=0)], tokens)
        laws[:, depth].copy_(law)
        return tokens

    def predictions_device(self, view, logits, draft_laws=None):
        """The token each verify column of the round predicts, written to view.predictions[:, :width + 1] (and
        returned): logits [rows][width + 1][vocab] FP32 (or [rows * (width + 1)][vocab]); the argmax in greedy
        rounds and for greedy rows, rejection sampling against draft_laws (the round's DraftLaws, or dense
        [rows][width][vocab] FP32 rows, or None for greedy drafts) for sampled rows, by Core's law and draws as
        predictions(). A column whose draft has no law (a cleared or open sparse law, no dense row) or no mass in it
        takes the target's own draw. The log-probabilities are logprobs_device's, after the accept. Capture-safe."""
        rows, width = view.rows, view.width
        cols = width + 1
        vocab = logits.shape[-1]
        x = logits.reshape(rows * cols, vocab)
        x = x if x.dtype == torch.float32 and x.stride(-1) == 1 else x.float().contiguous()
        out = view.predictions[:rows, :cols]
        if view.mode != "sampled":
            out.copy_(torch.argmax(x, dim=-1).view(rows, cols))
            return out
        state = self._state(view, vocab)
        if getattr(view, "tree_depths", None) is not None:   # draft-trees' rounds (a tree's node depths)
            return self._tree_predictions(view, state, x, out)
        if state.native is not None:
            state.positions(view)
            drafts = state.drafts[:rows * width]
            if width:
                drafts.view(rows, width).copy_(view.drafts[:rows, :width])
            sparse = dense = None
            if isinstance(draft_laws, DraftLaws):
                if draft_laws.sparse:
                    sparse = draft_laws.view(rows, width)
                else:
                    dense = draft_laws.view(rows, width)
            elif draft_laws is not None:
                dense = draft_laws[:rows, :width].contiguous().float()
            state.native.verify(x, vocab, state.requests, rows, cols, drafts if width else None, sparse, dense,
                                state.predictions[:rows * cols], state.scratch)
            out.copy_(state.predictions[:rows * cols].view(rows, cols))
            return out
        dense = None
        if isinstance(draft_laws, DraftLaws):
            dense = draft_laws.view(rows, width) if not draft_laws.sparse else None
        elif draft_laws is not None:
            dense = draft_laws[:rows, :width].float()
        out.copy_(_torch_verify(x.view(rows, cols, vocab), state, rows, width, view.drafts[:rows, :width],
                                _out_count(view)[:rows], dense))
        return out

    def _tree_predictions(self, view, state, x, out):
        """A sampled round over a token tree (view.tree_parents, view.tree_depths; node 0 the pending token): each
        node predicts the target's own draw at output position out_count + its depth (tree children are deterministic,
        so a node stands exactly when it equals that draw, and the walk emits what plain sampling would)."""
        rows, width = view.rows, view.width
        nodes = width + 1
        depths = view.tree_depths[:rows, :nodes]
        if state.native is not None:
            state.positions(view)
            index = torch.arange(nodes, device=x.device)[None, :]
            parents = torch.cat([torch.zeros_like(depths[:, :1]), view.tree_parents[:rows, :width]], dim=1)
            bit = torch.ones_like(depths) << index
            masks = bit
            for _ in range(width):   # each pass adds one more ancestor to every node's path mask
                masks = bit | torch.where(index > 0, masks.gather(1, parents), torch.zeros_like(masks))
            tokens = torch.cat([view.pending[:rows, None], view.drafts[:rows, :width]], dim=1).clamp(min=0)
            found = state.predictions[:rows * nodes]
            state.native.sample(x, x.shape[-1], state.requests, rows, nodes, 0, SAMPLE, found, scratch=state.scratch,
                                tree=masks.contiguous(), tree_tokens=tokens.to(torch.int32).contiguous())
            out.copy_(found.view(rows, nodes))
        else:
            positions = (_out_count(view)[:rows, None] + depths).reshape(-1)
            tokens, _ = _torch_sample(x, state, rows, positions, SAMPLE, repeat=nodes)
            out.copy_(tokens.view(rows, nodes))
        return out

    def logprobs_device(self, view, logits, tokens):
        """Core's TokenLogprob of tokens [rows][width + 1] over the raw logits rows ([rows * (width + 1)][vocab] or
        [rows][width + 1][vocab], each row the one that predicted its token: the round driver gathers a tree's along
        its path): view.logprobs[:, :width + 1] and, where the view keeps them (view.top_ids, view.top_logprobs), the
        ranked alternatives, in Core's order (lower id first among equals). The round driver calls it after the
        accept, over the emitted tokens. Capture-safe."""
        rows, cols = view.rows, view.width + 1
        vocab = logits.shape[-1]
        x = logits.reshape(rows * cols, vocab)
        x = x if x.dtype == torch.float32 and x.stride(-1) == 1 else x.float().contiguous()
        state = self._round
        if state is not None and state.native is not None and rows <= state.rows and cols <= state.widest + 1:
            chosen = state.predictions[:rows * cols]
            chosen.view(rows, cols).copy_(tokens[:rows, :cols])
            n = state.top
            state.native.logprobs(x, vocab, chosen, n, state.chosen[:rows * cols],
                                  state.top_ids[:rows * cols] if n else None,
                                  state.top_values[:rows * cols] if n else None, state.lp_scratch)
            self._write_logprobs(view, state, rows, cols)
            return
        values = torch.log_softmax(x.double(), dim=-1)
        view.logprobs[:rows, :cols].copy_(values.gather(1, tokens[:rows, :cols].reshape(-1, 1).long()).view(rows, cols))
        if view.top_ids is not None and view.top_ids.shape[-1]:
            k = view.top_ids.shape[-1]
            ranked, order = values.sort(dim=-1, descending=True, stable=True)   # Core's order: lower id first
            view.top_ids[:rows, :cols].copy_(order[:, :k].view(rows, cols, k))
            view.top_logprobs[:rows, :cols].copy_(ranked[:, :k].view(rows, cols, k))

    def _write_logprobs(self, view, state, rows, cols):
        view.logprobs[:rows, :cols].copy_(state.chosen[:rows * cols].view(rows, cols))
        if view.top_ids is not None and state.top:
            k = min(view.top_ids.shape[-1], state.top)
            view.top_ids[:rows, :cols, :k].copy_(state.top_ids[:rows * cols, :k].view(rows, cols, k))
            view.top_logprobs[:rows, :cols, :k].copy_(state.top_values[:rows * cols, :k].view(rows, cols, k))


# --- The torch law (no CUDA extension): Core's steps 7 to 12 over sorted rows in FP64 -------------------------------

def _law(x, temperature, top_p, min_p, top_k):
    """Sorted kept weights of rows x ([n][vocab]) under per-row rules ([n] tensors): (order, weights, total, greedy)."""
    values = x.double()
    ranked, order = torch.sort(values, dim=-1, descending=True, stable=True)
    vocab = ranked.shape[-1]
    position = torch.arange(vocab, device=x.device)[None, :]
    finite = torch.isfinite(ranked).sum(-1)
    greedy = ~(temperature > 0)
    t = torch.where(greedy, torch.ones_like(temperature), temperature)[:, None]
    weights = torch.exp((ranked - ranked[:, :1]) / t)
    k = top_k.long()
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
    weights = torch.where(greedy[:, None], (position == 0).double().expand_as(weights), weights)
    return order, weights, weights.sum(-1), greedy


def _walk(weights, target):
    """The first index whose cumulative weight exceeds target ([n][vocab], [n]), the last kept one otherwise."""
    cumulative = torch.cumsum(weights, -1)
    last = (weights > 0).long().cumsum(-1).argmax(-1)
    pick = torch.searchsorted(cumulative, target[:, None], right=True)[:, 0]
    return torch.minimum(pick, last)


def _row_rules(state, rows, repeat):
    rules = state.rules[:rows].repeat_interleave(repeat, 0)
    ints = state.ints[:rows].repeat_interleave(repeat, 0)
    return rules[:, 0], rules[:, 1], rules[:, 2], rules[:, 3], ints[:, 0], ints[:, 1]


def _torch_sample(x, state, rows, positions, purpose, repeat=1):
    """One draw per row of x ([rows * repeat][vocab], `repeat` rows per request) at positions [rows * repeat]:
    (tokens int64, dense laws [rows * repeat][vocab])."""
    temperature, top_p, min_p, top_k, seed, stream = _row_rules(state, rows, repeat)
    order, weights, total, greedy = _law(x, temperature, top_p, min_p, top_k)
    u = uniforms(seed, positions, stream, purpose)
    pick = _walk(weights, u * total)
    tokens = order.gather(1, pick[:, None])[:, 0]
    law = torch.zeros_like(weights).scatter_(1, order, weights / total[:, None]).float()
    return tokens, law


def _torch_verify(x, state, rows, width, drafts, out_count, dense):
    """sampling.verify over x [rows][width + 1][vocab] in torch: predictions [rows][width + 1] int64."""
    cols = width + 1
    vocab = x.shape[-1]
    flat = x.reshape(rows * cols, vocab)
    temperature, top_p, min_p, top_k, seed, stream = _row_rules(state, rows, cols)
    column = torch.arange(cols, device=x.device).repeat(rows)
    positions = out_count.repeat_interleave(cols) + column
    order, weights, total, greedy = _law(flat, temperature, top_p, min_p, top_k)
    own = order.gather(1, _walk(weights, uniforms(seed, positions, stream, SAMPLE) * total)[:, None])[:, 0]
    if not width or dense is None:
        return own.view(rows, cols)
    d = torch.cat([drafts, torch.full_like(drafts[:, :1], -1)], dim=1).reshape(-1)
    q_rows = torch.cat([dense, torch.zeros_like(dense[:, :1])], dim=1).reshape(rows * cols, vocab).double()
    dd = d.clamp(0, vocab - 1)
    p_sorted = weights / total[:, None]
    p_full = torch.zeros_like(p_sorted).scatter_(1, order, p_sorted)
    pd = p_full.gather(1, dd[:, None])[:, 0]
    qd = q_rows.gather(1, dd[:, None])[:, 0]
    # A draft its law gives no mass (no law: a copy) takes the target's own draw, as a greedy draft does.
    valid = (d >= 0) & (d < vocab) & (column < width) & ~greedy & (qd > 0)
    accept = uniforms(seed, positions, stream, ACCEPT) * qd < pd
    residual = (p_sorted - q_rows.gather(1, order)).clamp(min=0) * (weights > 0)
    mass = residual.sum(-1)
    rejected = order.gather(1, _walk(residual, uniforms(seed, positions, stream, RESIDUAL) * mass)[:, None])[:, 0]
    rejected = torch.where(mass > 0, rejected, dd)
    found = torch.where(valid, torch.where(accept, dd, rejected), own)
    return found.view(rows, cols)
