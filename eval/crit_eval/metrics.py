"""Speed and quality metrics. Pure Python so results can be computed on any host.

Quality has two forms: `text_quality` compares an engine's scores of a fixed text with the reference's, position
by position (top-k on both sides plus a lumped tail, a lower bound on KL), and `log_softmax`, `kl_divergence`,
`Summary` and `compare` compare full logit rows when an engine can supply them (FP64; probe/logprob_compare.cu is
the device reference).
"""
import math


def median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def geometric_mean(values):
    values = [v for v in values if v]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def percentile(values, fraction):
    """Nearest-rank percentile (fraction in (0, 1])."""
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))]


def speed_sample(prompt_tokens, n, started, first, last, rounds):
    """One timed request of n output tokens. Times are host seconds observed by the harness when the
    engine delivered tokens: started = call, first = first token, last = last token."""
    decode_seconds = last - first
    return {
        "output_tokens": n,
        "ttft_ms": (first - started) * 1000,
        "request_ms": (last - started) * 1000,
        "prefill_tps": prompt_tokens / (first - started) if first > started else None,
        # The first token comes from the prefill logits; decode covers the other n - 1.
        "decode_tps": (n - 1) / decode_seconds if n > 1 and decode_seconds > 0 else None,
        "rounds": rounds,
        "tokens_per_round": n / rounds if rounds else None,
        # Time per target pass (a whole speculative round); one token per pass without rounds.
        "round_ms": (decode_seconds * 1000 / rounds if rounds else
                     decode_seconds * 1000 / (n - 1) if n > 1 and decode_seconds > 0 else None),
    }


def partial_kl(reference_ids, reference_logprobs, engine):
    """KL(reference || engine) over the tokens in both top-k lists plus one lumped bucket for the rest.

    reference_ids and reference_logprobs: the reference's top-k; engine: {token id: log-probability} of the
    engine's top-k. Each token in both is its own bucket and everything else one more, holding each
    distribution's remaining mass. By the log-sum inequality this is a lower bound on the full-vocabulary KL,
    and it is exact when the rest is negligible.
    """
    kl = 0.0
    p_sum = q_sum = 0.0
    for token, lp in zip(reference_ids, reference_logprobs):
        lq = engine.get(token)
        if lq is None:
            continue
        p = math.exp(lp)
        p_sum += p
        q_sum += math.exp(lq)
        kl += p * (lp - lq)
    p_tail, q_tail = max(0.0, 1.0 - p_sum), max(0.0, 1.0 - q_sum)
    if p_tail > 1e-12:
        kl += p_tail * (math.log(p_tail) - math.log(max(q_tail, 1e-30)))
    return max(0.0, kl)


def log_softmax(logits, valid=None):
    """FP64 log-probabilities over the first `valid` entries (all when None); padding past `valid` is never read."""
    n = len(logits) if valid is None else valid
    if n < 1 or n > len(logits):
        raise ValueError("valid must cover 1..len(logits) entries")
    row = [float(x) for x in logits[:n]]
    if not all(math.isfinite(x) for x in row):
        raise ValueError("logits must be finite")
    top = max(row)
    lse = top + math.log(sum(math.exp(x - top) for x in row))
    return [x - lse for x in row]


def nll(logprobs, target):
    """Negative log-likelihood of `target` under a log-probability row."""
    if not 0 <= target < len(logprobs):
        raise ValueError("target outside the valid outputs")
    return -logprobs[target]


def kl_divergence(reference, logprobs):
    """Full-vocabulary KL(reference || candidate) in nats, clamped at zero: sum p_ref (log p_ref - log p)."""
    if len(reference) != len(logprobs):
        raise ValueError("rows must cover the same outputs")
    return max(0.0, sum(math.exp(r) * (r - q) for r, q in zip(reference, logprobs)))


class Summary:
    """Streaming per-target NLL and KL totals; quantiles keep every value (bounded by the corpus)."""

    def __init__(self):
        self.count, self.nll_sum, self.kld = 0, 0.0, []

    def add(self, nll_value, kld_value=None):
        if not math.isfinite(nll_value) or nll_value < 0:
            raise ValueError("NLL must be finite and non-negative")
        self.count += 1
        self.nll_sum += nll_value
        if kld_value is not None:
            if not math.isfinite(kld_value) or kld_value < 0:
                raise ValueError("KL divergence must be finite and non-negative")
            self.kld.append(kld_value)

    def perplexity(self):
        if not self.count:
            raise ValueError("no targets")
        return math.exp(self.nll_sum / self.count)

    def kld_mean(self):
        if not self.kld:
            raise ValueError("no KL divergence values")
        return math.fsum(self.kld) / len(self.kld)

    def kld_quantile(self, q):
        """Nearest-rank quantile (q in (0, 1]) of the KL divergence values."""
        if not self.kld or not 0 < q <= 1:
            raise ValueError("values and 0 < q <= 1 required")
        return percentile(self.kld, q)

    def kld_max(self):
        return max(self.kld)


def compare(reference_rows, candidate_rows, targets, valid=None):
    """Full-vocabulary comparison of one path with a reference on the same targets: summaries of
    both, with the candidate's per-target KL divergence from the reference."""
    if not len(reference_rows) == len(candidate_rows) == len(targets):
        raise ValueError("one reference row, candidate row and target per position")
    ref, cand = Summary(), Summary()
    for r_logits, c_logits, t in zip(reference_rows, candidate_rows, targets):
        r, c = log_softmax(r_logits, valid), log_softmax(c_logits, valid)
        ref.add(nll(r, t))
        cand.add(nll(c, t), kl_divergence(r, c))
    return ref, cand


def text_quality(reference_rows, engine_rows):
    """Compare an engine's scores of a fixed text with the reference's, position by position.

    reference_rows: per position {"logprob": the reference's log-probability of the text's token, "top_ids",
    "top_logprobs"}; engine_rows: per position {"logprob", "top": [[id, logprob], ...]}. Neither chose the token,
    so top-1 agreement is the share of positions where both rank the same token first. KL is partial_kl's: the
    top-k on both sides plus a lumped tail.
    """
    if len(reference_rows) != len(engine_rows) or not engine_rows:
        raise ValueError(f"the reference scored {len(reference_rows)} positions, the engine {len(engine_rows)}")
    ref_nll = [-r["logprob"] for r in reference_rows]
    eng_nll = [-e["logprob"] for e in engine_rows]
    kls = [partial_kl(r["top_ids"], r["top_logprobs"], dict(e["top"])) for r, e in zip(reference_rows, engine_rows)]
    ref_ppl = math.exp(sum(ref_nll) / len(ref_nll))
    ppl = math.exp(sum(eng_nll) / len(eng_nll))
    first = [max(e["top"], key=lambda pair: pair[1])[0] if e["top"] else None for e in engine_rows]
    return {
        "targets": len(kls),
        "perplexity": ppl,
        "reference_perplexity": ref_ppl,
        "ppl_ratio": ppl / ref_ppl,
        "kld_mean": sum(kls) / len(kls),
        "kld_p99": percentile(kls, 0.99),
        "kld_max": max(kls),
        "top1_agreement": sum(r["top_ids"][0] == f for r, f in zip(reference_rows, first)) / len(kls),
        "max_abs_target_logprob_diff": max(abs(a - b) for a, b in zip(ref_nll, eng_nll)),
    }


def log2_bucket(value, smallest):
    """The bucket of a position or offset (>= 1): 1..smallest-1, then powers of two (512-1023, 1024-2047, ...)."""
    if value < smallest:
        return f"1-{smallest - 1}"
    low = 1 << (int(value).bit_length() - 1)
    return f"{low}-{2 * low - 1}"


def bucketed(reference_rows, engine_rows, labels):
    """text_quality of the positions in each bucket ({label: quality}, in the labels' order of appearance)."""
    groups = {}
    for reference, engine, label in zip(reference_rows, engine_rows, labels):
        groups.setdefault(label, ([], []))
        groups[label][0].append(reference)
        groups[label][1].append(engine)
    return {label: text_quality(*rows) for label, rows in groups.items()}


def budget_used(quality, thresholds):
    """Largest fraction of the drift budget a set uses (1.0 at a limit); None when nothing is limited."""
    used = []
    for key, limit in thresholds.items():
        name = key[:-4] if key.endswith("_max") else key
        value = quality.get(name)
        if value is None:
            continue
        # A perplexity ratio drifts from 1, so its budget is the distance from 1.
        base = 1.0 if name == "ppl_ratio" else 0.0
        room = limit - base
        used.append(max(value - base, 0.0) / room if room > 0 else (0.0 if value <= limit else float("inf")))
    return max(used) if used else None


def judge(quality, thresholds):
    failures = []
    for key, limit in thresholds.items():
        name = key[:-4] if key.endswith("_max") else key
        value = quality.get(name)
        if value is None or value > limit:
            failures.append(f"{name} {value} > {limit}")
    return failures
