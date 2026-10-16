"""Speed and quality metrics. Pure Python so results can be computed on any host.

Quality has two forms: `quality_path` compares the suite's compact reference (top-k log-probabilities
plus a lumped tail, a lower bound on KL), and `log_softmax`, `kl_divergence`, `Summary` and `compare` compare
full logit rows when an engine can supply them (FP64; probe/logprob_compare.cu is the device reference).
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


def speed_sample(prompt_tokens, tokens, started, first, last, rounds):
    """One timed request. Times are host seconds observed by the harness when the
    engine delivered tokens: started = call, first = first token, last = last token."""
    n = len(tokens)
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


def lumped_kl(reference_logprobs, engine_logprobs):
    """KL(reference || engine) over the reference's top-k tokens plus one lumped tail bucket.

    reference_logprobs and engine_logprobs are log-probabilities of the same k token ids.
    The tail bucket holds each distribution's remaining mass. By the log-sum inequality this
    is a lower bound on the full-vocabulary KL, and it is exact when the tail is negligible.
    """
    kl = 0.0
    p_sum = q_sum = 0.0
    for lp, lq in zip(reference_logprobs, engine_logprobs):
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


def quality_path(reference_targets, engine_rows):
    """Compare one scoring path with the reference.

    reference_targets: list of {"target": id, "logprob": float, "top_ids": [...], "top_logprobs": [...]}
    engine_rows: list of {"target_logprob": float, "top_logprobs": [...], "argmax": id or None}
    with top_logprobs aligned to the reference's top_ids.
    """
    if len(reference_targets) != len(engine_rows):
        raise ValueError(f"engine scored {len(engine_rows)} targets, reference has {len(reference_targets)}")
    ref_nll = [-r["logprob"] for r in reference_targets]
    eng_nll = [-e["target_logprob"] for e in engine_rows]
    kls = [lumped_kl(r["top_logprobs"], e["top_logprobs"]) for r, e in zip(reference_targets, engine_rows)]
    agree = 0
    for r, e in zip(reference_targets, engine_rows):
        best = e.get("argmax")
        if best is None:  # best of the reference's top-k under the engine's distribution
            best = r["top_ids"][max(range(len(r["top_ids"])), key=lambda i: e["top_logprobs"][i])]
        agree += int(best == r["top_ids"][0])
    ref_ppl = math.exp(sum(ref_nll) / len(ref_nll))
    ppl = math.exp(sum(eng_nll) / len(eng_nll))
    return {
        "targets": len(kls),
        "perplexity": ppl,
        "reference_perplexity": ref_ppl,
        "ppl_ratio": ppl / ref_ppl,
        "kld_mean": sum(kls) / len(kls),
        "kld_p99": percentile(kls, 0.99),
        "kld_max": max(kls),
        "top1_agreement": agree / len(kls),
        "max_abs_target_logprob_diff": max(abs(a - b) for a, b in zip(ref_nll, eng_nll)),
    }


def budget_used(path_metrics, thresholds):
    """Largest fraction of the drift budget a path uses (1.0 at a limit); None when nothing is limited."""
    used = []
    for key, limit in thresholds.items():
        name = key[:-4] if key.endswith("_max") else key
        value = path_metrics.get(name)
        if value is None:
            continue
        # A perplexity ratio drifts from 1, so its budget is the distance from 1.
        base = 1.0 if name == "ppl_ratio" else 0.0
        room = limit - base
        used.append(max(value - base, 0.0) / room if room > 0 else (0.0 if value <= limit else float("inf")))
    return max(used) if used else None


def judge(path_metrics, thresholds):
    failures = []
    for key, limit in thresholds.items():
        name = key[:-4] if key.endswith("_max") else key
        value = path_metrics.get(name)
        if value is None or value > limit:
            failures.append(f"{name} {value} > {limit}")
    return failures
