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


def score_error(case_values):
    """Relative standard error of a geometric mean of per-case centres, from each case's measured repetitions: the
    spread of their logarithms over the square root of their count, combined over the cases. None when a case has
    fewer than two. Sampled repetitions decode different texts, so this includes how much the text moves the score."""
    terms = []
    for values in case_values:
        logs = [math.log(v) for v in values if v]
        if len(logs) < 2:
            return None
        mean = sum(logs) / len(logs)
        terms.append(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1) / len(logs))
    return math.sqrt(sum(terms)) / len(terms) if terms else None


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


def processed_distribution(logprobs, settings):
    """The distribution a sampler draws from at one position, over the alternatives an engine reported: their raw
    log-probabilities, highest first (a final entry may lump the unreported tail). Temperature scales them, then
    top_k keeps the k most likely, top_p the smallest prefix holding top_p of the mass, and min_p those at least
    min_p times the most likely. Returns probabilities aligned with `logprobs` (0 for dropped entries)."""
    temperature = float(settings.get("temperature") or 1.0)
    order = sorted(range(len(logprobs)), key=lambda i: (-logprobs[i], i))
    kept = order[:int(settings["top_k"])] if settings.get("top_k") else order
    scaled = {i: logprobs[i] / temperature for i in kept}
    top = max(scaled.values())
    weights = {i: math.exp(v - top) for i, v in scaled.items()}
    total = sum(weights.values())
    probs = {i: w / total for i, w in weights.items()}
    if settings.get("top_p") is not None and float(settings["top_p"]) < 1:
        mass, prefix = 0.0, []
        for i in kept:
            prefix.append(i)
            mass += probs[i]
            if mass >= float(settings["top_p"]):
                break
        probs = {i: probs[i] / mass for i in prefix}
    if settings.get("min_p"):
        floor = float(settings["min_p"]) * max(probs.values())
        probs = {i: p for i, p in probs.items() if p >= floor}
        total = sum(probs.values())
        probs = {i: p / total for i, p in probs.items()}
    return [probs.get(i, 0.0) for i in range(len(logprobs))]


TAIL_TOLERANCE = 1e-3  # the most the unreported tail's processed mass may be uncertain (protocol tail_tolerance)
TOP_P_MARGIN = 1e-6    # cumulative mass this close to top_p leaves the nucleus edge to rounding


def _tail_weight(lumped, count, ratio, temperature):
    """Bounds of sum (q / p0) ** (1 / temperature) over unreported tokens q holding `lumped` raw mass (relative to the
    likeliest token's p0), each at most the last reported token (`ratio` = p_last / p0), `count` of them (None:
    unknown). Each q <= p_last bounds one side; spreading the mass evenly over `count` tokens bounds the other."""
    if lumped <= 0:
        return 0.0, 0.0
    a = 1.0 / temperature
    if a == 1.0:
        return lumped, lumped
    capped = lumped * ratio ** (a - 1)     # every q at p_last: the most for a > 1, the least for a < 1
    spread = count ** (1 - a) * lumped ** a if count else None
    if a > 1:
        return (spread or 0.0), capped
    return capped, (spread if spread is not None else math.inf)


def resolved_distribution(logprobs, settings, vocab=None, tolerance=TAIL_TOLERANCE):
    """The processed distribution at one sampled position, over the alternatives an engine reported (their raw
    log-probabilities, the model's own before temperature and filters) plus one last entry for the unreported rest of
    the vocabulary, or None when the reported alternatives do not determine it. Temperature scales the logits, then
    top_k keeps the k most likely, top_p the smallest prefix holding top_p of the mass, and min_p those at least min_p
    times the most likely. The unreported tokens are each at most as likely as the last reported one; their raw mass
    is what the reported ones leave, and `vocab` (the vocabulary size) bounds how far it spreads. A position is
    resolved when top_k keeps only reported tokens, or the top_p or min_p edge falls among them whatever the
    unreported tokens hold, or the unreported tail's processed mass is known within `tolerance` of the whole
    (temperature 1 without filters: exactly). Returns probabilities aligned with `logprobs`, then the tail's."""
    n = len(logprobs)
    if not n:
        return None
    temperature = float(settings.get("temperature") or 1.0)
    top_k = int(settings.get("top_k") or 0)
    top_p = float(settings["top_p"]) if settings.get("top_p") is not None else 1.0
    min_p = float(settings.get("min_p") or 0.0)
    order = sorted(range(n), key=lambda i: (-logprobs[i], i))
    ranked = [float(logprobs[i]) for i in order]
    kept = min(n, top_k) if top_k else n
    weights = [math.exp((v - ranked[0]) / temperature) for v in ranked[:kept]]   # the likeliest weighs 1
    p0, ratio = math.exp(ranked[0]), math.exp(ranked[-1] - ranked[0])
    rest = max(0.0, 1.0 - sum(math.exp(v) for v in ranked))
    low = high = 0.0
    if not (top_k and top_k <= n) and rest > 0:
        others = None if vocab is None else max(0, int(vocab) - n)
        allowed = others if not top_k else min(top_k - n, others) if others is not None else top_k - n
        # top_k beyond the reported list keeps an unknown part of the tail: from none of it to `allowed` tokens.
        lumped_low, lumped_high = (rest, rest) if not top_k else (0.0, min(rest, allowed * math.exp(ranked[-1])))
        low = _tail_weight(lumped_low / p0, allowed, ratio, temperature)[0]
        high = _tail_weight(lumped_high / p0, allowed, ratio, temperature)[1]
    if min_p and ratio ** (1 / temperature) < min_p * (1 - TOP_P_MARGIN):
        low = high = 0.0   # every unreported token is below the min_p floor
    total = sum(weights)
    if top_p < 1:
        def edge(normalizer):
            mass = 0.0
            for i, w in enumerate(weights):
                mass += w / normalizer
                if abs(mass - top_p) <= TOP_P_MARGIN:
                    return "ambiguous"
                if mass > top_p:
                    return i
            return None   # the nucleus reaches into the unreported tokens
        first, last = edge(total + low), edge(total + high)
        if first != last or first in (None, "ambiguous"):
            return None
        weights, low = weights[:first + 1], 0.0
        high = 0.0
        total = sum(weights)
    if high - low > tolerance * (total + low):
        return None
    tail = (low + high) / 2
    if min_p:
        weights = [w if w >= min_p * (1 + TOP_P_MARGIN) else (None if w > min_p * (1 - TOP_P_MARGIN) else 0.0)
                   for w in weights]
        if None in weights:
            return None   # a token at the min_p edge, within rounding
    total = sum(weights) + tail
    probs = [0.0] * n
    for rank, w in enumerate(weights):
        probs[order[rank]] = w / total
    return probs + [tail / total]


def sampling_calibration(rows, settings, seed=0, vocab=None, tolerance=TAIL_TOLERANCE):
    """Whether sampled tokens follow the distribution the settings define (protocol sampling_check). rows: one per
    sampled token, {"top": raw log-probabilities of the reported alternatives, "chosen": the emitted token's index
    among them (None when it is not reported, so it is one of the unreported rest)}. At every position whose processed
    distribution the alternatives determine (resolved_distribution) the token gets a randomized probability integral
    transform u = (mass of the more likely alternatives, the unreported rest last) + V * p(token), V uniform: u is
    uniform on [0, 1] exactly when every token was drawn from its processed distribution. A sampler that favours
    likely tokens (greedy acceptance of drafts) pulls the mean of u below 1/2. Returns the judged token count, the
    mean of u and its z-score against 1/2, the Kolmogorov-Smirnov distance from uniform, the tokens outside the
    processed support and the positions left unresolved."""
    import random
    rng = random.Random(seed)
    us, outside, unresolved = [], 0, 0
    for row in rows:
        top = list(row["top"])
        probs = resolved_distribution(top, settings, vocab, tolerance) if top else None
        if probs is None:
            unresolved += 1
            continue
        chosen = row.get("chosen")
        chosen = len(top) if chosen is None else chosen
        if probs[chosen] <= 0:
            outside += 1
            continue
        order = sorted(range(len(top)), key=lambda i: (-probs[i], i)) + [len(top)]
        before = sum(probs[i] for i in order[:order.index(chosen)])
        us.append(min(1.0, before + rng.random() * probs[chosen]))
    n = len(us)
    if not n:
        return {"tokens": 0, "outside": outside, "unresolved": unresolved, "mean_u": None, "z": None, "ks": None}
    mean = sum(us) / n
    ordered = sorted(us)
    ks = max(max((i + 1) / n - u, u - i / n) for i, u in enumerate(ordered))
    return {"tokens": n, "outside": outside, "unresolved": unresolved, "mean_u": mean,
            "z": (mean - 0.5) / math.sqrt(1 / (12 * n)), "ks": ks}
