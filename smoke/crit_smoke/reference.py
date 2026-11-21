"""The checkpoint's own reference, which smoke's quality limits are relative to.

crit Eval's reference (`crit eval reference`, the checkpoint run teacher-forced in FP32 once per model) scores the
quality texts of its prepared data; `crit run ... recon --data DATA` copies a few of those texts (their token ids)
and the reference's log-probability of each written-answer token into recon.json (`gathered.smoke_reference`). The
`fluent` test scores the same ids and is judged by how far the engine lands from the checkpoint, not by how well
the checkpoint reads: a quantized checkpoint whose own reference reads prose at a perplexity of 74 passes, and an
engine that reads it worse than its checkpoint does still fails. Without a reference, fluent judges smoke's prose by
an absolute perplexity (ABSOLUTE_PERPLEXITY), as before.
"""
import json
import math
from pathlib import Path

# The fallback without a reference: smoke's prose read at most at this perplexity. A working 16-bit or 8-bit model
# reads it far lower; broken weights, dequantisation or norms read it at hundreds. A checkpoint that reads it higher
# itself fails every engine here, which is what a reference fixes.
ABSOLUTE_PERPLEXITY = 40.0
# Against the reference, per profile: the engine's perplexity of the answer tokens over the reference's, and the
# drift per target (|engine log-probability - reference log-probability| of each answer token, in nats) as its
# median and p90. loose sits above Eval's speed mode (ppl_ratio_max 1.4) and strict above preserve's (1.05): smoke is a
# floor that catches a broken component, never a gate Eval's own would pass an engine through. A wrong weight, scale
# or norm moves the median by nats; rounding of 4-bit activations moves single targets, which p90 bounds.
LIMITS = {
    "loose": {"ppl_ratio": 1.5, "median": 0.5, "p90": 2.0},
    "strict": {"ppl_ratio": 1.1, "median": 0.1, "p90": 0.5},
}


def validate(block):
    """A smoke reference block {source, perplexity, tokens, texts: [{id, set, prompt_tokens, ids, target}]}: each
    text's ids are its prompt's and its answer's, and target holds the reference's log-probability of every answer
    token (len(ids) - prompt_tokens of them)."""
    if not isinstance(block, dict) or not isinstance(block.get("texts"), list) or not block["texts"]:
        raise ValueError("a smoke reference holds texts: [{id, prompt_tokens, ids, target}]")
    for text in block["texts"]:
        ids, target, prompt = text.get("ids"), text.get("target"), text.get("prompt_tokens")
        if not (isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids)):
            raise ValueError(f"smoke reference text {text.get('id')!r}: ids must be token ids")
        if type(prompt) is not int or not 1 <= prompt < len(ids):
            raise ValueError(f"smoke reference text {text.get('id')!r}: prompt_tokens must leave answer tokens")
        if not (isinstance(target, list) and len(target) == len(ids) - prompt
                and all(isinstance(v, (int, float)) and math.isfinite(v) and v <= 1e-4 for v in target)):
            raise ValueError(f"smoke reference text {text.get('id')!r}: target must hold a log-probability for "
                             "each answer token")
    return block


def load(path):
    """The smoke reference in `path`: a run's recon.json (its gathered.smoke_reference) or the block itself.
    ValueError when it holds none."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and value.get("record_kind") == "model_recon":
        value = (value.get("gathered") or {}).get("smoke_reference")
        if value is None:
            raise ValueError(f"{path} records no smoke reference: run crit run --id RUN recon --data DATA, with the "
                             "data's reference scores (crit eval reference)")
    return validate(value)


def quantile(values, share):
    """The nearest-rank quantile of `values` (sorted or not) at `share` in (0, 1]."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def judge(engine, reference, limits):
    """Judge the engine's log-probabilities of the answer tokens against the reference's (equal-length lists).
    Returns (ok, summary): the two perplexities and their ratio, and the drift per target as median, p90 and max."""
    drift = [abs(a - b) for a, b in zip(engine, reference)]
    mine, theirs = math.exp(-sum(engine) / len(engine)), math.exp(-sum(reference) / len(reference))
    ratio = mine / theirs
    median, p90 = quantile(drift, 0.5), quantile(drift, 0.9)
    ok = ratio <= limits["ppl_ratio"] and median <= limits["median"] and p90 <= limits["p90"]
    summary = (f"perplexity {mine:.2f} against the reference's {theirs:.2f} (ratio {ratio:.3f}, limit "
               f"{limits['ppl_ratio']:g}); drift per target median {median:.3f}, p90 {p90:.3f}, max {max(drift):.3f} "
               f"nats (limits {limits['median']:g}, {limits['p90']:g})")
    return ok, summary
