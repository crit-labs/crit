"""Pinned continuations: `crit eval pin --adapter FILE --model DIR --data DIR`.

Speculative decoding makes decode speed depend on the text being drafted. Two builds whose arithmetic
differs slightly generate different texts and accept drafts at different rates, so their decode scores
partly measure which text came out. A pinned continuation fixes the text: `pin` records one engine's
plain greedy output for every speed case into the data directory, once, and `run` then measures every
replay-capable engine decoding exactly that text (drafts accepted while they match it, each round's
correction taken from it). The continuation is the model's own greedy text as computed by the pinning
engine; every engine evaluated against the data directory decodes the same tokens.
"""
import datetime
import hashlib
import json
from pathlib import Path

from . import CONTRACT, load_json, write_json
from .prepare import PREPARED

CONTINUATIONS = "continuations.json"


def continuation_sha256(cases):
    """Identity of the pinned text: sha256 over the canonical JSON of {case id: token ids}."""
    canonical = json.dumps({cid: [int(t) for t in tokens] for cid, tokens in sorted(cases.items())},
                           separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_continuations(data, prepared):
    """The data directory's pinned continuation, checked against its prepared data; None when absent."""
    path = Path(data) / CONTINUATIONS
    if not path.is_file():
        return None
    pinned = load_json(path)
    if pinned.get("prepared_model_sha256") != prepared["model"]["sha256"] or pinned.get("suite_sha256") != prepared["suite_sha256"]:
        raise SystemExit(f"{path} was pinned for different prepared data")
    if pinned.get("sha256") != continuation_sha256(pinned.get("cases", {})):
        raise SystemExit(f"{path} does not match its recorded sha256; pin again into a new data directory")
    return pinned


def pin(adapter, model, data, config=None, log=print, context_tokens=None):
    """Record the adapter's plain greedy output of every speed case as the data directory's continuation.
    Refuses to replace an existing one: a data directory has one pinned text. context_tokens: the context
    the engine is built for (the run's required context)."""
    from .runner import load_adapter, open_engine, timed_generate
    data = Path(data)
    prepared = load_json(data / PREPARED)
    out = data / CONTINUATIONS
    if out.exists():
        raise SystemExit(f"{out} already exists; a data directory keeps one pinned continuation")
    module = load_adapter(adapter)
    engine, info = open_engine(module, model, config or {}, speculative=False, context_tokens=context_tokens)
    cases, skipped = {}, {}
    try:
        for case in prepared["speed"]:
            prompt, n = case["prompt_ids"], case["output_tokens"]
            if len(prompt) + n > info["max_context"]:
                skipped[case["id"]] = f"needs {len(prompt) + n} tokens of context"
                log(f"pin {case['id']}: skipped (context)")
                continue
            first, _ = timed_generate(engine, prompt, n)
            second, _ = timed_generate(engine, prompt, n)
            if first != second:
                raise SystemExit(f"plain greedy output of speed case {case['id']} changes between runs; "
                                 "a pinned continuation needs a repeatable engine")
            cases[case["id"]] = first
            log(f"pin {case['id']}: {n} tokens")
    finally:
        engine.close()
    if skipped:
        raise SystemExit("every speed case needs a continuation; skipped: "
                         + ", ".join(f"{cid} ({why})" for cid, why in skipped.items()))
    pinned = {
        "contract": CONTRACT, "prepared_model_sha256": prepared["model"]["sha256"],
        "suite_sha256": prepared["suite_sha256"],
        "pinned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": {"name": info["name"], "configuration": info["configuration"]},
        "sha256": continuation_sha256(cases), "cases": cases,
    }
    write_json(out, pinned)
    return pinned
