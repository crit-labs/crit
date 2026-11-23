"""What the engine calls a plugin in each of crit's slots with, and the cases a plugin is compared with its reference on.

A slot's interface is a few methods over plain rows: Python lists of numbers, one row per token, request or matrix row.
The plugins of crit's library (its Inference techniques and base plugins) take Python lists; an engine's own plugin
in the slot takes its own tensors in the same layout and keeps these methods, so the engine calls every plugin of a
slot alike (which is what lets an ablation switch them) and its tests compare it with the library plugin it replaces
on the cases below.
"""
from collections import namedtuple
import math
import random

# slot -> {method: what it takes and returns}. A plugin of the slot has every method.
INTERFACES = {
    "model": {
        "info": "info() -> the engine's facts and Core's settings (crit_core.native.STEP_SETTINGS): vocabulary and "
                "context, max_concurrency (2 or more: every engine batches), max_pending, max_batch_tokens, "
                "prefill_chunk, max_row_tokens, stop_tokens, sampling, token_logprobs, max_top_logprobs, scoring, "
                "score_paths; and facts such as implementation, speculative, plain, replay",
        "open": "open(request) -> session: a StepRequest (key, prompt, output_tokens, sampling, stop_tokens, "
                "logprobs, replay, plain, inputs) becomes a session of the model's",
        "step": "step(rows) -> [StepResult]: one per StepRow (session, prefill_tokens, output_tokens, token_work, "
                "prefill_complete, allowed), prompt and decode rows of many sessions computed together; a row's tokens "
                "never depend on the other rows of its step; a constrained row's allowed (crit_plugins.Allowed) goes "
                "to the sampler as its params['allowed']",
        "commit": "commit(decisions): Core's StepDecision (session, accepted, terminal, cancelled) for each "
                  "row of the last step",
        "retire": "retire(session, finish): the request ended (length, stop, cancelled or failed); its state goes",
        "score": "score(ids, first_target, path, queries, top_k) -> Scores (target, query, argmax, top): fixed text "
                 "scored alone on the engine, every target's log-probability, its query row's and the ranked top_k",
    },
    "placement": {
        "plan": "plan(tensors, budget) -> {name: place}: where each tensor lives, place one of device, host or files; "
                "tensors {name: bytes}, budget {device, host}: the bytes free for them at the context the engine "
                "serves",
    },
    "sampler": {
        "select": "select(rows, params) -> [token]: one token per logit row; params is one dict for every row or a "
                  "list of one dict per row: temperature (0 greedy), top_p, top_k, min_p, seed, position (the row's "
                  "output position), stream, allowed (the token ids the row may take, sorted ids or a crit_plugins.Allowed, "
                  "absent for all)",
        "law": "law(row, params) -> [(token, probability)]: the tokens a draw chooses from, in rank order",
        "predictions": "predictions(rows, drafts, params, draft_laws=None) -> [token]: the token each of the K + 1 "
                       "verify columns predicts for K drafts (column c at output position position + c), under "
                       "rejection sampling when draft_laws gives the law each draft was drawn from",
    },
    "drafter": {
        "start": "start(sequence, tokens): a sequence's tokens so far, its last one the pending token",
        "propose": "propose(sequence, width, head=None) -> {tokens, parents, laws}: the draft tree to verify, node 0 "
                   "the pending token and parents[i] < i (a chain has parents -1, 0, 1, ...), laws None or one "
                   "token -> probability dict per draft node; head(paths, rows=None) is the engine's draft model "
                   "(fact draft:head): one logit row per token path after the pending token, over `rows` of the "
                   "vocabulary when given (fact draft:head-rows)",
        "observe": "observe(sequence, emitted): the tokens the round emitted, the next pending token last",
        "release": "release(sequence)",
    },
    "verifier": {
        "accept": "accept(proposal, predictions, remaining, stops=()) -> {tokens, accepted, path}: the tokens a round "
                  "emits from a drafter's proposal and the target's prediction at every node (at most `remaining`, "
                  "ending after a stop token), how many drafts were accepted and the accepted node path from 0",
    },
    "kv_cache": {
        "open": "open(sequence, key_width, value_width, capacity)",
        "stage": "stage(sequence, keys, values): rows appended after the sequence's last, pending until commit",
        "commit": "commit(sequence, count=None): keep the first count pending rows (all by default), drop the rest",
        "rollback": "rollback(sequence): drop the pending rows",
        "rows": "rows(sequence, pending=False) -> [(position, key, value)]: the committed rows, the pending ones too",
        "fork": "fork(sequence, new): a new sequence sharing the committed rows",
        "release": "release(sequence): drop the sequence and its rows (close() is the plugin's own teardown)",
    },
}


# The places a placement plan puts a tensor.
PLACES = ("device", "host", "files")
# Beside its interface, a model may answer admission(request) -> "ready", "deferred" or "infeasible" (else Core admits
# by its settings), offer native_rounds(library) -> crit_core.native.NativeRounds (decode rounds Core launches
# without calling step()) and, providing IMAGES, take images (prepare).
MODEL_OPTIONAL = ("admission", "native_rounds", "prepare")
# The fact a model plugin provides when it takes images in its prompts: it implements
#   prepare(ids, images, cancelled) -> (ids, inputs)
# ids as the chat template rendered them, one placeholder per image where the template put it, and the request's
# images in order (Image); it returns the ids the engine prefills, each image's positions in place of its placeholder
# (as many as the image takes), and the request's inputs (pixels, layout), which open() gets as StepRequest.inputs.
# It runs on the request's thread beside the steps: decoding and layout only, polling cancelled(); the vision
# arithmetic belongs to the request's prompt rows.
IMAGES = "inputs:images"
# One image of a request, as prepare() gets it: its bytes (decoded from the request's data: URL), their media type
# (image/png, ...) and the request's detail (auto, low or high).
Image = namedtuple("Image", "data media_type detail")


def missing(slot, plugin, contracts=None):
    """The interface methods of `slot` that `plugin` lacks (none for a slot nothing defines): Core's slots', and
    with `contracts` (crit_plugins.contracts.Contracts) those its plugins define."""
    if contracts is not None:
        return contracts.host_missing(slot, plugin)
    return [name for name in INTERFACES.get(slot, {}) if not callable(getattr(plugin, name, None))]


def model_problems(model):
    """Why Core's concurrent runtime cannot run `model` as an engine's step model, or []: the model interface's
    methods, and an info() with a vocabulary, a context and max_concurrency of 2 or more. Core's loader refuses a
    model with any; one request at a time is a batch of one, never an engine of its own."""
    lacking = missing("model", model)
    if lacking:
        return [f"it lacks {', '.join(lacking)} of the model interface"]
    info = model.info()
    if not isinstance(info, dict):
        return ["info() returns a dict"]
    problems = [f"info() names no {key}" for key in ("vocabulary", "context")
                if type(info.get(key)) is not int or info[key] < 1]
    concurrency = info.get("max_concurrency")
    if type(concurrency) is not int or concurrency < 2:
        problems.append(f"info() declares max_concurrency {concurrency!r}: every engine batches, so its step runs at "
                        "least 2 rows of different requests together")
    return problems


# --- The cases a plugin is compared with its reference on: case(assembly) -> a value, equal for both up to the
# tolerance the test gives (token ids and counts exactly).

def rows(rng, count, width, scale=3.0):
    """`count` rows of `width` numbers in [-scale, scale], 3 decimals: the inputs of a slot's comparison cases."""
    return [[round(rng.uniform(-scale, scale), 3) for _ in range(width)] for _ in range(count)]


_rows = rows


def _sampler(parts):
    s, rng = parts["sampler"], random.Random(7)
    rows = _rows(rng, 6, 11)
    rows[2][4] = rows[2][7] = max(rows[2]) + 1   # a tie: the lower id ranks first
    out = {"greedy": s.select(rows, {"temperature": 0.0})}
    rules = [dict(temperature=0.8, top_p=0.9, seed=3), dict(temperature=1.3, top_k=4, seed=5),
             dict(temperature=0.6, min_p=0.2, seed=9, stream=1), dict(temperature=1.0, top_p=0.5, top_k=6, seed=11)]
    for i, rule in enumerate(rules):
        out[f"sampled{i}"] = s.select(rows, [dict(rule, position=p) for p in range(len(rows))])
        out[f"law{i}"] = s.law(rows[i], rule)
    drafts = [s.select([rows[c]], dict(rules[0], position=c))[0] for c in range(3)]
    out["verify_greedy"] = s.predictions(rows[:4], drafts, {"temperature": 0.0})
    laws = [dict(s.law(rows[c], rules[1])) for c in range(3)]
    out["verify_sampled"] = s.predictions(rows[:4], drafts, dict(rules[1], position=2), laws)
    allowed = [1, 3, 4, 8]
    out["allowed"] = s.select(rows, [dict(rules[2], position=p, allowed=allowed) for p in range(len(rows))])
    out["allowed_law"] = s.law(rows[0], dict(rules[3], allowed=allowed))
    return out


def _verifier(parts):
    v, out = parts["verifier"], {}
    chain = {"tokens": [9, 4, 5, 6, 7], "parents": [-1, 0, 1, 2, 3], "laws": None}
    for name, predictions, remaining, stops in (("all", [4, 5, 6, 7, 8], 10, ()), ("first", [3, 5, 6, 7, 8], 10, ()),
                                                ("middle", [4, 5, 2, 7, 8], 10, ()), ("budget", [4, 5, 6, 7, 8], 2, ()),
                                                ("stop", [4, 5, 6, 7, 8], 10, (5,)), ("none", [4, 5, 6, 7, 8], 0, ())):
        result = v.accept(chain, predictions, remaining, stops)
        out[name] = {"tokens": list(result["tokens"]), "accepted": result["accepted"]}
    return out


def _kv(parts):
    kv, rng, out = parts["kv_cache"], random.Random(3), {}
    kv.open(1, 4, 3, 32)
    kv.stage(1, _rows(rng, 3, 4), _rows(rng, 3, 3))
    kv.commit(1)
    kv.stage(1, _rows(rng, 2, 4), _rows(rng, 2, 3))
    out["pending"] = kv.rows(1, pending=True)
    kv.commit(1, 1)
    out["committed"] = kv.rows(1)
    kv.fork(1, 2)
    kv.stage(2, _rows(rng, 1, 4), _rows(rng, 1, 3))
    kv.rollback(2)
    kv.stage(2, _rows(rng, 2, 4), _rows(rng, 2, 3))
    kv.commit(2)
    out["fork"], out["parent"] = kv.rows(2), kv.rows(1)
    kv.release(2)
    kv.release(1)
    return {k: [[p, list(key), list(value)] for p, key, value in rows] for k, rows in out.items()}












def _placement(parts):
    placement = parts["placement"]
    tensors = {"embed": 600, "layers.0": 300, "layers.1": 300, "layers.2": 300, "head": 600, "kv": 1000}
    # With room for everything, every tensor sits on the device.
    return {"roomy": placement.plan(dict(tensors), {"device": 10_000, "host": 10_000}),
            "exact": placement.plan(dict(tensors), {"device": sum(tensors.values()), "host": 0})}






# The comparison cases of Core's slots; a defined slot's are its definition's (crit_plugins.contracts: cases).
CASES = {"sampler": _sampler, "verifier": _verifier, "kv_cache": _kv, "placement": _placement}


def compare(actual, expected, atol=1e-9, rtol=1e-7, path="value"):
    """The first difference between two case values, or None: ints, bools, strings and None exactly, floats within
    atol + rtol * |expected|, lists, tuples and dicts element by element."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return f"{path}: keys {sorted(actual) if isinstance(actual, dict) else actual!r}, expected {sorted(expected)}"
        for key in expected:
            found = compare(actual[key], expected[key], atol, rtol, f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return f"{path}: {actual!r}, expected {len(expected)} items"
        for i, (a, e) in enumerate(zip(actual, expected)):
            found = compare(a, e, atol, rtol, f"{path}[{i}]")
            if found:
                return found
        return None
    if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if math.isfinite(expected) and math.isfinite(actual) and abs(actual - expected) <= atol + rtol * abs(expected):
            return None
        if actual == expected:
            return None
        return f"{path}: {actual!r}, expected {expected!r} (atol {atol}, rtol {rtol})"
    if actual != expected or type(actual) is bool and type(expected) is not bool:
        return f"{path}: {actual!r}, expected {expected!r}"
    return None
