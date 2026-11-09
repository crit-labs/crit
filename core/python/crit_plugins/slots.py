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
        "commit": "commit(decisions): Core's StepDecision (session, accepted, terminal, cancelled, discard) for each "
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
    "attention": {
        "attend": "attend(sequence, queries, positions, scale=None, pending=False) -> [output row]: each query "
                  "attends the sequence's rows at positions up to its own (scale 1 / sqrt(key width) by default)",
    },
    "projections": {
        "prepare": "prepare(rows) -> weight: a dense [out][in] matrix in the plugin's format",
        "project": "project(weight, activations) -> [[out] per token]",
        "dense": "dense(weight) -> rows: the values project multiplies with (the format's decoded matrix)",
    },
    "router": {
        "route": "route(logits, top, bias=None) -> [(expert ids, weights)]: one per token's router logits row",
    },
    "experts": {
        "run": "run(layer, x, ids, weights) -> rows: a routed SwiGLU block's output per token row of x, each token's "
               "routes ids[t] with weights[t] (id -1 skipped), weights from the expert_source slot",
    },
    "expert_source": {
        "add": "add(layer, expert, gate_up, down): an expert's [2 * inter][hidden] gate/up and [hidden][inter] down",
        "matrices": "matrices(layer, expert) -> (gate_up, down)",
    },
    "expert_residency": {
        "plan": "plan(layer, experts) -> [resident]: for each of one group's distinct routed experts, whether its "
                "weights are on the device once the plan's admissions are made",
        "resident": "resident(layer) -> set of resident experts",
    },
    "recurrence": {
        "prefill": "prefill(state, steps, geometry) -> (outputs, state): a gated-delta state ([keys][values] "
                   "flattened) over steps {query, key, value, log_decay, beta}; geometry {keys, values, epsilon, "
                   "query_scale, normalize_keys, normalize_queries}",
    },
    "graphs": {
        "choose": "choose(shape, offers, free_bytes, uses=1) -> offer name: shape {phase, tokens, rows, identity}, "
                  "each offer {name, mode (eager, full, partial), shape, run_ns, workspace_bytes, resident_bytes, "
                  "setup_ns, setup_peak_bytes, ready}",
    },
    "embeddings": {
        "load": "load(table): the lookup table, one [width] row per id (token or hashed n-gram embeddings)",
        "rows": "rows(ids) -> [row]: the table rows of the ids, in request order",
    },
    # The elementwise work around the matmuls, each row on its own: a row is a token's hidden row or one head of one
    # token. In FP64 from the inputs as given (the oracle of every device form).
    "norms": {
        "rms_norm_rows": "rms_norm_rows(rows, weight, eps, offset=0.0, residual=None) -> rows, or (rows, sums) with "
                         "residual rows: y = x / sqrt(mean(x^2) + eps) * (offset + weight) for each row x, or for x = "
                         "residual + row (sums: those x); weight one row, G rows (row i takes weight row i % G: G "
                         "heads or streams a token, each its own weight) or None; offset 1 for a weight stored as "
                         "weight - 1",
        "gated_rms_norm_rows": "gated_rms_norm_rows(rows, gates, weight, eps, offset=0.0, activation='silu') -> rows: "
                               "rms_norm_rows times activation(gate) element by element (silu or sigmoid)",
        "rope_rows": "rope_rows(rows, positions, freqs, rotary_dim=None, style='half', scale=1.0) -> rows: each row "
                     "(one head) rotated at its position in pairs on its first rotary_dim values, the rest passed "
                     "through; freqs rotary_dim / 2 inverse frequencies (the angle of pair j fp32(fp32(position) x "
                     "fp32(freqs[j]))) or a table of [cos of every pair, sin of every pair] rows by position; style "
                     "half pairs (j, j + rotary_dim / 2), interleaved (2j, 2j + 1); scale multiplies cos and sin",
        "act_mul_rows": "act_mul_rows(gates, values, activation) -> rows: activation(gate) * value element by element; "
                        "activation silu, gelu (erf), gelu_tanh or sigmoid",
        "l2_norm_rows": "l2_norm_rows(rows, eps=1e-6) -> rows: x / sqrt(sum(x^2) + eps)",
        "stream_sum_rows": "stream_sum_rows(streams, gates=None, activation='none', scale=1.0) -> rows: per token "
                           "(streams[t], its S stream rows) scale * the sum in stream order of activation(gate) * row; "
                           "gates[t] S rows (one gate per value) or S numbers (one per stream) or None (the plain "
                           "sum); activation none (the gate itself) or act_mul_rows' (several residual streams mixed "
                           "into one row)",
        "stream_add_rows": "stream_add_rows(streams, rows, weights=None) -> streams: per token every stream row plus "
                           "weights[t][s] * rows[t] (1 without weights): one row added into every residual stream",
        "softplus_decay_rows": "softplus_decay_rows(rows, bias, rate, gates=None) -> rows, or (rows, sigmoid rows) "
                               "with gates: -rate[i] * softplus(x[i] + bias[i]) for each value of each row "
                               "(softplus(v) = log(1 + exp(v))): the log-decay a gated delta or state-space layer "
                               "discretizes (rate exp(A_log)), and sigmoid(gate) of its beta gates",
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


def missing(slot, plugin):
    """The interface methods of `slot` that `plugin` lacks (none for a slot crit does not name)."""
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

def _rows(rng, count, width, scale=3.0):
    return [[round(rng.uniform(-scale, scale), 3) for _ in range(width)] for _ in range(count)]


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


def _attention(parts):
    kv, attention, rng, out = parts["kv_cache"], parts["attention"], random.Random(5), {}
    kv.open(11, 8, 5, 64)
    kv.stage(11, _rows(rng, 13, 8, 1.0), _rows(rng, 13, 5))
    kv.commit(11)
    queries = _rows(rng, 4, 8, 1.0)
    out["decode"] = attention.attend(11, queries[:1], [12])
    out["prefill"] = attention.attend(11, queries, [0, 5, 9, 12])
    out["scaled"] = attention.attend(11, queries[1:3], [7, 12], scale=0.5)
    kv.stage(11, _rows(rng, 3, 8, 1.0), _rows(rng, 3, 5))
    out["pending"] = attention.attend(11, queries[:3], [13, 14, 15], pending=True)
    kv.rollback(11)
    kv.release(11)
    return out


def _router(parts):
    router, rng = parts["router"], random.Random(13)
    logits = _rows(rng, 5, 8)
    out = {}
    for top in (1, 2, 3):
        out[f"top{top}"] = [[list(ids), list(weights)] for ids, weights in router.route(logits, top)]
    bias = [round(rng.uniform(-0.5, 0.5), 3) for _ in range(8)]
    out["bias"] = [[list(ids), list(weights)] for ids, weights in router.route(logits, 2, bias)]
    return out


def expert_weights(layer, expert, hidden=6, inter=4):
    """The synthetic weights the experts cases add to the expert_source slot."""
    rng = random.Random(1000 * layer + expert)
    return _rows(rng, 2 * inter, hidden, 0.5), _rows(rng, hidden, inter, 0.5)


def _experts(parts):
    source, experts, rng = parts["expert_source"], parts["experts"], random.Random(17)
    for layer in (0, 1):
        for expert in range(6):
            source.add(layer, expert, *expert_weights(layer, expert))
    x = _rows(rng, 5, 6, 1.0)
    ids = [[0, 3], [3, 5], [1, -1], [2, 4], [5, 0]]
    weights = [[0.7, 0.3], [0.5, 0.5], [1.0, 0.0], [0.6, 0.4], [0.9, 0.1]]
    out = {"layer0": experts.run(0, x, ids, weights), "layer1": experts.run(1, x[:3], ids[:3], weights[:3])}
    out["again"] = experts.run(0, x[2:], ids[2:], weights[2:])
    return out


def _recurrence(parts):
    recurrence, rng = parts["recurrence"], random.Random(19)
    geometry = {"keys": 3, "values": 2, "epsilon": 1e-6, "query_scale": 0.5, "normalize_keys": True,
                "normalize_queries": True}
    steps = [{"query": [rng.uniform(-1, 1) for _ in range(3)], "key": [rng.uniform(-1, 1) for _ in range(3)],
              "value": [rng.uniform(-1, 1) for _ in range(2)], "log_decay": rng.uniform(-0.4, -0.01),
              "beta": rng.uniform(0.1, 0.9)} for _ in range(9)]
    state = tuple(0.0 for _ in range(6))
    outputs, state = recurrence.prefill(state, steps[:7], geometry)
    more, state = recurrence.prefill(state, steps[7:], geometry)
    return {"outputs": [list(o) for o in outputs + more], "state": list(state)}


def _placement(parts):
    placement = parts["placement"]
    tensors = {"embed": 600, "layers.0": 300, "layers.1": 300, "layers.2": 300, "head": 600, "kv": 1000}
    # With room for everything, every tensor sits on the device.
    return {"roomy": placement.plan(dict(tensors), {"device": 10_000, "host": 10_000}),
            "exact": placement.plan(dict(tensors), {"device": sum(tensors.values()), "host": 0})}


def _embeddings(parts):
    embeddings, rng = parts["embeddings"], random.Random(23)
    embeddings.load(_rows(rng, 40, 6, 0.5))
    return {"decode": embeddings.rows([7]), "prefill": embeddings.rows([0, 39, 7, 7, 12, 3]),
            "again": embeddings.rows([12, 7, 25])}


def _norms(parts):
    norms, rng, out = parts["norms"], random.Random(29), {}
    x, residual = _rows(rng, 4, 12), _rows(rng, 4, 12)
    weight = [round(rng.uniform(-0.5, 1.5), 3) for _ in range(12)]
    out["rms"] = norms.rms_norm_rows(x, weight, 1e-6)
    out["rms_plain"] = norms.rms_norm_rows(x[:1], None, 1e-5)
    out["rms_offset"] = norms.rms_norm_rows(x, [w - 1 for w in weight], 1e-6, offset=1.0)
    out["rms_residual"] = list(norms.rms_norm_rows(x, weight, 1e-6, residual=residual))
    out["rms_zero"] = norms.rms_norm_rows([[0.0] * 12], weight, 1e-6)
    gates = _rows(rng, 4, 12)
    out["gated_silu"] = norms.gated_rms_norm_rows(x, gates, weight, 1e-6)
    out["gated_sigmoid"] = norms.gated_rms_norm_rows(x, gates, weight, 1e-6, offset=0.5, activation="sigmoid")
    inv_freq = [1.0 / 10000 ** (2 * j / 8) for j in range(4)]
    positions = [0, 1, 17, 4099]
    out["rope_half"] = norms.rope_rows(x, positions, inv_freq, 8)
    out["rope_interleaved"] = norms.rope_rows(x, positions, inv_freq, style="interleaved", scale=1.25)
    table = [[math.cos(p * f) for f in inv_freq] + [math.sin(p * f) for f in inv_freq] for p in range(20)]
    out["rope_table"] = norms.rope_rows(x[:3], [0, 5, 19], table, 8)
    out["rope_full"] = norms.rope_rows(x, positions, [1.0 / 500 ** (2 * j / 12) for j in range(6)])
    for activation in ("silu", "gelu", "gelu_tanh", "sigmoid"):
        out[f"mul_{activation}"] = norms.act_mul_rows(gates, x, activation)
    out["l2"] = norms.l2_norm_rows(x, 1e-6)
    out["rms_grouped"] = norms.rms_norm_rows(x, [weight, [w * 0.5 for w in weight]], 1e-6, offset=1.0)
    streams = [_rows(rng, 3, 5) for _ in range(2)]                    # 2 tokens of 3 streams of 5
    out["stream_mean"] = norms.stream_sum_rows(streams, scale=1 / 3)
    out["stream_gated"] = norms.stream_sum_rows(streams, [_rows(rng, 3, 5) for _ in range(2)], "sigmoid")
    out["stream_weighted"] = norms.stream_sum_rows(streams, [[0.5, -1.0, 2.0], [1.0, 0.0, 0.25]])
    out["stream_add"] = norms.stream_add_rows(streams, _rows(rng, 2, 5), [[1.0, 0.5, -2.0], [0.0, 1.0, 3.0]])
    out["stream_add_plain"] = norms.stream_add_rows(streams, _rows(rng, 2, 5))
    bias, rate = [round(rng.uniform(-2, 2), 3) for _ in range(12)], [round(rng.uniform(0.01, 8), 3) for _ in range(12)]
    out["softplus_decay"] = norms.softplus_decay_rows([[v * 8 for v in row] for row in x], bias, rate)
    out["softplus_decay_gated"] = list(norms.softplus_decay_rows(x, bias, rate, gates))
    return out


CASES = {"sampler": _sampler, "verifier": _verifier, "kv_cache": _kv, "attention": _attention, "router": _router,
         "experts": _experts, "recurrence": _recurrence, "embeddings": _embeddings, "placement": _placement,
         "norms": _norms}


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
