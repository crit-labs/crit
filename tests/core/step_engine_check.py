"""Drive a toy Python step model through crit_core.native.StepEngine and Core's concurrent runtime (crit/step_engine.h,
the crit_core_steps library): requests decode the same tokens batched as one at a time, steps hold several requests
and mix prompt chunks with decode rows, cancellation and retirement reach the model and release every session,
page-reserving admission defers requests, log-probabilities, replay and stop tokens pass through, scores run as Core's
barrier, a failing step fails the engine, and the stock eval-v1 and chat-resident-v1 adapters serve the model with
max_concurrent_requests above 1."""
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "python"))

from crit_core.batch import DECODE, PREFILL, PageReservations, StepLayout, step_item  # noqa: E402
from crit_core.native import (STEP_LIBRARY, NativeError, ResidentEngine, Scores, StepEngine, StepResult,  # noqa: E402
                              eval_engine)

VOCAB, STOP = 1000, 7


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def successor(seen):
    """The toy model: the next token is a hash of every token a row sees, so wrong bookkeeping changes it."""
    h = 1469598103934665603
    for position, token in enumerate(seen):
        h = ((h ^ ((token + 1) * (position + 1))) * 1099511628211) % 2 ** 64
    return h % VOCAB


def oracle(prompt, output):
    rows, out = list(prompt), []
    for _ in range(output):
        out.append(successor(rows))
        rows.append(out[-1])
    return out


class Session:
    def __init__(self, request):
        self.request, self.prompt = request, list(request.prompt)
        self.rows, self.outputs = [], []        # materialized rows; published outputs
        self.staged, self.produced, self.pending, self.pages = None, [], None, 0


class ToyStepModel:
    """A batched model in Python: one step lays out every scheduled row with crit_core.batch (decode rows first, then
    prompt chunks) and computes each logits row from the row's own history only."""

    def __init__(self, concurrency=4, pages=None, fail_on=None, slow=None):
        # A step holding the `slow` prompt's request takes 20 ms, so a client can cancel that request mid-decode.
        self.concurrency, self.fail_on, self.slow = concurrency, fail_on, slow
        self.reservations = PageReservations(pages, 4, 64) if pages else None
        self.steps, self.opened, self.retired, self.cancelled, self.closed = [], [], [], 0, False
        self.lock = threading.Lock()   # the callbacks run on Core's one worker; this only guards the test's reads

    def info(self):
        return {"implementation": "toy step model", "vocabulary": VOCAB, "context": 64, "prefill_chunk": 8,
                "max_concurrency": self.concurrency, "max_pending": 16, "max_batch_tokens": 24, "stop_tokens": [STOP],
                "sampling": {"temperature": True}, "token_logprobs": True, "max_top_logprobs": 2, "scoring": True,
                "score_paths": ["prefill"], "replay": True, "speculative": False, "native_libraries": ["toy.so"]}

    def admission(self, request):
        return self.reservations.admission(len(request.prompt), request.output_tokens) if self.reservations else "ready"

    def open(self, request):
        session = Session(request)
        if self.reservations:
            session.pages = self.reservations.reserve(len(request.prompt), request.output_tokens)
        self.opened.append(request.key)
        return session

    def step(self, rows):
        layout = StepLayout(step_item(row, len(row.session.prompt), len(row.session.rows)) for row in rows)
        if self.slow is not None and any(row.session.prompt == self.slow for row in rows):
            time.sleep(0.02)
        with self.lock:
            self.steps.append((len(rows), layout.decode_items))
        results = [None] * len(rows)
        for placement in layout.placements:
            s = rows[placement.item].session
            start = len(s.rows)
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[start:start + placement.rows]
            if self.fail_on is not None and self.fail_on in inputs:
                raise RuntimeError("the toy model refuses token %d" % self.fail_on)
            s.staged, s.produced = inputs, []
            if placement.logits:
                replay = s.request.replay
                s.produced = [replay[len(s.outputs)] if replay else successor(s.rows + inputs)]
            logprobs = None
            if s.request.logprobs is not None and s.produced:
                token = s.produced[0]
                logprobs = [(-0.5, [(token, -0.5), ((token + 1) % VOCAB, -0.75)])]
            results[placement.item] = StepResult(placement.rows if placement.phase == PREFILL else 0, s.produced,
                                                 placement.logits > 0, 0, placement.rows, logprobs)
        return results

    def commit(self, decisions):
        for d in decisions:
            s = d.session
            if s.staged is None:
                continue
            if d.cancelled or d.discard:
                s.staged = None
                self.cancelled += d.cancelled
                continue
            s.rows += s.staged
            s.staged = None
            if d.accepted:
                s.pending = s.produced[0]
                s.outputs += s.produced[:d.accepted]

    def retire(self, session, finish):
        with self.lock:
            self.retired.append(finish)
        if self.reservations:
            self.reservations.release(session.pages)

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        targets = range(first_target, len(ids))
        return Scores([-0.5] * len(targets), [[-1.0] * len(row) for row in queries] if queries else None,
                      [successor(ids[:t]) for t in targets],
                      [[(ids[t], -0.5), ((ids[t] + 1) % VOCAB, -0.75)][:top_k] for t in targets] if top_k else None)

    def close(self):
        self.closed = True


def prompts():
    return [[(r * 37 + i * 11) % VOCAB for i in range(1 + r * 5)] for r in range(9)]


def run_all(engine, output):
    with ThreadPoolExecutor(9) as pool:
        return list(pool.map(lambda p: engine.generate(p, output, stop_tokens=[])[0], prompts()))


def batched_equals_serial(library):
    for concurrency in (1, 4):
        model = ToyStepModel(concurrency)
        engine = StepEngine(model, library)
        expect(engine.info()["max_concurrent_requests"] == concurrency + 16, engine.info())
        got = run_all(engine, 12)
        expect(got == [oracle(p, 12) for p in prompts()], ("tokens differ from the oracle", concurrency))
        largest = max(n for n, _ in model.steps)
        if concurrency > 1:
            expect(largest > 1 and any(0 < d < n for n, d in model.steps), ("steps hold several requests and mix", model.steps))
        else:
            expect(largest == 1, model.steps)
        engine.close()
        expect(model.closed and not engine._sessions, "close releases every session and closes the model")
        expect(len(model.retired) == len(model.opened) == 9 and set(model.retired) == {"length"}, model.retired)


def cancellation_and_retirement(library):
    model = ToyStepModel(4, slow=[1, 2, 3])
    engine = StepEngine(model, library)
    seen = []
    with ThreadPoolExecutor(3) as pool:
        long = pool.submit(engine.generate, [1, 2, 3], 40, stop_tokens=[], on_tokens=seen.extend,
                           cancelled=lambda: len(seen) >= 3)
        peers = [pool.submit(engine.generate, p, 10, stop_tokens=[]) for p in prompts()[:2]]
        expect(long.result(timeout=30) == (None, {"cancelled": True}), "the long request is cancelled")
        expect([f.result(timeout=30)[0] for f in peers] == [oracle(p, 10) for p in prompts()[:2]], "peers are unaffected")
    expect(3 <= len(seen) < 40, seen)
    expect("cancelled" in model.retired, ("the model retires the cancelled request", model.retired))
    tokens, result = engine.generate([5, 6], 4, stop_tokens=[])
    expect(tokens == oracle([5, 6], 4) and result["finish"] == "length", "the engine serves after a cancellation")
    expect(engine.generate([1], 4, cancelled=lambda: True) == (None, {"cancelled": True}), "early cancellation")
    engine.close()
    expect(not engine._sessions and len(model.retired) == len(model.opened), (model.retired, model.opened))


def admission(library):
    # 20 pages of 4 rows: these requests need 9 to 14 pages, so at most two run at once; the rest wait their turn.
    model = ToyStepModel(6, pages=20)
    engine = StepEngine(model, library)
    later = prompts()[4:]
    with ThreadPoolExecutor(len(later)) as pool:
        got = list(pool.map(lambda p: engine.generate(p, 12, stop_tokens=[])[0], later))
    expect(got == [oracle(p, 12) for p in later], "deferred requests decode the oracle's tokens")
    expect(max(n for n, _ in model.steps) <= 2 and model.reservations.reserved == 0, model.steps)
    try:
        engine.generate([1] * 62, 4)
    except NativeError as error:
        expect("prompt plus output exceeds context" in str(error), error)
    else:
        raise AssertionError("a request beyond the context must fail")
    engine.close()


def options(library):
    model = ToyStepModel(4)
    engine = StepEngine(model, library)
    rows = []
    tokens, result = engine.generate([1, 2, 3], 5, stop_tokens=[], logprobs=2, on_tokens=lambda ts, lps: rows.extend(zip(ts, lps)))
    expect(tokens == oracle([1, 2, 3], 5) and [t for t, _ in rows] == tokens, (tokens, rows))
    expect(all(lp == -0.5 and top == [(t, -0.5), ((t + 1) % VOCAB, -0.75)] for t, (lp, top) in rows), rows)
    # Replay decodes the continuation; Core ends it at the model's stop token unless stops are disabled.
    tokens, result = engine.generate([1, 2, 3], 6, replay=[12, 13, STOP, 14, 15, 16])
    expect(tokens == [12, 13, STOP] and result["finish"] == "stop", (tokens, result))
    tokens, result = engine.generate([1, 2, 3], 6, stop_tokens=[], replay=[12, 13, STOP, 14, 15, 16])
    expect(tokens == [12, 13, STOP, 14, 15, 16] and result["finish"] == "length", (tokens, result))
    expect(model.opened and model.retired.count("stop") == 1, model.retired)
    # Scores run on Core's worker as an exclusive barrier, with the engine's own top-k on a named path.
    scored = engine.score([1, 2, 3, 4], 2, "prefill", top_k=2)
    expect(scored.target == [-0.5, -0.5] and scored.argmax == [successor([1, 2]), successor([1, 2, 3])]
           and scored.top == [[(3, -0.5), (4, -0.75)], [(4, -0.5), (5, -0.75)]], scored)
    queried = engine.score([1, 2, 3, 4], 2, "prefill", [[7, 8], [9, 10]])
    expect(queried.query == [[-1.0, -1.0], [-1.0, -1.0]], queried)
    try:
        engine.generate([1, 2], 3, sampling={"top_p": 0.5})
    except NativeError as error:
        expect("does not support a requested sampling option" in str(error), error)
    else:
        raise AssertionError("Core refuses sampling fields the model does not apply")
    engine.close()


def failure(library):
    model = ToyStepModel(4, fail_on=666)
    engine = StepEngine(model, library)
    expect(engine.generate([1, 2], 3, stop_tokens=[])[0] == oracle([1, 2], 3), "a healthy request")
    try:
        engine.generate([5, 666, 6], 3)
    except NativeError as error:
        expect("step engine step failed: RuntimeError: the toy model refuses token 666" in str(error), error)
    else:
        raise AssertionError("a failing step must fail the request")
    try:
        engine.generate([1, 2], 3)
        raise AssertionError("a failed step engine refuses later requests")
    except NativeError:
        pass
    engine.close()
    try:
        StepEngine(types.SimpleNamespace(info=lambda: {"context": 64}, open=None, step=lambda rows: []), library)
    except NativeError as error:
        expect("vocabulary" in str(error), error)
    else:
        raise AssertionError("a step model without a vocabulary must not open")


class FakeText:
    """One token per character, as tests/core/native_engine_check.py's fixture."""
    generation = types.SimpleNamespace(stop_token_ids=())
    tokenizer = types.SimpleNamespace(special_ids=(), token_bytes=lambda: tuple(bytes([97 + t % 26]) for t in range(VOCAB)))

    def encode_chat(self, messages, **_variables):
        return [ord(c) % 40 + 1 for m in messages for c in m["content"]]

    def text_stream(self, ids, **_kwargs):
        delta = lambda text: types.SimpleNamespace(text=text)
        return types.SimpleNamespace(push=lambda tokens: delta("".join(chr(97 + t % 26) for t in tokens)),
                                     finish=lambda: delta(""))


def adapters(library):
    os.environ[STEP_LIBRARY] = library
    opened = []

    def open_engine(configuration):
        opened.append(ToyStepModel(4))
        return opened[-1]

    evaluated = eval_engine({"model": "/models/toy", "config": {"mode": "x"}, "speculative": False}, open_engine)
    info = evaluated.info()
    expect(info["native_libraries"] == [str(Path(library).resolve()), "toy.so"] and info["max_context"] == 64, info)
    out = evaluated.generate([1, 2, 3], 6, lambda tokens: None)
    expect(out["tokens"] == oracle([1, 2, 3], 6) and out["rounds"] == 5 and "prefill_ms" in out, out)
    expect(evaluated.replay([1, 2, 3], [11, STOP, 12], lambda tokens: None)["tokens"] == [11, STOP, 12], "replay")
    evaluated.close()
    expect(opened[-1].closed, "Eval's adapter closes the model")

    import crit_core.model
    import crit_core.text as text
    crit_core.model.open_model = lambda path: types.SimpleNamespace(files=path)
    text.load_model_text = lambda model, clock=None: FakeText()
    text.openai_messages = lambda messages, *, images=False: messages
    context = types.SimpleNamespace(configuration={}, model_file=lambda name: "/models/toy/" + name, limits={"kv_tokens": 64})
    resident = ResidentEngine(context, open_engine)
    expect(resident.max_concurrent_requests == 20 and resident.logprobs is False, "served concurrently")
    wrappers = [resident.create_request({}) for _ in range(4)]
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(lambda w_p: w_p[0].generate_tokens(w_p[1], 8, {"temperature": 0.0}, None, lambda: False)[0],
                            zip(wrappers, prompts()[:4])))
    expected = [oracle(p, 8) for p in prompts()[:4]]
    # The model's stop token ends a reply where the oracle produces it.
    expect(got == [e[:e.index(STOP) + 1] if STOP in e else e for e in expected], (got, expected))
    for w in wrappers:
        w.close()
    resident.close()


def main(library):
    batched_equals_serial(library)
    cancellation_and_retirement(library)
    admission(library)
    options(library)
    failure(library)
    adapters(library)
    print("step engine checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
