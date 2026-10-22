"""Drive the toy engine library (native_engine_fixture.cpp) through crit_core.native: the C ABI, the eval-v1
adapter and the chat-resident-v1 adapter."""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "python"))

from crit_core.native import EvalEngine, NativeEngine, NativeError, ResidentEngine  # noqa: E402


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def abi(library):
    engine = NativeEngine(library, {"context_tokens": 256})
    facts = engine.info()
    expect(facts["implementation"] == "toy native engine" and facts["context"] == 256, facts)
    expect(facts["stop_tokens"] == [5] and facts["score_paths"] == ["prefill", "decode"], facts)
    expect(facts["speculative"] is True and facts["replay"] is True and facts["draft_width"] == 1, facts)
    expect(facts["sampling"]["penalties"] is False and facts["sampling"]["logit_bias"] is False, facts)
    # The bridge must forward these options to Core, which rejects them for this greedy-only model.
    for options in ({"frequency_penalty": .5}, {"presence_penalty": .5}, {"repetition_penalty": 1.1},
                    {"logit_bias": {"12": -2}}):
        try:
            engine.generate([1, 2], 3, sampling=options)
            raise AssertionError("sampling options must never disappear in the ABI: " + str(options))
        except NativeError as error:
            expect("does not support a requested sampling option" in str(error), error)

    streamed = []
    tokens, result = engine.generate([1, 2, 3], 7, on_tokens=streamed.extend)
    expect(len(tokens) == 7 and streamed == tokens and result["finish"] == "length", (tokens, result))
    timing = result["timing"]
    expect(timing["prefill_tokens"] == 3 and timing["decode_steps"] >= 3 and result["accepted"] == [1, 1], result)
    expect(all(timing[k] >= 0 for k in ("queued_ms", "prefill_ms", "first_token_ms", "decode_ms", "total_ms")), timing)
    expect(engine.generate([1, 2, 3], 7)[0] == tokens, "generation is deterministic")

    # Replay decodes the given continuation; the model's stop token ends it unless stops are disabled.
    tokens, result = engine.generate([1, 2, 3], 6, replay=[12, 13, 5, 14, 15, 16])
    expect(tokens == [12, 13, 5] and result["finish"] == "stop", (tokens, result))
    tokens, result = engine.generate([1, 2, 3], 6, stop_tokens=[], replay=[12, 13, 5, 14, 15, 16])
    expect(tokens == [12, 13, 5, 14, 15, 16] and result["finish"] == "length", (tokens, result))

    # Per-token log-probabilities: the token's own and its n most likely alternatives, ranked.
    expect(facts["token_logprobs"] is True and facts["max_top_logprobs"] == 20, facts)
    rows = []
    tokens, result = engine.generate([1, 2, 3], 5, logprobs=3, on_tokens=lambda ts, lps: rows.extend(zip(ts, lps)))
    expect([t for t, _ in rows] == tokens and len(tokens) == 5 and result["finish"] == "length", (tokens, rows))
    for token, (logprob, top) in rows:
        expect(logprob == -0.5 and top == [(token, -0.5), ((token + 1) % 64, -0.75), ((token + 2) % 64, -1.0)], rows)
    expect(engine.generate([1, 2, 3], 5)[0] == tokens, "log-probabilities leave the tokens as they are")
    tokens, _ = engine.generate([1, 2, 3], 2, logprobs=0, on_tokens=lambda ts, lps: rows.extend(lps))
    expect(rows[-2:] == [(-0.5, []), (-0.5, [])], rows)
    seen = []
    tokens, result = engine.generate([1, 2, 3], 20, logprobs=1, on_tokens=lambda ts, lps: seen.extend(ts),
                                     cancelled=lambda: len(seen) >= 3)
    expect(tokens is None and result == {"cancelled": True} and len(seen) == 3, (tokens, result, seen))
    try:
        engine.generate([1, 2, 3], 2, logprobs=21)
        raise AssertionError("more alternatives than the model reports must fail")
    except NativeError as error:
        expect("max_top_logprobs" in str(error), error)

    seen = []
    tokens, result = engine.generate([1, 2, 3], 20, on_tokens=seen.extend, cancelled=lambda: len(seen) >= 3)
    expect(tokens is None and result == {"cancelled": True} and len(seen) == 3, (tokens, result, seen))
    expect(len(engine.generate([4], 2)[0]) == 2, "the engine serves requests after a cancellation")

    def boom(_tokens):
        raise KeyError("from the callback")
    try:
        engine.generate([1], 4, on_tokens=boom)
        raise AssertionError("a callback exception must propagate")
    except KeyError:
        pass

    scored = engine.score([1, 2, 3, 4], 2)
    expect(scored == ([-0.5, -0.5], None, [3, 4], None), scored)
    scored = engine.score([1, 2, 3, 4], 2, "decode", [[7, 8], [9, 10]])
    expect(scored.target == [-0.75, -0.75] and scored.query == [[-1.0, -1.125], [-1.25, -1.375]]
           and scored.argmax == [3, 4] and scored.top is None, scored)
    # The engine's own top-k with the scores (the target first, then the ids after it), alone or with queries.
    scored = engine.score([1, 2, 3, 4], 2, "decode", top_k=2)
    expect(scored.target == [-0.75, -0.75] and scored.query is None
           and scored.top == [[(3, -0.75), (4, -1.0)], [(4, -0.75), (5, -1.0)]], scored)
    both = engine.score([1, 2, 3, 4], 2, "decode", [[7, 8], [9, 10]], top_k=2)
    expect(both.query == [[-1.0, -1.125], [-1.25, -1.375]] and both.top == scored.top, both)
    for bad in (lambda: engine.score([1, 2], 2), lambda: engine.generate([], 3),
                lambda: engine.score([1, 2], 1, top_k=21)):
        try:
            bad()
            raise AssertionError("invalid requests must fail")
        except NativeError:
            pass
    engine.close()
    engine.close()

    try:
        NativeEngine(library, {"fail": True})
        raise AssertionError("a failing factory must not open")
    except NativeError as error:
        expect("toy engine refused to open" in str(error), error)


def concurrent_adapter(library):
    engine = NativeEngine(library, {"concurrent": True})
    expect(engine.info()["max_concurrent_requests"] == 3, engine.info())
    cancelled = threading.Event()
    seen = []
    with ThreadPoolExecutor(2) as pool:
        slow = pool.submit(engine.generate, [1] * 200, 20, on_tokens=seen.extend, cancelled=cancelled.is_set)
        time.sleep(.02)
        fast = pool.submit(engine.generate, [4], 4)
        tokens, result = fast.result(timeout=5)
        expect(len(tokens) == 4 and result["identity"] > 0, (tokens, result))
        expect(not slow.done(), "a short peer must finish while the long prefill is still running")
        cancelled.set()
        expect(slow.result(timeout=5) == (None, {"cancelled": True}), "independent pre-token cancellation")
    expect(not seen, ("cancelled prefill emitted tokens", seen))
    expect(len(engine.generate([2], 4)[0]) == 4, "peer engine survives cancellation")
    expect(engine.generate([1], 4, cancelled=lambda: True) == (None, {"cancelled": True}), "early cancellation")
    try:
        engine.generate([1], 4, logprobs=1)
        raise AssertionError("a model without token log-probabilities must refuse them")
    except NativeError as error:
        expect("does not support generation logprobs" in str(error), error)
    engine.close()


def eval_adapter(library):
    engine = EvalEngine({"model": "/models/toy", "config": {"mode": "x"}, "speculative": True, "context_tokens": 128},
                        library)
    info = engine.info()
    expect(info["configuration"] == {"mode": "x", "model": "/models/toy", "speculative": True, "context_tokens": 128},
           info)
    expect(info["max_context"] == 256 and info["speculative"] and info["replay"] and info["draft_width"] == 1, info)
    expect(info["native_libraries"] == [str(Path(library).resolve())] and "score_paths" not in info, info)
    out = engine.generate([1, 2, 3], 6, lambda tokens: None)
    # Rounds follow the engine's accepted list (one entry per target pass), not Core's decode steps.
    expect(len(out["tokens"]) == 6 and out["rounds"] == 2 and out["accepted"] == [1, 1], out)
    continuation = [11, 5, 12, 13]  # the stop token does not end Eval generation
    expect(engine.replay([1, 2, 3], continuation, lambda tokens: None)["tokens"] == continuation, "replay")
    # Plain decoding on the speculative engine: the same greedy text, one target pass per token after the first.
    expect(info["plain"] is True, info)
    plain_out = engine.generate_plain([1, 2, 3], 6, lambda tokens: None)
    expect(plain_out["tokens"] == out["tokens"] and plain_out["rounds"] == 5 and "accepted" not in plain_out, plain_out)
    expect(not hasattr(engine, "score"), "Eval measures quality over the chat API, not through the adapter")
    engine.close()
    # Eval's speculative=False turns drafting off; it never turns on drafting the configuration turned off.
    for config, harness, expected in (({"speculative": False}, True, False), ({}, False, False), ({}, True, True)):
        plain = EvalEngine({"model": "/models/toy", "config": config, "speculative": harness}, library)
        expect(plain.configuration["speculative"] is expected, plain.configuration)
        plain.close()


class FakeText:
    """Stands in for crit_core.text's model text: one token per character, stop token filtered by the adapter."""
    generation = types.SimpleNamespace(stop_token_ids=())
    tokenizer = types.SimpleNamespace(special_ids=(), token_bytes=lambda: tuple(bytes([97 + t % 26]) for t in range(64)))

    def encode_chat(self, messages, **variables):
        expect(variables == {"tools": None, "add_generation_prompt": True, "enable_thinking": False}, variables)
        return [ord(c) % 40 + 1 for m in messages for c in m["content"]]

    def text_stream(self, ids, **_kwargs):
        def delta(text):
            return types.SimpleNamespace(text=text)

        return types.SimpleNamespace(push=lambda tokens: delta("".join(chr(97 + t % 26) for t in tokens)),
                                     finish=lambda: delta(""))


class Cancellation:
    def __init__(self, after=None):
        self.after, self.calls = after, 0

    def is_set(self):
        self.calls += 1
        return self.after is not None and self.calls > self.after

    def check(self):
        if self.is_set():
            raise RuntimeError("cancelled by the client")


def resident_adapter(library):
    def text_messages(messages, *, images=False):
        expect(not images, "the native text fixture must not request image processing")
        return messages

    import crit_core.model
    import crit_core.text as module
    crit_core.model.open_model = lambda path: types.SimpleNamespace(files=path)   # no model folder behind it
    module.load_model_text = lambda model, clock=None: FakeText()
    module.openai_messages = text_messages
    context = types.SimpleNamespace(configuration={"chat_template_kwargs": {"enable_thinking": False}},
                                    model_file=lambda name: "/models/toy/" + name, limits={"kv_tokens": 200})
    engine = ResidentEngine(context, library)
    request = {"request_id": "r1", "candidate_id": "candidate-0001", "package_sha256": "0" * 64,
               "messages": [{"role": "user", "content": "hello"}], "limits": {"max_output_tokens": 6}}
    pieces = []
    response = engine.generate_stream(request, Cancellation(), pieces.append)
    telemetry = response["telemetry"]
    expect(response["assistant"] == "".join(pieces) and len(response["assistant"]) == 6, response)
    expect(telemetry["prompt_tokens"] == 5 and telemetry["prefill_tokens"] == 5 and telemetry["output_tokens"] == 6,
           telemetry)
    expect(telemetry["decode_tokens"] == 5 and telemetry["ttft_ms"] >= 0 and telemetry["missing"] == {}, telemetry)
    try:
        engine.generate(request, Cancellation(after=2))
        raise AssertionError("a cancelled request must not return a response")
    except RuntimeError as error:
        expect("cancelled" in str(error), error)
    expect(engine.logprobs is True and "logprobs" not in response, response)
    response = engine.generate(dict(request, logprobs=2), Cancellation())
    rows, named = response["logprobs"]["tokens"], response["logprobs"]["bytes"]
    expect(len(rows) == 6 and all(row[1] == -0.5 and [t for t, _ in row[2]] == [row[0], (row[0] + 1) % 64]
                                  for row in rows), rows)
    expect(response["assistant"] == "".join(bytes.fromhex(named[str(row[0])]).decode() for row in rows), response)
    # Fixed text (chat-score-v1): the prompt through the prefill score path, the continuation by replay.
    expect(engine.scores is True, "query scoring, replay and log-probabilities score fixed text")
    scored = engine.score(dict(request, prompt_ids=[1, 2, 3], continuation_ids=[11, 5, 12], top_logprobs=2,
                               prompt_rows=True), Cancellation())
    expect([row[0] for row in scored["prompt"]] == [2, 3] and [row[0] for row in scored["continuation"]] == [11, 5, 12],
           scored)
    expect(all(row[1] == -0.5 and [t for t, _ in row[2]] == [row[0], (row[0] + 1) % 64]
               for row in scored["prompt"] + scored["continuation"]), scored)
    too_long = dict(request, limits={"max_output_tokens": 196})
    try:
        engine.generate(too_long, Cancellation())
        raise AssertionError("a request beyond kv_tokens must be refused")
    except ValueError:
        pass
    engine.close()
    context.configuration["concurrent"] = True
    engine = ResidentEngine(context, library)
    expect(engine.logprobs is False and engine.scores is False, "a model without token log-probabilities declares none")
    first, second = engine.create_request(request), engine.create_request(request)
    expect(engine.max_concurrent_requests == 3 and first is not second, "private request wrappers")
    first.calls_format = object()
    expect(second.calls_format is None, "parser discovery is request-local")
    first.close()
    expect(len(second.generate_tokens([1], 4, {}, None, lambda: False)[0]) == 4, "closing a wrapper keeps weights")
    second.close()
    engine.close()


def main(library):
    abi(library)
    concurrent_adapter(library)
    eval_adapter(library)
    resident_adapter(library)
    print("native engine checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
