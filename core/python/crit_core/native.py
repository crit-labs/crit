"""Native engines from Python: the ctypes side of Core's engine ABI (crit/native_engine.h) and the stock adapters.

An engine library exports its core::Engine with CRIT_EXPORT_ENGINE(factory). Its package then needs no glue of
its own beyond two small entry files:

    # eval adapter (eval-v1, docs/eval/ADAPTER.md)
    from crit_core.native import eval_engine
    def create_eval_engine(context):
        return eval_engine(context, Path(__file__).with_name("libengine.so"), defaults={...})

    # crit serve adapter (chat-resident-v1, docs/SERVING.md)
    from crit_core.native import resident_engine
    def create_engine(context):
        return resident_engine(context, Path(__file__).with_name("libengine.so"), defaults={...})

The factory receives a JSON object: the adapter's configuration plus "model" (the model directory) and
"context_tokens" (the context the engine must serve); the Eval adapter also passes "speculative" (false when
Eval asks for plain one-token decoding on a second engine; an engine declaring "plain" decodes it per request).
"""
import ctypes
import json
from pathlib import Path

ABI = 1
RESULT_BYTES = 1 << 20  # a generation's result JSON: finish, timing and the engine's request facts
TOKEN_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int)
FACTS = ("draft_width", "bytes_per_pass", "draft_bytes_per_step", "draft_ms", "kv_bytes_per_token", "kernel_work")


class NativeError(RuntimeError):
    pass


class NativeEngine:
    """One engine opened from a library exported with CRIT_EXPORT_ENGINE."""

    def __init__(self, library, configuration):
        self.path = Path(library).resolve()
        lib = ctypes.CDLL(str(self.path))
        lib.crit_engine_abi.restype = ctypes.c_int
        if lib.crit_engine_abi() != ABI:
            raise NativeError(f"{self.path} exports engine ABI {lib.crit_engine_abi()}, this crit_core reads {ABI}")
        lib.crit_engine_open.restype = ctypes.c_void_p
        lib.crit_engine_open.argtypes = [ctypes.c_char_p]
        lib.crit_engine_close.restype = None
        lib.crit_engine_close.argtypes = [ctypes.c_void_p]
        lib.crit_engine_info.restype = ctypes.c_int
        lib.crit_engine_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.crit_engine_generate.restype = ctypes.c_int
        lib.crit_engine_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                             ctypes.c_char_p, TOKEN_CALLBACK, ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_int]
        lib.crit_engine_score.restype = ctypes.c_int
        lib.crit_engine_score.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                          ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int)]
        lib.crit_engine_last_error.restype = ctypes.c_char_p
        self.lib = lib
        self.handle = lib.crit_engine_open(json.dumps(configuration).encode("utf-8"))
        if not self.handle:
            raise NativeError("opening the engine failed: " + self._error())
        size = 1 << 16
        buffer = ctypes.create_string_buffer(size)
        length = lib.crit_engine_info(self.handle, buffer, size)
        if length >= size:
            buffer = ctypes.create_string_buffer(length + 1)
            length = lib.crit_engine_info(self.handle, buffer, length + 1)
        if length < 0:
            raise NativeError("engine info failed: " + self._error())
        self.facts = json.loads(buffer.value.decode("utf-8"))

    def _error(self):
        return (self.lib.crit_engine_last_error() or b"").decode("utf-8", "replace")

    def info(self):
        """ModelInfo fields (implementation, context, stop_tokens, score_paths, ...) plus the factory's facts."""
        return dict(self.facts)

    def generate(self, prompt, output_tokens, *, stop_tokens=None, sampling=None, replay=None, plain=False,
                 on_tokens=None, cancelled=None):
        """Returns (tokens, result) with result's "finish", "timing" (ms, decode_steps) and the engine's request
        facts, or (None, {"cancelled": True}) when cancelled() turned true. stop_tokens None keeps the model's;
        [] disables them. sampling: temperature, top_p, top_k, min_p, seed, penalties and logit_bias.
        plain: decode without drafting (engines
        that declare "plain")."""
        options = dict(sampling or {})
        if stop_tokens is not None:
            options["stop_tokens"] = [int(t) for t in stop_tokens]
        if replay is not None:
            options["replay"] = [int(t) for t in replay]
        if plain:
            options["plain"] = True
        failure = []

        def deliver(_user, tokens, count):
            try:
                if on_tokens:
                    on_tokens([tokens[i] for i in range(count)])
                return 1 if cancelled and cancelled() else 0
            except BaseException as error:  # never unwind through C; re-raised below
                failure.append(error)
                return 1

        callback = TOKEN_CALLBACK(deliver)
        ids = (ctypes.c_int32 * len(prompt))(*prompt)
        out = (ctypes.c_int32 * max(1, output_tokens))()
        result = ctypes.create_string_buffer(RESULT_BYTES)
        count = self.lib.crit_engine_generate(self.handle, ids, len(prompt), int(output_tokens),
                                              json.dumps(options).encode("utf-8"), callback, None, out, result,
                                              RESULT_BYTES)
        if failure:
            raise failure[0]
        if count == -2:
            return None, {"cancelled": True}
        if count < 0:
            raise NativeError("generation failed: " + self._error())
        return [out[i] for i in range(count)], json.loads(result.value.decode("utf-8"))

    def score(self, ids, first_target, path="", queries=None):
        """(target log-probabilities, query log-probability rows or None, argmax or None) for every target
        ids[i], i >= first_target. queries: one row of token ids per target, all the same length."""
        targets = len(ids) - first_target
        per = len(queries[0]) if queries else 0
        flat = [int(t) for row in (queries or []) for t in row]
        if len(flat) != targets * per:
            raise NativeError("score queries need one equally long row per target")
        target = (ctypes.c_float * targets)()
        query = (ctypes.c_float * max(1, targets * per))()
        argmax = (ctypes.c_int32 * targets)()
        has_argmax = ctypes.c_int(0)
        code = self.lib.crit_engine_score(self.handle, (ctypes.c_int32 * len(ids))(*ids), len(ids), int(first_target),
                                          (path or "").encode("utf-8"),
                                          (ctypes.c_int32 * max(1, len(flat)))(*flat) if flat else None, per,
                                          target, query, argmax, ctypes.byref(has_argmax))
        if code != 0:
            raise NativeError("scoring failed: " + self._error())
        rows = [[query[t * per + j] for j in range(per)] for t in range(targets)] if per else None
        return list(target), rows, (list(argmax) if has_argmax.value else None)

    def close(self):
        if self.handle:
            self.lib.crit_engine_close(self.handle)
            self.handle = None


class EvalEngine:
    """eval-v1 (docs/eval/ADAPTER.md) over a native engine: greedy generation with stop tokens disabled, the
    engine's scoring paths, replay when the engine declares it, and its headroom facts."""

    def __init__(self, context, library, defaults=None):
        configuration = {**(defaults or {}), **(context.get("config") or {})}
        # Eval's speculative=False (plain one-token decoding) overrides the configuration; it never turns drafting on.
        speculative = bool(configuration.get("speculative", True)) and bool(context.get("speculative", True))
        configuration.update(model=str(context["model"]), speculative=speculative)
        if context.get("context_tokens"):
            configuration["context_tokens"] = int(context["context_tokens"])
        self.configuration = configuration
        self.native = NativeEngine(library, configuration)
        self.facts = self.native.info()
        self.speculative = bool(self.facts.get("speculative"))
        # Eval decodes greedily: request temperature zero where the engine samples; a greedy-only engine needs none.
        self.sampling = {"temperature": 0.0} if (self.facts.get("sampling") or {}).get("temperature") else {}
        if self.facts.get("replay"):
            self.replay = self._replay
        if self.speculative and self.facts.get("plain"):
            self.generate_plain = self._generate_plain

    def info(self):
        info = {"name": self.facts.get("name") or self.facts["implementation"], "configuration": self.configuration,
                "max_context": self.facts["context"], "speculative": self.speculative,
                "score_paths": self.facts.get("score_paths") or ["prefill"],
                "native_libraries": [str(self.native.path)], "replay": bool(self.facts.get("replay")),
                "plain": bool(self.speculative and self.facts.get("plain"))}
        info.update({key: self.facts[key] for key in FACTS if key in self.facts})
        return info

    def _result(self, tokens, result):
        # Rounds are target passes after prefill: the engine's own count, else one per accepted entry, else Core's
        # decode steps (which also count steps that run no target pass, such as seeding a drafter).
        if "rounds" in result:
            rounds = result["rounds"]
        elif "accepted" in result:
            rounds = len(result["accepted"])
        else:
            rounds = result["timing"]["decode_steps"] if self.speculative else max(0, len(tokens) - 1)
        out = {"tokens": tokens, "rounds": rounds}
        out.update({key: result[key] for key in ("accepted", "verify_ms", "draft_ms") if key in result})
        return out

    def generate(self, prompt_ids, max_new_tokens, on_tokens):
        tokens, result = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[],
                                              sampling=self.sampling, on_tokens=on_tokens)
        return self._result(tokens, result)

    def _generate_plain(self, prompt_ids, max_new_tokens, on_tokens):
        # One target pass per token after the first; the engine's request facts describe drafting, so none apply.
        tokens, _ = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[], sampling=self.sampling,
                                         plain=True, on_tokens=on_tokens)
        return {"tokens": tokens, "rounds": max(0, len(tokens) - 1)}

    def _replay(self, prompt_ids, continuation, on_tokens):
        tokens, result = self.native.generate(list(prompt_ids), len(continuation), stop_tokens=[],
                                              sampling=self.sampling, replay=list(continuation),
                                              on_tokens=on_tokens)
        return self._result(tokens, result)

    def score(self, ids, first_target, path, queries):
        named = self.facts.get("score_paths") or []
        target, rows, argmax = self.native.score(list(ids), int(first_target), path if named else "", queries)
        out = {"target_logprobs": target, "query_logprobs": rows or [[] for _ in target]}
        if argmax is not None:
            out["argmax"] = argmax
        return out

    def close(self):
        self.native.close()


class ResidentEngine:
    """chat-resident-v1 (docs/SERVING.md) over a native engine: the model's chat template and tokenizer from
    crit_core.text, the model's stop tokens and sampling defaults, streamed text and Core's phase timing.
    Configuration "chat_template_kwargs" passes variables to the chat template (for example enable_thinking)."""

    def __init__(self, context, library, defaults=None):
        from crit_core.text import load_model_text
        self.configuration = {**(defaults or {}), **(context.configuration or {})}
        model = Path(context.model_file("config.json")).parent
        limits = getattr(context, "limits", {}) or {}
        native = dict(self.configuration, model=str(model))
        if limits.get("kv_tokens"):
            native["context_tokens"] = int(limits["kv_tokens"])
        self.text = load_model_text(str(model), clock=None)
        self.native = NativeEngine(library, native)
        self.facts = self.native.info()
        self.capacity = min(int(self.facts["context"]), int(limits.get("kv_tokens") or self.facts["context"]))
        self.stop_ids = set(self.facts.get("stop_tokens") or [])

    def effective_configuration(self):
        return dict(self.configuration)

    def _run(self, request, cancellation, emit=None):
        if request.get("tools") or any(m.get("tool_calls") or m["role"] == "tool" for m in request["messages"]):
            raise ValueError("Native tool calling needs a model-specific resident adapter and output parser")
        messages = request["messages"]
        ids = self.text.encode_chat(messages, add_generation_prompt=True,
                                    **(self.configuration.get("chat_template_kwargs") or {}))
        limits = request.get("limits", {})
        output_tokens = int(limits.get("max_output_tokens") or 1024)
        if len(ids) + output_tokens > min(int(limits.get("kv_tokens") or self.capacity), self.capacity):
            raise ValueError(f"prompt of {len(ids)} tokens plus {output_tokens} output tokens exceeds the context")
        stream, pieces = self.text.text_stream(ids), []

        def on_tokens(tokens):
            delta = stream.push([t for t in tokens if t not in self.stop_ids])
            if delta.text:
                pieces.append(delta.text)
                if emit:
                    emit(delta.text)

        sampling = dict(request.get("sampling") or {})
        support = self.facts.get("sampling") or {}
        for key in sampling:
            flag = "penalties" if key in {"repetition_penalty", "presence_penalty", "frequency_penalty"} else key
            if not support.get(flag):
                raise ValueError("Native engine does not support requested sampling option: " + key)
        tokens, result = self.native.generate(ids, output_tokens, on_tokens=on_tokens, sampling=sampling,
                                              cancelled=lambda: cancellation.is_set())
        if tokens is None:
            cancellation.check()
            raise RuntimeError("generation cancelled")
        tail = stream.finish()
        if tail.text:
            pieces.append(tail.text)
            if emit:
                emit(tail.text)
        timing, n = result["timing"], len(tokens)
        telemetry = {"prompt_tokens": len(ids), "output_tokens": n,
                     "prefill_tokens": timing["prefill_tokens"] - timing["reused_prompt_tokens"],
                     "prefill_ms": timing["prefill_ms"],
                     "decode_tokens": n - 1 if n > 1 else None, "decode_ms": timing["decode_ms"] if n > 1 else None,
                     "ttft_ms": timing["first_token_ms"], "missing": {}}
        if n <= 1:
            telemetry["missing"]["decode"] = "the reply has a single token"
        return "".join(pieces), telemetry

    def _response(self, request, text, telemetry):
        return {"schema_version": 1, "record_kind": "chat_response", "request_id": request["request_id"],
                "candidate_id": request["candidate_id"], "package_sha256": request["package_sha256"],
                "assistant": text, "telemetry": telemetry}

    def generate(self, request, cancellation):
        return self._response(request, *self._run(request, cancellation))

    def generate_stream(self, request, cancellation, emit):
        return self._response(request, *self._run(request, cancellation, emit))

    def reset(self):
        pass

    def close(self):
        self.native.close()


def eval_engine(context, library, defaults=None):
    return EvalEngine(context, library, defaults)


def resident_engine(context, library, defaults=None):
    return ResidentEngine(context, library, defaults)


__all__ = ["NativeEngine", "NativeError", "EvalEngine", "ResidentEngine", "eval_engine", "resident_engine"]
