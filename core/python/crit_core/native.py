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
import copy
import threading
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path

from crit_core.serving import SAMPLING, TOP_LOGPROBS, ResidentChat, chat_prompt  # noqa: F401  (chat_prompt is re-exported)

ABI = 1
RESULT_BYTES = 1 << 20  # a generation's result JSON: finish, timing and the engine's request facts
TOKEN_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int)
OUTPUT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_float,
                                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int)
PENALTY, PENALTIES = "penalties", {"repetition_penalty", "presence_penalty", "frequency_penalty"}
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
        self._condition = threading.Condition()
        self._serial = threading.RLock()
        self._active = 0
        self._closing = False
        self._requests = hasattr(lib, "crit_engine_request_open")
        if self._requests:
            lib.crit_engine_request_open.restype = ctypes.c_void_p
            lib.crit_engine_request_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
            for name in ("cancel", "close"):
                function = getattr(lib, "crit_engine_request_" + name)
                function.restype = None
                function.argtypes = [ctypes.c_void_p]
            lib.crit_engine_request_wait.restype = ctypes.c_int
            lib.crit_engine_request_wait.argtypes = [ctypes.c_void_p, TOKEN_CALLBACK, ctypes.c_void_p,
                                                    ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_int]
        # Per-token log-probabilities (libraries built against a header that exports request_wait_output).
        self.outputs = self._requests and hasattr(lib, "crit_engine_request_wait_output")
        if self.outputs:
            lib.crit_engine_request_wait_output.restype = ctypes.c_int
            lib.crit_engine_request_wait_output.argtypes = [ctypes.c_void_p, OUTPUT_CALLBACK, ctypes.c_void_p,
                                                           ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p,
                                                           ctypes.c_int]
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

    @contextmanager
    def _lease(self):
        with self._condition:
            if self._closing or not self.handle:
                raise NativeError("engine is closed")
            self._active += 1
        try:
            concurrent = self._requests and self.facts.get("max_concurrent_requests", 1) > 1
            with nullcontext() if concurrent else self._serial:
                yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def generate(self, prompt, output_tokens, **options):
        with self._lease():
            return self._generate(prompt, output_tokens, **options)

    def _generate(self, prompt, output_tokens, *, stop_tokens=None, sampling=None, replay=None, plain=False,
                 logprobs=None, on_tokens=None, cancelled=None):
        """Returns (tokens, result) with result's "finish", "timing" (ms, decode_steps) and the engine's request
        facts, or (None, {"cancelled": True}) when cancelled() turned true. stop_tokens None keeps the model's;
        [] disables them. sampling: temperature, top_p, top_k, min_p, seed, penalties and logit_bias.
        plain: decode without drafting (engines that declare "plain"). logprobs n: on_tokens(tokens, rows) gets one
        (logprob, [(token, logprob), ...n alternatives]) row per token (engines whose info reports token_logprobs
        and max_top_logprobs >= n)."""
        if cancelled and cancelled():
            return None, {"cancelled": True}
        options = dict(sampling or {})
        if stop_tokens is not None:
            options["stop_tokens"] = [int(t) for t in stop_tokens]
        if replay is not None:
            options["replay"] = [int(t) for t in replay]
        if plain:
            options["plain"] = True
        if logprobs is not None:
            if not self.outputs:
                raise NativeError(f"{self.path} has no per-token log-probabilities: rebuild it against this crit's "
                                  "crit/native_engine.h")
            options["logprobs"] = int(logprobs)
        failure = []

        def guarded(publish):
            try:
                if cancelled and cancelled():
                    return 1
                if on_tokens:
                    publish()
                return 1 if cancelled and cancelled() else 0
            except BaseException as error:  # never unwind through C; re-raised below
                failure.append(error)
                return 1

        def deliver(_user, tokens, count):
            return guarded(lambda: on_tokens([tokens[i] for i in range(count)]))

        def deliver_output(_user, token, logprob, top_tokens, top_logprobs, count):
            return guarded(lambda: on_tokens([token], [(logprob, [(top_tokens[i], top_logprobs[i])
                                                                  for i in range(count)])]))

        callback = TOKEN_CALLBACK(deliver) if logprobs is None else OUTPUT_CALLBACK(deliver_output)
        ids = (ctypes.c_int32 * len(prompt))(*prompt)
        out = (ctypes.c_int32 * max(1, output_tokens))()
        result = ctypes.create_string_buffer(RESULT_BYTES)
        encoded = json.dumps(options).encode("utf-8")
        if self._requests:
            request = self.lib.crit_engine_request_open(self.handle, ids, len(prompt), int(output_tokens), encoded)
            if not request:
                raise NativeError("generation failed: " + self._error())
            done = threading.Event()
            monitor = None

            def watch():
                while not done.is_set():
                    try:
                        stop = cancelled()
                    except BaseException as error:
                        failure.append(error)
                        stop = True
                    if stop:
                        self.lib.crit_engine_request_cancel(request)
                        return
                    done.wait(0.01)

            try:
                if cancelled:
                    monitor = threading.Thread(target=watch, name="crit-native-cancel", daemon=True)
                    monitor.start()
                wait = self.lib.crit_engine_request_wait if logprobs is None else self.lib.crit_engine_request_wait_output
                count = wait(request, callback, None, out, result, RESULT_BYTES)
                error = self._error() if count < 0 else ""
            finally:
                done.set()
                if monitor is not None and monitor.ident is not None:
                    monitor.join()
                self.lib.crit_engine_request_close(request)
        else:
            count = self.lib.crit_engine_generate(self.handle, ids, len(prompt), int(output_tokens),
                                                 encoded, callback, None, out, result, RESULT_BYTES)
            error = self._error() if count < 0 else ""
        if failure:
            raise failure[0]
        if count == -2:
            return None, {"cancelled": True}
        if count < 0:
            raise NativeError("generation failed: " + error)
        return [out[i] for i in range(count)], json.loads(result.value.decode("utf-8"))

    def score(self, ids, first_target, path="", queries=None):
        with self._lease():
            return self._score(ids, first_target, path, queries)

    def _score(self, ids, first_target, path="", queries=None):
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
        with self._condition:
            self._closing = True
            while self._active:
                self._condition.wait()
            if self.handle:
                self.lib.crit_engine_close(self.handle)
                self.handle = None


def decode_rounds(tokens, result, speculative):
    """Target passes after prefill: the engine's own count, else one per accepted entry, else Core's decode steps
    (which also count steps that run no target pass, such as seeding a drafter); one per token after the first
    without drafting."""
    if "rounds" in result:
        return result["rounds"]
    if "accepted" in result:
        return len(result["accepted"])
    return result["timing"]["decode_steps"] if speculative else max(0, len(tokens) - 1)


class EvalEngine:
    """eval-v1 (docs/eval/ADAPTER.md) over a native engine: greedy generation with stop tokens disabled, replay
    and plain decoding when the engine declares them, and its headroom facts."""

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
                "native_libraries": [str(self.native.path)], "replay": bool(self.facts.get("replay")),
                "plain": bool(self.speculative and self.facts.get("plain"))}
        info.update({key: self.facts[key] for key in FACTS if key in self.facts})
        return info

    def _result(self, tokens, result):
        out = {"tokens": tokens, "rounds": decode_rounds(tokens, result, self.speculative)}
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

    def close(self):
        self.native.close()


class ResidentEngine(ResidentChat):
    """chat-resident-v1 (docs/SERVING.md) over a native engine: crit_core.serving.ResidentChat with the engine's
    stop tokens, context and the sampling fields its info declares ("sampling": {temperature, top_p, top_k, min_p,
    seed, penalties, logit_bias}), and token log-probabilities (chat-logprobs-v1) when its info reports
    token_logprobs with max_top_logprobs of at least 20. Configuration "chat_template_kwargs" passes variables to the
    chat template."""

    def __init__(self, context, library, defaults=None):
        configuration = {**(defaults or {}), **(context.configuration or {})}
        model = Path(context.model_file("config.json")).parent
        super().__init__(context, model, configuration=configuration)
        limits = getattr(context, "limits", {}) or {}
        native = dict(self.configuration, model=str(model))
        if limits.get("kv_tokens"):
            native["context_tokens"] = int(limits["kv_tokens"])
        self.native = NativeEngine(library, native)
        self.facts = self.native.info()
        self.max_concurrent_requests = (int(self.facts.get("max_concurrent_requests", 1))
                                        if self.native._requests else 1)
        self._owns_native = True
        self.capacity = min(int(self.facts["context"]), int(limits.get("kv_tokens") or self.facts["context"]))
        self.stop_ids = set(self.facts.get("stop_tokens") or [])
        support = self.facts.get("sampling") or {}
        self.sampling = tuple(key for key in SAMPLING if support.get(PENALTY if key in PENALTIES else key))
        self.logprobs = bool(self.native.outputs and self.facts.get("token_logprobs")
                             and int(self.facts.get("max_top_logprobs", 0)) >= TOP_LOGPROBS)

    def create_request(self, request):
        # Share only the immutable tokenizer/configuration and the synchronized native
        # engine. ResidentChat's parser discovery caches belong to this wrapper.
        child = copy.copy(self)
        child._owns_native = False
        child.configuration = dict(self.configuration)
        child.template_variables = dict(self.template_variables)
        child.stop_ids = set(self.stop_ids)
        child.kept_special = child.calls_format = None
        child.__dict__.pop("_kept_cache", None)
        child.__dict__.pop("_thinking", None)
        return child

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, logprobs=None):
        tokens, result = self.native.generate(ids, max_new_tokens, on_tokens=on_tokens, sampling=sampling,
                                              logprobs=logprobs, cancelled=cancelled)
        if tokens is None:
            return None, {}
        timing = result["timing"]
        return tokens, {"prefill_tokens": timing["prefill_tokens"] - timing["reused_prompt_tokens"],
                        "prefill_ms": timing["prefill_ms"], "decode_ms": timing["decode_ms"],
                        "first_token_ms": timing["first_token_ms"],
                        "rounds": decode_rounds(tokens, result, bool(self.facts.get("speculative")))}

    def close(self):
        if self._owns_native:
            self.native.close()


def eval_engine(context, library, defaults=None):
    return EvalEngine(context, library, defaults)


def resident_engine(context, library, defaults=None):
    return ResidentEngine(context, library, defaults)


__all__ = ["NativeEngine", "NativeError", "EvalEngine", "ResidentEngine", "chat_prompt", "eval_engine",
           "resident_engine"]
