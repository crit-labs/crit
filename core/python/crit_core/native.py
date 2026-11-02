"""Native engines from Python: the ctypes side of Core's engine ABI (crit/native_engine.h) and the stock adapters.

An engine library exports its core::Engine with CRIT_EXPORT_ENGINE(factory). Its package then needs no glue of
its own beyond two small entry files:

    # eval adapter (eval-v1, crit-eval's docs/ADAPTER.md)
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

A Python engine gets the same two adapters by implementing NativeEngine's interface (`info()`, `generate()`,
`score()`, `close()`, with the same arguments, results and info facts) and passing the function that opens it in
place of the library path:

    def open_engine(configuration):      # the same JSON object a native factory receives
        return MyEngine(configuration)   # info(), generate(...), score(...), close()

    def create_eval_engine(context):
        return eval_engine(context, open_engine, defaults={...})

    def create_engine(context):
        return resident_engine(context, open_engine, defaults={...})

The adapters then treat both alike, so a Python engine writes no adapter code and cannot drift from what Eval and
crit serve expect. Its generations and scores run one at a time unless its info declares max_concurrent_requests
above 1, and close() waits for those in flight.

A Python engine that batches concurrent requests implements a step instead of generate(): a step model (StepEngine)
opens a session per request and computes one row per scheduled request in step(rows), and Core's concurrent runtime
(crit/step_engine.h, through the crit_core_steps library) schedules, admits, cancels and retires its requests:

    def open_engine(configuration):
        return MyStepModel(configuration)   # info(), open(request), step(rows), commit(...), retire(...), close()

The same eval_engine and resident_engine serve it, max_concurrent_requests at once.
"""
import array
import ctypes
import os
from collections import namedtuple
import copy
import gc
import threading
import time
from contextlib import contextmanager, nullcontext
import json
import sys
from pathlib import Path

from crit_core.machine import under_wsl
from crit_core.serving import SAMPLING, TOP_LOGPROBS, ResidentChat, ScoreRows

ABI = 2
RESULT_BYTES = 1 << 20  # a generation's result JSON: finish, timing and the engine's request facts
TOKEN_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int)
OUTPUT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_float,
                                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int)
PENALTY, PENALTIES = "penalties", {"repetition_penalty", "presence_penalty", "frequency_penalty"}
# The penalties at the values that change nothing, for requests that must not take an engine's defaults (Eval's).
NEUTRAL_PENALTIES = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
# Engine info facts the eval-v1 adapter passes on (crit-eval's docs/ADAPTER.md): headroom inputs, kernel classes, the
# prompt GEMMs' activation formats, the configuration overrides of --attribute-drift and of a wider-draft probe, and
# prompt_cache false for an engine without a prompt cache.
FACTS = ("draft_width", "bytes_per_pass", "draft_bytes_per_step", "draft_ms", "kv_bytes_per_token", "kernel_work",
         "prompt_activations", "exact_variants", "wider_draft", "prompt_cache")
# NativeEngine.score per target: its log-probability, the query rows, the argmax and the engine's own ranked top_k
# [(token, logprob), ...] (None for parts not asked for or not returned).
Scores = namedtuple("Scores", "target query argmax top")


# Whether CUDA runs under the Windows display driver model: native Windows, or WSL's paravirtualized GPU.
WDDM = sys.platform == "win32" or under_wsl()


class NativeError(RuntimeError):
    pass


class _Requests:
    """Generations and scores in flight on one engine: one at a time unless its facts declare
    max_concurrent_requests above 1; close() waits for them."""

    def _start_requests(self):
        self._condition = threading.Condition()
        self._serial = threading.RLock()
        self._active = 0
        self._closing = False

    @contextmanager
    def _lease(self):
        with self._condition:
            if self._closing or not self._usable():
                raise NativeError("engine is closed")
            self._active += 1
        try:
            concurrent = self.facts.get("max_concurrent_requests", 1) > 1
            with nullcontext() if concurrent else self._serial:
                yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _drain(self):
        """Refuse new requests and wait for those in flight; call with self._condition held."""
        self._closing = True
        while self._active:
            self._condition.wait()


class NativeEngine(_Requests):
    """One engine opened from a library exported with CRIT_EXPORT_ENGINE."""

    def __init__(self, library, configuration):
        self._load(library)
        handle = self.lib.crit_engine_open(json.dumps(configuration).encode("utf-8"))
        if not handle:
            raise NativeError("opening the engine failed: " + self._error())
        self._attach(handle)

    def _load(self, library):
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
        lib.crit_engine_score.restype = ctypes.c_int
        lib.crit_engine_score.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                          ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                          ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)]
        lib.crit_engine_last_error.restype = ctypes.c_char_p
        self._start_requests()
        lib.crit_engine_request_open.restype = ctypes.c_void_p
        lib.crit_engine_request_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                                ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        for name in ("cancel", "close"):
            function = getattr(lib, "crit_engine_request_" + name)
            function.restype = None
            function.argtypes = [ctypes.c_void_p]
        for name, callback in (("wait", TOKEN_CALLBACK), ("wait_output", OUTPUT_CALLBACK)):
            function = getattr(lib, "crit_engine_request_" + name)
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_void_p, callback, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                 ctypes.c_char_p, ctypes.c_int]
        self.lib = lib

    def _attach(self, handle):
        """Adopts an opened engine handle and reads its facts."""
        self.handle = handle
        size = 1 << 16
        buffer = ctypes.create_string_buffer(size)
        length = self.lib.crit_engine_info(self.handle, buffer, size)
        if length >= size:
            buffer = ctypes.create_string_buffer(length + 1)
            length = self.lib.crit_engine_info(self.handle, buffer, length + 1)
        if length < 0:
            raise NativeError("engine info failed: " + self._error())
        self.facts = json.loads(buffer.value.decode("utf-8"))

    def _error(self):
        return (self.lib.crit_engine_last_error() or b"").decode("utf-8", "replace")

    def info(self):
        """ModelInfo fields (implementation, context, stop_tokens, score_paths, ...) plus the factory's facts."""
        return dict(self.facts)

    def _usable(self):
        return bool(self.handle)

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
        if failure:
            raise failure[0]
        if count == -2:
            return None, {"cancelled": True}
        if count < 0:
            raise NativeError("generation failed: " + error)
        return [out[i] for i in range(count)], json.loads(result.value.decode("utf-8"))

    def score(self, ids, first_target, path="", queries=None, top_k=0, packed=False):
        """Scores (crit_engine_score) of every target ids[i], i >= first_target: its log-probability, the
        log-probabilities of its row of `queries` (one equally long row of token ids per target), the argmax when the
        engine returns one, and with top_k the engine's own top_k ranked [(token, logprob), ...]. path: one of the
        engine's score_paths ("": the default). With `packed`, `top` is the targets' rows as ScoreRows of the
        engine's arrays (float32 log-probabilities), copied without a Python object per value."""
        with self._lease():
            targets = len(ids) - first_target
            per = len(queries[0]) if queries else 0
            flat = [int(t) for row in (queries or []) for t in row]
            if len(flat) != targets * per:
                raise NativeError("score queries need one equally long row per target")
            target = (ctypes.c_float * max(1, targets))()
            query = (ctypes.c_float * max(1, targets * per))()
            argmax = (ctypes.c_int32 * max(1, targets))()
            has_argmax = ctypes.c_int(0)
            tokens = (ctypes.c_int32 * max(1, targets * top_k))()
            values = (ctypes.c_float * max(1, targets * top_k))()
            code = self.lib.crit_engine_score(self.handle, (ctypes.c_int32 * len(ids))(*ids), len(ids),
                                              int(first_target), (path or "").encode("utf-8"),
                                              (ctypes.c_int32 * max(1, len(flat)))(*flat) if flat else None, per,
                                              int(top_k), target, query, argmax, ctypes.byref(has_argmax), tokens,
                                              values)
            if code != 0:
                raise NativeError("scoring failed: " + self._error())
            if packed and top_k:
                # Byte copies of the engine's arrays; c_int32 and c_float are array.array's "i" and "f".
                copied = lambda kind, data, count: array.array(kind, bytes(data)[:count * ctypes.sizeof(data._type_)])
                top = ScoreRows(copied("f", target, targets), copied("i", tokens, targets * top_k),
                                copied("f", values, targets * top_k), top_k)
            else:
                top = ([[(tokens[t * top_k + k], values[t * top_k + k]) for k in range(top_k)] for t in range(targets)]
                       if top_k else None)
            return Scores(list(target)[:targets],
                          [[query[t * per + j] for j in range(per)] for t in range(targets)] if per else None,
                          list(argmax)[:targets] if has_argmax.value else None, top)

    def close(self):
        with self._condition:
            self._drain()
            if self.handle:
                self.lib.crit_engine_close(self.handle)
                self.handle = None


class PythonEngine(_Requests):
    """A Python engine with NativeEngine's interface, opened by `opener(configuration)`: info() returns the facts a
    native engine's info holds (implementation or name, context, stop_tokens, sampling, speculative, plain, replay,
    token_logprobs, max_top_logprobs, query_scoring, score_paths, the eval-v1 facts in FACTS; native_libraries for
    libraries it loads); generate(prompt, output_tokens, *, stop_tokens, sampling, replay, plain, logprobs,
    on_tokens, cancelled) and score(ids, first_target, path, queries, top_k, packed) as NativeEngine documents them;
    close(). This wrapper only adds NativeEngine's request accounting."""

    path = None   # no library: eval-v1 takes native_libraries from the engine's facts

    def __init__(self, opener, configuration):
        self.engine = opener(dict(configuration))
        self.facts = dict(self.engine.info())
        self._start_requests()
        self._oversubscribed = set()
        self._check_device_memory()

    def _usable(self):
        return self.engine is not None

    def info(self):
        return dict(self.facts)

    def generate(self, prompt, output_tokens, **options):
        with self._lease():
            if options.get("cancelled") and options["cancelled"]():
                return None, {"cancelled": True}
            try:
                return self.engine.generate(list(prompt), int(output_tokens), **options)
            finally:
                self._check_device_memory()

    def score(self, ids, first_target, path="", queries=None, top_k=0, packed=False):
        with self._lease():
            try:
                return self.engine.score(list(ids), int(first_target), path, queries=queries, top_k=top_k,
                                         packed=packed)
            finally:
                self._check_device_memory()

    def _check_device_memory(self):
        """Warn once per device when torch has reserved more than the device holds. Under the Windows display driver
        (WDDM, native Windows and WSL) such allocations do not fail: the driver moves the excess to system memory, and
        every kernel that reads it slows down by the host link's bandwidth, so the engine only gets slower."""
        if not WDDM:
            return
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return
        for device in range(torch.cuda.device_count()):
            if device in self._oversubscribed:
                continue
            peak, total = torch.cuda.max_memory_reserved(device), torch.cuda.get_device_properties(device).total_memory
            if peak > total:
                self._oversubscribed.add(device)
                sys.stderr.write(
                    f"crit_core: the engine reserved {peak / 2**30:.1f} GiB on cuda:{device}, which holds "
                    f"{total / 2**30:.1f} GiB; under WDDM the excess runs from system memory and slows every kernel "
                    "that reads it: keep the engine's peak (weights, caches, prompt activations) below the device's "
                    "memory\n")

    def close(self):
        with self._condition:
            self._drain()
            if self.engine is not None:
                self.engine.close()
                self.engine = None
                # A Python engine's device memory returns only once its objects are freed. Parts that refer to each
                # other (a drafter and its model) wait for a collection, and crit serve reopens the package in the
                # same host (Eval's lower-context comparison), so collect now and hand torch's cached blocks back.
                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()


# The crit_core_steps library (crit/step_engine.h) a StepEngine loads, when the model's info names no "step_library".
STEP_LIBRARY = "CRIT_CORE_STEP_LIBRARY"
# The settings of a step model's info that configure Core (crit/step_engine.h's "engine" object); the rest are facts.
STEP_SETTINGS = ("implementation", "vocabulary", "valid_vocabulary", "context", "prefill_chunk", "stop_tokens",
                 "max_concurrency", "max_pending", "max_batch_tokens", "batch_policy", "max_row_tokens",
                 "decode_token_work", "sampling", "token_logprobs", "max_top_logprobs", "scoring", "score_paths",
                 "host_limit")
FINISHES = ("length", "stop", "cancelled", "failed")
SAMPLING_OPTIONS = ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty", "frequency_penalty",
                    "presence_penalty", "logit_bias", "min_tokens")


class _StepRequest(ctypes.Structure):
    _fields_ = [("request", ctypes.c_uint64), ("generation", ctypes.c_uint64), ("lane", ctypes.c_uint32),
                ("prompt_tokens", ctypes.c_uint32), ("prompt", ctypes.POINTER(ctypes.c_int32)),
                ("output_tokens", ctypes.c_uint32), ("reserved", ctypes.c_uint32), ("options_json", ctypes.c_char_p)]


class _StepRow(ctypes.Structure):
    _fields_ = [("session", ctypes.c_uint64), ("prefill_tokens", ctypes.c_uint32), ("output_tokens", ctypes.c_uint32),
                ("token_work", ctypes.c_uint32), ("prefill_complete", ctypes.c_int32)]


class _StepProgress(ctypes.Structure):
    _fields_ = [("prompt_tokens", ctypes.c_uint32), ("reused_prompt_tokens", ctypes.c_uint32),
                ("token_work", ctypes.c_uint32), ("tokens", ctypes.c_uint32), ("prefill_complete", ctypes.c_int32),
                ("reserved", ctypes.c_int32)]


class _StepDecision(ctypes.Structure):
    _fields_ = [("session", ctypes.c_uint64), ("accepted", ctypes.c_uint32), ("terminal", ctypes.c_int32),
                ("cancelled", ctypes.c_int32), ("discard", ctypes.c_int32)]


_I32, _F32 = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
_ADMISSION = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_StepRequest))
_OPEN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_StepRequest), ctypes.POINTER(ctypes.c_uint64))
_STEP = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_StepRow), ctypes.c_uint32,
                         ctypes.POINTER(_StepProgress), _I32, _F32, _I32, _F32)
_COMMIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_StepDecision), ctypes.c_uint32)
_RETIRE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)
_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64)
_SCORE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, _I32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, _I32,
                          ctypes.c_uint32, ctypes.c_uint32, _F32, _F32, _I32, _I32, _I32, _F32)
_ERROR = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)


class _StepCallbacks(ctypes.Structure):
    _fields_ = [("user", ctypes.c_void_p), ("admission", _ADMISSION), ("open", _OPEN), ("step", _STEP),
                ("commit", _COMMIT), ("retire", _RETIRE), ("release", _RELEASE), ("score", _SCORE), ("error", _ERROR)]


# What a step model's open() and admission() receive: the request's key (request, generation, lane), prompt and output
# limit; the sampling fields it set; the effective stop tokens; alternatives per token (None: no log-probabilities); a
# forced continuation (replay) or plain decoding (crit/native_engine.h's ReplayContinuation and PlainDecoding).
StepRequest = namedtuple("StepRequest", "key prompt output_tokens sampling stop_tokens logprobs replay plain")
# One scheduled row of a step: the session open() returned and its WorkBudget (crit/core.h).
StepRow = namedtuple("StepRow", "session prefill_tokens output_tokens token_work prefill_complete")
# What step() returns per row, in the rows' order: prompt tokens consumed (reused ones included), the tokens produced
# (at most max_row_tokens), whether prefill is complete, the reused prefix, the token work computed (0: the prompt
# tokens computed, or one per token) and, for requests asking for them, one (logprob, [(token, logprob), ...]) per token.
StepResult = namedtuple("StepResult", "prompt_tokens tokens prefill_complete reused_prompt_tokens token_work logprobs",
                        defaults=(0, (), False, 0, 0, None))
# Core's decision for a row of the last step: the tokens it kept, whether the request ends, was cancelled, or the row is
# discarded.
StepDecision = namedtuple("StepDecision", "session accepted terminal cancelled discard")


class StepEngine(NativeEngine):
    """A Python step model served by Core's concurrent runtime (crit/step_engine.h, core/src/runtime/README.md "Python
    engines"): Core schedules its requests in mixed prefill/decode steps, admits, applies stop tokens and output
    limits, cancels and retires them, and runs scores as exclusive barriers; the model implements

        info()                 facts as a Python engine's (implementation, stop_tokens, sampling, speculative, replay,
                               ...) plus Core's settings: "vocabulary" and "context" (required), "max_concurrency",
                               "max_pending", "max_batch_tokens", "prefill_chunk", "batch_policy", "max_row_tokens",
                               "token_logprobs", "max_top_logprobs", "scoring" (STEP_SETTINGS)
        open(request)          a session for a StepRequest
        step(rows)             one StepResult per StepRow, computed together
        commit(decisions)      optional: Core's StepDecision per row of the last step
        retire(session, finish) optional: "length", "stop", "cancelled" or "failed"
        admission(request)     optional: "ready", "deferred" or "infeasible"
        score(ids, first_target, path, queries, top_k)  with "scoring": Scores, as NativeEngine.score returns them
        close()

    on Core's one engine worker, never two at once. It generates and scores through NativeEngine's interface, with
    max_concurrent_requests (max_concurrency + max_pending) requests at once. `library` is crit_core_steps, else the
    info's "step_library", else CRIT_CORE_STEP_LIBRARY."""

    def __init__(self, model, library=None):
        self.model = model
        info = dict(model.info())
        library = library or info.pop("step_library", None) or os.environ.get(STEP_LIBRARY)
        info.pop("step_library", None)
        if not library:
            raise NativeError("a step engine needs Core's crit_core_steps library: build Core (core/README.md) and set "
                              f"{STEP_LIBRARY} to it, or name it in the model's info as step_library")
        self._load(library)
        self.library, self.path = self.path, None   # Eval takes it from the facts' native_libraries
        self.lib.crit_engine_open_steps.restype = ctypes.c_void_p
        self.lib.crit_engine_open_steps.argtypes = [ctypes.c_char_p, ctypes.POINTER(_StepCallbacks)]
        settings = {key: info[key] for key in STEP_SETTINGS if key in info}
        facts = {key: value for key, value in info.items() if key not in STEP_SETTINGS}
        facts["native_libraries"] = [str(self.library)] + list(facts.get("native_libraries") or [])
        self._row_tokens = int(settings.get("max_row_tokens", 1))
        self._top = int(settings.get("max_top_logprobs", 0))
        self._logprobs = bool(settings.get("token_logprobs"))
        self._sessions, self._next, self._failure = {}, 1, ""
        optional = lambda name, kind, method: kind(method) if callable(getattr(model, name, None)) else kind()
        self._callbacks = _StepCallbacks(
            None, optional("admission", _ADMISSION, self._admission), _OPEN(self._open), _STEP(self._step),
            optional("commit", _COMMIT, self._commit), _RETIRE(self._retire), _RELEASE(self._release),
            _SCORE(self._score) if settings.get("scoring") else _SCORE(), _ERROR(self._error_message))
        handle = self.lib.crit_engine_open_steps(json.dumps({"engine": settings, "facts": facts}).encode("utf-8"),
                                                 ctypes.byref(self._callbacks))
        if not handle:
            raise NativeError("opening the step engine failed: " + self._error())
        self._attach(handle)

    def _guard(self, run):
        """Runs a callback body; an exception is kept for error() and fails the engine (never unwinds through C)."""
        try:
            result = run()
            return 0 if result is None else result
        except BaseException as error:
            self._failure = f"{type(error).__name__}: {error}"
            return -1

    def _error_message(self, _user, out, capacity):
        if capacity < 1:
            return 0
        data = self._failure.encode("utf-8", "replace")[:capacity - 1]
        ctypes.memmove(out, data + b"\0", len(data) + 1)
        return len(data)

    @staticmethod
    def _request(pointer):
        r = pointer.contents
        options = json.loads(r.options_json.decode("utf-8")) if r.options_json else {}
        return StepRequest((r.request, r.generation, r.lane), r.prompt[:r.prompt_tokens], r.output_tokens,
                           {key: options[key] for key in SAMPLING_OPTIONS if key in options},
                           options.get("stop_tokens", []), options.get("logprobs"), options.get("replay"),
                           bool(options.get("plain")))

    def _admission(self, _user, request):
        def run():
            answer = self.model.admission(self._request(request))
            if answer not in ("ready", "deferred", "infeasible"):
                raise ValueError(f"admission must answer ready, deferred or infeasible, not {answer!r}")
            return ("ready", "deferred", "infeasible").index(answer)
        return self._guard(run)

    def _open(self, _user, request, session):
        def run():
            handle = self._next
            self._sessions[handle] = self.model.open(self._request(request))
            self._next += 1
            session[0] = handle
        return self._guard(run)

    def _step(self, _user, rows, count, progress, tokens, logprobs, top_ids, top_logprobs):
        def run():
            scheduled = [StepRow(self._sessions[rows[i].session], rows[i].prefill_tokens, rows[i].output_tokens,
                                 rows[i].token_work, bool(rows[i].prefill_complete)) for i in range(count)]
            results = list(self.model.step(scheduled))
            if len(results) != count:
                raise ValueError(f"step returned {len(results)} results for {count} rows")
            width, top = self._row_tokens, self._top
            for i, result in enumerate(results):
                result = StepResult(*result) if not isinstance(result, StepResult) else result
                produced = list(result.tokens)
                if len(produced) > width:
                    raise ValueError(f"a row produced {len(produced)} tokens, more than max_row_tokens {width}")
                p = progress[i]
                p.prompt_tokens, p.reused_prompt_tokens = int(result.prompt_tokens), int(result.reused_prompt_tokens)
                p.token_work, p.tokens, p.prefill_complete = int(result.token_work), len(produced), int(bool(result.prefill_complete))
                for j, token in enumerate(produced):
                    tokens[i * width + j] = int(token)
                if self._logprobs and result.logprobs is not None:
                    for j, (logprob, alternatives) in enumerate(list(result.logprobs)[:len(produced)]):
                        logprobs[i * width + j] = float(logprob)
                        for a, (token, value) in enumerate(list(alternatives)[:top]):
                            top_ids[(i * width + j) * top + a] = int(token)
                            top_logprobs[(i * width + j) * top + a] = float(value)
        return self._guard(run)

    def _commit(self, _user, decisions, count):
        def run():
            self.model.commit([StepDecision(self._sessions[d.session], d.accepted, bool(d.terminal), bool(d.cancelled),
                                            bool(d.discard)) for d in (decisions[i] for i in range(count))])
        return self._guard(run)

    def _retire(self, _user, session, finish):
        def run():
            if callable(getattr(self.model, "retire", None)):
                self.model.retire(self._sessions[session], FINISHES[finish])
        return self._guard(run)

    def _release(self, _user, session):
        self._sessions.pop(session, None)

    def _score(self, _user, ids, count, first_target, path, queries, per_target, top_k, target, query, argmax,
               has_argmax, top_ids, top_logprobs):
        def run():
            targets = count - first_target
            rows = ([[queries[t * per_target + j] for j in range(per_target)] for t in range(targets)]
                    if per_target else None)
            scored = self.model.score(ids[:count], first_target, (path or b"").decode("utf-8"), queries=rows, top_k=top_k)
            if len(scored.target) != targets:
                raise ValueError(f"score returned {len(scored.target)} targets, expected {targets}")
            for t, value in enumerate(scored.target):
                target[t] = float(value)
            for t, row in enumerate(scored.query or []):
                for j, value in enumerate(row):
                    query[t * per_target + j] = float(value)
            has_argmax[0] = int(scored.argmax is not None)
            for t, token in enumerate(scored.argmax or []):
                argmax[t] = int(token)
            for t, row in enumerate((scored.top or []) if top_k else []):
                for a, (token, value) in enumerate(row):
                    top_ids[t * top_k + a] = int(token)
                    top_logprobs[t * top_k + a] = float(value)
        return self._guard(run)

    def close(self):
        super().close()
        if self.model is not None:
            if callable(getattr(self.model, "close", None)):
                self.model.close()
            self.model = None
            self._sessions.clear()


def open_engine(engine, configuration):
    """NativeEngine for a library path; for a callable that opens a Python engine, a StepEngine when the engine is a
    step model (it has step()), else a PythonEngine."""
    if not callable(engine):
        return NativeEngine(engine, configuration)
    opened = engine(dict(configuration))
    if callable(getattr(opened, "step", None)):
        return StepEngine(opened)
    return PythonEngine(lambda _configuration: opened, configuration)


def engine_times(result):
    """The engine's own prefill and first-token milliseconds from a generation's timing (Core's prefill_ms and
    first_token_ms), for crit-eval's time-to-first-token breakdown; the ones it does not report are left out."""
    timing = (result or {}).get("timing") or {}
    return {key: float(timing[key]) for key in ("prefill_ms", "first_token_ms")
            if isinstance(timing.get(key), (int, float)) and not isinstance(timing[key], bool)}


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
    """eval-v1 (crit-eval's docs/ADAPTER.md) over a native engine: greedy generation with stop tokens disabled, replay
    and plain decoding when the engine declares them, fixed-text scores (`score`, for `crit eval run --only quality`
    in process, development evidence) when it declares token_logprobs, query_scoring and replay, the engine's own
    prefill and first-token times with every generation, and its headroom facts."""

    def __init__(self, context, library, defaults=None):
        """`library`: an engine library's path, or a function opening a Python engine (open_engine)."""
        configuration = {**(defaults or {}), **(context.get("config") or {})}
        # Eval's speculative=False (plain one-token decoding) overrides the configuration; it never turns drafting on.
        speculative = bool(configuration.get("speculative", True)) and bool(context.get("speculative", True))
        configuration.update(model=str(context["model"]), speculative=speculative)
        if context.get("context_tokens"):
            configuration["context_tokens"] = int(context["context_tokens"])
        self.configuration = configuration
        self.native = open_engine(library, configuration)
        self.facts = self.native.info()
        self.speculative = bool(self.facts.get("speculative"))
        # Eval decodes greedily: request temperature zero where the engine samples (a greedy-only engine needs none),
        # and neutral penalties where it applies them, whatever its defaults.
        support = self.facts.get("sampling") or {}
        self.sampling = {**({"temperature": 0.0} if support.get("temperature") else {}),
                         **(NEUTRAL_PENALTIES if support.get(PENALTY) else {})}
        if self.facts.get("replay"):
            self.replay = self._replay
        if self.speculative and self.facts.get("plain"):
            self.generate_plain = self._generate_plain
        # Eval's sampled suite (crit-eval-v7): the engine samples at the settings Eval names, applying these fields.
        self.sampling_fields = {key for key in SAMPLING if support.get(PENALTY if key in PENALTIES else key)}
        if support.get("temperature"):
            self.generate_sampled = self._generate_sampled
        # Fixed text scored as crit serve scores it (chat-score-v1, ResidentEngine.score_tokens).
        if self.facts.get("token_logprobs") and self.facts.get("query_scoring") and self.facts.get("replay"):
            self.score = self._score

    def info(self):
        info = {"name": self.facts.get("name") or self.facts["implementation"], "configuration": self.configuration,
                "max_context": self.facts["context"], "speculative": self.speculative,
                "native_libraries": ([str(self.native.path)] if self.native.path is not None
                                     else list(self.facts.get("native_libraries") or [])),
                "replay": bool(self.facts.get("replay")),
                "plain": bool(self.speculative and self.facts.get("plain"))}
        info.update({key: self.facts[key] for key in FACTS if key in self.facts})
        return info

    def _result(self, tokens, result):
        out = {"tokens": tokens, "rounds": decode_rounds(tokens, result, self.speculative)}
        out.update({key: result[key] for key in ("accepted", "drafted", "verify_ms", "draft_ms", "expert_reads",
                                                 "expert_hits") if key in result})
        return {**out, **engine_times(result)}

    def generate(self, prompt_ids, max_new_tokens, on_tokens):
        tokens, result = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[],
                                              sampling=self.sampling, on_tokens=on_tokens)
        return self._result(tokens, result)

    def _generate_sampled(self, prompt_ids, max_new_tokens, on_tokens, sampling, logprobs=None):
        """A sampled request; with logprobs n, the emitted token's log-probability and n alternatives per token, when
        the engine reports them (token_logprobs, max_top_logprobs). Fields the engine does not apply
        (sampling_fields) are left out; Eval sends such fields only at their neutral values."""
        sampling = {key: value for key, value in sampling.items() if key in self.sampling_fields}
        rows = []
        reports = logprobs and self.facts.get("token_logprobs") and int(self.facts.get("max_top_logprobs", 0)) >= logprobs

        def deliver(tokens, found=None):
            on_tokens(tokens)
            if found:
                rows.extend(zip(tokens, found))
        tokens, result = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[], sampling=dict(sampling),
                                              logprobs=logprobs if reports else None, on_tokens=deliver)
        out = self._result(tokens, result)
        if reports:
            out["logprobs"] = [{"token": token, "logprob": logprob, "top": [list(pair) for pair in top]}
                               for token, (logprob, top) in rows]
        return out

    def _generate_plain(self, prompt_ids, max_new_tokens, on_tokens):
        # One target pass per token after the first; the engine's request facts describe drafting, so none apply.
        tokens, result = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[], sampling=self.sampling,
                                              plain=True, on_tokens=on_tokens)
        return {"tokens": tokens, "rounds": max(0, len(tokens) - 1), **engine_times(result)}

    def _score(self, prompt_ids, continuation_ids, top_k, prompt_rows=True):
        """Fixed text as crit serve scores it: every prompt token after the first by query scoring on the engine's
        "prefill" score path (its default path when it names none), every continuation token from a greedy replay
        forced to it with log-probabilities, greedy with neutral penalties as Eval's requests are. Rows {"logprob", "top": [[token, logprob], ...top_k]} (crit-eval's
        ADAPTER.md)."""
        ids, continuation, rows = list(prompt_ids), list(continuation_ids), {"prompt": [], "continuation": []}
        if prompt_rows and len(ids) > 1:
            paths = self.facts.get("score_paths") or []
            scored = self.native.score(ids, 1, "prefill" if "prefill" in paths else "", top_k=top_k)
            rows["prompt"] = [{"logprob": float(logprob), "top": [[int(t), float(v)] for t, v in top or []]}
                              for logprob, top in zip(scored.target, scored.top or [[] for _ in scored.target])]
        if continuation:
            found = []
            tokens, _ = self.native.generate(ids, len(continuation), stop_tokens=[], replay=continuation,
                                             sampling=self.sampling or None, logprobs=top_k,
                                             on_tokens=lambda _tokens, values: found.extend(values))
            if list(tokens) != continuation:
                raise NativeError("replay emitted other tokens than the continuation")
            rows["continuation"] = [{"logprob": float(logprob), "top": [[int(t), float(v)] for t, v in top]}
                                    for logprob, top in found]
        return rows

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
    token_logprobs with max_top_logprobs of at least 20. It scores fixed text (chat-score-v1) when the engine also
    declares query_scoring and replay: the prompt by query scoring with the engine's own top-k on its "prefill" score
    path (the default path when it names none), the continuation by replay with log-probabilities. A speculative
    engine that declares plain decodes plainly on request (chat-plain-v1), and its info is the resident engine's
    `facts` (crit_core.serving.engine_facts). Configuration "chat_template_kwargs" passes variables to the chat
    template."""

    def __init__(self, context, library, defaults=None):
        configuration = {**(defaults or {}), **(context.configuration or {})}
        model = Path(context.model_file("config.json")).parent
        super().__init__(context, model, configuration=configuration)
        limits = getattr(context, "limits", {}) or {}
        native = dict(self.configuration, model=str(model))
        if limits.get("kv_tokens"):
            native["context_tokens"] = int(limits["kv_tokens"])
        self.native = open_engine(library, native)
        self.facts = self.native.info()
        self.max_concurrent_requests = int(self.facts.get("max_concurrent_requests", 1))
        self._owns_native = True
        self.capacity = min(int(self.facts["context"]), int(limits.get("kv_tokens") or self.facts["context"]))
        if "stop_tokens" in self.facts:   # a native engine always reports them; a Python engine may keep the model's
            self.stop_ids = set(self.facts["stop_tokens"] or [])
        support = self.facts.get("sampling") or {}
        self.sampling = tuple(key for key in SAMPLING if support.get(PENALTY if key in PENALTIES else key))
        self.logprobs = bool(self.facts.get("token_logprobs")
                             and int(self.facts.get("max_top_logprobs", 0)) >= TOP_LOGPROBS)
        self.scores = bool(self.logprobs and self.facts.get("query_scoring") and self.facts.get("replay"))
        self.plain = bool(self.facts.get("speculative") and self.facts.get("plain"))

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

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, logprobs=None, plain=False):
        tokens, result = self.native.generate(ids, max_new_tokens, on_tokens=on_tokens, sampling=sampling,
                                              logprobs=logprobs, plain=plain, cancelled=cancelled)
        if tokens is None:
            return None, {}
        timing = result["timing"]
        out = {"prefill_tokens": timing["prefill_tokens"] - timing["reused_prompt_tokens"],
               "prefill_ms": timing["prefill_ms"], "decode_ms": timing["decode_ms"],
               "first_token_ms": timing["first_token_ms"],
               "rounds": decode_rounds(tokens, result, bool(self.facts.get("speculative")) and not plain)}
        if not plain:
            # The engine's request facts describe drafting, which plain decoding does none of.
            out.update({key: result[key] for key in ("accepted", "drafted", "verify_ms", "draft_ms") if key in result})
        return tokens, out

    def score_tokens(self, ids, continuation, top_k, cancelled, prompt_rows):
        started = time.monotonic()
        paths = self.facts.get("score_paths") or []
        # The engine's own arrays as ScoreRows (top_k > 0), which ResidentChat.score packs without a row loop.
        scored = (self.native.score(ids, 1, "prefill" if "prefill" in paths else "", top_k=top_k, packed=True)
                  if prompt_rows and len(ids) > 1 else None)
        prefill_ms = (time.monotonic() - started) * 1000
        rows = []
        if continuation:
            # Greedy, with neutral penalties: a sampled replay would accept drafts by the sampling rule instead of by
            # equality, and the engine's default penalties would move the scores.
            greedy = {**({"temperature": 0.0} if "temperature" in self.sampling else {}),
                      **{key: value for key, value in NEUTRAL_PENALTIES.items() if key in self.sampling}} or None
            tokens, result = self.native.generate(list(ids), len(continuation), stop_tokens=[], replay=list(continuation),
                                                  sampling=greedy, logprobs=top_k,
                                                  on_tokens=lambda _tokens, found: rows.extend(found),
                                                  cancelled=cancelled)
            if tokens is None:
                return None, None, {}
            if list(tokens) != list(continuation):
                raise NativeError("replay emitted other tokens than the continuation")
            decode_ms = result["timing"]["decode_ms"]
            # The replay prefills the prompt again before its first row: engine time like the score's prefill.
            prefill_ms += result["timing"].get("prefill_ms") or 0.0
        else:
            decode_ms = None
        if scored is None:
            found = []
        elif isinstance(scored.top, ScoreRows):
            found = scored.top
        else:
            found = list(zip(scored.target, scored.top if top_k else [[] for _ in scored.target]))
        return found, rows, {"prefill_ms": prefill_ms, "decode_ms": decode_ms}

    def close(self):
        if self._owns_native:
            self.native.close()


def eval_engine(context, library, defaults=None):
    return EvalEngine(context, library, defaults)


def resident_engine(context, library, defaults=None):
    return ResidentEngine(context, library, defaults)


__all__ = ["NativeEngine", "NativeError", "PythonEngine", "StepEngine", "StepRequest", "StepRow", "StepResult",
           "StepDecision", "EvalEngine", "ResidentEngine", "eval_engine", "open_engine", "resident_engine"]
