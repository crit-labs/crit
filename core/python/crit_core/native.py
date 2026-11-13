"""Core's concurrent runtime from Python: the ctypes side of crit/step_engine.h, through the crit_core_steps library.

StepEngine runs a step model, the object of an engine's model plugin (crit_plugins.slots "model"): Core schedules its
requests in mixed prefill/decode steps, admits them, applies stop tokens and output limits, cancels and retires them,
and runs scores as exclusive barriers; the model computes one row per scheduled request in step(rows). Core's loader
(crit_core.engine.load) is the only thing that opens one: an engine is Core plus a plugin selection, and there is no
other way to hand Core an engine.
"""
import array
import ctypes
import gc
import json
import os
import struct
import sys
import threading
import time
from collections import OrderedDict, deque, namedtuple
from contextlib import contextmanager, nullcontext
from pathlib import Path

from crit_core import jsonfile
from crit_core.machine import under_wsl
from crit_core.serving import SAMPLING, ScoreRows
from crit_plugins.device import Allowed

ABI = 5
RESULT_BYTES = 1 << 20  # a generation's result JSON: finish, timing and the engine's request facts
TOKEN_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int)
OUTPUT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_float,
                                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int)
# StepEngine.score per target: its log-probability, the query rows, the argmax and the engine's own ranked top_k
# [(token, logprob), ...] (None for parts not asked for or not returned).
Scores = namedtuple("Scores", "target query argmax top")


# Whether CUDA runs under the Windows display driver model: native Windows, or WSL's paravirtualized GPU.
WDDM = sys.platform == "win32" or under_wsl()


class NativeError(RuntimeError):
    pass


class _Requests:
    """Generations and scores in flight on one engine, max_concurrent_requests of them at once (Core's runtime
    schedules them); close() waits for them."""

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


class _Runtime(_Requests):
    """The requests of Core's engine ABI (crit/native_engine.h) on an engine crit_core_steps opened: generations,
    scores, info and close. StepEngine is the one engine it serves."""

    def _load(self, library):
        self.path = Path(library).resolve()
        lib = ctypes.CDLL(str(self.path))
        lib.crit_engine_abi.restype = ctypes.c_int
        if lib.crit_engine_abi() != ABI:
            raise NativeError(f"{self.path} exports engine ABI {lib.crit_engine_abi()}, this crit_core reads {ABI}")
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
        """ModelInfo fields (implementation, context, stop_tokens, score_paths, ...) plus the model's facts."""
        return dict(self.facts)

    def _usable(self):
        return bool(self.handle)

    def generate(self, prompt, output_tokens, **options):
        with self._lease():
            try:
                return self._generate(prompt, output_tokens, **options)
            finally:
                self._check_device_memory()

    def _generate(self, prompt, output_tokens, *, stop_tokens=None, sampling=None, replay=None, plain=False,
                  logprobs=None, on_tokens=None, cancelled=None, constraint=None, inputs=None):
        """Returns (tokens, result) with result's "finish", "timing" (ms, decode_steps) and the engine's request
        facts, or (None, {"cancelled": True}) when cancelled() turned true. stop_tokens None keeps the model's;
        [] disables them. sampling: temperature, top_p, top_k, min_p, seed, penalties and logit_bias.
        plain: decode without drafting (engines that declare "plain"). logprobs n: on_tokens(tokens, rows) gets one
        (logprob, [(token, logprob), ...n alternatives]) row per token (engines whose info reports token_logprobs
        and max_top_logprobs >= n). constraint: the language every token keeps to (StepEngine.CONSTRAINTS); a
        constraint that cannot continue, or a token outside it, fails the request with NativeError. inputs: what the
        model's prepare() made of the prompt's images, which its open() gets as StepRequest.inputs."""
        if cancelled and cancelled():
            return None, {"cancelled": True}
        if constraint is None and inputs is None:
            return self._run(prompt, output_tokens, stop_tokens, sampling, replay, plain, logprobs, on_tokens,
                             cancelled, None)
        if replay is not None:
            raise NativeError("a replay is its own continuation: it takes no constraint or inputs")
        holder = _Attached(None if constraint is None else self._matcher(constraint), inputs)
        stopped = cancelled
        try:
            tokens, result = self._run(prompt, output_tokens, stop_tokens, sampling, replay, plain, logprobs, on_tokens,
                                       lambda: holder.failure is not None or bool(stopped and stopped()), holder)
        finally:
            holder.close()
        if holder.failure is not None:
            raise NativeError("the constrained request failed: " + holder.failure)
        return tokens, result

    def _run(self, prompt, output_tokens, stop_tokens, sampling, replay, plain, logprobs, on_tokens, cancelled,
             holder):
        options = dict(sampling or {})
        if stop_tokens is not None:
            options["stop_tokens"] = [int(t) for t in stop_tokens]
        if replay is not None:
            options["replay"] = [int(t) for t in replay]
        if plain:
            options["plain"] = True
        if logprobs is not None:
            options["logprobs"] = int(logprobs)
        if holder is not None:
            options["attachment"] = self._hold(holder)
            if holder.matcher is not None:
                options["constrained"] = True
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

        # Without a listener (no on_tokens, nothing to cancel) Core delivers no token to Python at all, so its waiter
        # threads take no turn at the interpreter while the engine's worker runs the steps.
        if on_tokens is None and cancelled is None:
            callback = TOKEN_CALLBACK() if logprobs is None else OUTPUT_CALLBACK()
        else:
            callback = TOKEN_CALLBACK(deliver) if logprobs is None else OUTPUT_CALLBACK(deliver_output)
        ids = (ctypes.c_int32 * len(prompt))(*prompt)
        out = (ctypes.c_int32 * max(1, output_tokens))()
        result = ctypes.create_string_buffer(RESULT_BYTES)
        encoded = json.dumps(options).encode("utf-8")
        request = self.lib.crit_engine_request_open(self.handle, ids, len(prompt), int(output_tokens), encoded)
        if not request:
            if holder is not None:
                self._release_hold(options["attachment"])
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
            if holder is not None:
                self._release_hold(options["attachment"])
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
            self._check_device_memory()
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


class _Matchers:
    """Core's structured-output matchers through crit_core_steps (crit/native_engine.h, crit/constraints.h): the
    model's byte vocabulary, its languages (json mode, compiled grammars; the LANGUAGES most recent kept, each with the
    allowed sets its matchers share) and one matcher per constrained request."""

    LANGUAGES = 16

    def __init__(self, lib):
        for name, restype, argtypes in (
                ("crit_vocabulary_open", ctypes.c_void_p, [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64),
                                                           ctypes.c_uint32, ctypes.c_int32]),
                ("crit_vocabulary_close", None, [ctypes.c_void_p]),
                ("crit_language_open", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                                         ctypes.c_uint64]),
                ("crit_language_close", None, [ctypes.c_void_p]),
                ("crit_matcher_open", ctypes.c_void_p, [ctypes.c_void_p]),
                ("crit_matcher_allowed", ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                                        ctypes.c_uint32]),
                ("crit_matcher_append", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int32]),
                ("crit_matcher_close", None, [ctypes.c_void_p]),
                ("crit_constraint_last_error", ctypes.c_char_p, [])):
            function = getattr(lib, name)
            function.restype, function.argtypes = restype, argtypes
        self.lib, self.lock = lib, threading.Lock()
        self.vocabulary, self.identity, self.size = None, None, 0
        self.languages = OrderedDict()

    def error(self):
        return (self.lib.crit_constraint_last_error() or b"").decode("utf-8", "replace")

    def use(self, vocabulary):
        """The model's crit_core.constraints.ByteVocabulary (once; the same identity again is kept)."""
        with self.lock:
            if self.vocabulary is not None and self.identity == vocabulary.identity:
                return
            pieces = list(vocabulary.pieces)
            ends, total = (ctypes.c_uint64 * len(pieces))(), 0
            for i, piece in enumerate(pieces):
                total += len(piece)
                ends[i] = total
            handle = self.lib.crit_vocabulary_open(b"".join(pieces), ends, len(pieces), int(vocabulary.eos))
            if not handle:
                raise NativeError("the byte vocabulary was refused: " + self.error())
            self._close_vocabulary()
            self.vocabulary, self.identity, self.size = handle, vocabulary.identity, len(pieces)

    def matcher(self, kind, artifact=b""):
        with self.lock:
            if self.vocabulary is None:
                raise NativeError("a constrained request needs the model's byte vocabulary (StepEngine.constrain)")
            key = (kind, artifact)
            language = self.languages.pop(key, None)
            if language is None:
                language = self.lib.crit_language_open(self.vocabulary, kind.encode(), artifact, len(artifact))
                if not language:
                    raise NativeError(f"the {kind} constraint was refused: " + self.error())
                while len(self.languages) >= self.LANGUAGES:
                    self.lib.crit_language_close(self.languages.popitem(last=False)[1])
            self.languages[key] = language   # most recent last
            handle = self.lib.crit_matcher_open(language)
            if not handle:
                raise NativeError(f"the {kind} constraint has no matcher: " + self.error())
            return _NativeMatcher(self, handle, self.size)

    def _close_vocabulary(self):
        for language in self.languages.values():
            self.lib.crit_language_close(language)
        self.languages.clear()
        if self.vocabulary is not None:
            self.lib.crit_vocabulary_close(self.vocabulary)
            self.vocabulary = None

    def close(self):
        with self.lock:
            self._close_vocabulary()


def _fit(words, vocabulary):
    """An allowed bitmask as an Allowed over `vocabulary` tokens: padded, or cut with the tokens past it dropped."""
    vocabulary = int(vocabulary)
    count = (vocabulary + 31) // 32
    if len(words) < count:
        words.extend([0] * (count - len(words)))
    elif len(words) > count:
        del words[count:]
    if vocabulary % 32:
        words[-1] &= (1 << (vocabulary % 32)) - 1
    if not any(words):
        raise NativeError(f"no token of the model's {vocabulary} continues the constraint")
    return Allowed(words, vocabulary)


class _NativeMatcher:
    """One request's TokenMatcher (Core's json or grammar matcher), private to it; its language may close first."""

    def __init__(self, matchers, handle, size):
        self.matchers, self.handle, self.size = matchers, handle, size
        self.words = (ctypes.c_uint32 * ((size + 31) // 32))()

    def allowed(self, vocabulary):
        """The allowed set over the model's `vocabulary` (its logits' tokens): tokens of the byte vocabulary past it
        are never produced, and a set of only such tokens cannot continue."""
        if self.matchers.lib.crit_matcher_allowed(self.handle, self.words, len(self.words)) != 0:
            raise NativeError(self.matchers.error())
        return _fit(array.array("I", bytes(self.words)), vocabulary)

    def append(self, token):
        if not 0 <= token < self.size or self.matchers.lib.crit_matcher_append(self.handle, int(token)) != 0:
            raise NativeError(f"token {token} does not continue the constraint: " + self.matchers.error())

    def close(self):
        if self.handle:
            self.matchers.lib.crit_matcher_close(self.handle)
            self.handle = None


class _TokenSet:
    """A constraint that takes every token from one fixed set ({"type": "tokens"}): Core's loader checks with it that a
    model plugin hands each row's allowed set to its sampler."""

    def __init__(self, tokens):
        self.tokens = sorted({int(t) for t in tokens})
        if not self.tokens or self.tokens[0] < 0:
            raise NativeError("a token-set constraint holds one or more token ids")

    def allowed(self, vocabulary):
        return _fit(Allowed.of(self.tokens, max(int(vocabulary), self.tokens[-1] + 1)).words, vocabulary)

    def append(self, token):
        if token not in self.tokens:
            raise NativeError(f"token {token} is not in the constraint's set")

    def close(self):
        pass


class _Attached:
    """A request's state on the host side of Core's runtime (its HostAttachment): a constraint's matcher (None when
    unconstrained), the token its last row produced (appended once Core keeps it), the mask that row had and why it
    failed (None while it keeps to the constraint); and the inputs the model prepared from its prompt's images."""

    __slots__ = ("matcher", "inputs", "pending", "allowed", "failure")

    def __init__(self, matcher, inputs=None):
        self.matcher, self.inputs, self.pending, self.allowed, self.failure = matcher, inputs, None, None, None

    def close(self):
        if self.matcher is not None:
            self.matcher.close()


# Where Core's crit_core_steps library (crit/step_engine.h) is, when the loader is given none.
STEP_LIBRARY = "CRIT_CORE_STEP_LIBRARY"
# "1" makes every StepEngine probe its steps (StepEngine.telemetry's "probe"): crit run serve --check sets it for its
# engine. PROBE_RECENT: the probed steps per kind whose samples the telemetry keeps.
STEP_PROBE = "CRIT_CORE_STEP_PROBE"
PROBE_RECENT = 8192
# The library's file on each OS, and the folders of a crit checkout it is looked for in when STEP_LIBRARY names none:
# where `crit core build` builds it (build/core-steps), and Core's build folder of core/README.md (build/core), each
# with a multi-configuration generator's Release folder.
STEP_LIBRARY_FILES = ("libcrit_core_steps.so", "crit_core_steps.dll", "libcrit_core_steps.dylib")
STEP_LIBRARY_FOLDERS = ("build/core-steps", "build/core-steps/Release", "build/core", "build/core/Release")


def step_cache():
    """Where `crit core build` builds this Core's library outside a checkout: CRIT_CORE_CACHE (default
    ~/.cache/crit/core), one folder a Core version."""
    from . import __version__
    return Path(os.environ.get("CRIT_CORE_CACHE") or Path.home() / ".cache" / "crit" / "core") / __version__


def step_library():
    """The crit_core_steps library to load: the file STEP_LIBRARY names, else the one built in the crit checkout this
    Core runs from (STEP_LIBRARY_FOLDERS), else the one `crit core build` built for this Core's version (step_cache),
    else None."""
    named = os.environ.get(STEP_LIBRARY)
    if named:
        return named
    root = Path(__file__).resolve().parents[3]
    folders = [root / f for f in STEP_LIBRARY_FOLDERS] if (root / "core" / "include" / "crit" / "core.h").is_file() else []
    cache = step_cache() / "build"
    for folder in folders + [cache, cache / "Release"]:
        for name in STEP_LIBRARY_FILES:
            if (folder / name).is_file():
                return str(folder / name)
    return None
# The settings of a step model's info that configure Core (crit/step_engine.h's "engine" object), from the table Core's
# step engine reads too (step_settings.json); the rest are facts.
STEP_SETTINGS = tuple(jsonfile.read(Path(__file__).with_name("step_settings.json"))["settings"])
FINISHES = ("length", "stop", "cancelled", "failed")


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
                ("drafted", ctypes.c_int32)]


# The rows, progress and decisions of a step as packed bytes (their structures' layouts), which _step and _commit read
# and write in one call each instead of a ctypes field access per value.
_ROW = struct.Struct("=QIIIi")
_PROGRESS = struct.Struct("=IIIIii")
_DECISION = struct.Struct("=QIiii")


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
_FACTS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)


class _StepCallbacks(ctypes.Structure):
    _fields_ = [("user", ctypes.c_void_p), ("admission", _ADMISSION), ("open", _OPEN), ("step", _STEP),
                ("commit", _COMMIT), ("retire", _RETIRE), ("release", _RELEASE), ("score", _SCORE), ("error", _ERROR),
                ("facts", _FACTS)]


_LAUNCH = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
_WAIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class _StepRounds(ctypes.Structure):
    _fields_ = [("user", ctypes.c_void_p), ("launch", _LAUNCH), ("wait", _WAIT), ("block", ctypes.c_void_p),
                ("row_stride", ctypes.c_uint32), ("entry_bytes", ctypes.c_uint32), ("max_rows", ctypes.c_uint32),
                ("telemetry", ctypes.c_uint32), ("counters", ctypes.c_uint32), ("widest", ctypes.c_uint32),
                ("widths", ctypes.POINTER(ctypes.c_int32)), ("slot_session", ctypes.POINTER(ctypes.c_uint64)),
                ("last", ctypes.POINTER(ctypes.c_uint32)), ("runs", ctypes.POINTER(ctypes.c_uint64)),
                ("logprobs", ctypes.c_void_p), ("top_ids", ctypes.c_void_p), ("top_logprobs", ctypes.c_void_p),
                ("top", ctypes.c_uint32)]


class NativeRounds:
    """Native decode rounds for a StepEngine (crit/step_engine.h's StepRounds): Core runs every decode step whose rows
    all sit in these slots without calling the model's step(), so nothing in Python runs while its requests decode.
    The model keeps its decoding rows packed in slots 0 .. rows - 1 and says which session each slot holds with bind()
    and unbind(); a round of r rows runs at widths[r - 1] drafts, `widths` the address of the engine's own int32 array
    of max_rows entries (-1: r rows go to step()), which Core reads before each round: the engine may rewrite it between
    steps (a draft-width policy; a captured round's own copy of the next widths into it), each entry within `widest`,
    the widest round it runs, and a row count at -1 when Core opens stays at step(). After a native round Core calls the
    model's commit() only when it trims, ends, cancels or discards a row; `last` then names the round (rows, width,
    native). `launch(user, rows, width, logprobs)` runs a round in one of its two forms: logprobs 1 the
    log-probability form (it writes the log-probability buffers and runs rows that follow a continuation of their own),
    which Core launches when a row's request asks for log-probabilities or replays a continuation, 0 the plain form.

    cuda(): over captured CUDA graphs (graphs {(rows, width): CUgraphExec} of the plain rounds, logprob_graphs the same
    of the log-probability rounds (None: the rounds have one form, and graphs serve both), as torch.cuda.CUDAGraph's
    raw_cuda_graph_exec() reports them, launched on `stream`, a CUstream such as torch.cuda.Stream's cuda_stream) whose
    last node copies the result block to host memory: `block` its address (pinned), rows of `row_stride` entries of
    `entry_bytes`: tokens emitted, drafts matched, frontier, then the tokens, and from entry `telemetry` (0: none; past
    the widest round's tokens) the row's words: its drafts accepted, then `counters` counters of the engine's own
    (named by its info's round_counters), which Core adds to the request's facts. `runs` counts the native rounds Core
    ran, their rows and the rounds of them in the log-probability form. `logprobs` (an address, 0: none), with
    `top` > 0 also `top_ids` and `top_logprobs`: the log-probability rounds' pinned log-probabilities of their emitted
    tokens ([max_rows][widest + 1] float32) and their `top` ranked alternatives ([max_rows][widest + 1][top] int64 ids
    and float32 values), which Core gives the requests that ask (a request then decodes in native rounds too)."""

    def __init__(self, launch, wait, user, block, row_stride, entry_bytes, max_rows, widths, widest, keep=(),
                 telemetry=0, counters=0, logprobs=0, top_ids=0, top_logprobs=0, top=0):
        if entry_bytes not in (4, 8) or max_rows < 1 or not widths or widest < 0:
            raise ValueError("native rounds: entries of 4 or 8 bytes, the widths' array and the widest round")
        if top and not (logprobs and top_ids and top_logprobs):
            raise ValueError("native rounds that rank alternatives give their log-probabilities, ids and values")
        if telemetry and (telemetry < 3 + widest + 1 or telemetry + 1 + counters > row_stride):
            raise ValueError("native rounds: the words lie past the widest round's tokens and inside the block's rows")
        self._widths = ctypes.cast(int(widths), ctypes.POINTER(ctypes.c_int32))
        self._slots = (ctypes.c_uint64 * max_rows)()
        self._last = (ctypes.c_uint32 * 3)()
        self._runs = (ctypes.c_uint64 * 3)()
        self._launch, self._wait, self._keep = launch, wait, keep
        self.struct = _StepRounds(user, launch, wait, block, row_stride, entry_bytes, max_rows, int(telemetry),
                                  int(counters), int(widest), self._widths, self._slots, self._last, self._runs,
                                  int(logprobs or 0), int(top_ids or 0), int(top_logprobs or 0), int(top))
        self.max_rows, self._handles, self._close = max_rows, {}, None
        # The slots the captures of these rounds called (the round driver's): Core's loader counts a request's native
        # rounds as calls of them, as it counts a Python replay.
        self.captured_slots = ()

    @property
    def runs(self):
        """(native rounds Core ran, the rows they ran, the rounds of them in the log-probability form)."""
        return int(self._runs[0]), int(self._runs[1]), int(self._runs[2])

    @classmethod
    def cuda(cls, library, stream, graphs, block, row_stride, entry_bytes, max_rows, max_width, widths, widest,
             telemetry=0, counters=0, logprobs=0, top_ids=0, top_logprobs=0, top=0, logprob_graphs=None):
        lib = ctypes.CDLL(str(Path(library).resolve()))
        lib.crit_rounds_cuda_open.restype = ctypes.c_void_p
        lib.crit_rounds_cuda_open.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                                              ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_uint32]
        lib.crit_rounds_cuda_close.argtypes = [ctypes.c_void_p]
        lib.crit_rounds_cuda_error.argtypes = [ctypes.c_char_p, ctypes.c_int]

        def table(found):
            out = (ctypes.c_uint64 * (max_rows * (max_width + 1)))()
            for (rows, width), handle in found.items():
                if not (1 <= rows <= max_rows and 0 <= width <= max_width):
                    raise ValueError(f"a graph for {rows} rows at width {width} is outside the table")
                out[(rows - 1) * (max_width + 1) + width] = int(handle)
            return out
        tables = (table(graphs), table(graphs if logprob_graphs is None else logprob_graphs))
        user = lib.crit_rounds_cuda_open(int(stream), tables[0], tables[1], max_rows, max_width)
        if not user:
            message = ctypes.create_string_buffer(512)
            lib.crit_rounds_cuda_error(message, 512)
            raise NativeError("native rounds: " + message.value.decode("utf-8", "replace"))
        launch = _LAUNCH(ctypes.cast(lib.crit_rounds_cuda_launch, ctypes.c_void_p).value)
        wait = _WAIT(ctypes.cast(lib.crit_rounds_cuda_wait, ctypes.c_void_p).value)
        rounds = cls(launch, wait, user, block, row_stride, entry_bytes, max_rows, widths, widest, keep=(lib, tables),
                     telemetry=telemetry, counters=counters, logprobs=logprobs, top_ids=top_ids,
                     top_logprobs=top_logprobs, top=top)
        rounds._close = lambda: lib.crit_rounds_cuda_close(user)
        return rounds

    def bind(self, slot, session):
        """Slot `slot` holds `session` (an object open() returned) for native rounds."""
        self._slots[slot] = self._handles[id(session)]

    def unbind(self, slot):
        self._slots[slot] = 0

    @property
    def last(self):
        """(rows, width, native) of the last step's round."""
        return self._last[0], self._last[1], bool(self._last[2])

    def close(self):
        if self._close is not None:
            self._close()
            self._close = None


# What a step model's open() and admission() receive: the request's key (request, generation, lane), prompt and output
# limit; the sampling fields it set; the effective stop tokens; alternatives per token (None: no log-probabilities); a
# forced continuation (replay) or plain decoding (crit/native_engine.h's ReplayContinuation and PlainDecoding, which
# every constrained request has); and the inputs the model's prepare() made of the prompt's images (None: none).
StepRequest = namedtuple("StepRequest", "key prompt output_tokens sampling stop_tokens logprobs replay plain inputs",
                         defaults=(None,))
# One scheduled row of a step: the session open() returned, its WorkBudget (crit/core.h) and, for a constrained
# request, the tokens its next token is taken from (crit_plugins.Allowed; None: any), which the model hands its sampler
# as the row's params["allowed"].
StepRow = namedtuple("StepRow", "session prefill_tokens output_tokens token_work prefill_complete allowed",
                     defaults=(None,))
# What step() returns per row, in the rows' order: prompt tokens consumed (reused ones included), the tokens produced
# (at most max_row_tokens), whether prefill is complete, the reused prefix, the token work computed (0: the prompt
# tokens computed, or one per token), for requests asking for them one (logprob, [(token, logprob), ...]) per token,
# and for a decode row the drafts its round verified (None: not reported), which the request's facts list per round.
StepResult = namedtuple("StepResult",
                        "prompt_tokens tokens prefill_complete reused_prompt_tokens token_work logprobs drafted",
                        defaults=(0, (), False, 0, 0, None, None))


# Facts Core keeps per round: the model's own list of them covers its step() rounds.
ROUND_LISTS = ("accepted", "drafted")


def merge_facts(model, core, counters=()):
    """A request's facts: Core's of its decode rounds with the model plugin's request_facts(session) merged round by
    round. Core counts every decode row as one round (rounds) and lists each round's accepted drafts and drafts
    (accepted, drafted; `_native` in `core` marks the rounds Core ran natively). A model's accepted or drafted list
    holds one entry per round its step() ran, in order, which take the step() rounds' places in Core's list (a list of
    another length is set aside, with a warning, and Core's kept). The engine's round counters (`counters`, its
    round_counters, which native rounds sum) add the model's counts of its step() rounds; the model's other facts
    (verify_ms, ...) stand as given, and a result's own keys (finish, timing) are never a fact."""
    merged = dict(core or {})
    native = merged.pop("_native", None)
    step = [i for i, kind in enumerate(native or ()) if not kind]
    for key, value in dict(model or {}).items():
        if key in ("finish", "timing", "rounds", "_native"):
            continue
        if key in ROUND_LISTS:
            if not isinstance(value, list) or native is None or len(value) != len(step):
                sys.stderr.write(f"crit_core: request_facts' {key} has {len(value) if isinstance(value, list) else 'no'} "
                                 f"entries for the {len(step)} rounds the model ran in step(): Core's are kept\n")
                continue
            entries = list(merged.get(key) or [0 if key == "accepted" else -1] * len(native))
            for index, entry in zip(step, value):
                entries[index] = entry
            merged[key] = entries
        elif key in counters and type(value) is int:
            merged[key] = merged.get(key, 0) + value if type(merged.get(key, 0)) is int else value
        else:
            merged[key] = value
    if isinstance(merged.get("drafted"), list) and any(type(d) is not int or d < 0 for d in merged["drafted"]):
        del merged["drafted"]   # a round's drafts unknown: no list (Core's rule)
    return merged


# Core's decision for a row of the last step: the tokens it kept, whether the request ends, was cancelled, or the row is
# discarded.
StepDecision = namedtuple("StepDecision", "session accepted terminal cancelled discard")


def _keep_thread_state(ident, threads):
    """Keep the Python thread state of the calling thread, one of Core's: a callback from a thread Python did not
    start makes a thread state on entry and frees it on return, which costs more than most steps' Python work; one
    more hold on it (PyGILState_Ensure, never released) keeps it for the thread's life, about a kilobyte left behind
    when the engine's worker ends."""
    ctypes.pythonapi.PyGILState_Ensure.restype = ctypes.c_int
    ctypes.pythonapi.PyGILState_Ensure.argtypes = []
    ctypes.pythonapi.PyGILState_Ensure()
    threads.add(ident)


class StepEngine(_Runtime):
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

    on Core's one engine worker, never two at once. A constrained request (generate's constraint: structured output,
    forced calls) decodes plainly (StepRequest.plain), and each of its rows carries the tokens its constraint allows
    next (StepRow.allowed), which the model passes to its sampler; Core checks each token such a row produces against
    them, and a row that produces another token or more than one fails the request. CONSTRAINTS are the kinds;
    constrain(vocabulary) gives Core the model's byte vocabulary that json and grammar constraints are matched over. It generates and scores max_concurrent_requests
    (max_concurrency + max_pending) requests at once. `library` is crit_core_steps, else step_library()'s.
    facts: what the engine reports beside the model's info (the loader's plugins); close: what closes the model's
    parts, the assembly (else the model's own close()); telemetry: what reports the engine's work so far (the
    assembly's telemetry(), else the model's own). A model with native_rounds(library) returning NativeRounds decodes
    without calling step() (NativeRounds).

    Each generation's result carries its request's facts (crit_core.serving.REQUEST_FACTS, rounds): Core's of its
    decode rounds (every decode row a round: accepted drafts, drafted, and a native round's telemetry words), merged
    with the model's optional request_facts(session) (merge_facts), which Core calls as the session retires."""

    # What generate's constraint may be: one JSON object ({"type": "json_object"}), a compiled crit-byte-grammar-v1
    # artifact ({"grammar": ...}, as response_format's json_schema and grammar types carry it), or a fixed token set
    # ({"type": "tokens", "tokens": [...]}).
    CONSTRAINTS = ("json_object", "grammar", "tokens")

    def __init__(self, model, library=None, facts=None, close=None, telemetry=None):
        self.model, self._close_parts = model, close
        own = getattr(model, "telemetry", None)
        self._telemetry = telemetry if telemetry is not None else (own if callable(own) else None)
        self._request_facts = getattr(model, "request_facts", None)
        self._request_facts = self._request_facts if callable(self._request_facts) else None
        self._counted = {"steps": 0, "rows": 0, "step_s": 0.0}
        self._probed = {} if os.environ.get(STEP_PROBE) == "1" else None   # StepEngine.probe's samples
        self._threads = set()   # Core's threads that keep their Python thread state (_keep_thread_state)
        info = dict(model.info())
        library = library or step_library()
        if not library:
            raise NativeError("Core's runtime is its crit_core_steps library, which is not built: run `crit core build`, "
                              f"or build Core (core/README.md) and set {STEP_LIBRARY} to the "
                              "library")
        self._load(library)
        self.library, self.path = self.path, None   # Eval takes it from the facts' native_libraries
        self.lib.crit_engine_open_steps.restype = ctypes.c_void_p
        self.lib.crit_engine_open_steps.argtypes = [ctypes.c_char_p, ctypes.POINTER(_StepCallbacks)]
        settings = {key: info[key] for key in STEP_SETTINGS if key in info}
        self._counter_names = tuple(settings.get("round_counters") or ())
        self._pending_facts = {}   # a session's encoded facts while Core sizes its buffer for them
        facts = {**{key: value for key, value in info.items() if key not in STEP_SETTINGS}, **(facts or {})}
        facts["native_libraries"] = [str(self.library)] + list(facts.get("native_libraries") or [])
        self._oversubscribed = set()
        self._row_tokens = int(settings.get("max_row_tokens", 1))
        self._top = int(settings.get("max_top_logprobs", 0))
        self._logprobs = bool(settings.get("token_logprobs"))
        self._sessions, self._handles, self._next, self._failure = {}, {}, 1, ""
        self._vocabulary = int(settings.get("vocabulary") or 0)
        self._attachments, self._constrained, self._attaching = {}, {}, threading.Lock()
        self._next_attachment, self._matchers = 1, None
        self._model_commits = callable(getattr(model, "commit", None))
        optional = lambda name, kind, method: kind(method) if callable(getattr(model, name, None)) else kind()
        self._callbacks = _StepCallbacks(
            None, optional("admission", _ADMISSION, self._admission), _OPEN(self._open), _STEP(self._step),
            _COMMIT(self._commit), _RETIRE(self._retire), _RELEASE(self._release),
            _SCORE(self._score) if settings.get("scoring") else _SCORE(), _ERROR(self._error_message),
            _FACTS(self._facts) if self._request_facts is not None else _FACTS())
        configuration = json.dumps({"engine": settings, "facts": facts}).encode("utf-8")
        make = getattr(model, "native_rounds", None)
        self.rounds = make(self.library) if callable(make) else None
        if self.rounds is not None:
            self.rounds._handles = self._handles
            self.lib.crit_engine_open_steps_rounds.restype = ctypes.c_void_p
            self.lib.crit_engine_open_steps_rounds.argtypes = [ctypes.c_char_p, ctypes.POINTER(_StepCallbacks),
                                                               ctypes.POINTER(_StepRounds)]
            handle = self.lib.crit_engine_open_steps_rounds(configuration, ctypes.byref(self._callbacks),
                                                            ctypes.byref(self.rounds.struct))
        else:
            handle = self.lib.crit_engine_open_steps(configuration, ctypes.byref(self._callbacks))
        if not handle:
            raise NativeError("opening the step engine failed: " + self._error())
        self._attach(handle)

    def _guard(self, run):
        """Runs a callback body; an exception is kept for error() and fails the engine (never unwinds through C)."""
        ident = threading.get_ident()
        if ident not in self._threads:
            _keep_thread_state(ident, self._threads)
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
    def _options(pointer):
        r = pointer.contents
        return json.loads(r.options_json.decode("utf-8")) if r.options_json else {}

    def _request(self, pointer, options=None, holder=None):
        r = pointer.contents
        options = self._options(pointer) if options is None else options
        if holder is None and options.get("attachment"):
            holder = self._held(options["attachment"])
        return StepRequest((r.request, r.generation, r.lane), r.prompt[:r.prompt_tokens], r.output_tokens,
                           {key: options[key] for key in SAMPLING if key in options},
                           options.get("stop_tokens", []), options.get("logprobs"), options.get("replay"),
                           bool(options.get("plain")), None if holder is None else holder.inputs)

    def _held(self, attachment):
        with self._attaching:
            holder = self._attachments.get(attachment)
        if holder is None:
            raise ValueError("a request without its host attachment")
        return holder

    def prepare(self, ids, images, cancelled=None):
        """The model's prepare() of a prompt's images (crit_plugins.IMAGES), on the caller's thread: (ids, inputs)."""
        prepared = self.model.prepare(list(ids), list(images), cancelled or (lambda: False))
        found, inputs = prepared
        return [int(t) for t in found], inputs

    def constrain(self, vocabulary):
        """The model's byte vocabulary (crit_core.constraints.ByteVocabulary) for json and grammar constraints."""
        if self._matchers is None:
            self._matchers = _Matchers(self.lib)
        self._matchers.use(vocabulary)

    def _matcher(self, constraint):
        kind = constraint.get("type")
        if kind == "tokens":
            return _TokenSet(constraint.get("tokens") or ())
        if kind == "json_object":
            artifact, kind = b"", "json_object"
        elif isinstance(constraint.get("grammar"), str):
            artifact, kind = constraint["grammar"].encode("utf-8"), "grammar"
        else:
            raise NativeError("a constraint is a json_object, a compiled grammar or a token set")
        if self._matchers is None:
            raise NativeError("a constrained request needs the model's byte vocabulary (StepEngine.constrain)")
        return self._matchers.matcher(kind, artifact)

    def _hold(self, holder):
        with self._attaching:
            attachment = self._next_attachment
            self._next_attachment += 1
            self._attachments[attachment] = holder
            return attachment

    def _release_hold(self, attachment):
        with self._attaching:
            self._attachments.pop(attachment, None)

    def _admission(self, _user, request):
        def run():
            answer = self.model.admission(self._request(request))
            if answer not in ("ready", "deferred", "infeasible"):
                raise ValueError(f"admission must answer ready, deferred or infeasible, not {answer!r}")
            return ("ready", "deferred", "infeasible").index(answer)
        return self._guard(run)

    def _open(self, _user, request, session):
        def run():
            handle, options = self._next, self._options(request)
            holder = self._held(options["attachment"]) if options.get("attachment") else None
            if options.get("constrained"):
                if holder is None or holder.matcher is None:
                    raise ValueError("a constrained request without its constraint")
                self._constrained[handle] = holder
            self._sessions[handle] = self.model.open(self._request(request, options, holder))
            self._handles[id(self._sessions[handle])] = handle
            self._next += 1
            session[0] = handle
        return self._guard(run)

    def _mask(self, session):
        """The tokens a constrained session's next token is taken from (None once it failed: its tokens are never
        published)."""
        holder = self._constrained[session]
        if holder.failure is not None:
            return None
        try:
            holder.allowed = holder.matcher.allowed(self._vocabulary)
        except NativeError as error:
            holder.allowed, holder.failure = None, f"the constraint cannot continue: {error}"
        return holder.allowed

    def _check(self, session, produced):
        """Keeps a constrained row's token for commit, or fails the request when it is not one its mask allowed."""
        holder = self._constrained[session]
        if holder.failure is not None or not produced:
            return
        if len(produced) > 1:
            holder.failure = (f"a constrained row produced {len(produced)} tokens: a constrained request decodes one "
                              "token per step (StepRequest.plain)")
        elif holder.allowed is None or produced[0] not in holder.allowed:
            holder.failure = (f"the model plugin produced token {produced[0]}, which the request's constraint does not "
                              "allow: it hands each row's StepRow.allowed to the sampler as params['allowed']")
        else:
            holder.pending = int(produced[0])

    def _probe_start(self):
        """At a probed step's start: the device's work so far finished and torch's peak reset, so the step's time
        and peak are its own; None without CUDA."""
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return torch

    def _probe_end(self, torch, kind, started):
        """At a probed step's end: its seconds with its device work finished, and the free device memory at its peak
        (the free memory now, less what torch reserved during the step beyond what it holds now), kept per kind."""
        if torch is not None:
            torch.cuda.synchronize()
        seconds = round(time.perf_counter() - started, 6)
        found = self._probed.setdefault(kind, {"steps": 0, "slowest_seconds": 0.0,
                                               "recent": deque(maxlen=PROBE_RECENT)})
        found["steps"] += 1
        found["slowest_seconds"] = max(found["slowest_seconds"], seconds)
        free = None
        if torch is not None:
            peak = torch.cuda.max_memory_reserved() - torch.cuda.memory_reserved()
            free = int(torch.cuda.mem_get_info()[0] - max(0, peak))
            found["min_free_device_bytes"] = min(found.get("min_free_device_bytes", free), free)
        found["recent"].append((seconds, free))

    def _step(self, _user, rows, count, progress, tokens, logprobs, top_ids, top_logprobs):
        def run():
            probing = self._probed is not None
            torch = self._probe_start() if probing else None
            started = time.perf_counter()
            constrained, sessions = self._constrained, self._sessions
            packed = ctypes.string_at(rows, count * _ROW.size)
            handles = [entry[0] for entry in _ROW.iter_unpack(packed)]
            scheduled = [tuple.__new__(StepRow, (sessions[handle], prefill, output, work, complete != 0,
                                                 self._mask(handle) if constrained and handle in constrained else None))
                         for handle, prefill, output, work, complete in _ROW.iter_unpack(packed)]
            results = self.model.step(scheduled)
            if not isinstance(results, (list, tuple)):
                results = list(results)
            if len(results) != count:
                raise ValueError(f"step returned {len(results)} results for {count} rows")
            width, top = self._row_tokens, self._top
            out = (ctypes.c_char * (count * _PROGRESS.size)).from_address(ctypes.addressof(progress.contents))
            ids = (ctypes.c_int32 * (count * width)).from_address(ctypes.addressof(tokens.contents))
            for i, result in enumerate(results):
                if not isinstance(result, StepResult):
                    result = StepResult(*result)
                produced = result.tokens if isinstance(result.tokens, list) else list(result.tokens)
                n = len(produced)
                if n > width:
                    raise ValueError(f"a row produced {n} tokens, more than max_row_tokens {width}")
                drafted = result.drafted
                _PROGRESS.pack_into(out, i * _PROGRESS.size, result.prompt_tokens, result.reused_prompt_tokens,
                                    result.token_work, n, 1 if result.prefill_complete else 0,
                                    0 if drafted is None else int(drafted) + 1)
                if n:
                    try:
                        ids[i * width:i * width + n] = produced
                    except TypeError:   # ids of another integer type (numpy's, a tensor's items)
                        ids[i * width:i * width + n] = [int(token) for token in produced]
                if constrained and handles[i] in constrained:
                    self._check(handles[i], produced)
                if self._logprobs and result.logprobs is not None:
                    for j, (logprob, alternatives) in enumerate(list(result.logprobs)[:n]):
                        logprobs[i * width + j] = float(logprob)
                        for a, (token, value) in enumerate(list(alternatives)[:top]):
                            top_ids[(i * width + j) * top + a] = int(token)
                            top_logprobs[(i * width + j) * top + a] = float(value)
            counted = self._counted
            counted["steps"] += 1
            counted["rows"] += count
            counted["step_s"] += time.perf_counter() - started
            if probing:
                # A step that prefills any row is a prompt step: Core settles a cancel only between steps.
                self._probe_end(torch, "prompt" if any(row.prefill_tokens for row in scheduled) else "decode", started)
        return self._guard(run)

    def _commit(self, _user, decisions, count):
        def run():
            found = list(_DECISION.iter_unpack(ctypes.string_at(decisions, count * _DECISION.size)))
            for session, accepted, terminal, cancelled, discard in found if self._constrained else ():
                holder = self._constrained.get(session)
                if holder is None or holder.pending is None:
                    continue
                token, holder.pending = holder.pending, None
                if accepted and not (terminal or cancelled or discard) and holder.failure is None:
                    try:
                        holder.matcher.append(token)
                    except NativeError as error:
                        holder.failure = str(error)
            if self._model_commits:
                sessions = self._sessions
                self.model.commit([tuple.__new__(StepDecision, (sessions[session], accepted, terminal != 0,
                                                                cancelled != 0, discard != 0))
                                   for session, accepted, terminal, cancelled, discard in found])
        return self._guard(run)

    def _facts(self, _user, session, core, out, capacity):
        """The retiring session's facts for its result: Core's (`core`) with the model's request_facts(session)
        (merge_facts), the model asked once: facts longer than Core's buffer wait here for the larger one Core asks with
        next. A failure keeps Core's alone and is reported on stderr: facts never fail a request."""
        data = self._pending_facts.pop(session, None)
        if data is None:
            try:
                model = self._request_facts(self._sessions[session])
                if model is not None and not isinstance(model, dict):
                    raise TypeError(f"request_facts returned {type(model).__name__}, not a dict")
                data = json.dumps(merge_facts(model, json.loads(core.decode("utf-8")) if core else {},
                                              self._counter_names)).encode("utf-8")
            except BaseException as error:  # noqa: BLE001 - never unwinds through C
                sys.stderr.write(f"crit_core: the model's request_facts failed ({type(error).__name__}: {error}); the "
                                 "request reports Core's facts alone\n")
                return -1
        if len(data) >= capacity:
            self._pending_facts[session] = data
            return len(data)
        ctypes.memmove(out, data + b"\0", len(data) + 1)
        return len(data)

    def telemetry(self):
        """The engine's work so far, for crit serve's /health and Eval (docs/PLUGINS.md): what its plugins report
        (the assembly's telemetry(), crit_plugins.Assembly) and Core's own counters ("core": steps and rows, native
        rounds included, the native rounds alone, those of them in the log-probability form, and the seconds the
        Python steps took, the model's step included). A probed engine (STEP_PROBE) adds "probe": per kind of step
        ("prompt": any row prefills, else "decode") the steps probed, the slowest one's seconds with its device work
        finished, the least free device memory at a step's peak (torch's reservations; CUDA only), and "recent": the
        last PROBE_RECENT steps' [seconds, free bytes or None], so a caller can judge the steps of one request (those
        past the count it read before). Probing waits
        on the device at every step's start and end, so only checks turn it on; Core's native rounds are not probed.
        It may run on another thread while the engine steps (docs/PLUGINS.md, Facts and telemetry: a plugin's
        telemetry() reads its counters and does not wait on the device)."""
        found = dict(self._telemetry() or {}) if self._telemetry is not None else {}
        counted = self._counted
        native = self.rounds.runs if getattr(self, "rounds", None) is not None else (0, 0, 0)
        found["core"] = {**(found.get("core") or {}), "steps": counted["steps"] + native[0],
                         "rows": counted["rows"] + native[1], "native_rounds": native[0],
                         "native_logprob_rounds": native[2], "step_seconds": round(counted["step_s"], 6)}
        if self._probed is not None:
            found["core"]["probe"] = {kind: {**value, "recent": [list(sample) for sample in value["recent"]]}
                                      for kind, value in list(self._probed.items())}
        return found

    def _retire(self, _user, session, finish):
        def run():
            if callable(getattr(self.model, "retire", None)):
                self.model.retire(self._sessions[session], FINISHES[finish])
        return self._guard(run)

    def _release(self, _user, session):
        self._constrained.pop(session, None)
        released = self._sessions.pop(session, None)
        if released is not None:
            self._handles.pop(id(released), None)

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
        if getattr(self, "rounds", None) is not None:
            self.rounds.close()
        if getattr(self, "_matchers", None) is not None:
            self._matchers.close()
            self._matchers = None
        if self.model is not None:
            if self._close_parts is not None:
                self._close_parts()
            elif callable(getattr(self.model, "close", None)):
                self.model.close()
            self.model = self._telemetry = self._request_facts = None
            self._sessions.clear()
            # A model's device memory returns only once its objects are freed. Parts that refer to each other (a
            # drafter and its model) wait for a collection, and crit serve opens an engine again in the same host
            # (Eval's lower-context comparison), so collect now and hand torch's cached blocks back.
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    # Free device memory below which a WDDM device counts as full: other processes (a desktop) share it, so the engine
    # can page with its own peak below the device's size.
    FULL_BYTES = 64 << 20

    def _check_device_memory(self):
        """Warn once per device when torch has reserved more than the device holds, or the device has (almost) no
        free memory left. Under the Windows display driver (WDDM, native Windows and WSL) such allocations do not
        fail: the driver moves the excess to system memory, and every kernel that reads it slows down by the host
        link's bandwidth, so the engine only gets slower."""
        if not WDDM:
            return
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return
        for device in range(torch.cuda.device_count()):
            if device in self._oversubscribed:
                continue
            peak, total = torch.cuda.max_memory_reserved(device), torch.cuda.get_device_properties(device).total_memory
            free = torch.cuda.mem_get_info(device)[0]
            if peak > total or free < self.FULL_BYTES:
                self._oversubscribed.add(device)
                held = (f"reserved {peak / 2**30:.1f} GiB on cuda:{device}, which holds {total / 2**30:.1f} GiB"
                        if peak > total else
                        f"left {free / 2**20:.0f} MiB free on cuda:{device} (it reserved {peak / 2**30:.1f} GiB of "
                        f"{total / 2**30:.1f} GiB; other processes hold the rest)")
                sys.stderr.write(
                    f"crit_core: the engine {held}; under WDDM the excess runs from system memory and slows every "
                    "kernel that reads it: keep the engine's peak (weights, caches, prompt activations) below the "
                    "device's free memory (device-placement's reserve_mib keeps room beside the plan)\n")


__all__ = ["NativeError", "NativeRounds", "Scores", "StepDecision", "StepEngine", "StepRequest", "StepResult", "StepRow",
           "merge_facts"]
