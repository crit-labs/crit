"""The shared body of a crit serve adapter (chat-resident-v1, docs/SERVING.md): an engine supplies only decoding.

Every resident chat adapter has to render the conversation with the model's chat template (tools and tool history
included), validate and forward per-request sampling, stream text without the model's stop tokens, honor
cancellation and report Core's phase telemetry. ResidentChat does all of that once, so an adapter cannot ship
without a feature by forgetting it:

    from crit_core.serving import ResidentChat

    class Engine(ResidentChat):
        sampling = ("temperature", "top_p", "top_k", "seed")   # request fields the decoder applies

        def __init__(self, context):
            model = Path(context.model_file("config.json")).parent
            super().__init__(context, model)
            self.decoder = open_my_engine(model, self.capacity)

        def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
            tokens, timing = self.decoder.generate(ids, max_new_tokens, **sampling, on_tokens=on_tokens,
                                                   cancelled=cancelled)
            return tokens, timing   # tokens None when cancelled; timing: prefill_ms, decode_ms, first_token_ms

    def create_engine(context):
        return Engine(context)

Declare `capabilities(text, Engine.sampling, tool_format=Engine.tool_format)` in the package record; `crit run
package` checks that a resident chat package declares tools (when the model's template teaches a call format) and
temperature. ResidentChat is the one place calls are parsed: in the format the model's chat template teaches when
crit_core.text.tools registers it, else in the package's own `tool_format` (a crit_core.text.ToolFormat; crit
carries no model-specific syntax). Either way the adapter returns parsed calls (chat-tools-v1, streamed with
chat-tool-stream-v1). Reasoning in the template's markers (<think>...</think>) is returned apart from the reply in
the same way, as `reasoning`, which crit serve gives clients as reasoning_content. An engine that reports token
log-probabilities sets `logprobs = True` and declares capabilities(..., logprobs=True) (chat-logprobs-v1): a request
then carries "logprobs": n, ResidentChat passes logprobs=n to generate_tokens, and the decoder calls
on_tokens(tokens, rows) with one (logprob, [(token, logprob), ...n alternatives]) row per token. An engine that
scores fixed text through its serving paths sets `scores = True`, implements score_tokens() and declares
capabilities(..., score=True) (chat-score-v1): crit serve's POST /v1/crit/score then returns, for given prompt and
continuation ids, every prompt position's row from its prefill and every continuation token's from its decode,
forced to the continuation.
"""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "logit_bias")
LOGPROBS_CAPABILITY = "chat-logprobs-v1"
SCORE_CAPABILITY = "chat-score-v1"
TOP_LOGPROBS = 20  # the most alternatives per token a request asks for (OpenAI's top_logprobs range)
MIN_LOGPROB = -9999.0  # what a zero probability reads as, as OpenAI reports it (JSON has no infinity)


def sampling_capability(name):
    """The package capability that lets crit serve forward request field `name` (same rule as crit_api.options)."""
    return "chat-sampling-" + name.replace("_", "-") + "-v1"


def template_tool_format(text):
    """The registered tool-call format the model's chat templates teach, or None (crit_core.text.tools)."""
    from crit_core.text import templates_tool_format
    return templates_tool_format(text.templates[name].source for name in text.templates)


def special_marker_ids(text, markers):
    """The ids of the markers the tokenizer holds as single special tokens."""
    from crit_core.text.markers import marker_ids
    return frozenset(marker_ids(text.tokenizer, markers).values())


def kept_special_ids(text, tool_format=None):
    """Special-token ids among the tool-call markers (of `tool_format`, else of the template's registered format):
    the reply text keeps them so the calls can be parsed; every other special token is still skipped."""
    from crit_core.text.tools import format_markers
    format = tool_format or template_tool_format(text)
    return special_marker_ids(text, format_markers(format)) if format is not None else frozenset()


def capabilities(text, sampling=(), *, stream=True, config=True, tool_format=None, images=False, concurrent=False,
                 logprobs=False, score=False):
    """The serving capabilities a package built on ResidentChat declares, for a model's ModelText and the package's
    own `tool_format` (ResidentChat.tool_format), if any. logprobs: the engine sets ResidentChat.logprobs; score:
    it sets ResidentChat.scores and implements score_tokens()."""
    unknown = sorted(set(sampling) - set(SAMPLING))
    if unknown:
        raise ValueError("Unknown sampling fields: " + ", ".join(unknown))
    result = ["chat-json-v1", "chat-resident-v1"]
    result += ["chat-stream-v1"] if stream else []
    result += ["chat-config-v1"] if config else []
    result += ["chat-images-v1"] if images else []
    result += ["chat-concurrent-v1"] if concurrent else []
    result += [LOGPROBS_CAPABILITY] if logprobs else []
    result += [SCORE_CAPABILITY] if score else []
    if tool_format is not None or template_tool_format(text):
        result += ["chat-tools-v1"] + (["chat-tool-stream-v1"] if stream else [])
    return result + [sampling_capability(name) for name in SAMPLING if name in sampling]


def logprob_row(token, row, alternatives):
    """[token, logprob, [[token, logprob], ...]] for a response's "logprobs": exactly `alternatives` ranked
    alternatives, log-probabilities clamped into [MIN_LOGPROB, 0] (Core allows -inf and a rounding tolerance)."""
    logprob, top = row
    if len(top) != alternatives:
        raise ValueError(f"The engine returned {len(top)} alternatives for a token, not {alternatives}")
    clamp = lambda value: min(0.0, max(MIN_LOGPROB, float(value)))
    return [int(token), clamp(logprob), [[int(other), clamp(value)] for other, value in top]]


def chat_prompt(request, *, images=False):
    """(messages, tools) for ModelText.encode_chat: tool history keeps its calls and results, and the request's
    tools are rendered unless tool_choice is "none"; ResidentChat parses the calls."""
    from crit_core.text import openai_messages
    tools = request.get("tools") if request.get("tool_choice", "auto") != "none" else None
    return openai_messages(request["messages"], images=images), (tools or None)


@dataclass(frozen=True)
class PreparedPrompt:
    """Request-local token ids, optional engine inputs, and actual context positions.

    Image expansion may occupy more positions than the template's placeholder ids.
    The engine supplies that count; Crit never guesses it from an image's size.
    """
    ids: object
    inputs: object = None
    token_count: int | None = None


class ResidentChat:
    """chat-resident-v1 over an engine's token generator. Subclasses set `sampling` (the request fields their
    decoder applies) and implement generate_tokens(); `stop_ids` defaults to the model's generation stop tokens.
    Configuration "chat_template_kwargs" passes variables to the chat template (for example enable_thinking).
    Calls are parsed in the format the chat template teaches, or in `tool_format`, a crit_core.text.ToolFormat a
    package sets for a syntax crit does not register; replies carry them as tool_calls (chat-tools-v1). Reasoning is
    split from the reply in the markers the chat template uses (crit_core.text.reasoning) and returned as
    `reasoning`, streamed as {"reasoning": text}; `reasoning_format` sets a crit_core.text.ReasoningFormat for
    markers crit does not register, or False to leave reasoning in the reply. `logprobs` True: the decoder reports
    each token's log-probability and up to TOP_LOGPROBS alternatives (chat-logprobs-v1); the response then carries
    them for every generated token but stop tokens, reasoning and tool calls included."""
    sampling = ()
    logprobs = False
    scores = False
    tool_format = None
    reasoning_format = None
    images = False
    max_concurrent_requests = 1

    def create_request(self, request):
        """Return an independent ResidentChat sharing only safe model resources.

        Concurrent packages implement this factory. The host resets and closes
        its result after generation settles; the parent keeps shared weights.
        """
        raise NotImplementedError("Concurrent adapters must provide create_request(request)")

    def __init__(self, context, model, *, configuration=None, stop_ids=None, capacity=None, template_variables=None):
        from crit_core.model import open_model
        from crit_core.text import load_model_text
        self.context = context
        self.configuration = dict(configuration if configuration is not None else (context.configuration or {}))
        # A folder's files, or those a lone GGUF's metadata derives.
        self.text = load_model_text(open_model(model).files, clock=None)
        limits = getattr(context, "limits", {}) or {}
        self.capacity = int(capacity or limits.get("kv_tokens") or 0) or None
        self.stop_ids = set(self.text.generation.stop_token_ids if stop_ids is None else stop_ids)
        if template_variables is None:
            template_variables = self.configuration.get("chat_template_kwargs") or {}
        self.template_variables = dict(template_variables)
        self.kept_special = None   # tool-marker special tokens, found on the first request with tools
        self.calls_format = None   # tool_format, else the template's registered format, found with them
        unknown = sorted(set(self.sampling) - set(SAMPLING))
        if unknown:
            raise ValueError("Unknown sampling fields: " + ", ".join(unknown))

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        """Decode up to max_new_tokens after prompt `ids`, calling on_tokens(list of ids) as tokens are accepted.
        `sampling` holds only request fields from `self.sampling`; absent fields keep the engine's defaults.
        Return (tokens, timing), with tokens None when cancelled() turned true. timing: prefill_ms, decode_ms,
        first_token_ms, and optionally prefill_tokens (prompt tokens actually computed), decode_tokens (output
        tokens the decode phase produced, never drafting rounds; default: all but the first) and rounds (target
        passes after prefill, a speculative engine's verify rounds; crit serve reports output tokens per round).
        An engine with `logprobs` also takes logprobs=n when the request asks for them and then calls
        on_tokens(tokens, rows): per token (logprob, [(token, logprob), ...]) with its n most likely alternatives,
        ranked, natural-log probabilities of the raw logits (crit::core::TokenLogprob)."""
        raise NotImplementedError

    def score_tokens(self, ids, continuation, top_k, cancelled, prompt_rows):
        """chat-score-v1: (prompt rows, continuation rows, timing) for prompt `ids` followed by the fixed
        `continuation`, each row (logprob of the token, [(token, logprob), ...its top_k alternatives, ranked]).
        Prompt rows score ids[1:] from one prefill of `ids`, as the engine prefills a request, reading every
        position; with prompt_rows false there are none ([]), and the prompt is only prefilled as for a reply.
        Continuation rows score each continuation token as decoding after that prefill produces it, every step
        forced to the continuation (a speculative engine's drafts accepted only while they equal it, each
        correction and bonus token taken from it): the first from the prefill's last position, the others from
        decode steps. Return None for the rows when cancelled() turned true. timing: prefill_ms, decode_ms."""
        raise NotImplementedError

    def score(self, request, cancellation):
        """The response record of a score request (chat-score-v1): {"prompt": rows, "continuation": rows,
        "telemetry"}, rows [token, logprob, [[token, logprob], ...top_logprobs]] as logprob_row makes them."""
        ids, continuation = list(request["prompt_ids"]), list(request["continuation_ids"])
        top_k, wanted = request["top_logprobs"], request["prompt_rows"]
        prompt_rows, continuation_rows, timing = self.score_tokens(ids, continuation, top_k, cancellation.is_set,
                                                                   wanted)
        if prompt_rows is None or continuation_rows is None:
            cancellation.check()
            raise ValueError("The engine stopped scoring without being cancelled")
        expected = len(ids) - 1 if wanted else 0
        if len(prompt_rows) != expected or len(continuation_rows) != len(continuation):
            raise ValueError(f"The engine scored {len(prompt_rows)} prompt and {len(continuation_rows)} continuation "
                             f"positions, not {expected} and {len(continuation)}")
        return {"schema_version": 1, "record_kind": "score_response", "request_id": request["request_id"],
                "candidate_id": request["candidate_id"], "package_sha256": request["package_sha256"],
                "prompt": [logprob_row(token, row, top_k) for token, row in zip(ids[1:], prompt_rows)],
                "continuation": [logprob_row(token, row, top_k) for token, row in zip(continuation, continuation_rows)],
                "telemetry": {key: timing.get(key) for key in ("prefill_ms", "decode_ms")}}

    def effective_configuration(self):
        return dict(self.configuration)

    def prompt_ids(self, request):
        messages, tools = chat_prompt(request, images=self.images)
        return self.text.encode_chat(messages, tools=tools, add_generation_prompt=True, **self.template_variables)

    @contextmanager
    def prepare_prompt(self, request, cancellation):
        """Override for images; yield PreparedPrompt and release private inputs in finally.

        Use prompt_ids() to render ordered image/text parts through the checkpoint's
        template. The engine supplies image decoding, vision arithmetic and layout.
        """
        if any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
               for m in request["messages"]):
            raise ValueError("Image input requires the engine's prepare_prompt implementation")
        yield PreparedPrompt(self.prompt_ids(request))

    def generate_prepared(self, prompt, max_new_tokens, sampling, on_tokens, cancelled, **options):
        """Override to consume prepared image inputs with the engine's token generator; `options` (logprobs) go
        to generate_tokens."""
        if prompt.inputs is not None:
            raise NotImplementedError("Prepared image inputs require generate_prepared")
        return self.generate_tokens(prompt.ids, max_new_tokens, sampling, on_tokens, cancelled, **options)

    def _run(self, request, cancellation, emit=None):
        with self.prepare_prompt(request, cancellation) as prompt:
            return self._generate(request, cancellation, prompt, emit)

    def _generate(self, request, cancellation, prompt, emit):
        from crit_core.text import MarkerDecoder, ReasoningSplitter, TokenText, ToolReply
        from crit_core.text.markers import prompt_ends_with

        ids = prompt.ids
        prompt_tokens = len(ids) if prompt.token_count is None else prompt.token_count
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError("Prepared prompt token_count must be a positive integer")
        limits = request.get("limits", {})
        output_tokens = int(limits.get("max_output_tokens") or 1024)
        bounds = [c for c in (int(limits.get("kv_tokens") or 0), self.capacity or 0) if c]
        if bounds and prompt_tokens + output_tokens > min(bounds):
            raise ValueError(f"prompt of {prompt_tokens} tokens plus {output_tokens} output tokens exceeds the context")
        sampling = dict(request.get("sampling") or {})
        unsupported = sorted(set(sampling) - set(self.sampling))
        if unsupported:
            raise ValueError("This engine does not apply sampling field(s): " + ", ".join(unsupported))
        alternatives = request.get("logprobs")
        if alternatives is not None and not self.logprobs:
            raise ValueError("This engine does not report log-probabilities")
        if alternatives is not None and not (type(alternatives) is int and 0 <= alternatives <= TOP_LOGPROBS):
            raise ValueError(f"logprobs must be an integer in [0, {TOP_LOGPROBS}]")
        calls_format, reply = None, None
        if request.get("tools"):   # calls only come when tools are offered
            if getattr(self, "calls_format", None) is None:   # also subclasses that set up self.text themselves
                self.calls_format = (self.tool_format if self.tool_format is not None
                                     else template_tool_format(self.text))
            calls_format = self.calls_format
            if calls_format is not None and request.get("tool_choice", "auto") != "none":
                if request.get("tool_choice", "auto") != "auto":
                    raise ValueError("This engine parses tool calls from free generation: tool_choice must be auto "
                                     "or none")
                reply = ToolReply(calls_format, request, emit, tokenizer=self.text.tokenizer)
        thinking = self._reasoning_format()
        kept = self._kept_markers(calls_format, thinking)
        dropped = ((set(self.text.tokenizer.special_ids) - kept) | self.stop_ids) if kept else self.stop_ids
        splitter = None
        boundaries = dict(reply.parser.token_ids) if reply else {}
        if thinking is not None:
            splitter = ReasoningSplitter(thinking, tokenizer=self.text.tokenizer)
            if prompt_ends_with(self.text.tokenizer, ids, thinking.opening, splitter.token_ids):
                splitter.state = "reasoning"
            boundaries.update(splitter.token_ids)
        stream = MarkerDecoder(self.text.tokenizer, markers={v: k for k, v in boundaries.items()},
                               stream=self.text.text_stream(ids, skip_special_tokens=not kept))
        pieces, reasoning = [], []
        rows = None if alternatives is None else []

        def deliver(events):
            for kind, value in events:
                if kind == "reasoning":
                    reasoning.append(value)
                    if emit:
                        emit({"reasoning": value})
                    continue
                text = value.text if isinstance(value, TokenText) else value
                pieces.append(text)
                if reply:
                    reply.push(value)
                elif emit:
                    emit(text)

        def route(text):
            deliver(splitter.push(text) if splitter else [("text", text)])

        def on_tokens(tokens, logprobs=None):
            if rows is not None:
                if logprobs is None or len(logprobs) != len(tokens):
                    raise ValueError("The engine returned tokens without their log-probabilities")
                rows.extend(logprob_row(token, row, alternatives) for token, row in zip(tokens, logprobs)
                            if token not in self.stop_ids)
            for part in stream.push([t for t in tokens if t not in dropped]):
                route(part)

        options = {} if alternatives is None else {"logprobs": alternatives}
        tokens, timing = self.generate_prepared(prompt, output_tokens, sampling, on_tokens, cancellation.is_set,
                                                **options)
        if tokens is None:
            cancellation.check()
            raise RuntimeError("generation cancelled")
        for part in stream.finish():
            route(part)
        if splitter:
            deliver(splitter.finish())
        calls = None
        if reply:
            content, calls = reply.finish(cut=len(tokens) >= output_tokens)
            pieces = [content]
        n = len(tokens)
        decode = timing.get("decode_tokens", n - 1)
        telemetry = {"prompt_tokens": prompt_tokens, "output_tokens": n,
                     "prefill_tokens": int(timing.get("prefill_tokens", prompt_tokens)), "prefill_ms": timing["prefill_ms"],
                     "decode_tokens": min(int(decode), n - 1) if n > 1 else None,
                     "decode_ms": timing["decode_ms"] if n > 1 else None,
                     "ttft_ms": timing["first_token_ms"], "missing": {}}
        if n <= 1:
            telemetry["missing"]["decode"] = "the reply has a single token"
        elif timing.get("rounds"):
            telemetry["decode_rounds"] = min(int(timing["rounds"]), n)
        logprobs = None
        if rows is not None:
            # Each token named once by its bytes, so the server can give clients token text and bytes.
            data = self.text.tokenizer.token_bytes()
            ids = {row[0] for row in rows} | {other for row in rows for other, _ in row[2]}
            if ids and not 0 <= min(ids) <= max(ids) < len(data):
                raise ValueError("The engine reported log-probabilities for ids outside the tokenizer's vocabulary")
            logprobs = {"tokens": rows, "bytes": {str(token): data[token].hex() for token in sorted(ids)}}
        return "".join(pieces), telemetry, calls, "".join(reasoning), logprobs

    def _reasoning_format(self):
        """reasoning_format, else the reasoning format the chat templates use (None: no split; False disables it)."""
        if getattr(self, "_thinking", None) is None:
            chosen = getattr(self, "reasoning_format", None)
            sources = [self.text.templates[n].source for n in getattr(self.text, "templates", {})]
            if chosen is False or (chosen is None and not sources):
                self._thinking = False
            else:
                from crit_core.text.reasoning import reasoning_format, template_reasoning_format
                name = chosen if chosen is not None else template_reasoning_format(sources)
                self._thinking = reasoning_format(name) if name else False
        return self._thinking or None

    def _kept_markers(self, calls_format, thinking):
        """Special-token ids the reply text keeps: tool-call markers when tools are offered, reasoning markers."""
        cache = getattr(self, "_kept_cache", None)
        if cache is None:
            cache = self._kept_cache = {}
        key = (calls_format is not None, thinking is not None)
        if key not in cache:
            kept = set(kept_special_ids(self.text, calls_format)) if calls_format is not None else set()
            if thinking is not None:
                kept |= special_marker_ids(self.text, (thinking.opening, thinking.closing))
            cache[key] = frozenset(kept)
        return cache[key]

    def _response(self, request, text, telemetry, calls=None, reasoning="", logprobs=None):
        response = {"schema_version": 1, "record_kind": "chat_response", "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"], "package_sha256": request["package_sha256"],
                    "assistant": text, "telemetry": telemetry}
        if calls:
            response["tool_calls"] = calls
        if reasoning:
            response["reasoning"] = reasoning
        if logprobs is not None:
            response["logprobs"] = logprobs
        return response

    def generate(self, request, cancellation):
        return self._response(request, *self._run(request, cancellation))

    def generate_stream(self, request, cancellation, emit):
        return self._response(request, *self._run(request, cancellation, emit))

    def reset(self):
        pass

    def close(self):
        pass


__all__ = ["ResidentChat", "PreparedPrompt", "SAMPLING", "LOGPROBS_CAPABILITY", "SCORE_CAPABILITY", "TOP_LOGPROBS",
           "capabilities",
           "chat_prompt", "kept_special_ids", "logprob_row", "special_marker_ids", "sampling_capability",
           "template_tool_format"]
