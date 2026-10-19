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
the same way, as `reasoning`, which crit serve gives clients as reasoning_content.
"""
from pathlib import Path

SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "logit_bias")


def sampling_capability(name):
    """The package capability that lets crit serve forward request field `name` (same rule as crit_api.options)."""
    return "chat-sampling-" + name.replace("_", "-") + "-v1"


def template_tool_format(text):
    """The registered tool-call format the model's chat templates teach, or None (crit_core.text.tools)."""
    from crit_core.text import templates_tool_format
    return templates_tool_format(text.templates[name].source for name in text.templates)


def special_marker_ids(text, markers):
    """The ids of the markers the tokenizer holds as single special tokens."""
    kept = set()
    for marker in markers:
        ids = text.tokenizer.encode(marker, add_special_tokens=False, split_special_tokens=False)
        if len(ids) == 1 and ids[0] in text.tokenizer.special_ids:
            kept.add(ids[0])
    return frozenset(kept)


def kept_special_ids(text, tool_format=None):
    """Special-token ids among the tool-call markers (of `tool_format`, else of the template's registered format):
    the reply text keeps them so the calls can be parsed; every other special token is still skipped."""
    from crit_core.text.tools import format_markers
    format = tool_format or template_tool_format(text)
    return special_marker_ids(text, format_markers(format)) if format is not None else frozenset()


def capabilities(text, sampling=(), *, stream=True, config=True, tool_format=None):
    """The serving capabilities a package built on ResidentChat declares, for a model's ModelText and the package's
    own `tool_format` (ResidentChat.tool_format), if any."""
    unknown = sorted(set(sampling) - set(SAMPLING))
    if unknown:
        raise ValueError("Unknown sampling fields: " + ", ".join(unknown))
    result = ["chat-json-v1", "chat-resident-v1"]
    result += ["chat-stream-v1"] if stream else []
    result += ["chat-config-v1"] if config else []
    if tool_format is not None or template_tool_format(text):
        result += ["chat-tools-v1"] + (["chat-tool-stream-v1"] if stream else [])
    return result + [sampling_capability(name) for name in SAMPLING if name in sampling]


def chat_prompt(request):
    """(messages, tools) for ModelText.encode_chat: tool history keeps its calls and results, and the request's
    tools are rendered unless tool_choice is "none"; ResidentChat parses the calls."""
    from crit_core.text import openai_messages
    tools = request.get("tools") if request.get("tool_choice", "auto") != "none" else None
    return openai_messages(request["messages"]), (tools or None)


class ResidentChat:
    """chat-resident-v1 over an engine's token generator. Subclasses set `sampling` (the request fields their
    decoder applies) and implement generate_tokens(); `stop_ids` defaults to the model's generation stop tokens.
    Configuration "chat_template_kwargs" passes variables to the chat template (for example enable_thinking).
    Calls are parsed in the format the chat template teaches, or in `tool_format`, a crit_core.text.ToolFormat a
    package sets for a syntax crit does not register; replies carry them as tool_calls (chat-tools-v1). Reasoning is
    split from the reply in the markers the chat template uses (crit_core.text.reasoning) and returned as
    `reasoning`, streamed as {"reasoning": text}; `reasoning_format` sets a crit_core.text.ReasoningFormat for
    markers crit does not register, or False to leave reasoning in the reply."""
    sampling = ()
    tool_format = None
    reasoning_format = None

    def __init__(self, context, model, *, configuration=None, stop_ids=None, capacity=None, template_variables=None):
        from crit_core.text import load_model_text
        self.context = context
        self.configuration = dict(configuration if configuration is not None else (context.configuration or {}))
        self.text = load_model_text(str(Path(model)), clock=None)
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
        first_token_ms, and optionally prefill_tokens (prompt tokens actually computed) and decode_tokens (output
        tokens the decode phase produced, never drafting rounds; default: all but the first)."""
        raise NotImplementedError

    def effective_configuration(self):
        return dict(self.configuration)

    def prompt_ids(self, request):
        messages, tools = chat_prompt(request)
        return self.text.encode_chat(messages, tools=tools, add_generation_prompt=True, **self.template_variables)

    def _run(self, request, cancellation, emit=None):
        ids = self.prompt_ids(request)
        limits = request.get("limits", {})
        output_tokens = int(limits.get("max_output_tokens") or 1024)
        bounds = [c for c in (int(limits.get("kv_tokens") or 0), self.capacity or 0) if c]
        if bounds and len(ids) + output_tokens > min(bounds):
            raise ValueError(f"prompt of {len(ids)} tokens plus {output_tokens} output tokens exceeds the context")
        sampling = dict(request.get("sampling") or {})
        unsupported = sorted(set(sampling) - set(self.sampling))
        if unsupported:
            raise ValueError("This engine does not apply sampling field(s): " + ", ".join(unsupported))
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
                from crit_core.text import ToolReply
                reply = ToolReply(calls_format, request, emit)
        thinking = self._reasoning_format()
        kept = self._kept_markers(calls_format, thinking)
        if kept:
            stream = self.text.text_stream(ids, skip_special_tokens=False)
            dropped = (set(self.text.tokenizer.special_ids) - kept) | self.stop_ids
        else:
            stream, dropped = self.text.text_stream(ids), self.stop_ids
        splitter = None
        if thinking is not None:
            from crit_core.text import ReasoningSplitter
            prompt_end = self.text.tokenizer.decode(list(ids[-16:])).rstrip()
            splitter = ReasoningSplitter(thinking, opened=prompt_end.endswith(thinking.opening))
        pieces, reasoning = [], []

        def deliver(events):
            for kind, value in events:
                if kind == "reasoning":
                    reasoning.append(value)
                    if emit:
                        emit({"reasoning": value})
                    continue
                pieces.append(value)
                if reply:
                    reply.push(value)
                elif emit:
                    emit(value)

        def route(text):
            deliver(splitter.push(text) if splitter else [("text", text)])

        def on_tokens(tokens):
            delta = stream.push([t for t in tokens if t not in dropped])
            if delta.text:
                route(delta.text)

        tokens, timing = self.generate_tokens(ids, output_tokens, sampling, on_tokens, cancellation.is_set)
        if tokens is None:
            cancellation.check()
            raise RuntimeError("generation cancelled")
        tail = stream.finish()
        if tail.text:
            route(tail.text)
        if splitter:
            deliver(splitter.finish())
        calls = None
        if reply:
            content, calls = reply.finish()
            pieces = [content]
        n = len(tokens)
        decode = timing.get("decode_tokens", n - 1)
        telemetry = {"prompt_tokens": len(ids), "output_tokens": n,
                     "prefill_tokens": int(timing.get("prefill_tokens", len(ids))), "prefill_ms": timing["prefill_ms"],
                     "decode_tokens": min(int(decode), n - 1) if n > 1 else None,
                     "decode_ms": timing["decode_ms"] if n > 1 else None,
                     "ttft_ms": timing["first_token_ms"], "missing": {}}
        if n <= 1:
            telemetry["missing"]["decode"] = "the reply has a single token"
        return "".join(pieces), telemetry, calls, "".join(reasoning)

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
                kept |= special_marker_ids(self.text, thinking)
            cache[key] = frozenset(kept)
        return cache[key]

    def _response(self, request, text, telemetry, calls=None, reasoning=""):
        response = {"schema_version": 1, "record_kind": "chat_response", "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"], "package_sha256": request["package_sha256"],
                    "assistant": text, "telemetry": telemetry}
        if calls:
            response["tool_calls"] = calls
        if reasoning:
            response["reasoning"] = reasoning
        return response

    def generate(self, request, cancellation):
        return self._response(request, *self._run(request, cancellation))

    def generate_stream(self, request, cancellation, emit):
        return self._response(request, *self._run(request, cancellation, emit))

    def reset(self):
        pass

    def close(self):
        pass


__all__ = ["ResidentChat", "SAMPLING", "capabilities", "chat_prompt", "kept_special_ids", "special_marker_ids",
           "sampling_capability", "template_tool_format"]
