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
the same way, as `reasoning`, which crit serve gives clients as reasoning_content. The thinking switches the
templates read (TEMPLATE_KWARGS: enable_thinking, thinking, reasoning_effort) are declared with
chat-template-kwargs-v1, and a request's template_kwargs set them for that request. An engine that reports token
log-probabilities sets `logprobs = True` and declares capabilities(..., logprobs=True) (chat-logprobs-v1): a request
then carries "logprobs": n, ResidentChat passes logprobs=n to generate_tokens, and the decoder calls
on_tokens(tokens, rows) with one (logprob, [(token, logprob), ...n alternatives]) row per token. An engine that
scores fixed text through its serving paths sets `scores = True`, implements score_tokens() and declares
capabilities(..., score=True) (chat-score-v1): crit serve's POST /v1/crit/score then returns, for given prompt and
continuation ids, every prompt position's row from its prefill and every continuation token's from its decode,
forced to the continuation (score_tokens returns rows as a list or, from flat arrays, as ScoreRows). A speculative
engine that also decodes one token per target pass, without drafting, sets `plain = True` and declares
capabilities(..., plain=True) (chat-plain-v1): a request carrying "decode": "plain" then reaches generate_tokens as
plain=True (Eval's exactness check on the served engine). An engine whose decoder keeps a reply to a
crit-byte-grammar-v1 grammar, as for response_format, sets `grammars = True` and declares capabilities(...,
grammars=True), which adds chat-tool-choice-v1 where ResidentChat can write the call syntax (forces_calls):
tool_choice "required" or a named function then reaches generate_tokens as grammar=artifact, the
grammar of a reply that makes the forced calls (crit_core.text.tools.call_grammar). An engine keeps the facts Eval's
headroom reads in `facts` (engine_facts), and generate_tokens' timing may add accepted (drafts accepted per round),
verify_ms and draft_ms for crit serve's telemetry. Every ResidentChat fits a request's output to the context
(chat-fit-output-v1): with limits.fit_output it decodes at most the room the prompt leaves and names that cap in the
response as max_output_tokens. A request carrying "continue_final_message" (chat-continue-v1, which capabilities()
declares) ends with an assistant message: ResidentChat renders it left open with the chat template, and the reply is
what the model writes after it. A ResidentChat also continues a raw prompt (completion-v1, crit serve's
/v1/completions; an images engine declares it with capabilities(..., completion=True)): a request carrying "prompt",
its text or token ids, in place of "messages" is encoded with completion_ids() and reaches generate_prepared as
PreparedPrompt(ids), without the chat template or prepare_prompt. Every ResidentChat tokenizes text, renders a
conversation to the ids a reply prefills (its template_kwargs and continue_final_message included) and detokenizes
ids (chat-tokenize-v1, which capabilities() declares): crit serve's /tokenize, /detokenize, /apply-template and
/v1/messages/count_tokens call tokenize(), render() and detokenize() on their own thread, beside generation, so they
read only the tokenizer and the chat template.
"""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
import array
import base64
import sys

SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "logit_bias")
LOGPROBS_CAPABILITY = "chat-logprobs-v1"
SCORE_CAPABILITY = "chat-score-v1"
PLAIN_CAPABILITY = "chat-plain-v1"
# A request with limits.fit_output cuts its output cap to the room the prompt leaves in the context.
FIT_OUTPUT_CAPABILITY = "chat-fit-output-v1"
CONTINUE_CAPABILITY = "chat-continue-v1"
COMPLETION_CAPABILITY = "completion-v1"
TOKENIZE_CAPABILITY = "chat-tokenize-v1"
# What a resident engine reports for Eval (crit serve's /v1/models engine_facts): whether it speculates and can decode
# plainly, its draft width, the bytes one model pass and one draft step read, and the activation format of its
# quantized matrices' prompt GEMMs (crit-eval's prefill ceilings by activation format).
ENGINE_FACTS = ("speculative", "plain", "draft_width", "bytes_per_pass", "host_bytes_per_pass", "draft_bytes_per_step",
                "draft_ms", "kv_bytes_per_token", "prompt_activations")
# The activation formats crit-eval prices a quantized matrix group's prompt GEMMs at (its ceiling.ACTIVATIONS).
PROMPT_ACTIVATIONS = ("bf16", "fp16_acc", "fp8", "stored")
TOP_LOGPROBS = 20  # the most alternatives per token a request asks for (OpenAI's top_logprobs range)
MIN_LOGPROB = -9999.0  # what a zero probability reads as, as OpenAI reports it (JSON has no infinity)
TEMPLATE_KWARGS_CAPABILITY = "chat-template-kwargs-v1"
# The chat-template variables a request may set (chat-template-kwargs-v1) and the values each takes: the thinking
# switches model families' templates read (Qwen3 and GLM enable_thinking, DeepSeek-V3.1 and Granite thinking) and
# gpt-oss's reasoning_effort.
TEMPLATE_KWARGS = {"enable_thinking": (False, True), "thinking": (False, True),
                   "reasoning_effort": ("low", "medium", "high")}


def sampling_capability(name):
    """The package capability that lets crit serve forward request field `name` (same rule as crit_api.options)."""
    return "chat-sampling-" + name.replace("_", "-") + "-v1"


def template_kwarg_capability(name):
    """The package capability that lets a request set chat-template variable `name` (one of TEMPLATE_KWARGS)."""
    return "chat-template-" + name.replace("_", "-") + "-v1"


def template_kwargs(text):
    """The TEMPLATE_KWARGS the model's chat templates read, in TEMPLATE_KWARGS order."""
    from jinja2 import meta
    from crit_core.text import environment
    read = set()
    for name in text.templates:
        read |= meta.find_undeclared_variables(environment(None).parse(text.templates[name].source))
    return tuple(name for name in TEMPLATE_KWARGS if name in read)


def template_value(name, value):
    """Whether `value` is one TEMPLATE_KWARGS[name] takes, of the same JSON type (true is not 1)."""
    return any(type(value) is type(choice) and value == choice for choice in TEMPLATE_KWARGS[name])


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


def forces_calls(text, tool_format=None):
    """Whether ResidentChat can force calls (chat-tool-choice-v1) in the package's `tool_format`, else the format the
    model's chat template teaches: it writes grammars for the formats crit registers, and only when the tokenizer
    spells their markers as text, since a grammar cannot write a special token."""
    from crit_core.text.tools import FORMATS, format_markers
    format = tool_format if tool_format is not None else template_tool_format(text)
    return isinstance(format, str) and format in FORMATS and not special_marker_ids(text, format_markers(format))


def capabilities(text, sampling=(), *, stream=True, config=True, tool_format=None, images=False, concurrent=False,
                 logprobs=False, score=False, plain=False, grammars=False, continuation=True, completion=None,
                 tokenize=True):
    """The serving capabilities a package built on ResidentChat declares, for a model's ModelText and the package's
    own `tool_format` (ResidentChat.tool_format), if any. logprobs: the engine sets ResidentChat.logprobs; score:
    it sets ResidentChat.scores and implements score_tokens(); plain: it sets ResidentChat.plain; grammars: it sets
    ResidentChat.grammars; continuation False: it renders its own prompt and does not continue a final assistant
    message; completion: raw prompts (completion-v1), which ResidentChat continues with any decoder, by default
    unless `images`: an images engine passes True when its generate_prepared also takes PreparedPrompt(ids) without
    inputs; tokenize False: its prompt_ids() override is not safe to call beside generation (ResidentChat.render).
    The TEMPLATE_KWARGS the chat templates read are declared with chat-template-kwargs-v1. Every ResidentChat fits
    its output to the context (chat-fit-output-v1)."""
    unknown = sorted(set(sampling) - set(SAMPLING))
    if unknown:
        raise ValueError("Unknown sampling fields: " + ", ".join(unknown))
    result = ["chat-json-v1", "chat-resident-v1", FIT_OUTPUT_CAPABILITY]
    result += ["chat-stream-v1"] if stream else []
    result += ["chat-config-v1"] if config else []
    result += [CONTINUE_CAPABILITY] if continuation else []
    result += [COMPLETION_CAPABILITY] if (not images if completion is None else completion) else []
    result += [TOKENIZE_CAPABILITY] if tokenize else []
    result += ["chat-images-v1"] if images else []
    result += ["chat-concurrent-v1"] if concurrent else []
    result += [LOGPROBS_CAPABILITY] if logprobs else []
    result += [SCORE_CAPABILITY] if score else []
    result += [PLAIN_CAPABILITY] if plain else []
    if tool_format is not None or template_tool_format(text):
        result += ["chat-tools-v1"] + (["chat-tool-stream-v1"] if stream else [])
        result += ["chat-tool-choice-v1"] if grammars and forces_calls(text, tool_format) else []
    variables = template_kwargs(text)
    result += [TEMPLATE_KWARGS_CAPABILITY, *map(template_kwarg_capability, variables)] if variables else []
    return result + [sampling_capability(name) for name in SAMPLING if name in sampling]


def engine_facts(engine):
    """The ENGINE_FACTS a resident engine holds in its `facts` (crit_core.native's engines hold their info there):
    booleans for speculative and plain, {"link", "cpu"} byte counts for host_bytes_per_pass, non-negative finite
    numbers for the rest; plain only on a speculative engine that sets `plain`. None when it reports none."""
    import math
    found = getattr(engine, "facts", None)
    if not isinstance(found, dict):
        return None
    out = {}
    for key in ENGINE_FACTS:
        value = found.get(key)
        if key == "prompt_activations":
            if prompt_activations(value):
                out[key] = value
        elif key in ("speculative", "plain"):
            if isinstance(value, bool):
                out[key] = value
        elif key == "host_bytes_per_pass":
            # The part of bytes_per_pass read from host memory: {"link": copied over the host link, "cpu": computed
            # by the CPU}, which crit-eval prices at the probe's host rates.
            if isinstance(value, dict) and value and set(value) <= {"link", "cpu"} and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                    for v in value.values()):
                out[key] = dict(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            out[key] = value
    if "plain" in out:
        out["plain"] = bool(out["plain"] and out.get("speculative") and getattr(engine, "plain", False))
    return out or None


def prompt_activations(value):
    """Whether `value` declares prompt activations as crit-eval reads them: one of PROMPT_ACTIVATIONS for every quantized
    format, or per stored format (fp8, nvfp4) one of them or {activation: share} with non-negative shares."""
    import math
    if isinstance(value, str):
        return value in PROMPT_ACTIVATIONS
    if not isinstance(value, dict) or not value:
        return False
    for name, mix in value.items():
        if not isinstance(name, str):
            return False
        if isinstance(mix, str):
            if mix not in PROMPT_ACTIVATIONS:
                return False
        elif not isinstance(mix, dict) or not mix or any(
                key not in PROMPT_ACTIVATIONS or isinstance(share, bool) or not isinstance(share, (int, float))
                or not math.isfinite(share) or share < 0 for key, share in mix.items()):
            return False
    return True


def logprob_row(token, row, alternatives):
    """[token, logprob, [[token, logprob], ...]] for a response's "logprobs": exactly `alternatives` ranked
    alternatives, log-probabilities clamped into [MIN_LOGPROB, 0] (Core allows -inf and a rounding tolerance)."""
    logprob, top = row
    if len(top) != alternatives:
        raise ValueError(f"The engine returned {len(top)} alternatives for a token, not {alternatives}")
    return [int(token), clamp_logprob(logprob), [[int(other), clamp_logprob(value)] for other, value in top]]


def clamp_logprob(value):
    """A log-probability clamped into [MIN_LOGPROB, 0] (NaN reads as MIN_LOGPROB)."""
    return min(0.0, max(MIN_LOGPROB, float(value)))


class ScoreRows:
    """Score rows as flat arrays, which score_tokens() may return in place of a list of rows (crit_core.native's
    engines do): each row's log-probability in `logprobs` and its `alternatives` ranked alternatives row by row in
    `top_tokens` (array.array "i", int32) and `top_logprobs` (array.array "f" or "d"). It reads as the rows it holds
    (len, indexing and iteration give (logprob, [(token, logprob), ...])), and ResidentChat.score packs it without a
    loop over its rows."""

    def __init__(self, logprobs, top_tokens, top_logprobs, alternatives):
        if len(top_tokens) != len(top_logprobs) or len(top_tokens) != alternatives * len(logprobs):
            raise ValueError(f"Score rows need {alternatives} alternative tokens and log-probabilities per row")
        self.logprobs, self.top_tokens, self.top_logprobs = logprobs, top_tokens, top_logprobs
        self.alternatives = alternatives

    def __len__(self):
        return len(self.logprobs)

    def __getitem__(self, index):
        index, k = range(len(self))[index], self.alternatives
        return self.logprobs[index], list(zip(self.top_tokens[index * k:(index + 1) * k],
                                              self.top_logprobs[index * k:(index + 1) * k]))

    def __iter__(self):
        return (self[index] for index in range(len(self)))


def _encoded(typecode, values):
    """base64 of `values` as a little-endian array of `typecode`."""
    try:
        same = isinstance(values, array.array) and values.typecode == typecode
        data = values if same else array.array(typecode, values)
    except OverflowError:
        raise ValueError("Score token ids must be below 2**31") from None
    if sys.byteorder != "little":
        data = array.array(typecode, data)
        data.byteswap()
    return base64.b64encode(data.tobytes()).decode("ascii")


def _clamped(values):
    """array("d") of log-probabilities clamped as clamp_logprob clamps them; the comparison keeps the values already
    in range without a call per value."""
    return array.array("d", [value if MIN_LOGPROB <= value < 0.0 else clamp_logprob(value) for value in values])


def packed_scores(tokens, rows, alternatives):
    """One part of a score response record in the packed format (ResidentChat.score for a request with "format":
    "packed"): {"count", "alternatives", "tokens", "logprobs", "top_tokens", "top_logprobs"}, base64 of
    little-endian arrays, the tokens and alternative tokens int32 and the log-probabilities float64 (an engine's values
    exactly, as JSON rows hold them), the alternatives row by row. `rows` are a ScoreRows or a list of (logprob,
    [(token, logprob), ...]) rows, one per token; log-probabilities are clamped as logprob_row clamps them."""
    if isinstance(rows, ScoreRows):
        if rows.alternatives != alternatives:
            raise ValueError(f"The engine returned {rows.alternatives} alternatives for a token, not {alternatives}")
        logprobs, top_tokens, top_logprobs = rows.logprobs, rows.top_tokens, rows.top_logprobs
    else:
        if any(len(top) != alternatives for _, top in rows):
            raise ValueError(f"The engine returned other than {alternatives} alternatives for a token")
        logprobs = [logprob for logprob, _ in rows]
        top_tokens = [int(token) for _, top in rows for token, _ in top]
        top_logprobs = [value for _, top in rows for _, value in top]
    return {"count": len(tokens), "alternatives": alternatives, "tokens": _encoded("i", [int(t) for t in tokens]),
            "logprobs": _encoded("d", _clamped(logprobs)), "top_tokens": _encoded("i", top_tokens),
            "top_logprobs": _encoded("d", _clamped(top_logprobs))}


def unpacked_scores(part):
    """(tokens, logprobs, top_tokens, top_logprobs, alternatives) of a packed part (packed_scores): array.array "i"
    and "d" in this machine's byte order. ValueError when it is malformed or its arrays do not add up."""
    if not isinstance(part, dict) or set(part) != {"count", "alternatives", "tokens", "logprobs", "top_tokens",
                                                   "top_logprobs"}:
        raise ValueError("A packed score part holds count, alternatives, tokens, logprobs, top_tokens and top_logprobs")
    count, alternatives = part["count"], part["alternatives"]
    if type(count) is not int or type(alternatives) is not int or count < 0 or not 0 <= alternatives <= TOP_LOGPROBS:
        raise ValueError("A packed score part's count and alternatives are integers")

    def numbers(typecode, key, size):
        if not isinstance(part[key], str):
            raise ValueError(f"A packed score part's {key} is base64 text")
        try:
            data = array.array(typecode, base64.b64decode(part[key], validate=True))
        except ValueError as exc:
            raise ValueError(f"A packed score part's {key} is not a {typecode} array: {exc}") from None
        if sys.byteorder != "little":
            data.byteswap()
        if len(data) != size:
            raise ValueError(f"A packed score part's {key} holds {len(data)} values, not {size}")
        return data
    return (numbers("i", "tokens", count), numbers("d", "logprobs", count),
            numbers("i", "top_tokens", count * alternatives), numbers("d", "top_logprobs", count * alternatives),
            alternatives)


def chat_prompt(request, *, images=False):
    """(messages, tools) for ModelText.encode_chat: tool history keeps its calls and results, and the request's
    tools are rendered unless tool_choice is "none"; ResidentChat parses the calls."""
    from crit_core.text import openai_messages
    tools = request.get("tools") if request.get("tool_choice", "auto") != "none" else None
    return openai_messages(request["messages"], images=images), (tools or None)


class RequestRejected(ValueError):
    """A request the engine refuses before generating anything: crit serve answers it with a 400 and keeps the engine
    loaded, where any other failure retires the host. `reason` names the rule; `details` hold its facts (JSON values)
    for the API's error."""
    reason = "rejected"

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details

    def record(self):
        return {"reason": self.reason, **self.details}


class ContextExceeded(RequestRejected):
    """The prompt plus the requested output exceeds the engine's context."""
    reason = "context_exceeded"

    def __init__(self, prompt_tokens, output_tokens, capacity):
        super().__init__(f"prompt of {prompt_tokens} tokens plus {output_tokens} output tokens exceeds the context of "
                         f"{capacity} tokens", prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                         capacity=capacity)


@contextmanager
def refused_text():
    """Invalid text-layer input (a crit_core.text.TextError: a message, template variable or token id) as a
    RequestRejected."""
    from crit_core.text import TextError
    try:
        yield
    except TextError as exc:
        raise RequestRejected(str(exc)) from None


def image_parts(request):
    """Whether a request's messages carry image parts."""
    return any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
               for m in request["messages"])


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
    Configuration "chat_template_kwargs" passes variables to the chat template (for example enable_thinking); a
    request's "template_kwargs" (chat-template-kwargs-v1) set those of TEMPLATE_KWARGS the templates read over them.
    Calls are parsed in the format the chat template teaches, or in `tool_format`, a crit_core.text.ToolFormat a
    package sets for a syntax crit does not register; replies carry them as tool_calls (chat-tools-v1). Reasoning is
    split from the reply in the markers the chat template uses (crit_core.text.reasoning) and returned as
    `reasoning`, streamed as {"reasoning": text}; `reasoning_format` sets a crit_core.text.ReasoningFormat for
    markers crit does not register, or False to leave reasoning in the reply. `logprobs` True: the decoder reports
    each token's log-probability and up to TOP_LOGPROBS alternatives (chat-logprobs-v1); the response then carries
    them for every generated token but stop tokens, reasoning and tool calls included. `grammars` True: the decoder
    keeps a reply to a grammar, so tool_choice "required" or a named function forces the calls (chat-tool-choice-v1)
    in a registered format; the model may reason first when the tokenizer spells the reasoning markers as text. A raw
    prompt (completion-v1) skips the template, tools and the reasoning split: the reply is all the text the model
    writes, and its log-probabilities keep the stop token's row."""
    sampling = ()
    logprobs = False
    scores = False
    plain = False
    grammars = False
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
            # The engine's own chat_template_kwargs, else the variables its run's recon froze (configuration["recon"]).
            template_variables = (self.configuration["chat_template_kwargs"] if "chat_template_kwargs" in self.configuration
                                  else (self.configuration.get("recon") or {}).get("template_variables"))
        self.template_variables = dict(template_variables or {})
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
        ranked, natural-log probabilities of the raw logits (crit::core::TokenLogprob). An engine with `plain` also takes
        plain=True when the request asks for plain decoding: one token per target pass, no drafting. An engine with
        `grammars` also takes grammar=artifact when the request forces tool calls: a crit-byte-grammar-v1 artifact the
        reply's bytes must keep to, applied as for response_format (a fresh GrammarMatcher over
        ByteGrammar::load(artifact) masking every position, a stop token only where the grammar is complete), so the
        reply is the calls (docs/SERVING.md). timing may add accepted (drafts accepted in each round), verify_ms and
        draft_ms (device time in verify passes and draft steps), which crit serve reports in its telemetry."""
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
        "telemetry"}, rows [token, logprob, [[token, logprob], ...top_logprobs]] as logprob_row makes them; for a
        request offering "format": "packed" (crit serve's), each part packed (packed_scores) and the record naming
        "format": "packed", which holds the same values without a JSON row per position."""
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
        record = {"schema_version": 1, "record_kind": "score_response", "request_id": request["request_id"],
                  "candidate_id": request["candidate_id"], "package_sha256": request["package_sha256"],
                  "telemetry": {key: timing.get(key) for key in ("prefill_ms", "decode_ms")}}
        prompt_tokens = ids[1:] if wanted else []
        if request.get("format") == "packed":
            return {**record, "format": "packed", "prompt": packed_scores(prompt_tokens, prompt_rows, top_k),
                    "continuation": packed_scores(continuation, continuation_rows, top_k)}
        return {**record, "prompt": [logprob_row(token, row, top_k) for token, row in zip(prompt_tokens, prompt_rows)],
                "continuation": [logprob_row(token, row, top_k) for token, row in zip(continuation, continuation_rows)]}

    def effective_configuration(self):
        return dict(self.configuration)

    def prompt_ids(self, request):
        messages, tools, variables = self._chat_template(request)
        if not request.get("continue_final_message"):
            return self.text.encode_chat(messages, tools=tools, **variables)
        from crit_core.text import TextError
        try:
            return self.text.encode_chat(messages, tools=tools, **variables)
        except TextError as exc:
            raise RequestRejected("The model's chat template cannot continue the final message: " + str(exc)) from None

    def _chat_template(self, request):
        """(messages, tools, variables) a chat request's prompt renders with: the configuration's template variables
        under the request's template_kwargs, and add_generation_prompt, or continue_final_message (chat-continue-v1),
        which leaves the final assistant message open."""
        messages, tools = chat_prompt(request, images=self.images)
        variables = dict(self.template_variables, **self._template_kwargs(request))
        if not request.get("continue_final_message"):
            return messages, tools, dict(variables, add_generation_prompt=True)
        # chat-continue-v1: the final assistant message is left open; calls are only parsed from a reply's start.
        if messages[-1]["role"] != "assistant":
            raise RequestRejected("continue_final_message needs a final assistant message")
        if request.get("tools"):
            raise RequestRejected("This engine cannot continue a final message in a request with tools")
        return messages, tools, dict(variables, continue_final_message=True)

    def _template_kwargs(self, request):
        """The request's template_kwargs: variables of TEMPLATE_KWARGS the chat templates read, each set to a value it
        takes. RequestRejected otherwise."""
        found = request.get("template_kwargs") or {}
        if found and getattr(self, "_read_variables", None) is None:
            self._read_variables = template_kwargs(self.text)
        for name, value in found.items():
            if name not in self._read_variables or not template_value(name, value):
                raise RequestRejected(f"The chat template does not take {name} = {value!r}")
        return found

    def completion_ids(self, request):
        """completion-v1: the token ids of a request's raw prompt, {"ids": [...]} as given or {"text": ...} encoded
        with the tokenizer's special tokens (the BOS its post-processor adds; one when the text already starts with
        it), control-token text read as those tokens as in a template. Ids outside the vocabulary, text the tokenizer
        cannot encode and a prompt of no tokens are refused (RequestRejected)."""
        from crit_core.text import TextError
        prompt, tokenizer = request["prompt"], self.text.tokenizer
        try:
            ids = (tokenizer.validate(prompt["ids"]) if "ids" in prompt
                   else tokenizer.encode(prompt["text"], add_special_tokens=True))
        except TextError as exc:
            raise RequestRejected("prompt: " + str(exc)) from None
        if "text" in prompt and tokenizer.bos_token_id is not None and ids[:2] == [tokenizer.bos_token_id] * 2:
            ids = ids[1:]
        if not ids:
            raise RequestRejected("The prompt encodes to no tokens")
        return ids

    @contextmanager
    def prepare_prompt(self, request, cancellation):
        """Override for images; yield PreparedPrompt and release private inputs in finally.

        Use prompt_ids() to render ordered image/text parts through the checkpoint's
        template. The engine supplies image decoding, vision arithmetic and layout.
        """
        if image_parts(request):
            raise ValueError("Image input requires the engine's prepare_prompt implementation")
        yield PreparedPrompt(self.prompt_ids(request))

    def tokenize(self, text, add_special_tokens):
        """chat-tokenize-v1: the ids of `text`, with the tokenizer's own special tokens (a BOS) when
        add_special_tokens. tokenize, render and detokenize run on their own thread beside generation and read only
        the tokenizer and the chat template; invalid input is a RequestRejected."""
        with refused_text():
            return self.text.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def render(self, request):
        """chat-tokenize-v1: (text, ids) of the prompt a chat request with these messages, tools, template_kwargs and
        continue_final_message prefills: the chat template's text and prompt_ids(). Image parts are refused, since the
        engine's vision processing places their positions (prepare_prompt)."""
        if image_parts(request):
            raise RequestRejected("Image positions come from the engine's vision processing: render text messages only")
        with refused_text():
            ids = self.prompt_ids(request)
            messages, tools, variables = self._chat_template(request)
            return self.text.render(messages, tools=tools, **variables), ids

    def detokenize(self, ids):
        """chat-tokenize-v1: the text of `ids`, special tokens included."""
        with refused_text():
            return self.text.tokenizer.decode(ids)

    def generate_prepared(self, prompt, max_new_tokens, sampling, on_tokens, cancelled, **options):
        """Override to consume prepared image inputs with the engine's token generator; `options` (logprobs, plain,
        grammar) go to generate_tokens. A raw prompt (completion-v1) comes as PreparedPrompt(ids), without inputs."""
        if prompt.inputs is not None:
            raise NotImplementedError("Prepared image inputs require generate_prepared")
        return self.generate_tokens(prompt.ids, max_new_tokens, sampling, on_tokens, cancelled, **options)

    def _run(self, request, cancellation, emit=None):
        if "prompt" in request:   # completion-v1: no template, so nothing to prepare
            return self._generate(request, cancellation, PreparedPrompt(self.completion_ids(request)), emit)
        with self.prepare_prompt(request, cancellation) as prompt:
            return self._generate(request, cancellation, prompt, emit)

    def _generate(self, request, cancellation, prompt, emit):
        from crit_core.text import MarkerDecoder, ReasoningSplitter, TokenText, ToolReply
        from crit_core.text.markers import prompt_ends_with

        ids, raw = prompt.ids, "prompt" in request
        prompt_tokens = len(ids) if prompt.token_count is None else prompt.token_count
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError("Prepared prompt token_count must be a positive integer")
        limits = request.get("limits", {})
        output_tokens = int(limits.get("max_output_tokens") or 1024)
        bounds = [c for c in (int(limits.get("kv_tokens") or 0), self.capacity or 0) if c]
        if limits.get("fit_output") and bounds and prompt_tokens < min(bounds):
            # chat-fit-output-v1: the output stops where the context ends; a prompt that fills it is still refused.
            output_tokens = min(output_tokens, min(bounds) - prompt_tokens)
        if bounds and prompt_tokens + output_tokens > min(bounds):
            raise ContextExceeded(prompt_tokens, output_tokens, min(bounds))
        sampling = dict(request.get("sampling") or {})
        unsupported = sorted(set(sampling) - set(self.sampling))
        if unsupported:
            raise ValueError("This engine does not apply sampling field(s): " + ", ".join(unsupported))
        alternatives = request.get("logprobs")
        if alternatives is not None and not self.logprobs:
            raise ValueError("This engine does not report log-probabilities")
        plain = request.get("decode") == "plain"
        if request.get("decode") not in (None, "plain"):
            raise ValueError("decode must be plain")
        if plain and not self.plain:
            raise ValueError("This engine does not decode plainly (plain)")
        if alternatives is not None and not (type(alternatives) is int and 0 <= alternatives <= TOP_LOGPROBS):
            raise ValueError(f"logprobs must be an integer in [0, {TOP_LOGPROBS}]")
        echo, prompt_rows = raw and bool(request.get("echo")), None
        if echo and alternatives is not None:
            # The prompt's rows come from the score path: one prefill of the prompt reading every position.
            if not self.scores:
                raise RequestRejected("This engine does not score a prompt (chat-score-v1): echo cannot carry logprobs")
            scored, _, _ = self.score_tokens(list(ids), [], alternatives, cancellation.is_set, True)
            if scored is None:
                cancellation.check()
                raise ValueError("The engine stopped scoring without being cancelled")
            if len(scored) != len(ids) - 1:
                raise ValueError(f"The engine scored {len(scored)} prompt positions, not {len(ids) - 1}")
            # Nothing precedes the first token, so it has no log-probability.
            prompt_rows = [[ids[0], None, []]] + [logprob_row(t, row, alternatives) for t, row in zip(ids[1:], scored)]
        echoed = None
        if echo:
            echoed = request["prompt"]["text"] if "text" in request["prompt"] else self.text.tokenizer.decode(ids)
        calls_format, reply, grammar = None, None, None
        choice = request.get("tool_choice", "auto")
        if request.get("tools"):   # calls only come when tools are offered
            if getattr(self, "calls_format", None) is None:   # also subclasses that set up self.text themselves
                self.calls_format = (self.tool_format if self.tool_format is not None
                                     else template_tool_format(self.text))
            calls_format = self.calls_format
            if calls_format is not None and choice != "none":
                if choice != "auto" and not (self.grammars and forces_calls(self.text, calls_format)):
                    raise RequestRejected("This engine parses tool calls from free generation: tool_choice must be "
                                          "auto or none")
                reply = ToolReply(calls_format, request, emit, tokenizer=self.text.tokenizer)
        thinking = None if raw else self._reasoning_format()
        kept = self._kept_markers(calls_format, thinking)
        dropped = ((set(self.text.tokenizer.special_ids) - kept) | self.stop_ids) if kept else self.stop_ids
        splitter = None
        boundaries = dict(reply.parser.token_ids) if reply else {}
        if thinking is not None:
            splitter = ReasoningSplitter(thinking, tokenizer=self.text.tokenizer)
            if prompt_ends_with(self.text.tokenizer, ids, thinking.opening, splitter.token_ids):
                splitter.state = "reasoning"
            boundaries.update(splitter.token_ids)
        if reply and choice != "auto":
            grammar = self._calls_grammar(request, calls_format, splitter)
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
                # A raw prompt's rows keep the stop token, as legacy completions list it: lm-eval reads one row
                # past the prompt's.
                rows.extend(logprob_row(token, row, alternatives) for token, row in zip(tokens, logprobs)
                            if raw or token not in self.stop_ids)
            for part in stream.push([t for t in tokens if t not in dropped]):
                route(part)

        options = {} if alternatives is None else {"logprobs": alternatives}
        if plain:
            options["plain"] = True
        if grammar is not None:
            options["grammar"] = grammar
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
            if isinstance(timing.get("accepted"), list) and len(timing["accepted"]) == telemetry["decode_rounds"]:
                telemetry["accepted"] = [int(a) for a in timing["accepted"]]
                if isinstance(timing.get("drafted"), list) and len(timing["drafted"]) == len(telemetry["accepted"]):
                    telemetry["drafted"] = [int(d) for d in timing["drafted"]]
        if n > 1:
            telemetry.update({key: float(timing[key]) for key in ("verify_ms", "draft_ms")
                              if isinstance(timing.get(key), (int, float)) and not isinstance(timing[key], bool)})
        logprobs = None
        if rows is not None:
            # Each token named once by its bytes, so the server can give clients token text and bytes.
            data = self.text.tokenizer.token_bytes()
            every = rows + (prompt_rows or [])
            ids = {row[0] for row in every} | {other for row in every for other, _ in row[2]}
            if ids and not 0 <= min(ids) <= max(ids) < len(data):
                raise ValueError("The engine reported log-probabilities for ids outside the tokenizer's vocabulary")
            logprobs = {"tokens": rows, "bytes": {str(token): data[token].hex() for token in sorted(ids)}}
            if prompt_rows is not None:
                logprobs["prompt"] = prompt_rows
        # A reply that ends on a stop token finishes as stop even when that token is the last one the cap allows.
        finish = "length" if n >= output_tokens and not (tokens and tokens[-1] in self.stop_ids) else "stop"
        return ("".join(pieces), telemetry, calls, "".join(reasoning), logprobs,
                output_tokens if limits.get("fit_output") else None, echoed, finish)

    def _calls_grammar(self, request, calls_format, splitter):
        """The grammar of a reply that makes the calls tool_choice forces (crit_core.text.tools.call_grammar), with room
        to reason first when a grammar can write the reasoning markers (text, not special tokens)."""
        from crit_core.constraints import UnsupportedConstraint
        from crit_core.text import TextError
        from crit_core.text.tools import call_grammar
        markers = (splitter.format.opening, splitter.format.closing) if splitter is not None else ()
        reasoning = markers if markers and not splitter.token_ids and "".join(markers).isascii() else None
        opened = splitter is not None and splitter.state == "reasoning"
        if opened and reasoning is None:
            raise RequestRejected("The reply starts inside reasoning whose markers a grammar cannot write: tool_choice "
                                  "must be auto or none")
        try:
            return call_grammar(calls_format, request["tools"], request["tool_choice"],
                                request.get("parallel_tool_calls", True), reasoning=reasoning, opened=opened)
        except (TextError, UnsupportedConstraint) as exc:
            raise RequestRejected("This engine cannot force the call: " + str(exc)) from None

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

    def _response(self, request, text, telemetry, calls=None, reasoning="", logprobs=None, max_output_tokens=None,
                  prompt=None, finish=None):
        response = {"schema_version": 1, "record_kind": "chat_response", "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"], "package_sha256": request["package_sha256"],
                    "assistant": text, "telemetry": telemetry}
        if calls:
            response["tool_calls"] = calls
        if reasoning:
            response["reasoning"] = reasoning
        if logprobs is not None:
            response["logprobs"] = logprobs
        if max_output_tokens is not None:
            response["max_output_tokens"] = max_output_tokens  # the cap a fitted request decoded under
        if prompt is not None:
            response["prompt"] = prompt   # the echoed prompt of a completion-v1 request
        if finish is not None:
            response["finish"] = finish   # "stop" or "length": how the engine ended the reply
        return response

    def generate(self, request, cancellation):
        return self._response(request, *self._run(request, cancellation))

    def generate_stream(self, request, cancellation, emit):
        return self._response(request, *self._run(request, cancellation, emit))

    def reset(self):
        pass

    def close(self):
        pass


__all__ = ["ResidentChat", "PreparedPrompt", "RequestRejected", "ContextExceeded", "SAMPLING", "LOGPROBS_CAPABILITY", "SCORE_CAPABILITY", "PLAIN_CAPABILITY",
           "COMPLETION_CAPABILITY", "CONTINUE_CAPABILITY", "FIT_OUTPUT_CAPABILITY", "TEMPLATE_KWARGS_CAPABILITY",
           "TOKENIZE_CAPABILITY", "TEMPLATE_KWARGS", "ENGINE_FACTS", "TOP_LOGPROBS", "capabilities", "engine_facts",
           "forces_calls", "chat_prompt", "kept_special_ids", "logprob_row", "clamp_logprob", "ScoreRows",
           "packed_scores", "unpacked_scores", "special_marker_ids",
           "sampling_capability", "template_kwarg_capability", "template_kwargs", "template_tool_format",
           "template_value"]
