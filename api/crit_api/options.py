"""Validated, capability-gated request options shared by HTTP and resident serving."""
import json
import math
import re
from functools import lru_cache

from crit_core.constraints import ByteGrammar, GrammarState, UnsupportedConstraint, compile_language, matches
from crit_core.serving import (PLAIN_CAPABILITY, TEMPLATE_KWARGS, TEMPLATE_KWARGS_CAPABILITY, template_kwarg_capability,
                               template_value)

TOOLS_CAPABILITY = "chat-tools-v1"
STRICT_TOOLS_CAPABILITY = "chat-tools-strict-v1"
TOOL_STREAM_CAPABILITY = "chat-tool-stream-v1"
TOOL_CHOICE_CAPABILITY = "chat-tool-choice-v1"   # tool_choice required or a named function: the adapter forces a call
# Declared by packages sealed before crit 0.2.2 with the interim template tools. Their adapters are ResidentChats, which
# now parse calls themselves, so it reads as chat-tools-v1 (and chat-tool-stream-v1 when the package streams), with
# tool_choice auto or none.
TOOL_TEMPLATE_CAPABILITY = "chat-tool-template-v1"
STREAM_CAPABILITY = "chat-stream-v1"
MAX_CHAT_RECORD = 16 * 1024 * 1024  # bounded image data plus retained request metadata
MAX_TOOLS = 128  # functions one request may offer
SAMPLING = {"temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty",
            "frequency_penalty", "repetition_penalty", "logit_bias"}
# Samplers crit's sampling law does not have, by llama.cpp's and Ollama's names: (the value that turns it off, range).
# No engine applies them, so the off value is a no-op.
EXTRA_SAMPLING = {"typical_p": (1, 0, 1), "tfs_z": (1, 0, 1), "mirostat": (0, 0, 2),
                  "dry_multiplier": (0, 0, float("inf")), "xtc_probability": (0, 0, 1)}
SWITCHES = ("enable_thinking", "thinking")  # the TEMPLATE_KWARGS that turn a template's thinking on or off
# OpenAI's reasoning_effort as a template's reasoning_effort: none turns thinking off, minimal and xhigh take the
# lowest and highest it takes.
EFFORTS = {"none": None, "minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high"}


class RequestError(Exception):
    def __init__(self, message, status=400, param=None, code=None):
        super().__init__(message)
        self.status, self.param, self.code = status, param, code


def serving_capabilities(declared):
    """The capabilities crit serve offers for a package's declared ones: the earlier chat-tool-template-v1 adds
    chat-tools-v1, and chat-tool-stream-v1 when the package streams."""
    result = list(declared)
    if TOOL_TEMPLATE_CAPABILITY in result:
        for capability in (TOOLS_CAPABILITY, *((TOOL_STREAM_CAPABILITY,) if STREAM_CAPABILITY in result else ())):
            if capability not in result:
                result.append(capability)
    return result


def sampling_capability(name):
    return "chat-sampling-" + name.replace("_", "-") + "-v1"


def require(capabilities, capability, param):
    if capability not in capabilities:
        raise RequestError(f"This engine package does not declare {capability}", param=param,
                           code="unsupported_parameter")


def parameter_name(value):
    """Whether crit serve may ignore a parameter by this name: X-Crit-Ignored-Parameters and the server log carry it as
    sent, so it is 1 to 64 letters, digits, _, . or -."""
    return re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value)


def unknown_parameters(body, known, lenient, prefix=""):
    """The request's keys outside `known`, sorted and named with `prefix` (stream_options. for that object's keys).
    Strict crit serve refuses the first; --lenient refuses only a name it cannot list (parameter_name)."""
    unknown = sorted(k for k in body if k not in known)
    refused = [k for k in unknown if not (lenient and parameter_name(k))]
    if refused:
        raise RequestError("Unknown request parameter: " + prefix + refused[0], param=prefix + refused[0],
                           code="unsupported_parameter")
    return [prefix + k for k in unknown]


def sampling_options(body, capabilities, ignored=None):
    """The request's sampling fields the package applies, each checked against its range. An extra sampler
    (EXTRA_SAMPLING) at its off value is a no-op. Any other field the package does not apply is refused, or, given an
    `ignored` list (crit serve --lenient), named there and left out."""
    result = {}
    bounds = {"temperature": (0, 2), "top_p": (0, 1), "min_p": (0, 1),
              "frequency_penalty": (-2, 2), "presence_penalty": (-2, 2),
              "repetition_penalty": (0, float("inf")), "top_k": (0, 2**32 - 1), "seed": (0, 2**64 - 1)}
    for key in sorted(SAMPLING & body.keys()):
        value = body[key]
        if value is None:
            continue
        if key == "logit_bias":
            if not isinstance(value, dict) or len(value) > 1024:
                raise RequestError("logit_bias must be an object with at most 1024 entries", param=key)
            for token, bias in value.items():
                if (not isinstance(token, str) or not re.fullmatch(r"0|[1-9][0-9]{0,9}", token)
                        or int(token) > 2**31 - 1 or type(bias) not in (int, float)
                        or not -100 <= bias <= 100):
                    raise RequestError("logit_bias needs token-id keys and finite biases in [-100, 100]", param=key)
        else:
            low, high = bounds[key]
            if (type(value) not in ((int,) if key in {"seed", "top_k"} else (int, float))
                    or not low <= value <= high or (key in {"top_p", "repetition_penalty"} and value == 0)
                    or (type(value) is float and not math.isfinite(value))):
                raise RequestError(f"Invalid {key}: expected a finite number in [{low}, {high}]"
                                   + (" (strictly positive)" if key in {"top_p", "repetition_penalty"} else ""), param=key)
        if ignored is not None and sampling_capability(key) not in capabilities:
            ignored.append(key)
            continue
        require(capabilities, sampling_capability(key), key)
        result[key] = value
    for key in sorted(EXTRA_SAMPLING.keys() & body.keys()):
        value, (off, low, high) = body[key], EXTRA_SAMPLING[key]
        if value is None:
            continue
        if (type(value) not in ((int,) if key == "mirostat" else (int, float)) or not low <= value <= high
                or (type(value) is float and not math.isfinite(value))):
            raise RequestError(f"Invalid {key}: expected " + ("0, 1 or 2" if key == "mirostat"
                                                               else f"a finite number in [{low}, {high}]"), param=key)
        if value == off:
            continue
        if ignored is None:
            raise RequestError(f"This engine does not apply {key}; omit it or send {off}, which turns it off",
                               param=key, code="unsupported_parameter")
        ignored.append(key)
    return result


def crit_options(value, capabilities):
    """crit's own request extension, the body's "crit" object: {"decode": "plain"} decodes the reply one token per
    target pass, without drafting, on an engine that declares chat-plain-v1; {"timeout": seconds} gives the request
    less time than the server's --timeout. Returns {"decode", "timeout"}, None where unset."""
    if value is None:
        return {"decode": None, "timeout": None}
    if not isinstance(value, dict) or set(value) - {"decode", "timeout"}:
        raise RequestError("crit takes an object with decode and timeout", param="crit")
    decode, timeout = value.get("decode"), value.get("timeout")
    if decode is not None:
        if decode != "plain":
            raise RequestError("crit.decode must be plain", param="crit.decode")
        require(capabilities, PLAIN_CAPABILITY, "crit.decode")
    if timeout is not None and (type(timeout) not in (int, float) or timeout <= 0
                                or (type(timeout) is float and not math.isfinite(timeout))):
        raise RequestError("crit.timeout must be a positive number of seconds", param="crit.timeout")
    return {"decode": decode, "timeout": timeout}


def template_variables(capabilities):
    """The TEMPLATE_KWARGS a request may set: those the package declares, with chat-template-kwargs-v1."""
    if TEMPLATE_KWARGS_CAPABILITY not in capabilities:
        return []
    return [name for name in TEMPLATE_KWARGS if template_kwarg_capability(name) in capabilities]


def thinking_kwargs(capabilities, enabled=None, effort=None, param="reasoning_effort", ignored=None):
    """The chat-template variables that turn thinking on or off (`enabled`) and set its effort (`effort`: low, medium
    or high, which turns it on), through those the package declares: a switch (enable_thinking or thinking) and
    reasoning_effort. A template with reasoning_effort and no switch always thinks: turning it off is refused, or,
    given an `ignored` list (crit serve --lenient), `param` is named there and no variable is set."""
    require(capabilities, TEMPLATE_KWARGS_CAPABILITY, param)
    declared = template_variables(capabilities)
    switches = [name for name in SWITCHES if name in declared]
    if not declared or (enabled is False and not switches):
        if ignored is not None:
            ignored.append(param)
            return {}
        raise RequestError("This engine's chat template cannot turn thinking " + ("off" if declared else "on or off"),
                           param=param, code="unsupported_parameter")
    result = dict.fromkeys(switches, enabled is not False)
    if enabled is not False and effort is not None and "reasoning_effort" in declared:
        result["reasoning_effort"] = effort
    return result


def template_options(body, capabilities, ignored=None):
    """The chat-template variables a request sets (chat-template-kwargs-v1): its chat_template_kwargs, each one of the
    package's template_variables with a value TEMPLATE_KWARGS gives it, and its reasoning_effort (EFFORTS) through
    thinking_kwargs; the two must agree on a variable both set. Given an `ignored` list (crit serve --lenient), what
    the package cannot apply is named there and left out instead of refused: both fields, whatever they hold, without
    chat-template-kwargs-v1; otherwise a variable crit does not set (by a name parameter_name takes) or the package
    does not declare, once its value is checked, and a reasoning_effort its template cannot take."""
    value = body.get("chat_template_kwargs")
    if ignored is not None and TEMPLATE_KWARGS_CAPABILITY not in capabilities:
        ignored += ["chat_template_kwargs"] if value is not None and value != {} else []
        ignored += ["reasoning_effort"] if body.get("reasoning_effort") is not None else []
        return {}
    if value is not None and not isinstance(value, dict):
        raise RequestError("chat_template_kwargs must be an object", param="chat_template_kwargs")
    result = {}
    for name, item in (value or {}).items():
        require(capabilities, TEMPLATE_KWARGS_CAPABILITY, "chat_template_kwargs")
        if name not in TEMPLATE_KWARGS:
            if ignored is not None and parameter_name(name):
                ignored.append("chat_template_kwargs." + name)
                continue
            raise RequestError(f"chat_template_kwargs.{name} is not a template variable crit serve sets; it sets "
                               + ", ".join(TEMPLATE_KWARGS), param="chat_template_kwargs", code="unsupported_parameter")
        declared = template_kwarg_capability(name) in capabilities
        if not declared and ignored is None:
            require(capabilities, template_kwarg_capability(name), "chat_template_kwargs")
        if not template_value(name, item):
            raise RequestError(f"chat_template_kwargs.{name} must be one of "
                               + ", ".join(map(json.dumps, TEMPLATE_KWARGS[name])), param="chat_template_kwargs")
        if not declared:
            ignored.append("chat_template_kwargs." + name)
            continue
        result[name] = item
    effort = body.get("reasoning_effort")
    if effort is not None:
        if not isinstance(effort, str) or effort not in EFFORTS:
            raise RequestError("reasoning_effort must be " + ", ".join(EFFORTS), param="reasoning_effort")
        mapped = thinking_kwargs(capabilities, effort != "none", EFFORTS[effort], ignored=ignored)
        if any(result.get(name, item) != item for name, item in mapped.items()):
            raise RequestError("reasoning_effort and chat_template_kwargs set a template variable differently",
                               param="reasoning_effort")
        result = {**mapped, **result}
    return result


def function_name(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)


@lru_cache(maxsize=128)
def tool_grammar(source):
    return compile_language("schema", source)


def tool_options(body, capabilities):
    tools = body.get("tools")
    tools = [] if tools is None else tools
    if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
        raise RequestError(f"tools must be a list of at most {MAX_TOOLS} functions", param="tools")
    names, grammars = set(), {}
    for tool in tools:
        spec = tool.get("function") if isinstance(tool, dict) else None
        if (not isinstance(spec, dict) or tool.get("type") != "function"
                or not function_name(spec.get("name")) or spec["name"] in names):
            raise RequestError("tools must contain functions with unique valid names", param="tools")
        names.add(spec["name"])
        if "parameters" in spec and not isinstance(spec["parameters"], dict):
            raise RequestError("Function parameters must be a JSON schema object", param="tools")
        if "description" in spec and not isinstance(spec["description"], str):
            raise RequestError("Function description must be a string", param="tools")
        if spec.get("strict") is not None and type(spec["strict"]) is not bool:
            raise RequestError("Function strict must be a boolean", param="tools")
        if spec.get("strict"):
            require(capabilities, STRICT_TOOLS_CAPABILITY, "tools")
            schema = spec.get("parameters")
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise RequestError("Strict function parameters must be an object schema", param="tools")
            try:
                grammars[spec["name"]] = tool_grammar(json.dumps(schema, ensure_ascii=False, allow_nan=False))
            except (UnsupportedConstraint, ValueError) as exc:
                raise RequestError("Unsupported strict tool schema: " + str(exc), param="tools",
                                   code="unsupported_schema") from None
            if sum(map(len, grammars.values())) > 512 * 1024:
                raise RequestError("Compiled tool schemas exceed 512 KiB", param="tools", code="unsupported_schema")
    choice = body.get("tool_choice")
    if choice is None:
        choice = "auto" if tools else "none"
    if isinstance(choice, dict):
        spec = choice.get("function")
        if (choice.get("type") != "function" or not isinstance(spec, dict)
                or not function_name(spec.get("name")) or spec["name"] not in names):
            raise RequestError("tool_choice must name a supplied function", param="tool_choice")
    elif not isinstance(choice, str) or choice not in {"auto", "none", "required"}:
        raise RequestError("tool_choice must be auto, none, required or a named function", param="tool_choice")
    if not tools and choice != "none":
        raise RequestError("tool_choice requires tools", param="tool_choice")
    parallel = body.get("parallel_tool_calls", True)
    if type(parallel) is not bool:
        raise RequestError("parallel_tool_calls must be a boolean", param="parallel_tool_calls")
    if tools:
        require(capabilities, TOOLS_CAPABILITY, "tools")
        if choice not in ("auto", "none"):
            require(capabilities, TOOL_CHOICE_CAPABILITY, "tool_choice")
    options = {"tools": tools, "tool_choice": choice, "parallel_tool_calls": parallel} if tools else {}
    if grammars:
        options["tool_grammars"] = grammars
    return options


def _reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def tool_calls(value, truncated=False):
    """Validate the OpenAI wire representation; never interpret or execute a call. With `truncated`, a reply cut by
    its output limit, the last call's arguments may be the start of a JSON object, as the model wrote them."""
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise ValueError("tool_calls must be a non-empty list of at most 128 calls")
    ids = set()
    for index, call in enumerate(value):
        if not isinstance(call, dict):
            raise ValueError("Each tool call must be an object")
        ident, function = call.get("id"), call.get("function")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", ident) or ident in ids
                or call.get("type") != "function" or not isinstance(function, dict)
                or not function_name(function.get("name")) or not isinstance(function.get("arguments"), str)):
            raise ValueError("Tool calls need unique ids, type function, a name and JSON-string arguments")
        ids.add(ident)
        if truncated and index == len(value) - 1 and not complete_arguments(call):
            if not function["arguments"].startswith("{"):
                raise ValueError("A cut tool call's arguments must start a JSON object")
            continue
        arguments = json.loads(function["arguments"], parse_constant=_reject_constant)
        if not isinstance(arguments, dict):
            raise ValueError("Tool call arguments must encode a JSON object")
        json.dumps(arguments, allow_nan=False)  # Also reject overflowing exponents such as 1e400.
    return value


def complete_arguments(call):
    """Whether a call's arguments are a whole JSON value, not the start of one a cut reply left."""
    try:
        json.loads(call["function"]["arguments"], parse_constant=_reject_constant)
    except ValueError:
        return False
    return True


def validate_tool_reply(calls, options, truncated=False):
    """A reply's calls against the request's tool policy; `truncated`: the reply reached its output limit, so its last
    call may be cut (tool_calls), or a forced call not yet begun."""
    calls = [] if calls is None or calls == [] else tool_calls(calls, truncated)
    choice = options.get("tool_choice", "none")
    names = {t["function"]["name"] for t in options.get("tools", [])}
    if calls and (choice == "none" or any(c["function"]["name"] not in names for c in calls)):
        raise ValueError("Engine returned an unrequested tool call")
    if (choice == "required" or isinstance(choice, dict)) and not calls and not truncated:
        raise ValueError("Engine did not return the required tool call")
    if isinstance(choice, dict) and any(c["function"]["name"] != choice["function"]["name"] for c in calls):
        raise ValueError("Engine did not call the selected function")
    if not options.get("parallel_tool_calls", True) and len(calls) > 1:
        raise ValueError("Engine returned parallel tool calls when disabled")
    prior_ids = {call["id"] for message in options.get("messages", []) for call in message.get("tool_calls", [])}
    if any(call["id"] in prior_ids for call in calls):
        raise ValueError("Engine reused a tool call id from the conversation")
    for call in calls:
        grammar = options.get("tool_grammars", {}).get(call["function"]["name"])
        arguments = call["function"]["arguments"].encode("utf-8")
        if grammar and not (matches(grammar, arguments) if complete_arguments(call)
                            else GrammarState(ByteGrammar.load(grammar)).advance(arguments) is not None):
            raise ValueError("Tool arguments do not match the strict schema: " + call["function"]["name"])
    return calls
