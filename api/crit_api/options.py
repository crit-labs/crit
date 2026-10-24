"""Validated, capability-gated request options shared by HTTP and resident serving."""
import json
import math
import re
from functools import lru_cache

from crit_core.constraints import ByteGrammar, GrammarState, UnsupportedConstraint, compile_language, matches
from crit_core.serving import PLAIN_CAPABILITY

TOOLS_CAPABILITY = "chat-tools-v1"
STRICT_TOOLS_CAPABILITY = "chat-tools-strict-v1"
TOOL_STREAM_CAPABILITY = "chat-tool-stream-v1"
# Declared by packages sealed before crit 0.2.2 with the interim template tools. Their adapters are ResidentChats, which
# now parse calls themselves, so it reads as chat-tools-v1 (and chat-tool-stream-v1 when the package streams), with
# tool_choice auto or none.
TOOL_TEMPLATE_CAPABILITY = "chat-tool-template-v1"
STREAM_CAPABILITY = "chat-stream-v1"
MAX_CHAT_RECORD = 16 * 1024 * 1024  # bounded image data plus retained request metadata
SAMPLING = {"temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty",
            "frequency_penalty", "repetition_penalty", "logit_bias"}


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


def sampling_options(body, capabilities):
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
        require(capabilities, sampling_capability(key), key)
        result[key] = value
    return result


def crit_options(value, capabilities):
    """crit's own request extension, the body's "crit" object: {"decode": "plain"} decodes the reply one token per
    target pass, without drafting, on an engine that declares chat-plain-v1. Returns the decode mode or None."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"decode"}:
        raise RequestError("crit takes an object with decode", param="crit")
    decode = value.get("decode")
    if decode is None:
        return None
    if decode != "plain":
        raise RequestError("crit.decode must be plain", param="crit.decode")
    require(capabilities, PLAIN_CAPABILITY, "crit.decode")
    return decode


def function_name(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)


@lru_cache(maxsize=128)
def tool_grammar(source):
    return compile_language("schema", source)


def tool_options(body, capabilities):
    tools = body.get("tools")
    tools = [] if tools is None else tools
    if not isinstance(tools, list) or len(tools) > 128:
        raise RequestError("tools must be a list of at most 128 functions", param="tools")
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
        if TOOL_TEMPLATE_CAPABILITY in capabilities and choice not in ("auto", "none"):
            raise RequestError("This engine parses tool calls from free generation; tool_choice must be auto or none",
                               param="tool_choice", code="unsupported_parameter")
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
    call may be cut (tool_calls)."""
    calls = [] if calls is None or calls == [] else tool_calls(calls, truncated)
    choice = options.get("tool_choice", "none")
    names = {t["function"]["name"] for t in options.get("tools", [])}
    if calls and (choice == "none" or any(c["function"]["name"] not in names for c in calls)):
        raise ValueError("Engine returned an unrequested tool call")
    if (choice == "required" or isinstance(choice, dict)) and not calls:
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
