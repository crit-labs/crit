"""Append-only tool-call assembly and strict argument-prefix checks."""
import copy
import re

from crit_core.constraints import ByteGrammar, GrammarState

from .content import content_size, content_text, tool_size
from .options import function_name


class ToolCallStream:
    def __init__(self, request):
        # Adapters receive a mutable request. Their local edits must not change the policy already admitted.
        self.request = copy.deepcopy(request)
        self.calls, self.states = [], []

    @property
    def size(self):
        return tool_size(self.calls)

    def update(self, deltas):
        """Apply a nonempty list of indexed OpenAI deltas atomically, returning normalized deltas."""
        if not isinstance(deltas, list) or not deltas or len(deltas) > 128:
            raise ValueError("Tool stream needs 1–128 deltas")
        calls, states, result = copy.deepcopy(self.calls), list(self.states), []
        names = {t["function"]["name"] for t in self.request.get("tools", [])}
        prior = {c["id"] for m in self.request.get("messages", []) for c in m.get("tool_calls", [])}
        choice = self.request.get("tool_choice", "none")
        for delta in deltas:
            if not isinstance(delta, dict) or set(delta) - {"index", "id", "type", "function"}:
                raise ValueError("Invalid tool-call delta fields")
            index, function = delta.get("index"), delta.get("function")
            if type(index) is not int or not 0 <= index <= len(calls) or index >= 128:
                raise ValueError("Tool-call indices must be contiguous and between 0 and 127")
            if not isinstance(function, dict) or set(function) - {"name", "arguments"}:
                raise ValueError("Invalid streamed function")
            arguments = content_text(function.get("arguments", ""), "Tool arguments", allow_empty=True)
            if index == len(calls):
                ident, name = delta.get("id"), function.get("name")
                if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", ident)
                        or ident in prior or any(c["id"] == ident for c in calls)
                        or delta.get("type") != "function" or not function_name(name)):
                    raise ValueError("First tool delta needs a new id, type function and complete function name")
                if choice == "none" or name not in names or (isinstance(choice, dict) and name != choice["function"]["name"]):
                    raise ValueError("Streamed tool call violates tool_choice or names an unknown function")
                if calls and not self.request.get("parallel_tool_calls", True):
                    raise ValueError("Parallel tool calls are disabled")
                calls.append({"id": ident, "type": "function", "function": {"name": name, "arguments": ""}})
                grammar = self.request.get("tool_grammars", {}).get(name)
                states.append(GrammarState(ByteGrammar.load(grammar)) if grammar else None)
                out = {"index": index, "id": ident, "type": "function", "function": {"name": name, "arguments": arguments}}
            else:
                if set(delta) != {"index", "function"} or set(function) != {"arguments"} or not arguments:
                    raise ValueError("Later tool deltas may only append nonempty arguments")
                out = {"index": index, "function": {"arguments": arguments}}
            calls[index]["function"]["arguments"] += arguments
            content_size("", tool_size(calls))
            if states[index] is not None:
                states[index] = states[index].advance(arguments.encode("utf-8"))
                if states[index] is None:
                    raise ValueError("Streamed arguments violate the strict tool schema")
            result.append(out)
        self.calls, self.states = calls, states
        return result

    def snapshot(self, calls):
        """Validate a coalesced snapshot and return only unseen argument suffixes."""
        if not isinstance(calls, list) or not len(self.calls) <= len(calls) <= 128:
            raise ValueError("Streamed tool calls moved backwards")
        deltas = []
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
                raise ValueError("Invalid tool snapshot")
            function = call["function"]
            if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
                raise ValueError("Invalid tool snapshot function")
            if index >= len(self.calls):
                deltas.append(dict(call, index=index))
                continue
            old = self.calls[index]
            if (call["id"] != old["id"] or call["type"] != old["type"] or function["name"] != old["function"]["name"]
                    or not isinstance(function["arguments"], str)
                    or not function["arguments"].startswith(old["function"]["arguments"])):
                raise ValueError("Streamed tool call was rewritten")
            suffix = function["arguments"][len(old["function"]["arguments"]):]
            if suffix:
                deltas.append({"index": index, "function": {"arguments": suffix}})
        return self.update(deltas) if deltas else []
