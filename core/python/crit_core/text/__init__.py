"""Text layer: tokenizer, chat templates, generation config, streaming detokenization, stop strings and tool calls.

Names are re-exported lazily (PEP 562): `import crit_core.text` and `from crit_core.text import StopMatcher` load
only the standard library; `tokenizers` and `jinja2` are imported when a name that needs them is first used.
"""
import importlib

_EXPORTS = {
    "TextError": "errors", "ModelTextError": "errors", "TemplateError": "errors", "TokenError": "errors",
    "StopMatcher": "stops",
    "openai_messages": "messages",
    "ToolCallParser": "tools", "ToolFormat": "tools", "ToolReply": "tools", "tool_call_format": "tools",
    "templates_tool_format": "tools",
    "ReasoningFormat": "reasoning", "ReasoningSplitter": "reasoning", "template_reasoning_format": "reasoning",
    "GenerationConfig": "generation", "legacy_eos_ids": "generation",
    "AddedToken": "tokenizer", "Tokenizer": "tokenizer", "SPECIAL_TOKEN_NAMES": "tokenizer",
    "ChatTemplate": "template", "ChatTemplates": "template", "environment": "template",
    "template_sources": "template",
    "IncrementalDetokenizer": "detokenize",
    "TextDelta": "stream", "TextStream": "stream",
    "ModelText": "model", "TextIdentity": "model", "load_model_text": "model",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
