"""Chat templates rendered as Hugging Face transformers 5.19 renders them, in an immutable jinja2 sandbox.

The environment matches transformers: trim_blocks and lstrip_blocks, the loopcontrols extension, a
{% generation %} tag, `tojson` as json.dumps (ensure_ascii=False, insertion order kept, indent and separators
passed through), `raise_exception` and `strftime_now`. The clock behind strftime_now is injectable; with no clock
the global is undefined, so templates fall back to their own date. Template discovery and selection follow
transformers too (see template_sources and ChatTemplates.select). For a model family whose checkpoints carry no chat
template, crit ships one (SHIPPED_TEMPLATES, by config.json model_type; see chat_template_sources).
"""
from collections.abc import Mapping
import contextvars
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import jinja2
import jinja2.ext
from jinja2 import filters as jinja_filters
import jinja2.nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .errors import (MAX_CONFIG_BYTES, ModelTextError, TemplateError, TextError, list_names, model_files,
                     read_json_object, read_text)

CONTINUE_TAG = "CONTINUE_FINAL_MESSAGE_TAG "
MAX_RENDER_CHARS = 1 << 26        # rendered text, and single results of the guarded operations, above this are refused
TEMPLATE_FILE = "chat_template.jinja"
TEMPLATE_DIR = "additional_chat_templates/"
LEGACY_TEMPLATE_FILE = "chat_template.json"
SHIPPED_DIR = Path(__file__).resolve().parent / "templates"
# The chat templates crit ships, by config.json model_type, for families whose checkpoints carry none: DeepSeek V4
# publishes a Python encoder (encoding_dsv4.py) instead, which templates/deepseek-v4.jinja ports.
SHIPPED_TEMPLATES = MappingProxyType({"deepseek_v4": "deepseek-v4.jinja"})

_SPANS = contextvars.ContextVar("crit_core_text_generation_spans", default=None)


class _Spans:
    __slots__ = ("length", "spans")

    def __init__(self):
        self.length = 0
        self.spans = []


class GenerationTag(jinja2.ext.Extension):
    """{% generation %}...{% endgeneration %}: renders its body and, while a render records assistant spans,
    notes the body's (start, end) character offsets in the output (transformers' AssistantTracker)."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_generation"), [], [], body).set_lineno(lineno)

    def _generation(self, caller):
        text = caller()
        spans = _SPANS.get()
        if spans is not None:
            spans.spans.append((spans.length, spans.length + len(text)))
        return text


class _RaisedError(jinja2.exceptions.TemplateError):
    """raise_exception() called by the template."""


def _raise_exception(message):
    raise _RaisedError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _limit(size, what):
    if size > MAX_RENDER_CHARS:
        raise jinja2.exceptions.SecurityError(f"{what} exceeds the render limit")


def _center(value, width=80):
    if isinstance(width, int):
        _limit(width, "center")
    return jinja_filters.do_center(value, width)


def _indent(s, width=4, first=False, blank=False):
    text = str(s)
    _limit(len(text) + (text.count("\n") + 1) * (width if isinstance(width, int) else len(str(width))), "indent")
    return jinja_filters.do_indent(s, width, first, blank)


@jinja2.pass_eval_context
def _join(eval_ctx, value, d="", attribute=None):
    if attribute is not None:
        value = map(jinja_filters.make_attrgetter(eval_ctx.environment, attribute), value)
    items = list(value)
    _limit(sum(len(str(item)) for item in items) + len(str(d)) * max(len(items) - 1, 0), "join")
    return jinja_filters.do_join(eval_ctx, items, d)


@jinja2.pass_eval_context
def _replace(eval_ctx, s, old, new, count=None):
    text, old_text = str(s), str(old)
    found = text.count(old_text) if old_text else len(text) + 1
    if isinstance(count, int) and count >= 0:
        found = min(found, count)
    _limit(len(text) + found * max(len(str(new)) - len(old_text), 0), "replace")
    return jinja_filters.do_replace(eval_ctx, s, old, new, count)


class _Sandbox(ImmutableSandboxedEnvironment):
    """Also refuses `*`, `**` and `+` results, and center, indent, join and replace results, that would exceed the
    render limit before the output cap could apply."""
    intercepted_binops = frozenset(("*", "**", "+"))

    def call_binop(self, context, operator, left, right):
        if operator == "*":
            sequence, count = (right, left) if isinstance(left, int) else (left, right)
            if isinstance(sequence, (str, list, tuple)) and isinstance(count, int):
                _limit(len(sequence) * count, "sequence repetition")
        elif (operator == "**" and isinstance(left, int) and isinstance(right, int) and right > 0
              and abs(left) > 1 and abs(left).bit_length() * right > 1 << 16):
            raise jinja2.exceptions.SecurityError("integer power exceeds the render limit")
        elif operator == "+" and isinstance(left, (str, list, tuple)) and isinstance(right, (str, list, tuple)):
            _limit(len(left) + len(right), "concatenation")
        return super().call_binop(context, operator, left, right)


def environment(clock=datetime.now):
    """A new sandboxed environment configured like transformers' chat-template environment. `clock` returns the
    datetime behind strftime_now (local time by default); None leaves strftime_now undefined."""
    if clock is not None and not callable(clock):
        raise TextError("clock must be callable or None")
    env = _Sandbox(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationTag, jinja2.ext.loopcontrols])
    env.filters.update(tojson=_tojson, center=_center, indent=_indent, join=_join, replace=_replace)
    env.globals["raise_exception"] = _raise_exception
    if clock is not None:
        def strftime_now(format):
            return clock().strftime(format)
        env.globals["strftime_now"] = strftime_now
    return env


def _template_line(traceback):
    line = None
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename == "<template>":
            line = traceback.tb_lineno
        traceback = traceback.tb_next
    return line


def _json_objects(value, name):
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise TextError(f"{name} must be a list of JSON objects")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TextError(f"{name}[{index}] must be a JSON object")
    return list(value)


class ChatTemplate:
    """One compiled chat template. Compilation happens here, so syntax errors surface at load as TemplateError with
    the template name and line. `special_tokens` (bos_token, eos_token, ...) become template variables that the
    caller's variables override, as in transformers' apply_chat_template."""

    def __init__(self, source, *, name="default", special_tokens=None, clock=datetime.now):
        if not isinstance(source, str):
            raise TextError("chat template source must be a string")
        if not isinstance(name, str) or not name:
            raise TextError("chat template name must be a non-empty string")
        special = dict(special_tokens or {})
        for key, value in special.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TextError("special tokens must map names to strings")
        self.name = name
        self.source = source
        self.sha256 = hashlib.sha256(source.encode("utf-8", "surrogatepass")).hexdigest()
        self.special_tokens = MappingProxyType(special)
        self.clock = clock
        try:
            self._template = environment(clock).from_string(source)
        except jinja2.exceptions.TemplateSyntaxError as error:
            raise TemplateError(f"chat template {name!r} has a syntax error at line {error.lineno}: {error.message}",
                                template_name=name, lineno=error.lineno) from None
        except jinja2.exceptions.TemplateError as error:
            raise TemplateError(f"chat template {name!r} does not compile: {error}", template_name=name) from None
        except RecursionError:
            raise TemplateError(f"chat template {name!r} is nested too deeply", template_name=name) from None

    def __repr__(self):
        return f"ChatTemplate(name={self.name!r}, sha256={self.sha256[:12]!r})"

    def _fail(self, message, lineno=None):
        return TemplateError(f"chat template {self.name!r}: {message}", template_name=self.name, lineno=lineno)

    def _continue(self, messages, field):
        """transformers' continue_final_message: tag the final message's text so the render can be cut there."""
        if field not in self.source:
            raise self._fail(f'continue_final_message names "{field}", which the template never mentions')
        last = messages[-1]
        value = last.get(field)
        if value is None:
            raise self._fail(f'continue_final_message is set but the final message has no "{field}" to continue')
        last = dict(last)
        if isinstance(value, (list, tuple)):
            parts = list(value)
            for index in reversed(range(len(parts))):
                part = parts[index]
                if isinstance(part, Mapping) and "text" in part:
                    final = part["text"]
                    if not isinstance(final, str):
                        raise TextError(f'the final message\'s last text part in "{field}" must be a string')
                    parts[index] = {**part, "text": final + CONTINUE_TAG}
                    break
            else:
                raise self._fail("continue_final_message is set but the final message has no text part to continue")
            last[field] = parts
        elif isinstance(value, str):
            final = value
            last[field] = value + CONTINUE_TAG
        else:
            raise TextError(f'the final message\'s "{field}" must be a string or a list of parts')
        return messages[:-1] + [last], final

    def render(self, messages, *, tools=None, documents=None, add_generation_prompt=False,
               continue_final_message=False, assistant_spans=None, **variables):
        """Render a conversation.

        `messages` is a non-empty list of mappings; `tools` and `documents` are lists of JSON objects or None.
        continue_final_message (True, or the name of the field to continue, "content" by default) renders the
        final message open-ended so generation continues it; it cannot be combined with add_generation_prompt
        or assistant_spans. When `assistant_spans` is a list it is filled with the (start, end) character spans of
        {% generation %} blocks. Extra keyword arguments (enable_thinking, date_string, ...) become template
        variables and override the special-token strings. Every template failure raises TemplateError.
        """
        if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, (list, tuple)):
            raise TextError("messages must be a list of message mappings")
        if not messages:
            raise self._fail("cannot render an empty conversation")
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TextError(f"messages[{index}] must be a mapping")
        messages = list(messages)
        if not isinstance(add_generation_prompt, bool):
            raise TextError("add_generation_prompt must be True or False")
        if not isinstance(continue_final_message, (bool, str)) or continue_final_message == "":
            raise TextError("continue_final_message must be True, False or a field name")
        if assistant_spans is not None and not isinstance(assistant_spans, list):
            raise TextError("assistant_spans must be a list or None")
        if continue_final_message and add_generation_prompt:
            raise self._fail("continue_final_message and add_generation_prompt cannot be combined")
        if continue_final_message and assistant_spans is not None:
            raise self._fail("continue_final_message and assistant_spans cannot be combined")
        tools = _json_objects(tools, "tools")
        documents = _json_objects(documents, "documents")
        final = None
        if continue_final_message:
            field = "content" if continue_final_message is True else continue_final_message
            messages, final = self._continue(messages, field)
        context = {**self.special_tokens, **variables, "messages": messages, "tools": tools,
                   "documents": documents, "add_generation_prompt": add_generation_prompt}
        spans = _Spans() if assistant_spans is not None else None
        reset = _SPANS.set(spans)
        try:
            blocks = []
            length = 0
            for block in self._template.generate(context):
                blocks.append(block)
                length += len(block)
                if spans is not None:
                    spans.length = length
                if length > MAX_RENDER_CHARS:
                    raise self._fail(f"rendered text exceeds {MAX_RENDER_CHARS} characters")
            rendered = "".join(blocks)
        except TemplateError:
            raise
        except Exception as error:
            lineno = _template_line(error.__traceback__)
            where = f" at line {lineno}" if lineno else ""
            if isinstance(error, MemoryError):
                message = f"ran out of memory{where}"
            elif isinstance(error, _RaisedError):
                message = f"raised an exception{where}: {error.message}"
            elif isinstance(error, jinja2.exceptions.SecurityError):
                message = f"was stopped by the sandbox{where}: {error}"
            elif isinstance(error, jinja2.exceptions.TemplateError):
                message = f"failed{where}: {error}"
            else:
                message = f"failed{where}: {type(error).__name__}: {error}"
            raise self._fail(message, lineno) from None
        finally:
            _SPANS.reset(reset)
        if final is not None:
            tag = CONTINUE_TAG.strip()
            if final.strip() not in rendered or tag not in rendered:
                raise self._fail("continue_final_message is set but the final message does not appear in the "
                                 "rendered chat; the template changes or drops it")
            location = rendered.rindex(tag)
            if rendered[location:location + len(CONTINUE_TAG)] == CONTINUE_TAG:
                rendered = rendered[:location]
            else:
                rendered = rendered[:location].rstrip()  # the template trimmed the space after the tag
        if assistant_spans is not None:
            assistant_spans[:] = spans.spans
        return rendered


def _config_templates(value):
    if value is None:
        return {}
    if isinstance(value, str):
        return {"default": value}
    if isinstance(value, list):
        result = {}
        for index, entry in enumerate(value):
            if (not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]
                    or not isinstance(entry.get("template"), str)):
                raise ModelTextError(f"tokenizer_config.json: chat_template[{index}] must be an object with a "
                                     "name string and a template string")
            if entry["name"] in result:
                raise ModelTextError(f"tokenizer_config.json: chat template {entry['name']!r} is defined twice")
            result[entry["name"]] = entry["template"]
        return result
    raise ModelTextError("tokenizer_config.json: chat_template must be a string, a list of named templates or null")


def template_sources(files, tokenizer_config=None):
    """{name: source} of a model's chat templates, with transformers' discovery precedence.

    chat_template.jinja ("default") and additional_chat_templates/<name>.jinja replace every template of
    tokenizer_config.json, whose chat_template is a string ("default") or a list of {name, template}. Without
    those files the legacy chat_template.json {"chat_template": ...} supplies "default". A legacy file combined
    with additional templates is an error, as in transformers; so are a legacy file and chat_template.jinja that
    disagree, and additional_chat_templates/default.jinja next to chat_template.jinja (transformers' tokenizer and
    processor loaders resolve those two cases differently). `files` is a ModelFiles or a directory path;
    `tokenizer_config` is the parsed tokenizer_config.json when the caller already read it.
    """
    files = model_files(files)
    modern = {}
    default = read_text(files, TEMPLATE_FILE)
    if default is not None:
        modern["default"] = default
    for path in list_names(files, TEMPLATE_DIR):
        name = path[len(TEMPLATE_DIR):]
        if "/" in name or not name.endswith(".jinja") or name == ".jinja":
            continue
        name = name[:-len(".jinja")]
        if name in modern:
            raise ModelTextError(f"{path} and {TEMPLATE_FILE} both define the default chat template")
        source = read_text(files, path)
        if source is None:
            raise ModelTextError(f"{path} is listed but missing")
        modern[name] = source
    legacy = read_json_object(files, LEGACY_TEMPLATE_FILE)
    if legacy is not None:
        if not isinstance(legacy.get("chat_template"), str):
            raise ModelTextError(f"{LEGACY_TEMPLATE_FILE} must hold a chat_template string")
        if len(modern) > (1 if default is not None else 0):
            raise ModelTextError(f"conflicting chat template files: the legacy {LEGACY_TEMPLATE_FILE} cannot be "
                                 f"combined with {TEMPLATE_DIR}; replace it with {TEMPLATE_FILE}")
        if default is not None and default != legacy["chat_template"]:
            raise ModelTextError(f"conflicting chat template files: {TEMPLATE_FILE} and the legacy "
                                 f"{LEGACY_TEMPLATE_FILE} hold different templates")
    if modern:
        return modern
    if legacy is not None:
        return {"default": legacy["chat_template"]}
    if tokenizer_config is None:
        tokenizer_config = read_json_object(files, "tokenizer_config.json") or {}
    return _config_templates(tokenizer_config.get("chat_template"))


def shipped_template(model_type):
    """(file name, source) of the chat template crit ships for config.json model_type `model_type`, or None."""
    name = SHIPPED_TEMPLATES.get(model_type) if isinstance(model_type, str) else None
    if name is None:
        return None
    return name, (SHIPPED_DIR / name).read_text(encoding="utf-8")


def chat_template_sources(files, chat_template=None, tokenizer_config=None):
    """(origin, {name: source}, shipped file) of the chat templates a model renders with: its own (template_sources),
    origin "model"; for a model whose files hold none, `chat_template` (a template source), origin "given", else the
    one crit ships for its config.json model_type (shipped_template), origin "crit", with its file name; else (None,
    {}, None). A chat_template for a model that carries its own raises ModelTextError: engines serve the model's."""
    files = model_files(files)
    if chat_template is not None and not isinstance(chat_template, str):
        raise TextError("chat_template must be a template source string")
    own = template_sources(files, tokenizer_config)
    if own:
        if chat_template is not None:
            raise ModelTextError("the model carries its own chat template: a chat template is given only for a model "
                                 "without one")
        return "model", own, None
    if chat_template is not None:
        return "given", {"default": chat_template}, None
    config = read_json_object(files, "config.json", max_bytes=MAX_CONFIG_BYTES, allow_nonfinite=True) or {}
    shipped = shipped_template(config.get("model_type"))
    return ("crit", {"default": shipped[1]}, shipped[0]) if shipped is not None else (None, {}, None)


class ChatTemplates(Mapping):
    """A model's named chat templates (possibly none), selected the way transformers' get_chat_template does."""

    def __init__(self, templates=()):
        templates = dict(templates)
        for name, template in templates.items():
            if not isinstance(template, ChatTemplate) or template.name != name:
                raise TextError(f"chat template {name!r} must be a ChatTemplate with the same name")
        self._templates = dict(sorted(templates.items()))

    @classmethod
    def from_sources(cls, sources, *, special_tokens=None, clock=datetime.now):
        return cls({name: ChatTemplate(source, name=name, special_tokens=special_tokens, clock=clock)
                    for name, source in dict(sources).items()})

    @classmethod
    def from_files(cls, files, *, special_tokens=None, clock=datetime.now, tokenizer_config=None):
        return cls.from_sources(template_sources(files, tokenizer_config), special_tokens=special_tokens, clock=clock)

    def __getitem__(self, name):
        return self._templates[name]

    def __iter__(self):
        return iter(self._templates)

    def __len__(self):
        return len(self._templates)

    def __repr__(self):
        return f"ChatTemplates({list(self._templates)!r})"

    def select(self, name=None, *, tools=None):
        """The template to render: `name` when given; else "tool_use" when tools are given and it exists; else
        "default"; else ModelTextError listing the names (or saying the model has no chat template)."""
        if not self._templates:
            raise ModelTextError("the model has no chat template")
        names = ", ".join(self._templates)
        if name is not None:
            if name not in self._templates:
                raise ModelTextError(f"the model has no chat template named {name!r}; available: {names}")
            return self._templates[name]
        if tools is not None and "tool_use" in self._templates:
            return self._templates["tool_use"]
        if "default" in self._templates:
            return self._templates["default"]
        raise ModelTextError(f"the model has several chat templates and no default; pass one of: {names}")
