"""A model's text layer loaded from its files: tokenizer, chat templates, generation config and text identity."""
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from types import MappingProxyType

from crit_core.files import ModelFiles, file_name

from .errors import TextError, model_files, read_json_object
from .generation import GenerationConfig
from .stream import TextStream
from .template import ChatTemplates, chat_template_sources
from .tokenizer import Tokenizer


class _ReadNames(ModelFiles):
    """Passes reads and listings through to `files` (which records them) and remembers which files this load read."""

    def __init__(self, files):
        super().__init__()
        self.files = files
        self.read_names = set()

    def read(self, name, max_bytes=None):
        data = self.files.read(name, max_bytes=max_bytes)
        if data is not None:
            self.read_names.add(file_name(name))
        return data

    def names(self, prefix=""):
        return self.files.names(prefix)


def library_versions():
    """Versions that change token ids or rendered text: crit_core, jinja2 and tokenizers."""
    import crit_core
    import jinja2
    import tokenizers
    return {"crit_core": crit_core.__version__, "jinja2": jinja2.__version__, "tokenizers": tokenizers.__version__}


@dataclass(frozen=True)
class TextIdentity:
    """What the text layer depends on: a FileRecord (path, bytes, sha256) for exactly the files the loader read,
    the library versions, the chat templates that are none of the model's files ({"origin", "sha256": {name:
    sha256}}, or None) and a sha256 over them."""
    files: tuple
    versions: MappingProxyType = field(hash=False)
    sha256: str = ""
    template: MappingProxyType = field(default=None, hash=False)

    @classmethod
    def of(cls, records, versions, template=None):
        records = tuple(sorted(records, key=lambda record: record.path))
        versions = dict(sorted(versions.items()))
        value = {"files": [record.row() for record in records], "versions": versions}
        if template is not None:   # only then, so a model's own template keeps the identity it had
            value["template"] = template
        blob = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return cls(records, MappingProxyType(versions), hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                   None if template is None else MappingProxyType(dict(template)))

    def row(self):
        row = {"files": [record.row() for record in self.files], "versions": dict(self.versions),
               "sha256": self.sha256}
        if self.template is not None:
            row["template"] = dict(self.template)
        return row


@dataclass(frozen=True, eq=False)
class ModelText:
    """Tokenizer, chat templates, generation config and identity of one model; immutable and thread-safe.
    template_origin says where the templates come from (crit_core.text.chat_template_sources): "model", "given",
    "crit" or None (the model renders none)."""
    tokenizer: Tokenizer
    templates: ChatTemplates
    generation: GenerationConfig
    identity: TextIdentity
    template_origin: str = None

    def render(self, messages, *, template=None, tools=None, **variables):
        """Render with the selected chat template (ModelTextError when the model has none); see ChatTemplate.render."""
        return self.templates.select(template, tools=tools).render(messages, tools=tools, **variables)

    def encode_chat(self, messages, *, template=None, tools=None, **variables):
        """Render, then encode with add_special_tokens=False as transformers' apply_chat_template does: a BOS comes
        only from the template. Control-token text is always parsed as control tokens, whatever the tokenizer's
        split_special_tokens default, so the template's own markers stay single tokens; control-token text inside
        messages is parsed the same way, as in transformers. TextError when the ids start with two BOS tokens."""
        ids = self.tokenizer.encode(self.render(messages, template=template, tools=tools, **variables),
                                    add_special_tokens=False, split_special_tokens=False)
        bos = self.tokenizer.bos_token_id
        if bos is not None and len(ids) >= 2 and ids[0] == bos and ids[1] == bos:
            raise TextError(f"the encoded chat starts with two BOS tokens (id {bos}); the template or a message "
                            "adds a second one")
        return ids

    def text_stream(self, prompt_ids=(), **options):
        """A TextStream for one reply; stop_token_ids defaults to generation.stop_token_ids."""
        options.setdefault("stop_token_ids", self.generation.stop_token_ids)
        return TextStream(self.tokenizer, prompt_ids, **options)


def load_model_text(model, *, clock=datetime.now, split_special_tokens=False, chat_template=None):
    """Load a model's text layer from a directory path, a DirectoryFiles or any ModelFiles (TypeError otherwise).

    tokenizer.json is required; tokenizer_config.json, special_tokens_map.json, chat_template.jinja,
    additional_chat_templates/*.jinja, chat_template.json, generation_config.json and config.json are read when
    present. A model whose files hold no chat template renders with `chat_template` (a template source) when given,
    else with the one crit ships for its config.json model_type (crit_core.text.chat_template_sources); without
    either it loads, and render() raises ModelTextError. `clock` drives the templates' strftime_now (None leaves it
    undefined, for reproducible renders).
    """
    files = model_files(model)
    reader = _ReadNames(files)
    tokenizer_config = read_json_object(reader, "tokenizer_config.json") or {}
    tokenizer = Tokenizer.from_files(reader, split_special_tokens=split_special_tokens,
                                     tokenizer_config=tokenizer_config)
    origin, sources, _ = chat_template_sources(reader, chat_template, tokenizer_config)
    templates = ChatTemplates.from_sources(sources, special_tokens=tokenizer.special_tokens, clock=clock)
    generation = GenerationConfig.from_files(reader, tokenizer)
    records = [record for record in files.records if record.path in reader.read_names]
    template = None if origin in ("model", None) else {
        "origin": origin, "sha256": {name: templates[name].sha256 for name in templates}}
    return ModelText(tokenizer, templates, generation, TextIdentity.of(records, library_versions(), template), origin)
