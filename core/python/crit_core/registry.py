"""What Core reads about particular model families, given by the packages that know them (standard library only).

Core holds no rule or file that holds only for one model family (core/README.md), yet two of its readers need such
knowledge for the families they meet:

    gguf_config      general.architecture -> rule(model, get) returning config.json as a dict, the inverse of what a
                     GGUF converter writes for that architecture (crit_core.model.gguf_files: get is its
                     MetadataReader over the model)
    chat_templates   config.json model_type -> the path of the chat template for a family whose checkpoints carry
                     none (crit_core.text.chat_template_sources)

A package gives a kind a mapping, or "module:attribute" naming one, which is imported only when the kind is first
read: at run time with register(kind, source), and, installed, in an entry point of group crit_core.<kind> (any
name, the source as its value), so Core finds it without the package being imported. crit-eval's architectures give
both kinds, both ways. Of the sources that give a key, the first keeps it: registered ones in registration order,
then entry points by name.
"""
from collections.abc import Mapping
import importlib
import threading
from types import MappingProxyType

KINDS = ("gguf_config", "chat_templates")
GROUP = "crit_core."   # + kind: the entry point group

_lock = threading.RLock()   # reentrant: importing a source may register more
_sources = {kind: [] for kind in KINDS}
_generation = dict.fromkeys(KINDS, 0)
_merged = {}
_installed = {}


class RegistryError(ValueError):
    """A kind that does not exist, or a source that does not load or is not a mapping."""


def _kind(kind):
    if kind not in KINDS:
        raise RegistryError(f"no registry kind {kind!r} (kinds: {', '.join(KINDS)})")
    return kind


def register(kind, source):
    """Adds `source` (a mapping, or "module:attribute" naming one) to `kind`; a source already there is kept once."""
    if not isinstance(source, (str, Mapping)):
        raise RegistryError(f"a {kind} source is a mapping or 'module:attribute', not {type(source).__name__}")
    with _lock:
        sources = _sources[_kind(kind)]
        if not any(s is source or (isinstance(s, str) and s == source) for s in sources):
            sources.append(source)
            _generation[kind] += 1
            _merged.pop(kind, None)


def _load(source, label):
    if not isinstance(source, str):
        return source
    module, _, attribute = source.partition(":")
    try:
        value = importlib.import_module(module)
        for part in attribute.split(".") if attribute else ():
            value = getattr(value, part)
    except Exception as error:
        raise RegistryError(f"{label} {source!r} does not load: {error}") from error
    if not isinstance(value, Mapping):
        raise RegistryError(f"{label} {source!r} is not a mapping ({type(value).__name__})")
    return value


def _entry_points(kind):
    """The (name, source) of the installed entry points of group crit_core.<kind>, by name; read once."""
    if kind not in _installed:
        from importlib.metadata import entry_points
        _installed[kind] = sorted({(point.name, point.value) for point in entry_points(group=GROUP + kind)})
    return _installed[kind]


def entries(kind):
    """{key: value} of `kind` (read-only): what its sources give, the first source that gives a key keeping it."""
    with _lock:
        while _merged.get(_kind(kind)) is None:
            generation, merged = _generation[kind], {}
            labelled = [(source, f"crit_core.registry {kind} source") for source in list(_sources[kind])]
            labelled += [(source, f"entry point {name} of group {GROUP + kind}")
                         for name, source in _entry_points(kind)]
            for source, label in labelled:
                for key, value in _load(source, label).items():
                    merged.setdefault(key, value)
            if _generation[kind] == generation:   # else a source loaded registered another: merge again
                _merged[kind] = MappingProxyType(merged)
        return _merged[kind]
