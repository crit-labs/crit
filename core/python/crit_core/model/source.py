"""A model as crit reads it, whatever format its weights are stored in.

open_model(path) takes what the user points at, a model folder or a weight file, and returns a ModelSource:

    format        the weight format's name ("safetensors", "gguf", "torch", "onnx"), or None for a folder
                  without weights
    weights       the format's own header object (crit_core.model.Checkpoint, GgufModel, TorchCheckpoint,
                  OnnxModel), validated, never decoded
    files         the description files as a ModelFiles source (crit_core.files): config.json, tokenizer.json,
                  tokenizer_config.json, the chat templates, generation_config.json. A folder's own files are
                  read as they are; a format whose weights carry their own description (GGUF) derives the
                  files the folder lacks, so a lone weight file is a whole model.
    weight_files  the files holding the language model's tensors, each with .path and .tensors (name, dtype,
                  shape, nbytes), with or without draft (MTP) layers shipped in their own files
    draft_blocks  layer numbers of draft layers stored among the model's own blocks (GGUF's nextn layers)

Everything that is not a weight (Core's config rules, the text layer, Eval's ceiling and context, recon) reads
the model through it. A new weight format is one ModelFormat in FORMATS: its name, the file suffixes that hold
its weights, and open(path) -> ModelSource; a format whose suffix other files share (.bin) also says which
files are its own. Eval's reference also needs the weights as the Hugging Face
checkpoint they came from; that part needs torch and lives in Eval (crit_eval.quality.reference.open_checkpoint).

Description files come in groups (gguf_files.GROUPS: the config; the tokenizer with its templates; the
generation config). A folder that holds any file of a group supplies that whole group, and only the groups it
lacks are derived, so a derived tokenizer.json is never mixed with a folder's tokenizer_config.json.
"""
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import threading
from types import MappingProxyType

from crit_core.files import ModelFiles, _read_regular

from .config import CONFIG_NAME, load_model_config, parse_model_config
from .gguf import PROJECTOR, SUFFIX as GGUF_SUFFIX, GgufError, check_gguf_folder, inspect_gguf_model
from .gguf_files import GROUPS, derived_files
from .safetensors import INDEX_SUFFIX, inspect_model, inspect_safetensors
from . import onnx, torch_pickle

DRAFT_FILE = "mtp"   # draft (MTP) layers shipped in a file of their own carry this in its name


class ModelSourceError(ValueError):
    """A path that is not a model crit can read: nothing there, several weight formats, an unknown file."""


class SourceFiles(ModelFiles):
    """A folder's files (as DirectoryFiles reads them), or none for a lone weight file, plus derived files served
    from memory. Derived bytes are built on first read and kept."""

    def __init__(self, folder, derived):
        super().__init__()
        self.folder = None if folder is None else Path(folder)
        self._derived = derived
        self._built = {}
        self._build_lock = threading.Lock()

    def _read(self, name):
        return self._read_limited(name, None)

    def _read_limited(self, name, max_bytes):
        build = self._derived.get(name)
        if build is not None:
            with self._build_lock:
                if name not in self._built:
                    self._built[name] = build()
            return self._built[name]
        if self.folder is None:
            return None
        return _read_regular(self.folder / name, name, max_bytes)

    def _names(self):
        names = set(self._derived)
        if self.folder is not None:
            names.update(path.relative_to(self.folder).as_posix() for path in self.folder.rglob("*") if path.is_file())
        return names


@dataclass(frozen=True)
class Weights:
    header: object = None           # the format's header object (Checkpoint, GgufModel)
    shards: tuple = ()              # the weight files the checkpoint names, in its order
    extra_drafts: tuple = ()        # draft-layer files beside the checkpoint that it does not name
    draft_blocks: frozenset = frozenset()


@dataclass(frozen=True, eq=False)
class ModelSource:
    path: Path                      # what was opened, absolute
    format: str | None              # FORMATS key, or None: no weight files crit reads
    folder: Path | None = None      # the folder whose own files are read (None for a lone weight file)
    derived: object = field(default_factory=lambda: MappingProxyType({}))   # {file name: build()}
    notes: tuple = ()
    load: object = field(default=Weights, repr=False)   # () -> Weights, run once on first use

    def _weights(self):
        cached = self.__dict__.get("_loaded")
        if cached is None:
            cached = self.load()
            object.__setattr__(self, "_loaded", cached)
        return cached

    @property
    def weights(self):
        """The format's header object, validated when first used (never decoded)."""
        return self._weights().header

    @property
    def draft_blocks(self):
        return self._weights().draft_blocks

    @property
    def files(self):
        """The description files, a fresh ModelFiles source (its records cover what one caller read)."""
        return SourceFiles(self.folder, self.derived)

    @property
    def derived_names(self):
        return tuple(sorted(self.derived))

    def weight_files(self, drafts=False):
        """The weight files of the language model. Without `drafts`, files with DRAFT_FILE in their name are left
        out; with it, draft files beside the checkpoint join it."""
        weights = self._weights()
        if not drafts:
            return tuple(shard for shard in weights.shards if DRAFT_FILE not in shard.path.name)
        return weights.shards + weights.extra_drafts

    def config(self):
        """config.json read with Core's rules (crit_core.model.load_model_config); an empty config when the model
        has none."""
        if CONFIG_NAME not in self.derived and (self.folder is None or not os.path.lexists(self.folder / CONFIG_NAME)):
            return parse_model_config({})
        return load_model_config(self.files)


@dataclass(frozen=True)
class ModelFormat:
    name: str
    suffixes: tuple                 # names of files that hold this format's weights end with one of these
    open: object                    # open(path: Path) -> ModelSource, path a folder or one of its weight files
    holds: object = None            # holds(path) -> bool: whether a file with one of the suffixes is this format's
    fallback: bool = False          # a fallback's files are ignored beside another format's (as Hugging Face loads
                                    # a repository's safetensors over the pytorch_model.bin it also ships)

    def holds_file(self, path):
        return path.name.endswith(self.suffixes) and (self.holds is None or self.holds(path))

    def weight_names(self, folder):
        """The weight files of this format in `folder` (not recursive)."""
        return sorted(path.name for path in folder.iterdir()
                      if (path.is_file() or path.is_symlink()) and self.holds_file(path))


def _safetensors(path):
    def load():
        weights = inspect_model(path)
        if not path.is_dir():
            return Weights(weights, tuple(weights.shards))
        named = {shard.path.name for shard in weights.shards}
        extra = tuple(inspect_safetensors(file) for file in sorted(path.glob(f"*{DRAFT_FILE}*.safetensors"))
                      if file.name not in named)
        return Weights(weights, tuple(weights.shards), extra)
    # A lone .safetensors file carries no description files. The headers are read when first used.
    return ModelSource(path, "safetensors", folder=path if path.is_dir() else None, load=load)


def _folder_groups(folder):
    """The description-file groups a folder supplies itself."""
    if folder is None:
        return set()
    present = set()
    for group, names in GROUPS.items():
        for name in names:
            target = folder / name.rstrip("/")
            if (target.is_dir() and any(target.glob("*.jinja"))) if name.endswith("/") else os.path.lexists(target):
                present.add(group)
    return present


def _gguf(path):
    gguf = inspect_gguf_model(path)
    folder = path if path.is_dir() else None
    own = _folder_groups(folder)
    derived = {}
    for group, files in derived_files(gguf).items():
        if group not in own:
            derived.update(files)
    count, blocks = gguf.get(".nextn_predict_layers") or 0, gguf.get(".block_count") or 0
    weights = Weights(gguf, tuple(gguf.files),
                      draft_blocks=frozenset(range(blocks - count, blocks)) if count else frozenset())
    source = ModelSource(path, "gguf", folder, MappingProxyType(derived), load=lambda: weights)
    if own & {"config", "tokenizer"}:
        # The folder's own files describe the weights to everything else, so they must be the base model's.
        check = check_gguf_folder(gguf, source.config(), folder)
        check.raise_errors(folder)
        source = replace(source, notes=check.notes)
    return source


def _torch(path):
    def load():
        weights = torch_pickle.inspect_torch_model(path)
        return Weights(weights, weights.shards)
    # PyTorch state dicts carry no description files: a folder supplies them; a lone file has none.
    return ModelSource(path, "torch", folder=path if path.is_dir() else None, load=load)


def _onnx(path):
    file = onnx.find_onnx(path)

    def load():
        weights = onnx.inspect_onnx(file)
        return Weights(weights, weights.shards)
    # An ONNX graph carries no description files: its folder supplies them, or the repository above an onnx/
    # folder (crit_core.model.onnx.description_folder).
    return ModelSource(path, "onnx", folder=onnx.description_folder(path), load=load)


FORMATS = {
    "safetensors": ModelFormat("safetensors", (".safetensors", INDEX_SUFFIX), _safetensors),
    "gguf": ModelFormat("gguf", (GGUF_SUFFIX,), _gguf, holds=lambda path: not PROJECTOR.search(path.stem)),
    "torch": ModelFormat("torch", torch_pickle.SUFFIXES + (torch_pickle.INDEX_SUFFIX,), _torch,
                         holds=torch_pickle.is_state_dict_file, fallback=True),
    "onnx": ModelFormat("onnx", (onnx.SUFFIX,), _onnx, fallback=True),
}


def _detect(path):
    if path.is_dir():
        found = {name: fmt for name, fmt in FORMATS.items() if fmt.weight_names(path)}
        if any(not fmt.fallback for fmt in found.values()):
            found = {name: fmt for name, fmt in found.items() if not fmt.fallback}
        if len(found) > 1:
            raise ModelSourceError(f"{path}: holds {' and '.join(sorted(found))} weights; keep one model per folder")
        return next(iter(found.values()), None)
    return next((fmt for fmt in FORMATS.values() if fmt.holds_file(path)), None)


def _signature(path):
    """What the cache keys on: the folder's (or the file's folder's) entries with their sizes and times."""
    folder = path if path.is_dir() else path.parent
    entries = []
    for entry in os.scandir(folder):
        try:
            info = entry.stat()
        except OSError:
            continue
        entries.append((entry.name, info.st_size, info.st_mtime_ns))
    return tuple(sorted(entries))


_CACHE, _CACHE_LOCK, _CACHE_SIZE = OrderedDict(), threading.Lock(), 8


def open_model(path):
    """The ModelSource for a model folder or a weight file (see the module docstring). Raises ModelSourceError for
    a path that does not exist or a folder of two formats, and the format's own error (SafetensorsError, GgufError) for weights that
    break its contract. Results are cached while the folder's entries are unchanged."""
    path = Path(path).absolute()
    if not os.path.exists(path):
        raise ModelSourceError(f"{path}: no such model folder or file")
    key = (str(path), _signature(path))
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    fmt = _detect(path)
    if fmt is None:
        # A folder without weight files, or a file in a format crit does not read (an engine's own): no weights,
        # and description files only from the folder.
        source = ModelSource(path, None, folder=path if path.is_dir() else None)
    else:
        source = fmt.open(path)
    with _CACHE_LOCK:
        _CACHE[key] = source
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return source
