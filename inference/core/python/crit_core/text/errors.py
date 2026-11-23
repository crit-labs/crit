"""Error types of the text layer, and the strict model-file readers that raise them (standard library only).

Every validation error is a TextError, a ValueError subclass, so callers that already catch (ValueError, OSError)
keep working. Misuse of a closed decoder or stream raises RuntimeError instead.
"""
from crit_core import jsonfile
from crit_core.files import ModelFilesError, model_files as _model_files

MAX_FILE_BYTES = jsonfile.MAX_BYTES            # tokenizer.json, tokenizer_config.json, templates and the like
MAX_CONFIG_BYTES = jsonfile.MAX_CONFIG_BYTES   # config.json and generation_config.json, as crit_core.model reads them


class TextError(ValueError):
    """Invalid text-layer input: a model file, message, template variable, token id or stop string."""


class ModelTextError(TextError):
    """A missing or malformed model text file, or a missing, unknown or ambiguous chat template."""


class TemplateError(TextError):
    """A chat template that failed to compile or render; `template_name` names it, `lineno` is its line when known."""

    def __init__(self, message, *, template_name=None, lineno=None):
        super().__init__(message)
        self.template_name = template_name
        self.lineno = lineno


class TokenError(TextError):
    """A token id that is not an integer in [0, vocab_size)."""


def model_files(source):
    """crit_core.files.model_files: a ModelFiles as given or DirectoryFiles for a directory path; a path that is not
    a directory raises ModelTextError, any other type TypeError."""
    try:
        return _model_files(source)
    except ModelFilesError as error:
        raise ModelTextError(str(error)) from None


def read_bytes(files, name, max_bytes=MAX_FILE_BYTES):
    """The file's bytes (at most max_bytes), or None when nothing exists by that name."""
    try:
        return files.read(name, max_bytes=max_bytes)
    except ModelFilesError as error:
        raise ModelTextError(str(error)) from None


def read_text(files, name, max_bytes=MAX_FILE_BYTES):
    """The file decoded as strict UTF-8, or None when it is absent."""
    data = read_bytes(files, name, max_bytes)
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ModelTextError(f"{name} is not UTF-8 ({error.reason} at byte {error.start})") from None


def read_json_object(files, name, *, max_bytes=MAX_FILE_BYTES, allow_nonfinite=False):
    """The file parsed as strict JSON that must hold an object, or None when it is absent. allow_nonfinite reads
    NaN and Infinity literals (for config.json and generation_config.json only)."""
    data = read_bytes(files, name, max_bytes)
    if data is None:
        return None
    try:
        value = jsonfile.loads(data, name, max_bytes=max_bytes, allow_nonfinite=allow_nonfinite)
    except jsonfile.JsonError as error:
        raise ModelTextError(str(error)) from None
    if not isinstance(value, dict):
        raise ModelTextError(f"{name} must hold a JSON object")
    return value


def list_names(files, prefix):
    """Sorted names of the files directly in the folder `prefix` (such as "additional_chat_templates/"), through
    names(prefix): a model directory lists only that folder, and names that crit_core.files refuses are skipped.
    Failures raise ModelTextError."""
    try:
        return [name for name in files.names(prefix) if "/" not in name[len(prefix):]]
    except (ModelFilesError, OSError) as error:
        raise ModelTextError(f"cannot list {prefix}: {error}") from None
