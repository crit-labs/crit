"""Strict JSON for model files, with the rules of the native reader (crit/json.h).

Duplicate keys, non-finite numbers, unpaired surrogate escapes, nesting deeper than max_depth and oversize input
are errors. Integers stay exact Python ints of any size up to Python's digit limit (sys.get_int_max_str_digits);
typed readers decide which range they accept.
"""
import json
import math
import os
from pathlib import Path
import re
import stat

MAX_BYTES = 128 << 20          # json::Limits::max_bytes
MAX_DEPTH = 128                # json::Limits::max_depth: open objects and arrays, the root included
MAX_CONFIG_BYTES = 16 << 20    # config.json and generation_config.json, as the native config loaders read them

# One float object per literal, as the json module's default: two parses of a NaN compare equal as containers.
_NONFINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")   # may also find an escaped backslash; the check is exact
_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[{]|[\]}]', re.S)


class JsonError(ValueError):
    """Malformed, non-UTF-8 or oversize JSON in a model file."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JsonError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _constant(name):
    raise JsonError(f"non-finite number {name}")


def _finite(text):
    value = float(text)
    if not math.isfinite(value):
        raise JsonError(f"non-finite number {text}")
    return value


def _integer_only(text):
    raise JsonError(f"real number {text} where only integers are allowed")


def _limit(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"json limits must be positive integers, not {name}={value!r}")


def _utf8(text):
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _deeper(text, max_depth):
    """Whether containers outside strings nest more than max_depth deep (after json.loads ran out of stack)."""
    depth = 0
    for match in _TOKEN.finditer(text):
        token = match.group()
        if token in "[{":
            depth += 1
            if depth > max_depth:
                return True
        elif token in "]}":
            depth -= 1
    return False


def _check_tree(value, source, max_depth, strings):
    """Nesting depth of every container and, when `strings`, that every key and string encodes as UTF-8. Valid
    UTF-8 input leaves a lone surrogate only where an escape such as \\ud800 had no partner."""
    if strings and type(value) is str and not _utf8(value):
        raise JsonError(f"{source}: unpaired surrogate escape")
    stack = [(value, 1)] if type(value) in (dict, list) else []
    while stack:
        container, depth = stack.pop()
        if depth > max_depth:
            raise JsonError(f"{source}: nesting exceeds the depth limit of {max_depth}")
        if type(container) is dict:
            if strings and not all(_utf8(key) for key in container):
                raise JsonError(f"{source}: unpaired surrogate escape")
            children = container.values()
        else:
            children = container
        for child in children:
            kind = type(child)
            if kind is dict or kind is list:
                stack.append((child, depth + 1))
            elif strings and kind is str and not _utf8(child):
                raise JsonError(f"{source}: unpaired surrogate escape")


def loads(data, source="JSON", *, max_bytes=MAX_BYTES, allow_real=True, allow_nonfinite=False, max_depth=MAX_DEPTH):
    """Parse bytes or text. `source` names the file in error messages.

    allow_real=False refuses real numbers entirely. allow_nonfinite=True reads the literals NaN, Infinity and
    -Infinity as non-finite floats (only config.json and generation_config.json opt in); a numeric literal beyond
    a double's range stays an error either way. Text input must be encodable as UTF-8.
    """
    _limit(max_bytes, "max_bytes")
    _limit(max_depth, "max_depth")
    if isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
        if len(data) > max_bytes:
            raise JsonError(f"{source}: {len(data)} bytes exceeds the {max_bytes}-byte limit")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise JsonError(f"{source}: not UTF-8 ({error.reason} at byte {error.start})") from None
    elif isinstance(data, str):
        text = data
        if len(text) > max_bytes:   # every character takes at least one byte
            raise JsonError(f"{source}: text exceeds the {max_bytes}-byte limit")
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise JsonError(f"{source}: not UTF-8 ({error.reason} at character {error.start})") from None
        if size > max_bytes:
            raise JsonError(f"{source}: text exceeds the {max_bytes}-byte limit")
    else:
        raise TypeError("JSON input must be bytes or str")
    constant = (_NONFINITE.__getitem__ if allow_real else _integer_only) if allow_nonfinite else _constant
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=constant,
                           parse_float=_finite if allow_real else _integer_only)
    except JsonError as error:
        raise JsonError(f"{source}: {error}") from None
    except json.JSONDecodeError as error:
        raise JsonError(f"{source}: {error.msg} at line {error.lineno} column {error.colno}") from None
    except RecursionError:
        raise JsonError(f"{source}: nesting exceeds the depth limit of {max_depth}" if _deeper(text, max_depth)
                        else f"{source}: nesting is too deep") from None
    except ValueError as error:   # an integer beyond Python's digit limit; must follow its subclasses above
        raise JsonError(f"{source}: {error}") from None
    strings = _SURROGATE_ESCAPE.search(text) is not None
    if strings or text.count("[") + text.count("{") > max_depth:   # else no container can be too deep
        _check_tree(value, source, max_depth, strings)
    return value


def read(path, *, max_bytes=MAX_BYTES, allow_real=True, allow_nonfinite=False, max_depth=MAX_DEPTH):
    """Read and parse one regular file (symlinks followed), checking its size before reading it. A missing file
    raises OSError; a directory or other non-regular file raises JsonError."""
    _limit(max_bytes, "max_bytes")
    path = Path(path)
    if not stat.S_ISREG(os.stat(path).st_mode):
        raise JsonError(f"{path}: not a regular file")
    # O_NONBLOCK keeps a FIFO swapped in after the check from blocking the open; regular files ignore it.
    with open(os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)), "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise JsonError(f"{path}: not a regular file")
        if info.st_size > max_bytes:
            raise JsonError(f"{path}: {info.st_size} bytes exceeds the {max_bytes}-byte limit")
        data = handle.read(max_bytes + 1)
    return loads(data, str(path), max_bytes=max_bytes, allow_real=allow_real, allow_nonfinite=allow_nonfinite,
                 max_depth=max_depth)
