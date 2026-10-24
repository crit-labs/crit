"""crit eval: one benchmark and quality suite that every engine runs the same way."""
import json
import os
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
SUITE_DIR = EVAL_DIR / "suite"
CONTRACT = "eval-v1"
PROTOCOL_ID = re.compile(r"crit-eval-v(\d+)(?:\.(\d+))?")


def protocol_version(identifier):
    """crit-eval-v4 -> (4, 0) and crit-eval-v4.1 -> (4, 1). Scores compare within one major version: a change
    that can move a score bumps the major, anything else the minor (docs/eval/README.md)."""
    match = PROTOCOL_ID.fullmatch(str(identifier))
    if not match:
        raise ValueError(f"not an Eval protocol id: {identifier}")
    return int(match[1]), int(match[2] or 0)


# The context a user wants engines built for, unless settings.toml's context_tokens says otherwise; the model's
# positions and the device's memory may allow less.
DEFAULT_CONTEXT_TOKENS = 262_144


def machine_settings():
    """Memory thresholds from the crit checkout's settings.toml (vram_headroom_gib, context_reserve_gib) and the
    wanted context (context_tokens), when Eval runs from one; an Eval copied elsewhere without it uses the protocol's
    defaults and DEFAULT_CONTEXT_TOKENS."""
    path = EVAL_DIR.parent / "settings.toml"
    if not path.is_file():
        return {}
    import tomllib
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    found = {key: float(value[key]) for key in ("vram_headroom_gib", "context_reserve_gib")
             if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool)}
    if type(value.get("context_tokens")) is int and value["context_tokens"] > 0:
        found["context_tokens"] = value["context_tokens"]
    return found


def requested_context():
    """The context the user wants: settings.toml's context_tokens, else DEFAULT_CONTEXT_TOKENS."""
    return machine_settings().get("context_tokens", DEFAULT_CONTEXT_TOKENS)


def portable_path(path):
    """A path as results store it, naming no user: relative to the crit checkout (posix) inside it, `~/...` inside
    the home directory, the bare file name where another user directory (/home/NAME, C:/Users/NAME, WSL's
    /mnt/c/Users/NAME) holds it, else the absolute path (toolkits under /usr/local or Program Files). None stays
    None."""
    if path is None:
        return None
    path = Path(os.path.abspath(path))
    for root, prefix in ((EVAL_DIR.parent, ""), (Path.home(), "~/")):
        try:
            return prefix + path.relative_to(root).as_posix()
        except ValueError:
            pass
    if any(part.lower() in ("users", "home") for part in path.parts[:-2]):
        return path.name
    return path.as_posix()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1)
        handle.write("\n")
    os.replace(temporary, path)
