"""crit eval: one benchmark and quality suite that every engine runs the same way."""
import json
import os
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
SUITE_DIR = EVAL_DIR / "suite"
CONTRACT = "eval-v1"


def machine_settings():
    """Memory thresholds from the crit checkout's settings.toml (vram_headroom_gib, context_reserve_gib), when
    Eval runs from one; an Eval copied elsewhere without it uses the protocol's defaults."""
    path = EVAL_DIR.parent / "settings.toml"
    if not path.is_file():
        return {}
    import tomllib
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    return {key: float(value[key]) for key in ("vram_headroom_gib", "context_reserve_gib")
            if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool)}


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
