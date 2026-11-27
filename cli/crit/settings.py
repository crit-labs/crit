"""The machine's settings: PROJECT/settings.toml, read in one place for crit's commands.

crit run, crit pull and crit bench read the project's settings.toml through load(); a missing file means no settings,
and a key or type the file should not hold is refused. The headrooms a setting may leave out take their defaults
from crit Core: the device's crit_plugins.memory.HEADROOM (which Core's loader leaves free where a package names
none), the host's crit_plugins.memory.HOST_HEADROOM (RAM_HEADROOM_GIB). crit Eval's own commands read the same file
leniently through crit_eval.machine_settings(), which finds the project crit names to it (crit.cli.named_project).
"""
from pathlib import Path
import tomllib

GIB = 1 << 30
# What settings.toml may hold, with each value's type.
SETTINGS = {"wsl_run_root": str, "ram_headroom_gib": (int, float), "vram_headroom_gib": (int, float),
            "context_reserve_gib": (int, float), "device": str, "context_tokens": int,
            "eval_packages": str, "quality_mode": str, "gpu_lock": str}
# The host memory a run leaves available where settings.toml names no ram_headroom_gib: Core's
# (crit_plugins.memory.HOST_HEADROOM, the one default, which crit Eval and the engine's spec take too).
from crit_plugins.memory import HOST_HEADROOM  # noqa: E402
RAM_HEADROOM_GIB = HOST_HEADROOM / GIB


def load(project):
    """Machine settings from PROJECT/settings.toml; a missing file means none."""
    path = Path(project) / "settings.toml"
    if not path.is_file():
        return {}
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    for key, item in value.items():
        if key not in SETTINGS:
            raise ValueError(f"{path}: unknown setting {key!r}; known: {', '.join(SETTINGS)}")
        if isinstance(item, bool) or not isinstance(item, SETTINGS[key]):
            raise ValueError(f"{path}: {key} has the wrong type")
    return value


def vram_headroom_gib(settings):
    """The device memory an engine leaves free, in GiB: the settings' vram_headroom_gib, else Core's default."""
    from crit_plugins.memory import HEADROOM
    return settings.get("vram_headroom_gib", HEADROOM / GIB)


def vram_headroom_bytes(settings):
    """vram_headroom_gib(settings) in bytes."""
    return round(vram_headroom_gib(settings) * GIB)


def ram_headroom_bytes(settings):
    """The host memory a run leaves available, in bytes: the settings' ram_headroom_gib, else RAM_HEADROOM_GIB."""
    return round(settings.get("ram_headroom_gib", RAM_HEADROOM_GIB) * GIB)
