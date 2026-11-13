"""A toy engine package for tests that load engines the one way crit loads them, through Core's loader
(crit_core.engine): plugins/ holding the toy CPU plugins (inference/examples/plugin-engine/plugins, the plugin engine example) and configuration.json whose plugins selection
fills every slot an engine fills. Its model is toy-model, a batched step model on Core's concurrent runtime,
which needs Core's crit_core_steps library (STEP_LIBRARY)."""
import json
from pathlib import Path
import shutil
import sys

# Core beside this checkout, as support/runs.py puts it: a test module imports this one first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "python"))
from crit_core.native import step_library  # noqa: E402

TOY = Path(__file__).resolve().parents[2] / "inference" / "examples" / "plugin-engine" / "plugins"
USE = {"model": "toy-model", "placement": "toy-placement", "projections": "toy-projections",
       "attention": "toy-attention", "kv_cache": "toy-kv", "sampler": "toy-sampler"}
DRAFTING = {"drafter": "toy-drafter", "verifier": "toy-verifier"}
# Core's runtime: the crit_core_steps library the engine tests load, found as Core's loader finds it
# (crit_core.native.step_library: CRIT_CORE_STEP_LIBRARY, else the checkout's build, else Core's cache).
STEP_LIBRARY = step_library()


def selection(drafting=False, options=None, **use):
    chosen = {**USE, **(DRAFTING if drafting else {}), **use}
    return {"use": chosen, "options": dict(options or {})}


def write_package(folder, drafting=False, options=None, configuration=None, **use):
    """A toy engine package in `folder`: its plugins and configuration.json ({"plugins": selection, ...})."""
    folder = Path(folder)
    plugins = folder / "plugins"
    if plugins.exists():
        shutil.rmtree(plugins)
    shutil.copytree(TOY, plugins, ignore=shutil.ignore_patterns("__pycache__"))
    value = {"plugins": selection(drafting, options, **use), **(configuration or {})}
    (folder / "configuration.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return folder
