"""The EngineSpec of a checkpoint before any package exists: the library-default engine's (crit_core.spec).

    from crit.spec import library_spec
    spec = library_spec(model, context_tokens=131072, quality_mode="speed")
    spec["values"]["kv_elements_per_token"]   # {"value": ..., "source": "resolved", "by": "a-model"}

A run is created, recon written and an offload or bench plan made before any candidate exists, so those readers size
the engine a build starts from: the library's plugins at their defaults for the checkpoint
(crit.preflight.candidate.library_selection, the preflight's candidate), packaged in a temporary folder and resolved
from the checkpoint's headers alone (crit_core.spec.engine_spec: the model plugin's KV pools, the projections slot's
held matrix bytes, every option with its source). crit Eval reads its prices (crit_eval.limits.context.spec_prices)
and keeps its own benchmark rules: the required context, the protocol's KV precision, the reserve.

None where the checkpoint has no finished model plugin (a build writes one first), or the selection does not resolve:
the readers then size by Eval's reading of the config, as before, and say so."""
import json
from pathlib import Path
import shutil
import tempfile

from crit_plugins.exactness import DEFAULT_QUALITY_MODE

_CACHE = {}   # (model, context_tokens, quality_mode, tables_off_device) -> spec, for one process


def library_spec(model, context_tokens=None, quality_mode=DEFAULT_QUALITY_MODE, tables_off_device=False):
    """The EngineSpec of the library-default engine for `model` at `context_tokens` (the module's description), or
    None with no finished model plugin or a selection that does not resolve. quality_mode: the run's (the
    precisions it leaves to the mode: speed's FP8 trunk and head); tables_off_device: the placement keeps the lookup
    tables off the device."""
    key = (str(Path(model).resolve()), context_tokens, quality_mode, bool(tables_off_device))
    if key in _CACHE:
        return _CACHE[key]
    found = None
    try:
        found = _resolved(Path(model), context_tokens, quality_mode, tables_off_device)
    except (ValueError, OSError, KeyError, ImportError):   # no model plugin, or a selection the checkpoint refuses
        found = None
    _CACHE[key] = found
    return found


def _resolved(model, context_tokens, quality_mode, tables_off_device):
    from crit_core.spec import engine_spec
    from .plugins import add
    from .preflight.candidate import library_selection
    selection, plugins = library_selection(model, tables_off_device=tables_off_device)
    folder = Path(tempfile.mkdtemp(prefix="crit-spec-"))
    try:
        add(folder / "plugins", plugins)
        configuration = {"plugins": selection, **({"quality_mode": quality_mode} if quality_mode else {})}
        (folder / "configuration.json").write_text(json.dumps(configuration), encoding="utf-8")
        spec = engine_spec(folder, model, configuration, context_tokens=context_tokens,
                           context_source="protocol" if context_tokens else None)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    spec["package"] = "the library's defaults (crit.preflight.candidate.library_selection)"
    return spec


def prices(spec):
    """The spec's prices crit Eval sizes by ({} without a spec): crit_eval.limits.context.spec_prices."""
    from crit_eval.limits.context import spec_prices
    return spec_prices(spec)
