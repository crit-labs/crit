"""A model plugin's folder read without a checkpoint or a device: for crit models and for a model plugin's tests."""
import importlib
from pathlib import Path

CORE = "model-template"   # the models core, which a model plugin extends


def composing(folder):
    """The models core's compose module as the model plugin in `folder` imports it, after the plugin's own plugin.py
    ran (so compose.plugin_layout(folder) answers). PluginError where the folder holds no plugin that extends the
    core."""
    from crit_plugins.assemble import load_module, plugin_package
    from crit_plugins.manifest import PluginError, load_manifest
    manifest = load_manifest(Path(folder).resolve())
    extends = manifest.get("extends") or {}
    if manifest["fills"] != ["model"] or extends.get("id") != CORE:
        raise PluginError(f"{manifest['id']} is no model plugin that extends the models core ({CORE}): it fills "
                          f"{', '.join(manifest['fills']) or 'no slot'} and extends {extends.get('id') or 'no plugin'}")
    load_module(manifest)
    return importlib.import_module(f"{plugin_package(manifest).__name__}.{extends['source']}.template.compose")


def layout(folder):
    """The layout the model plugin in `folder` composes, checked against the techniques the core mounts and the
    plugin's own (ValueError naming what does not check)."""
    return composing(folder).plugin_layout(Path(folder).resolve())
