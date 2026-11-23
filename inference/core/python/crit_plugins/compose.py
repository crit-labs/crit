"""Plugins built on other plugins: a plugin that subclasses a library plugin's object, or wraps one, instead of copying
it (a fused kernel over a library router's rule, a device form over another plugin's host form).

A plugin names the plugin it builds on among its sources (a folder holding that plugin's plugin.json) and, to take its
options, in `extends` (crit_plugins.manifest):

    "sources": {"base": "../experts/ROUTER"},
    "extends": "base"

The base's modules then import as in their own plugin (`from .base.plugin import Router`, its own sources nested
under it), so a subclass overrides the methods it changes and keeps the rest. A wrapper creates the base's object as
Core's loader would (create_base: the base's factory with the options it declares), and a plugin that composes a
plugin of the engine chosen by an option creates it by id (package_plugin). The extension states its own fills, needs
and provides: a fact of the base (rounds:<slot>, a capture-safe form) is the extension's when its own tests show it.
"""
import importlib
import sys
from pathlib import Path

from .manifest import PluginError, options_for


def _package(module):
    """The plugin package (crit_plugins.assemble.load_module's) a module of plugin code belongs to."""
    name = module if isinstance(module, str) else module.__name__
    top = sys.modules.get(name.split(".")[0])
    if top is None or getattr(top, "__crit_manifest__", None) is None:
        raise PluginError(f"{name} is no module of a plugin Core's loader loaded")
    return top


def _base(module, source):
    """(the plugin package, the source's name, the manifest of the plugin it holds)."""
    top = _package(module)
    own = top.__crit_manifest__
    if source is None:
        if not own.get("extends"):
            raise PluginError(f"{own['id']} extends no plugin: name the source that holds the plugin to create")
        source = own["extends"]["source"]
    manifest = getattr(getattr(top, source, None), "__crit_manifest__", None)
    if manifest is None:
        raise PluginError(f"{own['id']}'s source {source} is no plugin's folder")
    return top, source, manifest


def base_manifest(module, source=None):
    """The manifest of the plugin that source `source` of `module`'s plugin holds (default the source its manifest
    extends). module: the calling plugin's module or its __name__."""
    return _base(module, source)[2]


def create_base(module, context, options, parts, source=None):
    """The object of the plugin `module`'s plugin builds on (base_manifest), created as Core's loader creates a
    plugin: its factory, given the options of `options` it declares (an extension's own options left out), checked
    against its manifest, and the factory's context and parts."""
    top, source, manifest = _base(module, source)
    if manifest["entry"] is None:
        raise PluginError(f"{manifest['id']} fills no slot and has no factory: its extensions call its code")
    file, function = manifest["entry"]
    entry = importlib.import_module(f"{top.__name__}.{source}.{file[:-3].replace('/', '.')}")
    given = {name: value for name, value in (options or {}).items() if name in manifest["options"]}
    where = f"{top.__crit_manifest__['id']}'s base {manifest['id']}"
    return getattr(entry, function)(context, options_for(manifest, given, where), parts)


def package_plugin(context, plugin_id, parts, options=None):
    """The engine package's plugin `plugin_id` (another plugin of the same package, context.package/plugins), created
    as Core's loader creates a plugin, with `options` over its defaults: for a plugin that composes one the
    configuration names by an option."""
    from .assemble import Registry, load_module
    manifest = Registry.discover(Path(context.package) / "plugins")[plugin_id]
    factory = getattr(load_module(manifest), manifest["entry"][1])
    return factory(context, options_for(manifest, options or {}, f"plugins.options.{plugin_id}"), parts)


__all__ = ["base_manifest", "create_base", "package_plugin"]
