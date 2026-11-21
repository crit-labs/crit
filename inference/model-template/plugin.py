"""The library's model plugin composed from blocks (code/: compose, blocks, the serving contracts): the checkpoint's
architecture from the layout for its model_type (code/layouts), or the layout file the option `layout` names beside
this plugin. README.md says how an engine composes its own."""
from pathlib import Path

from .template.compose import compose, resolve as resolve_composed, tuning

HERE = Path(__file__).resolve().parent


def resolve(context, options, dry):
    """The engine's KV before any load: the layout's pools over the checkpoint's headers, priced by the kv_cache
    plugin (compose.resolve). Core's loader holds the load to it."""
    return resolve_composed(context, options, dry, folder=HERE)


def memory(context, options):
    tuning(options, HERE)
    return {"device": 0, "host": 0}


def create(context, options, parts):
    tuning(options, HERE)
    return compose(context, options, parts, folder=HERE)
