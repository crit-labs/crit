"""A test double of Core's resident adapter for crit serve's protocol tests: they check how crit serve turns HTTP
bodies into the request records it hands an adapter and how its host treats an adapter's replies and failures, so
they run a scripted chat engine at that boundary, where they can read the record and inject failures. Only crit's
tests put this folder on PYTHONPATH (support/serve.py's scripted_serving); crit serve itself has one way to run an
engine, Core's loader, which tests/test_serve_end_to_end.py serves over HTTP with no double.

CRIT_TEST_RESIDENT names a file of the served package (support/serve.py's engine.py): a resident host creates that
file's engine where Core's loader would assemble the package's plugins (crit_core.engine.resident).
CRIT_TEST_CAPABILITIES (a JSON list) is what crit serve offers for every package in place of the capabilities Core
derives from the package and the model (crit_api.engine.package_capabilities), in a process of crit's own, such as a
crit run command a test starts."""
import os

NAME = os.environ.get("CRIT_TEST_RESIDENT")
OFFERED = os.environ.get("CRIT_TEST_CAPABILITIES")


def _scripted(context, package):
    import hashlib
    import importlib.util
    from pathlib import Path
    import sys
    path = Path(package) / NAME
    name = "crit_test_resident_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].create_engine(context)


def _offered(package, model, configuration):
    import json
    from crit_api.options import serving_capabilities
    return serving_capabilities(json.loads(OFFERED))


# module -> (attribute, replacement), patched as each module finishes loading.
_PATCHES = {}
if NAME:
    _PATCHES["crit_core.engine"] = ("resident", _scripted)
if OFFERED:
    _PATCHES["crit_api.engine"] = ("package_capabilities", _offered)
    _PATCHES["crit_api.installed"] = ("package_capabilities", _offered)

if _PATCHES:
    import importlib.abc
    import sys

    class _Patching(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _PATCHES:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = getattr(finder, "find_spec", lambda *a: None)(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            run = spec.loader.exec_module
            attribute, replacement = _PATCHES[name]

            def exec_module(module):
                run(module)
                setattr(module, attribute, replacement)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Patching())
