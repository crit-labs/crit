"""crit: run records, Kernels, Eval dispatch and the serving API's command line."""
__version__ = "0.25.0"


def _models():
    """crit Models' Python package (crit_models): installed, else a checkout's models/core/python, put on the path of
    this process and of the processes it starts (a plugin's tests, an engine's own process)."""
    import importlib.util
    import os
    from pathlib import Path
    if importlib.util.find_spec("crit_models") is not None:
        return
    folder = Path(__file__).resolve().parents[2] / "models" / "core" / "python"
    if (folder / "crit_models").is_dir():
        import sys
        sys.path.insert(0, str(folder))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(folder)] + ([os.environ["PYTHONPATH"]]
                                                                     if os.environ.get("PYTHONPATH") else []))


_models()
try:
    # A decoder's mechanisms (crit_models): its GGUF checks join Core's (crit_core.registry gguf_checks) on import.
    import crit_models  # noqa: F401
except ImportError:   # crit installed without crit-models: Core checks a GGUF folder's generic shape only
    pass

try:
    # crit Eval gives Core the model-family knowledge Core holds none of (crit_core.registry): the GGUF config rules
    # of the architectures its reference reads, so a lone GGUF of them is a whole model to every crit command.
    import crit_eval  # noqa: F401
except ImportError:   # a checkout without crit Eval beside it: Core derives the generic config keys only
    pass
