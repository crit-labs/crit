"""crit: run records, Kernels, Eval dispatch and the serving API's command line."""
__version__ = "0.17.2"

try:
    # crit Eval gives Core the model-family knowledge Core holds none of (crit_core.registry): the GGUF config rules
    # of the architectures its reference reads, so a lone GGUF of them is a whole model to every crit command.
    import crit_eval  # noqa: F401
except ImportError:   # a checkout without crit Eval beside it: Core derives the generic config keys only
    pass
