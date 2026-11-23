"""crit Models' Python side: a decoder's mechanisms as its config.json states them and the rules over them
(crit_models.config), and what a GGUF's metadata must agree with for them (crit_models.gguf). Eval's limits and the
model plugins read them; Core reads only the generic fields (crit_core.model)."""
from crit_core import registry as _registry

__version__ = "0.23.0"

if "gguf_checks" in _registry.KINDS:   # a Core that runs a mechanism's GGUF checks (crit-core 0.22 on)
    _registry.register("gguf_checks", "crit_models.gguf:CHECKS")
