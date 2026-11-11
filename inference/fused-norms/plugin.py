"""fused-norms in the norms slot: the host form is the slot's FP64 rows (code/rows.py). The device form is crit's
kernel fused-norms for every call (Triton, one launch per call, built on this machine with the engine's tuning or its
default knobs), and where that kernel cannot be built here (no Triton or CUDA, crit's kernels package not installed)
torch-norms' arithmetic (code/torch_norms.py). No library kernel runs a call: FlashInfer's norms and activation
products are the kernel's yardstick (its bench), not a path of the plugin.

The fused forms (norms:fused: rms_norm_rope_pair, rope_pair, stream_add_norm) are one launch of the kernel where the
separate calls take two; each output has the bits the plugin's separate calls give it.

The choice of kernel or torch is made once, when the plugin is created, so a process never mixes the two."""
from pathlib import Path

from crit_plugins.kernels import load_kernel

from .norms import rows, torch_norms

KERNEL = "fused-norms"


class FusedNorms(rows.Rows):
    def __init__(self, kernel):
        # A kernel package without every operation the plugin calls (a library older than the plugin) serves
        # none of them: the fused forms keep the separate calls' bits only from one
        # source, so the plugin runs torch-norms' arithmetic throughout and the facts say why.
        self.lacking = [name for name in torch_norms.OPERATIONS if kernel is not None and not hasattr(kernel, name)]
        if self.lacking:
            kernel = None
        self.kernel = kernel
        self.source = kernel if kernel is not None else torch_norms
        for name in torch_norms.OPERATIONS:
            setattr(self, name, getattr(self.source, name))

    def facts(self):
        """Which code serves the device form (the engine's facts carry it, beside the kernel's record: tuned, built or
        fallback)."""
        found = {"kernel": KERNEL if self.kernel is not None else "torch fallback"}
        if self.lacking:
            found["kernel_lacks"] = self.lacking
        return {"norms": found}


def create(context, options, parts):
    return FusedNorms(load_kernel(KERNEL, folder=Path(__file__).parent))
