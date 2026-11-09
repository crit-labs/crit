"""Plain attention in the attention slot: attention-state's softmax over the visible rows (implementation.py) as the
host form. On the device (attention-state's device_attention.py) each token attends its sequence's positions up to its
own, read where the pool keeps them: FlashInfer's paged kernels over 16-bit and FP8 pages when FlashInfer is installed,
the registry's paged-kv-attention kernels (a split-KV reader of decode and verify windows over every page type, trees
and block lists included, and a prompt reader of fp8:row and INT8 pages) where they load, and torch otherwise: each
row's KV gathered and torch's scaled dot product attention, the plain path the others are tested against (option
kernels: torch forces it). Captured rounds (rounds:attention) read through the pool's round tables. Its memory()
declares exactly what the device form allocates when it is made."""
from pathlib import Path

from .state import device_attention
from .state import implementation as reference

KERNELS = "crit-labs/paged-kv-attention"


class SoftmaxAttention:
    def __init__(self, kv, device_form):
        self.kv, self.device_form = kv, device_form

    def attend(self, sequence, queries, positions, scale=None, pending=False):
        rows = self.kv.rows(sequence, pending)
        if len(queries) != len(positions):
            raise ValueError("one position per query")
        out = []
        for query, position in zip(queries, positions):
            visible = [(tuple(key), tuple(value)) for p, key, value in rows if p <= position]
            if not visible:
                raise ValueError(f"no row at or before position {position}")
            state = reference.KVState(len(visible), len(query), len(visible[0][1]))
            state.committed = visible
            out.append(list(state.attend(query, scale=scale)))
        return out

    # -- the device form ---------------------------------------------------------------------------------------------------
    def forward(self, layer, queries, pool, batch, scale=None, blocks=None, block_size=None, parents=None, lse=False):
        """queries [tokens][heads][dim] of a StepBatch or a RoundBatch over the pool's KV (written for this step
        already); each token sees its sequence's positions up to its own (device_attention.DeviceAttention.forward:
        blocks, parents, lse)."""
        return self.device_form.forward(layer, queries, pool, batch, scale, blocks, block_size, parents, lse)

    def facts(self):
        """The device form's engine facts (device_attention.DeviceAttention.facts)."""
        return self.device_form.facts()

    def close(self):
        self.device_form.close()


def _kernels(options):
    """The registry kernel the device form reads pages with, or None (option kernels torch, or not available)."""
    if options["kernels"] != "auto":
        return None
    from crit_plugins.kernels import registry_kernel
    return registry_kernel(KERNELS, folder=Path(__file__).parent)


def memory(context, options):
    """What the device form allocates when it is made (device_attention.device_memory), exactly: the split-KV reader's
    partials where the registry kernel loads, FlashInfer's workspaces where it serves, on a CUDA engine."""
    found, host = device_attention.device_memory(device_attention.engine_device(context), options["kernels"],
                                                 _kernels(options) is not None, options["workspace_mib"],
                                                 options["flashinfer"], options["flashinfer_mib"], options["decode"],
                                                 options["prompt_reader"])
    return {"device": found, "host": host}


def create(context, options, parts):
    form = device_attention.DeviceAttention(_kernels(options), options["kernels"], options["int8_qk"],
                                            options["decode"], options["partitions"], options["piece"],
                                            device_attention.engine_device(context), options["workspace_mib"],
                                            options["flashinfer"], options["flashinfer_mib"],
                                            options["prompt_reader"])
    return SoftmaxAttention(parts["kv_cache"], form)
