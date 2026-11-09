"""Partitioned attention in the attention slot: partitioned-attention's partials and merge (implementation.py) as the
host form. On the device, the same technique on one GPU through attention-state's device form (device_attention.py):
the registry's split-KV reader splits each row's keys into partitions (piece i of its positions to partition i mod P),
computes each partition's partial output with its log-sum-exp in parallel and merges them in partition order by their
log-sum-exps, so a long row's work spreads over the GPU and follows its length; merge() combines partials computed
elsewhere (a sequence's shards on other devices) the same way, in torch."""
from pathlib import Path

from .partitioned import implementation as partitioned
from .state import device_attention

KERNELS = "crit-labs/paged-kv-attention"


class PartitionedAttention:
    def __init__(self, kv, options, device_form):
        self.kv, self.device_form = kv, device_form
        self.ranks = [partitioned.SequencePartition(options["world"], rank, options["page_size"])
                      for rank in range(options["world"])]

    def attend(self, sequence, queries, positions, scale=None, pending=False):
        rows = self.kv.rows(sequence, pending)
        if not rows:
            raise ValueError("no rows to attend")
        width = len(rows[0][2])
        out = []
        for query, position in zip(queries, positions):
            partials = [partitioned.attention_partial(query, [row for row in rows if rank.owner(row[0]) == rank.rank],
                                                      value_width=width, query_position=position, scale=scale)
                        for rank in self.ranks]
            out.append(list(partitioned.merge_partials(partials).output))
        return out

    # -- the device form ---------------------------------------------------------------------------------------------------
    def forward(self, layer, queries, pool, batch, scale=None, blocks=None, block_size=None, parents=None, lse=False):
        """As softmax-attention's device form (device_attention.DeviceAttention.forward): the rows' keys split over KV
        partitions merged by their log-sum-exps; lse=True also returns each token's log-sum-exp, the normalizer a
        merge across devices needs."""
        return self.device_form.forward(layer, queries, pool, batch, scale, blocks, block_size, parents, lse)

    def facts(self):
        """The device form's engine facts (device_attention.DeviceAttention.facts)."""
        return self.device_form.facts()

    def close(self):
        self.device_form.close()

    def merge(self, outs, lses):
        """Partials of the same queries over disjoint keys (outs [parts][tokens][heads][dim], lses [parts][tokens]
        [heads] FP32, -inf for a part with no key) -> (out, lse): merge_partials on the device, the parts weighted by
        exp(lse - max) in part order. A token no part saw gets zeros and -inf."""
        import torch
        outs, lses = torch.stack(list(outs)).float(), torch.stack(list(lses)).float()
        top = lses.amax(0)
        sub = torch.where(top == float("-inf"), 0.0, top)
        weights = torch.where(lses == float("-inf"), 0.0, torch.exp(lses - sub))
        total = weights.sum(0)
        out = (weights[..., None] * outs).sum(0) / total.clamp(min=1e-30)[..., None]
        out = torch.where(total[..., None] > 0, out, 0.0)
        return out, torch.where(total > 0, sub + torch.log(total.clamp(min=1e-30)), float("-inf"))


def _kernels(options):
    """The registry kernel the device form reads pages with, or None (option kernels torch, or not available)."""
    if options["kernels"] != "auto":
        return None
    from crit_plugins.kernels import registry_kernel
    return registry_kernel(KERNELS, folder=Path(__file__).parent)


def memory(context, options):
    """What the device form allocates when it is made (device_attention.device_memory), exactly: the split-KV reader's
    partials where the registry kernel loads, FlashInfer's prompt workspaces where it serves, on a CUDA engine."""
    found, host = device_attention.device_memory(device_attention.engine_device(context), options["kernels"],
                                                 _kernels(options) is not None, options["workspace_mib"],
                                                 options["flashinfer"], options["flashinfer_mib"], "split",
                                                 options["prompt_reader"])
    return {"device": found, "host": host}


def create(context, options, parts):
    form = device_attention.DeviceAttention(_kernels(options), options["kernels"], decode="split",
                                            partitions=options["partitions"], piece=options["piece"],
                                            device=device_attention.engine_device(context),
                                            kernels_mib=options["workspace_mib"], flashinfer=options["flashinfer"],
                                            flashinfer_mib=options["flashinfer_mib"],
                                            prompt_reader=options["prompt_reader"])
    return PartitionedAttention(parts["kv_cache"], options, form)
