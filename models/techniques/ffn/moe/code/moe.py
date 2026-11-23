"""A mixture of experts over the router and experts slots (any experts plugin, expert-offload included), a shared expert and its gate; parallel-moe: a dense MLP and the experts side by side.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `moe`
and composes the block moe, parallel-moe a layout names (README.md)."""
from ..template.blocks import _tensor
from ..swiglu.swiglu import SwiGLU


def _mx_values(torch, stored):
    """An MXFP8 matrix's exact values in BF16: each E4M3 code times 2 ** (its E8M0 scale - 127), one scale a 32."""
    codes, scales = stored["weight"], stored["scale"]
    factors = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)[..., :codes.shape[-1]]
    values = codes.float() * factors
    found = values.to(torch.bfloat16)
    if not torch.equal(found.float(), values):
        raise ValueError("model-template: an MXFP8 matrix's values do not fit BF16 exactly")
    return found.contiguous()


def router_rule(moe, defaults=None):
    """The routing rule a checkpoint's config states, in topk-router's option names; defaults: what the family's
    reference assumes where config.json is silent (the layout's `defaults`: norm_topk_prob)."""
    defaults = defaults or {}
    scoring = moe.scoring_func or defaults.get("scoring_func") or "softmax"
    renormalize = moe.norm_topk_prob if moe.norm_topk_prob is not None else defaults.get("norm_topk_prob")
    rule = {"scoring": scoring, "score_before_selection": True, "renormalize": bool(renormalize),
            "scaling": float(moe.routed_scaling_factor or 1.0)}
    if moe.n_group and moe.n_group > 1:
        rule.update(groups=moe.n_group, selected_groups=moe.topk_group or 1)
        if defaults.get("group_score_top"):   # a group's score: the sum of its this many best experts
            rule["group_score_top"] = int(defaults["group_score_top"])
    return rule


class MoE:
    """A mixture of experts: the router slot's selection over the router's logits (the config's rule, checked against
    the router plugin's), the experts slot's routed SwiGLU block over every expert the layer loads (per-expert gate,
    up and down matrices, or fused [experts] tensors), and the shared expert where the checkpoint has one (gated by
    sigmoid of its own projection where it stores that gate). The experts plugin may hold the experts anywhere
    (expert-offload: device, host memory or a file): with `experts:offload` the model plans no device bytes for them
    and calls begin() before the loads and open() after its pools."""

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        self.layer = layer
        self.slot = m.expert_ids[layer]   # the experts slot's id of the layer
        moe = m.config.moe
        self.top, self.count = moe.experts_per_token, moe.experts
        self.router = m.matrices.load([prefix + spec.get("router", "gate")], "router")
        bias = _tensor(m, prefix + spec.get("router", "gate") + ".e_score_correction_bias", False)
        self.bias = None if bias is None else bias.float()
        self.shared = self.shared_gate = None
        for name in ("shared_expert.", "shared_experts."):
            if m.matrices.has(prefix + name + "gate_proj"):
                self.shared = SwiGLU(m, layer, prefix + name, {})
        if m.matrices.has(prefix + "shared_expert_gate"):
            self.shared_gate = m.matrices.load([prefix + "shared_expert_gate"], "shared_gate")
        self.prefix = prefix + spec.get("experts", "experts.")
        # A gate_up stack that keeps each gate row beside its up row (the spec's gate_up: interleaved)
        self.interleaved = spec.get("gate_up") == "interleaved"
        # The routing weights at the hidden dtype, as most references round them, or FP32 (the spec's weights)
        self.fp32_weights = spec.get("weights") == "fp32"

    def load_experts(self):
        """Every routed expert of the layer into the experts slot: per-expert matrices, or a fused gate_up_proj
        [experts][2 * inter][hidden] and down_proj [experts][hidden][inter]."""
        m = self.m
        # An experts plugin that packs its bundles on the host (host_loads: expert-offload) takes host tensors: on
        # the device first, every expert crossed the link twice on its way to host memory.
        device = "cpu" if getattr(m.experts, "host_loads", False) else m.device
        hold = getattr(m.ck, "hold", None)
        if hold is not None:   # the layer's experts, read once and packed elsewhere: past the page cache, at once
            hold(self.prefix)
        try:
            self._load_experts(device)
        finally:
            if hold is not None:
                m.ck.release()

    def _load_experts(self, device):
        m, torch = self.m, self.m.torch
        fused = self.prefix + "gate_up_proj"
        if fused in m.ck:
            gate_up, down = m.ck.get(fused, device), m.ck.get(self.prefix + "down_proj", device)
            if gate_up.shape[1] == m.hidden != gate_up.shape[2]:   # stored [experts][hidden][2 * inter]: transposed
                gate_up, down = gate_up.transpose(1, 2), down.transpose(1, 2)
            for e in range(self.count):
                m.experts.load(self.slot, e, gate_up[e].contiguous().to(torch.bfloat16),
                               down[e].contiguous().to(torch.bfloat16))
            return
        if m.matrices.has(fused):   # two stacks as matrices (fused.weight with its scales): each expert as stored
            for e in range(self.count):
                fmt, gate_up, down = m.matrices.stacked_expert(fused, self.prefix + "down_proj", e, self.interleaved,
                                                               device)
                if fmt == "mxfp8" and "experts:mxfp8" not in m.parts.facts("experts"):
                    # an experts plugin without MXFP8 bundles: the values exactly, in BF16 (an E4M3 code times a
                    # power of two is a BF16 value)
                    fmt, gate_up, down = "bf16", _mx_values(torch, gate_up), _mx_values(torch, down)
                    m.__dict__.setdefault("experts_decoded", set()).add(self.layer)
                if fmt == "bf16":
                    m.experts.load(self.slot, e, gate_up, down)
                else:
                    m.experts.load(self.slot, e, gate_up, down, fmt)
            return
        # The draft layer's experts in the format the slot holds the others in, re-encoded where stored otherwise.
        from crit_plugins.encodings import SOURCES
        want = m.expert_format if self.layer == m.layers else None
        stored = m.matrices.format(f"{self.prefix}0.gate_proj")
        if stored not in SOURCES:   # read as its own format reads it (a GGUF's layers keep theirs)
            want = None
        if want is not None and stored != want:
            m.draft_experts = f"stored {stored}, re-encoded to {want}"
        for e in range(self.count):
            p = f"{self.prefix}{e}."
            fmt, gate_up, down = m.matrices.expert(p + "gate_proj", p + "up_proj", p + "down_proj", want, device)
            if fmt == "bf16":
                m.experts.load(self.slot, e, gate_up, down)
            else:
                m.experts.load(self.slot, e, gate_up, down, fmt)

    def routed(self, h, ids, weights, ctx):
        """The experts' block of rows h over their routes (FP32), through the experts slot."""
        torch, m = self.m.torch, self.m
        if m.offload or getattr(m, "phased_experts", False):
            # decode rows and prompt chunks apart: a decode row keeps its decode arithmetic whatever shares the step
            # (a step's decode rows come first, StepLayout); experts:phases plugins may give prompt rows another
            split = next((first for first, _, prompt in ctx.groups if prompt), None)
            if ctx.prompt and split and not getattr(ctx.batch, "prompt", False):
                routed = torch.cat([m.experts.forward(self.slot, h[:split], ids[:split], weights[:split], prompt=False),
                                    m.experts.forward(self.slot, h[split:], ids[split:], weights[split:],
                                                      prompt=True)])
            else:
                # Each token's row for the cache's per-row reads and hits, which a round's count_round hands to its
                # requests: a round's rows only (a step's rows are not round slots, and their counts would wait in
                # the cache for the next round's requests).
                routed = m.experts.forward(self.slot, h, ids, weights, prompt=ctx.prompt,
                                           rows=ctx.batch.rows_t if ctx.batch.round else None)
        else:
            routed = m.experts.forward(self.slot, h, ids, weights)
        return routed

    def forward(self, h, ctx):
        torch, m = self.m.torch, self.m
        ids, weights = m.router.select(ctx.mm(h, self.router).float(), self.top, bias=self.bias)
        if not self.fp32_weights:
            weights = weights.to(h.dtype).float()   # the routing weights at the hidden dtype, as the reference rounds them
        routed = self.routed(h, ids, weights, ctx)
        if self.shared is None:
            return routed.to(h.dtype)
        shared = self.shared.forward(h, ctx).float()
        if self.shared_gate is not None:
            shared = torch.sigmoid(ctx.mm(h, self.shared_gate).float()) * shared
        return (routed + shared).to(h.dtype)


class ParallelMoE(MoE):
    """A dense gated MLP and a mixture of experts side by side, each normed, summed:

        dense   = norm_1(mlp(h))                              h: the layer's normed input, as any FFN's
        r       = the residual the layer adds into            (Context.residual: a sandwich residual gives it)
        routes  = top-k of softmax(proj(rms_norm(r) * scale / sqrt(hidden))), renormalized by the router slot,
                  each weight times its expert's scale
        routed  = norm_3(experts(norm_2(r), routes))
        out     = dense + routed

    The block sits at its layer's prefix: the MLP under `mlp.`, the router's `proj`, `scale` and `per_expert_scale`
    under `router.`, the experts fused under `experts.`, the three norms `post_feedforward_layernorm_1`,
    `pre_feedforward_layernorm_2` and `post_feedforward_layernorm_2` (the spec's `dense`, `router`, `experts` and
    `norms` rename them). The MLP's and the experts' gate is the model's activation."""

    at_layer = True   # made with its layer's prefix, not the FFN's

    def __init__(self, model, layer, prefix, spec):
        spec = {**spec, "router": spec.get("router", "router.") + "proj"}
        super().__init__(model, layer, prefix, spec)
        m, torch = self.m, self.m.torch
        self.dense = SwiGLU(m, layer, prefix + spec.get("dense", "mlp."), {})
        at = prefix + spec["router"][:-len("proj")]
        self.route_scale = _tensor(m, at + "scale").float() * m.hidden ** -0.5
        self.expert_scale = _tensor(m, at + "per_expert_scale").float()
        names = spec.get("norms", ("post_feedforward_layernorm_1", "pre_feedforward_layernorm_2",
                                   "post_feedforward_layernorm_2"))
        self.norm_dense, self.norm_in, self.norm_out = (_tensor(m, prefix + n + ".weight").float() for n in names)
        self.unit = torch.ones(m.hidden, dtype=torch.float32, device=m.device)   # an RMSNorm without a weight

    def forward(self, h, ctx):
        m, r = self.m, ctx.residual
        if r is None:
            raise RuntimeError("model-template: a parallel-moe block reads the stream its layer adds into, which "
                               "this residual does not give (the layout's residual: sandwich)")
        dense = m._norm(self.dense.forward(h, ctx), self.norm_dense)
        scored = (m.norms.rms_norm(r, self.unit, m.eps).float() * self.route_scale).to(r.dtype)
        ids, weights = m.router.select(ctx.mm(scored, self.router).float(), self.top, bias=self.bias)
        weights = weights.float() * self.expert_scale[ids]
        routed = self.routed(m._norm(r, self.norm_in), ids, weights, ctx)
        return (dense.float() + m._norm(routed.to(r.dtype), self.norm_out).float()).to(h.dtype)


MoE.router_rule = staticmethod(router_rule)   # what the model checks the router plugin's options against
BLOCKS = {"moe": MoE, "parallel-moe": ParallelMoE}
