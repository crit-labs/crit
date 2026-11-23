# moe

A mixture of experts over the router and experts slots (any experts plugin, expert-offload included), a shared expert and its gate; parallel-moe: a dense MLP and the experts side by side.

## How it works

A dense gated MLP and a mixture of experts side by side, each normed, summed:

    dense   = norm_1(mlp(h))                              h: the layer's normed input, as any FFN's
    r       = the residual the layer adds into            (Context.residual: a sandwich residual gives it)
    routes  = top-k of softmax(proj(rms_norm(r) * scale / sqrt(hidden))), renormalized by the router slot,
              each weight times its expert's scale
    routed  = norm_3(experts(norm_2(r), routes))
    out     = dense + routed

The block sits at its layer's prefix: the MLP under `mlp.`, the router's `proj`, `scale` and `per_expert_scale`
under `router.`, the experts fused under `experts.`, the three norms `post_feedforward_layernorm_1`,
`pre_feedforward_layernorm_2` and `post_feedforward_layernorm_2` (the spec's `dense`, `router`, `experts` and
`norms` rename them). The MLP's and the experts' gate is the model's activation.

## Use it

A layout names it as a mixer, an FFN or a `before` block (`moe, parallel-moe`); the model plugin mounts [code/moe.py](code/moe.py) as its source `moe`, beside the sources it imports (swiglu). What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
