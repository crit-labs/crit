# Grouped experts over bundles

Run a layer's routed SwiGLU experts in the format the checkpoint stores them, for decode rows and prompt chunks alike,
from wherever the experts sit: the plugin's own loaded experts or a residency cache's slot store. Each expert is one
contiguous **bundle**; a call names each expert's bundle through a **slot table**, groups the routes by expert on the
device, runs every expert's gate/up and down rows once for all of its routes, and sums each token's routes in route
order. A route's result is one function of the route and its expert: the same bits alone or batched, in any chunk,
in a decode row or a prompt chunk, so batched decoding matches serial decoding and prompts may share passes.

## When to use it

Use it as the experts slot's plugin of a model with routed SwiGLU experts (top-k routing, a [2 * inter][hidden]
gate/up matrix with the gate rows first, a [hidden][inter] down matrix, both multiples of 64) in bf16, fp8 with
channel or 128 x 128 block scales, nvfp4, int4 with an E4M3 scale per 16, mxfp4, or AWQ / GPTQ int4 with a zero
point a group (bundle format `int4_zp:<group>`, loaded as `awq` or `gptq` tensors `qweight`, `qzeros`, `scales`;
act order is refused, since the kernel reads x and h in order). It serves at candidate 1: its
arithmetic is the stored format's, its decode and prompt paths are one kernel, and a call at a fixed token count is
captured in a CUDA graph (rounds:experts).

Don't use it for experts that are not SwiGLU or shapes that are not multiples of 64; a provider's grouped GEMM that is
faster on the target is worth measuring against it, but only one that keeps a route's bits independent of the rows
beside it can replace it without changing what batched requests decode.

## How it works

**Bundles** (code/bundles.py). `layout(format, hidden, inter)` gives one expert's components and their offsets, each on
a 16-byte boundary, the matrices row major exactly as stored (so a CPU tier reads a bundle in place); `pack(gate_up,
down, format)` builds a bundle from what `load()` takes: a bfloat16 tensor, or the stored tensors by role (`weight`,
`scale`, `global_scale`, `input_scale`, in crit's divisor convention for the global and input scales). It copies bytes
and casts nothing: a byte component comes as raw bytes or its own dtype (float8_e4m3fn, float4_e2m1fn_x2,
float8_e8m0fnu). Per-row factors (1 / global scale, fp8 channel scales) and the input scales live in the bundle, so an
expert moves between slots, stores and tiers as one block; fp8:block keeps gate's and up's 128 x 128 scale grids apart
(an inter of 64 mod 128 included), and an E8M0 byte is 2^(byte - 127), byte 255 NaN, as the format defines.
`layout()` is the one definition of a bundle: a store builder or a CPU tier calls it through experts:bundles and keeps
no copy.

**The slot table.** `forward(layer, x, ids, weights)` runs the layer's loaded bundles (a load after the layer's first
forward writes into its store in place). `forward_bundles(layer, x, ids, weights, store, slots, format)` runs a store
`[slots][bundle_bytes]` of `format` bundles with `slots [experts]` int32: a slot >= 0 holds the expert, -2 means
another tier computes it (its routes are skipped here), -1 that it is not routed. `route_outputs(layer, x, ids, store,
slots, format)` gives each route's unweighted rows and `combine` sums a token's routes in route order with
`fmaf(weight, row, sum)`: a CPU tier writes its routes' rows into the same rows and the combine gives the bits an
all-device call gives (experts:routes).

**Activations, layer by layer.** With `activations: nvfp4` each NVFP4 layer runs as its stored matrices declare: W4A4
where they carry `input_scale`, W4A16 where they do not, so a checkpoint whose main experts declare W4A4 beside
experts that do not (an MTP layer's, say) runs each as declared. A layer's experts declare it at `load()`, or at
`pack(gate_up, down, format, layer=layer)` for a store; an expert's two matrices, and a layer's experts, must agree. The
first eager call over a store for a layer reads the input scales of the bundles its slot table names: all of them
carry theirs or none does, and they match what the layer's loads or packs declared (a W4A4 layer's bundles packed
without their scales are refused, never run as zeros); without a declaration the bundles decide. The facts name each
layer's activations (`experts.layers`). `activations: bf16` runs every layer W4A16.

**The arithmetic.** crit's kernel grouped-experts (`kernels/techniques/experts/grouped-experts`: BF16 tensor cores, sm_80 and
newer), built for the engine's device with the engine's tuning or its default knobs (no knob changes a route's bits):

- the routes sorted by expert, stable (a one-block sort for decode-sized calls, a stable device sort beyond 1,024
  routes), cut into items of 8 routes (decode) or 32 (prompt chunks), every grid launched at its largest size and
  exiting on the device's item count: no host synchronization;
- each weight decoded exactly to BF16 (E4M3 codes; E2M1 or INT4 times its E4M3 block scale; E2M1 times its E8M0
  scale), activations in BF16; a warp owns 16 rows of one expert and every route of its item, and each output is one
  chain of m16n8k16 MMAs over k in ascending 64-blocks whatever the item size; fp8 block scales multiply each 128-k
  block's sum into the running one; row factors multiply the FP32 sums;
- crit's declared SwiGLU (IEEE operations only), the intermediate in BF16;
- in a W4A4 layer (NVFP4 experts carrying input scales, with `activations: nvfp4`), the inputs and the intermediate
  quantized per route at the expert's input scales by crit's declared quantizer (host-expert-arithmetic's
  quantize_e2m1x2), the operands e2m1 times the decoded block scale;
- the down rows' FP32 outputs per route, then the route-order combine.

Where the kernel is not there (it does not build on the machine, a CPU device) the plugin runs the same operands
and factors in torch (code/fallback.py): each route's bundle gathered and decoded, batched matrix products in chunks of
one fixed size (so a route runs the same shapes in any call) with BF16 operands and FP32 sums chosen per call (no
global precision flag read or set), the declared quantizer and SwiGLU. It is correct, row invariant and capture-safe,
and much slower; the engine's facts say which one runs (`kernels`, and the plugin's `experts.path`).

**The declared arithmetic** (`arithmetic: declared`) is crit's integer block order (host-expert-arithmetic's
host_int.h) for nvfp4 W4A4 and int4 W4A8 experts: the bits crit's CPU expert tier computes on the same bundles, so a
dispatch that sends some routes to the CPU changes no token. Only the kernel computes it: without the kernel the
plugin refuses the option, naming the kernel and why it is not here.

A long prompt chunk runs in parts of the tokens `workspace_mib` holds (the route rows of a call, at most 65,535 kernel
items a part), which changes no route's bits; `forward_bundles` combines each part's routes into its tokens' rows as
it goes, so a call never holds a row of every route at once (an 8,192-token chunk of a top-10 model at a hidden size of
2,560 would hold 840 MB of them). A prompt-sized call of fewer tokens than a kept prompt workspace of its shape (a
chunk's last part, a draft layer's pass over the chunk's tokens but one) runs in that workspace's leading rows rather
than one of its own. A call captured in a CUDA graph (after one eager call at its token
count) replays into its workspace, which the plugin keeps for its life. Inside a captured prompt chunk (prompts:experts,
crit_plugins.prompt_graphs) a bucket of 16, 64, 128 or 512 tokens is such a call: the model masks the padding's ids to
-1 by the batch's valid_t, and those routes read no expert (the kernel never reads a skipped route's bundle; the torch
fallback's fixed-shape gather discards what it read) and add nothing, so a padded token's output is 0 and a real
token's the unpadded call's.

## What the code supplies

- [bundles.py](code/bundles.py): `layout`, `kernel_layout`, `inter_of`, `pack`, `operands`, `matrices`, `inputs`.
- [fallback.py](code/fallback.py): `route_outputs`, `combine`, `quantize`, `swiglu`.
- plugin.py: the experts plugin (host form: expert-streamed-prefill's FP64 block on the expert_source slot's
  weights; device form as above).

## What the engine supplies

The router's ids and weights, each expert's stored tensors at load (or a residency cache's store and slot table, its
bundles made with `pack`, given the layer), the model's shared expert and residual, and the quality mode's choice of
W4A16 everywhere or each layer as its checkpoint declares.

## Worked example

Four NVFP4 experts packed into a store, a call of three tokens through the torch arithmetic (the kernel computes
the same operands on tensor cores), checked against the decoded matrices by hand, then the same experts moved to other
slots of a store and found through a slot table:

```python
import torch
import bundles, fallback

H, I = 64, 64
g = torch.Generator().manual_seed(0)


def stored(rows, cols):
    return {"weight": torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g),
            "scale": torch.full((rows, cols // 16), 0.125).to(torch.float8_e4m3fn), "global_scale": torch.tensor(4.0)}


store = torch.stack([bundles.pack(stored(2 * I, H), stored(H, I), "nvfp4") for _ in range(4)])
found = bundles.layout("nvfp4", H, I)
print(found["bundle_bytes"], list(found["components"]))
x = torch.randn(3, H, generator=g).bfloat16()
ids = torch.tensor([[0, 2], [1, -1], [3, 0]])
w = torch.tensor([[0.6, 0.4], [1.0, 0.0], [0.5, 0.5]])
out = fallback.combine(fallback.route_outputs(x, ids, store, None, found, False, torch.zeros(6, H)), ids, w)
gate_up, down = bundles.matrices(store, found)
xt = x[0].float()
by_hand = sum(w[0, j] * (down[e] @ (torch.nn.functional.silu(gate_up[e][:I] @ xt) * (gate_up[e][I:] @ xt)))
              for j, e in enumerate(ids[0].tolist()))
assert (out[0] - by_hand).abs().max() <= 1e-2 * by_hand.abs().max()
slots = torch.tensor([2, 3, 0, 1], dtype=torch.int32)   # expert e now in slot slots[e]
moved = torch.zeros_like(store)
moved[slots.long()] = store
again = fallback.combine(fallback.route_outputs(x, ids, moved, slots, found, False, torch.zeros(6, H)), ids, w, slots)
assert torch.equal(again, out)
```

## Costs and validation

tests/test_plugin.py: the host form equals experts-reference on crit's experts cases. tests/test_device.py, through
the kernel (where it builds) and the fallback: every format within 5e-3 of the largest output of an FP64 block over the
same decoded weights (3e-2 for W4A4, where an FP32 intermediate a rounding away from an E2M1 boundary takes the other
code); a route's bits the same alone, batched and in prompt-sized calls; a permuted slot store and another tier's rows
joined in route order give the all-device bits; a captured call replays new routes and its workspace outlives later
prompt shapes; a chunk in parts and a shorter prompt-sized call keep the bits of one call in one prompt workspace; a load after a forward keeps the layer's other experts; E8M0 bytes 0 and 255 decode alike on both
paths; a model with a W4A4 layer and a W4A16 layer runs each as declared, loaded and from one store; unknown formats
and shapes, casts, bundles and stores that do not carry what their layer declares and the declared arithmetic without
the kernel are refused.

Measured on an RTX 5090 with every expert in device memory (the kernel's card has the table): a decode layer of 512
NVFP4 experts (2560 x 640, top 10) takes 42 us for one token and 131 us for four (about 0.66-0.86 TB/s of weights);
128 FP8 experts (4096 x 1536, top 8) move 1.25-1.42 TB/s in decode and run prompt chunks at 96-112 TFLOPS. The torch
fallback takes 20 ms for one token of the NVFP4 shape: a correct path, not a fast one.
