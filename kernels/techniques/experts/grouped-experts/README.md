Routed SwiGLU experts of mixture-of-experts models over a slot store of expert **bundles**, for decode rows and prompt
chunks alike (operation `grouped-expert-projection`): crit's grouped-experts plugin (`inference/grouped-experts`)
loads it, built for the engine's device with the engine's tuning or the default knobs. Each expert is one contiguous bundle in crit's experts:bundles layout (its gate/up and
down matrices row major as stored, their scales and per-row factors); a call names each expert's bundle through a slot
table, so an expert can sit in the plugin's own store, a residency cache's slots or a staging ring.

A call groups its routes by expert on the device (a one-block stable sort for up to 1,024 routes, a stable device
sort beyond), runs every routed expert's gate/up rows, SwiGLU and down rows once for all of its routes on BF16 tensor
cores, and sums each token's routes in route order. No host synchronization: every grid is launched at its largest
size and exits on the device's item count, so a call at a fixed token count is captured in a CUDA graph.

**A route's bits never depend on the rows beside it**: a warp owns rows of one expert and every route of its item,
and each output is one chain of MMAs over k in ascending 64-element blocks whatever the item size, the chunk or the
other routes. Decode rows (items of 8 routes: 8 inter rows a warp, 4-bit codes decoded through a shared byte table,
operands read straight from global memory) and prompt chunks (items of 32 routes, 64 on request: operands staged in
shared memory by cp.async, the next chunk landing while the current one computes) compute the same function bit for
bit. W4A4 inputs are quantized once per route (a pre-pass) for both.

**Two arithmetics.** `tensor` (the default): BF16 tensor cores for every format. `declared`: crit's integer block order
(host-expert-arithmetic's host_int.h, which crit's CPU expert tier computes) for nvfp4 W4A4 and int4 W4A8, so a route
computed on the CPU has the device's bits: exact int32 block dots (dp4a), block b to lane b % 16, fmaf of the dot by the
rounded product of the scales, the halving tree, times the row's post; the declared SwiGLU and quantizers.

**A baseline to replace**: measure a provider's grouped GEMM on the format's own tensor cores against it on the
engine's shapes, and keep the faster one that keeps the same row invariance.

**Built from the kernel core** (`kernels/core`): the format decoders, the BF16 MMA, cp.async staging, the block scan and
halving tree, crit's NVFP4 quantizer steps and the declared SwiGLU are `crit_kcore`'s blocks; the kernel holds the
expert pass, the plans and the declared order.

## Applicability

| | |
| --- | --- |
| Weight formats | bf16; fp8 E4M3 with one FP32 scale per row or per 128 x 128 block (gate's and up's block grids apart); nvfp4 (E2M1, E4M3 scale per 16, a 1 / global scale per row); int4 (two's complement, E4M3 scale per 16, optional 1 / global scale per row); mxfp4 (E2M1, E8M0 scale per 32: 2^(byte - 127), byte 0 is 2^-127 and byte 255 NaN); int4_zp (format 6: INT4 with a zero point a group, AWQ and GPTQ in crit's rows as decode-gemv's repack_awq / repack_gptq write them: unsigned codes, FP16 scales in *_scale and uint8 zero points in *_inverse, [rows][K / group] each, the group a 12th layout integer, a multiple of 16 dividing hidden and inter; a value (code - zero) * scale does not fit BF16, so it enters as two BF16 parts, hi + lo exactly, each MMA run twice on them) |
| Activations | BF16 (W4A16 / W8A16 / W16A16); nvfp4 W4A4: inputs and the intermediate quantized per route by crit's declared NVFP4 quantizer at each expert's input scales |
| GPUs | NVIDIA sm_80 and newer (mma.sync m16n8k16 BF16, no architecture-specific instruction; E4M3 conversion in hardware from sm_89) |
| Shapes | hidden and inter multiples of 64; any token count up to 65,535 items a call, any top-k, up to 65,535 experts (MAX_EXPERTS) |
| Numerics | weights decoded exactly to BF16 (every format's code times its block scale is exact in BF16; int4_zp's as two BF16 parts); FP32 MMA sums; per-row factors on the FP32 sums; fp8 block scales fmaf'd per 128-k block; crit's declared SwiGLU (IEEE operations only); the intermediate rounded to BF16 (or quantized, W4A4); the combine sum = fmaf(weight, row, sum) in route order |
| Determinism | the same bits on a repeat, alone or batched, in 8-, 32- and 64-route items, through either plan, and on graph replay (tests); the declared arithmetic equal to an exact oracle of host_int's order (tests) |
| Accuracy | within 5e-3 of the largest output of an FP64 block over the same decoded weights (3e-2 for W4A4) in every format (tests, RTX 5090) |
| Measured speed | see below |

## Interface

```python
from crit_kernels.build import load      # a plugin: crit_plugins.kernels.load_kernel("grouped-experts", folder=...)
gx = load("grouped-experts")
ws = gx.Workspace(tokens, top, experts, hidden, inter, device)     # the buffers of one call shape, made once
out = gx.forward(x, ids, weights, store, slots, layout, ws, a4)    # [tokens][hidden] FP32
y = gx.route_outputs(x, ids, store, slots, layout, ws, a4)        # unweighted rows [tokens * top][hidden]
y = gx.route_outputs(x, ids, store, slots, layout, ws, arithmetic="declared")   # nvfp4 W4A4 / int4 W4A8, CPU-exact
out = gx.combine(y, ids, weights, slots, experts)                  # route order; negative slots skipped
values, scales = gx.quantize_e2m1x2(x, global_scale)               # crit's declared NVFP4 quantizer
```

`layout` is `[format, bundle bytes, gate_up, gate_up_scale, gate_up_inverse, down, down_scale, down_inverse, input,
hidden, inter]` (byte offsets, -1 absent; formats 0 bf16, 1 fp8 channel, 2 fp8 block, 3 nvfp4, 4 int4, 5 mxfp4, 6
int4_zp with its group as a 12th integer), as
crit's `bundles.kernel_layout` gives it. x is BF16 [tokens][hidden], ids int64 or int32 [tokens][top], weights FP32,
store uint8 [slots][bundle bytes] (16-byte aligned), slots int32 [experts] or None (expert e in slot e). A route whose
id is out of range or whose slot is negative is skipped (its rows of y keep what they held).

## Structures and knobs

One structure per item width, each keyed by the format, W4A4 or not, the pass (gate/up or down), hidden and inter, and
the call's token count:

| Structure | Items | Tuned at (tokens) | WARPS | TILES | DEPTH | CHUNK |
| --- | --- | --- | --- | --- | --- | --- |
| `decode` | 8 routes, operands from global memory | 1, 2, 4, 8, 32 | 2, 4, 8 (4) | 1, 2, 4 (1) | 0-4, 6, 8 (0) | |
| `prompt` | 32 routes, operands staged by cp.async | 256, 1,024, 4,096 | 2, 4, 8 (4) | 1, 2, 4 (2) | 0-4 (0) | 64, 128, 256 (256) |
| `prompt-wide` | 64 routes (`Workspace(width=64)`) | 1,024, 4,096 | 2, 4, 8 (4) | 1, 2, 4 (1) | 0-4 (0) | 64, 128 (128) |
| `declared` | the integer block order | | | | | |
| `plan-combine` | grouping and the route-order sum | | | | | |

WARPS: warps a block; TILES: MMA row tiles a warp owns (gate/up 8 inter rows each, down 16 output rows), sharing each
B fragment; DEPTH: 64-k blocks of weights in flight a warp (0 is the format's own depth, 2 to 6); CHUNK: k values of the
inputs staged a pass (two halves of shared memory). Defaults in parentheses: the launches the kernel had before it had
knobs. No knob changes a route's MMA chain or any other operation (no order knob), so every configuration gives the
same bits; the sweep keeps one only when every route equals the untuned build's. `declared` and `plan-combine` have
no knobs. Which width a call takes stays the caller's (`width_for`, or `Workspace(width=...)`).

## Build, test, tune

```sh
crit kernels build grouped-experts                                # this GPU, default knobs (or --tuning FILE)
python -m pytest kernels/techniques/experts/grouped-experts/tests -q         # every format against the FP64 block, invariance
crit kernels sweep grouped-experts --model CHECKPOINT --out TUNING_DIR [--structure decode]
```

Run the tests on the target first: every format against the FP64 block, bitwise invariance, slot tables, capture, the
quantizer and a build with other knobs. The sweep's benches (`bench.py`) take the experts' format, hidden, inter,
count and top-k from the checkpoint and time each pass on a store of random bundles of that format (at most 2 GB of
them), the routes planned by the untuned build. A sweep build compiles only the structure it tunes.

## Measured speed

RTX 5090 (sm_120), default knobs (torch 2.14.1, CUDA 13.1), every expert in device memory; torch.mm BF16 8192^3 runs
207.5 TFLOPS on the same GPU. Decode: a CUDA graph of 48 layer calls, each with its own uniform random routes over one
store of every expert (so a layer's experts are mostly not in L2), time per layer call (plan, gate/up, down,
combine) and the weight bytes it moves. Prompt: one call of a chunk.

| Experts | Decode 1 / 2 / 4 / 8 tokens (us, weight GB/s) | Prompt 2,048 / 8,192 tokens (ms, TFLOPS) |
| --- | --- | --- |
| nvfp4 W4A16, 512 x top 10, 2560 x 640 | 42.2 (659) / 73.0 (753) / 131.3 (821) / 242.8 (858) | 3.84 (52.4) / 10.25 (78.6) |
| nvfp4 W4A4, same | 45.9 (605) / 76.8 (715) / 136.2 (791) / 247.9 (840) | 4.17 (48.3) / 10.60 (75.9) |
| mxfp4, same | 41.3 / 72.0 / 127.2 / 237.4 | |
| int4, same | 42.0 / 72.4 / 130.3 / 242.1 | |
| fp8 channel, 128 x top 8, 4096 x 1536 | 120.6 (1254) / 226.8 (1293) / 407.1 (1349) / 692.2 (1419) | 6.42 (96.3) / 22.16 (111.6) |
| fp8 128 x 128 block, same | 140.3 (1077) / 276.6 (1059) / 481.4 (1140) / 831.1 (1180) | 7.67 (80.6) / 23.97 (103.2) |
| bf16, same | 196.4 (1538) / 369.8 (1585) / 668.2 (1642) / 1183.2 (1658) | 6.58 (94.0) / 22.91 (108.0) |

Per-kernel (nvfp4 W4A16, 512 experts): one decode token: gate/up 27.8 us, down 9.6, plan 2.9, combine 2.2. The
small experts' gate/up pass (a k loop of 2,560 against down's 640) moves weights at about 600 GB/s against down's
about 1 TB/s: the decode limit of this baseline at that shape.

## Against the library

`decode`, `prompt` and `prompt-wide` name their yardstick in `kernel.json`, which the sweep times on the same shapes
(`library_us`, `vs_library` = library time / tuned time, above 1 the kernel is faster): FlashInfer's grouped GEMM
(`flashinfer.grouped_mm.grouped_mm_bf16`, its cuDNN MoE backend, faster here than `torch.nn.functional.grouped_mm`)
on the pass's weights widened to BF16 once, the routes' inputs gathered in expert order and, for gate/up, the SwiGLU
in torch, the outputs left in expert order. No library takes every stored format against BF16 inputs one pass at a
time (FlashInfer's fused MoE runners run whole layers, most with quantized activations), so for a 4-bit format the
library reads four times the weight bytes. `declared` and `plan-combine` are not tuned and have no yardstick.

RTX 5090, torch 2.14.1, FlashInfer 0.7.3, a sweep on the nvfp4 W4A16 experts of a 512-expert, top-10 model (hidden
2,560, inter 640), random bundles, one pass a call, tuned kernel against library (us, vs_library):

| Structure | Pass | Tokens: tuned / library (vs_library) |
| --- | --- | --- |
| `decode` | gate/up | 1: 25.2 / 60.7 (2.4), 2: 45.1 / 91.5 (2.0), 4: 100.4 / 171.4 (1.7), 8: 180.2 / 325.0 (1.8), 32: 538.0 / 1,032.6 (1.9) |
| `decode` | down | 1: 12.3 / 50.1 (4.1), 2: 16.4 / 50.5 (3.1), 4: 30.7 / 159.1 (5.2), 8: 55.3 / 165.2 (3.0), 32: 155.6 / 542.9 (3.5) |
| `prompt` | gate/up | 256: 1,291.6 / 2,265.8 (1.8), 1,024: 1,390.0 / 5,355.8 (3.9), 4,096: 2,544.7 / 11,053.7 (4.3) |
| `prompt` | down | 256: 358.4 / 1,033.6 (2.9), 1,024: 466.4 / 1,123.7 (2.4), 4,096: 1,308.1 / 1,692.6 (1.3) |
| `prompt-wide` | gate/up | 1,024: 1,396.8 / 5,339.5 (3.8), 4,096: 2,391.0 / 11,073.1 (4.6) |
| `prompt-wide` | down | 1,024: 518.2 / 1,174.3 (2.3), 4,096: 1,206.3 / 1,699.0 (1.4) |

The margin is mostly bytes: the kernel decodes 4-bit weights in registers where the library streams them widened. The
down pass at 4,096 tokens, the most compute-bound case, is where the library comes closest (1.3x).
