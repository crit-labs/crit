Routed SwiGLU experts of mixture-of-experts models over a slot store of expert **bundles**, for decode rows and prompt
chunks alike (operation `grouped-expert-projection`): crit's grouped-experts plugin (`inference/grouped-experts`)
loads it, built for the engine's device with the engine's tuning or the default knobs. Each expert is one contiguous bundle in crit's experts:bundles layout (its gate/up and
down matrices row major as stored, their scales and per-row factors); a call names each expert's bundle through a slot
table, so an expert can sit in the plugin's own store, a residency cache's slots or a staging ring.

A call groups its routes by expert on the device (a one-block stable sort for up to 1,024 routes, a stable device
sort beyond), runs every routed expert's gate/up rows, SwiGLU and down rows once for all of its routes on BF16 tensor
cores, and sums each token's routes in route order. No host synchronization: every grid is launched at its largest
size and exits on the device's item count, so a call at a fixed token count is captured in a CUDA graph.

The gate is the layout's (`Layout.activation`, the module's `ACTIVATIONS`): SiLU, or GELU in its tanh form, one
branch in the gate/up rows' epilogue (`crit_kcore`'s `silu_mul` and `gelu_tanh_mul`). The declared intermediate
(W4A4) is defined for SiLU and refuses another gate.

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
`fp4` (nvfp4 bundles with input scales, hidden and inter multiples of 128, the sm_120 family): W4A4 on the block-scaled
FP4 tensor cores (mma.sync kind::mxf4nvf4 m16n8k64, `mma_experts.cu`'s FP4 atom). Every route's gate/up inputs are quantized once by
crit's declared NVFP4 quantizer at its expert's input scale into E2M1 codes and UE4M3 block scales (the operands the
tensor arithmetic's W4A4 multiplies as BF16 values, the same values), the weights enter as their own codes and scales,
and the gate/up epilogue quantizes the intermediate the same way at the down's input scale. K is cut into k_parts(K)
parts at fixed 128-k block boundaries (4 from 2,048 k, 2 from 512, else 1: K's alone, so both structures cut it the
same); each output is one chain of MMAs a part over its 64-element blocks in order from zero, the parts' sums added in
part order in FP32. Decode items run a part a warp (a block's warps are row-tile groups of k_parts(K) warps) and add
the parts through shared memory; prompt items run the parts one after another in a warp, into a second accumulator.
The split spreads a decode pass's few row tiles over more warps (one token's grid is small): at one token on 2048 x 768
a layer went from 37.2 to 33.9 us, the BF16 path's speed, and FP8 from 68.8 to 48.4 us, while FP4 prompts lost 12-16%
to the second accumulator (Measured speed). Decode items (8 routes, operands from global memory,
`fp4-decode`), prompt items of 32 (codes and scales staged a chunk at a time, `fp4-prompt`) and of 64 (`fp4-tile`, below)
give a route the same bits, another arithmetic than `tensor`'s W4A4 (the tensor core's own sums of the same operands).
`fp4-tile` runs FP4 items of 64 routes a call asks it for (`Workspace(width=64, tile=True)`; without `tile` they run
`fp4-prompt`'s 64-route form, and the grouped-experts plugin picks between `fp4-prompt` and `fp4-tile` from the
engine's sweep report on its device, `prompt_items`) through a block tile:
WM x WN warps, warp (wm, wn) TILES row tiles of row group wm for 8 / WN of the item's n-tiles, the block's weight rows (a
128-k step's codes and scales) staged with the item's routes a step at a time through a STAGES-deep cp.async ring, so
a weight fragment leaves global memory once a block and feeds WN warps and no weight waits in registers (`fp4-prompt`'s
warps each read their rows from global memory into a ring of registers: 241 registers a thread, two blocks of 4 warps
an SM on an RTX 5090, latency-bound there). The chains, parts and part-order sums are `fp4-prompt`'s, so the bits are
too. Evidence, one GPU and shape (RTX 5090, 128 experts of 2,304 x 1,152, top 8, 4,096 tokens): gate/up 1.51 -> 1.14
ms, a layer's fp4-row experts 2.59 -> 2.13 ms, its sweep a further 8-13%, the tile L2-bound there (70-90% of its
throughput); another GPU or shape may take `fp4-prompt`, which stays beside it for the sweep and the plugin to choose. A block holding a NaN
or an infinity gets the format's NaN scale, so its route carries NaN.
`fp8` (fp8:channel and fp8:block bundles, hidden and inter multiples of 128, 8.9 on): W8A8 on the E4M3 tensor cores
(mma.sync e4m3 m16n8k32; kind::mxf8f6f4 with unit scales on the sm_120 family), the same mainloop (`mma_experts.cu`) with
its FP8 atom. Every route's inputs and the intermediate (after the gate/up epilogue writes it in BF16) are quantized
by crit's declared group quantizer (each 128 values of a row in E4M3 at amax / 448, quantized-projections'
`fp8_block.h` rule); K is cut into the same k_parts(K) parts as `fp4`'s, and each 128-k block's sum (four MMAs from
zero) is folded into its part's running sum with one fmaf by the activation's scale times the weight's (fp8:block's
grid entry; fp8:channel's row scale multiplies the finished sum), the parts' sums added in part order in FP32.
`fp8-decode` and `fp8-prompt` give a route the same bits.

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
# rows=f: its row buffers come from f(name, count, width, dtype), for a caller whose shapes share them
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

| Structure | Items | Tuned at (tokens) | WARPS | TILES | DEPTH | CHUNK | READ / FETCH |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `decode` | 8 routes, operands from global memory | 1, 2, 4, 8, 32 | 2, 4, 8 (4) | 1, 2, 4 (1) | 0-4, 6, 8 (0) | | 0, 1, 2 (0) |
| `prompt` | 32 routes, operands staged by cp.async | 256, 1,024, 4,096 | 2, 4, 8 (4) | 1, 2, 4 (2) | 0-4 (0) | 64, 128, 256 (256) | |
| `prompt-wide` | 64 routes (`Workspace(width=64)`) | 1,024, 4,096 | 2, 4, 8 (4) | 1, 2, 4 (1) | 0-4 (0) | 64, 128 (128) | |
| `fp4-decode` | 8 routes, FP4 MMA, operands from global memory | 1, 2, 4, 8, 32 | 2, 4, 8 (4) | 1, 2, 4 (1; gate/up 2) | 1-4, 6 (3) | | FETCH 0, 1 (1 at one token, else 0) |
| `fp4-prompt` | 32 or 64 routes, FP4 MMA, staged by cp.async | 256, 1,024, 4,096 | 2, 4, 8 (4) | 1, 2, 4 (2) | 1-4 (2) | 128, 256, 512 (256) | |
| `fp4-tile` | 64 routes a call asks it for (tile), FP4 MMA, weights and routes staged by cp.async (WM, WN, STAGES below) | 2,048, 4,096, 8,192 | | 2, 4 (2) | | | |
| `fp8-decode` | 8 routes, E4M3 MMA, operands from global memory | 1, 2, 4, 8, 32 | 2, 4, 8 (4) | 1, 2, 4 (1) | 1-4, 6 (4) | | FETCH 0, 1 (1 at one token, else 0) |
| `fp8-prompt` | 32 or 64 routes, E4M3 MMA, staged by cp.async | 256, 1,024, 4,096 | 2, 4, 8 (8) | 1, 2, 4 (4) | 1-4 (1) | 128, 256, 512 (128) | |
| `declared` | the integer block order | | | | | | |
| `plan-combine` | grouping and the route-order sum | | | | | | |

fp4-tile: WM (1, 2, 4; 4) warps along the block's matrix rows, WN (1, 2, 4; 2) along the item's routes, STAGES
(2, 3, 4; 2) 128-k steps in its ring; its defaults (TILES 2, STAGES 2: 36 KiB a block) fit every GPU's shared memory
without opting in, and the sweep tunes the rest per GPU and shape. WARPS: warps a block (fp4-decode, fp8-decode: WARPS / k_parts(K) row-tile groups, at least one, of k_parts(K) warps
each, one a K part, so a pass of 2,048 k at WARPS 2 runs 4 warps a block); TILES: MMA row tiles a warp owns (gate/up 8 inter rows each, down 16 output rows), sharing each
B fragment (fp4 gate/up at least 2: a warp holds the intermediate's 16-row blocks); DEPTH: 64-k blocks of weights in
flight a warp (0 is the format's own depth, 2 to 6; fp4: 128-k steps, fp8: 64-k steps); CHUNK: k values of the
inputs staged a pass (two halves of shared memory); READ: how a decode warp asks for its weights, 1 a 64-k block at a
time (32 bytes of a row of a 4-bit format, 64 of FP8), 2 128 bytes of a row at a time (4 blocks of a 4-bit format, 2
of FP8, loaded together; BF16 is 128 bytes a block either way), 0 the launch's choice (2 for a pass of 256 k values or
more). FETCH (fp4-decode, fp8-decode): 0, a lane loads 16 bytes of a step's row and a quad transpose gives it its
fragment words; 1, it loads its four fragment words directly, the same registers with fewer instructions (one token
of 2048 x 768: FP4 1.34x, FP8 1.17x a layer before K was split; level or slower from 2 tokens, fetch_ab in the
evidence). Defaults in parentheses: the launches the kernel had before it had knobs, but for READ, which before was 1. No
knob changes a route's MMA chain or any other operation (no order knob), so every configuration gives the same bits; the sweep keeps one only when every route equals the untuned build's. `declared` and `plan-combine` have
no knobs. Which width a call takes stays the caller's (`width_for`: 32 from 8 routes an expert, else 8; or
`Workspace(width=..., tile=...)`); the grouped-experts plugin's `prompt_items` chooses an FP4 prompt call's from the
engine's sweep report on its device.

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
| nvfp4 W4A16, 512 x top 10, 2560 x 640 | 30.4 (916) / 48.9 (1136) / 84.9 (1310) / 152.3 (1460) | 3.84 (52.4) / 10.25 (78.6) |
| nvfp4 W4A4, same | 33.9 (820) / 52.8 (1052) / 90.1 (1234) / 159.6 (1394) | 4.17 (48.3) / 10.60 (75.9) |
| mxfp4, same | 27.8 / 44.7 / 78.1 / 140.2 | |
| int4, same | 30.5 / 48.7 / 84.8 / 152.0 | |
| fp8 channel, 113 x top 8, 4096 x 1536 | 109.1 (1386) / 189.6 / 340.7 / 589.7 | 6.42 (96.3) / 22.16 (111.6) |
| fp8 128 x 128 block, same | 106.7 (1416) / 184.0 / 334.2 / 581.0 | 7.67 (80.6) / 23.97 (103.2) |
| bf16, 128 x top 8, same shape | 196.4 (1538) / 369.8 (1585) / 668.2 (1642) / 1183.2 (1658) | 6.58 (94.0) / 22.91 (108.0) |

The decode columns of the 4-bit and FP8 rows are with READ's default (128 bytes of a row a request), measured
2026-11-15 on the same GPU (the FP8 store holds 113 experts in the bench's 2 GB, so calls of 2 tokens and more reuse
some and their GB/s is not given). With READ 1, the kernel's reads until then, the same runs took: nvfp4 W4A16 42.9 /
74.3 / 131.7 / 243.2 us (648 to 914 GB/s), W4A4 46.8 / 78.1 / 136.6 / 248.4, mxfp4 43.9 / 74.5 / 132.6 / 244.1, int4
43.0 / 75.0 / 132.8 / 244.9, fp8 channel 122.2 / 227.0 / 398.9 / 681.3, fp8 block 141.7 / 263.3 / 464.4 / 787.6: a
layer call is 29 to 43% shorter for the 4-bit formats, 11 to 30% for FP8.

Per-kernel (nvfp4 W4A16, 512 experts, weights from DRAM), one decode token: gate/up 16.8 us, down 10.2, plan 2.9,
combine 2.2 (READ 1: gate/up 28.4, down 10.5); gate/up at 2 / 4 / 8 tokens 28.2 / 50.2 / 95.9 us (READ 1: 51.8 / 93.2
/ 182.5). A decode pass is a few warps a multiprocessor (one token's gate/up: 800 warps on 170), each waiting on its
own chain of weight requests, and what a request brings is the pass's rate: 32 bytes of a row a request moved the
weights at 0.6 to 0.9 TB/s, 64 bytes (FP8's blocks, or two 4-bit blocks) at 1.25 to 1.4, 128 bytes at 1.4 to 1.7, where
the BF16 experts (128 bytes a block) always were; 192 and 256 bytes a request are no faster, and neither are more
blocks in flight (DEPTH), a prefetch instruction or copies ahead through shared memory.

**The `fp4` and `fp8` arithmetics** (RTX 5090, 2026-11-21, default knobs, nvfp4 W4A4 and fp8 experts of 128 x top 8 at
2048 x 768 and 64 x top 8 at 4096 x 1536; decode: a layer, gate/up and down, CUDA graph replays after a 256 MB L2
flush, us; in parentheses the same structures before K was split):

| Shape, tokens | BF16 path, nvfp4 W4A4 | `fp4` | BF16 path, fp8 W8A16 | `fp8` |
| --- | --- | --- | --- | --- |
| 2048 x 768, 1 | 33.6 | 33.9 (37.2) | 39.7 | 48.4 (68.8) |
| 2048 x 768, 16 | 189.0 | 195.7 (241.4) | 292.2 | 302.5 (392.3) |
| 4096 x 1536, 1 | 84.1 | 97.5 (98.0) | 113.8 | 147.2 (178.0) |
| 4096 x 1536, 16 | 455.8 | 449.1 (605.2) | 708.9 | 814.3 (967.5) |

FP8 decode stays at 0.77-0.97x of the BF16 path's W8A16: it reads twice FP4's weight bytes and folds a scale every
128 k. FP4 prompts (a gate/up pass) take 603 us at 2,048 tokens and 2,046 at 8,192
on 2048 x 768 (the BF16 path 1,115 and 3,980) and 7,653 at 8,192 tokens on 4096 x 1536 (15,493): 1.8-2.0x the BF16
path, 12-16% slower than before K was split (539, 1,760, 6,590). Both arithmetics stay opt-in (the plugin's
`arithmetic`).

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
