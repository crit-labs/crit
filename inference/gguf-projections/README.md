# GGUF projections

A GGUF checkpoint's quantized matrices at their stored bits, in the projections slot: a model plugin reads each
matrix's stored rows from the file (`crit_core.model.gguf_blocks.GgufTensors`, mapped, never copied) and loads them
here as `gguf:<TYPE>`; decode rows multiply the stored blocks on crit's kernel
[gguf-rows](../../kernels/techniques/projections/gguf-rows/README.md), prompt rows the matrix decoded to BF16 through
cuBLAS or, with `prompt` exact, gguf-rows' tensor-core GEMM with every prompt row's decode-row bits.

**Why it exists.** Without it, an engine on a GGUF checkpoint decodes every quantized matrix to 16 bits at load (four
times the device memory of a 4-bit matrix, and four times the bytes a decode token reads) or writes its own block
readers and kernels. Here the weights stay as stored and every format runs one kernel through Core's and the kernel
core's decoders.

## Formats

Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and MXFP4 (`crit_core.model.gguf_blocks.TYPES`): a matrix's rows as the file stores
them, uint8 `[N][row bytes]`, or a stack of experts `[E][N][row bytes]` (`matmul_experts`). A GGUF file's dense
tensors (F32, F16, BF16) load as `bf16`. Other GGUF types are refused.

## Use

```text
reader = GgufTensors(inspect_gguf_model(checkpoint))       # crit_core.model, crit_core.model.gguf_blocks
info = reader.info("blk.0.ffn_up.weight")                   # dtype "Q4_K", shape [N][row bytes as stored]
up = projections.load({"weight": reader.rows("blk.0.ffn_up.weight")}, "gguf:" + info.dtype)
y = projections.matmul(x, up)                                # decode rows: gguf-rows' gemv; prompt rows: as `prompt` says
experts = projections.load({"weight": reader.rows("blk.0.ffn_up_exps.weight")}, "gguf:IQ3_S")
y = projections.matmul_experts(x, experts, ids)              # row t times expert ids[t]
```

The host form, runnable on its own:

```python
from host import GgufWeight, dense_rows, encode_q8_0, project_rows

rows = [[(i + j) % 7 - 3.0 for j in range(64)] for i in range(2)]
w = GgufWeight(encode_q8_0(rows), 8, 64)         # Q8_0: 2 blocks a row, 34 bytes each
assert w.blocks.shape == (2, 68)
values = dense_rows(w)                           # the stored values, as the format's decoder gives them
assert all(abs(a - b) <= 3.0 / 127 for r, v in zip(rows, values) for a, b in zip(r, v))
y = project_rows(w, [[1.0] * 64])                # exact sums over the stored values
assert len(y) == 1 and len(y[0]) == 2
print(y)
```

The tensor names, any reordering the file's converter applied (some architectures' query and key rows are permuted)
and which matrices to stack are the model plugin's: this plugin multiplies rows as they are stored.

## Arithmetic

- **Decode rows** (calls of up to 8 rows): gguf-rows' `gemv`. Each row is quantized to int8 per 32 values
  (`d = max |x| / 127`), each stored 32-value sub-block's integer dot is exact (`dp4a`), and the terms add in FP32 in a
  declared order, so a row's bits are its own whatever rows share the call; an expert stack's rows each have the
  plain call's bits on their expert.
- **Prompt rows**, option `prompt`: `dequantize` (the default) decodes the matrix to BF16 on each call (exactly, then one
  rounding) and runs cuBLAS's BF16 GEMM with FP32 sums, a prompt row's bits not its decode row's; `exact` runs
  gguf-rows' `gemm` in the decode rows' order, every prompt row with its decode row's bits (the facts'
  `prompt_exact`), which on an RTX 5090 is faster at a few hundred prompt rows and level or a little behind at
  thousands (gguf-rows' README gives the numbers).
- **Fallback**: where gguf-rows cannot be built, the stored values decoded on the host at load (Core's exact
  decoders) run as a BF16 matrix through torch, one row a call for decode rows.

## Host form

`prepare` encodes the slot's dense cases to Q8_0 (`code/host.py`: `encode_q8_0`), `dense` gives the stored values
(Core's decoders, the formats' own decoders bit for bit) and `project` sums over them exactly.

## Checked

`tests/test_plugin.py` on an RTX 5090: every format loaded from seeded stored rows; decode rows equal gguf-rows'
gemv bit for bit and a row alone equals it among others; dequantize-path prompt rows equal cuBLAS on the stored values
in BF16; exact prompt rows have the decode rows' bits; an expert stack's rows equal the plain call on their expert;
the fallback runs the host-decoded values; dense tensors and refusals.
