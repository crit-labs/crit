# Where kernels come from

crit hosts no compute kernels. An engine takes each operation's kernel from one of the providers
below, or writes it. Search them in this order, check each candidate with the three checks in the
[Kernels guide](README.md#find-an-implementation), and measure the one you choose against its floor
in Eval's kernel profile and against the best other provider for the same operation and arithmetic.

| Provider | Covers | How an engine uses it | License |
| --- | --- | --- | --- |
| [Hugging Face Kernel Hub](https://huggingface.co/kernels) | Community kernels for attention, GEMMs, activations, normalization, MoE and more, built per framework version and backend (CUDA, ROCm, Metal, XPU, CPU) | `kernels.get_kernel("org/name")` from Python; the repository's source for native builds | Per repository |
| [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api) | GEMMs in BF16/FP16, FP8 (per-tensor and per-row scales), block-scaled NVFP4 and MXFP8 on the GPUs that have them; epilogues | C API in the CUDA Toolkit; row-major stored codes, NVFP4 scales in its 128 x 4 tile order | NVIDIA CUDA EULA |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | GEMM and convolution templates for layouts and fusions cuBLASLt does not take, grouped GEMMs, mixed-input GEMMs (4- or 8-bit weights widened to 16 bits inside the main loop) | Header-only C++ templates instantiated by the engine | BSD-3-Clause |
| [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/) | Fused attention (SDPA), normalization | C API and graph frontend | NVIDIA cuDNN license |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Prompt and decode attention, paged KV, sliding windows, softcap | Python package; CUDA source for native builds | BSD-3-Clause |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Paged and ragged attention, FP8 KV, MLA, cascade attention, top-k / top-p / min-p sampling, some GEMMs | Python package with a C++ header library | Apache-2.0 |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Chunked prefill and recurrent decode for gated delta, Mamba-style, RWKV and other linear attention | Python package of Triton kernels (`fla.ops`) | MIT |
| [Triton](https://github.com/triton-lang/triton) | Writing a missing kernel in Python, compiled per device; ahead-of-time compilation for native engines | Python package | MIT |
| [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) kernels | Fused MoE, quantized GEMMs (GPTQ, AWQ, Marlin's 4-bit weights against 16-bit activations, FP8), attention backends, custom all-reduce | Some repackaged on the Hub (`vllm-flash-attn3`, `sgl-flash-attn3`); otherwise source inside the engine | Apache-2.0 |
| [ggml](https://github.com/ggml-org/llama.cpp) | GGUF block formats (Q4_0 .. Q8_0, K-quants, IQ) on CUDA, Metal, Vulkan, ROCm and CPU | C library and source | MIT |
| [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), oneDNN | CPU GEMMs and fused CPU operators | C libraries ([example](examples/openblas/README.md)) | BSD-3-Clause, Apache-2.0 |
| [Composable Kernel](https://github.com/ROCm/composable_kernel), [hipBLASLt](https://github.com/ROCm/hipBLASLt) | GEMMs and attention on AMD GPUs | C++ / C APIs in ROCm | MIT |
| [MLX](https://github.com/ml-explore/mlx), Metal Performance Shaders | GEMMs, attention and quantized matmuls on Apple GPUs | Python and C++ APIs | MIT |

A provider is a candidate, not a result: a library that covers the format may still lack a build
for the device (consumer Blackwell's sm_120 is the usual gap) or reduce in an order that breaks
speculative exactness. When none fits, the engine writes the kernel (Triton or CUDA) and records
why the providers above were rejected.

Prompts in the `preserve` quality mode keep 16-bit activations on a checkpoint's 4-bit layers, and
cuBLASLt multiplies NVFP4 only against NVFP4. Widening each chunk's weights to BF16 before a 16-bit
GEMM reads and writes them again; a mixed-input GEMM (CUTLASS's, or Marlin-style kernels from vLLM
and SGLang) widens them in registers instead. Measure one against the widened path on the engine's
own shapes before choosing; any GPU with 16-bit tensor cores runs either.

On one RTX 5090 (sm_120), for example, cuBLASLt's FP8 GEMM ran Qwen3.8-27B's prompt projections at
410-465 TFLOPS at 2,048 tokens, against 220-300 for a hand-written stored-tile kernel an earlier
crit version shipped.
