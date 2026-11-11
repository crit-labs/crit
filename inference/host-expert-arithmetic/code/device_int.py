"""host-expert-arithmetic's device form on CUDA: a checkpoint's stored NVFP4 or INT4 matrices (E4M3 scale per 16)
against activations quantized per block, on crit's kernel int-block-experts (built for the engine's device with the
engine's tuning or the default knobs).

- Decode rows (calls of up to DECODE_ROWS rows) on its `rows`: the integer block order declared in host_int.h, every
  output the bits crit's CPU rows give (the native rows or the torch order of declared.py), whatever rows share the
  call.
- Prompt rows, with prompt_gemm declared, on `gemm_staged`: the same order and bits. With prompt_gemm native (the
  default) W4A4 prompt rows (activations e2m1x2) on `gemm_native`: the activations by its quantize_nvfp4 (the same E2M1
  codes and E4M3 scale bytes as the declared quantizer), accumulated in the tensor cores (block-scaled FP4 MMAs on
  sm_120a for NVFP4 weights; the s8 MMA a 16-block for INT4 weights): NOT the declared order and not row invariant (a
  prompt row's bits may change with the prompt's length and the tuning, and are not the CPU rows'), within FP32
  accumulation of the exact product. W4A8 prompt rows (activations int8) stay on gemm_staged: no 4-bit tensor-core
  kind multiplies INT8 activations exactly.

Each matrix's factor of the E2M1 x 2 representation (0.25 for W4A4 NVFP4, 0.5 for one E2M1 side, 1 for INT4 x INT8) is
the table's post, and each row's inverse global scale multiplies the FP32 output after: both exact powers-of-two
scalings apart, so the output is the CPU rows' total * (factor * inverse) bit for bit. Every call runs on tensors made at
load or by the call itself (no host synchronization; a call at a fixed row count captures in a CUDA graph)."""

DECODE_ROWS = 16   # rows a decode call takes: jobs of up to 8 tokens each
JOB_TOKENS = 8


class CudaRows:
    def __init__(self, module, prompt_gemm):
        self.ib, self.prompt_gemm = module, prompt_gemm
        self._row_jobs = {}

    def row_jobs(self, tokens, device):
        """RowJob records for `tokens` decode rows of matrix 0 (made once per count and device, at load)."""
        key = (tokens, str(device))
        if key not in self._row_jobs:
            jobs = [(0, list(range(t0, min(tokens, t0 + JOB_TOKENS))), list(range(t0, min(tokens, t0 + JOB_TOKENS))))
                    for t0 in range(0, tokens, JOB_TOKENS)]
            self._row_jobs[key] = self.ib.row_jobs(jobs, device)
        return self._row_jobs[key]

    def load(self, codes, scales, inverse, weights, activations, input_scale, device):
        """The matrix's tensors on `device` and its kernel table: codes [N][K / 2], E4M3 scales [N][K / 16] (bytes,
        row major), inverse [N] FP32."""
        import torch
        codes = codes.to(device).contiguous()
        scales = scales.to(device).contiguous()
        inverse = inverse.to(device=device, dtype=torch.float32).contiguous()
        factor = (0.25 if activations == "e2m1x2" else 0.5) if weights == "e2m1x2" else \
            (0.5 if activations == "e2m1x2" else 1.0)
        table = self.ib.matrix_table([(codes, scales, factor, weights == "int4")], device)
        found = {"codes": codes, "scales": scales, "inverse": inverse, "table": table,
                 "jobs": torch.zeros(3, dtype=torch.int32, device=device), "int4": weights == "int4"}
        if input_scale is not None:
            found["input"] = float(input_scale)
        for tokens in range(1, DECODE_ROWS + 1):
            self.row_jobs(tokens, device)
        return found

    def native_prompts(self, activations):
        """Whether prompt rows of this activation format take gemm_native."""
        return self.prompt_gemm == "native" and activations == "e2m1x2"

    def matmul(self, x, matrix, out_dtype):
        """x [T][K] (CUDA) times the matrix -> [T][N] in out_dtype (x's by default) from FP32 outputs."""
        import torch
        t, ib = matrix.tensors, self.ib
        tokens, rows = x.shape[0], matrix.rows
        xf = x.float().contiguous()
        out = torch.empty(tokens, rows, dtype=torch.float32, device=x.device)
        if tokens > DECODE_ROWS and self.native_prompts(matrix.activations):
            codes, scales = ib.quantize_nvfp4(xf, t["input"])
            t["jobs"][2:3].fill_(tokens)
            ib.gemm_native(t["table"], t["jobs"].view(torch.uint8), rows, tokens, codes, scales, t["input"], out,
                           int4=t["int4"])
        else:
            if matrix.activations == "e2m1x2":
                values, scales = ib.quantize_e2m1x2(xf, t["input"])
                block = 16
            else:
                values, scales = ib.quantize_int8(xf, 32)
                block = 32
            if tokens <= DECODE_ROWS:
                ib.rows(t["table"], self.row_jobs(tokens, x.device), rows, values, scales, block, out)
            else:
                t["jobs"][2:3].fill_(tokens)
                ib.gemm_staged(t["table"], t["jobs"].view(torch.uint8), rows, tokens, values, scales, block, out)
        out.mul_(t["inverse"][None, :])
        return out.to(out_dtype or x.dtype)
