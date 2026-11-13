"""The plain projection: a dense matrix, y[t][n] = fsum(W[n][k] * x[t][k]) as the host form; on the device, a BF16
matrix multiplied by torch (cuBLAS). cuBLAS picks its kernel by the row count, so a call of up to DECODE_ROWS rows
(decode rows, verify windows) multiplies one row at a time: each row's bits are those of the row alone, as the
projections slot's device form promises, at the cost of reading the matrix once a row. Its matmul does no host work
and allocates only through torch, so captured decode rounds run it (rounds:projections)."""
import math

DECODE_ROWS = 8


class DenseProjections:
    def prepare(self, rows):
        rows = [[float(v) for v in row] for row in rows]
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("a nonempty rectangular matrix")
        return rows

    def project(self, weight, activations):
        if any(len(x) != len(weight[0]) for x in activations):
            raise ValueError("activation width differs from the matrix columns")
        return [[math.fsum(w * a for w, a in zip(row, x)) for row in weight] for x in activations]

    def dense(self, weight):
        return [list(row) for row in weight]

    def load(self, tensors, format, role=None, encode=None):
        """The device form: a BF16 matrix [out][in] from a 16- or 32-bit weight (whatever its role). encode (a
        draft-only matrix's re-encoding) other than bf16 is quantized-projections'."""
        import torch
        if format != "bf16":
            raise ValueError(f"projections-dense loads bf16 matrices; {format} is quantized-projections'")
        if encode not in (None, "bf16"):
            raise ValueError(f"projections-dense keeps matrices bf16; re-encoding to {encode} is quantized-projections'")
        return tensors["weight"].to(torch.bfloat16).contiguous()

    def matmul(self, x, matrix, out_dtype=None, prompt=False):
        """y = x W^T in BF16, or in out_dtype (FP32 for logits) from the same FP32 sums. prompt: a prompt chunk, which
        multiplies as one GEMM at every length (a short one is not split into decode rows)."""
        import torch
        x = x.to(torch.bfloat16)
        if x.shape[0] <= DECODE_ROWS and x.shape[0] > 1 and not prompt:
            return torch.cat([self.matmul(x[r:r + 1], matrix, out_dtype) for r in range(x.shape[0])])
        if out_dtype is not None and out_dtype != torch.bfloat16:
            if not x.is_cuda:   # torch has no CPU kernel for a BF16 GEMM into FP32: the same BF16 products in FP32
                return (x.float() @ matrix.float().t()).to(out_dtype)
            return torch.mm(x, matrix.t(), out_dtype=out_dtype)
        return torch.nn.functional.linear(x, matrix)

    def facts(self):
        """Calls of up to DECODE_ROWS rows keep each row's bits; longer ones (prompt rows) do not promise it."""
        return {"row_invariance": {"rows": DECODE_ROWS, "prompt_rows": False}}


def create(context, options, parts):
    return DenseProjections()
