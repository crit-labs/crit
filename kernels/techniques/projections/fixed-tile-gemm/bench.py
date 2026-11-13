"""fixed-tile-gemm's sweep bench (crit_kernels.sweep): every linear weight of the model as a BF16 [N][K] matrix (the
plugin runs BF16 matrices and NVFP4 or MXFP8 ones decoded to BF16 through it), BF16 activations whose row i is seeded
by i alone (one set a K, sliced for every row count), an FP64 reference of the first rows."""
import torch

from crit_kernels.gates import shared_memory_optin

MAX_ROWS = 8192     # the most rows tune_rows asks for
CHECKED = 64        # rows of each call checked against the FP64 reference


class Mm:
    structure = "mm"

    def __init__(self):
        self._weights, self._activations, self._references, self._smem = {}, {}, {}, None

    def shapes(self, model):
        return [{"N": m["N"], "K": m["K"]} for m in model.matrices()]

    def valid(self, knobs):
        # FP32 accumulators a thread holds (more spill), and the shared memory Triton's pipeliner takes (num_stages - 1
        # buffers of both operands' tiles)
        if knobs["BLOCK_M"] * knobs["BLOCK_N"] // (32 * knobs["WARPS"]) > 128:
            return False
        if self._smem is None:
            props = torch.cuda.get_device_properties(0)
            self._smem = getattr(props, "shared_memory_per_block_optin", 0) or 99 * 1024
        return (knobs["BLOCK_M"] + knobs["BLOCK_N"]) * knobs["BLOCK_K"] * 2 * max(knobs["STAGES"] - 1, 1) <= self._smem

    def inputs(self, shape, rows, device):
        n, k = int(shape["N"]), int(shape["K"])
        if (n, k) not in self._weights:   # one matrix a shape, kept for every configuration the sweep times
            g = torch.Generator().manual_seed(n * 31 + k)
            self._weights[(n, k)] = (torch.randn(n, k, generator=g) * 0.05).to(device, torch.bfloat16)
        if k not in self._activations:
            self._activations[k] = torch.cat([torch.randn(1, k, generator=torch.Generator().manual_seed(1000 + i))
                                              for i in range(MAX_ROWS)]).to(device, torch.bfloat16)
        return {"x": self._activations[k][:rows], "dense": self._weights[(n, k)], "key": (n, k), "rows": rows}

    def run(self, module, s):
        return module.mm(s["x"], s["dense"])

    def yardstick(self, s):
        """cuBLAS's BF16 GEMM on the same operands (torch.matmul: its heuristic picks the kernel, split-K included,
        by the row count, which is what changes a row's bits)."""
        x, w = s["x"], s["dense"]
        return lambda: torch.matmul(x, w.t())

    def reference(self, s):
        if s["key"] not in self._references:   # the first CHECKED rows, the same for every row count
            x = self._activations[s["key"][1]][:CHECKED].double()
            w = s["dense"]
            self._references[s["key"]] = torch.cat([x @ w[n0:n0 + 4096].double().t()
                                                    for n0 in range(0, w.shape[0], 4096)], dim=1)
        return self._references[s["key"]]

    def check(self, got, want, s):
        rows = min(s["rows"], CHECKED)
        error = float((got[:rows].double() - want[:rows]).abs().max() / want[:rows].abs().max())
        # FP32 accumulation, then one BF16 rounding of each output
        return None if error < 2.0 ** -7 else f"relative error {error:.2e} beyond FP32 sums and one BF16 rounding"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k = int(shape["N"]), int(shape["K"])
        return (n * k + rows * k + rows * n) * 2


class MmSplit(Mm):
    """mm_split on the same operands (structure mm-split: K split over programs, not row invariant)."""
    structure = "mm-split"

    def run(self, module, s):
        return module.mm_split(s["x"], s["dense"])


class CudaMm(Mm):
    """mm_cuda on the same operands (structure cuda-mm: the CUDA GEMM, row invariant)."""
    structure = "cuda-mm"

    def variants(self, shape):
        return [{"T": 0}]

    def valid(self, knobs):
        """Warp tiles of whole 16 x 16 tiles with at most 32 accumulator tiles (128 FP32 registers) a thread, 4 to 8
        warps whose tile copies cover whole 8-row groups (THREADS a multiple of 8 BK / 16), and the ring within a CTA's
        shared memory (the GPU's opt-in limit: 99 KB on an RTX 5090)."""
        tm, tn, wm, wn, bk = knobs["TM"], knobs["TN"], knobs["WM"], knobs["WN"], knobs["BK"]
        if tm % (16 * wm) or tn % (16 * wn) or (tm // wm // 16) * (tn // wn // 8) > 32 or not 4 <= wm * wn <= 8:
            return False
        if (32 * wm * wn) % (8 * bk // 16):
            return False
        return knobs["STAGES"] * (tm + tn) * bk <= shared_memory_optin()

    def run(self, module, s):
        return module.mm_cuda(s["x"], s["dense"])


class CudaMmSplit(CudaMm):
    """mm_cuda_split on the same operands (structure cuda-mm-split: K split over CTAs, not row invariant)."""
    structure = "cuda-mm-split"

    def valid(self, knobs):
        """cuda-mm's, narrowed to the tiles worth a split (the split multiplies the space by five): 128-byte stages of
        3 or 4, the staged epilogue, TN 64 or 128 (and 128 x 256)."""
        if not super().valid(knobs) or knobs["BK"] != 128 or knobs["EPI"] != 1 or knobs["STAGES"] == 2:
            return False
        return knobs["TN"] in (64, 128) or (knobs["TM"], knobs["TN"]) == (128, 256)

    def run(self, module, s):
        return module.mm_cuda_split(s["x"], s["dense"])
