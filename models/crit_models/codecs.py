"""Weight codecs and the GPTQ solver (torch).

Both formats are the compressed-tensors layouts Eval's reference and common engines read:

- fp8-channel: E4M3FN codes in `weight` and one FP32 `weight_scale` per output row ([rows, 1]), the row's
  amax / 448 (1 for an all-zero row); codes are round-to-nearest-even of w / scale, saturated at 448.
- nvfp4: the Inference NVFP4 codec (inference/quantized-projections/code/nvfp4.py) with FP32 arithmetic.
  `weight_global_scale` = 6 * 448 / amax(tensor), one E4M3FN `weight_scale` per 16 row values from the block
  amax, E2M1 codes packed two per byte in `weight_packed` (even element in the low nibble); a value decodes as
  e2m1(code) * e4m3(scale) / global.

A codec quantizes a whole matrix (round to nearest) or one column at a time for GPTQ; GPTQ with an identity
Hessian gives exactly the round-to-nearest codes.
"""
import torch

E4M3_MAX, E2M1_MAX, BLOCK = 448.0, 6.0, 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_encode(x):
    """4-bit codes (bit 3 the sign) of x rounded to the nearest E2M1 value, ties to the even code, saturating at 6."""
    m = x.abs()
    # Midpoints between neighbouring magnitudes; at a tie the lower code wins where it is even (>), else the upper (>=).
    index = ((m > 0.25).to(torch.uint8) + (m >= 0.75) + (m > 1.25) + (m >= 1.75) + (m > 2.5) + (m >= 3.5) + (m > 5.0))
    return index.to(torch.uint8) | (((x < 0) & (index > 0)).to(torch.uint8) << 3)


def e2m1_decode(codes):
    table = torch.tensor(E2M1, dtype=torch.float32, device=codes.device)
    values = table[(codes & 7).long()]
    return torch.where((codes & 8) > 0, -values, values)


class Fp8Channel:
    grouped = False

    def __init__(self, weight):
        amax = weight.abs().amax(dim=1)
        self.scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax)).float()

    def _codes(self, values, scale):
        return (values / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

    def rtn(self, weight):
        codes = self._codes(weight, self.scale[:, None])
        return codes.view(torch.uint8), codes.float() * self.scale[:, None]

    def column(self, values, column):
        codes = self._codes(values, self.scale)
        return codes.view(torch.uint8), codes.float() * self.scale

    def tensors(self, codes):
        return {"weight": codes.view(torch.float8_e4m3fn).contiguous(), "weight_scale": self.scale[:, None].contiguous()}


class Nvfp4:
    grouped = True

    def __init__(self, weight):
        amax = weight.abs().amax().float()
        self.global_scale = torch.where(amax > 0, (E2M1_MAX * E4M3_MAX) / amax, torch.ones_like(amax))
        self.scales = torch.zeros(weight.shape[0], weight.shape[1] // BLOCK, dtype=torch.uint8, device=weight.device)
        self.current = None

    def _block_scale(self, block_amax):
        return (block_amax * self.global_scale / E2M1_MAX).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)

    def _quantize(self, values, decoded):
        scaled = torch.where(decoded > 0, values * self.global_scale / torch.where(decoded > 0, decoded, 1.0), 0.0)
        codes = e2m1_encode(scaled)
        return codes, e2m1_decode(codes) * decoded / self.global_scale

    def rtn(self, weight):
        rows, cols = weight.shape
        blocks = weight.reshape(rows, cols // BLOCK, BLOCK)
        scales = self._block_scale(blocks.abs().amax(dim=2))
        self.scales = scales.view(torch.uint8)
        codes, values = self._quantize(blocks, scales.float()[..., None])
        return codes.reshape(rows, cols), values.reshape(rows, cols)

    def start_group(self, block, column):
        """Scale of the 16 columns starting at `column`, from their current (GPTQ-updated) values."""
        scales = self._block_scale(block.abs().amax(dim=1))
        self.scales[:, column // BLOCK] = scales.view(torch.uint8)
        self.current = scales.float()

    def column(self, values, column):
        return self._quantize(values, self.current)

    def tensors(self, codes):
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return {"weight_packed": packed.contiguous(), "weight_scale": self.scales.view(torch.float8_e4m3fn).contiguous(),
                "weight_global_scale": self.global_scale.reshape(1).contiguous()}


CODECS = {"fp8-channel": Fp8Channel, "nvfp4": Nvfp4}


def gptq(weight, hessian, codec, dampening, block):
    """GPTQ (Frantar et al., 2022) in column order: quantize each column, spread its error over the remaining
    columns through the inverse Hessian's Cholesky factor. Returns codes [rows, cols] and dequantized values.
    `block` must be a multiple of 16 so NVFP4 groups never straddle a block."""
    W = weight.clone().float()
    H = hessian.clone().float()
    rows, cols = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += dampening * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.zeros(rows, cols, dtype=torch.uint8, device=W.device)
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for i in range(i2 - i1):
            column = i1 + i
            if codec.grouped and column % BLOCK == 0:
                codec.start_group(W1[:, i:i + BLOCK], column)
            w = W1[:, i]
            c, q = codec.column(w, column)
            codes[:, column], Q1[:, i] = c, q
            error = (w - q) / Hinv1[i, i]
            W1[:, i:] -= error[:, None] @ Hinv1[i, i:][None, :]
            E1[:, i] = error
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return codes, Q


def quantize(format, weight, hessian=None, settings=None):
    """Export tensors and error measures for one weight: round to nearest without a Hessian, GPTQ with one.
    weight_error is ||W - Q|| / ||W||; output_error, with a Hessian, tr(D H D^T) / tr(W H W^T) for D = W - Q."""
    weight = weight.float()
    if not torch.isfinite(weight).all():
        raise ValueError("weight holds non-finite values")
    codec = CODECS[format](weight)
    if hessian is None:
        codes, values = codec.rtn(weight)
    else:
        codes, values = gptq(weight, hessian, codec, settings["dampening"], settings["block"])
    difference = weight - values
    norm = weight.norm()
    errors = {"weight_error": float(difference.norm() / norm) if norm > 0 else 0.0}
    if hessian is not None:
        h = hessian.float()
        total = float(torch.einsum("ij,jk,ik->", weight, h, weight))
        errors["output_error"] = float(torch.einsum("ij,jk,ik->", difference, h, difference)) / total if total > 0 else 0.0
    return codec.tensors(codes), values, errors
