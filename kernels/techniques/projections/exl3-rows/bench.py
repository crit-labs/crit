"""exl3-rows' sweep bench (crit_kernels.sweep): the model's EXL3 matrices (their N, K and bits; seeded trellis codes and
signs of that layout in the MCG codebook), BF16 activations whose row i is seeded by i alone, and an FP64 reference of
the call (product64: the rotated rows rounded to FP16 as the tensor cores take them, the decoded tiles, the rotations).
The check is FP32 accumulation and one output rounding."""
import torch

from crit_kernels.gates import shared_memory_optin

MOST = 16


def _activations(rows, k, device):
    return torch.stack([torch.randn(k, generator=torch.Generator().manual_seed(1000 + i)) for i in range(rows)]).to(
        device, torch.bfloat16)


def hadamard(device):
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / 128 ** 0.5


def dense64(package, w):
    """W [in][out] in FP64 from the decoded tiles and the signs (the definition, apart from the kernels' rotations)."""
    q = package.decode(w).double()
    k, n = q.shape
    h = hadamard(q.device)
    q = (h @ q.view(k // 128, 128, n)).view(k, n)
    q = (q.view(k, n // 128, 128) @ h).view(k, n)
    return w.suh.double()[:, None] * q * w.svh.double()[None, :]


def product64(package, w, x):
    """y [M][out] in FP64 as gemv computes it: x su H in FP64, rounded to FP16 (the tensor cores' operand), times the
    decoded tiles, then H sv."""
    q = package.decode(w).double()
    k, n = q.shape
    h = hadamard(q.device)
    xr = ((x.double() * w.suh.double()).view(-1, k // 128, 128) @ h).view(-1, k).half().double()
    y = xr @ q
    return (y.view(-1, n // 128, 128) @ h).view(-1, n) * w.svh.double()


class Gemv:
    structure = "gemv"

    def __init__(self):
        self._matrices, self._references = {}, {}

    def shapes(self, model):
        return [m for m in model.matrices() if m["format"] == "exl3"]

    def matrix(self, package, shape, device):
        n, k, bits = int(shape["N"]), int(shape["K"]), float(shape.get("bits", 4))
        key = (n, k, bits)
        if key not in self._matrices:
            g = torch.Generator().manual_seed(n * 31 + k)
            trellis = torch.randint(-32768, 32767, (k // 16, n // 16, int(16 * bits)), generator=g, dtype=torch.int16)
            suh = (torch.randint(0, 2, (k,), generator=g) * 2 - 1).to(torch.float16)
            svh = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float16)
            self._matrices[key] = package.Exl3Matrix(trellis.to(device), suh.to(device), svh.to(device),
                                                     2 if bits != int(bits) else 1)
        return self._matrices[key]

    def valid_for(self, knobs, shape):
        """A block's shared memory (its stages of tile slices and activations) within what a block may take on this
        GPU (its opt-in limit)."""
        bits = float(shape.get("bits", 4))
        words = 8 * int(bits) + (4 if bits != int(bits) else 0)
        bytes_ = knobs["STAGES"] * knobs["ROWS"] * (knobs["WARPS"] * knobs["STRIPS"] * words + 128) * 4
        return bytes_ <= shared_memory_optin()

    def inputs(self, shape, rows, device):
        return {"shape": shape, "x": _activations(rows, int(shape["K"]), device),
                "key": (int(shape["N"]), int(shape["K"]), float(shape.get("bits", 4)), rows)}

    def run(self, module, s):
        s["w"] = self.matrix(module, s["shape"], s["x"].device)
        s["module"] = module
        return module.gemv(s["x"], s["w"])

    def reference(self, s):
        if s["key"] not in self._references:
            w = s.get("w") or self.matrix(s["module"], s["shape"], s["x"].device)
            self._references[s["key"]] = product64(s["module"], w, s["x"])
        return self._references[s["key"]]

    def check(self, got, want, s):
        bound = 1e-4 * float(want.abs().max()) + 2.0 ** -8 * want.abs()
        bad = int(((got.double() - want).abs() > bound).sum())
        return None if bad == 0 else f"{bad} values beyond FP32 sums and one BF16 rounding"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        n, k, bits = int(shape["N"]), int(shape["K"]), float(shape.get("bits", 4))
        return int(n * k * bits / 8) + rows * (k + n) * 2
