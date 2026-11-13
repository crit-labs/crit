"""exl3-rows on this GPU: the decoded tiles equal the format's own decoder bit for bit for every codebook and bitrate
(the SHA-256 of a 128 x 128 matrix decoded from seeded codes, as crit-eval's CPU decoder's tests record it), gemv
within FP32 sums of its FP64 reference (FP16 rotated rows times the tiles) and within FP16 activations' rounding of the
exact product, a row's bits the same alone and among up to 16, dequantize
within one BF16 rounding, the rotation its own inverse, and what the kernel cannot read refused."""
import hashlib
import sys
from pathlib import Path

import pytest
import torch

from crit_kernels.build import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import dense64, product64  # noqa: E402

er = load("exl3-rows")
dev = torch.device("cuda")

# "codebook:K" -> SHA-256 of the [128, 128] FP16 tiles of torch.randint(-32768, 32767, (8, 8, 16 K), seed
# 1000 codebook + 10 K, int16).
DECODED = {
    "0:1": "1cd90a60d11ebf31684802b54f03ace891106e6a88462a72b6d9379c5d360063",
    "0:2": "c8e1cde868436333f5f110ba643a436ace6e21c01855b29df94112099193bb44",
    "0:3": "5d489c802444606f171659589339fcf363993182313fdb68e7e0b77f0e29d57a",
    "0:4": "fd7570af77b1dcc4b22f09c0857d28a00382988514d31817ad73567f6956c52f",
    "0:5": "f05fba35c75ecb27a0c291d5cb10a43872f49fab0be7a796e1aa6edfbeb2ecd8",
    "0:6": "17e4d770229b3c0c2d06620e8a05e0467031d549680c89a3bd5ef537611c3eb4",
    "0:7": "73b078fb980cdc48f5f37371cb41a2d806533cbddaa03976da9236c183f6bc7c",
    "0:8": "0ceb2d00e0d122a5977bdfa61fa4491206174cced4236b59ef376a05fdf5629f",
    "1:1": "63d387a096a37597193d92672c40bc51c281b52f0eb789f1d3cb2ca5d58821a4",
    "1:2": "68417590be358ac82f7e86cc36bfbb44281d01c4fc62311a8b0b98fde9678346",
    "1:3": "10cc7fe1ada8282c6d9127fc95070d5fa110a2199a4b15595771fe01e9a427ac",
    "1:4": "c9f8eb93bcfe4f8e4690e1daf9ec0c75b03d95c3c065912ef057f36fd70b87ba",
    "1:5": "b7c565519a8b9e7f88a5528d50906dcbbb0f86449a1d91815d0fb286ffaa7291",
    "1:6": "5b42be632fd73b37d226624d84438c81b3e8a221d9f1c7bf4f764328b26b3064",
    "1:7": "ab8a2cb1b91ae76f8373a97853b840b7d46a57071ff112fe14ea46eca0fbb8ed",
    "1:8": "703749d697f6dbc0f8104d3fec7e0015d71a103ff15ccf0b078e9beb40a7842f",
    "2:1": "df4d91f0ce59fc730806e0da2d49b4b3c21daf6aba319228ee87920af40fef88",
    "2:2": "84a95da45be2a4171d1045cb3b866d1786b705b91fca9c94f48d6358e7881338",
    "2:3": "fdfb0f0c6987cca25f502d2ccba09e6730d0db34d440053274397ed4058a03fb",
    "2:4": "6955cc0214c740dfb86dfb2e8b1e443ca6f60bee96c34ea293455dae89d12aae",
    "2:5": "23af6dae3884ee1e04fc8b16d28d25521fe3a9e94ce224a334071383b7a97fa4",
    "2:6": "8a1f5e511e59cab04f647e8bdfc323583dd8f062c90db1572c1b7a4b3b866278",
    "2:7": "c5fe49682b0f3f357d9f4510121849fcc0da2a84e25a545ef0fffcf751c279fc",
    "2:8": "f4bc4c26352b01395e98d4e96f469171d3cb92208f1a5b8748efde07332ffd8c",
    "2:1.5": "2f5aea3de03133c5e7c08259ecb687a555e4e7104c276f0c9c9027b78e73a3a7",
    "2:2.5": "63ec3b13137797155adb4b190d301e49f2909338cf2b2ca2669637043c9bb975",
    "2:3.5": "c659573613094bb3962c3fa47b562d4e1349a77293fda3573250a151cd336878"
}


def matrix(k, n, bits, codebook, seed=3):
    g = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, int(16 * bits)), generator=g, dtype=torch.int16)
    suh = (torch.randint(0, 2, (k,), generator=g) * 2 - 1).to(torch.float16)
    svh = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.float16)
    return er.Exl3Matrix(trellis.to(dev), suh.to(dev), svh.to(dev), codebook)


def activations(m, k, seed=8):
    return torch.randn(m, k, generator=torch.Generator().manual_seed(seed)).to(dev, torch.bfloat16)


@pytest.mark.parametrize("key", sorted(DECODED))
def test_decode_gives_the_formats_bits(key):
    codebook, rate = key.split(":")
    codebook, rate = int(codebook), float(rate)
    g = torch.Generator().manual_seed(int(1000 * codebook + 10 * rate))
    trellis = torch.randint(-32768, 32767, (8, 8, int(16 * rate)), generator=g, dtype=torch.int16).to(dev)
    q = er.decode(er.Exl3Matrix(trellis, None, None, codebook))
    assert hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() == DECODED[key]


@pytest.mark.parametrize("codebook,bits", [(0, 2), (1, 4), (2, 3), (2, 2.5), (1, 8)])
@pytest.mark.parametrize("shape", [(256, 384), (1024, 512)])
def test_gemv_is_the_product_with_the_dense_matrix_and_keeps_a_rows_bits(codebook, bits, shape):
    k, n = shape
    w = matrix(k, n, bits, codebook)
    x = activations(16, k)
    got = er.gemv(x, w)
    want = product64(er, w, x)
    bound = 1e-4 * float(want.abs().max()) + 2.0 ** -8 * want.abs()
    assert not ((got.double() - want).abs() > bound).any()
    # and within FP16 activations' rounding of the exact product with the dense matrix
    exact = x.double() @ dense64(er, w)
    assert ((got.double() - exact).abs() <= 2.0 ** -9 * (x.double().abs() @ dense64(er, w).abs()) + bound).all()
    for rows in (1, 3, 9):
        part = er.gemv(x[:rows].clone(), w)
        assert torch.equal(part.view(torch.int16), got[:rows].view(torch.int16))


def test_dequantize_and_the_rotation():
    w = matrix(256, 384, 3, 2)
    dense = er.dequantize(w)
    want = dense64(er, w).t()
    assert dense.shape == (384, 256) and dense.dtype == torch.bfloat16
    assert ((dense.double() - want).abs() <= 2.0 ** -8 * want.abs() + 1e-6).all()
    x = torch.randn(5, 256, device=dev)
    back = er.rotate(er.rotate(x))
    assert torch.allclose(back, x, atol=1e-5)
    signs = er.unpack_signs(torch.tensor([0b101, -1], dtype=torch.int16), 32)
    assert signs[:3].tolist() == [-1.0, 1.0, -1.0] and signs[16:].eq(-1).all()


def test_what_it_cannot_read_is_refused():
    with pytest.raises(RuntimeError):
        er.gemv(activations(2, 96), er.Exl3Matrix(torch.zeros(6, 8, 64, dtype=torch.int16, device=dev), None, None, 0))
    with pytest.raises(RuntimeError):
        er.decode(er.Exl3Matrix(torch.zeros(8, 8, 40, dtype=torch.int16, device=dev), None, None, 1))   # K 2.5, MCG
