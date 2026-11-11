"""The decode rows and the prompt GEMM give the same bits for every output, on a repeat too, and so does a build with
other knobs. tests/int_block_tests.cu checks the raw kernels against the declared order's scalar oracle in C++; it is
compiled here with nvcc against the default knobs' crit_tuning.h and against another configuration's."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from crit_kernels import build as kbuild

HERE = Path(__file__).resolve().parent
ib = kbuild.load("int-block-experts")
dev = torch.device("cuda")
OTHER = {"rows": {"THREADS": 128, "ROWS": 4, "STREAM": 1, "UNROLL": 2}, "gemm": {"WARPS": 4, "TILES": 2, "TOKZ": 2},
         "gemm-staged": {"BM": 64, "BN": 32, "WM": 2, "WN": 2, "GROUP": 1, "RUN": 2}}


def decode_and_prompt(module, rows=96, cols=704, tokens=21):
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g).to(dev)
    scales = (torch.randint(0x20, 0x5F, (rows, cols // 16), generator=g)).to(torch.uint8).to(dev)
    x = torch.randn(tokens, cols, generator=g).to(dev)
    values, act_scales = module.quantize_int8(x, 32)
    table = module.matrix_table([(codes, scales, 0.5, False)])
    decode = torch.zeros(tokens, rows, device=dev)
    module.rows(table, module.row_jobs([(0, list(range(t, min(t + 8, tokens))), list(range(t, min(t + 8, tokens))))
                                        for t in range(0, tokens, 8)], dev), rows, values, act_scales, 32, decode)
    prompt = torch.zeros(tokens, rows, device=dev)
    module.gemm(table, module.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, values, act_scales, 32, prompt)
    staged = torch.zeros(tokens, rows, device=dev)
    module.gemm_staged(table, module.gemm_jobs([(0, 0, tokens)], dev), rows, tokens, values, act_scales, 32, staged)
    return decode, prompt, staged


@pytest.mark.kernels_ci
def test_rows_and_gemms_agree_bit_for_bit():
    decode, prompt, staged = decode_and_prompt(ib)
    assert torch.equal(decode.view(torch.int32), prompt.view(torch.int32))
    assert torch.equal(staged.view(torch.int32), prompt.view(torch.int32))
    again = decode_and_prompt(ib)
    assert torch.equal(again[1], prompt) and torch.equal(again[2], staged)


@pytest.mark.kernels_ci
def test_the_staged_gemm_has_the_declared_bits_at_every_width_and_format():
    """Fewer than 16 blocks (empty lanes), rows and tokens off the tiles, INT4 weights, 16-element activation blocks,
    E2M1 x 2 activations and two jobs on one call."""
    g = torch.Generator().manual_seed(3)
    for rows, cols, tokens, int4, e2m1 in [(200, 64, 37, False, False), (130, 640, 300, True, False),
                                           (77, 1024, 70, False, True)]:
        codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g).to(dev)
        scales = (torch.randint(0x20, 0x5F, (rows, cols // 16), generator=g) |
                  torch.randint(0, 2, (rows, cols // 16), generator=g) * 0x80).to(torch.uint8).to(dev)
        x = torch.randn(tokens, cols, generator=g).to(dev)
        values, act_scales = ib.quantize_e2m1x2(x, 3.0) if e2m1 else ib.quantize_int8(x, 16 if int4 else 32)
        block = 16 if (e2m1 or int4) else 32
        table = ib.matrix_table([(codes, scales, 0.5, int4)])
        jobs = ib.gemm_jobs([(0, 0, tokens // 3), (0, tokens // 3, tokens - tokens // 3)], dev)
        want, got = torch.zeros(tokens, rows, device=dev), torch.zeros(tokens, rows, device=dev)
        ib.gemm(table, jobs, rows, tokens - tokens // 3, values, act_scales, block, want)
        ib.gemm_staged(table, jobs, rows, tokens - tokens // 3, values, act_scales, block, got)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (rows, cols, tokens)


@pytest.mark.kernels_ci
def test_other_knobs_give_the_same_bits():
    other = kbuild.load("int-block-experts", knobs=OTHER)
    # rows forward to decode-gemv's rows-int-block (the structure's alias), built with this tuning's rows table
    target = sys.modules[other.__name__ + "._crit_aliases"].module("rows")
    tuned = sys.modules[target.__name__ + "._crit_tuning"].TUNING["rows-int-block"]
    _, manifest = kbuild.find("int-block-experts")
    assert tuned["configs"][tuned["table"][0]["config"]] == dict(kbuild.defaults(manifest, "rows"), **OTHER["rows"])
    for got, want in zip(decode_and_prompt(other, rows=200, cols=2560, tokens=37),
                         decode_and_prompt(ib, rows=200, cols=2560, tokens=37)):
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def _tuning(knobs):
    _, manifest = kbuild.find("int-block-experts")
    if not knobs:
        return kbuild.default_tuning(manifest)
    tuning = {"schema": kbuild.TUNING_SCHEMA, "kernel": "int-block-experts",
              "structures": {s: {"configs": [dict(kbuild.defaults(manifest, s), **c)],
                                 "table": [{"key": None, "rows": [0, 1 << 30], "config": 0}]}
                             for s, c in knobs.items()}}
    return kbuild.check_tuning(manifest, tuning)


@pytest.mark.kernels_ci
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs nvcc")
@pytest.mark.parametrize("knobs", [None, OTHER], ids=["defaults", "other"])
def test_the_kernels_match_the_scalar_oracle(tmp_path, knobs):
    _, manifest = kbuild.find("int-block-experts")
    (tmp_path / "crit_tuning.h").write_text(kbuild.tuning_header(manifest, _tuning(knobs)))
    arch = kbuild.device_arch()[0]
    binary = tmp_path / "int_block_tests"
    subprocess.run(["nvcc", "-std=c++17", "-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-I", str(tmp_path),
                    "-I", str(HERE.parent / "int_block_cuda"), "-I", str(kbuild.core_folder() / "include"),
                    str(HERE.parent / "int_block_cuda" / "int_block.cu"),
                    str(HERE / "int_block_tests.cu"), "-o", str(binary)], check=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0 and "int_block tests passed" in run.stdout, run.stdout + run.stderr


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _values(codes, int4):
    """[rows][k / 2] codes as the integers the kernels take: E2M1 x 2 or INT4."""
    if int4:
        table = torch.tensor(list(range(8)) + list(range(-8, 0)), dtype=torch.float64, device=codes.device)
    else:
        table = torch.tensor([2 * v for v in E2M1] + [-2 * v for v in E2M1], dtype=torch.float64, device=codes.device)
    return table[torch.stack((codes & 15, codes >> 4), -1).long()].reshape(codes.shape[0], -1)


@pytest.mark.kernels_ci
def test_quantize_nvfp4_packs_the_e2m1x2_quantizers_values():
    x = torch.randn(33, 640, generator=torch.Generator().manual_seed(4)).to(dev) * 3
    values, scales = ib.quantize_e2m1x2(x, 5.0)
    codes, scale_bytes = ib.quantize_nvfp4(x, 5.0)
    assert torch.equal(_values(codes, False), values.double())
    decoded = scale_bytes.view(torch.float8_e4m3fn).float()
    # an elementwise division (torch turns a division by a Python scalar into a product with its reciprocal)
    assert torch.equal(decoded / torch.full_like(decoded, 5.0), scales)


@pytest.mark.kernels_ci
def test_the_native_gemm_is_within_fp32_accumulation_of_the_exact_product():
    """gemm-native is not the declared order: no bits to match, only the FP64 product of the stored values, E2M1 weights
    (the block-scaled path) and INT4 ones (the s8 path), two jobs, tokens and rows off the tiles, a K tail."""
    g = torch.Generator().manual_seed(6)
    for rows, cols, tokens, int4 in [(200, 640, 37, False), (130, 2560, 300, False), (77, 1024, 70, True)]:
        codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g).to(dev)
        scales = torch.randint(0x20, 0x5F, (rows, cols // 16), generator=g).to(torch.uint8).to(dev)
        x = torch.randn(tokens, cols, generator=g).to(dev)
        act_codes, act_scales = ib.quantize_nvfp4(x, 3.0)
        table = ib.matrix_table([(codes, scales, 0.5, int4)])
        jobs = ib.gemm_jobs([(0, 0, tokens // 3), (0, tokens // 3, tokens - tokens // 3)], dev)
        out = torch.full((tokens, rows), float("nan"), device=dev)
        ib.gemm_native(table, jobs, rows, tokens - tokens // 3, act_codes, act_scales, 3.0, out, int4=int4)
        w = _values(codes, int4) * scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) * 0.5
        a = _values(act_codes, False) * act_scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) / 3.0
        value, magnitude = a @ w.t(), a.abs() @ w.abs().t()
        error = (out.double() - value).abs()
        assert not out.isnan().any() and bool((error <= magnitude * 2e-6 + 1e-30).all()), (rows, cols, tokens, int4)


def _blocked(scales):
    """[rows][k / 16] scale bytes in cuBLASLt's 128 x 4 tiled layout (rows padded to 128), flat."""
    rows, cols = scales.shape
    padded = torch.zeros(-(-rows // 128) * 128, cols, dtype=scales.dtype, device=scales.device)
    padded[:rows] = scales
    r, c = padded.shape[0] // 128, cols // 4
    return padded.view(r, 4, 32, c, 4).permute(0, 3, 2, 1, 4).flatten()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("knobs", [None, {"EPI": 0, "SPLIT": 2}, {"TM": 192, "WM": 6, "WN": 2, "STAGES": 3}],
                         ids=["defaults", "register-stores", "192-rows"])
def test_the_native_gemm_writes_bf16_with_column_scales_and_reads_tiled_scales(knobs):
    """BF16 output (the FP32 result rounded once), column scales after the global scale, tiled scale bytes for the
    activations and the weights (sm_120a's path), the tensor-map copies (a matrix_table table) and the cp.async ones (a
    table without its host record), K split over CTAs (a short prompt on a long K), the same bits on a repeat; with
    the untuned knobs (TMA-store epilogue), register stores with stream-K past the last wave, and 192-row tiles (12
    consumer warps, B fragments streamed, tiles starting mid scale tile)."""
    ib = kbuild.load("int-block-experts", knobs={"gemm-native": knobs}) if knobs else globals()["ib"]
    g = torch.Generator().manual_seed(7)
    for rows, cols, tokens in [(200, 640, 37), (384, 4096, 300), (5120, 8192, 128), (512, 1024, 704)]:
        codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g).to(dev)
        scales = torch.randint(0x20, 0x5F, (rows, cols // 16), generator=g).to(torch.uint8).to(dev)
        x = torch.randn(tokens, cols, generator=g).to(dev).bfloat16()
        col = (torch.rand(rows, generator=g) + 0.5).to(dev)
        act_codes, act_scales = ib.quantize_nvfp4(x, 3.0)
        w = _values(codes, False) * scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) * 0.5
        a = _values(act_codes, False) * act_scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) / 3.0
        value, magnitude = (a @ w.t()) * col.double(), (a.abs() @ w.abs().t()) * col.double()
        layouts = [0] + ([1, 2, 3] if ib.tiled_supported() else [])
        for tiled in layouts:
            codes_x, scales_x = ib.quantize_nvfp4(x, 3.0, tiled=bool(tiled & ib.TILED_ACTS))
            assert torch.equal(codes_x, act_codes)
            w_scales = _blocked(scales) if tiled & ib.TILED_WEIGHTS else scales   # held while the table points at it
            table = ib.matrix_table([(codes, w_scales, 0.5, False)])
            jobs = ib.gemm_jobs([(0, 0, tokens)], dev)
            for host in (True, False):
                t = table if host else table.clone()   # a clone has no host record: cp.async copies
                outs = []
                for _ in range(2):
                    out = torch.full((tokens, rows), float("nan"), dtype=torch.bfloat16, device=dev)
                    ib.gemm_native(t, jobs, rows, tokens, codes_x, scales_x, 3.0, out, col_scale=col, tiled=tiled)
                    outs.append(out)
                error = (outs[0].double() - value).abs()
                assert bool((error <= magnitude * 2e-6 + value.abs() * 2.0 ** -8 + 1e-30).all()), (rows, cols, tiled, host)
                assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
