"""crit_kernels.floors: a call's floor (bytes at the bandwidth or multiply-adds at the format's rate, the longer) and
the share of it a timed call reaches."""
import json

import pytest

from crit_kernels import floors

CEILINGS = {"bandwidth_gbs": 1000.0, "tflops": {"bf16": 100.0, "nvfp4": 400.0}, "source": "test"}


class _Gemv:
    def bytes(self, shape, rows):
        return 2 * shape["N"] * shape["K"]


class _Gemm(_Gemv):
    arithmetic = "bf16"

    def flops(self, shape, rows):
        return 2 * rows * shape["N"] * shape["K"], self.arithmetic


def test_memory_bound_floor_and_share():
    found = floors.floor(_Gemv(), {"N": 1000, "K": 1000}, 1, CEILINGS)
    assert found == {"memory_us": 2.0, "floor_us": 2.0, "bound": "memory"}   # 2 MB at 1 TB/s
    entry = floors.stamp({"us": 4.0}, _Gemv(), {"N": 1000, "K": 1000}, 1, CEILINGS)
    assert entry["of_floor"] == 0.5


def test_compute_bound_floor_takes_the_longer_time():
    shape = {"N": 4096, "K": 4096}
    found = floors.floor(_Gemm(), shape, 8192, CEILINGS)   # 275 GFLOP at 100 TFLOPS: 2,749 us; 33.5 MB: 33.6 us
    assert found["bound"] == "compute" and found["floor_us"] == pytest.approx(2748.779, rel=1e-6)
    assert floors.floor(_Gemm(), shape, 1, CEILINGS)["bound"] == "memory"


def test_a_format_without_a_rate_stays_unpriced():
    gemm = _Gemm()
    gemm.arithmetic = "fp8"
    found = floors.floor(gemm, {"N": 4096, "K": 4096}, 8192, CEILINGS)
    assert found["bound"] == "memory" and found["unpriced"] == "fp8"


def test_a_bench_without_work_has_no_floor():
    assert floors.floor(object(), {}, 1, CEILINGS) is None
    assert "of_floor" not in floors.stamp({"us": 1.0}, object(), {}, 1, CEILINGS)


def test_probe_ceilings_take_the_larger_rate(tmp_path):
    probe = {"device": {"bandwidth_gbs": 1790.0, "tflops": {"bf16": 185.0, "fp8": 350.0},
                        "mma_tflops": {"bf16": 252.0, "nvfp4": 900.0}}}
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe))
    found = floors.from_probe(path)
    assert found["bandwidth_gbs"] == 1790.0
    assert found["tflops"] == {"bf16": 252.0, "fp8": 350.0, "nvfp4": 900.0}
    with pytest.raises(ValueError):
        floors.from_probe({"tflops": {}})
