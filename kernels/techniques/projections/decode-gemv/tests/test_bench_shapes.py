"""decode-gemv's sweep bench finds its shapes on a model without a GPU: every structure's shapes() over the model's
matrices and its AWQ or GPTQ tensors, keyed by crit_kernels.weight_formats, which mma-rows' bench reads too."""
import importlib.util
import types
from pathlib import Path

from crit_kernels import weight_formats


def _bench(kernel):
    path = Path(__file__).resolve().parents[2] / kernel / "bench.py"
    spec = importlib.util.spec_from_file_location(f"{kernel.replace('-', '_')}_bench_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(quant=None, tensors=None):
    matrices = [{"N": 2560, "K": 6144, "format": "fp8"}, {"N": 2560, "K": 2560, "format": "nvfp4"},
                {"N": 640, "K": 2560, "format": "bf16"}, {"N": 96, "K": 2560, "format": "mxfp8"}]
    return types.SimpleNamespace(matrices=lambda: matrices, quantization=quant or {}, tensors=tensors or {})


def test_the_cuda_and_triton_structures_key_each_format_by_its_weights():
    bench = _bench("decode-gemv")
    for structure in (bench.GemvCuda(), bench.GemvTriton()):
        shapes = structure.shapes(_model())
        assert {(s["format"], s["weights"]) for s in shapes} == {("fp8", 1), ("nvfp4", 0), ("bf16", 3)}


def test_awq_qweights_become_int4_keys():
    bench = _bench("decode-gemv")
    model = _model({"quant_method": "awq", "bits": 4, "group_size": 128},
                   {"layers.0.up.qweight": ("I32", [2560, 640])})
    int4 = [s for s in bench.GemvCuda().shapes(model) if s.get("weights") == weight_formats.WEIGHTS["int4_zp"]]
    assert int4 == [{"name": "layers.0.up", "weights": 4, "N": 5120, "K": 2560, "group": 128}]


def test_both_benches_take_their_format_codes_from_one_table():
    decode, rows = _bench("decode-gemv"), _bench("mma-rows")
    assert decode.WEIGHTS is weight_formats.WEIGHTS
    assert rows.WEIGHTS == {k: v for k, v in weight_formats.WEIGHTS.items() if k != "mxfp8"}
    assert decode.int4_shapes is rows.int4_shapes is weight_formats.int4_shapes
