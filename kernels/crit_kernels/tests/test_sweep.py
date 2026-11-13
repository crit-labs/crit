"""crit_kernels.sweep's table choice and crit_kernels.build's lookups, without a GPU: a structure's declared lengths
are swept as part of its shapes, one order group serves every length and row count, the speed knobs follow the length,
and tunings without length ranges load and look up as before."""
import json

from crit_kernels import build as kbuild
from crit_kernels import sweep

MANIFEST = {
    "id": "toy",
    "structures": {
        "s": {
            "key": ["D"],
            "lengths": {"field": "N", "values": [4096, 32768]},
            "rows": [1, 64],
            "tune_rows": [1, 16],
            "knobs": {"CHUNK": {"values": [0, 256], "default": 0, "order": True},
                      "STAGES": {"values": [0, 3], "default": 0}},
        }
    },
}


def _results(times):
    """times[(N, rows, CHUNK, STAGES)] -> the sweep's results dict for shape D 64."""
    return {("s", (64, n), r, json.dumps({"CHUNK": c, "STAGES": st}, sort_keys=True)): t
            for (n, r, c, st), t in times.items()}


def _times():
    # CHUNK 256 wins at 4K, 0 at 32K (by more); STAGES 3 wins at 4K, 0 at 32K
    t = {}
    for n in (4096, 32768):
        for r in (1, 16):
            for c in (0, 256):
                for st in (0, 3):
                    base = 10.0 if n == 4096 else 100.0
                    base *= 0.9 if (c == 256) == (n == 4096) else (1.0 if n == 4096 else 1.5)
                    base *= 0.95 if (st == 3) == (n == 4096) else 1.0
                    t[(n, r, c, st)] = base
    return t


def test_declared_lengths_are_swept_as_shapes():
    spec = MANIFEST["structures"]["s"]
    assert sweep._at_lengths(spec, [{"D": 64}]) == [{"D": 64, "N": 4096}, {"D": 64, "N": 32768}]
    assert sweep._at_lengths(spec, [{"D": 64, "N": 7}]) == [{"D": 64, "N": 7}]
    assert sweep._fields(spec) == ["D", "N"]
    assert sweep._fields({"key": ["D"]}) == ["D"]


def test_one_order_group_for_every_length_and_speed_knobs_by_length():
    shapes = {"s": sweep._at_lengths(MANIFEST["structures"]["s"], [{"D": 64}])}
    tuning, report = sweep.choose(MANIFEST, ["s"], shapes, _results(_times()), {}, {"s": object()})
    t = tuning["structures"]["s"]
    picked = [t["configs"][e["config"]] for e in t["table"]]
    assert len({c["CHUNK"] for c in picked}) == 1            # one order group (CHUNK 0: 32K loses least overall)
    assert picked[0]["CHUNK"] == 0
    by_length = {tuple(e["length"]): t["configs"][e["config"]]["STAGES"] for e in t["table"]}
    edge = int((4096 * 32768) ** 0.5)
    assert by_length == {(0, edge): 3, (edge + 1, 1 << 62): 0}
    assert {e["length"] for e in report["structures"]["s"]} == {4096, 32768}
    # the build's lookups: the Python tuning module and the CUDA header take the length
    checked = kbuild.check_tuning(MANIFEST, tuning)
    ns = {}
    exec(kbuild.tuning_module(MANIFEST, checked), ns)
    assert ns["knobs"]("s", {"D": 64}, 1, 4160)["STAGES"] == 3
    assert ns["knobs"]("s", {"D": 64}, 1, 32832)["STAGES"] == 0
    assert ns["knobs"]("s", {"D": 64}, 1)["STAGES"] == 3          # no length: the first entry that holds
    header = kbuild.tuning_header(MANIFEST, checked)
    assert "int rows, long long length = -1)" in header and f"{edge}" in header


def test_a_tuning_without_lengths_loads_and_holds_for_every_length():
    old = {"schema": kbuild.TUNING_SCHEMA, "kernel": "toy",
           "structures": {"s": {"configs": [{"CHUNK": 0, "STAGES": 0}, {"CHUNK": 256, "STAGES": 3}],
                                "table": [{"key": {"D": 64}, "rows": [1, 64], "config": 1}]}}}
    checked = kbuild.check_tuning(MANIFEST, old)
    ns = {}
    exec(kbuild.tuning_module(MANIFEST, checked), ns)
    for length in (None, 1, 1 << 40):
        assert ns["knobs"]("s", {"D": 64}, 8, length) == {"CHUNK": 256, "STAGES": 3}
    assert "{0, 64, 1, 64, 1, 0, 4611686018427387904}" in kbuild.tuning_header(MANIFEST, checked)


def test_without_lengths_the_table_is_as_before():
    manifest = json.loads(json.dumps(MANIFEST))
    del manifest["structures"]["s"]["lengths"]
    times = {k: v for k, v in _times().items() if k[0] == 4096}
    results = {("s", (64,), r, json.dumps({"CHUNK": c, "STAGES": st}, sort_keys=True)): t
               for (_, r, c, st), t in times.items()}
    tuning, report = sweep.choose(manifest, ["s"], {"s": [{"D": 64}]}, results, {}, {"s": object()})
    table = tuning["structures"]["s"]["table"]
    assert table == [{"key": {"D": 64}, "rows": [1, 64], "config": 1}]
    assert tuning["structures"]["s"]["configs"][1] == {"CHUNK": 256, "STAGES": 3}
    assert all("length" not in e for e in report["structures"]["s"])


def test_a_models_matrices_include_the_stacks_engines_load(tmp_path):
    """Siblings of one module that read the same input (q, k, v; gate and up) are also swept as one stacked matrix."""
    from crit_kernels.model import Model
    import struct
    header = {"model.layers.0.self_attn.q_proj.weight": {"dtype": "F8_E4M3", "shape": [4096, 5120], "data_offsets": [0, 0]},
              "model.layers.0.self_attn.k_proj.weight": {"dtype": "F8_E4M3", "shape": [1024, 5120], "data_offsets": [0, 0]},
              "model.layers.0.self_attn.v_proj.weight": {"dtype": "F8_E4M3", "shape": [1024, 5120], "data_offsets": [0, 0]},
              "model.layers.0.self_attn.o_proj.weight": {"dtype": "F8_E4M3", "shape": [5120, 4096], "data_offsets": [0, 0]}}
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    found = {m["name"]: m for m in Model(tmp_path).matrices()}
    stacked = found["model.layers.0.self_attn.q_proj|k_proj|v_proj"]
    assert (stacked["N"], stacked["K"], stacked["format"]) == (6144, 5120, "fp8")
    assert len(stacked["stacked"]) == 3 and len(found) == 5   # o_proj reads another input: no stack


def test_a_gguf_checkpoints_block_format_matrices_are_its_shapes(tmp_path):
    """A GGUF checkpoint's matrices in a GGUF block format, by their GGUF type, an expert stack with its count; dense
    tensors and types no kernel reads are left out."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    from gguf_writer import gguf_bytes
    from crit_kernels.model import Model
    tensors = [("blk.0.attn_q.weight", "Q4_K", (4, 256)), ("blk.0.ffn_up_exps.weight", "IQ3_S", (2, 3, 256)),
               ("blk.0.attn_norm.weight", "F32", (32,)), ("token_embd.weight", "F16", (8, 32)),
               ("blk.0.attn_k.weight", "TQ2_0", (4, 256))]
    (tmp_path / "model.gguf").write_bytes(gguf_bytes([("general.architecture", "llama")], tensors))
    found = {m["name"]: m for m in Model(tmp_path).matrices()}
    assert found == {"blk.0.attn_q": {"name": "blk.0.attn_q", "N": 4, "K": 256, "format": "Q4_K"},
                     "blk.0.ffn_up_exps": {"name": "blk.0.ffn_up_exps", "N": 3, "K": 256, "format": "IQ3_S",
                                           "experts": 2}}


def test_a_stop_while_measuring_ends_the_measuring_not_the_sweep():
    """SIGINT while measuring is caught and restored after; the sweep then writes what it measured."""
    import os
    import signal
    before = signal.getsignal(signal.SIGINT)
    stop = sweep._Stop()
    os.kill(os.getpid(), signal.SIGINT) if os.name != "nt" else stop._stop()
    assert stop.now
    stop.restore()
    assert signal.getsignal(signal.SIGINT) is before


def test_a_sweep_takes_the_matrices_an_engine_loaded_instead_of_guessing(tmp_path):
    """crit run serve --check's report keeps the served engine's facts; their matrices replace the checkpoint's own."""
    from crit_kernels.model import Model, loaded_matrices
    report = {"attempt_id": "attempt-0002", "checks": [
        {"name": "consecutive", "state": "passed"},
        {"name": "draft_bytes", "state": "passed",
         "engine_facts": {"matrices": [{"N": 6144, "K": 5120, "format": "fp8"}, {"N": 34816, "K": 5120, "format": "nvfp4"}]}}]}
    path = tmp_path / "serve-check-attempt-0002.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    found = Model(tmp_path, loaded_matrices(path)).matrices()
    assert [(m["N"], m["K"], m["format"]) for m in found] == [(6144, 5120, "fp8"), (34816, 5120, "nvfp4")]
    (tmp_path / "empty.json").write_text("[]", encoding="utf-8")
    try:
        loaded_matrices(tmp_path / "empty.json")
    except ValueError as error:
        assert "holds no loaded matrices" in str(error)
    else:
        raise AssertionError("an empty list is refused")


def test_a_stop_during_the_builds_ends_them_and_writes_the_defaults_table(tmp_path, monkeypatch):
    """SIGTERM or Ctrl+C while the configurations build: the builds not started are cancelled, nothing is measured,
    and the table of every structure's defaults is written, its report saying how far the builds got."""
    import threading
    import torch
    from concurrent.futures import ThreadPoolExecutor
    stops, calls, lock = [], [], threading.Lock()

    class Stop(sweep._Stop):
        def __init__(self):
            super().__init__()
            stops.append(self)

    def build_one(args):
        with lock:
            calls.append(args)
            if len(calls) == 1:
                stops[0]._stop()   # the stop's signal arrives during the first build
        return args[1], None

    monkeypatch.setattr(sweep, "_Stop", Stop)
    monkeypatch.setattr(sweep, "_build_one", build_one)
    monkeypatch.setattr(sweep, "ProcessPoolExecutor",
                        lambda max_workers, initializer: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(sweep, "_measure_all", lambda *a, **k: (_ for _ in ()).throw(AssertionError("measured")))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "test device")
    monkeypatch.setattr(kbuild, "device_arch", lambda: ("sm_120", "12.0"))
    shapes = {"gemm": [{"N": 4096, "K": 4096, "format": "Q4_K"}]}
    tuning, report = sweep.sweep("gguf-rows", shapes=shapes, structures=["gemm"], out=tmp_path, chunk=4, log=lambda *_: None)
    assert len(calls) == 1   # one build ran, the rest were cancelled
    assert tuning["structures"]["gemm"] == {"configs": [kbuild.defaults(kbuild.find("gguf-rows")[1], "gemm")], "table": []}
    reasons = set(report["dropped"].values())
    assert len(reasons) == 1 and "stopped during its builds (1 of" in reasons.pop()
    assert json.loads((tmp_path / "gguf-rows.json").read_text())["structures"]["gemm"]["table"] == []
