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


def test_a_first_launch_compiles_every_knob_set_its_shape_key_can_read():
    tuning = {"schema": kbuild.TUNING_SCHEMA, "kernel": "toy",
              "structures": {"s": {"configs": [{"CHUNK": 0, "STAGES": 0}, {"CHUNK": 256, "STAGES": 3},
                                               {"CHUNK": 256, "STAGES": 0}],
                                   "table": [{"key": {"D": 64}, "rows": [1, 8], "config": 1},
                                             {"key": {"D": 128}, "rows": [1, 8], "config": 2},
                                             {"key": None, "rows": [9, 64], "config": 2}]}}}
    ns = {}
    exec(kbuild.tuning_module(MANIFEST, kbuild.check_tuning(MANIFEST, tuning)), ns)
    knobs, variants, launch = ns["knobs"], ns["variants"], ns["launch"]
    assert variants("s", {"D": 64}) == [{"CHUNK": 0, "STAGES": 0}, {"CHUNK": 256, "STAGES": 3},
                                        {"CHUNK": 256, "STAGES": 0}]
    assert variants("s", {"D": 32}) == [{"CHUNK": 0, "STAGES": 0}, {"CHUNK": 256, "STAGES": 0}]

    class Kernel:   # a Triton kernel's run(): one compiled object a specialization, launched unless warming
        def __init__(self):
            self.compiled, self.launched = {}, []

        def run(self, *args, grid, warmup, **options):
            if not warmup:
                self.launched.append((grid, args, options))
            return self.compiled.setdefault((args, tuple(options.items())), object())

    kernel = Kernel()
    call = lambda k: ((4 if k["CHUNK"] else 1,), ("x", k["CHUNK"] or 1024), dict(STAGES=k["STAGES"] or 2))
    key = {"D": 64}
    launch(kernel, "s", key, knobs("s", key, 4), call)
    assert kernel.launched == [((4,), ("x", 256), {"STAGES": 3})]     # the call's own knobs, launched once
    assert len(kernel.compiled) == 3                                  # and the key's two other knob sets, compiled
    for rows in (4, 32, 500):
        launch(kernel, "s", key, knobs("s", key, rows), call)
    assert len(kernel.compiled) == 3 and len(kernel.launched) == 4    # another row count compiles nothing
    ns["SELECTED"]["s"] = 1                                           # a sweep's selection: that configuration alone
    assert variants("s", key) == [{"CHUNK": 256, "STAGES": 3}]


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
                                           "experts": 2},
                     "blk.0.attn_k": {"name": "blk.0.attn_k", "N": 4, "K": 256, "format": "TQ2_0"}}


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
                        lambda max_workers, initializer, mp_context=None: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(sweep, "_measure_build", lambda *a, **k: (_ for _ in ()).throw(AssertionError("measured")))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "test device")
    monkeypatch.setattr(kbuild, "device_arch", lambda: ("sm_120", "12.0"))
    shapes = {"gemm": [{"N": 4096, "K": 4096, "format": "Q4_K"}]}
    tuning, report = sweep.sweep("gguf-rows", shapes=shapes, structures=["gemm"], out=tmp_path, chunk=4, log=lambda *_: None)
    assert len(calls) == 1   # one build ran, the rest were cancelled
    assert tuning["structures"]["gemm"] == {"configs": [kbuild.defaults(kbuild.find("gguf-rows")[1], "gemm")], "table": []}
    reasons = set(report["dropped"].values())
    assert len(reasons) == 1 and "the sweep was stopped (1 of" in reasons.pop()
    assert json.loads((tmp_path / "gguf-rows.json").read_text())["structures"]["gemm"]["table"] == []


def _pipeline(monkeypatch, stop_after=None):
    """_measure_all over three builds a thread pool finishes one at a time (each waits for its gate), with fakes for
    the GPU: what was timed, in what order, and whether the GPU lock was held then."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    events, lock_held = [], []
    gates = [threading.Event() for _ in range(3)]
    plan = [("s", [], [(0, {"K": 0})]), ("s", [{"K": 1}], [(1, {"K": 1})]), ("s", [{"K": 2}], [(1, {"K": 2})])]

    class Handle:
        def __init__(self):
            lock_held.append(True)

        def close(self):
            lock_held.append(False)

    def build(i):
        gates[i].wait(10)
        events.append(("built", i))
        return {"s": plan[i][1]}, ("does not compile" if i == 2 else None)

    def measure(kernel_id, root, manifest, item, benches, shape_sets, results, dropped, where, timer, torch):
        i = plan.index(item)
        events.append(("timed", i, lock_held[-1]))
        if all(("s", json.dumps(c, sort_keys=True)) in dropped for _, c in item[2]):
            return
        results[("s", (1,), 1, json.dumps(item[2][0][1], sort_keys=True))] = 1.0
        if i + 1 < len(gates):
            gates[i + 1].set()   # the next build finishes only once this one was timed: timing does not wait on all

    stopped = sweep._Stop()
    if stop_after is not None:
        def measure_then_stop(*args):
            measure(*args)
            if len([e for e in events if e[0] == "timed"]) == stop_after:
                stopped._stop()
        monkeypatch.setattr(sweep, "_measure_build", measure_then_stop)
    else:
        monkeypatch.setattr(sweep, "_measure_build", measure)
    monkeypatch.setattr(kbuild, "load", lambda *a, **k: "module")
    monkeypatch.setattr(sweep, "_lock", lambda path: Handle())
    monkeypatch.setattr(sweep, "Timer", lambda *a: "timer")
    monkeypatch.setattr(sweep, "choose", lambda *a: ({}, {}))
    monkeypatch.setattr(sweep, "_confirm", lambda *a: events.append(("confirmed", lock_held[-1])))
    monkeypatch.setattr(sweep, "_yardsticks", lambda *a: {})
    results, dropped = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(build, i): i for i in range(3)}

        def built(future, result):
            if result[1]:
                for _, c in plan[futures[future]][2]:
                    dropped[("s", json.dumps(c, sort_keys=True))] = result[1]
        gates[0].set()
        found = sweep._measure_all("toy", None, "lock", None, 3, 25, {}, plan, ["s"], {"s": []}, {}, results, dropped,
                                   {}, stopped, lambda *_: None, pool, futures, None, built)
        for gate in gates:
            gate.set()
    return found, events, results, dropped


def test_each_build_is_timed_as_it_finishes_under_the_lock(monkeypatch):
    """The sweep times a build's configurations as soon as that build finishes, not after every build: build 1 only
    finishes once build 0 was timed. Timing and the confirmation hold the GPU lock; a build that failed is not timed."""
    (builds, finished), events, results, dropped = _pipeline(monkeypatch)
    assert (builds, finished) == (3, 3)
    assert events[:4] == [("built", 0), ("timed", 0, True), ("built", 1), ("timed", 1, True)]
    assert ("timed", 2, True) in events and events[-1] == ("confirmed", True)
    assert len(results) == 2 and set(dropped.values()) == {"does not compile"}


def test_a_stop_while_builds_run_keeps_what_was_timed(monkeypatch):
    """A stop after the first build was timed ends the builds; the table is chosen from that build's timings and the
    rest is recorded as not measured, saying how many builds were done."""
    (builds, finished), events, results, dropped = _pipeline(monkeypatch, stop_after=1)
    assert len(results) == 1 and ("confirmed", True) not in events
    assert finished <= 2 and builds == 3
    assert all("the sweep was stopped" in why and "builds built" in why for why in dropped.values())
    assert len(dropped) == 2


class _Valid:
    """A bench whose STAGES 3 does not run 256-wide heads and whose CONVERT runs only over 8-bit pages; variants by
    page kind."""

    def valid_for(self, knobs, shape):
        return not (knobs["STAGES"] == 3 and shape["D"] == 256) and not (knobs["CONVERT"] and shape["kind"] == "bf16")

    def variants(self, shape):
        return [{"KIND": {"bf16": 0, "fp8": 1}[shape["kind"]]}]


VALID = {"id": "toy2", "structures": {"s": {"key": ["D"], "knobs": {
    "STAGES": {"values": [0, 3], "default": 0}, "CONVERT": {"values": [0, 1], "default": 0}}}}}


def test_configurations_that_run_no_shape_are_not_built():
    shapes = [{"D": 256, "kind": "bf16"}, {"D": 256, "kind": "fp8"}]
    found = sweep.configurations(VALID, "s")
    kept, left = sweep.runnable(VALID, "s", found, _Valid(), shapes)
    assert left == 2 and kept == [{"STAGES": 0, "CONVERT": 0}, {"STAGES": 0, "CONVERT": 1}]
    # the defaults are built even where they run nothing (index 0 of every build), a bench without valid_for keeps all
    kept, _ = sweep.runnable(VALID, "s", found, _Valid(), [{"D": 256, "kind": "bf16"}][:0])
    assert kept == [{"STAGES": 0, "CONVERT": 0}]
    assert sweep.runnable(VALID, "s", found, object(), shapes) == (found, 0)


def test_a_mixed_kernels_builds_compile_only_the_variants_their_configurations_run():
    shapes = [{"D": 128, "kind": "bf16"}, {"D": 128, "kind": "fp8"}]
    bench = _Valid()
    variants = sweep._variants(bench, shapes)
    assert variants == [{"KIND": 0}, {"KIND": 1}]
    found = sweep.configurations(VALID, "s")
    needs = {json.dumps(c, sort_keys=True): sweep._needs(bench, c, shapes, variants) for c in found}
    assert needs[json.dumps({"STAGES": 3, "CONVERT": 1}, sort_keys=True)] == (1,)       # fp8 pages only
    assert needs[json.dumps({"STAGES": 3, "CONVERT": 0}, sort_keys=True)] == (0, 1)
    chunks = sweep._chunks(VALID, "s", found, 8, lambda c: needs[json.dumps(c, sort_keys=True)])
    # the defaults first, with those that run both kinds; the CONVERT ones (fp8 only) share a build of their own
    assert [[c for _, c in indices] for _, indices in chunks] == [
        [{"STAGES": 0, "CONVERT": 0}, {"STAGES": 3, "CONVERT": 0}],
        [{"STAGES": 0, "CONVERT": 1}, {"STAGES": 3, "CONVERT": 1}]]
    assert [i for i, _ in chunks[1][1]] == [1, 2]          # their indices in their build (0 is the defaults)
    # without groups: one build in kernel.json's order, as before
    assert [[c for _, c in indices] for _, indices in sweep._chunks(VALID, "s", found, 8)] == [found]


def test_configurations_that_compile_alike_are_next_to_each_other():
    manifest = {"id": "toy3", "structures": {"s": {"knobs": {
        "CHUNK": {"values": [0, 128, 256], "default": 0},
        "WARPS": {"values": [4, 8], "default": 4},
        "RAMP": {"values": [0, 1], "default": 0}}}}}

    class Bench:
        runtime = ("CHUNK", "RAMP")
    found = sweep.configurations(manifest, "s", Bench())
    assert len(found) == 12
    # the compiled knob (WARPS) varies slowest: a build of six holds one compiled tile
    assert [c["WARPS"] for c in found] == [4] * 6 + [8] * 6
    assert [(c["CHUNK"], c["RAMP"]) for c in found[:6]] == [(0, 0), (0, 1), (128, 0), (128, 1), (256, 0), (256, 1)]
    # a bench that names none: kernel.json's product order, as before
    assert [c["WARPS"] for c in sweep.configurations(manifest, "s")[:4]] == [4, 4, 8, 8]
