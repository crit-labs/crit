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
