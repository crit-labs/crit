"""crit_kernels.build.retire: a tuning written before a structure retired (kernel.json "retired") has its tables moved
into the structure that replaced them, so check_tuning takes it; decode-gemv's retired per-format GEMVs as the case."""
import pytest

from crit_kernels import build


def _manifest():
    return build.find("decode-gemv")[1]


def _old(structure, configs, table):
    return {"schema": build.TUNING_SCHEMA, "kernel": "decode-gemv",
            "structures": {structure: {"configs": configs, "table": table}}}


def test_an_old_cuda_table_becomes_entries_of_gemv_cuda_a_weight_format_each():
    m = _manifest()
    old = _old("gemv-fp8", [{"WARPS": 2, "ROWS": 8, "PREF": 1, "STREAM": 1}],
               [{"key": {"N": 1000, "K": 2048}, "rows": [1, 16], "config": 0},
                {"key": None, "rows": [1, 16], "config": 0}])   # an entry for every shape: dropped
    got = build.check_tuning(m, old)["structures"]["gemv-cuda"]
    assert got["configs"][0] == build.defaults(m, "gemv-cuda")
    assert [e["key"] for e in got["table"]] == [{"N": 1000, "K": 2048, "weights": 1},
                                               {"N": 1000, "K": 2048, "weights": 2}]
    assert got["configs"][got["table"][0]["config"]] == {"WARPS": 2, "ROWS": 8, "WPR": 1, "PREF": 1, "STREAM": 1}
    assert "gemv-fp8" not in build.check_tuning(m, old)["structures"]


def test_old_triton_tables_rename_their_knobs_and_fix_the_multiply_path():
    m = _manifest()
    tuning = _old("gemv-nvfp4-triton", [{"BLOCK_N": 16, "WARPS": 8, "STAGES": 2}],
                  [{"key": {"N": 5000, "K": 4096}, "rows": [1, 8], "config": 0}])
    tuning["structures"]["gemv-fp8-dot"] = {"configs": [{"BN": 16, "BK": 128, "WARPS": 2, "SPLIT": 2}],
                                            "table": [{"key": {"N": 5000, "K": 4096}, "rows": [1, 1], "config": 0}]}
    got = build.check_tuning(m, tuning)["structures"]["gemv-triton"]
    by_weights = {e["key"]["weights"]: (e["rows"], got["configs"][e["config"]]) for e in got["table"]}
    assert by_weights[0] == ([1, 8], {"MUL": 1, "BN": 16, "BK": 512, "WARPS": 8, "STAGES": 2, "SPLIT": 0})
    assert by_weights[1] == ([1, 16], {"MUL": 2, "BN": 16, "BK": 128, "WARPS": 2, "STAGES": 0, "SPLIT": 2})


def test_where_the_new_structure_is_tuned_too_only_the_old_non_default_entries_move_first():
    m = _manifest()
    tuning = _old("gemv-nvfp4", [{"WARPS": 4, "ROWS": 2, "WPR": 1, "PREF": 0, "STREAM": 0},   # its defaults
                                 {"WARPS": 8, "ROWS": 1, "WPR": 1, "PREF": 2, "STREAM": 1}],
                  [{"key": {"N": 64, "K": 256}, "rows": [1, 16], "config": 0},
                   {"key": {"N": 128, "K": 256}, "rows": [1, 16], "config": 1}])
    mine = dict(build.defaults(m, "gemv-cuda"), WARPS=2)
    tuning["structures"]["gemv-cuda"] = {"configs": [mine],
                                         "table": [{"key": {"weights": 0, "N": 64, "K": 256}, "rows": [1, 16],
                                                    "config": 0}]}
    got = build.check_tuning(m, tuning)["structures"]["gemv-cuda"]
    keys = [(e["key"], got["configs"][e["config"]]) for e in got["table"]]
    assert keys[0][0] == {"N": 128, "K": 256, "weights": 0} and keys[0][1]["WARPS"] == 8
    assert keys[1] == ({"weights": 0, "N": 64, "K": 256}, mine) and len(keys) == 2


def test_a_retired_structure_that_names_no_existing_one_is_refused():
    m = dict(_manifest(), retired={"gone": {"into": "nothing"}})
    with pytest.raises(ValueError, match="names nothing"):
        build.check_tuning(m, _old("gone", [], []))


def test_a_tuning_without_retired_structures_is_untouched():
    m = _manifest()
    tuning = build.default_tuning(m)
    assert build.retire(m, tuning) is tuning


def test_a_tuning_written_with_a_retired_knob_loads_with_the_knob_dropped():
    # mma-rows retired SPLITS (the sums' order is the matrix's, never a tuning's): an engine's tuning swept before
    # still loads, each configuration without it and every entry naming the same configuration
    m = build.find("mma-rows")[1]
    old = {"schema": build.TUNING_SCHEMA, "kernel": "mma-rows",
           "structures": {"decode-tile": {"configs": [{"GROUPS": 1, "WARPS": 4, "SPLITS": 3, "SW": 0, "SPF": 0}],
                                          "table": [{"key": {"weights": 0, "N": 34816, "K": 5120}, "rows": [1, 8],
                                                     "config": 0}]}}}
    got = build.check_tuning(m, old)["structures"]["decode-tile"]
    assert all("SPLITS" not in config for config in got["configs"])
    assert got["configs"][got["table"][0]["config"]] == {"GROUPS": 1, "WARPS": 4, "SW": 0, "SPF": 0}
    with pytest.raises(ValueError, match="unknown knob 'SPLIT'"):
        build.check_tuning(m, {**old, "structures": {"decode-tile": {"configs": [{"SPLIT": 3}], "table": []}}})
