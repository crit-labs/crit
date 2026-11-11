"""A kernel structure that aliases another kernel's (rows here, and fp8-block-rows' rows, of decode-gemv's declared rows),
loaded the way a plugin loads it (crit_plugins.kernels.load_kernel): the target takes the engine's tuning folder
(use_tuning()), and it is recorded in loaded() under its own id, for the plugin, with the alias it came through.

Run on its own (a process whose forwarded targets are not loaded yet): an alias builds its target once a process."""
import json
import sys
from pathlib import Path

import torch

from crit_kernels import build as kbuild
from crit_plugins import kernels as pk

ROOT = Path(__file__).resolve().parents[5]
dev = torch.device("cuda")


def _target(module):
    return sys.modules[module.__name__ + "._crit_aliases"].module("rows")


def test_a_forwarded_target_takes_the_tuning_folder_a_plugin_sets(tmp_path):
    _, manifest = kbuild.find("decode-gemv")
    base = kbuild.defaults(manifest, "rows-fp8-block")
    other = dict(base, WARPS=8, AHEAD=3, STREAM=1)
    tuning = kbuild.default_tuning(manifest)
    tuning["structures"]["rows-fp8-block"] = {"configs": [base, other],
                                              "table": [{"key": None, "rows": [1, 8], "config": 1}]}
    (tmp_path / "decode-gemv.json").write_text(json.dumps(tuning))
    pk.forget()
    pk.use_tuning(tmp_path)
    try:
        fb = pk.load_kernel("fp8-block-rows", folder=ROOT / "inference" / "quantized-projections")
        target = _target(fb)
        held = sys.modules[target.__name__ + "._crit_tuning"].TUNING["rows-fp8-block"]
        assert held["configs"][held["table"][0]["config"]] == other
        record = pk.loaded()["decode-gemv"]
        assert record["source"] == "tuned" and record["tuning"] == str(tmp_path / "decode-gemv.json")
    finally:
        pk.use_tuning(None)
        pk.forget()


def test_a_forwarded_target_is_recorded_for_the_plugin():
    pk.forget()
    try:
        folder = ROOT / "inference" / "host-expert-arithmetic"
        ib = pk.load_kernel("int-block-experts", folder=folder)
        g = torch.Generator().manual_seed(1)
        codes = torch.randint(0, 256, (64, 64), dtype=torch.uint8, generator=g).to(dev)
        scales = torch.randint(0x20, 0x5F, (64, 8), generator=g).to(torch.uint8).to(dev)
        values, act_scales = ib.quantize_int8(torch.randn(2, 128, generator=g).to(dev), 32)
        out = torch.zeros(2, 64, device=dev)
        ib.rows(ib.matrix_table([(codes, scales, 0.5, False)]), ib.row_jobs([(0, [0, 1], [0, 1])], dev), 64, values,
                act_scales, 32, out)
        record = pk.loaded()["decode-gemv"]
        plugin = pk.loaded()["int-block-experts"]["plugins"]
        assert record["via"] == "int-block-experts/rows" and record["source"] in ("built", "tuned")
        assert set(plugin) <= set(record["plugins"])
    finally:
        pk.forget()
