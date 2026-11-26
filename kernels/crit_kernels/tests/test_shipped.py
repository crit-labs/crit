"""crit_kernels.shipped: the tuning tables a kernel folder carries (tunings/<capability>.json), applied only on the
compute capability they were swept on, after the engine's own tuning and before the default knobs; the sweep leaves out
the shapes they cover; `crit kernels ship` merges a sweep into them. Without a GPU."""
import json

import pytest

from crit_kernels import build as kbuild
from crit_kernels import shipped
from crit_kernels import sweep as ksweep

KERNEL = "mma-rows"
STRUCTURE = "decode-tile"
A = {"weights": 0, "N": 4096, "K": 4096}
B = {"weights": 1, "N": 1024, "K": 4096}
C = {"weights": 3, "N": 2048, "K": 2048}
SWEEP = {"source": "a" * 64, "device": "a GPU of capability 12.0", "capability": "12.0", "date": "2026-11-20"}


def _manifest():
    return kbuild.find(KERNEL)[1]


def _knobs(**given):
    return dict(kbuild.defaults(_manifest(), STRUCTURE), **given)


def _table(entries, sweeps=(SWEEP,), capability="12.0"):
    """A shipped table: entries [(key, knobs, sweep index)], each for every row count."""
    configs, table = [_knobs()], []
    for key, knobs, i in entries:
        if knobs not in configs:
            configs.append(knobs)
        table.append({"key": key, "rows": [1, 64], "config": configs.index(knobs), "sweep": i})
    return {"schema": kbuild.TUNING_SCHEMA, "kernel": KERNEL, "capability": capability, "sweeps": list(sweeps),
            "structures": {STRUCTURE: {"configs": configs, "table": table, "report": []}}}


def _tuning(entries, **fields):
    configs, table = [_knobs()], []
    for key, knobs in entries:
        if knobs not in configs:
            configs.append(knobs)
        table.append({"key": key, "rows": [1, 64], "config": configs.index(knobs)})
    return dict({"schema": kbuild.TUNING_SCHEMA, "kernel": KERNEL,
                 "structures": {STRUCTURE: {"configs": configs, "table": table}}}, **fields)


def _picked(tuning, key):
    t = tuning["structures"][STRUCTURE]
    found = [t["configs"][e["config"]] for e in t["table"] if e["key"] == key]
    return found[0] if found else None


def test_every_shipped_table_passes_check_tuning_against_its_kernel():
    """Every table in crit's kernel folders is a sweep's measurement of the current kernel.json's knobs: its file named
    by its capability, each entry keyed by its structure's key fields and naming a sweep with its provenance."""
    seen = 0
    for kid, (folder, manifest) in kbuild.manifests().items():
        for found in sorted((folder / shipped.FOLDER).glob("*.json")):
            seen += 1
            table = json.loads(found.read_text(encoding="utf-8"))
            assert table["kernel"] == kid and found.stem == str(table["capability"]), found
            kbuild.check_tuning(manifest, table)   # raises on a knob or value kernel.json no longer has
            assert table["sweeps"], found
            for sweep in table["sweeps"]:
                assert set(sweep) >= {"source", "device", "capability", "date"} and sweep["capability"] == found.stem
            for sid, t in table["structures"].items():
                spec = manifest["structures"][sid]
                assert t["configs"][0] == kbuild.defaults(manifest, sid), (found, sid)
                for e in t["table"]:
                    assert sorted(e["key"]) == sorted(spec["key"]), (found, sid, e)
                    assert 0 <= e["sweep"] < len(table["sweeps"])
    assert seen >= 0


def test_the_engine_tuning_comes_first_the_table_fills_what_it_lacks():
    table = _table([(A, _knobs(WARPS=2), 0), (B, _knobs(WARPS=1), 0)])
    mine = _tuning([(A, _knobs(WARPS=4, GROUPS=2))])
    got, taken = shipped.fill(_manifest(), mine, table)
    assert _picked(got, A) == _knobs(WARPS=4, GROUPS=2)   # the engine's own entry for A
    assert _picked(got, B) == _knobs(WARPS=1)             # the table's for B, which the engine lacks
    assert _picked(got, C) is None                         # neither: the defaults (configuration 0)
    assert taken == {STRUCTURE: {shipped.shape_key(_manifest()["structures"][STRUCTURE], B)}}


def test_without_an_engine_tuning_the_table_comes_before_the_default_rules():
    manifest = _manifest()
    manifest = dict(manifest, structures=dict(manifest["structures"], **{STRUCTURE: dict(
        manifest["structures"][STRUCTURE], default_rows=[{"rows": [1, 64], "knobs": {"WARPS": 2}}])}))
    got, _ = shipped.fill(manifest, None, _table([(A, _knobs(WARPS=1), 0)]))
    table = got["structures"][STRUCTURE]["table"]
    assert table[0]["key"] == A and table[-1]["key"] is None


def test_a_table_applies_only_on_its_capability(monkeypatch, tmp_path):
    target = tmp_path / "12.0.json"
    target.write_text(json.dumps(_table([(A, _knobs(WARPS=2), 0)])), encoding="utf-8")
    monkeypatch.setattr(shipped, "path", lambda folder, capability: tmp_path / f"{capability}.json")
    manifest = _manifest()
    here, where, taken = shipped.apply(KERNEL, manifest, None, capability="12.0")
    assert where == target and _picked(here, A) == _knobs(WARPS=2) and taken
    other, where, taken = shipped.apply(KERNEL, manifest, None, capability="8.9")
    assert where is None and not taken and other == kbuild.default_tuning(manifest)
    monkeypatch.setattr(kbuild, "device_capability", lambda: None)   # no GPU to ask: no table
    assert shipped.apply(KERNEL, manifest, None)[1] is None


def test_a_table_that_fails_check_tuning_is_ignored_with_a_warning(monkeypatch, tmp_path, capsys):
    bad = _table([(A, _knobs(WARPS=2), 0)])
    bad["structures"][STRUCTURE]["configs"][1]["WARPS"] = 3   # a value kernel.json does not have
    (tmp_path / "12.0.json").write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(shipped, "path", lambda folder, capability: tmp_path / f"{capability}.json")
    monkeypatch.setattr(shipped, "_warned", set())
    assert shipped.table(KERNEL, "12.0") == (None, None)
    assert "not used" in capsys.readouterr().err


def test_the_sweep_leaves_out_covered_shapes_and_writes_their_entries(monkeypatch):
    manifest = _manifest()
    table = _table([(A, _knobs(WARPS=2), 0)])
    table["structures"][STRUCTURE]["report"] = [{"shape": A, "rows": 1, "tuned_us": 5.0, "sweep": 0}]
    monkeypatch.setattr(shipped, "table", lambda kernel_id, capability=None, root=None: (table, "tunings/12.0.json"))
    sets = {STRUCTURE: [dict(A, name="a"), dict(B, name="b")]}
    lines = []
    got, covered = ksweep._covered(KERNEL, manifest, [STRUCTURE], sets, None, lines.append, capability="12.0")
    assert [s["name"] for s in sets[STRUCTURE]] == ["b"] and "1 of 2 shapes covered" in lines[0]
    swept = _tuning([(B, _knobs(WARPS=1))])
    tuning, report = ksweep._with_shipped(manifest, [STRUCTURE], swept, {"structures": {STRUCTURE: []}}, got,
                                          covered)
    assert _picked(tuning, A) == _knobs(WARPS=2) and _picked(tuning, B) == _knobs(WARPS=1)
    assert tuning["shipped"] == {STRUCTURE: [A]}
    library = (manifest["structures"][STRUCTURE].get("yardstick") or {}).get("library")
    assert report["structures"][STRUCTURE] == [{"shape": A, "rows": 1, "tuned_us": 5.0, "sweep": 0, "library": library,
                                                "shipped": True}]


def test_merge_takes_a_sweep_and_prefers_the_newest_source():
    manifest = _manifest()
    current = "c" * 64
    old = _table([(A, _knobs(WARPS=2), 0), (B, _knobs(WARPS=2), 0)])
    sweep = _tuning([(A, _knobs(WARPS=1)), (C, _knobs(WARPS=1))], source=current, device="a GPU",
                    capability="12.0", date="2026-11-25")
    report = {"structures": {STRUCTURE: [{"shape": C, "rows": 1, "tuned_us": 3.0}]}}
    table, stats = shipped.merge(manifest, old, sweep, report, current=current)
    assert stats[STRUCTURE] == {"added": 1, "replaced": 1, "kept": 1, "dropped": 0}
    assert _picked(table, A) == _knobs(WARPS=1) and _picked(table, B) == _knobs(WARPS=2)
    assert table["structures"][STRUCTURE]["report"] == [{"shape": C, "rows": 1, "tuned_us": 3.0, "sweep": 1}]
    assert [s["source"] for s in table["sweeps"]] == ["a" * 64, current]
    # an older source's sweep does not displace an entry swept on the current source, whatever its date
    late = _tuning([(A, _knobs(WARPS=4, GROUPS=2))], source="b" * 64, device="a GPU", capability="12.0",
                   date="2026-12-01")
    again, stats = shipped.merge(manifest, table, late, current=current)
    assert _picked(again, A) == _knobs(WARPS=1) and stats[STRUCTURE]["kept"] == 3


def test_merge_skips_the_entries_a_sweep_copied_from_the_table():
    sweep = _tuning([(A, _knobs(WARPS=1)), (B, _knobs(WARPS=1))], source="c" * 64, device="a GPU",
                    capability="12.0", date="2026-11-25", shipped={STRUCTURE: [A]})
    table, stats = shipped.merge(_manifest(), _table([(A, _knobs(WARPS=2), 0)]), sweep)
    assert _picked(table, A) == _knobs(WARPS=2) and stats[STRUCTURE]["added"] == 1


def test_merge_refuses_a_tuning_without_its_provenance_or_from_another_capability():
    with pytest.raises(ValueError, match="written by hand"):
        shipped.merge(_manifest(), None, _tuning([(A, _knobs(WARPS=1))]))
    other = _tuning([(A, _knobs(WARPS=1))], source="c" * 64, device="a GPU", capability="8.9", date="2026-11-25")
    with pytest.raises(ValueError, match="capability 8.9"):
        shipped.merge(_manifest(), _table([]), other)


def test_merge_drops_table_entries_whose_knobs_no_longer_exist():
    old = _table([(A, _knobs(WARPS=2), 0), (B, _knobs(WARPS=2), 0)])
    old["structures"][STRUCTURE]["configs"].append(dict(_knobs(), WARPS=3))
    old["structures"][STRUCTURE]["table"][1]["config"] = 2
    sweep = _tuning([(C, _knobs(WARPS=1))], source="c" * 64, device="a GPU", capability="12.0", date="2026-11-25")
    table, stats = shipped.merge(_manifest(), old, sweep)
    assert stats[STRUCTURE]["dropped"] == 1 and _picked(table, B) is None and _picked(table, A) == _knobs(WARPS=2)


def _plugin_load(monkeypatch, tmp_path, capability, tuned=True):
    from crit_plugins import kernels as pk
    (tmp_path / "12.0.json").write_text(json.dumps(_table([(A, _knobs(WARPS=2), 0)])), encoding="utf-8")
    monkeypatch.setattr(shipped, "path", lambda folder, cap: tmp_path / f"{cap}.json")
    given = {}

    def load(name, tuning=None, shipped=True):
        given.update(tuning=tuning, shipped=shipped)
        return object()

    monkeypatch.setattr(kbuild, "device_capability", lambda: capability)
    monkeypatch.setattr(kbuild, "load", load)
    monkeypatch.setattr(pk, "declared", lambda folder, name: ({"name": name, "fallback": "torch"}, "toy"))
    pk.forget()
    try:
        assert pk.load_kernel(KERNEL, folder=tmp_path, tuned=tuned) is not None
        return given, pk.loaded()[KERNEL if tuned else KERNEL + pk.UNTUNED]
    finally:
        pk.forget()


def test_the_plugin_loader_records_the_shipped_table_on_its_capability(monkeypatch, tmp_path):
    given, record = _plugin_load(monkeypatch, tmp_path, "12.0")
    assert given["shipped"] is True and record["source"] == "tuned" and record["shipped"].endswith("12.0.json")
    given, record = _plugin_load(monkeypatch, tmp_path, "8.9")
    assert record["source"] == "built" and "shipped" not in record


def test_a_kernel_asked_for_untuned_takes_no_table(monkeypatch, tmp_path):
    given, record = _plugin_load(monkeypatch, tmp_path, "12.0", tuned=False)
    assert given["shipped"] is False and record["source"] == "built" and "shipped" not in record
