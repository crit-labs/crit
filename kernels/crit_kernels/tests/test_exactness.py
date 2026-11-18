"""crit_kernels.exactness without a GPU: every structure of crit's kernels states its level, the levels order as they
say, a prompt structure is matched with the decode structure whose bits it has, an engine's setting takes what it
leaves out from the run's quality mode, and the sweep runs its row check on the structures that are row-exact or
higher."""
import json
from types import SimpleNamespace

import pytest

from crit_kernels import build as kbuild
from crit_kernels import exactness, sweep

TOY = {
    "id": "toy",
    "structures": {
        "decode": {"exactness": "row-exact"},
        "prompt": {"exactness": "exact", "bits_of": "decode"},
        "prompt-wide": {"exactness": "exact", "bits_of": "prompt"},
        "other-decode": {"exactness": "row-exact"},
        "host-order": {"exactness": "declared"},
        "host-order-gemm": {"exactness": "declared"},
        "native": {"exactness": "repeatable"},
    },
}


def test_every_structure_of_crits_kernels_states_a_level():
    found = kbuild.manifests()
    assert len(found) >= 16
    for kid, (_, manifest) in found.items():
        assert exactness.problems(manifest, explicit=True) == [], kid
        for sid, spec in manifest["structures"].items():
            assert "row_invariant" not in spec, (kid, sid)
            if spec["exactness"] == "exact":   # the bits of a structure that fixes them by itself
                root = manifest["structures"][exactness.reference(manifest, sid)]
                assert root["exactness"] in ("row-exact", "declared"), (kid, sid)


def test_a_level_keeps_the_levels_below_it():
    assert exactness.LEVELS == ("repeatable", "row-exact", "exact", "declared")
    for i, name in enumerate(exactness.LEVELS):
        spec = {"exactness": name, "bits_of": "x"} if name == "exact" else {"exactness": name}
        assert [exactness.keeps(spec, asked) for asked in exactness.LEVELS] == [j <= i for j in range(4)]
    with pytest.raises(ValueError):
        exactness.rank("bitwise")
    with pytest.raises(ValueError):
        exactness.level({"exactness": "bitwise"})


def test_a_manifest_from_before_the_levels_reads_by_its_flag():
    assert exactness.level({"row_invariant": True}) == "row-exact"
    assert exactness.level({"row_invariant": False}) == "repeatable"
    assert exactness.level({}) == "repeatable"
    assert exactness.problems({"id": "old", "structures": {"s": {"row_invariant": True}, "t": {}}}) == []
    assert len(exactness.problems({"id": "old", "structures": {"s": {"row_invariant": True}, "t": {}}}, explicit=True)) == 2


def test_bits_follow_bits_of_to_the_structure_that_fixes_them():
    assert exactness.reference(TOY, "prompt-wide") == "decode"
    assert exactness.reference(TOY, "decode") == "decode"
    assert exactness.same_bits(TOY, "prompt", "prompt-wide") and exactness.same_bits(TOY, "decode", "prompt-wide")
    assert not exactness.same_bits(TOY, "prompt", "other-decode")
    assert exactness.same_bits(TOY, "host-order", "host-order-gemm")       # one declared order
    assert not exactness.same_bits(TOY, "host-order", "decode")
    assert not exactness.same_bits(TOY, "native", "decode")


def test_the_structures_a_phase_may_run_at_each_level():
    every = list(TOY["structures"])
    assert exactness.choices(TOY, "repeatable") == every
    assert exactness.choices(TOY, "repeatable", decode="decode") == every
    assert exactness.choices(TOY, "row-exact", decode="decode") == [s for s in every if s != "native"]
    assert exactness.choices(TOY, "exact", decode="decode") == ["decode", "prompt", "prompt-wide"]
    assert exactness.choices(TOY, "exact", decode="other-decode") == ["other-decode"]
    assert exactness.choices(TOY, "exact", decode="host-order") == ["host-order", "host-order-gemm"]
    assert exactness.choices(TOY, "declared", decode="host-order") == ["host-order", "host-order-gemm"]
    assert exactness.choices(TOY, "declared", decode="decode") == []       # its decode rows follow no declared order
    # the decode phase itself: exact is row-exact there
    assert exactness.choices(TOY, "exact") == [s for s in every if s != "native"]
    assert exactness.choices(TOY, "declared") == ["host-order", "host-order-gemm"]


def test_a_real_kernels_prompt_structures_by_level():
    manifest = kbuild.manifests()["mma-rows"][1]
    assert exactness.choices(manifest, "exact", decode="decode-tile") == ["decode-tile", "prompt-tile"]
    assert exactness.choices(manifest, "exact", decode="fp4-tile") == ["fp4-tile", "fp4-prompt-tile"]
    blocks = kbuild.manifests()["fp8-block-rows"][1]
    assert exactness.choices(blocks, "declared", decode="rows") == ["rows", "gemm", "gemm-staged"]
    fast = set(exactness.choices(blocks, "repeatable", decode="rows")) - set(exactness.choices(blocks, "row-exact"))
    assert fast == {"gemm-native", "gemm-native-channel", "gemm-native-blocks"}


def test_a_setting_takes_what_it_leaves_out_from_the_quality_mode():
    assert exactness.for_quality_mode("preserve") == {"decode": "exact", "prompt": "exact"}
    assert exactness.for_quality_mode("speed") == {"decode": "repeatable", "prompt": "repeatable"}
    assert exactness.setting(quality_mode="preserve") == {"decode": "row-exact", "prompt": "exact"}
    assert exactness.setting(quality_mode="speed") == {"decode": "repeatable", "prompt": "repeatable"}
    assert exactness.setting("declared") == {"decode": "declared", "prompt": "declared"}
    assert exactness.setting({"prompt": "repeatable"}, "preserve") == {"decode": "row-exact", "prompt": "repeatable"}
    assert exactness.setting({"decode": "row-exact", "prompt": "exact"}) == {"decode": "row-exact", "prompt": "exact"}
    for wrong in ({"prompt": "exact"}, {"verify": "exact"}, {"decode": "bitwise", "prompt": "exact"}, "fast"):
        with pytest.raises(ValueError):
            exactness.setting(wrong)
    with pytest.raises(ValueError):
        exactness.for_quality_mode("balanced")


@pytest.mark.parametrize("structures,word", [
    ({"s": {"exactness": "bitwise"}}, "not one of"),
    ({"s": {"exactness": "exact"}}, "bits_of"),
    ({"s": {"exactness": "row-exact", "bits_of": "t"}, "t": {"exactness": "row-exact"}}, "bits_of"),
    ({"s": {"exactness": "exact", "bits_of": "missing"}}, "not another structure"),
    ({"s": {"exactness": "exact", "bits_of": "s"}}, "not another structure"),
    ({"s": {"exactness": "exact", "bits_of": "t"}, "t": {"exactness": "exact", "bits_of": "s"}}, "circle"),
    ({"s": {"exactness": "exact", "bits_of": "t"}, "t": {"exactness": "repeatable"}}, "row-exact or declared"),
    ({"s": {"exactness": "repeatable", "row_invariant": True}}, "row_invariant"),
    ({"s": {"bits_of": "t"}, "t": {"exactness": "row-exact"}}, "bits_of without"),
])
def test_a_wrong_level_is_named(structures, word):
    manifest = {"id": "toy", "structures": structures}
    found = exactness.problems(manifest)
    assert found and any(word in line for line in found), found
    with pytest.raises(ValueError):
        exactness.check(manifest)


def test_manifests_refuse_a_kernel_with_a_wrong_level(tmp_path):
    folder = tmp_path / "toy"
    folder.mkdir()
    manifest = {"schema": kbuild.SCHEMA, "id": "toy", "structures": {"s": {"exactness": "exact"}}}
    (folder / "kernel.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bits_of"):
        kbuild.manifests(tmp_path)
    manifest["structures"]["s"] = {"exactness": "row-exact"}
    (folder / "kernel.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert list(kbuild.manifests(tmp_path)) == ["toy"]


class _Row:
    """A row's bytes, as much of a tensor as the sweep's row check reads."""
    dtype = SimpleNamespace(is_floating_point=False)

    def __init__(self, value):
        self.value = value

    def clone(self):
        return _Row(self.value)


class _Bench:
    """A kernel whose row 0 follows the row count: the sum of a split that follows it."""

    def inputs(self, shape, rows, device):
        return {"rows": rows}

    def run(self, module, state):
        return state["rows"]

    def reference(self, state):
        return None

    def check(self, got, want, state):
        return None

    def first_row(self, out):
        return _Row(1.0 if out == 1 else 1.5)


@pytest.mark.parametrize("level,kept", [("declared", [1]), ("exact", [1]), ("row-exact", [1]), ("repeatable", [1, 8]),
                                        (None, [1, 8])])
def test_the_sweep_checks_row_zero_alone_for_row_exact_structures_and_higher(level, kept):
    spec = {"key": ["N"], "tune_rows": [1, 8]}
    if level:
        spec["exactness"] = level
    if level == "exact":
        spec["bits_of"] = "decode"
    manifest = {"id": "toy", "structures": {"s": spec, "decode": {"exactness": "row-exact"}}}
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None), uint8=None,
                            equal=lambda a, b: a.value == b.value)
    results, dropped = {}, {}
    sweep._measure("s", "{}", None, manifest, {"s": _Bench()}, {"s": [{"N": 4}]}, results, dropped,
                   lambda call: 1.0, torch)
    assert sorted(rows for (_, _, rows, _) in results) == kept
    assert bool(dropped) == (kept == [1])
    if dropped:
        assert "row 0 differs from row 0 alone" in next(iter(dropped.values()))
