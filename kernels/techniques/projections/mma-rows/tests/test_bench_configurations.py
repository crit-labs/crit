"""The sweep's configurations of the prompt tiles: no two that compile to the same kernel (PR 2 runs with PF 0 whatever
PF says), and PK, the launch's parts of K, a run-time knob (configurations that differ in it alone share their compiled
code). Without a GPU."""
from crit_kernels import build as kbuild
from crit_kernels import sweep as ksweep


def _configurations(structure):
    folder, manifest = kbuild.find("mma-rows")
    bench = ksweep._bench_class(folder, manifest["structures"][structure]["bench"])
    return manifest, bench, ksweep.configurations(manifest, structure, bench)


def test_prompt_tile_sweeps_pr_2_once_and_keeps_its_defaults():
    manifest, bench, found = _configurations("prompt-tile")
    assert not [c for c in found if c["PR"] == 2 and c["PF"] == 1]
    assert [c for c in found if c["PR"] == 2 and c["PF"] == 0]
    assert kbuild.defaults(manifest, "prompt-tile") in found
    knobs = manifest["structures"]["prompt-tile"]["knobs"]
    raw = 1
    for spec in knobs.values():
        raw *= len(spec["values"])
    dropped = raw // len(knobs["PR"]["values"]) // len(knobs["PF"]["values"])   # PR 2 with PF 1, every other knob
    assert len(found) == raw - dropped


def test_pk_is_a_run_time_knob_of_both_prompt_tiles():
    for structure in ("prompt-tile", "fp4-prompt-tile"):
        _, bench, found = _configurations(structure)
        assert tuple(bench.runtime) == ("PK",)
        compiled = [tuple((n, v) for n, v in c.items() if n != "PK") for c in found]
        # consecutive: a build of consecutive configurations compiles each kernel once
        runs = [k for i, k in enumerate(compiled) if i == 0 or compiled[i - 1] != k]
        assert len(runs) == len(set(compiled))
