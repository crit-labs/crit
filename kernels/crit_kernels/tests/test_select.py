"""crit_kernels.select: the structures kernel.json says compute an operation at a level and row count, the times a
sweep report holds for them, and the fastest by those times with today's choice where they say nothing."""
from crit_kernels import select


def test_fitting_follows_the_operation_the_level_and_the_rows():
    found = select.fitting("gated-delta", kernels=["recurrent-scans"])
    assert ("recurrent-scans", "delta-prefill-tc") in found and ("recurrent-scans", "delta-chunk-rows") in found
    exact = select.fitting("gated-delta", asked="row-exact", kernels=["recurrent-scans"])
    assert ("recurrent-scans", "delta-prefill-tc") not in exact   # repeatable: its bits follow the prompt's chunking
    assert ("recurrent-scans", "delta-chunk-rows") in exact and ("recurrent-scans", "pool-delta") in exact
    assert ("recurrent-scans", "delta-columns") not in select.fitting("gated-delta", rows=100,
                                                                      kernels=["recurrent-scans"])   # rows 1-64
    assert select.fitting("no-such-operation") == []


REPORT = {"structures": {
    "gemm-native": [{"shape": {"N": 512, "K": 256}, "rows": 128, "tuned_us": 10.0, "library_us": 12.0},
                    {"shape": {"N": 512, "K": 256}, "rows": 2048, "tuned_us": 90.0, "library_us": 80.0},
                    {"shape": {"N": 1024, "K": 256}, "rows": 128, "tuned_us": 30.0}]}}


def test_times_read_a_structure_at_a_shape():
    assert select.times(REPORT, "gemm-native", {"N": 512, "K": 256}) == {128: 10.0, 2048: 90.0}
    assert select.times(REPORT, "gemm-native", {"N": 512, "K": 256}, library=True) == {128: 12.0, 2048: 80.0}
    assert select.times(REPORT, "gemm-native", {"N": 1024, "K": 256}, library=True) == {}
    assert select.times(None, "gemm-native", {"N": 512}) == {}


def test_fastest_by_row_count_and_over_every_count_with_the_default_where_unmeasured():
    mine, library = {128: 10.0, 2048: 90.0}, {128: 12.0, 2048: 80.0}
    options = [("native", mine), ("provider", library)]
    assert select.fastest(options, 128) == "native"
    assert select.fastest(options, 1000) == "provider"      # the smallest timed count at or above it: 2048
    assert select.fastest(options, 99999) == "provider"     # past every count: the largest timed
    assert select.fastest(options) == "provider"            # the sums over the counts both timed: 100 against 92
    assert select.fastest([("native", mine), ("provider", {})], 128, "native") == "native"   # unmeasured: default
    assert select.fastest([("a", {1: 5.0}), ("b", {2: 4.0})], None, "a") == "a"            # no shared count
    assert select.fastest([("a", {1: 5.0}), ("b", {1: 5.0})], 1) == "a"                    # ties: the earlier


def test_crossover_is_the_last_count_the_first_is_no_slower_at_every_count_up_to():
    gemv, tile = {1: 5.0, 2: 6.0, 4: 9.0, 8: 15.0}, {1: 6.0, 2: 6.0, 4: 8.0, 8: 9.0}
    assert select.crossover(gemv, tile, 16) == 2
    assert select.crossover(gemv, tile, 1) == 1
    assert select.crossover(tile, gemv, 16) == 0
    assert select.crossover({}, tile, 16) == 0
