"""crit_kernels.gates: the planted controls fail exactly their gate (on a CUDA device), and the host parts (edge rows,
resource records, limits) need none."""
import pytest

from crit_kernels import gates


def test_edge_rows_walk_every_power_of_two_edge():
    assert gates.edge_rows(1, 16) == [1, 2, 3, 4, 5, 7, 8, 9, 15, 16]
    assert gates.edge_rows(1, 17)[-3:] == [15, 16, 17]
    assert gates.edge_rows(4, 8) == [4, 5, 7, 8]


def test_resource_records_and_limits():
    records = gates.resource_usage(gates.LIMITS_CONTROL)
    assert [(r[0], r[1]) for r in records] == [("120", "control_too_many_registers"), ("120", "control_too_much_shared"),
                                              ("120", "control_fits")]
    failures, spills = gates.check_limits(records)
    assert len(failures) == 2 and len(spills) == 1
    assert gates.check_limits([("90a", "k", 255, 48 * 1024, 0)]) == ([], [])


def test_controls_fail_their_gates():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the controls run on a CUDA device")
    report = gates.controls()
    assert all(c["ok"] for c in report.values()), report
