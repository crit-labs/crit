"""The operation contract (kernels/crit_kernels/contracts.py): every operation of crit's techniques has its FP64
reference and tolerance there, and every structure of every kernel maps to operations a technique defines."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "cli", "kernels"):   # crit_kernels reads crit's records
    sys.path.insert(0, str(ROOT / _folder))
from crit_kernels import contracts  # noqa: E402


class KernelContractTests(unittest.TestCase):
    def test_every_operation_has_a_contract_and_every_structure_maps_to_defined_operations(self):
        self.assertEqual(contracts.problems(ROOT), [])


if __name__ == "__main__":
    unittest.main()
