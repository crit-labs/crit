"""crit_core.batch, the Python twin of Core's crit/batch.h: the same layouts, rows and reservations as
tests/core/batch_tests.cpp checks for the C++ library."""
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

from crit_core.batch import (DECODE, DEFERRED, INFEASIBLE, PREFILL, READY, PageReservations, StepItem,  # noqa: E402
                             StepLayout, row_pages, step_item)


def row(prefill_tokens=0, token_work=0, prefill_complete=False):
    return types.SimpleNamespace(prefill_tokens=prefill_tokens, output_tokens=9, token_work=token_work,
                                 prefill_complete=prefill_complete)


class StepLayoutTests(unittest.TestCase):
    def test_decode_items_come_first_with_their_logits_rows(self):
        items = [StepItem(1, PREFILL, 3, 1), StepItem(2, DECODE, 1, 1), StepItem(3, PREFILL, 4, 0),
                 StepItem(4, DECODE, 1, 1), StepItem(5, DECODE, 3, 3)]
        layout = StepLayout(items)
        self.assertEqual((layout.rows, layout.decode_rows, layout.decode_items), (12, 5, 3))
        self.assertEqual([p.key for p in layout.placements], [2, 4, 5, 1, 3])
        self.assertEqual(layout.placement(4)[3:], (2, 3, 2, 3, 3))
        self.assertEqual(layout.placement(0)[3:], (5, 3, 5, 1, 1))
        self.assertEqual(layout.logit_rows, [0, 1, 2, 3, 4, 7])
        self.assertEqual(layout.read_rows, layout.logit_rows)
        self.assertEqual(layout.placement(2).read, 0)

    def test_readers_keep_every_last_layer_row(self):
        layout = StepLayout([StepItem(1, PREFILL, 4, 0, True), StepItem(2, DECODE, 2, 2), StepItem(3, PREFILL, 3, 1)])
        self.assertEqual(layout.logit_rows, [0, 1, 8])
        self.assertEqual(layout.read_rows, [0, 1, 2, 3, 4, 5, 8])
        self.assertEqual([p.read for p in layout.placements], [2, 4, 1])

    def test_invalid_items_leave_the_layout(self):
        layout = StepLayout([StepItem(9, PREFILL, 2, 1)])
        for bad in ([StepItem(1, PREFILL, 0, 0)], [StepItem(1, PREFILL, 2, 3)], [StepItem(1, DECODE, 1, 0)],
                    [StepItem(1, "verify", 1, 1)], [StepItem(1, PREFILL, 2 ** 31 - 1, 0), StepItem(2, DECODE, 1, 1)]):
            with self.assertRaises(ValueError):
                layout.arrange(bad)
        self.assertEqual((layout.rows, layout.logit_rows), (2, [1]))
        self.assertEqual(StepLayout().rows, 0)


class StepItemTests(unittest.TestCase):
    def test_decode_rows_follow_the_grant(self):
        self.assertEqual(step_item(row(prefill_complete=True), 10, 10), StepItem(None, DECODE, 1, 1))
        self.assertEqual(step_item(row(prefill_complete=True), 10, 10, 5).rows, 5)
        self.assertEqual(step_item(row(token_work=3, prefill_complete=True), 10, 10, 5).rows, 3)
        self.assertEqual(step_item(row(token_work=1, prefill_complete=True), 10, 10, 5).rows, 1)
        with self.assertRaises(ValueError):
            step_item(row(prefill_complete=True), 10, 10, 0)

    def test_prompt_rows_end_the_prompt_with_logits(self):
        self.assertEqual(step_item(row(16), 40, 0, key="r"), StepItem("r", PREFILL, 16, 0))
        self.assertEqual(step_item(row(16), 40, 30), StepItem(None, PREFILL, 10, 1))
        self.assertEqual(step_item(row(16, token_work=4), 40, 30), StepItem(None, PREFILL, 4, 0))
        for bad in ((row(16), 40, 40), (row(0), 40, 0)):
            with self.assertRaises(ValueError):
                step_item(*bad)


class PageTests(unittest.TestCase):
    def test_page_reservations(self):
        self.assertEqual([row_pages(r, 16) for r in (0, 1, 16, 17)], [0, 1, 1, 2])
        pool = PageReservations(10, 16, 160)
        self.assertEqual((pool.need(53, 24), pool.admission(53, 24)), (5, READY))
        self.assertEqual((pool.reserve(53, 24), pool.reserve(53, 24), pool.reserved), (5, 5, 10))
        self.assertEqual(pool.admission(53, 24), DEFERRED)
        with self.assertRaises(RuntimeError):
            pool.reserve(53, 24)
        pool.release(5)
        self.assertEqual(pool.admission(53, 24), READY)
        with self.assertRaises(RuntimeError):
            pool.release(6)
        self.assertEqual((pool.admission(0, 4), pool.admission(53, 108), pool.need(53, 107)), (INFEASIBLE, INFEASIBLE, 10))
        speculative = PageReservations(10, 16, 160, 4)
        self.assertEqual((speculative.need(13, 3), speculative.need(150, 10)), (2, 10))
        for bad in ((0, 16, 16), (1, 16, 17), (1, 16, 16, 17)):
            with self.assertRaises(ValueError):
                PageReservations(*bad)


if __name__ == "__main__":
    unittest.main()
