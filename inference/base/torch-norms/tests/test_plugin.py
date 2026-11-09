"""torch-norms' host form: the norms slot's FP64 rows, on worked values and crit's cases."""
import math

from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase


class TorchNormsHostTests(PluginTestCase):
    def test_it_has_the_norms_interface_and_its_device_form(self):
        self.assertImplements()

    def test_an_rms_norm_scales_a_row_to_unit_root_mean_square(self):
        norms = self.assemble()["norms"]
        (y,), = [norms.rms_norm_rows([[3.0, 4.0]], [1.0, 2.0], 0.0)]
        rms = math.sqrt((9 + 16) / 2)
        self.assertClose(y, [3 / rms, 2 * 4 / rms], atol=1e-15)
        (z,) = norms.rms_norm_rows([[3.0, 4.0]], [0.0, 1.0], 0.0, offset=1.0)   # weight stored as weight - 1
        self.assertClose(z, y, atol=1e-15)
        normed, sums = norms.rms_norm_rows([[1.0, 2.0]], None, 0.0, residual=[[2.0, 2.0]])
        self.assertEqual(sums, [[3.0, 4.0]])
        self.assertClose(normed[0], [3 / rms, 4 / rms], atol=1e-15)

    def test_rope_rotates_pairs_and_passes_the_rest_through(self):
        norms = self.assemble()["norms"]
        row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        (half,) = norms.rope_rows([row], [1], [1.0, 0.5], 4)
        c1, s1, c2, s2 = math.cos(1.0), math.sin(1.0), math.cos(0.5), math.sin(0.5)
        self.assertClose(half, [1 * c1 - 3 * s1, 2 * c2 - 4 * s2, 3 * c1 + 1 * s1, 4 * c2 + 2 * s2, 5.0, 6.0],
                         atol=1e-7)
        (pairs,) = norms.rope_rows([row], [1], [1.0, 0.5], 4, style="interleaved")
        self.assertClose(pairs, [1 * c1 - 2 * s1, 2 * c1 + 1 * s1, 3 * c2 - 4 * s2, 4 * c2 + 3 * s2, 5.0, 6.0],
                         atol=1e-7)
        (at_zero,) = norms.rope_rows([row], [0], [1.0, 0.5])
        self.assertEqual(at_zero, row)
        with self.assertRaises(ValueError):
            norms.rope_rows([row], [1], [1.0, 0.5, 0.25, 0.125])   # 8 rotated values in a row of 6

    def test_products_and_l2_norms(self):
        norms = self.assemble()["norms"]
        (silu,) = norms.act_mul_rows([[2.0]], [[3.0]], "silu")
        self.assertAlmostEqual(silu[0], 2.0 / (1 + math.exp(-2.0)) * 3.0)
        (gelu,) = norms.act_mul_rows([[1.0]], [[1.0]], "gelu")
        self.assertAlmostEqual(gelu[0], 0.5 * (1 + math.erf(1 / math.sqrt(2))))
        (l2,) = norms.l2_norm_rows([[3.0, 4.0]], 0.0)
        self.assertClose(l2, [0.6, 0.8], atol=1e-15)
        (gated,) = norms.gated_rms_norm_rows([[3.0, 4.0]], [[0.0, 100.0]], None, 0.0, activation="sigmoid")
        rms = math.sqrt(12.5)
        self.assertClose(gated, [0.5 * 3 / rms, 4 / rms], atol=1e-12)

    def test_residual_streams_mix_and_take_a_row(self):
        norms = self.assemble()["norms"]
        streams = [[[1.0, 2.0], [3.0, 4.0]]]                      # one token, two streams of two values
        self.assertEqual(norms.stream_sum_rows(streams, scale=0.5), [[2.0, 3.0]])
        self.assertEqual(norms.stream_sum_rows(streams, [[2.0, -1.0]]), [[-1.0, 0.0]])        # a weight per stream
        self.assertEqual(norms.stream_sum_rows(streams, [[[1.0, 0.0], [0.5, 2.0]]]), [[2.5, 8.0]])
        (gated,) = norms.stream_sum_rows(streams, [[[0.0, 0.0], [0.0, 0.0]]], "sigmoid")
        self.assertEqual(gated, [2.0, 3.0])
        self.assertEqual(norms.stream_add_rows(streams, [[10.0, 20.0]], [[1.0, 0.5]]), [[[11.0, 22.0], [8.0, 14.0]]])
        self.assertEqual(norms.stream_add_rows(streams, [[1.0, 1.0]]), [[[2.0, 3.0], [4.0, 5.0]]])
        # one weight per stream: the rows of two streams of a token, each normalized and weighted by its own
        rows = norms.rms_norm_rows([[3.0, 4.0], [3.0, 4.0]], [[1.0, 1.0], [2.0, 2.0]], 0.0)
        self.assertClose(rows[1], [2 * v for v in rows[0]], atol=1e-15)

    def test_the_norms_cases_run(self):
        out = CASES["norms"](self.assemble())
        self.assertEqual(len(out["rms"]), 4)
        self.assertEqual(out["rms_zero"], [[0.0] * 12])
        self.assertEqual(out["rope_half"][0], CASES["norms"](self.assemble())["rope_half"][0])
