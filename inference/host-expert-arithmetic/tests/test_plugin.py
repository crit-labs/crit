import random

from crit_plugins.testing import PluginTestCase


def matrix(rows, cols, seed=1):
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


class HostExpertArithmeticTests(PluginTestCase):
    def test_it_has_the_projections_interface(self):
        self.assertImplements()

    def test_every_declared_order_is_within_fp32_rounding_of_the_fp64_projection(self):
        reference = self.reference(options={"quantized-projections": {"format": "nvfp4"}})["projections"]
        w, x = matrix(4, 64), matrix(2, 64, seed=2)
        want = reference.project(reference.prepare(w), x)
        for options in ({}, {"reduce": "sequential"}, {"product": "block_dot", "lanes": 4}):
            p = self.assemble(options=options)["projections"]
            self.assertClose(p.project(p.prepare(w), x), want, atol=1e-5, rtol=1e-5)

    def test_the_order_is_deterministic(self):
        p = self.assemble()["projections"]
        w = p.prepare(matrix(2, 32))
        x = matrix(1, 32, seed=5)
        self.assertEqual(p.project(w, x), p.project(w, x))
