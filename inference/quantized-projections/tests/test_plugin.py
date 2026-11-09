import random

from crit_plugins.testing import PluginTestCase


def matrix(rows, cols, seed=1):
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


class QuantizedProjectionsTests(PluginTestCase):
    # each format's options and how far its decoded weights may lie from the dense ones
    FORMATS = [({}, 0.1), ({"format": "int4"}, 0.1), ({"format": "int8", "group": 16}, 0.1),
               ({"format": "nvfp4"}, 0.2), ({"format": "nvfp4", "four_over_six": True}, 0.2)]

    def test_it_has_the_projections_interface(self):
        for options, _ in self.FORMATS:
            self.assertImplements(self.assemble(options=options))

    def test_it_projects_with_its_decoded_weights_as_the_dense_projection_does(self):
        dense = self.reference()["projections"]
        x = matrix(3, 64, seed=2)
        for options, _ in self.FORMATS:
            p = self.assemble(options=options)["projections"]
            w = p.prepare(matrix(5, 64))
            got, want = p.project(w, x), dense.project(dense.prepare(p.dense(w)), x)
            self.assertClose(got, want, atol=1e-9, rtol=1e-9)

    def test_its_decoded_weights_stay_near_the_dense_ones(self):
        for options, bound in self.FORMATS:
            p = self.assemble(options=options)["projections"]
            original = matrix(4, 64, seed=3)
            decoded = p.dense(p.prepare(original))
            error = max(abs(a - b) for ra, rb in zip(decoded, original) for a, b in zip(ra, rb))
            self.assertLess(error, bound, options)

    def test_the_formats_differ(self):
        w = matrix(2, 64, seed=4)
        decoded = {options.get("format", "int8"): self.assemble(options=options)["projections"]
                   for options, _ in self.FORMATS[:2] + self.FORMATS[3:4]}
        rows = {name: p.dense(p.prepare(w)) for name, p in decoded.items()}
        self.assertNotEqual(rows["int8"], rows["int4"])
        self.assertNotEqual(rows["int8"], rows["nvfp4"])
