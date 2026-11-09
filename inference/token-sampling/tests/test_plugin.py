import math

from crit_plugins.testing import PluginTestCase

ROW = [2.0, 1.0, 2.0, -1.0, 0.5, 1.5]


class TokenSamplingTests(PluginTestCase):
    def test_it_has_the_sampler_interface(self):
        self.assertImplements()

    def test_greedy_takes_the_best_token_and_the_lower_id_on_a_tie(self):
        sampler = self.assemble()["sampler"]
        self.assertEqual(sampler.select([ROW, [0.0, 3.0, 1.0]], {"temperature": 0.0}), [0, 1])
        self.assertEqual(sampler.law(ROW, {}), [(0, 1.0)])

    def test_a_draw_is_one_of_the_law_and_repeats_at_its_position(self):
        sampler = self.assemble()["sampler"]
        rule = {"temperature": 0.7, "top_p": 0.9, "seed": 3}
        law = sampler.law(ROW, rule)
        self.assertAlmostEqual(math.fsum(p for _, p in law), 1.0)
        draws = [sampler.select([ROW], dict(rule, position=i))[0] for i in range(40)]
        self.assertTrue(set(draws) <= {t for t, _ in law})
        self.assertGreater(len(set(draws)), 1)
        self.assertEqual(draws, [sampler.select([ROW], dict(rule, position=i))[0] for i in range(40)])

    def test_an_allowed_set_keeps_the_draw_inside_it(self):
        sampler = self.assemble()["sampler"]
        rule = {"temperature": 1.0, "allowed": [1, 3, 4]}
        self.assertEqual({t for t, _ in sampler.law(ROW, rule)}, {1, 3, 4})
        self.assertTrue(all(sampler.select([ROW], dict(rule, position=i))[0] in (1, 3, 4) for i in range(30)))

    def test_greedy_predictions_are_the_columns_argmax(self):
        sampler = self.assemble()["sampler"]
        rows = [ROW, [0.0, 3.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 4.0]]
        self.assertEqual(sampler.predictions(rows, [0, 2], {"temperature": 0.0}), [0, 1, 5])
