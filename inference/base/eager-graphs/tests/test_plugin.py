from crit_plugins.testing import PluginTestCase

SHAPE = {"phase": "decode", "tokens": 1, "rows": 4, "identity": ("bf16", "cuda", 1)}


def offer(name, mode, run_ns, shape=None, **costs):
    return {"name": name, "mode": mode, "shape": dict(shape or SHAPE), "run_ns": run_ns, **costs}


class GraphsChecks:
    """What every graphs plugin must do."""

    def graphs(self, **options):
        return self.assemble(options=options)["graphs"]

    def test_it_has_the_graphs_interface(self):
        self.assertImplements()

    def test_an_eager_offer_alone_is_chosen_and_none_is_refused(self):
        self.assertEqual(self.graphs().choose(SHAPE, [offer("eager", "eager", 900)], 0), "eager")
        with self.assertRaises(ValueError):
            self.graphs().choose(SHAPE, [offer("other", "eager", 900, dict(SHAPE, phase="prefill"))], 1 << 30)


class EagerGraphsTests(GraphsChecks, PluginTestCase):
    def test_a_faster_captured_graph_is_not_taken(self):
        offers = [offer("full", "full", 100, ready=True), offer("eager", "eager", 900)]
        self.assertEqual(self.graphs().choose(SHAPE, offers, 1 << 30, uses=100), "eager")
