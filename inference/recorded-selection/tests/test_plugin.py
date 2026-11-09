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


class RecordedSelectionTests(GraphsChecks, PluginTestCase):
    def test_capture_pays_when_it_is_used_often_enough(self):
        offers = [offer("eager", "eager", 900), offer("full", "full", 100, setup_ns=50000, resident_bytes=1000)]
        self.assertEqual(self.graphs().choose(SHAPE, offers, 1 << 20, uses=10), "eager")
        self.assertEqual(self.graphs().choose(SHAPE, offers, 1 << 20, uses=100), "full")
        self.assertEqual(self.graphs().choose(SHAPE, offers, 500, uses=100), "eager")

    def test_a_captured_graph_costs_no_setup_and_modes_are_honoured(self):
        offers = [offer("eager", "eager", 900), offer("full", "full", 100, setup_ns=50000, ready=True)]
        self.assertEqual(self.graphs().choose(SHAPE, offers, 0), "full")
        self.assertEqual(self.graphs(modes="eager").choose(SHAPE, offers, 0), "eager")

    def test_a_time_chooses_only_among_offers_of_the_same_bits(self):
        # Prompt graphs' offers: the unpadded eager path and padded buckets compute other bits, so the options fix the
        # arithmetic (allow_padding: the smallest padded bucket's) and a faster offer of other bits is never taken.
        shape = dict(SHAPE, phase="prefill", tokens=10, rows=1)
        eager = offer("eager", "eager", 50, shape, ready=True, bits="eager")
        small = offer("graph:16", "full", 100, dict(shape, tokens=16), ready=True, bits="padded:16")
        large = offer("graph:64", "full", 10, dict(shape, tokens=64), ready=True, bits="padded:64")
        offers = [eager, small, large]
        self.assertEqual(self.graphs(allow_padding=True).choose(shape, offers, 0), "graph:16")
        self.assertEqual(self.graphs().choose(shape, offers, 0), "eager")
        # Among offers of the same bits the measured times still choose: a bucket of exactly the chunk's tokens
        # replays the eager path's calls.
        exact = offer("graph:10", "full", 20, shape, ready=True, bits="eager")
        self.assertEqual(self.graphs().choose(shape, [eager, exact, large], 0), "graph:10")

    def test_padding_takes_the_bucket_of_exactly_the_chunk_and_the_modes_come_first(self):
        # A bucket of exactly the chunk's tokens is the smallest padding (none): allow_padding takes it, not a larger
        # one. A graph the plugin's modes exclude is no choice: with modes eager the eager path serves, padding or not.
        shape = dict(SHAPE, phase="prefill", tokens=64, rows=1)
        eager = offer("eager", "eager", 900, shape, ready=True, bits="eager")
        own = offer("graph:64", "full", 300, dict(shape, tokens=64), ready=True, bits="padded:64")
        larger = offer("graph:128", "full", 100, dict(shape, tokens=128), ready=True, bits="padded:128")
        offers = [eager, own, larger]
        self.assertEqual(self.graphs(allow_padding=True).choose(shape, offers, 0), "graph:64")
        self.assertEqual(self.graphs(allow_padding=True, modes="eager").choose(shape, offers, 0), "eager")
        self.assertEqual(self.graphs().choose(shape, offers, 0), "eager")

    def test_padding_only_where_allowed(self):
        padded = offer("full8", "full", 100, dict(SHAPE, rows=8), ready=True)
        offers = [offer("eager", "eager", 900), padded]
        self.assertEqual(self.graphs().choose(SHAPE, offers, 0), "eager")
        self.assertEqual(self.graphs(allow_padding=True).choose(SHAPE, offers, 0), "full8")
