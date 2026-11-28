"""crit.spec.library_spec: the EngineSpec of a checkpoint before any package exists, the library-default engine's
(the preflight's candidate) resolved from the headers alone, and the readers that size an engine by it (crit run
create's KV capacity through crit Eval's max context, recon's placement, crit plugins offload)."""
import json
from pathlib import Path
import tempfile
import unittest

import support.runs  # noqa: F401 - the product folders on sys.path
from support.tiny_checkpoints import checkpoint
from crit import spec as library


class LibrarySpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows refuses to delete a checkpoint file the process still maps (the spec's checkpoint reads)
        cls.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dense = checkpoint("dense", Path(cls.temp.name) / "dense")
        if cls.dense is None:
            raise unittest.SkipTest("transformers lacks the dense architecture")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_a_checkpoint_has_the_spec_of_the_engine_a_build_starts_from(self):
        from crit.preflight.candidate import library_selection
        found = library.library_spec(self.dense)
        selection, _ = library_selection(self.dense)
        self.assertEqual(found["selection"], {slot: plugin for slot, plugin in selection["use"].items() if plugin})
        kv = found["values"]["kv_elements_per_token"]
        self.assertEqual((kv["source"], kv["by"]), ("resolved", selection["use"]["model"]))
        from crit_models.config import architecture
        from crit_core.model import open_model
        config = architecture(open_model(self.dense).config())
        self.assertEqual(kv["value"], config.kv_elements_per_token())   # every layer's keys and values a token
        self.assertEqual(found["values"]["quality_mode"]["value"], "speed")
        self.assertIs(library.library_spec(self.dense), found)   # resolved once a process
        # no sliding window: no ring (kv_ring_bytes 0)
        self.assertEqual(library.prices(found), {"kv_elements_per_token": kv["value"], "kv_ring_bytes": 0})

    def test_a_checkpoint_no_model_plugin_serves_has_none(self):
        folder = Path(self.temp.name) / "unknown"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"model_type": "no-such-model", "hidden_size": 8}))
        self.assertIsNone(library.library_spec(folder))

    def test_eval_sizes_a_run_by_the_library_spec(self):
        from crit.run.records import eval_max_context
        device = {"device": "fixture", "memory_free_bytes": 8 << 30, "memory_total_bytes": 8 << 30}
        found = eval_max_context(self.dense, device, 1 << 28, requested=4096)
        self.assertEqual(found["basis"]["sized_by"], ["kv_elements_per_token", "kv_ring_bytes"])
        self.assertTrue(found["tokens"])

    def test_a_windowed_checkpoint_prices_its_rings_and_eval_counts_them_as_fixed_state(self):
        # The engine keeps a ring of each sliding window a sequence it runs at once (whole pages), where the config
        # counts one window: the spec prices the rings (kv_ring_bytes, the models core's planned pools) and crit
        # Eval's fixed state takes them in place of the config's window.
        windows = checkpoint("windows", Path(self.temp.name) / "windows")
        if windows is None:
            self.skipTest("transformers lacks the windowed architecture")
        found = library.library_spec(windows)
        ring = found["values"]["kv_ring_bytes"]
        self.assertEqual((ring["source"], ring["by"]), ("resolved", found["selection"]["model"]))
        self.assertGreater(ring["value"], 0)
        from crit.run.records import eval_max_context
        from crit_models.config import architecture
        from crit_core.model import open_model
        config = architecture(open_model(windows).config())
        device = {"device": "fixture", "memory_free_bytes": 8 << 30, "memory_total_bytes": 8 << 30}
        sized = eval_max_context(windows, device, 1 << 28, requested=4096)
        self.assertIn("kv_ring_bytes", sized["basis"]["sized_by"])
        element = 1   # the protocol's KV precision (fp8) a value
        sized_ring = library.library_spec(windows, quality_mode=None)["values"]["kv_ring_bytes"]["value"]
        self.assertEqual(sized["basis"]["fixed_state_bytes"],   # the spec it was sized by: no quality mode named
                         config.fixed_state_bytes(element, sliding=False) + sized_ring)


if __name__ == "__main__":
    unittest.main()
