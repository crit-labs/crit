"""crit run ... record-routes end to end on the toy engine of test_run_ablate: its attention slot holds a plugin that
declares record_routes and trace_path and writes the trace when it closes, as expert-offload does. Every prompt of
the corpus is answered once with the recording asked of that plugin, and the trace is the attempt's evidence."""
import json
import tempfile
from pathlib import Path
import unittest

import test_run_ablate as ablate_tests
from crit.run.routes import corpus_requests, record_routes

ROUTES_PLUGIN = '''
import json
from pathlib import Path


class Plugin:
    def __init__(self, options):
        self.seconds_per_token = 0.0
        self.tokens_per_round = 1
        self.options = options

    def close(self):
        if self.options.get("record_routes") and self.options.get("trace_path"):
            Path(self.options["trace_path"]).write_text(json.dumps([[0, [1, 2]]] * self.options["record_routes"]))


def create(context, options, parts):
    return Plugin(options)
'''
OPTIONS = {"record_routes": {"type": "int", "default": 0, "min": 0}, "trace_path": {"type": "str", "default": ""}}


class RecordRoutesTests(ablate_tests.AblateTests):
    def write_plugins(self, package):
        super().write_plugins(package)
        folder = package / "plugins" / "attn-routes"
        (folder / "tests").mkdir(parents=True, exist_ok=True)
        (folder / "plugin.json").write_text(json.dumps({
            "schema": "crit-plugin-v1", "id": "attn-routes", "title": "attn-routes", "fills": ["attention"],
            "entry": "plugin.py:create", "options": OPTIONS, "needs": [], "devices": ["cpu"]}), encoding="utf-8")
        (folder / "plugin.py").write_text(ROUTES_PLUGIN, encoding="utf-8")
        (folder / "tests" / "test_plugin.py").write_text(ablate_tests.TEST, encoding="utf-8")

    def test_the_corpus_is_answered_with_the_recording_on_and_the_trace_kept(self):
        selection = {**ablate_tests.SELECTION, "use": {**ablate_tests.SELECTION["use"], "attention": "attn-routes"}}
        candidate = self.plugin_package(selection=selection)
        with tempfile.TemporaryDirectory() as temp:
            corpus = Path(temp) / "corpus.json"
            corpus.write_text(json.dumps([{"lang": "de", "text": "Erzähl von Flüssen."}, "Explain a hash map.",
                                          {"messages": [{"role": "user", "content": "Write a haiku."}]}]),
                              encoding="utf-8")
            result = record_routes(self.store, self.run_id, candidate, corpus, routes=5, max_tokens=4)
        self.assertEqual((result["plugin"], result["prompts"], result["routes"]), ("attn-routes", 3, 5))
        self.assertEqual(result["tokens"], 12)
        self.assertEqual(result["attempt"]["outcome"], "completed")
        trace = Path(result["trace"])
        self.assertEqual(trace, (self.directory / "work" / "routes" / f"{candidate}.json").resolve())
        self.assertIn(trace.relative_to(self.directory.resolve()).as_posix(),
                      [log["path"] for log in result["attempt"]["logs"]])

    def test_a_selection_without_a_recording_plugin_is_refused(self):
        candidate = self.plugin_package()
        with tempfile.TemporaryDirectory() as temp:
            corpus = Path(temp) / "corpus.json"
            corpus.write_text(json.dumps(["hi"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "records routes"):
                record_routes(self.store, self.run_id, candidate, corpus)

    def test_a_corpus_item_is_a_text_or_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus = Path(temp) / "corpus.json"
            corpus.write_text(json.dumps([{"lang": "en", "text": "a"}, "b"]), encoding="utf-8")
            self.assertEqual(corpus_requests(corpus), [("en", [{"role": "user", "content": "a"}]),
                                                       ("2", [{"role": "user", "content": "b"}])])
            corpus.write_text(json.dumps([{"lang": "en"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "item 0"):
                corpus_requests(corpus)


# The ablation tests themselves run in test_run_ablate, not again here.
for name in [n for n in vars(ablate_tests.AblateTests) if n.startswith("test_")]:
    setattr(RecordRoutesTests, name, None)

if __name__ == "__main__":
    unittest.main()
