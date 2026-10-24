"""Several sealed model identities share one runtime and one-model memory budgets."""
import http.client
import contextlib
import io
import json
import threading
import unittest
from unittest.mock import patch

from test_chat_residency import ServeFixture
from crit.run.records import write_json
from crit_api.catalog import ModelCatalog
from crit_api.server import Api, RequestError, Server
from crit_api.cli import main, parser, write_launchers
from crit_api.engine import EngineError
from crit.process.ownership import process_identity


class CatalogTests(ServeFixture, unittest.TestCase):
    stream = True

    def setUp(self):
        super().setUp()
        primary = (self.run, self.run_id, self.directory)
        second_model = self.root / "other" / "model.fixture"
        second_model.parent.mkdir()
        second_model.write_text("a second synthetic model")
        view = self.store.create(self.data | {"model_path": str(second_model), "kv_tokens": 4096})
        self.run, self.run_id = view["run"], view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        candidate, *_ = self.measured()
        self.store.seal(self.run_id, candidate["candidate_id"], "completed")
        self.store.select(self.run_id, candidate["candidate_id"])
        self.other_run = self.run_id
        self.run, self.run_id, self.directory = primary
        self.catalog = self.root / "models.json"
        write_json(self.catalog, {"default": "first", "models": [
            {"name": "first", "workspace": "runs", "run": self.run_id, "candidate": self.a},
            {"name": "second", "workspace": "runs", "run": self.other_run}]})

    def catalog_engine(self):
        # Each model fits, their sum does not.
        engine = ModelCatalog(self.catalog, host_limit=30, device_limit=30)
        self.addCleanup(engine.close)
        engine.open()
        return engine

    def test_http_model_routing_unloads_before_loading_and_applies_each_context(self):
        engine = self.catalog_engine()
        api = Api(engine)
        self.assertEqual([(m["id"], m["context_length"], m["limits"]["context_length"], m["limits"]["max_output_tokens"])
                          for m in api.models()], [("first", 8192, 8192, 8191), ("second", 4096, 4096, 4095)])
        with self.assertRaises(RequestError) as missing:
            api.completion_request({"model": "typo", "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(missing.exception.status, 404)
        with self.assertRaises(RequestError):
            api.completion_request({"model": "second", "messages": [{"role": "user", "content": "hello"}],
                                    "max_tokens": 4096})
        server = Server(("127.0.0.1", 0), api)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        results = []
        for model, streaming in (("first", False), ("second", True), ("first", False)):
            client = http.client.HTTPConnection(*server.server_address, timeout=20)
            self.addCleanup(client.close)
            client.request("POST", "/v1/chat/completions", json.dumps(
                {"model": model, "messages": [{"role": "user", "content": "hello"}], "stream": streaming}))
            response = client.getresponse()
            data = response.read()
            self.assertEqual(response.status, 200, data)
            if streaming:
                frames = [json.loads(line[6:]) for line in data.splitlines()
                          if line.startswith(b"data: ") and line != b"data: [DONE]"]
                self.assertTrue(all(f["model"] == model for f in frames))
                text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames)
            else:
                body = json.loads(data)
                self.assertEqual(body["model"], model)
                text = body["choices"][0]["message"]["content"]
            results.append(json.loads(text))
            self.assertLessEqual(engine.status()["host_bytes"], 30)
            self.assertLessEqual(engine.status()["device_bytes"], 30)
        self.assertEqual(len({r["pid"] for r in results}), 1)
        self.assertEqual(results[0]["weights"], results[2]["weights"])
        self.assertNotEqual(results[0]["weights"], results[1]["weights"])
        self.assertEqual(len({r["device"] for r in results}), 3)
        self.assertEqual(len({r["generation"] for r in results}), 3)

    def test_ps_and_engine_facts_follow_the_loaded_name(self):
        engine = self.catalog_engine()
        status = engine.runtime.status
        facts = {"speculative": False}
        api = Api(engine)
        server = Server(("127.0.0.1", 0), api)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)

        def get(path):
            client = http.client.HTTPConnection(*server.server_address, timeout=20)
            self.addCleanup(client.close)
            client.request("GET", path)
            return json.loads(client.getresponse().read())

        with patch.object(engine.runtime, "status", lambda: dict(status(), engine_facts=facts)):
            for name, other in (("first", "second"), ("second", "first")):
                if name == "second":
                    self.ask(engine, candidate="second")
                self.assertEqual({m["id"]: m["engine_facts"] for m in api.models()}, {name: facts, other: None})
                loaded = get("/api/ps")["models"]
                self.assertEqual([(m["name"], m["digest"], m["context_length"]) for m in loaded],
                                 [(name, engine.spec(name)["package_sha256"], engine.spec(name)["limits"]["kv_tokens"])])
        self.assertEqual({m["name"]: m["digest"] for m in get("/api/tags")["models"]},
                         {n: engine.spec(n)["package_sha256"] for n in ("first", "second")})

    def test_aliases_and_an_opt_in_fallback_select_configured_names(self):
        value = json.loads(self.catalog.read_text())
        value["models"][0]["aliases"] = ["gpt-4o", "claude-*", "llama3:8b"]
        value["models"][1]["aliases"] = ["claude-haiku-*", "*-mini", "openai/gpt-4o-mini"]
        write_json(self.catalog, value)
        engine = ModelCatalog(self.catalog)
        # Names and exact aliases first, then the pattern with the most characters besides *.
        for model, name in (("first", "first"), ("second", "second"), ("gpt-4o", "first"), ("llama3:8b", "first"),
                            ("claude-sonnet-4-5-20250929", "first"), ("claude-haiku-4-5-20251001", "second"),
                            ("o3-mini", "second"), ("openai/gpt-4o-mini", "second"), ("claude-x-mini", "first")):
            with self.subTest(model=model):
                self.assertEqual(engine.resolve_model(model), name)
        for model in ("gpt-4", "gpt-4o-2024-08-06", "Claude-sonnet-4", "claude", "llama3:70b"):
            with self.subTest(model=model), self.assertRaises(EngineError) as unknown:
                engine.resolve_model(model)
            self.assertEqual(unknown.exception.status, 404)
        write_json(self.catalog, value | {"fallback": "second"})
        engine = ModelCatalog(self.catalog)
        self.assertEqual([engine.resolve_model(m) for m in ("gpt-4", "gpt-4o", "first")], ["second", "first", "first"])

    def test_catalog_refuses_clashing_or_malformed_aliases_and_fallbacks(self):
        value = json.loads(self.catalog.read_text())
        for rows, catalog, message in (
                ([{"aliases": ["second"]}, {}], {}, "also a model name"),
                ([{"aliases": ["gpt-4o"]}, {"aliases": ["claude-*", "gpt-4o"]}], {}, "Duplicate model alias"),
                ([{"aliases": "gpt-4o"}, {}], {}, "aliases must be"),
                ([{"aliases": ["claude 3"]}, {}], {}, "aliases must be"),
                ([{"aliases": ["gpt-[45]"]}, {}], {}, "aliases must be"),
                ([{"display_name": " "}, {}], {}, "display_name"),
                ([{"aliases": ["gpt-4o"]}, {}], {"fallback": "gpt-4o"}, "fallback"),
                ([{}, {}], {"fallback": None}, "fallback")):
            with self.subTest(rows=rows, catalog=catalog):
                write_json(self.catalog, value | catalog | {"models": [m | r for m, r in zip(value["models"], rows)]})
                with self.assertRaisesRegex(ValueError, message):
                    ModelCatalog(self.catalog)

    def test_cards_carry_display_name_and_max_model_len_and_aliases_reach_their_model(self):
        value = json.loads(self.catalog.read_text())
        value["models"][0] |= {"aliases": ["gpt-4o"], "display_name": "First model"}
        value["models"][1]["aliases"] = ["claude-haiku-*"]
        write_json(self.catalog, value)
        engine = self.catalog_engine()
        server = Server(("127.0.0.1", 0), Api(engine))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)

        def call(method, path, body=None):
            client = http.client.HTTPConnection(*server.server_address, timeout=20)
            self.addCleanup(client.close)
            client.request(method, path, None if body is None else json.dumps(body))
            response = client.getresponse()
            return response.status, json.loads(response.read())

        cards = call("GET", "/v1/models")[1]["data"]
        self.assertEqual([(m["id"], m["display_name"], m["label"], m["context_length"], m["max_model_len"])
                          for m in cards], [("first", "First model", "Fixture engine", 8192, 8192),
                                            ("second", "Fixture engine", "Fixture engine", 4096, 4096)])
        # Clients list the configured names only; an alias still finds its model's details.
        self.assertEqual([m["name"] for m in call("GET", "/api/tags")[1]["models"]], ["first", "second"])
        self.assertEqual(call("GET", "/v1/models/gpt-4o")[0], 404)
        shown = call("POST", "/api/show", {"model": "claude-haiku-4-5"})[1]
        self.assertEqual(shown["model_info"]["crit.context_length"], 4096)
        hello = [{"role": "user", "content": "hello"}]
        status, reply = call("POST", "/v1/messages", {"model": "claude-haiku-4-5-20251001", "max_tokens": 64,
                                                      "messages": hello})
        self.assertEqual((status, reply["model"]), (200, "claude-haiku-4-5-20251001"))
        self.assertEqual(json.loads(reply["content"][0]["text"])["weights"], "a second synthetic model")
        status, reply = call("POST", "/v1/chat/completions", {"model": "gpt-4o", "messages": hello})
        self.assertEqual((status, reply["model"]), (200, "first"))
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"])["weights"],
                         "synthetic input, not a model")
        status, reply = call("POST", "/api/chat", {"model": "gpt-4", "stream": False, "messages": hello})
        self.assertEqual(status, 404, reply)

    def test_duplicate_names_default_and_launchers(self):
        args = parser().parse_args(["--models", str(self.catalog), "--write-launchers"])
        windows, posix = write_launchers(args, ["--models", str(self.catalog), "--write-launchers"])
        self.assertEqual(windows.parent, self.root)
        self.assertIn("--models", posix.read_text())
        self.assertNotIn("--workspace", posix.read_text())
        value = json.loads(self.catalog.read_text())
        value["models"].append(value["models"][0])
        write_json(self.catalog, value)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ModelCatalog(self.catalog)
        value["models"].pop()
        value["default"] = "missing"
        write_json(self.catalog, value)
        with self.assertRaisesRegex(ValueError, "default"):
            ModelCatalog(self.catalog)

    def test_failed_old_engine_cleanup_retires_before_retrying_the_new_model(self):
        engine = self.catalog_engine()
        self.ask(engine, "badclose", candidate="first")
        pid = engine.runtime.tree.process.pid
        with self.assertRaisesRegex(EngineError, "detach failure"):
            self.ask(engine, candidate="second")
        self.assertIsNone(process_identity(pid))
        self.assertIsNone(engine.runtime.tree)
        next_reply = self.ask(engine, candidate="second")
        self.assertNotEqual(next_reply["pid"], pid)
        self.assertEqual(next_reply["weights"], "a second synthetic model")

    def test_serve_command_opens_catalog_routes_requests_and_releases_runtime(self):
        outcomes = []

        class ExercisedServer(Server):
            def serve_forever(server):
                def client():
                    try:
                        for model in ("first", "second"):
                            connection = http.client.HTTPConnection(*server.server_address, timeout=20)
                            try:
                                connection.request("POST", "/v1/chat/completions", json.dumps({
                                    "model": model, "messages": [{"role": "user", "content": "hello"}]}))
                                response = connection.getresponse()
                                outcomes.append((response.status, json.loads(response.read())))
                            finally:
                                connection.close()
                    except BaseException as exc:
                        outcomes.append(exc)
                    finally:
                        server.shutdown()
                worker = threading.Thread(target=client)
                worker.start()
                try:
                    super().serve_forever()
                finally:
                    worker.join(30)

        # Signal delivery and the machine lock have separate platform tests.
        # This exercises CLI selection, real HTTP and the guarded resident host.
        with patch("crit_api.cli.Server", ExercisedServer), patch("crit_api.cli.signal.signal"), \
                patch("crit_core.machine.engine_lock", return_value=contextlib.nullcontext()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--models", str(self.catalog), "--port", "0"]), 0)
        self.assertEqual(len(outcomes), 2, outcomes)
        self.assertTrue(all(isinstance(value, tuple) for value in outcomes), outcomes)
        self.assertEqual([status for status, _ in outcomes], [200, 200])
        self.assertEqual([value["model"] for _, value in outcomes], ["first", "second"])
        self.assertFalse((self.root / "runtime-owner" / "owner.json").exists())
