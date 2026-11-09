"""crit mcp: tools read from crit's parsers, command lines rebuilt from tool arguments, the stdio protocol end to
end over a real run, and jobs that end with their whole process tree."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("core/python", "api", "../crit-eval", "kernels", "models", "cli", "mcp")
for _folder in FOLDERS: sys.path.insert(0, str(ROOT / _folder))
from crit.process.ownership import process_identity
from crit_mcp.jobs import Jobs
from crit_mcp.server import Server
from crit_mcp.tools import family_parser, tools


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools, cls.missing = tools(("run", "serve", "kernels"))

    def test_every_run_action_is_a_tool_with_the_parsers_options(self):
        self.assertEqual(self.missing, {})
        parser = family_parser("run")
        actions = next(a for a in parser._actions if hasattr(a, "choices") and isinstance(a.choices, dict)).choices
        self.assertEqual({name for name in self.tools if name.startswith("crit_run_")},
                         {"crit_run_" + action for action in actions})
        plan = self.tools["crit_run_plan"].definition(30)
        self.assertEqual(plan["title"], "crit run plan")
        schema = plan["inputSchema"]
        self.assertEqual(schema["properties"]["check"]["type"], "boolean")
        self.assertEqual(schema["properties"]["record"]["type"], ["object", "array", "string"])
        self.assertEqual(sorted(schema["required"]), ["id", "record"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(self.tools["crit_run_check"].definition(30)["inputSchema"]["properties"]["from_eval"]["type"],
                         ["boolean", "string"])
        self.assertEqual(self.tools["crit_run_reserve"].definition(30)["inputSchema"]["properties"]["operation"]["enum"],
                         ["discovery", "build", "check", "benchmark"])
        # Only create names no existing run.
        self.assertEqual(self.tools["crit_run_create"].definition(30)["inputSchema"]["required"], ["record"])
        self.assertIn("crit_serve", self.tools)
        self.assertEqual(self.tools["crit_kernels_new"].definition(30)["inputSchema"]["required"], ["name", "op"])

    def test_arguments_become_the_command_line(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            check = self.tools["crit_run_check"]
            self.assertEqual(check.argv({"id": "run-1", "candidate": "candidate-0001", "from_eval": True}, {}, scratch),
                             ["run", "--id=run-1", "check", "--candidate=candidate-0001", "--from-eval"])
            self.assertEqual(check.argv({"id": "run-1", "candidate": "c", "from_eval": "attempt-0003"}, {}, scratch)[-1],
                             "--from-eval=attempt-0003")
            build = self.tools["crit_run_build"].argv({"id": "r", "candidate": "c", "record": {"build": {"command": ["make"]}},
                                                       "drop": ["obj/*", "-tmp"], "no_package": True}, {}, scratch)
            record = next(word for word in build if word.startswith("--record="))[len("--record="):]
            self.assertEqual(json.loads(Path(record).read_text(encoding="utf-8")), {"build": {"command": ["make"]}})
            self.assertEqual([w for w in build if w.startswith("--drop")], ["--drop=obj/*", "--drop=-tmp"])
            self.assertIn("--no-package", build)
            # An object for a JSON text option is its JSON; the server's defaults fill what the call leaves out.
            resume = self.tools["crit_run_resume"].argv({"id": "r", "quality_budget": {"ppl_ratio_max": 1.1}},
                                                        {"root": "R"}, scratch)
            self.assertEqual(resume[:3], ["run", "--root=R", "--id=r"])
            self.assertIn('--quality-budget={"ppl_ratio_max": 1.1}', resume)
            self.assertEqual(self.tools["crit_kernels_new"].argv({"name": "gemv", "op": "gemv", "shape": ["8,16"]}, {},
                                                                 scratch), ["kernels", "new", "gemv", "--op=gemv",
                                                                            "--shape=8,16"])
            with self.assertRaisesRegex(ValueError, "has no option nope"):
                check.argv({"id": "r", "candidate": "c", "nope": 1}, {}, scratch)
            with self.assertRaisesRegex(ValueError, "needs id"):
                self.tools["crit_run_show"].argv({}, {}, scratch)
            with self.assertRaisesRegex(ValueError, "switch"):
                self.tools["crit_run_plan"].argv({"id": "r", "record": "plan.json", "check": "yes"}, {}, scratch)

    def test_rebuilt_command_lines_parse_as_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            words = self.tools["crit_run_eval"].argv({"id": "r", "candidate": "c", "data": "D", "only": ["speed", "quality"],
                                                      "label": "-dash", "budget_minutes": 30, "resume": True},
                                                     {}, Path(directory))
        args = family_parser("run").parse_args(words[1:])
        self.assertEqual((args.id, args.action, args.candidate, args.only, args.label, args.budget_minutes, args.resume),
                         ("r", "eval", "c", ["speed", "quality"], "-dash", 30.0, True))


class Client:
    """The server as an MCP client sees it: one JSON-RPC message per line."""

    def __init__(self, *arguments, cwd):
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(str(ROOT / f) for f in FOLDERS)}
        self.process = subprocess.Popen([sys.executable, "-B", "-m", "crit_mcp", *arguments], cwd=cwd, env=environment,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.number = 0

    def send(self, message):
        self.process.stdin.write(json.dumps(message).encode() + b"\n")
        self.process.stdin.flush()

    def request(self, method, params=None):
        self.number += 1
        self.send({"jsonrpc": "2.0", "id": self.number, "method": method, **({"params": params} if params else {})})
        return self.answer(self.number)

    def answer(self, number):
        while True:
            message = json.loads(self.process.stdout.readline())
            if message.get("id") == number:
                return message

    def call(self, name, arguments):
        result = self.request("tools/call", {"name": name, "arguments": arguments})["result"]
        return json.loads(result["content"][0]["text"]), result["isError"]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=60)
        self.process.stdout.close()


class ProtocolTests(unittest.TestCase):
    def test_a_run_over_stdio(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            model = folder / "model.fixture"
            model.write_text("synthetic input, not a model", encoding="utf-8")
            client = Client("--root", str(folder / "runs"), "--jobs", str(folder / "jobs"), "--only", "run", cwd=folder)
            try:
                started = client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                        "clientInfo": {"name": "test", "version": "1"}})["result"]
                self.assertEqual(started["protocolVersion"], "2025-06-18")
                self.assertEqual(started["serverInfo"]["name"], "crit")
                self.assertIn("tools", started["capabilities"])
                client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
                self.assertEqual(client.request("initialize", {"protocolVersion": "1999-01-01"})["result"]["protocolVersion"],
                                 "2025-11-25")
                names = [tool["name"] for tool in client.request("tools/list")["result"]["tools"]]
                self.assertIn("crit_run_plan", names)
                self.assertIn("crit_job_wait", names)
                self.assertFalse(any(name.startswith("crit_eval_") for name in names))  # --only run

                created, failed = client.call("crit_run_create", {"record": {
                    "model_path": str(model), "correctness_criteria": "Fixed fixture gates", "kv_tokens": 8192,
                    "candidate_budget": 1}})
                self.assertFalse(failed, created)
                self.assertEqual((created["state"], created["exit_code"]), ("exited", 0))
                run_id = created["result"]["run"]["run_id"]
                self.assertTrue((folder / "runs" / run_id).is_dir())

                running, _ = client.call("crit_run_show", {"id": run_id, "wait_seconds": 0})
                self.assertIn(running["state"], {"running", "exited"})
                shown, failed = client.call("crit_job_wait", {"job": running["job"], "seconds": 120})
                self.assertFalse(failed)
                self.assertEqual(shown["result"]["run"]["run_id"], run_id)
                output, _ = client.call("crit_job_output", {"job": running["job"], "stream": "output", "offset": 0})
                self.assertEqual(json.loads(output["text"])["run"]["run_id"], run_id)

                refused, failed = client.call("crit_run_seal", {"id": run_id, "candidate": "candidate-0009",
                                                                "lifecycle": "completed"})
                self.assertTrue(failed)
                self.assertEqual(refused["exit_code"], 1)
                self.assertIn("ERROR", refused["errors"])
                invalid, failed = client.call("crit_run_show", {"id": run_id, "verbose": True})
                self.assertTrue(failed)
                self.assertIn("has no option verbose", invalid["error"])
                listed, _ = client.call("crit_jobs", {})
                self.assertEqual([job["tool"] for job in listed["jobs"]], ["crit_run_create", "crit_run_show", "crit_run_seal"])

                guide = client.request("resources/read", {"uri": "crit://AGENTS.md"})["result"]["contents"][0]
                self.assertEqual(guide["text"], (ROOT / "AGENTS.md").read_text(encoding="utf-8"))
                self.assertEqual(client.request("resources/read", {"uri": "crit://../secret"})["error"]["code"], -32002)
                self.assertEqual(client.request("tools/call", {"name": "crit_nothing"})["error"]["code"], -32602)
                self.assertEqual(client.request("nothing/here")["error"]["code"], -32601)
                client.process.stdin.write(b"{not json\n")
                client.process.stdin.flush()
                self.assertEqual(client.answer(None)["error"]["code"], -32700)
                self.assertEqual(client.request("ping")["result"], {})
            finally:
                client.close()
            self.assertEqual(client.process.returncode, 0)


class JobTests(unittest.TestCase):
    # A command that starts a child of its own, says both process ids and waits.
    TREE = ("import subprocess, sys, time; child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            "print(child.pid, flush=True); print('working', file=sys.stderr, flush=True); time.sleep(120)")

    def started(self, jobs):
        job = jobs.start("test", ["-c", self.TREE], Path.cwd())
        deadline = time.monotonic() + 30
        while not job.read("output")[0].strip() and time.monotonic() < deadline:
            time.sleep(.05)
        return job, int(job.read("output")[0])

    def test_a_servers_project_reaches_its_jobs(self):
        # crit eval has no --project: a job learns the server's from CRIT_PROJECT, as crit run's commands name theirs.
        from crit.cli import PROJECT_VARIABLE
        with tempfile.TemporaryDirectory() as directory:
            server = Server(project=directory, jobs=Path(directory) / "jobs")
            self.addCleanup(server.jobs.close, 5)
            job = server.jobs.start("test", ["-c", f"import os; print(os.environ.get({PROJECT_VARIABLE!r}))"],
                                    Path.cwd())
            self.assertTrue(job.done.wait(30))
            self.assertEqual(job.read("output")[0].strip(), str(Path(directory).resolve()))
            server.jobs.close(grace=5)

    def test_a_stopped_job_ends_with_its_whole_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory)
            job, child = self.started(jobs)
            self.assertEqual(job.state, "running")
            self.assertIsNotNone(process_identity(child))
            self.assertEqual(job.last_line(), "working")
            job.stop(grace=10)
            self.assertEqual(job.state, "stopped")
            self.assertIsNotNone(job.exit_code)
            deadline = time.monotonic() + 10
            while process_identity(child) is not None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertIsNone(process_identity(child))

    def test_closing_the_server_stops_its_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            server = Server(jobs=Path(directory) / "jobs", families=("run",))
            job, child = self.started(server.jobs)
            ended = server.jobs.start("quick", ["-c", "print('{\"done\": true}')"], Path.cwd())
            self.assertTrue(ended.done.wait(30))
            server.close()
            self.assertEqual((job.state, ended.state), ("stopped", "exited"))
            deadline = time.monotonic() + 10
            while process_identity(child) is not None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertIsNone(process_identity(child))
            result = json.loads(server.outcome(ended, True)["content"][0]["text"])
            self.assertEqual((result["result"], result["exit_code"]), ({"done": True}, 0))


class ServeStatusTests(unittest.TestCase):
    def test_reads_health_and_models(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"status": "ok", "ready": True} if self.path == "/health" else
                                  {"object": "list", "data": [{"id": "run/candidate-0001"}]}).encode()
                self.send_response(200 if self.headers.get("Authorization") == "Bearer key" or self.path == "/health" else 401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            server = Server(jobs=directory, families=())
            url = f"http://127.0.0.1:{http.server_address[1]}"
            answer = server.serve_status({"url": url, "api_key": "key"}, {}, None)
            status = json.loads(answer["content"][0]["text"])
            self.assertFalse(answer["isError"])
            self.assertEqual((status["health"]["ready"], status["models"]["data"][0]["id"]), (True, "run/candidate-0001"))
            status = json.loads(server.serve_status({"url": url}, {}, None)["content"][0]["text"])
            self.assertEqual(status["models"]["status"], 401)


if __name__ == "__main__":
    unittest.main()
