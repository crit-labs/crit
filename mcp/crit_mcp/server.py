"""crit mcp: crit's commands as Model Context Protocol tools, served over stdio to an agent's harness.

Every crit run, eval, pull, serve, kernels, models and inference subcommand is a tool, its input schema read from
the command's own parser. Each call runs the command as a job: the call returns its result when the command ends
within wait_seconds, else the job, which crit_job_wait, crit_job_output and crit_job_stop follow. crit's guides
(AGENTS.md and the docs it links) are resources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from crit import __version__
from crit.cli import PROJECT_VARIABLE
from crit.inference import ROOT as LIBRARY_ROOT
from crit.run.store import ROOT

from .jobs import Jobs
from .tools import FAMILIES, defaults_for, tools

# Newest first; a client asking for another version gets the newest.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# The guides an agent reads before driving a run, as crit://PATH resources.
GUIDES = ("AGENTS.md", "docs/BUILDING_ENGINES.md", "docs/RUN_RECORDS.md", "docs/SERVING.md",
          "docs/CONTRIBUTION_CRITERIA.md", "cli/README.md", "inference/README.md", "kernels/README.md",
          "models/README.md", "mcp/README.md")
# Characters of a command's output and errors a result carries; the rest stays in the job's logs.
OUTPUT_CHARACTERS = 60000
ERROR_CHARACTERS = 8000
# Output up to this size is parsed as the command's JSON result.
PARSED_BYTES = 8 << 20
MAX_WAIT = 3600
PROGRESS_SECONDS = 10
INSTRUCTIONS = """crit builds bespoke inference engines: these tools are its command line. Read crit://AGENTS.md \
first; it is the workflow (recon, plan, candidate, build, eval, check, seal, select, stop) and its rules. \
crit_run_* keep a run's records, crit_eval_* measure, crit_pull installs a crit-hub engine, crit_serve and \
crit_run_serve serve one, crit_serve_status asks a running server how it is. Every command runs as a job: a call \
returns the command's result (its JSON output parsed) when it ends within wait_seconds, else the job, and the \
command keeps running. Follow a job with crit_job_wait (bounded waits; call it again until the job has ended), \
crit_job_output for its full logs and crit_job_stop to interrupt it as Ctrl+C would. While Eval or a build holds \
the GPU, do the next round's CPU work and wait on the job afterwards. A record option takes the JSON object itself \
or a file's path. Paths are on the machine this server runs on; give them absolute."""


class Failure(Exception):
    """A JSON-RPC error."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Server:
    def __init__(self, project=None, root=None, cwd=None, jobs=None, wait=30, families=FAMILIES):
        # Without --project the commands find their own, as on the command line: this checkout, else the folder
        # they run in.
        checkout = ROOT if (ROOT / "core/include/crit/core.h").is_file() else None
        self.project = Path(project).resolve() if project else checkout
        self.settings = {"project": self.project, "root": Path(root).resolve() if root else None}
        self.cwd = Path(cwd or os.getcwd()).resolve()
        folder = Path(jobs) if jobs else Path(tempfile.mkdtemp(prefix="crit-mcp-"))
        # A --project names itself to every job, so a crit eval job reads its settings.toml as crit run's do.
        self.jobs = Jobs(folder, {PROJECT_VARIABLE: str(self.project)} if project else None)
        self.scratch = folder / "records"
        self.wait = wait
        self.tools, self.missing = tools(families)
        self.extra = {"crit_job_wait": self.job_wait, "crit_job_output": self.job_output,
                      "crit_job_stop": self.job_stop, "crit_jobs": self.job_list,
                      "crit_serve_status": self.serve_status}
        self.cancelled = {}
        self.send = None

    # --- MCP methods -------------------------------------------------------------------------------------------

    def handle(self, method, params, request_id=None):
        if method == "initialize":
            asked = params.get("protocolVersion")
            instructions = INSTRUCTIONS + "".join(
                f" crit {family} is unavailable here ({reason})." for family, reason in self.missing.items())
            return {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                    "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
                    "serverInfo": {"name": "crit", "title": "crit", "version": __version__},
                    "instructions": instructions}
        if method in {"ping", "logging/setLevel"}:
            return {}
        if method == "tools/list":
            return {"tools": [tool.definition(self.wait) for tool in self.tools.values()] + EXTRA_TOOLS}
        if method == "tools/call":
            return self.call(params.get("name"), params.get("arguments") or {}, params.get("_meta") or {}, request_id)
        if method == "resources/list":
            return {"resources": [{"uri": "crit://" + guide, "name": guide, "title": guide,
                                   "mimeType": "text/markdown"} for guide in GUIDES if self.guide(guide)]}
        if method == "resources/read":
            uri = params.get("uri", "")
            guide = uri[len("crit://"):] if uri.startswith("crit://") else None
            path = self.guide(guide) if guide in GUIDES else None
            if path is None:
                raise Failure(-32002, f"Resource not found: {uri}")
            return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": path.read_text(encoding="utf-8")}]}
        if method == "resources/templates/list":
            return {"resourceTemplates": []}
        if method == "prompts/list":
            return {"prompts": []}
        raise Failure(-32601, f"Method not found: {method}")

    def guide(self, name):
        """A guide's file in the crit checkout the commands use, else in this one, else (inference/README.md) in the
        plugin library this crit carries."""
        for folder in (self.project, ROOT, LIBRARY_ROOT):
            if folder and (folder / name).is_file():
                return folder / name
        return None

    def call(self, name, arguments, meta, request_id):
        if not isinstance(arguments, dict):
            raise Failure(-32602, "arguments must be an object")
        try:
            if name in self.extra:
                return self.extra[name](arguments, meta, request_id)
            tool = self.tools.get(name)
            if tool is None:
                raise Failure(-32602, f"Unknown tool: {name}")
            arguments = dict(arguments)
            wait = arguments.pop("wait_seconds", None)
            words = tool.argv(arguments, defaults_for(tool, self.settings), self.scratch)
            job = self.jobs.start(name, ["-m", "crit.cli", *words], self.cwd)
            return self.outcome(job, self.waited(job, wait, meta, request_id))
        except (ValueError, OSError) as error:
            return reply({"error": str(error)}, error=True)

    # --- jobs --------------------------------------------------------------------------------------------------

    def waited(self, job, seconds, meta, request_id):
        """Wait up to SECONDS for JOB, reporting progress to a client that asked for it; whether it ended."""
        seconds = self.wait if seconds is None else seconds
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
            raise ValueError("wait_seconds must be a number of seconds, 0 or more")
        end = time.monotonic() + min(seconds, MAX_WAIT)
        token = meta.get("progressToken")
        cancelled = self.cancelled.get(request_id)
        while not job.done.is_set():
            left = end - time.monotonic()
            if left <= 0 or (cancelled and cancelled.is_set()):
                break
            if job.done.wait(min(left, PROGRESS_SECONDS)) or token is None or self.send is None:
                continue
            message = job.last_line()
            self.send({"jsonrpc": "2.0", "method": "notifications/progress", "params": {
                "progressToken": token, "progress": round(time.time() - job.started, 1),
                **({"message": message} if message else {})}})
        return job.done.is_set()

    def outcome(self, job, ended):
        result = job.summary()
        # As this machine's shell would take it.
        result["command"] = (subprocess.list2cmdline if os.name == "nt" else shlex.join)(["crit", *job.command[2:]])
        if not ended:
            text, _ = job.tail("errors", ERROR_CHARACTERS)
            result["errors_tail"] = text
            result["next"] = f"crit_job_wait {{\"job\": \"{job.id}\"}} until it ends; crit_job_stop to interrupt it"
            return reply(result)
        output, _ = job.read("output", 0, PARSED_BYTES)
        try:
            value = json.loads(output) if output.strip() else None
        except json.JSONDecodeError:
            value = output
        if not isinstance(value, str) and len(json.dumps(value)) <= OUTPUT_CHARACTERS:
            result["result"] = value
        else:
            result["output"] = output[:OUTPUT_CHARACTERS]
            if len(output) > OUTPUT_CHARACTERS:
                result["output_truncated"] = (f"{job.size('output')} bytes; the rest is in logs.output "
                                              "(crit_job_output)")
        errors, cut = job.tail("errors", ERROR_CHARACTERS)
        if errors:
            result["errors"] = errors
            if cut:
                result["errors_truncated"] = "the beginning is in logs.errors (crit_job_output)"
        return reply(result, error=job.exit_code != 0)

    def job_wait(self, arguments, meta, request_id):
        job = self.jobs.get(arguments.get("job"))
        return self.outcome(job, self.waited(job, arguments.get("seconds", self.wait), meta, request_id))

    def job_output(self, arguments, meta, request_id):
        job = self.jobs.get(arguments.get("job"))
        stream = arguments.get("stream", "errors")
        if stream not in {"output", "errors"}:
            raise ValueError("stream is output or errors")
        offset, limit = arguments.get("offset", 0), arguments.get("limit", OUTPUT_CHARACTERS)
        if offset is None or offset < 0:
            offset = max(0, job.size(stream) - limit)  # the end of the log
        text, after = job.read(stream, offset, limit)
        return reply({"job": job.id, "state": job.state, "stream": stream, "offset": offset, "next_offset": after,
                      "size": job.size(stream), "text": text})

    def job_stop(self, arguments, meta, request_id):
        job = self.jobs.get(arguments.get("job"))
        job.stop(arguments.get("grace_seconds", 30))
        return self.outcome(job, job.done.is_set())

    def job_list(self, arguments, meta, request_id):
        return reply({"jobs": [job.summary() for job in self.jobs.jobs.values()]})

    def serve_status(self, arguments, meta, request_id):
        url = arguments.get("url", "http://127.0.0.1:8000").rstrip("/")
        headers = {"Authorization": "Bearer " + arguments["api_key"]} if arguments.get("api_key") else {}
        result = {"url": url}
        for name, path in (("health", "/health"), ("models", "/v1/models")):
            request = urllib.request.Request(url + path, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    result[name] = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                result[name] = {"status": error.code, "body": error.read().decode("utf-8", errors="replace")[:2000]}
            except (urllib.error.URLError, OSError, ValueError) as error:
                result[name] = {"error": str(getattr(error, "reason", error))}
        return reply(result, error="error" in result["health"])

    def close(self):
        self.jobs.close(grace=5)


def reply(value, error=False):
    return {"content": [{"type": "text", "text": json.dumps(value, indent=2, default=str)}], "isError": error}


JOB = {"type": "string", "description": "The job a command's call returned, such as job-3"}
EXTRA_TOOLS = [
    {"name": "crit_job_wait", "title": "Wait on a job", "description":
     "Wait for a job to end, at most SECONDS; returns its result once it ended, else its state and newest errors",
     "inputSchema": {"type": "object", "properties": {"job": JOB, "seconds": {
         "type": "number", "minimum": 0, "description": f"Longest wait (at most {MAX_WAIT}); call again to go on"}},
         "required": ["job"], "additionalProperties": False}},
    {"name": "crit_job_output", "title": "Read a job's logs", "description":
     "A job's command output (its JSON result) or errors (progress and reports) from a byte offset; offset -1 "
     "reads the end", "inputSchema": {"type": "object", "properties": {
         "job": JOB, "stream": {"type": "string", "enum": ["errors", "output"], "default": "errors"},
         "offset": {"type": "integer", "default": 0}, "limit": {"type": "integer", "minimum": 1,
                                                               "default": OUTPUT_CHARACTERS}},
         "required": ["job"], "additionalProperties": False}},
    {"name": "crit_job_stop", "title": "Stop a job", "description":
     "Interrupt a running job as Ctrl+C would (crit releases its leases and settles what it can), ending its "
     "process tree after grace_seconds; an Eval stopped so goes on with crit_run_eval resume",
     "inputSchema": {"type": "object", "properties": {"job": JOB, "grace_seconds": {
         "type": "number", "minimum": 0, "default": 30}}, "required": ["job"], "additionalProperties": False}},
    {"name": "crit_jobs", "title": "List jobs", "description": "This server's jobs: state, exit code, logs",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "crit_serve_status", "title": "Ask crit serve", "description":
     "A running crit serve's /health (loaded engine, readiness, capacity) and /v1/models",
     "inputSchema": {"type": "object", "properties": {
         "url": {"type": "string", "default": "http://127.0.0.1:8000", "description": "The server, without /v1"},
         "api_key": {"type": "string", "description": "One of the server's --api-key values, if it has any"}},
         "additionalProperties": False}},
]


def serve(server, reader, writer):
    """Answer newline-delimited JSON-RPC messages from READER on WRITER until READER ends."""
    lock = threading.Lock()

    def send(message):
        data = json.dumps(message, separators=(",", ":"), default=str).encode("utf-8") + b"\n"
        with lock:
            writer.write(data)
            writer.flush()

    server.send = send

    def answer(message):
        request_id = message.get("id")
        try:
            params = message.get("params") or {}
            if not isinstance(params, dict):
                raise Failure(-32602, "params must be an object")
            response = {"jsonrpc": "2.0", "id": request_id, "result": server.handle(message["method"], params, request_id)}
        except Failure as failure:
            response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": failure.code, "message": str(failure)}}
        except Exception as error:  # a bug answers its request instead of ending the server
            response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": f"{type(error).__name__}: {error}"}}
        cancelled = server.cancelled.pop(request_id, None)
        if not (cancelled and cancelled.is_set()):
            send(response)

    with ThreadPoolExecutor(32, thread_name_prefix="crit-mcp") as pool:
        try:
            read(server, reader, send, pool, answer)
        finally:
            for event in list(server.cancelled.values()):
                event.set()  # the client is gone: stop waiting on its calls


def read(server, reader, send, pool, answer):
    """Hand each request to the pool and act on notifications, until the client closes its end."""
    for line in reader:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {error}"}})
            continue
        for item in message if isinstance(message, list) else [message]:
            if not isinstance(item, dict) or ("method" not in item and "id" not in item):
                send({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid request"}})
            elif "method" not in item:
                continue  # a response to a request this server never sends
            elif "id" not in item:
                if item["method"] == "notifications/cancelled":
                    event = server.cancelled.get((item.get("params") or {}).get("requestId"))
                    if event:
                        event.set()
            else:
                server.cancelled[item["id"]] = threading.Event()
                pool.submit(answer, item)


def parser():
    """The arguments of crit mcp (crit mcp offers its tools from them)."""
    parser = argparse.ArgumentParser(prog="crit mcp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, help="crit repository the commands use, whose settings.toml gives the "
                                                     "device and headroom (default: this checkout; a crit installed "
                                                     "without one takes any folder, default the folder the commands "
                                                     "run in)")
    parser.add_argument("--root", type=Path, help="Run root crit run and crit serve use when a call names none "
                                                  "(default: PROJECT/runs)")
    parser.add_argument("--cwd", type=Path, help="Folder the commands run in (default: the current one)")
    parser.add_argument("--jobs", type=Path, metavar="DIR", help="Folder of the jobs' logs and record files "
                                                                 "(default: a new temporary folder)")
    parser.add_argument("--wait", type=float, default=30, metavar="SECONDS",
                        help="How long a call waits for its command by default (default: %(default)s)")
    parser.add_argument("--only", action="append", choices=FAMILIES, metavar="FAMILY",
                        help=f"Offer only these command families' tools (repeatable; any of {', '.join(FAMILIES)})")
    return parser


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    # The protocol owns stdout: anything else printed goes to stderr.
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    if os.name != "nt":
        # A client that ends the server with SIGTERM still gets its jobs stopped. A server started with SIGINT
        # ignored would pass that on to its jobs, and crit_job_stop's interrupt would never reach them.
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
            signal.signal(signal.SIGINT, signal.default_int_handler)
    server = Server(args.project, args.root, args.cwd, args.jobs, args.wait, tuple(args.only or FAMILIES))
    try:
        serve(server, reader, writer)
    finally:
        if os.name != "nt":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second SIGTERM does not cut the jobs' stop short
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
