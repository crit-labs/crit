"""crit serve: load a sealed candidate once and answer OpenAI-compatible requests until stopped."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import ssl
import subprocess
import sys

from crit.process.ownership import RuntimeOwnership
from crit.run.records import GIB
from crit.run.store import ROOT, RunStore

from .engine import EngineError, ServedEngine, single_run
from .catalog import ModelCatalog
from .server import Api, Server


def parser():
    result = argparse.ArgumentParser(prog="crit serve", description=__doc__)
    result.add_argument("--workspace", type=Path, help="Folder containing the run-* project directories")
    result.add_argument("--models", type=Path, help="JSON catalog of named models from sealed runs (docs/SERVING.md)")
    result.add_argument("--engine", metavar="OWNER/NAME", help="Serve an engine crit pull installed, such as "
                        "crit-labs/nvfp4-gemv-int8kv")
    result.add_argument("--store", type=Path, help="Installed-engine store --engine reads (default: PROJECT/engines, "
                                                   "where crit pull installs)")
    result.add_argument("--run", help="Run ID (default: the workspace's only run)")
    result.add_argument("--candidate", help="Candidate ID (default: the run's selected candidate)")
    result.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: 127.0.0.1, this machine only)")
    result.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    result.add_argument("--api-key", action="append",
                        help="Require this bearer key; repeat for several (default: the comma-separated keys in the "
                             "CRIT_API_KEY environment variable, else none)")
    result.add_argument("--api-key-file", type=Path,
                        help="Also accept the keys in this file, one per line; blank lines and lines starting with # "
                             "are skipped")
    result.add_argument("--ssl-certfile", type=Path, help="Serve HTTPS with this PEM certificate chain")
    result.add_argument("--ssl-keyfile", type=Path,
                        help="The certificate's PEM private key (default: the key in --ssl-certfile)")
    result.add_argument("--cors-origin", help="Allow browser requests from this origin, for example http://localhost:3000 or *")
    result.add_argument("--model-name", help="Model ID reported to clients (default: RUN_ID/CANDIDATE_ID)")
    result.add_argument("--reasoning-alias", action="store_true",
                        help='Return reasoning under "reasoning" as well as "reasoning_content", the name vLLM and '
                             "OpenRouter clients read (off by default: a client that joins every field of a delta, "
                             "as crit Eval does, would read it twice)")
    result.add_argument("--lenient", action="store_true",
                        help="Ignore request parameters crit does not apply (unknown ones, sampling fields the package "
                             "does not declare, extra samplers) instead of refusing them with 400; responses list them "
                             "in X-Crit-Ignored-Parameters and the server logs each once. Values are still checked")
    result.add_argument("--timeout", type=float, default=600, help="Seconds one request may run (default: 600)")
    result.add_argument("--max-queue", type=int, default=64,
                        help="Requests that may wait while the engine is busy; one more is refused at once with 429 "
                             "(Anthropic 529, Ollama 503) and Retry-After (default: 64)")
    result.add_argument("--queue-timeout", type=float,
                        help="Seconds a request may wait for the engine before it is refused the same way "
                             "(default: no limit; --timeout starts when the engine takes the request)")
    result.add_argument("--sse-ping-interval", type=float, default=15, metavar="SECONDS",
                        help="Send a keep-alive ping on an OpenAI or Anthropic stream after this many quiet seconds, "
                             "so proxies and clients do not close it while a model loads or a prompt prefills; 0 "
                             "sends none (default: 15)")
    result.add_argument("--load-timeout", type=float,
                        help="Seconds loading the candidate at startup may take (default: 600 plus the model's size at "
                             "200 MB/s, or at 50 MB/s with --verify-model)")
    result.add_argument("--verify-model", action="store_true",
                        help="Read and SHA-256 every model file against the run's frozen manifest before loading "
                             "(default: check file sizes only and load right away)")
    result.add_argument("--idle-unload", type=float, metavar="SECONDS",
                        help="Unload the engine's weights after this many seconds without a request; the next request "
                             "loads it again (default: stay loaded)")
    result.add_argument("--resident-ram-gib", type=float, help="Optional cap on retained model bytes in host memory")
    result.add_argument("--resident-vram-gib", type=float, help="Optional cap on retained model bytes in device memory")
    result.add_argument("--project", type=Path, default=ROOT if (ROOT / "core/include/crit/core.h").is_file() else Path.cwd(),
                        help="crit repository (default: this checkout)")
    result.add_argument("--write-launchers", action="store_true",
                        help="Write serve.cmd and serve.sh into the run folder, which start this exact command, then exit")
    result.add_argument("--recover-runtime", action="store_true",
                        help="Recover a dead runtime owner after verifying its processes settled, then exit")
    return result


def write_launchers(args, argv):
    """Double-clickable scripts in the run folder that start the server with the same options."""
    if args.models:
        folder = args.models.resolve().parent
    else:
        store = RunStore(args.workspace, args.project)
        folder = store.directory(args.run or single_run(store))
    options, skip = [], False
    for argument in argv:
        # A key in a script file would be readable by anyone who can read the run folder; use CRIT_API_KEY or
        # --api-key-file.
        if skip or argument == "--write-launchers" or argument.startswith("--api-key="):
            skip = False
            continue
        if argument == "--api-key":
            skip = True
            continue
        options.append(argument)
    if not args.models and not args.run:
        options += ["--run", folder.name]
    if not args.models and "--workspace" not in options:
        options += ["--workspace", str(args.workspace)]
    command = [sys.executable, "-m", "crit.cli"]
    paths = {"--models", "--workspace", "--api-key-file", "--ssl-certfile", "--ssl-keyfile"}
    def absolute_option(previous, argument):
        name, equals, value = argument.partition("=")
        if equals and name in paths:
            return name + "=" + str(Path(value).absolute())
        return str(Path(argument).absolute()) if previous in paths else argument
    command += ["serve", *map(absolute_option, [None, *options], options)]
    windows = folder / "serve.cmd"
    posix = folder / "serve.sh"
    # Under WSL the command holds Linux paths, so the Windows launcher starts serve.sh in the same distribution.
    distro = os.environ.get("WSL_DISTRO_NAME") if os.name != "nt" else None
    start = ["wsl.exe", "-d", distro, "-e", str(posix)] if distro else command
    windows.write_text("@echo off\r\n" + subprocess.list2cmdline(start) + "\r\npause\r\n", encoding="utf-8")
    posix.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    posix.chmod(0o755)
    return windows, posix


def api_keys(args):
    """The keys a request may carry: each --api-key, else the comma-separated keys in CRIT_API_KEY, and the lines of
    --api-key-file other than blank lines and # comments."""
    keys = args.api_key or os.environ.get("CRIT_API_KEY", "").split(",")
    if args.api_key_file:
        lines = [line.strip() for line in args.api_key_file.read_text(encoding="utf-8-sig").splitlines()]
        found = [line for line in lines if line and not line.startswith("#")]
        if not found:
            raise ValueError(f"{args.api_key_file} holds no API key")
        keys = [*keys, *found]
    return [key.strip() for key in keys if key.strip()]


def tls_context(args):
    """The server's TLS context for --ssl-certfile and --ssl-keyfile, or None to serve plain HTTP."""
    if args.ssl_certfile is None:
        return None
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    try:
        context.load_cert_chain(args.ssl_certfile, args.ssl_keyfile)
    except OSError as exc:
        files = " and ".join(str(path) for path in (args.ssl_certfile, args.ssl_keyfile) if path)
        raise ValueError(f"cannot load the TLS certificate from {files}: {exc}") from exc
    return context


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    args = parser().parse_args(argv)
    if args.recover_runtime:
        print(json.dumps(RuntimeOwnership().recover()))
        return 0
    if args.engine:
        if any((args.models, args.workspace, args.run, args.candidate)):
            parser().error("--engine replaces --models, --workspace, --run and --candidate")
        if args.write_launchers:
            parser().error("--write-launchers writes into a run folder; start an installed engine with crit serve --engine")
    elif args.store:
        parser().error("--store goes with --engine")
    elif args.models:
        if any((args.workspace, args.run, args.candidate, args.model_name)):
            parser().error("--models replaces --workspace, --run, --candidate and --model-name")
    elif args.workspace is None:
        parser().error("--workspace, --models or --engine is required")
    if not 0 <= args.sse_ping_interval <= 86400:
        parser().error("--sse-ping-interval takes 0 (no pings) to 86400 seconds")
    if args.ssl_keyfile and not args.ssl_certfile:
        parser().error("--ssl-keyfile needs --ssl-certfile")
    try:
        keys, tls = api_keys(args), tls_context(args)
    except (OSError, ValueError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not keys:
        print("warning: listening on %s without an API key; anyone who can reach this port can use the engine" % args.host,
              file=sys.stderr)
    if args.write_launchers:
        for path in write_launchers(args, argv if argv is not None else sys.argv[1:]):
            print(path)
        return 0
    from crit_core.machine import engine_lock
    with engine_lock(f"crit serve {args.engine or args.run or ''} {args.candidate or ''}".strip()):
        return _serve(args, keys, tls)


def _serve(args, keys, tls):
    gib = lambda value: None if value is None else round(value * GIB)
    try:
        options = dict(project=args.project, host_limit=gib(args.resident_ram_gib),
                       device_limit=gib(args.resident_vram_gib), timeout=args.timeout,
                       load_timeout=args.load_timeout, verify_model=args.verify_model, max_queue=args.max_queue,
                       queue_timeout=args.queue_timeout, idle_unload=args.idle_unload)
        if args.engine:
            from .installed import InstalledEngine
            engine = InstalledEngine(args.store or args.project / "engines", args.engine, **options)
        else:
            engine = (ModelCatalog(args.models, **options) if args.models else
                      ServedEngine(args.workspace, args.run, args.candidate, **options))
        print("Loading %s ..." % (args.engine or args.candidate or "the selected candidate"), file=sys.stderr, flush=True)
        engine.open()
    except (ValueError, EngineError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    try:
        api = Api(engine, args.model_name, keys, args.cors_origin,
                  log=lambda line: print(line, file=sys.stderr, flush=True), reasoning_alias=args.reasoning_alias,
                  lenient=args.lenient, ping_interval=args.sse_ping_interval)
        server = Server((args.host, args.port), api, tls)
        host, port = server.server_address[:2]
        # Ctrl+C, and SIGTERM from a supervisor, unload the engine and release ownership. A shell's background job
        # starts with SIGINT ignored, so restore Python's handler for it too.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _interrupt)
        print("Serving %s at %s://%s:%d/v1 (Ctrl+C to stop)" % (api.model, "https" if tls else "http", host, port),
              file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
