"""crit serve: load a sealed candidate once and answer OpenAI-compatible requests until stopped."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
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
    result.add_argument("--run", help="Run ID (default: the workspace's only run)")
    result.add_argument("--candidate", help="Candidate ID (default: the run's selected candidate)")
    result.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: 127.0.0.1, this machine only)")
    result.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    result.add_argument("--api-key", default=os.environ.get("CRIT_API_KEY"),
                        help="Require this bearer key (default: the CRIT_API_KEY environment variable, else none)")
    result.add_argument("--cors-origin", help="Allow browser requests from this origin, for example http://localhost:3000 or *")
    result.add_argument("--model-name", help="Model ID reported to clients (default: RUN_ID/CANDIDATE_ID)")
    result.add_argument("--timeout", type=float, default=600, help="Seconds one request may run (default: 600)")
    result.add_argument("--load-timeout", type=float,
                        help="Seconds loading the candidate at startup may take (default: 600 plus the model's size at "
                             "200 MB/s, or at 50 MB/s with --verify-model)")
    result.add_argument("--verify-model", action="store_true",
                        help="Read and SHA-256 every model file against the run's frozen manifest before loading "
                             "(default: check file sizes only and load right away)")
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
        # A key in a script file would be readable by anyone who can read the run folder; use CRIT_API_KEY.
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
    def absolute_option(argument):
        for prefix in ("--models=", "--workspace="):
            if argument.startswith(prefix):
                return prefix + str(Path(argument[len(prefix):]).absolute())
        return str(Path(argument).absolute()) if argument in {str(args.workspace), str(args.models)} else argument
    command += ["serve", *map(absolute_option, options)]
    windows = folder / "serve.cmd"
    posix = folder / "serve.sh"
    # Under WSL the command holds Linux paths, so the Windows launcher starts serve.sh in the same distribution.
    distro = os.environ.get("WSL_DISTRO_NAME") if os.name != "nt" else None
    start = ["wsl.exe", "-d", distro, "-e", str(posix)] if distro else command
    windows.write_text("@echo off\r\n" + subprocess.list2cmdline(start) + "\r\npause\r\n", encoding="utf-8")
    posix.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    posix.chmod(0o755)
    return windows, posix


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    args = parser().parse_args(argv)
    if args.recover_runtime:
        print(json.dumps(RuntimeOwnership().recover()))
        return 0
    if args.models:
        if any((args.workspace, args.run, args.candidate, args.model_name)):
            parser().error("--models replaces --workspace, --run, --candidate and --model-name")
    elif args.workspace is None:
        parser().error("--workspace or --models is required")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.api_key:
        print("warning: listening on %s without an API key; anyone who can reach this port can use the engine" % args.host,
              file=sys.stderr)
    if args.write_launchers:
        for path in write_launchers(args, argv if argv is not None else sys.argv[1:]):
            print(path)
        return 0
    from crit_core.machine import engine_lock
    with engine_lock(f"crit serve {args.run or ''} {args.candidate or ''}".strip()):
        return _serve(args)


def _serve(args):
    gib = lambda value: None if value is None else round(value * GIB)
    try:
        options = dict(project=args.project, host_limit=gib(args.resident_ram_gib),
                       device_limit=gib(args.resident_vram_gib), timeout=args.timeout,
                       load_timeout=args.load_timeout, verify_model=args.verify_model)
        engine = (ModelCatalog(args.models, **options) if args.models else
                  ServedEngine(args.workspace, args.run, args.candidate, **options))
        print("Loading %s ..." % (args.candidate or "the selected candidate"), file=sys.stderr, flush=True)
        engine.open()
    except (ValueError, EngineError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    try:
        api = Api(engine, args.model_name, args.api_key, args.cors_origin,
                  log=lambda line: print(line, file=sys.stderr, flush=True))
        server = Server((args.host, args.port), api)
        host, port = server.server_address[:2]
        # Ctrl+C, and SIGTERM from a supervisor, unload the engine and release ownership. A shell's background job
        # starts with SIGINT ignored, so restore Python's handler for it too.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _interrupt)
        print("Serving %s at http://%s:%d/v1 (Ctrl+C to stop)" % (api.model, host, port), file=sys.stderr, flush=True)
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
