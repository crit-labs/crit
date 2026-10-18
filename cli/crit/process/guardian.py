"""Launch gate and parent-death monitor, inside the supervisor's process tree."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crit.process.ownership import process_identity, read_ownership
from crit.run.records import read_json, write_json


def main():
    launch = read_json(Path(sys.argv[1]))
    owner_path = Path(launch["owner_path"])
    owner = read_ownership(owner_path)
    identity = process_identity(os.getpid())
    if (identity in {None, "unknown"} or owner["token"] != launch["token"] or
            process_identity(owner["owner_pid"]) != owner["owner_identity"]):
        return 125
    # A Windows virtual-environment launcher is a different process from the
    # interpreter running this file. Report the actual guardian before the
    # supervisor verifies containment and grants permission to start the engine.
    write_json(Path(sys.argv[1]).with_suffix(".ready.json"),
               {"token": launch["token"], "pid": os.getpid(), "identity": identity}, immutable=True)
    end = time.monotonic() + 10
    while True:
        owner = read_ownership(owner_path)
        if owner["token"] != launch["token"] or process_identity(owner["owner_pid"]) != owner["owner_identity"]:
            return 125
        child = owner.get("process")
        if child and child["pid"] == os.getpid() and child["identity"] == process_identity(os.getpid()):
            break
        if time.monotonic() >= end:
            return 125
        time.sleep(.02)
    # Inherit the guardian's POSIX process group / Windows Job Object.
    try:
        engine = subprocess.Popen(launch["command"], cwd=launch["cwd"], stdin=subprocess.DEVNULL)
    except OSError as exc:
        write_json(Path(sys.argv[1]).with_suffix(".error.json"), {"error": str(exc)}, immutable=True)
        return 125
    while engine.poll() is None:
        try:
            owner = read_ownership(owner_path)
            alive = owner["token"] == launch["token"] and process_identity(owner["owner_pid"]) == owner["owner_identity"]
        except (OSError, ValueError, KeyError) as exc:
            print("Launch guardian could not read ownership: " + str(exc), file=sys.stderr, flush=True)
            alive = False
        if not alive:
            if os.name != "nt":
                os.killpg(os.getpgrp(), signal.SIGKILL)
            engine.kill()  # Windows job also closes when its owner exits.
            engine.wait(timeout=5)
            return 125
        time.sleep(.05)
    return engine.returncode


if __name__ == "__main__":
    raise SystemExit(main())
