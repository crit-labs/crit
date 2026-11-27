"""Process supervisor for the opt-in resident host; commands, streams and results over its channel."""
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
import uuid

from crit.process.ownership import clean_snapshots
from crit.process.tree import guarded_tree
from crit.records import MAX_JSON_BYTES, write_json

from .channel import Supervisor
from .logprobs import RESPONDING, response_bytes
from .options import MAX_CHAT_RECORD
from .stream import StreamReader

RESULT = {"kind", "command_id", "status", "error", "max_concurrent_requests", "cleanup_settled"}


# Seconds a cancelled command has to settle before its host is retired (docs/SERVING.md): fixed, not an option. It is
# Core's cancel grace, the one a prompt step ends inside (crit_plugins.steps.GRACE).
from crit_plugins.steps import GRACE as CANCEL_GRACE_SECONDS  # noqa: E402
# Seconds between the lines crit serve prints of a load's reported work (crit_core.engine.LoadContext.working).
WORK_REPORT_SECONDS = 30


def work_line(work):
    """A load's reported work as one line: what, and how far where it counts."""
    done, total = work.get("done"), work.get("total")
    counted = (f" {done:,} of {total:,}" if total is not None else f" {done:,}") if done is not None else ""
    return f"{work.get('what', 'working')}{counted}"


def engine_key(spec):
    """What makes an engine another one for its host: the candidate, its package, its limits and its configuration
    (the identity the host's attach compares)."""
    from crit.records import digest
    spec = spec or {}
    return digest({"identity": {k: spec.get(k) for k in ("run_id", "candidate_id", "package_sha256")},
                   "limits": spec.get("limits") or {}, "configuration": spec.get("configuration")})


class ResidentRuntime:
    """The resident host process crit serve runs its engine in. one_process: one engine, one process: an engine
    other than the one the host holds (another candidate, other limits, another configuration) loads in a fresh host,
    and an unload ends the host, so everything a closed engine held returns to the system with its process (its CUDA
    context, allocator caches, captured graphs). The default keeps one host across reloads, whose model pool keeps
    the model's mapped files and host-side stores for the next engine."""

    def __init__(self, ownership, host_limit, device_limit, one_process=False):
        self.ownership = ownership
        self.host_limit, self.device_limit = host_limit, device_limit
        self.one_process = bool(one_process)
        self.attached = None   # engine_key() of the engine the host holds (one_process)
        # one_process: how the last engine whose host ended went (its host's own close report, or the process's end:
        # its memory went back with the process), which outlasts the host it was in (ServedEngine.last_close)
        self.last_exit = None
        self.exits = 0
        self.hosts = 0         # the hosts started, each a process
        self.tree = self.log = self.root = self.link = None
        self.last_work = None   # the work the newest load last reported (its status's `work`), kept past a retire
        self.lifecycle = threading.RLock()
        self.on_retire = lambda: None

    def start(self):
        with self.lifecycle:
            self._start()

    def _start(self):
        if self.tree:
            return
        self.root = self.ownership.directory / "runtimes" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.ownership.update(runtime_id=self.root.name)
        self.link = Supervisor(self.ownership.token)
        config = self.root / "host.json"
        write_json(config, {"root": str(self.root), "token": self.ownership.token, "port": self.link.port,
                   "host_limit": self.host_limit, "device_limit": self.device_limit}, immutable=True)
        self.log = (self.root / "runtime.log").open("xb")
        self.hosts += 1
        env = None
        if self.one_process:   # the files kept stores outlast each host in (crit_core.engine.LoadContext.kept_file)
            from crit_core.engine import KEPT_FILES
            env = {**os.environ, KEPT_FILES: str(self.kept_files())}
        self.tree = guarded_tree([sys.executable, "-B", str(Path(__file__).with_name("host.py")), str(config)],
                                 self.root, self.log, self.ownership, self.root, env=env)

    def kept_files(self):
        """The folder of the files kept stores write themselves to under one_process, which outlasts each host and
        goes with the server (forget_kept_files)."""
        return self.ownership.directory / "kept-files"

    def forget_kept_files(self):
        """Delete the kept stores' files: the server stops, and no later host reopens them."""
        import shutil
        shutil.rmtree(self.kept_files(), ignore_errors=True)

    def _ended(self, reported, closed=False):
        """One engine's host ends (one_process): how its engine's close went, kept past the host. closed: the host
        closed the engine itself first (an unload), whose own report (Core's close check) is the close's; else the
        engine ends with the process, which returns everything it held to the system."""
        self.exits += 1
        if closed and isinstance(reported, dict):
            self.last_exit = {**reported, "close": f"host-{self.exits}", "process_exit": True}
        else:
            self.last_exit = {"returned": True, "judged": True, "close": f"host-{self.exits}", "process_exit": True,
                              "message": "the engine's host process ended with it (one engine, one process): its "
                                         "device memory went back with the process"}

    def status(self):
        with self.lifecycle:
            tree, link = self.tree, self.link
        if not tree:
            return {"state": "unloaded", "host_bytes": 0, "device_bytes": 0}
        if tree.process.poll() is not None:
            return {"state": "interrupted", "host_bytes": 0, "device_bytes": 0,
                    "error": "Resident host exited; Unload verifies remaining process cleanup"}
        status = link.status
        return dict(status) if status is not None else {"state": "starting", "host_bytes": 0, "device_bytes": 0}

    def announce(self, why):
        """One stderr line naming why the running host is retired or closed; nothing when no host runs."""
        with self.lifecycle:
            tree = self.tree
        if tree is not None:
            print(f"crit serve: the engine host (process {tree.process.pid}) is {why}", file=sys.stderr, flush=True)

    def host_pid(self):
        """The resident host's process id once it has connected, else None."""
        with self.lifecycle:
            link = self.link
        return link.pid if link else None

    def call(self, action, payload, stop, timeout, progress=None, request=None, progress_tools=None,
             progress_reasoning=None, on_started=None, stall=None):
        """Run one command on the host. A generation's payload carries its request record; its result carries the
        engine's response, and progress, progress_tools and progress_reasoning receive each streamed piece. A piece
        whose consumer raises OSError (its client is gone) cancels the command as a stop does: the result is then
        "cancelled" with that error as "consumer_error", and the host stays loaded once the command settles.
        stall (a load's): past `timeout` the command goes on while the host's reported work (its status's `work`)
        last advanced under that many seconds ago, and crit serve prints where the work stands."""
        if stop.is_set():
            return {"status": "cancelled", "error": "Stopped", "cleanup_settled": True}
        key = engine_key(payload.get("spec")) if action == "load" else None
        if self.one_process and key is not None:
            with self.lifecycle:
                fresh = self.tree is not None and self.attached is not None and self.attached != key
            if fresh:   # another engine: the host's process ends with the engine it holds
                self.announce("closed: one engine, one process: the next engine loads in a fresh host")
                self._ended(self.status().get("last_close"))
                self.close()
        result = self._command(action, payload, stop, timeout, progress, request, progress_tools,
                               progress_reasoning, on_started, stall)
        if self.one_process and result.get("status") == "completed":
            if action == "load":
                self.attached = key
            elif action in ("unload", "close"):   # a closed engine's memory goes back with its host's process
                self.announce(f"closed: one engine, one process: the {action}ed engine's host ends")
                self.attached = None
                self._ended(self.status().get("last_close"), closed=True)
                self.close()
        return result

    def _command(self, action, payload, stop, timeout, progress=None, request=None, progress_tools=None,
                 progress_reasoning=None, on_started=None, stall=None):
        with self.lifecycle:
            if action not in RESPONDING:
                self.start()
            if self.tree is None:
                raise ValueError("Resident host was retired")
            root, tree, link = self.root, self.tree, self.link
        stream = (StreamReader(request, progress or (lambda text: None), progress_tools, progress_reasoning)
                  if progress or progress_tools or progress_reasoning else None)
        limit = response_bytes(request) + MAX_JSON_BYTES if action in RESPONDING else MAX_JSON_BYTES
        return self._call(uuid.uuid4().hex, action, payload, stop, timeout, stream, on_started, root, tree, link,
                          limit, stall)

    def _call(self, cid, action, payload, stop, timeout, stream, on_started=None, root=None, tree=None, link=None,
              limit=MAX_JSON_BYTES, stall=None):
        root, tree, link = root or self.root, tree or self.tree, link or self.link
        if stop.is_set():
            return {"status": "cancelled", "error": "Stopped", "cleanup_settled": True}
        box = link.register(cid, limit)
        try:
            link.send({"kind": "command", "command_id": cid, "action": action, "payload": payload},
                      MAX_CHAT_RECORD + MAX_JSON_BYTES)
            deadline, cancelled_at, timed_out, consumer = time.monotonic() + timeout, None, False, None
            if on_started:
                on_started(deadline)
            shown, shown_at = None, time.monotonic()
            while True:
                stamp = time.monotonic()
                advancing = False
                if stall:
                    work, worked = link.work, link.worked
                    self.last_work = work
                    advancing = worked is not None and stamp - worked < stall
                    if work is not None and work != shown and stamp - shown_at >= WORK_REPORT_SECONDS:
                        shown, shown_at = work, stamp
                        print(f"crit serve: loading: {work_line(work)}", file=sys.stderr, flush=True)
                if stop.is_set() or (stamp >= deadline and not advancing):
                    if cancelled_at is None:
                        timed_out = not stop.is_set()
                        cancelled_at = stamp
                        try:
                            link.send({"kind": "cancel", "command_id": cid})
                        except ValueError:
                            pass  # a broken channel is reported below
                if cancelled_at is not None and stamp - cancelled_at >= CANCEL_GRACE_SECONDS:
                    # An unload is no request to cancel: one past its bound is a close that did not end.
                    self.announce(f"retired: the engine did not close within {timeout:g} s"
                                  if action in ("unload", "close") else
                                  f"retired: the engine settled no cancel within {CANCEL_GRACE_SECONDS:g} s")
                    self.close()
                    return self._cancelled({"status": "timed_out" if timed_out else "cancelled",
                                            "error": "Reply timed out" if timed_out else "Stopped",
                                            "cleanup_settled": True}, consumer)
                if (root / "runtime.log").stat().st_size > 4 * 1024 * 1024:
                    raise ValueError("Resident diagnostic output limit exceeded")
                pieces, result = link.take(box)
                if stream and cancelled_at is None:
                    try:
                        for piece in pieces:  # every piece the host sent before its result
                            stream.apply(piece)
                    except OSError as exc:
                        # The client is gone (a failed write): cancel the turn as its stop would, and let the host
                        # settle it and keep its weights, instead of unwinding past the cancel.
                        consumer, cancelled_at = exc, time.monotonic()
                        try:
                            link.send({"kind": "cancel", "command_id": cid})
                        except ValueError:
                            pass  # a broken channel is reported below
                if result is not None:
                    if (set(result) - {"response", "rejection"} != RESULT or type(result["cleanup_settled"]) is not bool
                            or result["status"] not in {"completed", "cancelled", "failed", "rejected"}
                            or ("response" in result) != (action in RESPONDING and result["status"] == "completed")
                            or ("rejection" in result) != (action in RESPONDING and result["status"] == "rejected")
                            or ("rejection" in result and not (isinstance(result["rejection"], dict)
                                                              and isinstance(result["rejection"].get("reason"), str)))):
                        raise ValueError("Resident completion identity/cleanup mismatch")
                    result = {k: v for k, v in result.items() if k not in {"kind", "command_id"}}
                    if not result["cleanup_settled"]:
                        self.announce("retired: the engine did not settle its cleanup")
                        self.close()  # Retire the entire owner before admitting again.
                        result["cleanup_settled"] = True
                    if cancelled_at is not None and result["status"] != "failed":
                        result.update(status="timed_out" if timed_out else "cancelled", error="Reply timed out" if timed_out else "Stopped")
                        result.pop("response", None)
                        result.pop("rejection", None)
                        return self._cancelled(result, consumer)
                    return result
                if tree.process.poll() is not None or link.failure:
                    try:
                        tree.process.wait(1)  # a host that crashed also broke its channel
                    except subprocess.TimeoutExpired:
                        raise ValueError("Resident host channel failed: " + link.failure) from None
                    # Identify retained diagnostics without exposing arbitrary adapter log contents to API clients.
                    raise ValueError("Resident host exited before completing the request (exit %s); diagnostics: %s" %
                                     (tree.process.returncode, root / "runtime.log"))
                box.ready.wait(.02)
        finally:
            link.unregister(cid)

    @staticmethod
    def _cancelled(result, consumer):
        return {**result, "consumer_error": consumer} if consumer is not None else result

    def close(self):
        self.on_retire()
        with self.lifecycle:
            self._close()

    def _close(self):
        self.attached = None
        if self.tree:
            self.tree.close()
            self.tree = None
        if self.link:
            self.link.close()
            self.link = None
        if self.log:
            self.log.close()
            self.log = None
        if self.root:
            clean_snapshots(self.ownership.directory, self.root.name)
