"""Process supervisor for the opt-in resident host; commands, streams and results over its channel."""
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from crit.process.ownership import clean_snapshots
from crit.process.tree import guarded_tree
from crit.run.records import MAX_JSON_BYTES, write_json

from .channel import Supervisor
from .logprobs import RESPONDING, response_bytes
from .options import MAX_CHAT_RECORD
from .stream import StreamReader

CAPABILITY = "chat-resident-v1"
RESULT = {"kind", "command_id", "status", "error", "max_concurrent_requests", "cleanup_settled"}


class ResidentRuntime:
    def __init__(self, ownership, host_limit, device_limit):
        self.ownership = ownership
        self.host_limit, self.device_limit = host_limit, device_limit
        self.tree = self.log = self.root = self.link = None
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
        self.tree = guarded_tree([sys.executable, "-B", str(Path(__file__).with_name("host.py")), str(config)],
                                 self.root, self.log, self.ownership, self.root)

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

    def host_pid(self):
        """The resident host's process id once it has connected, else None."""
        with self.lifecycle:
            link = self.link
        return link.pid if link else None

    def call(self, action, payload, stop, timeout, progress=None, request=None, progress_tools=None,
             progress_reasoning=None, on_started=None):
        """Run one command on the host. A generation's payload carries its request record; its result carries the
        engine's response, and progress, progress_tools and progress_reasoning receive each streamed piece."""
        if stop.is_set():
            return {"status": "cancelled", "error": "Stopped", "cleanup_settled": True}
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
                          limit)

    def _call(self, cid, action, payload, stop, timeout, stream, on_started=None, root=None, tree=None, link=None,
              limit=MAX_JSON_BYTES):
        root, tree, link = root or self.root, tree or self.tree, link or self.link
        if stop.is_set():
            return {"status": "cancelled", "error": "Stopped", "cleanup_settled": True}
        box = link.register(cid, limit)
        try:
            link.send({"kind": "command", "command_id": cid, "action": action, "payload": payload},
                      MAX_CHAT_RECORD + MAX_JSON_BYTES)
            deadline, cancelled_at, timed_out = time.monotonic() + timeout, None, False
            if on_started:
                on_started(deadline)
            while True:
                stamp = time.monotonic()
                if stop.is_set() or stamp >= deadline:
                    if cancelled_at is None:
                        timed_out = not stop.is_set()
                        cancelled_at = stamp
                        try:
                            link.send({"kind": "cancel", "command_id": cid})
                        except ValueError:
                            pass  # a broken channel is reported below
                    if stamp - cancelled_at >= 1:
                        self.close()
                        return {"status": "timed_out" if timed_out else "cancelled", "error": "Reply timed out" if timed_out else "Stopped", "cleanup_settled": True}
                if (root / "runtime.log").stat().st_size > 4 * 1024 * 1024:
                    raise ValueError("Resident diagnostic output limit exceeded")
                pieces, result = link.take(box)
                if stream and cancelled_at is None:
                    for piece in pieces:  # every piece the host sent before its result
                        stream.apply(piece)
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
                        self.close()  # Retire the entire owner before admitting again.
                        result["cleanup_settled"] = True
                    if cancelled_at is not None and result["status"] != "failed":
                        result.update(status="timed_out" if timed_out else "cancelled", error="Reply timed out" if timed_out else "Stopped")
                        result.pop("response", None)
                        result.pop("rejection", None)
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

    def close(self):
        self.on_retire()
        with self.lifecycle:
            self._close()

    def _close(self):
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
