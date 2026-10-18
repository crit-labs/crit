"""Process supervisor for the opt-in resident host; immutable per-command IPC."""
from pathlib import Path
import errno
import sys
import time
import uuid

from crit.process.ownership import clean_snapshots
from crit.process.tree import guarded_tree
from crit.run.records import read_json, write_json

from .stream import StreamReader

CAPABILITY = "chat-resident-v1"


class ResidentRuntime:
    def __init__(self, ownership, host_limit, device_limit):
        self.ownership = ownership
        self.host_limit, self.device_limit = host_limit, device_limit
        self.tree = self.log = self.root = None

    def start(self):
        if self.tree:
            return
        self.root = self.ownership.directory / "runtimes" / uuid.uuid4().hex
        for name in ("commands", "accepted", "results", "cancel"):
            (self.root / name).mkdir(parents=True)
        self.ownership.update(runtime_id=self.root.name)
        config = self.root / "host.json"
        write_json(config, {"root": str(self.root), "token": self.ownership.token,
                   "host_limit": self.host_limit, "device_limit": self.device_limit}, immutable=True)
        self.log = (self.root / "runtime.log").open("xb")
        self.tree = guarded_tree([sys.executable, "-B", str(Path(__file__).with_name("host.py")), str(config)],
                                 self.root, self.log, self.ownership, self.root)

    def status(self):
        if not self.tree:
            return {"state": "unloaded", "host_bytes": 0, "device_bytes": 0}
        if self.tree.process.poll() is not None:
            return {"state": "interrupted", "host_bytes": 0, "device_bytes": 0,
                    "error": "Resident host exited; Unload verifies remaining process cleanup"}
        path = self.root / "status.json"
        deadline = time.monotonic() + .25
        while True:
            try:
                return read_json(path) if path.exists() else {"state": "starting", "host_bytes": 0, "device_bytes": 0}
            except PermissionError as exc:
                # Atomic replacement can briefly deny a Windows reader's open.
                # Retrying never supplies a cached or fabricated ready status.
                code = getattr(exc, "winerror", None)
                # Python's Windows file open can expose only EACCES, omitting
                # the native sharing code. Persistent access denial still fails.
                retryable = code in {5, 32, 33} or (code is None and exc.errno == errno.EACCES)
                if sys.platform != "win32" or not retryable or time.monotonic() >= deadline:
                    raise
                time.sleep(.005)

    def call(self, action, payload, stop, timeout, progress=None, request=None, progress_tools=None):
        self.start()
        cid = uuid.uuid4().hex
        stream_path = self.root / "stream.json"
        stream = (StreamReader(stream_path, cid, self.ownership.token, request, progress or (lambda text: None),
                               progress_tools) if progress or progress_tools else None)
        result = self._call(cid, action, payload, stop, timeout, stream)
        # Completion proves producer settlement. On an exception the owner
        # retires the host, then close()/recovery removes this private snapshot.
        stream_path.unlink(missing_ok=True)
        return result

    def _call(self, cid, action, payload, stop, timeout, stream):
        write_json(self.root / "commands" / (cid + ".json"), {"command_id": cid, "token": self.ownership.token,
                   "action": action, "payload": payload}, immutable=True)
        deadline, cancelled_at, timed_out = time.monotonic() + timeout, None, False
        result_path = self.root / "results" / (cid + ".json")
        while True:
            stamp = time.monotonic()
            if stop.is_set() or stamp >= deadline:
                if cancelled_at is None:
                    timed_out = not stop.is_set()
                    cancelled_at = stamp
                    write_json(self.root / "cancel" / (cid + ".json"), {"command_id": cid}, immutable=True)
                if stamp - cancelled_at >= 1:
                    self.close()
                    return {"status": "timed_out" if timed_out else "cancelled", "error": "Reply timed out" if timed_out else "Stopped", "cleanup_settled": True}
            if (self.root / "runtime.log").stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Resident diagnostic output limit exceeded")
            if stream and cancelled_at is None:
                stream.poll()
            if result_path.exists():
                result = read_json(result_path)
                if result.get("schema_version") != 1 or result.get("command_id") != cid or result.get("token") != self.ownership.token or type(result.get("cleanup_settled")) is not bool or result.get("status") not in {"completed", "cancelled", "failed"}:
                    raise ValueError("Resident completion identity/cleanup mismatch")
                if not result["cleanup_settled"]:
                    self.close()  # Retire the entire owner before admitting again.
                    result["cleanup_settled"] = True
                if cancelled_at is not None and result["status"] != "failed":
                    result.update(status="timed_out" if timed_out else "cancelled", error="Reply timed out" if timed_out else "Stopped")
                if stream and result["status"] == "completed":
                    # The final snapshot precedes completion publication. Poll
                    # once more in case it appeared after this loop's read.
                    stream.poll()
                return result
            if self.tree.process.poll() is not None:
                # Identify retained diagnostics without exposing arbitrary adapter log contents to API clients.
                raise ValueError("Resident host exited before completing the request (exit %s); diagnostics: %s" %
                                 (self.tree.process.returncode, self.root / "runtime.log"))
            time.sleep(.02)

    def close(self):
        if self.tree:
            self.tree.close()
            self.tree = None
        if self.log:
            self.log.close()
            self.log = None
        if self.root:
            clean_snapshots(self.ownership.directory, self.root.name)
