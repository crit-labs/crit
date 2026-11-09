"""Commands the server runs as jobs: an owned process tree each, its output and errors kept in files, waited on
for a bounded time per call, stopped like Ctrl+C in a terminal and never left running when the server exits."""
import itertools
import os
from pathlib import Path
import signal
import sys
import threading
import time

from crit.process.tree import ProcessTree


def python():
    """This interpreter as a command, and the environment it needs. A Windows venv's python.exe is a launcher that
    starts the base interpreter: the job's tree starts the base directly (crit.process.tree.ProcessTree)."""
    return sys.executable, {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}


class Job:
    def __init__(self, number, label, command, cwd, folder, environment=None):
        self.id = f"job-{number}"
        self.label = label
        self.command = command
        self.cwd = cwd
        self.output = folder / f"{self.id}.out"
        self.errors = folder / f"{self.id}.err"
        self.started = time.time()
        self.ended = None
        self.exit_code = None
        self.stopping = False
        self.cleanup_error = None
        self.done = threading.Event()
        interpreter, base = python()
        environment = {**base, **(environment or {})}
        with open(self.output, "wb") as output, open(self.errors, "wb") as errors:
            self.tree = ProcessTree([interpreter, *command], cwd, output, env=environment, errors=errors)
        threading.Thread(target=self._watch, name=self.id, daemon=True).start()

    def _watch(self):
        # The only closer of the tree: on Windows closing it releases the handle a wait would be using.
        self.exit_code = self.tree.process.wait()
        try:
            self.tree.close()  # the children the command left behind end with it
        except (OSError, ValueError) as error:
            self.cleanup_error = str(error)
        self.ended = time.time()
        self.done.set()

    @property
    def state(self):
        if not self.done.is_set():
            return "stopping" if self.stopping else "running"
        return "stopped" if self.stopping else "exited"

    def stop(self, grace):
        """Interrupt the whole tree as Ctrl+C would, so crit releases its leases; end it after GRACE seconds."""
        if self.done.is_set():
            return
        self.stopping = True
        if os.name == "nt":
            # Windows has no Ctrl+C for a tree without a console: end the command, and the watcher ends its job.
            self.tree.process.kill()
        else:
            try:
                os.killpg(self.tree.process.pid, signal.SIGINT)
                if not self.done.wait(grace):
                    os.killpg(self.tree.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.done.wait(60)

    def read(self, stream, offset=0, limit=None):
        """Text of STREAM ("output" or "errors") from byte OFFSET, at most LIMIT bytes, and the offset after it."""
        path = self.output if stream == "output" else self.errors
        with open(path, "rb") as file:
            file.seek(offset)
            data = file.read() if limit is None else file.read(limit)
        return data.decode("utf-8", errors="replace"), offset + len(data)

    def size(self, stream):
        return (self.output if stream == "output" else self.errors).stat().st_size

    def tail(self, stream, limit):
        size = self.size(stream)
        text, _ = self.read(stream, max(0, size - limit))
        return text, size > limit

    def last_line(self):
        text, _ = self.tail("errors", 4096)
        lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
        return lines[-1][-200:] if lines else ""

    def summary(self):
        return {"job": self.id, "tool": self.label, "state": self.state, "exit_code": self.exit_code,
                "seconds": round((self.ended or time.time()) - self.started, 1),
                "logs": {"output": str(self.output), "errors": str(self.errors)},
                **({"cleanup_error": self.cleanup_error} if self.cleanup_error else {})}


class Jobs:
    def __init__(self, folder, environment=None):
        """environment: variables every job gets beside this process's (the server's project for crit-eval)."""
        self.folder = Path(folder)
        self.environment = dict(environment or {})
        self.folder.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.numbers = itertools.count(1)

    def start(self, label, command, cwd):
        with self.lock:
            job = Job(next(self.numbers), label, command, cwd, self.folder, self.environment)
            self.jobs[job.id] = job
        return job

    def get(self, name):
        job = self.jobs.get(name)
        if job is None:
            raise ValueError(f"No job {name!r}; this server's jobs: {', '.join(self.jobs) or 'none'}")
        return job

    def close(self, grace=10):
        """Stop every job still running, all at once."""
        threads = [threading.Thread(target=job.stop, args=(grace,)) for job in self.jobs.values() if not job.done.is_set()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
