"""One reserved attempt of a candidate, as crit run build, measure, eval and serve run it: check the candidate, reserve,
run the work with its log, record how it ended, settle, then raise when it did not complete.

An attempt ends the same way for each of them: an Expired or TimeoutError is timed_out; Ctrl+C and SystemExit (crit run
turns SIGTERM into one) are interrupted and go on once the attempt is settled; an exception whose `settled` is False
leaves its cleanup unproven; anything else failed. A completed attempt that ran past its reservation's deadline is
timed_out too. The log and the attempt's other files are evidence only once cleanup is proven: a process that was not
stopped may still write to them.
"""
from contextlib import contextmanager
import sys

from ..records import utc_now
from .records import file_sha, ident, now, read_json, safe_path, timestamp

SEALED = "Sealed candidate is immutable; allocate a new candidate"


class Expired(Exception):
    """The reservation's deadline passed while the candidate was served."""


def remaining(deadline_at):
    """Seconds left before the reservation's deadline."""
    return (timestamp(deadline_at, "deadline") - utc_now()).total_seconds()


def unsealed(root, candidate_id, sealed=SEALED):
    """The candidate's folder; `sealed` is the error for a sealed candidate, which takes no new attempt."""
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    if (folder / "seal.json").exists():
        raise ValueError(sealed)
    return folder


def packaged(folder, doing, who):
    """The candidate's package manifest; the package must be a plugin package, the only kind Core's loader runs.
    `doing` and `who` word the errors ("measuring", "Eval measures")."""
    path = folder / "package" / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Package the candidate before {doing} it")
    if not (folder / "package" / "plugins").is_dir():
        raise ValueError(f"{who} the candidate through crit serve, which runs plugin packages only: package it as "
                         "plugins/ and the configuration that selects them (docs/PLUGINS.md)")
    return read_json(path)


class Attempt:
    """A reserved attempt of `operation` on a candidate, logged to candidates/ID/logs/NAME-ATTEMPT.log. `noun` names it
    in the error an attempt that did not complete raises ("Measurement timed_out: ..."), `subject` in the deadline's
    message ("The measurement passed its reserved deadline"). With `deadline` False a completed attempt is never
    timed_out (crit run serve serves until its deadline). With `launch`, an OSError is launch_failed (the build's
    command did not start)."""

    def __init__(self, store, run_id, candidate_id, operation, seconds, name, noun, subject, deadline=True,
                 launch=False):
        self.store, self.run_id, self.noun, self.subject = store, run_id, noun, subject
        self.check_deadline, self.launch = deadline, launch
        self.root = store.directory(run_id)
        self.reservation = store.reserve(run_id, candidate_id, operation, seconds)
        self.id = self.reservation["attempt_id"]
        self.log_path = safe_path(self.root, f"candidates/{candidate_id}/logs/{name}-{self.id}.log")
        self.outcome, self.failure, self.settled, self.interrupted, self.exit_code = "completed", None, True, None, None

    @property
    def deadline_at(self):
        return self.reservation["deadline_at"]

    @contextmanager
    def running(self, tee=True):
        """Run the attempt's work in the block. With `tee` the block gets log(message), which prints a line and appends
        it to the log; without, None (the work writes the log itself). How the block ends is recorded, not raised:
        finish() raises."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not tee:
                yield None
                return
            with self.log_path.open("a", encoding="utf-8") as stream:
                def log(message):
                    print(message, file=sys.stderr, flush=True)
                    stream.write(message + "\n")
                    stream.flush()
                yield log
        except BaseException as exc:
            self.ended(exc)

    def ended(self, exc):
        """Record how an exception ended the attempt."""
        self.settled = getattr(exc, "settled", True)
        if isinstance(exc, (Expired, TimeoutError)):
            self.outcome, self.failure = "timed_out", str(exc) or f"{self.subject} passed its reserved deadline"
        elif isinstance(exc, (KeyboardInterrupt, SystemExit)):
            self.outcome, self.failure, self.interrupted = "interrupted", "Interrupted", exc
        else:
            self.outcome = "launch_failed" if self.launch and isinstance(exc, OSError) else "failed"
            self.failure = f"{type(exc).__name__}: {exc}"[:8000]

    def end(self, outcome, failure, settled, exit_code):
        """Record the outcome the work reported itself (a build's command and its process tree)."""
        self.outcome, self.failure, self.settled, self.exit_code = outcome, failure, settled, exit_code

    def settle(self, evidence=()):
        """Settle the attempt and return its result: a completed attempt past its deadline is timed_out, and the log
        and `evidence` (other files the attempt wrote) are kept once cleanup is proven."""
        ended = now()
        if (self.check_deadline and self.outcome == "completed"
                and timestamp(ended, "ended") > timestamp(self.deadline_at, "deadline")):
            self.outcome, self.failure = "timed_out", f"{self.subject} passed its reserved deadline"
        logs = ([{"path": path.relative_to(self.root).as_posix(), "sha256": file_sha(path)}
                 for path in (self.log_path, *evidence) if path.is_file()] if self.settled else [])
        exit_code = self.exit_code if self.exit_code is not None or self.outcome != "completed" else 0
        return self.store.settle(self.run_id, self.reservation, {
            "outcome": self.outcome, "ended_at": ended, "exit_code": exit_code, "cleanup_settled": self.settled,
            "logs": logs, "failure": self.failure})

    def finish(self, detail=""):
        """After settle(): an interrupt goes on, and an attempt that did not complete raises ValueError naming its
        log; `detail` follows the log's path."""
        if self.interrupted is not None:
            raise self.interrupted
        if self.outcome != "completed":
            raise ValueError(f"{self.noun} {self.outcome}: {self.failure} "
                             f"(log {self.log_path.relative_to(self.root).as_posix()}){detail}")
