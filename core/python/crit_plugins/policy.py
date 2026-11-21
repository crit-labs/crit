"""An engine's policy: what one load gives its plugins, held by that engine and never by the process.

Core's loader reads a package's configuration once and gives its plugins what it asks: the kernel tuning its kernels
build with (configuration "kernel_tuning"), the record its measured decisions replay (configuration "decisions"), the
exactness of its rows, its quality mode and its prompt activations, and the folder for files a plugin can build again
(crit_core.engine.PLUGIN_CACHE). That is a Policy, one per engine: LoadContext.policy, and the engine's `policy`.

The plugins read it through the functions they always called (crit_plugins.kernels' asks(), allows(),
quality_mode(), activation(), tuning_file(), load_kernel(); crit_plugins.decisions' settle()), which answer from the
*active* policy: the one Core activates while it loads the engine and around every call it makes into the engine's
plugins (each step, admission, commit, retirement and score on the engine's worker, prepare(), telemetry(), close()),
and that the host's worker pool carries into its tasks. So two engines in one process (crit serve's host building
the next engine while one serves, ablation variants, crit eval ab's two engines) never see each other's policy.

Outside any engine (a test, a quick try in a shell) the functions answer from the process's own policy, which the
use_* functions set (crit_plugins.kernels.use_tuning, use_exactness, use_quality_mode, use_activations;
crit_plugins.decisions.use_record): inside an engine they set that engine's. The environment's CRIT_KERNEL_TUNING
and CRIT_DECISIONS stay the fallbacks for a quick try where the policy names no tuning and no record, read when they
are used, as before.
"""
from contextlib import contextmanager
import contextvars
import itertools
import os
from pathlib import Path
import threading

TUNING = "CRIT_KERNEL_TUNING"   # a folder of kernel tuning tables for a quick try, where the policy names none
DECISIONS = "CRIT_DECISIONS"     # a decisions record for a quick try, where the policy names none

_ACTIVE = contextvars.ContextVar("crit_plugins.policy", default=None)
_LOADS = itertools.count(1)   # the engines this process has loaded: a decision measured in its first load says so


class Policy:
    """One engine's policy (the module's description). tuning: the folder of kernel tuning tables, or None;
    exactness: {"decode": level, "prompt": level} the engine asks, or None; quality_mode: "preserve", "speed" or None;
    activations: the prompt activations asked (crit_plugins.kernels.use_activations' form), or None; decisions: the
    file of the decisions record, or None; cache: the folder for files plugins can build again, or None. load: this
    engine's number among the process's loads (None: count the next one). The kernels the engine's plugins asked for
    are recorded here too (kernels: kernel id -> record, crit_plugins.kernels.loaded)."""

    def __init__(self, tuning=None, exactness=None, quality_mode=None, activations=None, decisions=None, cache=None,
                 load=None):
        from .decisions import Record
        self.tuning = None if tuning is None else Path(tuning)
        self.exactness = None if exactness is None else dict(exactness)
        self.quality_mode = quality_mode
        self.activations = activations if activations is None or isinstance(activations, str) else dict(activations)
        self.cache = None if cache is None else Path(cache)
        self.decisions = Record(decisions, next(_LOADS) if load is None else int(load))
        self.kernels = {}
        self.lock = threading.RLock()

    def tuning_folder(self):
        """The folder of tuning tables the engine's kernels build with: its own, else CRIT_KERNEL_TUNING's, else
        None."""
        if self.tuning is not None:
            return self.tuning
        return Path(os.environ[TUNING]) if os.environ.get(TUNING) else None

    @contextmanager
    def active(self):
        """This policy as the active one for the calls made inside the block, on this thread."""
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def report(self):
        """What the engine was given, for its facts and a tool's view (JSON)."""
        return {"tuning": None if self.tuning is None else str(self.tuning), "exactness": self.exactness,
                "quality_mode": self.quality_mode, "activations": self.activations,
                "decisions": None if self.decisions.path is None else str(self.decisions.path),
                "cache": None if self.cache is None else str(self.cache), "load": self.decisions.load}


_process = None   # the policy outside any engine (tests, a quick try), made at its first use
_made = threading.Lock()


def process():
    """The process's own policy, which answers outside any engine."""
    global _process
    with _made:
        if _process is None:
            _process = Policy(load=0)
        return _process


def active():
    """The engine policy active on this thread, or None outside any engine."""
    return _ACTIVE.get()


def current():
    """The policy the plugins' calls answer from: the active engine's, else the process's own."""
    found = _ACTIVE.get()
    return found if found is not None else process()


def carried(function):
    """function wrapped to run with the policy active where carried() was called, on whichever thread calls it (the
    host's worker pool runs a caller's tasks on its own threads)."""
    policy = _ACTIVE.get()
    if policy is None:
        return function

    def run(*args, **kwargs):
        token = _ACTIVE.set(policy)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    return run


__all__ = ["DECISIONS", "TUNING", "Policy", "active", "carried", "current", "process"]
