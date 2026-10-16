"""Caller-paced prefetch decisions over existing private, receipt-bound transfers.

Control calls and callbacks are serialized. Deadlines stop waiting/submission,
never physical readers. No worker thread, model layout or device is assumed.
"""
from dataclasses import dataclass, replace
from itertools import islice
import math
import time


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative seconds required")
    return value


@dataclass(frozen=True)
class ReadyPrefix:
    source: object
    acquire: object  # Revalidate the exact snapshot and return an owned lease.


@dataclass
class PreparedPrefetch:
    source: object
    transfer: object
    publish: object  # Return ReadyPrefix after complete validated publication.
    discard: object  # May raise while readers or ambiguous work remain.
    valid: object
    remaining_seconds: object  # Live measured estimate, or None when unknown.


@dataclass(frozen=True)
class PrefetchDecision:
    action: str  # wait, reuse, recompute
    source: object = None
    lease: object = None
    predicted_seconds: float = 0
    reason: str = ""


@dataclass
class _Job:
    prepared: PreparedPrefetch
    deadline: float
    abandoned: bool = False
    error: str = ""


class PrefetchController:
    """Bounded live jobs, round-robin I/O quotas and cost/deadline decisions.

    Each key belongs to one queued request. Shared immutable source leases may
    back several keys, but cancelling a key never cancels another transfer.
    Keep polling retained jobs after their request chooses another path.
    """
    def __init__(self, *, planning, max_jobs=16, clock=time.monotonic):
        if type(max_jobs) is not int or not 0 < max_jobs <= 16 or not callable(clock):
            raise ValueError("one through sixteen jobs and a monotonic clock required")
        self.planning, self.max_jobs, self.clock = planning, max_jobs, clock
        self._jobs, self._cursor = {}, 0

    @property
    def retained(self):
        return tuple(self._jobs)

    def start(self, key, prepare, *, deadline):
        """Reserve a slot before preparation; prepare must not submit work."""
        _seconds(deadline)
        if (type(key) is not str or not key or len(key) > 256 or key in self._jobs
                or len(self._jobs) >= self.max_jobs or not callable(prepare)):
            raise ValueError("unique bounded key, available slot and preparation required")
        if _seconds(self.clock()) >= deadline:
            raise ValueError("prefetch deadline has already expired")
        prepared = prepare()
        # Retain even an invalid prepared object until explicit discard succeeds.
        job = self._jobs[key] = _Job(prepared, deadline)
        if (not isinstance(prepared, PreparedPrefetch)
                or not isinstance(prepared.source, self.planning.PrefixSource)
                or any(not callable(getattr(prepared, field, None)) for field in
                       ("publish", "discard", "valid", "remaining_seconds"))
                or any(not callable(getattr(prepared.transfer, field, None)) for field in
                       ("pump", "done", "result"))
                or type(getattr(getattr(prepared.transfer, 'limits', None), 'window_bytes', None)) is not int):
            job.abandoned, job.error = True, "invalid prepared transfer"
            raise ValueError(job.error)

    def cancel(self, key):
        self._jobs[key].abandoned = True
        self.collect()

    def collect(self):
        """Try to discard abandoned jobs; failures remain bounded and owned."""
        for key, job in tuple(self._jobs.items()):
            if job.abandoned:
                try:
                    job.prepared.discard()
                except Exception as error:
                    job.error = str(error)[:256]
                else:
                    del self._jobs[key]

    def error(self, key):
        return self._jobs[key].error

    def pump(self, *, max_bytes, max_submissions=1):
        """Poll all jobs, then allow at most the supplied aggregate I/O budget.

        One conservative submission attempt per turn in round-robin order;
        read and write bytes both count. Quotas smaller than a window only poll.
        """
        if (type(max_bytes) is not int or not 0 <= max_bytes <= 1 << 40
                or type(max_submissions) is not int or not 0 <= max_submissions <= 128):
            raise ValueError("bounded aggregate bytes and submission count required")
        now = _seconds(self.clock())
        for job in self._jobs.values():
            if job.abandoned:
                continue
            try:
                if now >= job.deadline or job.prepared.valid() is not True:
                    job.abandoned = True
                else:
                    job.prepared.transfer.pump(max_bytes=0, max_submissions=0)
            except Exception as error:
                job.abandoned, job.error = True, str(error)[:256]
        self.collect()
        jobs = [j for j in self._jobs.values() if not j.abandoned]
        submitted = 0
        if not jobs:
            return submitted
        for _ in range(max_submissions):
            job = jobs[self._cursor % len(jobs)]
            self._cursor = (self._cursor + 1) % len(jobs)
            if job.abandoned or max_bytes - submitted < job.prepared.transfer.limits.window_bytes:
                continue
            try:
                submitted += job.prepared.transfer.pump(max_bytes=max_bytes - submitted, max_submissions=1)
            except Exception as error:
                job.abandoned, job.error = True, str(error)[:256]
                # An ambiguous submission may have consumed the remaining quota.
                break
        self.collect()
        return submitted

    def decide(self, key, query, cost, *, fingerprint, codec, components, alternatives=()):
        """Choose an owned complete prefix, bounded wait, or recomputation.

        Ready alternatives are revalidated at acquisition. A stale alternative
        is removed and the rest ranked again. Pending bytes are never published
        or returned as a cache hit. Acquired leases belong to the caller.
        """
        alternatives = tuple(islice(alternatives, 65))
        if (len(alternatives) > 64 or any(not isinstance(a, ReadyPrefix) or not callable(a.acquire)
                                        for a in alternatives)):
            raise ValueError("at most 64 ready snapshot lease providers required")
        query, components = tuple(query), frozenset(components)
        identity = dict(fingerprint=fingerprint, codec=codec, components=components)
        # Validate once before any transfer state mutation or publication.
        self.planning.choose_prefix(query, [a.source for a in alternatives], cost, **identity)
        job = self._jobs.get(key)
        if job is not None and not job.abandoned:
            prepared = job.prepared
            try:
                candidate = replace(prepared.source, ready=True, transfers=())
                plan = self.planning.choose_prefix(query, (candidate,), cost, **identity)
                if prepared.valid() is not True or plan.source is None:
                    job.abandoned = True
                elif prepared.transfer.done():
                    prepared.transfer.result()
                    ready = prepared.publish()
                    if not isinstance(ready, ReadyPrefix) or ready.source != candidate:
                        raise ValueError("publication changed complete prefix identity")
                    # Successful publication has transferred ownership to the tier.
                    del self._jobs[key]
                    job = None
                    alternatives = (ready,) + tuple(a for a in alternatives if a.source.name != ready.source.name)
            except Exception as error:
                if job is not None:
                    job.abandoned, job.error = True, str(error)[:256]
        while True:
            fallback = self.planning.choose_prefix(query, [a.source for a in alternatives], cost, **identity)
            if job is not None and not job.abandoned:
                now = _seconds(self.clock())
                try:
                    remaining = job.prepared.remaining_seconds()
                    if remaining is not None:
                        _seconds(remaining)
                        candidate = replace(job.prepared.source, ready=True, transfers=())
                        pending = self.planning.choose_prefix(query, (candidate,), cost, **identity)
                        total = remaining + pending.predicted_seconds
                        if (now < job.deadline and remaining <= job.deadline - now
                                and pending.source is not None and total < fallback.predicted_seconds):
                            return PrefetchDecision("wait", predicted_seconds=total, reason="beneficial prefetch within deadline")
                except Exception as error:
                    job.error = str(error)[:256]
                job.abandoned = True
            self.collect()
            if fallback.source is None:
                return PrefetchDecision("recompute", predicted_seconds=fallback.predicted_seconds,
                                        reason="no beneficial complete prefix")
            provider = next(a for a in alternatives if a.source is fallback.source)
            try:
                lease = provider.acquire()
                if lease is None:
                    raise ValueError("prefix acquisition returned no owner")
            except Exception:
                alternatives = tuple(a for a in alternatives if a is not provider)
                continue
            return PrefetchDecision("reuse", fallback.source, lease, fallback.predicted_seconds,
                                    "complete generation revalidated and leased")


def tier_prefix(source, tier, key, identity):
    """Bind a complete source to an exact host-tier generation, then lease it."""
    generation = tier.generation(key)
    if generation is None:
        raise KeyError(key)
    def acquire():
        if tier.generation(key) != generation:
            raise ValueError("stale ready tier generation")
        return tier.acquire(key, identity)
    return ReadyPrefix(source, acquire)


def tier_prefetch(source, move, remaining_seconds, *, settle=None):
    """Adapt a prepared TierMove without changing whole-state publication."""
    generation = move.source.generation(move.key)
    def valid():
        return move.lease.current and move.source.generation(move.key) == generation
    def publish():
        move.publish()
        return tier_prefix(replace(source, ready=True, transfers=()), move.destination, move.key, move.identity)
    return PreparedPrefetch(source, move.transfer, publish, lambda: move.discard(settle), valid, remaining_seconds)
