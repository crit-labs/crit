"""Bounded single-flight grammar preparation; only immutable artifacts are cached.

Compiles on CPU workers while requests wait. Poll ready() from an adapter's
deferred admission; construct one private native matcher when admitted.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from threading import RLock
import time


class PreparationPending(RuntimeError):
    pass


class PreparationFailed(RuntimeError):
    pass


@dataclass(eq=False)
class _Entry:
    key: object
    reserved: int
    queued_at: float
    future: object = None
    result: object = None
    error: str = ""
    clients: int = 0
    started_at: float | None = None
    completed_at: float | None = None


class PreparedMatcher:
    """Private mutable matcher plus its charged immutable artifact owner."""
    def __init__(self, handle, matcher):
        self._handle, self.matcher = handle, matcher

    def close(self):
        if self._handle is None:
            raise ValueError("matcher lease already closed")
        self._handle.close()
        self._handle, self.matcher = None, None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Preparation:
    def __init__(self, service, entry, deadline):
        self._service, self._entry, self._deadline = service, entry, deadline
        self._state, self._error = 'pending', ''

    @property
    def state(self):
        self._service.poll()
        with self._service._lock:
            return self._state

    def ready(self):
        return self.state == 'ready'

    def result(self):
        """Borrow the immutable artifact; keep this handle open while using it."""
        self._service.poll()
        with self._service._lock:
            if self._state == 'pending':
                raise PreparationPending('grammar preparation pending')
            if self._state not in {'ready', 'leased'}:
                raise PreparationFailed(self._error or self._state)
            return self._entry.result

    def new_matcher(self, factory):
        """Call the engine's fresh-matcher factory once, outside the service lock."""
        self._service.poll()
        with self._service._lock:
            if self._state != 'ready' or not callable(factory):
                raise PreparationFailed(self._error or 'fresh ready preparation required')
            result = self._entry.result
            self._state = 'creating'
        try:
            matcher = factory(result)
            if matcher is None:
                raise ValueError('factory returned no matcher')
            lease = PreparedMatcher(self, matcher)
        except BaseException:
            with self._service._lock:
                if self._service._closed:
                    self._service._detach(self, 'closed')
                else:
                    self._state = 'ready'
            raise
        with self._service._lock:
            self._state = 'leased'
        return lease

    def cancel(self):
        with self._service._lock:
            if self._state in {'creating', 'leased'}:
                raise ValueError('close the settled private matcher lease')
            self._service._detach(self, 'cancelled')
        self._service.poll()

    def close(self):
        with self._service._lock:
            if self._state == 'creating':
                raise ValueError('matcher construction still owns the artifact')
            self._service._detach(self, 'closed')
        self._service.poll()

    @property
    def timings(self):
        with self._service._lock:
            entry = self._entry
            if entry is None:
                raise ValueError('preparation has released its timing record')
            return dict(queued_at=entry.queued_at, started_at=entry.started_at, completed_at=entry.completed_at)


class PreparationService:
    """Poll-driven bounded queue/cache with real CPU compilation workers.

    Limits include active leases: pinned entries cannot be evicted to admit new
    jobs. Each pending entry reserves input bytes plus max_artifact_bytes. This
    bounds retained payload, not Python/compiler/native matcher workspace.
    """
    def __init__(self, *, compiler=None, workers=2, max_pending=16, max_waiters=128,
                 max_entries=32, max_bytes=67108864, max_artifact_bytes=4194304,
                 clock=time.monotonic):
        for name, value, cap in (('workers', workers, 32), ('max_pending', max_pending, 1024),
                                 ('max_waiters', max_waiters, 4096), ('max_entries', max_entries, 4096),
                                 ('max_bytes', max_bytes, 1 << 32), ('max_artifact_bytes', max_artifact_bytes, 4194304)):
            if type(value) is not int or not 0 < value <= cap:
                raise ValueError(f'bounded positive {name} required')
        if not callable(clock):
            raise ValueError('monotonic clock required')
        if compiler is None:
            from crit_core.constraints import compiler
        self._request, self._compile, self._result_type = compiler.make_compile_request, compiler.compile_language, compiler.CompiledConstraint
        self._version = compiler.VERSION
        self.workers, self.max_pending, self.max_waiters = workers, max_pending, max_waiters
        self.max_entries, self.max_bytes, self.max_artifact_bytes = max_entries, max_bytes, max_artifact_bytes
        self._clock, self._lock = clock, RLock()
        self._entries, self._handles, self._bytes = OrderedDict(), set(), 0
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='crit-grammar')

    def _now(self):
        now = self._clock()
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            raise ValueError('finite monotonic time required')
        return now

    def _detach(self, handle, state, error=''):
        if handle in self._handles:
            handle._entry.clients -= 1
            self._handles.remove(handle)
        handle._entry, handle._state, handle._error = None, state, error

    def _drop(self, entry):
        del self._entries[entry.key]
        self._bytes -= entry.reserved

    def _poll(self):
        now = self._now()
        for entry in tuple(self._entries.values()):
            if entry.future is not None and entry.future.done():
                try:
                    artifact = entry.future.result()
                    if type(artifact) is not str or len(artifact.encode()) > self.max_artifact_bytes:
                        raise ValueError('compiled artifact exceeds admitted byte bound')
                    entry.result = self._result_type(artifact, entry.key.tokenizer_identity, entry.key.pieces, entry.key.eos)
                    size = entry.key.input_bytes + len(artifact.encode())
                except BaseException as error:
                    entry.error, size = f'{type(error).__name__}: {error}'[:512], entry.key.input_bytes
                entry.future = None
                self._bytes -= entry.reserved - size
                entry.reserved, entry.completed_at = size, now
        for handle in tuple(self._handles):
            if handle._state in {'creating', 'leased'}:
                continue
            entry = handle._entry
            if now >= handle._deadline:
                self._detach(handle, 'timeout', 'grammar preparation deadline expired')
            elif entry.error:
                self._detach(handle, 'failed', entry.error)
            elif entry.result is not None:
                handle._state = 'ready'
        for entry in tuple(self._entries.values()):
            if not entry.clients and entry.future is None and (entry.result is None or self._closed):
                self._drop(entry)
        if self._closed:
            return
        running = sum(entry.future is not None for entry in self._entries.values())
        for entry in self._entries.values():
            if running == self.workers:
                break
            if entry.started_at is None and entry.clients:
                # Only these bounded slots ever reach the executor's queue.
                try:
                    entry.future = self._executor.submit(self._compile, entry.key.kind, entry.key.source)
                except Exception as error:
                    entry.error, entry.completed_at = str(error)[:512], now
                entry.started_at = now
                running += int(entry.future is not None)

    def poll(self):
        with self._lock:
            self._poll()

    def request(self, kind, source, *, tokenizer_identity, pieces, eos, timeout=30):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ValueError('positive preparation timeout up to one day required')
        key = self._request(kind, source, tokenizer_identity=tokenizer_identity, pieces=pieces, eos=eos, version=self._version)
        with self._lock:
            self._poll()
            if self._closed or len(self._handles) >= self.max_waiters:
                raise ValueError('preparation service closed or waiter capacity exhausted')
            entry = self._entries.get(key)
            if entry is None:
                pending = sum(e.result is None and not e.error for e in self._entries.values())
                reserve = key.input_bytes + self.max_artifact_bytes
                if pending >= self.max_pending or reserve > self.max_bytes:
                    raise ValueError('preparation queue or byte capacity exhausted')
                reclaim = [e for e in self._entries.values() if not e.clients and e.future is None]
                count, needed = len(self._entries), self._bytes + reserve
                victims = []
                for old in reclaim:
                    if count < self.max_entries and needed <= self.max_bytes:
                        break
                    victims.append(old)
                    count, needed = count - 1, needed - old.reserved
                if count >= self.max_entries or needed > self.max_bytes:
                    raise ValueError('live artifact leases exhaust preparation cache capacity')
                for old in victims:
                    self._drop(old)
                entry = _Entry(key, reserve, self._now())
                self._entries[key] = entry
                self._bytes += reserve
            else:
                self._entries.move_to_end(key)
            handle = Preparation(self, entry, self._now() + timeout)
            self._handles.add(handle)
            entry.clients += 1
            self._poll()
            return handle

    @property
    def stats(self):
        with self._lock:
            self._poll()
            return dict(entries=len(self._entries), waiters=len(self._handles), reserved_bytes=self._bytes,
                        running=sum(e.future is not None for e in self._entries.values()),
                        pending=sum(e.result is None and not e.error for e in self._entries.values()))

    def close(self):
        """Stop admission and join actual compilers; live matcher leases stay charged."""
        with self._lock:
            self._closed = True
            for handle in tuple(self._handles):
                if handle._state not in {'creating', 'leased'}:
                    self._detach(handle, 'closed')
            self._poll()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.poll()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
