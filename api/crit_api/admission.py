"""Admission of requests for one resident identity, and exclusive lifecycle work: crit serve's side of scheduling.

The engine schedules the requests it holds (Core's runtime: which run at once, in what order by their priority,
how many: crit_core.native.StepEngine). crit serve's queue keeps what the engine cannot: exclusivity of the loaded
model (requests for another one wait for a switch, and lifecycle work for the requests before it), back-pressure at
the engine's capacity (`capacity`: the requests the engine holds at once, as it reports them; Core applies crit
serve's --max-concurrency), and the queue's bound and timeout that become HTTP 429. Those for one identity queued with
no switch between them leave in Core's order (crit_core.native.next_request: lower priority values first, then
arrival), so a request that waits here keeps the place it would have in the engine. The queue is unbounded unless
`max_queue` or `queue_timeout` is set: then a request that finds `max_queue` already waiting, or that has waited
`queue_timeout` seconds, is refused (Refused).
"""
import itertools
import threading
import time

from crit_core.native import next_request


class Refused(Exception):
    """A request turned away: max_queue requests were already waiting, or it waited queue_timeout seconds."""


class Waiter:
    """A request waiting for admission: the identity it needs and its priority (lower values go first). Exclusive
    lifecycle work (Admission.acquire) waits in the same queue under an identity of its own, so no request overtakes
    it and it overtakes none, but it counts toward neither max_queue nor waiting()."""
    def __init__(self, identity, priority, exclusive=False):
        self.identity, self.priority, self.exclusive = identity, priority, exclusive


class Admission:
    def __init__(self, max_queue=None, queue_timeout=None):
        self.condition = threading.Condition()
        self.active = 0
        self.exclusive = False
        self.identity = None
        self.capacity = 1
        self.ready = False
        self.invalid = False
        self.waiters = []
        self.max_queue, self.queue_timeout = max_queue, queue_timeout
        self.used = time.monotonic()  # when a request last held the engine, or it last loaded

    def configure(self, identity, capacity):
        with self.condition:
            self.identity, self.capacity, self.ready = identity, capacity, True
            self.used = time.monotonic()
            self.condition.notify_all()

    def invalidate(self):
        with self.condition:
            self.invalid = True
            self.ready = False
            self.condition.notify_all()

    def admit(self, identity, stop, priority=0):
        """Return whether this request must load; None means cancelled while waiting.

        A queued identity switch stops later requests overtaking it. A new or
        retired identity admits only its loader until capacity is confirmed.
        Waiters for one identity queued with no switch between them go by
        priority, lower first, then arrival.
        A request that cannot start at once raises Refused when max_queue
        requests already wait, or once it has waited queue_timeout seconds.
        """
        ticket, arriving = Waiter(identity, priority), True
        deadline = None if self.queue_timeout is None else time.monotonic() + self.queue_timeout
        with self.condition:
            self.waiters.append(ticket)
            try:
                while not stop.is_set():
                    if not self.exclusive and self.head() is ticket:
                        if not self.active:
                            if identity != self.identity or self.invalid:
                                self.ready, self.capacity = False, 1
                            self.invalid = False
                            self.identity = identity
                        if (not self.invalid and self.identity == identity and self.active < self.capacity
                                and (not self.active or self.ready)):
                            self.active += 1
                            self.used = time.monotonic()
                            return not self.ready
                    if arriving and self.max_queue is not None and self.queued() > self.max_queue:
                        raise Refused("The engine's queue is full (%d waiting, --max-queue); retry later"
                                      % self.max_queue)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise Refused("The request waited %g s for the engine (--queue-timeout); retry later"
                                      % self.queue_timeout)
                    arriving = False
                    self.condition.wait(.05)
                return None
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def head(self):
        """The waiter admitted next: the lowest priority value, then the oldest, of the first waiter and those after it
        that need its identity, up to the first that needs another. A waiter never overtakes a queued switch, and a
        switch never overtakes earlier waiters, so priority adds no switches."""
        first = self.waiters[0].identity
        group = list(itertools.takewhile(lambda waiter: waiter.identity == first, self.waiters))
        return group[next_request(waiter.priority for waiter in group)]

    def acquire(self, blocking=True, timeout=-1):
        """Exclusive lifecycle work, queued with requests: it waits for the active ones to drain, and requests that
        arrive later wait for it."""
        deadline = None if timeout < 0 else time.monotonic() + timeout
        ticket = Waiter(object(), 0, exclusive=True)
        with self.condition:
            self.waiters.append(ticket)
            try:
                while self.exclusive or self.active or self.head() is not ticket:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if not blocking or (remaining is not None and remaining <= 0):
                        return False
                    self.condition.wait(remaining)
                self.exclusive = True
                return True
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def release(self):
        with self.condition:
            if self.exclusive:
                self.exclusive = False
            elif self.active:
                self.active -= 1
                self.used = time.monotonic()
            else:
                raise RuntimeError("Release of unowned admission")
            self.condition.notify_all()

    def locked(self):
        with self.condition:
            return self.exclusive or bool(self.active)

    def waiting(self):
        """Requests queued for admission."""
        with self.condition:
            return self.queued()

    def queued(self):
        """The requests in the queue, without exclusive lifecycle work; the caller holds the condition."""
        return sum(not waiter.exclusive for waiter in self.waiters)

    def loading(self):
        """Whether the engine is not settled for requests: lifecycle work holds it, or an admitted request is loading
        or retiring it (a switch, a reload). A settled retire leaves no engine and a cancelled load the one the host
        settled with, which its status reports."""
        with self.condition:
            return self.exclusive or (bool(self.active) and not self.ready)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
