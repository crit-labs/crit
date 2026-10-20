"""Fair bounded admission for one resident identity and exclusive lifecycle work."""
import threading
import time


class Admission:
    def __init__(self):
        self.condition = threading.Condition()
        self.active = 0
        self.exclusive = False
        self.identity = None
        self.capacity = 1
        self.ready = False
        self.invalid = False
        self.waiters = []

    def configure(self, identity, capacity):
        with self.condition:
            self.identity, self.capacity, self.ready = identity, capacity, True
            self.condition.notify_all()

    def invalidate(self):
        with self.condition:
            self.invalid = True
            self.ready = False
            self.condition.notify_all()

    def admit(self, identity, stop):
        """Return whether this request must load; None means cancelled while waiting.

        A queued identity switch stops later requests overtaking it. A new or
        retired identity admits only its loader until capacity is confirmed.
        """
        ticket = object()
        with self.condition:
            self.waiters.append(ticket)
            try:
                while not stop.is_set():
                    if not self.exclusive and self.waiters[0] is ticket:
                        if not self.active:
                            if identity != self.identity or self.invalid:
                                self.ready, self.capacity = False, 1
                            self.invalid = False
                            self.identity = identity
                        if (not self.invalid and self.identity == identity and self.active < self.capacity
                                and (not self.active or self.ready)):
                            self.active += 1
                            return not self.ready
                    self.condition.wait(.05)
                return None
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def acquire(self, blocking=True, timeout=-1):
        deadline = None if timeout < 0 else time.monotonic() + timeout
        with self.condition:
            while self.exclusive or self.active:
                remaining = None if deadline is None else deadline - time.monotonic()
                if not blocking or (remaining is not None and remaining <= 0):
                    return False
                self.condition.wait(remaining)
            self.exclusive = True
            return True

    def release(self):
        with self.condition:
            if self.exclusive:
                self.exclusive = False
            elif self.active:
                self.active -= 1
            else:
                raise RuntimeError("Release of unowned admission")
            self.condition.notify_all()

    def locked(self):
        with self.condition:
            return self.exclusive or bool(self.active)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
