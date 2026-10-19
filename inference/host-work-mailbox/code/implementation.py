"""Host reference of the in-stream host-work mailbox protocol (mailbox.h).

A captured program is a list of operations whose arguments were fixed at capture, as CUDA stream memory
operations are. Replaying it shows why the device, not the host, resets the done word before each doorbell.
"""

import threading

PENDING, SERVED, SKIPPED, FAILED, RELEASED = range(5)
RELEASED_DONE = 0x7FFFFFFF


class Slot:
    def __init__(self):
        self.ready = 0
        self.done = 0
        self.status = PENDING
        self.payload = {}


class Mailbox:
    """Words in mapped memory plus a host service; `serve` is what the service thread runs per doorbell."""

    def __init__(self, slots, handler):
        if slots < 1:
            raise ValueError("at least one slot")
        self.slots = [Slot() for _ in range(slots)]
        self.handler = handler
        self.failed = False
        self.released = False
        self.error = None
        self.lock = threading.Lock()

    # Device side (captured): payload first, then done = 0, then ready = 1.
    def post(self, index, **inputs):
        slot = self.slots[index]
        slot.payload.update(inputs)
        slot.status = PENDING
        slot.done = 0
        slot.ready = 1

    def skip(self, index):
        slot = self.slots[index]
        slot.status = SKIPPED
        slot.done = 1

    # The captured join: done >= 1, a value fixed at capture.
    def joined(self, index):
        return self.slots[index].done >= 1

    def _answer(self, slot, status):
        slot.status = status
        slot.done = RELEASED_DONE if status == RELEASED else 1

    # Host side: claim a raised doorbell and answer it.
    def serve(self, index):
        slot = self.slots[index]
        with self.lock:
            if not slot.ready:
                return False
            slot.ready = 0
            if self.failed:
                self._answer(slot, FAILED)
            elif self.released:
                self._answer(slot, RELEASED)
            else:
                try:
                    self.handler(index, slot.payload)
                except Exception as exc:  # the first failure is terminal
                    self.failed, self.error = True, exc
                    self._answer(slot, FAILED)
                else:
                    self._answer(slot, SERVED)
            return True

    def serve_all(self):
        return sum(self.serve(i) for i in range(len(self.slots)))

    def release(self):
        """Complete every pending and later join; outputs of windows that saw it are invalid."""
        self.released = True
        for slot in self.slots:
            self._answer(slot, RELEASED)

    def rearm(self):
        """Only with every joining stream drained; failure stays terminal."""
        if self.failed:
            return False
        with self.lock:
            self.released = False
            for slot in self.slots:
                slot.ready, slot.done, slot.status = 0, 0, PENDING
        return True



class Stream:
    """A captured program replayed in order; an unsatisfied join stalls it until run() is called again."""

    def __init__(self, program):
        self.program = tuple(program)
        self.pc = len(self.program)

    def launch(self):
        if self.pc != len(self.program):
            raise ValueError("the previous replay has not finished")
        self.pc = 0

    def run(self):
        while self.pc < len(self.program):
            if self.program[self.pc]() is False:
                return False
            self.pc += 1
        return True
