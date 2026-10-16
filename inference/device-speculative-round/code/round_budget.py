"""Whole-round cost check from Core device timings; host reference."""
from dataclasses import dataclass
import math

DEFAULT_THRESHOLD = 1.3


@dataclass(frozen=True)
class DeviceTiming:
    stage: str
    index: int
    duration_ns: int


@dataclass(frozen=True)
class RoundBudgetReport:
    step_ns: float
    round_ns: float
    tokens_per_round: float
    ratio: float
    speedup: float
    threshold: float
    passed: bool


def total_ns(timings):
    total = 0
    for timing in timings:
        value = timing.duration_ns if isinstance(timing, DeviceTiming) else timing
        if not isinstance(value, int) or value < 0:
            raise ValueError("device timings are nonnegative integer nanoseconds")
        total += value
    return total


class RoundBudget:
    """Compare one speculative round with one ordinary decode step.

    A failing ratio means round overhead (host syncs, serial launches, extra
    readbacks, uncaptured work). Fix it before any kernel tuning.
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("positive finite round/step threshold required")
        self.threshold = threshold
        self._step_ns = self._round_ns = 0
        self._steps = self._rounds = self._tokens = 0

    def observe_step(self, timings):
        ns = total_ns(timings)
        if not ns:
            raise ValueError("ordinary step timing must be positive")
        self._step_ns += ns
        self._steps += 1

    def observe_round(self, timings, emitted):
        ns = total_ns(timings)
        if not ns or not isinstance(emitted, int) or emitted < 0:
            raise ValueError("positive round timing and emitted token count required")
        self._round_ns += ns
        self._tokens += emitted
        self._rounds += 1

    def report(self):
        if not self._steps or not self._rounds:
            raise ValueError("observe at least one ordinary step and one round")
        step = self._step_ns / self._steps
        round_ns = self._round_ns / self._rounds
        tokens = self._tokens / self._rounds
        ratio = round_ns / step
        return RoundBudgetReport(step, round_ns, tokens, ratio, tokens / ratio,
                                 self.threshold, ratio <= self.threshold)
