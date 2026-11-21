"""Speculative recovery of recurrent state, with a two-pass chunked prefill, in the recurrence slot: the host form is
recurrent-state-recovery's two-pass chunked gated-delta prefill (chunked_prefill.two_pass); on the device every
sequence's state and convolution window in a pool (device_states.py) whose verify rows keep their accepted prefix by the
technique's strategy (deferred replay, per-column checkpoints or recompute, chosen by `commit`), prompt chunks through
the column scan up to exact_rows tokens, the configuration's (default 512), and through recurrent-scans' chunked
prefill past it (crit's kernel, built for the engine's device: on tensor cores or in FP32, the prefill option), the
scan alone where the kernel does not build."""
from pathlib import Path

from .recurrence import chunked_prefill as recurrence
from .recurrence import device_states
from .recurrence.slot import RecurrenceSlot, check_state, geometry_of, steps_of

HERE = Path(__file__).resolve().parent


class ChunkedRecurrence(RecurrenceSlot):
    """The device form (slot.RecurrenceSlot) with the commit strategy of the `commit` option; the host form the
    two-pass chunked prefill."""

    def __init__(self, chunk_size, exact_rows=device_states.DEFAULT_EXACT_ROWS, commit="deferred", kernels=True,
                 folder=HERE, prefill="tensor_cores", scan="pool-delta-pipe", decode="scan"):
        super().__init__("recurrent-state-recovery", folder, exact_rows, commit, kernels, prefill, scan, decode=decode)
        self.chunk_size = chunk_size

    def prefill(self, state, steps, geometry):
        geometry = geometry_of(geometry)
        state = check_state(state, geometry)
        if not steps:
            return [], state
        outputs, state = recurrence.two_pass(state, steps_of(steps), geometry, self.chunk_size)
        return [list(o) for o in outputs], tuple(state)


def create(context, options, parts):
    found = ChunkedRecurrence(options["chunk_size"], options["exact_rows"], options["commit"], options["kernels"],
                              prefill=options["prefill"], scan=options["scan"], decode=options.get("decode", "scan"))
    found.elastic = bool(options["elastic"])
    # prompt_rows: scan gives every prompt row the scan; auto follows the level the engine asks of its prompt rows
    found.prompt = device_states.asked_prompt(options["prompt_rows"], found.prompt)
    return found
