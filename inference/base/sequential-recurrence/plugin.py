"""The gated-delta recurrence in the recurrence slot, token by token: recurrent-state-recovery's reference
(chunked_prefill.sequential over reference.py) as the host form; on the device, every sequence's state and convolution
window in a pool (device_states.py), decode tokens and verify windows through the column scan with a device commit of
their accepted prefix (crit's kernel recurrent-scans' pool rows where it builds, the torch formula otherwise), prompt
chunks through the same scan up to exact_rows tokens, the configuration's (default 512), and through recurrent-scans'
chunked prefill past it (on tensor cores or in FP32, the prefill option). Option fla (default off, a yardstick): fla's
chunked kernel instead where fla is installed, each of its Triton kernels on a stated tile (device_states.FLA_TILES)
rather than the one fla's autotune times best."""
from pathlib import Path

from .recurrence import chunked_prefill as recurrence
from .recurrence import device_states
from .recurrence.slot import RecurrenceSlot, check_state, geometry_of, steps_of

HERE = Path(__file__).resolve().parent


class SequentialRecurrence(RecurrenceSlot):
    """The device form is recurrent-state-recovery's (slot.RecurrenceSlot); the host form the token-by-token
    recurrence."""
    prompt = "two_pass"   # prompt rows: recurrent-scans' chunked prefill (fla's with option fla), else the scan

    def __init__(self, exact_rows=device_states.DEFAULT_EXACT_ROWS, chunk_fp32=True, commit="deferred", kernels=True,
                 folder=HERE, prefill="tensor_cores", scan="pool-delta-pipe"):
        super().__init__("sequential-recurrence", folder, exact_rows, commit, kernels, prefill, scan, chunk_fp32)

    def prefill(self, state, steps, geometry):
        geometry = geometry_of(geometry)
        outputs, state = recurrence.sequential(check_state(state, geometry), steps_of(steps), geometry)
        return [list(o) for o in outputs], tuple(state)


def create(context, options, parts):
    found = SequentialRecurrence(options["exact_rows"], options["chunk_fp32"], options["commit"], options["kernels"],
                                 prefill=options["prefill"], scan=options["scan"])
    if options.get("fla"):
        found.prompt = "fla"   # the yardstick: fla's chunked kernel where it is installed
    found.elastic = bool(options["elastic"])
    # prompt_rows: scan gives every prompt row the scan; auto follows the level the engine asks of its prompt rows,
    # but keeps fla where the option names it
    found.prompt = device_states.asked_prompt(options["prompt_rows"], found.prompt, pinned=bool(options.get("fla")))
    return found
