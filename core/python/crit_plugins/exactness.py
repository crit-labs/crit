"""The exactness of an engine's rows and the quality modes: Core's levels, which everything above Core imports.

An engine says how exact its rows are, per phase ("decode": its decode and verify rows; "prompt": its prompt rows),
on one scale of levels, from the most exact:

    declared    the kernel's declared order: the bits every implementation of that order gives (the CPU tier's rows,
                the decode rows and a prompt GEMM in the same order)
    exact       the bits of another computation of the same rows (a kernel structure's "bits_of", usually the decode
                one): a prompt row then has the bits of the same token decoded
    row-exact   an order of its own that follows the shape and the tuning only: a row's bits do not depend on the rows
                beside it, the row count or a prompt chunk's length
    repeatable  the same call gives the same bits again; a row's bits may follow the row count, the chunk's length or
                the tile

A level keeps what every level below it keeps, and none is below repeatable. For the decode rows exact and row-exact
are one level: they are the bits a prompt row is compared with, so a decode row's exact is written row-exact. The
level is about the order of the sums, not the operands' precision: activations quantized to 4 or 8 bits, 8-bit KV
pages or FP16 operands are arithmetic, which the quality mode sets.

The quality mode is the trade a run makes between quality and speed (speed: the stored format's activations, FP8 KV
and an FP8 head are the starting arithmetic; preserve: each precision change is measured first). Where an engine
states no level, its quality mode gives one (for_quality_mode): exact in both modes. A prompt row then has the bits
of the same token decoded, which is what the serving checks hold an engine to (crit smoke: a prefix against its
prompt, decoding against prefilling), and what they can only judge without a tolerance: an engine whose rows multiply
quantized activations turns an order's last bit into tenths of a nat (a 4-bit rounding that falls the other way), so
at a lower level its sound rows and a broken path's read alike. The modes differ in the precision of what is summed,
not in the order. A lower level is a lever an engine pulls itself, with its speed measured and its checks passed.

crit_kernels reads a kernel.json's structures against these levels (crit_kernels.exactness: a structure's level, the
structures a phase may run); the levels themselves are Core's (core/README.md#what-core-decides).
"""

LEVELS = ("repeatable", "row-exact", "exact", "declared")   # from the least exact
PHASES = ("decode", "prompt")
QUALITY_MODES = ("preserve", "speed")
# The level each quality mode asks where an engine states none (the module's description: exact in both).
MODE_LEVELS = {"preserve": "exact", "speed": "exact"}


def rank(level):
    """The level's place in LEVELS (a higher rank keeps more); ValueError for another name."""
    if level not in LEVELS:
        raise ValueError(f"exactness {level!r} is not one of {', '.join(reversed(LEVELS))}")
    return LEVELS.index(level)


def phase_rank(phase, level):
    """The level's rank for a phase's rows: for the decode rows exact is row-exact (the one level it is there)."""
    return rank("row-exact" if phase == "decode" and level == "exact" else level)


def keeps(phase, level, asked):
    """Whether rows of `level` keep the `asked` level in the phase: their level is that one or a higher one."""
    return phase_rank(phase, level) >= phase_rank(phase, asked)


def check_mode(mode):
    """The quality mode, or ValueError where it is not one of QUALITY_MODES."""
    if mode not in QUALITY_MODES:
        raise ValueError(f"quality mode {mode!r} is not one of {', '.join(QUALITY_MODES)}")
    return mode


def for_quality_mode(mode):
    """The level each phase gets from a run's quality mode when the engine asks for none: exact in both modes
    (MODE_LEVELS)."""
    check_mode(mode)
    return {phase: MODE_LEVELS[mode] for phase in PHASES}


def setting(value=None, quality_mode=None):
    """An engine's exactness setting as {"decode": level, "prompt": level}. value: one level for both phases, or a
    level per phase; a phase it leaves out takes the quality mode's (for_quality_mode), which must then be given.
    The decode rows' exact is written row-exact, the one level it is there."""
    if isinstance(value, str):
        value = {phase: value for phase in PHASES}
    value = dict(value or {})
    unknown = set(value) - set(PHASES)
    if unknown:
        raise ValueError(f"exactness is set per phase ({', '.join(PHASES)}), not {', '.join(sorted(unknown))}")
    out = {}
    for phase in PHASES:
        if value.get(phase) is None:
            if quality_mode is None:
                raise ValueError(f"no exactness for the {phase} phase and no quality mode to take it from")
            out[phase] = for_quality_mode(quality_mode)[phase]
        else:
            rank(value[phase])
            out[phase] = value[phase]
    if out["decode"] == "exact":
        out["decode"] = "row-exact"
    return out


__all__ = ["LEVELS", "MODE_LEVELS", "PHASES", "QUALITY_MODES", "check_mode", "for_quality_mode", "keeps",
           "phase_rank", "rank", "setting"]
