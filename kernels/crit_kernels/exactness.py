"""The exactness a kernel structure keeps: what the bits of a row may depend on.

Each structure in kernel.json declares its level ("exactness"), from the most exact:

    declared    the kernel's declared order: the bits every implementation of that order gives (the CPU tier's rows,
                the decode rows and a prompt GEMM in the same order)
    exact       the bits of another structure of the kernel, named by "bits_of" (usually the decode structure): a
                prompt row then has the bits of the same token decoded
    row-exact   an order of its own that follows the shape and the tuning only: a row's bits do not depend on the rows
                beside it, the row count or a prompt chunk's length
    repeatable  the same call gives the same bits again; a row's bits may follow the row count, the chunk's length or
                the tile

A level keeps what every level below it keeps, and none is below repeatable. The faster structures are usually the
lower ones (a split of K that follows the row count, the tensor cores' own order, a reduction over a whole chunk), so
the level is the trade an engine makes between exactness and speed: it asks for one per phase ("decode": its decode
and verify rows; "prompt": its prompt rows), and without one the run's quality mode gives it (for_quality_mode:
both modes ask exact). For the decode rows exact and row-exact are one level: they are the bits a prompt row is
compared with.

The level is about the order of the sums, not the operands' precision: activations quantized to 4 or 8 bits, 8-bit KV
pages or FP16 operands are a structure's arithmetic, which its "what" and its operation say.
"""

LEVELS = ("repeatable", "row-exact", "exact", "declared")   # from the least exact
PHASES = ("decode", "prompt")
# The level each of a run's quality modes asks where an engine states none: exact in both. A prompt row then has the
# bits of the same token decoded, which is what the serving checks hold an engine to (crit smoke: a prefix against
# its prompt, decoding against prefilling), and what they can only judge without a tolerance: an engine whose rows
# multiply quantized activations turns an order's last bit into tenths of a nat (a 4-bit rounding that falls the
# other way), so at a lower level its sound rows and a broken path's read alike. The modes differ in the precision
# of what is summed, not in the order. A lower level is a lever an engine pulls itself, with its speed measured and
# its checks passed.
QUALITY_MODES = {"preserve": "exact", "speed": "exact"}


def rank(level):
    """The level's place in LEVELS (a higher rank keeps more); ValueError for another name."""
    if level not in LEVELS:
        raise ValueError(f"exactness {level!r} is not one of {', '.join(reversed(LEVELS))}")
    return LEVELS.index(level)


def level(spec):
    """A structure's level (spec: its kernel.json entry). A manifest written before the levels says "row_invariant":
    true is row-exact, anything else repeatable."""
    if "exactness" in spec:
        rank(spec["exactness"])
        return spec["exactness"]
    return "row-exact" if spec.get("row_invariant") else "repeatable"


def keeps(spec, asked):
    """Whether the structure keeps the asked level: its own level is that one or a higher one."""
    return rank(level(spec)) >= rank(asked)


def reference(manifest, structure):
    """The structure whose bits `structure` has: the end of its "bits_of" chain, itself when it names none."""
    structures = manifest["structures"]
    seen = [structure]
    while structures[seen[-1]].get("bits_of") is not None:
        target = structures[seen[-1]]["bits_of"]
        if target not in structures or target in seen:
            raise ValueError(f"{manifest.get('id')}/{structure}: bits_of {target!r} "
                             f"{'names itself in a circle' if target in seen else 'is not a structure of the kernel'}")
        seen.append(target)
    return seen[-1]


def same_bits(manifest, a, b):
    """Whether structures a and b of one kernel give a row the same bits where both compute it: both follow the
    declared order, or their bits are one structure's (reference())."""
    structures = manifest["structures"]
    if a == b or reference(manifest, a) == reference(manifest, b):
        return True
    return level(structures[a]) == "declared" and level(structures[b]) == "declared"


def choices(manifest, asked, decode=None):
    """The kernel's structures a phase may run at the asked level, in kernel.json's order. decode: the structure the
    engine's decode rows run, for its prompt phase; at exact and declared only the structures with its bits remain.
    Without it (the decode phase itself) exact is row-exact. The caller picks among them by what each computes."""
    structures = manifest["structures"]
    floor = asked if decode is not None or asked != "exact" else "row-exact"
    out = []
    for sid, spec in structures.items():
        if not keeps(spec, "row-exact" if floor == "exact" else floor):
            continue
        if decode is not None and rank(asked) >= rank("exact") and not same_bits(manifest, sid, decode):
            continue
        out.append(sid)
    return out


def for_quality_mode(mode):
    """The level each phase gets from a run's quality mode when the engine asks for none: exact in both modes
    (QUALITY_MODES)."""
    if mode not in QUALITY_MODES:
        raise ValueError(f"quality mode {mode!r} is not one of {', '.join(QUALITY_MODES)}")
    return {phase: QUALITY_MODES[mode] for phase in PHASES}


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


def problems(manifest, explicit=False):
    """What is wrong with a kernel.json's levels, as a list of sentences (empty: nothing). explicit: every structure
    must say its level itself (crit's own kernels), with no "row_invariant" left."""
    kid, out = manifest.get("id"), []
    structures = manifest.get("structures", {})
    for sid, spec in structures.items():
        where = f"{kid}/{sid}"
        stated = spec.get("exactness")
        if stated is None:
            if explicit:
                out.append(f"{where}: no exactness ({', '.join(reversed(LEVELS))})")
            if "bits_of" in spec:
                out.append(f"{where}: bits_of without exactness exact")
            continue
        if stated not in LEVELS:
            out.append(f"{where}: exactness {stated!r} is not one of {', '.join(reversed(LEVELS))}")
            continue
        if "row_invariant" in spec and (explicit or bool(spec["row_invariant"]) != (rank(stated) >= rank("row-exact"))):
            out.append(f"{where}: row_invariant beside exactness {stated} (the level says it)")
        target = spec.get("bits_of")
        if (stated == "exact") != (target is not None):
            out.append(f"{where}: exactness exact names the structure whose bits it has in bits_of, and no other "
                       "level does")
            continue
        if target is None:
            continue
        if target not in structures or target == sid:
            out.append(f"{where}: bits_of {target!r} is not another structure of the kernel")
            continue
        try:
            root = reference(manifest, sid)
        except ValueError as error:
            out.append(str(error))
            continue
        if structures[root].get("exactness") not in ("row-exact", "declared"):
            out.append(f"{where}: its bits are {root}'s, which is {structures[root].get('exactness')}: an exact "
                       "structure has the bits of a row-exact or declared one")
    return out


def check(manifest, explicit=False):
    """ValueError naming everything problems() finds."""
    found = problems(manifest, explicit)
    if found:
        raise ValueError("; ".join(found))
    return manifest


__all__ = ["LEVELS", "PHASES", "QUALITY_MODES", "check", "choices", "for_quality_mode", "keeps", "level", "problems",
           "rank", "reference", "same_bits", "setting"]
