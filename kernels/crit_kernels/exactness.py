"""The exactness a kernel structure keeps: what the bits of a row may depend on.

Each structure in kernel.json declares its level ("exactness"), on Core's scale (crit_plugins.exactness, from the most
exact):

    declared    the kernel's declared order: the bits every implementation of that order gives (the CPU tier's rows,
                the decode rows and a prompt GEMM in the same order)
    exact       the bits of another structure of the kernel, named by "bits_of" (usually the decode structure): a
                prompt row then has the bits of the same token decoded
    row-exact   an order of its own that follows the shape and the tuning only: a row's bits do not depend on the rows
                beside it, the row count or a prompt chunk's length
    repeatable  the same call gives the same bits again; a row's bits may follow the row count, the chunk's length or
                the tile

The faster structures are usually the lower ones (a split of K that follows the row count, the tensor cores' own
order, a reduction over a whole chunk), so the level is the trade an engine makes between exactness and speed: it asks
for one per phase, and without one the run's quality mode gives it. The levels, the phases, the quality modes and an
engine's setting are Core's (crit_plugins.exactness, imported here as they were defined here before); this module
reads a kernel.json against them: a structure's level, whether it keeps a level, whose bits it has and the structures
a phase may run.
"""
from crit_plugins.exactness import LEVELS, MODE_LEVELS, PHASES, for_quality_mode, rank, setting

QUALITY_MODES = MODE_LEVELS   # the mode -> level table, under the name it had here


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


__all__ = ["LEVELS", "MODE_LEVELS", "PHASES", "QUALITY_MODES", "check", "choices", "for_quality_mode", "keeps", "level",
           "problems", "rank", "reference", "same_bits", "setting"]
