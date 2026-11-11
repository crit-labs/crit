"""How crit's commands report an error: a message for what the user can fix, and for what is a defect in crit (a
missing key, a wrong type, runaway recursion) the place it happened, so it is never mistaken for a user's mistake.
CRIT_DEBUG=1 prints the whole traceback."""
import os
import traceback

DEBUG = "CRIT_DEBUG"
DEFECTS = (KeyError, TypeError, AttributeError, IndexError, RecursionError)


def describe(exc):
    """The text a command prints for EXC: its message, and for a defect its type and crit's innermost frame."""
    if not isinstance(exc, DEFECTS):
        return str(exc)
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if "crit" in f.filename.replace("\\", "/").split("/site-packages/")[-1]] or frames
    where = f" at {os.path.basename(ours[-1].filename)}:{ours[-1].lineno} in {ours[-1].name}" if ours else ""
    return (f"{type(exc).__name__} {exc}{where}; this is a defect in crit or a malformed record "
            f"(set {DEBUG}=1 for the traceback)")


def report(exc, prefix, stream):
    """Print EXC on STREAM after PREFIX, with the traceback when CRIT_DEBUG is set."""
    if os.environ.get(DEBUG) not in (None, "", "0"):
        traceback.print_exception(exc, file=stream)
    print(prefix + describe(exc), file=stream)
