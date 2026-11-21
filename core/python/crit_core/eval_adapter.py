"""Core's eval-v1 adapter (crit-eval's docs/ADAPTER.md), the one crit eval loads for every crit engine: `crit eval run
--package DIR` names the package, and create_eval_engine opens its engine through Core's loader (crit_core.engine).
release() closes the host-side stores the engines kept in the process between them (crit_core.engine.eval_release)."""
from crit_core.engine import eval_release, eval_v1


def create_eval_engine(context):
    return eval_v1(context)


def release():
    eval_release()
