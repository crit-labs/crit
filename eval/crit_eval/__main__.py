import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from crit_eval.cli import main
else:
    from .cli import main

raise SystemExit(main())
