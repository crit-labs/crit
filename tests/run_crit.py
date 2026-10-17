"""Test launcher: run the crit command from this checkout without installing it."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in ("core/python", "api", "eval", "kernels", "models", "cli"):
    sys.path.insert(0, str(ROOT / folder))
from crit.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
