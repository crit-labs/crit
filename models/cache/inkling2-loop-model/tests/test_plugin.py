"""inkling2-loop-model on the host: its manifest and its layout (crit_models.testing.Manifest), without a checkpoint."""
from pathlib import Path
import unittest

from crit_models import testing as t


class ManifestTests(t.Manifest):
    plugin = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    unittest.main()
