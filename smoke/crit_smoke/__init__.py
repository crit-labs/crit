"""crit smoke: an engine's quality components checked in seconds against crit serve (smoke/README.md)."""
from .client import Client, Unsupported
from .suite import TESTS, run

__all__ = ["Client", "TESTS", "Unsupported", "run"]
