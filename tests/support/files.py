"""Files a test opens for the code under test but whose bytes it must not read."""


class Unreadable:
    """A file whose bytes must not be read (it can still be stat-ed); see unreadable_open."""

    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *details):
        self.handle.close()

    def fileno(self):
        return self.handle.fileno()

    def read(self, *details):
        raise AssertionError("read before the size check")


def unreadable_open(file, mode="r", *details, **options):
    """Patched in for open() (in crit_core.files and crit_core.jsonfile): opens the file but fails any read."""
    return Unreadable(open(file, mode, *details, **options))
