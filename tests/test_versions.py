"""crit's version, written wherever a package, a build or the docs need it: every place agrees with crit's own
(cli/crit/__init__.py), and every file naming that version is one of these places, so a release changes them all."""
import json
from pathlib import Path
import re
import subprocess
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"([0-9]+\.[0-9]+\.[0-9]+)"
# Each version-bearing file and the patterns its version is read with (a path's own reader where it has one).
PLACES = {
    "cli/crit/__init__.py": [r'^__version__ = "' + NUMBER + '"'],
    "core/python/crit_core/__init__.py": [r'^__version__ = "' + NUMBER + '"'],
    "pyproject.toml": [lambda text: tomllib.loads(text)["project"]["version"], r'"crit-core==' + NUMBER + '"'],
    "core/python/pyproject.toml": [lambda text: tomllib.loads(text)["project"]["version"]],
    "core/CMakeLists.txt": [r"project\(crit_core VERSION " + NUMBER],
    "core/contracts/CONTRACTS.json": [lambda text: json.loads(text)["version"]],
    "core/contracts/README.md": [r"the current source version is " + NUMBER],
    "core/README.md": [r"The SDK is version " + NUMBER, r"find_package\(critCore " + NUMBER + " EXACT"],
    "inference/CMakeLists.txt": [r"project\(crit_inference VERSION " + NUMBER, r"find_package\(critCore " + NUMBER],
    "inference/recurrent-state-recovery/code/CMakeLists.txt": [r"find_package\(critCore " + NUMBER],
    "tests/core/sdk_consumer/CMakeLists.txt": [r"find_package\(critCore " + NUMBER + " EXACT"],
    "tests/core/concurrency_consumer/CMakeLists.txt": [r"find_package\(critCore " + NUMBER + " EXACT"],
    "tests/core/concurrency_consumer/main.cpp": [r"crit " + NUMBER + " installed SDK"],
    "README.md": [r"\[!\[Version " + NUMBER + r"\]"],
    "docs/CONTRIBUTING.md": [r"crit-" + NUMBER + r"-py3-none-any\.whl"],
    "docs/assets/badge-version.svg": [r'aria-label="version: ' + NUMBER + '"', r"<title>version: " + NUMBER + "</title>",
                                      r'<text x="75\.5" y="14">' + NUMBER + "</text>"],
}
# Files that name earlier versions too: the changelog's history.
HISTORY = {"CHANGELOG.md"}


def read(path, reader):
    text = (ROOT / path).read_text(encoding="utf-8")
    if callable(reader):
        return reader(text)
    found = re.search(reader, text, re.M)
    return found[1] if found else None


class VersionTests(unittest.TestCase):
    def test_every_place_names_crits_version(self):
        version = read("cli/crit/__init__.py", PLACES["cli/crit/__init__.py"][0])
        for path, readers in PLACES.items():
            for reader in readers:
                with self.subTest(path=path, reader=reader if isinstance(reader, str) else "reader"):
                    self.assertEqual(read(path, reader), version)

    def test_every_file_that_names_the_version_is_a_known_place(self):
        version = read("cli/crit/__init__.py", PLACES["cli/crit/__init__.py"][0])
        listed = subprocess.run(["git", "grep", "-l", "-F", version], cwd=ROOT, capture_output=True, text=True)
        if listed.returncode not in (0, 1):
            self.skipTest("needs a git checkout: " + listed.stderr.strip())
        named = set(listed.stdout.split()) - HISTORY
        self.assertEqual(named - set(PLACES), set(), "add these to PLACES, which a release changes")


if __name__ == "__main__":
    unittest.main()
