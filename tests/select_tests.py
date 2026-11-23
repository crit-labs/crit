"""The test modules a change touches: python tests/select_tests.py [BASE] prints them, and
tests/run_parallel.py --changed [BASE] runs them. For the loop while a change is being written; the whole suite stays
the check of a finished change (CI runs all of it: most of crit's tests reach most of its packages).

The changed files are git's: what differs from BASE in the working tree (HEAD by default, so the uncommitted change;
`main` gives a branch's commits too, from where it left main) and every untracked file. A test module is chosen when

- it is a changed file, or it imports one: its imports are followed through the tests' own modules (tests/support) and
  crit's packages (ROOTS), read from the source without running it, imports inside functions included;
- or its own code or the test modules it imports name a changed file that no import reaches: a plugin's, technique's
  or kernel's files by the name of its folder (the folder with its plugin.json, entry.json or kernel.json), any other
  file by its own name. The library is loaded by path, not imported, so its tests are found by name.

A changed file no test module imports or names is listed as unmatched: nothing here says which tests cover it."""
import argparse
import ast
from pathlib import Path
import subprocess
import sys

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
# The folders crit's packages and the tests' modules import from, as the tests put them on sys.path.
ROOTS = ("tests", "inference/core/python", "models/core/python", "api", "cli", "kernels", "mcp", "eval", "smoke")
# A folder holding one of these is one library item: its files are named by the folder.
ITEMS = ("plugin.json", "entry.json", "kernel.json")


def index(root=ROOT, roots=ROOTS):
    """{module name: path} of every Python module under the roots; a package is its __init__.py."""
    found = {}
    for folder in roots:
        base = Path(root, folder)
        for path in sorted(base.rglob("*.py")) if base.is_dir() else ():
            parts = path.relative_to(base).with_suffix("").parts
            if "__pycache__" in parts:
                continue
            name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            if name:
                found.setdefault(name, path)
    return found


def imports(path, name, modules):
    """The modules of `modules` that the module `name` at `path` imports, wherever in its code, with the packages
    they lie in (importing a.b runs a's __init__ too)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
    asked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            asked.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1] + (node.module.split(".") if node.module else [])
            else:
                base = node.module.split(".")
            asked.add(".".join(base))
            asked.update(".".join(base + [alias.name]) for alias in node.names)   # a submodule, where it is one
    found = set()
    for module in asked:
        parts = module.split(".")
        found.update(".".join(parts[:end]) for end in range(1, len(parts) + 1))
    return found & modules.keys()


def reached(modules, tests=TESTS):
    """{test module: the modules it imports, directly or through others, itself included}, for the test modules of
    the folder 	ests."""
    direct = {}

    def closure(name):
        seen, queue = {name}, [name]
        while queue:
            current = queue.pop()
            if current not in direct:
                direct[current] = imports(modules[current], current, modules)
            for other in direct[current] - seen:
                seen.add(other)
                queue.append(other)
        return seen

    return {name: closure(name) for name, path in modules.items()
            if path.parent == tests and name.startswith("test")}


def item_name(path, root=ROOT):
    """What a changed file that no import reaches is named by: its library item's folder, else its own name."""
    folder = Path(root, path).parent
    while folder != root and root in folder.parents:
        if any((folder / marker).is_file() for marker in ITEMS):
            return folder.name
        folder = folder.parent
    return Path(path).name


def select(changed, root=ROOT, roots=ROOTS):
    """(test modules, unmatched files) for the changed files (paths from the repository's root)."""
    root = Path(root).resolve()
    modules = index(root, roots)
    by_path = {path.resolve(): name for name, path in modules.items()}
    tests = reached(modules, Path(root, "tests"))
    text = {}

    def names(test):
        """The code of a test module and of the tests' modules it imports."""
        if test not in text:
            own = [modules[name] for name in tests[test] if Path(root, "tests") in modules[name].parents]
            text[test] = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in own)
        return text[test]

    chosen, unmatched = set(), []
    for file in sorted(set(changed)):
        module = by_path.get(Path(root, file).resolve())
        found = {test for test, seen in tests.items() if module in seen} if module else set()
        if not found:
            name = item_name(file, Path(root))
            found = {test for test in tests if name in names(test)}
        if found:
            chosen |= found
        else:
            unmatched.append(file)
    return sorted(chosen), unmatched


def changed_files(base="HEAD", root=ROOT):
    """The files that differ from `base` in the working tree (from where HEAD left it, for a branch or a commit
    other than HEAD) and the untracked ones, as git lists them."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, text=True, encoding="utf-8",
                              check=True).stdout.split("\n")
    if base != "HEAD":
        base = git("merge-base", base, "HEAD")[0]
    found = git("diff", "--name-only", base, "--") + git("ls-files", "--others", "--exclude-standard")
    return sorted({line for line in found if line})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", nargs="?", default="HEAD", help="the commit or branch the change is compared with")
    options = parser.parse_args()
    chosen, unmatched = select(changed_files(options.base))
    for file in unmatched:
        print(f"no test module imports or names {file}", file=sys.stderr)
    print(" ".join(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
