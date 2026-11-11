"""crit core: build Core's native runtime, the crit_core_steps library every engine runs on (crit_core.native), where
Core's loader finds it without CRIT_CORE_STEP_LIBRARY (crit_core.native.step_library): in a crit checkout its
build/core-steps; for a crit installed from its wheel, which carries no C++, Core's source at the installed release's
tag (crit X.Y.Z is published from vX.Y.Z), fetched once into Core's cache and built there."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BUILD = "build/core-steps"


def library(folder):
    """The crit_core_steps file a build folder holds, or None."""
    from crit_core.native import STEP_LIBRARY_FILES
    for sub in (folder, folder / "Release"):
        for name in STEP_LIBRARY_FILES:
            if (sub / name).is_file():
                return sub / name
    return None


def source():
    """(Core's source folder, the build folder): this checkout's, else the installed release's in Core's cache,
    fetched with git at the release's tag when it is not there yet."""
    if (ROOT / "core" / "CMakeLists.txt").is_file():
        return ROOT / "core", ROOT / BUILD
    from crit_core.native import step_cache
    from . import __version__
    from .cli import REPOSITORY
    cache = step_cache()
    checkout = cache / "source"
    if not (checkout / "core" / "CMakeLists.txt").is_file():
        git = shutil.which("git")
        if git is None:
            raise ValueError("crit core build fetches Core's source with git, which is not on PATH")
        cache.mkdir(parents=True, exist_ok=True)
        print(f"crit core: fetching Core's source, {REPOSITORY} at v{__version__}", file=sys.stderr, flush=True)
        if subprocess.run([git, "clone", "--quiet", "--depth", "1", "--branch", f"v{__version__}", "--",
                           REPOSITORY, str(checkout)]).returncode:
            raise ValueError(f"could not fetch {REPOSITORY} at v{__version__}")
    return checkout / "core", cache / "build"


def build(build_type="Release", jobs=None):
    """Configure and build the crit_core_steps target (source()); returns the library's path. ValueError without
    CMake or git, or when the build fails (CMake's output is on the terminal)."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise ValueError("crit core build needs CMake 3.28 or newer and a C++20 compiler on PATH")
    core, folder = source()
    steps = [[cmake, "-S", str(core), "-B", str(folder), f"-DCMAKE_BUILD_TYPE={build_type}"],
             [cmake, "--build", str(folder), "--config", build_type, "--target", "crit_core_steps",
              "--parallel", *([str(jobs)] if jobs else [])]]
    for step in steps:
        if subprocess.run(step).returncode:
            raise ValueError("Core's build failed: " + " ".join(step))
    built = library(folder)
    if built is None:
        raise ValueError(f"Core's build finished without a crit_core_steps library in {folder}")
    return built


def parser():
    result = argparse.ArgumentParser(prog="crit core", description=__doc__)
    subs = result.add_subparsers(dest="command", required=True)
    sub = subs.add_parser("build", help="Build crit_core_steps where Core's loader finds it (a checkout's build/core-steps, else Core's cache)")
    sub.add_argument("--build-type", default="Release", help="CMake build type (default Release)")
    sub.add_argument("--jobs", type=int, help="Parallel build jobs (default: CMake's)")
    subs.add_parser("where", help="Print the crit_core_steps library Core's loader would load, or exit 1")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        import crit_core  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(ROOT / "core" / "python"))
    if args.command == "where":
        from crit_core.native import step_library
        found = step_library()
        if not found:
            print("crit_core_steps is not built: run crit core build", file=sys.stderr)
            return 1
        print(found)
        return 0
    try:
        built = build(args.build_type, args.jobs)
    except ValueError as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    print(built)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
