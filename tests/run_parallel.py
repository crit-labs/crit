"""Run the test modules in parallel processes, one module per process: python tests/run_parallel.py [--jobs N] [-v].
Most of the suite waits on served engines and their processes, so a runner's cores run several modules at once. The
largest modules start first; each module's output prints when it ends, and the exit status fails when any module
fails. Each module gets its own engine lock and lease (CRIT_ENGINE_LOCK), so modules that load engines do not wait on
each other or on an engine this machine runs outside the tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

TESTS = Path(__file__).resolve().parent


def run(module, verbose, locks):
    start = time.monotonic()
    environment = {**os.environ, "CRIT_ENGINE_LOCK": str(Path(locks) / f"{module}.lock")}
    environment.pop("CRIT_ENGINE_LEASE", None)   # the lease lives next to the module's lock
    environment.pop("CRIT_ENGINE_LOCK_HELD", None)
    result = subprocess.run([sys.executable, "-B", "-m", "unittest", *(["-v"] if verbose else []), module], cwd=TESTS,
                            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    return module, result.returncode, result.stdout, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("-v", "--verbose", action="store_true")
    options = parser.parse_args()
    modules = sorted(TESTS.glob("test*.py"), key=lambda path: -path.stat().st_size)
    start, failed = time.monotonic(), []
    with tempfile.TemporaryDirectory(prefix="crit-test-locks-") as locks, ThreadPoolExecutor(options.jobs) as pool:
        for future in as_completed([pool.submit(run, path.stem, options.verbose, locks) for path in modules]):
            module, code, output, seconds = future.result()
            print(f"===== {module}: {'ok' if code == 0 else 'FAILED'} in {seconds:.0f} s", flush=True)
            if code != 0 or options.verbose:
                print(output, flush=True)
            if code != 0:
                failed.append(module)
    print(f"{len(modules)} modules in {time.monotonic() - start:.0f} s with {options.jobs} jobs"
          + (f"; failed: {', '.join(sorted(failed))}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
