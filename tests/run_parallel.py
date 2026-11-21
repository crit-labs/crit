"""Run the tests in parallel processes: python tests/run_parallel.py [--jobs N] [--shard I/N] [--only MODULE ...] [-v].
Most of the suite waits on served engines and their processes, so a runner's cores run several units at once. A unit is
a test module, or one test class of a module the recorded times split (tests/durations.json), each in its own process.
The longest units start first; each unit's output prints when it ends, and the exit status fails when any unit fails.
Each unit gets its own engine lock and lease (CRIT_ENGINE_LOCK), so units that load engines do not wait on each other or
on an engine this machine runs outside the tests.

--shard I/N runs the I-th of N shards: every unit goes to the shard with the least recorded time so far, longest unit
first, so the shards end together and every unit runs in exactly one of them. --only runs just the named modules (CI's
selection for a pull request, tests/select_tests.py). --record FILE writes the time each unit took on this platform,
in durations.json's form; --merge FILE ... folds such records into durations.json."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

TESTS = Path(__file__).resolve().parent
DURATIONS = TESTS / "durations.json"
PLATFORM = "windows" if os.name == "nt" else "linux"
# A unit with no recorded time (a new module or class) counts as this many seconds when shards are filled.
UNRECORDED = 20.0


def recorded():
    """{platform: {unit: seconds}} from durations.json, empty where it is missing."""
    try:
        return json.loads(DURATIONS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def classes(modules):
    """{module: [module.Class, ...]}: the test classes each module's tests load from, as unittest runs them."""
    if not modules:
        return {}
    script = ("import json, sys, unittest\n"
              "found = {}\n"
              "for name in sys.argv[1:]:\n"
              "    names = []\n"
              "    def walk(suite):\n"
              "        for test in suite:\n"
              "            if isinstance(test, unittest.TestSuite):\n"
              "                walk(test)\n"
              "            elif type(test).__name__ not in names:\n"
              "                names.append(type(test).__name__)\n"
              "    walk(unittest.defaultTestLoader.loadTestsFromName(name))\n"
              "    found[name] = [name + '.' + test for test in names]\n"
              "print(json.dumps(found))\n")
    result = subprocess.run([sys.executable, "-B", "-c", script, *modules], cwd=TESTS, stdout=subprocess.PIPE,
                            text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout.splitlines()[-1])


def units(modules, times):
    """[(unit, seconds)]: each module whole, or its classes where the recorded times name them."""
    split = sorted({unit.partition(".")[0] for table in times.values() for unit in table if "." in unit} & set(modules))
    found = classes(split)
    own = times.get(PLATFORM) or next(iter(times.values()), {})
    other = [table for name, table in times.items() if name != PLATFORM]

    def seconds(unit):
        if unit in own:
            return own[unit]
        for table in other:
            if unit in table:
                return table[unit]
        return UNRECORDED

    chosen = [unit for module in modules for unit in found.get(module) or [module]]
    return [(unit, seconds(unit)) for unit in chosen]


def shard(chosen, index, count):
    """The units of shard index (1-based) of count, filling the shard with the least time first."""
    loads, shards = [0.0] * count, [[] for _ in range(count)]
    for unit, seconds in sorted(chosen, key=lambda item: (-item[1], item[0])):
        least = min(range(count), key=lambda number: (loads[number], number))
        loads[least] += seconds
        shards[least].append((unit, seconds))
    return shards[index - 1]


def run(unit, verbose, locks):
    start = time.monotonic()
    environment = {**os.environ, "CRIT_ENGINE_LOCK": str(Path(locks) / f"{unit}.lock")}
    environment.pop("CRIT_ENGINE_LEASE", None)   # the lease lives next to the unit's lock
    environment.pop("CRIT_ENGINE_LOCK_HELD", None)
    result = subprocess.run([sys.executable, "-B", "-m", "unittest", *(["-v"] if verbose else []), unit], cwd=TESTS,
                            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    return unit, result.returncode, result.stdout, time.monotonic() - start


def merge(records):
    """Fold --record files into durations.json: each platform's units take their newest times."""
    times = recorded()
    for record in records:
        for platform, table in json.loads(Path(record).read_text(encoding="utf-8")).items():
            times.setdefault(platform, {}).update(table)
    ordered = {platform: dict(sorted(table.items())) for platform, table in sorted(times.items())}
    DURATIONS.write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--shard", default="1/1", help="I/N: run the I-th of N shards")
    parser.add_argument("--only", nargs="*", help="run only these test modules")
    parser.add_argument("--record", type=Path, help="write each unit's time on this platform here")
    parser.add_argument("--merge", nargs="+", type=Path, help="fold --record files into durations.json, run nothing")
    parser.add_argument("-v", "--verbose", action="store_true")
    options = parser.parse_args()
    if options.merge:
        merge(options.merge)
        return 0
    index, count = map(int, options.shard.split("/"))
    if not 1 <= index <= count:
        parser.error(f"--shard {options.shard}: I must be 1 to N")
    modules = sorted(path.stem for path in TESTS.glob("test*.py"))
    if options.only is not None:
        unknown = sorted(set(options.only) - set(modules))
        if unknown:
            parser.error(f"--only: no test module {', '.join(unknown)}")
        modules = sorted(set(options.only))
    chosen = shard(units(modules, recorded()), index, count)
    start, failed, took = time.monotonic(), [], {}
    with tempfile.TemporaryDirectory(prefix="crit-test-locks-") as locks, ThreadPoolExecutor(options.jobs) as pool:
        for future in as_completed([pool.submit(run, unit, options.verbose, locks) for unit, _ in chosen]):
            unit, code, output, seconds = future.result()
            took[unit] = round(seconds, 1)
            print(f"===== {unit}: {'ok' if code == 0 else f'FAILED (exit code {code}' + (f' = {code & 0xFFFFFFFF:#x}' if code < 0 or code > 255 else '') + ')'} in {seconds:.0f} s", flush=True)
            if code != 0 or options.verbose:
                print(output, flush=True)
            if code != 0:
                failed.append(unit)
    print(f"{len(chosen)} units (shard {index}/{count}, {sum(s for _, s in chosen):.0f} s recorded) in "
          f"{time.monotonic() - start:.0f} s with {options.jobs} jobs"
          + (f"; failed: {', '.join(sorted(failed))}" if failed else ""))
    if options.record:
        options.record.write_text(json.dumps({PLATFORM: dict(sorted(took.items()))}, indent=1) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
