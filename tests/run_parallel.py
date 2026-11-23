"""Run the tests in parallel processes: python tests/run_parallel.py [--jobs N] [--shard I/N] [--only MODULE ...] [-v].
Most of the suite waits on served engines and their processes, so a runner's cores run several units at once. A unit is
a test module, or one test class of a module the recorded times split (tests/durations.json), each in its own process;
the short classes of such a module run several a process (grouped).
The two guards CI's lint job runs (LINT) are left out unless --only names them.
The longest units start first; each unit's output prints when it ends, and the exit status fails when any unit fails.
Each unit gets its own engine lock and lease (CRIT_ENGINE_LOCK), so units that load engines do not wait on each other or
on an engine this machine runs outside the tests.

--shard I/N runs the I-th of N shards: every unit goes to the shard with the least recorded time so far, longest unit
first, so the shards end together and every unit runs in exactly one of them; --times FILE lays newer recorded times
over durations.json's (CI: main's last run), and every shard of a run must be given the same ones. --only runs just
the named modules. --changed [BASE] runs the modules the working tree's change touches (tests/select_tests.py: the
loop while writing a change, not the check of a finished one) and --fast SECONDS leaves out the units recorded as
longer. --record FILE writes the time each unit took on this platform, in durations.json's form; --merge FILE ...
folds such records into durations.json."""
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
# The guards CI's lint job runs as scripts on every pull request (python -B tests/test_model_neutral.py): they are no
# host tests, so a run takes them only when --only names them.
LINT = ("test_model_neutral", "test_technique_reads")
# A split module's classes recorded as no longer than SHORT seconds run together, GROUP recorded seconds a process.
SHORT, GROUP = 15.0, 60.0


def recorded(newer=None):
    """{platform: {unit: seconds}} from durations.json, empty where it is missing, under the times of the file
    `newer` where it exists (--times: what main's last run recorded)."""
    try:
        times = json.loads(DURATIONS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        times = {}
    if newer is not None and Path(newer).is_file():
        for platform, table in json.loads(Path(newer).read_text(encoding="utf-8")).items():
            times.setdefault(platform, {}).update(table)
    return times


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


def grouped(chosen):
    """The units with a split module's short classes (SHORT) joined, in their order, into units of several classes
    (their names with spaces between, up to GROUP recorded seconds): a class that starts a process to run for a
    fraction of a second, or to skip where there is no GPU, costs the process's start and its imports, seconds each.
    Every shard joins them alike from the same recorded times."""
    found, open_groups = [], {}
    for unit, seconds in chosen:
        module, _, name = unit.partition(".")
        if not name or seconds > SHORT:
            found.append([[unit], seconds])
            continue
        group = open_groups.get(module)
        if group is None or group[1] + seconds > GROUP:
            group = open_groups[module] = [[], 0.0]
            found.append(group)
        group[0].append(unit)
        group[1] += seconds
    return [(" ".join(names), seconds) for names, seconds in found]


def label(unit):
    """A unit's name in the output: its class or module, or its module and how many classes it joins."""
    names = unit.split()
    return unit if len(names) == 1 else f"{names[0].partition('.')[0]} ({len(names)} short classes: {', '.join(n.partition('.')[2] for n in names)})"


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
    names = unit.split()   # one class or module, or the short classes joined into one process (grouped)
    environment = {**os.environ, "CRIT_ENGINE_LOCK": str(Path(locks) / f"{names[0]}+{len(names) - 1}.lock")}
    environment.pop("CRIT_ENGINE_LEASE", None)   # the lease lives next to the unit's lock
    environment.pop("CRIT_ENGINE_LOCK_HELD", None)
    # One math thread per process unless the caller chose: torch starts a thread per core in every unit and in every
    # process a unit starts (plugin tests, resident hosts), so units side by side fight over the cores, and the tests'
    # tiny models gain nothing from threads.
    environment.setdefault("OMP_NUM_THREADS", "1")
    # The checkout's crit Models (models/core/python), which the model plugin, crit models and Eval's limits import,
    # for every process a unit starts as for the unit (a checkout without its editable install).
    environment["PYTHONPATH"] = os.pathsep.join([str(TESTS.parent / "models" / "core" / "python")]
                                                + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
    result = subprocess.run([sys.executable, "-B", "-m", "unittest", *(["-v"] if verbose else []), *names], cwd=TESTS,
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
    parser.add_argument("--changed", nargs="?", const="HEAD", metavar="BASE",
                        help="run only the test modules the change touches: the working tree against BASE (HEAD; a "
                             "branch name adds this branch's commits since it left that branch)")
    parser.add_argument("--fast", type=float, metavar="SECONDS",
                        help="leave out the units whose recorded time is over SECONDS, and list them")
    parser.add_argument("--times", type=Path, metavar="FILE",
                        help="recorded times newer than durations.json's, in its form (a missing file is none)")
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
    modules = sorted(path.stem for path in TESTS.glob("test*.py") if path.stem not in LINT)
    if options.only is not None:
        unknown = sorted(set(options.only) - set(modules) - set(LINT))
        if unknown:
            parser.error(f"--only: no test module {', '.join(unknown)}")
        modules = sorted(set(options.only))
    if options.changed:
        import select_tests
        touched, unmatched = select_tests.select(select_tests.changed_files(options.changed))
        for file in unmatched:
            print(f"no test module imports or names {file}", flush=True)
        modules = sorted(set(modules) & set(touched))
        print(f"--changed {options.changed}: {len(modules)} test modules" + (": " + " ".join(modules) if modules else ""),
              flush=True)
    chosen = units(modules, recorded(options.times))
    if options.fast is not None:
        slow = sorted(unit for unit, seconds in chosen if seconds > options.fast)
        chosen = [(unit, seconds) for unit, seconds in chosen if seconds <= options.fast]
        if slow:
            print(f"--fast {options.fast:g}: left out {', '.join(slow)}", flush=True)
    chosen = shard(grouped(chosen), index, count)
    start, failed, took = time.monotonic(), [], {}
    with tempfile.TemporaryDirectory(prefix="crit-test-locks-") as locks, ThreadPoolExecutor(options.jobs) as pool:
        for future in as_completed([pool.submit(run, unit, options.verbose, locks) for unit, _ in chosen]):
            unit, code, output, seconds = future.result()
            for name in unit.split():   # joined classes share their process's time
                took[name] = round(seconds / len(unit.split()), 1)
            print(f"===== {label(unit)}: {'ok' if code == 0 else f'FAILED (exit code {code}' + (f' = {code & 0xFFFFFFFF:#x}' if code < 0 or code > 255 else '') + ')'} in {seconds:.0f} s", flush=True)
            if code != 0 or options.verbose:
                print(output, flush=True)
            if code != 0:
                failed.append(label(unit))
    print(f"{len(chosen)} units (shard {index}/{count}, {sum(s for _, s in chosen):.0f} s recorded) in "
          f"{time.monotonic() - start:.0f} s with {options.jobs} jobs"
          + (f"; failed: {', '.join(sorted(failed))}" if failed else ""))
    if options.record:
        options.record.write_text(json.dumps({PLATFORM: dict(sorted(took.items()))}, indent=1) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
