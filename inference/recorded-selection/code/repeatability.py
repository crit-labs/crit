"""Cross-process repeatability: run the same command in fresh processes and
compare their outputs. A repeated request inside one process cannot detect
nondeterminism that is fixed at process start (timed tuning, allocation
addresses, thread counts), so each run here is a new process.

  python repeatability.py --runs 2 --report report.json -- COMMAND ... {output} ...

`{output}` in the command is replaced by a per-run file path; the command writes
its result there as JSON (for example generated token IDs per case). The report
names the first differing JSON path and, for sequences, the first differing
index. Exit status: 0 identical, 1 different, 2 a run failed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time


def first_difference(a, b, path="$"):
    """First JSON path where a and b differ, with the index for sequences, or None."""
    if type(a) is not type(b):
        return {"path": path, "reason": "type"}
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return {"path": path + "." + str(k), "reason": "missing"}
            d = first_difference(a[k], b[k], path + "." + str(k))
            if d:
                return d
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, "%s[%d]" % (path, i))
            if d:
                return d if "index" in d else dict(d, index=i)  # innermost sequence index
        if len(a) != len(b):
            return {"path": path, "reason": "length", "index": min(len(a), len(b))}
        return None
    return None if a == b else {"path": path, "reason": "value"}


def compare(outputs):
    """Compare every run with the first; returns the report fields."""
    digests = [hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for o in outputs]
    differences = [dict(first_difference(outputs[0], o), run=i) for i, o in enumerate(outputs[1:], 1)
                   if first_difference(outputs[0], o)]
    return {"runs": len(outputs), "identical": not differences, "output_sha256": digests, "differences": differences}


def run(command, runs, timeout):
    if runs < 2:
        raise ValueError("repeatability needs at least two runs")
    if not any("{output}" in part for part in command):
        raise ValueError("the command must contain {output}")
    outputs, durations = [], []
    with tempfile.TemporaryDirectory(prefix="crit-repeat-") as scratch:
        for i in range(runs):
            out = os.path.join(scratch, "run-%d.json" % i)
            argv = [part.replace("{output}", out) for part in command]
            t0 = time.monotonic()
            proc = subprocess.run(argv, timeout=timeout)
            durations.append(round(time.monotonic() - t0, 3))
            if proc.returncode != 0 or not os.path.exists(out):
                return {"runs": i + 1, "identical": False, "failure": "run %d exited %d" % (i, proc.returncode), "seconds": durations}
            with open(out, encoding="utf-8") as f:
                outputs.append(json.load(f))
    return dict(compare(outputs), seconds=durations)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--report", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    report = dict(run(command, a.runs, a.timeout), command=command)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return 2 if "failure" in report else 0 if report["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
