"""The preflight's records: what a report is keyed by, its files, and the newest one a run keeps.

A report is keyed by the code it ran (crit_key: the trees of what an engine is built from, with a digest of their
uncommitted changes) and the checkpoint it cut (checkpoint_revision), so a commit that changes no engine code, or a
change to the docs and tests, keeps it, and a new commit of the library or another download of the model asks for a
new one. A run keeps its reports in RUN/preflight/KEY/ (preflight.json and preflight.md: a scenario a row), the cut,
its candidate and the plugin cache in the run's .plugin-cache/preflight/, which crit run stop removes with the rest.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from ..cli import ROOT, checkout

FOLDER = "preflight"      # RUN/preflight/KEY/: the run's reports
CACHE = "preflight"       # RUN/.plugin-cache/preflight/: the cut, its runs and the plugin cache, rebuilt at will
REPORT = "preflight"      # preflight.json and preflight.md
# What an engine is built from: a change outside these (the docs, the tests, the harnesses) keeps a report.
BUILT = ("inference", "cli", "api", "smoke", "kernels", "models", "pyproject.toml")
STATES = ("passed", "failed", "skipped", "info")


def git(folder, *args):
    """git in `folder`, also where it is a worktree whose .git names a Windows path seen from WSL (C:/... ->
    /mnt/c/...)."""
    command = ["git", "-c", "safe.directory=*", *args]
    found = subprocess.run(command, cwd=folder, capture_output=True)
    if found.returncode and (Path(folder) / ".git").is_file():
        text = (Path(folder) / ".git").read_text(encoding="utf-8").strip()
        gitdir = text.split(":", 1)[1].strip() if text.startswith("gitdir:") else ""
        if len(gitdir) > 2 and gitdir[1] == ":":
            gitdir = "/mnt/" + gitdir[0].lower() + gitdir[2:]
            common = (Path(gitdir) / "commondir").read_text(encoding="utf-8").strip()
            env = {**os.environ, "GIT_DIR": gitdir, "GIT_COMMON_DIR": str((Path(gitdir) / common).resolve()),
                   "GIT_WORK_TREE": str(folder)}
            found = subprocess.run(command, cwd=folder, capture_output=True, env=env)
    if found.returncode:
        raise RuntimeError(f"git {' '.join(args)} in {folder}: {found.stderr.decode()[:300]}")
    return found.stdout


def crit_key(root=ROOT, built=BUILT):
    """The key of the code a load runs: the commit's trees of `built` in crit's checkout, with a digest of their
    uncommitted changes; an installed crit's version where crit runs without a checkout."""
    if not checkout(root):
        from .. import __version__
        return "crit-" + __version__
    present = [path for path in built if (Path(root) / path).exists()]
    trees = git(root, "rev-parse", *[f"HEAD:{path}" for path in present])
    diff = git(root, "diff", "HEAD", "--", *present)
    head = hashlib.sha256(trees).hexdigest()[:12]
    return head + ("-dirty-" + hashlib.sha256(diff).hexdigest()[:8] if diff else "")


def crit_commit(root=ROOT):
    """crit's commit (12 characters), or its version without a checkout."""
    if not checkout(root):
        from .. import __version__
        return __version__
    return git(root, "rev-parse", "--short=12", "HEAD").decode().strip()


def checkpoint_revision(model):
    """The checkpoint's revision: a digest of its config.json, its index and every weight file's name and size (its
    headers name the tensors; their bytes are the run's resolve to hash). A new download of the model changes it."""
    model = Path(model)
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "hf_quant_config.json"):
        if (model / name).is_file():
            digest.update(name.encode() + b"\0" + (model / name).read_bytes() + b"\0")
    for path in sorted(model.rglob("*.safetensors")):
        digest.update(f"{path.relative_to(model).as_posix()}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()[:12]


def key(model, root=ROOT):
    """A report's key: crit's code and the checkpoint's revision."""
    return f"{crit_key(root)}-{checkpoint_revision(model)}"


def summary(items, **extra):
    """The report's counts by state, beside `extra` and the items."""
    counts = {state: sum(item["state"] == state for item in items) for state in STATES if state != "info"}
    return {**extra, **counts, "items": items}


def write(items, out, name=REPORT, title="crit preflight", **extra):
    """The report as NAME.json and NAME.md (scenario, state, seconds, detail) in `out`; returns the JSON's value."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    value = summary(items, **extra)
    (out / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    seconds = extra.get("seconds")
    builds = extra.get("builds") or {}
    lines = [f"# {title}", "",
             f"{value['passed']} passed, {value['failed']} failed, {value['skipped']} skipped"
             + (f" in {seconds / 60:.1f} min" if seconds is not None else "")
             + (f" (builds: {', '.join(f'{k} {v:.0f} s' for k, v in builds.items()) or 'cached'})"
                if "builds" in extra else ""),
             "", "| Scenario | State | Seconds | Detail |", "| --- | --- | --- | --- |"]
    for item in items:
        detail = item["detail"].replace("|", "\\|").replace("\n", " ")[:300]
        lines.append(f"| {item['scenario']} | {item['state']} | {item['seconds']} | {detail} |")
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def folder(root, report_key):
    """The run's folder of the report with this key."""
    return Path(root) / FOLDER / report_key


def reports(root):
    """The run's reports, oldest first: [(key, value)]."""
    found = []
    for path in sorted((Path(root) / FOLDER).glob(f"*/{REPORT}.json")):
        try:
            found.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda row: row[1].get("finished_at") or "")


def brief(root, model=None):
    """The run's preflight as `crit run ... show` and `next` state it: {key, current, report, passed, failed,
    skipped, failed_scenarios, finished_at} of the newest report (current: it is keyed by this crit and checkpoint),
    {"key": KEY, "current": False, "report": None} where the run has none, None where the checkpoint cannot be cut
    (no weight files to cut, or no folder)."""
    if model is not None and not cuttable(model):
        return None
    wanted = key(model) if model is not None else None
    found = reports(root)
    mine = [row for row in found if row[0] == wanted] or found
    if not mine:
        return {"key": wanted, "current": False, "report": None}
    name, value = mine[-1]
    return {"key": name, "current": wanted is None or name == wanted,
            "report": (folder(root, name) / f"{REPORT}.md").relative_to(root).as_posix(),
            **{state: value.get(state, 0) for state in ("passed", "failed", "skipped")},
            "failed_scenarios": [item["scenario"] for item in value.get("items", []) if item["state"] == "failed"],
            "finished_at": value.get("finished_at")}


def cuttable(model):
    """Whether the preflight can cut the checkpoint: a folder with a config.json and safetensors weight files."""
    model = Path(model)
    return model.is_dir() and (model / "config.json").is_file() and any(model.glob("*.safetensors"))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
