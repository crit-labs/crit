"""crit pull OWNER/NAME: install a crit-hub engine, so that crit serve --engine OWNER/NAME serves it.

It reads the engine's card (from a crit-hub checkout with --hub, else crit-hub's index.json), checks the card's
requirements against this machine (a crit eval probe result, and nvcc), fetches the engine's source at the commit
the card pins (from a crit-engines checkout with --engines, else its repository), checks its configuration against
the sealed run's in card.lock.json, fetches the checkpoint at its pinned Hugging Face revision (or takes a local
folder with --model) and checks every file against card.lock.json, runs the card's build command, and installs the
built package in the engine store (crit.pull.store). Nothing is installed until every step has passed.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from crit.run.records import GIB, digest, file_sha, inventory, now, read_json, safe_path, write_json
from crit.run.store import ROOT

from . import requirements as machine
from .store import ENTRY, INSTALL, MANIFEST, InstallStore, checkpoint_folder, engine_id

INDEX = "https://crit-hub.dev/index.json"
BUILD_SECONDS = 1800


def log(message):
    print(message, file=sys.stderr, flush=True)


def read_card(engine, hub=None, index=None):
    """The engine's card, its lock and where they came from. A crit-hub checkout gives card.yaml and card.lock.json;
    index.json gives the same fields as crit-hub's website carries them."""
    if hub is not None:
        folder = Path(hub).expanduser().resolve() / "engines" / engine
        if not (folder / "card.yaml").is_file():
            raise ValueError(f"{folder} holds no card.yaml: --hub names a crit-hub checkout with engines/{engine}/")
        from crit.run.hub import _yaml
        card = _yaml().safe_load((folder / "card.yaml").read_text(encoding="utf-8"))
        lock = read_json(folder / "card.lock.json", max_bytes=16 << 20)
        revision = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True)
        origin = {"from": "crit-hub checkout", "revision": revision.stdout.strip() or None,
                  "card_sha256": file_sha(folder / "card.yaml"), "lock_sha256": file_sha(folder / "card.lock.json")}
    else:
        index = index or INDEX
        if index.startswith("https://"):
            with urllib.request.urlopen(index, timeout=60) as response:
                value = json.loads(response.read().decode("utf-8"))
        else:
            value = json.loads(Path(index).expanduser().read_text(encoding="utf-8"))
        entry = next((e for e in value.get("engines", []) if e.get("path") == f"engines/{engine}"), None)
        if entry is None:
            raise ValueError(f"{index} lists no engine {engine}")
        pairs = entry.get("pairs") or []
        card = {"title": entry.get("name"), **{k: entry.get(k) for k in ("summary", "origin", "contributor", "license",
                                                                          "features", "source", "requirements",
                                                                          "build", "downloads")},
                "checkpoints": {p["model"]: p["checkpoint"] for p in pairs if p.get("checkpoint")},
                "results": [p["result"] for p in pairs if p.get("status") == "measured" and p.get("result")]}
        if not card.get("build"):
            card.pop("build")
        lock = entry.get("lock") or {}
        origin = {"from": index, "revision": None, "card_sha256": digest(card), "lock_sha256": digest(lock)}
    if not isinstance(card, dict) or not isinstance(lock, dict):
        raise ValueError(f"{engine}'s card and lock must be objects")
    if card.get("origin") != "crit":
        raise ValueError(f"{engine} was built with another toolkit (origin: {card.get('origin')}); crit pull installs "
                         "engines exported from crit, whose source has a crit serve adapter")
    source = card.get("source") or {}
    if not all(isinstance(source.get(k), str) for k in ("repository", "revision")):
        raise ValueError(f"{engine}'s card pins no source repository and revision")
    if not isinstance(lock.get("checkpoints"), dict) or not lock["checkpoints"]:
        raise ValueError(f"{engine}'s card.lock.json lists no checkpoint files")
    return card, lock, origin


def choose_checkpoint(engine, lock, wanted=None):
    ids = sorted(lock["checkpoints"])
    if wanted is not None:
        if wanted not in ids:
            raise ValueError(f"{engine} was measured on {', '.join(ids)}, not {wanted}")
        return wanted
    if len(ids) != 1:
        raise ValueError(f"{engine} was measured on several checkpoints; name one with --checkpoint: {', '.join(ids)}")
    return ids[0]


def context_tokens(card, model, device, requested=None):
    """The context the engine serves at: --context, else the context its card measured on this checkpoint (on this
    GPU when the card measured on it)."""
    if requested is not None:
        return requested
    name = "".join(c for c in str(device.get("device", "")).lower() if c.isalnum())
    results = [r for r in card.get("results") or [] if r.get("model") == model
               and isinstance((r.get("stats") or {}).get("context_tokens"), int)]
    results.sort(key=lambda r: str(r.get("gpu", "")).replace("-", "") not in name)
    if not results:
        raise ValueError(f"The card states no measured context for {model}; give one with --context")
    return results[0]["stats"]["context_tokens"]


def device_probe(project, store, given=None):
    """A crit eval probe result: --device, else settings.toml's device, else one this command runs into the store."""
    from crit.run.cli import load_settings
    path = given
    if path is None and "device" in load_settings(project):
        path = Path(load_settings(project)["device"]).expanduser()
        path = path if path.is_absolute() else Path(project) / path
    if path is None:
        path = store.root / "device.json"
        if not path.is_file():
            log(f"crit pull: probing the GPU (crit eval probe --out {path}); --device takes an earlier probe result")
            try:
                from crit_eval.cli import main as eval_main
            except ImportError:
                raise ValueError("Checking the card's requirements needs crit Eval's device probe: install crit-eval, "
                                 "pass --device with a crit eval probe result or --ignore-requirements") from None
            if eval_main(["probe", "--out", str(path)]):
                raise ValueError("crit eval probe failed; pass --device with a probe result or --ignore-requirements")
    return read_json(Path(path), max_bytes=16 << 20)


def git(*arguments, check=True):
    result = subprocess.run(["git", *arguments], capture_output=True)
    if check and result.returncode:
        raise ValueError(f"git {' '.join(arguments)} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result


def fetch_source(source, store, checkout=None):
    """The engine's folder at the pinned commit as a tar archive's bytes: from a local crit-engines checkout, else
    fetched (that commit only) into a cache under the store."""
    repository, revision, path = source["repository"], source["revision"], source.get("path") or "."
    if checkout is not None:
        repo = str(Path(checkout).expanduser().resolve())
        if git("-C", repo, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            raise ValueError(f"{repo} does not hold commit {revision}; fetch it there, or leave out --engines to fetch "
                             f"it from {repository}")
    else:
        repo = str(store.root / ".sources" / hashlib.sha256(repository.encode()).hexdigest()[:16])
        if not Path(repo).is_dir():
            git("init", "--bare", "--quiet", repo)
        if git("-C", repo, "cat-file", "-e", revision + "^{commit}", check=False).returncode:
            log(f"crit pull: fetching {repository} at {revision[:12]}")
            git("-C", repo, "fetch", "--quiet", "--depth", "1", repository, revision)
    return git("-C", repo, "archive", "--format=tar", revision, path).stdout


def unpack(archive, path, target):
    """The archive's `path`/ folder written to `target`: regular files and folders only, inside it."""
    prefix = "" if path in (None, ".") else path.rstrip("/") + "/"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.name.startswith(prefix) or member.name == prefix.rstrip("/"):
                continue
            relative = member.name[len(prefix):]
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"The engine's source holds {member.name}, which is not a regular file")
            destination = safe_path(target, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(tar.extractfile(member).read())


def verified(folder, rows):
    """Files of `rows` that `folder` lacks or holds with other bytes."""
    bad = []
    for row in rows:
        path = safe_path(folder, row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            bad.append(f"{row['path']}: missing or {row['bytes']} bytes expected")
        elif file_sha(path) != row["sha256"]:
            bad.append(f"{row['path']}: SHA-256 differs")
    return bad


def fetch_checkpoint(store, entry, local=None):
    """The checkpoint's folder in the store with every file card.lock.json lists, checked by size and SHA-256: taken
    from `local` (hard-linked, else copied) or downloaded at the pinned revision. A folder already checked keeps its
    stamp (sizes and modification times) and is not hashed again."""
    rows = entry["files"]
    relative = checkpoint_folder(entry["repository"], entry["revision"])
    folder = store.root / relative
    stamp = store.root / (relative + ".verified.json")
    def stat(row):
        info = (folder / row["path"]).stat()
        return [row["path"], info.st_size, info.st_mtime_ns]
    if stamp.is_file() and folder.is_dir():
        try:
            if read_json(stamp) == {"files_sha256": digest(rows), "files": [stat(r) for r in rows]}:
                return relative, "already in the store"
        except (OSError, ValueError):
            pass
    name = f"{entry['repository']}@{entry['revision']}"
    if local is not None:
        local = Path(local).expanduser().resolve()
        log(f"crit pull: checking {local} against {name} ({len(rows)} files)")
        bad = verified(local, rows)
        if bad:
            raise ValueError(f"{local} is not {name}:\n  " + "\n  ".join(bad))
        for row in rows:
            target = safe_path(folder, row["path"])
            if target.is_file() and target.stat().st_size == row["bytes"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            try:
                os.link(local / row["path"], target)
            except OSError:
                shutil.copy2(local / row["path"], target)
        found = "local folder"
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ValueError("Fetching the checkpoint needs huggingface_hub (python -m pip install huggingface_hub); "
                             "or pass a local copy with --model") from None
        log(f"crit pull: downloading {name} ({sum(r['bytes'] for r in rows) / GIB:.1f} GiB) into {folder}")
        snapshot_download(repo_id=entry["repository"], revision=entry["revision"], local_dir=str(folder),
                          allow_patterns=[row["path"] for row in rows])
        found = "Hugging Face"
    log(f"crit pull: checking {name} against card.lock.json")
    bad = verified(folder, rows)
    if bad:
        raise ValueError(f"{folder} is not {name}:\n  " + "\n  ".join(bad))
    write_json(stamp, {"files_sha256": digest(rows), "files": [stat(r) for r in rows]})
    return relative, found


def build(package, command, log_path, seconds):
    """Run the card's build command in the package (a first argument python is this Python), with the CUDA
    toolkit's bin folder on PATH when nvcc is found off it (CUDA_HOME, /usr/local/cuda)."""
    command = [sys.executable if i == 0 and arg == "python" else arg for i, arg in enumerate(command)]
    environment = dict(os.environ)
    compiler = machine.nvcc()[0]
    if compiler and not shutil.which("nvcc"):
        environment["PATH"] = str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
    if not shutil.which(command[0], path=environment.get("PATH")) and not Path(package, command[0]).is_file():
        raise ValueError(f"The build needs {command[0]}, which is not on PATH")
    log(f"crit pull: building: {' '.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as stream:
        try:
            code = subprocess.run(command, cwd=package, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds,
                                  env=environment).returncode
        except subprocess.TimeoutExpired:
            code = "timed out after %g s" % seconds
        except OSError as exc:
            code = f"could not start: {exc}"
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise ValueError(f"The build failed ({code}); its log, kept at {log_path}:\n" + "\n".join(tail))
    for cache in sorted(package.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)


def pull(engine, store, project=ROOT, hub=None, index=None, checkout=None, model=None, checkpoint=None, device=None,
         context=None, force=False, ignore_requirements=False, build_seconds=BUILD_SECONDS):
    """Install the engine; returns its install record."""
    engine = engine_id(engine)
    if not force:
        try:
            store.spec(engine)
            log(f"crit pull: {engine} is installed and checks out; --force installs it again")
            return store.get(engine)
        except (ValueError, KeyError, OSError):
            pass
    card, lock, origin = read_card(engine, hub, index)
    model_id = choose_checkpoint(engine, lock, checkpoint)
    entry = lock["checkpoints"][model_id]
    probe = {} if ignore_requirements and device is None else device_probe(project, store, device)
    rows = machine.check(card.get("requirements") or {}, probe) if probe else []
    if machine.unmet(rows) and not ignore_requirements:
        raise ValueError(f"This machine does not meet {engine}'s requirements (--ignore-requirements installs it "
                         "anyway):\n  " + "\n  ".join(machine.unmet(rows)))
    for row in rows:
        if not row["met"]:
            log(f"crit pull: warning: {row['requirement']} needs {row['required']}, this machine has {row['found']}")
    if card.get("downloads"):
        log("crit pull: warning: the card lists downloads crit pull does not fetch yet: "
            + ", ".join(f"{d.get('kind')} {d.get('repository')}" for d in card["downloads"]))
    from crit.run.cli import load_settings
    settings = load_settings(project)
    limits = {"kv_tokens": context_tokens(card, model_id, probe, context),
              "vram_headroom_bytes": round(settings.get("vram_headroom_gib", 1) * GIB),
              "ram_headroom_bytes": round(settings.get("ram_headroom_gib", 2) * GIB)}
    staging = Path(tempfile.mkdtemp(dir=store.root, prefix=".pull-"))
    try:
        folder = staging / "engine"
        package = folder / "package"
        source = staging / "source"
        source.mkdir()
        unpack(fetch_source(card["source"], store, checkout), card["source"].get("path"), source)
        if not (source / "engine" / ENTRY).is_file():
            raise ValueError(f"{card['source']['repository']} at {card['source']['revision'][:12]} holds no "
                             f"{card['source'].get('path') or '.'}/engine/{ENTRY}: not a crit engine export")
        shutil.copytree(source / "engine", package)
        for notice in ("THIRD_PARTY.md", "LICENSES.txt"):
            if (source / notice).is_file():
                shutil.copy2(source / notice, folder / notice)
        configuration = read_json(package / "configuration.json")
        expected = (lock.get("provenance") or {}).get("configuration_sha256")
        if expected and digest(configuration) != expected:
            raise ValueError(f"{engine}'s engine/configuration.json at {card['source']['revision'][:12]} is not the "
                             "configuration its sealed run measured (card.lock.json provenance)")
        sources = {row["path"]: row["sha256"] for row in inventory(package)}
        model_folder, model_from = fetch_checkpoint(store, entry, model)
        command = (card.get("build") or {}).get("command")
        if command:
            build(package, command, folder / "logs" / "build.log", build_seconds)
        files = inventory(package)
        outputs = [row["path"] for row in files if sources.get(row["path"]) != row["sha256"]]
        from crit.run.build import adapter_capabilities
        capabilities = adapter_capabilities(package, ENTRY, str(store.root / model_folder))
        for cache in sorted(package.rglob("__pycache__"), reverse=True):
            shutil.rmtree(cache)
        files = inventory(package)
        manifest = {"schema_version": 1, "record_kind": "installed_package", "engine": engine, "files": files,
                    "launch": [ENTRY], "capabilities": capabilities,
                    "build": {"command": command, "outputs": outputs} if command else None,
                    "configuration_sha256": digest(configuration)}
        write_json(package / MANIFEST, manifest)
        from crit import __version__
        record = {"schema_version": 1, "record_kind": "engine_install", "engine": engine, "title": card.get("title") or engine,
                  "installed_at": now(), "crit_version": __version__,
                  "card": {**origin, "summary": card.get("summary"), "contributor": card.get("contributor")},
                  "source": {k: card["source"].get(k) for k in ("repository", "revision", "path")},
                  "checkpoint": {"model": model_id, "repository": entry["repository"], "revision": entry["revision"],
                                 "folder": model_folder, "files_sha256": digest(entry["files"]), "from": model_from},
                  "package": {"sha256": digest(manifest), "files": len(files)},
                  "limits": limits,
                  "requirements": {"checks": rows, "ignored": bool(ignore_requirements)},
                  "build": {"command": command, "log": "logs/build.log", "outputs": outputs} if command else None}
        write_json(folder / "card.json", card)
        write_json(folder / "card.lock.json", lock, max_bytes=16 << 20)
        write_json(folder / INSTALL, record)
        target = store.folder(engine)
        target.parent.mkdir(parents=True, exist_ok=True)
        old = staging / "replaced"
        if target.exists():
            target.rename(old)
        try:
            folder.rename(target)
        except OSError:
            if old.exists():
                old.rename(target)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    store.spec(engine)  # the installed engine serves as recorded
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crit pull", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", nargs="?", metavar="OWNER/NAME", help="The crit-hub engine, such as crit-labs/nvfp4-gemv-int8kv")
    parser.add_argument("--hub", type=Path, help="Read the card from this crit-hub checkout (default: crit-hub's index.json)")
    parser.add_argument("--index", help=f"crit-hub's index.json, a URL or file (default: {INDEX})")
    parser.add_argument("--engines", type=Path, metavar="CHECKOUT",
                        help="Take the source from this crit-engines checkout, which holds the pinned commit "
                             "(default: fetch that commit from the card's source repository)")
    parser.add_argument("--model", type=Path, metavar="FOLDER",
                        help="A local copy of the checkpoint, checked file by file and hard-linked into the store "
                             "(default: download it at the pinned Hugging Face revision)")
    parser.add_argument("--checkpoint", metavar="MODEL", help="The checkpoint, when the engine was measured on several")
    parser.add_argument("--device", type=Path, help="A crit eval probe result to check the requirements against "
                                                     "(default: settings.toml's device, else a probe run now)")
    parser.add_argument("--context", type=int, metavar="TOKENS",
                        help="Context the engine serves at (default: the context its card measured)")
    parser.add_argument("--store", type=Path, help="Engine store (default: PROJECT/engines)")
    parser.add_argument("--project", type=Path, default=ROOT if (ROOT / "core/include/crit/core.h").is_file() else Path.cwd(),
                        help="crit repository, whose settings.toml gives the device and headroom (default: this checkout)")
    parser.add_argument("--force", action="store_true", help="Install again over an installed engine")
    parser.add_argument("--ignore-requirements", action="store_true",
                        help="Install even when this machine does not meet the card's requirements")
    parser.add_argument("--build-timeout", type=float, default=BUILD_SECONDS, metavar="SECONDS",
                        help="Seconds the build may take (default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="List the installed engines instead")
    args = parser.parse_args(argv)
    if bool(args.list) == bool(args.engine):
        parser.error("name an engine, OWNER/NAME, or --list")
    if args.hub and args.index:
        parser.error("--hub and --index are two sources of the card; give one")
    if args.context is not None and args.context < 2:
        parser.error("--context is at least 2 tokens")
    try:
        store = InstallStore(args.store or args.project / "engines")
        if args.list:
            print(json.dumps(store.list(), indent=2))
            return 0
        record = pull(args.engine, store, args.project, args.hub, args.index, args.engines, args.model,
                      args.checkpoint, args.device, args.context, args.force, args.ignore_requirements,
                      args.build_timeout)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2))
    serve = "crit serve --engine " + record["engine"] + (f" --store {args.store}" if args.store else "")
    log(f"crit pull: installed {record['engine']}; serve it with: {serve}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
