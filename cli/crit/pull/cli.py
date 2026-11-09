"""crit pull OWNER/NAME: install a crit-hub engine, so that crit serve --engine OWNER/NAME serves it.

It reads the engine's card (from a crit-hub checkout with --hub, else crit-hub's index.json), checks the card's
requirements against this machine (a crit eval probe result, and nvcc), fetches the engine's source at the commit
the card pins (from a crit-engines checkout with --engines, else its repository), checks its configuration against
the sealed run's in card.lock.json, fetches the checkpoint at its pinned Hugging Face revision (or takes a local
folder with --model) and checks every file against card.lock.json, fetches and checks the card's drafter downloads
the same way, runs the card's build command (or unpacks a prebuilt artifact whose toolchain matches this machine
instead), and installs the built package in the engine store (crit.pull.store). Nothing is installed until every
step has passed.
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

from crit.card import CARD, LOCK, check_source, read_folder
from crit.cli import default_project
from crit.records import GIB, digest, file_sha, inventory, now, read_json, safe_path, write_json
from crit.run.store import ROOT

from . import requirements as machine
from .store import INSTALL, MANIFEST, PLUGINS, InstallStore, checkpoint_folder, engine_id

INDEX = "https://crit-hub.dev/index.json"
BUILD_SECONDS = 1800


def log(message):
    print(message, file=sys.stderr, flush=True)


def read_card(engine, hub=None, index=None):
    """The engine's card, its lock and where they came from. A crit-hub checkout gives card.yaml and card.lock.json;
    index.json gives the same fields as crit-hub's website carries them."""
    if hub is not None:
        folder = Path(hub).expanduser().resolve() / "engines" / engine
        if not (folder / CARD).is_file():
            raise ValueError(f"{folder} holds no {CARD}: --hub names a crit-hub checkout with engines/{engine}/")
        card, lock = read_folder(folder)
        revision = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True)
        origin = {"from": "crit-hub checkout", "revision": revision.stdout.strip() or None,
                  "card_sha256": file_sha(folder / CARD), "lock_sha256": file_sha(folder / LOCK)}
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
                         "engines exported from crit, which are plugin packages")
    check_source(card, f"{engine}'s card")
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
    """Files of `rows` that `folder` lacks or holds with other bytes (a row without bytes is checked by SHA-256)."""
    bad = []
    for row in rows:
        path = safe_path(folder, row["path"])
        if not path.is_file() or ("bytes" in row and path.stat().st_size != row["bytes"]):
            bad.append(f"{row['path']}: missing" + (f" or {row['bytes']} bytes expected" if "bytes" in row else ""))
        elif file_sha(path) != row["sha256"]:
            bad.append(f"{row['path']}: SHA-256 differs")
    return bad


def fetch_checkpoint(store, entry, local=None, path=None):
    """The checkpoint's folder in the store with every file card.lock.json lists, checked by size and SHA-256: taken
    from `local` (hard-linked, else copied) or downloaded at the pinned revision. A folder already checked keeps its
    stamp (sizes and modification times) and is not hashed again. With `path` the files are that folder of the
    repository (a drafter download's), and the folder returned is it."""
    rows = entry["files"]
    relative = checkpoint_folder(entry["repository"], entry.get("revision") or "sha256-" + rows[0]["sha256"][:16])
    within = relative + "/" + path.strip("/") if path else relative
    folder = store.root / within
    stamp = store.root / (within + ".verified.json")
    def stat(row):
        info = (folder / row["path"]).stat()
        return [row["path"], info.st_size, info.st_mtime_ns]
    if stamp.is_file() and folder.is_dir():
        try:
            if read_json(stamp) == {"files_sha256": digest(rows), "files": [stat(r) for r in rows]}:
                return within, "already in the store"
        except (OSError, ValueError):
            pass
    name = f"{entry['repository']}@{entry.get('revision') or 'its pinned file'}" + (f"/{path}" if path else "")
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
        size = sum(r.get("bytes", 0) for r in rows) / GIB
        log(f"crit pull: downloading {name}" + (f" ({size:.1f} GiB)" if size else "") + f" into {folder}")
        prefix = path.strip("/") + "/" if path else ""
        snapshot_download(repo_id=entry["repository"], revision=entry.get("revision"), local_dir=str(store.root / relative),
                          allow_patterns=[prefix + row["path"] for row in rows])
        found = "Hugging Face"
    log(f"crit pull: checking {name} against card.lock.json")
    bad = verified(folder, rows)
    if bad:
        raise ValueError(f"{folder} is not {name}:\n  " + "\n  ".join(bad))
    write_json(stamp, {"files_sha256": digest(rows), "files": [stat(r) for r in rows]})
    return within, found


def drafter_files(download, lock):
    """The files a drafter download is checked against: card.lock.json's rows for it (pinned by revision), else the
    one file the card pins by its SHA-256."""
    key = (download["repository"], download.get("revision"), download.get("path"))
    for entry in lock.get("downloads") or []:
        if (entry.get("repository"), entry.get("revision"), entry.get("path")) == key and entry.get("files"):
            return entry["files"]
    if download.get("file") and download.get("sha256"):
        return [{"path": download["file"], "sha256": download["sha256"]}]
    raise ValueError(f"card.lock.json lists no files of the drafter {download['repository']}"
                     + (f"@{download['revision'][:12]}" if download.get("revision") else "")
                     + ", and the card pins no file of it by its SHA-256")


def fetch_drafters(store, card, lock, local):
    """Every drafter download in the store, checked like the checkpoint: one record per drafter, in card order, whose
    folder (relative to the store) the engine receives as context.drafters. `local` maps a repository to a local copy."""
    records = []
    for download in card.get("downloads") or []:
        if download.get("kind") != "drafter":
            continue
        rows = drafter_files(download, lock)
        entry = {"repository": download["repository"], "revision": download.get("revision"), "files": rows}
        folder, found = fetch_checkpoint(store, entry, local.get(download["repository"]), download.get("path"))
        records.append({"kind": "drafter", **{k: download.get(k) for k in ("repository", "revision", "path")},
                        "folder": folder, "files_sha256": digest(rows), "from": found})
    return records


def choose_artifact(card, device):
    """The first artifact download whose toolchain this machine matches and its rows, else (None, rows of each one
    that does not)."""
    missed = []
    for download in card.get("downloads") or []:
        if download.get("kind") != "artifact":
            continue
        rows = machine.toolchain(download.get("toolchain") or {}, device)
        if not machine.unmet(rows):
            return download, rows
        missed.append((download, rows))
    return None, missed


def fetch_artifact(store, download, local=None):
    """The artifact's file, checked by its SHA-256: `local`, else downloaded from its Hugging Face repository (at its
    revision, when the card pins one) into a cache under the store."""
    if not download.get("file") or not download.get("sha256"):
        raise ValueError(f"The artifact {download['repository']} names no file and SHA-256 to check it by")
    if local is not None:
        path, found = Path(local).expanduser().resolve(), "local file"
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ValueError("Fetching the prebuilt artifact needs huggingface_hub (python -m pip install "
                             "huggingface_hub); or pass a local copy with --download, or --build-from-source") from None
        name = (download["path"].strip("/") + "/" if download.get("path") else "") + download["file"]
        log(f"crit pull: downloading the prebuilt {download['repository']}/{name}")
        path = Path(hf_hub_download(repo_id=download["repository"], filename=name, revision=download.get("revision"),
                                    local_dir=str(store.root / ".artifacts" / download["sha256"])))
        found = "Hugging Face"
    if not path.is_file() or file_sha(path) != download["sha256"]:
        raise ValueError(f"{path} is not the artifact the card pins: its SHA-256 differs")
    return path, found


def unpack_artifact(archive, package):
    """A prebuilt artifact (a tar archive, compressed or not, of what the build adds to engine/) unpacked into the
    package: regular files and folders only, none of them replacing a file of the pinned source. Returns the files."""
    added = []
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"The artifact holds {member.name}, which is not a regular file")
            destination = safe_path(package, member.name)
            if destination.exists():
                raise ValueError(f"The artifact's {member.name} would replace a file of the engine's pinned source")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(tar.extractfile(member).read())
            added.append(Path(member.name).as_posix())
    return sorted(added)


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
    from crit.process.tree import ProcessTree
    with log_path.open("wb") as stream:
        try:
            tree = ProcessTree(command, package, stream, env=environment)
        except OSError as exc:
            code = f"could not start: {exc}"
        else:
            code = None
            try:
                code = tree.process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                code = "timed out after %g s" % seconds
            finally:
                # The whole tree, as crit run build stops it: what the build left running, and a build that timed out.
                try:
                    tree.close()
                except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                    code = code or f"left processes that could not be stopped: {exc}"
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise ValueError(f"The build failed ({code}); its log, kept at {log_path}:\n" + "\n".join(tail))
    for cache in sorted(package.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)


def pull(engine, store, project=ROOT, hub=None, index=None, checkout=None, model=None, checkpoint=None, device=None,
         context=None, force=False, ignore_requirements=False, build_seconds=BUILD_SECONDS, downloads=None,
         from_source=False):
    """Install the engine; returns its install record. `downloads` maps a download's repository to a local copy (a
    drafter's folder, an artifact's file); `from_source` builds even when a prebuilt artifact matches this machine."""
    downloads = dict(downloads or {})
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
    unknown = sorted(set(downloads) - {d.get("repository") for d in card.get("downloads") or []})
    if unknown:
        raise ValueError(f"--download names {', '.join(unknown)}, which {engine}'s card does not download")
    artifact, toolchain = (None, []) if from_source else choose_artifact(card, probe)
    if artifact is None:
        for download, missed in toolchain:
            log(f"crit pull: the prebuilt {download['repository']} does not match this machine ("
                + "; ".join(machine.unmet(missed)) + "); building from source")
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
        if not (source / "engine" / PLUGINS).is_dir():
            raise ValueError(f"{card['source']['repository']} at {card['source']['revision'][:12]} holds no "
                             f"{card['source'].get('path') or '.'}/engine/{PLUGINS}/: not a crit engine export, which "
                             "is a plugin package")
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
        drafters = fetch_drafters(store, card, lock, downloads)
        command = (card.get("build") or {}).get("command")
        built = None
        if artifact is not None:
            path, found = fetch_artifact(store, artifact, downloads.get(artifact["repository"]))
            log(f"crit pull: the prebuilt {artifact['repository']} matches this machine; unpacking it instead of "
                "building")
            unpack_artifact(path, package)
            built = {"artifact": {k: artifact.get(k) for k in ("repository", "revision", "path", "file", "sha256")
                                  if artifact.get(k)} | {"from": found}, "toolchain": toolchain}
            command = None
        elif command:
            build(package, command, folder / "logs" / "build.log", build_seconds)
        files = inventory(package)
        outputs = [row["path"] for row in files if sources.get(row["path"]) != row["sha256"]]
        from crit_core.engine import check_package
        from crit_plugins import PluginError
        try:
            check_package(package, store.root / model_folder, configuration)
        except PluginError as exc:
            raise ValueError(f"{engine} is no engine Core can load: {exc}") from None
        for cache in sorted(package.rglob("__pycache__"), reverse=True):
            shutil.rmtree(cache)
        files = inventory(package)
        manifest = {"schema_version": 1, "record_kind": "installed_package", "engine": engine, "files": files,
                    "build": ({"command": command, "outputs": outputs} if command else
                              {"artifact": built["artifact"], "outputs": outputs} if built else None),
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
                  "build": ({"command": command, "log": "logs/build.log", "outputs": outputs} if command else
                            built | {"outputs": outputs} if built else None),
                  "downloads": drafters}
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


def parser():
    """The arguments of crit pull (crit mcp offers its tools from them)."""
    parser = argparse.ArgumentParser(prog="crit pull", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", nargs="?", metavar="OWNER/NAME", help="The crit-hub engine, as OWNER/NAME")
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
    parser.add_argument("--project", type=Path, default=default_project(),
                        help="crit checkout whose settings.toml gives the device and headroom and whose engines/ is the "
                             "default store (default: this checkout); a crit installed without a checkout takes any "
                             "folder (default: the current one)")
    parser.add_argument("--force", action="store_true", help="Install again over an installed engine")
    parser.add_argument("--ignore-requirements", action="store_true",
                        help="Install even when this machine does not meet the card's requirements")
    parser.add_argument("--build-timeout", type=float, default=BUILD_SECONDS, metavar="SECONDS",
                        help="Seconds the build may take (default: %(default)s)")
    parser.add_argument("--build-from-source", action="store_true",
                        help="Build even when one of the card's prebuilt artifacts matches this machine")
    parser.add_argument("--download", action="append", default=[], metavar="REPOSITORY=PATH",
                        help="A local copy of one of the card's downloads (a drafter's folder, an artifact's file), "
                             "checked like a download (repeatable; default: fetch it from Hugging Face)")
    parser.add_argument("--list", action="store_true", help="List the installed engines instead")
    return parser


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if bool(args.list) == bool(args.engine):
        command.error("name an engine, OWNER/NAME, or --list")
    if args.hub and args.index:
        command.error("--hub and --index are two sources of the card; give one")
    if args.context is not None and args.context < 2:
        command.error("--context is at least 2 tokens")
    if not all("=" in value and all(value.split("=", 1)) for value in args.download):
        command.error("--download is REPOSITORY=PATH, such as owner/drafter=./drafter")
    local = dict(value.split("=", 1) for value in args.download)
    try:
        store = InstallStore(args.store or args.project / "engines")
        if args.list:
            print(json.dumps(store.list(), indent=2))
            return 0
        record = pull(args.engine, store, args.project, args.hub, args.index, args.engines, args.model,
                      args.checkpoint, args.device, args.context, args.force, args.ignore_requirements,
                      args.build_timeout, local, args.build_from_source)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2))
    serve = "crit serve --engine " + record["engine"] + (f" --store {args.store}" if args.store else "")
    log(f"crit pull: installed {record['engine']}; serve it with: {serve}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
