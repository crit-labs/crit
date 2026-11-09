"""Start a run from a crit-hub engine: read its card, pin its source and keep its evidence as the before numbers.

`crit run create --record request.json --from-engine ENGINE_DIR` takes an engine folder of a crit-hub checkout
(`engines/<owner>/<name>/`: card.yaml, card.lock.json, evidence/) and records it as the run's `origin_engine`: the
engine's identity and contributor, the repository, commit and folder that hold its source, the exact checkpoint
files it was measured on, its sealed provenance, and its published crit Eval results, copied into the run under
`captured/origin/`. `resolve` then records whether the run's model is that exact checkpoint, the first candidate
declares `"lineage": "origin-engine"` to say it is the imported source re-measured, and the engine exported from
the run says which engine it derives from.
"""
from pathlib import Path
import shutil

from ..card import CARD, LOCK, NAME, REPOSITORY, check_source, read_folder
from .records import ORIGIN_DIR, QUALITY_MODES, digest, file_sha, safe_path, sha, text


def describe_engine(folder):
    """The origin_engine record of a crit-hub engine folder, with its evidence files still at their source paths
    (`source`); store_evidence copies them into the run."""
    folder = Path(folder).expanduser().resolve()
    card_path, lock_path = folder / CARD, folder / LOCK
    card, lock = read_folder(folder)
    for key in ("title", "origin", "contributor", "source", "checkpoints", "results"):
        if key not in card:
            raise ValueError(f"{card_path}: missing {key}")
    if card["origin"] not in ("crit", "external"):
        raise ValueError(f"{card_path}: origin is crit or external")
    contributor, source = card["contributor"], card["source"]
    if not isinstance(contributor, dict) or not NAME.fullmatch(str(contributor.get("name", ""))) \
            or not REPOSITORY.fullmatch(str(contributor.get("repository", ""))):
        raise ValueError(f"{card_path}: contributor needs name and an https repository")
    check_source(card, card_path)
    if folder.parent.parent.name == "engines":
        engine = f"{folder.parent.name}/{folder.name}"
    else:
        engine = f"{contributor['name']}/{folder.name}"
    locked = lock.get("checkpoints")
    if not isinstance(locked, dict) or not locked:
        raise ValueError(f"{lock_path}: checkpoints must list every measured checkpoint's files")
    checkpoints = {}
    for model, entry in sorted(locked.items()):
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list) or not entry["files"]:
            raise ValueError(f"{lock_path}: checkpoint {model} needs repository, revision and its files")
        files = []
        for row in entry["files"]:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or type(row.get("bytes")) is not int:
                raise ValueError(f"{lock_path}: checkpoint {model} lists files as {{path, bytes, sha256}}")
            files.append({"path": row["path"], "bytes": row["bytes"], "sha256": sha(row.get("sha256"), f"{model} file sha256")})
        checkpoints[model] = {"repository": str(entry.get("repository")), "revision": str(entry.get("revision")),
                              "files": files, "files_sha256": digest(files)}
    evidence = []
    for result in card["results"] if isinstance(card["results"], list) else []:
        if not isinstance(result, dict):
            continue
        if result.get("evidence") and result.get("quality_mode") not in QUALITY_MODES:
            raise ValueError(f"{card_path}: a result states the quality mode it was judged in, quality_mode "
                             f"{' or '.join(QUALITY_MODES)}; found {result.get('quality_mode')!r}")
        for ref in result.get("evidence") or []:
            if not isinstance(ref, dict) or not isinstance(ref.get("file"), str):
                raise ValueError(f"{card_path}: evidence entries are {{file, sha256}}")
            path = safe_path(folder, ref["file"])
            if not path.is_file():
                raise ValueError(f"{folder}: evidence {ref['file']} is missing")
            found = file_sha(path)
            if found != sha(ref.get("sha256"), "evidence sha256"):
                raise ValueError(f"{folder}: evidence {ref['file']} does not match its sha256 in {CARD}")
            evidence.append({"file": ref["file"], "sha256": found, "measured_with": result.get("measured_with"),
                             "gpu": result.get("gpu"), "model": result.get("model"),
                             "quality_mode": result["quality_mode"], "source": str(path)})
    return {
        "engine": engine, "title": text(str(card["title"]), "card title", 400), "origin": card["origin"],
        "summary": card.get("summary") if isinstance(card.get("summary"), str) else None,
        "contributor": {"name": contributor["name"], "repository": contributor["repository"]},
        "source": {"repository": source["repository"], "revision": source["revision"],
                   "path": source["path"] if isinstance(source.get("path"), str) else None},
        "card": {"path": str(card_path), "sha256": file_sha(card_path)},
        "lock": {"path": str(lock_path), "sha256": file_sha(lock_path)},
        "checkpoints": checkpoints,
        "provenance": lock.get("provenance") if isinstance(lock.get("provenance"), dict) else {},
        "evidence": evidence,
    }


def store_evidence(origin, root):
    """Copy the origin's evidence files into the run's captured/origin/ and return the stored record."""
    target = safe_path(root, ORIGIN_DIR)
    target.mkdir(parents=True, exist_ok=True)
    stored = []
    for ref in origin["evidence"]:
        source = Path(ref["source"])
        destination = safe_path(target, Path(ref["file"]).name)
        if destination.exists():
            raise ValueError(f"{destination} exists; the origin's evidence files must have distinct names")
        shutil.copyfile(source, destination)
        if file_sha(destination) != ref["sha256"]:
            raise ValueError(f"{ref['file']} changed while it was copied")
        stored.append({key: ref[key] for key in ("file", "sha256", "measured_with", "gpu", "model", "quality_mode")}
                      | {"path": f"{ORIGIN_DIR}/{destination.name}"})
    return {**origin, "evidence": stored}


def checkpoint_match(origin, model_manifest):
    """Whether the run's model is a checkpoint the origin engine was measured on: per checkpoint the files the
    model lacks or stores with other bytes, and the matching checkpoint's id, or null."""
    rows = {row["path"]: row for row in model_manifest}
    report = {}
    for model, entry in origin["checkpoints"].items():
        missing, differing = [], []
        for row in entry["files"]:
            found = rows.get(row["path"])
            if found is None:
                missing.append(row["path"])
            elif found["sha256"] != row["sha256"] or found["bytes"] != row["bytes"]:
                differing.append(row["path"])
        extra = sorted(set(rows) - {row["path"] for row in entry["files"]})
        report[model] = {"matches": not missing and not differing, "missing": missing, "differing": differing, "extra": extra}
    matched = next((model for model, entry in report.items() if entry["matches"]), None)
    return {"matched": matched, "checkpoints": report}
