"""Provider-neutral kernel inspection and retained dependency bytes; no imports or downloads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

from crit.run.records import digest, file_sha, inventory, number, read_json, safe_path, sha, text, write_json

MAX_ENTRIES = 256
MAX_ARTIFACTS = 4096
REQUIRED_FACTS = {"operation.name", "operation.dtype", "operation.layout", "target.os",
                  "target.arch", "target.backend", "target.abi"}
ROLES = {"source", "header", "binary", "runtime", "notice", "build", "metadata", "test"}
FACT = r"(?:operation|target)\.[a-z][a-z0-9_]{0,63}"
PIN = r"(?:git:[0-9a-f]{40}|git:[0-9a-f]{64}|sha256:[0-9a-f]{64})"
RETAINED_QUALIFICATION = "retained declared-compatible dependency; arithmetic/performance unqualified"


def fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Expected fields: " + ", ".join(sorted(required)) +
                         ("; optional: " + ", ".join(sorted(optional)) if optional else ""))


def header(value, kind, required):
    fields(value, {"schema_version", "record_kind", *required})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != kind:
        raise ValueError("Expected schema_version 1, record_kind " + kind)


def strings(value, name, maximum=128):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(name + " requires a bounded list of strings")
    for item in value:
        text(item, name)
    if len(value) != len(set(value)):
        raise ValueError(name + " contains duplicates")
    return value


def scalar(value):
    if type(value) is str:
        text(value, "Fact value")
    elif type(value) not in {bool, int, float} or (type(value) in {int, float} and
            (abs(value) > 2 ** 63 or not math.isfinite(value))):
        raise ValueError("Facts require finite bounded numbers, booleans or nonempty strings")


def request(value):
    header(value, "kernels_request", {"operation", "target"})
    facts = {}
    for namespace in ("operation", "target"):
        items = value[namespace]
        if not isinstance(items, dict) or len(items) > 128:
            raise ValueError(namespace + " requires at most 128 facts")
        for name, item in items.items():
            key = namespace + "." + name
            if not re.fullmatch(FACT, key):
                raise ValueError("Invalid fact name: " + key)
            if isinstance(item, list):
                strings(item, key)
            elif item is not None:
                scalar(item)
            facts[key] = item
    text(facts.get("operation.name"), "Operation name")
    return facts


def implementation(value):
    header(value, "kernels_implementation", {"id", "summary", "provenance", "license", "requirements",
        "contract", "artifacts", "runtime_dependencies", "unverified"})
    if not isinstance(value["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value["id"]):
        raise ValueError("Implementation id requires 1–128 lowercase letters, digits, dots, underscores or hyphens")
    text(value["summary"], "Implementation summary")
    fields(value["provenance"], {"repository", "revision", "variant"})
    for key, item in value["provenance"].items():
        text(item, "Provenance " + key)
    text(value["license"], "License declaration")
    strings(value["runtime_dependencies"], "Runtime dependencies")
    strings(value["unverified"], "Unverified applicability", 64)
    fields(value["contract"], {"semantics", "invocation", "workspace", "completion", "ownership"})
    for name, item in value["contract"].items():
        text(item, "Contract " + name, 8000)
    requirements = value["requirements"]
    if not isinstance(requirements, dict) or len(requirements) > 256:
        raise ValueError("Requirements need at most 256 named facts")
    for name, rule in requirements.items():
        if not re.fullmatch(FACT, name) or not isinstance(rule, dict) or not rule:
            raise ValueError("Invalid requirement: " + name)
        if set(rule) == {"one_of"}:
            if not isinstance(rule["one_of"], list) or not 1 <= len(rule["one_of"]) <= 128:
                raise ValueError(name + " one_of needs 1–128 scalar values")
            for item in rule["one_of"]:
                scalar(item)
        elif set(rule) == {"contains"}:
            strings(rule["contains"], name)
        elif set(rule) == {"any_value"} and rule["any_value"] is True:
            pass
        elif set(rule) <= {"minimum", "maximum", "multiple_of"}:
            for key, item in rule.items():
                number(item, name + " " + key, -2 ** 63, 2 ** 63)
            if "multiple_of" in rule:
                number(rule["multiple_of"], name + " multiple_of", 1, 2 ** 63, integer=True)
            if rule.get("minimum", -2 ** 63) > rule.get("maximum", 2 ** 63):
                raise ValueError("Reversed requirement range: " + name)
        else:
            raise ValueError("Unknown or mixed requirement rule: " + name)
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) > MAX_ARTIFACTS:
        raise ValueError("Artifact inventory exceeds its bounded list")
    paths, spellings = set(), {}
    for item in artifacts:
        fields(item, {"path", "bytes", "sha256", "role"})
        # Lexical paths are checked again against the real source/destination.
        path = item["path"]
        if not isinstance(path, str) or not path or "\\" in path or ":" in path or any(p in {"", ".", ".."} for p in path.split("/")):
            raise ValueError("Artifact paths must be canonical relative paths")
        parts = path.split("/")
        for i, part in enumerate(parts):
            if re.search(r'[\x00-\x1f<>"|?*]', part) or part.endswith((" ", ".")) or re.fullmatch(
                    r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part, re.IGNORECASE):
                raise ValueError("Artifact path is not portable: " + path)
            prefix = "/".join(parts[:i + 1])
            previous = spellings.setdefault(prefix.casefold(), prefix)
            if previous != prefix:
                raise ValueError("Case-aliased artifact path: " + path)
        if path.casefold() in paths:
            raise ValueError("Duplicate/case-aliased artifact path: " + path)
        paths.add(path.casefold())
        if item["role"] not in ROLES:
            raise ValueError("Unknown artifact role")
        number(item["bytes"], "Artifact bytes", 0, 2 ** 60, integer=True)
        if item["role"] == "notice" and not item["bytes"]:
            raise ValueError("License/notice artifact must not be empty")
        sha(item["sha256"], "Artifact digest")
    for path in paths:
        if any("/".join(path.split("/")[:i]) in paths for i in range(1, len(path.split("/")))):
            raise ValueError("Artifact path is both a file and a directory: " + path)
    return value


def catalog(root):
    root = Path(root).absolute()
    paths = sorted(root.glob("*.json"))
    if not 1 <= len(paths) <= MAX_ENTRIES:
        raise ValueError("Catalog requires 1–256 top-level JSON implementation files")
    entries = {}
    for path in paths:
        value = implementation(read_json(safe_path(root, path.name)))
        if value["id"] in entries:
            raise ValueError("Duplicate implementation id: " + value["id"])
        entries[value["id"]] = value
    return entries


def matches(actual, rule):
    if "any_value" in rule:
        return True
    if "one_of" in rule:
        return any(type(actual) is type(item) and actual == item for item in rule["one_of"])
    if "contains" in rule:
        return isinstance(actual, list) and set(rule["contains"]) <= set(actual)
    if type(actual) not in {int, float}:
        return False
    return (rule.get("minimum", -2 ** 63) <= actual <= rule.get("maximum", 2 ** 63) and
            ("multiple_of" not in rule or (type(actual) is int and actual % rule["multiple_of"] == 0)))


def inspect(request_record, entry):
    facts = request(request_record)
    implementation(entry)
    checks = []
    for name in sorted(REQUIRED_FACTS | facts.keys() | entry["requirements"].keys()):
        actual, rule = facts.get(name), entry["requirements"].get(name)
        status = "unknown" if actual is None or rule is None else "matched" if matches(actual, rule) else "rejected"
        checks.append({"fact": name, "actual": actual, "required": rule, "status": status,
                       "reason": "Target/request fact missing" if actual is None else
                       "Implementation does not cover this fact" if rule is None else
                       "Matches declared requirement" if status == "matched" else "Does not match declared requirement"})
    missing = list(entry["unverified"])
    if not re.fullmatch(PIN, entry["provenance"]["revision"]):
        missing.append("Source revision is not pinned to a Git commit or SHA-256")
    if entry["license"].lower() in {"unknown", "unlicensed", "unspecified"}:
        missing.append("License declaration needs review")
    roles = {item["role"] for item in entry["artifacts"]}
    if "notice" not in roles:
        missing.append("No retained license/notice file is inventoried")
    if not roles & {"source", "binary"}:
        missing.append("No implementation source or binary is inventoried")
    status = ("rejected" if any(c["status"] == "rejected" for c in checks) else
              "unknown" if missing or any(c["status"] == "unknown" for c in checks) else "declared_compatible")
    return {"id": entry["id"], "status": status, "checks": checks, "unverified": missing,
            "implementation_sha256": digest(entry), "license": entry["license"],
            "provenance": entry["provenance"], "qualification": "declarations only; no operator or engine execution"}


def report(request_record, entries):
    request(request_record)
    return {"schema_version": 1, "record_kind": "kernels_assessment", "request_sha256": digest(request_record),
            "decisions": [inspect(request_record, entry) for entry in entries.values()]}


def verify_files(root, artifacts):
    for item in artifacts:
        path = safe_path(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
            raise ValueError("Artifact missing or changed: " + item["path"])


def retain(request_record, entries, selected, source, destination, rationale, max_bytes=8 * 1024 ** 3):
    """Publish a new dependency directory, with a final record only after verification."""
    assessment = report(request_record, entries)
    entry = entries.get(selected)
    decision = next((d for d in assessment["decisions"] if d["id"] == selected), None)
    if decision is None or decision["status"] != "declared_compatible":
        raise ValueError("Retain requires a declared-compatible selection; resolve unknown/rejected facts first")
    text(rationale, "Selection rationale", 8000)
    number(max_bytes, "Retention byte limit", 1, 2 ** 60, integer=True)
    if sum(item["bytes"] for item in entry["artifacts"]) > max_bytes:
        raise ValueError("Selected artifacts exceed the retention byte limit")
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if destination.resolve() != destination or destination.exists() or destination.is_relative_to(source):
        raise ValueError("Choose a new destination outside the artifact source, without symlink/reparse traversal")
    verify_files(source, entry["artifacts"])
    saved_inputs = (("request.json", request_record), ("catalog.json", list(entries.values())), ("assessment.json", assessment))
    for name, value in saved_inputs:
        if len((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")) > 1_048_576:
            raise ValueError("Retained " + name + " exceeds the 1 MiB record limit")
    destination.mkdir(parents=True, exist_ok=False)
    # Failed attempts remain visibly incomplete; never overwrite/delete a user's directory.
    for item in entry["artifacts"]:
        original = safe_path(source, item["path"])
        copied = safe_path(destination, "artifacts/" + item["path"])
        copied.parent.mkdir(parents=True, exist_ok=True)
        count, hasher = 0, hashlib.sha256()
        with original.open("rb") as reader, copied.open("xb") as writer:
            while block := reader.read(min(1024 * 1024, item["bytes"] - count + 1)):
                count += len(block)
                if count > item["bytes"]:
                    raise ValueError("Artifact grew during retention: " + item["path"])
                hasher.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if count != item["bytes"] or hasher.hexdigest() != item["sha256"]:
            raise ValueError("Artifact changed during retention: " + item["path"])
    for name, value in saved_inputs:
        write_json(destination / name, value, immutable=True)
    retained = inventory(destination)
    result = {"schema_version": 1, "record_kind": "kernels_dependency", "selected": selected,
              "request_sha256": digest(request_record), "implementation_sha256": digest(entry),
              "rationale": rationale, "files": retained,
              "qualification": RETAINED_QUALIFICATION}
    write_json(destination / "dependency.json", result, immutable=True)
    verify(destination)
    return result


def verify(root):
    root = Path(root).absolute()
    record = read_json(safe_path(root, "dependency.json"))
    header(record, "kernels_dependency", {"selected", "request_sha256", "implementation_sha256", "rationale", "files", "qualification"})
    text(record["selected"], "Selected implementation")
    text(record["rationale"], "Selection rationale", 8000)
    if record["qualification"] != RETAINED_QUALIFICATION:
        raise ValueError("Retained dependency qualification must remain arithmetic/performance unqualified")
    if record["files"] != inventory(root, exclude=("dependency.json",)):
        raise ValueError("Retained dependency bytes differ from inventory")
    req = read_json(safe_path(root, "request.json"))
    values = read_json(safe_path(root, "catalog.json"))
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_ENTRIES:
        raise ValueError("Invalid retained catalog")
    entries = {}
    for value in values:
        implementation(value)
        if value["id"] in entries:
            raise ValueError("Duplicate retained implementation")
        entries[value["id"]] = value
    assessment = report(req, entries)
    if assessment != read_json(safe_path(root, "assessment.json")):
        raise ValueError("Retained assessment differs from its request/catalog")
    entry = entries.get(record["selected"])
    if entry is None or record["request_sha256"] != digest(req) or record["implementation_sha256"] != digest(entry):
        raise ValueError("Retained dependency identity mismatch")
    if inspect(req, entry)["status"] != "declared_compatible":
        raise ValueError("Retained dependency is not declared compatible")
    verify_files(root / "artifacts", entry["artifacts"])
    return {"selected": record["selected"], "request_sha256": digest(req), "implementation_sha256": digest(entry),
            "dependency_sha256": digest(record), "files_verified": len(record["files"]),
            "qualification": "bytes and declared applicability verified; arithmetic/performance unqualified"}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("target", help="Observe host OS/architecture; ABI remains explicit")
    listing = sub.add_parser("inventory", help="Hash local artifact files without importing them")
    listing.add_argument("--root", type=Path, required=True)
    search = sub.add_parser("search", help="Search local implementation declarations")
    search.add_argument("query")
    search.add_argument("--catalog", type=Path, required=True)
    check = sub.add_parser("inspect", help="Explain every declared applicability match, rejection or unknown")
    check.add_argument("--request", type=Path, required=True)
    check.add_argument("--catalog", type=Path, required=True)
    save = sub.add_parser("retain", help="Copy and verify a selected dependency into a new directory")
    for name in ("request", "catalog", "root", "destination"):
        save.add_argument("--" + name, type=Path, required=True)
    save.add_argument("--select", required=True)
    save.add_argument("--reason", required=True)
    save.add_argument("--max-bytes", type=int, default=8 * 1024 ** 3)
    check = sub.add_parser("verify", help="Recheck a retained dependency's bytes and applicability")
    check.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            print("crit Kernels — kernel resources, local applicability checks and retained dependencies")
            print("Browse: https://huggingface.co/kernels\nGuide: kernels/README.md in the crit repository")
            print("Use kernels --help. Navigation/inspection never downloads, imports or executes kernels.")
            return 0
        if args.command == "target":
            machine = platform.machine().lower()
            result = {"schema_version": 1, "record_kind": "kernels_target", "target": {
                "os": platform.system().lower(), "arch": {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine),
                "backend": "cpu", "abi": None}, "note": "Host CPU context only; supply the kernel ABI and any device/toolchain facts explicitly"}
        elif args.command == "inventory":
            if not args.root.is_dir():
                raise ValueError("Artifact root must be a directory")
            result = inventory(args.root.absolute())
        elif args.command == "search":
            query = args.query.casefold()
            result = [entry for entry in catalog(args.catalog).values() if query in json.dumps(entry, ensure_ascii=False).casefold()]
        elif args.command == "inspect":
            result = report(read_json(args.request), catalog(args.catalog))
        elif args.command == "retain":
            result = retain(read_json(args.request), catalog(args.catalog), args.select, args.root,
                            args.destination, args.reason, args.max_bytes)
        else:
            result = verify(args.directory)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        print("crit kernels: " + str(exc), file=sys.stderr)
        return 2
