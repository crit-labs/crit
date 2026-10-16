"""Query/check the crit inference techniques; never import implementation code."""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[2]
GUIDANCE_TEXT = ("use_when", "avoid_when")
GUIDANCE_LISTS = ("provides", "engine_supplies", "preserve", "adapt")
# Navigation metadata, not folders: every entry names one category; `list` groups by it in this order.
CATEGORIES = ("attention", "projections", "experts", "speculation", "sampling-and-constraints",
              "execution", "memory-and-loading")


def under(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("canonical repository-relative path required")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(p in {"", ".", ".."} for p in relative.split("/")):
        raise ValueError("canonical repository-relative path required")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("path escapes the repository")
    return target


def load(root=ROOT):
    """Read authored entries directly, without generated files or code imports."""
    entries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "inference").glob("*/entry.json"))]
    return {"schema_version": 3, "collection_name": "crit Inference", "entries": entries}


def check(root=ROOT):
    errors = []
    store = load(root)
    ids = set()
    if not store["entries"]:
        errors.append("Inference contains no techniques")
    for entry in store["entries"]:
        ident = entry["id"]
        under(root, "inference/" + ident)
        if ident in ids:
            errors.append("duplicate technique ID")
        ids.add(ident)
        if any(field in entry for field in ("donors", "origin_path", "source_bundle_path", "code_locations")):
            errors.append(f"{ident}: external source-bundle metadata is not a crit technique")
        if entry.get("category") not in CATEGORIES:
            errors.append(f"{ident}: category must be one of " + ", ".join(CATEGORIES))
        if not entry["summary"] or not entry["implementation_scope"] or not entry["code_files"]:
            errors.append(f"{ident}: description and actual implementation required")
        if entry.get("implementation_status") not in {"reference", "native_source"}:
            errors.append(f"{ident}: explicit reference/native-source scope required")
        if any(not entry.get("validation", {}).get(field) for field in ("host", "native", "performance")):
            errors.append(f"{ident}: host/native/performance status must be explicit")
        guidance = entry.get("agent_guidance")
        if not isinstance(guidance, dict):
            errors.append(f"{ident}: agent selection, implementation and adaptation guidance required")
        else:
            for field in GUIDANCE_TEXT:
                value = guidance.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{ident}: agent_guidance.{field} must be nonempty text")
            for field in GUIDANCE_LISTS:
                value = guidance.get(field)
                if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
                    errors.append(f"{ident}: agent_guidance.{field} must list concrete responsibilities or choices")
        if entry["code_path"] != f"inference/{ident}/code" or entry["path"] != f"inference/{ident}/README.md":
            errors.append(f"{ident}: technique-local description/code required")
        if not under(root, entry["path"]).is_file():
            errors.append(f"{ident}: description missing")
        for field in ("input", "output", "ownership", "failure", "core_boundary"):
            if not entry["contract"].get(field):
                errors.append(f"{ident}: missing {field} contract")
        for field in ("hardware", "model", "workload"):
            if not entry["requires"].get(field):
                errors.append(f"{ident}: missing {field} requirement")
        actual = False
        listed = set()
        for code in entry["code_files"]:
            path = under(root, code["path"])
            if not path.is_relative_to(under(root, entry["code_path"])) or not path.is_file():
                errors.append(f"{ident}: actual code must be inside the feature")
            elif path.suffix in {".py", ".cpp", ".cu", ".h"} and path.stat().st_size:
                actual = True
            listed.add(path)
        if not actual:
            errors.append(f"{ident}: code folder contains no actual implementation")
        for path in under(root, entry["code_path"]).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".h"} and path.resolve() not in listed:
                errors.append(f"{ident}: implementation file missing from metadata")
    return errors


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "search", "show"):
        sub = subs.add_parser(name)
        if name != "list":
            sub.add_argument("query")
        sub.add_argument("--json", action="store_true")
    subs.add_parser("check")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            errors = check(root)
            if errors:
                raise ValueError("\n".join(errors))
            print("crit Inference passed: actual technique code and authored metadata.")
        else:
            entries = load(root)["entries"]
            if args.command == "show":
                entries = [e for e in entries if e["id"] == args.query]
                if not entries:
                    raise ValueError("unknown technique")
            elif args.command == "search":
                words = args.query.casefold().split()
                entries = [e for e in entries if all(w in json.dumps(e).casefold() for w in words)]
            if args.json:
                print(json.dumps(entries[0] if args.command == "show" else entries, indent=2))
            elif args.command == "list":
                for category in CATEGORIES + tuple(sorted({e.get("category") or "" for e in entries} - set(CATEGORIES))):
                    group = [e for e in entries if (e.get("category") or "") == category]
                    if group:
                        print((category or "uncategorized") + ":")
                        for entry in group:
                            print("  " + entry["id"] + ": " + entry["summary"])
            else:
                for entry in entries:
                    print(entry["id"] + ": " + entry["summary"])
                    if args.command == "show":
                        print("Category: " + str(entry.get("category")))
                        print("Scope: " + entry["implementation_scope"])
                        guidance = entry["agent_guidance"]
                        print("Use when: " + guidance["use_when"])
                        print("Consider another approach when: " + guidance["avoid_when"])
                        for title, field in (("Supplied here", "provides"), ("Engine supplies", "engine_supplies"),
                                             ("Preserve", "preserve"), ("Adapt", "adapt")):
                            print(title + ":")
                            for item in guidance[field]:
                                print("  - " + item)
                        print("Description and worked example: " + entry["path"])
                        print("Code:")
                        for code in entry["code_files"]:
                            print("  " + code["path"])
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
