"""Query/check the crit inference techniques; never import implementation code."""
import argparse
import json
from pathlib import Path, PurePosixPath
import sys

# crit's plugin library, LIBRARY (inference/), and ROOT, the folder holding it, which the entries' paths (inference/ID/...)
# are relative to: the checkout's, beside the cli/ folder this crit runs from (installed editable or not), else the copy
# crit's wheel ships as package data, crit/library/inference (pyproject.toml).
CHECKOUT = Path(__file__).resolve().parents[2]
ROOT = CHECKOUT if (CHECKOUT / "inference").is_dir() else Path(__file__).resolve().parent / "library"
LIBRARY = ROOT / "inference"
GUIDANCE_TEXT = ("use_when", "avoid_when")
GUIDANCE_LISTS = ("provides", "engine_supplies", "preserve", "adapt")
# A technique's operations.json: per operation it needs on a device, what it computes, its CPU reference and how a
# kernel is verified against it. No kernel lists: kernels come from libraries (kernels/README.md#libraries) and crit's kernel
# templates (kernels/techniques/<family>/<id>); a plugin declares crit's kernels it loads in its plugin.json.
OPERATIONS_KIND = "crit-technique-operations"
OPERATIONS_FILE = "operations.json"
OPERATION_TEXT = ("what", "verify")
OPERATION_FIELDS = {"id", "what", "reference", "verify", "invariance"}
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


def operations(root, ident):
    """The technique's operations.json, or None when it has none (no device arithmetic of its own)."""
    path = root / "inference" / ident / OPERATIONS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def plugins(root, ident):
    """[(id, title, folder)] of the technique's own plugin (its plugin.json), then of the other plugins of crit's
    library that import its code (their manifests' sources: other techniques and base plugins), read as data."""
    inference = root / "inference"
    code = (inference / ident / "code").resolve()
    own, others = [], []
    for path in sorted([*inference.glob("*/plugin.json"), *inference.glob("base/*/plugin.json")]):
        manifest, folder = json.loads(path.read_text(encoding="utf-8")), path.parent.relative_to(root).as_posix()
        if path.parent == inference / ident:
            own.append((manifest["id"], manifest["title"], folder))
        elif any((path.parent / source).resolve() == code for source in manifest.get("sources", {}).values()):
            others.append((manifest["id"], manifest["title"], folder))
    return own + others


def reference_symbols_missing(path, symbol):
    """The names of a reference's symbol text ("a, b::c (a note)": comma-separated, parenthesized notes aside, the last
    part of a qualified name) that the file does not contain as a word."""
    import re
    text = path.read_text(encoding="utf-8", errors="replace")
    names = [n.strip() for n in re.sub(r"\([^)]*\)", "", symbol).split(",") if n.strip()]
    return [n for n in names if not re.search(r"\b" + re.escape(re.split(r"::|\.", n)[-1]) + r"\b", text)]


def check_operations(root, ident, found):
    """Errors in one operations.json: every operation says what it computes, names its CPU reference (a file of the
    technique) and how to verify a kernel against it (tolerance; invariance where it promises one), and nothing else:
    no lists of kernels, providers or kernel source paths."""
    errors, where = [], f"{ident}/{OPERATIONS_FILE}"
    if not isinstance(found, dict) or found.get("kind") != OPERATIONS_KIND or found.get("technique") != ident:
        return [f"{where}: kind {OPERATIONS_KIND} and technique {ident} required"]
    extra = sorted(set(found) - {"kind", "technique", "operations"})
    if extra:
        errors.append(f"{where}: unknown fields {', '.join(extra)} (an operation's contract only: no kernel lists)")
    listed = found.get("operations")
    if not isinstance(listed, list) or not listed:
        return errors + [f"{where}: operations required"]
    seen = set()
    for op in listed:
        name = op.get("id") if isinstance(op, dict) else None
        if not isinstance(name, str) or not name or name in seen:
            errors.append(f"{where}: every operation needs a distinct id")
            continue
        seen.add(name)
        label = f"{where} {name}"
        unknown = sorted(set(op) - OPERATION_FIELDS)
        if unknown:
            errors.append(f"{label}: unknown fields {', '.join(unknown)}: a technique keeps no kernel lists or kernel "
                          "source paths (kernels come from crit's kernel templates, kernels/, and libraries)")
        for field in OPERATION_TEXT:
            if not isinstance(op.get(field), str) or not op[field].strip():
                errors.append(f"{label}: {field} required")
        if "invariance" in op and (not isinstance(op["invariance"], str) or not op["invariance"].strip()):
            errors.append(f"{label}: invariance is a line of text when given")
        reference = op.get("reference")
        if not isinstance(reference, dict) or not isinstance(reference.get("symbol"), str) or not reference["symbol"]:
            errors.append(f"{label}: reference with path and symbol required")
        elif set(reference) != {"path", "symbol"}:
            errors.append(f"{label}: reference is {{path, symbol}}, not {sorted(reference)}")
        else:
            try:
                path = under(root, reference.get("path"))
                if not path.is_file() or not path.is_relative_to(under(root, "inference/" + ident)):
                    errors.append(f"{label}: reference must be a file of the technique")
                else:
                    missing = reference_symbols_missing(path, reference["symbol"])
                    if missing:
                        errors.append(f"{label}: reference symbols {', '.join(missing)} are not in {reference['path']}")
            except ValueError as exc:
                errors.append(f"{label}: reference path: {exc}")
    return errors


def leftover_kernel_lists(root):
    """Errors for a kernels.json left in a base plugin's folder (a technique's is checked with its entry)."""
    return [f"{p.parent.relative_to(root / 'inference').as_posix()}/kernels.json: a plugin keeps no kernel lists; "
            "the kernels it loads are its plugin.json's \"kernels\"" for p in sorted((root / "inference").glob("base/*/kernels.json"))]


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
        if (root / "inference" / ident / "kernels.json").exists():
            errors.append(f"{ident}/kernels.json: a technique keeps no kernel lists; its per-operation contracts go in "
                          f"{OPERATIONS_FILE}")
        try:
            found = operations(root, ident)
        except ValueError as exc:
            errors.append(f"{ident}/{OPERATIONS_FILE}: {exc}")
            found = None
        if found is not None:
            errors += check_operations(root, ident, found)
    return errors + leftover_kernel_lists(root)


def parser():
    """The arguments of crit inference (crit mcp offers its tools from them)."""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name, text in (("list", "List the techniques by category"),
                       ("search", "The techniques whose metadata holds every word of the query"),
                       ("show", "One technique's guidance, symbols and files")):
        sub = subs.add_parser(name, help=text)
        if name != "list":
            sub.add_argument("query")
        sub.add_argument("--json", action="store_true")
    subs.add_parser("check", help="Check the techniques' code against their metadata")
    return parser


def missing(library):
    """What a crit whose plugin library is gone says: its install is broken."""
    return f"crit's plugin library is missing ({library} is no folder): this crit install is broken; reinstall crit"


def main(argv=None, root=ROOT):
    command = parser()
    args = command.parse_args(argv)
    try:
        if not (root / "inference").is_dir():
            raise ValueError(missing(root / "inference"))
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
                        built = plugins(root, entry["id"])
                        if built:
                            print("Plugins (crit plugins add ENGINE/plugins ID copies one into an engine):")
                            for plugin_id, title, folder in built:
                                where = "this technique" if folder == "inference/" + entry["id"] else folder
                                print("  " + plugin_id + " (" + where + "): " + title)
                        found = operations(root, entry["id"])
                        if found:
                            print("Operations (inference/" + entry["id"] + "/" + OPERATIONS_FILE + "; kernels for them: "
                                  "crit's kernel templates, crit kernels list, and kernels/README.md#libraries):")
                            for op in found["operations"]:
                                print("  " + op["id"] + ": " + op["what"])
                                print("    verify against " + op["reference"]["path"] + " " + op["reference"]["symbol"]
                                      + ": " + op["verify"])
                                if op.get("invariance"):
                                    print("    invariance: " + op["invariance"])
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
