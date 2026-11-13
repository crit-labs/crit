"""crit plugins: list, check and test an engine's plugins (by default crit's library, its Inference folder: every
technique that fills a slot is one plugin, inference/ID, beside the base plugins, inference/base/ID), copy library
plugins into an engine (add), and show the variants an ablation serves. The plugin contract itself, which an engine
loads with, is Core's crit_plugins (docs/PLUGINS.md)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

try:
    import crit_plugins  # noqa: F401 - Core's, from crit-core
except ImportError:   # a checkout whose crit-core was installed before Core held crit_plugins
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "python"))
from crit_plugins.ablation import variants  # noqa: E402
from crit_plugins.assemble import BASE, Registry, check, fitting  # noqa: E402
from crit_plugins.manifest import MANIFEST, POLICY_SLOTS, REQUIRED_SLOTS, SLOTS, PluginError  # noqa: E402
from crit_plugins.testing import run_tests  # noqa: E402

# crit's library: the checkout's inference/, or the copy crit's wheel ships
from .inference import LIBRARY as INFERENCE, missing  # noqa: E402
SOURCES = "_sources"   # an engine's copies of code its plugins import that none of its plugins holds
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")

# The Inference entries that are no plugin: code a model plugin calls rather than a technique in one of the other
# slots, or what Core's runtime does for every engine. Every other entry is a plugin or lends its code to one
# (tests/test_plugin_library.py holds every entry to that).
ENGINE_SERVICES = {
    "artifact-loading": "reads the checkpoint's tensors for the model plugin, before it hands them to the slots",
    "model-import-audit": "checks a checkpoint's headers and tensor roles before the model plugin loads it",
    "checkpoint-storage": "where the checkpoint and its prefetch live, which the model plugin reads from",
    "host-memory-admission": "admits the host memory the plugins' budget is checked against",
    "device-speculative-round": "the technique of a speculative round on the device (accept, frontier advance, one "
                                "readback a round) and python_round's rejection and copy-draft helpers, which Core's "
                                "round driver (crit_plugins.rounds) runs for a model plugin's captured rounds",
    "session-scheduling": "prefill chunks and physical batches: Core's concurrent runtime schedules every engine's steps",
    "concurrent-request-execution": "requests in flight at once, admission and compact ready rows: Core's concurrent "
                                    "runtime",
}


def library(inference=INFERENCE):
    """The library's registry: the plugin of every Inference technique that fills a slot, and the base plugins. crit's
    own library is a checkout's inference/ or the copy its wheel ships (crit.inference.LIBRARY): without it, the install
    is broken."""
    inference = Path(inference)
    if inference == INFERENCE and not inference.is_dir():
        raise PluginError(missing(inference))
    return Registry.discover(inference)


def closure(registry, ids):
    """[(id, why)]: the plugins `ids`, then transitively their references and the plugins their tests use."""
    out, seen, queue = [], set(), [(plugin_id, "named") for plugin_id in ids]
    while queue:
        plugin_id, why = queue.pop(0)
        if plugin_id in seen:
            continue
        manifest = registry[plugin_id]
        seen.add(plugin_id)
        out.append((plugin_id, why))
        queue += [(other, f"the reference of {plugin_id}") for other in manifest["reference"] or []]
        queue += [(other, f"the tests of {plugin_id} use it") for other in manifest["test_with"]]
    return out


def add(dest, ids, inference=INFERENCE, update=False):
    """Copy library plugins into the engine's plugins folder `dest`, with their references and the plugins their tests
    use, so that each runs its tests there (crit plugins test). A plugin the engine has already is kept as the engine's;
    naming one is refused, unless `update` asks to replace the engine's copy of it with the library's current one (at
    the place crit plugins add put it; reported as `replaced`, and the dest/_sources code the added plugins import is
    copied again from the library). Each plugin's folder is copied whole to its place in
    the library (dest/ID, a base plugin dest/base/ID): a technique brings its README, code and operations.json. Code a
    plugin imports from another plugin of
    the engine is that plugin's copy, as in the library; other code it imports (a technique that only lends code) is
    copied to dest/_sources at its path in the Inference folder, and the copied manifest points there. Returns
    {added: [(id, why)], kept: [(id, why)], sources: [path], unmet: {need: [library plugins providing it]}}, unmet being
    the needs no plugin of the engine provides (the engine's own facts, or a plugin still to choose)."""
    dest, inference = Path(dest), Path(inference).resolve()
    registry = library(inference)
    if not ids:
        raise PluginError("name the library plugins to add (crit plugins list shows them)")
    engine = Registry.discover(dest) if dest.is_dir() else Registry([])
    present = [plugin_id for plugin_id in ids if plugin_id in engine]
    if present and not update:
        raise PluginError(f"the engine has {', '.join(present)} already: remove its folder to add the library's again, "
                          "or add --update to replace its copy")
    replaced = []
    if present:
        library_place = {m["id"]: Path(m["folder"]).resolve().relative_to(inference) for m in registry}
        for plugin_id in present:
            if plugin_id not in library_place:
                raise PluginError(f"{plugin_id} is the engine's own plugin, no library plugin: --update replaces "
                                  "library copies only")
            folder = Path(engine[plugin_id]["folder"]).resolve()
            if folder != (dest / library_place[plugin_id]).resolve():
                raise PluginError(f"the engine keeps {plugin_id} at {folder}, not where crit plugins add puts it")
        for plugin_id in present:
            shutil.rmtree(Path(engine[plugin_id]["folder"]))
            replaced.append(plugin_id)
        engine = {m["id"]: m for m in engine if m["id"] not in present}   # the rest: its references are rechecked last
    wanted = closure(registry, ids)
    added = [(plugin_id, why) for plugin_id, why in wanted if plugin_id not in engine]
    kept = [(plugin_id, why) for plugin_id, why in wanted if plugin_id in engine]
    library_folder = {m["id"]: Path(m["folder"]).resolve() for m in registry}
    place = {plugin_id: library_folder[plugin_id].relative_to(inference) for plugin_id, _ in added}
    taken = [place[plugin_id].as_posix() for plugin_id, _ in added if (dest / place[plugin_id]).exists()]
    if taken:
        raise PluginError(f"{dest} holds {', '.join(taken)}, which is no plugin: move it first")
    # where each wanted plugin's folder is once the copy is done: an added one at its place, a kept one the engine's
    there = {**{plugin_id: dest / place[plugin_id] for plugin_id in place},
             **{plugin_id: Path(engine[plugin_id]["folder"]) for plugin_id, _ in kept}}
    plans, copies = [], {}
    for plugin_id, _ in added:
        for name, source in registry[plugin_id]["sources"].items():
            folder = Path(source["folder"]).resolve()
            if folder.is_relative_to(library_folder[plugin_id]):
                continue   # its own code, copied with it
            target = next((there[other] / folder.relative_to(library_folder[other]) for other in there
                           if folder.is_relative_to(library_folder[other])
                           and (other in place or (there[other] / folder.relative_to(library_folder[other])).is_dir())),
                          None)
            if target is None:
                if not folder.is_relative_to(inference):
                    raise PluginError(f"{plugin_id}: source {name} lies outside {inference}, so crit plugins add "
                                      "cannot place its copy")
                relative = folder.relative_to(inference)
                target = dest / SOURCES / relative
                copies.setdefault(relative, folder)
            plans.append((plugin_id, name, target))
    dest.mkdir(parents=True, exist_ok=True)
    sources = []
    for relative, folder in copies.items():
        target = dest / SOURCES / relative
        if target.exists() and update:
            shutil.rmtree(target)   # --update brings the library's current copy of the code its plugins import too
        if not target.exists():
            shutil.copytree(folder, target, ignore=IGNORE)
            sources.append(f"{SOURCES}/{relative.as_posix()}")
    for plugin_id, _ in added:
        shutil.copytree(library_folder[plugin_id], dest / place[plugin_id], ignore=IGNORE)
        path = dest / place[plugin_id] / MANIFEST
        data = json.loads(path.read_text(encoding="utf-8"))
        for owner, name, target in plans:
            if owner == plugin_id:
                data["sources"][name] = Path(os.path.relpath(target, dest / place[plugin_id])).as_posix()
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    engine = Registry.discover(dest)
    provided = {fact for manifest in engine for fact in manifest["provides"]}
    unmet = {}
    for plugin_id, _ in added:
        for alternatives in engine[plugin_id]["needs"]:
            if not provided & set(alternatives):
                need = " or ".join(alternatives)
                unmet[need] = sorted(m["id"] for m in registry if set(m["provides"]) & set(alternatives))
    return {"added": added, "kept": kept, "sources": sources, "unmet": unmet, "replaced": replaced}


def _selection(path):
    """configuration.plugins from a JSON file: the configuration object holding it, or the selection itself."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "plugins" in data and "use" not in data:
        data = data["plugins"]
    return data


def _list_fitting(registry, args):
    """crit plugins list --model DIR --device D: the facts the model and the device provide, then per slot the plugins
    that fit them (needs that another plugin provides named), the ones that do not with --all, and each required
    slot nothing fits: the plugin to write before the engine can load."""
    from crit_core.engine import architecture_facts, machine_facts
    if not (args.model and args.device):
        raise PluginError("list a model's fitting plugins with both --model and --device")
    probe = json.loads(args.device_file.read_text(encoding="utf-8")) if args.device_file else None
    facts = architecture_facts(args.model) + machine_facts(args.device, probe)
    print("facts: " + ", ".join(facts))
    unfilled = []
    for slot, plugins in fitting(registry, facts, args.device).items():
        fits = [(plugin_id, reasons) for plugin_id, reasons in plugins if not reasons]
        kind = "required" if slot in REQUIRED_SLOTS else "policy" if slot in POLICY_SLOTS else "optional"
        if not plugins:
            if slot in REQUIRED_SLOTS:
                unfilled.append(slot)
            continue
        print(f"{slot} ({kind}): " + (", ".join(plugin_id for plugin_id, _ in fits) if fits else "nothing fits"))
        for plugin_id, _ in fits:
            needs = [" or ".join(alternatives) for alternatives in registry[plugin_id]["needs"]
                     if not any(need in facts for need in alternatives)]
            if needs:
                print(f"    {plugin_id} needs {', '.join(needs)} from another plugin of the selection")
        for plugin_id, reasons in plugins if args.all else ():
            if reasons:
                print(f"    not {plugin_id}: {'; '.join(reasons)}")
        if slot in REQUIRED_SLOTS and not fits:
            unfilled.append(slot)
    if unfilled:
        print("nothing fits " + ", ".join(unfilled) + ": write that plugin (docs/PLUGINS.md) before the engine loads")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crit plugins", description=(
        "An engine's plugins (docs/PLUGINS.md): each fills slots of the engine, says what it needs and provides, "
        "and carries its own tests; the engine assembles them when it loads. Without a folder, list, check and test "
        "act on crit's library: every Inference technique that fills a slot is a plugin (inference/ID), beside the "
        "base plugins, the plain plugin of each slot (inference/base/ID)."))
    subs = parser.add_subparsers(dest="action", required=True)
    listed = subs.add_parser("list", help="List the plugins in a folder of plugins and the slots crit names; with "
                                          "--model and --device, per slot the plugins that fit them")
    listed.add_argument("folder", type=Path, nargs="?", help="The engine's plugins folder (default: crit's library)")
    listed.add_argument("--model", type=Path, help="A checkpoint folder: its architecture and stored formats are facts")
    listed.add_argument("--device", help="The device the engine serves on (cuda, rocm, metal, vulkan, cpu)")
    listed.add_argument("--device-file", type=Path, help="A crit eval probe device file for the device's facts "
                                                          "(default: the machine's)")
    listed.add_argument("--all", action="store_true", help="Also list the plugins that do not fit, with why")
    checked = subs.add_parser("check", help="Check every plugin's manifest and, with --selection, a selection")
    checked.add_argument("folder", type=Path, nargs="?")
    checked.add_argument("--selection", type=Path, help="A configuration (or its plugins object) to check")
    checked.add_argument("--fact", action="append", default=[], help="A fact the engine provides (repeatable)")
    tested = subs.add_parser("test", help="Run each plugin's own tests, one plugin per process")
    tested.add_argument("folder", type=Path, nargs="?")
    tested.add_argument("--only", action="append", help="Test only this plugin (repeatable)")
    tested.add_argument("--timeout", type=float, default=600, help="Seconds per plugin (default %(default)s)")
    tested.add_argument("--no-cache", action="store_true", help="Run every plugin's tests, also those that passed "
                        "before with the same files and environment (by default they are reported as cached)")
    shown = subs.add_parser("variants", help="Show the variants crit run ... ablate serves for a selection")
    shown.add_argument("folder", type=Path)
    shown.add_argument("--selection", type=Path, required=True)
    shown.add_argument("--fact", action="append", default=[], help="A fact the engine provides (repeatable; default: "
                       "what the selection's plugins need and none of them provides)")
    added = subs.add_parser("add", help="Copy library plugins into an engine with their references, the plugins "
                                        "their tests use and the code they import")
    added.add_argument("folder", type=Path, help="The engine's plugins folder")
    added.add_argument("ids", nargs="+", help="The library plugins to add")
    added.add_argument("--inference", type=Path, default=INFERENCE,
                       help="The Inference folder whose plugins to copy from (default: crit's)")
    added.add_argument("--update", action="store_true",
                       help="Replace the engine's copies of the named library plugins with the library's current ones "
                            "(a copy the engine changed is replaced too)")
    args = parser.parse_args(argv)
    try:
        if args.action == "add":
            report = add(args.folder, args.ids, args.inference, update=args.update)
            for plugin_id in report.get("replaced", []):
                print(f"replaced {plugin_id}")
            for plugin_id, why in report["added"]:
                print(f"added   {plugin_id:<30} {why}")
            for plugin_id, why in report["kept"]:
                print(f"kept    {plugin_id:<30} the engine's own ({why})")
            for path in report["sources"]:
                print(f"copied  {path}")
            for need, providers in report["unmet"].items():
                print(f"needs   {need:<30} " + (f"from the engine, or a plugin: {', '.join(providers)}" if providers
                                                  else "from the engine (no library plugin provides it)"))
            print(f"crit plugins test {args.folder} runs their tests in the engine")
            return 0
        registry = library() if args.folder is None else Registry.discover(args.folder)
        if args.action == "list" and (args.model or args.device):
            return _list_fitting(registry, args)
        if args.action == "list":
            for manifest in registry:
                fused = "fused; " if len(manifest["fills"]) > 1 else ""
                reference = f" ({fused}reference {', '.join(manifest['reference'])})" if manifest["reference"] else ""
                base = Path(manifest["folder"]).parent.name == BASE
                kind = f"{'base' if base else 'technique' if args.folder is None else 'plugin':<10} "
                print(f"{manifest['id']:<30} {kind}{', '.join(manifest['fills']):<18} {manifest['title']}{reference}")
            unnamed = sorted({s for m in registry for s in m["fills"]} - set(SLOTS))
            if unnamed:
                print("slots crit does not name: " + ", ".join(unnamed))
            return 0
        if args.action == "check":
            print(f"{len(registry)} plugins checked")
            if args.selection:
                use, order, _ = check(_selection(args.selection), registry, args.fact)
                print("selection checked; created in the order " + ", ".join(order))
                for slot, plugin_id in use.items():
                    print(f"  {slot:<16} {plugin_id or '-'}")
            return 0
        if args.action == "test":
            items = run_tests(registry, set(args.only or ()), args.timeout, cache=not args.no_cache)
            failed = sum(i["state"] == "failed" for i in items)
            skipped = [(i["name"], skip) for i in items for skip in i.get("skipped") or ()]
            print(f"{len(items) - failed} passed, {failed} failed"
                  + (f"; {len(skipped)} tests skipped, which these passes do not cover:" if skipped else ""))
            for name, skip in skipped:
                print(f"  {name}: {skip['test']}: {skip['reason']}")
            return 1 if failed else 0
        for variant in variants(_selection(args.selection), registry, args.fact or None):
            switched = ", ".join(f"{slot}={pid or '-'}" for slot, pid in variant["plugins"]["use"].items())
            print(f"{variant['name']:<32} {switched}" + (f"  (refused: {variant['error']})" if "error" in variant else ""))
        return 0
    except PluginError as error:
        print(f"crit plugins: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
