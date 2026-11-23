"""crit plugins: list, check and test an engine's plugins (by default crit's library: every Inference technique that
fills a slot is one plugin, inference/techniques/CATEGORY/ID, beside the plain plugin of each slot in the same
categories, the finished model plugins of models/cache and the models core they extend, models/core), copy library
plugins into an engine (add; a model plugin of the user's own cache too), say how an engine's copies stand to the
library since (status), bring the ones the engine did not change up to it (update), and show the variants an ablation
serves. The plugin contract itself, which an engine loads with, is Core's crit_plugins (docs/PLUGINS.md)."""
import argparse
import hashlib
import json
import os
import posixpath
from pathlib import Path
import shutil
import subprocess
import sys

try:
    import crit_plugins  # noqa: F401 - Core's, from crit-core
except ImportError:   # a checkout whose crit-core was installed before Core held crit_plugins
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inference" / "core" / "python"))
from crit_plugins.ablation import variants  # noqa: E402
from crit_plugins.assemble import TOGETHER, Registry, check, fitting, together  # noqa: E402
from crit_plugins.manifest import MANIFEST, REQUIRED_SLOTS, PluginError  # noqa: E402
from crit_plugins.testing import run_tests  # noqa: E402

# crit's library: the checkout's inference/techniques/ and models/, or the copies crit's wheel ships
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
    "device-speculative-round": "the technique of a speculative round on the device (accept, frontier advance, one "
                                "readback a round) and python_round's rejection and copy-draft helpers, which Core's "
                                "round driver (crit_plugins.rounds) runs for a model plugin's captured rounds",
}


MODELS = "models"   # the folder beside inference/ whose plugins (models/core, models/cache) the library holds too


class Library:
    """crit's plugin library as folders of plugins: the Inference techniques folder (inference/techniques, its plugins
    grouped by category) and, beside its inference/ folder, models/core (the models core, model-template) and
    models/cache (the finished model plugins, each extending that core). A plugin is copied into an engine at its
    place there: CATEGORY/ID, or ID for a plugin of models/. Code a plugin imports that no plugin holds goes to the
    engine's _sources at its path in the techniques folder, or at models/PATH for code in models/ (a model
    technique's)."""

    def __init__(self, inference=INFERENCE):
        self.inference = Path(inference).resolve()
        self.roots = together(self.inference)   # its plugins.json names models/core, which names models/cache
        self.models = next((root.parent for root in self.roots if root.parent.name == MODELS), None)

    def registry(self):
        return Registry.discover(*self.roots)

    def place(self, manifest):
        """Where crit plugins add puts a library plugin in an engine's plugins folder."""
        folder = Path(manifest["folder"]).resolve()
        root = next(root for root in self.roots if folder.is_relative_to(root))
        return folder.relative_to(root)

    def relative(self, folder):
        """A folder of code's path under an engine's _sources, or None for a folder outside the library."""
        folder = Path(folder).resolve()
        if folder.is_relative_to(self.inference):
            return folder.relative_to(self.inference)
        if self.models and folder.is_relative_to(self.models):
            return Path(MODELS) / folder.relative_to(self.models)
        return None

    def folder(self, relative):
        """The library folder of a path under _sources (relative()'s inverse)."""
        relative = Path(relative)
        if self.models and relative.parts[:1] == (MODELS,):
            return self.models / Path(*relative.parts[1:])
        return self.inference / relative


def library(inference=INFERENCE):
    """The library's registry: the plugin of every Inference technique that fills a slot, each slot's plain plugin,
    the finished model plugins of models/cache and the models core. crit's own library is a checkout's or the copy its wheel ships (crit.inference.LIBRARY):
    without it, the install is broken."""
    inference = Path(inference)
    if inference == INFERENCE and not inference.is_dir():
        raise PluginError(missing(inference))
    return Library(inference).registry()


def user_models():
    """{id: folder} of the model plugins in the user's cache (crit_core.cache_home("models")): the ones built on this
    machine and kept with crit models cache add, each as it lay in the engine that built it (ENGINE/plugins/ID)."""
    from crit_core import cache_home
    root = Path(cache_home("models"))
    return {path.parent.name: path.parent for path in sorted(root.glob(f"*/{MANIFEST}"))} if root.is_dir() else {}


def cached_needs(folder, places, registry):
    """What a model plugin of the user's cache takes from the library in an engine: (ids of library plugins: the
    one it extends, those whose code it imports, those its tests use; paths under _sources of other code it
    imports). Its manifest is the one it had at ENGINE/plugins/ID, so its sources say where crit plugins add puts
    the library's plugins and code in any engine."""
    data = json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))
    ident = Path(folder).name
    place = {places.place(manifest).as_posix(): manifest["id"] for manifest in registry}
    ids, lent = set(data.get("test_with", [])) | set(data.get("reference") or []), []
    for name, source in data.get("sources", {}).items():
        path = posixpath.normpath(posixpath.join(ident, source))   # from the engine's plugins folder
        if path == ident or path.startswith(ident + "/"):
            continue   # its own code
        if path.startswith(SOURCES + "/"):
            lent.append(Path(path[len(SOURCES) + 1:]))
            continue
        owner = next((other for where, other in place.items() if path == where or path.startswith(where + "/")), None)
        if owner is None:
            raise PluginError(f"{ident} (your cache of model plugins): source {name} is {source}, which is no place "
                              "crit plugins add puts a library plugin or its code")
        ids.add(owner)
    unknown = sorted(ids - {manifest["id"] for manifest in registry})
    if unknown:
        raise PluginError(f"{ident} (your cache of model plugins) uses {', '.join(unknown)}, which the library does "
                          "not hold")
    return sorted(ids), lent


def closure(registry, ids, why=None):
    """[(id, why)]: the plugins `ids` (each named, or for the reason `why` gives it), then transitively their
    references, the plugins their tests use and the plugins they extend."""
    out, seen, queue = [], set(), [(plugin_id, (why or {}).get(plugin_id, "named")) for plugin_id in ids]
    while queue:
        plugin_id, why = queue.pop(0)
        if plugin_id in seen:
            continue
        manifest = registry[plugin_id]
        seen.add(plugin_id)
        out.append((plugin_id, why))
        queue += [(other, f"the reference of {plugin_id}") for other in manifest["reference"] or []]
        queue += [(other, f"the tests of {plugin_id} use it") for other in manifest["test_with"]]
        if manifest.get("extends"):
            queue.append((manifest["extends"]["id"], f"{plugin_id} extends it"))
    return out


def add(dest, ids, inference=INFERENCE, update=False):
    """Copy library plugins into the engine's plugins folder `dest`, with their references and the plugins their tests
    use, so that each runs its tests there (crit plugins test). A plugin the engine has already is kept as the engine's;
    naming one is refused, unless `update` asks to replace the engine's copy of it with the library's current one (at
    the place crit plugins add put it; reported as `replaced`, and the dest/_sources code the added plugins import is
    copied again from the library). Each plugin's folder is copied whole to its place in
    the library (dest/CATEGORY/ID; dest/plugins.json marks dest as the folder of plugins): a technique brings its
    README, code and operations.json. Code a
    plugin imports from another plugin of
    the engine is that plugin's copy, as in the library; other code it imports (a technique that only lends code) is
    copied to dest/_sources at its path in the Inference folder, and the copied manifest points there. An id the
    library does not hold names a model plugin of the user's cache (user_models): it is copied to dest/ID as it is,
    with the library plugins and code it uses (cached_needs). Returns
    {added: [(id, why)], kept: [(id, why)], sources: [path], unmet: {need: [library plugins providing it]}}, unmet being
    the needs no plugin of the engine provides (the engine's own facts, or a plugin still to choose)."""
    dest, inference = Path(dest), Path(inference).resolve()
    registry, places = library(inference), Library(inference)
    if not ids:
        raise PluginError("name the library plugins to add (crit plugins list shows them)")
    engine = Registry.discover(dest) if dest.is_dir() else Registry([])
    kept_models = {plugin_id: folder for plugin_id, folder in user_models().items() if plugin_id not in registry}
    cached = [plugin_id for plugin_id in ids if plugin_id in kept_models]   # the user's own model plugins
    ids, why, lent = [plugin_id for plugin_id in ids if plugin_id not in kept_models], {}, {}
    for plugin_id in cached:
        used, lent[plugin_id] = cached_needs(kept_models[plugin_id], places, registry)
        for other in used:
            if other not in ids and other not in engine:
                ids.append(other)
                why[other] = f"{plugin_id} uses it"
    present = [plugin_id for plugin_id in ids + cached if plugin_id in engine and plugin_id not in why]
    if present and not update:
        raise PluginError(f"the engine has {', '.join(present)} already: remove its folder to add the library's again, "
                          "or add --update to replace its copy")
    replaced = []
    if present:
        library_place = {**{m["id"]: places.place(m) for m in registry}, **{i: Path(i) for i in cached}}
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
    wanted = closure(registry, ids, why)
    added = [(plugin_id, why) for plugin_id, why in wanted if plugin_id not in engine]
    kept = [(plugin_id, why) for plugin_id, why in wanted if plugin_id in engine]
    library_folder = {m["id"]: Path(m["folder"]).resolve() for m in registry}
    place = {plugin_id: places.place(registry[plugin_id]) for plugin_id, _ in added}
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
                relative = places.relative(folder)
                if relative is None:
                    raise PluginError(f"{plugin_id}: source {name} lies outside the library ({inference}), so crit "
                                      "plugins add cannot place its copy")
                target = dest / SOURCES / relative
                copies.setdefault(relative, folder)
            plans.append((plugin_id, name, target))
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / TOGETHER).is_file():   # the folder of plugins, which a plugin in a group finds its registry by
        (dest / TOGETHER).write_bytes(b"{}\n")
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
        path.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))   # the same bytes on every OS
    for plugin_id in cached:   # the user's own model plugins, as their engines held them
        if (dest / plugin_id).exists():
            raise PluginError(f"{dest} holds {plugin_id}, which is no plugin: move it first")
        for relative in lent[plugin_id]:
            target = dest / SOURCES / relative
            if not places.folder(relative).is_dir():
                raise PluginError(f"{plugin_id} imports {SOURCES}/{relative.as_posix()}, code the library holds no more")
            if not target.exists():
                shutil.copytree(places.folder(relative), target, ignore=IGNORE)
                sources.append(f"{SOURCES}/{relative.as_posix()}")
        shutil.copytree(kept_models[plugin_id], dest / plugin_id, ignore=IGNORE)
        added.append((plugin_id, "named, from your cache of model plugins"))
    engine = Registry.discover(dest)
    provided = {fact for manifest in engine for fact in manifest["provides"]}
    unmet = {}
    for plugin_id, _ in added:
        for alternatives in engine[plugin_id]["needs"]:
            if all(fact.startswith("model_type:") for fact in alternatives):
                continue   # the checkpoint's, which the engine gives when it loads
            if not provided & set(alternatives):
                need = " or ".join(alternatives)
                unmet[need] = sorted(m["id"] for m in registry if set(m["provides"]) & set(alternatives))
    return {"added": added, "kept": kept, "sources": sources, "unmet": unmet, "replaced": replaced}


def _tree(folder):
    """{path in the folder: bytes} of a plugin's or a source folder's files, caches aside."""
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def _same(name, mine, theirs):
    """Whether a copied file is the library's, whatever the line ends of either: a plugin's manifest but for its
    sources, whose paths crit plugins add rewrites for the engine."""
    mine, theirs = mine.replace(b"\r\n", b"\n"), theirs.replace(b"\r\n", b"\n")
    if name == MANIFEST:
        try:
            mine, theirs = ({key: value for key, value in json.loads(data).items() if key != "sources"}
                            for data in (mine, theirs))
        except ValueError:
            pass
    return mine == theirs


def _git(folder, *args):
    try:
        done = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _checkout(inference):
    """The git checkout holding the library, or None for a library that is no checkout (the copy crit's wheel
    ships), whose history cannot say where a copy came from."""
    top = _git(inference, "rev-parse", "--show-toplevel")
    if not top:
        return None
    root = Path(top.decode("utf-8").strip()).resolve()
    return root if Path(inference).resolve().is_relative_to(root) else None


def _version(root, folder, mine, limit):
    """The library's newest version of its folder `folder` that the engine's files `mine` are a copy of: {commit,
    date, changes} with the library commits that changed it since, or None when none of its last `limit` versions
    is."""
    path = Path(folder).resolve().relative_to(root).as_posix()
    log = _git(root, "log", f"-{limit}", "--format=%H %cs", "--", path)
    if not log:
        return None
    def blob(data):
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

    # a file as the engine holds it, or with the line ends a checkout stores
    wanted = {name: {blob(data), blob(data.replace(b"\r\n", b"\n"))}
              for name, data in mine.items() if name != MANIFEST}
    for later, line in enumerate(log.decode("utf-8").splitlines()):
        commit, date = line.split()
        listed = _git(root, "ls-tree", "-r", commit, "--", path)
        if listed is None:
            continue
        tree = {}
        for row in listed.decode("utf-8").splitlines():
            meta, name = row.split("\t", 1)
            name = name[len(path) + 1:]
            if "__pycache__" not in name and not name.endswith(".pyc"):
                tree[name] = meta.split()[2]
        if set(tree) != set(mine) or any(tree[name] not in blobs for name, blobs in wanted.items()):
            continue
        if MANIFEST in mine:
            theirs = _git(root, "cat-file", "blob", tree[MANIFEST])
            if theirs is None or not _same(MANIFEST, mine[MANIFEST], theirs):
                continue
        return {"commit": commit[:8], "date": date, "changes": later}
    return None


def status(dest, inference=INFERENCE, limit=500):
    """How the plugins of the engine's plugins folder `dest`, and the code under dest/_sources, stand to the library
    they were copied from: [{name, state, files, ...}], one per plugin and per source folder. state `current`: the
    library's files as they are now; `behind`: a copy of an earlier library version (commit, date, and `changes`,
    the library commits that changed it since), which the engine did not change, so crit plugins add --update
    brings the library's; `changed`: a copy of none of the library's last `limit` versions, so the engine changed
    it; `differs`: not the library's files, and the library is no git checkout whose history could say which of the
    two it is; `own`: no library plugin; `gone`: code the library holds no more. files: the paths that differ from
    the library now."""
    dest, inference = Path(dest), Path(inference).resolve()
    registry, engine, places = library(inference), Registry.discover(dest), Library(inference)
    checkout = _checkout(inference)
    place = {m["id"]: Path(m["folder"]).resolve() for m in registry}

    def compare(name, folder, theirs):
        mine, current = _tree(folder), _tree(theirs)
        files = sorted(path for path in set(mine) | set(current)
                       if path not in mine or path not in current or not _same(path, mine[path], current[path]))
        if not files:
            return {"name": name, "state": "current", "files": []}
        found = _version(checkout, theirs, mine, limit) if checkout else None
        if found:
            return {"name": name, "state": "behind", "files": files, **found}
        return {"name": name, "state": "changed" if checkout else "differs", "files": files}

    rows = []
    for manifest in engine:
        plugin_id = manifest["id"]
        if plugin_id not in place:
            rows.append({"name": plugin_id, "state": "own", "files": []})
            continue
        rows.append(compare(plugin_id, manifest["folder"], place[plugin_id]))
    sources = dest / SOURCES
    # the source folders: those the engine's plugins import from _sources, and any other code folder left there
    found = {Path(source["folder"]).resolve().relative_to(sources.resolve()) for manifest in engine
             for source in manifest["sources"].values()
             if sources.is_dir() and Path(source["folder"]).resolve().is_relative_to(sources.resolve())}
    found |= {folder.relative_to(sources) for folder in (sources.rglob("code") if sources.is_dir() else ())
              if folder.is_dir() and not any(folder.relative_to(sources).is_relative_to(other) for other in found)}
    for relative in sorted(found):
        name = f"{SOURCES}/{relative.as_posix()}"
        theirs = places.folder(relative)
        rows.append(compare(name, sources / relative, theirs) if theirs.is_dir()
                    else {"name": name, "state": "gone", "files": []})
    return rows


def update(dest, inference=INFERENCE, dry_run=False):
    """Bring the engine's copies that are behind the library up to it: each plugin and each folder of dest/_sources
    that status() calls `behind`, a copy of an earlier library version the engine did not change, is replaced by
    the library's current one (a plugin as crit plugins add --update replaces it, with the plugins its tests and
    reference now name). A copy the engine changed, its own plugins and a copy that only `differs` stay as they
    are; so does a plugin that is behind while code under _sources it imports is the engine's changed copy, which
    the replacement would overwrite. Returns {updated: [name], kept: [(name, why)], added: [(id, why)]}; with
    dry_run nothing is written and `updated` says what would be."""
    dest, inference = Path(dest), Path(inference).resolve()
    registry, places = library(inference), Library(inference)
    rows = {row["name"]: row for row in status(dest, inference)}
    behind = [name for name, row in rows.items() if row["state"] == "behind"]
    kept = [(name, "the engine changed it") for name, row in rows.items() if row["state"] == "changed"]
    kept += [(name, "not the library's files, and the library is no checkout to say whether the engine changed it")
             for name, row in rows.items() if row["state"] == "differs"]
    plugins, lent = [], set()
    for name in behind:
        if name.startswith(SOURCES + "/"):
            continue
        # The code under _sources the library's plugin imports, which its replacement copies again.
        folder = Path(registry[name]["folder"]).resolve()
        imports = {f"{SOURCES}/{places.relative(source['folder']).as_posix()}"
                   for source in registry[name]["sources"].values()
                   if not Path(source["folder"]).resolve().is_relative_to(folder)}
        changed = sorted(other for other in imports if rows.get(other, {}).get("state") in ("changed", "differs"))
        if changed:
            kept.append((name, f"it is behind, but it imports {', '.join(changed)}, which the engine changed"))
            continue
        plugins.append(name)
        lent |= imports
    sources = [name for name in behind if name.startswith(SOURCES + "/")]
    if dry_run:
        return {"updated": plugins + sources, "kept": kept, "added": []}
    added = []
    if plugins:
        report = add(dest, plugins, inference, update=True)
        added = [(plugin_id, why) for plugin_id, why in report["added"] if plugin_id not in plugins]
    for name in sources:
        relative = name[len(SOURCES) + 1:]
        target = dest / SOURCES / relative
        shutil.rmtree(target)
        shutil.copytree(places.folder(relative), target, ignore=IGNORE)
    return {"updated": plugins + sources, "kept": kept, "added": added}


def _print_status(rows, folder):
    def files(row):
        shown = ", ".join(row["files"][:4])
        return shown + (f" and {len(row['files']) - 4} more" if len(row["files"]) > 4 else "")

    for row in rows:
        detail = {
            "current": "",
            "behind": lambda: (f"the library's of {row['date']} ({row['commit']}); {row['changes']} library "
                               f"change{'s' if row['changes'] != 1 else ''} since: {files(row)}"),
            "changed": lambda: f"the engine changed it, or it is older than the library's history read: {files(row)}",
            "differs": lambda: f"not the library's files, and the library is no checkout to say why: {files(row)}",
            "own": "the engine's own",
            "gone": "the library holds this code no more",
        }[row["state"]]
        print(f"{row['state']:<8} {row['name']:<38} {detail() if callable(detail) else detail}".rstrip())
    counts = {state: sum(row["state"] == state for row in rows) for state in ("current", "behind", "changed", "differs")}
    print(", ".join(f"{count} {state}" for state, count in counts.items() if count) or "no library plugins")
    if any(row["state"] == "behind" for row in rows):
        print(f"crit plugins update {folder} replaces the copies that are behind with the library's; the engine's "
              "tests and its Eval then say whether it keeps its results")


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
    # the user's own model plugins (crit plugins add takes them by id): {id: needs} of those the library lacks
    kept_models = {} if args.folder is not None else {
        plugin_id: json.loads((folder / MANIFEST).read_text(encoding="utf-8")).get("needs", [])
        for plugin_id, folder in user_models().items() if plugin_id not in registry}
    kept_models = {plugin_id: [[need] if isinstance(need, str) else need for need in needs]
                   for plugin_id, needs in kept_models.items()}
    for slot, plugins in fitting(registry, facts, args.device).items():
        fits = [(plugin_id, reasons) for plugin_id, reasons in plugins if not reasons]
        kind = ("required" if slot in REQUIRED_SLOTS else "policy" if registry.contracts.is_policy(slot)
                else "optional")
        if not plugins:
            if slot in REQUIRED_SLOTS:
                unfilled.append(slot)
            continue
        kept = [f"{plugin_id} (your cache of model plugins)" for plugin_id in sorted(kept_models)
                if slot == "model" and all(set(alternatives) & set(facts) for alternatives in kept_models[plugin_id])]
        named = [plugin_id for plugin_id, _ in fits] + kept
        print(f"{slot} ({kind}): " + (", ".join(named) if named else "nothing fits"))
        for plugin_id, _ in fits:
            needs = [" or ".join(alternatives) for alternatives in registry[plugin_id]["needs"]
                     if not any(need in facts for need in alternatives)]
            if needs:
                print(f"    {plugin_id} needs {', '.join(needs)} from another plugin of the selection")
        for plugin_id, reasons in plugins if args.all else ():
            if reasons:
                print(f"    not {plugin_id}: {'; '.join(reasons)}")
        if slot in REQUIRED_SLOTS and not named:
            unfilled.append(slot)
    if unfilled:
        print("nothing fits " + ", ".join(unfilled) + ": write that plugin (docs/PLUGINS.md) before the engine loads")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crit plugins", description=(
        "An engine's plugins (docs/PLUGINS.md): each fills slots of the engine, says what it needs and provides, "
        "and carries its own tests; the engine assembles them when it loads. Without a folder, list, check and test "
        "act on crit's library: every Inference technique that fills a slot is a plugin "
        "(inference/techniques/CATEGORY/ID), beside the plain plugin of each slot in the same categories and the "
        "finished model plugins (models/cache/ID), which extend the models core."))
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
    tested.add_argument("--jobs", type=int, help="Plugins tested at once, a process each (default: "
                        "CRIT_PLUGIN_TEST_JOBS, else half the cores up to 4)")
    tested.add_argument("--no-cache", action="store_true", help="Run every plugin's tests, also those that passed "
                        "before with the same files and environment (by default they are reported as cached)")
    shown = subs.add_parser("variants", help="Show the variants crit run ... ablate serves for a selection")
    shown.add_argument("folder", type=Path)
    shown.add_argument("--selection", type=Path, required=True)
    shown.add_argument("--fact", action="append", default=[], help="A fact the engine provides (repeatable; default: "
                       "what the selection's plugins need and none of them provides)")
    added = subs.add_parser("add", help="Copy library plugins into an engine with their references, the plugins "
                                        "their tests use, the plugins they extend and the code they import; a "
                                        "model plugin of your own cache (crit models cache add) by its id too")
    added.add_argument("folder", type=Path, help="The engine's plugins folder")
    added.add_argument("ids", nargs="+", help="The library plugins to add")
    added.add_argument("--inference", type=Path, default=INFERENCE,
                       help="The Inference folder whose plugins to copy from (default: crit's)")
    added.add_argument("--update", action="store_true",
                       help="Replace the engine's copies of the named library plugins with the library's current ones "
                            "(a copy the engine changed is replaced too)")
    stood = subs.add_parser("status", help="How an engine's copies of library plugins stand to the library: "
                                             "current, behind (the library changed, the engine did not), changed "
                                             "by the engine, or the engine's own")
    stood.add_argument("folder", type=Path, help="The engine's plugins folder")
    stood.add_argument("--inference", type=Path, default=INFERENCE,
                       help="The Inference folder the copies came from (default: crit's)")
    stood.add_argument("--json", action="store_true", help="Print the rows as JSON")
    updated = subs.add_parser("update", help="Replace the engine's copies that are behind the library (status) "
                                               "with the library's current ones; copies the engine changed stay")
    updated.add_argument("folder", type=Path, help="The engine's plugins folder")
    updated.add_argument("--inference", type=Path, default=INFERENCE,
                         help="The Inference folder the copies came from (default: crit's)")
    updated.add_argument("--dry-run", action="store_true", help="Say what would be replaced, and write nothing")
    from . import audit, decisions, offload, resolve
    offload.add_parser(subs)
    audit.add_parser(subs)
    decisions.add_parser(subs)
    resolve.add_parser(subs)
    args = parser.parse_args(argv)
    if args.action in ("offload", "memory", "decisions", "resolve"):
        try:
            return {"offload": offload, "memory": audit, "decisions": decisions,
                    "resolve": resolve}[args.action].main(args)
        except (OSError, ValueError, ImportError, PluginError) as error:
            print(f"crit plugins {args.action}: {error}", file=sys.stderr)
            return 1
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
        if args.action == "update":
            report = update(args.folder, args.inference, dry_run=args.dry_run)
            for name in report["updated"]:
                print(f"{'would update' if args.dry_run else 'updated'} {name}")
            for plugin_id, why in report["added"]:
                print(f"added   {plugin_id:<30} {why}")
            for name, why in report["kept"]:
                print(f"kept    {name:<30} {why}")
            if not report["updated"]:
                print("no copy is behind the library")
            elif not args.dry_run:
                print(f"crit plugins test {args.folder} runs their tests in the engine; its Eval says whether it "
                      "keeps its results")
            return 0
        if args.action == "status":
            rows = status(args.folder, args.inference)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                _print_status(rows, args.folder)
            return 0
        registry = library() if args.folder is None else Registry.discover(args.folder)
        if args.action == "list" and (args.model or args.device):
            return _list_fitting(registry, args)
        if args.action == "list":
            for manifest in registry:
                fused = "fused; " if len(manifest["fills"]) > 1 else ""
                reference = f" ({fused}reference {', '.join(manifest['reference'])})" if manifest["reference"] else ""
                technique = (Path(manifest["folder"]) / "entry.json").is_file()
                # else a core other plugins extend, a finished model plugin, or a slot's plain plugin
                kind = ("technique" if technique else "core" if not manifest["fills"] else "plugin"
                        if args.folder is not None else "model" if manifest["fills"] == ["model"] else "plain")
                print(f"{manifest['id']:<30} {kind:<10} {', '.join(manifest['fills']) or '-':<18} "
                      f"{manifest['title']}{reference}")
            for plugin_id, folder in (user_models().items() if args.folder is None else ()):
                if plugin_id not in registry:
                    title = json.loads((folder / MANIFEST).read_text(encoding="utf-8")).get("title", "")
                    print(f"{plugin_id:<30} {'cached':<10} {'model':<18} {title} (your cache: {folder})")
            unnamed = sorted({s for m in registry for s in m["fills"]} - set(registry.contracts.slots))
            if unnamed:
                print("slots neither Core nor a plugin defines: " + ", ".join(unnamed))
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
            items = run_tests(registry, set(args.only or ()), args.timeout, cache=not args.no_cache, jobs=args.jobs)
            failed = sum(i["state"] == "failed" for i in items)
            skipped = [(i["name"], skip) for i in items for skip in i.get("skipped") or ()]
            print(f"{len(items) - failed} passed, {failed} failed"
                  + (f"; {len(skipped)} tests skipped, which these passes do not cover:" if skipped else ""))
            for name, skip in skipped:
                print(f"  {name}: {skip['test']}: {skip['reason']}"
                      + (" (unchecked: fails the plugin)" if skip.get("unchecked") else ""))
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
