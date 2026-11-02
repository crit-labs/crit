"""crit run build: a candidate built from its retained source under a build attempt, then packaged.

The retained source is copied to work/builds/CANDIDATE-ATTEMPT and the recipe's literal build.command runs there,
its output kept as candidates/CANDIDATE/logs/build-ATTEMPT.log, bounded by the reservation's deadline (the whole
process tree is stopped at it). The attempt settles with the command's outcome. A completed build is packaged
from that folder: every file the build added or changed is bound as a dependency (path and sha256) beside the
recipe's own, and capabilities missing from the recipe come from the adapter's declared_capabilities(model_path).
"""
import shutil
import subprocess
import sys
import types

from .attempt import Attempt, remaining, unsealed
from .records import file_sha, inventory, keys, load_run, read_json, safe_path


def build(store, run_id, candidate_id, recipe, seconds=None, drop=(), package=True, log=None):
    """Reserve, build, settle and (when the build completed and `package`) package. Returns the attempt result, the
    build folder (relative to the run) and the package manifest when packaged."""
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    root = store.directory(run_id)
    folder = unsealed(root, candidate_id)
    keys(recipe, {"build", "launch"}, {"capabilities"})
    keys(recipe["build"], {"toolchain", "command"}, {"dependencies"})
    command = recipe["build"]["command"]
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg for arg in command):
        raise ValueError("build.command must be a list of literal arguments")
    if recipe.get("capabilities") == []:
        # An empty list is a list the recipe names, so the adapter's would not be asked, and a package that declares
        # nothing can be neither served nor evaluated.
        raise ValueError("The recipe names no capabilities: leave capabilities out to take the adapter's "
                         "declared_capabilities(model_path), or name them")
    attempt = Attempt(store, run_id, candidate_id, "build", seconds, "build", "Build", "The build", launch=True)
    work = safe_path(root, f"work/builds/{candidate_id}-{attempt.id}")
    with attempt.running(tee=False):
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(folder / "source", work)
        log(f"crit run build: {' '.join(command)} in {work.relative_to(root).as_posix()}")
        attempt.end(*_run(command, work, attempt.log_path, attempt.deadline_at))
    result = attempt.settle()
    log_path = attempt.log_path
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:] if log_path.is_file() else []
    attempt.finish("\n" + "\n".join(tail) if tail else "")
    built = {"attempt": result, "build_folder": work.relative_to(root).as_posix()}
    for pattern in drop:
        for path in sorted(work.glob(pattern), reverse=True):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for cache in sorted(work.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)
    if package:
        built["package"] = store.package(run_id, candidate_id, work, package_recipe(root, folder, work, recipe,
                                                                                    attempt.id))
    return built


def _run(command, cwd, log_path, deadline_at):
    """The command's outcome, failure, whether its process tree was stopped, and its exit code (Attempt.end)."""
    from crit.process.tree import ProcessTree
    with log_path.open("ab") as stream:
        tree = ProcessTree(command, cwd, stream)
        code, outcome, failure = None, "completed", None
        try:
            code = tree.process.wait(timeout=max(0.0, remaining(deadline_at)))
            if code != 0:
                outcome, failure = "failed", f"{command[0]} exited {code}"
        except subprocess.TimeoutExpired:
            outcome, failure = "timed_out", "The build passed its reserved deadline"
        except KeyboardInterrupt:
            outcome, failure = "interrupted", "Interrupted"
        finally:
            try:
                tree.close()  # descendants the command left behind
                settled = True
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                settled = False
                failure = failure or f"Build cleanup failed: {exc}"
                outcome = outcome if outcome != "completed" else "failed"
    return outcome, failure, settled, code


def package_recipe(root, folder, work, recipe, attempt):
    """The recipe with the build's outputs bound as dependencies and the adapter's capabilities when it names none."""
    source = {row["path"]: row["sha256"] for row in read_json(folder / "source.manifest.json")["files"]}
    declared = {d["path"]: d for d in recipe["build"].get("dependencies", [])}
    outputs = [{"name": row["path"], "version": f"built by {attempt}", "path": row["path"], "sha256": row["sha256"]}
               for row in inventory(work) if source.get(row["path"]) != row["sha256"] and row["path"] not in declared
               and row["path"] != "configuration.json"]
    capabilities = recipe.get("capabilities")
    if capabilities is None:
        capabilities = adapter_capabilities(work, recipe["launch"][0], load_run(root)["spec"]["model_path"])
    return {"build": {**recipe["build"], "dependencies": list(declared.values()) + outputs},
            "launch": recipe["launch"], "capabilities": capabilities}


def adapter_capabilities(package, entry, model_path):
    """declared_capabilities(model_path) of the package's adapter, loaded as crit serve loads it (a module of a
    package of its folder), without writing bytecode into it."""
    import importlib.util
    path = safe_path(package, entry)
    namespace = "crit_build_" + file_sha(path)[:16]
    module_package = types.ModuleType(namespace)
    module_package.__path__ = [str(package)]
    sys.modules[namespace] = module_package
    spec = importlib.util.spec_from_file_location(namespace + ".entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    declared = getattr(module, "declared_capabilities", None)
    if not callable(declared):
        raise ValueError(f"{entry} exports no declared_capabilities(model_path): name the package's capabilities in "
                         "the recipe")
    value = declared(model_path)
    if not isinstance(value, list) or not all(isinstance(c, str) for c in value):
        raise ValueError(f"{entry}: declared_capabilities(model_path) must return a list of capability names")
    return value
