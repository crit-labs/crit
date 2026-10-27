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

from .records import file_sha, ident, inventory, keys, load_run, now, read_json, safe_path, timestamp
from .served import remaining


def build(store, run_id, candidate_id, recipe, seconds=None, drop=(), package=True, log=None):
    """Reserve, build, settle and (when the build completed and `package`) package. Returns the attempt result, the
    build folder (relative to the run) and the package manifest when packaged."""
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    root = store.directory(run_id)
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    if (folder / "seal.json").exists():
        raise ValueError("Sealed candidate is immutable; allocate a new candidate")
    keys(recipe, {"build", "launch"}, {"capabilities"})
    keys(recipe["build"], {"toolchain", "command"}, {"dependencies"})
    command = recipe["build"]["command"]
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg for arg in command):
        raise ValueError("build.command must be a list of literal arguments")
    reservation = store.reserve(run_id, candidate_id, "build", seconds)
    attempt = reservation["attempt_id"]
    work = safe_path(root, f"work/builds/{candidate_id}-{attempt}")
    log_path = folder / "logs" / f"build-{attempt}.log"
    outcome, failure, code, settled, interrupted = "completed", None, None, True, None
    try:
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(folder / "source", work)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"crit run build: {' '.join(command)} in {work.relative_to(root).as_posix()}")
        code, outcome, failure, settled = _run(command, work, log_path, reservation["deadline_at"])
    except (KeyboardInterrupt, SystemExit) as exc:
        outcome, failure, interrupted = "interrupted", "Interrupted", exc
    except BaseException as exc:
        outcome = "launch_failed" if isinstance(exc, OSError) else "failed"
        failure = f"{type(exc).__name__}: {exc}"[:8000]
    ended = now()
    if outcome == "completed" and timestamp(ended, "ended") > timestamp(reservation["deadline_at"], "deadline"):
        outcome, failure = "timed_out", "The build passed its reserved deadline"
    logs = [{"path": log_path.relative_to(root).as_posix(), "sha256": file_sha(log_path)}] if log_path.is_file() else []
    result = store.settle(run_id, reservation, {"outcome": outcome, "ended_at": ended, "exit_code": code,
                                                "cleanup_settled": settled, "logs": logs, "failure": failure})
    if interrupted is not None:
        raise interrupted
    built = {"attempt": result, "build_folder": work.relative_to(root).as_posix()}
    if outcome != "completed":
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:] if log_path.is_file() else []
        raise ValueError(f"Build {outcome}: {failure} (log {log_path.relative_to(root).as_posix()})"
                         + ("\n" + "\n".join(tail) if tail else ""))
    for pattern in drop:
        for path in sorted(work.glob(pattern), reverse=True):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for cache in sorted(work.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)
    if package:
        built["package"] = store.package(run_id, candidate_id, work, package_recipe(root, folder, work, recipe,
                                                                                    attempt))
    return built


def _run(command, cwd, log_path, deadline_at):
    """The command's exit code, outcome, failure and whether its process tree was stopped."""
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
    return code, outcome, failure, settled


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
