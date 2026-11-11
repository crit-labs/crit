"""A plugin's own tests: helpers for writing them, and the runner crit plugins test uses (each plugin's tests in a
process of their own, so plugins never share imports, device state or a failure)."""
from contextlib import contextmanager
import copy
import functools
import inspect
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest

from .assemble import Registry, assemble, plugins_root
from .device import DEVICE_EXTENSIONS, POOL_EXTENSIONS, StepBatch, device_missing, extension_missing, pool_missing
from .manifest import HOST, MANIFEST
from .slots import CASES, compare, missing, model_problems


def _entries(folder):
    try:
        return set(os.listdir(folder)) if folder and os.path.isdir(folder) else set()
    except OSError:
        return set()


@contextmanager
def compiles():
    """Every kernel compile and extension build while the block runs, as a list of descriptions it fills: a Triton
    JIT compile (Triton's jit_cache_hook, which runs whenever a kernel's specialization is not yet in memory, compiled
    or read from Triton's cache), new entries in Triton's cache beyond those (kernels compiled outside triton.jit,
    such as torch.compile's), and a torch extension build or load (torch.utils.cpp_extension). It uses only what is
    already loaded (triton, torch's cpp_extension), so it reads nothing a device without them would have. A plugin
    compiles at load, never on a request: a test wraps a request after load in it and expects no entry."""
    found, restore = [], []
    triton = sys.modules.get("triton")
    runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
    hooked = runtime is not None and hasattr(runtime, "jit_cache_hook")
    jit_compiles = []
    if hooked:
        previous = runtime.jit_cache_hook

        def hook(*args, **kwargs):
            fn, options = kwargs.get("fn"), kwargs.get("compile") or {}
            names = list(getattr(getattr(fn, "jit_function", None), "arg_names", None) or [])
            constants = ", ".join(
                f"{names[key[0]] if isinstance(key, tuple) and len(key) == 1 and key[0] < len(names) else key}="
                f"{value}" for key, value in (options.get("constants") or {}).items())
            signature = ", ".join(f"{name}: {kind}" for name, kind in (options.get("signature") or {}).items()
                                  if kind != "constexpr")
            what = (f"Triton compile of {getattr(fn, 'module', '?')}.{getattr(fn, 'name', '?')}({signature}) "
                    f"constexprs {constants or 'none'}")
            found.append(what)
            jit_compiles.append(what)
            return previous(*args, **kwargs) if previous else False
        runtime.jit_cache_hook = hook
        restore.append(lambda: setattr(runtime, "jit_cache_hook", previous))
    cache = getattr(getattr(getattr(triton, "knobs", None), "cache", None), "dir", None) if triton is not None else None
    cached = _entries(cache)
    extension = sys.modules.get("torch.utils.cpp_extension")
    built = None
    if extension is not None and hasattr(extension, "_jit_compile"):
        jit = extension._jit_compile

        def building(name, *args, **kwargs):
            found.append(f"torch extension build or load of {name}")
            return jit(name, *args, **kwargs)
        extension._jit_compile = building
        restore.append(lambda: setattr(extension, "_jit_compile", jit))
        built = os.environ.get("TORCH_EXTENSIONS_DIR") or extension.get_default_build_root()
    extensions = _entries(built)
    try:
        yield found
    finally:
        for undo in reversed(restore):
            undo()
        new = sorted(_entries(cache) - cached)
        if len(new) > len(jit_compiles):
            found.append(f"{len(new) - len(jit_compiles)} new Triton cache entries beyond the JIT compiles "
                         f"(compiled outside triton.jit): {', '.join(new[:8])}")
        for name in sorted(_entries(built) - extensions):
            found.append(f"a new torch extension build directory {name}")


@functools.lru_cache(maxsize=None)
def _plugin_folder(path):
    for folder in Path(path).resolve().parents:
        if (folder / MANIFEST).is_file():
            return folder
    raise ValueError(f"{path} lies in no plugin folder")


# A test process reads its plugins folder's manifests once: discovery stats and reads every plugin's files, which took
# almost all of a device test's time where the checkout sits on a slow file system (a Windows drive seen from WSL).
# Each test gets its own copy, so a test that changes a manifest it was given changes nothing for the next.
_DISCOVERED = {}


class PluginTestCase(unittest.TestCase):
    """A test of the plugin whose folder holds the test file. assemble() creates it (with the plugins and engine facts
    it needs, which a test supplies); reference() creates its reference plugins in the same slots, and
    assertMatchesReference() compares the two on the cases crit keeps for those slots (crit_plugins.slots)."""

    def plugin_folder(self):
        return _plugin_folder(inspect.getfile(type(self)))

    def registry(self):
        """The plugins beside it: those of the folder of plugins it lies in (an engine's plugins folder, or crit's
        Inference folder for a technique of crit's library), base plugins included."""
        root = plugins_root(self.plugin_folder())
        found = _DISCOVERED.get(str(root))
        if found is None:
            found = _DISCOVERED[str(root)] = Registry.discover(root)
        return Registry(copy.deepcopy(list(found.manifests.values())))

    def plugin_id(self):
        return self.plugin_folder().name

    def assemble(self, use=None, options=None, context=None, facts=(), budget=None, use_options=None):
        """The plugin assembled in its own slots, beside `use` (slot -> other plugin) for what it needs. options: its
        own; use_options: the other plugins', by id."""
        registry = self.registry()
        own = {slot: self.plugin_id() for slot in registry[self.plugin_id()]["fills"]}
        selection = {"use": {**(use or {}), **own},
                     "options": {**(use_options or {}), self.plugin_id(): options or {}}}
        assembly = assemble(selection, registry, context, facts, budget)
        self.addCleanup(assembly.close)
        return assembly

    def reference(self, use=None, options=None, context=None, facts=(), use_options=None):
        """The plugin's reference (its manifest's `reference`) in its slots: for a fused plugin the separate plugins it
        must equal, for another the plugin it replaces. use: other slots, as for assemble(); options: the reference
        plugins', by id; use_options: the other plugins', by id."""
        registry = self.registry()
        selection = {"use": {**(use or {}), **registry.reference_use(self.plugin_id())},
                     "options": {**(use_options or {}), **(options or {})}}
        assembly = assemble(selection, registry, context, facts)
        self.addCleanup(assembly.close)
        return assembly

    def device(self):
        """The device a test of a plugin's device form runs on, one of the devices its manifest names: CUDA when the
        plugin runs there and torch has it, else torch's CPU when the plugin runs there; a policy plugin running on the
        host (or a host reference) takes CUDA when torch has it, else the CPU. The test is skipped without torch, and
        when none of the plugin's devices is here (a CUDA plugin on a machine without CUDA)."""
        try:
            import torch
        except ImportError:
            self.skipTest("the device form's tensors are torch's, which is not installed")
        devices = self.registry()[self.plugin_id()]["devices"]
        cuda = torch.cuda.is_available()
        if not devices or HOST in devices:
            return "cuda" if cuda else "cpu"
        if "cuda" in devices and cuda:
            return "cuda"
        if "cpu" in devices:
            return "cpu"
        self.skipTest(f"{self.plugin_id()} runs on {', '.join(devices)}, none of which this machine has")

    def assertNoCompileAfter(self, function, *args, **kwargs):
        """function(*args, **kwargs), failing when it compiles a kernel or builds an extension (compiles()): a plugin
        compiles every kernel it calls at load (or at the warm-up its model runs at load), never on a request.
        Returns what the function returns."""
        with compiles() as found:
            result = function(*args, **kwargs)
        if found:
            self.fail("compiled after load: " + "; ".join(found))
        return result

    def assertImplements(self, assembly=None):
        """The plugin's object in each slot it fills has the slot's interface methods (crit_plugins.slots), and a
        plugin that runs on a device also the slot's device form (crit_plugins.device); a model plugin also declares
        what Core's concurrent runtime needs to batch it (model_problems)."""
        assembly = assembly or self.assemble()
        manifest = self.registry()[self.plugin_id()]
        on_device = [d for d in manifest["devices"] if d != HOST]
        for slot in manifest["fills"]:
            lacking = missing(slot, assembly[slot])
            if lacking:
                self.fail(f"{self.plugin_id()} in {slot} lacks {', '.join(lacking)} of the {slot} interface")
            lacking = device_missing(slot, assembly[slot]) if on_device else []
            if lacking:
                self.fail(f"{self.plugin_id()} runs on {', '.join(on_device)} but lacks {', '.join(lacking)} of the "
                          f"{slot} device form")
            if slot == "kv_cache" and on_device:
                self.assertKeepsKV(assembly[slot])
            if slot == "projections" and on_device:
                self.assertKeepsRows(assembly[slot])
            if slot == "model":
                problems = model_problems(assembly[slot])
                if problems:
                    self.fail(f"{self.plugin_id()}: Core cannot run it: {'; '.join(problems)}")
            for fact in manifest["provides"]:
                if fact not in DEVICE_EXTENSIONS:
                    continue
                lacking = extension_missing(fact, assembly[slot], self._made(slot, assembly[slot], fact))
                if lacking:
                    self.fail(f"{self.plugin_id()} provides {fact} but lacks {', '.join(lacking)} "
                              "(crit_plugins.device.DEVICE_EXTENSIONS)")

    def _made(self, slot, plugin, fact):
        """The objects the slot's device form returns that a pool fact's methods live on (POOL_EXTENSIONS): a small
        pool."""
        if POOL_EXTENSIONS.get(fact) != slot:
            return ()
        device = self.device()
        if slot == "kv_cache":
            return (plugin.pool([0], 1, 8, 64, device),)
        return (plugin.pool([0], 1, 4, 4, device),)

    def assertKeepsRows(self, plugin):
        """The projections plugin's matmul keeps each row's bits whatever rows share a call of up to
        crit_plugins.device.invariant_rows(plugin) rows (DECODE_ROWS, or the facts' row_invariance rows, up to 32 here),
        on a BF16 matrix, every plugin's format; where its facts say row_invariance prompt_rows, in calls of more
        rows than that (prompt rows: slices of a long call against the call); and where they say prompt_exact, a long
        call's rows equal the same rows alone (a prompt row has its decode row's bits)."""
        import torch
        from .device import invariant_rows
        device = self.device()
        generator = torch.Generator().manual_seed(9)
        matrix = plugin.load({"weight": torch.randn(48, 256, generator=generator).to(device) * 0.05}, "bf16")
        x = torch.randn(300, 256, generator=generator).to(device).to(torch.bfloat16)
        facts = plugin.facts() if callable(getattr(plugin, "facts", None)) else {}
        most = min(32, invariant_rows(plugin))
        for out_dtype in (None, torch.float32):
            alone = [plugin.matmul(x[r:r + 1].clone(), matrix, out_dtype)[0] for r in range(most)]
            for rows in range(2, most + 1):
                batch = plugin.matmul(x[:rows], matrix, out_dtype)
                for r in range(rows):
                    if not torch.equal(batch[r], alone[r]):
                        self.fail(f"{self.plugin_id()}: row {r} of a {rows}-row matmul differs from the row alone "
                                  f"({out_dtype or 'default'} out): batched decoding would not match serial")
        if (facts.get("row_invariance") or {}).get("prompt_rows"):
            whole = plugin.matmul(x, matrix)
            for first, rows in ((0, most + 1), (7, 40), (100, 200)):
                if not torch.equal(plugin.matmul(x[first:first + rows].clone(), matrix), whole[first:first + rows]):
                    self.fail(f"{self.plugin_id()} says its prompt rows keep their bits, but rows {first}.. of "
                              f"{rows} differ from the same rows in a call of {len(x)}")
        if (facts.get("row_invariance") or {}).get("prompt_exact"):
            whole = plugin.matmul(x, matrix)
            for r in (0, 1, most, 150, len(x) - 1):
                if not torch.equal(whole[r], plugin.matmul(x[r:r + 1].clone(), matrix)[0]):
                    self.fail(f"{self.plugin_id()} says its prompt rows have the decode rows' bits, but row {r} of a "
                              f"call of {len(x)} differs from the row alone")

    def assertKeepsKV(self, plugin):
        """The kv_cache plugin's pool keeps crit_plugins.device.KV_POOL: every method and attribute, and what a step
        writes is what gather() reads, across a fork and a rollback."""
        import torch
        device = self.device()
        pool = plugin.pool([0], 2, 8, 64, device)
        lacking = pool_missing(pool)
        if lacking:
            self.fail(f"{self.plugin_id()}'s pool lacks {', '.join(lacking)} of crit_plugins.device.KV_POOL")
        generator = torch.Generator().manual_seed(5)
        keys = torch.randn(5, 2, 8, generator=generator).to(device)
        values = torch.randn(5, 2, pool.value_dim, generator=generator).to(device)
        pool.open("a")
        pool.write(0, pool.reserve(StepBatch.build([("a", 0, 5)], device)), keys, values)
        pool.keep("a", 5)
        pool.fork("a", "b")
        pool.write(0, pool.reserve(StepBatch.build([("b", 3, 1)], device)), keys[:1] * 2, values[:1] * 2)
        pool.keep("b", 4)
        found = {name: pool.gather(0, name, pool.length(name)) for name in ("a", "b")}
        expected = {"a": (keys, values), "b": (torch.cat([keys[:3], keys[:1] * 2]), torch.cat([values[:3], values[:1] * 2]))}
        for name, (k, v) in expected.items():
            got_k, got_v = found[name]
            self.assertEqual((tuple(got_k.shape), tuple(got_v.shape)), (tuple(k.shape), tuple(v.shape)), name)
            torch.testing.assert_close(got_k.float(), k.float(), atol=0.07, rtol=0.02)
            torch.testing.assert_close(got_v.float(), v.float(), atol=0.07, rtol=0.02)
        pool.close("a")
        pool.close("b")

    def assertMatchesReference(self, use=None, options=None, context=None, facts=(), atol=1e-9, rtol=1e-7,
                               reference_options=None, cases=None, use_options=None):
        """The plugin and its reference give the same values on the cases of every slot it fills that crit keeps
        cases for (token ids exactly, numbers within atol + rtol * |expected|). use: the plugins of other slots both
        need, e.g. {"kv_cache": "attention-state"} for an attention plugin, and use_options their options by id.
        cases: the slots whose cases to run instead, for a plugin seen through another slot (an expert_source through
        the experts cases)."""
        fills = self.registry()[self.plugin_id()]["fills"]
        slots = [slot for slot in (cases or fills) if slot in CASES]
        if not slots:
            self.fail(f"crit keeps no cases for {', '.join(cases or fills)}: compare the plugin with its reference in "
                      "its own tests, or name the slots whose cases see it (cases=)")
        for slot in slots:
            plugin = self.assemble(use, options, context, facts, use_options=use_options)
            reference = self.reference(use, reference_options, context, facts, use_options)
            found = compare(CASES[slot](plugin), CASES[slot](reference), atol, rtol, slot)
            if found:
                self.fail(f"{self.plugin_id()} differs from its reference: {found}")

    def assertClose(self, actual, expected, atol=1e-6, rtol=1e-5, path="value"):
        """Numbers, or nested lists and tuples of them, equal within atol + rtol * |expected| element by element."""
        if isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, (list, tuple), path)
            self.assertEqual(len(actual), len(expected), f"{path}: length")
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.assertClose(a, e, atol, rtol, f"{path}[{i}]")
            return
        if not (math.isfinite(actual) and abs(actual - expected) <= atol + rtol * abs(expected)):
            self.fail(f"{path}: {actual!r} differs from {expected!r} by more than atol {atol} + rtol {rtol}")


RAN = re.compile(r"^Ran (\d+) tests? in", re.M)


def _tree_hash(digest, folder):
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())


def _environment():
    """What a plugin's test result depends on beside its files: Python, torch and its CUDA, the device, the kernel
    tuning named (CRIT_KERNEL_TUNING) and the transformers and provider packages' versions."""
    import importlib.metadata as metadata
    found = {"python": sys.version.split()[0], "kernel_tuning": os.environ.get("CRIT_KERNEL_TUNING", "")}
    for name in ("torch", "transformers", "flash-linear-attention", "flashinfer-python", "triton"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    torch = sys.modules.get("torch")
    if found["torch"] is not None:
        try:
            import torch
            found["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:   # noqa: BLE001 - an environment without a usable device is keyed as such
            found["device"] = "unknown"
    return found


def test_key(registry, manifest):
    """The key of a plugin's test result: its folder's files, those of the plugins its tests name (reference, test_with),
    the source of crit's kernels it declares, crit's own plugin and Core packages (crit_plugins, crit_core) and the
    environment (_environment). A passing result under the same key is that test's result again."""
    import hashlib
    import json
    digest = hashlib.sha256()
    _tree_hash(digest, manifest["folder"])
    for other in sorted(set(manifest.get("reference") or ()) | set(manifest.get("test_with") or ())):
        if other in registry:
            _tree_hash(digest, registry[other]["folder"])
    if manifest.get("kernels"):
        try:
            from crit_kernels.build import manifests
            library = manifests()
        except Exception:   # noqa: BLE001 - without crit's kernel library the plugin tests its fallbacks
            library = {}
        for kernel in sorted(k["name"] for k in manifest["kernels"]):
            digest.update(kernel.encode())
            if kernel in library:
                _tree_hash(digest, library[kernel][0])
    package_root = Path(__file__).resolve().parents[1]
    for package in ("crit_plugins", "crit_core"):
        if (package_root / package).is_dir():
            _tree_hash(digest, package_root / package)
    digest.update(json.dumps(_environment(), sort_keys=True).encode())
    return digest.hexdigest()


def cache_folder():
    """Where passing plugin test results are kept: CRIT_PLUGIN_TEST_CACHE, else ~/.cache/crit/plugin-tests."""
    return Path(os.environ.get("CRIT_PLUGIN_TEST_CACHE") or Path.home() / ".cache" / "crit" / "plugin-tests")


def run_tests(registry, only=None, timeout=600, log=print, cache=False):
    """Each plugin's tests (unittest files test*.py in its tests folder), one plugin per process. Returns one item per
    plugin: {name, state, detail, seconds}; a plugin without tests fails, since every plugin is checked on its own.
    cache: skip a plugin whose tests passed before under the same key (test_key: its files, the plugins its tests
    use, crit's packages and the environment), reported as passed with `cached` true, and keep each new pass
    (cache_folder()); a library plugin copied unchanged into engine after engine is tested once per machine."""
    if not isinstance(registry, Registry):
        registry = Registry.discover(registry)
    package_root = str(Path(__file__).resolve().parents[1])
    items = []
    for manifest in registry:
        if only and manifest["id"] not in only:
            continue
        key = test_key(registry, manifest) if cache else None
        stored = cache_folder() / f"{key}.json" if key else None
        if stored is not None and stored.is_file():
            import json
            try:
                found = json.loads(stored.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                found = None
            if found and found.get("state") == "passed":
                items.append({"name": manifest["id"], "state": "passed", "detail": found.get("detail", ""),
                              "seconds": 0.0, "cached": True})
                log(f" passed  {manifest['id']:<32}   cached  {found.get('detail', '')} (unchanged since they passed)")
                continue
        started = time.perf_counter()
        tests = Path(manifest["tests"]).resolve()   # the tests run in the plugin's folder, so never relative
        files = sorted(tests.glob("test*.py")) if tests.is_dir() else []
        if not files:
            items.append({"name": manifest["id"], "state": "failed", "seconds": 0.0,
                          "detail": f"no tests: write unittest files test*.py in {tests.name}/, against the plugin's "
                                    "reference" + (" plugins" if manifest["reference"] else "")})
            log(f" failed  {manifest['id']}: no tests")
            continue
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([package_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            done = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(tests), "-t", str(tests),
                                   "-p", "test*.py"], cwd=Path(manifest["folder"]).resolve(), env=env, capture_output=True, text=True,
                                  timeout=timeout)
            output = done.stdout + done.stderr
            ran = RAN.search(output)
            count = int(ran.group(1)) if ran else 0
            passed = done.returncode == 0 and count > 0
            detail = f"{count} tests" if passed else (output.strip()[-2000:] or f"exit code {done.returncode}")
        except subprocess.TimeoutExpired:
            passed, detail = False, f"the tests ran past {timeout:g} s"
        seconds = time.perf_counter() - started
        items.append({"name": manifest["id"], "state": "passed" if passed else "failed", "detail": detail,
                      "seconds": round(seconds, 2)})
        if passed and stored is not None:
            import json
            try:
                stored.parent.mkdir(parents=True, exist_ok=True)
                stored.write_text(json.dumps({"state": "passed", "detail": detail, "plugin": manifest["id"]}),
                                  encoding="utf-8")
            except OSError:
                pass   # a cache that cannot be written leaves the next run to test again
        log(f" {'passed' if passed else 'failed'}  {manifest['id']:<32} {seconds:6.2f}s  "
            f"{detail if passed else detail.splitlines()[-1]}")
    return items
