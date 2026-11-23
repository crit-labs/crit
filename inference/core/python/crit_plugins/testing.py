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
from .device import StepBatch, pool_missing
from .manifest import HOST, MANIFEST
from .memory import POISON
from .slots import compare, model_problems


@contextmanager
def compiles():
    """Every kernel compile and extension build while the block runs, as a list of descriptions it fills
    (crit_plugins.compiles.Watch: Triton JIT compiles, Triton cache entries made outside triton.jit, torch extension
    builds and loads). It uses only what is already loaded (triton, torch's cpp_extension), so it reads nothing a
    device without them would have. A plugin compiles at load, never on a request: a test wraps a request after load
    in it and expects no entry, as Core's loader watches every engine once it serves."""
    from .compiles import Watch
    found = []
    watch = Watch()
    try:
        yield found
    finally:
        found.extend(watch.stop())


@functools.lru_cache(maxsize=None)
def _plugin_folder(path):
    for folder in Path(path).resolve().parents:
        if (folder / MANIFEST).is_file():
            return folder
    raise ValueError(f"{path} lies in no plugin folder")


# A skip that leaves the plugin's correctness unchecked says so in its reason (unchecked()): the plugin's comparison
# with its reference, or the engine run on Core's runtime, did not run although the plugin runs on a device here.
# crit plugins test fails such a skip. Any other skip is the machine's (a CUDA-only plugin's tests without CUDA) and
# passes, listed.
UNCHECKED = "unchecked: "
NO_CORE_RUNTIME = ("Core's runtime crit_core_steps is not built: `crit core build` builds it "
                   "(CRIT_CORE_STEP_LIBRARY names another)")


def unchecked(reason):
    """A unittest.SkipTest that crit plugins test fails: raise it where a test skips without checking what the plugin
    is tested for (a model plugin's comparison with its reference, the engine run on Core's runtime) although the
    plugin runs on a device here. A skip because no device the plugin runs on is here stays a plain skipTest()."""
    return unittest.SkipTest(UNCHECKED + reason)


def core_runtime():
    """The crit_core_steps library a test that runs the engine on Core's runtime loads (crit_core.native's
    step_library: CRIT_CORE_STEP_LIBRARY, else the checkout's build, else the one `crit core build` built). Without
    one it raises unchecked(NO_CORE_RUNTIME): the engine does not run, so the test checks nothing, and crit plugins
    test fails, saying how to build it. For setUpClass; a test method calls PluginTestCase.core_runtime()."""
    from crit_core.native import step_library
    found = step_library()
    if found is None:
        raise unchecked(NO_CORE_RUNTIME)
    return found


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
        Inference folder for a technique of crit's library), whatever group each lies in."""
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

    def core_runtime(self):
        """The crit_core_steps library a test that runs the engine on Core's runtime loads (crit_core.native's
        step_library: CRIT_CORE_STEP_LIBRARY, else the checkout's build, else the one `crit core build` built). The
        test is skipped, with how to build it, when there is none, and crit plugins test fails that skip
        (core_runtime(): unchecked)."""
        return core_runtime()

    def skipUnchecked(self, reason):
        """Skip the test as unchecked(reason), which crit plugins test fails: for a skip that leaves the plugin's
        correctness unchecked although it runs here, never for a missing device (skipTest, device())."""
        raise unchecked(reason)

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
        registry = self.registry()
        contracts = registry.contracts
        manifest = registry[self.plugin_id()]
        on_device = [d for d in manifest["devices"] if d != HOST]
        for slot in manifest["fills"]:
            lacking = contracts.host_missing(slot, assembly[slot])
            if lacking:
                self.fail(f"{self.plugin_id()} in {slot} lacks {', '.join(lacking)} of the {slot} interface")
            lacking = contracts.device_missing(slot, assembly[slot]) if on_device else []
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
                if not contracts.known(fact):
                    continue
                lacking = contracts.extension_missing(fact, assembly[slot],
                                                      self._made(slot, assembly[slot], fact, contracts))
                if lacking:
                    self.fail(f"{self.plugin_id()} provides {fact} but lacks {', '.join(lacking)} "
                              f"(crit_plugins.contracts: {contracts.by.get(('the fact', fact), 'Core')} defines it)")

    def _made(self, slot, plugin, fact, contracts):
        """The objects the slot's device form returns that a pool fact's methods live on (the fact's pool): a small
        pool."""
        if contracts.pools.get(fact) != slot:
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
        registry = self.registry()
        fills = registry[self.plugin_id()]["fills"]
        slots = [slot for slot in (cases or fills) if registry.contracts.case(slot) is not None]
        if not slots:
            self.fail(f"crit keeps no cases for {', '.join(cases or fills)}: compare the plugin with its reference in "
                      "its own tests, or name the slots whose cases see it (cases=)")
        for slot in slots:
            plugin = self.assemble(use, options, context, facts, use_options=use_options)
            reference = self.reference(use, reference_options, context, facts, use_options)
            made = registry.contracts.case(slot)
            found = compare(made(plugin), made(reference), atol, rtol, slot)
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
# A skipped test in unittest's verbose output: "name (module.Class.name)", its docstring's first line on the next
# line when it has one, then " ... skipped 'reason'".
SKIPPED = re.compile(r"^(\w+) \(([\w.]+?)(?:\.\1)?\)(?:\n[^\n]*?)? \.\.\. skipped (.*)$", re.M)


def _skips(output):
    """[{test, reason, unchecked}] of the tests unittest's verbose output reports skipped; unchecked: the skip was
    unchecked()'s (its UNCHECKED prefix taken off the reason)."""
    import ast
    found = []
    for name, where, reason in SKIPPED.findall(output):
        try:
            reason = str(ast.literal_eval(reason))
        except (ValueError, SyntaxError):
            pass   # shown as printed
        hides = reason.startswith(UNCHECKED)
        found.append({"test": f"{where}.{name}", "reason": reason[len(UNCHECKED):] if hides else reason,
                      "unchecked": hides})
    return found


def _skip_line(skip):
    return (f"           skipped {skip['test']}: {skip['reason']}"
            + (" (unchecked: fails the plugin)" if skip.get("unchecked") else ""))


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
    """Where passing plugin test results are kept: CRIT_PLUGIN_TEST_CACHE, else crit_core.cache_home()'s plugin-tests."""
    from crit_core import cache_home
    return Path(os.environ.get("CRIT_PLUGIN_TEST_CACHE") or cache_home("plugin-tests"))


# Plugins whose tests run at once (one process each): CRIT_PLUGIN_TEST_JOBS, else half the cores up to 4 (a plugin's
# tests may use the GPU, which the processes share).
TEST_JOBS = "CRIT_PLUGIN_TEST_JOBS"


def test_jobs():
    given = os.environ.get(TEST_JOBS, "").strip()
    if given:
        return max(1, int(given))
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def run_tests(registry, only=None, timeout=600, log=print, cache=False, jobs=None):
    """Each plugin's tests (unittest files test*.py in its tests folder), one plugin per process, `jobs` plugins at
    once (test_jobs()). Returns one item per plugin, in the registry's order, each logged in that order:
    {name, state, detail, seconds, skipped}; a plugin without tests fails, since every plugin is checked on its
    own. skipped lists the tests that skipped ({test, reason, unchecked}), each also logged. The machine's skip is no
    failure (a CUDA-only plugin's tests skip without CUDA), but "passed" does not cover what it skipped; an unchecked
    skip (unchecked(): Core's runtime missing, a model plugin's reference comparison not run) fails a plugin whose
    tests otherwise passed, and its item lists those skips again as `unchecked`.
    cache: skip a plugin whose tests passed before under the same key (test_key: its files, the plugins its tests
    use, crit's packages and the environment), reported as passed with `cached` true, and keep each new pass
    (cache_folder()); a library plugin copied unchanged into engine after engine is tested once per machine."""
    from concurrent.futures import ThreadPoolExecutor
    if not isinstance(registry, Registry):
        registry = Registry.discover(registry)
    package_root = str(Path(__file__).resolve().parents[1])
    chosen = [m for m in registry if not only or m["id"] in only]
    jobs = max(1, jobs or test_jobs())

    def one(manifest):
        """(item, its log lines)."""
        lines = []
        key = test_key(registry, manifest) if cache else None
        stored = cache_folder() / f"{key}.json" if key else None
        if stored is not None and stored.is_file():
            import json
            try:
                found = json.loads(stored.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                found = None
            if found and found.get("state") == "passed":
                skipped = found.get("skipped") or []
                lines.append(f" passed  {manifest['id']:<32}   cached  {found.get('detail', '')} (unchanged since they "
                             "passed)")
                lines += [_skip_line(skip) for skip in skipped]
                return {"name": manifest["id"], "state": "passed", "detail": found.get("detail", ""),
                        "seconds": 0.0, "cached": True, "skipped": skipped}, lines
        started = time.perf_counter()
        tests = Path(manifest["tests"]).resolve()   # the tests run in the plugin's folder, so never relative
        files = sorted(tests.glob("test*.py")) if tests.is_dir() else []
        if not files:
            return {"name": manifest["id"], "state": "failed", "seconds": 0.0,
                    "detail": f"no tests: write unittest files test*.py in {tests.name}/, against the plugin's "
                              "reference" + (" plugins" if manifest["reference"] else "")}, \
                [f" failed  {manifest['id']}: no tests"]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([package_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault(POISON, "1")   # a scratch hands out poisoned bytes: no call may read what another left
        try:
            done = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-v", "-s", str(tests),
                                   "-t", str(tests), "-p", "test*.py"], cwd=Path(manifest["folder"]).resolve(),
                                  env=env, capture_output=True, text=True, timeout=timeout)
            output = done.stdout + done.stderr
            ran = RAN.search(output)
            count = int(ran.group(1)) if ran else 0
            ran = done.returncode == 0 and count > 0
            skipped = _skips(output)
            hidden = [skip for skip in skipped if skip["unchecked"]] if ran else []
            passed = ran and not hidden
            detail = (f"{count} tests" + (f", {len(skipped)} skipped" if skipped else "") if ran
                      else f"exit code {done.returncode}\n{output.strip()[-2000:]}".rstrip())
            if hidden:
                reasons = {}
                for skip in hidden:
                    reasons.setdefault(skip["reason"], []).append(skip["test"])
                detail += f"; {len(hidden)} of them leave its correctness unchecked: " + "; ".join(
                    f"{reason} ({tests[0]}" + (f" and {len(tests) - 1} more)" if len(tests) > 1 else ")")
                    for reason, tests in reasons.items())
        except subprocess.TimeoutExpired:
            passed, detail, skipped, hidden = False, f"the tests ran past {timeout:g} s", [], []
        seconds = time.perf_counter() - started
        item = {"name": manifest["id"], "state": "passed" if passed else "failed", "detail": detail,
                "seconds": round(seconds, 2), "skipped": skipped, **({"unchecked": hidden} if hidden else {})}
        if passed and stored is not None:
            import json
            try:
                stored.parent.mkdir(parents=True, exist_ok=True)
                stored.write_text(json.dumps({"state": "passed", "detail": detail, "plugin": manifest["id"],
                                              "skipped": skipped}), encoding="utf-8")
            except OSError:
                pass   # a cache that cannot be written leaves the next run to test again
        lines.append(f" {'passed' if passed else 'failed'}  {manifest['id']:<32} {seconds:6.2f}s  "
                     f"{detail if passed or hidden else detail.splitlines()[-1]}")
        lines += [_skip_line(skip) for skip in skipped]
        return item, lines

    items = []
    with ThreadPoolExecutor(max_workers=min(jobs, max(1, len(chosen)))) as pool:
        for item, lines in pool.map(one, chosen):   # in the registry's order, each as soon as those before it are
            items.append(item)
            for line in lines:
                log(line)
    return items


class _Cases:
    """The comparison cases of the slots of the library a test file lies in, by slot (cases_for)."""

    def __init__(self, path):
        self._path = path
        self._found = None

    def _contracts(self):
        if self._found is None:
            from .contracts import Contracts
            self._found = Contracts.of(Registry.discover(plugins_root(_plugin_folder(self._path))))
        return self._found

    def __getitem__(self, slot):
        found = self._contracts().case(slot)
        if found is None:
            raise KeyError(f"no plugin of the library defines cases for {slot}")
        return found

    def __contains__(self, slot):
        return self._contracts().case(slot) is not None

    def __iter__(self):
        return iter(self._contracts().cases)

    def module(self, slot):
        """The module of `slot`'s cases, with the helpers they share (Contracts.case_module)."""
        return self._contracts().case_module(slot)


def cases_for(path):
    """The comparison cases of the slots of the folder of plugins the file at `path` lies in, as {slot:
    function(parts)}: Core's slots' and those its plugins define (crit_plugins.contracts)."""
    return _Cases(path)
