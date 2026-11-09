"""Loading an engine's plugins: their modules write no bytecode into the package (a sealed package keeps its bytes),
the counting proxies leave the hot path once the loader's checks pass (every proxy a plugin keeps is replaced by its
target, one left in a closure passes calls through), the factories' parts tell which plugin fills each slot and the
facts it provides, and the assembly gathers its plugins' telemetry."""
from collections import namedtuple
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "cli"):
    sys.path.insert(0, str(ROOT / _folder))

from crit_plugins import PluginError, Registry, assemble  # noqa: E402
from crit_plugins.observe import Calls, Counted, Observed  # noqa: E402

KV = '''
from .helpers import first


class Pool:
    def write(self, value):
        return value + first.VALUE


class Kv:
    def value(self, x):
        return x + 1

    def pool(self):
        return Pool()

    def later(self):
        from .helpers import second   # imported once the engine runs, not when it loads
        return second.VALUE

    def telemetry(self):
        return {"kv": {"pages": 3}, "core": {"kv": True}}


def create(context, options, parts):
    return Kv()
'''

ATTENTION = '''
from collections import deque, namedtuple
import functools
import types

Pair = namedtuple("Pair", "kv pool")


class Holder:
    __slots__ = ("kv", "extra")

    def __init__(self, kv):
        self.kv, self.extra = kv, [kv]


class Attention:
    def __init__(self, parts):
        kv = parts["kv_cache"]
        self.parts, self.kv, self.pool = parts, kv, kv.pool()
        self.listed, self.paired, self.keyed = [kv, 1], (kv, (kv, 2)), {"kv": kv, "pool": self.pool}
        self.make, self.pair, self.holder = kv.pool, Pair(kv, self.pool), Holder(kv)
        self.closure = lambda x: kv.value(x)
        self.queue, self.space = deque([kv]), types.SimpleNamespace(kv=kv, deeper={"kv": [kv]})
        self.partial = functools.partial(lambda kv, x: kv.value(x), kv)
        self.by_key = {kv: "the kv plugin as a key"}
        self.bound = self.space.__init__   # a bound method: its object is looked into

    def run(self, x):
        return self.kv.value(x) + self.pool.write(x)

    def telemetry(self):
        return {"kv": {"pages": 99, "free_pages": 1}, "attention": {"runs": 2}}


def create(context, options, parts):
    return Attention(parts)
'''


def write_package(root):
    plugins = Path(root) / "plugins"
    for plugin_id, slot, text, extra in (("kv", "kv_cache", KV, {"sources": {"helpers": "code"}, "provides": ["kv:toy"]}),
                                         ("attn", "attention", ATTENTION, {"needs": ["kv:toy"]})):
        folder = plugins / plugin_id
        folder.mkdir(parents=True)
        manifest = {"schema": "crit-plugin-v1", "id": plugin_id, "title": f"Fixture {plugin_id}", "fills": [slot],
                    "entry": "plugin.py:create", **extra}
        (folder / "plugin.json").write_bytes(json.dumps(manifest).encode())
        (folder / "plugin.py").write_bytes(textwrap.dedent(text).encode())
    code = plugins / "kv" / "code"
    code.mkdir()
    (code / "first.py").write_bytes(b"VALUE = 10\n")
    (code / "second.py").write_bytes(b"VALUE = 20\n")
    return plugins


SELECTION = {"use": {"kv_cache": "kv", "attention": "attn"}}


# Plugins whose factories take device memory: `takes` bytes added to the context's held counter (what the loader's
# measure reads), or with `cuda` a real allocation; `declares` what their memory() says; the attention plugin's
# factory may first plan `plans` bytes onto the device through the placement plugin.
MEASURED = '''
def memory(context, options):
    return {"device": options["declares"]}


class Plugin:
    def plan(self, tensors, budget):
        return {name: "device" for name in tensors}


def create(context, options, parts):
    plugin = Plugin()
    if options["plans"]:
        parts["placement"].plan({"weights": options["plans"]}, {"device": 1 << 40, "host": 0})
    if options["cuda"]:
        import torch
        plugin.held = torch.zeros(options["takes"] + options["plans"], dtype=torch.uint8, device="cuda")
    elif context is not None:
        context["held"][0] += options["takes"] + options["plans"]
    return plugin
'''
OPTIONS = {name: {"type": "int", "default": 0, "min": 0} for name in ("declares", "takes", "plans", "cuda")}
MIB = 1 << 20


def write_measured(root):
    plugins = Path(root) / "plugins"
    for plugin_id, slot, extra in (("place", "placement", {}), ("kv", "kv_cache", {"provides": ["kv:toy"]}),
                                   ("attn", "attention", {"needs": ["kv:toy"]})):
        folder = plugins / plugin_id
        folder.mkdir(parents=True)
        manifest = {"schema": "crit-plugin-v1", "id": plugin_id, "title": f"Fixture {plugin_id}", "fills": [slot],
                    "entry": "plugin.py:create", "options": OPTIONS, **extra}
        (folder / "plugin.json").write_bytes(json.dumps(manifest).encode())
        (folder / "plugin.py").write_bytes(textwrap.dedent(MEASURED).encode())
    return plugins


class MeasuredMemoryTests(unittest.TestCase):
    """A plugin's declared device memory counts as planned only as far as its factory was measured to take it."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.registry = Registry.discover(write_measured(temp.name))
        self.context = {"held": [0]}

    def assemble(self, kv, attn, measure="counter", observe=None):
        selection = {"use": {"placement": "place", "kv_cache": "kv", "attention": "attn"},
                     "options": {"kv": kv, "attn": attn}}
        measure = (lambda: self.context["held"][0]) if measure == "counter" else measure
        return assemble(selection, self.registry, self.context, observe=observe, measure=measure)

    def test_an_honest_plugin_passes_and_an_over_declared_one_counts_what_it_took(self):
        assembly = self.assemble({"declares": 100 * MIB, "takes": 100 * MIB}, {"declares": 128 * MIB})
        self.assertEqual(assembly.measured, {"place": 0, "kv": 100 * MIB, "attn": 0})
        self.assertEqual(assembly.declared_device(), 100 * MIB, "the 128 MiB attn declared and never took cover nothing")
        self.assertEqual(assembly.unmeasured(), [])
        # Within the slack (a math library's workspace, small tables) a plugin passes, counted at its declaration.
        assembly = self.assemble({"declares": 100 * MIB, "takes": 130 * MIB}, {})
        self.assertEqual(assembly.declared_device(), 100 * MIB)

    def test_a_plugin_that_takes_more_than_it_declares_is_refused_naming_both(self):
        with self.assertRaisesRegex(PluginError, r"attn: its factory took 200\.0 MiB .* declares 0\.0 MiB"):
            self.assemble({}, {"takes": 200 * MIB})
        with self.assertRaisesRegex(PluginError, r"kv: its factory took 300\.0 MiB .* declares 100\.0 MiB"):
            self.assemble({"declares": 100 * MIB, "takes": 300 * MIB}, {})

    def test_what_a_factory_plans_through_the_placement_plugin_is_not_its_own_memory(self):
        # The model's weights: planned, then loaded in its factory. Plans are seen through the loader's proxies.
        assembly = self.assemble({}, {"plans": 2 << 30, "takes": 8 * MIB}, observe=Calls())
        self.assertEqual(assembly.measured["attn"], 8 * MIB)
        with self.assertRaisesRegex(PluginError, "attn: its factory took"):
            self.assemble({}, {"plans": 2 << 30, "takes": 200 * MIB}, observe=Calls())

    def test_a_device_whose_memory_cannot_be_read_leaves_the_declarations_unmeasured(self):
        assembly = self.assemble({"declares": 100 * MIB, "takes": 300 * MIB}, {"declares": 8 * MIB},
                                 measure=lambda: None)
        self.assertEqual(assembly.measured, {"place": None, "kv": None, "attn": None})
        self.assertEqual(assembly.declared_device(), 108 * MIB)
        self.assertEqual(assembly.unmeasured(), ["kv", "attn"])

    def test_the_default_measure_reads_cuda_allocations(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device")
        assembly = self.assemble({"declares": 16 * MIB, "takes": 16 * MIB, "cuda": 1}, {"declares": 64 * MIB},
                                 measure=None)
        self.assertEqual(assembly.measured["kv"], 16 * MIB)
        self.assertEqual(assembly.declared_device(), 16 * MIB)
        assembly.close()
        with self.assertRaisesRegex(PluginError, "attn: its factory took 256.0 MiB"):
            self.assemble({}, {"takes": 256 * MIB, "cuda": 1}, measure=None)


class BytecodeTests(unittest.TestCase):
    def test_loading_runs_the_source_never_cached_bytecode_beside_it(self):
        # A __pycache__ beside a plugin's source (or under PYTHONPYCACHEPREFIX) may hold other code than the package's
        # bytes: the loader compiles the source every time.
        import importlib.util
        import marshal
        with tempfile.TemporaryDirectory() as temp:
            plugins = write_package(temp)
            first = plugins / "kv" / "code" / "first.py"
            stale = Path(importlib.util.cache_from_source(str(first)))
            stale.parent.mkdir(parents=True)
            code = compile("VALUE = 999\n", str(first), "exec")
            stat = first.stat()
            header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little") + \
                int(stat.st_mtime).to_bytes(4, "little") + (stat.st_size & 0xFFFFFFFF).to_bytes(4, "little")
            stale.write_bytes(header + marshal.dumps(code))   # bytecode a source-checking loader would accept
            assembly = assemble(SELECTION, Registry.discover(plugins))
            self.assertEqual(assembly["attention"].run(1), 13, "first.VALUE is the source's 10, not the cache's 999")

    def test_loading_a_packages_plugins_writes_no_bytecode_into_it(self):
        from crit.records import inventory, verify_inventory
        before = sys.dont_write_bytecode
        sys.dont_write_bytecode = False   # a debug script run without -B or PYTHONDONTWRITEBYTECODE
        try:
            with tempfile.TemporaryDirectory() as temp:
                plugins = write_package(temp)
                sealed = inventory(Path(temp))
                assembly = assemble(SELECTION, Registry.discover(plugins))
                self.assertEqual(assembly["kv_cache"].later(), 20)
                self.assertEqual(assembly["attention"].run(1), 13)
                self.assertEqual(list(Path(temp).rglob("__pycache__")), [])
                verify_inventory(Path(temp), sealed)   # what crit run and crit serve check before they serve it
        finally:
            sys.dont_write_bytecode = before


class UnobserveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.calls = Calls()
        self.assembly = assemble(SELECTION, Registry.discover(write_package(temp.name)), observe=self.calls)
        self.attention = self.assembly["attention"]

    def test_the_proxies_count_until_the_checks_pass(self):
        a = self.attention
        self.assertIsInstance(a.kv, Observed)
        self.assertIsInstance(a.pool, Observed, "a pool the kv plugin made is behind its slot's proxy")
        self.assertIsInstance(a.make, Counted)
        self.calls.reset()
        a.run(1)
        self.assertEqual(self.calls["kv_cache"], 2)

    def test_unobserve_puts_every_proxy_a_plugin_keeps_back_to_its_target(self):
        a, kv = self.attention, self.assembly["kv_cache"]
        replaced = self.assembly.unobserve()
        self.assertGreaterEqual(replaced, 10)
        self.assertIs(a.kv, kv)
        self.assertIs(type(a.pool).__name__, "Pool")
        self.assertIs(a.listed[0], kv)
        self.assertIs(a.paired[0], kv)
        self.assertIs(a.paired[1][0], kv)
        self.assertIs(a.keyed["kv"], kv)
        self.assertIs(type(a.keyed["pool"]).__name__, "Pool")
        self.assertEqual(a.make.__self__, kv)
        self.assertIs(a.pair.kv, kv)
        self.assertEqual(type(a.pair).__name__, "Pair", "a named tuple stays one")
        self.assertIs(a.holder.kv, kv)
        self.assertIs(a.holder.extra[0], kv)
        self.assertIs(a.parts["kv_cache"], kv, "the factories' parts hand out the plugins themselves")
        self.assertIs(a.closure.__closure__[0].cell_contents, kv, "a closure's proxy too")
        self.assertIs(a.queue[0], kv)
        self.assertIs(a.space.kv, kv)
        self.assertIs(a.space.deeper["kv"][0], kv)
        self.assertIs(a.partial.args[0], kv)
        self.assertEqual(list(a.by_key), [kv])
        self.calls.reset()
        self.assertEqual(a.closure(1), 2)
        self.assertEqual(a.partial(1), 2)
        self.assertEqual(a.run(1), 13)
        self.assertEqual(sum(self.calls.values()), 0)
        self.assertEqual(self.assembly.unobserve(), 0, "once is enough")

    def test_a_proxy_left_where_core_cannot_reach_passes_calls_through(self):
        kept = self.attention.kv   # a reference held outside every plugin object
        self.assembly.unobserve()
        self.calls.reset()
        self.assertIsInstance(kept, Observed)
        self.assertEqual(kept.value(1), 2)
        self.assertEqual(sum(self.calls.values()), 0)

    def test_unobserve_looks_inside_torch_modules(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from crit_plugins.observe import unobserve
        kv = self.assembly["kv_cache"]
        proxy = self.attention.kv
        layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        layers[0].kv = proxy
        holder = torch.nn.ModuleDict({"layers": layers})
        holder.weights = torch.zeros(3)
        self.assertGreaterEqual(unobserve([holder]), 1)
        self.assertIs(holder["layers"][0].kv, kv)

    def test_the_parts_name_each_slots_plugin_and_its_facts(self):
        parts = self.assembly.parts
        self.assertEqual(parts.slots(), ["kv_cache", "attention"])
        self.assertEqual(parts.plugin("attention"), "attn")
        self.assertIsNone(parts.plugin("sampler"))
        self.assertEqual(parts.facts("kv_cache"), ("kv:toy",))
        self.assertEqual(parts.facts("sampler"), ())

    def test_the_assembly_gathers_its_plugins_telemetry(self):
        self.assertEqual(self.assembly.telemetry(), {"kv": {"pages": 3, "free_pages": 1}, "core": {"kv": True},
                                                     "attention": {"runs": 2}})


class ObservedTupleTests(unittest.TestCase):
    def test_a_tuple_holding_no_proxy_is_kept_as_it_is(self):
        from crit_plugins.observe import unobserve
        Pair = namedtuple("Pair", "a b")
        holder = {"pair": Pair(1, (2, 3)), "list": [Pair(4, 5)]}
        kept = holder["pair"]
        self.assertEqual(unobserve([holder]), 0)
        self.assertIs(holder["pair"], kept)


if __name__ == "__main__":
    unittest.main()
