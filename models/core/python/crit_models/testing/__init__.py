"""Whole engines of a model plugin, for its tests: a small random checkpoint of a transformers architecture, the model
plugin beside library plugins in a package, run by Core's loader on Core's concurrent runtime (which refuses an engine
that misses a serving contract: its batching, every computing slot reached, its plans, its warm requests).

    from pathlib import Path
    from crit_models import testing as t

    class MyModelTests(t.Layout):
        plugin = Path(__file__).resolve().parents[1]          # the model plugin's folder (tests/ lies in it)

        @classmethod
        def checkpoint(cls, root):
            config = t.transformers.MyConfig(vocab_size=256, hidden_size=128, ...)
            return cls.save(t.transformers.MyForCausalLM(config), root)

    class MyManifestTests(t.Manifest):                        # the host checks of its manifest and layout
        plugin = Path(__file__).resolve().parents[1]

GgufEngines (the checkpoint written as GGUF, by the layout's gguf entry) and Respelled (its tensors under another
spelling, which names rules map back: the plugin's own, or ones the test's layout adds) put a class over another
checkpoint layout, as each model plugin's tests/checkpoints.json lists them (tests/test_model_neutral.py).

Layout holds the checks every model plugin's engine passes: its fixed-text scores follow transformers'
log-probabilities of the same checkpoint, its greedy continuations are transformers' choices up to BF16 rounding,
requests in flight together decode what each decodes alone, its prompt spans are timed at load, and the KV it
resolves before a load is the KV it loads. The engines run on CUDA where torch has it and on the CPU otherwise
(CRIT_TEMPLATE_DEVICE names one): there the norms are torch-norms', decoding goes through step() (captured rounds are
a GPU's) and no span is timed, the arithmetic checked against transformers is the same. Needs transformers,
safetensors and Core's crit_core_steps: without them the engines skip unchecked, which fails crit plugins test
(crit_plugins.testing.unchecked). A class skips plainly where transformers lacks its architecture or its plugins do
not run on the device.

The package is the model plugin as crit plugins add copies it (with the models core it extends) where the plugin is
crit's library's, and the engine's own plugins folder where it is an engine's; the plugins the class names (library)
and the plugin's tests use (test_with) are added from the library."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest

from crit_plugins.testing import PluginTestCase, core_runtime, unchecked

try:
    import torch
except ImportError:
    torch = None
try:
    import transformers
    import safetensors.torch  # noqa: F401
except ImportError:
    transformers = None

GREEDY = {"temperature": 0.0}
BASE = ["attention-state", "softmax-attention", "projections-dense", "token-sampling", "device-placement",
        "fused-norms", "memory-table-rows"]
# The library plugins every engine here may load (fused-norms' base, and its stand-in on the CPU, is torch-norms):
# a model plugin's test_with names them beside the ones its own classes name, so crit plugins add brings them.
PLUGINS = BASE + ["torch-norms"]
# The plain plugin of each slot a model plugin asks; the model slot is the class's plugin.
USE = {"placement": "device-placement", "projections": "projections-dense",
       "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
       "norms": "fused-norms", "embeddings": "memory-table-rows"}
from crit_models.engines import ROUTED as MOE, SMALL_OPTIONS as OPTIONS   # noqa: E402,F401 - one definition (MOE re-exported for the model plugins' tests)


def engine_device():
    """The device the engines run on: CRIT_TEMPLATE_DEVICE, else CUDA where torch has it, else the CPU."""
    return os.environ.get("CRIT_TEMPLATE_DEVICE") or ("cuda" if torch is not None and torch.cuda.is_available()
                                                      else "cpu")


class Manifest(PluginTestCase):
    """What a model plugin's folder says, checked on the host without a checkpoint: a class names the plugin's
    folder (plugin) and gets these."""
    plugin = None

    def plugin_folder(self):
        return Path(self.plugin)

    def setUp(self):
        if self.plugin is None:
            self.skipTest("the harness: a model plugin's class names its folder")

    def test_it_fills_the_model_slot_for_the_model_types_it_names_over_the_models_core(self):
        from crit_models.plugin import CORE
        manifest = self.registry()[self.plugin_id()]
        self.assertEqual(manifest["fills"], ["model"])
        self.assertEqual(manifest["extends"]["id"], CORE)
        self.assertEqual(manifest["options"].keys(), self.registry()[CORE]["options"].keys())
        types = [need for need in manifest["needs"] if all(fact.startswith("model_type:") for fact in need)]
        self.assertTrue(types, "needs names the config model_types the plugin serves")

    def test_its_layout_composes_techniques_the_core_or_the_plugin_gives(self):
        from crit_models.plugin import layout
        self.assertIn("mixers", layout(self.plugin))   # checked: a problem raises with its name

    def test_it_provides_the_draft_head_exactly_where_its_layout_has_a_draft_layer(self):
        from crit_models.plugin import layout
        provides = self.registry()[self.plugin_id()]["provides"]
        drafts = "draft" in layout(self.plugin)
        self.assertEqual(("draft:head" in provides, "draft:head-device" in provides), (drafts, drafts))


class Engines(PluginTestCase):
    """A package of a model plugin and the library plugins it needs, and engines of it over one checkpoint."""
    plugin = None   # the model plugin's folder: a class names it (tests/ lies in it)
    library, use, options, plugin_options = BASE, USE, OPTIONS, {}
    devices = ("cuda", "cpu")   # where the selection's plugins run: a class over a CUDA-only plugin names cuda alone
    reference_fp32 = False
    # Scores judged on average and by each position's most likely token rather than position by position: where a
    # BF16 rounding of the recurrent state or a route moves a token far down by up to a nat under the sharpened head
    # (crit's hybrid engine test judges the same architecture so).
    on_average = False
    sharp_routes = False   # the router's weights scaled up: a near-tie route no rounding flips
    # Attention that picks: the query and key norms' weights set to 3 (the projections scaled where a model has no
    # such norms) and the value and output projections scaled by 3. A small random model attends every position
    # almost alike, so its logits barely depend on which positions a token may read, and a test of a model that
    # limits them (an indexer's blocks, a window) passes whatever the limit picks. With sharp attention the pick
    # decides the logits; rounding then moves a few positions further, so such a class judges on_average.
    sharp_attention = False
    # The share of a text's positions that may miss the per-position check (on_average): where an indexer's cut falls
    # between two blocks that score alike, a rounding picks the other block and the position reads other KV.
    flips = 0.0
    mean_limit = 0.15   # on_average: the most the positions' log-probabilities may differ from the reference's, mean

    @classmethod
    def checkpoint(cls, root):
        raise NotImplementedError

    def plugin_folder(self):
        return Path(self.plugin)   # also for a class in another folder that varies a plugin's class

    @classmethod
    def setUpClass(cls):
        if cls.checkpoint.__func__ is Engines.checkpoint.__func__:
            raise unittest.SkipTest("the harness: a model plugin's class names its checkpoint")
        if torch is None:
            raise unittest.SkipTest("whole engines need torch")
        cls.device = engine_device()
        if cls.device not in cls.devices:
            raise unittest.SkipTest(f"its plugins run on {', '.join(cls.devices)}, the engines here on {cls.device}")
        # The engines run here, so what keeps them from running leaves the plugin's correctness unchecked: Core's
        # runtime, and transformers, the reference its scores and continuations are compared with.
        core_runtime()
        if transformers is None:
            raise unchecked("transformers and safetensors, the reference the engines are compared with "
                            "(crit's reference extra)")
        try:
            from crit.plugins import IGNORE, add, library
        except ImportError:
            raise unittest.SkipTest("crit's command line (crit plugins add)") from None
        from crit_plugins.assemble import Registry, plugins_root
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        model = cls.checkpoint(root)
        if model is None:
            raise unittest.SkipTest(f"transformers {transformers.__version__} lacks the architecture")
        cls.package = root / "package"
        folder = Path(cls.plugin).resolve()
        manifest = json.loads((folder / "plugin.json").read_text(encoding="utf-8"))
        cls.model_id = manifest["id"]
        wanted = set(cls.library) | set(manifest.get("test_with", []))
        registry = library()
        if cls.model_id in registry and Path(registry[cls.model_id]["folder"]).resolve() == folder:
            add(cls.package / "plugins", sorted(wanted | {cls.model_id}))   # with the core it extends
        else:   # an engine's model plugin: the engine's plugins as they are, and what the class adds to them
            shutil.copytree(plugins_root(folder), cls.package / "plugins", ignore=IGNORE)
            lacking = sorted(wanted - {held["id"] for held in Registry.discover(cls.package / "plugins")})
            if lacking:
                add(cls.package / "plugins", lacking)
        (cls.package / "configuration.json").write_text(json.dumps({"plugins": cls.selection()}), encoding="utf-8")
        cls._serve_layout()
        cls.engines = []
        cls.engine = cls.load()

    @classmethod
    def served_layout(cls, layout):
        """The layout the engines compose (a copy of the plugin's own, layout.json): a class over another checkpoint
        layout may extend it for the test (Respelled: the names rules that map its spelling back), never the
        plugin's file."""
        return layout

    @classmethod
    def _serve_layout(cls):
        """The package's copy of the model plugin given served_layout()'s layout where it differs from the plugin's."""
        from crit_plugins.assemble import Registry
        path = Path(Registry.discover(cls.package / "plugins")[cls.model_id]["folder"]) / "layout.json"
        own = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        served = cls.served_layout(json.loads(json.dumps(own)))
        if served != own:
            path.write_text(json.dumps(served, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def selection(cls, use=None, plugins=None, **options):
        """The engine's selection: `use` over the class's, `plugins` the other plugins' options over the class's
        ({plugin: {option: value}}), `options` the model plugin's."""
        chosen = {"model": cls.model_id, **cls.use, **(use or {})}
        if cls.device != "cuda" and chosen.get("norms") == "fused-norms":
            chosen["norms"] = "torch-norms"   # fused-norms is a GPU's kernel: the base plugin's arithmetic on the CPU
        others = {name: {**cls.plugin_options.get(name, {}), **(plugins or {}).get(name, {})}
                  for name in {**cls.plugin_options, **(plugins or {})}}
        return {"use": chosen, "base": {"norms": "torch-norms"},
                "options": {cls.model_id: {**cls.options, **options}, **others}}

    @classmethod
    def load(cls, use=None, plugins=None, **options):
        from crit_core.engine import load
        changed = use or plugins or options
        engine = load(cls.package, cls.model, {"plugins": cls.selection(use, plugins, **options)} if changed else None,
                      device=cls.device)
        cls.engines.append(engine)
        return engine

    @classmethod
    def tearDownClass(cls):
        for engine in getattr(cls, "engines", []):
            engine.close()
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    @classmethod
    def save(cls, model, root, fp32=False):
        with torch.no_grad():
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
            names = [name for name, _ in model.named_parameters()]
            normed = any(name.endswith("self_attn.q_norm.weight") for name in names)
            for name, parameter in model.named_parameters():
                if cls.sharp_routes and name.endswith("mlp.gate.weight"):
                    parameter.mul_(20)
                if cls.sharp_attention:
                    if name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
                        parameter.fill_(3.0)
                    elif name.endswith(("self_attn.v_proj.weight", "self_attn.o_proj.weight")) or (
                            not normed and name.endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight"))):
                        parameter.mul_(3.0)
        model.to(torch.bfloat16)
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        cls.reference = (model.float() if cls.reference_fp32 or fp32 else model).to(cls.device).eval()
        return cls.model

    def prompts(self, count, length=17):
        g = torch.Generator().manual_seed(count * 101 + length)
        return [torch.randint(3, 256, (length + 4 * i,), generator=g).tolist() for i in range(count)]

    def logits(self, ids):
        with torch.no_grad():
            return self.reference(torch.tensor([ids], device=self.device)).logits[0].float()

    def assertScores(self, engine, ids, limit=0.25):
        scored = engine.score(ids, 1, "prefill", top_k=1)
        logp = self.logits(ids)[:len(ids) - 1].log_softmax(-1)
        expected = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].tolist()
        differences = [abs(a - b) for a, b in zip(scored.target, expected)]
        if not self.on_average:
            self.assertLess(max(differences), limit, differences)
            return
        self.assertLess(sum(differences) / len(differences), self.mean_limit, differences)
        best, token = logp.max(-1)
        missed = []
        for i, ((found, value),) in enumerate(scored.top):
            if found != int(token[i]):   # only a near-tie may rank them the other way
                if float(best[i]) - float(logp[i].topk(2).values[1]) >= 0.5:
                    missed.append((i, found, int(token[i])))
            elif abs(value - float(best[i])) >= 0.5:
                missed.append((i, value, float(best[i])))
        self.assertLessEqual(len(missed), int(self.flips * len(differences)), missed)


class Layout(Engines):
    """The checks every model plugin's engine passes."""

    def test_the_kv_it_resolves_before_any_load_is_the_kv_it_loads(self):
        # Core asks the plugins what the engine will be before a load (their resolve()): the layout's pools from the
        # checkpoint's headers alone, priced by the kv_cache plugin, are the pools the load makes, so a plan's
        # kv_bytes_per_token is checked before any load (and the loader refuses an engine that is not what it
        # resolved).
        from crit_core.engine import resolve_package
        resolved = resolve_package(self.package, self.model, device=self.device)
        stated = self.engine.info().get("kv_bytes_per_token")
        self.assertIsNotNone(stated)
        self.assertEqual(resolved["facts"].get("kv_bytes_per_token"), stated)
        self.assertEqual(resolved["by"]["kv_bytes_per_token"], self.model_id)
        self.assertEqual(self.engine.telemetry()["load"]["resolved"]["kv_bytes_per_token"], stated)

    def test_scores_follow_transformers_log_probabilities(self):
        (ids,) = self.prompts(1, 150)   # three prompt spans, the KV and any recurrent state carried between them
        self.assertScores(self.engine, ids)

    def test_greedy_continuations_are_transformers_choices(self):
        # Each token, given the ones before it, is transformers' argmax unless two logits tie within BF16's step. A
        # class that says a rounding may flip a selection (flips) is allowed that share of the tokens, one at least.
        missed, count = [], 0
        for prompt in self.prompts(3):
            tokens, _ = self.engine.generate(prompt, 12, stop_tokens=[], sampling=GREEDY)
            logits = self.logits(prompt + tokens)
            for i, token in enumerate(tokens):
                row, count = logits[len(prompt) - 1 + i], count + 1
                if float(row[token]) < float(row.max()) - 0.25:
                    missed.append((i, token, int(row.argmax()), float(row[token]), float(row.max())))
        self.assertLessEqual(len(missed), max(1, int(self.flips * count)) if self.flips else 0, missed)

    def test_requests_in_flight_together_decode_what_each_decodes_alone(self):
        prompts = self.prompts(6, 9)
        alone = [self.engine.generate(p, 10, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        with ThreadPoolExecutor(6) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 10, stop_tokens=[], sampling=GREEDY)[0],
                                     prompts))
        self.assertEqual(together, alone)

    def test_a_seeded_sampled_request_repeats_and_decodes_alone_as_together(self):
        sampling = {"temperature": 1.0, "top_p": 0.9, "seed": 7}
        prompts = self.prompts(3, 11)
        alone = [self.engine.generate(p, 12, stop_tokens=[], sampling=sampling)[0] for p in prompts]
        self.assertEqual(self.engine.generate(prompts[0], 12, stop_tokens=[], sampling=sampling)[0], alone[0])
        with ThreadPoolExecutor(3) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 12, stop_tokens=[], sampling=sampling)[0],
                                     prompts))
        self.assertEqual(together, alone)

    def test_prompts_run_layer_by_layer_decode_and_score_what_they_do_chunk_by_chunk(self):
        # Passes (passes.py): a prompt's spans run through one layer, then the next, over as many of Core's steps as
        # that takes (here a chunk through a layer a step). Every chunk through a layer is the call it is chunk by
        # chunk, so replies and scores are the same bits; a pass's steps consume nothing until its last, requests
        # beside it decode on, and a request cancelled inside a pass leaves no state behind.
        # The prompts are a few chunks of 16 tokens each, the last one partial (40, 44 and 20 tokens: 3, 3 and 2
        # chunks; the scored text 70: 5): a pass's steps are its chunks x the model's layers, so longer prompts run
        # the same calls more often, not other ones. The cancelled prompt stays 200 tokens (13 chunks): at 90 (6) a
        # fast host ended its pass before the cancel looked, and the request finished instead.
        options = dict(prefill_chunk=16, span_seconds=1e-6)
        passed, plain = self.load(passes="on", **options), self.load(passes="off", **options)
        self.assertEqual((passed.model.passes, plain.model.passes), (True, False))
        prompts = self.prompts(2, 40) + self.prompts(1, 20)
        want = [plain.generate(p, 8, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        before, warm = passed.telemetry()["core"]["steps"], dict(passed.model.passed)   # the load's requests passed too
        self.assertEqual([passed.generate(p, 8, stop_tokens=[], sampling=GREEDY)[0] for p in prompts], want)
        seen = {key: value - warm[key] for key, value in passed.model.telemetry()["prompt_passes"].items()
                if isinstance(value, int)}
        self.assertGreater(seen["passes"], 0, seen)
        self.assertEqual(passed.model.passed["open"], 0, seen)
        # a chunk through a layer a step: every pass took its chunks x the model's layers steps, or one fewer
        self.assertGreaterEqual(seen["steps"], seen["chunks"] * passed.model.layers - seen["passes"], seen)
        self.assertGreater(passed.telemetry()["core"]["steps"] - before, seen["chunks"])
        (ids,) = self.prompts(1, 70)
        self.assertEqual(passed.score(ids, 1, "prefill", top_k=1), plain.score(ids, 1, "prefill", top_k=1))
        with ThreadPoolExecutor(3) as pool:   # passes of several requests, and rows decoding beside them
            together = list(pool.map(lambda p: passed.generate(p, 8, stop_tokens=[], sampling=GREEDY)[0], prompts))
        self.assertEqual(together, want)
        # cancelled inside a pass: the request ends, its sequence is dropped, and the prompt decodes as before
        (long,) = self.prompts(1, 200)
        dropped = passed.model.passed["dropped"]
        # The cancel is looked for every 10 ms (crit_core.native), and a host can run the whole pass in less: each
        # step of an open pass waits until the cancel has been asked, so it lands inside the pass on any host.
        asked, step = threading.Event(), passed.model._pass_step

        def waiting(row, deadline):
            asked.wait(1.0)
            return step(row, deadline)

        def cancelled():
            if passed.model.passed["open"] > 0:
                asked.set()
            return asked.is_set()
        passed.model._pass_step = waiting
        try:
            found = passed.generate(long, 4, stop_tokens=[], sampling=GREEDY, cancelled=cancelled)
        finally:
            passed.model._pass_step = step
        self.assertTrue(asked.is_set())
        self.assertEqual(found, (None, {"cancelled": True}))
        # Core retires the cancelled session on its engine worker, which may come just after the request returned
        settle = time.monotonic() + 10
        while (passed.model.passed["dropped"], passed.model.passed["open"]) != (dropped + 1, 0)                 and time.monotonic() < settle:
            time.sleep(0.01)
        self.assertEqual((passed.model.passed["dropped"], passed.model.passed["open"]), (dropped + 1, 0))
        self.assertEqual(passed.generate(long, 8, stop_tokens=[], sampling=GREEDY)[0],
                         plain.generate(long, 8, stop_tokens=[], sampling=GREEDY)[0])
        for engine in (passed, plain):
            engine.close()
            self.engines.remove(engine)

    def test_the_model_has_cores_step_model_interface(self):
        self.assertImplements({"model": self.engine.model})

    def test_prompt_spans_are_timed_at_load_and_decode_runs_in_rounds(self):
        if self.device != "cuda":
            self.skipTest("spans are timed and rounds captured on a GPU")
        facts = self.engine.model.facts()
        self.assertTrue(facts["spans"]["calibrated"], facts["spans"])
        self.assertEqual(set(facts["spans"]["costs"]), {"c", "a", "b"})
        # the calibration's steps are Core's measured steps: their peak bytes a row, from Core's allocator, state
        # the deepest prompt chunk Core's arena holds (info's step_bytes)
        model = self.engine.model
        self.assertGreater(facts["spans"]["row_bytes"], 0)
        self.assertEqual(model.info()["step_bytes"], int(facts["spans"]["row_bytes"] * model.chunk))
        self.assertEqual(facts["rounds"]["problems"], [])
        self.assertIsNotNone(self.engine.model.rounds.native)


class GgufEngines:
    """A model plugin's class over the same checkpoint written as GGUF (gguf_checkpoint.py beside this file: the
    converter's renames, stacks, orders and transforms, by the plugin's layout's gguf entry for `arch`, its matrices
    Q8_0) and served through gguf-projections: the models core reads the file under the names its blocks read, and the
    reference computes the values the file stores. Put it before the plugin's class: class GgufX(GgufEngines, XTests),
    with `arch` the GGUF architecture."""
    arch = None
    dense = ()   # Hugging Face name endings of matrices the file keeps F32 (write_gguf's dense)

    @classmethod
    def checkpoint(cls, root):
        """The plugin class's checkpoint, finished as it makes it (a draft layer added after the save included), then
        written as GGUF."""
        path = super().checkpoint(root)
        if path is None:
            return None
        from . import gguf_checkpoint as fixture
        layout = json.loads((Path(cls.plugin) / "layout.json").read_text(encoding="utf-8"))
        gguf, values = fixture.write_gguf(path, root / "gguf", cls.arch, dense=cls.dense, layout=layout)
        fixture.stored_reference(cls.reference, values)
        cls.model = gguf.parent
        return cls.model


GGUF_USE = {**USE, "projections": "gguf-projections"}


def _name_rules(entries):
    """[(pattern, replacement)] of a layout's names rules ([pattern, replacement] or [pattern, replacement, shift]: the
    first group's number shifted), as the models core's checkpoint applies them."""
    import re

    def rule(entry):
        pattern, replacement = re.compile(entry[0]), entry[1]
        if len(entry) < 3:
            return pattern, replacement
        shift = int(entry[2])
        return pattern, lambda m: m.expand(replacement.replace("\\1", str(int(m.group(1)) + shift)))
    return [rule(entry) for entry in entries]


def _renamed(name, rules):
    for pattern, replacement in rules:
        name = pattern.sub(replacement, name)
    return name


class Respelled:
    """A model plugin's class over the same checkpoint with its tensors under another spelling: `spelling`, rules
    [pattern, replacement] (or [pattern, replacement, shift]: the first group's number shifted) applied in order to
    every saved name, as a checkpoint that names its tensors its own way (a release's own names: another decoder
    prefix, other part names) stores them; layout `names` rules map them back to the names the techniques read, so the
    engine is held to the same reference through them. `names`: those rules, which the engines' layout runs before the
    plugin's own names rules (served_layout: the layout of the test, as a plugin that serves such a release carries
    them; the plugin's layout.json keeps the rules of the checkpoints it does serve); without them the plugin's own
    rules must map the spelling back. Before any engine loads, every respelled tensor is checked to read, through the
    rules, as the name it reads under in the checkpoint as saved. Put it before the plugin's class:
    class X(Respelled, XTests). `files`: the safetensors files respelled (a glob in the checkpoint's folder). Every
    rule must rename some tensor (a respelling that misses is no second spelling), or with every_rule False some rule
    must (a spelling that is the layout's own rules, some of which another file's tensors meet)."""
    spelling = ()
    names = ()
    files = "*.safetensors"
    every_rule = True

    @classmethod
    def served_layout(cls, layout):
        layout = super().served_layout(layout)
        if cls.names:
            layout["names"] = [list(rule) for rule in cls.names] + list(layout.get("names", []))
        return layout

    @classmethod
    def checkpoint(cls, root):
        path = super().checkpoint(root)
        if path is None:
            return None
        from safetensors.torch import load_file, save_file
        own = Path(cls.plugin) / "layout.json"
        own = _name_rules(json.loads(own.read_text(encoding="utf-8")).get("names", []) if own.is_file() else [])
        served = _name_rules(cls.names) + own
        rules, used, wrong = _name_rules(cls.spelling), set(), []
        for file in sorted(Path(path).glob(cls.files)):
            tensors, renamed = load_file(str(file)), {}
            for name, tensor in tensors.items():
                new = name
                for i, (pattern, replacement) in enumerate(rules):
                    changed = pattern.sub(replacement, new)
                    if changed != new:
                        used.add(i)
                    new = changed
                if new in renamed:
                    raise ValueError(f"the spelling gives two tensors one name, {new}")
                renamed[new] = tensor
                if _renamed(new, served) != _renamed(name, own):
                    wrong.append(f"{new} reads as {_renamed(new, served)}, not {_renamed(name, own)}")
            save_file(renamed, str(file), metadata={"format": "pt"})
        index = Path(path) / "model.safetensors.index.json"
        if index.is_file():
            index.unlink()   # the files are read without their index (each file's own header names its tensors)
        unused = [cls.spelling[i][0] for i in range(len(rules)) if i not in used]
        if not used or (unused and cls.every_rule):
            raise ValueError(f"spelling rules that rename no tensor: {unused}")
        if wrong:
            raise ValueError(f"the layout's names rules do not map the spelling back: {'; '.join(wrong[:4])}")
        return path
