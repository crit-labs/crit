"""model-template as whole engines: for each library layout a small random checkpoint of a transformers
architecture, the template beside the library's device plugins in a package, run by Core's loader on Core's
concurrent runtime (which refuses an engine that misses a serving contract: its batching, every computing slot reached,
its plans, its warm requests). Each engine's fixed-text scores follow transformers' log-probabilities of the same
checkpoint, its greedy continuations are transformers' choices up to BF16 rounding, requests in flight together decode
what each decodes alone, and its prompt spans are timed at load; spans of 16 tokens score as whole chunks do. These are
the dense layouts over the library's base plugins; crit's tests/test_model_template.py runs the same engines for the
mixture-of-experts, hybrid, streams and drafting layouts. The engines run on CUDA where torch has it and on the CPU
otherwise (CRIT_TEMPLATE_DEVICE names one): there the norms are torch-norms', decoding goes through step() (captured
rounds are a GPU's) and no span is timed, the arithmetic checked against transformers is the same. Needs
transformers, safetensors and Core's crit_core_steps: without them the engines skip unchecked, which fails crit plugins
test (crit_plugins.testing.unchecked). Each class skips plainly where transformers lacks its architecture or its plugins
do not run on the device."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
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

HERE = Path(__file__).resolve().parents[1]
GREEDY = {"temperature": 0.0}
BASE = ["attention-state", "softmax-attention", "projections-dense", "token-sampling", "device-placement",
        "fused-norms", "memory-table-rows"]
USE = {"model": "model-template", "placement": "device-placement", "projections": "projections-dense",
       "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
       "norms": "fused-norms", "embeddings": "memory-table-rows"}
OPTIONS = {"context": 1024, "kv_margin": 256, "prefill_chunk": 64, "activations_mib": 64, "concurrency": 4}


def engine_device():
    """The device the engines run on: CRIT_TEMPLATE_DEVICE, else CUDA where torch has it, else the CPU."""
    return os.environ.get("CRIT_TEMPLATE_DEVICE") or ("cuda" if torch is not None and torch.cuda.is_available()
                                                      else "cpu")


class Engines(PluginTestCase):
    """A package of the template and the library plugins a layout needs, and engines of it over one checkpoint."""
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
        return HERE   # this plugin's, also for the engines crit's own tests build on this harness

    @classmethod
    def setUpClass(cls):
        if cls.checkpoint.__func__ is Engines.checkpoint.__func__:
            raise unittest.SkipTest("the harness: a layout's class names its checkpoint")
        if torch is None:
            raise unittest.SkipTest("whole engines need torch")
        cls.device = engine_device()
        if cls.device not in cls.devices:
            raise unittest.SkipTest(f"its plugins run on {', '.join(cls.devices)}, the engines here on {cls.device}")
        # The engines run here, so what keeps them from running leaves the template's correctness unchecked: Core's
        # runtime, and transformers, the reference its scores and continuations are compared with.
        core_runtime()
        if transformers is None:
            raise unchecked("transformers and safetensors, the reference the engines are compared with "
                            "(crit's reference extra)")
        try:
            from crit.plugins import add
        except ImportError:
            raise unittest.SkipTest("crit's command line (crit plugins add)") from None
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        model = cls.checkpoint(root)
        if model is None:
            raise unittest.SkipTest(f"transformers {transformers.__version__} lacks the architecture")
        cls.package = root / "package"
        # model-template is copied from here, so the plugins its tests use (test_with) are added as crit plugins add
        # adds them beside a plugin it copies
        test_with = json.loads((HERE / "plugin.json").read_text(encoding="utf-8")).get("test_with", [])
        add(cls.package / "plugins", sorted(set(cls.library) | set(test_with) - {"model-template"}))
        shutil.copytree(HERE, cls.package / "plugins" / "model-template",
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))
        (cls.package / "configuration.json").write_text(json.dumps({"plugins": cls.selection()}), encoding="utf-8")
        cls.engines = []
        cls.engine = cls.load()

    @classmethod
    def selection(cls, use=None, plugins=None, **options):
        """The engine's selection: `use` over the class's, `plugins` the other plugins' options over the class's
        ({plugin: {option: value}}), `options` the template's."""
        chosen = {**cls.use, **(use or {})}
        if cls.device != "cuda" and chosen.get("norms") == "fused-norms":
            chosen["norms"] = "torch-norms"   # fused-norms is a GPU's kernel: the base plugin's arithmetic on the CPU
        others = {name: {**cls.plugin_options.get(name, {}), **(plugins or {}).get(name, {})}
                  for name in {**cls.plugin_options, **(plugins or {})}}
        return {"use": chosen, "base": {"norms": "torch-norms"},
                "options": {"model-template": {**cls.options, **options}, **others}}

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
    """The checks every layout's engine passes."""

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
        self.assertEqual(resolved["by"]["kv_bytes_per_token"], "model-template")
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
        options = dict(prefill_chunk=16, span_seconds=1e-6)
        passed, plain = self.load(passes="on", **options), self.load(passes="off", **options)
        self.assertEqual((passed.model.passes, plain.model.passes), (True, False))
        prompts = self.prompts(2, 75) + self.prompts(1, 35)
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
        (ids,) = self.prompts(1, 150)
        self.assertEqual(passed.score(ids, 1, "prefill", top_k=1), plain.score(ids, 1, "prefill", top_k=1))
        with ThreadPoolExecutor(3) as pool:   # passes of several requests, and rows decoding beside them
            together = list(pool.map(lambda p: passed.generate(p, 8, stop_tokens=[], sampling=GREEDY)[0], prompts))
        self.assertEqual(together, want)
        # cancelled inside a pass: the request ends, its sequence is dropped, and the prompt decodes as before
        (long,) = self.prompts(1, 200)
        dropped = passed.model.passed["dropped"]
        found = passed.generate(long, 4, stop_tokens=[], sampling=GREEDY,
                                cancelled=lambda: passed.model.passed["open"] > 0)
        self.assertEqual(found, (None, {"cancelled": True}))
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


@unittest.skipIf(torch is None, "torch")
class DecoderTests(Layout):
    """The decoder layout: a Llama checkpoint (grouped-query attention, rotary positions, SwiGLU)."""

    @classmethod
    def checkpoint(cls, root):
        config = transformers.LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                                          rms_norm_eps=1e-6, tie_word_embeddings=False)
        return cls.save(transformers.LlamaForCausalLM(config), root)

    def test_short_spans_score_as_whole_chunks_and_prefill_in_spans(self):
        # A prefill chunk of QUANTUM tokens: every prompt step is that long, and a text scored in such spans gets
        # the whole chunks' scores within prompt-row rounding (other chunk boundaries, the same arithmetic).
        short = self.load(prefill_chunk=16)
        spans = short.model.spans
        self.assertEqual([spans.rows(p) for p in (0, 100, 900)], [16, 16, 16])
        (ids,) = self.prompts(1, 150)
        self.assertScores(short, ids)
        before = short.telemetry()["core"]["steps"]
        short.generate(ids, 2, stop_tokens=[], sampling=GREEDY)
        self.assertGreaterEqual(short.telemetry()["core"]["steps"] - before, -(-len(ids) // 16))
        short.close()
        self.engines.remove(short)

    def test_a_score_s_slices_of_logits_live_in_the_step_arena_it_states(self):
        # A score's logits are its rows' FP32 [rows][vocabulary] beside their log-softmax and, for top_k, a mask of
        # the entries at or past each row's k-th value (crit_plugins.ranking: no sorted vocabulary row): on a large
        # vocabulary hundreds of megabytes a slice, which no prompt step of its size takes. Core's warm requests
        # score a text of a full chunk with 20 alternatives at load, its arena holds what that took, and a score's
        # segments come out of it: no slice passes the arena.
        from crit_plugins import memory
        model = self.engine.model
        if model.device.type != "cuda":
            self.skipTest("Core allocates a CUDA device's step memory; this engine runs on the CPU")
        device = memory.of(model.device)
        overflows = device.steps()["overflows"]
        (ids,) = self.prompts(1, 300)
        self.engine.score(ids, 1, "prefill", top_k=4)
        steps = device.steps()
        if steps.get("unaccounted"):
            self.skipTest(steps["unaccounted"])
        self.assertGreater(steps["most"].get("score", 0), 0)
        self.assertEqual(steps["overflows"], overflows)
        self.assertGreaterEqual(device.arena, steps["most"]["score"])

    def test_a_request_asking_decode_via_prompt_decodes_as_one_token_prompt_chunks(self):
        # The debug hook crit smoke tells a decode gap's rounding from a fault by (StepRequest.decode_via_prompt): a
        # replay's decode rows multiply as one-token prompt chunks where they would multiply as decode rows, and
        # score what the prompt path scores. (The head multiplies every logits row as a decode row on both paths.)
        model = self.engine.model
        self.assertIs(model.info()["decode_via_prompt"], True)
        groups, matmul = [], model.rows.matmul

        def spy(x, matrix, found, out_dtype=None):
            if matrix is not model.head:
                groups.append(list(found))
            return matmul(x, matrix, found, out_dtype)
        model.rows.matmul = spy
        prompt, continuation = self.prompts(2, 21)
        continuation = continuation[:12]

        def replay(via_prompt):
            rows = []
            groups.clear()
            tokens, _ = self.engine.generate(prompt, len(continuation), stop_tokens=[], replay=continuation,
                                             sampling=GREEDY, logprobs=1, decode_via_prompt=via_prompt,
                                             on_tokens=lambda _tokens, found: rows.extend(found))
            self.assertEqual(tokens, continuation)
            return [logprob for logprob, _ in rows], [g for found in groups for g in found if g[1] - g[0] == 1]
        def rounds():   # decode rounds Core ran in its captured rounds (none where the model has no rounds)
            return ((model.telemetry() or {}).get("drafts") or {}).get("rounds") or 0
        try:
            began = rounds()
            plain, plain_rows = replay(False)
            plain_rounds = rounds() - began
            via, via_rows = replay(True)
            via_rounds = rounds() - began - plain_rounds
        finally:
            model.rows.matmul = matmul
        # Plain decode rows multiply as decode rows: eagerly (the spy sees them), or inside Core's captured rounds,
        # whose replays no Python spy sees (a replayed request decodes there too since rounds mask), never as prompt
        # chunks. Through the hook they are one-token prompt chunks, run eagerly, in no round.
        self.assertFalse(any(prompt for _, _, prompt in plain_rows), plain_rows)
        self.assertTrue(plain_rows or plain_rounds >= len(continuation) - 1, (plain_rows, plain_rounds))
        self.assertTrue(via_rows and all(prompt for _, _, prompt in via_rows), via_rows)
        self.assertEqual(via_rounds, 0)
        scored = self.engine.score(prompt + continuation, len(prompt), "prefill").target
        self.assertLess(max(abs(a - b) for a, b in zip(via, scored)), 0.25, (via, scored))
        self.assertLess(max(abs(a - b) for a, b in zip(via, plain)), 0.25, (via, plain))

    def test_a_selection_without_a_slot_the_layout_calls_is_refused_before_load(self):
        from crit_plugins import PluginError
        from crit_core.engine import load
        selection = self.selection()
        del selection["use"]["embeddings"]
        with self.assertRaisesRegex(PluginError, "model-template requires embeddings"):
            load(self.package, self.model, {"plugins": selection}, device=self.device)


@unittest.skipIf(torch is None, "torch")
class SlidingTests(Layout):
    """The decoder layout with sliding-window layers: a Qwen3 checkpoint (query/key norms) whose first layer sees its
    last 24 positions, its prompts longer than the window."""

    @classmethod
    def checkpoint(cls, root):
        if not hasattr(transformers, "Qwen3ForCausalLM"):
            return None
        config = transformers.Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                                          max_position_embeddings=2048, rms_norm_eps=1e-6, tie_word_embeddings=False,
                                          use_sliding_window=True, sliding_window=24, max_window_layers=1,
                                          layer_types=["sliding_attention", "full_attention"])
        model = transformers.Qwen3ForCausalLM(config)
        if list(getattr(model.config, "layer_types", [])) != ["sliding_attention", "full_attention"]:
            return None
        return cls.save(model, root)


def gguf_fixture():
    """tests/gguf_checkpoint.py: a checkpoint written as GGUF's converter writes it."""
    spec = importlib.util.spec_from_file_location("model_template_gguf_checkpoint",
                                                  Path(__file__).resolve().with_name("gguf_checkpoint.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GgufEngines:
    """A layout's class over the same checkpoint written as GGUF (tests/gguf_checkpoint.py: the converter's renames,
    stacks, orders and transforms, its matrices Q8_0) and served through gguf-projections: the template reads the file
    under the names its blocks read, and the reference computes the values the file stores. Put it before the
    layout's class: class GgufX(GgufEngines, XTests), with `arch` the GGUF architecture."""
    arch = None
    dense = ()   # Hugging Face name endings of matrices the file keeps F32 (write_gguf's dense)

    @classmethod
    def checkpoint(cls, root):
        """The layout class's checkpoint, finished as it makes it (a draft layer added after the save included), then
        written as GGUF."""
        path = super().checkpoint(root)
        if path is None:
            return None
        fixture = gguf_fixture()
        gguf, values = fixture.write_gguf(path, root / "gguf", cls.arch, dense=cls.dense)
        fixture.stored_reference(cls.reference, values)
        cls.model = gguf.parent
        return cls.model


GGUF_USE = {**USE, "projections": "gguf-projections"}


@unittest.skipIf(torch is None, "torch")
class GgufDecoderTests(GgufEngines, DecoderTests):
    """The decoder layout over a Llama checkpoint written as GGUF: its query and key rows as the converter permutes
    them, every matrix Q8_0, the token embeddings kept as the file's Q8_0 rows (lookup-table-rows decodes the rows a
    lookup reads)."""
    arch = "llama"
    library = BASE + ["gguf-projections", "lookup-table-rows"]
    use = {**GGUF_USE, "embeddings": "lookup-table-rows"}

    def test_the_matrices_load_as_the_file_stores_them(self):
        found = self.engine.facts
        self.assertIn("Q8_0", found["gguf_formats"], found)
        self.assertEqual(self.engine.model.embed.format, "gguf:Q8_0")   # stored rows, never decoded whole


@unittest.skipIf(torch is None, "torch")
class GgufDenseDecoderTests(GgufEngines, DecoderTests):
    """The decoder layout over a Llama checkpoint written as GGUF with some matrices kept F32 (the attention's output
    and the MLP's down projection), as GGUF files keep routers and small gates: gguf-projections loads them dense
    beside the Q8_0 ones, and an engine of the run's preserve mode, which asks exact prompt rows, is taken by Core's
    loader and serves."""
    arch = "llama"
    dense = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
    library = BASE + ["gguf-projections", "lookup-table-rows"]
    use = {**GGUF_USE, "embeddings": "lookup-table-rows"}

    def test_a_preserve_mode_engine_keeps_the_exact_rows_it_asks(self):
        from crit_core.engine import load
        found = self.engine.facts
        self.assertIn("Q8_0", found["gguf_formats"], found)
        self.assertIn("bf16", found["gguf_formats"], found)   # the F32 matrices, dense
        engine = load(self.package, self.model, {"plugins": self.selection(), "quality_mode": "preserve"},
                      device=self.device)
        self.engines.append(engine)
        facts = engine.facts
        self.assertEqual(facts["row_invariance"]["prompt_exact"], True, facts)
        # Its prompt rows have its decode rows' bits: each greedy reply is the top token its own prefill gives at
        # every position. The engine without a mode multiplies other bits on CUDA (prompt rows cuBLAS's GEMM over
        # the decoded matrices, the exact path's decode rows int8 activations), so its reply may part from this one
        # only where the reference has a near-tie; its scores are the reference's.
        prompts = self.prompts(2, 40)
        for p in prompts:
            reply = engine.generate(p, 6, stop_tokens=[], sampling=GREEDY)[0]
            scored = engine.score(p + reply, 1, "prefill", top_k=1)
            self.assertEqual([scored.top[len(p) - 1 + k][0][0] for k in range(len(reply))], reply)
            other = self.engine.generate(p, 6, stop_tokens=[], sampling=GREEDY)[0]
            if other != reply:
                at = next(k for k, (a, b) in enumerate(zip(reply, other)) if a != b)
                logp = self.logits(p + reply[:at])[len(p) - 1 + at].log_softmax(-1)
                self.assertLess(abs(float(logp[reply[at]]) - float(logp[other[at]])), 0.5, (reply, other))
        self.assertScores(engine, prompts[0])


if __name__ == "__main__":
    unittest.main()
