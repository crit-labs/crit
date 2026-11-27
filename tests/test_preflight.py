"""crit run ... preflight on the CPU: what it plans for small random checkpoints of real architectures (dense, routed
experts, sliding windows with rings, a draft layer, a companion drafter), sized from the run; the cut of those
checkpoints; the library-defaults selection; the time budget; and the records crit run ... show and next read. The
loads themselves need a GPU (the end-of-cycle checks)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import support.runs  # noqa: F401 - the product folders on sys.path
from support.tiny_checkpoints import checkpoint
from crit.preflight import candidate, cut, plan, records
from crit.preflight import run as preflight_run
from crit.run.served import smoke_lower
from crit.run.store import RunStore

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 131072
MIB = 1 << 20


class Tiny:
    """One folder of tiny checkpoints for the module's tests, each written once."""
    folder = None
    made = {}

    @classmethod
    def get(cls, name, **options):
        key = (name, tuple(sorted(options.items())))
        if key not in cls.made:
            if cls.folder is None:
                cls.temp = tempfile.TemporaryDirectory()
                cls.folder = Path(cls.temp.name)
            cls.made[key] = checkpoint(name, cls.folder / f"{name}-{len(cls.made)}", **options)
        if cls.made[key] is None:
            raise unittest.SkipTest(f"transformers lacks the {name} architecture")
        return cls.made[key]


def tearDownModule():
    if Tiny.folder is not None:
        Tiny.temp.cleanup()


class Runs:
    def run_of(self, model, mode="speed", context=CONTEXT):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store = RunStore(Path(temp.name) / "runs", project=ROOT)
        view = store.create({"model_path": str(model), "correctness_criteria": "preflight test", "kv_tokens": context,
                             "quality_mode": mode})
        return store, view["run"]


def placement(share):
    experts = 1000 * MIB
    return {"scenario": "sparse_device_host", "bytes": {"experts": experts, "experts_on_device": int(share * experts)}}


class PlanTests(Runs, unittest.TestCase):
    def plan(self, name, mode="speed", context=CONTEXT, placement=None, companion=False, data=True, **options):
        model = Tiny.get(name, **options)
        _, run = self.run_of(model, mode, context)
        return plan.plan_loads(run, candidate.facts(model), placement, companion, data)

    def test_a_dense_checkpoint_takes_one_load_of_every_general_scenario_sized_from_the_run(self):
        found = self.plan("dense")
        self.assertEqual([load.name for load in found.loads], ["speed/natural"])
        load = found.loads[0]
        self.assertEqual(load.extras, {"short", "cancel", "followup", "concurrency", "32k", "lower", "same",
                                       "precheck"})
        self.assertEqual((load.sizes.context, load.sizes.long, load.sizes.lower, load.sizes.cancel,
                          load.sizes.followup, load.sizes.calibrate),
                         (CONTEXT, 32768, smoke_lower(CONTEXT), 65536, 4096, True))
        self.assertEqual((load.share, load.smoke, load.precheck), (None, True, True))
        left = dict(found.left_out)
        self.assertIn("draft layer", left["drafting_draft_layer"])
        self.assertIn("routes no experts", left["pressure"])

    def test_a_short_context_leaves_out_what_it_cannot_hold_and_sizes_the_rest_within_it(self):
        found = self.plan("dense", context=16384)
        load = found.loads[0]
        self.assertNotIn("32k", load.extras)
        self.assertIn("prompt_32k", dict(found.left_out))
        self.assertEqual((load.sizes.cancel, load.sizes.long, load.sizes.lower), (16384 - 1024, 16384 - 1024, 5120))
        tiny = self.plan("dense", context=5000)
        self.assertNotIn("lower", tiny.loads[0].extras)   # smoke's lower context is its floor, 5,120: none lower
        self.assertIn("same", tiny.loads[0].extras)   # a close and reopen at the run's context holds at any context
        self.assertEqual(tiny.loads[0].sizes.followup, 2500)

    def test_the_quality_mode_is_the_runs(self):
        self.assertEqual([load.mode for load in self.plan("dense", mode="preserve").loads], ["preserve"])

    def test_routed_experts_add_a_pressure_load_at_the_whole_models_share_on_the_device(self):
        found = self.plan("routed", placement=placement(0.6))
        self.assertEqual([load.name for load in found.loads], ["speed/natural", "speed/pressure"])
        pressed = found.loads[1]
        self.assertAlmostEqual(pressed.share, 0.6, places=3)
        self.assertEqual(pressed.extras, {"short", "ladder", "cancel", "smoke", "32k"})
        self.assertEqual((pressed.smoke, pressed.precheck), (False, False))
        fits = self.plan("routed", placement=placement(1.0))
        self.assertEqual(len(fits.loads), 1)
        self.assertIn("fit the device", dict(fits.left_out)["pressure"])
        unsized = self.plan("routed", placement={"scenario": None, "reason": "no device"})
        self.assertIn("device file", dict(unsized.left_out)["pressure"])

    def test_sliding_windows_size_the_follow_up_past_their_ring(self):
        found = self.plan("windows", window=4096)
        self.assertEqual(found.loads[0].sizes.followup, 8192)
        self.assertIn("followup", found.loads[0].extras)
        self.assertEqual(self.plan("windows", window=24).loads[0].sizes.followup, 4096)

    def test_a_draft_layer_drafts_and_a_companion_drafter_drafts_through_its_plugin(self):
        self.assertIn("drafting", self.plan("draft").loads[0].extras)
        found = self.plan("dense", companion=True)
        self.assertIn("companion", found.loads[0].extras)
        self.assertNotIn("drafting", found.loads[0].extras)

    def test_without_crit_evals_data_there_is_no_precheck(self):
        found = self.plan("dense", data=False)
        self.assertFalse(found.loads[0].precheck)
        self.assertIn("precheck", dict(found.left_out))


class SelectionTests(unittest.TestCase):
    def test_the_library_defaults_follow_the_checkpoints_facts(self):
        dense, _ = candidate.library_selection(Tiny.get("dense"))
        self.assertEqual(dense["use"], {"model": "llama-qwen3-model", "placement": "device-placement",
                                        "projections": "projections-dense", "attention": "partitioned-attention",
                                        "kv_cache": "attention-state", "sampler": "token-constraints",
                                        "norms": "fused-norms", "embeddings": "lookup-table-rows"})
        self.assertEqual(dense["options"], {})
        routed, plugins = candidate.library_selection(Tiny.get("routed"), tables_off_device=True)
        self.assertEqual({slot: routed["use"][slot] for slot in ("router", "experts", "expert_source")},
                         {"router": "topk-router", "experts": "expert-offload", "expert_source": "expert-tensors"})
        self.assertEqual(routed["options"]["attention-state"], {"elastic": True})
        self.assertEqual(routed["options"]["device-placement"], {"off_device": "tables."})
        self.assertIn("score_before_selection", routed["options"]["topk-router"])   # the checkpoint's own rule
        self.assertEqual(plugins, list(dict.fromkeys(routed["use"].values())))
        draft, plugins = candidate.library_selection(Tiny.get("draft"))
        self.assertEqual((draft["use"]["drafter"], draft["use"]["recurrence"]),
                         ("greedy-head-drafter", "sequential-recurrence"))
        self.assertEqual(draft["options"]["qwen3-5-model"], {"mtp": True})
        self.assertEqual(plugins[-1], candidate.SEAM_DRAFTER)
        companion, plugins = candidate.library_selection(Tiny.get("dense"), companion=True)
        self.assertNotIn("drafter", companion["use"])   # a variant of the load drafts with it
        self.assertEqual(plugins[-1], candidate.COMPANION_DRAFTER)
        # A model plugin whose layout reads a companion as its own draft block drafts through the model's draft head
        sliding = candidate.facts(Tiny.get("windows"))
        self.assertEqual(sliding["draft_companion"], "gemma4_assistant")
        _, plugins = candidate.library_selection(Tiny.get("windows"), sliding, companion=True)
        self.assertEqual(plugins[-1], candidate.DEVICE_DEFAULTS["drafter"])

    def test_every_layer_kind_the_checkpoint_declares_gets_an_attention_plugin_that_serves_it(self):
        # partitioned-attention provides no attention:sliding: a sliding-window checkpoint takes softmax-attention,
        # whose manifest provides every fact its layers need, and the selection's model plugin loads beside it
        sliding = candidate.facts(Tiny.get("windows"))
        self.assertIn("sliding_attention", sliding["layer_kinds"])
        self.assertEqual(candidate.needed_facts(sliding)["attention"], {"attention:sliding"})
        selection, plugins = candidate.library_selection(Tiny.get("windows"), sliding)
        self.assertEqual(selection["use"]["attention"], "softmax-attention")
        self.assertIn("softmax-attention", plugins)
        from crit.plugins import library
        provides = set(library()[selection["use"]["attention"]]["provides"])
        self.assertLessEqual(candidate.needed_facts(sliding)["attention"], provides)
        # a dense checkpoint keeps the device default; latent attention and sinks are needs too
        dense = candidate.facts(Tiny.get("dense"))
        self.assertEqual(candidate.needed_facts(dense)["attention"], set())
        self.assertEqual(candidate.needed_facts({"latent": True, "sinks": True})["attention"],
                         {"attention:latent", "attention:sinks"})
        self.assertEqual(candidate.serving("attention", {"attention:latent"}), "softmax-attention")
        with self.assertRaisesRegex(ValueError, "no library plugin of the attention slot provides attention:nothing"):
            candidate.serving("attention", {"attention:nothing"})

    def test_recurrent_states_are_elastic_beside_an_expert_cache_as_the_kv_pools_are(self):
        draft = Tiny.get("draft")
        found = {**candidate.facts(draft), "routed": True}
        with mock.patch("crit.models.options_for", return_value={}):
            selection, _ = candidate.library_selection(draft, found)
        self.assertEqual(selection["options"]["attention-state"], {"elastic": True})
        self.assertEqual(selection["options"]["sequential-recurrence"], {"elastic": True})

    def test_a_checkpoint_no_model_plugin_serves_is_refused_by_name(self):
        found = {**candidate.facts(Tiny.get("dense")), "model_plugin": None}
        with self.assertRaisesRegex(ValueError, "no finished model plugin"):
            candidate.library_selection(Tiny.get("dense"), found)


class CutTests(unittest.TestCase):
    def test_the_default_cut_keeps_the_first_layer_of_every_kind_the_draft_layer_and_links_whole_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            windows = cut.cut(Tiny.get("windows", window=24), Path(folder, "windows"))
            self.assertEqual(windows["layers"], [0, 1, 2])   # up to the first full layer, the period kept
            config = json.loads(Path(folder, "windows", "config.json").read_text(encoding="utf-8"))
            text = config.get("text_config") or config
            self.assertEqual(text["layer_types"], ["sliding_attention", "sliding_attention", "full_attention"])
            draft = cut.cut(Tiny.get("draft"), Path(folder, "draft"), draft_prefixes=("mtp.",))
            self.assertEqual(draft["layers"], [0, 1, 2, 3])   # one full layer in four: the whole period
            self.assertIn("mtp.safetensors", draft["linked"])   # kept whole under its own names: the same file
            self.assertTrue(any(name.startswith("mtp.") for name in cut.tensor_names(Path(folder, "draft"))))
            dense = cut.cut(Tiny.get("dense"), Path(folder, "dense"))
            self.assertEqual(dense["layers"], [0])
            self.assertEqual(json.loads(Path(folder, "dense", "config.json").read_text())["num_hidden_layers"], 1)
            self.assertTrue(Path(folder, "dense", "tokenizer.json").is_file())


class BudgetTests(unittest.TestCase):
    def test_a_scenario_runs_only_where_its_estimate_fits_what_is_left(self):
        now = [0.0]
        budget = plan.Budget(100, {"prompt_32k": 50.0}, clock=lambda: now[0])
        self.assertTrue(budget.fits("prompt_32k"))
        now[0] = 60.0
        self.assertFalse(budget.fits("prompt_32k"))
        self.assertTrue(budget.fits("short_request"))   # the default guess, 2 s
        self.assertIn("40 s left", budget.describe("prompt_32k"))

    def test_a_previous_report_measures_each_scenario_and_the_load_less_what_ran_inside_it(self):
        report = {"items": [{"scenario": "speed/natural/load", "state": "passed", "seconds": 300.0},
                            {"scenario": "speed/natural/prompt_32k", "state": "passed", "seconds": 12.0},
                            {"scenario": "speed/natural/smoke_decode_long", "state": "passed", "seconds": 30.0},
                            {"scenario": "speed/natural/precheck/honesty", "state": "passed", "seconds": 40.0},
                            {"scenario": "speed/natural/ready", "state": "info", "seconds": 100.0},
                            {"scenario": "cut", "state": "info", "seconds": 5.0}]}
        found = plan.measured(report)
        self.assertEqual((found["prompt_32k"], found["smoke"], found["precheck"], found["load"]),
                         (12.0, 30.0, 40.0, 300.0 - 82.0))
        load = plan.Load("speed/natural", "speed", {"short", "32k"}, plan.sizes_for(CONTEXT, {}), precheck=True)
        estimate = plan.Budget(900, found).estimate(load)
        self.assertEqual(estimate, 218.0 + 2.0 + 12.0 + 2 * 30.0 + 40.0)

    def test_the_ballast_leaves_the_cut_its_own_room_and_the_whole_models_share_of_its_experts(self):
        basis = {"vram_headroom_bytes": 1, "reserve_bytes": 10, "resident_weight_bytes": 100,
                 "fixed_state_bytes": 1000, "expert_bytes": 10000}
        self.assertEqual(plan.leave_bytes(basis, 2, 50000, 0.5), 1 + 10 + 100 + 1000 + 100000 + 5000)


class RecordTests(Runs, unittest.TestCase):
    def report_for(self, root, model, key=None, failed=False):
        items = [{"scenario": "speed/natural/load", "state": "passed", "seconds": 1.0, "detail": ""},
                 {"scenario": "speed/natural/prompt_cache_followup", "state": "failed" if failed else "passed",
                  "seconds": 1.0, "detail": ""}]
        return records.write(items, records.folder(root, key or records.key(model)), finished_at=records.now())

    def test_a_report_is_keyed_by_crits_code_and_the_checkpoints_revision(self):
        model = Tiny.get("dense")
        self.assertEqual(records.key(model), f"{records.crit_key()}-{records.checkpoint_revision(model)}")
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder, "copy")
            cut.cut(model, copy, [0, 1], link=False)
            self.assertNotEqual(records.checkpoint_revision(copy), records.checkpoint_revision(model))

    def test_show_and_next_name_the_preflight_until_a_report_for_this_crit_and_checkpoint_exists(self):
        from crit.run.cli import main
        from crit.run.state import render
        model = Tiny.get("dense")
        store, run = self.run_of(model)
        root = store.directory(run["run_id"])
        found = store.next(run["run_id"])
        self.assertEqual((found["step"], found["command"].rsplit(" ", 1)[-1]), ("preflight", "preflight"))
        self.report_for(root, model, key="0123456789ab-feedfacecafe")   # another crit's: not this one's
        found = store.next(run["run_id"])
        self.assertEqual(found["step"], "preflight")
        self.assertIn("0123456789ab-feedfacecafe", found["why"])
        self.report_for(root, model, failed=True)
        found = store.next(run["run_id"])
        self.assertEqual(found["step"], "recon")   # a failed scenario is feedback, never a gate
        self.assertEqual(found["state"]["preflight"]["failed_scenarios"], ["speed/natural/prompt_cache_followup"])
        self.assertIn("at the library's defaults: speed/natural/prompt_cache_followup", render(found))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--root", str(store.root), "--id", run["run_id"], "show"]), 0)
        shown = json.loads(output.getvalue())["preflight"]
        self.assertEqual((shown["current"], shown["failed"], shown["report"]),
                         (True, 1, f"preflight/{records.key(model)}/preflight.md"))

    def test_a_model_the_preflight_cannot_cut_takes_no_step(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder, "model.gguf")
            weights.write_bytes(b"GGUF")
            self.assertIsNone(records.brief(folder, weights))


class PreflightTests(Runs, unittest.TestCase):
    def test_it_cuts_packages_plans_and_serves_each_load_under_the_lock_and_keeps_the_report(self):
        from crit.preflight import scenarios
        model = Tiny.get("dense")
        store, run = self.run_of(model, context=16384)
        served = []

        def load(report, store_, run_id, candidate_id, prefix, extras, sizes, data, plugin_tests, **options):
            served.append((prefix, sorted(extras), sizes.context, data is not None, options["smoke"]))
            report.add(prefix + "load", "passed", 1.0, "fixture")
            report.add(prefix + "concurrency", "failed", 1.0, "fixture")

        locks = []

        @contextlib.contextmanager
        def gpu(shared, label, log=None):
            locks.append((shared, label))
            yield

        with mock.patch.object(scenarios, "load", load), mock.patch.object(scenarios, "gpu", gpu), \
                mock.patch.object(scenarios, "prepare_data", lambda model, data: Path(data)):
            found = preflight_run.preflight(store, run["run_id"], lock="/tmp/gpu.lock", log=lambda line: None)
        root = store.directory(run["run_id"])
        self.assertEqual(served, [("speed/natural/", ["cancel", "concurrency", "followup", "lower", "precheck",
                                                      "same", "short"], 16384, True, True)])
        self.assertEqual(locks, [("/tmp/gpu.lock", f"crit run preflight {run['run_id']} speed/natural")])
        self.assertEqual((found["key"], found["failed"], found["failed_scenarios"]),
                         (records.key(model), 1, ["speed/natural/concurrency"]))
        value = json.loads((records.folder(root, found["key"]) / "preflight.json").read_text(encoding="utf-8"))
        self.assertEqual(value["cut"]["layers"], [0])
        self.assertEqual(value["selection"]["use"]["model"], "llama-qwen3-model")
        names = [item["scenario"] for item in value["items"]]
        self.assertIn("prompt_32k", names)   # left out with why: the context holds none
        self.assertTrue((records.folder(root, found["key"]) / "preflight.md").is_file())
        self.assertTrue((root / ".plugin-cache" / "preflight" / records.checkpoint_revision(model) / "cut").is_dir())
        self.assertEqual(store.next(run["run_id"])["step"], "recon")
        self.assertEqual(json.loads((root / "state.json").read_text())["candidates"], [])   # nothing else recorded

    def test_the_gpu_lock_is_the_options_then_the_environments_then_the_settings(self):
        with mock.patch.dict("os.environ", {}, clear=False) as environment:
            environment.pop("CRIT_GPU_LOCK", None)
            self.assertEqual(preflight_run.gpu_lock(None, {"gpu_lock": "/x.lock"}), "/x.lock")
            self.assertEqual(preflight_run.gpu_lock("", {"gpu_lock": "/x.lock"}), "")
            environment["CRIT_GPU_LOCK"] = ""   # gpu-job's jobs hold the lock already
            self.assertEqual(preflight_run.gpu_lock(None, {"gpu_lock": "/x.lock"}), "")



class ReopenCloseTests(unittest.TestCase):
    """A reopen scenario judges the engine closes it made (crit serve's last_close, read into Served.closes)."""

    def judged(self, closes):
        from crit.preflight.scenarios import judge_closes
        found = {"state": "passed", "detail": "reopened"}
        judge_closes(found, closes)
        return found

    def test_a_close_that_kept_memory_fails_the_reopen_and_an_unjudged_one_does_too(self):
        returned = {"returned": True, "judged": True, "message": "returned"}
        self.assertEqual(self.judged([returned])["state"], "passed")
        kept = {"returned": False, "judged": True, "message": "elastic range attention-state/pages is open"}
        found = self.judged([returned, kept])
        self.assertEqual(found["state"], "failed")
        self.assertIn("did not return its memory: elastic range attention-state/pages is open", found["detail"])
        self.assertEqual(self.judged([{"returned": None, "message": "not judged"}])["state"], "failed")
        self.assertIn("reported no engine close", self.judged([])["detail"])


if __name__ == "__main__":
    unittest.main()
