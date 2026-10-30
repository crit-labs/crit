"""crit pull and crit serve --engine: a crit-hub engine installed from its card, its pinned source and its locked
checkpoint, then served from the store without a run."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit.pull import requirements as machine
from crit.pull.cli import main, pull
from crit.pull.store import InstallStore
from crit.run.records import digest, file_sha, inventory, read_json, write_json
from test_chat_residency import ADAPTER

try:
    import yaml
except ImportError:  # crit-hub cards are YAML
    yaml = None

CAPABILITIES = ["prefill", "decode", "chat-json-v1", "chat-resident-v1", "chat-sampling-temperature-v1"]
ENGINE = "someone/fast-engine"
DEVICE = {"device": "NVIDIA GeForce RTX 5090", "compute_capability": "12.0", "cuda_runtime": 13010,
          "memory_total_bytes": 34190458880}


def git(folder, *arguments):
    return subprocess.run(["git", "-C", str(folder), *arguments], check=True, capture_output=True, text=True).stdout.strip()


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class PullTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        toolkit = patch("crit.pull.requirements.toolkit", return_value="13.1")
        toolkit.start()
        self.addCleanup(toolkit.stop)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model.fixture").write_text("synthetic input, not a model", encoding="utf-8")
        (self.model / "README.md").write_text("# Fixture model\n", encoding="utf-8")
        self.configuration = {"batch": 1}
        self.revision = self.engines_repository()
        self.hub = self.root / "hub"
        self.card = self.hub_card()
        self.device = self.root / "device.json"
        write_json(self.device, DEVICE)
        self.store = InstallStore(self.root / "store")

    def engines_repository(self):
        """A crit-engines checkout with fast-engine/: an adapter, its configuration and a build script."""
        repository = self.root / "crit-engines"
        engine = repository / "fast-engine" / "engine"
        engine.mkdir(parents=True)
        (engine / "adapter.py").write_text(ADAPTER + f"\ndef declared_capabilities(model_path):\n    return {CAPABILITIES!r}\n",
                                           encoding="utf-8")
        (engine / "configuration.json").write_text(json.dumps(self.configuration, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        (engine / "build.py").write_text("from pathlib import Path\nPath('built.txt').write_text('built')\n",
                                         encoding="utf-8")
        (repository / "fast-engine" / "THIRD_PARTY.md").write_text("# Credits\n", encoding="utf-8")
        git(repository.parent, "init", "--quiet", str(repository))
        git(repository, "add", "-A")
        git(repository, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "--quiet", "-m", "fixture")
        return git(repository, "rev-parse", "HEAD")

    def hub_card(self, **changes):
        folder = self.hub / "engines" / ENGINE
        folder.mkdir(parents=True, exist_ok=True)
        card = {
            "title": "fast-engine", "summary": "A fixture engine", "origin": "crit",
            "contributor": {"name": "someone", "repository": "https://github.com/someone/crit"}, "license": "Apache-2.0",
            "features": ["openai-api"],
            "source": {"repository": "https://github.com/crit-labs/crit-engines", "revision": self.revision, "path": "fast-engine"},
            "requirements": {"platforms": [machine.platform()], "gpu_arch": ["sm_120a"], "gpu_memory_gb": 32, "cuda": "13.1"},
            "build": {"command": ["python", "build.py"]},
            "checkpoints": {"fixture@BF16": {"repository": "fixture/model", "revision": "f" * 40}},
            "results": [{"gpu": "rtx5090", "model": "fixture@BF16", "measured_with": "crit-eval-v0.2.1",
                         "stats": {"decode_tok_s": 100.0, "context_tokens": 4096}, "evidence": []}],
            "readme": {"intro": "An engine."},
        } | changes
        (folder / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")
        rows = inventory(self.model)
        write_json(folder / "card.lock.json", {
            "checkpoints": {"fixture@BF16": {"repository": "fixture/model", "revision": "f" * 40, "files": rows}},
            "provenance": {"run_id": "run-20261023T011625Z-3141e655", "candidate_id": "candidate-0007",
                           "configuration_sha256": digest(self.configuration)}})
        return card

    def install(self, **options):
        defaults = dict(hub=self.hub, checkout=self.root / "crit-engines", model=self.model, device=self.device,
                        project=ROOT)
        return pull(ENGINE, self.store, **(defaults | options))

    def test_pull_installs_the_built_engine_without_absolute_paths(self):
        record = self.install()
        folder = self.store.folder(ENGINE)
        self.assertEqual(record["source"], {"repository": "https://github.com/crit-labs/crit-engines",
                                            "revision": self.revision, "path": "fast-engine"})
        self.assertEqual(record["checkpoint"]["folder"], f"models/fixture--model@{'f' * 40}")
        self.assertEqual(record["checkpoint"]["from"], "local folder")
        self.assertEqual(record["limits"]["kv_tokens"], 4096, "the context the card measured")
        self.assertTrue(all(row["met"] for row in record["requirements"]["checks"]))
        self.assertEqual(record["build"]["outputs"], ["built.txt"])
        self.assertEqual((folder / "package/built.txt").read_text(), "built")
        self.assertTrue((folder / "THIRD_PARTY.md").is_file())
        manifest = read_json(folder / "package/manifest.json")
        self.assertEqual(manifest["capabilities"], CAPABILITIES)
        self.assertEqual(manifest["launch"], ["adapter.py"])
        self.assertEqual(file_sha(self.store.root / record["checkpoint"]["folder"] / "model.fixture"),
                         file_sha(self.model / "model.fixture"))
        for path in folder.rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            for local in (str(self.root), self.root.as_posix(), json.dumps(str(self.root))[1:-1]):
                self.assertNotIn(local, text, f"{path.name} names a local path")
        self.assertEqual([row["engine"] for row in self.store.list()], [ENGINE])

    def test_the_installed_engine_serves_and_still_serves_after_the_store_moves(self):
        self.install()
        from crit_api.installed import InstalledEngine
        environment = patch.dict("os.environ", {"CRIT_RUNTIME_DIRECTORY": str(self.root / "runtime-owner")})
        environment.start()
        self.addCleanup(environment.stop)
        moved = self.root / "elsewhere"
        shutil.move(str(self.store.root), str(moved))
        engine = InstalledEngine(moved, ENGINE, host_limit=1 << 20, device_limit=1 << 20)
        self.addCleanup(engine.close)
        engine.open()
        self.assertEqual(engine.model_id(), ENGINE)
        self.assertEqual(engine.resolve_model("anything"), ENGINE)
        self.assertEqual(engine.candidates(), [{"candidate_id": ENGINE, "label": "fast-engine"}])
        status, assistant, _ = engine.generate([{"role": "user", "content": "hello"}], 16, threading.Event())
        self.assertEqual(status, "completed")
        self.assertEqual(json.loads(assistant)["weights"], "synthetic input, not a model")
        self.assertEqual(engine.spec()["limits"]["kv_tokens"], 4096)

    def test_a_second_pull_keeps_a_sound_install_and_a_changed_package_fails_to_serve(self):
        first = self.install()
        self.assertEqual(self.install()["installed_at"], first["installed_at"])
        (self.store.folder(ENGINE) / "package/adapter.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ"):
            self.store.spec(ENGINE)
        self.assertNotEqual(self.install()["installed_at"], first["installed_at"], "a broken install is pulled again")
        self.store.spec(ENGINE)

    def test_a_machine_that_misses_the_requirements_installs_nothing_unless_asked(self):
        write_json(self.device, DEVICE | {"compute_capability": "8.9", "memory_total_bytes": 24 * 10**9})
        with self.assertRaisesRegex(ValueError, r"gpu_arch: needs \['sm_120a'\].*\n.*gpu_memory_gb"):
            self.install()
        self.assertEqual(self.store.list(), [])
        record = self.install(ignore_requirements=True)
        self.assertTrue(record["requirements"]["ignored"])
        self.assertEqual([r["requirement"] for r in record["requirements"]["checks"] if not r["met"]],
                         ["gpu_arch", "gpu_memory_gb"])

    def test_a_checkpoint_that_differs_from_the_lock_is_refused(self):
        (self.model / "README.md").write_text("# Finetune\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "README.md: (SHA-256 differs|missing)"):
            self.install()
        self.assertEqual(self.store.list(), [])

    def test_a_configuration_other_than_the_sealed_runs_is_refused(self):
        lock = self.hub / "engines" / ENGINE / "card.lock.json"
        value = read_json(lock)
        value["provenance"]["configuration_sha256"] = "0" * 64
        write_json(lock, value)
        with self.assertRaisesRegex(ValueError, "not the configuration its sealed run measured"):
            self.install()

    def test_a_failed_build_keeps_its_log_tail_and_installs_nothing(self):
        self.hub_card(build={"command": ["python", "-c", "import sys; print('no compiler'); sys.exit(3)"]})
        with self.assertRaisesRegex(ValueError, r"(?s)The build failed \(3\).*no compiler"):
            self.install()
        self.assertEqual(self.store.list(), [])

    def test_the_card_can_come_from_crit_hubs_index(self):
        card = dict(self.card)
        lock = read_json(self.hub / "engines" / ENGINE / "card.lock.json")
        index = self.root / "index.json"
        write_json(index, {"engines": [{
            "path": f"engines/{ENGINE}", "name": "fast-engine", **{k: card[k] for k in (
                "summary", "origin", "contributor", "license", "features", "source", "requirements", "build")},
            "downloads": [], "lock": lock,
            "pairs": [{"gpu": "rtx5090", "model": "fixture@BF16", "status": "measured", "result": card["results"][0],
                       "checkpoint": card["checkpoints"]["fixture@BF16"]}]}]})
        record = self.install(hub=None, index=str(index))
        self.assertEqual(record["card"]["from"], str(index))
        self.assertEqual(record["build"]["outputs"], ["built.txt"])

    def test_an_engine_built_elsewhere_is_not_pulled(self):
        self.hub_card(origin="external")
        with self.assertRaisesRegex(ValueError, "crit pull installs engines exported from crit"):
            self.install()

    def test_the_command_line_installs_and_lists(self):
        arguments = [ENGINE, "--hub", str(self.hub), "--engines", str(self.root / "crit-engines"), "--model",
                     str(self.model), "--device", str(self.device), "--store", str(self.store.root), "--context", "2048"]
        with patch("sys.stdout"), patch("sys.stderr"):
            self.assertEqual(main(arguments), 0)
        self.assertEqual(self.store.get(ENGINE)["limits"]["kv_tokens"], 2048)
        with patch("sys.stderr"):
            self.assertEqual(main(["other/engine", "--hub", str(self.hub), "--store", str(self.store.root),
                                   "--device", str(self.device)]), 1)


class RequirementTests(unittest.TestCase):
    def test_architectures(self):
        self.assertTrue(machine.arch_runs("sm_120a", "12.0"))
        self.assertFalse(machine.arch_runs("sm_120a", "12.1"))
        self.assertTrue(machine.arch_runs("sm_120", "12.1"))
        self.assertFalse(machine.arch_runs("sm_120", "8.9"))
        self.assertTrue(machine.arch_runs("sm_90a", "9.0"))
        self.assertTrue(machine.arch_runs("sm_86", "8.9"))

    def test_toolkits_and_platforms(self):
        self.assertTrue(machine.at_least("13.2", "13.1"))
        self.assertFalse(machine.at_least("12.9", "13.1"))
        self.assertFalse(machine.at_least(None, "13.1"))
        rows = machine.check({"platforms": ["wsl2"], "gpu_arch": ["sm_120a"], "gpu_memory_gb": 32, "cuda": "13.1"},
                             DEVICE, cuda="13.1", here="linux")
        self.assertEqual(machine.unmet(rows), [])
        self.assertEqual(rows[0]["note"], "measured on wsl2")
        rows = machine.check({"platforms": ["wsl2"], "gpu_arch": ["sm_120a"], "gpu_memory_gb": 32, "cuda": "13.1"},
                             DEVICE, cuda="13.1", here="windows")
        self.assertEqual(machine.unmet(rows), ["platform: needs ['wsl2'], this machine has windows"])


if __name__ == "__main__":
    unittest.main()
