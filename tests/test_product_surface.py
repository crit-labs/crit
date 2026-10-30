"""Navigation, CLI and Inference regressions using temporary host fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import os
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cli"))
from crit import inference as store
LAUNCHER = str(ROOT / "tests/run_crit.py")
PRODUCT_FOLDERS = ("core", "inference", "kernels", "models", "eval", "api", "tests", "docs", "cli", ".github")


class ProductTests(unittest.TestCase):
    def fixture(self, root):
        entry = copy.deepcopy(store.load()["entries"][0])
        entry.update(id="example", path="inference/example/README.md", code_path="inference/example/code",
                     summary="Generic contract </script><img src=x>",
                     code_files=[{"path": "inference/example/code/implementation.py", "purpose": "Never executed by navigation"}])
        feature = root / "inference/example"
        (feature / "code").mkdir(parents=True)
        (feature / "entry.json").write_text(json.dumps(entry), encoding="utf-8")
        (feature / "README.md").write_text("# Scoped contract\n", encoding="utf-8")
        (feature / "code/implementation.py").write_text("raise RuntimeError('must never import during navigation')\n", encoding="utf-8")
        return entry

    def test_navigation_and_checks_work_without_generated_files_or_code_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            before = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
            self.assertEqual(store.check(root), [])
            self.assertEqual(store.load(root)["entries"], [entry])
            self.assertEqual(sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file()), before)

    def test_queries_see_authored_metadata_changes_without_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            self.assertEqual(store.load(root)["entries"][0]["summary"], entry["summary"])
            entry["summary"] = "Newly authored search guidance"
            (root / "inference/example/entry.json").write_text(json.dumps(entry), encoding="utf-8")
            self.assertEqual(store.load(root)["entries"][0]["summary"], entry["summary"])
            self.assertEqual(store.check(root), [])

    def test_metadata_only_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            (root / entry["code_files"][0]["path"]).unlink()
            errors = store.check(root)
            self.assertTrue(any("no actual implementation" in error for error in errors))

    def test_agent_guidance_must_distinguish_code_from_engine_responsibilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            for field, value in (("provides", []), ("engine_supplies", "backend"),
                                 ("preserve", [None]), ("adapt", [""]), ("use_when", " ")):
                with self.subTest(field=field):
                    entry = copy.deepcopy(original)
                    entry["agent_guidance"][field] = value
                    (root / "inference/example/entry.json").write_text(json.dumps(entry), encoding="utf-8")
                    self.assertTrue(any("agent_guidance." + field in error for error in store.check(root)))
            entry.pop("agent_guidance")
            (root / "inference/example/entry.json").write_text(json.dumps(entry), encoding="utf-8")
            self.assertTrue(any("agent selection" in error for error in store.check(root)))

    def test_category_is_required_from_the_fixed_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.fixture(root)
            for value in (None, "", "Attention", "kernels", ["attention"]):
                with self.subTest(category=value):
                    entry = copy.deepcopy(original)
                    if value is None:
                        entry.pop("category")
                    else:
                        entry["category"] = value
                    (root / "inference/example/entry.json").write_text(json.dumps(entry), encoding="utf-8")
                    self.assertTrue(any("category must be one of" in error for error in store.check(root)))

    def test_list_groups_entries_by_category(self):
        entries = store.load()["entries"]
        self.assertEqual({e["category"] for e in entries}, set(store.CATEGORIES))
        listed = subprocess.run([sys.executable, "-B", LAUNCHER, "inference", "list"],
                                capture_output=True, text=True, timeout=20, check=True).stdout.splitlines()
        headers = [line[:-1] for line in listed if not line.startswith(" ")]
        self.assertEqual(headers, list(store.CATEGORIES))
        current = None
        for line in listed:
            if not line.startswith(" "):
                current = line[:-1]
            else:
                ident = line.strip().split(":", 1)[0]
                self.assertEqual(next(e["category"] for e in entries if e["id"] == ident), current)
        self.assertEqual(sum(line.startswith("  ") for line in listed), len(entries))

    def test_external_bundle_metadata_and_nonlocal_code_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            entry["donors"] = ["external-source"]
            entry["code_files"][0]["path"] = "research/implementation.py"
            (root / "inference/example/entry.json").write_text(json.dumps(entry), encoding="utf-8")
            errors = store.check(root)
            self.assertTrue(any("source-bundle" in error for error in errors))
            self.assertTrue(any("inside the feature" in error for error in errors))

    def test_paths_cannot_escape_or_address_external_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ("../private", "/private", "C:/private", "a\\b", "a//b", "a/../b", "https://example.invalid/source"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    store.under(root, path)


    def test_whole_folders_include_unknown_extensions_and_exclude_local_roots(self):
        allowed = ["README.md", "CLAUDE.md", "inference/example/code/new.extension", "core/deep/data.json",
                   "tests/fixture.raw", "docs/new.record", "cli/new.extension", "api/new.extension", "kernels/new.extension", "models/new.extension", ".github/workflows/ci.yml"]
        ignored = ["scratch.txt", "tools/deep/source.anything", "private/source.py", "archive/history.json", "evidence/result.json", "build/output.cpp",
                   "dist/package.json", "runs/ledger.json", ".idea/workspace.xml", "research/source.cpp",
                   "temp/new-ui.py", "other-ui-proposal/source.py", "ui/radar.py"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            result = subprocess.run(["git", "-c", "core.excludesfile=", "check-ignore", "--no-index", "--stdin", "-z"],
                                    cwd=root, input=b"\0".join(p.encode() for p in allowed + ignored) + b"\0", capture_output=True)
            self.assertIn(result.returncode, (0, 1))
            actual = {p.decode() for p in result.stdout.split(b"\0") if p}
            self.assertEqual(actual, set(ignored))

    def test_real_inference_entries_are_generic_local_implementations(self):
        self.assertEqual(store.check(), [])
        entries = store.load()["entries"]
        self.assertEqual(len(entries), 28)
        for entry in entries:
            self.assertNotIn("donors", entry)
            self.assertNotIn("code_locations", entry)
            self.assertTrue(all(c["path"].startswith(entry["code_path"] + "/") for c in entry["code_files"]))
            self.assertTrue(entry["validation"]["native"].strip())

    def test_cli_queries_only_the_current_product(self):
        result = subprocess.run([sys.executable, "-B", LAUNCHER, "inference", "show",
                                 "expert-residency", "--json"], capture_output=True, timeout=20, check=True)
        entry = json.loads(result.stdout)
        self.assertEqual(entry["id"], "expert-residency")
        self.assertIn("ExpertCache", entry["symbols"])
        shown = subprocess.run([sys.executable, "-B", LAUNCHER, "inference", "show",
                                "expert-residency"], capture_output=True, text=True, timeout=20, check=True)
        self.assertIn("Engine supplies:", shown.stdout)
        self.assertIn(entry["agent_guidance"]["provides"][0], shown.stdout)
        self.assertIn(entry["agent_guidance"]["engine_supplies"][0], shown.stdout)
        self.assertIn(entry["path"], shown.stdout)
        searched = subprocess.run([sys.executable, "-B", LAUNCHER, "inference", "search",
                                   "state reconciliation", "--json"], capture_output=True, timeout=20, check=True)
        self.assertIn("mtp-decode", [item["id"] for item in json.loads(searched.stdout)])
        contracts = subprocess.run([sys.executable, "-B", LAUNCHER, "contracts"],
                                   capture_output=True, timeout=20, check=True)
        self.assertIn(b"core/include/crit/core.h", contracts.stdout)
        with tempfile.TemporaryDirectory() as directory:
            kernels = subprocess.run([sys.executable, "-B", "-S", LAUNCHER, "kernels"],
                                   cwd=directory, capture_output=True, text=True, timeout=10, check=True)
            self.assertIn("https://huggingface.co/kernels", kernels.stdout)
            self.assertIn("kernels/README.md", kernels.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_native_build_source_and_test_paths_exist(self):
        cmake = (ROOT / "core/CMakeLists.txt").read_text(encoding="utf-8")
        for path in ("src/runtime/runtime.cpp", "src/memory/resources.cpp", "src/backends/cuda/cuda.cpp",
                     "../tests/core/runtime_tests.cpp", "../tests/core/concurrency_tests.cpp", "../tests/core/cuda_tests.cpp"):
            self.assertIn(path, cmake)
            self.assertTrue((ROOT / "core" / path).is_file())

    def test_core_never_builds_or_includes_inference_code(self):
        for path in (ROOT / "core").rglob("*"):
            if path.is_file() and path.suffix in {".txt", ".cmake", ".in", ".h", ".hpp", ".cpp", ".cu", ".py"}:
                self.assertNotRegex(path.read_text(encoding="utf-8").lower(), r"inference/|crit_inference|crit::inference", path)
        for path in (ROOT / "tests/core").rglob("*"):
            if path.is_file():
                self.assertNotIn("inference/", path.read_text(encoding="utf-8"), path)
        cmake = (ROOT / "inference/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../core"', cmake)
        for source in re.findall(r"\$\{FIXTURES\}/([\w./]+)", cmake):
            self.assertTrue((ROOT / "tests/inference" / source).is_file(), source)

    def test_cli_only_package_has_one_entrypoint_and_no_browser_sources(self):
        package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(package["project"]["scripts"], {"crit": "crit.cli:main"})
        self.assertEqual(package["tool"]["setuptools"]["packages"]["find"]["where"], ["cli", "api", "kernels", "models"])
        self.assertEqual({p.name for p in (ROOT / "cli").iterdir() if p.is_dir()}, {"crit"})
        self.assertEqual({p.name for p in (ROOT / "api").iterdir() if p.is_dir()}, {"crit_api"})
        self.assertEqual({p.name for p in (ROOT / "kernels").iterdir() if p.is_dir()}, {"crit_kernels", "examples"})
        self.assertEqual({p.name for p in (ROOT / "models").iterdir() if p.is_dir()}, {"crit_models"})
        assets = [p for folder in PRODUCT_FOLDERS for p in (ROOT / folder).rglob("*")
                  if p.is_file() and p.suffix in {".html", ".css", ".js"}]
        self.assertEqual(assets, [])
        rejected = subprocess.run([sys.executable, "-S", "-B", LAUNCHER, "gui", "--help"],
                                  capture_output=True, timeout=10)
        self.assertEqual(rejected.returncode, 2)

    def test_core_python_package_ships_the_root_license(self):
        self.assertEqual((ROOT / "core/python/LICENSE").read_bytes(), (ROOT / "LICENSE").read_bytes())

if __name__ == "__main__":
    unittest.main()
