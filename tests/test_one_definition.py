"""Each calculation Core owns has one definition: a static check that no second copy of it comes back (the "Core owns
it" pass, wsl-evidence CORE-OWNS.md). Every rule here names the one definition and the pattern a copy would take;
the scan reads crit's Python and CUDA sources (case studies and the scanned definition itself aside)."""
import ast
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SKIP = ("case-studies", "__pycache__", ".git", "build", "node_modules")


SOURCES = ("api", "cli", "inference", "kernels", "mcp", "models", "smoke", "tests", "tools")


def files(*globs):
    """The source files of the globs ("**/*.py": every source folder's), case studies and builds aside."""
    for pattern in globs:
        roots = [ROOT / folder for folder in SOURCES] if pattern.startswith("**/") else [ROOT]
        for path in sorted(p for root in roots for p in root.glob(pattern)):
            if not any(part in SKIP for part in path.relative_to(ROOT).parts):
                yield path


def text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def functions(path, names):
    """The source text of the module-level functions `names` of a Python file."""
    source = text(path)
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
    return found


class OneDefinitionTests(unittest.TestCase):
    def assertNoCopies(self, pattern, globs, allowed=(), why=""):
        rule = re.compile(pattern, re.M)
        copies = [f"{rel(p)}:{text(p)[:m.start()].count(chr(10)) + 1}" for p in files(*globs)
                  if rel(p) not in allowed for m in rule.finditer(text(p))]
        self.assertEqual(copies, [], why)

    def test_exactness_levels_quality_modes_and_the_grace_are_cores(self):
        self.assertNoCopies(r"^(LEVELS|QUALITY_MODES|EXACTNESS_LEVELS)\s*=\s*\(", ["**/*.py"],
                            ("inference/core/python/crit_plugins/exactness.py", "tests/test_one_definition.py"),
                            "crit_plugins.exactness defines the levels and modes")
        self.assertNoCopies(r'^DEFAULT_QUALITY_MODE\s*=\s*"', ["**/*.py"],
                            ("inference/core/python/crit_plugins/exactness.py",), "crit_plugins.exactness")
        self.assertNoCopies(r"^(GRACE|CANCEL_GRACE_SECONDS)\s*=\s*\d", ["**/*.py"],
                            ("inference/core/python/crit_plugins/steps.py",), "crit_plugins.steps.GRACE")

    def test_headrooms_and_what_fits_are_cores(self):
        memory = "inference/core/python/crit_plugins/memory.py"
        self.assertNoCopies(r"^(HEADROOM|HOST_HEADROOM)\s*=", ["**/*.py"], (memory,), "crit_plugins.memory's")
        # the KV that fits and the bytes a context needs: kv_fit and kv_need, never a sum of the basis' parts
        self.assertNoCopies(r'\[\s*"vram_headroom_bytes"\s*\]\s*\+', ["cli/**/*.py", "api/**/*.py"], (),
                            "crit_plugins.memory.kv_need")
        self.assertNoCopies(r"ram_headroom_bytes\s*=\s*2\s*\*|--ram-headroom-gib.*default\s*=", ["cli/**/*.py"], (),
                            "crit.settings.ram_headroom_bytes, else crit_plugins.memory.HOST_HEADROOM")

    def test_the_small_engine_and_the_routed_slots_have_one_definition(self):
        self.assertNoCopies(r'"kv_margin"\s*:\s*256', ["**/*.py"],
                            ("models/core/python/crit_models/engines.py",), "crit_models.engines.SMALL_OPTIONS")

    def test_the_step_settings_are_cores_and_the_model_declares_what_it_takes(self):
        options = json.loads(text(ROOT / "models/core/model-template/plugin.json"))["options"]
        for name in ("concurrency", "pending", "activations_mib", "span_seconds"):
            self.assertEqual(options[name]["default"], -1, name)   # crit_plugins.steps.CORE: Core decides
        self.assertNoCopies(r"^(LANES|PENDING)\s*=|^ACTIVATIONS\s*=\s*\d+\s*<<",
                            ["models/**/*.py", "cli/**/*.py", "api/**/*.py", "inference/core/**/*.py"],
                            ("inference/core/python/crit_plugins/steps.py",), "crit_plugins.steps")

    def test_the_kv_page_is_the_kv_cache_plugins(self):
        self.assertNoCopies(r"page=PAGE\b|//\s*PAGE\b|\*\s*PAGE\b", ["models/core/model-template/code/*.py"], (),
                            "the model's pools take the kv_cache plugin's page (ComposedModel.page)")

    def test_the_decode_rows_are_cores_and_a_kernels_row_limit_its_own(self):
        self.assertNoCopies(r"^DECODE_ROWS\s*=", ["inference/**/*.py", "models/**/*.py"],
                            ("inference/core/python/crit_plugins/device.py",), "crit_plugins.device.DECODE_ROWS")
        self.assertNoCopies(r"^MAX_ROWS\s*=\s*\d", ["inference/**/*.py", "models/**/*.py"], (),
                            "a kernel's row limit is its module's (gguf_rows.MAX_ROWS)")

    def test_the_draft_width_rule_is_cores_round_drivers(self):
        self.assertNoCopies(r"^class\s+\w*Width\w*Policy\b|^class\s+WidthPolicy\b", ["inference/techniques/**/*.py",
                            "models/**/*.py"], (), "crit_plugins.rounds' width policies")

    def test_summation_order_rules_count_the_orders_multiprocessors(self):
        rules = {"kernels/techniques/attention/paged-kv-attention/torch-ext/paged_kv_attention/__init__.py":
                 ("partitions", "_cuda_chunk_rule"),
                 "kernels/techniques/attention/latent-attention/torch-ext/latent_attention/__init__.py":
                 ("partitions",),
                 "kernels/techniques/projections/fixed-tile-gemm/torch-ext/fixed_tile_gemm/__init__.py":
                 ("split_tile",)}
        device = re.compile(r"multi_processor_count|_multiprocessors\(|_sms\(|_capability\([^)]*\)\[1\]")
        for path, names in rules.items():
            found = functions(ROOT / path, names)
            self.assertEqual(sorted(found), sorted(names), path)
            for name, source in found.items():
                self.assertIsNone(device.search(source), f"{path} {name}: ORDER_MULTIPROCESSORS, never the device's")
        for path in ("kernels/techniques/projections/gguf-rows/gguf_rows_cuda/binding.cu",
                     "kernels/techniques/projections/exl3-rows/exl3_rows_cuda/binding.cu"):
            self.assertNotIn("cudaDevAttrMultiProcessorCount", text(ROOT / path), path)


if __name__ == "__main__":
    unittest.main()
