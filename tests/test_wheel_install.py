"""crit's wheel as pip installs it: built from a clean snapshot of HEAD (git archive, as CI's packaging step builds it)
and installed with crit-core's into a fresh virtual environment, it carries the plugin library, so crit plugins and crit
inference act on it there, plugins copy out of it and a library plugin's tests pass on the CPU, and crit run keeps a
run without a checkout. The environment borrows this Python's installed packages (torch, tokenizers, crit Eval) behind
its own. The build uses this Python's setuptools when it is new enough (offline), else pip fetches one; the test skips
without git, pip or a builder."""
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
import venv

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tests" / "run_crit.py"
PLUGIN = "projections-dense"   # a base plugin on torch's CPU tensors, whose tests take seconds
BUILD = "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])"
# This process's variables less the ones that would put this checkout's sources first, with CUDA hidden.
ENVIRONMENT = {**{name: value for name, value in os.environ.items() if name not in {"PYTHONPATH", "PYTHONHOME"}},
               "CUDA_VISIBLE_DEVICES": "", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}


def run(command, **options):
    """The command's completed process, its output read as text unless text=False."""
    if options.setdefault("text", True):
        options.update(encoding="utf-8", errors="replace")
    return subprocess.run(command, env=ENVIRONMENT, capture_output=True, **options)


def version(text):
    return tuple(int(part) for part in re.findall(r"\d+", text)[:2])


def files(folder):
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


class WheelInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory(prefix="crit-wheel-", ignore_cleanup_errors=True)
        cls.addClassCleanup(temp.cleanup)
        cls.work = Path(temp.name)
        cls.source, dist, cls.prefix = cls.work / "source", cls.work / "dist", cls.work / "venv"
        cls.snapshot()
        cls.build(cls.source, dist)
        cls.build(cls.source / "core" / "python", dist)
        cls.install(sorted(dist.glob("*.whl")))

    @classmethod
    def snapshot(cls):
        if shutil.which("git") is None:
            raise unittest.SkipTest("git makes the snapshot (git archive HEAD)")
        archive = run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD"], text=False, timeout=60)
        if archive.returncode != 0:
            raise unittest.SkipTest(f"no git checkout to snapshot: {archive.stderr.decode(errors='replace').strip()}")
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(cls.source, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))

    @classmethod
    def build(cls, project, dist):
        """The project's wheel: with this Python's setuptools when it builds crit (build-system: setuptools>=77), as
        CI's packaging step does, else with pip's isolated build, which fetches setuptools."""
        try:
            local = version(metadata.version("setuptools")) >= (77,)
        except metadata.PackageNotFoundError:
            local = False
        if local:
            built = run([sys.executable, "-B", "-c", BUILD, str(dist)], cwd=project, timeout=300)
        else:
            built = run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(dist), str(project)],
                        timeout=300)
            if built.returncode != 0 and re.search(r"install(ing)? build dependencies", built.stdout + built.stderr):
                raise unittest.SkipTest("no setuptools>=77 here, and pip could not fetch one (offline?)")
        if built.returncode != 0:
            raise AssertionError(f"building {project}'s wheel failed:\n{built.stdout[-3000:]}{built.stderr[-3000:]}")

    @classmethod
    def install(cls, wheels):
        """A fresh environment holding the wheels, which this Python's pip installs into it (--python: it needs no pip
        of its own), with this Python's site folders behind its own for their dependencies."""
        pip = run([sys.executable, "-m", "pip", "--version"], timeout=60)
        if pip.returncode != 0 or version(pip.stdout.split()[1]) < (22, 3):
            raise unittest.SkipTest("pip>=22.3 installs the wheels (pip --python)")
        venv.EnvBuilder(with_pip=False).create(cls.prefix)
        cls.python = cls.prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        found = run([str(cls.python), "-c", "import sysconfig; print(sysconfig.get_path('purelib')); "
                     "print(sysconfig.get_path('scripts'))"], timeout=60)
        cls.purelib, scripts = (Path(line).resolve() for line in found.stdout.splitlines())
        installed = run([sys.executable, "-m", "pip", "--python", str(cls.python), "install", "--no-deps", "--no-index",
                         *map(str, wheels)], timeout=300)
        if installed.returncode != 0:
            raise AssertionError(f"pip install failed:\n{installed.stdout[-3000:]}{installed.stderr[-3000:]}")
        borrowed = site.getsitepackages() + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else [])
        (cls.purelib / "crit-test-borrowed.pth").write_text(
            f"import site; list(map(site.addsitedir, {borrowed!r}))\n", encoding="utf-8")
        cls.crit = shutil.which("crit", path=str(scripts))
        if cls.crit is None:
            raise AssertionError(f"pip installed no crit command in {scripts}")

    def crit_(self, *arguments, timeout=120, cwd=None):
        """crit ... from the installed environment, run in a folder outside every checkout (default the test's)."""
        result = run([self.crit, *arguments], cwd=cwd or self.work, timeout=timeout)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_the_installed_crit_and_its_library_are_the_wheel_s(self):
        probe = run([str(self.python), "-c", "import crit, crit.inference, crit_plugins; print(crit.__file__); "
                     "print(crit.inference.LIBRARY); print(crit_plugins.__file__)"], cwd=self.work, timeout=60)
        self.assertEqual(probe.returncode, 0, probe.stderr)
        package, library, plugins = (Path(line).resolve() for line in probe.stdout.splitlines())
        self.assertEqual(package.parent, self.purelib / "crit")
        self.assertEqual(library, self.purelib / "crit" / "library" / "inference")
        self.assertEqual(plugins.parent, self.purelib / "crit_plugins")
        # Every file of the library ships but Inference's native build entry and walkthrough (pyproject.toml).
        checkout, shipped = files(self.source / "inference"), files(library)
        self.assertEqual(checkout - shipped, {"CMakeLists.txt"} | {p for p in checkout if p.startswith("examples/")})
        self.assertEqual(shipped - checkout, set())

    def test_an_installed_crit_mcp_serves_every_guide(self):
        # Outside every checkout, crit mcp's resources are the guides the wheel carries: AGENTS.md first, as its
        # instructions ask, the same bytes as the checkout's.
        messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
                    {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
                    {"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": "crit://AGENTS.md"}}]
        server = subprocess.Popen([self.crit, "mcp", "--only", "inference"], cwd=self.work, env=ENVIRONMENT,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        answers = {}
        try:
            for message in messages:  # one at a time: each answer before the next request, and before the input ends
                server.stdin.write(json.dumps(message).encode() + b"\n")
                server.stdin.flush()
                answer = json.loads(server.stdout.readline())
                answers[answer["id"]] = answer["result"]
        finally:
            server.stdin.close()
            server.wait(timeout=60)
            server.stdout.close()
        shipped = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        guides = shipped["tool"]["setuptools"]["package-data"]["crit.library.guides"] + ["inference/README.md"]
        listed = [resource["uri"] for resource in answers[2]["resources"]]
        self.assertEqual(listed[0], "crit://AGENTS.md")
        self.assertEqual(sorted(listed), sorted("crit://" + guide for guide in guides))
        self.assertEqual(answers[3]["contents"][0]["text"], (self.source / "AGENTS.md").read_text(encoding="utf-8"))

    def test_an_installed_crit_names_the_crit_eval_release_it_pins(self):
        # Without crit Eval, an installed crit says to install the release it pins, not a checkout beside it.
        found = run([str(self.python), "-c", "from crit.cli import eval_install; print(eval_install())"],
                    cwd=self.work, timeout=60)
        self.assertEqual(found.returncode, 0, found.stderr)
        pinned = re.search(r'"crit-eval==([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))[1]
        self.assertEqual(found.stdout.strip(), f"python -m pip install crit-eval=={pinned}")

    def test_crit_plugins_list_and_crit_inference_act_on_the_shipped_library(self):
        listed = self.crit_("plugins", "list")
        source = subprocess.run([sys.executable, "-B", str(LAUNCHER), "plugins", "list"], capture_output=True,
                                text=True, encoding="utf-8", timeout=120)
        self.assertEqual(source.returncode, 0, source.stderr)
        self.assertEqual(listed, source.stdout)   # the checkout's library, plugin for plugin
        self.assertRegex(listed, rf"(?m)^{PLUGIN} +base +projections ")
        self.assertIn("crit Inference passed", self.crit_("inference", "check"))

    def test_a_library_plugin_s_tests_pass_in_the_installed_crit_on_the_cpu(self):
        if run([str(self.python), "-c", "import torch"], cwd=self.work, timeout=120).returncode != 0:
            self.skipTest(f"{PLUGIN}'s tests compute on torch's tensors, and this Python has no torch")
        out = self.crit_("plugins", "test", "--only", PLUGIN, timeout=600)
        self.assertRegex(out, rf"passed +{PLUGIN} ")
        self.assertIn("1 passed, 0 failed", out)

    def test_crit_plugins_add_copies_a_plugin_out_of_the_installed_crit(self):
        engine = self.work / "engine" / "plugins"
        self.assertRegex(self.crit_("plugins", "add", str(engine), PLUGIN), rf"added +{PLUGIN} ")
        self.assertEqual(files(engine / "base" / PLUGIN), files(self.source / "inference" / "base" / PLUGIN))
        self.assertEqual(list((engine / "base" / PLUGIN).rglob("__pycache__")), [])

    def test_crit_run_keeps_a_run_of_the_toy_engine_without_a_checkout(self):
        """The folder crit run runs in is its project, holding settings.toml and runs/; the run's brief names the
        installed crit's release; and a run of the CPU toy engine goes from create through recon, capture, resolve and
        its first plan to a candidate, built and packaged, with no record in error."""
        from core_text_fixtures import llama_files, write_files
        from support.engines import TOY, selection
        project = self.work / "project"
        model = write_files(project / "model", llama_files())
        (project / "settings.toml").write_text("ram_headroom_gib = 3\n", encoding="utf-8")
        (project / "request.json").write_text(json.dumps({"model_path": str(model), "kv_tokens": 8192,
                                                          "correctness_criteria": "Fixture gates"}), encoding="utf-8")

        def crit(*arguments, record=None):
            if record is not None:
                (project / (arguments[-1])).write_text(json.dumps(record), encoding="utf-8")
            return json.loads(self.crit_("run", *arguments, cwd=project))

        run = crit("create", "--record", "request.json")["run"]
        run_id, folder = run["run_id"], project / "runs" / run["run_id"]
        self.assertEqual(run["spec"]["hardware_constraints"]["ram_headroom_bytes"], 3 << 30)   # the project's setting
        version = self.crit_("--version").split()[1]
        self.assertIn(f"\nRepository: https://github.com/crit-labs/crit/tree/v{version}, the release of the installed "
                      f"crit {version} ({self.purelib / 'crit'})\n",
                      (folder / "agent_request.md").read_text(encoding="utf-8"))
        crit("--id", run_id, "recon")
        notes = folder / "work" / "recon.md"
        notes.write_bytes(re.sub(rb"(?m)^TODO\.?", b"Written for this test.", notes.read_bytes()))
        self.assertEqual(crit("--id", run_id, "capture", "--inputs", "inputs.json")["prompt_tokens"], 512)
        inputs = json.loads((project / "inputs.json").read_text(encoding="utf-8"))
        crit("--id", run_id, "resolve", "--record", "inputs.json", record={
            **inputs, "hardware_manifest": {"device": "host"}, "environment_manifest": {"benchmark": "synthetic"}})
        plan = crit("--id", run_id, "plan", "--record", "plan.json", record={
            "parent_candidate_id": None, "decision": "build", "gaps": [], "levers": [], "rejected": [],
            "needed": {kpi: {"needed": True, "evidence": "no Eval yet"} for kpi in ("decode", "prefill")},
            "strategy": {"current": {"name": "plain decode", "ceiling": "unknown", "evidence": "none yet"},
                         "alternative": {"name": "drafting", "ceiling": "unknown", "evidence": "none yet"},
                         "choice": "current", "reason": "the first candidate"},
            "forecast": None, "gates": "Quality stays inside the budget", "prompt_activations": "stored"})
        shutil.copytree(TOY, project / "source" / "plugins", ignore=shutil.ignore_patterns("__pycache__"))
        candidate = crit("--id", run_id, "candidate", "--source", "source", "--record", "proposal.json", record={
            "label": "Toy engine", "rationale": "The CPU toy engine", "strategy": "plain decode",
            "configuration": {"plugins": selection()}, "store_decisions": {}, "plan_id": plan["plan_id"],
            "lever_fates": {}})["candidate_id"]
        built = crit("--id", run_id, "build", "--candidate", candidate, "--record", "recipe.json", record={"build": {
            "toolchain": {"python": sys.version.split()[0]},
            "command": [str(self.python), "-c", "open('built.txt', 'w').write('built')"]}})
        self.assertEqual(built["attempt"]["outcome"], "completed")
        self.assertEqual([d["path"] for d in built["package"]["build"]["dependencies"]], ["built.txt"])
        view = crit("--id", run_id, "show")
        self.assertEqual(view["errors"], [])
        self.assertEqual([(c["candidate_id"], c["artifacts"]["package"]) for c in view["candidates"]],
                         [(candidate, f"candidates/{candidate}/package")])


if __name__ == "__main__":
    unittest.main()
