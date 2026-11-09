"""crit's wheel as pip installs it: built from a clean snapshot of HEAD (git archive, as CI's packaging step builds it)
and installed with crit-core's into a fresh virtual environment, it carries the plugin library, so crit plugins and crit
inference act on it there, plugins copy out of it and a library plugin's tests pass on the CPU. The environment borrows
this Python's installed packages (torch, tokenizers, crit Eval) behind its own. The build uses this Python's
setuptools when it is new enough (offline), else pip fetches one; the test skips without git, pip or a builder."""
from importlib import metadata
import io
import os
from pathlib import Path
import re
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
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
            f"import site; [site.addsitedir(p) for p in {borrowed!r}]\n", encoding="utf-8")
        cls.crit = shutil.which("crit", path=str(scripts))
        if cls.crit is None:
            raise AssertionError(f"pip installed no crit command in {scripts}")

    def crit_(self, *arguments, timeout=120):
        """crit ... from the installed environment, run in a folder outside every checkout."""
        result = run([self.crit, *arguments], cwd=self.work, timeout=timeout)
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


if __name__ == "__main__":
    unittest.main()
