"""Native CTest entry: retained synthetic ABI fixture -> guarded operator check."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
library, executable, bits = sys.argv[1:4]
profile = sys.argv[4] if len(sys.argv) == 5 else "gemm"
cases = 2304 if profile == "attention" else 648
with tempfile.TemporaryDirectory(prefix="crit-blas-") as folder:
    root = Path(folder)
    notice = root / "NOTICE.txt"
    notice.write_text("Synthetic test fixture, not OpenBLAS. No performance or license claim.\n")
    env = dict(os.environ, CRIT_RUNTIME_DIRECTORY=str(root / "runtime"))
    def run(*args):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "kernels/examples/openblas/example.py"), *map(str, args)],
            cwd=root, env=env, capture_output=True, encoding="utf-8", timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(result.stdout)
    run("prepare", "--library", library, "--work", root / "work", "--notice", notice,
        "--integer-bits", bits, "--sgemm-symbol", "fixture_sgemm", "--config-symbol", "fixture_config",
        "--threads-symbol", "fixture_threads", "--variant", "synthetic", "--license", "LicenseRef-SyntheticFixture")
    result = run("check", "--profile", profile, "--dependency", root / "work/dependency", "--executable", executable, "--output", root / "results")
    if result["status"] != "passed" or not result["cleanup_settled"] or result["native"]["cases"] != cases:
        raise RuntimeError("Native pipeline did not pass all checks and cleanup")
    if (root / "runtime/owner.json").exists():
        raise RuntimeError("Settled native pipeline retained runtime ownership")
    print(f"Retained dependency, {cases} native {bits}-bit ABI {profile} cases and guarded cleanup passed")
