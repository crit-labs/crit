"""Compare crit::core::rope_table (crit_core_model_tests model_rope_print) with its Python twin on the configs of
tests/test_core_rope.py, refusals included: equal tables to 1e-12 relative, and both refusing the same configs."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "core" / "python"), str(ROOT / "tests")]

from crit_core.model import ConfigError, parse_model_config, rope_table  # noqa: E402
from test_core_rope import BASE, CASES  # noqa: E402

REFUSED = [({"rope_scaling": {"rope_type": "su", "factor": 2.0}}, 128),
           ({"rope_scaling": {"rope_type": "llama3", "factor": 8.0}}, 128),
           ({"partial_rotary_factor": 0.3}, 10)]


def main(executable):
    checked = 0
    with tempfile.TemporaryDirectory() as folder:
        runs = [(keys, 128, length) for _, keys, lengths in CASES for length in lengths]
        runs += [(dict(BASE, **keys), head_dim, None) for keys, head_dim in REFUSED]
        for keys, head_dim, length in runs:
            document = dict(BASE, **keys)
            (Path(folder) / "config.json").write_text(json.dumps(document), encoding="utf-8")
            line = subprocess.run([executable, "model_rope_print", folder, str(head_dim), "-" if length is None else str(length)],
                                  capture_output=True, text=True, check=True).stdout.strip()
            parsed = parse_model_config(document)
            try:
                table = rope_table(parsed.rope, head_dim, parsed.max_positions, length)
            except ConfigError:
                if not line.startswith("error "):
                    raise SystemExit(f"native accepts what Python refuses: {keys}: {line}")
                checked += 1
                continue
            if line.startswith("error "):
                raise SystemExit(f"native refuses what Python accepts: {keys}: {line}")
            native = [float(x) for x in line.split()]
            want = [table.attention_factor, *table.inv_freq]
            if len(native) != len(want) or not all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(native, want)):
                raise SystemExit(f"tables differ for {keys} at length {length}")
            checked += 1
    print(f"{checked} rotary tables agree with crit_core.model.rope_table")


if __name__ == "__main__":
    main(sys.argv[1])
