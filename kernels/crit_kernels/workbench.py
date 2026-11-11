"""The kernel workbench: a folder per kernel an engine writes or adapts, with its reference, the model's own shapes and a
yardstick provider. crit-eval measures it (`crit eval workbench check|bench|profile FOLDER`).

`crit kernels new NAME --op OP` scaffolds work/kernels/NAME (kernel.json, kernel.py, crit-eval's harness.py, src/):
for a matmul (gemv, gemm) in bf16, fp8 or nvfp4 its FP64 reference, a library yardstick and a candidate that calls the
kernel's op run_<format>; for any other operation the harness with the functions to fill in. src/ is a torch
extension project (sources plus a torch-ext binding): `--from DIR` copies an existing one (one of crit's kernels in
kernels/techniques/<family>/<id>, a provider's kernel source, a library sample) with its license, else kernel-builder's template
(`kernel-builder init`, on PATH or named by CRIT_KERNEL_BUILDER) starts it. `--model DIR` takes the shapes from the
checkpoint's matrices in the format. `--starter` starts from crit's starter matmul (kernels/starters/workbench-starter:
simple, correct, bf16 and fp8; or the kernel --from names); in a run it is allowed from the run's fourth candidate on
(`--run RUN_FOLDER`), so the first three candidates find their kernels themselves. Scaffolding imports nothing; it
runs only kernel-builder init. A kernel that proves out belongs in crit's kernel templates, with knobs (TEMPLATES.md).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

TEMPLATES = Path(__file__).resolve().parent / "templates"
MATMUL_OPS = {"gemv": [1, 2, 4, 8], "gemm": [64, 512, 2048]}
# format: (crit_core scheme, rate key). nvfp4 is W4A16: NVFP4 weights against 16-bit activations, the arithmetic of a
# decode GEMV on stored weights and of a preserve run's prompts; its yardstick reads weights widened to BF16 once.
MATMUL_FORMATS = {"bf16": ("plain", "bf16"), "fp8": ("fp8", "fp8"), "nvfp4": ("nvfp4", "bf16")}
YARDSTICKS = {"bf16": "lib:cuBLAS (torch.matmul in bf16)", "fp8": "lib:cuBLASLt (torch._scaled_mm, row-wise scales)",
              "nvfp4": "lib:cuBLAS (torch.matmul in bf16 on the weights widened to BF16 once)"}
STARTERS = ("bf16", "fp8")   # formats crit's workbench-starter has a starter kernel for
NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
LICENSES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "NOTICE")


def model_shapes(model, scheme):
    """The (N, K) of every matrix of `model` stored in `scheme` (crit_core's schemes), most frequent first."""
    from crit_core.model.safetensors import inspect_model
    from crit_core.model.schemes import stored
    tensors = inspect_model(Path(model), header_only=True).tensors
    config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") or config.get("quantization") or {}   # as quantization_of reads it
    counts = {}
    for name in tensors:
        for suffix in (".weight", ".weight_packed", ".qweight"):
            if not name.endswith(suffix) or len(tensors[name].shape) != 2:   # matrices, not norms or biases
                continue
            base = name[:-len(suffix)]
            found = stored(tensors, base, quantization)
            if found and found[0] == scheme:
                rows = tensors[name].shape[0]
                shape = (rows, found[2] // rows)
                counts[shape] = counts.get(shape, 0) + 1
    return [{"shape": list(shape), "matrices": n} for shape, n in sorted(counts.items(), key=lambda kv: -kv[1])]


def _fill(path, out, values):
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    out.write_text(text, encoding="utf-8", newline="\n")


STARTER_FROM = 4   # a run's candidate number from which `--starter` is allowed


def kernel_builder():
    """The kernel-builder executable: CRIT_KERNEL_BUILDER, else kernel-builder on PATH; None without one."""
    return os.environ.get("CRIT_KERNEL_BUILDER") or shutil.which("kernel-builder")


def init_project(src, name):
    """src/ as kernel-builder's template project for kernel NAME (`kernel-builder init`)."""
    tool = kernel_builder()
    if tool is None:
        raise ValueError("src/ is a kernel-builder project: install kernel-builder (or set CRIT_KERNEL_BUILDER) so "
                         "`kernel-builder init` starts it, or pass --from DIR with an existing project (crit's kernels "
                         "are in kernels/techniques/<family>/<id>)")
    done = subprocess.run([tool, "init", "--name", "crit-labs/" + name.replace("_", "-"), str(src)],
                          capture_output=True, text=True, timeout=120)
    if done.returncode != 0 or not (src / "build.toml").is_file():
        raise ValueError(f"kernel-builder init failed: {(done.stderr or done.stdout).strip()[-500:]}")
    shutil.rmtree(src / ".git", ignore_errors=True)   # the workbench folder belongs to the run's source, not a repo


def harness():
    """crit-eval's workbench harness, which every folder carries."""
    import crit_eval.workbench
    return crit_eval.workbench.HARNESS


def starter_allowed(run):
    """Whether run folder `run` has built enough candidates for a starter kernel: its next one is STARTER_FROM or later."""
    candidates = [p for p in (Path(run) / "candidates").glob("candidate-*") if (p / "candidate.json").is_file()]
    return len(candidates) + 1 >= STARTER_FROM


def new(name, op, root, fmt=None, shapes=(), model=None, columns=None, source=None, yardstick=None,
        column_invariant=None, tolerance=None, starter=False, run=None, repeatable=True):
    """Scaffold the workbench folder root/NAME. Returns the kernel.json it wrote. repeatable False: the kernel may give
    other bits on a repeat (atomics, a split the workload chooses), for engines in quality mode speed; the check
    then judges each repeat against the reference within the tolerance."""
    if not NAME.fullmatch(name):
        raise ValueError("NAME is lower-case letters, digits and _, starting with a letter")
    if starter:
        if source is None:
            from .build import find
            source = find("workbench-starter")[0]   # crit's starter matmul, kernels/starters/workbench-starter
        if run is None:
            raise ValueError("--starter needs --run RUN_FOLDER: a starter kernel is allowed from a run's candidate "
                             f"{STARTER_FROM} on")
        if not starter_allowed(run):
            raise ValueError(f"--starter is allowed from candidate {STARTER_FROM} on; {run} has not built "
                             f"{STARTER_FROM - 1} candidates yet")
        if op not in MATMUL_OPS or fmt not in STARTERS:
            raise ValueError("--starter has a starter kernel for gemv and gemm in " + " or ".join(STARTERS) + " only")
    folder = Path(root) / name
    if folder.exists():
        raise ValueError(f"{folder} exists: choose another name or remove it")
    matmul = op in MATMUL_OPS and fmt in MATMUL_FORMATS
    if op in MATMUL_OPS and fmt not in MATMUL_FORMATS:
        raise ValueError(f"--op {op} scaffolds a working kernel for --format " + " or ".join(MATMUL_FORMATS)
                         + "; use --op custom for another format and fill in kernel.py")
    found = [{"shape": [int(v) for v in s.split(",")], "matrices": None} for s in shapes]
    if model is not None:
        if not matmul:
            raise ValueError("--model reads matrix shapes: it goes with --op gemv or gemm")
        found += model_shapes(model, MATMUL_FORMATS[fmt][0])
    if not found:
        raise ValueError("Give the shapes: --shape N,K (repeatable) or --model MODEL_DIR")
    copied = []
    if source is None:
        folder.mkdir(parents=True)
        init_project(folder / "src", name)
    else:
        (folder / "src").mkdir(parents=True)
        source = Path(source)
        if not source.is_dir():
            raise ValueError(f"--from {source} is no directory")
        for path in sorted(source.rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                target = folder / "src" / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append({"path": path.relative_to(source).as_posix(),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if not any(Path(row["path"]).name in LICENSES for row in copied):
            raise ValueError(f"--from {source} holds no license file ({', '.join(LICENSES)}): check the source's terms "
                             "and copy its license beside it first")
    if any(path.suffix == ".cpp" and "TORCH_LIBRARY_EXPAND" in path.read_text(errors="replace")
           for path in (folder / "src").rglob("*.cpp")):
        shutil.copy2(TEMPLATES / "registration.h", folder / "src" / "registration.h")   # a kernel-builder torch-ext
    values = {"name": name, "op": op, "format": fmt or "custom", "module": "crit_wb_" + name,
              "yardstick": yardstick or YARDSTICKS.get(fmt, "the best existing provider"),
              "core_include": str(Path(__file__).resolve().parent.parent / "core" / "include")}
    if matmul:
        _fill(TEMPLATES / "matmul_kernel.py", folder / "kernel.py", values)
    else:
        _fill(TEMPLATES / "custom_kernel.py", folder / "kernel.py", values)
    shutil.copy2(harness(), folder / "harness.py")
    spec = {"schema_version": 1, "record_kind": "kernels_workbench", "name": name, "op": op, "format": fmt,
            "shapes": found, "columns": columns or MATMUL_OPS.get(op, [1]),
            "tolerance": tolerance if tolerance is not None else (1e-2 if fmt == "bf16" else 1e-3 if matmul else 1e-3),
            "column_invariant": bool(op == "gemv" if column_invariant is None else column_invariant),
            "repeatable": bool(repeatable),
            "rate": MATMUL_FORMATS[fmt][1] if matmul else None, "yardstick": values["yardstick"],
            "from": {"path": str(source), "files": copied} if source is not None else None,
            "starter": bool(starter)}
    (folder / "kernel.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8", newline="\n")
    return spec
