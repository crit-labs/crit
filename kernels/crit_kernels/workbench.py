"""The kernel workbench: a folder per kernel an engine writes or adapts, with its reference, the model's own shapes and a
yardstick provider. crit-eval measures it (`crit eval workbench check|bench|profile FOLDER`).

`crit kernels new NAME --op OP` scaffolds work/kernels/NAME (kernel.json, kernel.py, crit-eval's harness.py, src/):
for a matmul (gemv, gemm) in bf16, fp8 or nvfp4 its FP64 reference, a library yardstick and a binding whose kernel the engine
writes; for any other operation the harness with the functions to fill in. `--from DIR` copies an existing kernel's
source (a Hub kernel, a library sample) into src/ with its license, as the starting point. `--model DIR` takes the
shapes from the checkpoint's matrices in the format. `--starter` writes a simple, correct, portable kernel to start
from instead of the empty binding; in a run it is allowed from the run's fourth candidate on (`--run RUN_FOLDER`), so
the first three candidates find their kernels themselves. Scaffolding imports and runs nothing.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

TEMPLATES = Path(__file__).resolve().parent / "templates"
MATMUL_OPS = {"gemv": [1, 2, 4, 8], "gemm": [64, 512, 2048]}
# format: (crit_core scheme, rate key). nvfp4 is W4A16: NVFP4 weights against 16-bit activations, the arithmetic of a
# decode GEMV on stored weights and of a preserve run's prompts; its yardstick reads weights widened to BF16 once.
MATMUL_FORMATS = {"bf16": ("plain", "bf16"), "fp8": ("fp8", "fp8"), "nvfp4": ("nvfp4", "bf16")}
YARDSTICKS = {"bf16": "lib:cuBLAS (torch.matmul in bf16)", "fp8": "lib:cuBLASLt (torch._scaled_mm, row-wise scales)",
              "nvfp4": "lib:cuBLAS (torch.matmul in bf16 on the weights widened to BF16 once)"}
STARTERS = ("bf16", "fp8")   # formats --starter has a starter kernel for
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


def harness():
    """crit-eval's workbench harness, which every folder carries."""
    import crit_eval.workbench
    return crit_eval.workbench.HARNESS


def starter_allowed(run):
    """Whether run folder `run` has built enough candidates for a starter kernel: its next one is STARTER_FROM or later."""
    candidates = [p for p in (Path(run) / "candidates").glob("candidate-*") if (p / "candidate.json").is_file()]
    return len(candidates) + 1 >= STARTER_FROM


def new(name, op, root, fmt=None, shapes=(), model=None, columns=None, source=None, yardstick=None,
        column_invariant=None, tolerance=None, starter=False, run=None):
    """Scaffold the workbench folder root/NAME. Returns the kernel.json it wrote."""
    if not NAME.fullmatch(name):
        raise ValueError("NAME is lower-case letters, digits and _, starting with a letter")
    if starter:
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
    (folder / "src").mkdir(parents=True)
    copied = []
    if source is not None:
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
            shutil.copy2(TEMPLATES / "registration.h", folder / "src" / "registration.h")   # a Hub kernel's torch-ext
    values = {"name": name, "op": op, "format": fmt or "custom", "module": "crit_wb_" + name,
              "yardstick": yardstick or YARDSTICKS.get(fmt, "the best existing provider")}
    if matmul and source is None:
        _fill(TEMPLATES / "matmul_kernel.py", folder / "kernel.py", values)
        _fill(TEMPLATES / ("matmul_kernel.cu" if starter else "matmul_stub.cu"), folder / "src" / f"{name}.cu", values)
    else:
        _fill(TEMPLATES / "custom_kernel.py", folder / "kernel.py", values)
    shutil.copy2(harness(), folder / "harness.py")
    spec = {"schema_version": 1, "record_kind": "kernels_workbench", "name": name, "op": op, "format": fmt,
            "shapes": found, "columns": columns or MATMUL_OPS.get(op, [1]),
            "tolerance": tolerance if tolerance is not None else (1e-2 if fmt == "bf16" else 1e-3 if matmul else 1e-3),
            "column_invariant": bool(op == "gemv" if column_invariant is None else column_invariant),
            "rate": MATMUL_FORMATS[fmt][1] if matmul else None, "yardstick": values["yardstick"],
            "from": {"path": str(source), "files": copied} if source is not None else None,
            "starter": bool(starter)}
    (folder / "kernel.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8", newline="\n")
    return spec
