"""Quantized weights in the projections slot: quantized-projections' grouped integer quantize, dequantize and project
(implementation.py) or its NVFP4 codec and projection (nvfp4.py), as the `format` option chooses, as the host form;
on the device, a checkpoint's stored BF16, FP8 or NVFP4 matrices on the engine's device, multiplied through cuBLASLt
(device_matrices.py), NVFP4 ones with NVFP4 or 16-bit activations as `activations` says (device_nvfp4.py: crit's
kernels that plugin.json declares, built on this machine's GPU, or torch fallbacks)."""
from pathlib import Path

from .quantized import device_matrices, implementation as grouped, nvfp4

device_matrices.KERNEL_FOLDER = Path(__file__).resolve().parent   # its plugin.json declares crit's kernels


class DeviceMatrices:
    """The device form, whatever the host form's format: load(tensors, format) and matmul(x, matrix). Matrices land
    on the engine's device whatever device the given tensors are on."""

    device, activations, prompt, gemm, gemv, loaded, roles = None, "nvfp4", "bf16", "auto", "auto", frozenset(), {}
    decode, decode_rows, rows, tile_rows, invariant = "prompt", None, device_matrices.GEMV_ROWS, "auto", None
    fp8_activations = "fp8"
    decode_formats = {}   # decode_formats: a format's decode activations where they differ from decode_activations

    def load(self, tensors, format, role=None, encode=None):
        """role: the matrix's role in the model, as the model plugin names it (qkv, gate_up, ...), which picks its
        prompt arithmetic where prompt_roles names it. encode: for a matrix only a drafter reads (an MTP layer, a
        draft head; the model plugin passes it only for draft-only roles), the format to re-encode the stored one to
        at load (device_matrices.ENCODINGS: nvfp4, fp8); an nvfp4 one multiplies 16-bit activations (no calibrated
        activation scale), and facts() records the group under draft_encoding (a matrix already stored in the
        encode format stays as stored and is recorded the same way: draft-only all the same)."""
        activations, draft = self.activations, encode is not None
        if draft:
            stored, before = format, device_matrices.stored_bytes(tensors, format)
            if encode != format.split(":")[0]:
                tensors, format = device_matrices.reencode(tensors, format, encode, self.device)
                activations = "bf16"
            group = self.encodings.setdefault(role or "draft", {"stored": stored, "encoded": format, "matrices": 0,
                                                                  "stored_bytes": 0, "encoded_bytes": 0})
            if group["stored"] != stored or group["encoded"] != format:
                raise ValueError(f"quantized-projections: draft-only role {role or 'draft'} re-encodes "
                                 f"{group['stored']} to {group['encoded']}, not {stored} to {format}")
            group["matrices"] += 1
            group["stored_bytes"] += before
            group["encoded_bytes"] += device_matrices.stored_bytes(tensors, format)
        prompt = self.roles.get(role, self.prompt)
        decode = self.decode_formats.get(format.split(":")[0], self.decode)
        found = device_matrices.load(tensors, format, self.device, activations, prompt,
                                     self.gemm if self.gemm in ("pinned", "pinned_cublaslt", "tile", "native", "auto") or prompt == "bf16" or
                                     format in ("bf16", "mxfp8") or
                                     (format.startswith("fp8") and self.fp8_activations == "bf16") else "provider",
                                     self.gemv, decode, self.rows, self.tile_rows, self.fp8_activations,
                                     self.mx_activations)
        format = "int4_zp" if found.format == "int4_zp" else format   # awq, gptq: crit's INT4 rows
        rows = found.decode_rows
        self.decode_rows = rows if self.decode_rows is None else min(self.decode_rows, rows)
        self.invariant = found.invariant_rows if self.invariant is None else min(self.invariant, found.invariant_rows)
        if not draft and format != "bf16":
            self.paths = {**self.paths, format.split(":")[0]: sorted({*self.paths.get(format.split(":")[0], ()),
                                                                     found.path})}
        # Which decode kernel each role's matrices run and on which activations (the model plugin's role names, or
        # the format's).
        name = role or format.split(":")[0]
        acts = found.mma_acts["decode"] if getattr(found, "mma", None) is not None else \
            "bf16" if found.decode == "bf16" or found.activations in (None, "bf16") else found.activations
        if found.format == "mxfp8" and found.activations == "mxfp8" and found.decode == "prompt":
            acts = "mxfp8"   # the MXFP8 values, as 16-bit rows
        acts = {"fp8_rows": "fp8", "nvfp4_mma": "nvfp4"}.get(acts, acts)
        entry = self.kernels.get(name, {"kernels": [], "activations": []})
        self.kernels = {**self.kernels, name: {"kernels": sorted({*entry["kernels"], found.path}),
                                               "activations": sorted({*entry["activations"], acts})}}
        if getattr(found, "gemv_rows", 0):   # gemv crossover: the calls of up to these rows on the GEMV
            self.crossovers = {**self.crossovers, name: min(self.crossovers.get(name, found.gemv_rows),
                                                            found.gemv_rows)}
        # The formats of the target's matrices (crit-eval prices prefill by them; draft-only ones are not the target's).
        self.loaded = self.loaded | ({"draft"} if draft else {format.split(":")[0]})
        # Prompt rows keep their bits whatever rows share a call through the fixed-tile GEMM (BF16 matrices and
        # weight-only ones with BF16 prompts), the pinned GEMMs (every format: crit's in one configuration a shape, or
        # cuBLASLt's pinned algorithms) or the prompt tile, not through
        # cuBLAS(Lt)'s choice by the row count; through the prompt tile they have the decode rows' bits too (exact),
        # where every target matrix runs it.
        self.prompt_invariant = getattr(self, "prompt_invariant", True) and found.prompt_invariant
        if not draft:
            self.prompt_exact = getattr(self, "prompt_exact", True) and found.prompt_exact
        # The kernel each format's (each role's) prompt rows go through: native GEMMs are not row invariant.
        entry = self.prompt_kernels.get(name, [])
        self.prompt_kernels = {**self.prompt_kernels, name: sorted({*entry, found.prompt_kernel})}
        return found

    paths = {}
    kernels = {}
    prompt_kernels = {}
    mx_activations = "bf16"
    crossovers = {}

    def arithmetic(self, format):
        """A format's prompt rows' arithmetic class, for a model plugin to declare per group: the weights' values
        (exact: the stored values, or what the prompt GEMM multiplies them as), the activations' (bf16: 16-bit, as
        given; fp8: each row in E4M3 by its own scale; nvfp4: E2M1 blocks of 16 under the input scale) and the sums'
        (fp32, or fp16 for fp16_acc). With prompt_gemm tile, prompt rows multiply the decode rows' activations."""
        if format == "mxfp8":
            return {"weights": "exact", "activations": self.mx_activations, "sums": "fp32"}
        if format in ("bf16", "int4_zp"):
            return {"weights": "exact", "activations": "bf16", "sums": "fp32"}
        if self.exact():
            return {"weights": "exact", "activations": self.decode_arithmetic(format), "sums": "fp32"}
        if format == "fp8":
            return {"weights": "exact", "activations": self.fp8_activations, "sums": "fp32"}
        if self.activations == "nvfp4":
            return {"weights": "exact", "activations": "nvfp4", "sums": "fp32"}
        return {"bf16": {"weights": "exact", "activations": "bf16", "sums": "fp32"},
                "fp16_acc": {"weights": "exact", "activations": "fp16", "sums": "fp16"},
                "fp8": {"weights": "fp8_rows", "activations": "fp8", "sums": "fp32"}}[self.prompt]

    def exact(self):
        """Whether every target matrix loaded so far runs its prompt rows on the prompt tile (prompt_gemm tile, the
        decode rows on the tile, the kernel's prompt tile here): each prompt row's output then has the bits of the
        same row decoded, so a scored token's log-probability is the served one's."""
        return bool(self.loaded - {"draft"}) and getattr(self, "prompt_exact", False)

    def decode_arithmetic(self, format):
        """The activations a format's decode rows multiply: bf16 for weight-only matrices and with decode_activations
        bf16 (decode_formats' for the formats it names), else the prompt rows' quantization (fp8 rows, nvfp4
        blocks)."""
        weight_only = {"mxfp8": self.mx_activations == "bf16", "bf16": True, "int4_zp": True,
                       "nvfp4": self.activations == "bf16", "fp8": self.fp8_activations == "bf16"}
        return "bf16" if weight_only[format] or self.decode_formats.get(format, self.decode) == "bf16" else format

    @property
    def encodings(self):
        if "_encodings" not in self.__dict__:
            self._encodings = {}
        return self._encodings

    def matmul(self, x, matrix, out_dtype=None):
        return device_matrices.matmul(x, matrix, out_dtype)

    def round_problems(self):
        """Why this plugin's matmul cannot run inside a captured round, [] when it can (rounds:projections): every
        decode and prompt path is device work on buffers sized at load, without host synchronization. The Triton
        kernels (the Triton GEMV, the decoders, the fixed-tile GEMM) compile on their first call for a shape: a round
        runs once eagerly before it is captured, and a first call inside a capture raises with that reason."""
        return []

    def facts(self):
        """The engine facts of the matrices loaded so far: each quantized format's prompt activations, as crit-eval
        prices prefill by them (stored: the format's own, W4A4 or W8A8), their rows' invariance (calls of up to
        `rows` rows keep each row's bits, the fewest any loaded matrix keeps: its decode rows (device_matrices.GEMV_ROWS,
        TRITON_ROWS on the Triton GEMV) or, on mma-rows' tile with the same arithmetic for both, its tile rows
        (tile_rows), so a padded prompt bucket of up to `rows` rows gives the unpadded chunk's bits; longer calls keep
        them through prompt_gemm fixed_tiles, pinned or pinned_cublaslt: prompt_rows, a row's bits the same at every
        prompt length,
        chunk and padded bucket), the decode rows'
        activations per quantized format (decode_activations: the activations they multiply, the prompt rows' or bf16,
        decode_formats' for the formats it names, else decode_activations')
        and their path (decode_paths: mma, mma_fp4, dot, gemv, triton, cublaslt one row a call, torch; gemv_mma and
        gemv_mma_fp4 with gemv crossover, decode_crossover naming per role the most rows a call sends through the
        GEMV before the tile;
        decode_kernels: the kernels and the decode activations (bf16, fp8, nvfp4) by the role the model plugin loaded
        each matrix with, or by format), the kernels each role's (format's) prompt rows go through (prompt_kernels:
        fp8-block-rows gemm-native and int-block-experts gemm-native with prompt_gemm native, accumulated in the
        tensor cores, so row_invariance prompt_rows is false; the same "pinned" with prompt_gemm pinned, one
        configuration a matrix shape, row invariant; mma-rows prompt-tile, pinned-gemm, fixed-tile-gemm,
        provider, torch), and
        each draft-only role re-encoded at load (draft_encoding: stored and encoded format, matrices, bytes)."""
        exact = self.exact()
        found = {"row_invariance": {"rows": self.invariant or device_matrices.GEMV_ROWS,
                                    "prompt_rows": bool(self.loaded) and getattr(self, "prompt_invariant", False),
                                    "prompt_exact": exact}}
        if "nvfp4" in self.loaded:
            found["nvfp4"] = ("stored" if self.decode_arithmetic("nvfp4") == "nvfp4" else "bf16") if exact else \
                "stored" if self.activations == "nvfp4" else self.prompt
            if self.activations != "nvfp4" and self.roles and not exact:
                found["nvfp4_by_role"] = dict(self.roles)
        if "fp8" in self.loaded:
            found["fp8"] = ("stored" if self.decode_arithmetic("fp8") == "fp8" else "bf16") if exact else \
                "stored" if self.fp8_activations == "fp8" else "bf16"
        if "mxfp8" in self.loaded:   # its prompt rows multiply 16-bit rows or MXFP8 ones (mxfp8_activations)
            found["mxfp8"] = "stored" if self.mx_activations == "mxfp8" else "bf16"
        if "int4_zp" in self.loaded:
            found["int4_zp"] = "bf16"   # weight only (W4A16)
        prompt = {k: v for k, v in found.items() if k != "row_invariance"}
        draft = {role: dict(group) for role, group in self.encodings.items()}
        decode = {f: self.decode_arithmetic(f) for f in sorted(self.loaded & {"fp8", "mxfp8", "nvfp4", "int4_zp"})}
        arithmetic = {f: self.arithmetic(f) for f in sorted(self.loaded & {"bf16", "fp8", "mxfp8", "nvfp4",
                                                                           "int4_zp"})}
        return {"row_invariance": found["row_invariance"], **({"prompt_activations": prompt} if prompt else {}),
                **({"prompt_arithmetic": arithmetic} if arithmetic else {}),
                **({"decode_activations": decode, "decode_paths": dict(self.paths)} if decode else {}),
                **({"decode_kernels": dict(self.kernels)} if self.kernels else {}),
                **({"decode_crossover": dict(self.crossovers)} if self.crossovers else {}),
                **({"prompt_kernels": dict(self.prompt_kernels)} if self.prompt_kernels else {}),
                **({"draft_encoding": draft} if draft else {})}


class GroupedIntProjections(DeviceMatrices):
    def __init__(self, bits, group):
        self.bits, self.group = bits, group

    def prepare(self, rows):
        return grouped.quantize(rows, self.group, self.bits)

    def project(self, weight, activations):
        return [list(grouped.project(weight, x)) for x in activations]

    def dense(self, weight):
        return [list(row) for row in grouped.dequantize(weight)]


class NVFP4Projections(DeviceMatrices):
    def __init__(self, four_over_six):
        self.four_over_six = four_over_six

    def prepare(self, rows):
        return nvfp4.Matrix.quantize([[float(v) for v in row] for row in rows], self.four_over_six)

    def project(self, weight, activations):
        return nvfp4.project(weight, [list(x) for x in activations])

    def dense(self, weight):
        return [weight.row(n) for n in range(weight.rows)]


def create(context, options, parts):
    if options["format"] == "nvfp4":
        found = NVFP4Projections(options["four_over_six"])
    else:
        found = GroupedIntProjections(int(options["format"][3:]), options["group"])
    found.device = getattr(context, "device", None)
    found.activations, found.prompt, found.gemm = options["activations"], options["prompt"], options["prompt_gemm"]
    if found.gemm == "fixed_tiles" and found.prompt != "bf16":
        raise ValueError("quantized-projections: prompt_gemm fixed_tiles runs bf16 prompts, not " + found.prompt)
    if found.gemm in ("pinned", "pinned_cublaslt", "tile") and found.activations == "bf16" and found.prompt != "bf16":
        raise ValueError(f"quantized-projections: prompt_gemm {found.gemm} runs weight-only NVFP4 prompts in bf16, "
                         f"not {found.prompt}")
    if found.gemm == "tile" and options["gemv"] != "mma":
        raise ValueError("quantized-projections: prompt_gemm tile gives prompt rows the decode tile's bits: gemv mma, "
                         "not " + options["gemv"])
    found.roles = prompt_roles(options["prompt_roles"])
    found.gemv = options["gemv"]
    found.decode = options["decode_activations"]
    found.decode_formats = decode_formats(options["decode_formats"])
    found.rows = options["decode_rows"]
    found.tile_rows = options["tile_rows"] if options["tile_rows"] == "auto" else int(options["tile_rows"])
    found.fp8_activations = options["fp8_activations"]
    found.mx_activations = options["mxfp8_activations"]
    if found.device is not None and str(found.device).startswith("cuda"):
        device_matrices.buffers(found.device)   # the shared buffers memory() declares, made here
    return found


def memory(context, options):
    """The device bytes the factory takes: the buffers every matrix on the device shares (device_matrices.buffers:
    a prompt GEMM's decode workspace, mma-rows' activations, partial sums and counters, the pinned GEMMs' cuBLASLt
    workspace), on a CUDA device. Matrices
    are the model's tensors and go through its plan."""
    device = getattr(context, "device", None)
    return {"device": device_matrices.BUFFER_BYTES} if device is not None and str(device).startswith("cuda") else {}


def decode_formats(text):
    """'format=activations,...' (decode_formats) as {format: activations}: fp8 or nvfp4, each prompt or bf16 (the
    formats whose decode rows can take either: MXFP8, BF16 and weight-only NVFP4 decode rows multiply 16-bit
    activations whatever the option)."""
    found = {}
    for pair in filter(None, (p.strip() for p in text.split(","))):
        format, _, activations = (x.strip() for x in pair.partition("="))
        if format not in ("fp8", "nvfp4") or activations not in ("prompt", "bf16") or format in found:
            raise ValueError("quantized-projections: decode_formats takes format=activations pairs (formats fp8, "
                             f"nvfp4, each once; activations prompt, bf16), not {pair!r}")
        found[format] = activations
    return found


def prompt_roles(text):
    """'role=arithmetic,...' (prompt_roles) as {role: arithmetic}, each arithmetic one of device_matrices.PROMPTS."""
    found = {}
    for pair in filter(None, (p.strip() for p in text.split(","))):
        role, _, arith = (x.strip() for x in pair.partition("="))
        if not role or arith not in device_matrices.PROMPTS:
            raise ValueError(f"quantized-projections: prompt_roles takes role=arithmetic pairs "
                             f"({', '.join(device_matrices.PROMPTS)}), not {pair!r}")
        found[role] = arith
    return found
