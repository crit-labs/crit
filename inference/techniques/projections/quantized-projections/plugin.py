"""Quantized weights in the projections slot: quantized-projections' grouped integer quantize, dequantize and project
(implementation.py) or its NVFP4 codec and projection (nvfp4.py), as the `format` option chooses, as the host form;
on the device, a checkpoint's stored BF16, FP8 or NVFP4 matrices on the engine's device, multiplied through cuBLASLt
(device_matrices.py), NVFP4 ones with NVFP4 or 16-bit activations as `activations` says (device_nvfp4.py: crit's
kernels that plugin.json declares, built on this machine's GPU, or torch fallbacks)."""
import weakref
from pathlib import Path

from .quantized import device_matrices, implementation as grouped, nvfp4

device_matrices.KERNEL_FOLDER = Path(__file__).resolve().parent   # its plugin.json declares crit's kernels


class DeviceMatrices:
    """The device form, whatever the host form's format: load(tensors, format) and matmul(x, matrix). Matrices land
    on the engine's device whatever device the given tensors are on."""

    device, activations, prompt, gemm, gemv, loaded, roles = None, "nvfp4", "bf16", "auto", "auto", frozenset(), {}
    decode, decode_rows, rows, tile_rows, invariant = "prompt", None, device_matrices.GEMV_ROWS, "auto", None
    fp8_activations = "fp8"
    formats = device_matrices.FORMATS   # the stored formats load() takes (model-template reads it: fp8:block)
    decode_formats = {}   # decode_formats: a format's decode activations where they differ from decode_activations
    head, head_by, head_encoding = "stored", None, None   # the `head` option as create() resolved it (asked_head)
    head_kept = None   # why an FP8 head stayed as stored (a shape cuBLASLt does not take), for facts()
    trunk, trunk_by, trunk_encoding = "stored", None, None   # the `trunk` option as create() resolved it (asked_trunk)
    trunk_keep = frozenset()   # the roles `trunk_keep` names: held as stored whatever `trunk` says
    trunk_kept = {}   # role -> why its 16-bit matrices stayed as stored under trunk fp8, for facts()
    stored_copies = ()   # (weak reference, device) of each stored 16-bit target matrix re-encoded here, until
    # release_dropped()
    scratch = None        # the engine's scratch (context.scratch), which the calls on its device compute in

    def close(self):
        if self.scratch is not None:   # the engine closes: its scratch is used no more
            device_matrices.use_scratch(self.device, self.scratch, use=False)
            self.scratch = None

    def encoding_problem(self, encode, rows, columns):
        """Why a draft-only matrix of rows x columns stays as stored where `encode` is asked (a shape the encoded
        format's kernels do not take), or None: load() keeps such a matrix as stored and facts() name it."""
        return device_matrices.encoding_problem(encode, rows, columns)

    def load(self, tensors, format, role=None, encode=None):
        """role: the matrix's role in the model, as the model plugin names it (qkv, gate_up, ...), which picks its
        prompt arithmetic where prompt_roles names it. Role "head" (HEAD_ROLE) is the model's output head, the
        vocabulary-wide matrix that makes the logits: with head fp8 (the `head` option, or its auto in quality mode
        speed: asked_head) a head stored in 16 bits is re-encoded to FP8 rows at load (fp8:channel, one FP32 scale a
        row: device_matrices.reencode) and is from then on an FP8 target matrix like a stored one, which facts()
        records under head_encoding; a head stored quantized stays as stored. Every other target matrix stored in 16
        bits is re-encoded the same way with trunk fp8 (the `trunk` option, or its auto in quality mode speed:
        asked_trunk), but those of a role trunk_keep names (the matrices whose outputs pick: a router's experts, an
        indexer's keys), recorded under trunk_encoding; one whose rows alone the FP8 kernels do not take (not a
        multiple of 16: a stacked product with a sink's 24 rows) is padded with zero rows first (row_padding; its
        products its own rows, a view); another shape the FP8 kernels do not take stays as stored, under trunk_kept
        (head_kept for the head). encode: for a matrix only a drafter
        reads (an MTP layer, a draft head; the model plugin passes it only for draft-only roles), the format to
        re-encode the stored one to at load (device_matrices.ENCODINGS: nvfp4, fp8); an nvfp4 one multiplies 16-bit
        activations (no calibrated activation scale), and facts() records the group under draft_encoding (a matrix
        already stored in the encode format stays as stored and is recorded the same way: draft-only all the
        same)."""
        self.release_dropped()
        activations, draft, encoded_on = self.activations, encode is not None, None
        kept = None   # why a matrix asked to re-encode stays as stored: a shape the encoded format does not take
        target = None if draft or format not in device_matrices.ENCODINGS["fp8"] else self.target_encoding(role)
        out_rows = None   # a trunk matrix padded with zero rows to a shape FP8 takes: its own rows
        # A format stored without a 16-bit weight (int4's qweight) has no target, so no shape to read
        rows, columns = tensors["weight"].shape[:2] if target else (0, 0)
        pad = device_matrices.row_padding(target, rows, columns) if target and role != HEAD_ROLE else 0
        problem = target and device_matrices.encoding_problem(target, rows + pad, columns)
        if problem and role == HEAD_ROLE:
            self.head_kept = problem
        elif problem:
            self.trunk_kept = {**self.trunk_kept, role or format: problem}
        elif target:
            stored, before = format, device_matrices.stored_bytes(tensors, format)
            self.stored_copies = (*self.stored_copies, (weakref.ref(tensors["weight"]), tensors["weight"].device))
            if pad:   # zero rows up to the shape FP8 takes, encoded with the rest (matmul returns its own rows)
                weight, out_rows = tensors["weight"], rows
                tensors = {**tensors, "weight": weight.new_zeros(rows + pad, columns)}
                tensors["weight"][:rows] = weight
            tensors, format = device_matrices.reencode(tensors, format, target, self.device)
            encoded_on = tensors["weight"].device
            head = role == HEAD_ROLE
            option, by = ("head", self.head_by) if head else ("trunk", self.trunk_by)
            group = (self.head_encoding if head else self.trunk_encoding) or {
                "stored": stored, "encoded": format, "matrices": 0, "stored_bytes": 0, "encoded_bytes": 0,
                "chosen": f"the {option} option" if by is None else f"{option} auto in quality mode {by}"}
            group = {**group, "matrices": group["matrices"] + 1, "stored_bytes": group["stored_bytes"] + before,
                     "encoded_bytes": group["encoded_bytes"] + device_matrices.stored_bytes(tensors, format)}
            if head:
                self.head_encoding = group
            else:
                self.trunk_encoding = {**group, "roles": sorted({*group.get("roles", ()), role or stored})}
        if draft:
            stored, before = format, device_matrices.stored_bytes(tensors, format)
            if encode != format.split(":")[0]:
                # A matrix whose shape the encoded format does not take stays as stored (a draft layer's 324 x 10240
                # matrix in FP8): the rest of its role re-encodes, and the facts name it (draft_encoding's kept).
                kept = (device_matrices.encoding_problem(encode, *tensors["weight"].shape[:2])
                        if format in device_matrices.ENCODINGS.get(encode, ()) else None)
                if kept is None:
                    tensors, format = device_matrices.reencode(tensors, format, encode, self.device)
                    activations = "bf16"
            # One group per role and stored format: a role whose matrices are stored in two formats (a draft layer's
            # BF16 and FP8 matrices) records each apart, under role:stored for the second; a matrix kept as stored,
            # under role:kept.
            key = role or "draft"
            if kept is not None:
                key = f"{key}:kept"
            elif key in self.encodings and self.encodings[key]["stored"] != stored:
                key = f"{key}:{stored}"
            group = self.encodings.setdefault(key, {"stored": stored, "encoded": format, "matrices": 0,
                                                    "stored_bytes": 0, "encoded_bytes": 0,
                                                    **({"asked": encode, "kept": kept} if kept else {})})
            if group["encoded"] != format:
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
        found.out_rows = out_rows
        format = "int4_zp" if found.format == "int4_zp" else format   # awq, gptq: crit's INT4 rows
        # Every matrix shape as loaded (stacked ones as the model plugin stacked them), for crit kernels sweep.
        self.shapes = self.shapes | {(found.rows, found.columns, format.split(":")[0])}
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
        if encoded_on is not None:
            # What the re-encoded matrix's load left in torch's cache (its warm-up) goes back to the device before a
            # next matrix is cut from those blocks.
            device_matrices.release(encoded_on)
        return found

    def target_encoding(self, role):
        """The format a target matrix of `role` stored in 16 bits is re-encoded to at load (fp8), or None to hold it
        as stored: the `head` option for the output head (HEAD_ROLE), the `trunk` option for every other role but
        those trunk_keep names."""
        if role == HEAD_ROLE:
            return "fp8" if self.head == "fp8" else None
        return "fp8" if self.trunk == "fp8" and role not in self.trunk_keep else None

    def release_dropped(self):
        """The stored 16-bit copy of a re-encoded target matrix (the head, a trunk matrix), once the model plugin
        dropped it: its device block goes back to the device instead of staying in torch's cache, where the next
        matrices would be cut from it and keep it reserved for the engine's lifetime; what sizes itself by the
        device's free memory (an expert cache) then gets the bytes the FP8 matrices gave up. A tensor's block is
        free only once the tensor is gone, so nothing can hand it back at the drop itself: load() and facts() call
        this, the next calls a model plugin makes. A copy something still holds (embeddings that share the head's
        tensor) is waited for."""
        held = tuple((stored, device) for stored, device in self.stored_copies if stored() is not None)
        for device in {device for stored, device in self.stored_copies if stored() is None}:
            device_matrices.release(device)
        self.stored_copies = held

    shapes = frozenset()
    paths = {}
    kernels = {}
    prompt_kernels = {}
    mx_activations, mx_by = "bf16", None   # the `mxfp8_activations` option as create() resolved it (asked_mx)
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

    def matmul(self, x, matrix, out_dtype=None, prompt=False):
        return device_matrices.matmul(x, matrix, out_dtype, prompt=prompt)

    def exactness(self):
        """The level the loaded matrices' rows keep (crit_plugins.exactness; Core's facts carry it and its loader
        compares it with the level the configuration asks). Decode rows are row-exact: in a call of up to the
        facts' row_invariance rows each row has the bits of the row alone. Prompt rows are exact where every target
        matrix runs the prompt tile (the bits of the same row decoded), row-exact through the fixed-tile and pinned
        GEMMs (the same bits at every prompt length), and repeatable through the native GEMMs and the provider,
        whose sums follow the tile or the row count. Nothing before a matrix is loaded."""
        if not self.loaded:
            return {}
        prompt = "exact" if self.exact() else "row-exact" if getattr(self, "prompt_invariant", False) else "repeatable"
        return {"decode": "row-exact", "prompt": prompt}

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
        each draft-only role re-encoded at load (draft_encoding: stored and encoded format, matrices, bytes), and the
        output head where it was re-encoded to FP8 at load (head_encoding: the same, and what chose it), the other
        16-bit target matrices re-encoded to FP8 (trunk_encoding: the same, and their roles) and the roles whose
        matrices trunk fp8 left as stored for their shape (trunk_kept)."""
        self.release_dropped()
        exact = self.exact()
        chosen = {}   # what an option left to choose became, and by what
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
            if self.mx_by is not None:
                chosen["mxfp8_activations"] = f"mxfp8_activations auto by {self.mx_by}"
        if "int4_zp" in self.loaded:
            found["int4_zp"] = "bf16"   # weight only (W4A16)
        prompt = {k: v for k, v in found.items() if k != "row_invariance"}
        draft = {role: dict(group) for role, group in self.encodings.items()}
        decode = {f: self.decode_arithmetic(f) for f in sorted(self.loaded & {"fp8", "mxfp8", "nvfp4", "int4_zp"})}
        arithmetic = {f: self.arithmetic(f) for f in sorted(self.loaded & {"bf16", "fp8", "mxfp8", "nvfp4",
                                                                           "int4_zp"})}
        return {"row_invariance": found["row_invariance"], **({"prompt_activations": prompt} if prompt else {}),
                **({"chosen": chosen} if chosen else {}),
                **({"prompt_arithmetic": arithmetic} if arithmetic else {}),
                **({"decode_activations": decode, "decode_paths": dict(self.paths)} if decode else {}),
                **({"decode_kernels": dict(self.kernels)} if self.kernels else {}),
                **({"decode_crossover": dict(self.crossovers)} if self.crossovers else {}),
                **({"prompt_kernels": dict(self.prompt_kernels)} if self.prompt_kernels else {}),
                **({"draft_encoding": draft} if draft else {}),
                **({"head_encoding": dict(self.head_encoding)} if self.head_encoding else {}),
                **({"head_kept": self.head_kept} if self.head_kept else {}),
                **({"trunk_encoding": dict(self.trunk_encoding)} if self.trunk_encoding else {}),
                **({"trunk_kept": dict(self.trunk_kept)} if self.trunk_kept else {}),
                **({"matrices": [{"N": n, "K": k, "format": f} for n, k, f in sorted(self.shapes)]} if self.shapes else {})}


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


HEAD_ROLE = "head"   # the role a model plugin loads its output head with
HEADS = ("auto", "stored", "fp8")


def asked_head(head, mode):
    """The `head` option under the quality mode the engine states (crit_plugins.kernels.quality_mode): auto is fp8
    in speed, whose budget starts from an FP8 head, and stored in preserve or where no mode is stated (an engine
    built outside a run, or before the option). A head set to one way is that way. Returns (head, the mode that
    chose it or None)."""
    if head not in HEADS:
        raise ValueError(f"quantized-projections: head is one of {', '.join(HEADS)}, not {head!r}")
    if head != "auto":
        return head, None
    return ("fp8", mode) if mode == "speed" else ("stored", None)


TRUNKS = ("auto", "stored", "fp8")


def asked_trunk(trunk, mode):
    """The `trunk` option under the quality mode the engine states, as asked_head reads `head`: auto is fp8 in
    speed, which starts from 8-bit target weights wherever they are stored in 16 bits (every decode step reads each
    of them, and a W8A8 prompt row multiplies at twice the BF16 rate), and stored in preserve or where no mode is
    stated. Returns (trunk, the mode that chose it or None)."""
    if trunk not in TRUNKS:
        raise ValueError(f"quantized-projections: trunk is one of {', '.join(TRUNKS)}, not {trunk!r}")
    if trunk != "auto":
        return trunk, None
    return ("fp8", mode) if mode == "speed" else ("stored", None)


def role_names(text):
    """'role,role,...' (trunk_keep) as a frozenset of role names."""
    return frozenset(filter(None, (r.strip() for r in text.split(","))))


MX_ACTIVATIONS = ("auto", "bf16", "mxfp8")


# The MXFP8 prompt activation each asked activation (crit_plugins.kernels.activation) is served by.
MX_ASKED = {"stored": "mxfp8", "bf16": "bf16"}


def asked_mx(activations, mode, asked=None):
    """The `mxfp8_activations` option under what the engine is asked: auto serves the prompt activation the
    configuration asks for MXFP8 matrices (`asked`, crit_plugins.kernels.activation("mxfp8"): what the build plan
    declared; stored is mxfp8, W8A8), and where nothing is asked follows the quality mode: mxfp8 in speed, where a
    quantized format's prompt rows start at its stored activations, and bf16 in preserve or where no mode is stated.
    An option set to one way is that way (Core's loader refuses it where it is not what is asked). Returns
    (activations, what chose them or None)."""
    if activations not in MX_ACTIVATIONS:
        raise ValueError(f"quantized-projections: mxfp8_activations is one of {', '.join(MX_ACTIVATIONS)}, not "
                         f"{activations!r}")
    if activations != "auto":
        return activations, None
    if asked in MX_ASKED:
        return MX_ASKED[asked], f"the configuration's prompt_activations ({asked})"
    return ("mxfp8", f"quality mode {mode}") if mode == "speed" else ("bf16", None)


def asked_gemm(gemm, gemv, asked):
    """prompt_gemm auto and gemv auto under the level the engine asks of its prompt rows (crit_plugins.kernels.asks):
    the fastest prompt GEMM that keeps it. exact (and declared, which no GEMM here keeps: the loader then says so):
    the prompt tile, each prompt row the bits of the row decoded, with the decode rows on the tile (gemv auto as fp4:
    the FP4 MMA where it is, the tile elsewhere); row-exact: the pinned GEMMs, one configuration a matrix shape;
    repeatable, or nothing asked: auto as it was, crit's native GEMMs. A prompt_gemm set to one way is that way.
    Returns (prompt_gemm, gemv, the level that chose them or None)."""
    if gemm != "auto" or asked in (None, "repeatable"):
        return gemm, gemv, None
    if asked == "row-exact":
        return "pinned", gemv, asked
    return "tile", "fp4" if gemv == "auto" else gemv, asked


def create(context, options, parts):
    from crit_plugins.kernels import activation, asks, quality_mode
    if options["format"] == "nvfp4":
        found = NVFP4Projections(options["four_over_six"])
    else:
        found = GroupedIntProjections(int(options["format"][3:]), options["group"])
    found.device = getattr(context, "device", None)
    found.activations, found.prompt = options["activations"], options["prompt"]
    found.gemm, gemv, asked = asked_gemm(options["prompt_gemm"], options["gemv"], asks("prompt"))
    how = f"prompt_gemm {found.gemm}" + (f" (auto, for the {asked} prompt rows the configuration's exactness asks)"
                                         if asked else "")
    if found.gemm == "fixed_tiles" and found.prompt != "bf16":
        raise ValueError("quantized-projections: prompt_gemm fixed_tiles runs bf16 prompts, not " + found.prompt)
    if found.gemm in ("pinned", "pinned_cublaslt", "tile") and found.activations == "bf16" and found.prompt != "bf16":
        raise ValueError(f"quantized-projections: {how} runs weight-only NVFP4 prompts in bf16, "
                         f"not {found.prompt}")
    if found.gemm == "tile" and gemv not in ("mma", "fp4"):
        raise ValueError(f"quantized-projections: {how} gives prompt rows the decode tile's bits: gemv mma "
                         "or fp4, not " + gemv)
    found.roles = prompt_roles(options["prompt_roles"])
    found.head, found.head_by = asked_head(options["head"], quality_mode())
    found.trunk, found.trunk_by = asked_trunk(options["trunk"], quality_mode())
    found.trunk_keep = role_names(options["trunk_keep"])
    found.gemv = gemv
    found.decode = options["decode_activations"]
    found.decode_formats = decode_formats(options["decode_formats"])
    found.rows = options["decode_rows"]
    found.tile_rows = options["tile_rows"] if options["tile_rows"] == "auto" else int(options["tile_rows"])
    found.fp8_activations = options["fp8_activations"]
    found.mx_activations, found.mx_by = asked_mx(options["mxfp8_activations"], quality_mode(), activation("mxfp8"))
    if found.device is not None and str(found.device).startswith("cuda"):
        device_matrices.buffers(found.device)   # the shared buffers memory() declares: each made at its first use
        found.scratch = getattr(context, "scratch", None)
        if found.scratch is not None:   # the engine's scratch: a call's activations and partial sums are taken there
            device_matrices.use_scratch(found.device, found.scratch)
    return found


def memory(context, options):
    """The device bytes the plugin's shared buffers take at most (device_matrices.buffers: a prompt GEMM's decode
    workspace, mma-rows' activations, partial sums and counters, the pinned GEMMs' cuBLASLt workspace; each made
    when a matrix first uses it), on a CUDA device. Matrices are the model's tensors and go through its plan."""
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
