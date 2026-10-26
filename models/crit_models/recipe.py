"""Transform recipes and the plan they resolve to against a checkpoint's headers (standard library only)."""
import re

from crit.run.records import digest, number

METHODS = ("ptq", "drafter")
FORMATS = ("fp8-channel", "nvfp4")
ALGORITHMS = ("rtn", "gptq")
DEFAULT_INCLUDE = (r"^layers\.\d+\..+_proj$",)
DEFAULT_CALIBRATION = {"samples": 128, "sequence_tokens": 2048}
DEFAULT_GPTQ = {"dampening": 0.01, "block": 128}
DRAFTER_KINDS = ("mtp-head",)
DRAFTER_DTYPES = ("bf16", "fp32")
DEFAULT_DRAFTER_CALIBRATION = {"samples": 64, "sequence_tokens": 1024}
DEFAULT_TRAIN = {"steps": 1000, "windows_per_step": 4, "learning_rate": 1e-4, "warmup_steps": 50, "weight_decay": 0.0}
LAYER_ANCHOR = "layers.0.input_layernorm.weight"
# A decoder layer's tensor: the prefix before `layers.<index>.` at a name boundary.
LAYER_TENSOR = re.compile(r"(?:^|(?<=\.))layers\.(\d+)\.")
SOURCE_DTYPES = {"BF16", "F16", "F32"}
# Tensors a quantized module may carry; a target's stored ones are replaced as a whole.
QUANTIZED_SUFFIXES = ("weight", "weight_scale", "weight_packed", "weight_global_scale", "input_scale",
                      "input_global_scale", "weight_zero_point", "weight_shape")


def _fields(value, name, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f"{name} expects fields " + ", ".join(sorted(required)) +
                         ("; optional: " + ", ".join(sorted(optional)) if optional else ""))


def _patterns(value, name):
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError(f"{name} requires a list of at most 64 regular expressions")
    for item in value:
        if not isinstance(item, str) or not item or len(item) > 512:
            raise ValueError(f"{name} requires nonempty patterns")
        try:
            re.compile(item)
        except re.error as error:
            raise ValueError(f"{name}: invalid pattern {item!r}: {error}") from None
    return list(value)


def load_recipe(value):
    """Validate a models_recipe and return it with every default filled in (the recipe a record freezes)."""
    if not isinstance(value, dict) or value.get("record_kind") != "models_recipe" or value.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1, record_kind models_recipe")
    if value.get("method") not in METHODS:
        raise ValueError("method must be one of " + ", ".join(METHODS) + " (pruning and QAD are not available yet)")
    if value["method"] == "drafter":
        return _drafter_recipe(value)
    _fields(value, "models_recipe", {"schema_version", "record_kind", "method", "format", "algorithm"},
            {"include", "exclude", "seed", "calibration", "gptq"})
    if value["format"] not in FORMATS:
        raise ValueError("format must be one of " + ", ".join(FORMATS))
    if value["algorithm"] not in ALGORITHMS:
        raise ValueError("algorithm must be one of " + ", ".join(ALGORITHMS))
    gptq = value["algorithm"] == "gptq"
    if not gptq and ("calibration" in value or "gptq" in value):
        raise ValueError("rtn uses no calibration data; calibration and gptq settings need algorithm gptq")
    recipe = {"schema_version": 1, "record_kind": "models_recipe", "method": "ptq", "format": value["format"],
              "algorithm": value["algorithm"],
              "include": _patterns(value.get("include", list(DEFAULT_INCLUDE)), "include"),
              "exclude": _patterns(value.get("exclude", []), "exclude"),
              "seed": number(value.get("seed", 0), "seed", 0, 2 ** 32 - 1, integer=True)}
    if not recipe["include"]:
        raise ValueError("include needs at least one pattern")
    if gptq:
        calibration = {**DEFAULT_CALIBRATION, **_checked(value.get("calibration", {}), "calibration", DEFAULT_CALIBRATION)}
        number(calibration["samples"], "calibration.samples", 1, 65536, integer=True)
        number(calibration["sequence_tokens"], "calibration.sequence_tokens", 16, 1 << 20, integer=True)
        settings = {**DEFAULT_GPTQ, **_checked(value.get("gptq", {}), "gptq", DEFAULT_GPTQ)}
        number(settings["dampening"], "gptq.dampening", 0, 1)
        number(settings["block"], "gptq.block", 16, 65536, integer=True)
        if settings["block"] % 16:
            raise ValueError("gptq.block must be a multiple of 16")
        recipe.update(calibration=calibration, gptq=settings)
    return recipe


def _drafter_recipe(value):
    """A drafter recipe: the kind of head, the target layer it starts from, the calibration windows, the share held
    out, the training settings and the stored dtype, every default filled in."""
    _fields(value, "models_recipe", {"schema_version", "record_kind", "method"},
            {"kind", "init_layer", "seed", "calibration", "heldout_fraction", "train", "dtype"})
    kind = value.get("kind", DRAFTER_KINDS[0])
    if kind not in DRAFTER_KINDS:
        raise ValueError("kind must be one of " + ", ".join(DRAFTER_KINDS))
    dtype = value.get("dtype", "bf16")
    if dtype not in DRAFTER_DTYPES:
        raise ValueError("dtype must be one of " + ", ".join(DRAFTER_DTYPES))
    init_layer = value.get("init_layer", -1)
    if type(init_layer) is not int or not -1024 <= init_layer <= 1024:
        raise ValueError("init_layer is the index of the target layer the drafter starts from (-1: the last)")
    calibration = {**DEFAULT_DRAFTER_CALIBRATION, **_checked(value.get("calibration", {}), "calibration", DEFAULT_DRAFTER_CALIBRATION)}
    number(calibration["samples"], "calibration.samples", 2, 65536, integer=True)
    number(calibration["sequence_tokens"], "calibration.sequence_tokens", 16, 1 << 20, integer=True)
    heldout = value.get("heldout_fraction", 0.1)
    if type(heldout) not in (int, float) or not 0 < heldout < 0.5:
        raise ValueError("heldout_fraction is the share of windows held out to measure the drafter, above 0 and below 0.5")
    train = {**DEFAULT_TRAIN, **_checked(value.get("train", {}), "train", DEFAULT_TRAIN)}
    number(train["steps"], "train.steps", 1, 1_000_000, integer=True)
    number(train["windows_per_step"], "train.windows_per_step", 1, 1024, integer=True)
    number(train["learning_rate"], "train.learning_rate", 1e-9, 1.0)
    number(train["warmup_steps"], "train.warmup_steps", 0, 1_000_000, integer=True)
    number(train["weight_decay"], "train.weight_decay", 0, 1.0)
    return {"schema_version": 1, "record_kind": "models_recipe", "method": "drafter", "kind": kind, "init_layer": init_layer,
            "seed": number(value.get("seed", 0), "seed", 0, 2 ** 32 - 1, integer=True), "calibration": calibration,
            "heldout_fraction": heldout, "train": train, "dtype": dtype}


def _checked(value, name, defaults):
    _fields(value, name, (), defaults)
    return value


def layer_prefix(names):
    """The decoder-layer prefix (for example `model.` before `layers.`) the reference also uses: the one of its layer
    0's input norm, else, for layers without one (such as hyper-connection layers, which norm their streams), of the
    prefixes under which a layer 0 is stored (draft and vision layers have their own), the one holding the most layer
    tensors."""
    names = list(names)
    for name in names:
        if name.endswith(LAYER_ANCHOR):
            return name[: -len(LAYER_ANCHOR)]
    counts, zero = {}, set()
    for name in names:
        found = LAYER_TENSOR.search(name)
        if found:
            prefix = name[: found.start()]
            counts[prefix] = counts.get(prefix, 0) + 1
            if found.group(1) == "0":
                zero.add(prefix)
    if not zero:
        raise ValueError(f"cannot find the decoder layers: no tensor ends with {LAYER_ANCHOR} or holds layers.0.")
    return max(sorted(zero), key=lambda prefix: counts[prefix])


def plan(recipe, checkpoint):
    """Targets of a recipe in a crit_core.model.Checkpoint: module names relative to the decoder-layer prefix,
    matched by include and not exclude, with their stored and transformed bytes. Raises when a match cannot be
    transformed from its stored form."""
    tensors = checkpoint.tensors
    prefix = layer_prefix(tensors)
    include = [re.compile(p) for p in recipe["include"]]
    exclude = [re.compile(p) for p in recipe["exclude"]]
    modules = sorted({name[len(prefix):].rsplit(".", 1)[0] for name in tensors
                      if name.startswith(prefix) and name.rsplit(".", 1)[-1] in QUANTIZED_SUFFIXES})
    targets = []
    for module in modules:
        if not any(p.search(module) for p in include) or any(p.search(module) for p in exclude):
            continue
        base = prefix + module
        weight = tensors.get(base + ".weight")
        stored = [s for s in QUANTIZED_SUFFIXES if base + "." + s in tensors]
        if weight is None or stored != ["weight"]:
            raise ValueError(f"{module}: PTQ starts from unquantized weights; it stores {', '.join(stored)}")
        if weight.dtype not in SOURCE_DTYPES or len(weight.shape) != 2:
            raise ValueError(f"{module}: needs a 2-D BF16, F16 or F32 weight, found {weight.dtype} {list(weight.shape)}")
        rows, cols = weight.shape
        if recipe["format"] == "nvfp4" and cols % 16:
            raise ValueError(f"{module}: nvfp4 needs input width divisible by 16, found {cols}")
        targets.append({"module": module, "name": base, "shape": [rows, cols], "dtype": weight.dtype,
                        "stored_bytes": weight.nbytes, "transformed_bytes": transformed_bytes(recipe["format"], rows, cols)})
    if not targets:
        raise ValueError("the recipe matches no module")
    return {"prefix": prefix, "targets": targets,
            "stored_bytes": sum(t["stored_bytes"] for t in targets),
            "transformed_bytes": sum(t["transformed_bytes"] for t in targets),
            "checkpoint_bytes": checkpoint.tensor_bytes,
            "modules_sha256": digest([t["name"] for t in targets])}


def transformed_bytes(format, rows, cols):
    if format == "fp8-channel":
        return rows * cols + rows * 4                       # E4M3 codes and one FP32 scale per row
    return rows * cols // 2 + rows * cols // 16 + 4         # E2M1 pairs, E4M3 per 16 values, FP32 global
