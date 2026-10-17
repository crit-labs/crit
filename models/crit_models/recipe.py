"""Transform recipes and the plan they resolve to against a checkpoint's headers (standard library only)."""
import re

from crit.run.records import digest, number

FORMATS = ("fp8-channel", "nvfp4")
ALGORITHMS = ("rtn", "gptq")
DEFAULT_INCLUDE = (r"^layers\.\d+\..+_proj$",)
DEFAULT_CALIBRATION = {"samples": 128, "sequence_tokens": 2048}
DEFAULT_GPTQ = {"dampening": 0.01, "block": 128}
LAYER_ANCHOR = "layers.0.input_layernorm.weight"
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
    _fields(value, "models_recipe", {"schema_version", "record_kind", "method", "format", "algorithm"},
            {"include", "exclude", "seed", "calibration", "gptq"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["record_kind"] != "models_recipe":
        raise ValueError("Expected schema_version 1, record_kind models_recipe")
    if value["method"] != "ptq":
        raise ValueError("method must be ptq (pruning and QAD are not available yet)")
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


def _checked(value, name, defaults):
    _fields(value, name, (), defaults)
    return value


def layer_prefix(names):
    """The decoder-layer prefix (for example `model.layers.` without `layers.`) the reference also uses."""
    for name in names:
        if name.endswith(LAYER_ANCHOR):
            return name[: -len(LAYER_ANCHOR)]
    raise ValueError(f"cannot find the decoder layers: no tensor ends with {LAYER_ANCHOR}")


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
