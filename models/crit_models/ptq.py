"""The PTQ workflow: source checkpoint, calibration, quantization, export and the transform record."""
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

from crit.run.records import digest, file_sha, inventory, now, write_json

from .recipe import plan as make_plan

INDEX = "model.safetensors.index.json"
STAGING = ".crit-staging"


def _directory(path, name):
    path = Path(path).absolute()
    if not path.is_dir():
        raise ValueError(f"{name} {path} is not a directory")
    return path


def _inside(path, root):
    try:
        Path(path).absolute().relative_to(Path(root).absolute())
        return True
    except ValueError:
        return False


def _config_quantization(config):
    sections = [config] + ([config["text_config"]] if isinstance(config.get("text_config"), dict) else [])
    return any("quantization_config" in s for s in sections)


def load_checkpoint(source):
    from crit_core.model import inspect_model
    source = _directory(source, "source")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if _config_quantization(config):
        raise ValueError(f"{source}: config.json declares quantization_config; PTQ starts from an unquantized checkpoint")
    return source, config, inspect_model(source)


def quantization_config(format, targets):
    """compressed-tensors description of the exported targets (weights only; activations stay with the engine)."""
    if format == "fp8-channel":
        weights = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel", "group_size": None,
                   "dynamic": False}
        layout = "float-quantized"
    else:
        weights = {"num_bits": 4, "type": "float", "symmetric": True, "strategy": "tensor_group", "group_size": 16,
                   "dynamic": False}
        layout = "nvfp4-pack-quantized"
    group = {"targets": [t["name"] for t in targets], "weights": weights, "input_activations": None,
             "output_activations": None, "format": layout}
    return {"quant_method": "compressed-tensors", "format": layout, "quantization_status": "compressed",
            "config_groups": {"group_0": group}, "ignore": []}


def peak_host_bytes():
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters(cb=ctypes.sizeof(Counters))
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
            return None
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (OSError, AttributeError, ImportError):
        return None


def _layer_targets(plan, layers):
    grouped = defaultdict(list)
    for target in plan["targets"]:
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", target["module"])
        if not match or int(match.group(1)) >= layers:
            raise ValueError(f"{target['module']}: gptq calibrates decoder layers only; exclude it or use rtn")
        grouped[int(match.group(1))].append((match.group(2), target))
    return grouped


def calibrate(source, plan, recipe, windows, device, staging, log):
    """Sequential GPTQ: stream decoder layers through the official transformers modeling code (as Eval's reference
    does), collect each target's input Hessian over the calibration windows, quantize it, and feed the next layer
    the outputs of the quantized layer. Writes each layer's export tensors to `staging`."""
    import torch
    from safetensors.torch import save_file
    from crit_eval.reference import StreamingReference
    from .codecs import quantize
    reference = StreamingReference(source, device)
    if reference.prefix != plan["prefix"]:
        raise ValueError(f"decoder prefix {reference.prefix!r} differs from the plan's {plan['prefix']!r}")
    config = reference.config
    by_layer = _layer_targets(plan, config.num_hidden_layers)
    embed = reference.checkpoint.raw(plan["prefix"] + "embed_tokens.weight")
    states = [embed[torch.tensor(ids)].float()[None] for ids in windows]
    del embed
    length = len(windows[0])
    positions = torch.arange(length, device=device)[None]
    # The rotary module keeps its computed frequency buffers on the reference's skeleton.
    embeddings = reference.skeleton.rotary_emb.to(device)(states[0].to(device), positions)
    mask = torch.full((length, length), float("-inf"), device=device).triu(1)[None, None]
    layer_types = getattr(config, "layer_types", None)
    errors, last = {}, max(by_layer)

    def forward(layer, index):
        full = layer_types is None or layer_types[index] != "linear_attention"
        for s, state in enumerate(states):
            out = layer(state.to(device), position_embeddings=embeddings, attention_mask=mask if full else None,
                        position_ids=positions)
            states[s] = (out[0] if isinstance(out, tuple) else out).to("cpu")

    with torch.no_grad():
        for index in range(last + 1):
            started = time.time()
            layer = reference.layer(index, index + 1 if index < last else None)
            targets = by_layer.get(index, [])
            if targets:
                hessians, counts, hooks = {}, defaultdict(int), []
                for name, target in targets:
                    module = layer.get_submodule(name)
                    if not isinstance(module, torch.nn.Linear) or list(module.weight.shape) != target["shape"]:
                        raise ValueError(f"{target['module']}: not a linear layer of shape {target['shape']}")
                    hessians[name] = torch.zeros(target["shape"][1], target["shape"][1], device=device)

                    def hook(module, args, name=name):
                        x = args[0].reshape(-1, args[0].shape[-1]).float()
                        hessians[name] += x.T @ x
                        counts[name] += x.shape[0]
                    hooks.append(module.register_forward_pre_hook(hook))
                full = layer_types is None or layer_types[index] != "linear_attention"
                for state in states:
                    layer(state.to(device), position_embeddings=embeddings, attention_mask=mask if full else None,
                          position_ids=positions)
                for handle in hooks:
                    handle.remove()
                exported = {}
                for name, target in targets:
                    module = layer.get_submodule(name)
                    hessian = hessians.pop(name) * (2.0 / counts[name])
                    tensors, values, error = quantize(recipe["format"], module.weight.data, hessian, recipe["gptq"])
                    module.weight.data = values.to(module.weight.dtype)
                    exported.update({f"{target['name']}.{k}": v.cpu() for k, v in tensors.items()})
                    errors[target["module"]] = error
                    del hessian
                save_file(exported, str(staging / f"layer-{index}.safetensors"))
            if index < last:
                forward(layer, index)
            reference.skeleton.layers[index] = torch.nn.Identity()
            del layer
            if device != "cpu":
                torch.cuda.empty_cache()
            log(f"layer {index + 1}/{last + 1} {time.time() - started:.1f}s"
                + (f", {len(targets)} modules quantized" if targets else ""))
    return errors


def export(source, checkpoint, config, out, plan, recipe, provider, log):
    """Write every source shard under its own name: targets as their export tensors, everything else as stored."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    targets = {t["name"]: t for t in plan["targets"]}
    weight_map, total = {}, 0
    for shard in checkpoint.shards:
        tensors = {}
        with safe_open(str(shard.path), "pt", device="cpu") as handle:
            for info in shard.tensors:
                base, _, leaf = info.name.rpartition(".")
                if base in targets and leaf == "weight":
                    for suffix, tensor in provider(base, handle.get_tensor(info.name)).items():
                        tensors[f"{base}.{suffix}"] = tensor.cpu()
                else:
                    tensors[info.name] = handle.get_tensor(info.name)
        save_file(tensors, str(out / shard.path.name), metadata=dict(shard.metadata) or {"format": "pt"})
        weight_map.update({name: shard.path.name for name in tensors})
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        log(f"wrote {shard.path.name}")
    if checkpoint.source.name == INDEX:
        write_json(out / INDEX, {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))})
    copied, skipped = [], []
    for path in sorted(source.iterdir()):
        if path.is_dir():
            skipped.append(path.name)
        elif path.suffix == ".safetensors" or path.name == INDEX:
            continue
        elif path.name == "config.json":
            written = {**config, "quantization_config": quantization_config(recipe["format"], plan["targets"])}
            (out / path.name).write_text(json.dumps(written, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(path, out / path.name)
            copied.append(path.name)
    return copied, skipped


def summarize(errors):
    summary = {}
    for key in ("weight_error", "output_error"):
        values = {module: e[key] for module, e in errors.items() if key in e}
        if values:
            worst = max(values, key=values.get)
            summary[key] = {"mean": sum(values.values()) / len(values), "max": values[worst], "max_module": worst}
    return summary


def run_ptq(recipe, source, out, record_path, calibration=None, device=None, log=print):
    """Transform `source` into `out` by `recipe` and write the models_transform record to `record_path`."""
    import torch
    from crit import __version__
    from crit_eval.prepare import model_identity
    from .calibration import load_calibration
    started_at, clock = now(), time.perf_counter()
    phases = {}
    out, record_path = Path(out).absolute(), Path(record_path).absolute()
    source, config, checkpoint = load_checkpoint(source)
    if _inside(out, source) or _inside(source, out):
        raise ValueError("the output and source directories must not contain each other")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"{out} must be a new or empty directory")
    if record_path.exists() or _inside(record_path, out) or _inside(record_path, source):
        raise ValueError("the record must be a new file outside the source and output directories")
    if (calibration is None) != (recipe["algorithm"] == "rtn"):
        raise ValueError("gptq needs --calibration; rtn takes none")
    plan = make_plan(recipe, checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(recipe["seed"])
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    log(f"{len(plan['targets'])} modules, {plan['stored_bytes'] / 2**30:.2f} GiB stored -> "
        f"{plan['transformed_bytes'] / 2**30:.2f} GiB {recipe['format']} ({recipe['algorithm']} on {device})")
    mark = time.perf_counter()
    source_files = inventory(source)
    phases["hash_source"] = time.perf_counter() - mark
    out.mkdir(parents=True, exist_ok=True)
    staging = out / STAGING
    errors, calibration_facts = {}, None
    try:
        if recipe["algorithm"] == "gptq":
            mark = time.perf_counter()
            windows, calibration_facts = load_calibration(calibration, source, recipe)
            staging.mkdir()
            errors = calibrate(source, plan, recipe, windows, device, staging, log)
            phases["calibrate_and_quantize"] = time.perf_counter() - mark
            from safetensors import safe_open
            staged = {}
            for path in sorted(staging.glob("*.safetensors")):
                with safe_open(str(path), "pt", device="cpu") as handle:
                    for key in handle.keys():
                        base, _, suffix = key.rpartition(".")
                        staged.setdefault(base, (path, []))[1].append(suffix)

            def provider(base, weight):
                path, suffixes = staged[base]
                with safe_open(str(path), "pt", device="cpu") as handle:
                    return {s: handle.get_tensor(f"{base}.{s}") for s in suffixes}
        else:
            from .codecs import quantize

            def provider(base, weight):
                tensors, _, error = quantize(recipe["format"], weight.to(device))
                errors[base[len(plan["prefix"]):]] = error
                return tensors
        mark = time.perf_counter()
        copied, skipped = export(source, checkpoint, config, out, plan, recipe, provider, log)
        phases["export" if recipe["algorithm"] == "gptq" else "quantize_and_export"] = time.perf_counter() - mark
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if staging.exists():
        raise ValueError(f"could not remove {staging}")
    from crit_core.model import inspect_model
    inspect_model(out)  # the exported headers and index pass Core's checks
    mark = time.perf_counter()
    output_files = inventory(out)
    phases["hash_output"] = time.perf_counter() - mark
    unchanged = {r["path"]: r["sha256"] for r in source_files}
    for name in copied:
        if unchanged.get(name) != next(r["sha256"] for r in output_files if r["path"] == name):
            raise ValueError(f"{name} changed while it was copied")
    record = {
        "schema_version": 1, "record_kind": "models_transform", "crit_version": __version__,
        "recipe": recipe, "recipe_sha256": digest(recipe), "seed": recipe["seed"],
        "source": {"path": str(source), "files": source_files, "manifest_sha256": digest(source_files),
                   "eval_model_sha256": model_identity(source)["sha256"]},
        "calibration": calibration_facts,
        "plan": {k: plan[k] for k in ("prefix", "modules_sha256", "stored_bytes", "transformed_bytes",
                                      "checkpoint_bytes")} | {"modules": len(plan["targets"])},
        "output": {"path": str(out), "files": output_files, "manifest_sha256": digest(output_files),
                   "eval_model_sha256": model_identity(out)["sha256"], "copied_files": copied,
                   "skipped_directories": skipped},
        "errors": summarize(errors),
        "cost": {"wall_seconds": time.perf_counter() - clock, "phases": phases, "device": device,
                 "device_name": torch.cuda.get_device_name(device) if device != "cpu" else platform.processor() or None,
                 "peak_device_bytes": torch.cuda.max_memory_allocated(device) if device != "cpu" else None,
                 "peak_host_bytes": peak_host_bytes()},
        "environment": _environment(),
        "started_at": started_at, "ended_at": now(),
    }
    write_json(record_path, record, immutable=True)
    log(f"wrote {out} and {record_path} ({record['cost']['wall_seconds']:.1f}s)")
    return record


def _environment():
    import torch
    environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                   "cuda": torch.version.cuda}
    for name in ("safetensors", "transformers"):
        try:
            environment[name] = __import__(name).__version__
        except ImportError:
            environment[name] = None
    return environment


def verify(record):
    """Re-hash the output (and the source when it is still present); returns a list of findings, empty when exact."""
    findings = []
    for side in ("output", "source"):
        root = Path(record[side]["path"])
        if not root.is_dir():
            findings.append(f"{side} {root} is missing")
            continue
        rows = inventory(root)
        if rows != record[side]["files"]:
            recorded = {r["path"]: r for r in record[side]["files"]}
            actual = {r["path"]: r for r in rows}
            changed = sorted(p for p in recorded.keys() | actual.keys() if recorded.get(p) != actual.get(p))
            findings.append(f"{side} differs from the record: {', '.join(changed[:8])}")
    return findings


def record_sha(path):
    return file_sha(Path(path))
