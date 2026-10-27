"""Drafter production: distill a draft head for speculative decoding from the target model itself, as a recorded
transform with lineage, so a model that ships no MTP layer can still draft.

The drafter is an MTP-style head (kind `mtp-head`): at position t it reads the target's final hidden state (before the
final norm) and the embedding of token t+1, normalizes both, fuses them with one linear layer, runs one decoder layer
of the target's own architecture (initialized from a target layer, the last by default) and the target's final norm,
and predicts through the target's own output head, which is not trained and not stored with the drafter. It is
trained to match the target's next-token distribution at position t+1 (the token t+2) over calibration windows
unrelated to Eval, by the KL divergence from the target's distribution; a held-out share of the windows measures
how often its greedy draft equals the target's (top-1 agreement) and the KL left. The target's weights never change,
so an engine that verifies every draft with the stored target keeps the target's output; `weights: as_stored`
still holds for the target and the drafter rides along like a companion draft model, with its record.

The target pass follows the PTQ calibration pass (crit_models.ptq.calibrate): token embeddings through the decoder
layers with the model's rotary embeddings and a causal mask, layer by layer so a model larger than device memory
still runs; models whose forward does more around their layers are outside this pass, as they are for GPTQ.
"""
import copy
import json
from pathlib import Path
import platform
import re
import shutil
import time

from crit.run.records import digest, file_sha, inventory, now, write_json

from .ptq import _directory, _environment, _inside, peak_host_bytes

KIND = "mtp-head"
FORMAT_VERSION = "crit-drafter-v1"
WEIGHTS = "model.safetensors"
LAYER = re.compile(r"layers\.(\d+)\.")
CHUNK = 256  # positions per logits chunk when the target and draft distributions are compared


def plan_drafter(recipe, checkpoint):
    """What the recipe builds from a checkpoint's headers: the decoder prefix, the layer the drafter starts from, its
    tensors, parameter count and stored bytes."""
    from crit_core.model.schemes import layer_prefix
    tensors = checkpoint.tensors
    prefix = layer_prefix(tensors)
    layers = sorted({int(m.group(1)) for name in tensors for m in [LAYER.match(name[len(prefix):])] if m and name.startswith(prefix)})
    if not layers:
        raise ValueError("the checkpoint has no decoder layers")
    count = layers[-1] + 1
    init = recipe["init_layer"] if recipe["init_layer"] >= 0 else count + recipe["init_layer"]
    if not 0 <= init < count:
        raise ValueError(f"init_layer {recipe['init_layer']} is outside the model's {count} layers")
    names = sorted(name for name in tensors if name.startswith(f"{prefix}layers.{init}."))
    parameters = 0
    for name in names:
        size = 1
        for dim in tensors[name].shape:
            size *= dim
        parameters += size
    return {"prefix": prefix, "layers": count, "init_layer": init, "layer_tensors": names, "parameters": parameters,
            "stored_bytes": sum(tensors[n].nbytes for n in names), "modules_sha256": digest(names),
            "checkpoint_bytes": checkpoint.tensor_bytes}


def _causal(length, device, torch):
    return torch.full((length, length), float("-inf"), device=device).triu(1)[None, None]


def _layer_forward(layer, state, embeddings, mask, positions):
    out = layer(state, position_embeddings=embeddings, attention_mask=mask, position_ids=positions)
    return out[0] if isinstance(out, tuple) else out


def target_pass(reference, plan, windows, device, log):
    """Run the target over every window, layer by layer: returns (final hidden states per window on the host, a
    deep copy of the init layer, the loaded final norm, the head weight on the device, the embedding table)."""
    import torch
    config = reference.config
    prefix = plan["prefix"]
    if reference.prefix != prefix:
        raise ValueError(f"decoder prefix {reference.prefix!r} differs from the plan's {prefix!r}")
    table = reference.checkpoint.embedding(prefix + "embed_tokens").float()
    states = [table[torch.tensor(ids)][None] for ids in windows]
    length = len(windows[0])
    positions = torch.arange(length, device=device)[None]
    embeddings = reference.skeleton.rotary_emb.to(device)(states[0].to(device), positions)
    mask = _causal(length, device, torch)
    layer_types = getattr(config, "layer_types", None)
    init_copy = None
    with torch.no_grad():
        for index in range(plan["layers"]):
            started = time.time()
            layer = reference.layer(index, index + 1 if index + 1 < plan["layers"] else None)
            if index == plan["init_layer"]:
                init_copy = copy.deepcopy(layer)
            full = layer_types is None or layer_types[index] != "linear_attention"
            for s, state in enumerate(states):
                states[s] = _layer_forward(layer, state.to(device), embeddings, mask if full else None, positions).to("cpu")
            reference.skeleton.layers[index] = torch.nn.Identity()
            del layer
            if device != "cpu":
                torch.cuda.empty_cache()
            log(f"target layer {index + 1}/{plan['layers']} {time.time() - started:.1f}s")
        norm = reference._load(reference.skeleton.norm, "norm.")
        if reference.checkpoint.has("lm_head.weight") or any(reference.checkpoint.has("lm_head." + s) for s in ("weight_packed", "qweight", "weight_scale")):
            head = reference.checkpoint.dense("lm_head", device).float()
        else:
            head = table.to(device)  # tied embeddings
    return states, init_copy, norm, head, table, embeddings, mask, positions


class Drafter:
    """The draft head as a torch module: built from the target's pieces, saved as crit-drafter-v1 tensors."""

    def __init__(self, layer, norm, torch):
        hidden = norm.weight.shape[0]
        self.torch = torch
        self.module = torch.nn.Module()
        self.module.pre_fc_norm_hidden = copy.deepcopy(norm)
        self.module.pre_fc_norm_embedding = copy.deepcopy(norm)
        self.module.fc = torch.nn.Linear(2 * hidden, hidden, bias=False, device=norm.weight.device, dtype=torch.float32)
        with torch.no_grad():
            eye = torch.eye(hidden, device=norm.weight.device)
            self.module.fc.weight.copy_(torch.cat([eye, eye], dim=1) * 0.5)
        self.module.layers = torch.nn.ModuleList([layer])
        self.module.norm = copy.deepcopy(norm)
        for parameter in self.module.parameters():
            parameter.requires_grad_(True)
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()

    def forward(self, hidden, embedded, rotary, mask, positions):
        fused = self.module.fc(self.torch.cat([self.module.pre_fc_norm_hidden(hidden), self.module.pre_fc_norm_embedding(embedded)], -1))
        return self.module.norm(_layer_forward(self.module.layers[0], fused, rotary, mask, positions))

    def tensors(self, dtype):
        return {"drafter." + name: value.detach().to("cpu").to(dtype).contiguous()
                for name, value in self.module.state_dict().items()}


def _divergence(target_logits, draft_logits, torch):
    """Mean KL(target || draft) over positions and the share where both argmaxes agree."""
    log_p = torch.log_softmax(target_logits.float(), -1)
    log_q = torch.log_softmax(draft_logits.float(), -1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    agree = (target_logits.argmax(-1) == draft_logits.argmax(-1)).float()
    return kl.mean(), agree.mean()


def _compare(drafter, window, final, table, head, norm, rotary, mask, positions, device, torch, train=False):
    """Loss (KL) and agreement of the drafter on one window: draft at t from (h_t, e_{t+1}) against the target at
    t+1, in chunks of positions so the two vocabularies' logits never sit in memory at once."""
    ids = torch.tensor(window, device=device)
    final = final.to(device)
    hidden, embedded = final[:, :-1], table.to(device)[ids[1:]][None]
    length = hidden.shape[1]
    draft = drafter.forward(hidden, embedded, (rotary[0][:, :length], rotary[1][:, :length]), mask[..., :length, :length],
                            positions[:, :length])
    with torch.no_grad():
        target = norm(final[:, 1:])
    kls, agrees = [], []
    for start in range(0, length, CHUNK):
        stop = min(length, start + CHUNK)
        with torch.no_grad():
            target_logits = target[0, start:stop] @ head.T
        draft_logits = draft[0, start:stop] @ head.T
        kl, agree = _divergence(target_logits, draft_logits, torch)
        if train:
            (kl * (stop - start) / length).backward(retain_graph=stop < length)
        kls.append(kl.detach() * (stop - start))
        agrees.append(agree.detach() * (stop - start))
    return float(sum(kls) / length), float(sum(agrees) / length)


def train(drafter, windows, finals, table, head, norm, rotary, mask, positions, settings, seed, device, log):
    import torch
    optimizer = torch.optim.AdamW(drafter.module.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    losses = []
    head.requires_grad_(False)
    for parameter in norm.parameters():
        parameter.requires_grad_(False)
    for step in range(settings["steps"]):
        scale = min(1.0, (step + 1) / max(1, settings["warmup_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = settings["learning_rate"] * scale
        optimizer.zero_grad(set_to_none=True)
        picks = torch.randperm(len(windows), generator=generator)[:settings["windows_per_step"]].tolist()
        total = 0.0
        for pick in picks:
            kl, _ = _compare(drafter, windows[pick], finals[pick], table, head, norm, rotary, mask, positions, device, torch, train=True)
            total += kl / len(picks)
        for parameter in drafter.module.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(len(picks))
        torch.nn.utils.clip_grad_norm_(drafter.module.parameters(), 1.0)
        optimizer.step()
        losses.append(total)
        if step % max(1, settings["steps"] // 10) == 0 or step == settings["steps"] - 1:
            log(f"step {step + 1}/{settings['steps']} loss {total:.4f}")
    return losses


def evaluate(drafter, windows, finals, table, head, norm, rotary, mask, positions, device):
    import torch
    kls, agrees = [], []
    with torch.no_grad():
        for window, final in zip(windows, finals):
            kl, agree = _compare(drafter, window, final, table, head, norm, rotary, mask, positions, device, torch)
            kls.append(kl)
            agrees.append(agree)
    return {"kl_mean": sum(kls) / len(kls), "top1_agreement": sum(agrees) / len(agrees), "windows": len(windows)}


def export(out, drafter, source, config, plan, recipe, record_pointer):
    """Write the drafter: config.json (the target's, with one layer and the crit_drafter description) and its
    tensors."""
    from safetensors.torch import save_file
    import torch
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[recipe["dtype"]]
    tensors = drafter.tensors(dtype)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / WEIGHTS), metadata={"format": "pt"})
    written = {**config, "num_hidden_layers": 1, "crit_drafter": {
        "format": FORMAT_VERSION, "kind": KIND, "init_layer": plan["init_layer"], "target_layers": plan["layers"],
        "inputs": "the target's final hidden state before its final norm at position t, and the embedding of token t+1",
        "fusion": "pre_fc_norm_hidden and pre_fc_norm_embedding (copies of the target's final norm), then fc "
                  "(2 x hidden -> hidden, no bias)",
        "layer": "layers.0: one decoder layer of the target's architecture, then norm (a copy of the target's final norm)",
        "head": "the target's own output head (lm_head, or the tied embeddings), not stored here",
        "target": {"path": str(source), "eval_model_sha256": record_pointer}, "dtype": recipe["dtype"]}}
    for key in ("quantization_config",):
        written.pop(key, None)  # the drafter's tensors are stored plain
    (out / "config.json").write_text(json.dumps(written, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sorted(tensors)


def run_drafter(recipe, source, out, record_path, calibration, device=None, log=print):
    """Produce a drafter for `source` by `recipe` into `out` and write the models_transform record."""
    import torch
    from crit import __version__
    from crit_core.model import inspect_model
    from crit_eval.quality.prepare import model_identity
    from crit_eval.quality.reference import StreamingReference
    from .calibration import load_calibration
    started_at, clock = now(), time.perf_counter()
    phases = {}
    out, record_path = Path(out).absolute(), Path(record_path).absolute()
    source = _directory(source, "source")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if _inside(out, source) or _inside(source, out):
        raise ValueError("the output and source directories must not contain each other")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"{out} must be a new or empty directory")
    if record_path.exists() or _inside(record_path, out) or _inside(record_path, source):
        raise ValueError("the record must be a new file outside the source and output directories")
    if calibration is None:
        raise ValueError("a drafter needs --calibration: documents unrelated to Eval")
    plan = plan_drafter(recipe, inspect_model(source))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(recipe["seed"])
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    mark = time.perf_counter()
    source_files = inventory(source)
    source_identity = model_identity(source)["sha256"]
    phases["hash_source"] = time.perf_counter() - mark
    mark = time.perf_counter()
    windows, facts = load_calibration(calibration, source, recipe)
    heldout_count = max(1, round(len(windows) * recipe["heldout_fraction"]))
    if len(windows) - heldout_count < 1:
        raise ValueError("the calibration needs at least two windows: one to train on and one held out")
    train_windows, heldout_windows = windows[:-heldout_count], windows[-heldout_count:]
    facts = {**facts, "train_windows": len(train_windows), "heldout_windows": heldout_count}
    log(f"{len(train_windows)} training windows, {heldout_count} held out; drafter from layer {plan['init_layer']} "
        f"of {plan['layers']} ({plan['parameters']:,} parameters) on {device}")
    reference = StreamingReference(source, device)
    finals, init_layer, norm, head, table, rotary, mask, positions = target_pass(reference, plan, windows, device, log)
    phases["target_pass"] = time.perf_counter() - mark
    drafter = Drafter(init_layer, norm, torch)
    mark = time.perf_counter()
    before = evaluate(drafter, heldout_windows, finals[-heldout_count:], table, head, norm, rotary, mask, positions, device)
    losses = train(drafter, train_windows, finals[:-heldout_count], table, head, norm, rotary, mask, positions,
                   recipe["train"], recipe["seed"], device, log)
    after = evaluate(drafter, heldout_windows, finals[-heldout_count:], table, head, norm, rotary, mask, positions, device)
    phases["train"] = time.perf_counter() - mark
    mark = time.perf_counter()
    names = export(out, drafter, source, config, plan, recipe, source_identity)
    inspect_model(out)
    output_files = inventory(out)
    phases["export"] = time.perf_counter() - mark
    head_count = max(1, len(losses) // 10)
    record = {
        "schema_version": 1, "record_kind": "models_transform", "crit_version": __version__,
        "recipe": recipe, "recipe_sha256": digest(recipe), "seed": recipe["seed"],
        "source": {"path": str(source), "files": source_files, "manifest_sha256": digest(source_files),
                   "eval_model_sha256": source_identity},
        "calibration": facts,
        "plan": {k: plan[k] for k in ("prefix", "layers", "init_layer", "parameters", "stored_bytes", "modules_sha256",
                                      "checkpoint_bytes")} | {"kind": KIND, "format": FORMAT_VERSION, "tensors": names},
        "output": {"path": str(out), "files": output_files, "manifest_sha256": digest(output_files),
                   "eval_model_sha256": model_identity(out)["sha256"], "copied_files": [], "skipped_directories": []},
        "errors": {"train_loss": {"start": sum(losses[:head_count]) / head_count, "end": sum(losses[-head_count:]) / head_count,
                                  "steps": len(losses)},
                   "heldout_before": before, "heldout": after},
        "cost": {"wall_seconds": time.perf_counter() - clock, "phases": phases, "device": device,
                 "device_name": torch.cuda.get_device_name(device) if device != "cpu" else platform.processor() or None,
                 "peak_device_bytes": torch.cuda.max_memory_allocated(device) if device != "cpu" else None,
                 "peak_host_bytes": peak_host_bytes()},
        "environment": _environment(),
        "started_at": started_at, "ended_at": now(),
    }
    write_json(record_path, record, immutable=True)
    log(f"wrote {out} and {record_path}: held-out top-1 agreement {before['top1_agreement']:.3f} -> "
        f"{after['top1_agreement']:.3f}, KL {before['kl_mean']:.4f} -> {after['kl_mean']:.4f} "
        f"({record['cost']['wall_seconds']:.1f}s)")
    return record
