"""crit diff: where an engine leaves the reference, layer by layer.

    crit diff --package PACKAGE --model MODEL [--tokens 32] [--seed 0] [--reference-device cpu] [--json]

The package's engine (a candidate's package folder, or any plugin package) and the model's official forward
(transformers, in FP32) read the same token sequence as one prompt. The engine gives the residual's running row on
entry to every layer and its final-normed rows (its model's layer_rows(): the tap the models core offers), the
reference its hidden states at the same places (output_hidden_states). Each place's gap is the largest difference of a
row, relative to the reference row's root mean square. The 16-bit baseline is the same gap of the reference run in
BF16 against itself in FP32: how far rounding alone moves that place. Where the checkpoint declares the formats its
own inference quantizes a matrix's input activations to (a W4A4 or W8A8 checkpoint: crit-eval's
limits.schemes.activations, the formats crit eval reference's allowance pass applies), the declared-format allowance
is the same gap of the FP32 reference with every such matrix's inputs quantized as declared. A gap within FACTOR times
the larger of the two (or within FLOOR, where both are smaller) is format-sized; the first place past it is where the
engine's own arithmetic
leaves the reference, and the layer before it produced it: crit diff names that layer, its kind, its mixer and FFN
blocks and the selection's plugins they call. Nothing is judged: it is the localiser a failed gate or a smoke check
points at (crit smoke's decode_matches_prefill names it)."""
import json
from pathlib import Path
import random

# A place's gap is format-sized within FACTOR times the larger of the 16-bit baseline's and the declared-format
# allowance's (where the checkpoint declares one), or FLOOR, whichever is larger: the rule is one factor over the
# largest gap a format the engine may use moves the reference by. Rounding at 16 bits, and a declared activation
# format's rounding, grow layer by layer about as their own gaps do; a wrong product or a misread weight by far more.
FACTOR = 8.0
FLOOR = 1e-3
# The slots a layer's blocks call, by the kind of block: the plugins a divergence there may lie in.
SLOTS = {"mixer": ("attention", "kv_cache", "projections", "norms", "recurrence"),
         "ffn": ("projections", "norms", "router", "experts")}


def gap(found, want):
    """The largest difference of a row of `found` from `want` ([tokens][hidden]), relative to the RMS of `want`'s."""
    found, want = found.float(), want.float()
    scale = want.square().mean(-1).sqrt().clamp(min=1e-12)
    return float(((found - want).abs().amax(-1) / scale).max())


def reference_rows(model, ids, dtype, trust_remote_code=False, declared=False, device="cpu"):
    """The reference's hidden states of ids: [rows on entry to each layer] + [final-normed rows], each [tokens][hidden]
    FP32. declared: every matrix whose checkpoint declares its input activations' format multiplies its inputs
    quantized as declared (declared_inputs); None where the checkpoint declares none. device: where the reference runs
    (TF32 off: FP32 is FP32 on a GPU too)."""
    import torch
    import transformers
    reference = transformers.AutoModelForCausalLM.from_pretrained(str(model), dtype=dtype,
                                                                  trust_remote_code=trust_remote_code).to(device)
    if declared and not declared_inputs(reference, model):
        return None
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            out = reference(torch.tensor([ids], device=device), output_hidden_states=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn
    states = [h[0].float().cpu() for h in out.hidden_states]
    return states   # embeddings (layer 0's entry), each later layer's entry, then the final-normed rows


def declared_inputs(reference, model):
    """Quantize the inputs of every nn.Linear of `reference` whose stored matrix the checkpoint declares input
    activations for, as its own inference does (crit-eval's limits.schemes.activations and quantize_activations, the
    allowance pass's rule), by forward pre-hooks. Returns how many matrices it hooked (0: the checkpoint declares
    none)."""
    import torch
    from crit_eval.limits.schemes import activations, quantize_activations
    from crit_eval.quality.checkpoints import open_checkpoint
    checkpoint = open_checkpoint(model, reference.config.get_text_config())
    hooked = 0
    for name, module in reference.named_modules():
        if not isinstance(module, torch.nn.Linear) or not checkpoint.has(name + ".weight"):
            continue
        found = activations(checkpoint, name)
        if not found:
            continue
        kind, scale = found
        module.register_forward_pre_hook(lambda _, args, kind=kind, scale=scale:
                                         (quantize_activations(args[0].float(), kind, scale).to(args[0].dtype),)
                                         + tuple(args[1:]))
        hooked += 1
    return hooked


def sequence(model, tokens, seed):
    config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    vocab = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
    rng = random.Random(seed)
    return [rng.randrange(3, min(vocab, 250)) for _ in range(tokens)]


def compare(engine_rows, reference32, reference16, parts=None, selection=None, declared=None):
    """The places' gaps and the first one past the format-sized bound: {"places": [{name, gap, baseline, declared,
    bound, sized}], "first_divergent": {name, layer, kind, mixer, ffn, plugins} or None, "verdict": "format-sized" or
    "bug-sized"}. declared: the declared-format reference's rows (reference_rows declared), or None."""
    if len(engine_rows) != len(reference32):
        raise ValueError(f"the engine gives {len(engine_rows)} places, the reference {len(reference32)}: the engine's "
                         "layers are not the reference's")
    places, first = [], None
    for i, ((name, rows), want, rounded) in enumerate(zip(engine_rows, reference32, reference16)):
        baseline = gap(rounded, want)
        allowance = gap(declared[i], want) if declared is not None else None
        bound = max(FACTOR * max(baseline, allowance or 0.0), FLOOR)
        found = gap(rows, want)
        sized = "format-sized" if found <= bound else "bug-sized"
        places.append({"name": name, "gap": found, "baseline": baseline, "declared": allowance, "bound": bound,
                       "sized": sized})
        if first is None and sized == "bug-sized":
            first = i
    divergent = None
    if first is not None:
        # The rows on entry to place `first` are made by the layer before it (the embedding rows for layer 0).
        layer = first - 1
        divergent = {"name": places[first]["name"], "made_by": "the embedding" if layer < 0 else f"layer {layer}"}
        if layer >= 0 and parts:
            part = parts[layer]
            divergent.update(layer=layer, kind=part.get("kind"), mixer=part.get("mixer"), ffn=part.get("ffn"))
            if selection:
                divergent["plugins"] = sorted({selection[slot] for slot in SLOTS["mixer"] + SLOTS["ffn"]
                                               if selection.get(slot)})
    return {"places": places, "first_divergent": divergent,
            "verdict": "bug-sized" if divergent else "format-sized"}


def layer_diff(package, model, tokens=32, seed=0, trust_remote_code=False, device=None, reference=None,
               reference_device="cpu"):
    """The diff of the package's engine against the reference on `model` (the module's description). reference: the
    checkpoint the official forward reads, `model` by default (a test gives the engine a checkpoint it altered).
    reference_device: where the reference's FP32, BF16 and declared-format passes run."""
    import torch
    from crit_core.engine import load, read_configuration, selection
    ids = sequence(model, tokens, seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = load(package, model, device=device)
    try:
        rows = engine.layer_rows(ids)   # Core's tap fact: any model plugin that offers layer_rows (crit_core.native)
        parts = engine.layer_parts()
    finally:
        engine.close()
    reference = model if reference is None else reference
    at = reference_device
    found = compare(rows, reference_rows(reference, ids, torch.float32, trust_remote_code, device=at),
                    reference_rows(reference, ids, torch.bfloat16, trust_remote_code, device=at), parts,
                    selection(read_configuration(package))["use"],
                    reference_rows(reference, ids, torch.float32, trust_remote_code, declared=True, device=at))
    return {"model": str(model), "package": str(package), "tokens": len(ids), "seed": seed, **found}


def render(found):
    lines = [f"{p['name']:<10} gap {p['gap']:.2e}  16-bit baseline {p['baseline']:.2e}"
             + (f"  declared formats {p['declared']:.2e}" if p.get("declared") is not None else "")
             + f"  {p['sized']}" for p in found["places"]]
    first = found["first_divergent"]
    if first is None:
        lines.append("every place is within the format-sized bound (16-bit baseline, declared formats): no layer "
                     "leaves the reference beyond rounding")
    else:
        where = ", ".join(f"{k} {first[k]}" for k in ("kind", "mixer", "ffn") if first.get(k))
        lines.append(f"first bug-sized gap: on entry to {first['name']}, made by {first['made_by']}"
                     + (f" ({where})" if where else "")
                     + (f"; the plugins it calls: {', '.join(first['plugins'])}" if first.get("plugins") else ""))
    return "\n".join(lines)


def parser():
    import argparse
    found = argparse.ArgumentParser(prog="crit diff", description="Where an engine leaves the reference, layer by "
                                    "layer: its rows on entry to every layer against the official forward's, beside "
                                    "the 16-bit baseline")
    found.add_argument("--package", type=Path, required=True, help="The engine's package (a candidate's package/)")
    found.add_argument("--model", type=Path, required=True, help="The checkpoint folder")
    found.add_argument("--tokens", type=int, default=32,
                       help="The text's tokens (any length: the engine runs them in prompt chunks)")
    found.add_argument("--seed", type=int, default=0, help="The random text's seed")
    found.add_argument("--trust-remote-code", action="store_true", help="Let transformers run the model's own code")
    found.add_argument("--reference-device", default="cpu",
                       help="Where the reference's passes run (cpu, or cuda for a large checkpoint; FP32 either way)")
    found.add_argument("--json", action="store_true", help="Print the diff as JSON")
    return found


def main(argv=None):
    args = parser().parse_args(argv)
    found = layer_diff(args.package, args.model, args.tokens, args.seed, args.trust_remote_code,
                       reference_device=args.reference_device)
    print(json.dumps(found, indent=2) if args.json else render(found))
    return 0
