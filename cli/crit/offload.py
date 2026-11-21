"""The settings of a mixture of experts larger than the device, derived instead of written: `crit plugins offload`.

From the checkpoint's config.json and a `crit eval probe` device file, at the context the engine must serve, it reads
crit Eval's placement scenario (which experts fit, the device expert cache's bytes, each miss source's plain-pass floor)
and turns it into expert-offload's selection and options: the miss path, the cache's policy, the prompt
arithmetic, the CPU tier's workers, device-placement's off-device prefixes for lookup tables. With a routing trace (one
(layer, experts) group per routed layer of each decoded token: the cache's own record, record_routes) it also replays
the cache's residency offline at several sizes and copy budgets (device-expert-cache's trace_replay: the rule the
device runs), so what a size or a budget is worth is read in seconds from real routing, not from an engine rebuild.
The cache's size stays expert-offload's own: every byte the device has free past what the device keeps (the run's
VRAM headroom, which Core's loader gives the device from the candidate's configuration: no option of the plan) while
the engine serves."""
import json
from pathlib import Path

GIB = 1 << 30
INFERENCE = Path(__file__).resolve().parents[2] / "inference"
CPU_FORMATS = ("nvfp4", "int4")   # the formats the CPU expert tier computes in the device's bits
SIZES = (0.75, 1.0, 1.25)         # cache sizes replayed, as fractions of the planned slots
SWAPS = (64, 128, 256)            # copies a round replayed (expert-offload's swaps)
PINNED = (0.25, 0.5, 0.75)        # the pinned core's shares of the slots a profile is replayed at


def _load(path, name):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _text_config(config):
    """The decoder's config: a multimodal checkpoint nests it under text_config."""
    return config.get("text_config") or config


def expert_shape(config):
    """(layers with routed experts, experts per layer, experts per token, hidden, expert intermediate), or None for a
    model without routed experts."""
    text = _text_config(config)
    experts = text.get("num_experts") or text.get("n_routed_experts") or text.get("num_local_experts")
    top = text.get("num_experts_per_tok") or text.get("moe_top_k")
    inter = text.get("moe_intermediate_size") or text.get("expert_intermediate_size")
    hidden, layers = text.get("hidden_size"), text.get("num_hidden_layers")
    if not (experts and top and inter and hidden and layers):
        return None
    from crit_core.model import routed_layers
    return len(routed_layers(text)), int(experts), int(top), int(hidden), int(inter)


def expert_format(config):
    """The routed experts' stored format as the experts plugins name it: nvfp4, int4 (AWQ/GPTQ 4-bit), fp8:block,
    fp8:channel or bf16, from config.json's quantization_config."""
    quant = config.get("quantization_config") or _text_config(config).get("quantization_config") or {}
    text = json.dumps(quant).lower()
    method = str(quant.get("quant_method", "")).lower()
    if "nvfp4" in text or "fp4" in str(quant.get("quant_algo", "")).lower():
        return "nvfp4"
    if method in ("awq", "gptq") or ('"bits": 4' in text and method != "fp8"):
        return "int4"
    if method == "fp8" or "fp8" in text:
        return "fp8:block" if quant.get("weight_block_size") else "fp8:channel"
    return "bf16"


def bundle_bytes(format, hidden, inter):
    """One expert's bundle in the experts:bundles layout (grouped-experts' bundles, the CPU tier's for its formats)."""
    bundles = _load(INFERENCE / "grouped-experts" / "code" / "bundles.py", "crit_offload_bundles")
    return int(bundles.layout(format, hidden, inter)["bundle_bytes"])


def _trace_replay():
    """device-expert-cache's trace replay, imported as a package of its own (it takes its rule from the file beside)."""
    import importlib
    import sys
    import types
    name = "crit_offload_cache"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(INFERENCE / "device-expert-cache" / "code")]
        sys.modules[name] = package
    return importlib.import_module(name + ".trace_replay")


def replay_trace(trace, layers, experts, slots, sizes=SIZES, swaps=SWAPS, warmup=0):
    """[{slots, policy, swaps, hit_rate, misses_per_round, copies_per_round}] for the trace replayed through the
    cache's recent residency at each fraction of `slots` and each copy budget, and through the same slots holding the
    trace's own most routed experts and never changing (static: what no profile alone can beat)."""
    for layer, routed in trace:
        if not 0 <= int(layer) < layers or any(not 0 <= int(e) < experts for e in routed):
            raise ValueError(f"a trace route lies outside {layers} layers x {experts} experts")
    replay = _trace_replay()
    rounds = replay.windows(trace)
    rows = []
    for fraction in sizes:
        count = max(1, int(slots * fraction))
        found = replay.sweep(rounds, layers, experts, (count,), swaps=swaps, warmup=warmup)
        found.append(replay.static(rounds, layers, experts, count, warmup=warmup))
        rows += [{"slots": count, "policy": r["policy"], "swaps": r["max_swaps"], "hit_rate": round(r["hit_rate"], 4),
                  "misses_per_round": round(r["misses_per_round"], 1),
                  "copies_per_round": round(r["swaps_per_round"], 1)} for r in found]
    return rows


def replay_pinned(trace, layers, experts, slots, profile, shares=PINNED, swaps=max(SWAPS), warmup=0):
    """[{slots, pinned, hit_rate, misses_per_round}] for the trace replayed through recent residency started from
    `profile` (keys, hottest first), with no core and with each share of the slots holding the profile's hottest
    experts for good (expert-offload's pinned option): what such a core costs in hits on routing other than the
    profile's, before an engine is loaded with one."""
    replay = _trace_replay()
    rounds = replay.windows(trace)
    profile = [int(key) for key in profile if 0 <= int(key) < layers * experts]
    rows = []
    for share in (0.0,) + tuple(shares):
        found = replay.recent(rounds, layers, experts, slots, swaps=swaps, preload=profile, warmup=warmup,
                              pinned=profile[:int(slots * share)])
        rows.append({"slots": found["slots"], "pinned": found["pinned"], "hit_rate": round(found["hit_rate"], 4),
                     "misses_per_round": round(found["misses_per_round"], 1)})
    return rows


def read_profile(path):
    """A profile's keys, hottest first: expert-offload's routing profile or a residency profile file."""
    return [int(key) for key in json.loads(Path(path).read_text(encoding="utf-8"))["keys"]]


def read_trace(path):
    """A routing trace: a JSON list of [layer, [experts]] groups, or JSON Lines of them, or a residency profile's
    trace object ({"trace": [...]})."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("trace", [])
    return [(int(group[0]), [int(e) for e in group[1]]) for group in data]


def plan(model, device, tokens, quality_mode="speed", vram_headroom_bytes=None, ram_headroom_bytes=2 * GIB,
         reserve_bytes=None, trace=None, settings=None, cores=None, profile=None):
    """The offload plan of MODEL (a checkpoint folder) on DEVICE (a crit eval probe result, a dict) at `tokens` of
    context: the placement Eval prices, the derived plugin options and, with a trace, the replayed hit rates."""
    config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    shape = expert_shape(config)
    if shape is None:
        raise ValueError("the model routes no experts: there is nothing to offload")
    layers, experts, top, hidden, inter = shape
    format = expert_format(config)
    size = bundle_bytes(format, hidden, inter)
    if vram_headroom_bytes is None:   # the machine's setting, else Core's default (crit_plugins.memory.HEADROOM)
        from crit.settings import vram_headroom_bytes as headroom_of
        from crit_eval import machine_settings
        vram_headroom_bytes = headroom_of(machine_settings())
    if settings is None:
        from crit_eval import EVAL_DIR
        settings = json.loads((Path(EVAL_DIR) / "protocol.json").read_text(encoding="utf-8"))["context"]
    if reserve_bytes is not None:
        settings = {**settings, "reserve_gib": reserve_bytes / GIB}
    from crit_eval.limits.context import placement
    found = placement(str(model), device, vram_headroom_bytes, ram_headroom_bytes, settings, int(tokens))
    if not found:
        raise ValueError("crit Eval cannot size the model on this device")
    floors = found.get("plain_pass_floor_ms") or {}
    out = {"schema": "crit-offload-plan-v1", "model": str(model), "tokens": int(tokens), "quality_mode": quality_mode,
           "experts": {"layers": layers, "per_layer": experts, "per_token": top, "hidden": hidden, "inter": inter,
                       "format": format, "bundle_bytes": size},
           "placement": found, "options": {}, "notes": []}
    if found["scenario"] in ("fits_device", "dense_device_host", "beyond_host"):
        out["notes"].append(f"scenario {found['scenario']}: the routed experts need no device expert cache")
        return out
    cache = int(found.get("expert_cache_bytes") or 0)
    slots = cache // size
    # The CPU computes misses where its floor (alone or beside the link) beats copying them over the link and the
    # format is one it computes in the device's bits; otherwise every miss is copied.
    path = min((k for k in ("link", "cpu", "split") if k in floors), key=floors.get, default="link")
    cpu = format in CPU_FORMATS and path in ("cpu", "split")
    if format not in CPU_FORMATS and path != "link":
        out["notes"].append(f"the CPU tier computes {' and '.join(CPU_FORMATS)} experts, not {format}: misses are "
                            "copied over the link")
    out["miss_path"] = "cpu" if cpu else "link"
    workers = -1 if cores is None else max(1, int(cores) - 1)
    # The device keeps the run's VRAM headroom free at every moment (Core's, from the candidate's configuration): what
    # the run's checks measure, which the plan sets nowhere. A prompt chunk's arithmetic is the plugin's own choice (prompt_arithmetic auto): the faster one that keeps the
    # level the engine asks of its prompt rows. So is the miss path where the placement prices the CPU first (misses
    # auto): the floors are lower bounds from a probe of an idle device, and the plugin times a missed expert both
    # ways in the loaded engine, where a turn to the host costs what it costs there, and takes the faster.
    plugin = {"misses": "auto" if cpu else "link"}
    if cores is not None:
        plugin["workers"] = workers
    # The KV pools hold only the pages sequences use (attention-state's elastic pools): the expert cache the rest.
    out["options"] = {"device-placement": {"off_device": ""}, "expert-offload": plugin,
                      "attention-state": {"elastic": True}}
    out["selection"] = {"experts": "expert-offload", "expert_source": "expert-tensors"}
    out["cache"] = {"slots": slots, "bytes": slots * size,
                    "share": round(slots / (layers * experts), 4) if layers * experts else 0}
    tables = found.get("lookup_tables")
    if tables and tables["tier"] != "device":
        out["options"]["device-placement"]["off_device"] = "tables."
        # Every table a pass gathers a few rows of, the token embeddings among them: the cache's bytes otherwise.
        out["notes"].append(f"lookup tables ({tables['bytes'] / 1e9:.1f} GB: the token embeddings and every other "
                            "table a pass gathers a few rows of) stay off the device, where each GiB of them would "
                            f"take {GIB // size:,} cache slots: the model plans each as a `tables.` group and opens "
                            "it through the embeddings slot (lookup-table-rows' open(): a tensor's rows in "
                            "registered host memory, gathered over the link at a device table's cost, or files)")
    out["notes"].append("the placement prices misses computed on the CPU below misses copied over the link; "
                        "expert-offload times both in the loaded engine and takes the faster (misses auto; its "
                        "facts say which and why). On the CPU tier every miss is computed while the link copies it "
                        "into the cache beside the round: it serves from the next round" if cpu else
                        "every miss is copied over the link into a cache slot before its layer reads it")
    out["notes"].append(f"about {slots:,} cache slots ({slots * size / 1e9:.1f} GB of the experts' "
                        f"{layers * experts * size / 1e9:.1f} GB, {slots / (layers * experts):.0%}) at Eval's margins: "
                        "expert-offload sizes the cache itself from the device's free memory while the engine "
                        "serves, and elastic KV pools (attention-state) leave it what their sequences do not use")
    out["notes"].append("a model that keeps recurrent state: give its recurrence plugin `elastic` too, so the states "
                        "of sequences that are not open are the cache's")
    if trace:
        rows = replay_trace(trace, layers, experts, slots)
        out["trace"] = {"groups": len(trace), "replay": rows}
        at = [r for r in rows if r["slots"] == max(1, slots) and r["policy"] == "recent"]
        if at:
            top = max(r["hit_rate"] for r in at)
            best = min((r for r in at if r["hit_rate"] >= top - 0.005), key=lambda r: r["swaps"])
            out["trace"]["best"] = best
            if cpu:
                plugin["swaps"] = best["swaps"]
            out["notes"].append(f"replayed routing hits {best['hit_rate']:.1%} at {slots:,} slots with up to "
                                f"{best['swaps']} copies a round ({best['misses_per_round']:.0f} misses and "
                                f"{best['copies_per_round']:.0f} copies a round)")
        if profile and cpu and slots > 1:
            out["trace"]["pinned"] = replay_pinned(trace, layers, experts, slots, profile)
            out["notes"].append("a pinned core (expert-offload's pinned) holds the profile's hottest experts on the "
                                "device alone, a bundle of host memory less for each: its rows above are this "
                                "trace's hits with such a core against none, both started from the profile")
    return out


def render(value):
    e, p = value["experts"], value["placement"]
    lines = [f"{e['layers']} routed layers x {e['per_layer']} experts ({e['per_token']} a token), {e['format']} "
             f"bundles of {e['bundle_bytes'] / 1e6:.2f} MB; scenario {p['scenario']} at {value['tokens']:,} tokens"]
    floors = p.get("plain_pass_ceiling_tps") or {}
    if floors:
        lines.append("plain-pass ceilings at even routing: " + ", ".join(f"{k} {v} tok/s" for k, v in floors.items()))
    if value.get("miss_path"):
        lines.append(f"misses: {value['miss_path']} by the placement's floors"
                     + (" (the plugin times both in the loaded engine and takes the faster)"
                        if value["miss_path"] == "cpu" else ""))
    for plugin, options in value["options"].items():
        lines.append(f"  {plugin}: " + ", ".join(f"{k}={json.dumps(v)}" for k, v in options.items()))
    trace = value.get("trace")
    if trace:
        lines.append(f"trace replay ({trace['groups']:,} groups):")
        for row in trace["replay"]:
            budget = f"{row['swaps']:>4} copies a round" if row["policy"] == "recent" else "fixed residents    "
            lines.append(f"  {row['slots']:>7,} slots {row['policy']:<7} {budget}: hits {row['hit_rate']:.1%}, "
                         f"{row['misses_per_round']:.0f} misses and {row['copies_per_round']:.0f} copies a round")
        for row in trace.get("pinned") or ():
            lines.append(f"  {row['slots']:>7,} slots recent, {row['pinned']:>6,} pinned "
                         f"({row['pinned'] * e['bundle_bytes'] / GIB:.1f} GiB of host memory not held): hits "
                         f"{row['hit_rate']:.1%}, {row['misses_per_round']:.0f} misses a round")
    lines += [f"note: {n}" for n in value["notes"]]
    return "\n".join(lines)


def main(args):
    device = json.loads(Path(args.device_file).read_text(encoding="utf-8"))
    trace = read_trace(args.trace) if args.trace else None
    headroom = None if args.vram_headroom_gib is None else round(args.vram_headroom_gib * GIB)
    value = plan(args.model, device, args.context, args.quality_mode, vram_headroom_bytes=headroom,
                 ram_headroom_bytes=round(args.ram_headroom_gib * GIB),
                 reserve_bytes=None if args.reserve_gib is None else round(args.reserve_gib * GIB),
                 trace=trace, cores=args.cores, profile=read_profile(args.profile) if args.profile else None)
    print(render(value))
    if args.out:
        Path(args.out).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def add_parser(subs):
    sub = subs.add_parser("offload", help="Derive the library plugins' options for experts larger than the device, "
                                          "and replay a routing trace through the expert cache")
    sub.add_argument("--model", type=Path, required=True, help="The checkpoint folder (its config.json)")
    sub.add_argument("--device-file", type=Path, required=True, help="A crit eval probe result for this machine")
    sub.add_argument("--context", type=int, required=True, help="The context tokens the engine must serve")
    from crit_plugins.exactness import QUALITY_MODES
    sub.add_argument("--quality-mode", choices=QUALITY_MODES, default="speed")
    sub.add_argument("--trace", type=Path, help="A routing trace: [layer, [experts]] groups per routed layer of each "
                                                 "decoded token (JSON or JSON Lines; expert-offload's trace_path)")
    sub.add_argument("--profile", type=Path, help="With --trace: a routing or residency profile (its keys, hottest "
                                                   "first); the trace is also replayed with shares of the slots "
                                                   "holding its hottest experts for good (expert-offload's pinned)")
    sub.add_argument("--cores", type=int, help="Physical CPU cores for the CPU tier (default: the tier's own count)")
    sub.add_argument("--vram-headroom-gib", type=float,
                     help="Device memory left free at every moment (default: settings.toml vram_headroom_gib, else "
                          "crit Eval's)")
    sub.add_argument("--ram-headroom-gib", type=float, default=2.0)
    sub.add_argument("--reserve-gib", type=float, help="Device memory kept out of the KV cache (default: Eval's)")
    sub.add_argument("--out", type=Path, help="Write the plan as JSON")
    return sub
