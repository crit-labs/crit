"""The candidate a build starts from, for any checkpoint: the library's plugins at their defaults, packaged in a run.

library_selection() reads the checkpoint (its config, its stored formats, its tensor names) and the finished model
plugin that serves its model_type, and fills every slot an engine fills with the library's device plugin for it
(DEVICE_DEFAULTS: the plugin AGENTS.md's starting designs take, the plain one where the checkpoint's facts rule it
out, and the next of SLOT_ORDER where the default does not provide a fact the checkpoint's layers need). Options are the defaults, except where the checkpoint states the rule (the router's) or the run's placement
does (lookup tables off the device; elastic KV pools and recurrence states beside an expert cache, as crit plugins offload gives every
engine whose experts may live off the device). build_run() makes a run of it in a run store and packages that one
candidate as an engine build does: crit run create, recon, capture, resolve, plan, candidate and package."""
import json
from pathlib import Path
import shutil

# The library's device plugin of each slot a build's first candidate takes, before any plan changes it: the plugins
# AGENTS.md's starting designs name (quantized projections over the stored formats, the split-KV reader, the native
# sampler, fused norms, the row reader of lookup tables, the expert cache that places experts by what fits). They are
# general: each runs every checkpoint its slot serves, and where one does not fit the checkpoint the plain plugin
# beside it does (PLAIN_FOR).
DEVICE_DEFAULTS = {"placement": "device-placement", "projections": "quantized-projections",
                   "attention": "partitioned-attention", "kv_cache": "attention-state",
                   "sampler": "token-constraints", "norms": "fused-norms", "embeddings": "lookup-table-rows",
                   "recurrence": "sequential-recurrence", "router": "topk-router", "experts": "expert-offload",
                   "expert_source": "expert-tensors", "drafter": "greedy-head-drafter"}
# The plain plugin where the device default does not fit: projections of matrices stored in 16 bits.
PLAIN_FOR = {"quantized-projections": "projections-dense"}
# The library's plugins of a slot whose facts a checkpoint's layers may need, in the order the selection tries them:
# the device default first, then the next one that provides every fact the checkpoint's layers need of the slot
# (needed_facts). A sliding-window or latent checkpoint so takes softmax-attention, which serves every layer kind,
# where partitioned-attention provides neither attention:sliding nor attention:latent.
SLOT_ORDER = {"attention": ("partitioned-attention", "softmax-attention")}
# The drafter a run's drafting through the drafter seam serves (scenarios: drafting_drafter_seam), copied beside.
SEAM_DRAFTER = "mtp-decode"
# The drafter of a companion drafter folder (EAGLE-3, DFlash, DSpark: hidden-state-drafters' README). A companion the
# model plugin's layout reads as its own draft block drafts through the model's draft head instead (the device default
# drafter, DEVICE_DEFAULTS["drafter"]).
COMPANION_DRAFTER = "hidden-state-drafters"


def model_plugin(model):
    """The finished model plugin that serves the checkpoint's model_type ({id, layout, ...}), or None."""
    from ..models import _types, plugin_for
    return plugin_for(_types(model))


def draft_prefix(plugin):
    """The prefix of the draft layer's tensors the model plugin's layout reads, or None where it reads none."""
    draft = ((plugin or {}).get("layout") or {}).get("draft") or {}
    return draft.get("prefix") or None


def facts(model, plugin=None):
    """What the selection and the plan read of a checkpoint: {model_plugin, routed, recurrent, draft_layer, stored,
    sliding_window, latent, sinks, layer_kinds, vocab_size, experts, draft_companion (the companion drafter's model_type the model
    plugin's layout reads as its draft block, or None)}. plugin: model_plugin(model) where already found."""
    from ..models import _config
    from crit_core.engine import architecture_facts
    from .cut import tensor_names
    plugin = model_plugin(model) if plugin is None else plugin
    config = _config(model)
    kinds = tuple(config.layer_kinds())
    prefix = draft_prefix(plugin)
    names = tensor_names(model)
    stored = sorted(fact for fact in architecture_facts(model) if fact.startswith("stored:"))
    return {"model_plugin": (plugin or {}).get("id"), "routed": bool(config.moe is not None and config.routed_layers()),
            "recurrent": "linear_attention" in kinds,
            "draft_layer": bool(prefix and any(name.startswith(prefix) for name in names)),
            "draft_prefix": prefix, "stored": stored, "sliding_window": config.sliding_window,
            "draft_companion": (((plugin or {}).get("layout") or {}).get("draft") or {}).get("companion"),
            "latent": config.mla is not None, "sinks": any(name.endswith(".sinks") for name in names),
            "layer_kinds": sorted(set(kinds)), "vocab_size": getattr(config, "vocab_size", None),
            "experts": getattr(config.moe, "experts", 0) if config.moe is not None else 0}


def needed_facts(found):
    """{slot: facts} every layer kind of the checkpoint needs a slot's plugin to provide (the facts the models core's
    mixers refuse to load without): attention:sliding for sliding-window layers, attention:latent for latent (MLA)
    attention, attention:sinks for stored attention sinks."""
    attention = set()
    if "sliding_attention" in found.get("layer_kinds", ()):
        attention.add("attention:sliding")
    if found.get("latent"):
        attention.add("attention:latent")
    if found.get("sinks"):
        attention.add("attention:sinks")
    return {"attention": attention}


def serving(slot, needs, registry=None):
    """The first plugin of SLOT_ORDER[slot] whose manifest provides every fact of `needs`. Raises ValueError where none
    does, naming the facts no library plugin of the slot provides."""
    if registry is None:
        from ..plugins import library
        registry = library()
    for plugin in SLOT_ORDER[slot]:
        if set(needs) <= set(registry[plugin].get("provides", ())):
            return plugin
    raise ValueError(f"no library plugin of the {slot} slot provides {', '.join(sorted(needs))}, which the "
                     f"checkpoint's layers need")


def library_selection(model, found=None, tables_off_device=False, companion=False):
    """configuration.plugins of the library's plugins at their defaults for the checkpoint, and the plugins a package
    copies: ({"use", "options"}, [plugin id]). found: facts(model). tables_off_device: the run's placement keeps the
    lookup tables off the device (device-placement's off_device "tables."). companion: the run names a companion
    drafter: the package also holds hidden-state-drafters (or, where the model plugin's layout reads a companion as
    its draft block, the device default drafter, which drafts from the model's draft head), which a load drafts with
    by a variant of the selection (scenarios: drafting_companion), so a drafter the library cannot read leaves the
    candidate as it is. Raises ValueError where no finished model plugin serves the
    checkpoint: the build writes one first (AGENTS.md: start from the selection)."""
    from ..models import options_for
    found = facts(model) if found is None else found
    if found["model_plugin"] is None:
        raise ValueError("no finished model plugin serves this checkpoint's model_type (crit models inspect names what "
                         "it needs): a build writes one first, from the model techniques")
    slots = ["placement", "projections", "attention", "kv_cache", "sampler", "norms", "embeddings"]
    if found["recurrent"]:
        slots.append("recurrence")
    if found["routed"]:
        slots += ["router", "experts", "expert_source"]
    if found["draft_layer"]:
        slots.append("drafter")
    use = {"model": found["model_plugin"], **{slot: DEVICE_DEFAULTS[slot] for slot in slots}}
    for slot, needs in needed_facts(found).items():
        if needs and slot in use:
            use[slot] = serving(slot, needs)
    if use["projections"] == "quantized-projections" and found["stored"] in ([], ["stored:plain"]):
        use["projections"] = PLAIN_FOR["quantized-projections"]
    options = {}
    if found["draft_layer"]:
        options[use["model"]] = {"mtp": True}
    options.update(options_for(model, use))
    if tables_off_device:
        options["device-placement"] = {"off_device": "tables."}
    if found["routed"]:
        # what crit plugins offload gives every engine whose experts may exceed the device (AGENTS.md): the KV pools
        # hold what their open sequences use and the expert cache the rest
        options["attention-state"] = {"elastic": True}
        if found["recurrent"]:
            options["sequential-recurrence"] = {"elastic": True}
    plugins = list(dict.fromkeys(use.values()))
    if found["draft_layer"]:
        plugins.append(SEAM_DRAFTER)
    if companion:
        plugins.append(DEVICE_DEFAULTS["drafter"] if found.get("draft_companion") else COMPANION_DRAFTER)
    return {"use": use, "options": options}, plugins


def build_run(store, model, mode, sources, selection, plugins, request=None, criteria="crit run preflight"):
    """A run of `model` in quality mode `mode` with one candidate packaged from the library's plugins `plugins` under
    `selection` (crit run create, recon, capture, resolve, plan, candidate and package, as an engine build does them).
    request: the run request's other values (kv_tokens, device, headrooms). Returns (run_id, candidate_id)."""
    from ..plugins import add
    from ..run import recon
    from ..run.capture import capture
    view = store.create({"model_path": str(model), "correctness_criteria": criteria, "quality_mode": mode,
                         "candidate_budget": 64,
                         "workload": {"prompt_tokens": 512, "output_tokens": 64, "repetitions": 1}, **(request or {})})
    run_id = view["run"]["run_id"]
    root = store.directory(run_id)
    store.recon(run_id)
    notes = root / "work" / "recon.md"
    text = notes.read_text(encoding="utf-8")
    for title in recon.PLANS + (recon.PLACEMENT_PLAN,):
        text = text.replace(f"## {title}\n\nTODO", f"## {title}\n\nThe library's plugins at their defaults "
                            f"({criteria}).")
    notes.write_text(text, encoding="utf-8")
    case = view["run"]["spec"]["cases"][0]["case_id"]
    row = capture(store, run_id, case)
    store.freeze_inputs(run_id, {"hardware_manifest": {"device": criteria},
                                 "environment_manifest": {"benchmark": criteria},
                                 "case_manifests": {case: [row["row"] if "row" in row else row]}})
    plan = {"parent_candidate_id": None, "decision": "build",
            "needed": {name: {"needed": True, "evidence": f"{criteria} serves every path"}
                       for name in ("decode", "prefill")},
            "gaps": [], "levers": [], "rejected": [], "forecast": None, "gates": f"{criteria}'s scenarios",
            "strategy": {"current": {"name": "library defaults", "ceiling": "unknown", "evidence": criteria},
                         "alternative": {"name": "none", "ceiling": "unknown", "evidence": criteria},
                         "choice": "current", "reason": f"{criteria} measures the library as it is"}}
    if mode == "speed":
        plan["prompt_activations"] = "stored"
    plan_id = store.plan(run_id, plan)["plan_id"]
    sources = Path(sources)
    if sources.exists():
        shutil.rmtree(sources)
    (sources / "plugins").mkdir(parents=True)
    add(sources / "plugins", plugins)
    configuration = {"plugins": selection}
    (sources / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    candidate = store.candidate(run_id, sources, {
        "label": "library defaults", "rationale": criteria, "strategy": "library defaults",
        "configuration": configuration, "plan_id": plan_id, "lever_fates": {}, "store_decisions": {}})
    (sources / "configuration.json").unlink()   # crit writes the candidate's completed configuration into the package
    store.package(run_id, candidate["candidate_id"], sources,
                  {"build": {"toolchain": {"identity": criteria}, "dependencies": [], "command": ["none"]}})
    return run_id, candidate["candidate_id"]
