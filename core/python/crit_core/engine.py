"""Core's engine loader: the one way crit runs an engine (docs/PLUGINS.md "Every engine is plugins").

Core knows no model and no device, so it is never an engine on its own: an engine is Core plus a plugin selection. A
package is a folder holding `plugins/` (its plugins, the plain ones in `plugins/base/`) and `configuration.json`, whose
`plugins` selection fills every slot an engine fills (crit_plugins.REQUIRED_SLOTS) with plugins that run on the
engine's device. load() takes what the checkpoint's config.json says of its architecture as facts
(`model_type:<type>`, `architecture:<class>`), assembles the selection as a whole engine, and runs the model plugin's
object, its batched step model, on Core's concurrent runtime (crit_core.native.StepEngine over crit_core_steps).

crit serve reaches an engine only through resident() (chat-resident-v1, docs/SERVING.md), crit-eval only through
eval_v1() (eval-v1, crit-eval's docs/ADAPTER.md; crit_core/eval_adapter.py is the adapter file crit eval loads). A
package has no entry of its own, and crit serve and crit-eval run no code of a package but its plugins.

Every engine serves the same contract: concurrent requests (chat-concurrent-v1), token log-probabilities with 20
alternatives (chat-logprobs-v1), fixed-text scores (chat-score-v1), Core's sampling fields (temperature, top_p, top_k,
min_p, seed, and the ones its sampler plugin provides), structured output and forced tool calls where the tokenizer's
bytes are known (chat-constraints-v1, chat-constraints-grammar-v1, chat-tool-choice-v1: Core's matchers mask each token
and the sampler applies the mask), images where the model plugin takes them (chat-images-v1: it provides
crit_plugins.IMAGES and prepares them) and, when the selection drafts, plain decoding on request (chat-plain-v1). A model
plugin whose info declares less is refused (serving_problems), and so is one whose rows do not keep to Core's masks
(slot_problems).
"""
import base64
import binascii
from contextlib import contextmanager
import copy
import json
import os
import re
import threading
import time
from pathlib import Path

from crit_core.native import NativeError, StepEngine
from crit_core.serving import (ENGINE_FACTS, EXPERT_FACTS, REQUEST_FACTS, SAMPLING, TOP_LOGPROBS, PreparedPrompt,
                               RequestRejected, ResidentChat, ScoreRows, image_parts)
from crit_core.serving import capabilities as serving_capabilities
from crit_plugins import IMAGES, Image, PluginError, Registry, assemble, check, model_problems
from crit_plugins.assemble import device_bytes, device_memory, normalize, release_cached
from crit_plugins.manifest import HOST, POLICY_SLOTS
from crit_plugins.observe import Calls

FOLDER = "plugins"                     # a package's plugins, as crit_plugins.Registry finds them
CONFIGURATION = "configuration.json"   # a package's configuration, its `plugins` the selection
DEVICE = "cuda"                        # the device an engine serves on unless told otherwise
# The sampling fields every engine applies, Core's sampling law over the sampler slot; a sampler plugin providing a
# fact of SAMPLER_FIELDS adds its fields.
CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
PENALTY, PENALTIES = "penalties", ("presence_penalty", "frequency_penalty", "repetition_penalty")
SAMPLER_FIELDS = {"sampling:penalties": PENALTIES, "sampling:logit_bias": ("logit_bias",),
                  "sampling:min_tokens": ("min_tokens",)}
# The slots a selection drafts with; Eval's plain engine (speculative False) leaves them empty.
SPECULATIVE_SLOTS = ("drafter", "verifier")
# The penalties at the values that change nothing, for requests that must not take an engine's defaults (Eval's).
NEUTRAL_PENALTIES = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
# Engine info facts the eval-v1 adapter passes on (crit-eval's docs/ADAPTER.md): crit serve's engine facts
# (crit_core.serving.ENGINE_FACTS) but speculative and plain, which the adapter states itself, and what only an
# in-process adapter reports: kernel classes and the configuration overrides of --attribute-drift and of a
# wider-draft probe.
FACTS = tuple(key for key in ENGINE_FACTS if key not in ("speculative", "plain")) + ("kernel_work", "exact_variants",
                                                                                   "wider_draft")


def read_configuration(package):
    """A package's configuration.json."""
    path = Path(package) / CONFIGURATION
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PluginError(f"{package} holds no {CONFIGURATION}: a package is its plugins/ and the configuration whose "
                          "plugins selection names them") from None
    if not isinstance(value, dict):
        raise PluginError(f"{path} is a JSON object")
    return value


def selection(configuration, speculative=True):
    """The configuration's plugin selection in its full form (crit_plugins.assemble.normalize); speculative False
    empties the drafting slots, for an engine that decodes one token per target pass."""
    plugins = (configuration or {}).get("plugins")
    if plugins is None:
        raise PluginError("the configuration names no plugins: an engine is Core plus the plugins its plugins.use "
                          "names in every slot an engine fills (docs/PLUGINS.md)")
    chosen = normalize(plugins)
    if not speculative:
        dropped = {chosen["use"][slot] for slot in SPECULATIVE_SLOTS if chosen["use"].get(slot)}
        for slot in SPECULATIVE_SLOTS:
            if slot in chosen["use"]:
                chosen["use"][slot] = chosen["base"][slot] = None
        kept = set(chosen["use"].values()) | set(chosen["base"].values())
        chosen["options"] = {k: v for k, v in chosen["options"].items() if k not in dropped - kept}
    return chosen


def architecture_facts(model):
    """What the checkpoint's config.json says of it, as facts plugins need: model_type:<type> (top level and
    text_config), architecture:<class> for each of its architectures, and the formats its matrices are stored in
    (stored_facts)."""
    path = Path(model) / "config.json"
    if not path.is_file():
        return []
    config = json.loads(path.read_text(encoding="utf-8"))
    facts = stored_facts(config)
    for section in (config, config.get("text_config") or {}):
        if isinstance(section, dict) and isinstance(section.get("model_type"), str) and section["model_type"]:
            facts.append("model_type:" + section["model_type"])
    for name in config.get("architectures") or []:
        if isinstance(name, str) and name:
            facts.append("architecture:" + name)
    return list(dict.fromkeys(facts))


def stored_facts(config):
    """The formats a checkpoint's config.json says its matrices are stored in, as facts: stored:nvfp4 (ModelOpt's
    NVFP4 and W4A16_NVFP4, compressed-tensors' nvfp4-pack-quantized, MLX's nvfp4), stored:fp8 (ModelOpt's and
    compressed-tensors' FP8, quant_method fp8), stored:mxfp8 and stored:mxfp4 (microscaled blocks), stored:<method> for
    another quantization method (awq, gptq, bitsandbytes, ...), and stored:plain without a quantization config. A
    checkpoint of mixed formats (ModelOpt's MIXED_PRECISION with its quantized_layers, compressed-tensors' config
    groups) has the fact of each format its layers use."""
    quantization = config.get("quantization_config") or config.get("quantization") or {}
    if not isinstance(quantization, dict) or not quantization:
        return ["stored:plain"]
    method = str(quantization.get("quant_method") or "").lower()
    names = {str(quantization.get(key) or "").lower() for key in ("quant_algo", "format", "mode")}
    for group in (quantization.get("config_groups") or {}).values():
        if isinstance(group, dict):
            names.add(str(group.get("format") or "").lower())
            names.add(str((group.get("weights") or {}).get("type") or "").lower())
    layers = quantization.get("quantized_layers")
    for layer in (layers.values() if isinstance(layers, dict) else ()):
        if isinstance(layer, dict):
            names.add(str(layer.get("quant_algo") or "").lower())
    found = []
    if any("nvfp4" in name for name in names):
        found.append("stored:nvfp4")
    if method == "fp8" or any(name.startswith("fp8") or name == "float-quantized" for name in names):
        found.append("stored:fp8")
    for block in ("mxfp8", "mxfp4"):
        if any(block in name for name in names):
            found.append("stored:" + block)
    if not found and method:
        found.append("stored:" + re.sub(r"[^a-z0-9_-]", "-", method))
    return found or ["stored:plain"]


# The tensor-core formats a CUDA compute capability brings, as facts plugins need: BF16 from 8.0, FP8 from 8.9 and
# block-scaled FP4 (NVFP4) from 10.0.
CUDA_FEATURES = (("cuda:bf16", 80), ("cuda:fp8", 89), ("cuda:fp4", 100))
_PROBED = {}


def cuda_capability(probe=None):
    """The compute capability (120 for 12.0) of the engine's GPU: from `probe`, a crit eval probe device file (its
    compute_capability, or its first device's), else by nvidia-smi of the GPU the engine runs on, the first that
    CUDA_VISIBLE_DEVICES names (an index or a UUID) or GPU 0 without it; None when unknown."""
    if probe is not None:
        found = probe.get("compute_capability") or next(
            (d.get("compute_capability") for d in probe.get("devices") or [] if isinstance(d, dict)), None)
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu = visible.split(",")[0].strip() if visible and visible.strip() else "0"
        if ("nvidia-smi", gpu) not in _PROBED:
            import subprocess
            try:
                done = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", gpu],
                                      capture_output=True, text=True, timeout=20)
                lines = done.stdout.strip().splitlines() if done.returncode == 0 else []
                _PROBED[("nvidia-smi", gpu)] = lines[0].strip() if lines else None
            except (OSError, subprocess.SubprocessError):
                _PROBED[("nvidia-smi", gpu)] = None
        found = _PROBED[("nvidia-smi", gpu)]
    try:
        major, minor = (int(part) for part in str(found).split("."))
    except (TypeError, ValueError):
        return None
    return major * 10 + minor


def machine_facts(device, probe=None):
    """The device an engine serves on, as facts plugins need: device:<kind> and, for CUDA, its compute capability
    (cuda:sm_120) and the formats its tensor cores have (CUDA_FEATURES), from `probe` (a crit eval probe device file)
    or the machine. A feature Core cannot read is left out, so a plugin that needs it does not fit."""
    facts = ["device:" + device]
    if device == "cuda":
        capability = cuda_capability(probe)
        if capability:
            facts += [f"cuda:sm_{capability}"] + [fact for fact, least in CUDA_FEATURES if capability >= least]
    return facts


def registry_of(package):
    folder = Path(package) / FOLDER
    if not folder.is_dir():
        raise PluginError(f"{package} has no {FOLDER}/ folder: an engine package is its plugins and the "
                          f"{CONFIGURATION} whose plugins selection names them")
    return Registry.discover(folder)


def engine_device(registry, use, configuration):
    """The device an engine serves on: the configuration's "device", else the first of the model plugin's devices
    that every other computing plugin of the selection runs on (a model plugin lists its devices in the order it
    prefers them)."""
    if (configuration or {}).get("device"):
        return configuration["device"]
    model = use.get("model")
    if model is None or model not in registry:
        return DEVICE
    others = [registry[p]["devices"] for p in set(use.values()) if p and p != model and p in registry
              and HOST not in registry[p]["devices"] and registry[p]["devices"]]
    for device in registry[model]["devices"]:
        if all(device in devices for devices in others):
            return device
    return (registry[model]["devices"] or [DEVICE])[0]


def check_package(package, model, configuration=None, device=None, probe=None):
    """Check, without running any plugin code, that a package is an engine for `model` on `device` (engine_device
    when not given): its plugins and its configuration's selection fill every slot an engine fills, every need met,
    the device's facts from `probe` (a crit eval probe device file: a run's) or the machine. Returns (use, order,
    options)."""
    configuration = read_configuration(package) if configuration is None else configuration
    chosen, registry = selection(configuration), registry_of(package)
    device = device or engine_device(registry, chosen["use"], configuration)
    return check(chosen, registry, architecture_facts(model) + machine_facts(device, probe), device=device)


def sampling_fields(registry, use):
    """The sampling fields the engine applies: Core's, and the ones its sampler plugin provides."""
    sampler = use.get("sampler")
    provided = set(registry[sampler]["provides"]) if sampler else set()
    extra = [field for fact, fields in SAMPLER_FIELDS.items() if fact in provided for field in fields]
    return tuple(field for field in SAMPLING if field in CORE_SAMPLING or field in extra)


def capabilities(package, model, configuration=None):
    """The serving capabilities of a package's engine for `model` (crit_core.serving.capabilities): the contract
    every engine serves, the sampling fields its sampler adds, plain decoding when it drafts, structured output and
    forced calls where Core can match the tokenizer's bytes (constrains), and what the model's chat templates teach
    (tools, reasoning, template variables)."""
    from crit_core.model import open_model
    from crit_core.text import load_model_text
    configuration = read_configuration(package) if configuration is None else configuration
    chosen = selection(configuration)
    text = load_model_text(open_model(Path(model)).files, clock=None)
    speculative = any(chosen["use"].get(slot) for slot in SPECULATIVE_SLOTS)
    kept = constrains(text, text.generation.stop_token_ids)
    registry = registry_of(package)
    images = IMAGES in registry[chosen["use"]["model"]]["provides"]
    return serving_capabilities(text, sampling_fields(registry, chosen["use"]), concurrent=True, logprobs=True,
                                score=True, plain=speculative, grammars=kept, constraints=kept, images=images,
                                completion=True)


def constrains(text, stop_tokens):
    """Whether Core keeps the model's replies to constraints: its matchers need every token's exact bytes
    (crit_core.constraints.constrainable) and a stop token to end a constrained reply."""
    from crit_core.constraints import constrainable
    return bool(stop_tokens) and constrainable(text.tokenizer)


DATA_URL = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", re.S)


def request_images(request):
    """The images of a request's messages in order (crit_plugins.Image), decoded from their base64 data: URLs: Core
    fetches no URL, so any other source is refused (RequestRejected)."""
    found = []
    for message in request["messages"]:
        content = message.get("content")
        for part in content if isinstance(content, list) else ():
            if part.get("type") != "image_url":
                continue
            source = part.get("image_url") or {}
            url = source.get("url") if isinstance(source, dict) else source
            match = DATA_URL.fullmatch(url) if isinstance(url, str) else None
            if not match:
                raise RequestRejected("Core's adapter takes images as base64 data: URLs: it fetches no URL")
            try:
                data = base64.b64decode(match[2], validate=True)
            except binascii.Error:
                raise RequestRejected("an image's data: URL does not hold base64") from None
            found.append(Image(data, match[1], (source.get("detail") if isinstance(source, dict) else None) or "auto"))
    return found


def reply_end(tokenizer, stop_tokens):
    """The stop token that ends a constrained reply: the tokenizer's EOS when it is one, else the lowest."""
    return tokenizer.eos_token_id if tokenizer.eos_token_id in stop_tokens else min(stop_tokens)


def serving_problems(info, speculative, fields):
    """Why the model's info falls short of the contract every engine serves, or []."""
    problems = []
    if not info.get("token_logprobs") or int(info.get("max_top_logprobs") or 0) < TOP_LOGPROBS:
        problems.append(f"info() reports token_logprobs with max_top_logprobs of at least {TOP_LOGPROBS} "
                        "(chat-logprobs-v1)")
    lacking = [key for key in ("scoring", "query_scoring", "replay") if not info.get(key)]
    if lacking:
        problems.append(f"info() declares no {', '.join(lacking)}: every engine scores fixed text (chat-score-v1)")
    support = info.get("sampling") or {}
    unapplied = [field for field in fields if not support.get(PENALTY if field in PENALTIES else field)]
    if unapplied:
        problems.append(f"info()'s sampling does not apply {', '.join(unapplied)}")
    if bool(info.get("speculative")) != speculative:
        problems.append(f"info() declares speculative {bool(info.get('speculative'))}, but the selection "
                        f"{'drafts' if speculative else 'drafts nothing'}")
    if speculative and not info.get("plain"):
        problems.append("info() declares no plain: an engine that drafts also decodes one token per target pass on "
                        "request (chat-plain-v1)")
    return problems


class LoadContext:
    """What every plugin's factory and memory() get as `context`: the package, the checkpoint (`model`, its
    config.json as `config`), the configuration, the context the engine serves (`context_tokens`, None for the
    model's own), the device, whether it drafts (`speculative`) and, under crit serve, crit serve's model context
    (`serving`: verified model files, device leases that stay loaded when the engine is built again; None in
    process)."""

    def __init__(self, package, model, configuration, context_tokens=None, device=DEVICE, speculative=True,
                 serving=None):
        self.package, self.model = Path(package), Path(model)
        path = self.model / "config.json"
        self.config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        self.configuration = dict(configuration)
        self.context_tokens = int(context_tokens) if context_tokens else None
        self.device, self.speculative, self.serving = device, bool(speculative), serving

    def model_file(self, name):
        """A file of the checkpoint: crit serve's verified copy when served, else the model folder's."""
        return Path(self.serving.model_file(name)) if self.serving is not None else self.model / name


def load(package, model, configuration=None, *, context_tokens=None, speculative=True, device=None, serving=None,
         library=None):
    """The package's engine for `model`: its selection assembled as a whole engine on `device` (crit_plugins.assemble;
    engine_device when not given; every plugin on a device speaks its slot's device form, crit_plugins.device) and its
    model plugin run on Core's concurrent runtime (StepEngine; `library` is crit_core_steps, else
    CRIT_CORE_STEP_LIBRARY). configuration: the package's configuration.json unless given (a run's candidate, an
    ablation's variant). A model plugin that cannot batch or falls short of the serving contract is refused, its
    plugins closed again, and so is one that does not use its slots: it must ask the placement plugin where its
    tensors live, and every computing slot of the selection must take part in a first request (slot_problems), and
    on CUDA it may hold no more device memory than its placement plans put there (plan_problems). Once every factory
    has run, the memory torch's allocator holds cached for nothing (the load's temporaries) goes back to the device
    (release_cached), before the engine opens (its native rounds capture) and before the plan check measures; the
    engine's telemetry carries the device memory at both points ("load"). The engine's facts carry the plugins it
    runs."""
    package = Path(package)
    configuration = read_configuration(package) if configuration is None else dict(configuration)
    chosen = selection(configuration, speculative)
    registry = registry_of(package)
    device = device or engine_device(registry, chosen["use"], configuration)
    context = LoadContext(package, model, configuration, context_tokens, device, speculative, serving)
    before = device_bytes(device)
    calls = Calls()
    assembly = assemble(chosen, registry, context, architecture_facts(model) + machine_facts(device), device=device,
                        observe=calls, measure=lambda: device_bytes(device))
    engine = None
    try:
        step_model = assembly["model"]
        problems = model_problems(step_model)
        if not problems:
            drafts = any(assembly.get(slot) is not None for slot in SPECULATIVE_SLOTS)
            problems = serving_problems(step_model.info(), drafts, sampling_fields(registry, assembly.use))
        if not problems and not calls["placement"]:
            problems = ["it never asked the placement plugin where its tensors live"]
        # The plugins' own facts (prompt_activations, ...), where the model's info states none of its own.
        stated = step_model.info()
        facts = {key: value for key, value in assembly.facts().items() if key == "plugins" or key not in stated}
        if IMAGES in registry[assembly.use["model"]]["provides"]:
            facts["images"] = True
            if not callable(getattr(step_model, "prepare", None)):
                problems.append(f"it provides {IMAGES} but has no prepare(ids, images, cancelled)")
        if problems:
            raise PluginError(f"{assembly.use['model']}: Core cannot serve it: " + "; ".join(problems))
        load = {"made": device_memory(device), "released_cached": release_cached(device)}
        load["opened"] = device_memory(device)
        engine = StepEngine(step_model, library, facts=facts, close=assembly.close,
                            telemetry=lambda: {**assembly.telemetry(), "load": dict(load)})
        problems = slot_problems(engine, assembly.use, calls)
        if problems:
            raise PluginError(f"{assembly.use['model']}: the engine is not its plugins: " + "; ".join(problems))
        load["warmed"] = warm_requests(engine)
        # What the warm requests left in torch's cache (their intermediates, a long prompt chunk's workspaces) goes back
        # to the device too: Core reads the device's free memory, and the engine keeps only what it plans.
        load["released_after_warm"] = release_cached(device)
        after = device_bytes(device)
        load["checked"] = device_memory(device)
        problems = plan_problems(calls.plans, None if after is None else after - (before or 0),
                                 assembly.declared_device(), assembly.unmeasured())
        if problems:
            raise PluginError(f"{assembly.use['model']}: the engine is not where its plan put it: " + "; ".join(problems))
        # The checks passed: the counting proxies leave the hot path, each plugin calling the others' objects directly.
        assembly.unobserve()
        return engine
    except BaseException:
        if engine is not None:
            engine.close()
        else:
            assembly.close()
        raise


# What a placement plan need not list: the small tensors (norms, tables) and the workspace a model keeps, as a share of
# the plan's device bytes plus a fixed amount.
PLAN_SLACK = (0.05, 64 << 20)


def plan_problems(plans, held, declared=0, unmeasured=()):
    """Why the model's device memory breaks its placement plans, or []: `held` is what the engine allocated on the
    device while it loaded and served its first requests, the plans' device bytes what its plugins asked to put
    there, `declared` the device bytes the plugins' own memory counts (Assembly.declared_device: each declaration at
    most what its factory was measured to take; `unmeasured` the plugins counted at their declaration); a model may
    hold PLAN_SLACK more."""
    if held is None:
        return []
    planned = sum(size for tensors, places in plans for name, size in tensors.items() if places.get(name) == "device")
    planned += int(declared or 0)
    if held <= planned * (1 + PLAN_SLACK[0]) + PLAN_SLACK[1]:
        return []
    counted = f"; unmeasured, counted at their declarations: {', '.join(unmeasured)}" if unmeasured else ""
    return [f"it holds {held / 2**30:.2f} GiB on the device, and its placement plans put {planned / 2**30:.2f} GiB "
            f"there (its plugins' own memory included, each declaration at most what its factory took{counted}): "
            "every tensor the engine keeps on the device (weights, KV, state) goes through the placement plugin's "
            "plan, and a plugin's own workspaces through its memory()"]


# The slots a first request must reach: every filled slot that computes. Policy slots decide, and may stay idle in a
# request (the placement plugin is asked at load).
def used_slots(use):
    return [slot for slot, plugin in use.items() if plugin and slot != "model" and slot not in POLICY_SLOTS]


def slot_problems(engine, use, calls, prompt_tokens=8, output_tokens=2):
    """Why the engine is not its plugins, or []: one short greedy request (token ids 1 to prompt_tokens) runs, and
    every computing slot of the selection must have been called during it (crit_plugins.observe): a call through its
    proxy, or a replay of a graph whose capture called it (a prompt chunk's, a decode round's, a native round's). Then the same
    prompt is kept to one token its greedy choice is not ({"type": "tokens"}, Core's mask on the row): the model plugin
    must hand each row's allowed set to its sampler, or no constrained reply keeps to its language."""
    vocabulary = int(engine.info().get("vocabulary") or 2)
    prompt = [1 + i % (vocabulary - 1) for i in range(prompt_tokens)]
    native = getattr(engine, "rounds", None)
    ran = native.runs[0] if native is not None else 0
    calls.reset()
    first, _ = engine.generate(prompt, output_tokens, stop_tokens=[], sampling={"temperature": 0.0})
    if native is not None and native.runs[0] > ran:
        calls.replayed(native.captured_slots)   # Core's native rounds replayed the captures of those slots' calls
    idle = [slot for slot in used_slots(use) if not calls[slot]]
    problems = [f"its model plugin never called the {slot} plugin ({use[slot]}) during a request: the plugin of a "
                "slot is the code that does the slot's work" for slot in idle]
    other = (first[0] + 1) % vocabulary if first else 0
    try:
        kept, _ = engine.generate(prompt, 1, stop_tokens=[], sampling={"temperature": 0.0},
                                  constraint={"type": "tokens", "tokens": [other]})
    except NativeError as error:
        kept = str(error)
    if kept != [other]:
        problems.append(f"its rows do not keep to Core's masks: a request allowed only token {other} gave {kept} (the "
                        "model plugin hands each row's StepRow.allowed to the sampler as params['allowed'])")
    return problems


# Prompt lengths the warm-up serves beside its short prompt: past an attention window and the default prompt buckets
# (eager prompt rows, longer chunks of the recurrence and the projections), none a multiple of 16.
WARM_PROMPTS = (37, 601)


def warm_requests(engine, prompt_tokens=8, output_tokens=4):
    """Serve at load every kind of request the first one (slot_problems) is not, so the kernels their paths launch
    compile here and none on a request (crit_plugins.testing.compiles): a sampled request, one with
    log-probabilities, a plain one (an engine that drafts), a replay of the greedy reply with log-probabilities and
    the prompt scored with alternatives (Eval's scored continuation, chat-score-v1), longer prompts (WARM_PROMPTS,
    within the engine's context: rows past an attention window and the prompt buckets run the eager prompt paths),
    and two greedy requests of other prompts together (a step of two rows). Each path's first request through a
    model's step(), its eager rounds and its prompt rows then runs at load. Returns the kinds run, which the
    engine's load telemetry records."""
    from concurrent.futures import ThreadPoolExecutor
    info = engine.info()
    vocabulary = int(info.get("vocabulary") or 2)
    context = int(info.get("context") or 0)

    def prompt_of(count, seed=0):   # token ids 1 .. vocabulary - 1, other ids for another seed
        return [1 + (seed * 104729 + i * 7919) % (vocabulary - 1) for i in range(count)]
    prompt = [1 + i % (vocabulary - 1) for i in range(prompt_tokens)]
    greedy = {"temperature": 0.0}
    ran = []
    first, _ = engine.generate(prompt, output_tokens, stop_tokens=[], sampling=greedy)
    engine.generate(prompt, output_tokens, stop_tokens=[], sampling={"temperature": 1.0})
    ran.append("sampled")
    engine.generate(prompt, output_tokens, stop_tokens=[], sampling=greedy, logprobs=TOP_LOGPROBS)
    ran.append("logprobs")
    if info.get("speculative") and info.get("plain"):
        engine.generate(prompt, output_tokens, stop_tokens=[], sampling=greedy, plain=True)
        ran.append("plain")
    if first:
        engine.generate(prompt, len(first), stop_tokens=[], sampling=greedy, replay=list(first), logprobs=TOP_LOGPROBS)
        ran.append("replay")
    paths = info.get("score_paths") or []
    engine.score(prompt, 1, "prefill" if "prefill" in paths else "", top_k=TOP_LOGPROBS)
    ran.append("score")
    for seed, length in enumerate(WARM_PROMPTS, 1):
        length = min(length, context - output_tokens - 1) if context else length
        if length > prompt_tokens:
            engine.generate(prompt_of(length, seed), output_tokens, stop_tokens=[], sampling=greedy)
            ran.append(f"prompt:{length}")
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda seed: engine.generate(prompt_of(prompt_tokens + seed, 10 + seed), output_tokens,
                                                   stop_tokens=[], sampling=greedy), range(2)))
    ran.append("together")
    return ran


def engine_times(result):
    """The engine's own prefill and first-token milliseconds from a generation's timing (Core's prefill_ms and
    first_token_ms), for crit-eval's time-to-first-token breakdown; the ones it does not report are left out."""
    timing = (result or {}).get("timing") or {}
    return {key: float(timing[key]) for key in ("prefill_ms", "first_token_ms")
            if isinstance(timing.get(key), (int, float)) and not isinstance(timing[key], bool)}


def request_facts(result, plain=False):
    """The facts a generation's result reports beside its timing (crit_core.serving.REQUEST_FACTS), as crit serve's
    telemetry and Eval read them: without the drafting ones when it decoded plainly, which drafts nothing."""
    return {key: result[key] for key in (EXPERT_FACTS if plain else REQUEST_FACTS) if key in result}


def decode_rounds(tokens, result, speculative):
    """Target passes after prefill: the engine's own count, else one per accepted entry, else Core's decode steps
    (which also count steps that run no target pass, such as seeding a drafter); one per token after the first
    without drafting."""
    if "rounds" in result:
        return result["rounds"]
    if "accepted" in result:
        return len(result["accepted"])
    return result["timing"]["decode_steps"] if speculative else max(0, len(tokens) - 1)


class EvalEngine:
    """eval-v1 (crit-eval's docs/ADAPTER.md) over a package's engine: greedy generation with stop tokens disabled,
    replay, plain decoding when it drafts, fixed-text scores (`score`, for `crit eval run --only quality` in process,
    development evidence), the engine's own prefill and first-token times with every generation, and its headroom
    facts. context: crit-eval's, its config the configuration over the package's, with `package` naming the
    package."""

    def __init__(self, context):
        given = dict(context.get("config") or {})
        package = given.pop("package", None)
        if not package:
            raise PluginError("Core's eval-v1 adapter serves a package: name it as config.package (crit eval run "
                              "--package DIR)")
        configuration = {**read_configuration(package), **given}
        # Eval's speculative=False (plain one-token decoding on a second engine) empties the drafting slots.
        speculative = bool(context.get("speculative", True))
        self.configuration = configuration
        self.native = load(package, context["model"], configuration, context_tokens=context.get("context_tokens"),
                           speculative=speculative)
        self.facts = self.native.info()
        self.speculative = bool(self.facts.get("speculative"))
        # Eval decodes greedily: temperature zero and neutral penalties where the engine applies them.
        support = self.facts.get("sampling") or {}
        self.sampling = {"temperature": 0.0, **(NEUTRAL_PENALTIES if support.get(PENALTY) else {})}
        self.replay = self._replay
        if self.speculative:
            self.generate_plain = self._generate_plain
        # Eval's sampled suite (crit-eval-v7): the engine samples at the settings Eval names, applying these fields.
        self.sampling_fields = {key for key in SAMPLING if support.get(PENALTY if key in PENALTIES else key)}
        self.generate_sampled = self._generate_sampled
        self.score = self._score

    def info(self):
        info = {"name": self.facts.get("name") or self.facts["implementation"], "configuration": self.configuration,
                "max_context": self.facts["context"], "speculative": self.speculative,
                "native_libraries": list(self.facts.get("native_libraries") or []), "replay": True,
                "plain": self.speculative, "plugins": self.facts.get("plugins")}
        info.update({key: self.facts[key] for key in FACTS if key in self.facts})
        return info

    def _result(self, tokens, result):
        out = {"tokens": tokens, "rounds": decode_rounds(tokens, result, self.speculative), **request_facts(result)}
        return {**out, **engine_times(result)}

    def generate(self, prompt_ids, max_new_tokens, on_tokens):
        tokens, result = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[],
                                              sampling=self.sampling, on_tokens=on_tokens)
        return self._result(tokens, result)

    def _generate_sampled(self, prompt_ids, max_new_tokens, on_tokens, sampling, logprobs=None):
        """A sampled request; with logprobs n, the emitted token's log-probability and n alternatives per token.
        Fields the engine does not apply (sampling_fields) are left out; Eval sends such fields only at their neutral
        values."""
        sampling = {key: value for key, value in sampling.items() if key in self.sampling_fields}
        rows = []

        def deliver(tokens, found=None):
            on_tokens(tokens)
            if found:
                rows.extend(zip(tokens, found))
        tokens, result = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[],
                                              sampling=dict(sampling), logprobs=logprobs, on_tokens=deliver)
        out = self._result(tokens, result)
        if logprobs:
            out["logprobs"] = [{"token": token, "logprob": logprob, "top": [list(pair) for pair in top]}
                               for token, (logprob, top) in rows]
        return out

    def _generate_plain(self, prompt_ids, max_new_tokens, on_tokens):
        # One target pass per token after the first, and no drafting facts.
        tokens, result = self.native.generate(list(prompt_ids), int(max_new_tokens), stop_tokens=[],
                                              sampling=self.sampling, plain=True, on_tokens=on_tokens)
        return {"tokens": tokens, "rounds": max(0, len(tokens) - 1), **request_facts(result, plain=True),
                **engine_times(result)}

    def _score(self, prompt_ids, continuation_ids, top_k, prompt_rows=True):
        """Fixed text as crit serve scores it: every prompt token after the first by query scoring on the engine's
        "prefill" score path (its default path when it names none), every continuation token from a greedy replay
        forced to it with log-probabilities, greedy with neutral penalties as Eval's requests are. Rows {"logprob",
        "top": [[token, logprob], ...top_k]} (crit-eval's ADAPTER.md)."""
        ids, continuation, rows = list(prompt_ids), list(continuation_ids), {"prompt": [], "continuation": []}
        if prompt_rows and len(ids) > 1:
            paths = self.facts.get("score_paths") or []
            scored = self.native.score(ids, 1, "prefill" if "prefill" in paths else "", top_k=top_k)
            rows["prompt"] = [{"logprob": float(logprob), "top": [[int(t), float(v)] for t, v in top or []]}
                              for logprob, top in zip(scored.target, scored.top or [[] for _ in scored.target])]
        if continuation:
            found = []
            tokens, _ = self.native.generate(ids, len(continuation), stop_tokens=[], replay=continuation,
                                             sampling=self.sampling, logprobs=top_k,
                                             on_tokens=lambda _tokens, values: found.extend(values))
            if list(tokens) != continuation:
                raise NativeError("replay emitted other tokens than the continuation")
            rows["continuation"] = [{"logprob": float(logprob), "top": [[int(t), float(v)] for t, v in top]}
                                    for logprob, top in found]
        return rows

    def _replay(self, prompt_ids, continuation, on_tokens):
        tokens, result = self.native.generate(list(prompt_ids), len(continuation), stop_tokens=[],
                                              sampling=self.sampling, replay=list(continuation),
                                              on_tokens=on_tokens)
        return self._result(tokens, result)

    def telemetry(self):
        """The engine's work so far (crit_core.native.StepEngine.telemetry: its plugins' telemetry, such as the device
        expert cache's {slots, resident, lent, reads, hits, fills, fill_bytes} and the drafter's accepted drafts by
        depth, and Core's step counters), which Eval prices its limits with."""
        return self.native.telemetry()

    def capacity(self):
        """The requests Core's runtime runs at once on the engine (its max_concurrent_requests)."""
        value = self.facts.get("max_concurrent_requests")
        return value if type(value) is int and value >= 1 else None

    def cohort(self, clients, max_new_tokens):
        """One closed-loop level of crit-eval's concurrency part, in process: clients[c] is client c's prompts, each
        generated greedily (as Eval's speed cases, stop tokens off) once the client's previous one ended, every client
        starting at once on its own thread. {"seconds": the level's wall interval, "requests": per request its client,
        index, prompt_tokens, the seconds from the level's start when it was sent, its first and last tokens came and
        it ended (started, first, last, finished), completion_tokens, finish_reason, cached_tokens (0: no prompt
        cache in process), errors (none: no stream to validate), its tokens and, where the engine reports them, its
        expert_reads and expert_hits}, as a served engine's cohort reports it."""
        start, rows, failures, epoch = threading.Event(), [], [], [0.0]

        def client(index, prompts):
            start.wait()
            for position, prompt in enumerate(prompts):
                stamps = []
                started = time.perf_counter() - epoch[0]
                try:
                    tokens, result = self.native.generate(
                        list(prompt), int(max_new_tokens), stop_tokens=[], sampling=self.sampling,
                        on_tokens=lambda found, *rest: stamps.append(time.perf_counter() - epoch[0]))
                except BaseException as error:  # noqa: BLE001 - the level stops with it
                    failures.append(error)
                    return
                rows.append({"client": index, "index": position, "prompt_tokens": len(prompt), "started": started,
                             "first": stamps[0] if stamps else None, "last": stamps[-1] if stamps else None,
                             "finished": time.perf_counter() - epoch[0], "completion_tokens": len(tokens),
                             "finish_reason": result.get("finish"), "cached_tokens": 0, "errors": [],
                             "tokens": list(tokens),
                             # The request's routed expert reads and cache hits, as a served cohort's telemetry has
                             # them, which Eval prices the level's speedup at.
                             **{key: result[key] for key in EXPERT_FACTS if type(result.get(key)) is int}})
        threads = [threading.Thread(target=client, args=(c, prompts), name=f"crit-eval-client-{c}", daemon=True)
                   for c, prompts in enumerate(clients)]
        for thread in threads:
            thread.start()
        epoch[0] = time.perf_counter()
        start.set()
        for thread in threads:
            thread.join()
        seconds = time.perf_counter() - epoch[0]
        if failures:
            raise failures[0]
        rows.sort(key=lambda row: (row["client"], row["index"]))
        return {"seconds": seconds, "requests": rows}

    def close(self):
        self.native.close()


class ResidentEngine(ResidentChat):
    """chat-resident-v1 (docs/SERVING.md) over a package's engine, for crit serve: crit_core.serving.ResidentChat with
    the engine's stop tokens, context and sampling fields, token log-probabilities (chat-logprobs-v1), fixed-text
    scores (chat-score-v1: the prompt by query scoring with the engine's own top-k on its "prefill" score path, the
    default path when it names none, the continuation by replay with log-probabilities), concurrent requests
    (chat-concurrent-v1), replies kept to a response_format or to forced calls where Core can match the tokenizer's
    bytes (constrains: Core's matchers over the model's byte vocabulary, built on the first such request, mask each
    token, a constrained reply ending at reply_end), images where the model plugin takes them (chat-images-v1: the
    request's data: URLs decoded, request_images, and laid out by the model plugin's prepare() on the request's
    thread; its inputs reach the model's open() with the request) and, when it drafts, plain decoding on request
    (chat-plain-v1). Its info is the resident
    engine's `facts` (crit_core.serving.engine_facts). Configuration "chat_template_kwargs" passes variables to the
    chat template."""

    def __init__(self, context, package):
        configuration = {**read_configuration(package), **(context.configuration or {})}
        model = Path(context.model_file("config.json")).parent
        super().__init__(context, model, configuration=configuration)
        limits = getattr(context, "limits", {}) or {}
        self.native = load(package, model, self.configuration, context_tokens=limits.get("kv_tokens"),
                           serving=context)
        self.facts = self.native.info()
        self.max_concurrent_requests = int(self.facts["max_concurrent_requests"])
        self._owns_native = True
        self.capacity = min(int(self.facts["context"]), int(limits.get("kv_tokens") or self.facts["context"]))
        self.stop_ids = set(self.facts.get("stop_tokens") or [])
        support = self.facts.get("sampling") or {}
        self.sampling = tuple(key for key in SAMPLING if support.get(PENALTY if key in PENALTIES else key))
        self.logprobs = self.scores = True
        self.plain = bool(self.facts.get("speculative"))
        self.grammars = self.constraints = constrains(self.text, self.stop_ids)
        self.images = bool(self.facts.get("images"))
        self._vocabulary = {"lock": threading.Lock(), "ready": False}   # the requests' copies share it

    def create_request(self, request):
        # Share only the immutable tokenizer/configuration and the synchronized native
        # engine. ResidentChat's parser discovery caches belong to this wrapper.
        child = copy.copy(self)
        child._owns_native = False
        child.configuration = dict(self.configuration)
        child.template_variables = dict(self.template_variables)
        child.stop_ids = set(self.stop_ids)
        child.kept_special = child.calls_format = None
        child.__dict__.pop("_kept_cache", None)
        child.__dict__.pop("_thinking", None)
        return child

    def _constrain(self):
        """Gives Core the model's byte vocabulary, once, before the first constrained request."""
        from crit_core.constraints import byte_vocabulary
        state = self._vocabulary
        with state["lock"]:
            if not state["ready"]:
                tokenizer = self.text.tokenizer
                self.native.constrain(byte_vocabulary(tokenizer, reply_end(tokenizer, self.stop_ids)))
                state["ready"] = True

    @contextmanager
    def prepare_prompt(self, request, cancellation):
        if not image_parts(request):
            yield PreparedPrompt(self.prompt_ids(request))
            return
        if not self.images:
            raise RequestRejected("This engine's model plugin takes no images")
        images = request_images(request)
        ids, inputs = self.native.prepare(self.prompt_ids(request), images, cancellation.is_set)
        yield PreparedPrompt(ids, inputs, len(ids))

    def generate_prepared(self, prompt, max_new_tokens, sampling, on_tokens, cancelled, **options):
        return self.generate_tokens(prompt.ids, max_new_tokens, sampling, on_tokens, cancelled, inputs=prompt.inputs,
                                    **options)

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, logprobs=None, plain=False,
                        constraint=None, inputs=None):
        if constraint is not None:
            self._constrain()
        tokens, result = self.native.generate(ids, max_new_tokens, on_tokens=on_tokens, sampling=sampling,
                                              logprobs=logprobs, plain=plain, cancelled=cancelled,
                                              constraint=constraint, inputs=inputs)
        if tokens is None:
            return None, {}
        timing = result["timing"]
        out = {"prefill_tokens": timing["prefill_tokens"] - timing["reused_prompt_tokens"],
               "prefill_ms": timing["prefill_ms"], "decode_ms": timing["decode_ms"],
               "first_token_ms": timing["first_token_ms"],
               "rounds": decode_rounds(tokens, result, bool(self.facts.get("speculative")) and not plain),
               **request_facts(result, plain)}
        return tokens, out

    def score_tokens(self, ids, continuation, top_k, cancelled, prompt_rows):
        started = time.monotonic()
        paths = self.facts.get("score_paths") or []
        # The engine's own arrays as ScoreRows (top_k > 0), which ResidentChat.score packs without a row loop.
        scored = (self.native.score(ids, 1, "prefill" if "prefill" in paths else "", top_k=top_k, packed=True)
                  if prompt_rows and len(ids) > 1 else None)
        prefill_ms = (time.monotonic() - started) * 1000
        rows = []
        if continuation:
            # Greedy, with neutral penalties: a sampled replay would accept drafts by the sampling rule instead of by
            # equality, and the engine's default penalties would move the scores.
            greedy = {"temperature": 0.0, **{key: value for key, value in NEUTRAL_PENALTIES.items()
                                             if key in self.sampling}}
            tokens, result = self.native.generate(list(ids), len(continuation), stop_tokens=[], replay=list(continuation),
                                                  sampling=greedy, logprobs=top_k,
                                                  on_tokens=lambda _tokens, found: rows.extend(found),
                                                  cancelled=cancelled)
            if tokens is None:
                return None, None, {}
            if list(tokens) != list(continuation):
                raise NativeError("replay emitted other tokens than the continuation")
            decode_ms = result["timing"]["decode_ms"]
            # The replay prefills the prompt again before its first row: engine time like the score's prefill.
            prefill_ms += result["timing"].get("prefill_ms") or 0.0
        else:
            decode_ms = None
        if scored is None:
            found = []
        elif isinstance(scored.top, ScoreRows):
            found = scored.top
        else:
            found = list(zip(scored.target, scored.top if top_k else [[] for _ in scored.target]))
        return found, rows, {"prefill_ms": prefill_ms, "decode_ms": decode_ms}

    def telemetry(self):
        """The engine's work so far (crit_core.native.StepEngine.telemetry), which crit serve's /health reports as
        engine_telemetry."""
        return self.native.telemetry()

    def close(self):
        if self._owns_native:
            self.native.close()


def resident(context, package):
    """crit serve's engine for a package (crit_api.host): ResidentEngine over load()."""
    return ResidentEngine(context, package)


def eval_v1(context):
    """crit-eval's engine for a package (eval-v1): EvalEngine over load()."""
    return EvalEngine(context)


__all__ = ["CONFIGURATION", "FOLDER", "EvalEngine", "LoadContext", "ResidentEngine", "capabilities", "check_package",
           "constrains", "eval_v1", "load", "machine_facts", "read_configuration", "reply_end", "request_images",
           "resident", "selection", "serving_problems", "stored_facts"]
