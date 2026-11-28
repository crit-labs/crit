"""Which loads and scenarios a run's checkpoint asks of the preflight, every size from the run.

plan_loads(run, found, placement, companion) reads the run (its context, quality mode, VRAM headroom, device file)
and the checkpoint's facts (candidate.facts) and returns the loads, each one serve --check of the library-defaults
candidate with the in-load scenarios that are meaningful for this checkpoint, and the scenarios left out with why:

| Scenario | Where | Size |
| --- | --- | --- |
| serve --check's items, crit smoke at the run's context and at smoke's lower one, the full-context prompt | every load | the run's context |
| short request, cancel in a prefill | every load | a prompt of CANCEL_TOKENS, within the context |
| 32k prompt | the context holds it | LONG_TOKENS |
| reopen at a lower context and back (elastic ranges, each close's memory) | the first load | serve --check's smoke lower context |
| close and reopen at the same context (each close returns its memory) | the first load | the run's context |
| the engine's lanes at once | the first load | the lanes the engine states |
| prompt-cache follow-up turn | the first load | past the sliding window's ring, within half the context |
| drafting from the draft layer and through the drafter seam | the checkpoint has a draft layer | |
| the companion drafter | the run names one | |
| gate pre-check (crit-eval) | the cut's prepared data | the run's context |
| memory pressure: prompt ladder, smoke at the context | routed experts the full model keeps partly off the device | the full model's share on the device |

One load per quality mode the run asks (a run asks one: its quality_mode). Budget holds the time budget: each
scenario's estimate (the run's previous report where it has one, else DEFAULT_SECONDS) against what is left, so a
scenario that cannot finish in it is left out and says so, and estimate() gives a load's ETA before it starts."""
from dataclasses import dataclass, field
import time

from .scenarios import Sizes

LONG_TOKENS = 32768        # the 32k prompt: past every prompt chunk and step buffer a prefill takes, the first size
#                            where a long prompt's own costs (a pass's spans, a ring's wrap) show
CANCEL_TOKENS = 65536      # a prompt long enough that a cancel CANCEL_AFTER_S in lands inside its prefill
TEMPLATE_TOKENS = 1024     # what a chat template and a short reply add to a prompt: a prompt sized to the context
#                            leaves this much
FOLLOWUP_TOKENS = 4096     # the follow-up's first turn where the model has no sliding window: past a few prompt
#                            chunks, so the prompt cache has a prefill to save
BUDGET_MINUTES = 15        # crit run ... preflight's default time budget, loads included
# A first guess at each scenario's seconds on a cut, until the run's own report has measured them (Budget): what the
# ETA starts from and the budget leaves scenarios out by.
DEFAULT_SECONDS = {"load": 240.0, "short_request": 2.0, "prompt_32k": 20.0, "cancel_in_prefill": 30.0,
                   "prompt_ladder": 30.0, "concurrency": 10.0, "prompt_cache_followup": 10.0,
                   "drafting_draft_layer": 30.0, "drafting_companion": 40.0, "reopen": 60.0, "reopen_same_context": 60.0,
                   "smoke": 120.0,
                   "precheck": 180.0}
EXTRA_SCENARIO = {"short": "short_request", "32k": "prompt_32k", "cancel": "cancel_in_prefill",
                  "ladder": "prompt_ladder", "concurrency": "concurrency", "followup": "prompt_cache_followup",
                  "drafting": "drafting_draft_layer", "companion": "drafting_companion", "lower": "reopen",
                  "same": "reopen_same_context"}


@dataclass
class Load:
    """One load: its name (MODE/PRESSURE), quality mode, the in-load scenarios (extras), their sizes, the share of
    the routed experts its ballast leaves room for (None: no ballast), and whether serve --check runs smoke and the
    gate pre-check."""
    name: str
    mode: str
    extras: set
    sizes: Sizes
    share: float = None
    smoke: bool = True
    precheck: bool = False


@dataclass
class Plan:
    loads: list
    left_out: list = field(default_factory=list)   # [(scenario, why)]


def lower_context(context):
    """The context the first load reopens at: serve --check's smoke lower context (crit.run.served.smoke_lower)."""
    from ..run.served import smoke_lower
    return smoke_lower(context)


def sizes_for(context, found):
    """The scenarios' sizes at the run's context for a checkpoint of facts `found`."""
    room = max(1, context - TEMPLATE_TOKENS)
    window = found.get("sliding_window") or 0
    followup = min(max(FOLLOWUP_TOKENS, 2 * window), max(1, context // 2))
    return Sizes(context=context, long=min(LONG_TOKENS, room), lower=lower_context(context),
                 cancel=min(CANCEL_TOKENS, room), followup=followup, calibrate=True)


def expert_share(placement):
    """The share of the routed experts the whole model keeps on the device at the run's context (crit-eval's
    placement of it on the run's device), or None where the placement says none or nothing."""
    found = (placement or {}).get("bytes") or {}
    experts, on = found.get("experts"), found.get("experts_on_device")
    if not experts or on is None:
        return None
    return min(1.0, on / experts)


def plan_loads(run, found, placement=None, companion=False, data=True):
    """The preflight's Plan for a run (its record) and its checkpoint's facts (candidate.facts). placement: crit-eval's
    placement of the whole model on the run's device at its context (crit.run.recon.placement), for the pressure;
    companion: the run names a companion drafter; data: the cut's crit-eval data is prepared (the gate pre-check)."""
    from ..run.records import quality_mode
    context = run["spec"]["hardware_constraints"]["kv_tokens"]
    mode = quality_mode(run["spec"]["correctness"])
    sizes = sizes_for(context, found)
    left_out = []
    extras = {"short", "cancel", "followup", "concurrency", "same"}
    if context >= LONG_TOKENS + TEMPLATE_TOKENS:
        extras.add("32k")
    else:
        left_out.append(("prompt_32k", f"the run's context ({context:,} tokens) holds no 32k prompt"))
    if sizes.lower < context:
        extras.add("lower")
    else:
        left_out.append(("reopen", f"the run's context ({context:,} tokens) is the lowest serve --check opens at"))
    if found["draft_layer"]:
        extras.add("drafting")
    else:
        left_out.append(("drafting_draft_layer", "the checkpoint stores no draft layer its model plugin reads"))
    if companion:
        extras.add("companion")
    if data:
        extras.add("precheck")
    else:
        left_out.append(("precheck", "no crit-eval data for the cut (crit Eval is not installed)"))
    loads = [Load(f"{mode}/natural", mode, extras, sizes, precheck=data)]
    share = expert_share(placement)
    if not found["routed"]:
        left_out.append(("pressure", "the checkpoint routes no experts"))
    elif share is None:
        left_out.append(("pressure", "the run's placement sizes no expert cache (it names no device file, or Eval "
                                     "cannot size the model): crit run ... recon records it"))
    elif share >= 1.0:
        left_out.append(("pressure", "the whole model's routed experts fit the device at the run's context"))
    else:
        pressed = {"short", "ladder", "cancel", "smoke"} | ({"32k"} & extras)
        loads.append(Load(f"{mode}/pressure", mode, pressed, sizes, share=share, smoke=False))
    return Plan(loads, left_out)


class Budget:
    """The preflight's time budget: `seconds` from its start, scenario estimates from `measured` ({scenario: seconds}
    of the run's previous report, its NAME without the load's prefix) over DEFAULT_SECONDS."""

    def __init__(self, seconds, measured=None, clock=time.monotonic):
        self.seconds, self.measured, self.clock = float(seconds), dict(measured or {}), clock
        self.started = clock()

    def left(self):
        return self.seconds - (self.clock() - self.started)

    def guess(self, name):
        key = "reopen" if name.startswith("reopen_at_") else name
        return float(self.measured.get(name, self.measured.get(key, DEFAULT_SECONDS.get(key, 30.0))))

    def fits(self, name):
        return self.guess(name) <= self.left()

    def describe(self, name):
        return f"about {self.guess(name):.0f} s, {max(0.0, self.left()):.0f} s left of {self.seconds / 60:.0f} min"

    def estimate(self, load):
        """A load's seconds: its load and serve --check's parts, and each in-load scenario."""
        total = self.guess("load") + sum(self.guess(EXTRA_SCENARIO[extra]) for extra in load.extras
                                         if extra in EXTRA_SCENARIO)
        total += (2 * self.guess("smoke") if load.smoke else 0) + ("smoke" in load.extras) * self.guess("smoke")
        return total + (self.guess("precheck") if load.precheck else 0)


def measured(report):
    """{scenario: seconds} of a previous report's items, the load's prefix (MODE/PRESSURE/) taken off; the load's
    own seconds less what ran inside it count as its "load"."""
    found, loads, inside = {}, {}, {}
    for item in (report or {}).get("items", []):
        parts = item["scenario"].split("/", 2)   # MODE/PRESSURE/NAME inside a load, NAME outside one
        prefix, name = ("/".join(parts[:2]), parts[2]) if len(parts) == 3 else ("", item["scenario"])
        seconds = float(item.get("seconds") or 0.0)
        if name == "load":
            loads[prefix] = seconds
            continue
        if name != "ready":
            inside[prefix] = inside.get(prefix, 0.0) + seconds
        if name in DEFAULT_SECONDS or name.startswith("reopen_at_") or name in EXTRA_SCENARIO.values():
            found[name] = max(found.get(name, 0.0), seconds)
        if name.startswith("smoke_"):
            found["smoke"] = found.get("smoke", 0.0) + float(item.get("seconds") or 0.0)
        if name.startswith("precheck/"):
            found["precheck"] = found.get("precheck", 0.0) + float(item.get("seconds") or 0.0)
    if loads:
        found["load"] = max(max(0.0, total - inside.get(prefix, 0.0)) for prefix, total in loads.items())
    return found


def leave_bytes(basis, kv_bytes_per_token, tokens, share):
    """The device bytes a ballast leaves free for the cut at the run's context so that its routed experts meet the
    whole model's pressure: what the cut holds beside its experts at the context (crit-eval's basis of it on the run's
    device: resident weights, fixed state, companion drafters, the reserve and the VRAM headroom, and its KV at
    `tokens`) and `share` of its routed experts."""
    from crit_plugins.memory import kv_need   # Core's one arithmetic of what fits, read the other way
    return int(kv_need(kv_bytes_per_token, tokens, headroom=basis["vram_headroom_bytes"],
                       reserve=basis["reserve_bytes"],
                       held=(basis["resident_weight_bytes"], basis["fixed_state_bytes"],
                             basis.get("companion_drafter_bytes", 0), share * basis["expert_bytes"])))
