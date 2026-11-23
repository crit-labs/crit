"""Ablation: the engine served with its plugins switched off one at a time and on one at a time, so that what each
plugin adds, costs or breaks is measured on the one engine before Eval measures it. Here the variants and the reading
of their results; crit run ... ablate serves and measures them (cli/crit/run/ablate.py)."""
import math

from .assemble import check, normalize
from .exactness import LEVELS, PHASES
from .manifest import PluginError

# Below this relative change a short bench cannot tell a plugin's effect from run-to-run noise.
NOISE = 0.03
# With alternating rounds (crit run ablate --rounds) two variants are compared round by round, each round's pair
# served minutes apart instead of an ablation apart, so the drift of a shared device between them cancels: a KPI has
# changed when every round's paired change lies on the same side and their median beyond this and beyond the rounds'
# own spread (their highest change less their lowest). Without the spread, four rounds of a KPI that scatters by more
# than this land on one side one time in eight, and the same engine served twice read as faster.
PAIRED_NOISE = 0.005
# The KPIs a variant reports, and whether more is better.
KPIS = {"decode_tps": True, "prefill_tps": True, "ttft_ms": False}
# The time per decode round, compared where both variants draft (more than DRAFTING tokens a round): a plugin outside
# the drafter changes how long a round takes, while a short bench's tokens per round move with what it samples.
ROUND, DRAFTING = "round_ms", 1.05
LOWER = {ROUND}
# The slots whose plugins choose the drafts: their effect on the tokens a round keeps is theirs, so their verdict reads
# decode even where both variants draft.
DRAFT_SLOTS = ("drafter", "verifier")


def toggles(selection):
    """The plugins an ablation switches: those in use that are not their slots' plain choice, in selection order."""
    selection = normalize(selection)
    found = []
    for slot, plugin_id in selection["use"].items():
        if plugin_id is not None and plugin_id != selection["base"][slot] and plugin_id not in found:
            found.append(plugin_id)
    return found


def _without(selection, plugin_ids):
    """The selection's use with plugin_ids at their slots' base. Drafting is one unit: without the plugin of one of
    DRAFT_SLOTS the others go to their base too (nothing verifies without a drafter, nothing drafts without a
    verifier), so the variant is a plain engine Core's load accepts and not a selection that drafts with no drafter."""
    use = dict(selection["use"])
    removed = {slot for slot, plugin_id in selection["use"].items() if plugin_id in plugin_ids}
    if removed & set(DRAFT_SLOTS):
        removed |= {slot for slot in DRAFT_SLOTS if slot in use}
    for slot in removed:
        use[slot] = selection["base"][slot]
    return use


def engine_facts(selection, registry):
    """The facts the engine itself provides, as the candidate's own selection shows them: what its plugins need and
    none of them provides. The candidate serves as built, so its engine supplies these. Of a need with alternatives
    that no plugin meets, every alternative is taken: the engine meets at least one."""
    selection = normalize(selection, registry)
    used = [plugin_id for plugin_id in dict.fromkeys(selection["use"].values()) if plugin_id is not None]
    providers = {}
    for plugin_id in used:
        for fact in registry[plugin_id]["provides"]:
            providers.setdefault(fact, set()).add(plugin_id)
    found = []
    for plugin_id in used:
        for alternatives in registry[plugin_id]["needs"]:
            if not any(providers.get(fact, set()) - {plugin_id} for fact in alternatives):
                found += [fact for fact in alternatives if fact not in found]
    return found


def variants(selection, registry=None, facts=None, names=None):
    """The selections an ablation serves: full (as built), base (every toggle off), without:ID for each toggle and
    only:ID (base with that one on) when there are two or more. With a registry each is checked, and one the plugins
    refuse carries its `error` instead of being served. facts: the engine's own facts (default engine_facts, those
    the full selection leaves to the engine). names: only the variants of these names."""
    selection = normalize(selection, registry)
    if registry is not None and facts is None:
        facts = engine_facts(selection, registry)
    switched = toggles(selection)
    rows = [("full", selection["use"])]
    if switched:
        rows.append(("base", _without(selection, switched)))
        rows += [(f"without:{p}", _without(selection, [p])) for p in switched]
        if len(switched) > 1:   # only:ID keeps its drafting unit whole (a drafter with its verifier)
            drafting = {selection["use"][slot] for slot in DRAFT_SLOTS if selection["use"].get(slot) is not None}
            rows += [(f"only:{p}", _without(selection, [q for q in switched if q != p
                                                         and not (p in drafting and q in drafting)]))
                     for p in switched]
    out, seen = [], {}
    for name, use in rows:
        key = tuple(sorted(use.items(), key=lambda item: item[0]))
        if key in seen:   # the same engine as an earlier variant (base is without:ID when ID is the only toggle)
            seen[key].setdefault("also", []).append(name)
            continue
        if names and name not in names:
            continue
        # A variant carries the options of the plugins it can run, so the engine's own assembly accepts it.
        runs = {p for p in [*use.values(), *selection["base"].values()] if p is not None}
        options = {p: given for p, given in selection["options"].items() if p in runs}
        variant = {"name": name, "plugins": {"use": use, "base": selection["base"], "options": options}}
        seen[key] = variant
        if registry is not None:
            try:
                check(variant["plugins"], registry, facts)
            except PluginError as error:
                variant["error"] = str(error)
        out.append(variant)
    if names:
        unknown = sorted(set(names) - {n for v in out for n in [v["name"], *v.get("also", [])]})
        if unknown:
            raise PluginError(f"no variant {', '.join(unknown)}: the variants are "
                              + ", ".join(name for name, _ in rows))
    return out


def option_variants(selection, assignments, registry=None, facts=None):
    """Variants of the full selection with plugin options set: each of `assignments` is one variant, a text of
    PLUGIN.OPTION=JSON pairs joined by ";" (mtp-decode.override=0;attention-state.fresh_mib=64), named "set:" and that text.
    A value that is no JSON is a string, and so is one an option declared as a string or a choice of strings takes as
    written (tile_rows=0 is the choice "0"). With a registry each is checked like variants(), one the plugins refuse
    carrying its `error`. An in-process A/B of one option is the same engine served twice, as built and with it set,
    measured on the same cases.
    exactness=LEVEL, exactness.decode=LEVEL and exactness.prompt=LEVEL are no plugin's option: they set the level the
    engine asks of its rows (the configuration's exactness, crit_plugins.exactness), which the variant carries as
    `exactness` ({phase: level}) for the engine to be served with, so what a level costs or gains is measured like
    an option."""
    import json
    selection = normalize(selection, registry)
    if registry is not None and facts is None:
        facts = engine_facts(selection, registry)
    out = []
    for text in assignments:
        options = {plugin_id: dict(given) for plugin_id, given in selection["options"].items()}
        levels = {}
        for pair in [part.strip() for part in text.split(";") if part.strip()]:
            key, sep, value = pair.partition("=")
            plugin_id, dot, option = key.strip().rpartition(".")
            if sep and (key.strip() == "exactness" or (plugin_id == "exactness" and option in PHASES)):
                if value.strip() not in LEVELS:
                    raise PluginError(f"--set {pair!r}: exactness is one of {', '.join(reversed(LEVELS))}")
                levels.update({phase: value.strip() for phase in (PHASES if not dot else (option,))})
                continue
            if not sep or not dot or not plugin_id or not option:
                raise PluginError(f"--set {pair!r}: give PLUGIN.OPTION=VALUE")
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = value
            # A string option, or a choice among strings, takes the text as written: tile_rows=0 is the choice "0".
            spec = (registry[plugin_id].get("options") or {}).get(option) or {}                 if registry is not None and plugin_id in registry else {}
            if not isinstance(parsed, str) and (spec.get("type") == "str" or (
                    spec.get("type") == "choice" and parsed not in spec["choices"] and value.strip() in spec["choices"])):
                parsed = value.strip()
            options.setdefault(plugin_id, {})[option] = parsed
        variant = {"name": f"set:{text}",
                   "plugins": {"use": dict(selection["use"]), "base": selection["base"], "options": options},
                   **({"exactness": levels} if levels else {})}
        if registry is not None:
            try:
                check(variant["plugins"], registry, facts)
            except PluginError as error:
                variant["error"] = str(error)
            # An option set that approximates the model (its manifest's approximates) changes what the engine
            # computes: speed alone does not judge it, its drift against the reference scores does (Eval's quality).
            from .assemble import approximations
            known = {p: o for p, o in options.items() if p in registry}
            was = {(a["plugin"], a["option"]) for a in approximations(registry, {p: o for p, o in selection["options"]
                                                                                  .items() if p in registry})}
            new = [a for a in approximations(registry, known) if (a["plugin"], a["option"]) not in was]
            if new:
                variant["approximates"] = new
        out.append(variant)
    return out


def _positive(value):
    return isinstance(value, (int, float)) and value > 0


def _mean(values):
    """The geometric mean, None when a value is missing."""
    if not values or not all(_positive(v) for v in values):
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summary(cases):
    """A variant's KPIs over the run's cases: the geometric mean of each case's median, None where a case lacks it,
    with its tokens per round and the time per decode round they give (1000 x tokens per round / decode_tps)."""
    out = {kpi: _mean([case.get(kpi) for case in cases.values()]) for kpi in KPIS}
    out["accepted_tokens_per_round"] = _mean([case.get("accepted_tokens_per_round") for case in cases.values()])
    out[ROUND] = _mean([1000 * case["accepted_tokens_per_round"] / case["decode_tps"]
                        if _positive(case.get("accepted_tokens_per_round")) and _positive(case.get("decode_tps"))
                        else None for case in cases.values()])
    return out


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def combine(rounds):
    """One variant's rows from alternating rounds (crit run ablate --rounds, one row a round, the first with its smoke
    items) as one row: its state (measured where every round measured it, else the first other round's, with its
    detail), each KPI and each case's figure the median over the measured rounds, and `rounds` each round's state, load
    time and KPIs, which blame() pairs round by round. A single round's row is returned as it is."""
    if len(rounds) == 1:
        return rounds[0]
    first = rounds[0]
    out = {key: value for key, value in first.items() if key not in ("state", "detail", "kpis", "cases", "load_ms")}
    out["rounds"] = [{key: row[key] for key in ("state", "detail", "load_ms", "kpis") if key in row} for row in rounds]
    measured = [row for row in rounds if row.get("state") == "measured"]
    other = next((row for row in rounds if row.get("state") != "measured"), None)
    out["state"] = "measured" if other is None else other.get("state")
    if other is not None and other.get("detail"):
        out["detail"] = other["detail"]
    if measured:
        def middle(values):
            values = [v for v in values if isinstance(v, (int, float))]
            return _median(values) if values else None
        out["load_ms"] = middle([row.get("load_ms") for row in measured])
        out["kpis"] = {kpi: middle([row["kpis"].get(kpi) for row in measured])
                       for kpi in dict.fromkeys(k for row in measured for k in row["kpis"])}
        cases = [row.get("cases") or {} for row in measured]
        out["cases"] = {case: {name: middle([(found.get(case) or {}).get(name) for found in cases])
                               for name in dict.fromkeys(n for found in cases for n in (found.get(case) or {}))}
                        for case in dict.fromkeys(c for found in cases for c in found)}
    return out


def _change(kpi, value, against):
    """The relative change of `value` over `against`, positive when it is better."""
    if value is None or against is None or against <= 0 or value <= 0:
        return None
    return value / against - 1 if KPIS.get(kpi, kpi not in LOWER) else against / value - 1


def _drafts(variant):
    return (variant["kpis"].get("accepted_tokens_per_round") or 0) > DRAFTING


def _failed(variant):
    return set((variant.get("smoke") or {}).get("failed") or [])


def _compare(value, against):
    """KPI -> (change, lowest, highest, changes) of `value` against `against`: the relative change of their KPIs and,
    where both were measured in alternating rounds (combine()), the median of each round's paired change, the lowest
    and highest of them and the list. Without rounds lowest and highest are the change itself; with fewer than two
    rounds both measured, None (no change resolved). The time per decode round is compared where both draft."""
    out = {}
    for kpi in list(KPIS) + ([ROUND] if _drafts(value) and _drafts(against) else []):
        if value.get("rounds") and against.get("rounds"):
            changes = [_change(kpi, a["kpis"].get(kpi), b["kpis"].get(kpi))
                       for a, b in zip(value["rounds"], against["rounds"])
                       if a.get("state") == "measured" and b.get("state") == "measured"]
            changes = [c for c in changes if c is not None]
            if len(changes) >= 2:
                out[kpi] = (_median(changes), min(changes), max(changes), changes)
            else:
                out[kpi] = (changes[0] if changes else None, None, None, changes)
        else:
            change = _change(kpi, value["kpis"].get(kpi), against["kpis"].get(kpi))
            out[kpi] = (change, change, change, None)
    return out


def _verdict(effects, noise, broken=False):
    """breaks, trades, pays, slows, neutral or unmeasured from `effects` (_compare's, the KPIs judged): a KPI is better
    when its change is above `noise` and its lowest change above 0, worse when its change is below -noise and its
    highest below 0 (where paired: the rounds' median, every round on that side, and the median beyond the rounds'
    spread, their highest change less their lowest; a single pass has no spread)."""
    def bound(lo, hi):
        return max(noise, hi - lo) if lo is not None and hi is not None else noise
    better = any(lo is not None and lo > 0 and change > bound(lo, hi) for change, lo, hi, _ in effects.values())
    worse = any(hi is not None and hi < 0 and change < -bound(lo, hi) for change, lo, hi, _ in effects.values())
    measured = any(change is not None for change, _, _, _ in effects.values())
    return ("breaks" if broken else "trades" if better and worse else "pays" if better else "slows" if worse else
            "neutral" if measured else "unmeasured")


def blame(report_variants, selection, noise=NOISE):
    """What each toggled plugin did, from the measured variants: its effect in the engine (full against without:ID)
    and alone (only:ID against base; full against base when it is the only toggle), with the time per decode round
    where both variants of the pair draft, the smoke failures it brings (failing with it, passing without it), and a
    verdict from its effect in the engine (alone when that is unmeasured): breaks, pays (some KPI better by more than
    `noise`, none worse), slows (some worse, none better), trades (some better, some worse), neutral or
    unmeasured. Where both variants of a pair draft, the verdict of a plugin outside the drafter and verifier slots
    reads the time per decode round in place of decode: a short bench's tokens per round move with what it samples, so
    its decode does too, by more than the plugin's own effect (`judged` names the KPIs the verdict read). Variants
    measured in alternating rounds (combine()) are compared round by round: each change is the median of the rounds'
    paired changes, `paired` holds them, and a KPI is better or worse only where every round's change lies on that
    side and their median beyond `noise`."""
    by_name = {name: v for v in report_variants for name in [v["name"], *v.get("also", [])]}
    measured = lambda name: by_name.get(name) if by_name.get(name, {}).get("state") == "measured" else None
    switched = toggles(selection)
    drafting = {plugin_id for slot, plugin_id in normalize(selection)["use"].items() if slot in DRAFT_SLOTS}
    full, base = measured("full"), measured("base")
    out = []
    for plugin_id in switched:
        without = measured(f"without:{plugin_id}")
        alone = measured(f"only:{plugin_id}") if len(switched) > 1 else full
        row = {"plugin": plugin_id, "in_engine": {}, "alone": {}, "breaks": []}
        effects = {"in_engine": {}, "alone": {}}
        for key, (value, against) in (("in_engine", (full, without)), ("alone", (alone, base))):
            if value and against:
                effects[key] = _compare(value, against)
                row[key] = {kpi: effect[0] for kpi, effect in effects[key].items()}
        _paired_rows(row, effects)
        breaks = set()
        if full and without:
            breaks |= _failed(full) - _failed(without)
        if alone and base:
            breaks |= _failed(alone) - _failed(base)
        if by_name.get(f"without:{plugin_id}", {}).get("state") == "measured" and full is None \
                and by_name.get("full", {}).get("state") == "failed":
            breaks.add("load")   # the engine loads without it and not with it
        row["breaks"] = sorted(breaks)
        def judged(effect):
            if ROUND in effect and plugin_id not in drafting:
                return {kpi: e for kpi, e in effect.items() if kpi != "decode_tps" and e[0] is not None}
            return {kpi: e for kpi, e in effect.items() if kpi != ROUND and e[0] is not None}
        read = judged(effects["in_engine"]) or judged(effects["alone"])
        row["judged"] = sorted(read)
        row["verdict"] = _verdict(read, noise, bool(row["breaks"]))
        out.append(row)
    return out


def _paired_rows(row, effects):
    """Where the comparisons were paired over rounds, each KPI's change in every round (`paired`: {in_engine, alone}
    -> {kpi: [changes]})."""
    paired = {key: {kpi: e[3] for kpi, e in found.items() if e[3] is not None} for key, found in effects.items()}
    if any(paired.values()):
        row["paired"] = paired


def settings(report_variants, noise=NOISE):
    """What each option variant (option_variants) did against the full selection: its KPI changes (the time per decode
    round where both draft, paired round by round where both were measured in rounds) and smoke failures it adds, with
    a verdict like blame()'s."""
    by_name = {v["name"]: v for v in report_variants}
    full = by_name.get("full") if by_name.get("full", {}).get("state") == "measured" else None
    out = []
    for variant in report_variants:
        if not variant["name"].startswith("set:"):
            continue
        row = {"plugin": variant["name"], "in_engine": {}, "alone": {}, "breaks": []}
        effects = {"in_engine": {}}
        if variant.get("state") == "measured" and full:
            effects["in_engine"] = _compare(variant, full)
            row["in_engine"] = {kpi: effect[0] for kpi, effect in effects["in_engine"].items()}
            row["breaks"] = sorted(_failed(variant) - _failed(full))
        _paired_rows(row, effects)
        row["verdict"] = _verdict(effects["in_engine"], noise, bool(row["breaks"]) or variant.get("state") == "failed")
        out.append(row)
    return out


def describe(row):
    """One line for a blame row."""
    def changes(values, paired):
        shown = [f"{kpi.split('_')[0]} {value:+.1%}" + (f" (rounds {min(paired[kpi]):+.1%} to {max(paired[kpi]):+.1%})"
                                                        if len(paired.get(kpi) or ()) > 1 else "")
                 for kpi, value in values.items() if value is not None]
        return ", ".join(shown) or "unmeasured"
    paired = row.get("paired") or {}
    engine, alone = (changes(row[key], paired.get(key) or {}) for key in ("in_engine", "alone"))
    line = f"{row['plugin']}: {row['verdict']}; in the engine {engine}; alone {alone}"
    if row["breaks"]:
        line += "; fails " + ", ".join(row["breaks"]) + " with it and passes without it"
    return line
