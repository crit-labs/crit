"""Ablation: the engine served with its plugins switched off one at a time and on one at a time, so that what each
plugin adds, costs or breaks is measured on the one engine before Eval measures it. Here the variants and the reading
of their results; crit run ... ablate serves and measures them (cli/crit/run/ablate.py)."""
import math

from .assemble import check, normalize
from .manifest import PluginError

# Below this relative change a short bench cannot tell a plugin's effect from run-to-run noise.
NOISE = 0.03
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
    use = dict(selection["use"])
    for slot, plugin_id in selection["use"].items():
        if plugin_id in plugin_ids:
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
        if len(switched) > 1:
            rows += [(f"only:{p}", _without(selection, [q for q in switched if q != p])) for p in switched]
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
    A value that is no JSON is a string. With a registry each is checked like variants(), one the plugins refuse
    carrying its `error`. An in-process A/B of one option is the same engine served twice, as built and with it set,
    measured on the same cases."""
    import json
    selection = normalize(selection, registry)
    if registry is not None and facts is None:
        facts = engine_facts(selection, registry)
    out = []
    for text in assignments:
        options = {plugin_id: dict(given) for plugin_id, given in selection["options"].items()}
        for pair in [part.strip() for part in text.split(";") if part.strip()]:
            key, sep, value = pair.partition("=")
            plugin_id, dot, option = key.strip().rpartition(".")
            if not sep or not dot or not plugin_id or not option:
                raise PluginError(f"--set {pair!r}: give PLUGIN.OPTION=VALUE")
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = value
            options.setdefault(plugin_id, {})[option] = parsed
        variant = {"name": f"set:{text}",
                   "plugins": {"use": dict(selection["use"]), "base": selection["base"], "options": options}}
        if registry is not None:
            try:
                check(variant["plugins"], registry, facts)
            except PluginError as error:
                variant["error"] = str(error)
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


def _change(kpi, value, against):
    """The relative change of `value` over `against`, positive when it is better."""
    if value is None or against is None or against <= 0 or value <= 0:
        return None
    return value / against - 1 if KPIS.get(kpi, kpi not in LOWER) else against / value - 1


def _drafts(variant):
    return (variant["kpis"].get("accepted_tokens_per_round") or 0) > DRAFTING


def _failed(variant):
    return set((variant.get("smoke") or {}).get("failed") or [])


def blame(report_variants, selection, noise=NOISE):
    """What each toggled plugin did, from the measured variants: its effect in the engine (full against without:ID)
    and alone (only:ID against base; full against base when it is the only toggle), with the time per decode round
    where both variants of the pair draft, the smoke failures it brings (failing with it, passing without it), and a
    verdict from its effect in the engine (alone when that is unmeasured): breaks, pays (some KPI better by more than
    `noise`, none worse), slows (some worse, none better), trades (some better, some worse), neutral or
    unmeasured. Where both variants of a pair draft, the verdict of a plugin outside the drafter and verifier slots
    reads the time per decode round in place of decode: a short bench's tokens per round move with what it samples, so
    its decode does too, by more than the plugin's own effect (`judged` names the KPIs the verdict read)."""
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
        for key, (value, against) in (("in_engine", (full, without)), ("alone", (alone, base))):
            if value and against:
                for kpi in KPIS:
                    row[key][kpi] = _change(kpi, value["kpis"][kpi], against["kpis"][kpi])
                if _drafts(value) and _drafts(against):
                    row[key][ROUND] = _change(ROUND, value["kpis"].get(ROUND), against["kpis"].get(ROUND))
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
                return {kpi: c for kpi, c in effect.items() if kpi != "decode_tps" and c is not None}
            return {kpi: c for kpi, c in effect.items() if kpi != ROUND and c is not None}
        read = judged(row["in_engine"]) or judged(row["alone"])
        row["judged"] = sorted(read)
        changes = list(read.values())
        better, worse = any(c > noise for c in changes), any(c < -noise for c in changes)
        row["verdict"] = ("breaks" if row["breaks"] else "trades" if better and worse else "pays" if better else
                          "slows" if worse else "neutral" if changes else "unmeasured")
        out.append(row)
    return out


def settings(report_variants, noise=NOISE):
    """What each option variant (option_variants) did against the full selection: its KPI changes (the time per decode
    round where both draft) and smoke failures it adds, with a verdict like blame()'s."""
    by_name = {v["name"]: v for v in report_variants}
    full = by_name.get("full") if by_name.get("full", {}).get("state") == "measured" else None
    out = []
    for variant in report_variants:
        if not variant["name"].startswith("set:"):
            continue
        row = {"plugin": variant["name"], "in_engine": {}, "alone": {}, "breaks": []}
        if variant.get("state") == "measured" and full:
            for kpi in KPIS:
                row["in_engine"][kpi] = _change(kpi, variant["kpis"][kpi], full["kpis"][kpi])
            if _drafts(variant) and _drafts(full):
                row["in_engine"][ROUND] = _change(ROUND, variant["kpis"].get(ROUND), full["kpis"].get(ROUND))
            row["breaks"] = sorted(_failed(variant) - _failed(full))
        changes = [c for c in row["in_engine"].values() if c is not None]
        better, worse = any(c > noise for c in changes), any(c < -noise for c in changes)
        row["verdict"] = ("breaks" if row["breaks"] or variant.get("state") == "failed" else "trades" if better and worse
                          else "pays" if better else "slows" if worse else "neutral" if changes else "unmeasured")
        out.append(row)
    return out


def describe(row):
    """One line for a blame row."""
    def changes(values):
        shown = [f"{kpi.split('_')[0]} {value:+.1%}" for kpi, value in values.items() if value is not None]
        return ", ".join(shown) or "unmeasured"
    line = f"{row['plugin']}: {row['verdict']}; in the engine {changes(row['in_engine'])}; alone {changes(row['alone'])}"
    if row["breaks"]:
        line += "; fails " + ", ".join(row["breaks"]) + " with it and passes without it"
    return line
