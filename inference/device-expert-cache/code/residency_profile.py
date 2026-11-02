"""Residency profiles: which experts a device expert cache holds before its first request.

A profile is the engine's own file in its package: the cache's geometry, the keys
(group * experts + expert) to preload in order (device_cache.h enqueue_preload,
ClockDirectory.preload), the corpus its routes came from and a held-out check.
Routing depends on content, so a profile is a content-dependent choice: build it from
router logs over a general corpus unrelated to Eval (never Eval's prompts, pinned
continuation, held-out prompts or look-alike text), and check it on held-out traces in
several languages and domains. The cache keeps adapting from the profile; it is a
starting point, not a frozen residency.

A trace is a sequence of (group, experts) plans, the same shape ClockDirectory.plan
and replay take: one plan per routed layer of each decoded token.
"""
from collections import Counter
import json

from device_cache import ClockDirectory, simulate

KIND = "device-expert-cache-profile"


def route_counts(trace, groups, experts):
    """{key: routes} over a trace; rejects routes outside the geometry."""
    counts = Counter()
    for group, routed in trace:
        for e in routed:
            if not (0 <= group < groups and 0 <= e < experts):
                raise ValueError("route outside the profile's geometry")
            counts[group * experts + e] += 1
    return counts


def choose_keys(counts, slots):
    """The `slots` most routed keys, most routed first, ties by key: the preload order."""
    return [k for k, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:slots]]


def held_out_check(keys, held_out, slots, groups, experts):
    """{name: {"cold", "profiled"}} hit rates per held-out trace: a cold clock directory
    against one preloaded with `keys`, the same replacement afterwards."""
    if len(held_out) < 2:
        raise ValueError("a held-out check needs traces in several languages or domains")
    out = {}
    for name, trace in held_out.items():
        trace = list(trace)
        cold = simulate(ClockDirectory(slots, groups, experts), trace)
        warm = ClockDirectory(slots, groups, experts)
        warm.preload(keys)
        out[name] = {"cold": cold, "profiled": simulate(warm, trace)}
    return out


def build_profile(trace, groups, experts, slots, corpus, held_out):
    """A profile from a tuning trace over `corpus` (a description of the general text it
    came from: source, languages, size), checked on `held_out` ({name: trace})."""
    if not isinstance(corpus, str) or not corpus.strip():
        raise ValueError("state the corpus the routes came from")
    if slots < 1 or groups < 1 or experts < 1:
        raise ValueError("profile geometry must be positive")
    keys = choose_keys(route_counts(trace, groups, experts), slots)
    return {"kind": KIND, "groups": groups, "experts": experts, "slots": slots, "keys": keys,
            "corpus": corpus, "held_out": held_out_check(keys, held_out, slots, groups, experts)}


def write_profile(path, profile):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=1)


def load_profile(path, groups, experts, slots):
    """The keys to preload, checked against the engine's geometry. A profile for fewer
    slots fills only its keys; one for more is cut to the engine's slots."""
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    if profile.get("kind") != KIND:
        raise ValueError("not a device expert cache profile")
    if (profile["groups"], profile["experts"]) != (groups, experts):
        raise ValueError("profile built for another routing geometry")
    keys = profile["keys"]
    if any(type(k) is not int or not 0 <= k < groups * experts for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("profile keys must be distinct keys of the geometry")
    return keys[:slots]
