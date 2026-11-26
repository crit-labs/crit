"""Which of crit's kernels a kernel trace's time went to: each kernel.json's `launches` names the kernels it launches
as a trace shows them, so a sweep starts where the engine spends its time and skips a kernel that launches nothing.

A trace (crit eval kernels, crit run ... profile) names each launch by its demangled name. A name is read without its
template arguments and without anonymous namespaces (crit::kernels::mma_rows::(anonymous namespace)::mma_kernel<2, 0>
is crit::kernels::mma_rows::mma_kernel). An entry of `launches` ending in "::" takes every name under that
namespace; any other entry is a name as written (a Triton kernel's function, a CUDA kernel outside crit's
namespaces). A name two kernels declare is ambiguous, and a name none declares is a library's or a plugin's own."""
import re

ANONYMOUS = "(anonymous namespace)::"


def plain(name):
    """A traced kernel name without anonymous namespaces and without template arguments (from the first '<' or '(')."""
    name = name.replace(ANONYMOUS, "")
    cut = re.search(r"[<(]", name)
    return (name[:cut.start()] if cut else name).strip()


def declared():
    """{kernel id: its launches} of every kernel of crit's (crit_kernels.build.manifests)."""
    from .build import manifests
    return {kernel: list(manifest.get("launches") or []) for kernel, (_, manifest) in manifests().items()}


def owners(name, launches):
    """The kernel ids whose launches take `name` (a traced name)."""
    found = plain(name)
    return sorted(kernel for kernel, entries in launches.items()
                  if any(found.startswith(e) if e.endswith("::") else found == e for e in entries))


def rank(trace, launches=None):
    """crit's kernels by the share of the trace's kernel time they took, per phase and over all of it:
    {"kernels": [{kernel, ms, share, phases: {phase: share}}] (most first), "absent": [kernels that launched nothing],
    "ambiguous": [{name, kernels, ms}], "other": [{name, ms, share}] (the largest names no kernel declares)}.
    `trace` is a kernels report: {"phases": {phase: {"kernels": [{kernel, ms}]}}}."""
    launches = declared() if launches is None else launches
    total = 0.0
    by_kernel, phases, ambiguous, other = {}, {}, {}, {}
    for phase, found in (trace.get("phases") or {}).items():
        rows = found.get("kernels") or []
        phase_ms = sum(float(row.get("ms") or 0) for row in rows) or 1.0
        for row in rows:
            ms = float(row.get("ms") or 0)
            total += ms
            who = owners(row["kernel"], launches)
            if len(who) == 1:
                by_kernel[who[0]] = by_kernel.get(who[0], 0.0) + ms
                share = phases.setdefault(who[0], {})
                share[phase] = share.get(phase, 0.0) + ms / phase_ms
            elif who:
                entry = ambiguous.setdefault(plain(row["kernel"]), {"name": plain(row["kernel"]), "kernels": who,
                                                                    "ms": 0.0})
                entry["ms"] += ms
            else:
                other[plain(row["kernel"])] = other.get(plain(row["kernel"]), 0.0) + ms
    total = total or 1.0
    kernels = [{"kernel": kernel, "ms": round(ms, 3), "share": round(ms / total, 4),
                "phases": {phase: round(share, 4) for phase, share in sorted(phases[kernel].items())}}
               for kernel, ms in sorted(by_kernel.items(), key=lambda item: -item[1])]
    return {"kernels": kernels,
            "absent": sorted(kernel for kernel, entries in launches.items() if entries and kernel not in by_kernel),
            "ambiguous": sorted(({**e, "ms": round(e["ms"], 3)} for e in ambiguous.values()), key=lambda e: -e["ms"]),
            "other": [{"name": name, "ms": round(ms, 3), "share": round(ms / total, 4)}
                      for name, ms in sorted(other.items(), key=lambda item: -item[1])[:12]]}


def render(found):
    lines = ["crit's kernels by the trace's kernel time (sweep from the top; a kernel below a few percent pays little):"]
    for row in found["kernels"]:
        lines.append(f"  {row['kernel']:<24} {row['share']:6.1%}  " + ", ".join(
            f"{phase} {share:.1%}" for phase, share in row["phases"].items()))
    if found["ambiguous"]:
        lines.append("declared by more than one kernel: " + "; ".join(
            f"{e['name']} ({', '.join(e['kernels'])}, {e['ms']:,.0f} ms)" for e in found["ambiguous"]))
    if found["other"]:
        lines.append("not crit's kernels (libraries, a plugin's own): " + "; ".join(
            f"{e['name']} {e['share']:.1%}" for e in found["other"][:6]))
    if found["absent"]:
        lines.append("launched nothing in this trace (no sweep pays): " + ", ".join(found["absent"]))
    return "\n".join(lines)
