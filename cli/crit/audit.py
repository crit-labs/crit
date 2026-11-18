"""What an engine's device memory really holds: `crit plugins memory`.

It loads a package's engine in process, serves a short workload of its own and reads the device before and after
(crit_plugins.audit): every buffer by the plugin that holds it, with the bytes the workload wrote, how far into it
the writes reach and what stayed zero; each captured graph's pool by the code whose allocations fill it; each
elastic range and, for a paged one, the units in use; and each step's allocations and transient peak. A buffer
nothing writes (dead) or one written in a small part (oversized) shows by itself, with no knowledge of the engine.

The workload is two prompt sets, so that scratch shows as written twice with other contents: each a long prompt
(--prompt-tokens) and three short chats, two sampled and one greedy (--reply-tokens each), one request at a time.
Its text is made here from two word lists and is none of crit Eval's. The device is read as a greedy chat ends
before the sets and as each set's last chat ends, each time before the engine closes the request's sequence, so
what a request keeps while it runs is seen in place. The first greedy chat runs again after the sets and the two
replies are compared, since the audit writes a pattern into the bytes Core knows to be scratch (a captured graph's
temporaries, growing buffers) and an engine on exact paths must not notice.

The verdicts hold for that workload: an engine that serves several requests at once, or longer prompts, writes
more. The report says what ran."""
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

GIB, MIB = 1 << 30, 1 << 20
# The two prompt sets' words: sentences are drawn from them by a seeded generator, so a set is the same text every
# run and the two share none.
WORDS = (
    {"who": ("the night shift", "an inspector", "the pump crew", "a contractor", "the district engineer"),
     "did": ("logged", "replaced", "measured", "sealed", "recalibrated", "flushed"),
     "what": ("the inlet valve", "a pressure gauge", "the third filter bed", "a worn gasket", "the chlorine feed",
              "the overflow weir"),
     "where": ("at the north station", "below the reservoir", "in the old gallery", "beside the river intake"),
     "ask": ("Summarise what was done at each place and what still needs a second visit.",
             "Which three repairs would you schedule first, and why?",
             "Write a short handover note for the next shift.",
             "List the parts that were replaced, grouped by station.")},
    {"who": ("the quartermaster", "a deckhand", "the second mate", "a customs clerk", "the ship's cook"),
     "did": ("counted", "stowed", "weighed", "relabelled", "condemned", "traded"),
     "what": ("forty sacks of rice", "a crate of lamp oil", "the spare canvas", "six barrels of salt pork",
              "a coil of tarred rope", "the citrus stores"),
     "where": ("in the forward hold", "under the galley", "on the quarterdeck", "against the stern bulkhead"),
     "ask": ("Draw up the inventory as a table with one line per kind of goods.",
             "What would run out first on a ten week passage, and what would you buy in port?",
             "Explain to a new deckhand how the hold is organised.",
             "Write the entry for the ship's log in four sentences.")},
)


def passage(words, sentences, seed):
    draw = random.Random(seed)
    return " ".join(f"On day {draw.randrange(1, 400)} {draw.choice(words['who'])} {draw.choice(words['did'])} "
                    f"{draw.choice(words['what'])} {draw.choice(words['where'])}." for _ in range(sentences))


def prompts(encode, prompt_tokens):
    """The workload's requests as token ids: per prompt set {"long": ids, "chats": [ids] * 3}. encode(text) gives a
    user message's prompt ids."""
    found = []
    for index, words in enumerate(WORDS):
        sentences = max(1, prompt_tokens // 12)
        for _ in range(4):   # sized by the tokenizer's own count
            ids = encode(passage(words, sentences, index) + " " + words["ask"][0])
            if len(ids) <= prompt_tokens or sentences == 1:
                break
            sentences = max(1, min(sentences - 1, sentences * prompt_tokens // len(ids)))
        found.append({"long": ids, "chats": [encode(passage(words, 2, 100 + index * 10 + i) + " " + words["ask"][1 + i])
                                             for i in range(3)]})
    return found


def encoder(model, configuration):
    """A user message's prompt ids with the model's chat template (and the configuration's chat_template_kwargs),
    the text alone where the model renders none."""
    from crit_core.model import open_model
    from crit_core.text import TextError, load_model_text
    text = load_model_text(open_model(Path(model)).files, clock=None)
    variables = dict(configuration.get("chat_template_kwargs") or {})

    def encode(content):
        try:
            return [int(t) for t in text.encode_chat([{"role": "user", "content": content}],
                                                     add_generation_prompt=True, **variables)]
        except TextError:
            return [int(t) for t in text.tokenizer.encode(content, add_special_tokens=True)]
    return encode


def audit(package, model, context=None, prompt_tokens=2048, reply_tokens=96, least=MIB, mark=True, log=None,
          encode=None):
    """Load the package's engine for `model`, serve the workload and return crit_plugins.audit's report with
    "workload": {"sets", "prompt_tokens": [each set's long prompt], "reply_tokens", "requests", "at_once": 1,
    "repeated": whether the greedy chat's reply was the same before the first reading and after the last,
    "replies": a hash of each greedy reply's tokens in the order served (the chat, each set's long prompt and
    chat, the chat again), which two audits of one engine on exact paths share, "seconds"}, and "telemetry": the
    engine's own (its plugins' and Core's counters) after the workload. mark False writes no pattern into the scratch bytes; encode(text) -> prompt ids replaces the
    model's tokenizer and chat template (a checkpoint without them)."""
    from crit_core.engine import load, read_configuration
    from crit_plugins.audit import Audit
    say = log or (lambda text: None)
    configuration = read_configuration(package)
    work = prompts(encode or encoder(model, configuration), prompt_tokens)
    greedy = {"temperature": 0.0}
    found = Audit(configuration.get("device") or "cuda")
    engine = None
    try:
        say("loading the engine")
        engine = load(package, model, configuration, context_tokens=context)
        found.attach(engine)
        started, requests = time.time(), 0
        replies = []

        def reply(ids, tokens):
            found, _ = engine.generate(ids, tokens, stop_tokens=[], sampling=greedy)
            replies.append(hashlib.sha1(str([int(token) for token in found]).encode()).hexdigest()[:8])
            return found
        found.read("load")   # the engine takes each reading as the next request ends: here the first chat's
        if mark:
            found.mark()
        reference = reply(work[0]["chats"][0], reply_tokens)
        for index, (label, one) in enumerate(zip(("first", "second"), work)):
            say(f"prompt set {index + 1}: {len(one['long']):,} prompt tokens, then three chats")
            reply(one["long"], 16)
            for seed, ids in enumerate(one["chats"][:2]):
                engine.generate(ids, reply_tokens, stop_tokens=[],
                                sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7 + seed})
            found.read(label)
            reply(one["chats"][2], reply_tokens)
            requests += 4
        again = reply(work[0]["chats"][0], reply_tokens)
        report = found.report(least)
        report["workload"] = {"sets": len(work), "prompt_tokens": [len(one["long"]) for one in work],
                              "reply_tokens": reply_tokens, "requests": requests + 2, "at_once": 1,
                              "repeated": list(reference) == list(again), "replies": replies,
                              "seconds": round(time.time() - started, 1)}
        # What the engine's plugins and Core counted over the load and the workload (hits, graphs, steps): read
        # beside the buffers, it says which path the requests took.
        report["telemetry"] = json.loads(json.dumps(engine.telemetry(), default=str))
        return report
    finally:
        found.close()
        if engine is not None:
            engine.close()


def size(nbytes):
    return f"{nbytes / GIB:.2f} GiB" if nbytes >= GIB else f"{nbytes / MIB:.1f} MiB"


def small(nbytes):
    return size(nbytes) if nbytes >= MIB // 10 else f"{nbytes / 1024:.1f} KiB" if nbytes >= 1024 else f"{nbytes} B"


def span(values):
    return f"{min(values):,}" if min(values) == max(values) else f"{min(values):,} to {max(values):,}"


def _grouped(buffers):
    """Buffers held alike (one name without its indices, one verdict) as one row."""
    from crit_plugins.audit import pattern
    groups = defaultdict(list)
    for found in buffers:
        groups[(found["plugin"] or "-", pattern(found["name"]), found["verdict"], found["pool"])].append(found)
    rows = []
    for (plugin, name, verdict, pool), members in groups.items():
        total = lambda key: sum(member.get(key, 0) for member in members)
        name = members[0]["name"] if len(members) == 1 else name   # one buffer: its own name, keys and all
        rows.append({"plugin": plugin, "name": name + (f" (in {pool})" if pool else ""), "verdict": verdict,
                     "count": len(members), "bytes": total("bytes"), "written": total("written"), "zero": total("zero"),
                     "reach": max(member.get("reach", 0) for member in members),
                     "largest": max(member["bytes"] for member in members),
                     "shape": members[0]["shape"] if len(members) == 1 else None,
                     "certain": all(member["certain"] for member in members)})
    return sorted(rows, key=lambda row: (-row["zero"], -row["bytes"]))


def _rows(lines, rows, limit):
    for row in rows[:limit]:
        what = row["name"] + (f" x{row['count']}" if row["count"] > 1 else
                              f" {row['shape']}" if row["shape"] else "")
        reach = f", writes reach {size(row['reach'])}" + (" of the largest" if row["count"] > 1 else "") \
            if row["written"] and row["reach"] < row["largest"] else ""
        lines.append(f"  {row['plugin']:<26} {size(row['bytes']):>11}  written {size(row['written']):>11}, zero "
                     f"{size(row['zero']):>11}{reach}  {what[:150]}")
    if len(rows) > limit:
        lines.append(f"  ... and {len(rows) - limit} more rows ({size(sum(row['bytes'] for row in rows[limit:]))})")


def render(report, limit=24):
    """The report as text for a reader: what is unused first."""
    memory, work = report["memory"], report["workload"]
    elastic = sum(found["backed"] for found in memory["elastic"].values())
    lines = [f"{memory['device']}: {size(memory['total'])}, free {size(memory['free'])}; torch holds "
             f"{size(memory['torch']['reserved'])} ({size(memory['torch']['allocated'])} allocated), elastic ranges "
             f"{size(elastic)}, everything else {size(memory['other'])}",
             f"workload: {work['sets']} prompt sets, each a prompt of {' and '.join(f'{n:,}' for n in work['prompt_tokens'])} "
             f"tokens and three chats of {work['reply_tokens']} reply tokens, one request at a time "
             f"({work['requests']} requests, {work['seconds']:.0f} s)",
             "the greedy chat repeated after the workload gave "
             + ("the same reply" if work["repeated"] else "ANOTHER reply: the engine's replies depend on what ran "
                "before (cache state), or something reads a graph's temporary before writing it")
             + f"; greedy replies {' '.join(work.get('replies') or [])}",
             f"readings: {', '.join(report['readings'])}; the audit held {size(report['audit_bytes'])} itself and "
             f"marked {size(report['marked_bytes'])} of scratch", ""]
    rows = _grouped(report["buffers"])
    totals = report["totals"]
    for verdict, title in (
            ("dead", f"Dead: all zero at every reading, never written ({size(totals['dead'])} with the small ones)"),
            ("oversized", f"Oversized: a part never written and all zero ({size(totals['oversized'])} unwritten with "
                          "the small ones)"),
            ("partly written", "Partly written, the rest unchanged and not zero (not certain: unused room, or state "
                               "the load set)"),
            ("written", "Written: scratch and state in use"),
            ("new", "New: first held at the last reading (made while the engine served)")):
        found = [row for row in rows if row["verdict"] == verdict]
        if found:
            lines.append(title)
            _rows(lines, found, limit)
            lines.append("")
    lines.append("By plugin (every buffer, the small ones too; constant: weights, tables, or what only the load wrote)")
    for plugin, held in sorted(report["plugins"].items(), key=lambda item: -item[1].get("bytes", 0)):
        lines.append(f"  {plugin or 'held by no plugin object':<26} {size(held.get('bytes', 0)):>11} in "
                     f"{held.get('buffers', 0):>5} buffers: written {size(held.get('written', 0))}, zero "
                     f"{size(held.get('zero', 0))}, constant {size(held.get('constant', 0))}")
    lines.append("")
    if report["pools"]:
        lines.append("Captured graphs' pools (kept: tensors a plugin holds; temporaries: what the graph's replays write)")
        for pool in report["pools"]:
            wrote = (f"of which the workload wrote {size(pool['written'])}" if pool["written"] or not pool["temporaries"]
                     else "which the workload never wrote: its graphs did not run")
            lines.append(f"  {pool['name']:<38} {size(pool['bytes']):>11}: kept {size(pool['kept'])}, temporaries "
                         f"{size(pool['temporaries'])} {wrote}")
            for made in pool["allocations"][:6]:
                lines.append(f"      {size(made['bytes']):>11} x{made['count']:<4} {made['where'][:150]}")
        lines.append("")
    if report["elastic"]:
        lines.append("Elastic ranges")
        for found in report["elastic"]:
            lines.append(f"  {found['name']:<38} {size(found['backed']):>11} backed of {size(found['reserved'])}: "
                         f"written {size(found['written'])}, zero {size(found['zero'])}, constant "
                         f"{size(found['constant'])}" + (f", first backed at the last reading {size(found['new'])}"
                                                         if found.get("new") else "")
                         + (f", and {size(found['host'])} in host memory (not read)" if found.get("host") else "")
                         + f"  ({found['verdict']})")
            planes = defaultdict(list)
            for plane in found.get("planes") or ():
                planes[(plane["unit_bytes"], plane["units"])].append(plane)
            for (unit, units), alike in sorted(planes.items(), key=lambda item: -item[0][0]):
                lines.append(f"      {len(alike)} planes of {units:,} units of {small(unit)}: "
                             f"{span([plane['backed'] for plane in alike])} backed, "
                             f"{span([plane['written'] for plane in alike])} written by the workload")
        lines.append("")
    lines.append("Steps (per step: graphs replayed, allocations, the most held above the start, bytes left allocated)")
    for step in report["steps"]:
        lines.append(f"  {step['kind']:<36} x{step['steps']:<5} replays {step['replays']:<5} allocations "
                     f"{step['allocations']:>6,} (most {step['most_allocations']:,}), peak {size(step['peak'])}, kept "
                     f"{size(max(0, step['kept']))}")
    return "\n".join(lines)


def main(args):
    report = audit(args.package, args.model, args.context, args.prompt_tokens, args.reply_tokens,
                   round(args.least_mib * MIB), mark=not args.no_mark, log=lambda text: print(text, file=sys.stderr, flush=True))
    print(render(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def add_parser(subs):
    sub = subs.add_parser("memory", help="Audit an engine's device memory: load the package, serve a short workload "
                                         "and report which buffers it never writes (dead), which it writes a small "
                                         "part of (oversized), what its captured graphs keep and what each step "
                                         "allocates")
    sub.add_argument("package", type=Path, help="The engine's package (its plugins/ and configuration.json)")
    sub.add_argument("--model", type=Path, required=True, help="The checkpoint folder")
    sub.add_argument("--context", type=int, help="The context tokens the engine serves (default: the engine's own)")
    sub.add_argument("--prompt-tokens", type=int, default=2048,
                     help="Each prompt set's long prompt, in tokens (default %(default)s): past the engine's largest "
                          "prompt chunk shows every chunk's workspaces")
    sub.add_argument("--reply-tokens", type=int, default=96, help="Tokens of each chat's reply (default %(default)s)")
    sub.add_argument("--least-mib", type=float, default=1.0,
                     help="List the buffers of at least this size one by one (default %(default)s)")
    sub.add_argument("--no-mark", action="store_true",
                     help="Write no pattern into the scratch bytes: every verdict then comes from comparing readings, "
                          "and nothing but the workload touches the device")
    sub.add_argument("--out", type=Path, help="Write the report as JSON")
    return sub
