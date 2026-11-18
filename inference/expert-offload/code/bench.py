"""expert-offload measured alone: a routing trace's rounds replayed through the plugin's own forward() on the engine's
real experts, one captured graph a round as an engine's rounds run it, with nothing else on the device. What an engine
loses to its experts is then a number of its own (ms a round, the CPU tier's share, the cache's hits), and a setting is
judged in a minute instead of an engine build and an Eval.

    experts = open_offload(options, model, store)          # the plugin as an engine assembles it, over a store file
    out = decode(experts, rounds, hidden, top)             # {"ms_per_round", "cpu_ms_per_round", "hit_rate", ...}
    out = prompt(experts, rounds, hidden, top, tokens=1024)  # prompt chunks: staging and expert rows, tokens a second

A trace is the cache's own record (record_routes; trace_replay.windows cuts it into rounds). The inputs are random
rows: an expert's time does not depend on their values. The rounds are synchronized one by one, as an engine reads
each round's result. With device memory held beside the plugin (--hold-gib) and device work after every layer's
experts (--model-us), the cache has the slots and its copies the time an engine gives them: the bench's round is
then the engine's. That device work is one kernel that only waits, and nothing running beside it slows it: the bench
shows what a setting costs the experts and the CPU tier, not what device work moved beside a model's own kernels
costs those kernels. Copies launched after a layer's CPU work won here (a round 48.2 against 49.5 ms) and lost
in the engine (41 against 53 tok/s): judge such a change on the engine."""
import json
import tempfile
import time
from pathlib import Path


class _View:
    """A round's counters as crit_plugins.rounds' RoundView gives them."""

    def __init__(self, torch, device, rows):
        self.names = ("expert_reads", "expert_hits")
        self.counters = torch.zeros(rows, len(self.names), dtype=torch.int64, device=device)

    def counter(self, name):
        return self.counters[:, self.names.index(name)]


def open_offload(options, model, store, device="cuda", layers=None):
    """expert-offload assembled from crit's library as an engine's loader does, over the complete bundle store file
    `store` of the checkpoint folder `model` (its index beside it names the geometry and the checkpoint)."""
    from crit.plugins import library
    from crit_core.engine import LoadContext
    from crit_plugins.assemble import assemble
    index = json.loads(Path(str(store) + ".json").read_text(encoding="utf-8"))
    package = Path(tempfile.mkdtemp(prefix="offload-bench-"))
    selection = {"use": {"experts": "expert-offload", "expert_source": "expert-tensors"},
                 "options": {"expert-offload": {**options, "store_path": str(store)}}}
    context = LoadContext(package, model, {"plugins": selection}, device=device)
    context.memory.load()   # an engine starts to load: until serve() below the cache keeps its reserve
    assembly = assemble(selection, library(), context)
    experts = assembly["experts"]
    start = time.perf_counter()
    experts.begin(layers or index["layers"], index["experts"], index["format"], index["hidden"], index["inter"],
                  identity=index["identity"], device=device)
    if not experts.complete:
        raise ValueError(f"{store} is not a complete bundle store of this layout")
    begun = time.perf_counter()
    experts.open()
    from crit_plugins.memory import of
    of(device).serve()   # the load is over, as Core's loader says after an engine's warm requests: the cache fills
    experts.reasons["seconds"] = f"store {begun - start:.1f} s, cache and CPU tier {time.perf_counter() - begun:.1f} s"
    return assembly, experts


def _template(rounds, most=64):
    """A round's plans in call order (the layer sequence most rounds have: a round of another shape runs the plans
    that fit it) and the token rows of a plan (what 999 plans in 1,000 have at most, never more than `most`: a stray
    long plan in a trace is a prompt chunk, not a decode round)."""
    from collections import Counter
    order = list(Counter(tuple(layer for layer, _ in plans) for plans in rounds).most_common(1)[0][0])
    counts = sorted(len(tokens) for plans in rounds for _, tokens in plans)
    rows = min(int(most), counts[min(len(counts) - 1, int(0.999 * len(counts)))])
    return order, rows


def _fill(torch, ids, order, plans, top):
    """ids [plans][rows][top] for one round: each plan's token rows, -1 where a row or a route is absent."""
    ids.fill_(-1)
    at = {}
    for layer, tokens in plans:
        i = order.index(layer, at.get(layer, 0)) if layer in order[at.get(layer, 0):] else None
        if i is None:
            continue
        at[layer] = i + 1
        for r, row in enumerate(tokens[:ids.shape[1]]):
            n = min(len(row), top)
            if n:
                ids[i, r, :n] = torch.from_numpy(row[:n])


def decode(experts, rounds, hidden, top, device="cuda", warmup=0, limit=None, sync=True, model_us=0.0):
    """The trace's rounds through forward() in a captured graph a round. Returns the round time and its parts.
    model_us: device work after every layer's experts, in microseconds (a kernel that only waits), as a model's
    attention and projections take the device between two layers' experts: copies beside the round then have the
    time an engine's round gives them."""
    import torch
    cycles = 0
    if model_us:   # the waiting kernel counts device clock cycles: how many a microsecond takes here
        torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        begun = time.perf_counter()
        torch.cuda._sleep(50_000_000)
        torch.cuda.synchronize()
        cycles = int(model_us * 50_000_000 / (1e6 * (time.perf_counter() - begun)))
    order, rows = _template(rounds, int(experts.options["max_tokens"]))
    rounds = rounds[:limit] if limit else rounds
    n = len(order)
    g = torch.Generator().manual_seed(11)
    x = torch.randn(n, rows, hidden, generator=g).bfloat16().to(device)
    weights = torch.full((n, rows, top), 1.0 / top, dtype=torch.float32, device=device)
    ids = torch.full((n, rows, top), -1, dtype=torch.int64, device=device)
    host = torch.full((n, rows, top), -1, dtype=torch.int64).pin_memory()
    row_of = torch.zeros(rows, dtype=torch.int64, device=device)
    view = _View(torch, device, 1)
    out = torch.zeros(n, rows, hidden, dtype=torch.float32, device=device)

    def one_round():
        experts.start_round(view)
        for i, layer in enumerate(order):
            out[i].copy_(experts.forward(layer, x[i], ids[i], weights[i], rows=row_of, prompt=False))
            if cycles:
                torch.cuda._sleep(cycles)
        experts.count_round(view)

    _fill(torch, host, order, rounds[0], top)
    ids.copy_(host)
    one_round()                      # every shape once outside a capture
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            one_round()
    torch.cuda.synchronize()
    def counters():   # the cache's counters now (synchronized) and the CPU tier's
        cache = experts.cache.telemetry(sync=True) if experts.cache is not None else {}
        return cache, (experts.dispatch.tier.stats() if experts.dispatch is not None else None)

    before, start = None, None
    for r, plans in enumerate(rounds):
        if r == warmup:
            before, start = counters(), time.perf_counter()
            view.counters.zero_()
        _fill(torch, host, order, plans, top)
        ids.copy_(host, non_blocking=True)
        graph.replay()
        if sync:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    after = counters()
    scored = len(rounds) - warmup
    reads, hits = int(view.counter("expert_reads").sum()), int(view.counter("expert_hits").sum())
    copies = sum(after[0].get(name, 0) - before[0].get(name, 0) for name in ("swaps", "fills"))
    result = {"rounds": scored, "plans": n, "rows": rows, "model_ms_per_round": model_us * n / 1000,
              "ms_per_round": 1000 * wall / scored,
              "reads": reads, "hit_rate": hits / reads if reads else 0.0, "misses_per_round": (reads - hits) / scored,
              "slots": after[0].get("slots"), "misses": experts.misses, "copies_per_round": copies / scored}
    tier_a, tier_b = after[1], before[1]
    if tier_a and tier_b:
        busy = tier_a["ms_total"] - tier_b["ms_total"]
        chunks = tier_a["expert_chunks"] - tier_b["expert_chunks"]
        byts = tier_a["weight_bytes"] - tier_b["weight_bytes"]
        result.update({"cpu_ms_per_round": busy / scored, "cpu_us_per_expert": 1000 * busy / chunks if chunks else 0.0,
                       "cpu_gbs": byts / busy / 1e6 if busy else 0.0,
                       "other_ms_per_round": 1000 * wall / scored - busy / scored})
    return result


def prompt(experts, rounds, hidden, top, tokens=1024, chunks=4, device="cuda"):
    """Prompt chunks of `tokens` tokens through every routed layer's forward(prompt=True): each layer's routes are the
    trace's own token rows of that layer, in order. Returns tokens a second through the experts alone."""
    import numpy as np
    import torch
    per_layer = {}
    for plans in rounds:
        for layer, rows in plans:
            per_layer.setdefault(layer, []).extend(row for row in rows if len(row) == top)
    layers = sorted(layer for layer, rows in per_layer.items() if len(rows) >= tokens * (chunks + 1))
    if not layers:
        raise ValueError(f"the trace holds too few routes for {chunks + 1} chunks of {tokens} tokens")
    g = torch.Generator().manual_seed(13)
    x = torch.randn(tokens, hidden, generator=g).bfloat16().to(device)
    weights = torch.full((tokens, top), 1.0 / top, dtype=torch.float32, device=device)

    def chunk(c):
        for layer in layers:
            block = np.stack(per_layer[layer][c * tokens:(c + 1) * tokens])
            experts.forward(layer, x, torch.from_numpy(block).to(device), weights, prompt=True)
        experts.end_step()

    chunk(0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for c in range(1, chunks + 1):
        chunk(c)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    return {"tokens": tokens, "chunks": chunks, "layers": len(layers), "ms_per_chunk": 1000 * wall / chunks,
            "tokens_per_second": tokens * chunks / wall}


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="expert-offload alone on a store file and a routing trace")
    parser.add_argument("--model", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--code", required=True, help="device-expert-cache's code folder (trace_replay)")
    parser.add_argument("--option", action="append", default=[], help="NAME=JSON for expert-offload (repeatable)")
    parser.add_argument("--variant", action="append", default=[],
                        help="NAME=JSON,NAME=JSON: one more run with these options changed (repeatable)")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=0)
    parser.add_argument("--hold-gib", type=float, default=0.0,
                        help="device memory held beside the plugin, as an engine's weights and pools hold it")
    parser.add_argument("--model-us", type=float, default=0.0,
                        help="device work after every layer's experts, in microseconds, as a model's own layers take")
    args = parser.parse_args(argv)
    sys.path.insert(0, args.code)
    import trace_replay
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    rounds = trace_replay.windows(trace, token_rows=True)
    top = max(len(row) for plans in rounds[:50] for _, rows in plans for row in rows)

    def options(text):
        return {k: json.loads(v) for k, v in (item.split("=", 1) for item in text)}

    base = options(args.option)
    _held = None
    if args.hold_gib:   # the cache then has what an engine leaves it, and misses as often as it does there
        import torch
        _held = torch.empty(int(args.hold_gib * 2**30), dtype=torch.uint8, device="cuda")
    for variant in [""] + args.variant:
        chosen = {**base, **options([v for v in variant.split(";") if v])}
        assembly, experts = open_offload(chosen, args.model, args.store)
        try:
            hidden = experts.shape[3]
            found = decode(experts, rounds, hidden, top, warmup=args.warmup, limit=args.limit or None,
                           model_us=args.model_us)
            print(json.dumps({"options": chosen, **{k: (round(v, 3) if isinstance(v, float) else v)
                                                   for k, v in found.items()}}), flush=True)
            if args.prompt_tokens:
                found = prompt(experts, rounds, hidden, top, tokens=args.prompt_tokens)
                print(json.dumps({"prompt": {k: (round(v, 2) if isinstance(v, float) else v)
                                             for k, v in found.items()}}), flush=True)
            print("FACTS " + json.dumps(experts.facts()["expert_offload"].get("reasons")), flush=True)
            if experts.dispatch is not None:   # what one post and join of the CPU tier costs here, by kind of join
                print("JOINS " + json.dumps(experts.dispatch.joins, default=str), flush=True)
        finally:
            assembly.close()


if __name__ == "__main__":
    main()
