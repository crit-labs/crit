"""Tune one of crit's kernels for a model on this GPU: build every knob configuration, check each against its reference,
time it on the model's own shapes, and write the tuning table the kernel's build reads.

Each structure in kernel.json names a bench (kernel.json "bench": "bench.py:Class" in the kernel's folder), a class
with:

    structure = "<structure id>"
    def shapes(self, model): the shape keys this structure meets in the model (a crit_kernels.model.Model), a list of
        dicts holding the structure's "key" fields (and anything else inputs() needs)
    def inputs(self, shape, rows, device): the call's inputs for that shape and row count; row 0's input must not depend
        on rows (seed each row on its own), so a row's result can be compared alone and batched
    def run(self, module, state): call the kernel's package once and return its output
    def reference(self, state): the expected output (FP64 or FP32, from the operation's reference)
    def check(self, got, want, state): None when got is within the operation's tolerance of want, else the reason
    def first_row(self, out): the part of out that belongs to row 0 (for the row-invariance check)
    def bytes(self, shape, rows): optional, bytes the call must move (for the GB/s column)
    def valid(self, knobs): optional, False for knob combinations that cannot build or run
    def valid_for(self, knobs, shape): optional, False where a configuration cannot run this shape (a split that does
        not divide K): the shape is not timed with it
    def variants(self, shape): optional, for a kernel with CUDA sources beside its Triton modules (kernel.json "cuda"):
        the CUDA variants (dicts of its "variant" fields) a call on this shape uses, compiled before the GPU is held
    default: set by the sweep before timing, the kernel's untuned build (its default knobs), for benches that compare
        a configuration's bits with it or plan their calls with it
    def yardstick(self, state): the call of the best vendor or open library at the same arithmetic (kernel.json's
        structure "yardstick" names it), as a function of no arguments ready to time, or None where that library is
        not installed here. A structure with no library at its arithmetic declares "yardstick": null and its report
        rows carry the achieved GB/s (bytes()) against the device instead.

A knob with "order": true changes the kernel's summation order. A row's bits must not depend on the rows beside it
(the operations' invariance contract), so one shape keeps one value of every order knob for all its row counts; the
others (speed only) may differ per row count. Configurations whose result is wrong, or whose row 0 differs from row 0
computed alone (for a structure whose kernel.json says "row_invariant": true), are dropped.

A structure may declare "lengths": {"field": "N", "values": [...]} (a shape field the bench reads: the positions a
row holds, for attention): every shape is swept at each of those lengths (a shape that gives the field keeps its own),
a length is part of the shape's results, and the table keeps one value of every order knob for a shape over all its
lengths and row counts, its speed knobs per length range and row count (table entries' "length": [lo, hi]; the build
looks them up by the call's length, tables without them hold for every length).

Timing: cold L2 (a buffer of twice the L2 is written before each sample), one call between CUDA events, the median of
`repeat` samples after `warmup` calls. The yardstick library is timed at the end paired with the configuration the
table picked for the same shape and row count, both as CUDA graph replays where they can be captured (a bench whose run
replays its own graph, or cannot be captured, as it runs), the two in turn for YARDSTICK_ROUNDS rounds, the lower
median of each kept: vs_library compares the two the same way and under the same conditions (a GPU that has run the whole sweep may clock lower than at
its start, where the picks were timed first).
"""
import importlib.util
import itertools
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import build as kbuild


def _bench_class(folder, ref):
    path, _, name = ref.partition(":")
    spec = importlib.util.spec_from_file_location(f"crit_kernel_bench_{folder.name.replace('-', '_')}", folder / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)()


def configurations(manifest, structure, bench=None):
    knobs = manifest["structures"][structure].get("knobs", {})
    names = list(knobs)
    out = []
    for values in itertools.product(*(knobs[n]["values"] for n in names)):
        config = dict(zip(names, values))
        if bench is None or not hasattr(bench, "valid") or bench.valid(config):
            out.append(config)
    return out


def _build_one(args):
    kernel_id, configs, root, *rest = args
    variants = rest[0] if rest else None
    os.environ.setdefault("MAX_JOBS", "1")
    try:
        if configs is None:   # the untuned build (a mixed kernel's CUDA variants the benches call)
            kbuild.build(kernel_id, root=root, variants=variants)
        else:
            kbuild.build(kernel_id, configs=configs, sweep=True, root=root, variants=variants)
        return configs, None
    except Exception as error:  # noqa: BLE001 - a configuration that does not build is dropped with its reason
        return configs, f"{type(error).__name__}: {str(error).strip().splitlines()[-1][:300] if str(error).strip() else ''}"


class Timer:
    def __init__(self, torch, warmup=3, repeat=25):
        self.torch, self.warmup, self.repeat = torch, warmup, repeat
        l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 64 << 20) or (64 << 20)
        self.flush = torch.empty(max(2 * l2, 64 << 20) // 4, dtype=torch.float32, device="cuda").fill_(1)
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        self.spin = int(torch.cuda.get_device_properties(0).clock_rate * 0.2) if hasattr(
            torch.cuda.get_device_properties(0), "clock_rate") else 400_000   # about 200 us of cycles (kHz * 0.2)

    def __call__(self, fn):
        samples = []
        for i in range(self.warmup + self.repeat):
            # A write, not a read: lines a call wrote back (a library's state) stay in L2 through a read flush, so a
            # kernel that writes its inputs would be timed warm against one that only reads them.
            self.flush.zero_()
            # A spin on the GPU while the host queues the call, so the time is the GPU's, not the host's launch path
            # (a Triton launch from Python takes tens of microseconds, more on a busy CPU).
            self.torch.cuda._sleep(self.spin)
            self.start.record()
            fn()
            self.stop.record()
            self.stop.synchronize()
            if i >= self.warmup:
                samples.append(self.start.elapsed_time(self.stop) * 1000.0)
        return statistics.median(samples)

    def graph(self, fn):
        """fn timed as a CUDA graph replay (eager when it cannot be captured): a library call that synchronizes or
        does host work past the spin (a DSL launch path, a tactic lookup) is then timed as the GPU runs it in a
        captured decode step, not with its host path in the sample. Returns (us, "graph" or "eager")."""
        replay = self.graph_of(fn)
        return self(replay), ("eager" if replay is fn else "graph")

    def graph_of(self, fn):
        """fn captured as a CUDA graph: its replay (bound to the graph, which it keeps alive), or fn itself where it
        cannot be captured."""
        torch = self.torch
        try:
            fn()
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - not capturable: time it eagerly
            torch.cuda.synchronize()
            return fn
        return graph.replay


def _variants(bench, shapes):
    """A mixed kernel's CUDA variants a bench calls on its shapes (bench.variants(shape), the kernel.json "cuda"
    "variant" fields), each once."""
    found = []
    for shape in shapes:
        for v in (bench.variants(shape) if hasattr(bench, "variants") else []):
            if dict(v) not in found:
                found.append(dict(v))
    return found


YARDSTICK_ROUNDS = 3


def _fields(spec):
    """A structure's result key fields: its key, and its lengths' field where it declares one."""
    fields = list(spec.get("key", []))
    lengths = spec.get("lengths")
    if lengths and lengths["field"] not in fields:
        fields.append(lengths["field"])
    return fields


def _at_lengths(spec, shapes):
    """The shapes at each of the structure's declared lengths (a shape that gives the length field keeps its own)."""
    lengths = spec.get("lengths")
    if not lengths:
        return list(shapes)
    out = []
    for shape in shapes:
        if lengths["field"] in shape:
            out.append(dict(shape))
        else:
            out.extend(dict(shape, **{lengths["field"]: int(v)}) for v in lengths["values"])
    return out


def _key(shape, fields):
    return tuple(int(shape[f]) for f in fields)


def _lock(path):
    if not path:
        return None
    import fcntl
    handle = open(path, "a")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def _chunks(manifest, structure, configs, size):
    """The structure's configurations in builds of `size`: each build holds the defaults (index 0) and its share of the
    rest in order, so a configuration's index in its build is known (select())."""
    base = kbuild.defaults(manifest, structure)
    rest = [c for c in configs if c != base]
    groups = [rest[i:i + size] for i in range(0, len(rest), size)] or [[]]
    out = []
    for group in groups:
        indices = [(0, base)] if base in configs and group is groups[0] else []
        indices += [(i + 1, c) for i, c in enumerate(group)]
        out.append((group, indices))
    return out


def sweep(kernel_id, *, model=None, shapes=None, structures=None, out=None, jobs=None, root=None, lock=None,
          warmup=3, repeat=25, chunk=24, log=print):
    """Tune kernel_id. model: a checkpoint folder (its shapes come from each bench's shapes()); shapes: {structure:
    [shape dicts]} instead. Returns the tuning dict (also written to out/<kernel_id>.json with a report beside it)."""
    import torch
    from .model import Model
    folder, manifest = kbuild.find(kernel_id, root)
    chosen = structures or [s for s, spec in manifest.get("structures", {}).items() if spec.get("bench")]
    if not chosen:
        raise ValueError(f"{kernel_id} declares no structure with a bench to sweep")
    unknown = [s for s in chosen if s not in manifest.get("structures", {})]
    benchless = [s for s in chosen if s not in unknown and not manifest["structures"][s].get("bench")]
    if unknown or benchless:
        swept = sorted(s for s, spec in manifest.get("structures", {}).items() if spec.get("bench"))
        raise ValueError(f"{kernel_id}: " + "; ".join(
            ([f"no structure {', '.join(unknown)}"] if unknown else [])
            + ([f"{', '.join(benchless)} has no bench (an alias is tuned through the structure it aliases)"]
               if benchless else [])) + f"; structures with a bench: {', '.join(swept)}")
    benches = {s: _bench_class(folder, manifest["structures"][s]["bench"]) for s in chosen}
    loaded_model = Model(model) if model else None
    shape_sets = {}
    for s in chosen:
        given = (shapes or {}).get(s) if shapes else None
        found = given if given is not None else (benches[s].shapes(loaded_model) if loaded_model else [])
        found = _at_lengths(manifest["structures"][s], found)
        unique, keys = [], set()
        fields = _fields(manifest["structures"][s])
        for shape in found:
            k = _key(shape, fields)
            if k not in keys:
                keys.add(k)
                unique.append(shape)
        shape_sets[s] = unique
        log(f"{kernel_id}/{s}: {len(unique)} shapes")
    # Each structure is swept with the others on their defaults, its configurations in builds of `chunk` (one compile
    # of the binding and the knob-independent sources a build), each picked at run time (select()).
    plan = []   # (structure, configurations the build holds, [(index in the build, configuration)])
    total = 0
    for s in chosen:
        if not shape_sets[s]:
            log(f"{kernel_id}/{s}: no shapes, not swept")
            continue
        found = configurations(manifest, s, benches[s])
        total += len(found)
        for group, indices in _chunks(manifest, s, found, chunk):
            plan.append((s, group, indices))
    log(f"{kernel_id}: building {total} configurations in {len(plan)} builds")
    t0 = time.time()
    failed = {}
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    if manifest["language"] == "cuda":
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_build_one, (kernel_id, {s: group}, root)): (s, indices)
                       for s, group, indices in plan}
            for future in as_completed(futures):
                s, indices = futures[future]
                _, error = future.result()
                if error:
                    for _, c in indices:
                        failed[(s, json.dumps(c, sort_keys=True))] = error
    elif kbuild._mixed(manifest):
        # A Triton kernel with CUDA sources: the builds of its CUDA structures compile the variants their benches call
        # (bench.variants(shape)), and the untuned build all of them, before the GPU is held; the Triton structures'
        # builds need no compile step.
        cuda = set(manifest["cuda"].get("structures", []))
        variants = {s: _variants(benches[s], shape_sets[s]) for s in chosen if s in cuda}
        every = [v for s in sorted(variants) for v in variants[s]]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_build_one, (kernel_id, {s: group}, root, variants[s])): (s, indices)
                       for s, group, indices in plan if s in cuda and variants[s]}
            if every:
                futures[pool.submit(_build_one, (kernel_id, None, root, every))] = (None, [])
            for future in as_completed(futures):
                s, indices = futures[future]
                _, error = future.result()
                if error and s is None:
                    log(f"{kernel_id}: the untuned build's CUDA variants did not compile: {error}")
                for _, c in indices if error else []:
                    failed[(s, json.dumps(c, sort_keys=True))] = error
    log(f"{kernel_id}: built in {time.time() - t0:.0f} s, {len(failed)} configurations failed to build")
    # The untuned build, before the GPU is held: benches may compare bits with it or plan calls with it.
    default_module = kbuild.load(kernel_id, root=root)
    for bench in benches.values():
        bench.default = default_module
    handle = _lock(lock)
    try:
        timer = Timer(torch, warmup, repeat)
        results = {}   # (structure, shape key, rows, config json) -> us
        dropped = dict(failed)
        where = {}     # (structure, config json) -> (module, index in its build)
        t0 = time.time()
        for s, group, indices in plan:
            try:
                module = kbuild.load(kernel_id, configs={s: group}, sweep=True, root=root)
            except Exception as error:  # noqa: BLE001
                for _, c in indices:
                    dropped.setdefault((s, json.dumps(c, sort_keys=True)), f"load: {error}")
                continue
            for index, config in indices:
                cj = json.dumps(config, sort_keys=True)
                if (s, cj) not in dropped:
                    kbuild.select(module, s, index)
                    _measure(s, cj, module, manifest, benches, shape_sets, results, dropped, timer, torch)
                    where[(s, cj)] = (module, index)
            kbuild.select(module, s, -1)
        log(f"{kernel_id}: measured in {time.time() - t0:.0f} s, {len(dropped)} configurations dropped")
        swept = [s for s in chosen if shape_sets[s]]
        # A median can still catch other work on the GPU: re-time the defaults and every pick once more and keep the
        # lower of the two medians, then choose again with them.
        tuning, report = choose(manifest, swept, shape_sets, results, dropped, benches)
        _confirm(tuning, manifest, swept, shape_sets, results, dropped, where, benches, timer)
        tuning, report = choose(manifest, swept, shape_sets, results, dropped, benches)
        library = _yardsticks(manifest, swept, shape_sets, benches, timer, report, where, results)
    finally:
        if handle:
            handle.close()
    tuning, report = choose(manifest, swept, shape_sets, results, dropped, benches)
    _add_yardsticks(manifest, report, library)
    tuning["source"] = kbuild.source_digest(kernel_id, root)
    tuning["device"] = torch.cuda.get_device_name()
    tuning["capability"] = kbuild.device_arch()[1]
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{kernel_id}.json").write_text(json.dumps(tuning, indent=1) + "\n")
        (out / f"{kernel_id}.report.json").write_text(json.dumps(report, indent=1) + "\n")
    return tuning, report


def _yardsticks(manifest, swept, shape_sets, benches, timer, report=None, where=None, results=None):
    """Time each structure's yardstick library on every shape and tuned row count: (structure, key, rows) -> (us,
    "graph" or "eager", the pick's us timed beside it or None), or a reason it was not timed. Where the report names
    the configuration the table picked (and the sweep still holds its build), the library and the pick are timed in
    turn, YARDSTICK_ROUNDS rounds, the lower median of each kept (the pick's only for vs_library: the table keeps
    the sweep's times, every configuration's taken the same way)."""
    import torch
    picks = {}
    for s2, rows_report in ((report or {}).get("structures") or {}).items():
        spec = manifest["structures"][s2]
        for entry in rows_report:
            k = tuple(int(entry["shape"][f]) for f in _fields(spec))
            picks[(s2, k, entry["rows"])] = json.dumps(entry["config"], sort_keys=True)
    found = {}
    for s in swept:
        spec = manifest["structures"][s]
        fields = _fields(spec)
        bench = benches[s]
        for shape in shape_sets[s]:
            for rows in spec.get("tune_rows", [1]):
                k = (s, _key(shape, fields), rows)
                if not spec.get("yardstick") or not hasattr(bench, "yardstick"):
                    continue
                try:
                    state = bench.inputs(shape, rows, "cuda")
                    call = bench.yardstick(state)
                    if call is None:
                        found[k] = "not installed here"
                        continue
                    replay = timer.graph_of(call)
                    how = "eager" if replay is call else "graph"
                    lib = timer(replay)
                    cj = picks.get(k)
                    pick = None
                    if cj is not None and where is not None and (s, cj) in where:
                        module, index = where[(s, cj)]
                        kbuild.select(module, s, index)
                        # the pick captured too where it can be (as the library is), else as the sweep timed it
                        mine = timer.graph_of(lambda: bench.run(module, state))
                        for _ in range(YARDSTICK_ROUNDS):
                            t = timer(mine)
                            pick = t if pick is None else min(pick, t)
                            lib = min(lib, timer(replay))
                        torch.cuda.synchronize()
                        kbuild.select(module, s, -1)
                    found[k] = (lib, how, pick)
                except Exception as error:  # noqa: BLE001 - a library that cannot take the shape is a fact to report
                    found[k] = f"{type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"
    return found


def _add_yardsticks(manifest, report, library):
    """Each report row gains its yardstick: the library's name, its time and the tuned kernel's time against it."""
    for s, rows_report in report["structures"].items():
        spec = manifest["structures"][s]
        name = (spec.get("yardstick") or {}).get("library")
        for entry in rows_report:
            entry["library"] = name
            if not name:
                continue
            k = (s, tuple(int(entry["shape"][f]) for f in _fields(spec)), entry["rows"])
            value = library.get(k)
            if isinstance(value, tuple):
                entry["library_us"] = round(value[0], 3)
                entry["library_timed"] = value[1]
                # against the pick timed beside the library (the same conditions) where it was, else its sweep time
                mine = value[2] if len(value) > 2 and value[2] else entry["tuned_us"]
                if len(value) > 2 and value[2]:
                    entry["paired_us"] = round(value[2], 3)
                entry["vs_library"] = round(value[0] / mine, 3)
            elif value:
                entry["library_us"] = None
                entry["library_note"] = value


def _confirm(tuning, manifest, swept, shape_sets, results, dropped, where, benches, timer):
    """Re-time each structure's defaults and the configurations its table picked, on every shape and tuned row count,
    keeping the lower median."""
    for s in swept:
        spec = manifest["structures"][s]
        fields = _fields(spec)
        picked = {json.dumps(c, sort_keys=True) for c in tuning["structures"][s]["configs"]}
        for cj in sorted(picked):
            if (s, cj) in dropped or (s, cj) not in where:
                continue
            module, index = where[(s, cj)]
            kbuild.select(module, s, index)
            for shape in shape_sets[s]:
                k = _key(shape, fields)
                for rows in spec.get("tune_rows", [1]):
                    if (s, k, rows, cj) in results:
                        state = benches[s].inputs(shape, rows, "cuda")
                        again = timer(lambda: benches[s].run(module, state))
                        results[(s, k, rows, cj)] = min(results[(s, k, rows, cj)], again)
            kbuild.select(module, s, -1)


def _measure(s, cj, module, manifest, benches, shape_sets, results, dropped, timer, torch):
    """Check one configuration on every shape and tuned row count, and time it where it passes. A wrong result or a
    bench error drops the configuration for that shape, with its reason, and the sweep goes on; a shape the bench says
    the knobs cannot run (valid_for(knobs, shape) False) is skipped for it."""
    spec = manifest["structures"][s]
    bench, fields = benches[s], _fields(spec)
    invariant = spec.get("row_invariant", False)
    knobs = json.loads(cj)
    for shape in shape_sets[s]:
        if hasattr(bench, "valid_for") and not bench.valid_for(knobs, shape):
            continue
        where = (s, f"{cj} @ {json.dumps({f: shape[f] for f in spec.get('key', [])}, sort_keys=True)}")
        first = None
        for rows in spec.get("tune_rows", [1]):
            try:
                state = bench.inputs(shape, rows, "cuda")
                got = bench.run(module, state)
                torch.cuda.synchronize()
                reason = bench.check(got, bench.reference(state), state)
                if reason:
                    dropped[where] = f"{shape} rows {rows}: {reason}"
                    break
                row0 = bench.first_row(got).clone()
                if invariant:
                    if first is None:
                        alone = bench.inputs(shape, 1, "cuda")
                        first = bench.first_row(bench.run(module, alone)).clone()
                    if not torch.equal(row0.view(torch.uint8) if row0.dtype.is_floating_point else row0,
                                       first.view(torch.uint8) if first.dtype.is_floating_point else first):
                        dropped[where] = f"{shape} rows {rows}: row 0 differs from row 0 alone"
                        break
                results[(s, _key(shape, fields), rows, cj)] = timer(lambda: bench.run(module, state))
            except Exception as error:  # noqa: BLE001 - one configuration's failure on one shape, not the sweep's
                dropped[where] = f"{shape} rows {rows}: {type(error).__name__}: {str(error).splitlines()[0][:300] if str(error) else ''}"
                break


def choose(manifest, chosen, shape_sets, results, dropped, benches):
    """The tuning table: per structure and shape, one value of every order knob (the lowest sum over the tuned row
    counts, and the declared lengths, of time relative to that count's best), and per row count (and length) the
    fastest configuration with it. Row ranges run from one tuned count to the next, length ranges likewise."""
    tuning = {"schema": kbuild.TUNING_SCHEMA, "kernel": manifest["id"], "structures": {}}
    report = {"kernel": manifest["id"], "structures": {}, "dropped": {f"{s} {c}": r for (s, c), r in dropped.items()}}
    for s in chosen:
        spec = manifest["structures"][s]
        fields, knobs = spec.get("key", []), spec.get("knobs", {})
        lengths = spec.get("lengths")
        all_fields = _fields(spec)
        order = [n for n, k in knobs.items() if k.get("order")]
        default = json.dumps(kbuild.defaults(manifest, s), sort_keys=True)
        tune_rows = sorted(spec.get("tune_rows", [1]))
        lo_all, hi_all = spec.get("rows", [1, tune_rows[-1]])
        configs, table, rows_report = [json.loads(default)], [], []
        # the shapes by their key, each at its lengths (one length, None, without declared lengths)
        by_key = {}
        for shape in shape_sets[s]:
            by_key.setdefault(_key(shape, fields), []).append(shape)
        for k, shapes in by_key.items():
            at = sorted(shapes, key=lambda sh: int(sh[lengths["field"]]) if lengths else 0)
            points = [(_key(sh, all_fields), r) for sh in at for r in tune_rows]
            candidates = {c for (s2, k2, _, c) in results if s2 == s and k2 in {p for p, _ in points}}
            complete = [c for c in candidates if all((s, p, r, c) in results for p, r in points)]
            if not complete:
                continue
            groups = {}
            for c in complete:
                g = tuple(json.loads(c)[n] for n in order)
                groups.setdefault(g, []).append(c)
            # Each row count weighs the same unless the structure gives tune_weights (and each length the same): the
            # order-knob group whose picks lose least against each point's best.
            best_at = {(p, r): min(results[(s, p, r, c)] for c in complete) for p, r in points}
            weight = dict(zip(tune_rows, spec.get("tune_weights") or [1] * len(tune_rows)))
            best_total, best_pick = None, None
            for members in groups.values():
                pick = {(p, r): min(members, key=lambda c: results[(s, p, r, c)]) for p, r in points}
                total = sum(weight[r] * results[(s, p, r, pick[(p, r)])] / best_at[(p, r)] for p, r in points)
                if best_total is None or total < best_total:
                    best_total, best_pick = total, pick
            # a length's range runs to the geometric middle between it and the next one (a call's length, the
            # positions its tables hold, is a little past the length its rows reach)
            length_lo = 0
            for j, shape in enumerate(at):
                p = _key(shape, all_fields)
                length = int(shape[lengths["field"]]) if lengths else None
                nxt = int(at[j + 1][lengths["field"]]) if j + 1 < len(at) else None
                length_hi = (1 << 62) if nxt is None else int((length * nxt) ** 0.5)
                previous = lo_all - 1
                for i, r in enumerate(tune_rows):
                    c = best_pick[(p, r)]
                    hi = hi_all if i == len(tune_rows) - 1 else r
                    config = json.loads(c)
                    if config not in configs:
                        configs.append(config)
                    index = configs.index(config)
                    key = {f: int(shape[f]) for f in fields}
                    span = [length_lo, length_hi] if lengths else None
                    last = table[-1] if table else None
                    if last and last["key"] == key and last["config"] == index and last["rows"][1] == previous and \
                            last.get("length") == span:
                        last["rows"][1] = hi
                    else:
                        entry = {"key": key, "rows": [previous + 1, hi], "config": index}
                        if span:
                            entry["length"] = span
                        table.append(entry)
                    previous = hi
                    base = results.get((s, p, r, default))
                    tuned = results[(s, p, r, c)]
                    shown = {f: int(shape[f]) for f in all_fields}
                    entry = {"shape": shown, "rows": r, "tuned_us": round(tuned, 3), "config": config,
                             "default_us": round(base, 3) if base else None,
                             "speedup": round(base / tuned, 3) if base else None}
                    if lengths:
                        entry["length"] = length
                    bench = benches[s]
                    if hasattr(bench, "bytes"):
                        try:
                            entry["tuned_GBs"] = round(bench.bytes(shape, r) / tuned / 1e3, 1)
                        except Exception:   # noqa: BLE001 - the GB/s column is a courtesy
                            pass
                    rows_report.append(entry)
                if lengths:
                    length_lo = length_hi + 1
        tuning["structures"][s] = {"configs": configs, "table": table}
        report["structures"][s] = rows_report
    return tuning, report


__all__ = ["choose", "configurations", "sweep"]
