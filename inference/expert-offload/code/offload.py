"""expert-offload's engine: one object that decides where a mixture of experts' routed experts live and who computes
each route, and runs a routed layer in one call.

The decision tree, at begin() (sizes) and open() (after the model's own pools, with the device's free memory then):

    tier      device  every expert fits on the device beside the model's own needs: one store per layer, no cache
              ram     host memory holds every bundle: a mapped bundle store the device reads, a device expert cache
                      in every byte left
              disk    host memory does not: a store file read on the host at a queue depth, a host tier of the
                      memory that is free holding what the device cache evicts, the cache in front of both
    misses    cpu     the CPU tier computes every miss beside the device (nvfp4 W4A4 / int4 bundles, the native tier
                      on the host's pool and the host's in-stream handoff, Core's crit_plugins.host) while the link copies it into the cache beside the round: it
                      serves from the next round, in the slot of the expert routed longest ago (policy recent). The
                      device's rows then take the CPU's integer order
              link    every miss copied into a cache slot before its layer reads it (the clock picks the slot), the
                      device's rows in the kernel's tensor arithmetic
    prompts   a prompt chunk stages its layer's non-resident experts through the cache's ring and runs in the tensor
              arithmetic (prompt_arithmetic) or the CPU's order

Nothing here is new arithmetic: the cache is device-expert-cache's ExpertCache, the store expert-tensors' BundleStore,
the CPU tier host-expert-tier's CpuTier behind heterogeneous-expert-dispatch's DeviceDispatch, the rows grouped-experts'
kernel (KernelRoutes)."""
import hashlib
import json
import os
from pathlib import Path

GIB = 1 << 30
CPU_FORMATS = ("nvfp4", "int4")
PROFILE_KIND = "expert-offload-routing"
# A prompt chunk is given what such steps were measured to take, an eighth more and SLACK: a step's peak varies from
# run to run, and the device's free memory never falls under margin_mib for it.
SLACK = 32 << 20


def host_available_bytes():
    """Host memory available now, as Core reads it (crit_plugins.host): 0 where the OS does not say."""
    from crit_plugins.host import memory_bytes
    return memory_bytes()["available"]


def choose_tier(total_bytes, device_free, device_needs, host_free, margin, ram_headroom, forced="auto"):
    """device, ram or disk for experts of `total_bytes`: on the device when they fit beside the model's own needs and
    the margin, in host memory when it holds them beside its headroom, else in a file."""
    if forced != "auto":
        return forced
    if device_free is not None and total_bytes + device_needs + margin <= device_free:
        return "device"
    if total_bytes + ram_headroom <= host_free:
        return "ram"
    return "disk"


def cache_slots(free, margin, bundle_bytes):
    """Own slots of a device expert cache in every byte the device has free past the margin (a scratch slot beside)."""
    return max(0, (int(free) - int(margin)) // int(bundle_bytes) - 1)


def prompt_bytes(peaks, tokens):
    """The device bytes a prompt step whose largest chunk has `tokens` tokens takes for its temporaries, from the
    steps measured so far (peaks: {chunk tokens: the most such a step took}): a measured size's own peak; between two
    measured sizes the line through them; and never less than a smaller chunk took. Past the largest measured size it
    is the largest's peak in proportion to the tokens: a step's temporaries have a part that does not grow with its
    tokens (workspaces, a first graph), so that is more than a chunk never seen takes, and it holds for one step:
    the chunk's own measurement follows."""
    seen = sorted(peaks.items())
    if not seen:
        return 0
    floor = max((p for n, p in seen if n <= tokens), default=seen[0][1])
    if tokens <= seen[0][0]:
        return int(seen[0][1])
    for (n0, p0), (n1, p1) in zip(seen, seen[1:]):
        if tokens <= n1:
            return int(max(floor, p0 + (p1 - p0) * (tokens - n0) / (n1 - n0)))
    n1, p1 = seen[-1]
    return int(max(floor, p1 * tokens / n1))


def learn(peaks, first, tokens, peak, serving):
    """A prompt step whose largest chunk had `tokens` tokens took `peak` device bytes: peaks ({chunk tokens: the
    most such a step took}) and first (the sizes measured while the engine loaded) after it, in place. A step of
    the load is a first step of its kind: its peak holds what its graphs' captures took beside the chunk's own
    temporaries, and stands only until a step the engine serves, of that size or larger, took less."""
    if serving:
        for size in [size for size in first if size <= tokens and peaks.get(size, 0) >= peak]:
            first.discard(size)
            del peaks[size]
        first.discard(tokens)
    else:
        first.add(tokens)
    peaks[tokens] = max(peaks.get(tokens, 0), peak)


# The CPU tier takes the misses where it computes a missed expert in under this share of the time the link copies
# one: beside the CPU tier every device row runs the declared order, slower than the tensor cores', and the tier
# needs its threads awake, so at equal times the link serves a round sooner and starts no slower after a pause.
CPU_WINS = 0.8


def choose_misses(format, cpu_ready, forced="auto", measured=None):
    """cpu or link. The link where the CPU tier does not compute the format or is not built here (the native tier
    and the in-stream join), or where it is asked for. Else, on auto, whichever serves a missed expert sooner on
    this machine: measured {"cpu": us, "link": us} a missed expert (the tier computing it, the link copying it),
    the CPU tier where it takes under CPU_WINS of the link's time; without a measurement, and where it is asked
    for, the CPU tier."""
    if forced == "link" or format not in CPU_FORMATS or not cpu_ready:
        return "link"
    if forced == "cpu" or not measured:
        return "cpu"
    return "cpu" if measured["cpu"] < CPU_WINS * measured["link"] else "link"


def diagnosis(telemetry):
    """One line on what bounds the engine's experts, from its telemetry: hit rate, where misses went, who holds the
    device's memory (Core's census of it, crit_plugins.memory), the next lever."""
    cache = telemetry.get("expert_cache") or {}
    tier, path = telemetry.get("tier"), telemetry.get("misses")
    if tier == "device":
        return "every expert on the device: experts cost only the device's expert rows"
    reads, hits = cache.get("reads", 0), cache.get("hits", 0)
    if not reads:
        return f"{tier} tier, misses {path}: no decode routes yet"
    rate = hits / reads
    cpu = (telemetry.get("cpu_routes") or {}).get("routes", 0)
    fills = cache.get("fills", 0)
    line = (f"{tier} tier: hits {rate:.1%} of {reads:,} expert reads over {cache.get('slots', 0):,} slots; misses "
            f"{fills:,} copied over the link" + (f", {cpu:,} routes on the CPU" if path == "cpu" else ""))
    device = telemetry.get("device")
    if device:   # the cache is what the model leaves: its share of the device is the experts' first lever
        holders = ", ".join(f"{plugin} {held / GIB:.1f}" for plugin, held in device.get("holders") or ())
        line += (f"; of the device's {device['total'] / GIB:.1f} GiB the cache holds {device['cache'] / GIB:.1f}, the "
                 f"model's weights, pools and workspaces {device['model'] / GIB:.1f}"
                 + (f" ({holders})" if holders else "") + f", {device['free'] / GIB:.1f} are "
                 f"free (a GiB more for the cache is {GIB // device['bundle']:,} slots)")
        tensors = ", ".join("".join(f"[{n}]" for n in found["shape"]) + f" {found['dtype']} "
                            f"{found['bytes'] / GIB:.1f} of {found['plugin']}" for found in device.get("tensors") or ())
        if tensors:   # by shape: a table a pass gathers rows of, or a pool sized past what is served, shows here
            line += f"; its largest tensors: {tensors}"
    if rate < 0.8:
        line += ("; next lever: more slots (device memory the model can do without: tables it can read from the "
                 "host, pools sized past what it serves, weights held wider than stored), then more copies a round "
                 "(swaps); crit plugins offload --trace replays both in seconds")
    return line


class FileStore:
    """A bundle store file read on the host (the disk tier), with a bounded host tier between it and the device.

    read(keys, into) fills a host buffer with the bundles the device cache is about to copy up. With host_bytes the
    store keeps as many bundles in host memory of its own and each bundle is held once: the device cache hands a
    bundle it evicts back (retain(), which the cache calls because `retains` is set), and a bundle read for a cache
    slot leaves the host tier as it goes up. The device's slots and the host's then add up to what is kept off the
    file, where copies on both sides would hold the same experts twice. The file is read past the OS's page cache
    (the host tier is the copy) from a few threads at once (Core's crit_plugins.host reads), so a drive works at a
    queue depth instead of one page fault at a time. Without host_bytes the reads are buffered and the page cache is
    the host copy, as before, still at a queue depth. No device mapping (mapped False: a device expert cache over it
    plans eagerly and cannot be captured)."""

    mapped = False
    DEPTH = 8

    def __init__(self, layout, layers, experts, path, host_bytes=0):
        self.layout = dict(layout)
        self.bundle_bytes = int(layout["bundle_bytes"])
        self.layers, self.experts = int(layers), int(experts)
        self.keys = self.layers * self.experts
        self.compressed = ()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        self._fd = os.open(self.path, flags, 0o644)
        os.ftruncate(self._fd, self.keys * self.bundle_bytes)
        self.complete = False
        self.slots = min(self.keys, max(0, int(host_bytes)) // self.bundle_bytes)
        self.retains = self.slots > 0     # the device cache hands its evicted bundles back
        self._host = None                 # [slots][bundle_bytes], made at the first bundle kept
        self._where = {}                  # key -> host slot, the bundle kept longest first
        self._free = list(range(self.slots))
        self._reads = None
        self.counted = {"host_hits": 0, "file_reads": 0, "retained": 0, "dropped": 0}

    def put(self, layer, expert, bundle, compressed=None):
        data = bundle.detach().contiguous().cpu().numpy().tobytes() if hasattr(bundle, "numpy") else bytes(bundle)
        if len(data) != self.bundle_bytes:
            raise ValueError(f"a bundle is {self.bundle_bytes} bytes, not {len(data)}")
        if self._reads is not None:   # a load after the first read: the readers open the file again
            self._reads.close()
            self._reads = None
        at = (layer * self.experts + expert) * self.bundle_bytes
        if hasattr(os, "pwrite"):
            os.pwrite(self._fd, data, at)
        else:   # Windows: the loads come one at a time
            os.lseek(self._fd, at, os.SEEK_SET)
            os.write(self._fd, data)

    def _reader(self):
        if self._reads is None:
            from crit_plugins.host import current
            os.fsync(self._fd)
            if self.retains and hasattr(os, "posix_fadvise"):   # the loads' pages: the host tier is the copy now
                os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
            self._reads = current().reads("expert-offload", self.path, depth=self.DEPTH, direct=self.retains)
        return self._reads

    def _arena(self):
        if self._host is None:
            import torch
            self._host = torch.empty(self.slots, self.bundle_bytes, dtype=torch.uint8)
        return self._host

    def read(self, keys, into, keep=False):
        """The bundles of `keys` into the rows of `into` (a host uint8 tensor). A bundle the host tier holds comes
        from it, and leaves it unless keep (a prompt's ring slot, which the cache does not keep either); the others
        are read from the file, all at once."""
        requests = []
        for i, key in enumerate(keys):
            key = int(key)
            slot = self._where.get(key)
            if slot is None:
                requests.append((key * self.bundle_bytes, memoryview(into[i].numpy())))
                continue
            into[i].copy_(self._host[slot])
            self.counted["host_hits"] += 1
            if not keep:
                del self._where[key]
                self._free.append(slot)
        if requests:
            self._reader().read(requests)
            self.counted["file_reads"] += len(requests)

    def retain(self, keys, rows):
        """Bundles the device cache evicts (rows: their bytes on the host, [n][bundle_bytes]) into the host tier,
        the bundles kept there longest leaving for them."""
        if not self.retains:
            return
        host = self._arena()
        for key, row in zip(keys, rows):
            key = int(key)
            if key in self._where:
                continue
            if self._free:
                slot = self._free.pop()
            else:
                slot = self._where.pop(next(iter(self._where)))
                self.counted["dropped"] += 1
            host[slot].copy_(row)
            self._where[key] = slot
            self.counted["retained"] += 1

    def stats(self):
        found = {**self.counted, "host_slots": self.slots, "host_held": len(self._where)}
        if self._reads is not None:
            found["reads"] = self._reads.stats()
        return found

    def close(self):
        if self._reads is not None:
            self._reads.close()
            self._reads = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._host = None


class ExpertOffload:
    """The experts slot's device form for experts of any size against the device: load() every expert as for experts
    that fit, open() once the model's pools are made, forward() per routed layer."""

    def __init__(self, source, placement, options, code, context):
        self.source, self.placement, self.options, self.code = source, placement, dict(options), code
        self.context = context
        self.settings = self._settings()
        from crit_plugins.kernels import asks
        self.asked = asks("prompt")   # the level the engine asks of its prompt rows, None where it asks nothing
        self.kernel, self.kernel_reason = None, "no CUDA device"
        self.shape = None          # (layers, experts, format, hidden, inter)
        self.tier = None
        self.misses = None
        self.miss_times = None     # {"cpu": us, "link": us} a missed expert, measured at open() where misses is auto
        self.miss_source = None    # "measured" in this process, or "recorded" in the package's decisions
        self.reasons = {}
        self.bundle_layout = None
        self.a4 = None             # nvfp4: the routed layers' experts carry their input scales (W4A4)
        self.a4_layers = {}        # nvfp4 layer -> whether its experts carry input scales (W4A4) or not (W4A16)
        self.resident = {}         # layer -> device store [experts][bundle_bytes]: W4A16 layers beside W4A4 ones
        self.stores = {}           # device tier: layer -> device store [experts][bundle_bytes]
        self.store = None          # ram / disk tier: the host store
        self.cache = self.dispatch = self.part = None
        self.policy = None         # the cache's residency: recent or ranked beside the CPU tier, clock over the link
        self.core = []             # the pinned core's keys: experts held on the device alone (the pinned option)
        self._peaks = {}         # {a prompt step's largest chunk in tokens: the most device bytes such a step took}
        self._chunk = 0          # the most tokens one prompt forward() ran
        self._frame = None       # the open step frame: [the chunk tokens it has room for, the most a forward() ran,
        #                          what torch's allocator held in use before it (None: no prompt in it so far)]
        self._first = set()      # the chunk sizes of _peaks measured while the engine loaded (learn)
        self._left = 0           # the most a prompt step left in use behind it (a first step's graphs and workspaces)
        self._unsettled = False  # a prompt step ended and the cache has not grown back since
        self._scratch = None     # a scratch of its own, where no engine gives one (a test, a bench)
        self.memory = None       # the device's memory (crit_plugins.memory): the ledger the cache is a giver of
        self.shrinks = 0
        self.loaded = 0
        self._prompt_open = False
        self._last_layer = None
        self.device = None

    # -- the host form: the reference block over the expert_source's matrices ----------------------------------------
    def run(self, layer, x, ids, weights):
        reference = self.code["reference"]
        used = [e for row in ids for e in row if e >= 0]
        if not used:
            return [[0.0] * len(row) for row in x]
        cache = {}

        def matrices(expert):
            if expert not in cache:
                cache[expert] = self.source.matrices(layer, expert)
            return cache[expert]
        return reference.moe_block(x, ids, weights, lambda e: matrices(e)[0], lambda e: matrices(e)[1],
                                   len(matrices(used[0])[0]) // 2)

    # -- sizes and the tier -----------------------------------------------------------------------------------------
    def _settings(self):
        """A tuned settings file (crit plugins offload's offload.json) from the package, or {}."""
        name = self.options.get("settings") or ""
        if not name:
            return {}
        path = Path(name)
        if not path.is_absolute() and getattr(self.context, "package", None) is not None:
            path = Path(self.context.package) / path
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("options", {}).get("expert-offload", value)

    def _option(self, name):
        return self.settings.get(name, self.options[name])

    def bundles(self, format):
        """The bundle code for a format: the CPU tier's for its formats (the device and the CPU read one layout),
        grouped-experts' for the rest."""
        return self.code["tier_bundles"] if format in CPU_FORMATS else self.code["grouped_bundles"]

    def layout_of(self, format, hidden, inter):
        return self.bundles(format).layout(format, hidden, inter)

    def begin(self, layers, experts, format, hidden, inter, device_bytes=0, identity=None, device="cuda"):
        """Before the loads: the routed experts' geometry (layers counts every layer whose experts load(), a draft
        layer's included), the device bytes the model will still allocate (its KV, state and prompt pools), and the
        checkpoint's identity for a store file. Decides the tier and makes its store."""
        import torch
        from crit_plugins.memory import of
        if self.shape is not None:
            raise ValueError("begin() once, before the first load()")
        self.device = torch.device(device)
        self.memory = of(self.device)
        self.shape = (int(layers), int(experts), format, int(hidden), int(inter))
        self.bundle_layout = self.layout_of(format, hidden, inter)
        size = int(self.bundle_layout["bundle_bytes"])
        total = int(layers) * int(experts) * size
        margin = int(self._option("reserve_mib")) << 20
        free = self.memory.free(release=True)
        self.tier = choose_tier(total, free, int(device_bytes), host_available_bytes(), margin,
                                int(self._option("ram_headroom_mib")) << 20, self._option("tier"))
        self.reasons["tier"] = (f"{total / 1e9:.1f} GB of experts; device free {free / 1e9:.1f} GB with "
                                f"{device_bytes / 1e9:.1f} GB still to come" if free is not None else
                                f"{total / 1e9:.1f} GB of experts")
        if self.tier == "device":
            for layer in range(int(layers)):
                self.stores[layer] = torch.zeros(int(experts), size, dtype=torch.uint8, device=self.device)
            if self.placement is not None:
                self.placement.plan({"expert_stores": total}, {"device": free, "host": 0})
            return self.tier
        path = self._store_path(identity)
        if self.tier == "ram":
            self.store = self.source.open_store(self.bundle_layout, int(layers), int(experts), str(self.device),
                                                path=path, identity=self._identity(identity) if path else None)
        else:
            if path is None:
                raise ValueError("the disk tier keeps the experts in a file: give expert-offload a store_dir")
            given = int(self._option("host_tier_mib"))
            room = max(0, host_available_bytes() - (int(self._option("ram_headroom_mib")) << 20))
            self.store = FileStore(self.bundle_layout, layers, experts, path,
                                   host_bytes=room if given < 0 else given << 20)
            self.reasons["host_tier"] = (
                f"{self.store.slots} of {self.store.keys} experts kept in host memory between the file and the "
                "device cache, each expert held once" if self.store.retains
                else "no host tier: the OS's page cache holds what was read from the file")
        return self.tier

    def _check_store(self, path, identity):
        """A store_path names a complete store file of this checkpoint and layout: it is opened as it is, never
        rebuilt (a store that does not match is refused, not overwritten)."""
        store = self.code["store"]
        layers, experts = self.shape[0], self.shape[1]
        layout = store.store_layout(self.bundle_layout, getattr(self.source, "csf", None))
        want = {"kind": store.BUNDLES, "layers": layers, "experts": experts, **store._jsonable(layout),
                "identity": store._jsonable(self._identity(identity))}
        index = Path(str(path) + ".json")
        try:
            found = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError(f"store_path {path} has no readable index ({index.name}): build the store with store_dir")
        if found != want or path.stat().st_size != layers * experts * int(self.bundle_layout["bundle_bytes"]):
            differ = sorted(k for k in set(found) | set(want) if found.get(k) != want.get(k))
            raise ValueError(f"store_path {path} holds another store (index differs in {', '.join(differ) or 'size'}): "
                             "it is opened only as it is, never rebuilt")

    def _identity(self, identity):
        if identity is not None:
            return identity
        config = getattr(self.context, "config", None) or {}
        return {"config": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "model": str(getattr(self.context, "model", ""))}

    def _store_path(self, identity):
        given = self._option("store_path")
        if given:
            self._check_store(Path(given).expanduser(), identity)
            return str(Path(given).expanduser())
        folder = self._option("store_dir")
        if not folder:
            return None
        layers, experts, format, hidden, inter = self.shape
        key = hashlib.sha256(json.dumps([self._identity(identity), layers, experts, format, hidden, inter],
                                        sort_keys=True, default=str).encode()).hexdigest()[:16]
        return str(Path(folder).expanduser() / f"experts-{key}.bin")

    # -- loads ------------------------------------------------------------------------------------------------------
    def layout(self, format, hidden, inter):
        return self.layout_of(format, hidden, inter)

    def pack(self, gate_up, down, format, layer=None):
        return self.bundles(format).pack(gate_up, down, format)

    @property
    def complete(self):
        """A store file reopened whole: the loads may be skipped (the model still calls begin() and open())."""
        return bool(getattr(self.store, "complete", False))

    def load(self, layer, expert, gate_up, down, format="bf16"):
        """One expert's stored tensors, as the experts slot's load() takes them: its bundle into the tier's store."""
        if self.shape is None:
            raise ValueError("call begin(layers, experts, format, hidden, inter) before the first load()")
        if format != self.shape[2]:
            raise ValueError(f"the experts were begun as {self.shape[2]}, not {format}")
        if self.cache is not None:
            raise ValueError("experts load before open()")
        if format == "nvfp4":   # a layer's experts carry their input scales (W4A4) or none does (W4A16)
            carries = all(isinstance(m, dict) and m.get("input_scale") is not None for m in (gate_up, down))
            if self.a4_layers.setdefault(int(layer), carries) != carries:
                raise ValueError(f"layer {layer}'s nvfp4 experts mix W4A4 and W4A16: each carries its input scales "
                                 "or none does")
        bundle = self.pack(gate_up, down, format)
        if self.tier == "device":
            self.stores[int(layer)][int(expert)].copy_(bundle.to(self.device).view(-1))
        elif not self.complete:
            self.store.put(int(layer), int(expert), bundle)
        self.loaded += 1

    # -- open: the cache and the CPU tier ---------------------------------------------------------------------------
    def spare_bytes(self, pool_bytes):
        """The spare bytes to make the model's KV pool with, so the cache takes its own slots there and also borrows
        the pool's pages no sequence holds (open(lend=pool)): every byte the device has free past the pool's own
        `pool_bytes` and margin_mib, whole bundles plus the scratch slot. 0 for the device tier."""
        if self.tier == "device" or self.shape is None:
            return 0
        size = int(self.bundle_layout["bundle_bytes"])
        free = self.memory.free(release=True) - int(pool_bytes)
        slots = min(cache_slots(free, int(self._option("reserve_mib")) << 20, size), self.shape[0] * self.shape[1])
        return (slots + 1) * size if slots > 0 else 0

    def open(self, lend=None):
        """After the model's own pools: the device kernel, and for the ram and disk tiers the cache in every byte left
        (or the spare bytes of `lend`, a KV pool made with spare bytes) and the CPU tier."""
        import torch
        if self.shape is None:
            raise ValueError("begin() and load() the experts before open()")
        if self.part is not None:
            return self
        layers, experts, format, hidden, inter = self.shape
        self._load_kernel()
        self._declared_by_store()
        a4 = set(self.a4_layers.values())
        self.a4 = True in a4 if format == "nvfp4" else False
        if self.tier == "device":
            self.misses = "device"
            self.part = self._part(format, hidden, inter)
            return self
        size = int(self.bundle_layout["bundle_bytes"])
        if lend is not None:
            first, end = lend.lendable()["spare"]
            slots = max(1, (end - first) // size - 1)
        else:
            free = self.memory.free(release=True)
            slots = int(self.settings.get("slots") or 0) or cache_slots(free, int(self._option("reserve_mib")) << 20, size)
            slots = min(slots, layers * experts)
            if self.placement is not None:
                self.placement.plan({"expert_cache": (slots + 1) * size}, {"device": free, "host": 0})
        # Weight-only (W4A16) layers beside W4A4 ones (a draft layer's experts, say): the CPU's order is W4A4, so
        # they stay on the device whole, outside the cache, and the cache keeps the W4A4 layers' slots.
        whole = ([layer for layer, carries in self.a4_layers.items() if not carries]
                 if format == "nvfp4" and a4 == {True, False} else [])
        if whole and not getattr(self.store, "mapped", False):
            raise ValueError("W4A16 layers beside W4A4 ones need a mapped store (the ram tier)")
        if whole:
            slots = max(1, slots - len(whole) * experts - 1)
        self.core = self._leave(slots, whole)   # before anything reads the store's region: that seals it
        for layer in whole:
            rows = torch.empty(experts, size, dtype=torch.uint8)
            self.store.read([layer * experts + e for e in range(experts)], rows)
            self.resident[layer] = rows.to(self.device)
        self.slots = slots
        cpu_ready = self._cpu(format, hidden, inter)
        if cpu_ready and self._option("misses") == "auto":
            # Who computes a miss changes the device rows' arithmetic, so the timed choice is a decision
            # (crit_plugins.decisions): a package that carries its record decides the same in every process.
            from crit_plugins import decisions
            self.misses = decisions.settle(
                "expert-offload", "misses",
                key={"device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
                     "format": format, "hidden": hidden, "inter": inter, "bundle_bytes": size,
                     "workers": self.options["workers"]},
                measure=lambda: self._miss_times(layers, experts, hidden, size) or {},
                decide=lambda found: choose_misses(format, True, "auto", found or None))
            row = decisions.settled("expert-offload", "misses")
            self.miss_times, self.miss_source = row["measurement"] or None, row["source"]
        else:
            self.misses = choose_misses(format, cpu_ready, self._option("misses"))
        if self.miss_times:
            self.reasons["misses"] = (
                f"a missed expert takes the CPU tier {self.miss_times['cpu']:.0f} us and the link "
                f"{self.miss_times['link']:.0f} us here ({self.miss_source}): "
                + ("every miss on the CPU tier; the link copies it into the cache beside the round"
                   if self.misses == "cpu" else "every miss copied over the link"))
        if self.misses == "link" and self.dispatch is not None:
            self.dispatch.close()
            self.dispatch.tier.close()
            self.dispatch = None
        self.part = self._part(format, hidden, inter, declared=self.misses == "cpu")
        cache_code = self.code["cache"]
        # Beside the CPU tier the cache follows the routing (recent, or ranked); over the link every miss is copied
        # on the step's path and the clock picks its slot.
        self.policy = self._option("policy") if self.misses == "cpu" else "clock"
        if self.core and self.policy != "recent":
            raise RuntimeError("the pinned core's experts are held on the device alone and only the recent policy "
                               f"beside the CPU tier keeps them there; here the misses go to the {self.misses} "
                               f"({self.reasons.get('misses', 'no reason given')}): set pinned to 0")
        swaps = int(self._option("swaps"))
        self.cache = cache_code.ExpertCache(
            layers, experts, self.bundle_layout, slots, self.device, self.store, lend, policy=self.policy,
            cpu_tier=self.misses == "cpu", record_routes=int(self._option("record_routes")),
            period=int(self._option("swap_period")), max_swaps=swaps, burst_swaps=4 * swaps,
            fill_blocks=int(self._option("fill_blocks")), side_swaps=bool(self._option("side_swaps")),
            rows=int(self._option("round_rows")), max_slots=self._max_slots(lend), owner="expert-offload")
        if self._elastic():
            self.memory.give(f"expert-offload:{id(self)}", self._release, self._regrow, keep=self._keep)
        if self.core:
            self.cache.pin(self.core)
        profile = self._profile_path(self._option("profile"))
        saved = self._profile_path(self._option("profile_save"))
        start = saved if saved is not None and saved.is_file() else profile
        if start is not None:
            self.cache.warm(self._read_profile(start))
            self.reasons["profile"] = f"cache started from {start.name}"
        return self

    def _leave(self, slots, whole=()):
        """Each expert in one place, decided before the store is sealed: the pinned core (the `pinned` share of the
        cache's `slots` at open(), the experts the `profile` option's file ranks hottest; never profile_save's,
        which follows whatever was served last) and the layers kept `whole` on the device are left out of host
        memory, where the store can read them from its file again (leave()). Returns the core's keys, none where
        nothing is pinned (reasons["pinned"] says why)."""
        layers, experts = self.shape[0], self.shape[1]
        size = int(self.bundle_layout["bundle_bytes"])
        leave = getattr(self.store, "leave", None)
        share, core, why = float(self._option("pinned")), [], None
        path = self._profile_path(self._option("profile"))
        if share > 0:
            if not 0 < share <= 0.9:
                raise ValueError(f"pinned is a share of the cache's slots up to 0.9, not {share}")
            if self._option("misses") != "cpu" or self._option("policy") != "recent":
                why = ("a pinned core is asked for with misses: cpu and the recent policy (misses: auto times the "
                       "CPU tier on the store, which is sealed by then)")
            elif path is None or not path.is_file():
                why = "no profile names the experts to pin"
            elif leave is None or getattr(self.store, "path", None) is None:
                why = "the store has no file to read a pinned expert from: every expert stays in host memory"
            else:
                ranked = self._read_profile(path)
                if not isinstance(ranked, dict):
                    ranked = {"keys": self.code["cache"].residency_profile_keys(ranked, layers, experts)}
                gone = set(whole)
                room = slots - self.code["cache"].ring_for(slots, experts, 0, bool(self._option("elastic"))) - 1
                core = [int(k) for k in ranked["keys"]
                        if int(k) // experts not in gone][:max(0, min(int(share * slots), room))]
                if not core:
                    why = f"the cache's {slots:,} slots hold no core beside its prompt ring"
        if leave is not None and (core or whole):
            left = leave(core + [layer * experts + e for layer in whole for e in range(experts)])
            if core and not set(core) <= left:
                core, why = [], "the store was sealed before the core was decided: every expert stays in host memory"
        if core:
            why = (f"the {len(core):,} experts {path.name} ranks hottest are held on the device alone, in slots no "
                   f"plan takes: {len(core) * size / 2**30:.1f} GiB of host memory the store does not hold")
        if why:
            self.reasons["pinned"] = why
        return core

    def _declared_by_store(self):
        """A store reopened whole needs no load(): each nvfp4 layer's activations (W4A4 where its bundles carry their
        input scales, else W4A16) are read from its first expert's bundle, as a load() of it would declare them."""
        layers, experts, format, hidden, inter = self.shape
        if format != "nvfp4" or not self.complete or not getattr(self.store, "mapped", False):
            return
        bundles = self.code["tier_bundles"]
        for layer in range(layers):
            if layer not in self.a4_layers:
                peek = getattr(self.store, "peek", None)    # from the file: the store is not sealed by it
                first = peek(layer, 0) if peek is not None else None
                gate_up, down = bundles.globals_of(first if first is not None else self.store.bundle(layer, 0),
                                                   self.bundle_layout)
                self.a4_layers[layer] = gate_up > 0 and down > 0

    def _profile_path(self, name):
        if not name:
            return None
        path = Path(name).expanduser()
        if not path.is_absolute() and getattr(self.context, "package", None) is not None:
            path = Path(self.context.package) / path
        return path

    def _read_profile(self, path):
        """The keys a profile names, hottest first: a saved routing profile ({"keys"}), or a residency profile file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("kind") == PROFILE_KIND:
            layers, experts = self.shape[0], self.shape[1]
            if (data.get("layers"), data.get("experts")) != (layers, experts):
                raise ValueError(f"the profile {path} is of {data.get('layers')} x {data.get('experts')} experts, "
                                 f"not {layers} x {experts}")
            return {"keys": data["keys"]}
        return str(path)

    def save_profile(self, path=None):
        """The routing this engine counted, hottest key first (decode routes since open, the resident keys breaking
        ties), written to `path` or the profile_save option: the next open() starts the cache from it."""
        path = Path(path) if path is not None else self._profile_path(self._option("profile_save"))
        if path is None or self.cache is None or not self.cache.directory.route_counts.numel():
            return None
        import torch
        d = self.cache.directory
        counts = d.route_counts[:d.keys].double()
        resident = (d.key_slot[:d.keys] >= 0).double()
        order = torch.argsort(counts * 2 + resident, descending=True, stable=True)
        keys = [int(k) for k, c in zip(order.tolist(), (counts * 2 + resident)[order].tolist()) if c > 0]
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        part.write_text(json.dumps({"kind": PROFILE_KIND, "layers": self.shape[0], "experts": self.shape[1],
                                    "routes": int(d.route_counts[:d.keys].sum()), "keys": keys}), encoding="utf-8")
        os.replace(part, path)
        return path

    def _max_slots(self, lend):
        """The slots an elastic cache reserves address space for (every expert), 0 when it cannot be elastic here
        (a KV pool lends its memory instead, or the driver's virtual memory API is not reachable)."""
        if lend is not None or not self._option("elastic"):
            return 0
        if not self.memory.elastic_ranges:
            self.reasons["elastic"] = "no elastic range here: the cache keeps its size from open()"
            return 0
        return self.shape[0] * self.shape[1]

    # -- the device's memory: the cache is every byte the requests in flight do not need ------------------------------
    def _elastic(self):
        return self.cache is not None and getattr(self.cache, "elastic", None) is not None

    def _release(self, nbytes):
        """The ledger's call (the device memory's need()): give back `nbytes` or more, the coldest experts first.
        Backing goes back by whole granules, so a granule more is asked of the slots."""
        size, elastic = int(self.bundle_layout["bundle_bytes"]), self.cache.elastic
        before, held = self.cache.backed_slots(), elastic.mapped
        after = self.cache.shrink(max(0, before - -(-(int(nbytes) + elastic.gran) // size)))
        self.slots, self.shrinks = after, self.shrinks + (after < before)
        return held - elastic.mapped

    def _regrow(self, room):
        """The ledger's call (the device memory's settle(), which does nothing while the engine loads: reserve_mib
        stays free for what the load still makes): grow into `room` free bytes, never past them (backing comes by
        whole granules, so a granule less is asked for the slots)."""
        size, elastic = int(self.bundle_layout["bundle_bytes"]), self.cache.elastic
        before, held = self.cache.backed_slots(), elastic.mapped
        slots = (int(room) - elastic.gran) // size
        self.slots = self.cache.grow(before + slots) if slots > 0 else before
        if self.slots > before and "fill" not in self.reasons:   # the first growth: Core's serve() after the load
            self.reasons["fill"] = (f"{before:,} -> {self.slots:,} slots once the load was over "
                                    f"({room / 2**30:.2f} GiB were free past what the cache keeps); prompt chunks "
                                    "measured by then: " + (", ".join(
                                        f"{n:,} tokens {p / 2**20:,.0f} MiB" for n, p in sorted(self._peaks.items()))
                                        or "none"))
        return elastic.mapped - held

    def _margin(self):
        return int(self._option("margin_mib")) << 20

    def _room(self, tokens):
        """The device bytes a prompt chunk of `tokens` tokens is given: what such steps were measured to take
        (prompt_bytes), an eighth more and SLACK. A chunk larger than any measured is a first step of its kind: it
        is also given the most such a step left in use behind it so far (graphs, workspaces made once). 0 before any
        step was measured."""
        room = prompt_bytes(self._peaks, tokens)
        if not room:
            return 0
        return room + room // 8 + SLACK + (self._left if tokens > max(self._peaks) else 0)

    def _keep(self):
        """What an elastic cache leaves free on the device (the ledger's keep): margin_mib at every moment, and the
        room of the prompt step in flight, or of a short prompt (max_tokens tokens) when none is, as far as torch's
        allocator does not hold it already: the whole blocks it keeps unused from earlier steps, and the blocks this
        step has taken so far, are that room in use. So a chat turn finds its memory without the cache giving any
        back, a prompt of many steps makes room once, and KV pages a step reserves (the ledger's need) never come
        out of the step's own room. While the engine loads it is margin_mib alone: the load's steps live on
        reserve_mib."""
        if not self.memory.serving:
            return self._margin()
        frame, held = self._frame, self.memory.torch()
        if frame is not None and frame[2] is not None:   # the step's workspace so far: what torch holds over its base
            return self._margin() + max(0, self._room(frame[0]) - (held["reserved"] - frame[2]))
        return self._margin() + max(0, self._room(int(self.options["max_tokens"])) - held["unused"])

    def begin_step(self, prompt_tokens):
        """Core's step frame (crit_plugins.device.STEP_FRAMES; a model outside Core may call it): a step that runs
        prompt tokens takes its temporaries from the device, so an elastic cache first makes room for a chunk of
        its size (_make_room). Decode rows need nothing. Between steps, never captured."""
        self._frame = [0, 0, None]
        if prompt_tokens and self._elastic():
            self._make_room(min(int(prompt_tokens), self._chunk) if self._chunk else int(prompt_tokens))

    def _make_room(self, tokens):
        """The step in flight gets the room of a prompt chunk of `tokens` tokens: the cache gives back what the
        device's free memory and the blocks torch holds of the prompt's earlier steps lack of it (the ledger's
        need, over _keep). The step's workspace is measured from its first call here: every block torch holds over
        what was in use then."""
        frame = self._frame
        if frame[2] is None:
            self.memory.mark()
            held = self.memory.torch()
            frame[2] = held["reserved"] - held["unused"]
        frame[0] = tokens
        self.memory.need(0)

    def _prompt_forward(self, tokens):
        """A prompt chunk of `tokens` tokens reaches the experts: a chunk larger than the frame has room for (a
        model that runs chunks of its own size, a step Core counted no prompt tokens in) gets it before its rows are
        made."""
        if self._frame is None:
            self.begin_step(tokens)   # no frame from Core: this call opens one, the next decode call ends it
        frame = self._frame
        frame[1] = max(frame[1], tokens)
        if tokens > frame[0] and self._elastic():
            self._make_room(tokens)

    def end_step(self, prompts_open=False):
        """Core's step frame: the step returned. A prompt step's staged experts go back to the cache and what its
        temporaries took is learned: the most torch's allocator held during it over what is in use, before it or
        after it (what a first step of its kind leaves behind, graphs and workspaces, is no step's room again), in
        device bytes (its blocks, not the tensors in them: a step's tensors come and go in blocks of other sizes).
        Once no prompt is in flight the blocks torch's allocator kept of it go back to the device and an elastic
        cache grows back into the free memory, less what it keeps (_keep): the next prompt starts on an empty
        allocator, so its peak is its own and not the blocks an earlier, larger one left (kept, they would count as
        every later prompt's room, and the cache would leave that much free for good). A step without prompt tokens
        changes nothing and costs nothing."""
        frame, self._frame = self._frame, None
        if self._prompt_open and self.cache is not None:
            self.cache.end_prompt()
        self._prompt_open, self._last_layer = False, None
        if not self._elastic():
            return
        if frame is not None and frame[2] is not None:
            if frame[1]:
                held = self.memory.torch()
                in_use = held["reserved"] - held["unused"]
                self._left = max(self._left, in_use - frame[2])
                peak = max(0, self.memory.peak() - max(frame[2], in_use))
                learn(self._peaks, self._first, frame[1], peak, self.memory.serving)
                self._chunk = max(self._chunk, frame[1])
            self._unsettled = True
        if self._unsettled and not prompts_open and self.memory.serving:
            self._unsettled = False
            self.memory.release()
            self.memory.settle()

    def _load_kernel(self):
        if self.kernel is not None or self.device.type != "cuda":
            return
        try:
            from crit_plugins.kernels import load_kernel
            self.kernel, self.kernel_reason = load_kernel("grouped-experts", folder=self.code["folder"]), None
        except Exception as error:   # the plugin refuses below: there is no exact fallback fast enough to serve
            self.kernel, self.kernel_reason = None, str(error).splitlines()[0] if str(error) else type(error).__name__
        if self.kernel is None:
            raise RuntimeError(f"expert-offload runs on crit's grouped-experts kernel, which is not here "
                               f"({self.kernel_reason or 'it does not build on this machine'})")

    def _part(self, format, hidden, inter, declared=False):
        dispatch = self.code["dispatch"]
        # A call's route rows are scratch (the engine's: Core's): the prompt chunk sizes a server meets cost the
        # largest call's rows, not a set each, in bytes every other plugin's calls compute in too.
        scratch = getattr(self.context, "scratch", None)
        if scratch is None:
            scratch = self._scratch = self._scratch or self.memory.scratch("expert-offload")
        part = dispatch.KernelRoutes(self.kernel, self.bundle_layout, self.code["grouped_bundles"].kernel_layout,
                                     format, hidden, inter, a4=self.a4, scratch=scratch)
        part.decode_arithmetic = "declared" if declared else "tensor"
        return part

    def _cpu(self, format, hidden, inter):
        """Whether the CPU tier and its in-stream join are built here (then self.dispatch holds them)."""
        o, c = self.options, self.code
        if format == "nvfp4" and not self.a4:
            self.reasons["misses"] = "the CPU tier's integer order is W4A4: these nvfp4 experts carry no input scales"
            return False
        if format not in CPU_FORMATS or self._option("misses") == "link" or not getattr(self.store, "mapped", False):
            self.reasons["misses"] = ("the CPU tier computes nvfp4 and int4 experts" if format not in CPU_FORMATS
                                      else "misses pinned to the link" if self._option("misses") == "link"
                                      else "a file tier's misses are read on the host")
            return False
        try:
            tier = c["cpu_tier"].CpuTier(hidden, inter, format, c["arithmetic"], c["quantized"], native=True,
                                         workers=o["workers"], pin=o["pin"])
        except Exception as error:   # no native tier here: misses over the link
            self.reasons["misses"] = f"the native CPU tier does not build: {str(error).splitlines()[0]}"
            return False
        layers, experts = self.shape[0], self.shape[1]
        host, _ = self.store.region()
        flat = host.view(-1)
        size = int(self.bundle_layout["bundle_bytes"])
        left = getattr(self.store, "left", ())   # held on the device alone: never a miss, and not in host memory
        for layer in range(layers):
            for expert in range(experts):
                if layer * experts + expert in left:
                    continue
                at = (layer * experts + expert) * size
                tier.add(layer * experts + expert, flat[at:at + size])
        ext, reason = c["dispatch"].extension()
        if ext is None:
            tier.close()
            self.reasons["misses"] = f"the in-stream join does not build: {reason}"
            return False
        part = self._part(format, hidden, inter, declared=True)
        self.dispatch = c["dispatch"].DeviceDispatch(
            self.device, hidden, inter, format, tier, part, ext, experts, set(range(layers)),
            max_tokens=int(o["max_tokens"]), prompt_arithmetic=self._asked_arithmetic(),
            owner="expert-offload")
        # The CPU tier is ready: open() times a missed expert on it and over the link and takes the sooner
        # (choose_misses), unless misses pins one.
        self.reasons["misses"] = "every miss on the CPU tier; the link copies it into the cache beside the round"
        return True

    def _miss_times(self, layers, experts, hidden, size, count=16, trials=3):
        """{"cpu": us, "link": us} a missed expert on this machine now: the CPU tier computing `count` experts spread
        over the layers for one token each, and the link copying their bundles to the device, the median of
        `trials` such batches of other experts (a first batch wakes the tier's threads and is not counted). None
        where either cannot be timed (no mapped store on a CUDA device)."""
        import time
        import torch
        try:
            _, mapped = self.store.region()
        except Exception:   # a store the device cannot read in place
            return None
        if mapped is None or self.device.type != "cuda" or self.dispatch is None:
            return None
        tier, flat, total = self.dispatch.tier, mapped.view(-1), layers * experts   # the link as the cache's fills read it
        x = torch.randn(1, hidden, generator=torch.Generator().manual_seed(0))
        token = torch.zeros(count, dtype=torch.int32)
        target = torch.empty(count, size, dtype=torch.uint8, device=self.device)
        cpu, link = [], []
        for trial in range(trials + 1):
            keys = torch.tensor([((trial * count + i) * 9973) % total for i in range(count)], dtype=torch.int32)
            started = time.perf_counter()
            tier.compute(keys, token, x)
            cpu.append((time.perf_counter() - started) * 1e6 / count)
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            for i, key in enumerate(keys.tolist()):
                target[i].copy_(flat[key * size:(key + 1) * size], non_blocking=True)
            torch.cuda.synchronize(self.device)
            link.append((time.perf_counter() - started) * 1e6 / count)
        del target
        tier.reset_stats()
        return {"cpu": sorted(cpu[1:])[trials // 2], "link": sorted(link[1:])[trials // 2]}

    # -- the routed layer -------------------------------------------------------------------------------------------
    def forward(self, layer, x, ids, weights, rows=None, prompt=None):
        """out [tokens][hidden] FP32: the routed SwiGLU block of each token over its routes. prompt: a prompt chunk
        (stages the layer's misses, admits nothing); None takes a call of more than max_tokens tokens for one. rows:
        the batch's row of each token in a captured round (its per-row reads and hits)."""
        if self.part is None:
            self.open()
        layer = int(layer)
        keep = int(self.options.get("keep_routes") or 0)
        if keep and ids.dim() == 2 and ids.shape[1] > keep:
            # An approximation of the model, asked for: each token's routes past the router's first `keep` are left
            # out (not renormalized), in every tier and for decode rows and prompt chunks alike, so a row's result
            # stays a function of its own routes and never of what the cache holds.
            ids, weights = ids[:, :keep].contiguous(), weights[:, :keep].contiguous()
        if self.tier == "device":
            return self._rows(layer, x, ids, weights, self.stores[layer], None, "tensor")
        if layer in self.resident:
            return self._rows(layer, x, ids, weights, self.resident[layer], None, "tensor")
        if prompt is None:
            prompt = ids.shape[0] > int(self.options["max_tokens"])
        if prompt:
            if self._prompt_open and self._last_layer is not None and layer <= self._last_layer:
                self.end_step(True)   # the next prompt step began without a frame from Core
            self._prompt_forward(ids.shape[0])
            self._prompt_open, self._last_layer = True, layer
            # The device part's rows first: a shape's first call grows the scratch, which takes device memory the
            # cache gives by releasing slots, and the chunk's experts must not be staged in them yet.
            self._a4(layer)
            self.part.prepare(ids.shape[0], ids.shape[1], self.shape[1], x.device, self._prompt_arithmetic())
            store, slots = self.cache.prompt(layer, ids)
            return self._rows(layer, x, ids, weights, store, slots, self._prompt_arithmetic())
        if self._prompt_open:
            self.end_step(False)   # a prompt step no frame closed: this decode call ends it
        if self.dispatch is not None:
            self.part.a4 = True
            decided = (lambda table: self.dispatch.post(layer, x, ids, table))
            slots = self.cache.plan(layer, ids, rows=rows, decided=decided)
            return self.dispatch.forward(layer, x, ids, weights, self.cache.store, slots)
        slots = self.cache.plan(layer, ids, rows=rows)
        return self._rows(layer, x, ids, weights, self.cache.store, slots, "tensor")

    def _asked_arithmetic(self):
        """The prompt_arithmetic option, declared or tensor. auto is the faster one that keeps the level the engine
        asks of its prompt rows (crit_plugins.kernels.asks, the configuration's exactness): tensor where it asks
        repeatable or row-exact (the kernel's tensor cores: a row's bits its own at every row count, not the decode
        rows'), declared where it asks exact or declared (the CPU's integer order, the decode rows' bits) and where it
        asks nothing, as before the option had auto."""
        chosen = self._option("prompt_arithmetic")
        if chosen == "auto":
            chosen = "tensor" if self.asked in ("repeatable", "row-exact") else "declared"
        return chosen

    def _prompt_arithmetic(self):
        """A prompt chunk's rows: the CPU's order only where the CPU computes decode misses and the option keeps it."""
        return self._asked_arithmetic() if self.misses == "cpu" else "tensor"

    def exactness(self):
        """The level its rows keep once open() decided who serves a miss (crit_kernels.exactness; Core compares it
        with the level the configuration asks). Where the CPU computes decode misses every decode row is the CPU's
        integer order (declared) and a prompt row is that order too, or with prompt_arithmetic tensor the kernel's
        tensor arithmetic: the same bits at every row count, not the decode rows' (row-exact). With misses over the
        link, or every expert on the device, every row is the kernel's tensor arithmetic: decode rows row-exact, a
        prompt row the bits of the same route decoded (exact). Nothing before open()."""
        if self.misses is None:
            return {}
        if self.misses == "cpu":
            return {"decode": "declared", "prompt": "declared" if self._prompt_arithmetic() == "declared" else "row-exact"}
        return {"decode": "row-exact", "prompt": "exact"}

    def _rows(self, layer, x, ids, weights, store, slots, arithmetic):
        if slots is None:
            slots = self._identity_slots(store.shape[0])
        self._a4(layer)
        y = self.part.route_outputs(layer, x, ids, store, slots, arithmetic)
        return self.part.combine(y, ids, weights)

    def _a4(self, layer):
        """The device part reads `layer` as its experts are stored: W4A4 where its nvfp4 bundles carry input scales."""
        if self.shape[2] == "nvfp4":
            self.part.a4 = self.a4_layers.get(layer, self.a4)

    def _identity_slots(self, n):
        import torch
        found = getattr(self, "_identity_table", None)
        if found is None or found.numel() != n:
            found = self._identity_table = torch.arange(n, dtype=torch.int32, device=self.device)
        return found

    # -- rounds -----------------------------------------------------------------------------------------------------
    def start_round(self, view):
        if self.cache is not None:
            self.cache.start_round(view)

    def count_round(self, view):
        if self.cache is not None:
            self.cache.count_round(view)

    def round_problems(self):
        if self.part is None:
            return ["open() the experts before capturing rounds"]
        return self.cache.round_problems() if self.cache is not None else []

    # -- what serves, and how it runs -------------------------------------------------------------------------------
    def routes(self):
        """The recorded decode routes (record_routes), the trace crit plugins offload --trace replays."""
        return self.cache.routes() if self.cache is not None else []

    def facts(self):
        if self.shape is None:
            return {"expert_offload": {"tier": None}}
        layers, experts, format, hidden, inter = self.shape
        out = {"tier": self.tier, "misses": self.misses, "format": format, "layers": layers, "experts": experts,
               "bundle_bytes": int(self.bundle_layout["bundle_bytes"]), "reasons": dict(self.reasons),
               "prompt_arithmetic": self._prompt_arithmetic() if self.misses is not None else None,
               "decode_arithmetic": getattr(self.part, "decode_arithmetic", None)}
        if int(self.options.get("keep_routes") or 0):
            out["keep_routes"] = int(self.options["keep_routes"])
            out["reasons"]["keep_routes"] = (f"each token keeps the router's first {out['keep_routes']} routes: an "
                                             "approximation of the model, which the run's drift budget judges")
        if self.cache is not None:
            out.update({"slots": self.slots, "elastic": getattr(self.cache, "elastic", None) is not None,
                        "prompt_bytes": dict(sorted(self._peaks.items())), "shrinks": self.shrinks,
                        "policy": self.policy, "settings": bool(self.settings)})
        return {"expert_offload": out}

    def telemetry(self):
        out = {"tier": self.tier, "misses": self.misses}
        if self.miss_times:
            out["miss_times_us"] = {k: round(v, 1) for k, v in self.miss_times.items()}
            out["miss_times_from"] = self.miss_source
        if self.cache is not None:
            out["expert_cache"] = self.cache.telemetry()
            if self._elastic():   # the cache's size now, the times it gave memory back, what a prompt token takes
                out["elastic"] = {"slots": self.cache.backed_slots(), "shrinks": self.shrinks,
                                  "prompt_bytes": dict(sorted(self._peaks.items()))}
        if getattr(self.store, "left", None):    # experts in one place: what the host store does not hold
            out["host_store"] = {"left": len(self.store.left), "left_bytes": int(self.store.left_bytes)}
        if isinstance(self.store, FileStore):   # the disk tier: the host tier's hits, the file's reads
            out["file_store"] = self.store.stats()
        if self.dispatch is not None:
            out["cpu_routes"] = self.dispatch.status()
            out["cpu_tier"] = self.dispatch.tier.stats()
        device = self._device_shares()
        if device is not None:
            out["device"] = device
        out["diagnosis"] = diagnosis(out)
        trace = self._option("trace_path")
        if trace and self.cache is not None:
            Path(trace).write_text(json.dumps(self.routes()), encoding="utf-8")
        return {"expert_offload": out}

    def _device_shares(self):
        """Who holds the device's memory now, from Core's reading of it (crit_plugins.memory), in bytes: the
        cache's slots, torch's allocator beside them (the model's weights, pools and workspaces; `holders`: the
        three plugins that hold most of it, and `tensors`: its three largest tensors, each by shape, dtype and
        plugin, both by the census Core takes when the load is over), what is free; and a bundle's bytes. None
        without a cache on a device whose memory can be read."""
        report = self.memory.report() if self.cache is not None and self.memory is not None else None
        if report is None:
            return None
        size, reserved = int(self.bundle_layout["bundle_bytes"]), report["torch"]["reserved"]
        if self._elastic():   # the slots are an elastic range, beside torch's allocator
            cache, model = self.cache.elastic.mapped, reserved
        else:                 # the slots are a torch allocation (the cache's own, or a KV pool's spare bytes)
            cache = (self.cache.backed_slots() + 1) * size
            model = max(0, reserved - cache)
        holders = sorted((report.get("plugins") or {}).items(), key=lambda item: -item[1])[:3]
        return {"total": report["total"], "cache": int(cache), "model": int(model), "free": report["free"],
                "bundle": size, "holders": [[plugin, int(held)] for plugin, held in holders],
                "tensors": (report.get("tensors") or [])[:3]}

    def close(self):
        try:
            self.save_profile()
        except Exception:   # a profile that cannot be written never keeps the engine from closing
            pass
        if self._elastic():
            self.memory.ungive(f"expert-offload:{id(self)}")
        if self.dispatch is not None:
            self.dispatch.close()
            self.dispatch.tier.close()
            self.dispatch = None
        self.part = None
        scratch, self._scratch = self._scratch, None
        if scratch is not None:
            scratch.close()
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        if self.store is not None:
            self.store.close()
            self.store = None
