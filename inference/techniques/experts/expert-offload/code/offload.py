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
    short     a prompt chunk of fewer routes than a layer has experts, where the CPU tier computes decode misses
    prompts   and prompt rows keep its order: its missed routes on the CPU tier where they cost less there than
              their experts over the link (both measured at open()), else copied into the ring as below
    prompts   a prompt chunk stages its layer's non-resident experts through the cache's ring and runs in the tensor
              arithmetic (prompt_arithmetic) or the CPU's order. A staged layer serves every chunk that reads it: a
              model that runs a span of a prompt layer by layer (stages_prompts tells it to) brings each layer's
              experts over the link once for the span, however many chunks and steps it takes

Nothing here is new arithmetic: the cache is device-expert-cache's ExpertCache, the store expert-tensors' BundleStore,
the CPU tier host-expert-tier's CpuTier behind heterogeneous-expert-dispatch's DeviceDispatch, the rows grouped-experts'
kernel (KernelRoutes), or for experts a GGUF checkpoint stores (format "gguf": each layer's formats as the file has
them, gguf_experts.py) gguf-rows' kernel over the bundles in their slots (GgufRoutes) and, for their misses, its
declared order on the host (GgufCpuTier)."""
import hashlib
import json
import os
from pathlib import Path

from crit_plugins.memory import COPIES   # the cache's rank in the device's eviction order: copies, whose loss costs speed

GIB = 1 << 30
# Files of a package's plugins that make a bundle what it is (the model plugin's reading of the checkpoint, the experts
# plugins' packing): a store in the host's cache is keyed by them.
CODE_SUFFIXES = (".py", ".json", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp")
# Filesystems a store file is not kept on unasked: every miss would read it over a network or a VM's file bridge.
REMOTE_FILESYSTEMS = ("9p", "drvfs", "nfs", "nfs4", "cifs", "smb3", "fuse.sshfs", "virtiofs")
CPU_FORMATS = ("nvfp4", "int4")
# The formats whose misses the CPU tier computes: its own bundles' (native), and a GGUF's experts as stored (in torch,
# gguf_experts.GgufCpuTier: gguf-rows' declared order on the host, joined after a synchronization)
MISS_FORMATS = CPU_FORMATS + ("gguf",)
PROFILE_KIND = "expert-offload-routing"


def host_available_bytes():
    """Host memory available now, as Core reads it (crit_plugins.host): 0 where the OS does not say."""
    from crit_plugins.host import memory_bytes
    return memory_bytes()["available"]


# The slots whose plugins' code makes an expert store's bytes: the model plugin reads the checkpoint's experts and
# hands them to load(), the expert source holds them, the experts plugin packs them.
PACKING_SLOTS = ("model", "expert_source", "experts")


def packing_folders(plugins, configuration):
    """The folders of code under `plugins` (a package's plugins folder) that make an expert store's bytes: the
    PACKING_SLOTS plugins the configuration's selection names and, through their plugin.json sources, every folder
    of code they import (another plugin's folder with its own sources too), resolved, in order. None where the
    selection does not name a plugin of each of those slots, or one of them is not in the package."""
    use = ((configuration or {}).get("plugins") or {}).get("use") or {}
    wanted = [use.get(slot) for slot in PACKING_SLOTS]
    if not all(isinstance(name, str) and name for name in wanted):
        return None
    found = {}
    for manifest in sorted(Path(plugins).rglob("plugin.json")):
        if "__pycache__" in manifest.parts:
            continue
        try:
            name = json.loads(manifest.read_text(encoding="utf-8")).get("id")
        except (OSError, ValueError):
            continue
        found.setdefault(name, manifest.parent.resolve())
    if not all(name in found for name in wanted):
        return None
    folders, todo = [], [found[name] for name in wanted]
    while todo:
        folder = todo.pop(0)
        if folder in folders:
            continue
        folders.append(folder)
        try:
            sources = json.loads((folder / "plugin.json").read_text(encoding="utf-8")).get("sources") or {}
        except (OSError, ValueError):
            sources = {}
        todo += [(folder / relative).resolve() for relative in sources.values() if isinstance(relative, str)]
    return folders


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


def _filesystem(folder):
    """The type of the filesystem `folder` is on, from /proc/mounts (its longest mount point); None where that
    cannot be read (not Linux)."""
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    folder, best = Path(folder).resolve(), (-1, None)
    for line in mounts:
        fields = line.split()
        if len(fields) >= 3 and (folder == Path(fields[1]) or Path(fields[1]) in folder.parents):
            best = max(best, (len(fields[1]), fields[2]))
    return best[1]


def cache_slots(free, margin, bundle_bytes):
    """Own slots of a device expert cache in every byte the device has free past the margin (a scratch slot beside)."""
    return max(0, (int(free) - int(margin)) // int(bundle_bytes) - 1)


# The CPU tier takes the misses where it computes a missed expert in under this share of the time the link copies
# one: beside the CPU tier every device row runs the declared order, slower than the tensor cores', and the tier
# needs its threads awake, so at equal times the link serves a round sooner and starts no slower after a pause.
# A short prompt chunk's misses go by the same share (the cache's prompt_cpu).
CPU_WINS = 0.8


def choose_misses(format, cpu_ready, forced="auto", measured=None):
    """cpu or link. The link where the CPU tier does not compute the format or is not built here (the native tier
    and the in-stream join), or where it is asked for. Else, on auto, whichever serves a missed expert sooner on
    this machine: measured {"cpu": us, "link": us} a missed expert (the tier computing it, the link copying it),
    the CPU tier where it takes under CPU_WINS of the link's time; without a measurement, and where it is asked
    for, the CPU tier."""
    if forced == "link" or format not in MISS_FORMATS or not cpu_ready:
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

    # Its bundles are packed on the host, whatever the tier: a model plugin hands load() host tensors (an expert
    # read onto the device first crossed the link twice on its way to host memory, 25,000 times a load).
    host_loads = True

    def __init__(self, source, placement, options, code, context):
        self.source, self.placement, self.options, self.code = source, placement, dict(options), code
        self.context = context
        self.settings = self._settings()
        from crit_plugins.kernels import asks, quality_mode
        self.asked = asks("prompt")   # the level the engine asks of its prompt rows, None where it asks nothing
        self.speed = quality_mode() == "speed"   # the device rows on the format's own tensor cores (_tensor)
        self.kernel, self.kernel_reason = None, "no CUDA device"
        self.shape = None          # (layers, experts, format, hidden, inter)
        self.tier = None
        self.misses = None
        self.miss_times = None     # {"cpu": us, "link": us} a missed expert, measured at open() where misses is auto
        self.miss_source = None    # "measured" in this process, or "recorded" in the package's decisions
        self.miss_trials = None    # each timed batch's microseconds a missed expert, the uncounted first included
        self.link_us = None        # microseconds the link takes a bundle, measured at open() (a mapped store on CUDA)
        self.link_trials = None    # each timed batch's microseconds a bundle over the link, the uncounted first included
        self.activation = "silu"   # the blocks' gate (activate): silu; gelu_tanh or softcapped_silu (caps) in the
        self.caps = None           # tensor arithmetic
        self.reasons = {}
        self.bundle_layout = None
        self.formats = None        # gguf: {layer: (gate_up type, down type)}, as the checkpoint stores each layer
        self.a4 = None             # nvfp4: the routed layers' experts carry their input scales (W4A4)
        self.a4_layers = {}        # nvfp4 layer -> whether its experts carry input scales (W4A4) or not (W4A16)
        self.resident = {}         # layer -> device store [experts][bundle_bytes]: layers kept whole on the device
        self.link_layers = []      # W4A16 layers beside W4A4 ones: the cache's link layers (their misses over the link)
        self.decode_routes = None  # the most routes a decode round makes in a layer (begin: decode_rows x top)
        self.stores = {}           # device tier: layer -> device store [experts][bundle_bytes]
        self.store = None          # ram / disk tier: the host store
        self.cache = self.dispatch = self.part = None
        self.policy = None         # the cache's residency: recent or ranked beside the CPU tier, clock over the link
        self.core = []             # the pinned core's keys: experts held on the device alone (the pinned option)
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
                                   len(matrices(used[0])[0]) // 2, activation=self.activation, caps=self.caps)

    def activate(self, name, caps=None):
        """The gate of the blocks' product, act(gate) * up: silu (SwiGLU, the default), gelu_tanh or softcapped_silu
        with caps (gate, up) (experts:activations), before open(). The CPU tier computes SwiGLU blocks alone, so a
        block gated otherwise has its misses copied over the link and every row in the kernel's tensor arithmetic
        (GGUF experts: silu and gelu_tanh)."""
        known = self.code["grouped_bundles"].ACTIVATIONS
        if name not in known:
            raise ValueError(f"expert-offload gates a block by {' or '.join(known)}, not {name!r}")
        if (name == "softcapped_silu") != (caps is not None) or (caps is not None and (len(caps) != 2
                                                                                         or min(caps) <= 0)):
            raise ValueError("expert-offload: softcapped_silu, and only it, takes two positive caps: (gate, up)")
        if self.part is not None:
            raise ValueError("the blocks' activation is set before open()")
        self.activation, self.caps = name, None if caps is None else (float(caps[0]), float(caps[1]))

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
        if format == "gguf":
            return self.code["gguf"].layout(self.formats, hidden, inter)
        return self.bundles(format).layout(format, hidden, inter)

    def begin(self, layers, experts, format, hidden, inter, device_bytes=0, identity=None, device="cuda",
              formats=None, decode_rows=None, top=None):
        """Before the loads: the routed experts' geometry (layers counts every layer whose experts load(), a draft
        layer's included), the device bytes the model will still allocate (its KV, state and prompt pools), and the
        checkpoint's identity for a store file. Format "gguf" (a GGUF checkpoint's experts as stored) takes each
        layer's formats, {layer: (gate_up type, down type)}. decode_rows and top: the most rows one decode round runs
        through a layer, every plan of the round counted (a captured round lists its victims once, at its start), and
        the routes a row takes: they bound the slots a link layer's misses take a round (link_reserve()). Decides the
        tier and makes its store."""
        import torch
        from crit_plugins.memory import of
        if self.shape is not None:
            raise ValueError("begin() once, before the first load()")
        if decode_rows and top:
            self.decode_routes = int(decode_rows) * int(top)
        if format == "gguf":
            self.formats = {int(k): tuple(v) for k, v in dict(formats or {}).items()}
            if sorted(self.formats) != list(range(int(layers))):
                raise ValueError(f"GGUF experts take the formats of every layer 0 .. {int(layers) - 1}, not of "
                                 f"{sorted(self.formats)}")
        elif formats is not None:
            raise ValueError("formats are a GGUF checkpoint's (format gguf)")
        self.device = torch.device(device)
        self.memory = of(self.device)
        self.shape = (int(layers), int(experts), format, int(hidden), int(inter))
        self.bundle_layout = self.layout_of(format, hidden, inter)
        size = int(self.bundle_layout["bundle_bytes"])
        sizes = self.bundle_layout.get("sizes") or {layer: size for layer in range(int(layers))}
        total = int(experts) * sum(sizes.values())
        margin = int(self._option("reserve_mib")) << 20
        free = self.memory.free(release=True)
        # A store kept from the engine before this one (the source's kept_bytes) is host memory this one has.
        kept = int(getattr(self.source, "kept_bytes", lambda: 0)())
        self.tier = choose_tier(total, free, int(device_bytes), host_available_bytes() + kept, margin,
                                int(self._option("ram_headroom_mib")) << 20, self._option("tier"))
        self.reasons["tier"] = (f"{total / 1e9:.1f} GB of experts; device free {free / 1e9:.1f} GB with "
                                f"{device_bytes / 1e9:.1f} GB still to come" if free is not None else
                                f"{total / 1e9:.1f} GB of experts")
        if kept and self.tier != "ram":   # the kept store is not this engine's: its memory goes first
            getattr(self.source, "drop_kept", lambda: None)()
        if self.tier == "device":
            for layer in range(int(layers)):
                self.stores[layer] = torch.zeros(int(experts), max(size, sizes[layer]), dtype=torch.uint8,
                                                 device=self.device)
            if self.placement is not None:
                self.placement.plan({"expert_stores": total}, {"device": free, "host": 0})
            return self.tier
        path = self._store_path(identity)
        if self.tier == "ram":
            # Kept for the next engine built for the model in this host (the source's open_store kept=): an ablation's
            # variant or Eval's next engine gets the store sealed and skips the loads, as from a complete store file.
            self.store = self.source.open_store(self.bundle_layout, self.store_layers, int(experts),
                                                str(self.device), path=path,
                                                identity=self._identity(identity) if path else None,
                                                kept=self._content(identity) + [self._code()])
            if getattr(self.store, "kept", False) and self.store.sealed:
                self.reasons["store"] = "kept in host memory from the engine built before this one for the model"
        else:
            if path is None:
                raise ValueError("the disk tier keeps the experts in a file: give expert-offload a store_dir")
            given = int(self._option("host_tier_mib"))
            room = max(0, host_available_bytes() - (int(self._option("ram_headroom_mib")) << 20))
            self.store = FileStore(self.bundle_layout, self.store_layers, experts, path,
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
        layers, experts = self.store_layers, self.shape[1]
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
        experts = self.shape[1]
        parts = self._content(identity)

        def key(parts):
            return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
        folder = self._option("store_dir")
        if folder:
            Path(folder).expanduser().mkdir(parents=True, exist_ok=True)   # a folder named and not there yet
            return str(Path(folder).expanduser() / f"experts-{key(parts)}.bin")
        # No folder named: the host's cache for files a plugin can build again (context.cache: crit run gives every
        # run one), so the loads after the first reopen the store instead of building it from the checkpoint again.
        cache = getattr(self.context, "cache", None)
        if cache is None:
            return None
        path = Path(cache) / "expert-offload" / f"experts-{key(parts + [self._code()])}.bin"
        nbytes = self.store_layers * experts * int(self.bundle_layout["bundle_bytes"])
        return str(path) if self._cached(path, nbytes, self._identity(identity)) else None

    def _content(self, identity):
        """What the store's bundles hold, beside the plugin code that packs them: the checkpoint and the experts'
        geometry and format."""
        layers, experts, format, hidden, inter = self.shape
        parts = [self._identity(identity), layers, experts, format, hidden, inter]
        if self.formats is not None:   # another quantization of the checkpoint is another store
            parts.append(sorted(self.formats.items()))
        return parts

    def _code(self):
        """A digest of the code that makes the store's bytes (CODE_SUFFIXES): the package's plugins in the slots
        PACKING_SLOTS (the model plugin that reads the experts, the expert source, this experts plugin) and every
        folder of code they take as sources, their tests apart. A store another model plugin read or another
        experts plugin packed is another store; a candidate that changes only other plugins (a sampler, a graphs
        plugin) reopens it. Where the selection does not name those plugins, every plugin's code. Empty outside a
        package."""
        package = getattr(self.context, "package", None)
        plugins = Path(package) / "plugins" if package is not None else None
        if plugins is None or not plugins.is_dir():
            return ""
        folders = packing_folders(plugins, (getattr(self.context, "configuration", None) or {}))
        digest = hashlib.sha256()
        for folder in folders or [plugins]:
            for path in sorted(p for p in folder.rglob("*") if p.suffix in CODE_SUFFIXES and p.is_file()
                               and "__pycache__" not in p.parts
                               and (not folders or "tests" not in p.relative_to(folder).parts)):
                digest.update(path.relative_to(plugins).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
        return digest.hexdigest()

    def _cached(self, path, nbytes, identity):
        """Whether the store is kept at `path` in the host's cache. The cache holds one store a checkpoint: this
        checkpoint's other stores there (built by other plugin code or in another layout) and files a build left
        unfinished are removed first. A store already there is reopened; a new one is written only where the folder
        is on a local disk and the machine's plugin cache ledger (crit_plugins.disk_cache) makes room for it: under
        its cap, with its floor of free space left, the least recently used stores of any run removed first.
        Otherwise the store is built in host memory, as without a cache, and the reason says why."""
        from crit_plugins import disk_cache
        store = self.code["store"]
        folder = path.parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
            for other in sorted(folder.glob("experts-*.bin")):
                if other == path:
                    continue
                index = Path(str(other) + ".json")
                try:
                    stale = json.loads(index.read_text(encoding="utf-8")).get("identity") == store._jsonable(identity)
                except (OSError, ValueError):
                    stale = True   # no index: a build that did not finish
                if stale:
                    disk_cache.forget(other)
                    for gone in (index, other):
                        gone.unlink(missing_ok=True)
            index = Path(str(path) + ".json")
            if path.is_file() and index.is_file():
                disk_cache.touch(path, [index])
                self.reasons["store"] = f"reopened from the host's cache ({path}) where it is complete"
                return True
            kind = _filesystem(folder)
            if kind in REMOTE_FILESYSTEMS:
                self.reasons["store"] = (f"built in host memory at every load: the host's cache ({folder}) is on a "
                                         f"{kind} filesystem; name a folder on a local disk with store_dir")
                return False
            admitted, reason = disk_cache.admit(path, nbytes, [index])
            if not admitted:
                self.reasons["store"] = f"built in host memory at every load: {reason}"
                return False
        except OSError as error:
            self.reasons["store"] = f"built in host memory at every load: the host's cache is not usable ({error})"
            return False
        self.reasons["store"] = f"kept in the host's cache ({path}, {nbytes / GIB:.1f} GiB): later loads reopen it"
        return True

    @property
    def store_layers(self):
        """The store's layers: the model's, and for GGUF experts the extra layers an outsized layer's pieces take."""
        return int((self.bundle_layout or {}).get("store_layers") or self.shape[0])

    def outsized(self):
        """The layers whose bundles are larger than the slot (GGUF experts in a wider format): kept whole on the
        device at their own size, outside the cache (gguf_experts.layout)."""
        return sorted((self.bundle_layout or {}).get("pieces") or ())

    # -- loads ------------------------------------------------------------------------------------------------------
    def layout(self, format, hidden, inter):
        return self.layout_of(format, hidden, inter)

    def pack(self, gate_up, down, format, layer=None):
        if format == "gguf":
            return self.code["gguf"].pack(gate_up, down, self.bundle_layout, layer)
        return self.bundles(format).pack(gate_up, down, format)

    @property
    def complete(self):
        """A store file reopened whole, or a kept store an earlier engine sealed: the loads may be skipped (the model
        still calls begin() and open())."""
        return bool(getattr(self.store, "complete", False) or getattr(self.store, "sealed", False))

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
        bundle = self.pack(gate_up, down, format, layer)
        if self.tier == "device":
            row = self.stores[int(layer)][int(expert)]
            row.copy_(bundle.to(self.device).view(-1)[:row.numel()])
        elif not self.complete and format == "gguf":   # an outsized layer's bundle in pieces of a slot each
            for where, piece in self.code["gguf"].pieces(bundle, self.bundle_layout, layer):
                self.store.put(where, int(expert), piece)
        elif not self.complete:
            self.store.put(int(layer), int(expert), bundle)
        self.loaded += 1
        if not self.loaded % 64:
            self._working("expert store: experts packed", self.loaded, self.shape[0] * self.shape[1])

    def _working(self, what, done=None, total=None):
        """What a long step of the load is doing, for whoever serves the engine (LoadContext.working: crit serve
        prints it, and does not time out a load whose work advances)."""
        report = getattr(self.context, "working", None)
        if callable(report):
            report(what, done, total)

    # -- open: the cache and the CPU tier ---------------------------------------------------------------------------
    def spare_bytes(self, pool_bytes):
        """The spare bytes to make the model's KV pool with, so the cache takes its own slots there and also borrows
        the pool's pages no sequence holds (open(lend=pool)): every byte the device has free past the pool's own
        `pool_bytes` and reserve_mib, whole bundles plus the scratch slot. 0 for the device tier."""
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
            reserve = int(self._option("reserve_mib")) << 20
            slots = min(int(self.settings.get("slots") or 0) or cache_slots(free, reserve, size), layers * experts)
            ceiling = self._max_slots(lend)
            if ceiling <= slots:
                # A cache that keeps its size (it cannot be elastic here, or it holds every expert) cannot give memory
                # back: the pools the model made in elastic ranges (KV sized for the whole context, backed as
                # sequences grow) keep the bytes they may still back. An elastic cache takes them meanwhile and gives
                # them back as the pools ask (the ledger's need); so does one that holds fewer experts for the pools'
                # sake where it can be elastic.
                later = self._pools_unbacked()
                if later:
                    free = max(0, free - later)
                    slots = min(slots, cache_slots(free, reserve, size))
                    if ceiling <= slots:
                        self.reasons["pools"] = (f"{later / GIB:.2f} GiB left free for the model's pools, which "
                                                 "back their pages as sequences grow: this cache keeps its size")
            if self.placement is not None:
                self.placement.plan({"expert_cache": (slots + 1) * size}, {"device": free, "host": 0})
        # Weight-only (W4A16) layers beside W4A4 ones (a draft layer's experts, say): the CPU's order is W4A4, so
        # where the CPU tier may compute misses it never computes theirs. Under recent residency they are the cache's
        # link layers: in the cache with the others, each miss admitted at once from a reserve of the round's victims
        # and copied over the link before it is read (what a decode round routes in a layer: decode_routes). Under
        # another policy beside the CPU tier they stay on the device whole, outside the cache and the tier. Where
        # every miss goes over the link (a file tier, misses pinned to the link) every row is the kernel's tensor
        # arithmetic, which reads each layer as it is stored: they are cached as the others.
        split = ([layer for layer, carries in self.a4_layers.items() if not carries]
                 if format == "nvfp4" and a4 == {True, False} and self._cpu_possible(format) else [])
        self.link_layers = split if self._option("policy") == "recent" else []
        whole = [layer for layer in split if layer not in self.link_layers]
        # GGUF layers stored wider than the slot (outsized) are kept whole too, at their own size: the slot is sized
        # for the rest, and no slot is padded to theirs.
        outsized = self.outsized()
        whole = whole + outsized
        if whole:
            own = self.bundle_layout.get("sizes") or {}
            held = sum(experts * max(size, int(own.get(layer, size))) for layer in whole)
            slots = max(1, slots - -(-held // size) - 1)
        if hasattr(self.store, "working"):   # the store's seal reads a store file in: it says how far it is
            self.store.working = self._working
        self.core = self._leave(slots, whole)   # before anything reads the store's region: that seals it
        for layer in whole:
            at = [layer] + list(self.bundle_layout.get("pieces", {}).get(layer, ()))
            rows = torch.empty(experts, len(at) * size, dtype=torch.uint8)
            for i, where in enumerate(at):
                piece = torch.empty(experts, size, dtype=torch.uint8)
                self.store.read([where * experts + e for e in range(experts)], piece)
                rows[:, i * size:(i + 1) * size] = piece
            own = int((self.bundle_layout.get("sizes") or {}).get(layer, size))
            self.resident[layer] = rows[:, :max(own, size)].contiguous().to(self.device)
        if outsized:
            gib = sum(self.resident[layer].numel() for layer in outsized) / GIB
            self.reasons["outsized"] = (f"layers {', '.join(map(str, outsized))} are stored wider than the slot "
                                        f"({size:,} bytes, the size most layers need): kept whole on the device "
                                        f"({gib:.2f} GiB), no slot padded to their size")
        self.slots = slots
        # The link is timed once, here: before the CPU tier exists (its threads take the processor the copies' host
        # side runs on) and before the cache takes the device's memory.
        self.link_trials = self._link_trials(size)
        self.link_us = min(self.link_trials[1:]) if self.link_trials else None
        cpu_ready = self._cpu(format, hidden, inter)
        # Over the link every miss takes a slot, so a plan keeps a layer's experts beside the prompt ring's: a cache
        # with fewer slots serves prompts only with the CPU tier computing the misses, whatever the link's time.
        link_slots = 2 * experts + (1 if self._max_slots(lend) > slots else 0)   # an elastic cache: one more
        small = slots < link_slots
        if cpu_ready and self._option("misses") == "auto" and small and format in MISS_FORMATS:
            self.misses = "cpu"
            self.miss_times = self._miss_times(layers, experts, hidden, size)
            self.miss_source = "measured" if self.miss_times else None
            self.reasons["misses_small"] = (
                f"every miss on the CPU tier: the cache's {slots:,} slots hold no prompt ring of a layer's {experts} "
                f"experts beside the {experts} a decode plan keeps when misses go over the link ({link_slots:,} slots)")
        elif cpu_ready and self._option("misses") == "auto":
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
            if self.misses == "cpu":   # pinned to the CPU tier: timed all the same, for a short prompt chunk's misses
                self.miss_times = self._miss_times(layers, experts, hidden, size)
                self.miss_source = "measured" if self.miss_times else None
        if self.miss_times:
            times = self.miss_times
            parts = (f" ({times['compute']:.0f} to compute it, {times['handoff']:.0f} for its layer's rows to reach "
                     "the host and come back)" if "handoff" in times else "")
            self.reasons["misses"] = (
                f"a missed expert takes the CPU tier {times['cpu']:.0f} us{parts} and the link "
                f"{times['link']:.0f} us here ({self.miss_source}): "
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
        # (the store's layers: an outsized GGUF layer's extra pieces are keys no plan reads)
        if self.misses != "cpu":
            self.link_layers = []   # over the link every layer is cached alike
        reserve = self.link_reserve()
        if self.link_layers:
            bound = (f"the {self.decode_routes:,} routes one decode round makes in a layer (decode rows x top)"
                     if self.decode_routes and self.decode_routes < experts else f"a layer's {experts} experts")
            self.reasons["link_layers"] = (
                f"layers {', '.join(map(str, self.link_layers))} carry no input scales (W4A16) beside W4A4 ones, and "
                f"the CPU tier's integer order is W4A4: they are in the cache, each miss admitted at once from a "
                f"reserve of {reserve:,} slots a round ({bound}) and copied over the link before the layer reads it; "
                "the CPU tier computes the other layers' misses")
        self.cache = cache_code.ExpertCache(
            self.store_layers, experts, self.bundle_layout, slots, self.device, self.store, lend, policy=self.policy,
            cpu_tier=self.misses == "cpu", record_routes=int(self._option("record_routes")),
            period=int(self._option("swap_period")), max_swaps=swaps, burst_swaps=4 * swaps,
            fill_blocks=int(self._option("fill_blocks")), side_swaps=bool(self._option("side_swaps")),
            rows=int(self._option("round_rows")), max_slots=self._max_slots(lend), owner="expert-offload",
            ring_slots=self.settings.get("ring_slots"),   # 0 in a settings file: a decode-only cache (tests)
            link_layers=self.link_layers, link_slots=reserve)
        if self._elastic():
            # A giver: the device keeps its headroom and a prompt step's room (crit_plugins.memory's step frames).
            self.memory.give(f"expert-offload:{id(self)}", self._release, self._regrow, rank=COPIES,
                             hungry=lambda: self.cache.backed_slots() < self.shape[0] * self.shape[1])
        if self.core:
            self.cache.pin(self.core)
        profile = self._profile_path(self._option("profile"))
        saved = self._profile_path(self._option("profile_save"))
        start = saved if saved is not None and saved.is_file() else profile
        if start is not None:
            self.cache.warm(self._read_profile(start))
            self.reasons["profile"] = f"cache started from {start.name}"
        return self

    def link_reserve(self):
        """The slots a round keeps for the link layers' misses: the routes one decode round makes in a layer
        (begin()'s decode_rows x top), at most a layer's experts; a layer's experts where begin() was not told. 0
        without link layers."""
        if not self.link_layers:
            return 0
        experts = self.shape[1]
        return min(experts, self.decode_routes or experts)

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
            pieces = (self.bundle_layout or {}).get("pieces") or {}
            stored = [where for layer in whole for where in [layer] + list(pieces.get(layer, ()))]
            left = leave(core + [where * experts + e for where in stored for e in range(experts)])
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

    def _pools_unbacked(self):
        """The device bytes other plugins' paged elastic ranges (crit_plugins.memory: a KV pool's pages, backed as
        sequences come to hold them) have reserved and not backed yet."""
        return sum(max(0, found.size - found.mapped - getattr(found, "host_mapped", 0))
                   for key, found in self.memory.ranges.items()
                   if getattr(found, "planes", None) and not key.startswith("expert-offload/"))

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
        stays free for what the load still makes; then the device keeps its headroom and a step's room): grow into
        `room` free bytes, never past them (backing comes by
        whole granules, so a granule less is asked for the slots)."""
        size, elastic = int(self.bundle_layout["bundle_bytes"]), self.cache.elastic
        before, held = self.cache.backed_slots(), elastic.mapped
        slots = (int(room) - elastic.gran) // size
        self.slots = self.cache.grow(before + slots) if slots > 0 else before
        if self.slots > before and "fill" not in self.reasons:   # the first growth: Core's serve() after the load
            self.reasons["fill"] = (f"{before:,} -> {self.slots:,} slots once the load was over "
                                    f"({room / 2**30:.2f} GiB were free past what the device keeps: the headroom "
                                    f"and a {self.memory.arena / 2**30:.2f} GiB step arena)")
        return elastic.mapped - held

    def end_step(self, prompts_open=False):
        """Core's step frame: the step returned. A prompt step's staged experts go back to the cache once no prompt
        is in flight; while one goes on (prompts_open) the ring and the layer staged in it stay (a span run layer by
        layer takes many steps, and its layer's experts cross the link once), and the ring goes back also when the
        prompt that held it ended without another step (a cancel). The cache grows back once the device says so
        (crit_plugins.memory's end_step, Core's frame after this one)."""
        if self.cache is not None and (self._prompt_open or self.cache.holding):
            self.cache.end_prompt(hold=bool(prompts_open))
        self._prompt_open, self._last_layer = False, None

    def _load_kernel(self):
        if self.kernel is not None or self.device.type != "cuda":
            return
        name = "gguf-rows" if self.shape[2] == "gguf" else "grouped-experts"
        try:
            from crit_plugins.kernels import load_kernel
            self.kernel, self.kernel_reason = load_kernel(name, folder=self.code["folder"]), None
        except Exception as error:   # the plugin refuses below: there is no exact fallback fast enough to serve
            self.kernel, self.kernel_reason = None, str(error).splitlines()[0] if str(error) else type(error).__name__
        if self.kernel is None:
            raise RuntimeError(f"expert-offload runs on crit's {name} kernel, which is not here "
                               f"({self.kernel_reason or 'it does not build on this machine'})")

    def _part(self, format, hidden, inter, declared=False):
        if format == "gguf":   # gguf-rows over the slots (the CPU tier's GGUF rows are its declared order)
            if self.caps is not None:
                raise ValueError("expert-offload: GGUF experts are gated by silu or gelu_tanh, not softcapped_silu")
            return self.code["gguf"].GgufRoutes(self.kernel, self.bundle_layout, activation=self.activation)
        dispatch = self.code["dispatch"]
        # A call's route rows are scratch (the engine's: Core's): the prompt chunk sizes a server meets cost the
        # largest call's rows, not a set each, in bytes every other plugin's calls compute in too.
        scratch = getattr(self.context, "scratch", None)
        if scratch is None:
            scratch = self._scratch = self._scratch or self.memory.scratch("expert-offload")
        part = dispatch.KernelRoutes(self.kernel, self.bundle_layout, self.code["grouped_bundles"].kernel_layout,
                                     format, hidden, inter, a4=self.a4, scratch=scratch, activation=self.activation,
                                     caps=self.caps)
        part.decode_arithmetic = "declared" if declared else self._tensor()
        return part

    def _cpu_possible(self, format):
        """Whether the CPU tier may compute this engine's misses at all: a format it computes, misses not pinned to
        the link, a store it reads bundles of in place (mapped, in the slot layout)."""
        return (format in MISS_FORMATS and self._option("misses") != "link" and getattr(self.store, "mapped", False)
                and not getattr(self.store, "compressed", ())
                and (self.activation == "silu" or format == "gguf" and self.activation == "gelu_tanh"))

    def _cpu(self, format, hidden, inter):
        """Whether the CPU tier and its in-stream join are built here (then self.dispatch holds them)."""
        o, c = self.options, self.code
        if format == "nvfp4" and not self.a4:
            self.reasons["misses"] = "the CPU tier's integer order is W4A4: these nvfp4 experts carry no input scales"
            return False
        if not self._cpu_possible(format):
            self.reasons["misses"] = (
                "the CPU tier computes nvfp4, int4 and GGUF experts" if format not in MISS_FORMATS
                else f"the CPU tier computes SwiGLU blocks, and these are gated by {self.activation}"
                if self.activation != "silu"
                else "misses pinned to the link" if self._option("misses") == "link"
                else "a file tier's misses are read on the host" if not getattr(self.store, "mapped", False)
                else "the store keeps scale components compressed (the expert source's csf_scales), which the CPU "
                     "tier does not read in place")
            if self._option("misses") == "cpu" and getattr(self.store, "compressed", ()):
                raise ValueError("misses: cpu over a store that keeps its scales compressed (the expert source's "
                                 "csf_scales): the CPU tier reads bundles in place in the slot layout. Leave "
                                 "csf_scales empty, or let the misses go over the link")
            return False
        try:
            if format == "gguf":   # in torch, gguf-rows' declared order with the kernel's splits on this device
                gguf = c["gguf"]
                tier = gguf.GgufCpuTier(self.bundle_layout, self.shape[1],
                                        gguf.tier_splits(self.kernel, self.bundle_layout, self.device),
                                        activation=self.activation)
            else:
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
        # The layers kept whole on the device (W4A16 beside W4A4) are never the tier's: it computes W4A4 experts, and
        # those layers' bundles carry no input scales.
        served = [layer for layer in range(layers) if layer not in self.resident and layer not in self.link_layers]
        try:
            for layer in served:
                for expert in range(experts):
                    if layer * experts + expert in left:
                        continue
                    at = (layer * experts + expert) * size
                    tier.add(layer * experts + expert, flat[at:at + size])
        except Exception as error:   # an expert the tier does not take: the misses go over the link, with the reason
            tier.close()
            if self._option("misses") == "cpu":
                raise
            self.reasons["misses"] = f"the CPU tier does not take these experts: {str(error).splitlines()[0]}"
            return False
        ext, reason = c["dispatch"].extension() if tier.native else (None, None)
        if ext is None and tier.native:
            tier.close()
            self.reasons["misses"] = f"the in-stream join does not build: {reason}"
            return False
        part = self._part(format, hidden, inter, declared=True)
        self.dispatch = c["dispatch"].DeviceDispatch(
            self.device, hidden, inter, format, tier, part, ext, experts, set(served),
            max_tokens=int(o["max_tokens"]), prompt_arithmetic=self._asked_arithmetic(),
            owner="expert-offload")
        if not tier.native:
            self.reasons["misses"] = ("every miss on the CPU tier (GGUF experts in torch, gguf-rows' declared order: "
                                      "joined after a synchronization, rounds with it not captured); the link copies "
                                      "it into the cache beside the round")
            return True
        # The CPU tier is ready: open() times a missed expert on it and over the link and takes the sooner
        # (choose_misses), unless misses pins one.
        self.reasons["misses"] = "every miss on the CPU tier; the link copies it into the cache beside the round"
        return True

    def _miss_times(self, layers, experts, hidden, size, count=64, trials=5):
        """{"cpu": us, "link": us, "compute": us, "handoff": us} a missed expert on this machine as the engine will
        serve it. compute: the CPU tier computing `count` experts spread over the store for one token each, the
        fastest of `trials` batches of the same experts after one that is not counted. link: the link copying a
        bundle to the device as open() timed it before the tier existed (_link_trials). handoff: the rows' way to
        the host and back (_handoff_us), which a decode layer with a miss on the CPU tier pays and a copy does not.
        cpu: compute and handoff, a decode layer's miss (its layer misses one expert or two). The fastest batch is
        what each side can do: other work on the host only ever slows one. The first batch is the experts' first use: it wakes the tier's threads and brings the
        store's pages in (a store just built or reopened pays that once an expert, several times a miss's own time).
        The batch is larger than the processor's caches, so the tier reads each expert from host memory again, as
        a miss does. None where either cannot be timed (no mapped store on a CUDA device)."""
        import time
        import torch
        if not self.link_trials or self.dispatch is None:
            return None
        tier = self.dispatch.tier
        held = sorted(tier.entries)   # the experts the tier computes: not a layer kept whole
        if not held:
            return None
        total = len(held)
        count = min(count, total if tier.native else 8)   # a tier in torch: a few experts time it
        x = torch.randn(1, hidden, generator=torch.Generator().manual_seed(0))
        token = torch.zeros(count, dtype=torch.int32)
        keys = torch.tensor([held[(i * 9973) % total] for i in range(count)], dtype=torch.int32)
        cpu = []
        for trial in range(trials + 1):
            started = time.perf_counter()
            tier.compute(keys, token, x)
            cpu.append((time.perf_counter() - started) * 1e6 / count)
        tier.reset_stats()
        link = self.link_trials if len(self.link_trials) <= 12 else self.link_trials[:6] + self.link_trials[-6:]
        self.miss_trials = {"cpu": [round(t, 1) for t in cpu], "link": [round(t, 1) for t in link]}
        compute, handoff = min(cpu[1:]), self._handoff_us(hidden)
        return {"cpu": compute + handoff, "link": min(self.link_trials[1:]), "compute": compute, "handoff": handoff}

    def _handoff_us(self, hidden, trials=16):
        """The microseconds a layer's rows take to reach the host and its answer to come back to the device on this
        machine: what a decode layer pays before the CPU tier's first missed expert of it is computed and after its
        last (the device stops for the host, the host for the device), and a copy over the link never does (it is
        one more operation in the device's order). The median of `trials` turns of one row after two that are not
        counted: a latency as a layer typically pays it, tens of microseconds where the device is the machine's own,
        most of a millisecond where it is reached through a virtual machine."""
        import time
        import torch
        x = torch.zeros(1, hidden, dtype=torch.bfloat16, device=self.device)
        found = []
        for trial in range(trials + 2):
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            x.float().cpu().to(self.device)
            torch.cuda.synchronize(self.device)
            found.append((time.perf_counter() - started) * 1e6)
        return sorted(found[2:])[trials // 2]

    def _link_trials(self, size, count=64, seconds=1.0, most=256):
        """The microseconds the link takes to bring one bundle from the store to the device on this machine, as
        the cache's fills read it: batches of the same `count` bundles spread over the store, one after another,
        the first counted in the list and never used (the bundles' first use brings the store's pages in for the
        host and for the device, which a served copy does not pay again). The link's time is the fastest of the
        others. A device left idle while the engine loaded has its link at its lowest speed and brings it up only
        under load, a fraction of a second later (an RTX 5090: 3.5 GiB/s for 0.2 to 0.4 s, then 25): its first
        batches agree with each other at a seventh of the rate it serves misses at, with the device at work. So
        the batches go on until four have run since the rate last doubled, for `seconds` after the first at
        most, `most` batches at most: no early agreement ends them. None without a store the device reads in
        place on a CUDA device."""
        import time
        import torch
        try:
            _, mapped = self.store.region()
        except Exception:   # a store the device cannot read in place (a file tier)
            return None
        if mapped is None or self.device.type != "cuda":
            return None
        flat, total = mapped.view(-1), self.shape[0] * self.shape[1]
        left = getattr(self.store, "left", ())   # held on the device alone: not in host memory
        keys = [key for key in (((7 + i) * 9973) % total for i in range(count)) if key not in left]
        if not keys:
            return None
        target = torch.empty(len(keys), size, dtype=torch.uint8, device=self.device)
        found, began, woke = [], None, None
        while len(found) < most and (began is None or len(found) < 5 or time.perf_counter() - began < seconds):
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            for i, key in enumerate(keys):
                target[i].copy_(flat[key * size:(key + 1) * size], non_blocking=True)
            torch.cuda.synchronize(self.device)
            found.append((time.perf_counter() - started) * 1e6 / len(keys))
            if began is None:
                began = time.perf_counter()   # the first batch brought the pages in: the clock starts after it
            elif 2 * found[-1] < max(found[1:]) and (woke is None or 2 * found[-1] < found[woke]):
                woke = len(found) - 1         # the link came up: four more batches at its speed
            if woke is not None and len(found) - woke > 4:
                break
        del target
        return found

    def prompts(self):
        """What a prompt costs these experts, from what was measured here: whether a chunk brings its layers'
        experts to the device (stages_prompts()), the link's rate, and a sweep, the bytes and seconds of bringing
        every expert the cache does not hold to the device once. A long chunk run through every layer pays a sweep;
        a span run layer by layer pays one for all its chunks. {} where every expert is on the device."""
        if self.cache is None:
            return {}
        size = int(self.bundle_layout["bundle_bytes"])
        cached = (self.shape[0] - len(self.resident)) * self.shape[1] - len(self.core)
        away = max(0, cached - (self.cache.backed_slots() - len(self.cache.ring_slots) - len(self.core)))
        found = {"staged": self.stages_prompts(), "sweep_bytes": away * size}
        if self._short_on_cpu():
            times = self.miss_times
            found["short"] = ("a short chunk's missed routes on the CPU tier where they cost under "
                              f"{CPU_WINS:g} of their experts' copies: {times.get('compute', times['cpu']):.0f} us "
                              f"a route there, {times['link']:.0f} us a bundle over the link")
            found["short_cpu_routes"] = self.cache.prompt_cpu_routes
        link = self.link_us or (self.miss_times or {}).get("link")
        if link:
            found.update(link_us=round(link, 1), link_gib_s=round(size / link * 1e6 / GIB, 2),
                         sweep_seconds=round(away * link / 1e6, 3))
        return found

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
        if prompt is None:
            prompt = ids.shape[0] > int(self.options["max_tokens"])
        if self.tier == "device":
            return self._rows(layer, x, ids, weights, self.stores[layer], None, self._tensor(layer, prompt))
        if layer in self.resident:
            return self._rows(layer, x, ids, weights, self.resident[layer], None, self._tensor(layer, prompt))
        if prompt:
            if self._prompt_open and self._last_layer is not None and layer < self._last_layer:
                self.end_step(True)   # the next prompt step began without a frame from Core
            self._prompt_open, self._last_layer = True, layer
            # The device part's rows first: a shape's first call grows the scratch, which takes device memory the
            # cache gives by releasing slots, and the chunk's experts must not be staged in them yet.
            self._a4(layer)
            # A link layer's rows are the kernel's tensor arithmetic, as a layer held whole on the device: the CPU
            # tier's order never computes its experts.
            link = layer in self.link_layers
            arithmetic = self._tensor(layer, True) if link else self._prompt_arithmetic(layer)
            self.part.prepare(ids.shape[0], ids.shape[1], self.shape[1], x.device, arithmetic)
            if self._short_on_cpu() and not link:
                # A short chunk where the CPU tier computes misses in the rows' own order: its missed routes go
                # where they cost less by what open() measured, the CPU tier or the link.
                # (a chunk's many routes share one way to the host and back: a route costs its computing alone)
                times = self.miss_times
                slots = self.cache.prompt_cpu(layer, ids, times.get("compute", times["cpu"]), times["link"], CPU_WINS)
                if slots is not None:
                    return self.dispatch.forward(layer, x, ids, weights, self.cache.store, slots)
            store, slots = self.cache.prompt(layer, ids)
            return self._rows(layer, x, ids, weights, store, slots, arithmetic)
        if self._prompt_open:
            self.end_step(False)   # a prompt step no frame closed: this decode call ends it
        if layer in self.link_layers and self._eager_after_others():
            # A step decodes eagerly, the decoder's layers before the draft layer in one round: their hits and the
            # CPU tier's copies can take every victim the round listed. A captured round plans the draft layer first
            # (the drafter before the verify); eagerly its plan starts a round of its own, with the whole list.
            self.cache.start_round(None)
        if self.dispatch is not None and layer not in self.link_layers:
            self.part.a4 = True
            decided = (lambda table: self.dispatch.post(layer, x, ids, table))
            slots = self.cache.plan(layer, ids, rows=rows, decided=decided)
            return self.dispatch.forward(layer, x, ids, weights, self.cache.store, slots)
        slots = self.cache.plan(layer, ids, rows=rows)
        return self._rows(layer, x, ids, weights, self.cache.store, slots, self._tensor(layer))

    def _eager_after_others(self):
        """Whether a decode plan now runs outside a capture after other plans of the cache's round."""
        import torch
        recent = getattr(self.cache, "_recent", None)
        if recent is None or not recent.listed or not recent.calls:
            return False
        return not (self.device.type == "cuda" and torch.cuda.is_current_stream_capturing())

    def _short_on_cpu(self):
        """Whether a short prompt chunk's misses may go to the CPU tier: it computes decode misses, prompt rows keep
        its order (declared: who computes a route then never changes a bit), and open() timed both ways."""
        return (self.dispatch is not None and self.misses == "cpu" and bool(self.miss_times)
                and self._prompt_arithmetic() == "declared")

    def _asked_arithmetic(self):
        """The prompt_arithmetic option, declared or tensor. auto is the faster one that keeps the level the engine
        asks of its prompt rows (crit_plugins.kernels.asks, the configuration's exactness): tensor where it asks
        repeatable or row-exact (the kernel's tensor cores: a row's bits its own at every row count, not the decode
        rows'), declared where it asks exact or declared (the CPU's integer order, the decode rows' bits) and where it
        asks nothing, as before the option had auto."""
        chosen = self._option("prompt_arithmetic")
        if self.shape is not None and self.shape[2] == "gguf":
            return "declared"   # gguf-rows' prompt rows are its decode rows' order: the CPU tier's
        if chosen == "auto":
            chosen = "tensor" if self.asked in ("repeatable", "row-exact") else "declared"
        return chosen

    def _prompt_arithmetic(self, layer=None):
        """A prompt chunk's rows: the CPU's order only where the CPU computes decode misses and the option keeps it,
        else the kernel's (_tensor)."""
        chosen = self._asked_arithmetic() if self.misses == "cpu" else "tensor"
        return self._tensor(layer, True) if chosen == "tensor" else chosen

    def _row_prompts(self):
        """Whether prompt rows of NVFP4 experts without input scales run fp4-row (W4A4 at each route's own scale on the
        FP4 tensor cores), as grouped-experts' prompt_arithmetic auto lets them: quality mode speed, an engine that asks
        its prompt rows no more than row-exact (their bits then their own, not the decoded route's), crit's kernel with
        FP4 tensor cores here, hidden and inter multiples of 128. Decode rows, bound by their weights' bytes, keep
        tensor."""
        if not self.speed or self.kernel is None or self.shape is None or self.asked not in (None, "row-exact",
                                                                                             "repeatable"):
            return False
        _, _, format, hidden, inter = self.shape
        return (format == "nvfp4" and hidden % 128 == 0 and inter % 128 == 0
                and getattr(self.kernel, "fp4_available", lambda device=None: False)(self.device))

    def _tensor(self, layer=None, prompt=False):
        """The kernel's arithmetic for device rows that are not the CPU tier's order: tensor (BF16 tensor cores), or
        in quality mode speed the experts' own tensor cores where crit's kernel has them on this device and the layer
        takes them, as grouped-experts' arithmetic auto chooses: fp4 for W4A4 NVFP4 experts (their bundles carry
        input scales), fp8 for FP8 ones, hidden and inter multiples of 128. Decode and prompt rows of a layer take
        the same one, so a prompt row keeps the bits of the same route decoded; but a prompt row of an NVFP4 layer
        without input scales takes fp4-row where _row_prompts allows it (its bits then its own)."""
        if not self.speed or self.kernel is None or self.shape is None:
            return "tensor"
        _, _, format, hidden, inter = self.shape
        a4 = self.a4_layers.get(layer, self.a4) if layer is not None else self.a4
        if prompt and not a4 and self._row_prompts():
            return "fp4-row"
        own = "fp4" if format == "nvfp4" and a4 else "fp8" if format in ("fp8:channel", "fp8:block") else None
        here = own is not None and getattr(self.kernel, own + "_available", lambda device=None: False)(self.device)
        return own if here and hidden % 128 == 0 and inter % 128 == 0 else "tensor"

    def exactness(self):
        """The level its rows keep once open() decided who serves a miss (crit_plugins.exactness; Core compares it
        with the level the configuration asks). Where the CPU computes decode misses every decode row is the CPU's
        integer order (declared) and a prompt row is that order too, or with prompt_arithmetic tensor the kernel's
        tensor arithmetic: the same bits at every row count, not the decode rows' (row-exact). With misses over the
        link, or every expert on the device, every row is the kernel's tensor arithmetic: decode rows row-exact, a
        prompt row the bits of the same route decoded (exact). Nothing before open()."""
        if self.misses is None:
            return {}
        if self.misses == "cpu":
            return {"decode": "declared", "prompt": "declared" if self._prompt_arithmetic() == "declared" else "row-exact"}
        apart = self._row_prompts() and (not self.a4 or False in self.a4_layers.values())   # fp4-row prompt rows
        return {"decode": "row-exact", "prompt": "row-exact" if apart else "exact"}

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

    def stages_prompts(self):
        """Whether a prompt chunk brings its layers' experts to the device (the ram and disk tiers: a long chunk every
        expert of every layer that is not resident, over the link). A model plugin then runs a prompt's spans layer
        by layer, every chunk of a span through one layer before the next layer: the layer's experts are staged once
        for the span (the cache serves the layer's later chunks from what its first staged), where chunk by chunk
        every layer's would cross the link again for each chunk. False for experts that fit on the device, and
        before open()."""
        return self.cache is not None and bool(self.cache.ring_slots)

    # -- rounds -----------------------------------------------------------------------------------------------------
    @property
    def round_counters(self):
        """The round counters its cache adds each row's expert reads and hits to (crit_plugins.rounds.Rounds gives a
        round the counters its parts name); none where every expert is on the device, which counts nothing."""
        return self.cache.round_counters if self.cache is not None else ()

    def start_round(self, view):
        if self.cache is not None:
            self.cache.start_round(view)

    def count_round(self, view):
        if self.cache is not None:
            self.cache.count_round(view)

    def round_problems(self):
        if self.part is None:
            return ["open() the experts before capturing rounds"]
        found = self.cache.round_problems() if self.cache is not None else []
        if self.dispatch is not None and not self.dispatch.in_stream:
            found.append("the CPU tier computes these misses in torch on the host: their rows join the device's after "
                         "a synchronization, outside a capture")
        return found

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
                        "shrinks": self.shrinks,
                        "policy": self.policy, "settings": bool(self.settings), "prompts": self.prompts()})
        return {"expert_offload": out}

    def telemetry(self):
        out = {"tier": self.tier, "misses": self.misses}
        if self.miss_times:
            out["miss_times_us"] = {k: round(v, 1) for k, v in self.miss_times.items()}
            out["miss_times_from"] = self.miss_source
            if self.miss_trials:
                out["miss_trials_us"] = self.miss_trials
        if self.cache is not None:
            out["expert_cache"] = self.cache.telemetry()
            out["prompts"] = self.prompts()
            if self._elastic():   # the cache's size now and the times it gave memory back, beside what the device
                # keeps free (crit_plugins.memory: the headroom and the step arena)
                steps = self.memory.steps()
                out["elastic"] = {"slots": self.cache.backed_slots(), "shrinks": self.shrinks,
                                  "headroom_bytes": steps["headroom"], "arena_bytes": steps["arena"]}
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
            if getattr(self.store, "kept", False):   # its keeper's: it stays for the next engine
                self.store.working = None
            else:
                self.store.close()
            self.store = None
