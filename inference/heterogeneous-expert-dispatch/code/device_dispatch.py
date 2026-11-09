"""Routed experts split between the device and the CPU inside one CUDA stream (captured graphs too).

A layer's routes whose expert the slot table marks -2 (the expert cache's admission sent them to the CPU) are posted to
the CPU tier through a host-work-mailbox slot (cpu_routes.cu: one post kernel, the tier's C handler on the mailbox's
service thread, one join, one scatter kernel); every other route is the device part's (its route_outputs over the slot
store); then the device part's combine sums each token's routes in route order. Both parts compute the same declared
order (host-expert-arithmetic's integer block rows) on bf16(x), so a token's bits never depend on which part computed
which route.

    dispatch = DeviceDispatch(device, hidden, inter, format, tier, part, ext, stride)
    dispatch.post(layer, x, ids, slots)     # optional, early: right after the slot table is known, before the fills
    out = dispatch.forward(layer, x, ids, weights, store, slots)

A call of up to max_tokens tokens posts once and joins once, capturably. A longer call (a prompt chunk, host-driven)
looks for -2 routes first (one synchronization): none skips the mailbox, else they are posted in pieces of max_tokens
tokens. The device part is any object with route_outputs(layer, x, ids, store, slots) -> y [tokens * top][hidden]
(routes of slot -2 and id -1 left as they were) and combine(y, ids, weights) (the experts plugin's fact experts:routes);
DeclaredRoutes is the baseline one here: the declared order in torch on the device, exact and slow, until an experts
plugin's own kernel takes its place. Without the native CPU tier or the CUDA build the CPU routes are computed on the
host after a synchronization (the same bits); a capture then raises.
"""
import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_BUILT = {}


def extension(mailbox_code, arch=None):
    """(module, None) for cpu_routes.cu with host-work-mailbox's mailbox.cu built for this device (or `arch`, e.g.
    "12.0a") through torch's extension builder under a name hashed from the sources' paths and contents, or
    (None, why) where it does not build."""
    import torch
    sources = [HERE / "cpu_routes.cu", Path(mailbox_code) / "mailbox.cu"]
    digest = hashlib.sha256(b"".join(str(s).encode() + s.read_bytes() for s in sources) +
                            (Path(mailbox_code) / "mailbox.h").read_bytes() +
                            (Path(mailbox_code) / "mailbox_device.cuh").read_bytes()).hexdigest()[:12]
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}" + ("a" if major >= 9 else "")
    key = (digest, arch)
    if key not in _BUILT:
        from torch.utils.cpp_extension import load
        before = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        try:
            _BUILT[key] = (load(name=f"crit_cpu_routes_{arch.replace('.', '_')}_{digest}",
                                sources=[str(s) for s in sources], extra_include_paths=[str(mailbox_code)],
                                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=["-O3", "-std=c++20"],
                                verbose=False), None)
        except Exception as error:  # noqa: BLE001 - no compiler or a header mismatch: the host path serves
            _BUILT[key] = (None, f"the CPU routes' mailbox did not build: {str(error).splitlines()[0][:300]}")
        finally:
            if before is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = before
    return _BUILT[key]


def capturing():
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class DeclaredRoutes:
    """The baseline device part: every device route's unweighted row in the declared integer order in torch (exact, the
    CPU tier's bits), over bundles in a slot store; capture-safe: its constants are made when it is, its workspace at
    the first call of a shape (refused inside a capture), and a capture's temporaries come from the graph's own pool."""

    def __init__(self, declared, bundles, format, hidden, inter, device):
        self.declared, self.bundles, self.format = declared, bundles, format
        self.plan = bundles.layout(format, hidden, inter)
        self.math = bundles.arithmetic(format)
        self.hidden = hidden
        self.workspace = {}
        declared.prepare(device)

    def route_outputs(self, layer, x, ids, store, slots):
        import torch
        tokens, top = ids.shape
        key = (tokens, top, str(x.device))
        if key not in self.workspace:
            if capturing():
                raise RuntimeError(f"call the experts once at {tokens} tokens x {top} routes outside a capture first: "
                                   "the device part's workspace is made then")
            self.workspace[key] = torch.zeros(tokens * top, self.hidden, dtype=torch.float32, device=x.device)
        y = self.workspace[key]
        flat = ids.reshape(-1).long()
        slot_of = slots.long()[flat.clamp(min=0)]
        valid = (flat >= 0) & (slot_of >= 0)
        safe = torch.where(valid, slot_of, torch.zeros_like(slot_of))
        rows = x.bfloat16().float()   # the declared order reads bf16(x)
        m, view, plan = self.math, self.bundles.view, self.plan
        for r in range(tokens * top):
            bundle = store.index_select(0, safe[r:r + 1])[0]
            g = view(bundle, plan, "input")
            row = self.declared.expert_rows(
                rows[r // top:r // top + 1], (view(bundle, plan, "gate_up"), view(bundle, plan, "gate_up_scale"),
                                              view(bundle, plan, "gate_up_inverse")),
                (view(bundle, plan, "down"), view(bundle, plan, "down_scale"), view(bundle, plan, "down_inverse")),
                m["weights"], m["factor"], m["input_kind"], g[0], m["inter_kind"], g[1], 32)
            y[r] = torch.where(valid[r], row[0], y[r])
        return y

    def combine(self, y, ids, weights):
        return self.declared.combine(y, ids, weights)


class DeviceDispatch:
    """The CUDA dispatch over a CPU tier (host-expert-tier's CpuTier) and a device part. ext: extension()'s module or
    None (the host path); stride: experts per layer in the tier's key space (key = layer * stride + expert), the length
    every slot table must have; layers: the layers whose experts the tier holds."""

    def __init__(self, device, hidden, inter, format, tier, part, ext, stride, layers, max_tokens=64, join=None,
                 mail_slots=2, idle_after_us=2000, idle_sleep_us=50, joins=None):
        import torch
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden, self.inter, self.format = hidden, inter, format
        self.tier, self.part, self.stride, self.layers = tier, part, int(stride), set(layers)
        self.max_tokens, self.mail_slots = int(max_tokens), int(mail_slots)
        self.idle_after_us, self.idle_sleep_us = int(idle_after_us), int(idle_sleep_us)
        self.ext = ext if tier.native else None
        self.box, self.top, self.joins, self.buffers, self.posted = None, 0, None, {}, set()
        self.eager_calls = 0
        if self.ext is not None:
            self.joins = joins or self.ext.measure_joins(self.device.index, 32, 50)
            self.join_kind = join or self.joins["choice"]
            self.counters = torch.zeros(2, dtype=torch.int64, device=self.device)   # jobs posted, routes posted

    @property
    def in_stream(self):
        return self.ext is not None

    def slot_of(self, layer):
        return layer % self.mail_slots

    def _box(self, top):
        """The mailbox, sized for max_tokens tokens of `top` routes; built at the first post, rebuilt (outside a capture)
        for a call with more routes a token."""
        import torch
        if self.box is not None and top <= self.top:
            return self.box
        if capturing():
            raise RuntimeError(f"post a call of {top} routes a token once outside a capture first: the mailbox is sized then")
        if self.box is not None:
            self.close_box()
        routes = self.max_tokens * top
        serve, context = self.tier.handler()
        box = self.ext.Box(self.mail_slots, self.hidden, routes, self.max_tokens, self.join_kind, serve, context,
                           self.idle_after_us, self.idle_sleep_us)
        tier_layout = self.tier.payload_layout(routes, self.max_tokens)
        mine = dict(zip(("experts", "token", "x", "out", "bytes"), box.layout()))
        if mine != tier_layout:
            box.release()
            raise RuntimeError(f"the mailbox payload {mine} is not the CPU tier's {tier_layout}")
        self.box, self.top, self.routes_cap = box, top, routes
        self.buffers = {s: {"routes": torch.zeros(routes, dtype=torch.int32, device=self.device),
                            "count": torch.zeros(2, dtype=torch.int32, device=self.device),
                            "status": torch.zeros(1, dtype=torch.int32, device=self.device)}
                        for s in range(self.mail_slots)}
        return box

    def _check(self, layer, ids, slots):
        if self.box is not None and self.box.failed():   # a host atomic: no synchronization
            raise RuntimeError(f"the CPU tier failed and its mailbox answers every job as failed: {self.box.error()}")
        if layer not in self.layers:
            raise KeyError(f"the CPU tier holds no expert of layer {layer}")
        if slots.numel() != self.stride:
            raise ValueError(f"a slot table has one entry per expert of the layer ({self.stride}), not {slots.numel()}")

    def post(self, layer, x, ids, slots):
        """Post the layer's CPU routes (slot -2) now, so the CPU computes while the stream does the rest; a call of up
        to max_tokens tokens."""
        if not self.in_stream:
            return
        self._check(layer, ids, slots)
        if ids.shape[0] > self.max_tokens:
            raise ValueError(f"post() takes up to {self.max_tokens} tokens (max_tokens); forward() posts longer calls "
                             "in pieces")
        box = self._box(ids.shape[1])
        b = self.buffers[self.slot_of(layer)]
        box.post(self.slot_of(layer), ids.contiguous(), slots, x.contiguous(), int(layer) * self.stride, b["routes"],
                 b["count"], self.counters)
        self.posted.add(layer)

    def forward(self, layer, x, ids, weights, store, slots):
        if not self.in_stream:
            if capturing():
                raise RuntimeError("the CPU routes join inside a capture only through the in-stream mailbox, which is "
                                   "not built here (see the plugin's facts): decode without capturing")
            y = self.part.route_outputs(layer, x, ids, store, slots)
            self._host(layer, x, ids, slots, y)
            return self.part.combine(y, ids, weights)
        self._check(layer, ids, slots)
        if ids.shape[0] <= self.max_tokens:
            if layer not in self.posted:
                self.post(layer, x, ids, slots)
            y = self.part.route_outputs(layer, x, ids, store, slots)
            s = self.slot_of(layer)
            b = self.buffers[s]
            self.box.join(s)
            self.box.scatter(s, y, b["routes"], b["count"], b["status"], ids.numel())
            self.posted.discard(layer)
            return self.part.combine(y, ids, weights)
        return self._long(layer, x, ids, weights, store, slots)

    def _long(self, layer, x, ids, weights, store, slots):
        """A call past max_tokens (prompt chunks; host-driven, never captured): the mailbox only when it has -2 routes,
        then in pieces of max_tokens tokens."""
        if capturing():
            raise RuntimeError(f"a captured call has at most {self.max_tokens} tokens (max_tokens)")
        flat = ids.reshape(-1).long()
        cpu = bool(((flat >= 0) & (slots.long()[flat.clamp(min=0)] == -2)).any())   # one synchronization
        y = self.part.route_outputs(layer, x, ids, store, slots)
        if cpu:
            top = ids.shape[1]
            box = self._box(top)
            for t0 in range(0, ids.shape[0], self.max_tokens):
                t1 = min(ids.shape[0], t0 + self.max_tokens)
                s = self.slot_of(layer)
                b = self.buffers[s]
                box.post(s, ids[t0:t1].contiguous(), slots, x[t0:t1].contiguous(), int(layer) * self.stride,
                         b["routes"], b["count"], self.counters)
                box.join(s)
                box.scatter(s, y[t0 * top:t1 * top], b["routes"], b["count"], b["status"], (t1 - t0) * top)
        return self.part.combine(y, ids, weights)

    def _host(self, layer, x, ids, slots, y):
        """The CPU routes on the host after a synchronization: the path without the in-stream join."""
        import torch
        self._check(layer, ids, slots)
        flat = ids.reshape(-1).long()
        slot_of = slots.long()[flat.clamp(min=0)]
        cpu = ((flat >= 0) & (slot_of == -2)).nonzero(as_tuple=True)[0]
        if cpu.numel() == 0:
            return
        self.eager_calls += 1
        keys = (flat[cpu] + int(layer) * self.stride).to(torch.int32).cpu()
        tokens = (cpu // ids.shape[1]).to(torch.int32).cpu()
        rows = self.tier.compute(keys, tokens, x.bfloat16().float().cpu())
        y[cpu] = rows.to(y.device)

    def calibrate(self, split, bundles, layers, tokens=1, misses=(1, 2, 3, 4), repeats=5, graph_layers=8):
        """The CPU/link split of a layer's misses at `tokens` tokens an expert, measured here through captured layers
        (never a constant): each of `graph_layers` layers posts j CPU experts (experts 0..j-1 of one of `layers`,
        rotated), copies k bundles over the link (pinned host -> device by the copy engine; `bundles` rotated), joins
        and scatters; split.py's SplitTimings from k alone, j alone and both at once, ms a layer, and its quota table
        (link misses per miss count). Provisional until it measures the expert cache's own fills (SM loads from mapped
        memory) behind its decided() hook."""
        import torch
        if not self.in_stream:
            raise ValueError("the split is measured through the in-stream join: it needs the CUDA build and the native tier")
        most = max(misses)
        top = most + 1
        if tokens > self.max_tokens:
            raise ValueError(f"{tokens} tokens are past max_tokens ({self.max_tokens})")
        dev = self.device
        self._box(max(top, self.top))
        scratch = torch.empty(most, bundles[0].numel(), dtype=torch.uint8, device=dev)
        x = torch.zeros(tokens, self.hidden, dtype=torch.bfloat16, device=dev)
        y = torch.zeros(tokens * top, self.hidden, dtype=torch.float32, device=dev)
        graphs = {}

        def inputs(j):
            ids = torch.full((tokens, top), most, dtype=torch.int64)   # expert `most`: resident (slot 0)
            ids[:, :j] = torch.arange(j)
            table = torch.zeros(self.stride, dtype=torch.int32)
            table[:j] = -2
            return ids.to(dev), table.to(dev)

        def graph(k, j):
            ids, table = inputs(j)
            stream = torch.cuda.Stream(dev)

            def body():
                for g in range(graph_layers):
                    s, b = self.slot_of(g), self.buffers[self.slot_of(g)]
                    self.box.post(s, ids, table, x, layers[g % len(layers)] * self.stride, b["routes"], b["count"],
                                  self.counters)
                    for i in range(k):
                        scratch[i].copy_(bundles[(g * most + i) % len(bundles)], non_blocking=True)
                    self.box.join(s)
                    self.box.scatter(s, y, b["routes"], b["count"], b["status"], ids.numel())

            with torch.cuda.stream(stream):
                body()
            stream.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                body()
            return captured, stream, (ids, table)   # the inputs live as long as the graph that reads their addresses

        def measure(k, j):
            if (k, j) not in graphs:
                graphs[(k, j)] = graph(k, j)
            captured, stream, _ = graphs[(k, j)]
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                a.record()
                captured.replay()
                b.record()
            b.synchronize()
            last = self.buffers[self.slot_of(graph_layers - 1)]
            if j and (int(last["status"][0]) != 1 or int(last["count"][0]) != j * tokens):
                raise RuntimeError(f"a calibration layer with {j} CPU experts was not served as posted (status "
                                   f"{int(last['status'][0])}, {int(last['count'][0])} routes)")
            return a.elapsed_time(b) / graph_layers

        measure(most, 0)
        measure(0, most)
        timings = split.calibrate(lambda k: measure(k, 0), lambda j: measure(0, j), measure, misses, repeats)
        return timings, split.quota_table(timings)

    def status(self):
        """The mailbox's state: join kind, failure and its error (host words, no synchronization), and, read on the
        device (synchronizes), each mail slot's last answer (1 served, 2 skipped; others: the window is invalid), its
        overflow flag and the jobs and routes the device posted (replays included)."""
        if not self.in_stream:
            return {"in_stream": False, "host_calls": self.eager_calls}
        out = {"in_stream": True, "join": self.join_kind, "jobs": int(self.counters[0]), "routes": int(self.counters[1])}
        if self.box is not None:
            out.update({"failed": self.box.failed(), "error": self.box.error(), "served": self.box.served(),
                        "status": [int(self.buffers[s]["status"][0]) for s in range(self.mail_slots)],
                        "overflow": [int(self.buffers[s]["count"][1]) for s in range(self.mail_slots)]})
        return out

    def close_box(self):
        """Answer every pending doorbell, let the streams drain, then drop the mailbox (before the tier it calls)."""
        import torch
        if self.box is not None:
            self.box.release()
            torch.cuda.synchronize(self.device)
            self.box = None

    def close(self):
        self.close_box()
