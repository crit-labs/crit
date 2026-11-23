"""Routed experts split between the device and the CPU inside one CUDA stream (captured graphs too).

A layer's routes whose expert the slot table marks -2 (the expert cache's admission sent them to the CPU) are posted to
the CPU tier through a slot of one of the host's handoffs (Core's, crit_plugins.host; crit's kernel cpu-routes: one post kernel,
the tier's C handler on the host's service thread, the handoff's join, one scatter kernel); every other route is the
device part's (its route_outputs over the slot store); then the device part's combine sums each token's routes in
route order. Both parts compute the same declared
order (host-expert-arithmetic's integer block rows) on bf16(x), so a token's bits never depend on which part computed
which route.

    dispatch = DeviceDispatch(device, hidden, inter, format, tier, part, ext, stride)
    dispatch.post(layer, x, ids, slots)     # optional, early: right after the slot table is known, before the fills
    out = dispatch.forward(layer, x, ids, weights, store, slots)

A call of up to max_tokens tokens posts once and joins once, capturably. A longer call (a prompt chunk, host-driven)
looks for -2 routes first (one synchronization): none skips the handoff, else they are posted in pieces of max_tokens
tokens. The device part is any object with route_outputs(layer, x, ids, store, slots) -> y [tokens * top][hidden]
(routes of slot -2 and id -1 left as they were) and combine(y, ids, weights) (the experts plugin's fact experts:routes);
DeclaredRoutes is the baseline one here: the declared order in torch on the device, exact and slow, until an experts
plugin's own kernel takes its place. Without the native CPU tier or the CUDA build the CPU routes are computed on the
host after a synchronization (the same bits); a capture then raises.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


KERNEL = "cpu-routes"   # crit's kernel of the in-stream CPU routes (plugin.json "kernels")


def extension():
    """(module, None) for crit's kernel cpu-routes built for this device by crit's kernel build
    (crit_plugins.kernels.load_kernel, as the plugin whose package imports this code: heterogeneous-expert-dispatch or
    expert-offload), or (None, why) where it does not build. It also asks Core for the device's joins, so Core's
    handoff is built and measured here too."""
    import torch
    from crit_plugins.host import NativeUnavailable, current
    from crit_plugins.kernels import fallbacks, load_kernel, plugin_folder
    module = load_kernel(KERNEL, folder=plugin_folder(__name__, HERE.parent))
    if module is None:   # no compiler or no crit_kernels: the host path serves
        return None, f"the CPU routes' in-stream join did not build: {fallbacks().get(KERNEL, '')}"[:400]
    try:
        current().joins(torch.cuda.current_device())
    except NativeUnavailable as error:   # Core's handoff did not build: the host path serves
        return None, f"the host's handoff did not build: {str(error).splitlines()[0][:300]}"
    return module, None


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


class KernelRoutes:
    """The device part on crit's grouped-experts kernel: every device route's unweighted row in the declared integer
    order (the CPU tier's bits) and, for a call no CPU route takes part in, optionally in the kernel's tensor arithmetic
    (BF16 tensor cores: the fast prompt path, other bits than the declared order). Routes to negative slots keep their
    rows (the CPU's are scattered into them). One workspace per call shape, made outside a capture. scratch: the
    engine's scratch (crit_plugins.memory's Scratch); with it a call's row buffers are scratch, taken by every
    call (they are read before the call returns its sum), so the shapes a server meets cost the largest call's
    bytes and not a set each, and they share those with every other plugin's calls. Without it each workspace has
    rows of its own."""

    SCRATCH_ROUTES = 1 << 16   # the routes a call's rows take from the scratch; a larger call has its own buffers

    def __init__(self, kernel, layout, kernel_layout, format, hidden, inter, a4=None, scratch=None,
                 activation="silu", caps=None):
        self.kernel, self.format = kernel, format
        # W4A4 at the bundles' input scales: the tier's nvfp4 bundles carry them; a4=False runs weight-only W4A16 rows
        # (tensor arithmetic only: the declared order is W4A4)
        self.a4 = format == "nvfp4" if a4 is None else bool(a4)
        # activation: the blocks' gate (silu: SwiGLU; gelu_tanh and softcapped_silu with its caps in the tensor
        # arithmetic only)
        self.layout = (kernel_layout(layout) if activation == "silu" else kernel_layout(layout, activation) if caps is None
                       else kernel_layout(layout, activation, caps))
        self.hidden, self.inter = hidden, inter
        self.workspaces = {}
        self.scratch = scratch
        self._packed = {}   # routes of a call -> {name: (first byte, bytes)} of its rows in the scratch, "end": bytes

    def _rows(self, name, count, width, dtype):
        """[count][width] of dtype for calls of `count` routes: the next bytes of such a call's scratch (a workspace
        asks once for each buffer it uses, and every call takes them all: route_outputs)."""
        import torch
        nbytes = torch.empty(0, dtype=dtype).element_size() * width * count
        if count > self.SCRATCH_ROUTES:
            return torch.zeros(count, width, dtype=dtype, device=self.scratch.memory.device)
        packed = self._packed.setdefault(count, {"end": 0})
        if name not in packed:
            first = -(-packed["end"] // 64) * 64
            packed[name], packed["end"] = (first, nbytes), first + nbytes
        first, nbytes = packed[name]
        return self.scratch.take(packed["end"])[first:first + nbytes].view(dtype).view(count, width)

    def close(self):
        self.workspaces, self._packed = {}, {}

    def prepare(self, tokens, top, experts, device, arithmetic="declared"):
        """The workspace of calls of `tokens` tokens x `top` routes over a table of `experts` slots, with every row
        buffer `arithmetic` uses, and the call's take of the scratch. A caller whose cache gives device memory back
        calls it before it stages a call's experts: the first call of a shape grows the scratch, the cache releases
        slots for that, and slots staged before would be among them. route_outputs calls it too."""
        import torch
        key = (tokens, top, experts, str(device))
        ws = self.workspaces.get(key)
        if ws is None:
            if capturing():
                raise RuntimeError(f"call the experts once at {tokens} tokens x {top} routes outside a capture first: "
                                   "the device part's workspace is made then")
            ws = self.workspaces[key] = self.kernel.Workspace(
                tokens, top, experts, self.hidden, self.inter, device,
                **({"rows": self._rows} if self.scratch is not None else {}))
        if arithmetic == "declared":      # the rows the kernel makes at an arithmetic's first call, made here
            ws.declared(self.layout)
        elif arithmetic in ("fp4", "fp8"):   # the quantized operands' rows of the own-tensor-core arithmetics
            getattr(ws, arithmetic)()
        elif arithmetic in ("fp4-row", "fp4-row2"):   # and of FP4 at each route's own scale (one or two terms)
            ws.fp4_rows({"fp4-row": 1, "fp4-row2": 2}[arithmetic])
        elif self.a4 and ws.xa is None:
            ws.xa = ws._rows("xa", ws.routes, ws.hidden, torch.bfloat16)
        if tokens * top in self._packed:   # the call's rows: its take of the scratch, as every call's
            self.scratch.take(self._packed[tokens * top]["end"])
        return ws

    def route_outputs(self, layer, x, ids, store, slots, arithmetic="declared"):
        import torch
        tokens, top = ids.shape
        ws = self.prepare(tokens, top, slots.numel(), x.device, arithmetic)
        xb = x if x.dtype == torch.bfloat16 and x.is_contiguous() else x.to(torch.bfloat16).contiguous()
        s = (slots if slots.dtype == torch.int32 else slots.int()).contiguous()
        return self.kernel.route_outputs(xb, ids.contiguous(), store, s, self.layout, ws, self.a4, arithmetic)

    def combine(self, y, ids, weights):
        import torch
        w = weights if weights.dtype == torch.float32 and weights.is_contiguous() else weights.float().contiguous()
        return self.kernel.combine(y, ids.contiguous(), w, None, (1 << 30) - 1, True)


def kernel_part(kernel, tier_bundles, grouped_bundles, format, hidden, inter, scratch=None):
    """KernelRoutes over the tier's bundles (its rows in `scratch`, the engine's), or None with the reason where the
    kernel cannot read them as they are."""
    if kernel is None:
        return None, "the grouped-experts kernel is not built here"
    theirs = tier_bundles.layout(format, hidden, inter)
    try:
        mine = grouped_bundles.layout(format, hidden, inter)
    except ValueError as error:
        return None, str(error)
    def offsets(plan):
        return plan["bundle_bytes"], {name: int(c[0]) for name, c in plan["components"].items()}
    if offsets(mine) != offsets(theirs):
        return None, f"grouped-experts' {format} bundle layout differs from the CPU tier's"
    return KernelRoutes(kernel, mine, grouped_bundles.kernel_layout, format, hidden, inter, scratch=scratch), None


class DeviceDispatch:
    """The CUDA dispatch over a CPU tier (host-expert-tier's CpuTier) and a device part. ext: extension()'s module or
    None (the host path); stride: experts per layer in the tier's key space (key = layer * stride + expert), the length
    every slot table must have; layers: the layers whose experts the tier holds."""

    def __init__(self, device, hidden, inter, format, tier, part, ext, stride, layers, max_tokens=64, join=None,
                 mail_slots=2, joins=None, prompt_arithmetic="declared", owner="heterogeneous-expert-dispatch"):
        import torch
        from crit_plugins.host import current
        if prompt_arithmetic not in ("declared", "tensor"):
            raise ValueError(f"prompt_arithmetic is declared or tensor, not {prompt_arithmetic!r}")
        if prompt_arithmetic == "tensor" and not isinstance(part, KernelRoutes):
            raise ValueError("prompt_arithmetic tensor needs the grouped-experts kernel's device part")
        self.prompt_arithmetic = prompt_arithmetic
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden, self.inter, self.format = hidden, inter, format
        self.tier, self.part, self.stride, self.layers = tier, part, int(stride), set(layers)
        self.max_tokens, self.mail_slots = int(max_tokens), int(mail_slots)
        self.host, self.owner = current(), owner   # the host's handoffs (Core's): this dispatch's is owner/cpu-routes
        self.ext = ext if tier.native else None
        # box: the handoff; kernels: crit's kernel cpu-routes' post and scatter over its slots
        self.box, self.kernels, self.top, self.joins, self.buffers, self.posted = None, None, 0, None, {}, set()
        self.eager_calls = 0
        self.tensor_calls = 0   # prompt calls past max_tokens run in the tensor arithmetic
        if self.ext is not None:
            self.joins = joins or self.host.joins(self.device)
            self.join_kind = join or self.joins["choice"]
            self.counters = torch.zeros(2, dtype=torch.int64, device=self.device)   # jobs posted, routes posted

    @property
    def in_stream(self):
        return self.ext is not None

    def slot_of(self, layer):
        return layer % self.mail_slots

    def _box(self, top):
        """The handoff (the host's, with the tier's handler) and the kernels over its slots, sized for max_tokens
        tokens of `top` routes; made at the first post, made again (outside a capture) for a call with more routes
        a token."""
        import torch
        if self.box is not None and top <= self.top:
            return self.kernels
        if capturing():
            raise RuntimeError(f"post a call of {top} routes a token once outside a capture first: the handoff is sized then")
        if self.box is not None:
            self.close_box()
        routes = self.max_tokens * top
        serve, context = self.tier.handler()
        tier_layout = self.tier.payload_layout(routes, self.max_tokens)
        box = self.host.handoff(self.device, self.owner, "cpu-routes", self.mail_slots, tier_layout["bytes"], serve,
                                context, join=self.join_kind)
        kernels = self.ext.Routes(self.hidden, routes, self.max_tokens, [box.words(s) for s in range(self.mail_slots)])
        mine = dict(zip(("experts", "token", "x", "out", "bytes"), kernels.layout()))
        if mine != tier_layout:
            box.close()
            raise RuntimeError(f"the kernels' payload {mine} is not the CPU tier's {tier_layout}")
        self.box, self.kernels, self.top, self.routes_cap = box, kernels, top, routes
        self.buffers = {s: {"routes": torch.zeros(routes, dtype=torch.int32, device=self.device),
                            "count": torch.zeros(2, dtype=torch.int32, device=self.device),
                            "status": torch.zeros(1, dtype=torch.int32, device=self.device)}
                        for s in range(self.mail_slots)}
        return kernels

    def _check(self, layer, ids, slots):
        if self.box is not None and self.box.failed():   # a host atomic: no synchronization
            raise RuntimeError(f"the CPU tier failed and its handoff answers every job as failed: {self.box.error()}")
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
        kernels = self._box(ids.shape[1])
        b = self.buffers[self.slot_of(layer)]
        kernels.post(self.slot_of(layer), ids.contiguous(), slots, x.contiguous(), int(layer) * self.stride,
                     b["routes"], b["count"], self.counters)
        self.posted.add(layer)

    def forward(self, layer, x, ids, weights, store, slots):
        if not self.in_stream:
            if capturing():
                raise RuntimeError("the CPU routes join inside a capture only through the in-stream handoff, which is "
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
            self.kernels.scatter(s, y, b["routes"], b["count"], b["status"], ids.numel())
            self.posted.discard(layer)
            return self.part.combine(y, ids, weights)
        return self._long(layer, x, ids, weights, store, slots)

    def _long(self, layer, x, ids, weights, store, slots):
        """A call past max_tokens (prompt chunks; host-driven, never captured): the handoff only when it has -2 routes,
        then in pieces of max_tokens tokens."""
        if capturing():
            raise RuntimeError(f"a captured call has at most {self.max_tokens} tokens (max_tokens)")
        flat = ids.reshape(-1).long()
        cpu = bool(((flat >= 0) & (slots.long()[flat.clamp(min=0)] == -2)).any())   # one synchronization
        if not cpu and self.prompt_arithmetic == "tensor":   # no CPU route to match: the kernel's tensor arithmetic
            self.tensor_calls += 1
            return self.part.combine(self.part.route_outputs(layer, x, ids, store, slots, "tensor"), ids, weights)
        y = self.part.route_outputs(layer, x, ids, store, slots)
        if cpu:
            top = ids.shape[1]
            kernels = self._box(top)
            for t0 in range(0, ids.shape[0], self.max_tokens):
                t1 = min(ids.shape[0], t0 + self.max_tokens)
                s = self.slot_of(layer)
                b = self.buffers[s]
                kernels.post(s, ids[t0:t1].contiguous(), slots, x[t0:t1].contiguous(), int(layer) * self.stride,
                             b["routes"], b["count"], self.counters)
                self.box.join(s)
                kernels.scatter(s, y[t0 * top:t1 * top], b["routes"], b["count"], b["status"], (t1 - t0) * top)
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
        y[cpu] = rows.to(device=y.device, dtype=y.dtype)   # the device part's rows: FP32, or BF16 (GGUF)

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
                    self.kernels.post(s, ids, table, x, layers[g % len(layers)] * self.stride, b["routes"], b["count"],
                                      self.counters)
                    for i in range(k):
                        scratch[i].copy_(bundles[(g * most + i) % len(bundles)], non_blocking=True)
                    self.box.join(s)
                    self.kernels.scatter(s, y, b["routes"], b["count"], b["status"], ids.numel())

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
        """The handoff's state: join kind, failure and its error (host words, no synchronization), and, read on the
        device (synchronizes), each of its slots' last answer (1 served, 2 skipped; others: the window is invalid), its
        overflow flag and the jobs and routes the device posted (replays included)."""
        if not self.in_stream:
            return {"in_stream": False, "host_calls": self.eager_calls, "tensor_calls": self.tensor_calls}
        out = {"in_stream": True, "join": self.join_kind, "jobs": int(self.counters[0]), "routes": int(self.counters[1]),
               "tensor_calls": self.tensor_calls}
        if self.box is not None:
            out.update({"failed": self.box.failed(), "error": self.box.error(), "served": self.box.served(),
                        "status": [int(self.buffers[s]["status"][0]) for s in range(self.mail_slots)],
                        "overflow": [int(self.buffers[s]["count"][1]) for s in range(self.mail_slots)]})
        return out

    def close_box(self):
        """Answer every pending doorbell, let the streams drain, then give the handoff back (before the tier it
        calls)."""
        if self.box is not None:
            self.box.close()
            self.box = self.kernels = None

    def close(self):
        self.close_box()
