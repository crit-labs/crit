"""A model plugin composed from blocks (blocks.py) by a layout (layouts/*.json, or an engine's own), on the serving
contracts every model plugin keeps (rows.py), as Core's step model.

The layout names a block for each layer kind of the config (layer_types: full_attention, sliding_attention,
linear_attention) and the FFN of each layer (`auto`: moe on the layers the config routes experts in,
crit_core.model.routed_layers, swiglu elsewhere); the residual stream runs RMSNorm (the layout's offset: 1 + weight
for zero-centred norms), mixer, RMSNorm with the residual add fused, FFN, the next layer's norm with its add fused.

Serving: a step of many sequences' rows is one batch, its matmuls in RowGroups (exact rows); a prompt chunk ends at
its span (Spans: within the cancel grace, its peak within the device's room) or at a place the prompt cache marks;
decode rows run in Core's captured rounds (crit_plugins.rounds.Rounds: sampled mode, greedy a row at temperature 0,
drafted from the checkpoint's MTP layer when the selection drafts); replays, constrained (plain) rows and what rounds
do not hold run through step(); fixed-text scores run the prompt path in the same spans; Core's prompt cache keeps
ended requests' sequences in the pools. Every tensor kept on the device is planned through the placement plugin
first, the experts excepted where the experts plugin places them itself (experts:offload)."""
import json
import time

from crit_core.batch import DECODE, DEFERRED, PREFILL, READY, PageReservations, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch
from crit_plugins.device import RoundBatch, free_device_bytes, release_cached_under
from crit_plugins.host import memory_bytes
from crit_plugins.prompt_cache import PromptCache

from .blocks import BLOCKS, Context, GatedDelta, MoE, router_rule
from .checkpoint import Matrices, open_checkpoint, stop_tokens
from .passes import Pass
from .residual import RESIDUALS
from .rows import GRACE, QUANTUM, SECONDS, RowGroups, Spans, fit

CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
CONTINUATION = 8192   # replayed tokens a round slot holds
STOPS = 8             # stop ids a row in a round slot holds
TOP = 20              # the alternatives a token's log-probabilities carry at most (max_top_logprobs)
PROMPT_TAIL = 8       # a prompt chunk ends this many tokens before the prompt's end, where the prompt cache marks
PROMPT_CUT = 256      # the shortest prompt cut there: a shorter one is prefilled whole in about the cut's time
SCORE_SLICE = 128     # rows of logits a score holds at once (FP32 [rows][vocabulary])
PAGE = 16             # KV page tokens
FILL = 8192           # positions of zero KV a calibration writes at once
PASS = "pass"         # a session's phase in a step of a pass that did not end it (passes.py)


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs, self.pending = 0, [], None    # rows committed; published outputs
        self.phase, self.produced, self.rows = None, [], 0
        self.slot, self.pages, self.last_hidden, self.mtp_done = None, 0, None, 0
        self.graphed = False   # its last step's prompt chunk ran as a captured graph (its pools settled already)
        self.reused = 0        # the prefix the prompt cache gave it, until its first prompt step reports it
        self.tail = None       # the target's hidden of its last prompt chunk's last row (the prompt cache's mark)
        self.resume = None     # the same of the row before the prefix it resumed: the MTP pool's row there is owed
        self.constrained = False
        self.passing = None    # the pass its prompt is in (passes.py): a span run layer by layer over several steps
        self.settled = False   # its last step ended a pass: its pools took the span's state as its layers ran
        self.broken = False    # a pass it did not finish moved its pools past its rows: its sequence is not kept


class _MtpRows:
    """The MTP pool as the prompt cache keeps it: its row of a position is written from the token after it, so of a
    prefix it holds one row fewer than the target's pool (the last is written when the next token is known)."""

    def __init__(self, pool):
        self.pool = pool
        self.open, self.close, self.fork = pool.open, pool.close, pool.fork

    def keep(self, sequence, length):
        self.pool.keep(sequence, min(max(length - 1, 0), self.pool.length(sequence)))


class Rope:
    """The config's rotary table on the device: FP32 inverse frequencies (partial rotation: rotary_dim / 2 of them) and
    the attention factor cos and sin are scaled by."""

    def __init__(self, config, head_dim, device, rope=None):
        """rope: the rope parameters of one layer type (a config that keys its rope by layer type), the config's own
        by default."""
        import torch
        from ..rope.rope_table import rope_table   # attention-state's derivation from the config's rope parameters
        table = rope_table(config.rope if rope is None else rope, head_dim, config.max_positions)
        self.freqs = torch.tensor(table.inv_freq, dtype=torch.float32, device=device)
        self.scale = float(table.attention_factor)


class Layer:
    """A decoder layer: its mixer and FFN blocks, what the residual keeps for it (residual.py: a pre-norm stream's two
    norm weights, the streams' two mixes) and the blocks that add rows into the residual before it."""

    def __init__(self, kind, mixer, ffn, res, before=()):
        self.kind, self.mixer, self.ffn, self.res, self.before = kind, mixer, ffn, res, list(before)


class MtpHead:
    """The model's device draft head (draft:head-device) over the checkpoint's MTP layer, for captured rounds of up to
    `rows` rows at up to `widest` drafts. features[r][c] is the target's draft feature (the residual's: its
    final-normed hidden state, or its streams) of the last round's verify column c of row r (the seed's: the prompt's
    last row), which the catch-up reads with the emitted token c at position frontier - n + c; hidden[r] is the
    chain's last MTP feature and out[r] the rows the draft head reads of it."""

    def __init__(self, model, rows, widest):
        torch = model.torch
        self.m, self.torch, self.rows, self.widest, self.cols = model, torch, rows, widest, widest + 1
        bf16 = dict(dtype=torch.bfloat16, device=model.device)
        self.width = model.residual.width
        self.features = torch.zeros(rows, self.cols, self.width, **bf16)
        self.hidden = torch.zeros(rows, self.width, **bf16)
        self.out = torch.zeros(rows, model.hidden, **bf16)
        self.catch = RoundBatch(rows, widest, model.device)
        self.chain = RoundBatch(rows, 0, model.device)

    def seed_feature(self, slot, hidden):
        self.features[slot, 0].copy_(hidden)

    def move(self, source, target):
        self.features[target].copy_(self.features[source])
        self.hidden[target].copy_(self.hidden[source])
        self.out[target].copy_(self.out[source])

    def keep_features(self, view, hidden):
        rows, cols = view.rows, view.width + 1
        held = (view.hold[:rows] > 0)[:, None, None]
        self.features[:rows, :cols] = self.torch.where(held, self.features[:rows, :cols], hidden)

    def catch_up(self, view):
        rows, cols, w = view.rows, self.cols, self.width
        count = view.catch_up_count[:rows]
        base = view.frontier[:rows] - count
        batch = self.catch.fill(rows, self.widest, base, view.catch_up_tokens[:rows])
        out, feature = self.m._mtp(batch.tokens_t, self.features[:rows].reshape(rows * cols, w), batch)
        last = (count - 1).clamp(min=0)[:, None, None]
        self.hidden[:rows] = feature.view(rows, cols, w).gather(1, last.expand(rows, 1, w)).squeeze(1)
        self.out[:rows] = out.view(rows, cols, -1).gather(1, last.expand(rows, 1, out.shape[-1])).squeeze(1)

    def logits(self, depth, tokens, view, parents=None):
        rows = view.rows
        if depth:
            batch = self.chain.fill(rows, 0, view.frontier[:rows] + (depth - 1), tokens[:rows, None])
            self.out[:rows], self.hidden[:rows] = self.m._mtp(batch.tokens_t, self.hidden[:rows], batch)
        return self.m._logits_rows(self.out[:rows], draft=True)

    def commit_round(self, view):
        """Nothing beyond the MTP pool's own commit: its positions are rewritten by the next catch-up."""


def _host_available():
    """The host memory a process can take without paging (Core's reading: Linux's MemAvailable, Windows'
    available physical memory)."""
    return memory_bytes()["available"]


class ComposedModel:
    def __init__(self, context, options, parts, layout, blocks=None, folder=None):
        import torch
        from crit_core.model import parse_model_config
        self.torch, self.options, self.parts, self.layout = torch, options, parts, layout
        self.folder = folder   # the plugin's folder, whose plugin.json declares the kernels its blocks load
        self.placed = {}       # where the placement plugin put each planned group (a block's lookup table)
        self._ropes = {}       # rotary tables of other head widths than the model's (rope_of)
        self.blocks = {**BLOCKS, **(blocks or {})}
        config = self.config = parse_model_config(context.config)
        self.quant = context.config.get("quantization_config") or config.get("quantization_config") or {}
        self.kinds = config.layer_kinds()
        # The config's own names of its layer types, which a layout may key its mixers by (a name Core's kinds do
        # not know is "other" there).
        self.types = tuple(config.layer_types) if len(config.layer_types) == len(self.kinds) else self.kinds
        self.layers, self.hidden = len(self.kinds), config.hidden_size
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.effective_head_dim()
        self.eps, self.vocabulary = config.rms_norm_eps or 1e-6, config.vocab_size
        self.norm_offset = float(layout.get("norm_offset", 0.0))
        res = dict(layout.get("residual") or {"block": "pre-norm"})
        if res.get("block") not in RESIDUALS:
            raise ValueError(f"model-template: no residual {res.get('block')!r} (residuals: "
                             f"{', '.join(sorted(RESIDUALS))})")
        self.device = torch.device(str(getattr(context, "device", None) or "cuda"))
        self.context = int(context.context_tokens or options["context"])
        self.concurrency, self.pending_max = int(options["concurrency"]), int(options["pending"])
        self.chunk = int(options["prefill_chunk"])
        self.free_below = int(options["release_below_mib"]) << 20
        self.sequences = self.concurrency + 1   # every request decoding at once, and a score's
        self.kept = int(options["prompt_cache"])
        self.proj, self.attention, self.sampler = parts["projections"], parts["attention"], parts["sampler"]
        self.norms, self.embeddings = parts["norms"], parts["embeddings"]
        self.recurrence, self.router, self.experts = parts.get("recurrence"), parts.get("router"), parts.get("experts")
        self.drafter = parts.get("drafter")
        self.fused_norms = "norms:fused" in parts.facts("norms")
        self.offload = "experts:offload" in parts.facts("experts")
        self.rows = RowGroups(self.proj)
        # The checkpoint's headers and tensors: the model folder's (its safetensors files, or a GGUF read under the same
        # names), or one the context hands over (a planner reading a checkpoint manifest, tests/support/dry_plan.py,
        # has headers and no weights).
        handed = getattr(context, "checkpoint", None)
        ck, matrices = (handed, Matrices) if handed is not None else open_checkpoint(context.model, context.config)
        self.ck = ck
        self.matrices = matrices(ck, self.proj, self.quant, self.device)
        # A GGUF's experts as stored where the experts plugin computes them so (experts:gguf), else their values
        self.matrices.stored_experts = "experts:gguf" in parts.facts("experts")
        self.residual = RESIDUALS[res["block"]](self, res)
        draft = layout.get("draft") or {}
        self.draft_prefix = draft.get("prefix")
        self.mtp = (bool(options["mtp"]) and self.drafter is not None and self.draft_prefix is not None
                    and self.residual.draft_stored(self.draft_prefix))
        if self.drafter is not None and not self.mtp:
            raise ValueError("model-template drafts from the checkpoint's MTP layer: the selection has a drafter, so "
                             "set option mtp and give a checkpoint with its draft layer's tensors (the layout's draft "
                             "prefix), or leave the drafter slot empty")
        if getattr(self.drafter, "uses_head", True) is False:
            # A drafter that never asks the draft head (mtp-decode's copy drafts alone) would leave the draft layer
            # loaded, written at every prompt position and caught up every round, while none of its drafts is one.
            raise ValueError(f"model-template drafts from the checkpoint's MTP layer, and the drafter "
                             f"{parts.plugin('drafter')}, as configured, never asks the model's draft head (mtp-decode "
                             "at override 0 proposes prompt copies alone): select greedy-head-drafter for the draft "
                             "layer's chain, give mtp-decode an override of 1 or more (the chain, long copies over "
                             "it), or leave the drafter slot empty")
        self.width = int(options["width"]) if self.mtp else 0
        for key, want in (layout.get("requires") or {}).items():   # parts of the family this layout does not compute
            found = config.get(key)
            if (found or type(want)()) != want:
                raise ValueError(f"model-template: the {layout.get('name')} layout computes checkpoints whose {key} is "
                                 f"{want!r}, and this one's is {found!r}: its layers have a part no block here computes")
        self.routed = set(config.routed_layers())
        if layout.get("routed_if") and not self.setting(layout["routed_if"]):
            self.routed = set()   # the family's switch for its experts (a config key) is off: every layer is dense
        # The gate of the MLPs' and the experts' product: SiLU (SwiGLU), or GELU in its tanh form.
        act = config.get("hidden_act") or config.get("hidden_activation") or "silu"
        self.activation = {"silu": "silu", "swish": "silu", "gelu_pytorch_tanh": "gelu_tanh"}.get(act)
        if self.activation is None:
            raise ValueError(f"model-template: the checkpoint's activation is {act}; its gated blocks compute silu "
                             "and gelu_pytorch_tanh")
        if self.routed and self.activation != "silu" and self.experts is not None:
            if "experts:activations" not in parts.facts("experts"):
                raise ValueError(f"model-template: the checkpoint's experts are gated by {act}, and the experts "
                                 f"plugin {parts.plugin('experts')} computes SwiGLU blocks alone (it does not provide "
                                 "experts:activations)")
            self.experts.activate(self.activation)
        # The experts slot's layer ids: the routed layers in order, then the draft layer's (every experts plugin
        # holds layers 0 .. n - 1)
        self.expert_ids = {layer: k for k, layer in enumerate(sorted(self.routed))}
        self._slots_needed()

        # -- names -----------------------------------------------------------------------------------------------------
        anchor = "layers.0." + self.residual.anchor()
        prefixes = [n[:-len(anchor)] for n in ck.names() if n.endswith(anchor)
                    and not (self.draft_prefix and n.startswith(self.draft_prefix))]
        if not prefixes:
            raise ValueError("model-template: no decoder layer 0 in the checkpoint")
        self.prefix = prefix = sorted(prefixes, key=len)[0]   # "model.language_model." or "model."
        tied = not self.matrices.has("lm_head")
        embed_name = prefix + "embed_tokens.weight"
        # Where a routed layer's experts are stored under its prefix: the FFN's prefix (none for a block made at
        # its layer's), then the experts' own.
        ffn = dict(layout.get("ffn") or {})
        routed_block = self._block(ffn.get("routed", "moe") if ffn.get("block", "auto") == "auto" else ffn["block"])
        self.experts_at = (("" if getattr(routed_block, "at_layer", False) else ffn.get("prefix", "mlp."))
                           + ffn.get("experts", "experts."))
        experts_prefixes = [f"{prefix}layers.{i}.{self.experts_at}".rstrip(".") for i in self.routed]
        if self.mtp:
            experts_prefixes.append(self.draft_prefix + "layers.0.mlp.experts")
        outside = set(self.residual.names(prefix))
        tables = {}   # the lookup tables the layout's blocks read rows of: plan name -> (bytes, tensor prefix)
        for i in range(self.layers):
            for spec in self._before(i):
                found = getattr(self._block(spec["block"]), "tables", None)
                if found is not None:
                    tables.update(found(self, f"{prefix}layers.{i}." + spec.get("prefix", ""), spec))
        tabled = tuple(at for _, at in tables.values())
        kept = [n for n in ck.names() if (n.startswith(prefix + "layers.") or n in outside
                                          or n.startswith("lm_head.") or (self.mtp and n.startswith(self.draft_prefix)))
                and not n.startswith(tabled)]
        offloaded = [n for n in kept if self.offload and any(n.startswith(p) for p in experts_prefixes)]
        self.kv_dtype = options["kv"]
        # The attention layers whose keys and values the model's KV pools hold (a mixer may keep its own: keeps_kv).
        full = [i for i, k in enumerate(self.kinds) if k != "linear_attention"
                and getattr(self._block(self._layer_spec(i)["block"]), "keeps_kv", True)]
        scales = bool(full) and all(f"{prefix}layers.{i}.self_attn.k_scale" in ck for i in full)
        if self.kv_dtype == "auto":
            self.kv_dtype = "fp8" if scales else "bf16"
        if self.kv_dtype == "fp8" and not scales:
            raise ValueError("model-template: fp8 KV pages take the checkpoint's static k_scale and v_scale, which it "
                             "does not store: choose kv bf16 or fp8:row")
        self.kv_scales = self.kv_dtype == "fp8"

        # -- the plan ----------------------------------------------------------------------------------------------------
        self.kv_tokens = self.context + int(options["kv_margin"])
        pages = -(-self.kv_tokens // PAGE)
        linear = [i for i, k in enumerate(self.kinds) if k == "linear_attention"]
        windows = {}
        for i in full:
            windows.setdefault(self._window(i), []).append(i)

        def pool_bytes(layers, dtype, tokens, shape=None):
            kv_heads, head_dim = shape or (self.kv_heads, self.head_dim)
            element = {"bf16": 2, "fp8": 1, "fp8:row": 1}[dtype]
            pages_ = -(-tokens // PAGE)
            found = 2 * layers * pages_ * PAGE * kv_heads * head_dim * element
            if dtype == "fp8:row":
                found += 2 * layers * pages_ * PAGE * kv_heads * 4
            return found

        # Each window's layers share a pool, so they share a head shape (a family whose layer kinds differ in shape
        # differs in window too: its full layers a pool, its sliding ones another).
        shapes = {}
        for window, layers in windows.items():
            found = {self.kv_shape(i, self._layer_spec(i)) for i in layers}
            if len(found) > 1:
                raise ValueError(f"model-template: the attention layers of window {window} have key/value heads x "
                                 f"widths {sorted(found)}: layers that share a KV pool share a head shape")
            shapes[window] = found.pop()
        kv_bytes = 0
        for window, layers in windows.items():
            tokens = self.kv_tokens if not window else self.sequences * (window + self.chunk + 2 * PAGE)
            kv_bytes += pool_bytes(len(layers), self.kv_dtype, tokens, shapes[window])
        if self.mtp:
            kv_bytes += pool_bytes(1, options["mtp_kv"], self.kv_tokens)
        state_bytes = 0
        if linear:
            lin = config.linear
            conv = 2 * lin.num_key_heads * lin.key_head_dim + lin.num_value_heads * lin.value_head_dim
            n, cap = self.sequences + 1, self.width + 1
            state_bytes = len(linear) * n * (lin.num_value_heads * lin.key_head_dim * lin.value_head_dim
                                             + conv * (lin.conv_kernel_dim - 1) + 2 * conv * cap
                                             + 2 * cap * (lin.num_key_heads * lin.key_head_dim + lin.num_value_heads
                                                          + lin.num_value_heads * lin.value_head_dim)) * 4
            state_bytes = int(state_bytes * 1.05) + (32 << 20)
        declared = self._declared()
        for d in declared:
            if d["kind"] == "keys":     # BF16 keys, a row a position
                kv_bytes += len(d["layers"]) * pages * PAGE * d["heads"] * d["dim"] * 2
            else:                       # a convolution's window of each sequence and round row, FP32
                span = (d["kernel"] - 1) * d.get("dilation", 1) + 2 * (self.width + 1)
                rows = len(d["layers"]) * (self.sequences + 1)
                state_bytes += int(rows * d["channels"] * span * 4 * 1.05) + (8 << 20)
        self.activations = int(options["activations_mib"]) << 20
        away = set(offloaded)   # once: a checkpoint of routed experts names hundreds of thousands of tensors
        tensors = {name: ck.nbytes(name) for name in kept if name not in away}
        # The token embeddings: a lookup table a placement may put off the device where they can be read in place,
        # else a device weight like any other.
        self.embed_plan = "tables.embed_tokens" if self._embed_in_place(embed_name) else "embed_tokens"
        tensors[self.embed_plan] = ck.nbytes(embed_name)
        tensors.update({name: nbytes for name, (nbytes, _) in tables.items()})
        tensors.update({"kv": kv_bytes, "states": state_bytes, "activations": self.activations})
        if self.mtp and options["draft_head"] != "shared":
            head_bytes = ck.nbytes("lm_head.weight" if not tied else embed_name)
            tensors["draft_head"] = head_bytes // 2 if options["draft_head"] == "nvfp4" else head_bytes
        if self.mtp and self._draft_moe():
            self.expert_ids[self.layers] = len(self.expert_ids)
        # The format the experts slot holds every layer's experts in: the first routed layer's. A draft layer's
        # experts stored otherwise are re-encoded to it (draft-only: the target verifies every draft).
        self.expert_format = None
        if self.routed:
            first = f"{prefix}layers.{min(self.routed)}.{self.experts_at}"
            self.expert_format = ("bf16" if first + "gate_up_proj" in ck
                                  else self.matrices.format(first + "0.gate_proj"))
        if self.offload:
            moe, fmt, stored = config.moe, self.expert_format, {}
            if fmt.startswith("gguf:") and not self.matrices.stored_experts:
                fmt = "bf16"
            if fmt.startswith("gguf:"):   # each layer's experts as the file stores them: its formats, a layer each
                fmt, stored["formats"] = "gguf", {
                    k: self.matrices.expert_formats(f"{prefix}layers.{i}.{self.experts_at}" if i < self.layers
                                                    else f"{self.draft_prefix}layers.0.mlp.experts.")
                    for i, k in self.expert_ids.items()}
            # what the slot is begun with; a GGUF's layers each keep their own formats
            self.expert_format = None if fmt == "gguf" else fmt
            self.experts.begin(len(self.expert_ids), moe.experts, fmt, self.hidden, moe.moe_intermediate_size,
                               device_bytes=kv_bytes + state_bytes + self.activations, **stored)
        free = self._free()   # after the load's own temporaries go back to the device
        placed = self.placed = parts["placement"].plan(tensors, {"device": free, "host": _host_available()})
        if any(place != "device" for name, place in placed.items() if not name.startswith("tables.")):
            raise ValueError("model-template keeps every weight, the KV and the states on the device (routed experts "
                             "off it through an experts plugin that places them: expert-offload)")

        # -- the embedding rows (first, while the device has the room for the table in transit) ------------------------
        self.embed = self._embedding(embed_name)

        # -- the layers ------------------------------------------------------------------------------------------------
        self.residual.load(prefix)
        self.head = self.matrices.load(["lm_head" if not tied else prefix + "embed_tokens"], "head")
        self.rope = Rope(config, self.head_dim, self.device)
        scale = layout.get("embed_scale")
        if scale not in (None, "sqrt_hidden"):
            raise ValueError(f"model-template: the layout's embed_scale is sqrt_hidden, not {scale!r}")
        self.embed_scale = float(torch.tensor(self.hidden ** 0.5).to(torch.bfloat16)) if scale else None
        self.softcap = float(self.setting(layout["logit_softcap"]) or 0.0) if layout.get("logit_softcap") else 0.0
        self.geometry = GatedDelta.geometry(self) if linear else None
        self.decoder = []
        for i, kind in enumerate(self.kinds):
            p = f"{prefix}layers.{i}."
            before = [self._block(spec["block"])(self, i, p + spec.get("prefix", ""), spec) for spec in self._before(i)]
            self.decoder.append(Layer(kind, self._mixer(i, p), self._ffn(i, p), self.residual.layer(p), before))
        # The ids some blocks read beside each token's own (hashed n-grams): each token's window of this many ids.
        self.window_tokens = max((getattr(block, "token_window", 0) for layer in self.decoder
                                  for block in layer.before), default=0)
        self._check_router()

        # -- the MTP layer (draft-only: its matrices may be re-encoded, the stored target verifies every draft) ----------
        self.mtp_layer, self.draft_matrix = None, self.head
        if self.mtp:
            self.mtp_layer = self._draft_layer(p=self.draft_prefix, encode=None if options["mtp_encode"] == "bf16"
                                               else options["mtp_encode"])
            if options["draft_head"] != "shared":
                self.draft_matrix = self._head_slices(int(options["draft_head_slices"]), options["draft_head"],
                                                      "lm_head" if not tied else prefix + "embed_tokens")
        self.draft_bytes = self._draft_bytes(tied, embed_name) if self.mtp else 0

        # -- the pools -------------------------------------------------------------------------------------------------
        kv = parts["kv_cache"]
        self.pools = {}
        for window, layers in windows.items():
            key = ("kv", window) if window else "kv"
            extra = dict(sliding=window, span=self.chunk + self.width + 1, positions=self.kv_tokens) if window else {}
            tokens = self.kv_tokens if not window else self.sequences * (window + self.chunk + 2 * PAGE)
            self.pools[key] = kv.pool(layers, *shapes[window], tokens, self.device, self.kv_dtype, page=PAGE, **extra)
            if self.kv_scales:
                for i in layers:
                    self.pools[key].set_scales(i, *self.decoder[i].mixer.kv_scales)
        if self.mtp:
            self.pools["mtp"] = kv.pool([self.layers], self.kv_heads, self.head_dim, self.kv_tokens, self.device,
                                        options["mtp_kv"], page=PAGE)
        if linear:
            lin = config.linear
            conv = 2 * lin.num_key_heads * lin.key_head_dim + lin.num_value_heads * lin.value_head_dim
            # (rows: the rounds' slots, which a commit that keeps a state per slot sizes itself by)
            self.pools["states"] = self.recurrence.pool(linear, lin.num_value_heads, lin.key_head_dim,
                                                        lin.value_head_dim, self.device, self.sequences,
                                                        widest=self.width, conv=(conv, lin.conv_kernel_dim),
                                                        key_heads=lin.num_key_heads, rows=self.concurrency)
        for d in declared:
            if d["kind"] == "keys":
                self.pools[d["key"]] = kv.pool(d["layers"], d["heads"], d["dim"], self.kv_tokens, self.device, "bf16",
                                               page=PAGE, value_dim=0)
            else:
                if self.recurrence is None:
                    raise ValueError(f"model-template: the selection leaves recurrence empty, which keeps the "
                                     f"convolution window {d['key']}")
                self.pools[d["key"]] = self.recurrence.pool(
                    d["layers"], 1, 1, 1, self.device, self.sequences, widest=self.width,
                    conv=(d["channels"], d["kernel"], d.get("dilation", 1)), rows=self.concurrency)
        # The pools by what follows them: the draft layer's (a row behind the target's: written from the token after),
        # the state pools (a sequence's recurrent state and convolution windows), and the rest, KV and keys a position.
        windows_declared = {d["key"] for d in declared if d["kind"] == "window"}
        self.draft_keys = [k for k in self.pools if k == "mtp" or (isinstance(k, str) and k.startswith("mtp-"))]
        self.state_keys = [k for k in self.pools if k == "states" or k in windows_declared]
        aside = set(self.draft_keys) | set(self.state_keys)
        self.kv_keys = [k for k in self.pools if k not in aside]
        self.pool = self.pools.get("kv") or self.pools[self.kv_keys[0]]
        self.kv_pools = [self.pools[k] for k in self.kv_keys]
        self.state_pools = [self.pools[k] for k in self.state_keys]
        self.draft_pools = [self.pools[k] for k in self.draft_keys]
        self.states = self.state_pools[0] if self.state_pools else None
        self.reservations = PageReservations(pages, PAGE, self.context, verify_rows=self.width + 1)
        self.stops = stop_tokens(context.model, context.config)
        self.handles = 0
        if self.offload:
            self.experts.open()

        # -- passes: prompt spans run layer by layer where a layer's experts come to the device for a chunk ---------------
        # What a pass asks of a layer: the KV pools its blocks write (their slots are reserved for each chunk as it
        # runs the layer) and the state pools that follow it (they accept each chunk as it ran).
        self.layer_pools = [[key for key in (getattr(layer.mixer, "pool", None), getattr(layer.mixer, "index", None))
                             if key in self.kv_keys] for layer in self.decoder]
        self.layer_states = [[self.pools[key] for key in dict.fromkeys(
            [getattr(layer.mixer, "pool", None)] + [getattr(block, "pool", None) for block in layer.before])
            if key in self.state_keys] for layer in self.decoder]
        self.passes, self.passes_reason = self._passes(options)
        self.pass_bytes = int(options["pass_mib"]) << 20
        # The most prompt tokens a pass holds (Core's prefill_pass_tokens): those whose states its budget keeps, a
        # prefill chunk at least.
        self.pass_tokens = max(self.chunk, min(self.context, self.pass_bytes // self.residual.carry(2))
                               // QUANTUM * QUANTUM)
        # A step of a pass ends within the span budget and half the cancel grace: a cancel then also gives back
        # what the pass holds (its states, the pages it reserved), which costs the driver time by the gigabyte.
        self.pass_seconds = min(float(options["span_seconds"]) or SECONDS, GRACE / 2)
        self.passed = {"passes": 0, "tokens": 0, "steps": 0, "chunks": 0, "dropped": 0, "open": 0}

        # -- spans: every prompt step within the cancel grace and the device's room ----------------------------------------
        self.spans = self._spans(options)
        self.ck.close()
        self.ck = None   # the file handles go: every tensor is where its plan put it

        # -- the prompt cache --------------------------------------------------------------------------------------------
        self.cache = None
        if self.kept:
            pools = list(self.kv_pools) + [_MtpRows(pool) for pool in self.draft_pools]
            self.cache = PromptCache(pools, states=list(self.state_pools),
                                     entries=self.kept, marks=2, marking=1, tail=PROMPT_TAIL, tail_least=PROMPT_CUT)

        # -- rounds ------------------------------------------------------------------------------------------------------
        uses = ["projections", "attention", "kv_cache", "norms", "embeddings"] + [
            slot for slot, part in (("recurrence", self.recurrence), ("router", self.router),
                                    ("experts", self.experts)) if part is not None]
        self.uses = uses
        self.rounds, self.slots, self.draft_head, self.ring = None, [], None, None
        if options["rounds"]:
            from crit_plugins.rounds import Rounds, TokenRing, width_policy
            pools = list(self.pools.values())
            if self.window_tokens > 1:   # each round slot's last ids, which a verify's windows start from
                self.ring = TokenRing(self.concurrency, self.window_tokens - 1, self.device)
                pools.append(self.ring)
            drafting = {}
            if self.mtp:
                self.draft_head = MtpHead(self, self.concurrency, self.width)
                widths = sorted({int(w) for w in str(options["widths"]).split(",") if w.strip()} | {self.width})
                drafting = dict(drafter=self.drafter, head=self.draft_head, widths=widths,
                                width_policy=width_policy(options["width_policy"]))
            self.rounds = Rounds(self, self.sampler, rows=self.concurrency, widest=self.width, device=self.device,
                                 pools=pools, parts=parts, uses=uses, stops=STOPS, logprobs=TOP,
                                 vocabulary=self.vocabulary, sampled=True, continuation=CONTINUATION, **drafting)

        # -- captured prompt chunks ----------------------------------------------------------------------------------------
        self.prompts, self.prompts_refused = None, None
        buckets = [int(b) for b in str(options["prompt_buckets"]).split(",") if b.strip()]
        eager = sorted({type(block).__name__ for layer in self.decoder for block in (layer.mixer, *layer.before)
                        if not getattr(block, "captures_prompts", True)})
        if buckets and (eager or (self.mtp and self.residual.width != self.hidden)):
            # A captured chunk's forward reads the batch's device tensors alone and gives the head's rows: a block
            # that plans a chunk on the host (its sequence's page table, its ids' windows) or a draft feature other
            # than those rows keeps every chunk eager.
            self.prompts_refused = ("prompt chunks run eager: " + (
                f"{', '.join(eager)} plan a chunk on the host" if eager else
                "a draft layer over residual streams reads the streams, which a captured chunk does not return"))
        elif buckets:
            from crit_plugins.prompt_graphs import PromptGraphs
            self.prompts = PromptGraphs(self._prompt_forward, device=self.device, buckets=buckets, parts=parts,
                                        uses=uses,
                                        pools=[p for k, p in self.pools.items() if k not in self.draft_keys])
            if not self.prompts.problems:
                self.prompts.prepare()

    # -- composing -----------------------------------------------------------------------------------------------------
    def _free(self):
        """The bytes the engine's device has free for a plan: the card's, or on the CPU the host's available memory."""
        return free_device_bytes(self.device) if self.device.type == "cuda" else _host_available()

    def _slots_needed(self):
        """The slots the layout's blocks call, refused here (naming the slot) when the selection leaves one empty."""
        missing = []
        if "linear_attention" in self.kinds and self.recurrence is None:
            missing.append("recurrence (the gated-delta layers)")
        if self.routed and (self.router is None or self.experts is None):
            missing.append("router and experts (the routed MLPs)")
        if missing:
            raise ValueError("model-template: the selection leaves " + " and ".join(missing) + " empty")

    def _spec(self, kind, name=None):
        """The layout's mixer of a layer kind (Core's: full_attention, sliding_attention, linear_attention), looked up
        first by the config's own name of the layer type where it has another."""
        mixers = self.layout.get("mixers") or {}
        spec = mixers.get(name) if name else None
        spec = mixers.get(kind) if spec is None else spec
        if spec is None:
            raise ValueError(f"model-template: the layout has no block for {name or kind} layers (its mixers: "
                             f"{', '.join(sorted(mixers)) or 'none'})")
        return {**spec, "kind": kind}

    def _layer_spec(self, layer):
        return self._spec(self.kinds[layer], self.types[layer])

    def setting(self, key):
        """The config's value of `key`, else what the family's reference assumes where config.json is silent (the
        layout's `defaults`)."""
        found = self.config.get(key)
        return (self.layout.get("defaults") or {}).get(key) if found is None else found

    def _before_layers(self, spec):
        """The layers (numbered from 1, as configs list them) a `before` block runs ahead of."""
        layers = spec["layers"]
        return set(self.config.get(layers) or ()) if isinstance(layers, str) else set(layers)

    def _before(self, layer):
        return [spec for spec in self.layout.get("before") or [] if layer + 1 in self._before_layers(spec)]

    def _declared(self):
        """The pools the layout's blocks keep beside their layers' (a block class's pools(model, spec, layers)): one
        declaration a key, for the layers that run the block; the draft layer's mixer declares its own."""
        groups = {}
        for i in range(self.layers):
            spec = self._layer_spec(i)
            groups.setdefault(json.dumps(spec, sort_keys=True), (spec, []))[1].append(i)
        groups = list(groups.values())
        for spec in self.layout.get("before") or []:
            layers = [i for i in range(self.layers) if spec in self._before(i)]
            if layers:
                groups.append((spec, layers))
        if self.mtp:
            groups.append(({**self._spec("full_attention"), "window": 0, "pool": "mtp"}, [self.layers]))
        found = {}
        for spec, layers in groups:
            declare = getattr(self._block(spec["block"]), "pools", None)
            for d in (declare(self, spec, layers) if declare is not None else []):
                if d["key"] in found or d["key"] in ("kv", "states", "mtp") or d["kind"] not in ("keys", "window"):
                    raise ValueError(f"model-template: block {spec['block']} declares the pool {d['key']!r} "
                                     f"({d['kind']}), which is taken or of no kind (keys, window)")
                found[d["key"]] = d
        return list(found.values())

    def kv_shape(self, layer, spec):
        """(key/value heads, head width) of a layer's attention: what the config states for the layer itself (its
        per_layer_config entry, by the layer's number or its type), else the config keys the mixer's spec names
        (`kv_heads`, `head_dim`), else the model's."""
        per = self.config.get("per_layer_config") or {}
        own = per.get(str(layer)) or per.get(self.types[layer] if layer < len(self.types) else None) or {}
        kv_heads = own.get("num_key_value_heads") or (self.setting(spec["kv_heads"]) if spec.get("kv_heads") else None)
        head_dim = own.get("head_dim") or (self.setting(spec["head_dim"]) if spec.get("head_dim") else None)
        return int(kv_heads or self.kv_heads), int(head_dim or self.head_dim)

    def rope_for(self, layer, head_dim):
        """The rotary table of a layer's attention: its layer type's where the config keys its rope by layer type
        (each type its own base, type and rotated share, at heads of `head_dim`), else the model's."""
        kind = self.types[layer] if layer < len(self.types) else None
        rope = next((entry.rope for entry in self.config.rope_by_layer_type if entry.layer_type == kind), None)
        if rope is None:
            return self.rope_of(head_dim)
        found = self._ropes.get((kind, head_dim))
        if found is None:
            found = self._ropes[(kind, head_dim)] = Rope(self.config, head_dim, self.device, rope)
        return found

    def rope_of(self, dim):
        """The config's rotary table for heads of `dim` rotated values (the model's own where that is its head)."""
        if 2 * self.rope.freqs.numel() == dim or self.head_dim == dim:
            return self.rope
        found = self._ropes.get(dim)
        if found is None:
            found = self._ropes[dim] = Rope(self.config, dim, self.device)
        return found

    def kernel(self, name):
        """crit's kernel `name` for a block, built for this machine, or None (the block then runs its torch form):
        the plugin's plugin.json declares it."""
        if self.device.type != "cuda" or self.folder is None:
            return None
        try:
            from crit_plugins.kernels import load_kernel
            return load_kernel(name, folder=self.folder)
        except Exception:   # a plugin over compose() that does not declare the kernel runs the torch form
            return None

    def _block(self, name):
        found = self.blocks.get(name)
        if found is None:
            raise ValueError(f"model-template: no block {name!r} (blocks: {', '.join(sorted(self.blocks))}); an engine "
                             "passes its own to compose()")
        return found

    def _window(self, layer):
        spec = self._layer_spec(layer)
        window = spec.get("window")
        if window is None and self.kinds[layer] == "sliding_attention":
            window = self.config.sliding_window
        return int(window or 0)

    def _mixer(self, layer, p):
        spec = self._layer_spec(layer)
        return self._block(spec["block"])(self, layer, p + spec.get("prefix", ""), spec)

    def _ffn(self, layer, p, routed=None):
        spec = dict(self.layout.get("ffn") or {"block": "auto"})
        name = spec.get("block", "auto")
        if name == "auto":   # the layout's block for a routed layer (moe) and for a dense one (swiglu)
            name = (spec.get("routed", "moe") if (layer in self.routed if routed is None else routed)
                    else spec.get("dense", "swiglu"))
        made = self._block(name)
        block = made(self, layer, p if getattr(made, "at_layer", False) else p + spec.get("prefix", "mlp."), spec)
        if isinstance(block, MoE):
            if not (self.offload and getattr(self.experts, "complete", False)):
                block.load_experts()
        return block

    def _check_router(self):
        if self.router is None or not self.routed:
            return
        routing = getattr(self.router, "routing", None)
        if routing is None:
            return
        wanted = router_rule(self.config.moe, self.layout.get("defaults"))
        found = {k: getattr(routing, k) for k in wanted if hasattr(routing, k)}
        wrong = {k: v for k, v in wanted.items() if k in found and found[k] != v}
        if wrong:
            raise ValueError("model-template: the checkpoint routes its experts by " + ", ".join(
                f"{k}={v}" for k, v in wanted.items()) + "; set the router plugin's options to it (it has " + ", ".join(
                f"{k}={found[k]}" for k in wrong) + ")")

    def _draft_moe(self):
        return self.draft_prefix is not None and f"{self.draft_prefix}layers.0.mlp.gate.weight" in self.ck

    def _draft_layer(self, p, encode):
        """The checkpoint's MTP layer: the residual's input from the next token's embedding and the target's feature
        (residual.py), one full-attention decoder layer of the model's own blocks in its own KV pool (layer id
        `layers`), the residual's own end."""
        layer = self.layers
        lp = p + "layers.0."
        spec = self._spec("full_attention")
        mixer = self._block(spec["block"])(self, layer, lp + spec.get("prefix", ""),
                                           {**spec, "window": 0, "pool": "mtp"})
        ffn = self._ffn(layer, lp, routed=self._draft_moe())
        return {"res": self.residual.draft(p, encode), "layer": Layer("full_attention", mixer, ffn, None)}

    def _stored_rows(self, name):
        """Whether `name` is a table of a GGUF block format the embeddings plugin decodes on lookup (embeddings:gguf):
        then the table is kept as the file stores its rows, never decoded whole."""
        return (getattr(self.ck.headers[name], "stored", False)
                and "embeddings:gguf" in self.parts.facts("embeddings"))

    def _embed_in_place(self, name):
        """Whether the token embeddings can be read where the checkpoint stores them (an embeddings plugin with
        embeddings:host reads rows from the checkpoint's file): BF16 rows, or a GGUF's stored rows the plugin decodes
        (embeddings:gguf), at a known place of a file."""
        info = self.ck.headers[name]
        rows = info.dtype == "BF16" and not getattr(info, "stored", False) or self._stored_rows(name)
        return (rows and hasattr(info, "data_begin") and len(info.shape) == 2
                and "embeddings:host" in self.parts.facts("embeddings"))

    def gguf_table(self, names, width, plan):
        """A table of GGUF stored rows (one block format: `names` its parts, their rows in order) through the
        embeddings slot, which decodes the rows a lookup reads (embeddings:gguf): on the device where the plan put
        `plan`, else read in place from the file (embeddings:host)."""
        torch, ck = self.torch, self.ck
        fmt = "gguf:" + ck.headers[names[0]].dtype
        if "embeddings:gguf" not in self.parts.facts("embeddings"):
            raise ValueError(f"model-template: {names[0]} is stored {fmt[5:]} (a GGUF block format), and the "
                             "embeddings plugin does not provide embeddings:gguf (lookup-table-rows decodes them)")
        if self.placed.get(plan, "device") == "device":
            rows = torch.cat([torch.from_numpy(ck.rows(n)) for n in names]).to(self.device)
            return self.embeddings.store(rows, format=fmt, dtype=torch.bfloat16)
        if "embeddings:host" not in self.parts.facts("embeddings"):
            raise ValueError(f"model-template: {names[0]} lies off the device, and the embeddings plugin does not "
                             "provide embeddings:host (lookup-table-rows reads a table where the checkpoint stores it)")
        files = {"paths": [str(ck.headers[n].shard) for n in names],
                 "offsets": [int(ck.headers[n].data_begin) for n in names], "rows": [int(ck.shape(n)[0]) for n in names]}
        return self.embeddings.open(files, width, torch.bfloat16, self.device, format=fmt)

    def _embedding(self, name):
        """The token embeddings through the embeddings slot: on the device where the plan put them, else read in
        place from the checkpoint's file (embeddings:host), as a lookup table the plan put off the device is. A GGUF's
        quantized embeddings stay its stored rows, decoded a lookup (embeddings:gguf)."""
        torch, ck = self.torch, self.ck
        if self._stored_rows(name):
            return self.gguf_table([name], int(ck.shape(name)[1]), self.embed_plan)
        if self.placed.get(self.embed_plan, "device") == "device":
            rows = ck.get(name, self.device).to(torch.bfloat16)
            found = self.embeddings.store(rows)
            del rows
            return found
        info = ck.headers[name]
        source = {"paths": [str(info.shard)], "offsets": [int(info.data_begin)], "rows": [int(info.shape[0])]}
        return self.embeddings.open(source, int(info.shape[1]), torch.bfloat16, self.device)

    def _draft_bytes(self, tied, embed_name):
        """The bytes one draft step reads: the draft layer's tensors (its routed experts at the share a token routes
        to) and the head it predicts through."""
        moe = self.config.moe
        share = (moe.experts_per_token / moe.experts) if moe else 1.0
        experts = self.draft_prefix + "layers.0.mlp.experts"
        total = sum(self.ck.nbytes(n) * (share if n.startswith(experts) else 1.0)
                    for n in self.ck.names() if n.startswith(self.draft_prefix))
        head = self.ck.nbytes("lm_head.weight" if not tied else embed_name)
        if self.options["draft_head"] == "nvfp4":
            head //= 4
        elif self.options["draft_head"] == "fp8":
            head //= 2
        return int(total + head)

    def _head_slices(self, count, encode, base):
        """The head as `count` row slices re-encoded to `encode` (draft-only matrices)."""
        fmt, tensors = self.matrices.stored(base)
        rows = tensors["weight"].shape[0]
        step = -(-rows // count)
        step = -(-step // 128) * 128
        found = []
        for first in range(0, rows, step):
            part = {k: (v[first:first + step] if getattr(v, "dim", lambda: 0)() and v.shape[0] == rows else v)
                    for k, v in tensors.items()}
            found.append(self.proj.load(part, fmt, role="draft_head", encode=encode))
            self.torch.cuda.empty_cache()
        del tensors
        return found

    # -- passes ----------------------------------------------------------------------------------------------------------
    def _passes(self, options):
        """(whether prompt spans run layer by layer, why): the option `passes`, auto where the experts plugin says a
        prompt chunk brings its layers' experts to the device (stages_prompts())."""
        asked = options["passes"]
        if asked not in ("auto", "on", "off"):
            raise ValueError(f"model-template: passes is auto, on or off, not {asked!r}")
        if asked == "off":
            return False, "off (the option passes)"
        if asked == "on":
            return True, "on (the option passes)"
        stages = getattr(self.experts, "stages_prompts", None)
        if callable(stages) and stages():
            return True, ("the experts plugin brings a prompt chunk's experts to the device: a span's chunks run "
                          "layer by layer, so each layer's cross the link once a span")
        return False, "nothing brings weights to the device for a prompt chunk"

    def _pass_counts(self, first, limit):
        """The chunks a pass would hold from position `first`: the spans up to `limit` or the pass's tokens, each a
        prompt row to every pool (more tokens than a verify window, which the state pools commit as decode rows);
        [] where no pass pays: a single chunk whose routes are fewer than a layer's experts runs as any chunk does."""
        if not self.passes:
            return []
        at, counts = first, []
        while at < limit:
            rows = min(self.spans.rows(at), limit - at)
            if rows <= self.width + 1 or (counts and at + rows - first > self.pass_tokens):
                break
            counts.append(rows)
            at += rows
        moe = self.config.moe if self.routed else None
        staged = moe is not None and counts and counts[0] * moe.experts_per_token >= moe.experts
        return counts if len(counts) > 1 or staged or (counts and self.options["passes"] == "on") else []

    def _passing(self, row):
        """Whether the row's prompt work is a pass's: one under way, or one that starts at its frontier here and
        runs to the place the prompt cache marks next, or the prompt's end."""
        s = row.session
        if row.prefill_complete or not self.passes:
            return False
        if s.passing is None:
            counts = self._pass_counts(s.done, self._stop(s, len(s.prompt)))
            if not counts:
                return False
            if s.resume is not None:   # as a prompt chunk after a resumed prefix: the MTP pool's row there is owed
                if self.pools.get("mtp") is not None and s.mtp_done < s.done:
                    self._mtp_prompt([(s, s.mtp_done, 1, 0)], s.resume[None])
                    s.mtp_done = s.done
                s.resume = None
            s.passing = Pass(self, s.handle, s.prompt, s.done, counts)
            self.passed["open"] += 1
        return True

    def _pass_step(self, row, deadline):
        """A step of the row's pass: chunks through layers until `deadline`. Until the pass ends the step consumes
        nothing and reports the span (Core's prefill passes); the step that ends it consumes the span, and where
        the span ends the prompt, emits its first token."""
        s, made = row.session, row.session.passing
        mtp = self.pools.get("mtp")

        def finish(chunk, h, feature):
            if mtp is not None:   # the draft layer's keys of the chunk's positions, from the target's feature there
                count = min(chunk.first + chunk.count, len(s.prompt) - 1) - chunk.first
                if count > 0:
                    self._mtp_prompt([(s, chunk.first, count, 0)], feature)
            if chunk.index + 1 == len(made.chunks):
                made.ended = (h[-1:].clone(), feature[-1].clone())
        reused, s.reused = s.reused, 0
        s.produced, s.graphed, s.settled = [], False, False
        if not made.done:
            made.run(deadline, finish)
        # Core counts a pass's tokens past a step's grant only once a step reported the pass.
        if not made.done or (made.tokens > row.prefill_tokens and not made.reported):
            made.reported = True
            s.phase, s.rows = PASS, 0
            return StepResult(reused, [], False, reused, pass_tokens=made.tokens)
        s.passing = None
        s.phase, s.rows, s.settled = PREFILL, made.tokens, True
        head, feature = made.ended
        if self.cache is not None and mtp is not None:
            s.tail = feature
        self._passed(made)
        ends = s.done + made.tokens == len(s.prompt)
        logprobs = None
        if ends:
            (token, logprobs), = self._emit([(s, row.allowed)], self._logits_rows(head))
            s.produced = [token]
            if self.draft_head is not None:
                s.last_hidden = feature.clone()   # the seed's feature: the prompt's last row
            release_cached_under(self.free_below, self.device)   # a prompt ended: its temporaries leave room
        return StepResult(made.tokens + reused, s.produced, ends, reused, 0, logprobs)

    def _passed(self, made, ended=True, stepped=True):
        """A pass ended, or was dropped: the telemetry's counts. stepped: it ran in Core's steps (a score's pass
        runs in one call), so its longest step and its longest chunk through a layer are steps' times."""
        counted = self.passed
        counted["open"] -= 1
        counted["passes" if ended else "dropped"] += 1
        if stepped and made.step_seconds > counted.get("slowest_step_seconds", 0.0):
            counted["slowest_step_seconds"] = round(made.step_seconds, 4)
            counted["slowest_unit"] = list(made.slowest)   # seconds, layer, chunk
        if ended:
            counted["tokens"] += made.tokens
            counted["steps"] += made.steps
            counted["chunks"] += len(made.chunks)

    # -- spans -----------------------------------------------------------------------------------------------------------
    def _spans(self, options):
        seconds = float(options["span_seconds"])
        if seconds <= 0 or not self.device.type == "cuda" or self.chunk <= QUANTUM:   # no shorter span to choose
            return Spans(self.chunk)
        from crit_plugins import decisions
        torch = self.torch
        key = {"device": torch.cuda.get_device_name(self.device), "layers": self.layers, "hidden": self.hidden,
               "kinds": sorted(set(self.kinds)), "chunk": self.chunk, "context": self.context,
               "vocabulary": self.vocabulary, "kv": self.kv_dtype,
               "plugins": {slot: self.parts.plugin(slot) for slot in self.parts.slots()}}
        costs = decisions.settle("model-template", "spans", key=key, measure=self._measure_spans,
                                 decide=lambda found: fit({(r, p): s for r, p, s in found["timings"]},
                                                          {r: b for r, b in found["peaks"]}))
        room = None
        if not self.offload:   # expert-offload makes room for a prompt step itself (its step frames)
            room = max(0, free_device_bytes() - (int(options["headroom_mib"]) << 20))
        deepest = max(0, (self.kv_tokens - self.width - 1 - self.chunk) // QUANTUM * QUANTUM)
        return Spans(self.chunk, costs, seconds, room, deepest, units=len(self.decoder) if self.passes else 1)

    def _measure_spans(self):
        """Three prompt steps on a sequence of its own: a short and a full chunk at position 0, a full chunk at the
        deepest position served (its earlier KV positions written with zeros); each run twice, the second timed and its
        peak measured above what the device held before it."""
        import time
        torch = self.torch
        memory = getattr(self.parts.context, "memory", None) if hasattr(self.parts, "context") else None
        if memory is None:
            from crit_plugins.memory import of
            memory = of(self.device)
        short = max(QUANTUM, (self.chunk // 4) // QUANTUM * QUANTUM)
        deep = max(0, (self.kv_tokens - self.width - 1 - self.chunk) // QUANTUM * QUANTUM)
        g = torch.Generator().manual_seed(0)
        timings, peaks = [], {}
        for number, (rows, position) in enumerate([(short, 0), (self.chunk, 0), (self.chunk, deep)]):
            sequence = ("model-template calibration", number)
            self._open(sequence)
            try:
                for key in self.kv_keys:
                    pool = self.pools[key]
                    # a sliding pool holds a ring: only its window's positions, a span at a time
                    window = key[1] if isinstance(key, tuple) else 0
                    step = self.chunk if window else FILL
                    for first in range(max(0, position - window - self.chunk) if window else 0, position, step):
                        count = min(step, position - first)
                        batch = StepBatch.build([(sequence, first, count)], self.device)
                        zeros = torch.zeros(count, pool.kv_heads, pool.head_dim, dtype=torch.bfloat16,
                                            device=self.device)
                        values = torch.zeros(count, pool.kv_heads, pool.value_dim, dtype=torch.bfloat16,
                                             device=self.device) if pool.value_dim else None
                        slots = pool.reserve(batch)
                        for layer in pool.layers:
                            pool.write(layer, slots, zeros, values)
                        pool.keep(sequence, first + count)
                ids = torch.randint(3, self.vocabulary, (rows,), generator=g).to(self.device)
                batch = StepBatch.build([(sequence, position, rows)], self.device)
                for run in range(2):
                    torch.cuda.synchronize(self.device)
                    before = torch.cuda.memory_reserved(self.device)
                    memory.mark()
                    started = time.perf_counter()
                    h = self._hidden(ids, batch, windows=self._windows([([], ids.tolist())]))
                    self._logits_rows(h[-1:])
                    if self.offload:
                        self.experts.end_step(False)
                    torch.cuda.synchronize(self.device)
                    seconds = time.perf_counter() - started
                    peak = max(0, (memory.peak() or before) - before)
                    del h
                timings.append([rows, position, seconds])
                peaks[rows] = max(peaks.get(rows, 0), peak)
            finally:
                self._close(sequence)
        release_cached_under(self.free_below, self.device)
        return {"timings": timings, "peaks": [[r, b] for r, b in sorted(peaks.items())]}

    # -- Core's settings -------------------------------------------------------------------------------------------------
    def info(self):
        found = {"implementation": f"model-template {self.layout.get('name', '')} decoder (crit device plugins)",
                 "vocabulary": self.vocabulary, "context": self.context, "prefill_chunk": self.chunk,
                 "prefill_work_quantum": self.chunk, "max_concurrency": self.concurrency,
                 "max_pending": self.pending_max, "max_batch_tokens": self.chunk + self.concurrency * (self.width + 1),
                 "max_row_tokens": self.width + 1, "decode_token_work": self.width + 1, "stop_tokens": self.stops,
                 "sampling": dict.fromkeys(CORE_SAMPLING, True), "token_logprobs": True, "max_top_logprobs": TOP,
                 "scoring": True, "query_scoring": True, "score_paths": ["prefill"], "replay": True,
                 "speculative": self.mtp, "plain": self.mtp,
                 **({"prompt_cache": True} if self.cache is not None else {})}
        if self.rounds is not None and self.rounds.counter_names:   # what the round's parts count (an expert cache)
            found["round_counters"] = list(self.rounds.counter_names)
        if self.passes:   # Core's prefill passes: a pass runs a chunk through a layer a step at least
            found.update(prefill_pass_tokens=self.pass_tokens,
                         prefill_pass_steps=self.layers * (self.pass_tokens // QUANTUM + 2))
        if self.mtp:
            found["draft_bytes_per_step"] = self.draft_bytes
            found["drafter_kind"] = "checkpoint MTP layer"
            if self.rounds is not None:
                found.update(self.rounds.width_facts())
        return found

    def facts(self):
        return {"layout": self.layout.get("name"), "kv_pages": self.kv_dtype, "kv_pool_tokens": self.kv_tokens,
                "state_sequences": self.sequences if self.states is not None else 0, "spans": self.spans.facts(),
                "passes": {"on": self.passes, "reason": self.passes_reason,
                           **({"tokens": self.pass_tokens, "state_bytes_per_token": self.residual.carry(2)}
                              if self.passes else {})},
                "mtp": {"width": self.width, "kv": self.options["mtp_kv"], "encode": self.options["mtp_encode"],
                        "draft_head": self.options["draft_head"],
                        **({"experts": self.draft_experts} if getattr(self, "draft_experts", None) else {})}
                if self.mtp else None,
                "rounds": None if self.rounds is None else {"rows": self.rounds.rows, "problems": self.rounds.problems},
                "prompt_graphs": ({"buckets": sorted(self.prompts.graphs), "problems": self.prompts.problems}
                                  if self.prompts is not None else
                                  {"buckets": [], "problems": [self.prompts_refused]} if self.prompts_refused else None),
                "block_selection": next((("sparse-indexer" if layer.mixer.kernel is not None else "torch")
                                         for layer in self.decoder if getattr(layer.mixer, "sparse", False)), None)}

    def telemetry(self):
        found = self.rounds.telemetry() if self.rounds is not None else {}
        if self.passes:
            found = {**found, "prompt_passes": dict(self.passed)}
        if self.prompts is not None:
            found = {**found, "prompt_graphs": self.prompts.telemetry()}
        if self.cache is not None:
            found = {**found, **self.cache.telemetry()}
        return found

    def native_rounds(self, library):
        if self.rounds is None:
            return None
        return self.rounds.native_rounds(library, [self.width] * self.rounds.rows, mode="sampled")

    # -- the arithmetic --------------------------------------------------------------------------------------------------
    def _norm(self, x, weight, residual=None):
        return self.norms.rms_norm(x, weight, self.eps, offset=self.norm_offset, residual=residual)

    def _reserve(self, batch, pools):
        return {key: self.pools[key].reserve(batch) for key in pools}

    def _windows(self, rows):
        """Each token's window of ids ([tokens][window_tokens] int64, its own last, -1 before its sequence) for rows
        of (the sequence's ids before the row, the row's ids), in batch order; None where no block reads windows."""
        if not self.window_tokens:
            return None
        k, found = self.window_tokens - 1, []
        for before, inputs in rows:
            joined = [-1] * (k - min(k, len(before))) + list(before[len(before) - min(k, len(before)):]) + list(inputs)
            found += [joined[i:i + k + 1] for i in range(len(inputs))]
        return self.torch.tensor(found, dtype=self.torch.int64, device=self.device).view(len(found), k + 1)

    def _hidden(self, ids, batch, feature=False, windows=None):
        """The hidden rows of the batch's tokens the head reads; with feature, also what a draft layer reads of them
        (the residual's: the same rows, or its streams). windows: _windows of the batch's rows (a round's come from
        its token ring)."""
        x = self.embedded(ids)       # the embedding rows the residual starts from
        if self.window_tokens and windows is None:
            if not batch.round or self.ring is None:
                raise RuntimeError("model-template: a step of a model whose blocks read token windows gives them")
            windows = self.ring.windows(ids.view(batch.rows, -1), self.window_tokens).reshape(-1, self.window_tokens)
        ctx = Context(self, batch, self.rows.of(batch),
                      self._reserve(batch, self.kv_keys), ids=ids, windows=windows)
        h, found = self.residual.forward(self.decoder, x, ctx)
        return (h, found) if feature else h

    def embedded(self, ids):
        """The embedding rows of ids, scaled where the family scales them (the layout's `embed_scale`: sqrt_hidden,
        the square root of the hidden width at the rows' precision)."""
        x = self.embed.lookup(ids)
        if self.embed_scale is None:
            return x
        return (x.float() * self.embed_scale).to(x.dtype)

    def _prompt_forward(self, batch):
        """A captured prompt chunk's forward (crit_plugins.prompt_graphs): the final-normed hidden rows of the bucket,
        the same prompt arithmetic as the eager path."""
        return self._hidden(batch.tokens_t, batch)

    def _mtp_input(self, tokens, feature, ctx):
        return self.residual.draft_input(self.mtp_layer["res"], self.embed.lookup(tokens), feature, ctx)

    def _mtp(self, tokens, feature, batch):
        """The MTP layer over a round batch (catch-up columns or a chain step) from the target's feature rows (or its
        own, a chain's): (out, feature), the rows the draft head reads and the next step's feature."""
        b = self.mtp_layer
        ctx = Context(self, batch, self.rows.rows(tokens.shape[0]), self._reserve(batch, self.draft_keys))
        x = self._mtp_input(tokens, feature, ctx)
        return self.residual.draft_forward(b["res"], b["layer"], x, ctx)

    def _mtp_prompt(self, rows, hidden):
        """The MTP pool's keys and values of prompt positions (`rows`: (session, first position, count, first row of
        `hidden`, the target's feature rows)), each from the token after it and the target's feature there: the
        residual's input, the mixer's input rows and its k/v projections (no attention: a one-layer drafter's KV reads
        nothing else)."""
        torch = self.torch
        packed, tokens, index = [], [], []
        for s, first, count, at in rows:
            packed.append((s.handle, first, count))
            tokens += s.prompt[first + 1:first + 1 + count]
            index += range(at, at + count)
        batch = StepBatch.build(packed, self.device)
        ctx = Context(self, batch, [(batch.starts[r], batch.starts[r + 1], True) for r in range(batch.rows)],
                      self._reserve(batch, self.draft_keys))
        x = self._mtp_input(torch.tensor(tokens, dtype=torch.int64, device=self.device),
                            hidden[torch.tensor(index, dtype=torch.int64, device=self.device)], ctx)
        b = self.mtp_layer
        b["layer"].mixer.write(self.residual.draft_keys(b["res"], x, ctx), ctx)

    def _logits_rows(self, h, draft=False):
        """FP32 logits of rows h, each multiplied as a decode row is; a draft head of row slices gives the slices'
        columns side by side."""
        groups = self.rows.rows(h.shape[0])
        matrix = self.draft_matrix if draft else self.head
        if isinstance(matrix, list):
            logits = self.torch.cat([self.rows.matmul(h, m, groups) for m in matrix], dim=-1).float()
        else:
            logits = self.rows.matmul(h, matrix, groups).float()
        if self.softcap:   # the family's cap on its logits: cap * tanh(logits / cap)
            logits = self.torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def _score_bytes(self, rows, top_k=0):
        """The device bytes a score's slice of `rows` rows holds at once: FP32 logits (twice with a cap on them) and
        their log-softmax, with top_k the whole sorted row beside its int64 order."""
        per = 4 * (2 + (1 if self.softcap else 0)) + (12 if top_k else 0)
        return int(rows) * self.vocabulary * per

    def _need(self, nbytes):
        """`nbytes` the step is about to take beyond what its experts plugin measured steps of its size to take:
        the device's ledger has its givers (an elastic expert cache) make room for them first, so the device's free
        memory stays at what they keep (crit_plugins.memory: need). Nothing on the CPU, or inside a capture."""
        if self.device.type != "cuda" or self.torch.cuda.is_current_stream_capturing():
            return
        from crit_plugins.memory import of
        of(self.device).need(nbytes)

    def _logits_at(self, h, logit_rows):
        torch = self.torch
        if not logit_rows:   # a step of prompt chunks none of which ends its prompt
            return torch.zeros(0, self.vocabulary, dtype=torch.float32, device=self.device)
        rows = torch.tensor(logit_rows, dtype=torch.int64, device=self.device)
        return self._logits_rows(h[rows])

    def verify(self, view, batch):
        """The round's target pass (crit_plugins.rounds): logits of every column of the RoundBatch, [rows][width + 1]
        [vocabulary]; the recurrence and the KV follow the round slots their pools bind; the draft head keeps every
        column's hidden state for its next catch-up."""
        if self.draft_head is None:
            h = self._hidden(batch.tokens_t, batch)
        else:
            h, feature = self._hidden(batch.tokens_t, batch, feature=True)
            self.draft_head.keep_features(view, feature.view(view.rows, view.width + 1, -1))
        return self._logits_rows(h).view(view.rows, view.width + 1, -1)

    # -- Core's step model -----------------------------------------------------------------------------------------------
    def admission(self, request):
        prompt, output = len(request.prompt), int(request.output_tokens)
        found = self.reservations.admission(prompt, output)
        if found != READY:
            return found
        return READY if self._state() else DEFERRED

    def _state(self):
        """Whether the state pool has a free sequence, for a request or a score: the prompt cache's marks leave for
        one, the kept sequences' first and then the open ones'."""
        if self.states is None:
            return True
        while not self.states.free and self.cache is not None and (self.cache.evict() or self.cache.unmark()):
            pass
        return bool(self.states.free)   # every state pool holds the same sequences

    def _open(self, sequence):
        for pool in self.pools.values():
            pool.open(sequence)

    def _close(self, sequence):
        for pool in self.pools.values():
            pool.close(sequence)

    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        prompt, output = len(request.prompt), int(request.output_tokens)
        if self.reservations.admission(prompt, output) == READY:
            session.pages = self.reservations.reserve(prompt, output)
        if self.cache is None:
            self._open(session.handle)
        else:
            self._state()
            session.done = session.reused = self.cache.open(session.handle, session.prompt, request.prompt_cache)
            session.resume = self.cache.resumed(session.handle)
            session.mtp_done = max(session.done - 1, 0)
            self._trim()
        return session

    def _trim(self, tokens=0):
        """Kept sequences hold only pages no open request has reserved, less those a score of `tokens` takes: the
        least recently used leave first."""
        pool = self.pool
        while self.cache.kept and (sum(-(-len(entry.tokens) // pool.page) for entry in self.cache.kept.values())
                                   > pool.pages - self.reservations.reserved - -(-tokens // pool.page)):
            self.cache.evict()

    def _stop(self, s, stop=None):
        """Where a prompt chunk of session s ends: its span (or `stop`), or before it a place the prompt cache marks."""
        stop = self.spans.stop(s.done) if stop is None else stop
        if self.cache is not None:
            marked = self.cache.stop(s.handle)
            if marked is not None and marked > s.done:
                stop = min(stop, marked)
        return stop

    def step(self, rows):
        started = time.perf_counter()
        results = [None] * len(rows)
        bound = [i for i, row in enumerate(rows) if row.session.slot is not None]
        if bound:
            scheduled = {rows[i].session.slot for i in bound}
            held = [slot for slot in range(len(self.slots)) if slot not in scheduled]
            width = self.rounds.width(len(self.slots))
            self.rounds.launch(len(self.slots), width, mode="sampled", hold=held)
            found = self.rounds.read()
            for i in bound:
                s = rows[i].session
                row = found[s.slot]
                s.phase, s.produced, s.rows = "round", row["tokens"], 0
                logprobs = None
                if s.request.logprobs is not None:
                    asked = int(s.request.logprobs)
                    logprobs = [(value, top[:asked]) for value, top in zip(row["logprobs"], row["top"])]
                results[i] = StepResult(0, s.produced, True, logprobs=logprobs, drafted=width)
        rest = [i for i, row in enumerate(rows) if row.session.slot is None]
        passing = [i for i in rest if self._passing(rows[i])]
        rest = [i for i in rest if i not in passing]
        if rest:
            for i, result in zip(rest, self._plain([rows[i] for i in rest])):
                results[i] = result
        for i in passing:   # after the step's other rows: a pass takes what is left of the step's time
            results[i] = self._pass_step(rows[i], started + self.pass_seconds)
        return results

    def _plain(self, rows):
        torch = self.torch
        layout = StepLayout(step_item(row, len(row.session.prompt), row.session.done,
                                      stop=None if row.session.done >= len(row.session.prompt) else self._stop(row.session))
                            for row in rows)
        packed, ids, logit_rows, placed, prompts, graphed, tails, history = [], [], [], [], [], [], [], []
        mtp = self.pools.get("mtp")
        marking = self.cache is not None and mtp is not None
        for placement in layout.placements:
            s = rows[placement.item].session
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[s.done:s.done + placement.rows]
            s.phase, s.rows, s.produced, s.graphed, s.settled = placement.phase, len(inputs), [], False, False
            if placement.phase == PREFILL and s.resume is not None:
                if mtp is not None and s.mtp_done < s.done:
                    self._mtp_prompt([(s, s.mtp_done, 1, 0)], s.resume[None])
                    s.mtp_done = s.done
                s.resume = None
            if placement.phase == PREFILL and self.prompts is not None and self.prompts.graphs:
                found = self.prompts.run(s.handle, s.done, inputs)
                if found is not None:
                    hidden = found[:len(inputs)].clone()   # the graph's output, overwritten by its next run
                    s.graphed = True
                    if marking:
                        s.tail = hidden[-1].clone()
                    graphed.append((placement, s, hidden, inputs))
                    placed.append((placement, s, bool(placement.logits)))
                    continue
            if placement.phase == PREFILL:
                count = min(s.done + len(inputs), len(s.prompt) - 1) - s.done
                if count > 0:
                    prompts.append((s, s.done, count, len(ids)))
            packed.append((s.handle, s.done, len(inputs)))
            if self.window_tokens:   # the row's ids and the sequence's before it (its prompt, then its outputs)
                before = s.prompt[:s.done] if s.done <= len(s.prompt) else s.prompt + s.outputs[:s.done - len(s.prompt)]
                history.append((before[-self.window_tokens:], inputs))
            ids += inputs
            if marking and placement.phase == PREFILL:
                tails.append((s, len(ids) - 1))
            if placement.logits:
                logit_rows.append(len(ids) - 1)
            placed.append((placement, s, bool(placement.logits)))
        eager_logits, hidden = None, None   # hidden: the eager rows' draft features (the head reads `head_rows`)
        if packed:
            batch = StepBatch.build(packed, self.device)
            head_rows, hidden = self._hidden(torch.tensor(ids, dtype=torch.int64, device=self.device), batch,
                                             feature=True, windows=self._windows(history))
            eager_logits = self._logits_at(head_rows, logit_rows)
            del head_rows
            if mtp is not None and prompts:
                self._mtp_prompt(prompts, hidden)
            for s, row in tails:
                s.tail = hidden[row].clone()
        last_hidden, rows_logits, e = {}, [], 0
        graphs = {id(s): (h, inputs) for _, s, h, inputs in graphed}
        for placement, s, has in placed:
            if not has:
                continue
            if id(s) in graphs:
                h, inputs = graphs[id(s)]
                rows_logits.append(self._logits_rows(h[-1:]))
                last_hidden[id(s)] = h[-1]
            else:
                rows_logits.append(eager_logits[e:e + 1])
                last_hidden[id(s)] = hidden[logit_rows[e]]
                e += 1
        for placement, s, h, inputs in graphed:
            if mtp is not None:
                count = min(s.done + len(inputs), len(s.prompt) - 1) - s.done
                if count > 0:
                    self._mtp_prompt([(s, s.done, count, 0)], h)
        logits = torch.cat(rows_logits) if rows_logits else None
        emitted = iter(self._emit([(s, rows[placement.item].allowed) for placement, s, has in placed if has], logits))
        if any(placement.phase == PREFILL and has for placement, _, has in placed):
            release_cached_under(self.free_below, self.device)   # a prompt ended: its temporaries leave room
        results = [None] * len(rows)
        for placement, s, has in placed:
            logprobs = None
            if has:
                token, logprobs = next(emitted)
                s.produced = [token]
                if placement.phase == PREFILL and self.draft_head is not None:
                    s.last_hidden = last_hidden[id(s)].clone()   # the seed's feature: the prompt's last row
            reused, s.reused = (s.reused, 0) if placement.phase == PREFILL else (0, s.reused)
            results[placement.item] = StepResult(placement.rows + reused if placement.phase == PREFILL else 0,
                                                 s.produced, has, reused, placement.rows, logprobs)
        return results

    def _emit(self, sampled, logits):
        """The token each of `sampled` (a session and the tokens its constraint allows, None: any) emits from its row
        of `logits`, with its log-probability rows where its request asks for them: [(token, logprobs)]."""
        torch = self.torch
        for s, allowed in sampled:
            if allowed is not None:
                s.constrained = True
        params = [dict({k: s.request.sampling[k] for k in CORE_SAMPLING if k in s.request.sampling},
                       position=len(s.outputs), **({} if allowed is None else {"allowed": allowed}))
                  for s, allowed in sampled]
        tokens = self.sampler.sample(logits, params).tolist() if sampled else []
        found = []
        for j, (s, _) in enumerate(sampled):
            token, logprobs = s.request.replay[len(s.outputs)] if s.request.replay else tokens[j], None
            if s.request.logprobs is not None:
                row = torch.log_softmax(logits[j], -1)
                ranked, order = row.sort(descending=True, stable=True)
                k = min(int(s.request.logprobs), TOP)
                logprobs = [(float(row[token]), list(zip(order[:k].tolist(), ranked[:k].tolist())))]
            found.append((token, logprobs))
        return found

    def _roundable(self, s):
        """Whether a request whose prompt is in decodes in round slots from here: rounds on and able to run, a free
        slot, no long replay, no constrained decoding, its stops within a slot's, its log-probabilities within the
        driver's alternatives."""
        r, rounds = s.request, self.rounds
        return (rounds is not None and rounds.native is not None and not rounds.problems
                and len(self.slots) < rounds.rows and (not r.replay or len(r.replay) <= CONTINUATION)
                and not s.constrained and (not r.plain or float(r.sampling.get("temperature", 1.0)) == 0.0)
                and len(r.stop_tokens) <= STOPS and (r.logprobs is None or int(r.logprobs) <= rounds.top)
                and (self.draft_head is None or s.last_hidden is not None))

    def commit(self, decisions):
        rounds = self.rounds
        native = rounds.native if rounds is not None else None
        mtp = self.pools.get("mtp")
        trims = {}
        for d in decisions:
            s = d.session
            if s.slot is not None:
                if not (d.terminal or d.cancelled):
                    trims[s.slot] = d.accepted
                continue
            if s.phase is None:
                continue
            if s.phase == PASS:   # a step inside a pass: its pools follow the pass, and nothing counts before its end
                s.phase = None
                continue
            completed = s.phase == PREFILL and s.done + s.rows == len(s.prompt)
            kept = not d.cancelled
            if kept:
                s.done += s.rows
                if d.accepted:
                    s.pending = s.produced[d.accepted - 1]
                    s.outputs += s.produced[:d.accepted]
            if s.graphed:
                if not kept:
                    raise RuntimeError("model-template: a captured prompt chunk settled its state, and Core discarded it")
            elif s.settled:   # a pass: its state pools accepted each chunk as its layers ran
                s.broken = s.broken or not kept
            else:
                for states in self.state_pools:
                    states.accept(s.handle, s.rows if kept else 0)   # the state and the convolution window
            for pool in self.kv_pools:
                pool.keep(s.handle, s.done)
            if mtp is not None:
                s.mtp_done = min(s.done, len(s.prompt) - 1) if kept and s.phase == PREFILL else s.mtp_done
                for pool in self.draft_pools:
                    pool.keep(s.handle, s.mtp_done)
            if self.cache is not None and kept and s.phase == PREFILL:
                self.cache.mark(s.handle, s.done, payload=s.tail)
            s.phase, s.produced = None, []
            if completed and kept and d.accepted and not d.terminal and self._roundable(s):
                self._bind(s)
        if trims:
            ran_natively = native is not None and native.last is not None and native.last[2]
            if ran_natively:
                rounds.adopt(*native.last[:2])
                emitted = rounds.n_emitted[:len(self.slots)].tolist()
            else:
                emitted = [len(s.produced) for s in self.slots]
            keep = [min(count, trims.get(slot, count)) for slot, count in enumerate(emitted)]
            if keep != emitted:
                rounds.truncate(keep)

    def _bind(self, s):
        s.slot = len(self.slots)
        self.slots.append(s)
        r = s.request
        params = {k: r.sampling[k] for k in CORE_SAMPLING if k in r.sampling}
        params.setdefault("temperature", 0.0)
        if self.draft_head is not None:
            self.draft_head.seed_feature(s.slot, s.last_hidden)
            s.last_hidden = None
        self.rounds.bind(s.slot, s.handle, len(s.prompt), s.pending, r.output_tokens - len(s.outputs),
                         len(s.prompt) + r.output_tokens + self.width + 1, stops=r.stop_tokens, params=params,
                         session=s, tokens=s.prompt + [s.pending], outputs=len(s.outputs),
                         logprobs=r.logprobs is not None, width=0 if r.plain else None,
                         continuation=list(r.replay or ()), follow="replay" if r.replay else None)

    def retire(self, session, finish):
        if session.slot is not None:
            slot, last = session.slot, len(self.slots) - 1
            self.rounds.unbind(slot)
            if slot != last:   # the decoding rows stay packed at the front
                moved = self.slots[last]
                self.rounds.move(last, slot)
                if self.draft_head is not None:
                    self.draft_head.move(last, slot)
                moved.slot = slot
                self.slots[slot] = moved
            self.slots.pop()
            session.slot = None
        made, session.passing = session.passing, None
        if made is not None:   # a pass that did not end: its pools stand past the rows the session committed
            made.close()
            self._passed(made, ended=False)
            session.broken = True
            if self.offload:   # the layer the experts plugin held staged for it goes back to its cache
                self.experts.end_step(False)
        if self.cache is None:
            self._close(session.handle)
        elif finish == "failed" or session.broken:
            self.cache.retire(session.handle, None)
        else:
            # Kept to its prompt: rounds publish the reply through Core, and the state stands past it.
            held = min(self.pool.length(session.handle), len(session.prompt))
            for pool in self.kv_pools:
                pool.keep(session.handle, held)
            for pool in self.draft_pools:
                pool.keep(session.handle, min(pool.length(session.handle), max(held - 1, 0)))
            self.cache.retire(session.handle, session.prompt[:held], stands=False)
        session.last_hidden = session.tail = session.resume = None
        if session.pages:
            self.reservations.release(session.pages)
            session.pages = 0
        if self.cache is not None:
            self._trim()

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        """A fixed text's scores through the prompt path, in the prompt's spans, each chunk's KV and recurrent state
        kept for the next; the logits of the scored rows SCORE_SLICE at a time."""
        torch = self.torch
        sequence = ("score", self.handles)
        wanted = range(first_target - 1, len(ids) - 1)
        target, argmax, top, query = [], [], [], [] if queries else None
        if self.cache is not None:   # the prompt cache makes room: the score's pages and its state
            self._trim(len(ids))
            self._state()
        for key, pool in self.pools.items():
            if key not in self.draft_keys:
                pool.open(sequence)

        def rows(start, end, h):
            """The scores of the scored positions among [start, end), from their head rows h."""
            for lo in range(max(start, wanted.start), end, SCORE_SLICE):
                hi = min(lo + SCORE_SLICE, end)
                self._need(self._score_bytes(hi - lo, top_k))
                logp = torch.log_softmax(self._logits_rows(h[lo - start:hi - start]), -1)
                nxt = torch.tensor(ids[lo + 1:hi + 1], dtype=torch.int64, device=self.device)
                target.extend(logp.gather(1, nxt[:, None])[:, 0].tolist())
                argmax.extend(logp.argmax(-1).tolist())
                if top_k:
                    ranked, order = logp.sort(dim=-1, descending=True, stable=True)
                    top.extend(list(zip(o, v)) for o, v in zip(order[:, :top_k].tolist(), ranked[:, :top_k].tolist()))
                if queries:
                    for r in range(lo, hi):
                        row = queries[r - wanted.start]
                        query.append(logp[r - lo, torch.tensor(row, dtype=torch.int64, device=self.device)]
                                     .tolist() if row else [])
                del logp

        try:
            start = 0
            while start < wanted.stop:   # the last token predicts nothing scored
                counts = self._pass_counts(start, wanted.stop)
                if counts:   # the spans from here layer by layer (passes.py), each chunk scored as its layers end
                    made = Pass(self, sequence, ids, start, counts)
                    self.passed["open"] += 1
                    try:
                        made.run(None, lambda chunk, h, feature: rows(chunk.first, chunk.first + chunk.count, h))
                    finally:
                        made.close()
                        self._passed(made, ended=made.done, stepped=False)
                    start += made.tokens
                    for pool in self.kv_pools:
                        pool.keep(sequence, start)
                    continue
                end = min(self.spans.stop(start), wanted.stop)
                batch = StepBatch.build([(sequence, start, end - start)], self.device)
                before = ids[max(0, start - self.window_tokens):start]
                h = self._hidden(torch.tensor(ids[start:end], dtype=torch.int64, device=self.device), batch,
                                 windows=self._windows([(before, ids[start:end])]))
                for states in self.state_pools:
                    states.accept(sequence, end - start)
                for pool in self.kv_pools:
                    pool.keep(sequence, end)
                rows(start, end, h)
                start = end
        finally:
            for key, pool in self.pools.items():
                if key not in self.draft_keys:
                    pool.close(sequence)
            release_cached_under(self.free_below, self.device)
        return Scores(target, query, argmax, top or None)

    def close(self):
        self.decoder.clear()
