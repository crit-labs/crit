"""Gemma 4's companion MTP drafter (a `gemma4_assistant` checkpoint, published beside the target in a folder of its
own) as the model's device draft head: a few decoder layers that read the target's own keys and values, fed the
target's final-normed row.

Its arithmetic, as transformers' Gemma4AssistantForCausalLM and its MTP candidate generator run it:

    seed      the pending token t (position P, not yet in the target's KV) and h, the target's final-normed row of
              the column that predicted t (a round's last emitted token's column; after a prompt, its last row)
    step      x = pre_projection([embed(t) * sqrt(hidden), h])                  (2 x target width -> drafter width)
              per layer (a decoder layer between two norms, its output scaled by layer_scalar), attention query-only:
              q = rope(q_norm(q_proj(input_layernorm(x)))) at position P, over the target's keys and values of its
              last layer of the same type (sliding or full) at the positions before P (a sliding layer's last window)
              out = norm(x); logits = embed_tokens(out) (its head tied to its embedding, no cap);
              h' = post_projection(out)
    chain     each next draft repeats the step at the same position P with its draft token and h'

step() is that arithmetic over the operations it is handed (Ops): the device head's (the projections slot's held
matrices, the norms slot's rotation, the attention slot over the target's pools) and the host reference's in the
package's tests (plain FP32 products over the official implementation's keys and values), so both run the same
code. The drafter only proposes: the target verifies every draft, so its precision (its matrices through the
projections slot's draft encoding, mtp_encode) changes which drafts are kept, never what a request emits.

A model technique of the gemma4-model package (models/cache/README.md, a family's own pieces): its manifest mounts
this code as its source `assistant_drafter`, the draft block gemma4-assistant, which the layout's draft names with
the companion's model_type (`"draft": {"block": "gemma4-assistant", "companion": "gemma4_assistant"}`); the models
core finds the folder (code/companion.py: option draft_folder, else the one beside the checkpoint that fits) and
hands it over as model.draft_companion."""
import weakref

# Each layer's tensors, under the companion's own names (a layout's draft `names` maps another spelling onto them).
LAYER = ("input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
         "post_feedforward_layernorm.weight", "layer_scalar", "self_attn.q_proj.weight", "self_attn.q_norm.weight",
         "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
MATRICES = ("self_attn.q_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight")
TOP = ("pre_projection.weight", "post_projection.weight", "model.norm.weight", "model.embed_tokens.weight")
# The bytes a held element takes, by the draft encoding (the formats' own sizes: NVFP4 an E2M1 code and an E4M3 scale
# a 16 of them, FP8 a byte, BF16 two).
ELEMENT = {"nvfp4": 0.5 + 1 / 16, "fp8": 1.0, "bf16": 2.0}


def text_config(companion):
    return companion.config.get("text_config") or companion.config


def sources(text, target_kinds):
    """{layer type: the target layer whose keys and values the drafter's layers of that type read}: the target's last
    layer of each type. KeyError naming a type the target has none of."""
    found = {}
    for kind in set(text["layer_types"]):
        layers = [i for i, k in enumerate(target_kinds) if k == kind]
        if not layers:
            raise KeyError(kind)
        found[kind] = layers[-1]
    return found


def weights(ck, layers, load):
    """(per layer {name: tensor}, {top name: tensor}): the drafter's tensors read from `ck` (its companion's
    checkpoint) through load(name, tensor) (the caller's holding: a matrix as the projections slot holds it, a
    vector in FP32)."""
    per = [{name: load(name, ck.get(f"model.layers.{i}.{name}", "cpu")) for name in LAYER} for i in range(layers)]
    top = {name: load(name, ck.get(name, "cpu")) for name in TOP}
    return per, top


class Ops:
    """The operations step() runs on: mm(x, held matrix) -> [n][out], rms(x, weight) FP32, rope(layer, q) the layer's
    query norm and rotation at the pending position, attend(layer, q) -> [n][heads][head_dim], gelu(x) (tanh)."""

    def __init__(self, mm, rms, rope, attend, gelu):
        self.mm, self.rms, self.rope, self.attend, self.gelu = mm, rms, rope, attend, gelu


def step(per, top, heads, embedded, h, ops):
    """One drafter pass of every row at its pending position: (logits FP32 [rows][vocabulary], next h). `per` and
    `top` as weights() gives them (a layer's `gate_up` the gate and up projections stacked, `head` the tied head),
    embedded the pending tokens' target embedding rows (scaled), h the rows the chain starts from."""
    import torch
    rows = embedded.shape[0]
    x = ops.mm(torch.cat([embedded.to(torch.bfloat16), h.to(torch.bfloat16)], dim=-1), top["pre"]).float()
    for i, w in enumerate(per):
        a = ops.rms(x, w["input_layernorm.weight"])
        q = ops.mm(a, w["q"]).view(rows, heads, -1)
        o = ops.attend(i, ops.rope(i, q))
        y = ops.mm(o.reshape(rows, -1).to(torch.bfloat16), w["o"]).float()
        x = x + ops.rms(y, w["post_attention_layernorm.weight"])
        a = ops.rms(x, w["pre_feedforward_layernorm.weight"])
        gate, up = ops.mm(a, w["gate_up"]).float().chunk(2, dim=-1)
        f = ops.mm((ops.gelu(gate) * up).to(torch.bfloat16), w["down"]).float()
        x = (x + ops.rms(f, w["post_feedforward_layernorm.weight"])) * w["scalar"]
        x = x.to(torch.bfloat16).float()   # the residual stream at the checkpoint's precision between layers
    out = ops.rms(x, top["model.norm.weight"]).to(torch.bfloat16)
    return ops.mm(out, top["head"]).float(), ops.mm(out, top["post"])


def arrange(per, top, stack):
    """The layers' matrices as step() reads them: q, o, gate_up (stack(gate, up)), down, scalar; and pre, post, head."""
    for w in per:
        w["q"], w["o"] = w.pop("self_attn.q_proj.weight"), w.pop("self_attn.o_proj.weight")
        w["gate_up"] = stack(w.pop("mlp.gate_proj.weight"), w.pop("mlp.up_proj.weight"))
        w["down"] = w.pop("mlp.down_proj.weight")
        w["scalar"] = float(w.pop("layer_scalar").reshape(-1)[0])
    top["pre"], top["post"] = top.pop("pre_projection.weight"), top.pop("post_projection.weight")
    top["head"] = top.pop("model.embed_tokens.weight")
    return per, top


class AssistantHead:
    """The drafter as the model's draft head over round slots 0 .. rows - 1 (draft:head-device)."""

    def __init__(self, model, companion, rows, widest):
        torch = self.torch = model.torch
        # The model and its layers weakly: the round driver and the drafter keep this head, and a strong reference back
        # would hold a closed model's device memory until a later collection.
        self.m, self.rows, self.widest, self.cols = weakref.proxy(model), rows, widest, widest + 1
        text = text_config(companion)
        self.hidden, self.heads = int(text["hidden_size"]), int(text["num_attention_heads"])
        self.eps = float(text.get("rms_norm_eps", 1e-6))
        self.types = list(text["layer_types"])
        self.source = sources(text, model.types)
        dev = model.device
        encode = None if model.options["mtp_encode"] == "bf16" else model.options["mtp_encode"]

        def load(name, tensor):
            if name in MATRICES or name in ("pre_projection.weight", "post_projection.weight"):
                return tensor.to(dev).to(torch.bfloat16)
            if name == "model.embed_tokens.weight":
                return tensor[:model.vocabulary].to(dev).to(torch.bfloat16)
            return tensor.to(dev).float()

        def held(t):
            return model.proj.load({"weight": t.contiguous()}, "bf16", role="draft", encode=encode)

        per, top = arrange(*weights(companion.ck, len(self.types), load), lambda g, u: torch.cat([g, u]))
        for i, w in enumerate(per):
            target = model.decoder[self.source[self.types[i]]].mixer
            head_dim = w["self_attn.q_norm.weight"].shape[0]
            if head_dim != target.head_dim:
                raise ValueError(f"model-template: drafter layer {i} has heads of {head_dim}, the target's layer "
                                 f"{self.source[self.types[i]]} of {target.head_dim}")
            w["target"], w["rope_table"] = weakref.proxy(target), model.rope_for(self.source[self.types[i]], head_dim)
            for name in ("q", "o", "gate_up", "down"):
                w[name] = held(w[name])
        for name in ("pre", "post", "head"):
            top[name] = held(top[name])
        self.per, self.top = per, top
        # Device state, allocated once (captured rounds read and write it in place).
        self.features = torch.zeros(rows, self.cols, model.hidden, dtype=torch.bfloat16, device=dev)
        self.state = torch.zeros(rows, model.hidden, dtype=torch.bfloat16, device=dev)   # the chain's h
        from crit_plugins.device import RoundBatch
        self.batch = RoundBatch(rows, 0, dev)

    def _ops(self, rows, frontier):
        torch, m = self.torch, self.m

        def mm(x, matrix):
            return m.rows.matmul(x.to(torch.bfloat16).contiguous(), matrix, m.rows.rows(x.shape[0]))

        def rms(x, weight):
            x = x.float()
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * weight

        def rope(i, q):
            w = self.per[i]
            table = w["rope_table"]
            return m.norms.rms_norm_rope(q.contiguous(), w["self_attn.q_norm.weight"], self.eps, frontier,
                                         table.freqs, offset=m.norm_offset, scale=table.scale)

        def attend(i, q):
            target = self.per[i]["target"]
            if callable(getattr(target, "read", None)):   # a target block that keeps its own rows (shared KV)
                return target.read(q, self.batch)
            extra = {"sliding": target.window} if target.window else {}
            return m.attention.forward(target.layer, q, m.pools[target.pool], self.batch, scale=1.0, **extra)

        def gelu(x):
            return torch.nn.functional.gelu(x, approximate="tanh")
        return Ops(mm, rms, rope, attend, gelu)

    def _step(self, tokens, h, view):
        rows = tokens.shape[0]
        frontier = view.frontier[:rows]
        # The attention reads the keys before the pending position: a query row at P - 1 reads positions up to it
        # (a sliding layer its last window); the query turns at P.
        self.batch.fill(rows, 0, (frontier - 1).clamp(min=0), tokens[:, None])
        logits, h = step(self.per, self.top, self.heads, self.m.embedded(tokens), h, self._ops(rows, frontier))
        return logits[..., :self.m.vocabulary], h

    # -- the draft head (draft:head-device) ------------------------------------------------------------------------------
    def seed_feature(self, slot, hidden):
        self.features[slot, 0].copy_(hidden)

    def seed_session(self, slot, session):
        """Nothing: the drafter keeps no cache of its own (it reads the target's)."""

    def prompt(self, rows, hidden):
        """Nothing: the drafter reads the target's keys and values, which the prompt's chunks wrote."""

    def move(self, source, target):
        self.features[target].copy_(self.features[source])
        self.state[target].copy_(self.state[source])

    def keep_features(self, view, hidden):
        rows, cols = view.rows, view.width + 1
        held = (view.hold[:rows] > 0)[:, None, None]
        self.features[:rows, :cols] = self.torch.where(held, self.features[:rows, :cols], hidden)

    def catch_up(self, view):
        """Nothing to write: the round's committed positions are in the target's pools already."""

    def logits(self, depth, tokens, view, parents=None):
        rows = view.rows
        if depth == 0:
            last = (view.catch_up_count[:rows] - 1).clamp(min=0)[:, None, None]
            h = self.features[:rows].gather(1, last.expand(rows, 1, self.m.hidden)).squeeze(1)
        else:
            h = self.state[:rows]
        logits, self.state[:rows] = self._step(tokens[:rows], h, view)
        return logits

    def commit_round(self, view):
        """Nothing: the next round seeds from the target's features."""


class AssistantDrafter:
    """The draft block gemma4-assistant as the models core asks a drafter technique (DRAFTERS)."""
    name = "companion MTP drafter (gemma4_assistant)"   # the facts' drafter_kind
    spare = 0   # it writes no cache positions past the pending token

    @staticmethod
    def stored(ck, prefix):
        """A companion drafter is stored in a folder of its own, never in the checkpoint."""
        return False

    @staticmethod
    def fits(config, shape):
        """None where the companion's config reads this model: as wide a backbone, no ordered embeddings, a
        vocabulary that holds the model's, the target layer types its layers read; else why not."""
        text = config.get("text_config") or config
        if int(config.get("backbone_hidden_size") or 0) != shape.hidden:
            return f"its backbone is {config.get('backbone_hidden_size')} wide, the model {shape.hidden}"
        if config.get("use_ordered_embeddings"):
            return "its logits go through ordered embeddings (centroids), which this drafter does not read"
        if int(text.get("vocab_size") or 0) < shape.vocabulary:
            return f"its vocabulary ({text.get('vocab_size')}) is smaller than the model's ({shape.vocabulary})"
        try:
            sources(text, shape.types)
        except KeyError as kind:
            return f"its layers read the target's last {kind.args[0]} layer, the model has none"
        except (TypeError, ValueError):
            return "its config names no layer_types"
        return None

    def __init__(self, model, prefix, spec):
        self.m, self.spec = model, dict(spec)   # the model owns this block: no cycle outlives it
        self.companion = model.draft_companion

    def taps(self, applications):
        return ()   # the target's final-normed row alone

    def _elements(self):
        """(matrix elements, other elements) of the companion's tensors."""
        ck, matrices, other = self.companion.ck, 0, 0
        for name in ck.names():
            count = 1
            for size in ck.shape(name):
                count *= size
            held = name.endswith(MATRICES) or name in ("pre_projection.weight", "post_projection.weight",
                                                       "model.embed_tokens.weight")
            matrices, other = (matrices + count, other) if held else (matrices, other + count)
        return matrices, other

    def weight_bytes(self, ck, vocabulary):
        """The bytes a draft step reads: its matrices as the draft encoding holds them, its norms in FP32."""
        matrices, other = self._elements()
        encode = self.m.options.get("mtp_encode")
        return int(matrices * ELEMENT.get(encode, ELEMENT["bf16"]) + other * 4)

    def device_bytes(self, ck, vocabulary, rows, tokens):
        """Its matrices as held and its round state (each slot's feature rows and chain row)."""
        return self.weight_bytes(ck, vocabulary) + rows * (self.m.width + 2) * self.m.hidden * 2

    def head(self, taps, rows, widest, tokens, head_matrix):
        return AssistantHead(self.m, self.companion, rows, widest)


DRAFTERS = {"gemma4-assistant": AssistantDrafter}
