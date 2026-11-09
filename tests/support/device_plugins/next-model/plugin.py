"""A hybrid mixture-of-experts decoder in Qwen3-Next's layout as a model plugin on a GPU, for the hybrid engine test:
gated-delta layers (a causal convolution over q, k and v, then the delta rule, both in the recurrence slot) beside gated
full-attention layers (partial rotary positions, KV in the kv_cache slot's pool, attention by the attention slot),
and a routed SwiGLU block in every layer (the router and experts slots, beside a gated shared expert). Every matrix
is in the projections slot, the norms (zero-centred weights, each residual add fused into the norm after it), the
query/key norms with their rotary positions, the gated norm, the attention output gate and the shared expert's SiLU
product are the norms slot's, and every token comes from the sampler; a step of many sequences' rows is one batch, its
matrices multiplied a prompt chunk alone and one-token rows DECODE_ROWS at a time, so a row's bits are those it has
alone. verify(view, batch) is its target pass over a crit_plugins.rounds RoundBatch (every column's logits), for the
round driver's tests of a hybrid model: the recurrence pool and the KV pool both follow the round slots."""
import json

from crit_core.batch import DECODE, PREFILL, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch
from crit_plugins.device import DECODE_ROWS

CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs, self.pending = 0, [], None    # rows committed; published outputs
        self.phase, self.produced, self.rows = None, [], 0


class NextModel:
    def __init__(self, context, options, parts):
        import torch
        from safetensors.torch import load_file
        self.torch, self.options = torch, options
        config = context.config.get("text_config") or context.config
        self.types = list(config["layer_types"])
        self.layers, self.hidden = len(self.types), config["hidden_size"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim = config.get("head_dim") or self.hidden // self.heads
        self.nk, self.nv = config["linear_num_key_heads"], config["linear_num_value_heads"]
        self.dk, self.dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
        self.kernel, self.top = config["linear_conv_kernel_dim"], config["num_experts_per_tok"]
        self.eps, self.vocabulary = config["rms_norm_eps"], config["vocab_size"]
        rope = config.get("rope_parameters") or {}
        self.rotary = int(self.head_dim * rope.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0)))
        theta = rope.get("rope_theta") or config.get("rope_theta", 10000.0)
        self.device = torch.device("cuda")
        self.linear = [i for i, t in enumerate(self.types) if t == "linear_attention"]
        self.full = [i for i, t in enumerate(self.types) if t == "full_attention"]
        self.context = int(context.context_tokens or options["context"])
        sequences = options["concurrency"] + 9   # every session Core holds at once, and a score's
        tensors = {}
        for path in sorted(context.model.glob("*.safetensors")):
            tensors.update(load_file(str(path), device="cpu"))
        kv_bytes = 2 * len(self.full) * self.context * sequences * self.kv_heads * self.head_dim * 2
        state_bytes = len(self.linear) * sequences * self.nv * self.dk * self.dv * 4
        free, _ = torch.cuda.mem_get_info()
        placed = parts["placement"].plan({**{n: t.numel() * t.element_size() for n, t in tensors.items()},
                                          "kv": kv_bytes, "states": state_bytes}, {"device": free, "host": 0})
        if set(placed.values()) != {"device"}:
            raise ValueError("this model plugin keeps every tensor on the device")
        self.proj, self.attention, self.sampler = parts["projections"], parts["attention"], parts["sampler"]
        self.recurrence, self.router, self.experts = parts["recurrence"], parts["router"], parts["experts"]
        self.norms = parts["norms"]
        self.fused = "norms:fused" in parts.facts("norms")      # a layer's q and k norms with rope in one call
        on = lambda name: tensors[name].to(self.device)  # noqa: E731
        matrix = lambda name: self.proj.load({"weight": on(name)}, "bf16")  # noqa: E731
        self.embed = on("model.embed_tokens.weight").to(torch.bfloat16)
        self.final_norm = on("model.norm.weight").float()
        self.head = matrix("lm_head.weight")
        self.blocks = []
        for i, kind in enumerate(self.types):
            p = f"model.layers.{i}."
            shared = p + "mlp.shared_expert."
            b = {"in": on(p + "input_layernorm.weight").float(), "post": on(p + "post_attention_layernorm.weight").float(),
                 "router": matrix(p + "mlp.gate.weight"), "shared_gate": matrix(p + "mlp.shared_expert_gate.weight"),
                 # the shared expert's gate and up as one matrix: one matmul, then silu_mul over its halves
                 "gate_up": self.proj.load({"weight": torch.cat([on(shared + "gate_proj.weight"),
                                                                 on(shared + "up_proj.weight")])}, "bf16"),
                 "down": matrix(shared + "down_proj.weight")}
            if kind == "linear_attention":
                b.update(qkvz=matrix(p + "linear_attn.in_proj_qkvz.weight"), ba=matrix(p + "linear_attn.in_proj_ba.weight"),
                         out=matrix(p + "linear_attn.out_proj.weight"), conv=on(p + "linear_attn.conv1d.weight")[:, 0].float(),
                         a_log=on(p + "linear_attn.A_log").float(), dt_bias=on(p + "linear_attn.dt_bias").float(),
                         norm=on(p + "linear_attn.norm.weight").float())
            else:
                b.update({n: matrix(p + f"self_attn.{n}_proj.weight") for n in ("q", "k", "v", "o")},
                         q_norm=on(p + "self_attn.q_norm.weight").float(), k_norm=on(p + "self_attn.k_norm.weight").float())
            e = 0
            while f"{p}mlp.experts.{e}.gate_proj.weight" in tensors:
                q = f"{p}mlp.experts.{e}."
                self.experts.load(i, e, torch.cat([on(q + "gate_proj.weight"), on(q + "up_proj.weight")]).to(torch.bfloat16),
                                  on(q + "down_proj.weight").to(torch.bfloat16))
                e += 1
            self.blocks.append(b)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary, 2, device=self.device).float() / self.rotary))
        self.pool = parts["kv_cache"].pool(self.full, self.kv_heads, self.head_dim, self.context * sequences,
                                           self.device, "bf16")
        self.states = self.recurrence.pool(self.linear, self.nv, self.dk, self.dv, self.device, sequences,
                                           widest=options["widest"],
                                           conv=(2 * self.nk * self.dk + self.nv * self.dv, self.kernel), key_heads=self.nk)
        self.geometry = {"keys": self.dk, "values": self.dv, "epsilon": 1e-6, "query_scale": self.dk ** -0.5,
                         "normalize_keys": True, "normalize_queries": True}
        generation = context.model / "generation_config.json"
        eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id") if generation.is_file() else None
        self.stops = sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []
        self.handles = 0

    def info(self):
        return {"implementation": "qwen3-next-layout hybrid MoE decoder (device plugins)", "vocabulary": self.vocabulary,
                "context": self.context, "prefill_chunk": self.options["prefill_chunk"],
                "max_concurrency": self.options["concurrency"], "max_pending": 8, "max_batch_tokens": 512,
                "stop_tokens": self.stops, "sampling": dict.fromkeys(CORE_SAMPLING, True), "token_logprobs": True,
                "max_top_logprobs": 20, "scoring": True, "query_scoring": True, "score_paths": ["prefill"],
                "replay": True, "speculative": False}

    # -- the arithmetic --------------------------------------------------------------------------------------------------
    def _norm(self, x, weight, residual=None):
        """Qwen3-Next's RMSNorm: (1 + weight) over the normalized row (residual + x first, in place, when given)."""
        return self.norms.rms_norm(x, weight, self.eps, offset=1.0, residual=residual)

    def _head_norm_rope(self, x, weight):
        """A query's or key's per-head RMSNorm (1 + weight), then rotary positions on its first `rotary` values."""
        return self.norms.rms_norm_rope(x, weight, self.eps, self.positions, self.inv_freq, offset=1.0)

    @staticmethod
    def _groups(batch):
        """The token ranges the batch's matrices multiply apart: each prompt chunk alone, one-token rows (decode rows,
        a round's columns) DECODE_ROWS at a time, so every row's bits are those it has alone whatever shares the step."""
        if batch.round:
            return [(first, min(first + DECODE_ROWS, batch.tokens)) for first in range(0, batch.tokens, DECODE_ROWS)]
        groups, run = [], None
        for r in range(batch.rows):
            first, end = batch.starts[r], batch.starts[r + 1]
            if end - first > 1:
                groups.append((first, end))
                run = None
            elif run is not None and run[1] == first and run[1] - run[0] < DECODE_ROWS:
                run = groups[-1] = (run[0], end)
            else:
                run = (first, end)
                groups.append(run)
        return groups

    def _mm(self, x, matrix):
        groups = self.groups
        if len(groups) == 1:
            return self.proj.matmul(x, matrix)
        return self.torch.cat([self.proj.matmul(x[first:end], matrix) for first, end in groups])

    def _linear(self, i, h, batch):
        torch, t, b = self.torch, h.shape[0], self.blocks[i]
        r = self.nv // self.nk
        qkvz = self._mm(h, b["qkvz"]).view(t, self.nk, 2 * self.dk + 2 * self.dv * r)
        q, k, v, z = torch.split(qkvz, [self.dk, self.dk, r * self.dv, r * self.dv], dim=-1)
        beta, a = torch.split(self._mm(h, b["ba"]).view(t, self.nk, 2 * r), [r, r], dim=-1)
        mixed = self.recurrence.convolve(i, torch.cat([q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)], -1),
                                         b["conv"], None, self.states, batch)
        q, k, v = torch.split(mixed, [self.nk * self.dk, self.nk * self.dk, self.nv * self.dv], dim=-1)
        q, k = q.reshape(t, self.nk, self.dk), k.reshape(t, self.nk, self.dk)   # value head h reads key head h // r
        v = v.reshape(t, self.nv, self.dv)
        decay, beta = self.norms.softplus_decay(a.reshape(t, self.nv), b["dt_bias"], b["a_log"].exp(),
                                                beta.reshape(t, self.nv))
        out = self.recurrence.forward(i, q, k, v, decay, beta, self.states, batch, self.geometry)
        # The gated RMSNorm: normalize, then weight, then SiLU of z, each value head of each key-head group (z is a
        # view of the projection's output, read in place).
        gated = self.norms.gated_rms_norm(out.reshape(t, self.nk, r, self.dv), z.view(t, self.nk, r, self.dv),
                                          b["norm"], self.eps, out_dtype=h.dtype)
        return self._mm(gated.reshape(t, -1), b["out"])

    def _full(self, i, h, batch, slots):
        torch, t, b = self.torch, h.shape[0], self.blocks[i]
        q, gate = torch.chunk(self._mm(h, b["q"]).view(t, self.heads, 2 * self.head_dim), 2, dim=-1)
        k = self._mm(h, b["k"]).view(t, self.kv_heads, self.head_dim)
        if self.fused:
            q, k = self.norms.rms_norm_rope_pair(q, k, b["q_norm"], b["k_norm"], self.eps, self.positions,
                                                 self.inv_freq, offset=1.0)
        else:
            q, k = self._head_norm_rope(q, b["q_norm"]), self._head_norm_rope(k, b["k_norm"])
        v = self._mm(h, b["v"]).view(t, self.kv_heads, self.head_dim)
        self.pool.write(i, slots, k, v)
        a = self.attention.forward(i, q, self.pool, batch, scale=self.head_dim ** -0.5)
        a = self.norms.sigmoid_mul(a, gate, out_dtype=h.dtype)       # each head's output times sigmoid(its gate)
        return self._mm(a.reshape(t, -1), b["o"])

    def _moe(self, i, h):
        torch, b = self.torch, self.blocks[i]
        shared = self._mm(self.norms.silu_mul(self._mm(h, b["gate_up"])), b["down"])
        ids, weights = self.router.select(self._mm(h, b["router"]).float(), self.top)
        routed = self.experts.forward(i, h, ids, weights.to(h.dtype).float())
        gate = torch.sigmoid(self._mm(h, b["shared_gate"]).float())
        return (routed + gate * shared.float()).to(h.dtype)

    def _forward(self, ids, batch, logit_rows):
        """Logits [len(logit_rows)][vocabulary] (FP32) of the batch's tokens at logit_rows (None: every token)."""
        torch = self.torch
        x = self.embed[ids]          # the residual stream, which each norm after a block's output advances in place
        self.positions, self.groups = batch.positions_t, self._groups(batch)
        slots = self.pool.reserve(batch)
        h = self._norm(x, self.blocks[0]["in"])
        for i, kind in enumerate(self.types):
            mixed = self._linear(i, h, batch) if kind == "linear_attention" else self._full(i, h, batch, slots)
            h, x = self._norm(mixed, self.blocks[i]["post"], residual=x)
            following = self.blocks[i + 1]["in"] if i + 1 < self.layers else self.final_norm
            h, x = self._norm(self._moe(i, h), following, residual=x)
        if logit_rows is None:
            return self._mm(h, self.head).float()
        if not logit_rows:   # a step of prompt chunks none of which ends its prompt
            return torch.zeros(0, self.vocabulary, dtype=torch.float32, device=self.device)
        rows = torch.tensor(logit_rows, dtype=torch.int64, device=self.device)
        # Each logits row alone: a prompt's last row, a decode row and a round's column multiply the head the same way.
        return torch.cat([self.proj.matmul(h[rows[j:j + 1]], self.head) for j in range(len(logit_rows))]).float()

    def verify(self, view, batch):
        """The round's target pass (crit_plugins.rounds): logits of every column of the RoundBatch, [rows][width + 1]
        [vocabulary]; the recurrence and the KV follow the round slots their pools bind."""
        return self._forward(batch.tokens_t, batch, None).view(view.rows, view.width + 1, -1)

    # -- Core's step model -----------------------------------------------------------------------------------------------
    def _open(self, sequence):
        self.pool.open(sequence)
        self.states.open(sequence)

    def _close(self, sequence):
        self.pool.close(sequence)
        self.states.close(sequence)

    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        self._open(session.handle)
        return session

    def step(self, rows):
        torch = self.torch
        layout = StepLayout(step_item(row, len(row.session.prompt), row.session.done) for row in rows)
        packed, ids, logit_rows, placed = [], [], [], []
        for placement in layout.placements:
            s = rows[placement.item].session
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[s.done:s.done + placement.rows]
            s.phase, s.rows, s.produced = placement.phase, len(inputs), []
            packed.append((s.handle, s.done, len(inputs)))
            ids += inputs
            if placement.logits:
                logit_rows.append(len(ids) - 1)
            placed.append((placement, s, bool(placement.logits)))
        batch = StepBatch.build(packed, self.device)
        logits = self._forward(torch.tensor(ids, dtype=torch.int64, device=self.device), batch, logit_rows)
        sampled = [(s, rows[placement.item].allowed) for placement, s, has in placed if has]
        params = [dict({k: s.request.sampling[k] for k in CORE_SAMPLING if k in s.request.sampling},
                       position=len(s.outputs), **({} if allowed is None else {"allowed": allowed}))
                  for s, allowed in sampled]
        tokens = self.sampler.sample(logits, params).tolist() if sampled else []
        results, j = [None] * len(rows), 0
        for placement, s, has in placed:
            logprobs = None
            if has:
                token = s.request.replay[len(s.outputs)] if s.request.replay else tokens[j]
                s.produced = [token]
                if s.request.logprobs is not None:
                    row = torch.log_softmax(logits[j], -1)
                    top = torch.topk(row, min(int(s.request.logprobs), 20))
                    logprobs = [(float(row[token]), list(zip(top.indices.tolist(), top.values.tolist())))]
                j += 1
            results[placement.item] = StepResult(placement.rows if placement.phase == PREFILL else 0, s.produced, has,
                                                 0, placement.rows, logprobs)
        return results

    def commit(self, decisions):
        for d in decisions:
            s = d.session
            if s.phase is None:
                continue
            kept = not (d.cancelled or d.discard)
            if kept:
                s.done += s.rows
                if d.accepted:
                    s.pending = s.produced[d.accepted - 1]
                    s.outputs += s.produced[:d.accepted]
            self.states.accept(s.handle, s.rows if kept else 0)   # the state and the convolution window
            self.pool.keep(s.handle, s.done)
            s.phase, s.produced = None, []

    def retire(self, session, finish):
        self._close(session.handle)

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        torch = self.torch
        sequence = ("score", self.handles)
        self._open(sequence)
        try:
            batch = StepBatch.build([(sequence, 0, len(ids))], self.device)
            rows = list(range(first_target - 1, len(ids) - 1))
            logp = torch.log_softmax(self._forward(torch.tensor(ids, device=self.device), batch, rows), -1)
        finally:
            self._close(sequence)
        targets = torch.tensor(ids[first_target:], device=self.device)
        target = logp.gather(1, targets[:, None])[:, 0].tolist()
        query = ([logp[t, torch.tensor(row, device=self.device)].tolist() for t, row in enumerate(queries)]
                 if queries else None)
        top = None
        if top_k:
            found = torch.topk(logp, top_k)
            top = [list(zip(i, v)) for i, v in zip(found.indices.tolist(), found.values.tolist())]
        return Scores(target, query, logp.argmax(-1).tolist(), top)

    def close(self):
        self.blocks.clear()


def create(context, options, parts):
    return NextModel(context, options, parts)
