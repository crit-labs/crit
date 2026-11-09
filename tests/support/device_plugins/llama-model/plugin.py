"""A Llama-layout decoder as a model plugin on a GPU, for the device engine tests: the tensors of a LlamaForCausalLM
checkpoint (RMSNorm, rotary positions, grouped-query attention, SwiGLU), every matrix in the projections slot, every
token's keys and values in the kv_cache slot's pool, attention by the attention slot, the norms, rotary positions and
SiLU product by the norms slot (each residual add fused into the norm after it) and tokens by the sampler, a step of
many sequences' rows computed as one batch.

With the option `rounds` its greedy requests decode through Core's round driver (crit_plugins.rounds.Rounds) once
their prompt is in: each decode step of the rows in round slots is one round (verify over a RoundBatch, the sampler's
device predictions, the drafter's drafts from the model's draft head when the selection drafts), captured per row count
and launched by Core natively. The draft head is a stub (draft:head-device): the target itself one token a row per
depth, drafting its second choice where a position p has p % draft_miss == 2, so rounds accept some drafts and reject
others."""
import json
import math

from crit_core.batch import DECODE, PREFILL, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch
from crit_plugins.device import DECODE_ROWS, RoundBatch

CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
STOPS = 8   # stop ids a row in a round slot holds


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs, self.pending = 0, [], None    # KV rows committed; published outputs
        self.phase, self.produced, self.rows = None, [], 0
        self.slot = None                                         # its round slot once it decodes in rounds


class StubHead:
    """The model's draft head (draft:head-device) for the round tests: the target itself, one token a row at each
    depth through the round slots' KV (the positions the verify then writes again with the same tokens), its second
    choice where the position p has p % miss == 2."""

    def __init__(self, model, rows, miss):
        torch = model.torch
        self.model, self.miss, self.torch = model, miss, torch
        self.batch = RoundBatch(rows, 0, model.device)
        self.positions = torch.zeros(rows, dtype=torch.int64, device=model.device)
        self.tokens = torch.zeros(rows, 1, dtype=torch.int64, device=model.device)

    def catch_up(self, view):
        """Nothing to ingest: the target's KV holds every emitted token."""

    def logits(self, depth, tokens, view):
        torch, rows = self.torch, view.rows
        self.positions[:rows] = view.frontier + depth
        self.tokens[:rows, 0] = tokens
        batch = self.batch.fill(rows, 0, self.positions, self.tokens)
        logits = self.model._forward(batch.tokens_t, batch, None)
        if not self.miss:
            return logits
        wrong = (self.positions[:rows] % self.miss == 2)[:, None]
        best = logits.argmax(-1, keepdim=True)
        columns = torch.arange(logits.shape[1], device=logits.device)[None, :]
        return torch.where(wrong & (columns == best), torch.full_like(logits, -math.inf), logits)

    def commit_round(self, view):
        """Nothing kept apart from the target's own KV."""


class LlamaModel:
    def __init__(self, context, options, parts):
        import torch
        from safetensors.torch import load_file
        self.torch, self.options = torch, options
        config = context.config.get("text_config") or context.config
        self.layers, self.hidden = config["num_hidden_layers"], config["hidden_size"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.head_dim = config.get("head_dim") or self.hidden // self.heads
        self.eps, self.vocabulary = config["rms_norm_eps"], config["vocab_size"]
        self.device = torch.device("cuda")
        tensors = {}
        for path in sorted(context.model.glob("*.safetensors")):
            tensors.update(load_file(str(path), device="cpu"))
        free, _ = torch.cuda.mem_get_info()
        kv_tokens = int(context.context_tokens or options["context"]) * options["concurrency"]
        kv_bytes = 2 * self.layers * kv_tokens * self.kv_heads * self.head_dim * 2   # BF16 keys and values
        placed = parts["placement"].plan({**{n: t.numel() * t.element_size() for n, t in tensors.items()},
                                          "kv": kv_bytes}, {"device": free, "host": 0})
        if set(placed.values()) != {"device"}:
            raise ValueError("this model plugin keeps every tensor on the device")
        tensors = {n: t.to(self.device) for n, t in tensors.items()}
        self.proj, self.attention, self.sampler = parts["projections"], parts["attention"], parts["sampler"]
        self.norms = parts["norms"]
        self.fused = "norms:fused" in parts.facts("norms")      # a layer's q and k rope in one call
        matrix = lambda name: self.proj.load({"weight": tensors[name]}, "bf16")  # noqa: E731
        self.embed = tensors["model.embed_tokens.weight"].to(torch.bfloat16)
        self.final_norm = tensors["model.norm.weight"].float()
        self.head = matrix("lm_head.weight" if "lm_head.weight" in tensors else "model.embed_tokens.weight")
        self.blocks = []
        for i in range(self.layers):
            p = f"model.layers.{i}."
            self.blocks.append({
                "in": tensors[p + "input_layernorm.weight"].float(), "post": tensors[p + "post_attention_layernorm.weight"].float(),
                **{n: matrix(p + "self_attn." + n + "_proj.weight") for n in ("q", "k", "v", "o")},
                # gate and up as one matrix: one matmul, then the norms slot's silu_mul over its halves
                "gate_up": self.proj.load({"weight": torch.cat([tensors[p + "mlp.gate_proj.weight"],
                                                                tensors[p + "mlp.up_proj.weight"]])}, "bf16"),
                "down": matrix(p + "mlp.down_proj.weight")})
        theta = (config.get("rope_parameters") or {}).get("rope_theta") or config.get("rope_theta", 10000.0)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        self.context = int(context.context_tokens or options["context"])
        self.pool = parts["kv_cache"].pool(list(range(self.layers)), self.kv_heads, self.head_dim,
                                           self.context * options["concurrency"], self.device, "bf16")
        generation = context.model / "generation_config.json"
        eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id") if generation.is_file() else None
        self.stops = sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []
        self.handles = 0
        # Device memory the plan does not list (a test of Core's loader): Core refuses an engine holding it.
        self.unplanned = torch.empty(options["unplanned_mib"] << 20, dtype=torch.uint8, device=self.device)
        self.drafter, self.rounds, self.width, self.slots = parts.get("drafter"), None, 0, []
        if options["rounds"]:
            from crit_plugins.rounds import Rounds
            self.width = options["width"] if self.drafter is not None else 0
            rows = options["round_rows"] or options["concurrency"]
            self.draft_head = StubHead(self, rows, options["draft_miss"]) if self.drafter is not None else None
            # The slots its verify calls: the matrices, the KV pool, attention and the norms (the sampler's the
            # driver's own). The driver plans the memory its captures take through the placement plugin.
            self.rounds = Rounds(self, self.sampler, rows=rows, widest=self.width, device=self.device,
                                 drafter=self.drafter, head=self.draft_head, pools=[self.pool], parts=parts,
                                 uses=["projections", "attention", "kv_cache", "norms"], stops=STOPS,
                                 vocabulary=self.vocabulary)

    def info(self):
        drafts = self.drafter is not None
        return {"implementation": "llama-layout decoder (device plugins)", "vocabulary": self.vocabulary,
                "context": self.context, "prefill_chunk": self.options["prefill_chunk"],
                "max_concurrency": self.options["concurrency"], "max_pending": 8, "max_batch_tokens": 512,
                "max_row_tokens": self.width + 1, "decode_token_work": self.width + 1, "stop_tokens": self.stops,
                "sampling": dict.fromkeys(CORE_SAMPLING, True), "token_logprobs": True, "max_top_logprobs": 20,
                "scoring": True, "query_scoring": True, "score_paths": ["prefill"], "replay": True,
                "speculative": drafts, "plain": drafts}

    def telemetry(self):
        """The round driver's counters (rounds, drafts by depth) when the model decodes in rounds."""
        return self.rounds.telemetry() if self.rounds is not None else {}

    def native_rounds(self, library):
        """Core's native rounds over the round driver's graphs (one per row count, at the model's width), or None
        without rounds or when the selection cannot run inside one (the rows then decode through step())."""
        if self.rounds is None:
            return None
        return self.rounds.native_rounds(library, [self.width] * self.rounds.rows)

    # -- the arithmetic --------------------------------------------------------------------------------------------------
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

    def _mm(self, x, matrix, groups):
        if len(groups) == 1:
            return self.proj.matmul(x, matrix)
        return self.torch.cat([self.proj.matmul(x[first:end], matrix) for first, end in groups])

    def _forward(self, ids, batch, logit_rows):
        """Logits [len(logit_rows)][vocabulary] (FP32) of the batch's tokens at logit_rows (None: every token)."""
        torch, norms, t, eps = self.torch, self.norms, len(ids), self.eps
        groups = self._groups(batch)
        x = self.embed[ids]          # the residual stream, which each norm after a block's output advances in place
        slots = self.pool.reserve(batch)
        h = norms.rms_norm(x, self.blocks[0]["in"], eps)
        for i, b in enumerate(self.blocks):
            q = self._mm(h, b["q"], groups).view(t, self.heads, self.head_dim)
            k = self._mm(h, b["k"], groups).view(t, self.kv_heads, self.head_dim)
            if self.fused:
                q, k = norms.rope_pair(q, k, batch.positions_t, self.inv_freq)
            else:
                q, k = norms.rope(q, batch.positions_t, self.inv_freq), norms.rope(k, batch.positions_t, self.inv_freq)
            v = self._mm(h, b["v"], groups).view(t, self.kv_heads, self.head_dim)
            self.pool.write(i, slots, k, v)
            a = self.attention.forward(i, q, self.pool, batch, scale=self.head_dim ** -0.5)
            h, x = norms.rms_norm(self._mm(a.reshape(t, -1), b["o"], groups), b["post"], eps, residual=x)
            mlp = self._mm(norms.silu_mul(self._mm(h, b["gate_up"], groups)), b["down"], groups)
            following = self.blocks[i + 1]["in"] if i + 1 < len(self.blocks) else self.final_norm
            h, x = norms.rms_norm(mlp, following, eps, residual=x)
        if logit_rows is None:
            return self._mm(h, self.head, groups).float()
        if not logit_rows:   # a step of prompt chunks none of which ends its prompt
            return torch.zeros(0, self.vocabulary, dtype=torch.float32, device=self.device)
        rows = torch.tensor(logit_rows, dtype=torch.int64, device=self.device)
        # Each logits row alone: a prompt's last row, a decode row and a round's column multiply the head the same way.
        alone = [(i, i + 1) for i in range(len(logit_rows))]
        return self._mm(h[rows], self.head, alone).float()

    def verify(self, view, batch):
        """The round's target pass (crit_plugins.rounds): logits of every column of the RoundBatch, [rows][width + 1]
        [vocabulary]."""
        return self._forward(batch.tokens_t, batch, None).view(view.rows, view.width + 1, -1)

    # -- Core's step model -----------------------------------------------------------------------------------------------
    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        self.pool.open(session.handle)
        return session

    def step(self, rows):
        results = [None] * len(rows)
        bound = [i for i, row in enumerate(rows) if row.session.slot is not None]
        if bound:
            # The rows in round slots: one round of every slot, the slots this step does not schedule held.
            scheduled = {rows[i].session.slot for i in bound}
            held = [slot for slot in range(len(self.slots)) if slot not in scheduled]
            self.rounds.launch(len(self.slots), self.width, hold=held)
            found = self.rounds.read()
            for i in bound:
                s = rows[i].session
                s.phase, s.produced, s.rows = "round", found[s.slot]["tokens"], 0
                results[i] = StepResult(0, s.produced, True, drafted=self.width)
        rest = [i for i, row in enumerate(rows) if row.session.slot is None]
        if rest:
            for i, result in zip(rest, self._plain([rows[i] for i in rest])):
                results[i] = result
        return results

    def _plain(self, rows):
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

    def _roundable(self, s):
        """Whether a request whose prompt is in decodes in round slots from here: rounds on, the selection able to run
        inside one, a free slot, and a greedy request without log-probabilities, replay or plain decoding."""
        r, rounds = s.request, self.rounds
        return (rounds is not None and not rounds.problems and len(self.slots) < rounds.rows
                and not float(r.sampling.get("temperature", 0.0)) > 0 and r.logprobs is None and not r.replay
                and not r.plain and len(r.stop_tokens) <= STOPS)

    def commit(self, decisions):
        rounds = self.rounds
        native = rounds.native if rounds is not None else None
        trims = {}
        for d in decisions:
            s = d.session
            if s.slot is not None:
                # A row in a round slot keeps its state on the device; a trim Core made (not an end) goes back there.
                if not (d.terminal or d.cancelled or d.discard):
                    trims[s.slot] = d.accepted
                continue
            if s.phase is None:
                continue
            completed = s.phase == PREFILL and s.done + s.rows == len(s.prompt)
            if not (d.cancelled or d.discard):
                s.done += s.rows
                if d.accepted:
                    s.pending = s.produced[d.accepted - 1]
                    s.outputs += s.produced[:d.accepted]
            self.pool.keep(s.handle, s.done)
            s.phase, s.produced = None, []
            if completed and d.accepted and not (d.terminal or d.cancelled or d.discard) and self._roundable(s):
                self._bind(s)
        if trims:
            ran_natively = native is not None and native.last[2]
            if ran_natively:
                rounds.adopt(*native.last[:2])
                emitted = rounds.n_emitted[:len(self.slots)].tolist()
            else:
                emitted = [len(s.produced) for s in self.slots]
            keep = [min(count, trims.get(slot, count)) for slot, count in enumerate(emitted)]
            if keep != emitted:
                rounds.truncate(keep)

    def _bind(self, s):
        """The session decodes in the next free round slot: its pending token (the first output) at the prompt's
        end, the rest of its output limit, its stop tokens."""
        s.slot = len(self.slots)
        self.slots.append(s)
        r = s.request
        self.rounds.bind(s.slot, s.handle, len(s.prompt), s.pending, r.output_tokens - len(s.outputs),
                         len(s.prompt) + r.output_tokens + self.width + 1, stops=r.stop_tokens,
                         params={"temperature": 0.0}, session=s, tokens=s.prompt + [s.pending],
                         outputs=len(s.outputs))

    def retire(self, session, finish):
        if session.slot is not None:
            slot, last = session.slot, len(self.slots) - 1
            self.rounds.unbind(slot)
            if slot != last:   # the decoding rows stay packed at the front
                moved = self.slots[last]
                self.rounds.move(last, slot)
                moved.slot = slot
                self.slots[slot] = moved
            self.slots.pop()
            session.slot = None
        self.pool.close(session.handle)

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        torch = self.torch
        sequence = ("score", self.handles)
        self.pool.open(sequence)
        try:
            batch = StepBatch.build([(sequence, 0, len(ids))], self.device)
            rows = list(range(first_target - 1, len(ids) - 1))
            logp = torch.log_softmax(self._forward(torch.tensor(ids, device=self.device), batch, rows), -1)
        finally:
            self.pool.close(sequence)
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
    model = LlamaModel(context, options, parts)
    if options["temporary_mib"]:   # a load's temporaries (a test of Core's loader): freed, left in torch's cache
        import torch
        torch.empty(options["temporary_mib"] << 20, dtype=torch.uint8, device=model.device).fill_(1)
    return model
