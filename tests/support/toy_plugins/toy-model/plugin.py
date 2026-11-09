"""A toy model plugin on the CPU, over its slots' device forms in torch: a batched step model whose next token is a hash
of every token its row sees, so wrong bookkeeping changes it. Its history lives in the kv_cache slot's pool (each
token as its key), its logits rows go to the sampler slot, and with a drafter in the selection a decode row verifies
the drafter's drafts: it emits the drafter's tokens_per_round (2 by default) a round, each computed after the one
before, as an engine whose drafts are always accepted. A step takes step_ms, plus the attention plugin's
seconds_per_token for each token a row of it emits, if it names one. A row's allowed set (a constrained request's)
goes to the sampler with the row's params."""
import json
from pathlib import Path
import time

from crit_core.batch import DECODE, PREFILL, StepLayout, step_item
from crit_core.native import Scores, StepResult
from crit_plugins import StepBatch

CORE_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")


def successor(seen, vocabulary):
    h = 1469598103934665603
    for position, token in enumerate(seen):
        h = ((h ^ ((token + 1) * (position + 1))) * 1099511628211) % 2 ** 64
    return h % vocabulary


class Session:
    def __init__(self, handle, request):
        self.handle, self.request, self.prompt = handle, request, list(request.prompt)
        self.done, self.outputs = 0, []          # KV rows kept; published outputs
        self.pending, self.produced, self.phase, self.rows = None, [], None, 0


class ToyModel:
    def __init__(self, context, options, parts):
        import torch
        self.torch, self.options, self.parts = torch, options, parts
        self.sampler = parts["sampler"]
        # Every slot takes part in a step, as Core's loader checks: the toy's one-weight projection and attention over
        # the step's rows read nothing the toy's tokens depend on. `skip` leaves one slot out (a test).
        self.skip = options["skip"]
        self.proj, self.attention = parts["projections"], parts["attention"]
        device = getattr(context, "device", None)   # the loader's device (the CPU for a context without one)
        self.device = torch.device(device if isinstance(device, str) else "cpu")
        self.weight = self.proj.load({"weight": torch.ones(1, 1, device=self.device)}, "fp32")
        self.drafts = parts.get("drafter") is not None
        self.width = int(getattr(parts.get("drafter"), "tokens_per_round", 2)) if self.drafts else 1
        self.per_token = float(getattr(parts["attention"], "seconds_per_token", 0.0))
        config = getattr(context, "config", None) or {}
        text = config.get("text_config") or {}
        self.vocabulary = int(text.get("vocab_size") or config.get("vocab_size") or options["vocabulary"])
        self.context = int(getattr(context, "context_tokens", None) or options["context"])
        model = getattr(context, "model", None)
        generation = Path(model) / "generation_config.json" if isinstance(model, (str, Path)) else None
        eos = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id") if generation and generation.is_file() else []
        self.stops = sorted(set(eos if isinstance(eos, list) else [eos])) if eos is not None else []
        if self.skip != "placement":
            parts["placement"].plan({"toy.table": 4 * self.vocabulary}, {"device": 1 << 40, "host": 1 << 40})
        self.pool = parts["kv_cache"].pool([0], 1, 1, self.context, self.device, dtype="fp32")
        self.handles, self.steps, self.closed = 0, [], False
        # prompt_graphs: the step's projection work as a captured chunk of its rows, replayed (a test of Core's loader,
        # which counts a replay as the calls its capture made).
        self.prompts = None
        if options.get("prompt_graphs", "off") != "off" and torch.cuda.is_available():   # toy-vision has no such option
            from crit_plugins.prompt_graphs import PromptGraphs
            self.graph_weight = self.proj.load({"weight": torch.ones(1, 1, device="cuda")}, "fp32")
            self.prompts = PromptGraphs(self._chunk, device="cuda", buckets=(16,), parts=parts, uses=["projections"])
            self.prompts.prepare()

    def _chunk(self, batch):
        x = batch.tokens_t.float()[:, None]
        if self.options["prompt_graphs"] == "skip":
            return x @ self.graph_weight.T
        return self.proj.matmul(x, self.graph_weight)

    def info(self):
        return {"implementation": "toy step model", "vocabulary": self.vocabulary, "context": self.context,
                "prefill_chunk": self.options["prefill_chunk"], "max_concurrency": self.options["concurrency"],
                "max_pending": self.options["pending"], "max_batch_tokens": 64, "max_row_tokens": self.width,
                "stop_tokens": self.stops, "sampling": {key: True for key in CORE_SAMPLING},
                "token_logprobs": True, "max_top_logprobs": 20, "scoring": True, "query_scoring": True,
                "score_paths": ["prefill"], "replay": True, "speculative": self.drafts, "plain": self.drafts}

    def open(self, request):
        self.handles += 1
        session = Session(self.handles, request)
        self.pool.open(session.handle)
        return session

    # -- the pool holds each token as its key ----------------------------------------------------------------------------
    def _write(self, rows):
        """Rows of (session, first position, tokens) into the pool; returns the StepBatch they form."""
        batch = StepBatch.build([(s.handle, first, len(tokens)) for s, first, tokens in rows], self.device)
        ids = self.torch.tensor([t for _, _, tokens in rows for t in tokens], dtype=self.torch.float32,
                                device=self.device).view(-1, 1, 1)
        self.pool.write(0, self.pool.reserve(batch), ids, ids)
        return batch

    def _history(self, session, count):
        return [int(t) for t in self.pool.gather(0, session.handle, count)[0].view(-1).tolist()]

    def _token(self, session, count, allowed=None):
        """The token after the first `count` positions of the session's KV."""
        replay = session.request.replay
        if replay:
            return replay[len(session.outputs) + len(session.produced)]
        logits = self.torch.zeros(1, self.vocabulary, device=self.device)
        logits[0, successor(self._history(session, count), self.vocabulary)] = 10.0
        params = {key: session.request.sampling[key] for key in CORE_SAMPLING if key in session.request.sampling}
        if allowed is not None and self.skip != "masks":
            params["allowed"] = allowed
        return int(self.sampler.sample(logits, [params])[0])

    def step(self, rows):
        if self.options["step_ms"]:
            time.sleep(self.options["step_ms"] / 1000)
        if self.per_token:
            time.sleep(self.per_token * max(min(self.width, row.output_tokens or 1) for row in rows))
        layout = StepLayout(step_item(row, len(row.session.prompt), row.session.done) for row in rows)
        self.steps.append((len(rows), layout.decode_items))
        placed = []
        for placement in layout.placements:
            s = rows[placement.item].session
            s.phase, s.produced = placement.phase, []
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[s.done:s.done + placement.rows]
            s.rows = len(inputs)
            placed.append((s, s.done, inputs))
        batch = self._write(placed)
        if self.prompts is not None:
            for s, first, inputs in placed:
                if self.prompts.run(s.handle, first, inputs) is None:
                    self.proj.matmul(self.torch.ones(1, 1, device=self.device), self.weight)
        elif self.skip != "projections":
            self.proj.matmul(self.torch.ones(1, 1, device=self.device), self.weight)
        if self.skip != "attention":
            self.attention.forward(0, self.torch.ones(batch.tokens, 1, 1, device=self.device), self.pool, batch)
        results = [None] * len(rows)
        for placement, (s, first, inputs) in zip(layout.placements, placed):
            row = rows[placement.item]
            if placement.logits:
                s.produced.append(self._token(s, first + len(inputs), row.allowed))
                width = self.width if placement.phase == DECODE and not s.request.plain else 1
                while len(s.produced) < min(width, row.output_tokens):
                    # A draft, verified: its token goes in as the next row before the next token is computed.
                    at = first + len(inputs) + len(s.produced) - 1
                    self._write([(s, at, [s.produced[-1]])])
                    s.produced.append(self._token(s, at + 1, row.allowed))
            logprobs = None
            if s.request.logprobs is not None:
                width = min(int(s.request.logprobs), 20)
                logprobs = [(-0.5, [((t + a) % self.vocabulary, -0.5 - 0.25 * a) for a in range(width)])
                            for t in s.produced]
            results[placement.item] = StepResult(placement.rows if placement.phase == PREFILL else 0, s.produced,
                                                 placement.logits > 0, 0, placement.rows, logprobs)
        return results

    def commit(self, decisions):
        for d in decisions:
            s = d.session
            if s.phase is None:
                continue
            if d.cancelled or d.discard:
                self.pool.keep(s.handle, s.done)
            elif s.phase == PREFILL:
                # A prompt chunk keeps every row it wrote.
                self.pool.keep(s.handle, s.done + s.rows)
            else:
                # A decode round keeps the pending token and the drafts Core accepted after it.
                self.pool.keep(s.handle, s.done + d.accepted)
            s.done = self.pool.length(s.handle)
            if not (d.cancelled or d.discard) and d.accepted:
                s.pending = s.produced[d.accepted - 1]
                s.outputs += s.produced[:d.accepted]
            s.phase, s.produced = None, []

    def retire(self, session, finish):
        self.pool.close(session.handle)

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        targets = range(first_target, len(ids))
        v = self.vocabulary
        return Scores([-0.5] * len(targets), [[-1.0] * len(row) for row in queries] if queries else None,
                      [successor(ids[:t], v) for t in targets],
                      [[((ids[t] + a) % v, -0.5 - 0.25 * a) for a in range(top_k)] for t in targets] if top_k else None)

    def close(self):
        self.closed = True


def create(context, options, parts):
    return ToyModel(context, options, parts)
