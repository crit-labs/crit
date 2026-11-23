"""Block-sparse attention: an indexer scores the complete blocks of positions before a token and the token reads its best ones within a budget, and its own block.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `indexed_attention`
and composes the block indexed-attention a layout names (README.md)."""
from ..template.blocks import _tensor
from ..attention.attention import Attention


INDEXER = {"heads": "indexer_n_heads", "kv_heads": "indexer_kv_heads", "dim": "indexer_head_dim",
           "budget": "indexer_budget", "block": "indexer_compress_ratio"}


class _BlockPlan:
    """One step's block bookkeeping, shared by its indexed layers (they keep one pool, so one set of slots): the
    tokens that end a block (`at`: their indices, None for every token), the slots of each such block's earlier
    positions (`previous` [n][block - 1], oldest first), the position it starts at (`starts`), the slot its key is
    written at (`target`: the token's own; -1 where a round's token ends none), and what selection reads: per row of
    a step (first token, end, blocks complete after it, the slots of its blocks' keys), or for a round `keys` [rows]
    [blocks] and `ends` [tokens]."""
    __slots__ = ("at", "previous", "starts", "target", "rows", "keys", "ends")


class IndexedAttention(Attention):
    """Block-sparse softmax attention: an indexer picks the KV blocks each token reads.

    The sequence's positions lie in blocks of `block` positions from position 0. A token at position p reads the
    complete blocks an indexer picks for it, up to budget / block of them, and the block it lies in (its tail), so a
    token with no more complete blocks than that reads every position up to its own: plain causal attention below
    the budget. The indexer projects each token's row to index queries (several heads) and one raw index key; a
    complete block's key is the mean of its positions' raw keys, normed and rotated at the block's first position; a
    token scores a block by the sum over its index heads of relu(query . key) / sqrt(dim), queries normed and rotated
    at the token's position, and keeps its highest-scoring blocks (crit's canonical choice: ties to the lower block).

    Keys: the layers keep one pool of keys only, a row a position (`pools`): a position's raw key, and at a block's
    last position, once the block is complete, the block's key in its place (the raw key of a block's last position
    is read by nothing else). So the pool follows a sequence exactly as its KV does, through steps, rounds, forks and
    truncations, and a block's key is rewritten whenever its last position is computed again.

    Selection runs on crit's kernel sparse-indexer where it loads and takes the indexer's shape (the keys read in
    place through their slots), else in torch a token at a time, so a token's blocks never depend on the tokens
    beside it. A prompt step's tokens are scored in groups whose scores fit `score_bytes` (a token's scores are one
    FP32 a block, and the choice works on a wider copy of them: a 4,096-token step over 50,000 blocks would hold
    gigabytes at once), which changes no choice: each token's is its own. Where no sequence the engine serves can pass the budget (its KV pool holds no more blocks) the block is
    its base: no index keys, no block lists. Otherwise every token's attention reads a block list, below the budget
    too, so a row's arithmetic is one whatever shares its step. Config keys (the spec's `config` renames them):
    indexer_n_heads, indexer_kv_heads (1), indexer_head_dim, indexer_budget, indexer_compress_ratio; the spec's
    `indexer` is the indexer's tensor prefix under the attention's (indexer.)."""

    score_bytes = 64 << 20   # the most a group of a step's tokens holds in scores and their sort keys at once

    @classmethod
    def _config(cls, model, spec):
        names = {**INDEXER, **(spec.get("config") or {})}
        found = {key: int(model.setting(name) or 0) for key, name in names.items()}
        if min(found.values()) < 1 or found["kv_heads"] != 1 or found["budget"] % found["block"]:
            raise ValueError("model-template: an indexed attention takes the config's " + ", ".join(
                f"{name}={found[key]}" for key, name in names.items()) + ": positive, one index key head, the budget "
                "a multiple of the block")
        found["sparse"] = model.kv_tokens // found["block"] > found["budget"] // found["block"]
        return found

    @classmethod
    def pools(cls, model, spec, layers):
        found = cls._config(model, spec)
        if not found["sparse"]:
            return []
        key = "index" if not spec.get("pool") else f"{spec['pool']}-index"
        return [{"key": key, "kind": "keys", "layers": list(layers), "heads": 1, "dim": found["dim"]}]

    def __init__(self, model, layer, prefix, spec):
        super().__init__(model, layer, prefix, spec)
        m = self.m
        if self.window:
            raise ValueError("model-template: an indexed attention reads block lists, never a sliding window")
        found = self._config(m, spec)
        self.sparse, self.block, self.top = found["sparse"], found["block"], found["budget"] // found["block"]
        self.index_heads, self.index_dim = found["heads"], found["dim"]
        self.captures_prompts = not self.sparse   # a chunk's blocks are planned from its sequence's page table
        if not self.sparse:
            return
        for slot, fact in (("attention", "attention:blocks"), ("kv_cache", "kv:keys")):
            if fact not in m.parts.facts(slot):
                raise ValueError(f"model-template: layer {layer} attends the blocks an indexer picks, and the "
                                 f"{slot} plugin does not provide {fact}")
        ip = prefix + spec.get("indexer", "indexer.")
        if m.matrices.rows(ip + "index_qk_proj") != (self.index_heads + 1) * self.index_dim:
            raise ValueError(f"model-template: {ip}index_qk_proj has {m.matrices.rows(ip + 'index_qk_proj')} rows, "
                             f"{self.index_heads} index heads and a key of {self.index_dim} take "
                             f"{(self.index_heads + 1) * self.index_dim}")
        self.index_qk = m.matrices.load([ip + "index_qk_proj"], "index_qk")
        self.index_q_norm = _tensor(m, ip + "q_layernorm.weight").float()
        self.index_k_norm = _tensor(m, ip + "k_layernorm.weight").float()
        self.index = "index" if self.pool == "kv" else f"{self.pool}-index"
        self.index_scale = self.index_dim ** -0.5
        self.kernel = m.kernel("sparse-indexer") if self._kernel_shape() else None

    def _kernel_shape(self):
        """Whether crit's kernel sparse-indexer takes this indexer: a power of two of heads up to 128 of 64, 128 or
        256 values, on a CUDA device (its keys are the pool's own BF16 rows)."""
        heads = self.index_heads
        return (self.m.device.type == "cuda" and heads <= 128 and heads & (heads - 1) == 0
                and self.index_dim in (64, 128, 256))

    # -- the step's blocks ---------------------------------------------------------------------------------------------
    def _plan(self, ctx):
        key = ("blocks", self.index, self.block, self.top)
        plan = ctx.shared.get(key)
        if plan is None:
            plan = ctx.shared[key] = (self._round_plan if ctx.batch.round else self._step_plan)(ctx)
        return plan

    def _step_plan(self, ctx):
        m, torch, batch, size = self.m, self.m.torch, ctx.batch, self.block
        pool, plan = m.pools[self.index], _BlockPlan()
        at, previous, starts, plan.rows = [], [], [], []
        for r in range(batch.rows):
            sequence, first, count, position = batch.row(r)
            table, page = pool.table(sequence), pool.page
            done = (position + count) // size   # the blocks complete once the row's tokens are in
            for b in range(position // size, done):
                last = b * size + size - 1
                at.append(first + last - position)
                previous.append([table[q // page] * page + q % page for q in range(b * size, last)])
                starts.append(b * size)
            keys = None
            if done > self.top:   # past the budget: the row's tokens choose among its blocks' keys
                pages = torch.tensor(table, dtype=torch.int64, device=m.device)
                where = torch.arange(done, dtype=torch.int64, device=m.device) * size + (size - 1)
                keys = pages[where // page] * page + where % page
            plan.rows.append((first, first + count, done, keys))
        long = dict(dtype=torch.int64, device=m.device)
        plan.at = torch.tensor(at, **long)
        plan.previous = torch.tensor(previous, **long).view(len(at), size - 1)
        plan.starts = torch.tensor(starts, **long)
        plan.target = ctx.slots[self.index].index_select(0, plan.at)
        plan.keys = plan.ends = None
        return plan

    def _round_plan(self, ctx):
        """A round's tokens at fixed shapes, on the device alone: every token is treated as the end of the block of
        positions up to it, and writes that block's key only where it does end one."""
        m, torch, batch, size = self.m, self.m.torch, ctx.batch, self.block
        pool, plan = m.pools[self.index], _BlockPlan()
        p, tokens, rows = ctx.positions, ctx.positions.shape[0], batch.rows
        slot = batch.slots_t.index_select(0, batch.rows_t)
        back = torch.arange(size - 1, 0, -1, dtype=torch.int64, device=m.device)
        plan.at = None
        plan.previous = pool.slots_at(slot[:, None].expand(tokens, size - 1), p[:, None] - back[None, :])
        plan.starts = (p - (size - 1)).clamp(min=0)
        plan.target = torch.where((p + 1) % size == 0, ctx.slots[self.index], -1)
        blocks = m.kv_tokens // size
        where = torch.arange(blocks, dtype=torch.int64, device=m.device) * size + (size - 1)
        plan.keys = pool.slots_at(batch.slots_t[:rows, None].expand(rows, blocks), where[None, :].expand(rows, blocks))
        plan.ends = ((p + 1) // size).to(torch.int32)
        plan.rows = None
        return plan

    def _index_rows(self, h, ctx):
        """(queries, raw): the indexer's queries [tokens][index heads][dim], normed and rotated at their positions,
        and the tokens' raw index keys [tokens][dim]."""
        m, t = self.m, h.shape[0]
        qk = ctx.mm(h, self.index_qk)
        width = self.index_heads * self.index_dim
        q = m.norms.rms_norm_rope(qk[:, :width].reshape(t, self.index_heads, self.index_dim), self.index_q_norm,
                                  m.eps, ctx.positions, m.rope.freqs, offset=m.norm_offset, scale=m.rope.scale)
        return q, qk[:, width:]

    def _write_index(self, raw, ctx):
        """The step's raw index keys at their positions, then the keys of the blocks its tokens complete at the
        blocks' last positions: the mean of the block's raw keys in position order (the earlier ones read back from
        the pool), normed and rotated at the block's first position."""
        m, torch = self.m, self.m.torch
        pool, plan, size, dim = m.pools[self.index], self._plan(ctx), self.block, self.index_dim
        pool.write(self.layer, ctx.slots[self.index], raw.reshape(-1, 1, dim).contiguous(), None)
        n = plan.target.shape[0]
        if not n:
            return
        earlier, _ = pool.rows_at(self.layer, plan.previous)
        own = raw if plan.at is None else raw.index_select(0, plan.at)
        group = torch.cat([earlier.view(n, size - 1, dim).to(raw.dtype), own[:, None, :]], dim=1)
        pooled = m.norms.stream_sum(group, scale=1.0 / size, out_dtype=raw.dtype)
        key = m.norms.rms_norm_rope(pooled[:, None, :], self.index_k_norm, m.eps, plan.starts, m.rope.freqs,
                                    offset=m.norm_offset, scale=m.rope.scale)
        pool.write(self.layer, plan.target, key.contiguous(), None)

    # -- selection ---------------------------------------------------------------------------------------------------
    def _tail(self, picked, positions):
        """The attention's block list of each token, [tokens][top + 1] int32: its chosen blocks (ascending, -1 past
        them) and the block it lies in where that one is not complete, in block order, -1 last."""
        torch, size = self.m.torch, self.block
        tail = torch.where((positions + 1) % size != 0, positions // size, -1)
        both = torch.cat([picked.long(), tail[:, None]], dim=1)
        big = torch.iinfo(torch.int64).max
        both = torch.where(both < 0, big, both).sort(dim=-1).values
        return torch.where(both == big, -1, both).to(torch.int32)

    def _scores(self, q, keys):
        """FP32 [tokens][blocks]: each token's scores of `keys` ([blocks][dim], or [rows][blocks][dim] for a round's
        rows of equal token counts), a token at a time and its heads summed in order, so a token's scores are its
        own."""
        torch = self.m.torch
        tokens = q.shape[0]
        run = tokens // keys.shape[0] if keys.dim() == 3 else 0
        q, keys = q.float(), keys.float()
        out = []
        for t in range(tokens):
            found = torch.relu(q[t] @ (keys[t // run] if run else keys).transpose(0, 1))
            total = found[0]
            for head in range(1, found.shape[0]):
                total = total + found[head]
            out.append(total * self.index_scale)
        return torch.stack(out)

    def _select(self, q, ctx):
        """The block list of every token of the step ([tokens][top + 1] int32)."""
        from crit_plugins.device import topk_blocks
        m, torch, size = self.m, self.m.torch, self.block
        pool, plan, p = m.pools[self.index], self._plan(ctx), ctx.positions
        store = pool.layer_pages(self.layer)[0].view(-1, self.index_dim) if self.kernel is not None else None
        if ctx.batch.round:
            if self.kernel is not None:
                ones = torch.ones(q.shape[0], self.index_heads, dtype=torch.float32, device=q.device)
                scores = self.kernel.scores(q.contiguous(), store, ones, plan.ends, slots=plan.keys)
                return self._tail(self.kernel.topk(scores, plan.ends, self.top), p)
            keys, _ = pool.rows_at(self.layer, plan.keys)
            return topk_blocks(self._scores(q, keys.view(*plan.keys.shape, self.index_dim)), self.top, p, size)
        found = []
        for first, end, done, slots in plan.rows:
            positions = p[first:end]
            if slots is None:   # no more blocks than the budget: every complete block, whatever its score
                zeros = torch.zeros(end - first, max(done, 1), dtype=torch.float32, device=q.device)
                found.append(topk_blocks(zeros, self.top, positions, size))
            elif self.kernel is not None:
                ones = torch.ones(end - first, self.index_heads, dtype=torch.float32, device=q.device)
                ends = ((positions + 1) // size).to(torch.int32)
                picked = self.kernel.select(q[first:end].contiguous(), store, ones, ends, self.top, slots=slots,
                                            workspace_bytes=self.score_bytes)
                found.append(self._tail(picked, positions))
            else:
                keys, _ = pool.rows_at(self.layer, slots)
                keys = keys.view(done, self.index_dim)
                # a token's scores and the choice's keys of them are 12 bytes a block: tokens in groups within the
                # bound, each token's choice its own whatever its group
                group = max(1, self.score_bytes // (12 * done))
                for at in range(first, end, group):
                    stop = min(at + group, end)
                    found.append(topk_blocks(self._scores(q[at:stop], keys), self.top, p[at:stop], size))
        return found[0] if len(found) == 1 else torch.cat(found)

    def write(self, h, ctx):
        q, gate = super().write(h, ctx)
        if self.sparse:
            self._write_index(self._index_rows(h, ctx)[1], ctx)
        return q, gate

    def forward(self, h, ctx):
        if not self.sparse:
            return super().forward(h, ctx)
        q, gate = super().write(h, ctx)
        queries, raw = self._index_rows(h, ctx)
        self._write_index(raw, ctx)
        return self.attend(q, gate, self.m.pools[self.pool], ctx.batch, ctx, h.shape[0], h.dtype,
                           blocks=self._select(queries, ctx), block_size=self.block)


BLOCKS = {"indexed-attention": IndexedAttention}
