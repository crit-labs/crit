"""The chunk-aligned exact form of a StatePool's gated delta (the plugin's decode option "chunk"): every row of a
sequence, decode, verify or prompt, runs recurrent-scans' gated_delta_chunk_rows (structure delta-chunk-rows) over
the sequence's tokens since its last 64-aligned position (CHUNK) from the state there, so a token's output has the
bits a prompt row gives it whatever rows it came in: prompt rows on tensor cores with the decode's bits (the
assessment's exact chunked recurrence, recurrent-scans' README).

A pool in this form keeps, per layer and sequence (and one spare, which unbound round rows write):

- boundary: the state at the sequence's last CHUNK-aligned position (the pool's `states` are its first `sequences`
  slots, so open, fork and the commit bookkeeping see it);
- the chunk's tokens since then (held of them, 0 .. CHUNK - 1, the same for every layer): keys and queries normalized
  as the rows give them, values, log decays and betas, room for CHUNK + capacity tokens;
- next: the state after a row's first complete chunk, which a commit that keeps that chunk's tokens makes the boundary.

A row writes its tokens into the sequence's buffer after the held ones and reads boundary + buffer; nothing it does is
kept until a commit (accept for a step's rows, commit_round for a captured round, commit_prompt for a captured prompt
chunk): held + count tokens, and where that reaches CHUNK, the boundary from next and the buffer moved down a chunk.
Every commit is device tensor work (torch.where on the counts), so a captured round commits in its graph. The scan's
stash, pending counts and replay are not used; the convolution window keeps its own deferred commit. A prompt row
longer than the buffer runs as one row over the held tokens and its own, its tail (the tokens past its last complete
chunk) kept for its accept.

Its decode costs more than the scan (each row recomputes its chunk so far: recurrent-scans' README has the numbers),
and its prompt rows run on tensor cores with the decode's bits: a configuration that asks exact prompt rows of long
prompts is where it pays."""

CHUNK = 64


class ChunkState:
    """The chunk form's buffers of a StatePool (module docstring)."""

    def __init__(self, pool):
        torch = pool.torch
        if pool.kernels is None or not hasattr(pool.kernels, "gated_delta_chunk_rows"):
            raise ValueError("decode chunk runs on recurrent-scans' gated_delta_chunk_rows, which did not load here")
        if pool.elastic is not None:
            raise ValueError("decode chunk keeps its states in one allocation: elastic pools take decode scan")
        if pool.keys not in (64, 128) or pool.values % 16:
            raise ValueError(f"decode chunk takes keys 64 or 128 and values a multiple of 16, not {pool.keys} x "
                             f"{pool.values}")
        self.pool, self.torch = pool, torch
        L, n = len(pool.layers), pool.sequences
        self.spare, self.room = n, CHUNK + pool.capacity
        rows = (n + 1) * self.room
        f32 = dict(dtype=torch.float32, device=pool.device)
        h, kh, dk, dv = pool.heads, pool.key_heads, pool.keys, pool.values
        self.boundary = torch.zeros(L, n + 1, h, dk, dv, **f32)
        self.next = torch.zeros(L, n + 1, h, dk, dv, **f32)
        self.q = torch.zeros(L, rows, kh, dk, **f32)
        self.k = torch.zeros(L, rows, kh, dk, **f32)
        self.v = torch.zeros(L, rows, h, dv, **f32)
        self.g = torch.zeros(L, rows, h, **f32)
        self.b = torch.zeros(L, rows, h, **f32)
        self.out = torch.zeros(rows, h, dv, **f32)
        self.held = torch.zeros(n + 1, dtype=torch.int64, device=pool.device)
        self.tail = torch.arange(pool.capacity, dtype=torch.int64, device=pool.device)   # a commit's moved tokens
        self.prompts = {}   # sequence -> (held after its prompt row, {layer: tail tensors}) until its accept
        pool.states = self.boundary[:, :n]

    # -- the buffers ----------------------------------------------------------------------------------------------------
    def buffers(self, li):
        return self.q[li], self.k[li], self.v[li], self.g[li], self.b[li]

    def open(self, i):
        self.boundary[:, i].zero_()
        self.held[i] = 0

    def fork(self, source, target):
        torch = self.torch
        span = torch.arange(self.room, device=self.held.device)
        for buf in (self.q, self.k, self.v, self.g, self.b):
            buf[:, target * self.room + span] = buf[:, source * self.room + span]
        self.boundary[:, target] = self.boundary[:, source]
        self.held[target] = self.held[source]

    def live_state(self, li, i, scale):
        """The state after the held tokens: the chunk's prefill from the boundary (a new tensor)."""
        state = self.boundary[li, i].clone()
        n = int(self.held[i])
        if n:
            lo = i * self.room
            q, k, v, g, b = (x[lo:lo + n] for x in self.buffers(li))
            self.pool.kernels.gated_delta_prefill(q, k, v, g, b, state, scale=scale, tensor_cores=True)
        return state

    # -- rows -----------------------------------------------------------------------------------------------------------
    def rows(self, li, q, k, v, g, b, seq, start, count, scale, out):
        """Rows of tokens (q, k normalized FP32 [T][key heads][keys], v [T][heads][values], g, b [T][heads]): row r of
        sequence index seq[r] (device int64; the spare for an unbound row) takes tokens start[r] .. start[r] +
        count[r] - 1; written after its held ones, read with them from the boundary, outputs to out [T][heads][values].
        Device work only; nothing kept."""
        torch = self.torch
        T, R = q.shape[0], seq.shape[0]
        token = torch.arange(T, device=q.device)
        row = torch.searchsorted(start[1:].contiguous(), token, right=True).clamp(max=R - 1)
        at = seq[row] * self.room + self.held[seq[row]] + (token - start[row])
        bq, bk, bv, bg, bb = self.buffers(li)
        for buf, x in ((bq, q), (bk, k), (bv, v), (bg, g), (bb, b)):
            buf.index_copy_(0, at, x)
        meta = torch.stack([seq * self.room, self.held[seq] + count, seq, seq], 1).to(torch.int32).contiguous()
        self.pool.kernels.gated_delta_chunk_rows(bq, bk, bv, bg, bb, meta, self.boundary[li], self.next[li],
                                                 scale=scale, widest=self.room, out=self.out)
        out.copy_(self.out.index_select(0, at))
        return out

    def prompt(self, li, q, k, v, g, b, i, sequence, scale, out):
        """A step's prompt row of sequence index i (host), longer than the buffer: one row over its held tokens and
        its own from the boundary; the state after its last complete chunk to next, its tail kept for its accept."""
        torch = self.torch
        n = int(self.held[i])
        lo = i * self.room
        packed = [torch.cat([buf[lo:lo + n], x]) for buf, x in zip(self.buffers(li), (q, k, v, g, b))]
        total = n + q.shape[0]
        meta = torch.tensor([[0, total, i, i]], dtype=torch.int32, device=q.device)
        found = self.pool.kernels.gated_delta_chunk_rows(*packed, meta, self.boundary[li], self.next[li], scale=scale,
                                                         widest=total)
        out.copy_(found[n:])
        cut = total // CHUNK * CHUNK
        self.prompts.setdefault(sequence, (total - cut, cut > 0, {}))[2][li] = [x[cut:total].clone() for x in packed]
        return out

    def captured_prompt(self, li, q, k, v, g, b, i, count, valid, scale, out):
        """A captured prompt chunk (bucket B, count real tokens, i the staged sequence index, all device tensors): the
        held tokens and the bucket as one row of room + B tokens; its boundary and buffer for this layer written in
        place (held, shared by the layers, moves at commit_prompt). Padding alone (valid all false) changes nothing."""
        torch = self.torch
        B = q.shape[0]
        lo = i * self.room
        n = self.held[i]
        span = torch.arange(self.room + B, device=q.device)
        at = n + torch.arange(B, device=q.device)
        packed = []
        for buf, x in zip(self.buffers(li), (q, k, v, g, b)):
            p = torch.zeros(self.room + B, *buf.shape[1:], dtype=buf.dtype, device=buf.device)
            p[:self.room] = buf.index_select(0, lo + span[:self.room])
            p.index_copy_(0, at, x)
            packed.append(p)
        total = n + count
        meta = torch.stack([torch.zeros_like(total), total, i, i], 1).to(torch.int32).contiguous()
        found = self.pool.kernels.gated_delta_chunk_rows(*packed, meta, self.boundary[li], self.next[li], scale=scale,
                                                         widest=self.room + B)
        out.copy_(torch.where(valid[:, None, None], found.index_select(0, at), 0.0))
        done = total >= CHUNK   # a chunk completed: the boundary moves to the state after the last complete one
        self.boundary[li, i] = torch.where(done[:, None, None, None], self.next[li, i], self.boundary[li, i])
        cut = total // CHUNK * CHUNK
        src = (cut + span[:self.room]).clamp(max=self.room + B - 1)
        for buf, p in zip(self.buffers(li), packed):
            buf.index_copy_(0, lo + span[:self.room], p.index_select(0, src))
        return out

    # -- commits --------------------------------------------------------------------------------------------------------
    def commit(self, seq, count):
        """Keep the first count[r] tokens of the last row of sequence index seq[r] (device int64), every layer:
        held + count, and where that completes the chunk, next becomes the boundary and the buffer moves down a
        chunk. Capture-safe."""
        torch = self.torch
        held = self.held[seq] + count
        done = held >= CHUNK
        self.boundary[:, seq] = torch.where(done.view(1, -1, 1, 1, 1), self.next[:, seq], self.boundary[:, seq])
        lo = (seq * self.room)[:, None]
        dst = (lo + self.tail[None, :]).reshape(-1)
        src = (lo + CHUNK + self.tail[None, :]).reshape(-1)
        moved = done[:, None].expand(-1, self.tail.numel()).reshape(-1)
        for buf in (self.q, self.k, self.v, self.g, self.b):
            keep = moved.view(1, -1, *([1] * (buf.dim() - 2)))
            buf[:, dst] = torch.where(keep, buf[:, src], buf[:, dst])
        self.held[seq] = torch.where(done, held - CHUNK, held)

    def accept_prompt(self, sequence, i):
        """A step's prompt row kept: the boundary from next where it completed a chunk, its tail the held tokens."""
        held, completed, tails = self.prompts.pop(sequence)
        lo = i * self.room
        for li, tail in tails.items():
            if completed:
                self.boundary[li, i] = self.next[li, i]
            for buf, x in zip(self.buffers(li), tail):
                buf[lo:lo + held] = x
        self.held[i] = held

    def drop_prompt(self, sequence):
        self.prompts.pop(sequence, None)

    def commit_captured(self, i, count):
        """A captured prompt chunk of `count` real tokens kept (its layers moved their boundaries and buffers in the
        graph): held moves past them."""
        self.held[i] = (self.held[i] + int(count)) % CHUNK
