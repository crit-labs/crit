"""A block drafter stored beside the model (the layout's draft {block: block-drafter, prefix, config, keys}): a few
bidirectional decoder layers that fill a whole block of drafts in one pass from the target's own hidden rows, and a
small recurrent head that orders the block's drafts token by token.

The drafter's config is the model config's entry the layout's draft `config` names, its keys read as `keys` maps them
(the drafter's name -> the config's; a key not mapped is read under its own name): hidden_size, num_hidden_layers,
num_attention_heads, num_key_value_heads, head_dim, intermediate_size, rms_norm_eps, rope_theta (a decoder's standard
keys), block_size, head_rank (the recurrent head's state width), tap_layers, local_layers and sliding_window. Its
tensors under the prefix are read under the names below; a checkpoint that spells them otherwise gets rules in its
layout's `names`. Its arithmetic, with `taps` the target's rows it reads (tap_layers: the running row on entry to
those layer applications, counted over every pass of a looped span, then the final-normed row):

    mix_i(t)     = rms(sum_j softmax(tap_weights)[i][j] * t_j) * tap_norm_i     a layer's (or the head's) view of taps t
    context      every committed position p of the sequence: per layer, k = rope(rms_head(context_k_proj(mix_i(t_p)))),
                 v = context_v_proj(mix_i(t_p)), kept in the drafter's own cache
    block        positions P .. P + B - 1 (P: the pending token's position): row 0 the pending token's embedding
                 (normed, by the residual's embedding norm where it has one), rows 1 .. B - 1 mask_embedding; each
                 layer a pre-norm decoder layer (q/k normed per head, RoPE, attention over the cache's positions and
                 the whole block, no causal mask inside it; local layers see the last sliding_window positions before
                 P), a dense SwiGLU, the final norm
    head         f = mix_last(t_{P-1}); per draft d (rows 1 .. B - 1, in order), with state s (zeros) and the token
                 before it (the pending token, then each draft), under recurrent_head.:
                     [gate | cand | out] = recurrent_proj([s, token_embedding[prev], hidden_d, f])
                     s = sigmoid(gate) * s + (1 - sigmoid(gate)) * tanh(cand)
                     logits = cap * tanh((head(hidden_d) + head(out_proj(tanh(out)))) / cap)
                 head(out_proj(.)) folded once at load into a [vocabulary][head_rank] matrix and its bias.

The drafter only proposes: the target verifies every draft, so its precision (BF16 weights, FP32 sums) changes which
drafts are kept, never what a request emits.

A model technique: the model plugin (models/core/model-template) mounts this code beside its own and finds the draft
block block-drafter in DRAFTERS (BlockDrafter: what the core asks a drafter technique). It reads the target's tapped
rows, which the models core keeps for any residual that gives its running row (residual.py: row(state)).

As the model's device draft head (draft:head-device) in Core's captured rounds: keep_features stores each verify
column's taps, catch_up writes the cache rows of the positions the round committed, logits(depth) runs the block pass
at depth 0 and one head step per depth. Each round slot keeps a full layer's rows of every position and a local
layer's of only the positions its block pass reads (its window and the block: a ring of R = window + block positions,
each written at p % R and p % R + R so any window reads as one span); a prompt's rows are written as its chunks run
(prompt), into the session, and copied into the slot it decodes in (seed_session)."""
import json
from collections.abc import Mapping
from pathlib import Path

# The tensors of the drafter under its prefix (its config names the shapes): each layer's, and the recurrent head's.
LAYER = ("tap_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
         "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
         "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.context_k_proj.weight",
         "self_attn.context_v_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
         "self_attn.context_k_norm.weight")
HEAD = ("tap_norm.weight", "token_embedding.weight", "recurrent_proj.weight", "recurrent_proj.bias", "out_proj.weight",
        "out_proj.bias")
# The drafter's config keys: a decoder's standard ones, and its own.
KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
        "intermediate_size", "rms_norm_eps", "rope_theta", "block_size", "head_rank", "tap_layers", "local_layers",
        "sliding_window")


def stored(ck, prefix):
    """Whether the checkpoint stores a block drafter under `prefix`."""
    return f"{prefix}tap_weights" in ck and f"{prefix}recurrent_head.recurrent_proj.weight" in ck and \
        f"{prefix}layers.0.self_attn.context_k_proj.weight" in ck


def config_of(model, spec):
    """The drafter's config under its own keys (KEYS): the model config's entry the layout's draft `config` names,
    read through the draft's `keys` ({drafter's key: the config's})."""
    key = spec.get("config")
    if not key:
        raise ValueError("model-template: a block drafter's layout entry names the model config's key that holds the "
                         "drafter's config (draft.config)")
    found = model.setting(key) or (getattr(model.config, "raw", None) or {}).get(key)
    folder = getattr(model.ck, "folder", None)
    if not found and folder is not None and (Path(folder) / "config.json").is_file():   # a multimodal config's root
        found = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8")).get(key)
    if isinstance(found, str):
        found = json.loads(found)
    if isinstance(found, Mapping):
        found = dict(found)
    names = dict(spec.get("keys") or {})
    unknown = sorted(set(names) - set(KEYS))
    if unknown:
        raise ValueError(f"model-template: a block drafter's draft.keys maps {', '.join(unknown)}, which it does not "
                         f"read ({', '.join(KEYS)})")
    if not isinstance(found, dict):
        raise ValueError(f"model-template: the block drafter's config is the model config's {key}, which it lacks")
    cfg = {name: found.get(names.get(name, name)) for name in KEYS}
    missing = [name for name in ("block_size", "head_rank", "tap_layers", "hidden_size", "num_hidden_layers")
               if cfg[name] is None]
    if missing:
        raise ValueError(f"model-template: the block drafter's config ({key}) gives no "
                         f"{', '.join(names.get(n, n) for n in missing)}")
    return cfg


def taps_of(cfg, applications):
    """The layer applications whose entry row the drafter reads (tap_layers below `applications`), the last id the
    final-normed row (== applications)."""
    ids = [int(i) for i in cfg.get("tap_layers") or ()]
    if not ids or ids[-1] != applications or any(not 0 <= i < applications for i in ids[:-1]):
        raise ValueError(f"model-template: a block drafter's tap_layers {ids} are layer applications below "
                         f"{applications}, then {applications} (the final-normed row)")
    return tuple(ids[:-1])


def nbytes(ck, prefix, vocabulary, rank):
    """The device bytes of the drafter's weights and the folded head (BF16 [vocabulary][rank] and FP32 bias)."""
    return int(sum(ck.nbytes(n) for n in ck.names() if n.startswith(prefix)) + vocabulary * (2 * rank + 4))


def rings(cfg, tokens):
    """Per layer, the ring a local layer keeps (R = sliding window + block: the positions a block pass reads) where
    its doubled ring is smaller than `tokens` positions, else 0 (the layer keeps every position)."""
    window = int(cfg.get("sliding_window") or 0)
    r = window + int(cfg["block_size"]) if window else 0
    local = set(int(i) for i in cfg.get("local_layers") or ())
    return [r if i in local and 2 * r + 1 < tokens else 0 for i in range(int(cfg["num_hidden_layers"]))]


def columns(r, tokens):
    """A layer's cache columns a round slot: its ring twice and one no read reaches (r), else `tokens` positions."""
    return 2 * r + 1 if r else tokens


def spots(torch, r, tokens, positions):
    """The columns a layer keeps positions [n] at: its own (r == 0); in a ring p % r and p % r + r, the slot's last
    position (tokens - 1: catch-up's unused columns) the column no read reaches."""
    if not r:
        return (positions,)
    unused, at = positions == tokens - 1, positions % r
    return tuple(torch.where(unused, torch.full_like(at, 2 * r), at + o) for o in (0, r))


def span(r, first, last):
    """Positions first .. last (last - first < r) as the columns they are read at (a ring's: one contiguous span)."""
    if not r:
        return first, last
    shift = first - first % r
    return first - shift, last - shift


def cache_bytes(cfg, rows, tokens):
    """The drafter's cache: every layer's keys and values, a round slot's columns each (rings), BF16."""
    per = sum(columns(r, tokens) for r in rings(cfg, tokens))
    return int(per * 2 * rows * cfg["num_key_value_heads"] * cfg["head_dim"] * 2)


class BlockHead:
    """The block drafter as the model's draft head over round slots 0 .. rows - 1, at up to `widest` drafts a round
    (at most block_size - 1)."""

    def __init__(self, model, prefix, cfg, taps, rows, widest, tokens, head_matrix):
        torch = self.torch = model.torch
        self.m, self.rows, self.widest, self.cols = model, rows, widest, widest + 1
        self.cfg = cfg
        self.block, self.rank = int(cfg["block_size"]), int(cfg["head_rank"])
        if widest > self.block - 1:
            raise ValueError(f"model-template: the block drafter drafts {self.block - 1} a pass, not {widest}")
        self.layers, self.heads = int(cfg["num_hidden_layers"]), int(cfg["num_attention_heads"])
        self.kv_heads, self.head_dim = int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
        self.hidden, self.eps, self.theta = int(cfg["hidden_size"]), float(cfg["rms_norm_eps"]), float(cfg["rope_theta"])
        self.local = set(int(i) for i in cfg.get("local_layers") or ())
        self.window = int(cfg.get("sliding_window") or 0)
        self.taps = tuple(taps)
        self.width = (len(self.taps) + 1) * model.hidden   # the target's feature: its taps, then its final row
        self.tokens = tokens                                 # cache positions a slot
        dev, bf16 = model.device, torch.bfloat16
        ck = model.ck

        def get(name, dtype=bf16):
            return ck.get(prefix + name, dev).to(dtype).contiguous()

        self.w = [{n: get(f"layers.{i}.{n}", torch.float32 if "norm" in n else bf16) for n in LAYER}
                  for i in range(self.layers)]
        for w in self.w:   # k|v context projections as one product
            w["ctx"] = torch.cat([w.pop("self_attn.context_k_proj.weight"), w.pop("self_attn.context_v_proj.weight")])
            w["qkv"] = torch.cat([w.pop("self_attn.q_proj.weight"), w.pop("self_attn.k_proj.weight"),
                                  w.pop("self_attn.v_proj.weight")])
            w["gate_up"] = torch.cat([w.pop("mlp.gate_proj.weight"), w.pop("mlp.up_proj.weight")])
        self.h = {n: get(f"recurrent_head.{n}", torch.float32 if n != "token_embedding.weight" else bf16) for n in HEAD}
        self.alpha = torch.softmax(get("tap_weights", torch.float32), dim=-1)       # [layers + 1][taps + 1]
        self.layer_alpha = self.alpha[:-1].to(bf16).contiguous()                     # the layers' rows
        self.mask = get("mask_embedding", torch.float32)
        self.norm = get("norm.weight", torch.float32)
        rnn = self.h.pop("recurrent_proj.weight")
        self.rnn_state = rnn[:, :2 * self.rank].contiguous()      # over [state, token_embedding[prev]]
        self.rnn_rows = rnn[:, 2 * self.rank:].to(bf16).contiguous()   # over [hidden, f]
        # The head's own term folded through the target's head: head(out_proj(x)) = fold x + fold_bias.
        head = head_matrix(ck)                                     # [vocabulary][hidden] as stored (BF16)
        self.vocabulary = head.shape[0]
        out_w, out_b = self.h.pop("out_proj.weight"), self.h.pop("out_proj.bias")
        fold, fold_b = [], []
        for first in range(0, head.shape[0], 16384):
            part = head[first:first + 16384].to(dev).float()
            fold.append((part @ out_w).to(bf16))
            fold_b.append(part @ out_b)
        self.fold, self.fold_b = torch.cat(fold).contiguous(), torch.cat(fold_b).contiguous()
        del head
        # The matrices through the model's projections slot, draft-only (re-encoded where option mtp_encode asks:
        # the target verifies every draft), so their rows run on the slot's decode kernels.
        encode = None if model.options["mtp_encode"] == "bf16" else model.options["mtp_encode"]

        def held(t):
            return model.proj.load({"weight": t.contiguous()}, "bf16", role="draft", encode=encode)

        for w in self.w:
            for n in ("ctx", "qkv", "self_attn.o_proj.weight", "gate_up", "mlp.down_proj.weight"):
                w[n] = held(w[n])
        self.rnn_rows, self.fold = held(self.rnn_rows), held(self.fold)
        self.embed_norm = getattr(model.residual, "embed_norm", None)   # the residual's embedding norm, if any
        inv = self.theta ** (-torch.arange(0, self.head_dim, 2, device=dev, dtype=torch.float64) / self.head_dim)
        self.inv_freq = inv.float()
        # Device state, allocated once (captured rounds read and write it in place).
        self.features = torch.zeros(rows, self.cols, self.width, dtype=bf16, device=dev)
        self.rings = rings(cfg, tokens)
        shapes = [(rows, self.kv_heads, columns(r, tokens), self.head_dim) for r in self.rings]
        self.k = [torch.zeros(shape, dtype=bf16, device=dev) for shape in shapes]
        self.v = [torch.zeros(shape, dtype=bf16, device=dev) for shape in shapes]
        self.base = torch.zeros(rows, self.block - 1, self.vocabulary, dtype=torch.float32, device=dev)
        self.context = torch.zeros(rows, self.block - 1, 3 * self.rank, dtype=torch.float32, device=dev)
        self.state = torch.zeros(rows, self.rank, dtype=torch.float32, device=dev)
        self.slots = torch.arange(rows, device=dev)
        self.offsets = torch.arange(self.block, device=dev)
        self.cap = float(model.softcap or 0.0)
        self.query_rows = max(16, 1 << (self.heads // self.kv_heads * self.block - 1).bit_length())
        self.ops = model.kernel("row-ops")   # crit's span attention on CUDA, None for torch's

    # -- arithmetic ----------------------------------------------------------------------------------------------------
    def _mm(self, x, matrix):
        """x [n][in] times a held matrix: FP32 [n][out]."""
        rows, n = self.m.rows, x.shape[0]
        groups = rows.rows(n) if n <= rows.decode_rows else [(0, n, True)]   # a prompt's rows: one prompt call
        return rows.matmul(x.to(self.torch.bfloat16).contiguous(), matrix, groups).float()

    def _rms(self, x, weight=None):
        x = x.float()
        y = x * self.torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return y if weight is None else y * weight

    def _rope(self, x, positions):
        """x [..][n][heads][dim] at positions [..][n] (the halves rotated)."""
        torch = self.torch
        angle = positions.float()[..., None] * self.inv_freq
        cos, sin = angle.cos()[..., None, :], angle.sin()[..., None, :]
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def _mix(self, taps, row, weight):
        """taps [n][taps + 1][hidden] -> [n][hidden] FP32: a layer's (row) weighted, normed view of them."""
        t = taps.view(taps.shape[0], -1, self.hidden).float()
        return self._rms((t * self.alpha[row][None, :, None]).sum(1), weight)

    def _mixes(self, taps):
        """taps [n][width] -> [n][layers][hidden] BF16: every layer's weighted sum of the taps in one product (no FP32
        copy of the taps: a prompt chunk's taps are its widest tensor)."""
        return self.torch.matmul(self.layer_alpha, taps.reshape(taps.shape[0], -1, self.hidden))

    def _context(self, taps, positions):
        """The cache rows of taps [n][width] at positions [n]: per layer (k [n][kv][dim] roped and normed, v)."""
        found = []
        mixed = self._mixes(taps)
        for i, w in enumerate(self.w):
            c = self._rms(mixed[:, i], w["tap_norm.weight"]).to(self.torch.bfloat16)
            kv = self._mm(c, w["ctx"]).view(-1, 2, self.kv_heads, self.head_dim)
            k = self._rope(self._rms(kv[:, 0], w["self_attn.context_k_norm.weight"]), positions)
            found.append((k.to(self.torch.bfloat16), kv[:, 1].to(self.torch.bfloat16)))
        return found

    def _put(self, i, slots, positions, k, v):
        """Layer i's rows k, v [n][kv][dim] into the slots' cache at positions (both [n])."""
        for at in spots(self.torch, self.rings[i], self.tokens, positions):
            self.k[i][slots, :, at] = k
            self.v[i][slots, :, at] = v

    def _write(self, slots, positions, rows):
        """rows (from _context) into the slots' cache at positions (both [n])."""
        for i, (k, v) in enumerate(rows):
            self._put(i, slots, positions, k, v)

    def _attend(self, i, q, frontier, rows):
        """q [rows][block][heads][dim] over slot r's cache positions up to frontier + block - 1 (local layers: the
        window before frontier on), every block row seeing the whole block."""
        torch = self.torch
        g = self.heads // self.kv_heads
        q = q.view(rows, self.block, self.kv_heads, g, self.head_dim).permute(0, 2, 3, 1, 4).reshape(
            rows, self.kv_heads, g * self.block, self.head_dim)
        last = frontier + self.block - 1
        first = (last - (self.window + self.block) + 1).clamp(min=0) if i in self.local and self.window \
            else torch.zeros_like(last)
        first, last = span(self.rings[i], first, last)
        if self.ops is not None:   # row-ops' span_attention: a power of two of queries, at least 16, a KV head
            wide = self.query_rows
            padded = q.new_zeros(rows, self.kv_heads, wide, self.head_dim, dtype=torch.bfloat16)
            padded[:, :, :g * self.block] = q
            out = self.ops.span_attention(padded, self.k[i][:rows], self.v[i][:rows], first,
                                          last)[:, :, :g * self.block]
        else:
            at = torch.arange(self.k[i].shape[2], device=q.device)[None, :]
            mask = ((at >= first[:, None]) & (at <= last[:, None]))[:, None, None, :]
            out = torch.nn.functional.scaled_dot_product_attention(q.to(torch.bfloat16), self.k[i][:rows],
                                                                   self.v[i][:rows], attn_mask=mask)
        return out.reshape(rows, self.kv_heads, g, self.block, self.head_dim).permute(0, 3, 1, 2, 4).reshape(
            rows * self.block, self.heads * self.head_dim)

    def _block(self, anchors, frontier, rows):
        """The block pass: [rows][block - 1][hidden] FP32 hidden rows of the drafts."""
        torch, bf16 = self.torch, self.torch.bfloat16
        x = self.mask.expand(rows, self.block, self.hidden).clone()
        x[:, 0] = self._rms(self.m.embed.lookup(anchors), self.embed_norm)
        x = x.reshape(rows * self.block, self.hidden)
        positions = (frontier[:, None] + self.offsets).reshape(-1)
        slots = self.slots[:rows, None].expand(rows, self.block).reshape(-1)
        qd, kd = self.heads * self.head_dim, self.kv_heads * self.head_dim
        for i, w in enumerate(self.w):
            h = self._rms(x, w["input_layernorm.weight"]).to(bf16)
            qkv = self._mm(h, w["qkv"])
            q = self._rope(self._rms(qkv[:, :qd].view(-1, self.heads, self.head_dim), w["self_attn.q_norm.weight"]),
                           positions)
            k = self._rope(self._rms(qkv[:, qd:qd + kd].view(-1, self.kv_heads, self.head_dim),
                                     w["self_attn.k_norm.weight"]), positions)
            self._put(i, slots, positions, k.to(bf16),   # the block's own rows, rewritten at the catch-up
                      qkv[:, qd + kd:].view(-1, self.kv_heads, self.head_dim).to(bf16))
            a = self._attend(i, q.view(rows, self.block, self.heads, self.head_dim), frontier, rows)
            x = x + self._mm(a, w["self_attn.o_proj.weight"])
            h = self._rms(x, w["post_attention_layernorm.weight"]).to(bf16)
            gu = self._mm(h, w["gate_up"])
            half = gu.shape[-1] // 2
            x = x + self._mm(torch.nn.functional.silu(gu[:, :half]) * gu[:, half:], w["mlp.down_proj.weight"])
        return self._rms(x, self.norm).view(rows, self.block, self.hidden)[:, 1:]

    # -- the draft head (draft:head-device) ------------------------------------------------------------------------------
    def seed_feature(self, slot, hidden):
        self.features[slot, 0].copy_(hidden)

    def seed_session(self, slot, session):
        """The prompt's cache rows (prompt) into round slot `slot`."""
        found = getattr(session, "draft_rows", None)
        if found is not None:
            for i in range(self.layers):   # a ring layer's whole ring, a full layer's prompt positions
                n = found[0][i].shape[1]
                self.k[i][slot, :, :n] = found[0][i]
                self.v[i][slot, :, :n] = found[1][i]
        session.draft_rows = None

    def prompt(self, rows, hidden):
        """The cache rows of prompt positions (`rows`: (session, first position, count, first row of `hidden`)), kept
        in the session until it takes a round slot."""
        torch = self.torch
        for s, first, count, at in rows:
            if getattr(s, "draft_rows", None) is None:   # per layer [kv][columns][dim], a ring layer's as in a slot
                shapes = [(self.kv_heads, columns(r, len(s.prompt)), self.head_dim)
                          for r in self.rings]
                s.draft_rows = tuple([torch.zeros(shape, dtype=torch.bfloat16, device=self.m.device)
                                      for shape in shapes] for _ in range(2))
            positions = torch.arange(first, first + count, device=self.m.device)
            for i, (k, v) in enumerate(self._context(hidden[at:at + count], positions)):
                r = self.rings[i]
                if not r:
                    s.draft_rows[0][i][:, first:first + count] = k.transpose(0, 1)
                    s.draft_rows[1][i][:, first:first + count] = v.transpose(0, 1)
                    continue
                keep = max(0, count - r)   # a ring holds the chunk's last r positions (distinct columns)
                for spot in spots(torch, r, self.tokens, positions[keep:]):
                    s.draft_rows[0][i][:, spot] = k[keep:].transpose(0, 1)
                    s.draft_rows[1][i][:, spot] = v[keep:].transpose(0, 1)

    def move(self, source, target):
        self.features[target].copy_(self.features[source])
        for k, v in zip(self.k, self.v):
            k[target].copy_(k[source])
            v[target].copy_(v[source])

    def keep_features(self, view, hidden):
        rows, cols = view.rows, view.width + 1
        held = (view.hold[:rows] > 0)[:, None, None]
        self.features[:rows, :cols] = self.torch.where(held, self.features[:rows, :cols], hidden)

    def catch_up(self, view):
        """The cache rows of the positions the last round committed: column c of row r at frontier - count + c, for
        c < count (the others land on the slot's last position, never read)."""
        torch = self.torch
        rows, cols = view.rows, self.cols
        count = view.catch_up_count[:rows]
        c = torch.arange(cols, device=count.device)[None, :]
        positions = torch.where(c < count[:, None], view.frontier[:rows, None] - count[:, None] + c, self.tokens - 1)
        slots = self.slots[:rows, None].expand(rows, cols).reshape(-1)
        self._write(slots, positions.reshape(-1), self._context(self.features[:rows].reshape(rows * cols, -1),
                                                                positions.reshape(-1)))

    def logits(self, depth, tokens, view, parents=None):
        torch = self.torch
        rows = view.rows
        if depth == 0:
            frontier = view.frontier[:rows]
            hidden = self._block(tokens[:rows], frontier, rows)                     # [rows][block - 1][hidden]
            last = (view.catch_up_count[:rows] - 1).clamp(min=0)[:, None, None]
            taps = self.features[:rows].gather(1, last.expand(rows, 1, self.width)).squeeze(1)
            f = self._mix(taps, self.layers, self.h["tap_norm.weight"])                # [rows][hidden]
            flat = hidden.reshape(-1, self.hidden).to(torch.bfloat16)
            self.base[:rows] = self.m.raw_logits(flat).view(rows, self.block - 1, -1)
            joined = torch.cat([flat, f[:, None].expand(rows, self.block - 1, self.hidden).reshape(-1, self.hidden)
                               .to(torch.bfloat16)], dim=-1)
            self.context[:rows] = (self._mm(joined, self.rnn_rows) + self.h["recurrent_proj.bias"]).view(
                rows, self.block - 1, -1)
            self.state[:rows] = 0.0
        prev = self.h["token_embedding.weight"][tokens[:rows]].float()
        raw = self.context[:rows, depth] + torch.cat([self.state[:rows], prev], dim=-1) @ self.rnn_state.T
        r = self.rank
        g = torch.sigmoid(raw[:, :r])
        self.state[:rows] = g * self.state[:rows] + (1.0 - g) * torch.tanh(raw[:, r:2 * r])
        latent = torch.tanh(raw[:, 2 * r:]).to(torch.bfloat16)
        logits = self.base[:rows, depth] + self._mm(latent, self.fold) + self.fold_b
        if self.cap:
            logits = torch.tanh(logits / self.cap) * self.cap
        return logits

    def commit_round(self, view):
        """Nothing beyond the next catch-up: it rewrites the positions the block pass wrote that were committed."""


class BlockDrafter:
    """The draft block block-drafter as the model core asks a drafter technique (DRAFTERS): whether a checkpoint stores
    it, the layer applications whose entry rows it reads, its device bytes, and its draft head (BlockHead), made at
    load while the checkpoint is open."""
    name = "checkpoint block drafter"   # the facts' drafter_kind

    @staticmethod
    def stored(ck, prefix):
        return stored(ck, prefix)

    def __init__(self, model, prefix, spec):
        if not callable(getattr(model.residual, "row", None)):
            raise ValueError(f"model-template: a block drafter reads the target's rows on entry to its tap layers, "
                             f"and the {model.residual.name} residual keeps no single row (residual.py: row)")
        self.m, self.prefix, self.cfg = model, prefix, config_of(model, spec)
        self.spare = int(self.cfg["block_size"])   # cache positions a block pass writes from the pending token on

    def taps(self, applications):
        return taps_of(self.cfg, applications)

    def weight_bytes(self, ck, vocabulary):
        return nbytes(ck, self.prefix, vocabulary, int(self.cfg["head_rank"]))

    def device_bytes(self, ck, vocabulary, rows, tokens):
        """Its weights, the folded head and every round slot's cache of `tokens` positions."""
        return self.weight_bytes(ck, vocabulary) + cache_bytes(self.cfg, rows, tokens)

    def head(self, taps, rows, widest, tokens, head_matrix):
        return BlockHead(self.m, self.prefix, self.cfg, taps, rows, widest, tokens, head_matrix)


DRAFTERS = {"block-drafter": BlockDrafter}
