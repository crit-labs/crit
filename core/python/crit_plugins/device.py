"""The device form of crit's slots: what a model plugin calls on the other slots' plugins with tensors on the engine's
device, inside its batched step.

A plugin that runs on a device (any of crit_plugins.DEVICES, the CPU too) keeps its slot's host form
(crit_plugins.slots.INTERFACES, rows of numbers), over the same arithmetic, so its tests compare it with the slot's host
reference on crit's cases; the model plugin calls its device form below. PluginTestCase.assertImplements checks it has
both, and Core's loader refuses an engine whose plugin in a computing slot lacks its slot's device form: the host form
is an oracle, and no engine serves one. Tensors are torch tensors on the engine's device and stream; nothing here imports torch until a
StepBatch is built.

A step's tokens are packed in row order: row r is a run of consecutive tokens of one sequence (a prompt chunk, a
decode token, a verify window), tokens StepBatch.starts[r] .. starts[r + 1] - 1, each at its own position in the
sequence. Every device plugin reads the same StepBatch, so a token's KV slot, its attention window and its recurrent
state all follow from it.
"""
import array
from dataclasses import dataclass, field

# The row count up to which every projections plugin's matmul keeps each row's bits whatever rows share the call (a
# step's decode rows, a verify window): DEVICE_INTERFACES["projections"]["matmul"]. A plugin may promise more in its
# facts (row_invariance rows); invariant_rows() reads it.
DECODE_ROWS = 8


def free_device_bytes(device="cuda"):
    """The device's free bytes for a placement plan or a pool's size, after handing back what torch's caching allocator
    holds and nothing uses (crit_plugins.assemble.release_cached): a model plugin that re-encodes or converts weights
    at load leaves those temporaries cached, and torch.cuda.mem_get_info alone counts them as taken, so a KV pool
    planned from it comes out smaller than the device allows, or refuses to fit. None where the device's memory
    cannot be read."""
    import torch
    from .assemble import release_cached
    if not torch.cuda.is_available():
        return None
    release_cached(device)
    return int(torch.cuda.mem_get_info(torch.device(device))[0])


def release_cached_under(free_below, device="cuda"):
    """Hand back what torch's caching allocator holds and nothing uses only when the device has fewer than
    `free_below` bytes free: for a model plugin where a request's temporaries end (a prompt's last chunk, a long
    score), so that blocks cached from one request leave room for the next one's allocations, the KV a longer prompt
    takes among them, without releasing after every step (each release waits for the device, and the allocations
    after it go back to the driver). Returns the bytes released (0 above the threshold), or None where the device's
    memory cannot be read."""
    import torch
    from .assemble import release_cached
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    if torch.cuda.mem_get_info(device)[0] >= free_below:
        return 0
    return release_cached("cuda")


def invariant_rows(projections):
    """The most rows one matmul call of this projections plugin takes as decode rows, each row's bits those of the row
    alone: its facts' row_invariance rows, at least DECODE_ROWS. A model plugin splits a step's decode rows (verify
    windows included) into calls of at most this many; a longer call is prompt rows."""
    facts = projections.facts() if callable(getattr(projections, "facts", None)) else {}
    rows = (facts.get("row_invariance") or {}).get("rows")
    return max(DECODE_ROWS, int(rows)) if rows else DECODE_ROWS

# slot -> {method: what it takes and returns}: the device form a plugin running on a device adds to its host form.
DEVICE_INTERFACES = {
    "projections": {
        "load": "load(tensors, format, role=None, encode=None) -> matrix: a matrix on the device from a checkpoint's "
                "stored tensors; format is the encoding (bf16, fp8:channel, fp8:tensor, mxfp8, nvfp4; a plugin "
                "refuses one it does not take), tensors as stored ({weight, scale, global_scale, input_scale} by "
                "role); role, where the model plugin gives one, is the matrix's part in the model in the model's own "
                "words (qkv, gate_up, ...), which a plugin may choose its arithmetic by; encode, which the model "
                "plugin passes only for a matrix only a drafter reads, names a format the plugin re-encodes the "
                "stored one to (nvfp4, fp8), recorded in its facts (draft_encoding)",
        "matmul": "matmul(x, matrix, out_dtype=None) -> y: y[t] = W x[t] for each token row of x ([tokens][in], the "
                  "activation dtype); in a call of up to invariant_rows(plugin) rows (decode rows, verify windows: "
                  "DECODE_ROWS, or the facts' row_invariance rows when more) every row's bits are those of the row "
                  "alone, so batched decoding matches serial; a longer call (prompt rows) may sum in another order "
                  "than a shorter one, unless the plugin's facts say row_invariance prompt_rows (then a row's bits are "
                  "the same in any call of more than invariant_rows(plugin) rows); out_dtype asks for the output in a "
                  "dtype of its own (FP32 for logits) from the same FP32 sums",
    },
    "kv_cache": {
        "pool": "pool(layers, kv_heads, head_dim, tokens, device, dtype='bf16', *, value_dim=None) -> KVPool: the "
                "KV of every sequence (KV_POOL: its methods and attributes, which the attention slot's plugin "
                "reads); value_dim, a keyword, the values' width (the keys' by default; 0 for a pool of keys only, "
                "with kv:keys)",
    },
    "attention": {
        "forward": "forward(layer, queries, pool, batch, scale=None) -> out: queries [tokens][heads][dim]; each token "
                   "attends its sequence's KV at positions up to its own, the batch's own rows included once "
                   "written; out [tokens][heads][value dim]. With the facts below also forward(..., blocks=None, "
                   "block_size=None, parents=None, lse=False): attention:blocks (blocks [tokens][k] KV block ids, "
                   "block_size positions each, -1 for none: a token reads only its listed blocks, still nothing past "
                   "its own position; topk_blocks chooses them), attention:tree (parents [tokens]: each token's parent "
                   "column in its row, -1 for none, parents before children: a verify window's columns see the "
                   "positions before it and their ancestors and themselves in it), attention:lse (lse=True also "
                   "returns each token's log-sum-exp [tokens][heads] FP32, natural log over the scaled scores)",
    },
    "recurrence": {
        "pool": "pool(layers, heads, keys, values, device, sequences=None, widest=0, conv=None, key_heads=None, "
                "rows=None) -> StatePool: each sequence's state per layer (heads value heads of [keys][values]) and, "
                "with conv=(channels, kernel) or (channels, kernel, dilation), the window of its causal depthwise "
                "convolution (the last (kernel - 1) * dilation inputs), with open(sequence), close(sequence), "
                "fork(sequence, new), accept(sequence, count) (the state and window after the first count tokens of "
                "the sequence's row in the last step: any prefix of a row of up to widest + 1 tokens, all or none of a "
                "longer one), and the rounds:recurrence methods where the plugin provides them; state(layer, sequence) "
                "and window(layer, sequence) (the committed state [heads][keys][values] and window "
                "[channels][(kernel - 1) * dilation]) and flush(sequence=None), which apply a deferred commit and make "
                "it final; key_heads: the heads of queries and keys (heads by default, value "
                "head h reading key head h // (heads / key_heads)); rows: round slots (sequences by default)",
        "forward": "forward(layer, query, key, value, log_decay, beta, pool, batch, geometry, out=None) -> out: each "
                   "row of a StepBatch or RoundBatch continues its sequence's committed state over its tokens "
                   "(query/key [tokens][key heads][keys], value [tokens][heads][values], log_decay and beta "
                   "[tokens][heads]); out [tokens][heads][values] FP32 (given, or allocated); geometry as the host "
                   "form's; a decode or verify row's bits its own, whatever rows share the call; a round row of an "
                   "unbound slot outputs zeros",
        "convolve": "convolve(layer, x, weight, bias, pool, batch, activation='silu', out=None) -> y: the causal "
                    "depthwise convolution of each row of x ([tokens][channels]) over its sequence's window at the "
                    "pool's dilation, y[t] = act(bias + sum_i weight[:, i] x[t - (kernel - 1 - i) * dilation]) (weight "
                    "[channels][kernel], bias [channels] or None, act silu or None), y "
                    "in x's dtype (out, when given, is not x); the row's inputs join the window when accept or "
                    "commit_round keeps them",
    },
    "router": {
        "select": "select(logits, top, bias=None) -> (ids, weights): each token's routed experts ([tokens][top] int64, "
                  "in selection order) and their weights ([tokens][top] FP32) from its router logits row "
                  "([tokens][experts]), by the host form's rule",
    },
    "experts": {
        "load": "load(layer, expert, gate_up, down, format='bf16'): an expert's [2 * inter][hidden] gate/up and "
                "[hidden][inter] down matrices on the device, each a bfloat16 tensor (format bf16) or, as projections' "
                "load takes them, its stored tensors {weight, scale, global_scale, input_scale} in `format` (fp8:channel, "
                "fp8:block, nvfp4, int4, mxfp4); a plugin refuses a format it does not compute",
        "forward": "forward(layer, x, ids, weights) -> out: the routed SwiGLU block of each token row of x "
                   "([tokens][hidden]) over its routes (ids and weights [tokens][top] as the router's, id -1 skipped) "
                   "and the layer's loaded experts, out [tokens][hidden] FP32; a token's result independent of the rows "
                   "beside it (a store of bundles goes through experts:bundles' forward_bundles)",
    },
    "embeddings": {
        "store": "store(rows, format=None, scales=None, scale=1.0, dtype=None) -> table: a lookup table on the device "
                 "([ids][width] of its dtype; format nvfp4: rows the E2M1 codes [ids][width / 2], scales the E4M3 "
                 "scales [ids][width / 16], scale the global scale, dtype the decoded dtype), returned as a handle with "
                 "lookup(ids) and close(); a plugin holds any number of tables",
        "lookup": "lookup(ids) -> rows: the last stored table's row of each id ([tokens] int64 -> [tokens][width])",
    },
    "sampler": {
        "sample": "sample(logits, params) -> tokens: one token per logits row ([rows][vocabulary]), params one dict "
                  "per row as select()'s (temperature, top_p, top_k, min_p, seed, position, stream, allowed: the "
                  "row's Allowed, whose mask(device, width) is the tokens it may take)",
    },
    # Every norms method: inputs BF16, FP16 or FP32 (each in its own dtype), FP32 math inside, one rounding to
    # out_dtype (bf16, fp16 or fp32 torch dtypes; by default the dtype of the tensor normalized or gated); the last
    # dimension contiguous, the others of any strides (views of a fused projection's output); each row's bits
    # independent of the rows beside it, whatever the row count (decode, verify and prompt rows agree); capture-safe
    # (no host synchronization; allocations only from torch's caching allocator, which a CUDA graph captures: the
    # result, unless `out` is given, whose last dimension is contiguous too, and any copies a layout needs); every
    # tensor of a call on one device; a new result tensor in the input's shape otherwise.
    "norms": {
        "rms_norm": "rms_norm(x, weight, eps, offset=0.0, residual=None, out_dtype=None, out=None) -> y, or (y, "
                    "residual) with a residual: y = x / sqrt(mean(x^2) + eps) * (offset + weight) per last-dimension "
                    "row (weight [dim], [groups][dim] for x [..., groups, dim]: each head or residual stream its own "
                    "weight, or None); with residual (x's shape, its own dtype: an FP32 residual stream beside BF16 "
                    "rows) the residual becomes residual + x in place, rounded to its dtype, and those rows are "
                    "normalized (y's default dtype then the residual's); the caller owns the residual, which the call "
                    "advances",
        "rms_norm_heads": "rms_norm_heads(x, weight, eps, offset=0.0, out_dtype=None, out=None) -> y: rms_norm of "
                          "each head of x [tokens][heads][dim] (query/key norms), weight [dim] or [heads][dim]",
        "rms_norm_rope": "rms_norm_rope(x, weight, eps, positions, freqs, offset=0.0, rotary_dim=None, style='half', "
                         "scale=1.0, out_dtype=None, out=None) -> y: rms_norm_heads then rope, in FP32 between them",
        "gated_rms_norm": "gated_rms_norm(x, z, weight, eps, offset=0.0, activation='silu', out_dtype=None, out=None) "
                          "-> y: rms_norm of each last-dimension row of x times activation(z) element by element "
                          "(silu or sigmoid; z x's shape), as gated linear-attention outputs are",
        "rope": "rope(x, positions, freqs, rotary_dim=None, style='half', scale=1.0, out_dtype=None, out=None) -> y: "
                "each token's heads (x [tokens][heads][dim] or [tokens][dim]) rotated at its position (positions "
                "[tokens] int64, a StepBatch's or RoundBatch's positions_t) on their first rotary_dim values, the rest "
                "passed through; freqs [rotary_dim / 2] FP32 inverse frequencies (angle fp32(position) x freq in FP32) "
                "or a [positions][rotary_dim] FP32 table (cos of every pair, then sin); style half (pairs j, j + "
                "rotary_dim / 2) or interleaved (2j, 2j + 1); scale multiplies cos and sin; out may be x (in place)",
        "silu_mul": "silu_mul(x, up=None, layout='halves', out_dtype=None, out=None) -> y: silu(gate) * up element by "
                    "element: x the gates and up the values, or x [..., 2n] both (layout halves [gate | up], "
                    "interleaved gate at even and up at odd columns) and y [..., n]",
        "gelu_mul": "gelu_mul(x, up=None, layout='halves', approximate='none', out_dtype=None, out=None) -> y: as "
                    "silu_mul with gelu (approximate none: erf; tanh)",
        "sigmoid_mul": "sigmoid_mul(x, gate, out_dtype=None, out=None) -> y: x * sigmoid(gate) element by element",
        "l2_norm": "l2_norm(x, eps=1e-6, out_dtype=None, out=None) -> y: x / sqrt(sum(x^2) + eps) per last-dimension "
                   "row (each head of a query or key)",
        "stream_sum": "stream_sum(x, gate=None, activation='none', scale=1.0, out_dtype=None, out=None) -> y: y[t] = "
                      "scale * the sum over streams in stream order of activation(gate[t][s]) * x[t][s], x [tokens]"
                      "[streams][dim], gate x's shape (one per value) or [tokens][streams] (one per stream) or None "
                      "(the plain sum; scale 1 / streams for their mean), activation none (the gate itself), silu, "
                      "gelu, gelu_tanh or sigmoid; y [tokens][dim]: several residual streams mixed into one row (a "
                      "per-token streams-to-streams mix, S rows from S streams, is S calls, each into its stream's "
                      "out= view)",
        "stream_add": "stream_add(streams, x, weights=None) -> streams: streams[t][s] += weights[t][s] * x[t] for every "
                      "stream, in place in the streams' dtype (streams [tokens][streams][dim], x [tokens][dim], weights "
                      "[tokens][streams] or None for 1): one row added into every residual stream",
        "softplus_decay": "softplus_decay(x, bias, rate, gate=None, out_dtype=None, out=None, gate_out=None) -> decay, "
                          "or (decay, beta) with gate: decay = -rate * softplus(x + bias) (x [tokens][n], any row "
                          "stride: a slice of a projection's output; bias and rate [n]), the log-decay a gated delta "
                          "or state-space layer discretizes (rate exp(A_log)); beta = sigmoid(gate) (x's shape); "
                          "out_dtype FP32 by default",
    },
}


# A kv_cache plugin's pool, which another slot's plugin (attention) reads: its methods and attributes. A pool of
# another kv_cache plugin serves any attention plugin that keeps to them.
KV_POOL = {
    "open": "open(sequence): a new sequence, no KV",
    "close": "close(sequence): the sequence and its KV go",
    "fork": "fork(sequence, new): a new sequence sharing the sequence's valid KV (copied when either writes into it)",
    "length": "length(sequence) -> its valid positions",
    "reserve": "reserve(batch) -> slots: the KV slot of each of the StepBatch's tokens at its position ([tokens] int64)",
    "write": "write(layer, slots, keys, values): a layer's keys [tokens][kv_heads][head_dim] and values "
             "[tokens][kv_heads][value_dim] at their slots; a slot below 0 is skipped (no KV a sequence reads "
             "changes)",
    "keep": "keep(sequence, length): the valid KV after a step (a verify's accepted rows kept, the rest rolled back)",
    "gather": "gather(layer, sequence, count) -> (keys, values): the sequence's first count positions of a layer, "
              "keys [count][kv_heads][head_dim] and values [count][kv_heads][value_dim] in the compute dtype",
}
KV_POOL_ATTRIBUTES = ("kv_heads", "head_dim", "value_dim")

# Device forms a plugin offers beyond its slot's, each named by the fact it provides: fact -> {method: what it takes and
# returns}. A plugin that provides one of these facts has its methods (PluginTestCase.assertImplements checks them, and
# Core's loader refuses a plugin that provides the fact without them), and a plugin that needs the fact may call them.
# The methods of a "rounds:<slot>" fact live on the object the slot's device form returns where it returns one (the KV
# pool of a kv_cache plugin, the state pool of a recurrence plugin) and on the plugin otherwise.
DEVICE_EXTENSIONS = {
    # Captured decode rounds (crit_plugins.rounds): the slot's device form takes a RoundBatch capture-safely, and its
    # per-sequence state follows the round's slots. A plugin whose configuration can keep it out of a captured round
    # (a device expert cache over a file tier, a table read through Python) also has round_problems() -> [reason, ...]
    # ([] when it can be captured, no arguments), which the round driver asks before any capture (round_problems):
    # a reason refuses the round, and the model decodes through step().
    # A pool of a rounds:<slot> fact with a fixed number of round slots says how many as round_rows (an int);
    # crit_plugins.rounds.Rounds refuses rounds of more rows when it is built.
    "rounds:kv_cache": {
        "bind": "bind(slot, sequence, capacity): round slot `slot` holds `sequence`; its pages are reserved up to "
                "`capacity` positions, so a captured round never allocates",
        "unbind": "unbind(slot): the slot holds no sequence",
        "commit_round": "commit_round(view): each bound slot's valid length becomes its frontier after the round "
                        "(view.frontier), on the device",
    },
    "rounds:attention": {},
    # Attention over chosen KV blocks, draft trees and log-sum-exps: forward's keywords (DEVICE_INTERFACES["attention"]).
    "attention:blocks": {},
    "attention:tree": {},
    "attention:lse": {},
    # A KV pool's static FP8 scales per layer (a checkpoint's k_scale and v_scale), on the pool.
    "kv:scales": {
        "set_scales": "set_scales(layer, key_scale, value_scale): the layer's stored = value / scale for its keys "
                      "and values (an fp8 pool), before the layer's first write",
    },
    # Rows at chosen positions, on the pool: a pool of keys only (pool(..., value_dim=0): an indexer's keys) and a
    # per-block row cache (a pool whose positions are block numbers: pooled block keys written once per block).
    "kv:keys": {
        "rows_at": "rows_at(layer, slots) -> (keys, values): the rows at KV slots (int64, any shape), FP32 "
                   "[..., kv_heads, dim] each (values [..., kv_heads, 0] in a pool of keys only); device work only",
        "slots_at": "slots_at(round_slots, positions) -> slots: the KV slots of `positions` of the sequences bound to "
                    "`round_slots` (both int64, one shape; a position outside its slot's capacity gets a slot no "
                    "sequence reads), for writing and reading rows inside a captured round; device work only",
    },
    "rounds:recurrence": {
        "bind": "bind(slot, sequence): round slot `slot` holds `sequence`'s state",
        "unbind": "unbind(slot): the slot holds no sequence",
        "move": "move(source, target): the sequence of round slot `source` moves to slot `target`",
        "commit_round": "commit_round(view): each slot's state after column view.n_emitted - 1 of the round (its conv "
                        "window too), selected on the device; again with smaller counts before the rows' next round "
                        "(a truncation) and before a flush (flush, state, window, fork), which makes a deferred commit "
                        "final",
    },
    "rounds:sampler": {
        "reserve": "reserve(rows, widest, vocab, device, logprobs=0): every buffer of rounds of up to `rows` rows at up "
                   "to `widest` drafts over `vocab` logits (with `logprobs` ranked alternatives), once, before any "
                   "stage or capture; the rules staged so far are kept",
        "stage": "stage(slot, params): the sampling rules of the request in round slot `slot` (select()'s params), "
                 "kept in device buffers the round reads; its draws are at the round's output positions "
                 "(view.out_count + column)",
        "laws": "laws(rows, widest) -> the laws a sampled round's drafts were drawn from (the driver's "
                "view.draft_laws): a draft without one (a copy) is verified by the target's own draw",
        "draft_device": "draft_device(view, logits, depth, targets=None) -> tokens: each row's draft at `depth` from "
                        "its draft logits ([rows][vocab] FP32): the argmax for greedy rows, a draw from the row's "
                        "processed law with Core's `draft` purpose for sampled ones, its law kept in view.draft_laws; "
                        "targets maps a draft vocabulary's columns to target ids; capture-safe",
        "predictions_device": "predictions_device(view, logits, draft_laws=None) -> predictions: the token each verify "
                              "column of the round predicts ([rows][width + 1], int64), logits [rows][width + 1][vocab] "
                              "FP32: the argmax for greedy rows, rejection sampling against draft_laws (the laws "
                              "draft_device kept; a draft without a law, or whose law gives it no mass, takes the "
                              "target's own draw) for sampled rows, by Core's law and draws as predictions(), "
                              "capture-safe",
        "logprobs_device": "logprobs_device(view, logits, tokens): the log-probabilities of the round's emitted tokens "
                           "[rows][width + 1] over the logits rows that predicted them, in the arithmetic of the "
                           "model's step and score paths (torch's FP32 log_softmax, as the driver's own), into "
                           "view.logprobs (and view.top_ids / view.top_logprobs in Core's order), capture-safe",
    },
    "rounds:embeddings": {},
    "rounds:experts": {},
    "rounds:router": {},
    "rounds:projections": {},
    # The norms device form inside a captured round: every method capture-safe on fixed shapes (positions_t of a
    # RoundBatch at its fixed address); a shape's first call may compile, so the model calls each shape before it
    # captures.
    "rounds:norms": {},
    # Fused forms of the norms slot's own calls where a layer runs them back to back: one launch where the calls take
    # two or more, each output the values the separate calls give (the norms slot's conformance checks they agree).
    "norms:fused": {
        "rms_norm_rope_pair": "rms_norm_rope_pair(q, k, q_weight, k_weight, eps, positions, freqs, offset=0.0, "
                              "rotary_dim=None, style='half', scale=1.0, out_dtype=None, q_out=None, k_out=None) -> "
                              "(q, k): rms_norm_rope of a layer's queries [tokens][q_heads][dim] and keys "
                              "[tokens][k_heads][dim] (each its own weight, both or neither) at the same positions",
        "rope_pair": "rope_pair(q, k, positions, freqs, rotary_dim=None, style='half', scale=1.0, out_dtype=None, "
                     "q_out=None, k_out=None) -> (q, k): rope of a layer's queries and keys at the same positions",
        "stream_add_norm": "stream_add_norm(streams, x, weights, weight, eps, offset=0.0, out_dtype=None, out=None) -> "
                           "(y, streams): stream_add(streams, x, weights), then rms_norm of every stream as stored "
                           "(weight [dim], [streams][dim] or None; y [tokens][streams][dim], the streams' dtype by "
                           "default)",
    },
    # Prompt chunks as captured graphs (crit_plugins.prompt_graphs): the slot's prompt form takes a PromptBatch
    # capture-safely: it reads the batch's device tensors only, its launches are fixed for the bucket whatever the
    # sequence's context, and its padding (the tokens past valid_t's real ones) changes no state: no KV but at scratch,
    # no recurrent step, no routed expert read. A slot whose plugin keeps per-sequence state stages the sequence into
    # its fixed buffers before each replay and settles it after; the methods live on the object the slot's device
    # form returns (the KV pool, the state pool) and on the plugin otherwise.
    "prompts:kv_cache": {
        "stage_prompt": "stage_prompt(batch): the PromptBatch's sequence (batch.sequence, None: padding only) in the "
                        "pool's fixed prompt buffers: its pages reserved to batch.frontier + batch.count, its real "
                        "tokens' KV slots, the padding's at scratch; outside the graph, before each replay",
        "commit_prompt": "commit_prompt(batch): the sequence's valid length becomes batch.frontier + batch.count; "
                         "outside the graph, after the replay",
    },
    "prompts:attention": {},
    "prompts:recurrence": {
        "stage_prompt": "stage_prompt(batch): the sequence's state row in the pool's fixed prompt buffers (scratch "
                        "for None), so the chunk's forward reads and writes it on the device; before each replay",
        "commit_prompt": "commit_prompt(batch): the sequence's state now covers batch.frontier + batch.count "
                         "positions (its conv window the last real inputs); after the replay",
    },
    "prompts:embeddings": {},
    "prompts:experts": {},
    "prompts:router": {},
    "prompts:projections": {},
    "prompts:norms": {},
    # Experts as bundles in a slot store (a device expert cache's slots, a host or file store). The plugin's layout() is
    # the one definition of a bundle: store builders and the CPU expert tier call it, none keeps a copy.
    "experts:bundles": {
        "layout": "layout(format, hidden, inter) -> {bundle_bytes, format, hidden, inter, components: {name: (offset, "
                  "dtype, shape)}}: one expert's gate/up and down matrices in `format` as one contiguous block, "
                  "components 16-byte aligned, matrices row major as stored (the plugin documents each component)",
        "pack": "pack(gate_up, down, format, layer=None) -> uint8 tensor [bundle_bytes]: an expert's bundle (load()'s "
                "arguments), refused as load() refuses it; with layer, the bundle declares that layer's activations as "
                "a load does, and the stores forward_bundles reads for the layer must match",
        "forward_bundles": "forward_bundles(layer, x, ids, weights, store, slots, format) -> out: forward() over "
                           "bundles: store a uint8 device tensor [S][bundle_bytes] of `format` bundles, slots an int32 "
                           "device tensor [experts] naming each expert's slot (>= 0; -2 another tier's, skipped; -1 not "
                           "routed) or None (expert e in slot e); capture-safe on a fixed token count",
    },
    # A layer's routes as rows, so another tier's rows join a token's sum in route order (a CPU tier's routes get the
    # same bits as the device's whatever the split).
    "experts:routes": {
        "route_outputs": "route_outputs(layer, x, ids, store, slots, format) -> y: each route's unweighted output row "
                         "([tokens * top][hidden] FP32, route t * top + j), the rows of routes with id -1 or a negative "
                         "slot left as they were; a workspace, valid until the next call of the shape",
        "combine": "combine(y, ids, weights, slots=None) -> out: [tokens][hidden] FP32, each token's sum = fmaf(weight, "
                   "row, sum) over its routes in route order with id >= 0 (and a slot >= 0 when slots are given); "
                   "forward_bundles is combine(route_outputs(...), ids, weights, slots)",
    },
    # A device expert cache in the expert_residency slot.
    "experts:cache": {
        "cache": "cache(layers, experts, layout, slots, device, source, lend=None) -> ExpertCache: store (uint8 [S]"
                 "[bundle_bytes] slot memory), plan(layer, ids, rows=None, decided=None) -> slots ([experts] int32: "
                 "admits the routed misses by its policy and fills them from the source, capture-safe), prompt(layer, "
                 "ids) -> (store, slots) for prompt chunks (streamed, never admits) and end_prompt() after a prompt "
                 "step, start_round(view) and count_round(view) for captured rounds (crit_plugins.rounds), "
                 "telemetry() -> {slots, resident, lent, reads, hits, fills, fill_bytes} without waiting for the device",
    },
    # Bundles in host memory or files, which a device expert cache fills its slots from.
    "experts:store": {
        "open_store": "open_store(layout, layers, experts, device, path=None, identity=None) -> a store of the "
                      "experts plugin's slot layout with layout, keys, bundle_bytes, mapped and compressed: "
                      "put(layer, expert, bundle, compressed=None) of a slot-layout bundle, read(keys, into); a mapped "
                      "store also region() -> (host tensor, device tensor over its device address) and "
                      "address(layer, expert); a store file at path is bound to the checkpoint's identity. The "
                      "plugin's csf_scales option keeps scale components CSF-compressed: store.layout is then the "
                      "store's own (a fill program the expert cache runs to rebuild the slot layout), store.compressed "
                      "names them, and put() encodes them from the bundle or takes the checkpoint's as compressed "
                      "{component: (fixed, exceptions), or a list of such parts stacked by rows}",
    },
    # A drafter that drafts inside a captured round, from the model's device draft head (draft:head-device).
    "draft:device": {
        "draft": "draft(view, head, sampler) -> None: view.drafts[:, :width] and, for sampled rows, view.draft_laws "
                 "([rows][width][vocab], the law each draft was drawn from; one-hot for a greedy or copied draft), "
                 "capture-safe",
        "commit_round": "commit_round(view): the drafter's own state keeps the round's accepted prefix",
    },
    # A verifier that accepts inside a captured round.
    "verify:device": {
        "accept_device": "accept_device(view) -> None: each row's emitted tokens from view.predictions and "
                         "view.drafts (view.n_emitted, view.n_matched, view.emitted), ending after a stop token and "
                         "within the row's allowance, capture-safe",
    },
    # Lookup-table rows kept in host memory or files, staged to the device.
    "embeddings:host": {
        "open": "open(source, width, dtype, device) -> a table (lookup(ids), prefetch(ids), close()): the table in "
                "host memory or files (source: a CPU tensor of rows, a file path, a list of paths each holding the "
                "next rows, or {paths, offsets, rows}); store(rows) also returns one",
        "prefetch": "prefetch(ids): host ids known ahead (the next prompt chunk, a round's tokens), read for the last "
                    "table's lookup",
    },
}


def _rank_of(value):
    """The order-preserving int32 rank of an FP32 value (its bits, negative ones mirrored), as topk_blocks ranks."""
    import struct
    bits = struct.unpack("<i", struct.pack("<f", value))[0]
    return bits ^ 0x7FFFFFFF if bits < 0 else bits


_NEGATIVE_INFINITY = _rank_of(float("-inf"))   # a score at or below it may not be read


def topk_blocks(scores, k, positions=None, block_size=None):
    """The KV blocks each query reads under block-sparse attention (DEVICE_INTERFACES["attention"] forward's blocks,
    fact attention:blocks), chosen canonically from an indexer's scores: scores [tokens][blocks] (FP32 or wider; -inf
    for a block the token may not read, NaN read as -inf) -> ids [tokens][k] int32, each token's k highest-scoring
    blocks, ties going to the lower block id, listed in ascending id order, -1 past the blocks it may read (fewer
    than k scores above -inf). The same scores give the same ids on any device and in any batch: each score of 32 bits
    or fewer and its block id become one distinct int64 key, so the order torch.topk leaves equal values in never
    matters; wider scores (FP64) are compared as they are, by a stable sort (no rounding to FP32 makes new ties).
    Device work only, so a captured round selects with it.

    With positions ([tokens] int64, each query's position) and block_size (positions per block: an indexer's own
    block, not the KV page), only the blocks complete at the query's position compete, and the incomplete block
    holding the position (its tail) is always read: ids [tokens][k + 1], the tail listed after the chosen blocks in
    id order (-1 where the position ends a block). A query with k or fewer complete blocks so reads every position up
    to its own: plain causal attention below the budget."""
    import torch
    if positions is not None:
        if not block_size or block_size < 1:
            raise ValueError("selecting with positions takes the blocks' size in positions (block_size)")
        complete = (positions.long() + 1) // block_size
        ids = torch.arange(scores.shape[1], device=scores.device)
        open_ = ids[None, :] >= complete[:, None]
        chosen = topk_blocks(scores.masked_fill(open_, float("-inf")), k)
        tail = torch.where((positions.long() + 1) % block_size != 0, positions.long() // block_size, -1)
        both = torch.cat([chosen.long(), tail[:, None]], dim=1)
        big = torch.iinfo(torch.int64).max
        both = torch.where(both < 0, big, both).sort(dim=-1).values
        return torch.where(both == big, -1, both).to(torch.int32)
    tokens, blocks = scores.shape
    count = min(k, blocks)
    if scores.dtype == torch.float64:
        x = torch.where(torch.isnan(scores), torch.full_like(scores, float("-inf")), scores)
        # Descending and stable: equal scores keep their ascending ids, so the lower id goes first.
        top, order = torch.sort(x, dim=-1, descending=True, stable=True)
        ids = torch.where(top[:, :count] > float("-inf"), order[:, :count], blocks)
    else:
        x = scores.float()      # 16-bit scores become FP32 exactly
        x = torch.where(torch.isnan(x), torch.full_like(x, float("-inf")), x) + 0.0     # -0.0 ranks as +0.0
        bits = x.view(torch.int32)
        rank = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).long()                   # monotone in the score
        low = blocks - 1 - torch.arange(blocks, device=scores.device)                   # the lower id wins a tie
        kept = (rank * (1 << 32) + low).topk(count, dim=-1).values
        ids = torch.where((kept >> 32) > _NEGATIVE_INFINITY, blocks - 1 - (kept & 0xFFFFFFFF), blocks)
    ids = ids.sort(dim=-1).values
    ids = torch.where(ids == blocks, -1, ids).to(torch.int32)
    if k > blocks:
        ids = torch.cat([ids, ids.new_full((tokens, k - blocks), -1)], dim=1)
    return ids


# The DEVICE_EXTENSIONS facts whose methods live on an object a slot's device form returns: fact -> that slot (its
# pool). The assembly cannot see them before the model plugin makes one; PluginTestCase.assertImplements checks them.
POOL_EXTENSIONS = {"rounds:kv_cache": "kv_cache", "rounds:recurrence": "recurrence", "prompts:kv_cache": "kv_cache",
                   "prompts:recurrence": "recurrence", "kv:scales": "kv_cache", "kv:keys": "kv_cache"}


def capturing():
    """Whether the current CUDA stream is capturing a graph: False where torch is not loaded or no CUDA device is
    (torch.cuda.is_current_stream_capturing raises without one), so device forms that also run on the CPU can ask."""
    import sys
    torch = sys.modules.get("torch")
    return bool(torch is not None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def extension_missing(fact, plugin, made=()):
    """The DEVICE_EXTENSIONS methods of `fact` that neither `plugin` nor any object of `made` (what its device form
    returned: a pool) has."""
    holders = [plugin, *made]
    return [name for name in DEVICE_EXTENSIONS.get(fact, {})
            if not any(callable(getattr(holder, name, None)) for holder in holders)]


def pool_missing(pool):
    """The KV_POOL methods and attributes a kv_cache plugin's pool lacks."""
    return ([name for name in KV_POOL if not callable(getattr(pool, name, None))]
            + [name for name in KV_POOL_ATTRIBUTES if not isinstance(getattr(pool, name, None), int)])


def device_missing(slot, plugin):
    """The device-form methods of `slot` that `plugin` lacks (none for a slot without a device form)."""
    return [name for name in DEVICE_INTERFACES.get(slot, {}) if not callable(getattr(plugin, name, None))]


@dataclass
class StepBatch:
    """One step's tokens for the device plugins: `sequences` (one per row), `starts` (the first token of each row,
    rows + 1 entries, the last the token count) and `positions` (each token's position in its sequence), on the host
    as lists and on the device as tensors (`starts_t` int32, `positions_t` int64, `rows_t` int64: each token's row)."""
    sequences: list
    starts: list
    positions: list
    device: object = None
    starts_t: object = field(default=None, repr=False)
    positions_t: object = field(default=None, repr=False)
    rows_t: object = field(default=None, repr=False)

    @classmethod
    def build(cls, rows, device):
        """A batch from rows of (sequence, first position, token count), in the order the model packed them."""
        import torch
        sequences, starts, positions, row_of = [], [0], [], []
        for r, (sequence, first, count) in enumerate(rows):
            if count < 1 or first < 0:
                raise ValueError("a row is one or more tokens at non-negative positions")
            sequences.append(sequence)
            starts.append(starts[-1] + count)
            positions += range(first, first + count)
            row_of += [r] * count
        batch = cls(sequences, starts, positions, device)
        batch.starts_t = torch.tensor(starts, dtype=torch.int32, device=device)
        batch.positions_t = torch.tensor(positions, dtype=torch.int64, device=device)
        batch.rows_t = torch.tensor(row_of, dtype=torch.int64, device=device)
        return batch

    @property
    def rows(self):
        return len(self.sequences)

    @property
    def tokens(self):
        return self.starts[-1]

    def row(self, r):
        """(sequence, first token, token count, first position) of row r."""
        first = self.starts[r]
        return self.sequences[r], first, self.starts[r + 1] - first, self.positions[first]

    # A StepBatch is built on the host for one step; a RoundBatch (below) lives on the device for captured rounds and
    # a PromptBatch for captured prompt chunks.
    round = False
    prompt = False


class RoundBatch(StepBatch):
    """The tokens of a captured decode round (crit_plugins.rounds) at fixed device addresses: `rows` rows (the round's
    slots 0 .. rows - 1, packed at the front) of `width + 1` columns each, the pending token then the drafts, at positions
    frontier[r] + c. Its host lists (sequences, starts, positions) are None: a plugin reads the device tensors only
    (tokens_t, positions_t, rows_t, starts_t, slots_t, frontier_t), and follows a sequence through the round slot its
    pools bound it to (DEVICE_EXTENSIONS "rounds:<slot>"). fill() runs inside the captured round; every buffer is
    allocated once, for up to rows_max rows at up to widest drafts."""

    round = True

    def __init__(self, rows_max, widest, device):
        import torch
        if rows_max < 1 or widest < 0:
            raise ValueError("a round batch holds one row or more at a width of zero or more")
        cols = widest + 1
        long = dict(dtype=torch.int64, device=device)
        super().__init__(None, None, None, device)
        self.rows_max, self.widest, self.width, self._count = rows_max, widest, 0, 0
        self._tokens = torch.zeros(rows_max * cols, **long)
        self._positions = torch.zeros(rows_max * cols, **long)
        self._rows = torch.zeros(rows_max * cols, **long)
        self._starts = torch.zeros(rows_max + 1, dtype=torch.int32, device=device)
        self._row_index = torch.arange(rows_max, **long)[:, None].expand(rows_max, cols).contiguous()
        self._columns = torch.arange(cols, **long)
        self._start_index = torch.arange(rows_max + 1, dtype=torch.int32, device=device)
        self._slots = torch.arange(rows_max, **long)
        self.tokens_t = self.frontier_t = self.slots_t = None

    def fill(self, rows, width, frontier, columns):
        """The round of `rows` rows at `width` drafts: frontier [rows] (the position of each row's column 0) and
        columns [rows][width + 1] (its input tokens), both device tensors; capture-safe. Returns the batch."""
        if not 1 <= rows <= self.rows_max or not 0 <= width <= self.widest:
            raise ValueError(f"a round of 1..{self.rows_max} rows at width 0..{self.widest}")
        cols = width + 1
        n = rows * cols
        self._count, self.width = rows, width
        self.tokens_t = self._tokens[:n]
        self.tokens_t.view(rows, cols).copy_(columns[:rows, :cols])
        self.positions_t = self._positions[:n]
        self.positions_t.view(rows, cols).copy_(frontier[:rows, None] + self._columns[None, :cols])
        self.rows_t = self._rows[:n]
        self.rows_t.view(rows, cols).copy_(self._row_index[:rows, :cols])
        self.starts_t = self._starts[:rows + 1]
        self.starts_t.copy_(self._start_index[:rows + 1] * cols)
        self.frontier_t = frontier[:rows]
        self.slots_t = self._slots[:rows]
        return self

    @property
    def rows(self):
        return self._count

    @property
    def tokens(self):
        return self._count * (self.width + 1)

    def row(self, r):
        raise ValueError("a round batch has no host rows: read its device tensors, and a sequence through its round slot")


class PromptBatch(StepBatch):
    """One sequence's prompt chunk for a captured graph (crit_plugins.prompt_graphs) at fixed device addresses: a
    bucket of `bucket` tokens (one row), the chunk's `count` real tokens first, then padding (token 0), every token at
    position frontier + i (the padding's positions continue the chunk's). valid_t marks the real tokens, count_t /
    last_t / frontier_t hold the count, the last real token's index and the frontier on the device; a plugin's prompt
    form (DEVICE_EXTENSIONS "prompts:<slot>") reads those tensors only, and its padding changes no state. The host
    fields (sequence, frontier, count, and the StepBatch fields of the whole bucket) describe the chunk for the plugins'
    stage_prompt / commit_prompt, outside the graph.

    fill() writes the chunk's host values into pinned staging and moves them in one copy: count_t, last_t, frontier_t
    and tokens_t are views of one device buffer, side by side. derive() then makes positions_t and valid_t from them on
    the device; a captured chunk derives them inside its graph, so a replay needs that one copy only. Every buffer is
    allocated once, for buckets up to tokens_max; a bucket's tensors are the same views at the same addresses on every
    fill."""

    prompt = True
    HEAD = 3   # count, last, frontier: the device buffer's first words, the tokens after them

    def __init__(self, tokens_max, device):
        import torch
        if tokens_max < 1:
            raise ValueError("a prompt batch holds one token or more")
        self.torch = torch
        long = dict(dtype=torch.int64, device=device)
        super().__init__([None], [0, 0], range(0), device)
        self.tokens_max, self.bucket = tokens_max, 0
        self.sequence, self.frontier, self.count = None, 0, 0
        self._packed = torch.zeros(self.HEAD + tokens_max, **long)
        pinned = torch.device(device).type == "cuda"
        self._staging = torch.zeros(self.HEAD + tokens_max, dtype=torch.int64, pin_memory=pinned)
        self._host = self._staging.numpy()
        self._copied = torch.cuda.Event() if pinned else None   # the last copy out of the staging
        self.count_t, self.last_t, self.frontier_t = self._packed[0:1], self._packed[1:2], self._packed[2:3]
        self._positions, self._valid = torch.zeros(tokens_max, **long), torch.zeros(tokens_max, dtype=torch.bool,
                                                                                    device=device)
        self._rows, self._index = torch.zeros(tokens_max, **long), torch.arange(tokens_max, **long)
        self._views = {}
        self.tokens_t = self.valid_t = self._columns = None

    def _use(self, bucket):
        views = self._views.get(bucket)
        if views is None:
            views = self._views[bucket] = (
                self._packed[self.HEAD:self.HEAD + bucket], self._positions[:bucket], self._valid[:bucket],
                self._rows[:bucket], self.torch.tensor([0, bucket], dtype=self.torch.int32, device=self.device),
                self._index[:bucket], [0, bucket])
        (self.tokens_t, self.positions_t, self.valid_t, self.rows_t, self.starts_t, self._columns,
         self.starts) = views
        self.bucket = bucket

    def fill(self, bucket, sequence, frontier, tokens, derive=True):
        """The chunk `tokens` (host ints; empty: padding only, as a capture's warm-ups run) of `sequence` from position
        `frontier`, padded to `bucket` tokens: one copy from pinned staging (it waits for the previous copy out of it
        to finish first), then derive() unless `derive` is false (a captured chunk derives inside its graph). Outside
        the graph."""
        count = len(tokens)
        if not 1 <= bucket <= self.tokens_max or count > bucket or frontier < 0:
            raise ValueError(f"a chunk of {count} tokens in a bucket of {bucket} (at most {self.tokens_max})")
        if bucket != self.bucket:
            self._use(bucket)
        self.sequence, self.frontier, self.count = sequence, frontier, count
        self.sequences[0] = sequence
        self.positions = range(frontier, frontier + bucket)
        if self._copied is not None:
            self._copied.synchronize()
        host, head = self._host, self.HEAD
        host[0], host[1], host[2] = count, max(count - 1, 0), frontier
        host[head:head + count] = tokens
        host[head + count:head + bucket] = 0
        self._packed[:head + bucket].copy_(self._staging[:head + bucket], non_blocking=True)
        if self._copied is not None:
            self._copied.record()
        if derive:
            self.derive()
        return self

    def derive(self):
        """positions_t (frontier + i) and valid_t (i < count) from frontier_t and count_t, on the device; capture-safe."""
        self.torch.add(self._columns, self.frontier_t, out=self.positions_t)
        self.torch.lt(self._columns, self.count_t, out=self.valid_t)

    def row(self, r):
        if r != 0:
            raise IndexError("a prompt batch has one row")
        return self.sequence, 0, self.bucket, self.frontier


class Allowed:
    """The tokens a constrained request's row may take next (structured output, forced calls): Core's matcher of the
    request's constraint computes them before each step (crit_core.native), the row carries them as StepRow.allowed,
    and the model plugin hands them to the sampler as the row's params["allowed"], which masks every other token
    before its rules. Core checks each token a constrained row produces against them. `words` is a bitmask, bit t % 32
    of word t // 32 for token t (crit::core::TokenMatcher::allowed_bits), over `vocabulary` tokens. It reads as the
    sorted ids it allows (iteration, len, `in`), the host form's allowed set, and as a device mask (mask)."""

    __slots__ = ("words", "vocabulary", "_ids")

    def __init__(self, words, vocabulary):
        self.words = words if isinstance(words, array.array) and words.typecode == "I" else array.array("I", words)
        self.vocabulary = int(vocabulary)
        if len(self.words) != (self.vocabulary + 31) // 32:
            raise ValueError(f"an allowed set over {self.vocabulary} tokens is {(self.vocabulary + 31) // 32} words")
        self._ids = None

    @classmethod
    def of(cls, tokens, vocabulary):
        """The allowed set of exactly `tokens`."""
        words = array.array("I", bytes(4 * ((int(vocabulary) + 31) // 32)))
        for token in tokens:
            if not 0 <= token < vocabulary:
                raise ValueError(f"token {token} is outside the vocabulary of {vocabulary}")
            words[token // 32] |= 1 << (token % 32)
        return cls(words, vocabulary)

    def ids(self):
        """The allowed token ids, ascending."""
        if self._ids is None:
            found = []
            for w, word in enumerate(self.words):
                while word:
                    low = word & -word
                    found.append(32 * w + low.bit_length() - 1)
                    word ^= low
            self._ids = found
        return self._ids

    def __iter__(self):
        return iter(self.ids())

    def __len__(self):
        return len(self.ids())

    def __bool__(self):
        return any(self.words)

    def __contains__(self, token):
        return 0 <= token < self.vocabulary and bool(self.words[token // 32] >> (token % 32) & 1)

    def mask(self, device, width=None):
        """A bool tensor of `width` entries (the vocabulary by default) on `device`, True for the tokens allowed; the
        columns past the vocabulary are never allowed."""
        import torch
        width = self.vocabulary if width is None else int(width)
        if width < self.vocabulary and any(t >= width for t in self.ids()):
            raise ValueError(f"the allowed set has tokens past the {width} columns of the logits")
        words = torch.frombuffer(bytearray(self.words.tobytes()), dtype=torch.int32).to(device)
        bits = torch.arange(32, dtype=torch.int32, device=words.device)
        found = ((words.unsqueeze(1) >> bits) & 1).flatten().bool()
        if found.numel() >= width:
            return found[:width]
        return torch.cat([found, torch.zeros(width - found.numel(), dtype=torch.bool, device=found.device)])

    def __repr__(self):
        return f"Allowed({len(self)} of {self.vocabulary})"
