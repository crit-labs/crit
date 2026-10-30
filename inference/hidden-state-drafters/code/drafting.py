"""Host reference for drafters that read the target's hidden states (EAGLE-3) or draft a block per pass (DFlash,
DSpark): which target hidden states and tokens feed the drafter after a round, the block a block drafter fills,
draft-vocabulary mapping and confidence truncation. Every rule here has a kernel in hidden_drafters.cu.

Taps are residual-stream indices (the hidden state after k target decoder layers); crit_core.model.drafters reads
them, and every other drafter fact, from the published config formats.

**EAGLE-3 alignment.** The drafter's input at position p is the target's fused taps at p (the hidden states the
target computed while reading the token at p) with the embedding of the token at p + 1, and it predicts the token
at p + 2. A verify column c sits at position base + c and reads the round's pending token (c = 0) or draft c; its
taps are the target's at that position. After the round emits n tokens e_0 .. e_{n-1}, the positions base ..
base + n - 1 are materialized and the token after position base + c is e_c, so the drafter ingests the pairs
(taps of column c, e_c) for c < n: exactly the accepted columns, whatever was drafted. Its first new draft comes from
the last pair; later draft steps of the same round feed the drafter's own output hidden state instead of target taps.

**Block drafting.** A block drafter fills [anchor, mask x (block_size - 1)] at positions frontier .. frontier +
block_size - 1 in one non-causal pass (prompt attention with causal = false over the block, the target's
context injected through its taps), where the anchor is the pending token. Positions 1 .. block_size - 1 give
block_size - 1 drafts; with sample_from_anchor the anchor's row gives one more, first.
"""

PAD = -1


def eagle_inputs(emitted, width):
    """The drafter's catch-up columns after a round: (column, token) for the n emitted tokens, PAD beyond, over a
    fixed width so the drafter's launch shape never depends on n."""
    if len(emitted) > width:
        raise ValueError("more emitted tokens than columns")
    return [(c, emitted[c]) if c < len(emitted) else (c, PAD) for c in range(width)]


def block_input(anchor, mask_token, block_size):
    """The block a block drafter fills: the pending token, then mask placeholders."""
    if block_size < 2:
        raise ValueError("a block holds the anchor and at least one mask")
    return [anchor] + [mask_token] * (block_size - 1)


def block_drafts(row_tokens, sample_from_anchor=False):
    """Drafts from one pass's per-position choices (row_tokens[i] = the drafter's token at block position i)."""
    return list(row_tokens if sample_from_anchor else row_tokens[1:])


def map_tokens(ids, d2t):
    """Draft-vocabulary ids to target ids, EAGLE-3's offset table: target = draft + d2t[draft]; PAD stays."""
    return [i if i < 0 else i + d2t[i] for i in ids]


def scatter_law(q_draft, d2t, target_vocab):
    """A draft-vocabulary law as a target-vocabulary row (zeros elsewhere), for rejection sampling against the target."""
    out = [0.0] * target_vocab
    for d, p in enumerate(q_draft):
        out[d + d2t[d]] += p
    return out


def truncate(drafts, confidence, threshold):
    """Drafts from the first one whose confidence (a confidence head's estimate, or the draft's own probability) is
    below the threshold become PAD, which no prediction matches, so the round stops there without a narrower graph."""
    out, cut = [], False
    for d, c in zip(drafts, confidence):
        cut = cut or c < threshold
        out.append(PAD if cut else d)
    return out
