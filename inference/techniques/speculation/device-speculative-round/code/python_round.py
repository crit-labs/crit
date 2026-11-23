"""Device speculative rounds in torch: the pieces of a round that belong to the technique, for engines whose rounds run
on Core's round driver (crit_plugins.rounds: SpeculativeRounds over an engine's own draft, verify and commit hooks, and
Rounds over the library's device forms, captured per (rows, width, mode) and launched by Core natively).

- rejection_predictions and draw: the predictions of a sampled verify under which the driver's chain accept is
  rejection sampling (accept a draft with probability min(1, p/q), else draw from max(p - q, 0); the last column draws
  from p), so a sampled round keeps the target's distribution; capture-safe.
- override_drafts and CopyDrafts: a copy drafter's tokens override a neural drafter's where the copy's match is long,
  their laws one-hot at the copied token for sampled rows; staged in pinned memory between rounds and applied inside a
  captured round.
"""
import torch


def draw(probabilities, uniforms):
    """A token from each distribution of `probabilities` ([..., vocabulary], non-negative, any scale) by inverse CDF
    with `uniforms` ([...] in [0, 1)): capture-safe, no host synchronization."""
    cdf = probabilities.cumsum(-1)
    target = (uniforms * cdf[..., -1]).unsqueeze(-1)
    return torch.searchsorted(cdf, target, right=True).squeeze(-1).clamp(max=probabilities.shape[-1] - 1)


def rejection_predictions(target, drafts, draft_probs, uniforms, residual_uniforms):
    """Predictions under which the greedy accept rule is rejection sampling, so sampled rounds keep the target's
    distribution. target: [rows, width + 1, vocabulary] the processed target distribution at each column (temperature
    and filters applied); drafts: [rows, width]; draft_probs: [rows, width, vocabulary] the drafter's distribution each
    draft came from (one-hot for a greedy drafter); uniforms: [rows, width] for the accept tests; residual_uniforms:
    [rows, width + 1] for the draws. Column j < width predicts its draft when u_j < p_j(d_j) / q_j(d_j), else a draw
    from max(p_j - q_j, 0) (which cannot be the draft, so acceptance stops there); the last column draws from p."""
    width = drafts.shape[1]
    predictions = draw(target, residual_uniforms)
    if width:
        p = target[:, :width].gather(2, drafts[..., None]).squeeze(2)
        q = draft_probs.gather(2, drafts[..., None]).squeeze(2)
        accept = uniforms * q < p
        residual = (target[:, :width] - draft_probs).clamp(min=0)
        # Where p <= q everywhere (the drafter's distribution is the target's) the residual is empty; then the accept
        # test always passes, and the draw is never used.
        residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, target[:, :width])
        rejected = draw(residual, residual_uniforms[:, :width])
        predictions[:, :width] = torch.where(accept, drafts, rejected)
    return predictions


def override_drafts(drafts, copies, counts, matched, threshold, draft_probs=None):
    """In place: where a row's copy match covers at least `threshold` tokens, its first counts[row] drafts become the
    copied tokens. drafts: [rows, width] a drafter's tokens; copies: [rows, >= width]; counts and matched: [rows].
    draft_probs ([rows, width, vocabulary], for sampled rows) becomes one-hot at each copied token, the law a copy was
    drawn from, so rejection_predictions stays exact; a draft kept after a shorter copy keeps its own law, which it was
    still drawn from. Capture-safe."""
    width = drafts.shape[1]
    if not width:
        return
    columns = torch.arange(width, device=drafts.device)
    use = (matched >= threshold)[:, None] & (columns[None, :] < counts[:, None])
    copied = copies[:, :width]
    drafts.copy_(torch.where(use, copied, drafts))
    if draft_probs is not None:
        one_hot = torch.zeros_like(draft_probs).scatter_(2, copied[..., None], 1.0)
        draft_probs.copy_(torch.where(use[..., None], one_hot, draft_probs))


class CopyDrafts:
    """A copy drafter's tokens for overriding a neural drafter's, staged on the host between rounds and applied on the
    device in the draft hook.

    Before each launch, set(slot, tokens, matched) for every decoding row from its copy proposal (mtp-decode's
    PromptCopyIndex.propose_matched over the row's accepted context), or clear(slot); the host buffers are pinned on a
    CUDA device. In the draft hook, after the drafter wrote view.drafts, apply(view) copies them to the device and
    overrides each row whose match covers at least `threshold` tokens (override_drafts); sampled rows pass their
    draft_probs. Every buffer is allocated once, so apply runs inside a captured round. The drafter catches up from
    the emitted tokens (view.catch_up_tokens) as always, never from an overridden draft."""

    def __init__(self, rows, widest, device, threshold):
        if rows < 1 or widest < 1 or threshold < 1:
            raise ValueError("at least one row, a width of one and a threshold of one")
        self.rows, self.widest, self.threshold = rows, widest, threshold
        device = torch.device(device)
        pinned = device.type == "cuda"
        self.host = torch.zeros(rows, widest + 2, dtype=torch.long, pin_memory=pinned)   # tokens, count, matched
        self.device = torch.zeros(rows, widest + 2, dtype=torch.long, device=device)

    def set(self, slot, tokens, matched):
        tokens = list(tokens)[:self.widest]
        row = self.host[slot]
        row.zero_()
        if tokens:
            row[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        row[self.widest], row[self.widest + 1] = len(tokens), int(matched)

    def clear(self, slot):
        self.host[slot].zero_()

    def apply(self, view, draft_probs=None):
        rows, width = view.rows, view.width
        self.device[:rows].copy_(self.host[:rows], non_blocking=True)
        staged = self.device[:rows]
        override_drafts(view.drafts[:, :width], staged[:, :self.widest], staged[:, self.widest],
                        staged[:, self.widest + 1], self.threshold, draft_probs)
