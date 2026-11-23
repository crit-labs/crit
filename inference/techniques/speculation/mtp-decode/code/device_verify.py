"""mtp-decode's verify (implementation.verify) for every row of a captured decode round at once, in torch on the
device: the verifier slot's "verify:device" extension (crit_plugins.device.DEVICE_EXTENSIONS).

A round of R rows at width W: row r's drafts view.drafts[r, :W] (-1: a cut draft, which no prediction matches) and the
target's prediction at each of its W + 1 columns, view.predictions[r, :W + 1]. The row emits its predictions while the
drafts match them, then the first prediction that differs (the correction, or the bonus after the last draft), ending
after the first predicted stop token (view.stops[r], -1 padded) and within its allowance (view.allowance[r], 0 for an
inactive row); a held row (view.hold) emits nothing:

    matched   = drafts equal to their column's prediction, from the first on
    n_emitted = min(matched + 1, first stop column + 1, allowance), 0 for a held row
    n_matched = min(matched, n_emitted - 1)         the drafts whose inputs stay materialized
    emitted   = predictions[:n_emitted], -1 beyond

The tokens are the host form's (chain-verifier's accept: mtp-decode's verify over the same drafts and predictions;
its `accepted`, the drafts emitted as tokens, is min(matched, n_emitted)). accept_chain writes only n_emitted, n_matched
and emitted: the round driver advances its own state from them and counts acceptance by depth (crit_plugins.rounds).
Fixed-shape torch work on the round's buffers: capture-safe, no host value inside.
"""
import torch


def accept_chain(view, cache=None):
    """The chain rule above for every row of the round (view.rows rows at view.width): view.n_emitted, view.n_matched
    and view.emitted. cache: a dict the caller keeps for the column indices. Returns n_emitted."""
    rows, width = view.rows, view.width
    predictions = view.predictions[:rows, :width + 1]
    device = predictions.device
    cache = {} if cache is None else cache
    key = (width + 1, str(device))
    if key not in cache:
        cache[key] = torch.arange(width + 1, dtype=torch.int64, device=device)
    columns = cache[key]
    if width:
        matched = (view.drafts[:rows, :width] == predictions[:, :width]).long().cumprod(1).sum(1)
    else:
        matched = torch.zeros(rows, dtype=torch.int64, device=device)
    stop = (predictions[:, :, None] == view.stops[:rows, None, :]).any(2) & (predictions >= 0)
    first_stop = torch.where(stop, columns[None, :], width + 1).min(1).values
    allowance = view.allowance[:rows]
    count = torch.minimum(torch.minimum(matched + 1, first_stop + 1), allowance).clamp(min=0)
    count = torch.where(view.hold[:rows] > 0, torch.zeros_like(count), count)
    view.n_matched[:rows] = torch.minimum(matched, (count - 1).clamp(min=0))
    view.n_emitted[:rows] = count
    kept = columns[None, :] < count[:, None]
    view.emitted[:rows].fill_(-1)
    view.emitted[:rows, :width + 1] = torch.where(kept, predictions, torch.full_like(predictions, -1))
    return count
