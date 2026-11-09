"""tree.py's draft tree and accept walk for every row of a captured decode round, in torch on the device: the
draft-trees plugin's "draft:device" and "verify:device" extensions.

Drafting (tree.draft): the root's `branch` best tokens by (logit descending, id ascending) with their log-softmax
scores, then for each newer depth the `beam` best candidates of the newest depth by (score descending, candidate index
ascending) expanded alike, and the `nodes - 1` best candidates of every depth kept; nodes ordered by depth, then rank
(tree.build), so parents come first. The model's device draft head drafts the frontier of every row at once:

    head.logits(depth, tokens [rows][F], view, parents=[rows][F]) -> [rows][F][vocab]    (draft:head-device, with parents)

where parents[r][f] is the frontier entry's parent node among the previous depth's frontier (-1 for the root's
children), so the head can attend along each entry's path. A round's tree lands in view.drafts[:, :width] (node i's
token at column i - 1), view.tree_parents [rows][width] (node i's parent node) and view.tree_depths [rows][width + 1],
the verifier's own buffers (TreeBuffers); with fewer candidates than nodes the rest are PAD nodes under the root, which
no prediction matches.

The target verifies the nodes in one pass at positions frontier + depth under ancestor masks (attention:tree, the
attention slot's); each node's prediction is its argmax, or in a sampled round the target's own draw at output position
out_count + depth (token-sampling's predictions_device reads view.tree_depths), so the walk emits what plain sampling
draws.

Accept (tree.accept): from the root, move to the child whose token equals the node's prediction (siblings are distinct
top-k tokens, so the accepted nodes form one path), emitting each prediction, until no child matches; ending after a
stop token and within the allowance, a held row (view.hold) emitting nothing. It writes view.n_emitted, view.n_matched
(accepted nodes kept materialized), view.emitted and view.tree_path ([rows][width + 1]: node 0, then the accepted nodes
by depth, -1 beyond: the rows a KV commit keeps, and the logits rows the driver reads each emitted token's
log-probability from), and rewrites view.predictions[:, :width + 1] in path order (column j the prediction of node
path[j], the emitted token j), so what the driver reads by column after the accept (a truncation, the result block) is
the path's; capture-safe. Logits are assumed finite (a -inf candidate the host form drops is kept here).
"""
import torch

PAD = -1
WORD = 0xFFFFFFFF


def ranked_top(values, k):
    """(ids, values) of each row's k best ([..., vocab]) by (value descending, id ascending), exact on ties: a
    composite of the value's order bits and the id; NaN ranks as -inf."""
    x = torch.nan_to_num(values.float(), nan=float("-inf")).contiguous()
    bits = x.view(torch.int32).long()
    key = torch.where(bits >= 0, bits | 0x80000000, (~bits) & WORD)
    ids = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64)
    picked = ((key << 31) | (0x7FFFFFFF - ids)).topk(k, dim=-1).values
    found = 0x7FFFFFFF - (picked & 0x7FFFFFFF)
    return found, x.gather(-1, found)


def rank_scores(scores, k):
    """Candidate indices of the k best scores per row ([rows][C] FP64) by (score descending, index ascending)."""
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :k]


def frontier_sizes(depth, branch, beam):
    """The frontier width at each depth and the candidates each depth adds (fixed by the options)."""
    sizes, added, f = [], [], 1
    for _ in range(depth):
        sizes.append(f)
        added.append(f * branch)
        f = min(beam, f * branch)
    return sizes, added


class TreeBuffers:
    """A tree round's own buffers for up to `rows` rows at up to `widest` drafts (nodes - 1), allocated once."""

    def __init__(self, rows, widest, device):
        long = dict(dtype=torch.int64, device=device)
        self.parents = torch.zeros(rows, max(widest, 1), **long)
        self.depths = torch.zeros(rows, widest + 1, **long)
        self.path = torch.full((rows, widest + 1), -1, **long)
        self.predictions = torch.zeros(rows, widest + 1, **long)


def draft_tree(view, head, depth, branch, beam, buffers):
    """The round's tree for every row (view.width drafts = nodes - 1) into view.drafts and the buffers' parents and
    depths (view.tree_parents, view.tree_depths)."""
    rows, width = view.rows, view.width
    device = view.drafts.device
    long = dict(dtype=torch.int64, device=device)
    head.catch_up(view)
    view.tree_parents, view.tree_depths = buffers.parents[:rows, :width], buffers.depths[:rows, :width + 1]
    if not width:
        buffers.depths[:rows, :1].zero_()
        return
    depth = min(depth, width)
    sizes, added = frontier_sizes(depth, branch, beam)
    total = sum(added)
    tokens = torch.zeros(rows, total, **long)
    parent = torch.full((rows, total), -1, **long)
    score = torch.zeros(rows, total, dtype=torch.float64, device=device)
    level = torch.zeros(rows, total, **long)
    frontier = torch.full((rows, 1), -1, **long)            # candidate indices, -1 the root
    frontier_tokens = view.pending[:rows, None]
    frontier_parents = torch.full((rows, 1), -1, **long)
    at = 0
    for d in range(depth):
        logits = head.logits(d, frontier_tokens, view, parents=frontier_parents)       # [rows][F][vocab]
        ids, values = ranked_top(logits, branch)
        lse = torch.logsumexp(torch.nan_to_num(logits.double(), nan=float("-inf")), dim=-1, keepdim=True)
        base = torch.where(frontier >= 0, score.gather(1, frontier.clamp(min=0)), torch.zeros_like(score[:, :1]))
        f = sizes[d]
        new = slice(at, at + f * branch)
        tokens[:, new] = ids.reshape(rows, -1)
        parent[:, new] = frontier[:, :, None].expand(rows, f, branch).reshape(rows, -1)
        score[:, new] = (base[:, :, None] + (values.double() - lse)).reshape(rows, -1)
        level[:, new] = d + 1
        if d + 1 < depth:
            pick = rank_scores(score[:, new], sizes[d + 1])
            frontier = pick + at
            frontier_tokens = tokens.gather(1, frontier)
            # The head's parent of each new frontier entry: its index in the previous frontier.
            frontier_parents = pick // branch
        at += f * branch
    kept = rank_scores(score, min(width, total))                                       # [rows][K] rank order
    count = kept.shape[1]
    # Nodes by (depth, rank): node i + 1 is the i-th kept candidate in that order.
    order = torch.sort(level.gather(1, kept) * (count + 1) + torch.arange(count, **long)[None, :], dim=1).indices
    nodes = kept.gather(1, order)                                                      # candidate of node i + 1
    node_of = torch.zeros(rows, total, **long).scatter_(1, nodes, torch.arange(1, count + 1, **long)[None, :]
                                                        .expand(rows, count).contiguous())
    cand_parent = parent.gather(1, nodes)
    parent_node = torch.where(cand_parent >= 0, node_of.gather(1, cand_parent.clamp(min=0)),
                              torch.zeros_like(cand_parent))
    view.drafts[:rows, :width].fill_(PAD)
    view.drafts[:rows, :count] = tokens.gather(1, nodes)
    parents, depths = buffers.parents[:rows, :width], buffers.depths[:rows, :width + 1]
    parents.zero_()
    parents[:, :count] = parent_node
    depths.fill_(1)
    depths[:, 0] = 0
    depths[:, 1:count + 1] = level.gather(1, nodes)
    view.tree_parents, view.tree_depths = parents, depths


def accept_tree(view, buffers, levels=None):
    """tree.accept's walk for every row: writes view.n_emitted, view.n_matched, view.emitted, view.tree_path (the
    buffers' path) and view.predictions in path order; returns n_emitted. levels: the tree's deepest depth (the width by
    default)."""
    rows, width = view.rows, view.width
    nodes = width + 1
    predictions = view.predictions[:rows, :nodes]
    device = predictions.device
    long = dict(dtype=torch.int64, device=device)
    column = torch.arange(nodes, **long)[None, :]
    tokens = torch.cat([view.pending[:rows, None], view.drafts[:rows, :width]], dim=1)
    if width:
        parents = torch.cat([torch.full((rows, 1), -1, **long), view.tree_parents[:rows, :width]], dim=1)
        depths = view.tree_depths[:rows, :nodes]
        # A node stands when its token is its parent's prediction and its parent stands (the root stands).
        good = (column == 0).expand(rows, nodes)
        ok = (tokens == predictions.gather(1, parents.clamp(min=0))) & (tokens >= 0) & (column > 0)
        for _ in range(int(min(width if levels is None else levels, width))):
            good = (column == 0) | (ok & good.gather(1, parents.clamp(min=0)))
        reached = torch.where(good, depths, torch.zeros_like(depths)).max(1).values      # accepted nodes
        # The accepted node at each depth (one per depth: siblings hold distinct tokens), column `nodes` a sink.
        path = torch.full((rows, nodes + 1), -1, **long)
        target = torch.where(good, depths, torch.full_like(depths, nodes))
        path.scatter_(1, target, torch.where(good, column.expand(rows, nodes), torch.full_like(target, -1)))
        path = path[:, :nodes]
    else:
        reached = torch.zeros(rows, **long)
        path = torch.where(column == 0, torch.zeros_like(column), torch.full_like(column, -1)).expand(rows, 1)
    along = predictions.gather(1, path.clamp(min=0))                                    # emitted j = pred[path[j]]
    stop = (along[:, :, None] == view.stops[:rows, None, :]).any(2) & (along >= 0) & (column <= reached[:, None])
    first_stop = torch.where(stop, column, torch.full_like(column, nodes)).min(1).values
    count = torch.minimum(torch.minimum(reached + 1, first_stop + 1), view.allowance[:rows]).clamp(min=0)
    active = (view.allowance[:rows] > 0) & (view.hold[:rows] == 0)
    count = torch.where(active, count, torch.zeros_like(count))
    view.n_emitted[:rows] = count
    view.n_matched[:rows] = torch.minimum(reached, (count - 1).clamp(min=0))
    view.emitted[:rows].fill_(-1)
    view.emitted[:rows, :nodes] = torch.where(column < count[:, None], along, torch.full_like(along, -1))
    buffers.path[:rows, :nodes] = path
    view.tree_path = buffers.path[:rows, :nodes]
    # Predictions in path order: column j is the prediction of node path[j], the emitted token j.
    buffers.predictions[:rows, :nodes] = along
    view.predictions[:rows, :nodes] = buffers.predictions[:rows, :nodes]
    return count
