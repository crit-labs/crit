"""Token trees for speculative decoding: dynamic draft trees, their attention masks, verification and compaction.

A tree has N nodes in topological order. Node 0 is the root, the pending token whose successors the round predicts;
node i > 0 holds a draft token and its parent parents[i] < i. mask[i] has bit j set for every node j on the
root-to-i path (i included) and depth[i] = popcount(mask[i]) - 1. Verification feeds all N nodes in one target pass
with an ancestor-only attention mask at RoPE positions frontier + depth[i] (the engine's
attention kernel takes the masks); prediction[v] is the target's next token after the path to v: its argmax for a
greedy request, or its own draw at output position outputs + depth[v] for a sampled one (token-sampling's Rows
with `tree`). The walk starts at the root and moves to the child whose token equals the node's prediction, emitting
it, until no child matches; that prediction is then emitted as the correction (the bonus at a leaf). Every emitted
token is the target's own choice at its position, so a tree changes how many tokens a round emits, never which.

Drafting follows EAGLE-2: expand the root's best `branch` tokens, then for `depth - 1` more steps expand the
`beam` best candidates of the newest depth, scoring each candidate by its path's draft log-probability, and keep
the `nodes - 1` best candidates of all depths. A child never outscores its parent (log-probabilities are <= 0) and
ties go to the earlier candidate, so the kept set always contains every kept node's parent.

After acceptance the materialized inputs are the root and the first emitted - 1 accepted nodes, whose rows were
written at slots base + node; compaction moves them to base + 0, 1, ..., in order of position, which never
overwrites a row a later move reads (path nodes increase).
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    token: int
    parent: int      # candidate index of the parent, -1 for the root
    score: float     # draft log-probability of the root-to-candidate path
    depth: int       # 1 for the root's children


@dataclass(frozen=True)
class Tree:
    tokens: tuple    # node tokens, node 0 the root
    parents: tuple   # parents[i] for i >= 1; parents[0] = -1
    masks: tuple     # root-to-node bitmasks
    depths: tuple

    @property
    def nodes(self):
        return len(self.tokens)


def expand(row, branch):
    """The `branch` best tokens of a logit row by (logit desc, id asc) with their log-softmax values. NaN ranks as
    -inf; a row without a finite logit scores every token -inf."""
    if not row or branch < 1:
        raise ValueError("a nonempty row and a positive branch are required")
    row = [x if x == x else -math.inf for x in row]
    top = max(row)
    best = sorted(range(len(row)), key=lambda t: (-row[t], t))[:branch]
    if top == -math.inf:
        return [(t, -math.inf) for t in best]
    lse = top + math.log(math.fsum(math.exp(x - top) for x in row))
    return [(t, row[t] - lse) for t in best]


def rank(candidates, ids, m):
    """The m best candidate ids by (score desc, index asc). A candidate scoring -inf is never kept."""
    return [i for i in sorted(ids, key=lambda i: (-candidates[i].score, i)) if candidates[i].score > -math.inf][:m]


def path_tokens(candidates, index):
    out = []
    while index >= 0:
        out.append(candidates[index].token)
        index = candidates[index].parent
    return out[::-1]


def draft(drafter, root, depth, branch, beam, nodes):
    """Candidates and the kept tree. drafter(paths) returns one logit row per path (token lists after the root)."""
    if not 1 <= nodes <= 64 or depth < 1 or beam < 1:
        raise ValueError("1 <= nodes <= 64, depth >= 1 and beam >= 1 are required")
    candidates, frontier = [], [-1]
    for d in range(depth):
        rows = drafter([path_tokens(candidates, f) if f >= 0 else [] for f in frontier])
        newest = []
        for f, row in zip(frontier, rows):
            base = candidates[f].score if f >= 0 else 0.0
            for token, logp in expand(row, branch):
                newest.append(len(candidates))
                candidates.append(Candidate(token, f, base + logp, d + 1))
        frontier = rank(candidates, newest, beam)
    return candidates, build(root, candidates, rank(candidates, range(len(candidates)), nodes - 1))


def build(root, candidates, kept):
    """The tree of the kept candidate ids (rank order): nodes by depth, then rank, so parents come first."""
    order = sorted(range(len(kept)), key=lambda r: (candidates[kept[r]].depth, r))
    node_of = {kept[r]: i + 1 for i, r in enumerate(order)}
    tokens, parents, masks, depths = [root], [-1], [1], [0]
    for r in order:
        c = candidates[kept[r]]
        if c.parent >= 0 and c.parent not in node_of:
            raise ValueError("a kept candidate's parent was not kept")
        parent = node_of.get(c.parent, 0)
        tokens.append(c.token)
        parents.append(parent)
        masks.append(masks[parent] | (1 << len(masks)))
        depths.append(depths[parent] + 1)
    return Tree(tuple(tokens), tuple(parents), tuple(masks), tuple(depths))


def chain(root, drafts):
    """A chain is the tree whose node i has parent i - 1."""
    tokens = (root,) + tuple(drafts)
    return Tree(tokens, tuple(range(-1, len(drafts))), tuple((1 << (i + 1)) - 1 for i in range(len(tokens))),
                tuple(range(len(tokens))))


def accept(tree, predictions, remaining, stops=()):
    """(emitted tokens, path): the walk of the device tree accept kernel; path[0] = 0, then the accepted nodes."""
    if len(predictions) != tree.nodes:
        raise ValueError("one prediction per node")
    out, path, v = [], [0], 0
    while len(out) < remaining:
        t = predictions[v]
        child = next((i for i in range(1, tree.nodes) if tree.parents[i] == v and tree.tokens[i] == t), None)
        out.append(t)
        if child is None:
            break
        path.append(child)
        v = child
        if t in stops:
            break
    return out, path


def compaction(path, emitted):
    """Row moves (from node, to position) that place the materialized nodes path[0 .. emitted - 1] in order."""
    return [(path[i], i) for i in range(1, min(emitted, len(path))) if path[i] != i]


def apply(rows, moves, base=0):
    """Apply moves in order to a row list (rows[base + node]), as the compaction kernel does."""
    out = list(rows)
    for src, dst in moves:
        out[base + dst] = out[base + src]
    return out


def draft_rank(row, token):
    """Rank of `token` in a drafter logit row by (logit desc, id asc), the order expand() takes children in."""
    return sum(1 for t, x in enumerate(row) if x > row[token] or (x == row[token] and t < token))


class DraftRanks:
    """Per draft position, where the target's token ranked among the drafter's candidates (enqueue_draft_ranks).

    counts[d][r] counts chain rounds that compared position d (every accepted draft, and the first rejected one) with the target's token at rank r of the drafter's row there; counts[d][ranks] counts ranks >= ranks.
    covered(d, w) is the share of those rounds a tree of width w at depth d would have held from the chain's context.
    """

    def __init__(self, positions, ranks):
        if positions < 1 or ranks < 1:
            raise ValueError("positive positions and ranks are required")
        self.ranks = ranks
        self.counts = [[0] * (ranks + 1) for _ in range(positions)]

    def record(self, rows, predictions, matched, emitted):
        """One chain round: rows[d] the drafter's logits at draft d, predictions[d] the target's token after the
        first d drafts, matched the accepted drafts and emitted the round's emitted tokens (accept's walk). Position
        matched counts only when a correction was emitted there (emitted > matched): a round cut off by its output
        allowance or ending on an accepted stop never compared it."""
        if len(rows) > len(self.counts) or len(predictions) < len(rows) or not 0 <= matched <= emitted <= matched + 1:
            raise ValueError("a row per draft position, a prediction per row and emitted in [matched, matched + 1]")
        for d in range(min(matched + (1 if emitted > matched else 0), len(rows))):
            self.counts[d][min(draft_rank(rows[d], predictions[d]), self.ranks)] += 1

    def reached(self, d):
        return sum(self.counts[d])

    def covered(self, d, width):
        return sum(self.counts[d][:width]) / self.reached(d) if self.reached(d) else 0.0
