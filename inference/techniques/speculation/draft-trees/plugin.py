"""Draft trees in the drafter and verifier slots: draft-trees' EAGLE-2 expansion and kept tree over the engine's
draft model, and its accept walk over the verified tree (tree.py). The engine verifies every node in one pass under
the ancestor masks of the proposal's parents. Inside captured decode rounds the same tree and walk on the device
(device_tree.py, the draft:device and verify:device extensions): the model's device draft head drafts every row's
frontier at once (head.logits(depth, tokens, view, parents=...)), and its verify pass reads the tree under ancestor
masks (attention:tree)."""
from .trees import tree


def as_tree(tokens, parents):
    if not tokens or parents[0] != -1 or any(not 0 <= parents[i] < i for i in range(1, len(tokens))):
        raise ValueError("a proposal's parents are -1 for the root and an earlier node for every other node")
    masks, depths = [1], [0]
    for i in range(1, len(tokens)):
        masks.append(masks[parents[i]] | (1 << i))
        depths.append(depths[parents[i]] + 1)
    return tree.Tree(tuple(tokens), tuple(parents), tuple(masks), tuple(depths))


class DraftTrees:
    def __init__(self, options, tree_attention):
        self.options, self.pending = options, {}
        self.buffers, self.tree_attention = None, tree_attention

    # drafter
    def start(self, sequence, tokens):
        self.pending[sequence] = list(tokens)[-1]

    def propose(self, sequence, width, head=None):
        if head is None:
            raise ValueError("draft-trees drafts from the engine's draft head: pass head")
        root = self.pending[sequence]
        if width < 1:
            return {"tokens": [root], "parents": [-1], "laws": None}
        o = self.options
        _, kept = tree.draft(lambda paths: head(paths), root, min(o["depth"], width), o["branch"], o["beam"],
                             width + 1)
        return {"tokens": list(kept.tokens), "parents": list(kept.parents), "laws": None}

    def observe(self, sequence, emitted):
        if emitted:
            self.pending[sequence] = emitted[-1]

    def release(self, sequence):
        self.pending.pop(sequence, None)

    # verifier: any proposal, a chain included
    def accept(self, proposal, predictions, remaining, stops=()):
        emitted, path = tree.accept(as_tree(list(proposal["tokens"]), list(proposal["parents"])), list(predictions),
                                    remaining, tuple(stops))
        return {"tokens": list(emitted), "accepted": len(path) - 1, "path": list(path)}


    # draft:device and verify:device (crit_plugins.device.DEVICE_EXTENSIONS)
    def reserve(self, rows, widest, vocab, device):
        """A tree round's parents, depths, path and path-order predictions for up to `rows` rows at up to `widest`
        drafts, once, before any capture (the round driver calls it)."""
        from .trees import device_tree
        self.buffers = device_tree.TreeBuffers(rows, widest, device)
        return self.buffers

    def _buffers(self):
        if self.buffers is None:
            raise ValueError("draft-trees: reserve(rows, widest, vocab, device) its round buffers first")
        if not self.tree_attention:
            raise ValueError("draft-trees: the attention plugin provides no attention:tree, so the target cannot verify "
                             "a tree's nodes under their ancestor masks inside a round")
        return self.buffers

    def draft(self, view, head, sampler=None):
        """The round's tree for every row: view.drafts[:, :width] (node i at column i - 1), view.tree_parents and
        view.tree_depths, from the model's device draft head; its children are top-k tokens in every round (a sampled
        round's verify draws each node at its depth: the sampler's predictions_device over view.tree_depths);
        capture-safe."""
        from .trees import device_tree
        if head is None:
            raise ValueError("draft-trees drafts from the model's device draft head: pass head")
        o = self.options
        device_tree.draft_tree(view, head, o["depth"], o["branch"], o["beam"], self._buffers())

    def commit_round(self, view):
        """Nothing of its own: the round driver commits the draft head (head.commit_round, view.tree_path)."""

    def accept_device(self, view):
        """The walk for every row (view.n_emitted, view.n_matched, view.emitted, view.tree_path, and view.predictions
        in path order), ending after a stop token and within the allowance, a held row emitting nothing;
        capture-safe."""
        from .trees import device_tree
        device_tree.accept_tree(view, self._buffers(), self.options["depth"])


def create(context, options, parts):
    # Inside rounds the target verifies the tree under ancestor masks: an attention plugin of the selection provides
    # attention:tree (a model that computes its attention itself fills no attention slot).
    attention = parts.plugin("attention") if callable(getattr(parts, "plugin", None)) else None
    return DraftTrees(options, attention is None or "attention:tree" in parts.facts("attention"))
