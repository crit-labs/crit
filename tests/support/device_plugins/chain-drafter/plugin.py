"""A test stand-in for a chain drafter's device form (draft:device), which greedy-head-drafter also provides: one
draft per depth, the argmax of the model's draft head (draft:head-device) after the previous draft, inside a captured
round. Its host form proposes nothing: it serves only inside rounds."""


class ChainDrafter:
    def start(self, sequence, tokens):
        pass

    def propose(self, sequence, width):
        return {"tokens": [], "parents": [], "laws": None}

    def observe(self, sequence, accepted):
        pass

    def release(self, sequence):
        pass

    def draft(self, view, head, sampler):
        head.catch_up(view)
        token = view.pending
        for depth in range(view.width):
            token = head.logits(depth, token, view).argmax(-1)
            view.drafts[:, depth] = token

    def commit_round(self, view):
        """Nothing of its own: the head keeps what it needs."""


def create(context, options, parts):
    return ChainDrafter()
