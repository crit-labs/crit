"""The plain drafter: a chain of greedy drafts from the engine's draft model (draft:head) as the host form; inside
captured decode rounds the chain over the model's device draft head (draft:head-device), argmax drafts in greedy rounds
and the sampler's draft draws with their laws in sampled ones (mtp-decode's device_drafts.head_chain, the draft:device
extension)."""


def argmax(row):
    return max(range(len(row)), key=lambda t: (row[t], -t))


class GreedyHeadDrafter:
    def __init__(self):
        self.pending = {}
        self.head = None

    def start(self, sequence, tokens):
        self.pending[sequence] = list(tokens)[-1]

    def propose(self, sequence, width, head=None):
        if head is None:
            raise ValueError("greedy-head-drafter drafts from the engine's draft head: pass head")
        path = []
        for _ in range(width):
            path.append(argmax(head([list(path)])[0]))
        return {"tokens": [self.pending[sequence]] + path, "parents": list(range(-1, width)), "laws": None}

    def observe(self, sequence, emitted):
        if emitted:
            self.pending[sequence] = emitted[-1]

    def release(self, sequence):
        self.pending.pop(sequence, None)

    # draft:device (crit_plugins.device.DEVICE_EXTENSIONS)
    def draft(self, view, head, sampler=None):
        """view.drafts[:, :width] from the model's device draft head, after its catch-up; capture-safe."""
        from .mtp import device_drafts
        if head is None:
            raise ValueError("greedy-head-drafter drafts from the model's device draft head: pass head")
        self.head = head
        device_drafts.head_chain(view, head, sampler)

    def commit_round(self, view):
        """Nothing of its own: the round driver commits the draft head (head.commit_round) after the drafter."""


def create(context, options, parts):
    return GreedyHeadDrafter()
