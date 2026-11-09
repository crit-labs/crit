"""The toy's drafter: a chain of the pending token alone (the toy model verifies its own successor)."""


class Drafter:
    def start(self, sequence, tokens):
        pass

    def propose(self, sequence, width, head=None):
        return {"tokens": [], "parents": [], "laws": None}

    def observe(self, sequence, emitted):
        pass

    def release(self, sequence):
        pass


def create(context, options, parts):
    return Drafter()
