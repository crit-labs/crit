"""Accepted asynchronous contributions retain their sources until every receipt settles."""


class ContributionGroup:
    def __init__(self, positions):
        positions = tuple(positions)
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("nonempty unique route positions required")
        if any(type(x) is not int or x < 0 for x in positions):
            raise ValueError("nonnegative route positions required")
        self.positions = positions
        self.accepted = {}
        self.completed = False

    def accept(self, position, receipt, owner):
        if self.completed or position not in self.positions or position in self.accepted:
            raise ValueError("contribution is duplicate, unknown or already settled")
        if owner is None:
            raise ValueError("accepted work must retain its source")
        self.accepted[position] = (receipt, owner)

    def ready(self):
        return set(self.accepted) == set(self.positions) and all(receipt.done() for receipt, _ in self.accepted.values())

    def settle(self):
        if self.completed or not self.ready():
            raise ValueError("all accepted route occurrences must be settled")
        vectors, errors = [], []
        for position in self.positions:
            try:
                vectors.append(tuple(self.accepted[position][0].result()))
            except BaseException as exc:
                errors.append(exc)
        # Every receipt has terminal settlement; only now may source owners go.
        self.accepted.clear()
        self.completed = True
        if errors:
            raise errors[0]
        if not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
            raise ValueError("contribution geometry differs")
        return tuple(sum(v[column] for v in vectors) for column in range(len(vectors[0])))

    def abort(self, quiesce):
        """quiesce must stop producers and settle physical reads; failure keeps owners."""
        quiesce(tuple(receipt for receipt, _ in self.accepted.values()))
        if not all(receipt.done() for receipt, _ in self.accepted.values()):
            raise ValueError("producer quiescence is not proven")
        self.accepted.clear()
        self.completed = True
