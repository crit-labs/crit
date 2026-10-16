"""Compact variable candidate rows and atomic group-publication staging."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    request_id: int
    generation: int
    candidates: tuple[int, ...]
    readiness: object


def compact_ready(rows, max_rows):
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("positive admitted row limit required")
    rows = tuple(rows)
    ids = set()
    for row in rows:
        if type(row.request_id) is not int or row.request_id <= 0 or type(row.generation) is not int or row.generation <= 0:
            raise ValueError("positive request/lane identity required")
        if row.request_id in ids:
            raise ValueError("a request may occur only once in a group")
        ids.add(row.request_id)
    return tuple(row for row in rows if row.readiness.done())[:max_rows]


class StagedGroup:
    def __init__(self, rows, budgets, reserve):
        self.rows = []
        self.owners = []
        self.used = False
        ids = set()
        for row in rows:
            if type(row.request_id) is not int or row.request_id <= 0 or type(row.generation) is not int or row.generation <= 0:
                raise ValueError("positive request/lane identity required")
            if row.request_id in ids or not row.readiness.done():
                raise ValueError("duplicate or unsettled row")
            ids.add(row.request_id)
            tokens = tuple(row.candidates)  # borrowed arrays may disappear at commit
            if not tokens or len(tokens) > budgets[row.request_id]:
                raise ValueError("candidate row exceeds admitted output budget")
            if any(type(token) is not int or not 0 <= token < 1 << 31 for token in tokens):
                raise ValueError("candidate token representation exceeded")
            self.owners.append(reserve(4 * len(tokens)))
            self.rows.append((row.request_id, row.generation, tokens))
        if not self.rows:
            raise ValueError("empty group")
        self.rows = tuple(self.rows)

    def commit(self, model_commit):
        if self.used:
            raise ValueError("group already committed or failed")
        self.used = True
        model_commit(self.rows)  # may invalidate borrowed model storage or throw
        return self.rows  # core may publish only after successful return


class RetirementBarrier:
    def __init__(self, owners, receipts):
        self.owners, self.receipts = list(owners), tuple(receipts)
        self.retired = False

    def poll(self):
        if self.retired:
            return True
        if not all(receipt.done() for receipt in self.receipts):
            return False  # elapsed time is not proof of physical settlement
        self.owners.clear()
        self.retired = True
        return True
