"""Host-side checks of complete outputs against a constraint language (standard library only).

`ByteGrammar.load(artifact)` reads a crit-byte-grammar-v1 artifact with the native loader's rules, and
`GrammarState` runs the same bounded Earley recognizer as the native `GrammarMatcher`, one byte string at a time.
`matches(language, data)` and `json_object_matches(data)` answer whether a whole output belongs to a language: a
server can check what an engine returned, and tests can use them as references. Per-token masks during generation
come from the native matchers (crit/constraints.h), which scan the whole vocabulary at every position.
"""
from dataclasses import dataclass
import json

RULE_LIMIT, PRODUCTION_LIMIT, SYMBOL_LIMIT, ARTIFACT_LIMIT = 4096, 16384, 262144, 4194304


class GrammarError(ValueError):
    """A malformed artifact or grammar, or an exhausted chart budget."""


@dataclass(frozen=True)
class ByteGrammar:
    """Symbols 0..255 are bytes; 256 + i is rule i. `rules[i]` lists the production indexes of rule i."""
    start: int
    productions: tuple  # (lhs, rhs tuple)
    rules: tuple
    nullable: frozenset

    @classmethod
    def build(cls, rule_count, start, productions):
        productions = tuple((lhs, tuple(rhs)) for lhs, rhs in productions)
        if (not 0 < rule_count <= RULE_LIMIT or not 0 <= start < rule_count or not productions
                or len(productions) > PRODUCTION_LIMIT or sum(len(rhs) for _, rhs in productions) > SYMBOL_LIMIT):
            raise GrammarError("invalid bounded byte grammar")
        rules = [[] for _ in range(rule_count)]
        for index, (lhs, rhs) in enumerate(productions):
            if not 0 <= lhs < rule_count or any(not 0 <= s < 256 + rule_count for s in rhs):
                raise GrammarError("undefined grammar symbol")
            rules[lhs].append(index)
        nullable, productive = set(), set()
        changed = True
        while changed:
            changed = False
            for lhs, rhs in productions:
                if lhs not in nullable and all(s >= 256 and s - 256 in nullable for s in rhs):
                    nullable.add(lhs); changed = True
                if lhs not in productive and all(s < 256 or s - 256 in productive for s in rhs):
                    productive.add(lhs); changed = True
        if len(productive) != rule_count:
            raise GrammarError("grammar has undefined or unproductive rules")
        return cls(start, productions, tuple(tuple(r) for r in rules), frozenset(nullable))

    @classmethod
    def load(cls, artifact):
        if isinstance(artifact, str):
            artifact = artifact.encode("ascii", "replace")
        if len(artifact) > ARTIFACT_LIMIT:
            raise GrammarError("grammar artifact exceeds 4 MiB")
        fields = artifact.split()
        if len(fields) < 4 or fields[0] != b"crit-byte-grammar-v1":
            raise GrammarError("invalid grammar artifact header")

        def number(at):
            text = fields[at] if at < len(fields) else b""
            if not text.isdigit() or len(text) > 9:
                raise GrammarError("invalid grammar artifact field")
            return int(text)
        rule_count, count, start = number(1), number(2), number(3)
        if not 0 < rule_count <= RULE_LIMIT or not 0 < count <= PRODUCTION_LIMIT or start >= rule_count:
            raise GrammarError("invalid grammar artifact header")
        at, productions = 4, []
        for _ in range(count):
            lhs, size = number(at), number(at + 1)
            productions.append((lhs, tuple(number(at + 2 + i) for i in range(size))))
            at += 2 + size
        if at != len(fields):
            raise GrammarError("trailing grammar artifact data")
        return cls.build(rule_count, start, productions)


class GrammarState:
    """An Earley frontier after some output bytes. `advance` returns a new state or None; states are immutable, so
    keeping one is a fork. Budgets match the native GrammarLimits defaults scaled for whole outputs."""
    def __init__(self, grammar, *, states_per_byte=65536, total_states=4194304, _charts=None, _total=0):
        self.grammar, self.states_per_byte, self.total_states = grammar, states_per_byte, total_states
        if _charts is None:
            charts = [[(p, 0, 0) for p in grammar.rules[grammar.start]]]
            self._total = self._close(charts, 0)
            self._charts = charts
        else:
            self._charts, self._total = _charts, _total

    def _close(self, charts, total):
        """Completes and predicts the last chart; returns the item count of all charts."""
        grammar, position = self.grammar, len(charts) - 1
        chart, seen = charts[-1], set()
        seeds, chart[:] = list(chart), []

        def add(item):
            nonlocal total
            if item in seen:
                return
            if len(chart) == self.states_per_byte or total == self.total_states:
                raise GrammarError("grammar chart budget exhausted")
            seen.add(item); chart.append(item); total += 1
        for seed in seeds:
            add(seed)
        at = 0
        while at < len(chart):
            production, dot, origin = chart[at]
            at += 1
            lhs, rhs = grammar.productions[production]
            if dot == len(rhs):
                parents = charts[origin]
                j = 0
                while j < len(parents):
                    parent_production, parent_dot, parent_origin = parents[j]
                    j += 1
                    parent_rhs = grammar.productions[parent_production][1]
                    if parent_dot < len(parent_rhs) and parent_rhs[parent_dot] == 256 + lhs:
                        add((parent_production, parent_dot + 1, parent_origin))
            elif rhs[dot] >= 256:
                symbol = rhs[dot] - 256
                for index in grammar.rules[symbol]:
                    add((index, 0, position))
                if symbol in grammar.nullable:
                    add((production, dot + 1, origin))
        return total

    def advance(self, data):
        charts, total = list(self._charts), self._total
        for byte in data:
            scanned = [(p, d + 1, o) for p, d, o in charts[-1]
                       if d < len(self.grammar.productions[p][1]) and self.grammar.productions[p][1][d] == byte]
            if not scanned:
                return None
            charts.append(scanned)
            total = self._close(charts, total)
        return GrammarState(self.grammar, states_per_byte=self.states_per_byte, total_states=self.total_states,
                            _charts=charts, _total=total)

    @property
    def complete(self):
        start = self.grammar.start
        return any(origin == 0 and self.grammar.productions[p][0] == start and dot == len(self.grammar.productions[p][1])
                   for p, dot, origin in self._charts[-1])


def matches(language, data):
    """Whether bytes `data` are a whole sentence of `language` (a ByteGrammar or an artifact)."""
    grammar = language if isinstance(language, ByteGrammar) else ByteGrammar.load(language)
    state = GrammarState(grammar).advance(data)
    return state is not None and state.complete


def json_object_matches(data):
    """Whether bytes `data` are one JSON object with optional JSON whitespace around it: the native
    JsonObjectMatcher's language (strict UTF-8, no NaN/Infinity, no raw control characters in strings)."""
    def constant(_):
        raise ValueError("non-JSON constant")
    try:
        return isinstance(json.loads(bytes(data).decode("utf-8"), parse_constant=constant), dict)
    except (UnicodeError, ValueError, RecursionError):
        return False
