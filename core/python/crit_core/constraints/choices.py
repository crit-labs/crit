"""Python reference for finite byte choices (the native ChoiceMatcher); mutable state is private to one request."""
from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class ChoiceGrammar:
    _nodes: tuple
    _terminal: frozenset
    vocabulary: tuple
    eos: int

    def __init__(self, choices, vocabulary, eos):
        choices, vocabulary = tuple(choices), tuple(vocabulary)
        if (not choices or len(choices) > 65536 or not vocabulary or len(vocabulary) > 1048576
                or any(type(x) is not bytes for x in choices + vocabulary)
                or type(eos) is not int or not 0 <= eos < len(vocabulary)
                or vocabulary[eos] or sum(map(len, choices)) > 1048576
                or sum(map(len, vocabulary)) > 67108864):
            raise ValueError("bounded byte choices/vocabulary and an empty EOS entry required")
        nodes, terminal = [{}], set()
        for choice in choices:
            node = 0
            for byte in choice:
                if byte not in nodes[node]:
                    nodes[node][byte] = len(nodes)
                    nodes.append({})
                node = nodes[node][byte]
            terminal.add(node)
        # Freeze all exposed inputs; states never mutate the compiled language.
        object.__setattr__(self, "_nodes", tuple(tuple(sorted(node.items())) for node in nodes))
        object.__setattr__(self, "_terminal", frozenset(terminal))
        object.__setattr__(self, "vocabulary", vocabulary)
        object.__setattr__(self, "eos", eos)

    def transition(self, node, token):
        if type(token) is not int or not 0 <= token < len(self.vocabulary):
            raise ValueError("token outside vocabulary")
        if token == self.eos:
            return -1 if node in self._terminal else None
        piece = self.vocabulary[token]
        if not piece or node == -1:
            return None
        for byte in piece:
            node = dict(self._nodes[node]).get(byte)
            if node is None:
                return None
        return node


class ChoiceState:
    def __init__(self, grammar):
        if not isinstance(grammar, ChoiceGrammar):
            raise ValueError("compiled choice grammar required")
        self.grammar, self._states = grammar, [0]

    @property
    def ended(self):
        return self._states[-1] == -1

    def fork(self):
        result = ChoiceState(self.grammar)
        result._states = self._states.copy()
        return result

    def allowed(self):
        if self.ended:
            return (self.grammar.eos,)  # absorbing next law for terminal verification rows
        result = tuple(i for i in range(len(self.grammar.vocabulary))
                       if self.grammar.transition(self._states[-1], i) is not None)
        if not result:
            raise ValueError("choice language is unreachable with this vocabulary")
        return result

    def append(self, token):
        node = self.grammar.transition(self._states[-1], token)
        if node is None:
            raise ValueError("token violates choice language")
        self._states.append(node)

    def rewind(self, count):
        if type(count) is not int or not 0 <= count < len(self._states):
            raise ValueError("rewind cannot extend output")
        del self._states[count + 1:]
