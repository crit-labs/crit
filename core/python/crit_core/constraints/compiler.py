"""Standard-library compiler for byte regex/EBNF and a strict JSON Schema subset.

Emits the data-only crit-byte-grammar-v1 format that crit::core::ByteGrammar::load
(crit/constraints.h) and crit_core.constraints.grammar read. This is not a full
JSON Schema/PCRE/GBNF implementation. Unsupported constructs raise
UnsupportedConstraint with the reason; schemas generate a documented canonical
subset of their valid JSON encodings. The artifact does not depend on a tokenizer.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
from itertools import islice
import re
import sys

VERSION = "crit-language-compiler-1"
MAX_SOURCE = 1048576


class UnsupportedConstraint(ValueError):
    pass


def _integer(value, name, maximum=1024):
    if type(value) is not int or not 0 <= value <= maximum:
        raise UnsupportedConstraint(f"{name} must be an integer in [0, {maximum}]")
    return value


def _json(source):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise UnsupportedConstraint("duplicate JSON key")
            result[k] = v
        return result
    def constant(_):
        raise UnsupportedConstraint("nonfinite JSON constant")
    return json.loads(source, object_pairs_hook=pairs, parse_constant=constant)


def _literal_json(value):
    # Reject unpaired surrogates even in nested enum/const values and keys.
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


class Builder:
    def __init__(self):
        self.rules, self.symbols = [], 0

    def new(self):
        if len(self.rules) == 4096:
            raise UnsupportedConstraint("grammar rule budget exceeded")
        self.rules.append([])
        return 256 + len(self.rules) - 1

    def define(self, rule, choices):
        choices = [tuple(c) for c in choices]
        added = sum(map(len, choices))
        if self.symbols + added > 262144 or sum(map(len, self.rules)) + len(choices) > 16384:
            raise UnsupportedConstraint("grammar production budget exceeded")
        self.rules[rule - 256].extend(choices)
        self.symbols += added
        return rule

    def rule(self, choices):
        return self.define(self.new(), choices)

    def literal(self, value):
        return self.rule([value.encode("utf-8") if isinstance(value, str) else value])

    def chars(self, values):
        return self.rule((v,) for v in sorted(set(values)))

    def sequence(self, parts):
        return self.rule([parts])

    def choice(self, parts):
        return self.rule((part,) for part in parts)

    def repeat(self, rule, low=0, high=None):
        _integer(low, "minimum repetition")
        if high is not None:
            _integer(high, "maximum repetition")
            if high < low:
                raise UnsupportedConstraint("inverted repetition bounds")
        tail = self.new()
        if high is None:
            self.define(tail, [(), (rule, tail)])
        else:
            self.define(tail, [()])
            for _ in range(high - low):
                tail = self.rule([(), (rule, tail)])
        return self.sequence([rule] * low + [tail])

    def artifact(self, start):
        # Reject undefined/unproductive cycles at compilation, before native use.
        productive = set()
        while True:
            more = {256 + i for i, choices in enumerate(self.rules)
                    if any(all(s < 256 or s in productive for s in c) for c in choices)}
            if more <= productive:
                break
            productive |= more
        if len(productive) != len(self.rules):
            raise UnsupportedConstraint("undefined or unproductive grammar rule")
        lines = [f"crit-byte-grammar-v1 {len(self.rules)} {sum(map(len, self.rules))} {start - 256}"]
        for i, choices in enumerate(self.rules):
            for rhs in choices:
                lines.append(f"{i} {len(rhs)}" + (" " + " ".join(map(str, rhs)) if rhs else ""))
        result = "\n".join(lines) + "\n"
        if len(result) > 4194304:
            raise UnsupportedConstraint("compiled artifact byte budget exceeded")
        return result


class Expression:
    """Recursive-descent byte-regex / line-EBNF frontend; bounded nesting."""
    def __init__(self, text, builder, names=None):
        self.text, self.b, self.names, self.at, self.depth = text, builder, names, 0, 0

    def peek(self):
        if self.names is not None:
            while self.at < len(self.text) and self.text[self.at].isspace():
                self.at += 1
            if self.at < len(self.text) and self.text[self.at] == "#":
                self.at = len(self.text)
        return self.text[self.at:self.at + 1]

    def escaped(self):
        if self.at == len(self.text):
            raise UnsupportedConstraint("trailing escape")
        char = self.text[self.at]; self.at += 1
        if char in "nrt0":
            if char == "0" and self.text[self.at:self.at + 1].isdigit():
                raise UnsupportedConstraint("octal escapes are unsupported; use hex bytes")
            return {"n": 10, "r": 13, "t": 9, "0": 0}[char]
        if char == "x":
            text = self.text[self.at:self.at + 2]
            if len(text) != 2 or not re.fullmatch("[0-9a-fA-F]{2}", text):
                raise UnsupportedConstraint("invalid byte escape")
            self.at += 2
            return int(text, 16)
        if char not in r"\.*+?{}[]()|^$-/#\"":
            raise UnsupportedConstraint("unsupported escape/backreference")
        return ord(char)

    def char_class(self):
        self.at += 1
        invert = self.text[self.at:self.at + 1] == "^"
        self.at += int(invert)
        values = set()
        def byte():
            if self.at == len(self.text):
                raise UnsupportedConstraint("unterminated character class")
            value = self.text[self.at]; self.at += 1
            if value == "\\":
                return self.escaped()
            if ord(value) > 127:
                raise UnsupportedConstraint("byte classes require ASCII or hex escapes")
            return ord(value)
        while self.text[self.at:self.at + 1] != "]":
            first = byte()
            if self.text[self.at:self.at + 1] == "-" and self.text[self.at + 1:self.at + 2] != "]":
                self.at += 1
                last = byte()
                if first > last:
                    raise UnsupportedConstraint("inverted character range")
                values.update(range(first, last + 1))
            else:
                values.add(first)
        self.at += 1
        if not values:
            raise UnsupportedConstraint("empty character class")
        if invert:
            values = set(range(256)) - values
        if not values:
            raise UnsupportedConstraint("empty character class")
        return self.b.chars(values)

    def atom(self):
        char = self.peek()
        if char == "(":
            self.at += 1; self.depth += 1
            if self.depth > 64:
                raise UnsupportedConstraint("expression nesting limit")
            rule = self.expression()
            if self.peek() != ")":
                raise UnsupportedConstraint("unclosed group")
            self.at += 1; self.depth -= 1
            return rule
        if char == "[":
            return self.char_class()
        if self.names is not None:
            if char == '"':
                value, end = json.JSONDecoder().raw_decode(self.text[self.at:])
                self.at += end
                return self.b.literal(value)
            found = re.match(r"[A-Za-z_][A-Za-z_0-9-]*", self.text[self.at:])
            if found and found[0] in self.names:
                self.at += len(found[0])
                return self.names[found[0]]
            raise UnsupportedConstraint("unknown rule or unsupported EBNF atom")
        self.at += 1
        if char == ".":
            return self.b.chars(set(range(256)) - {10})
        if char == "\\":
            special = self.text[self.at:self.at + 1]
            groups = {"d": b"0123456789", "w": b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_",
                      "s": b" \t\r\n\f\v"}
            if special in groups:
                self.at += 1
                return self.b.chars(groups[special])
            return self.b.chars([self.escaped()])
        if char in "*+?{}^$" or not char:
            raise UnsupportedConstraint("unsupported regex operator")
        return self.b.literal(char)

    def expression(self):
        choices, sequence = [], []
        while self.peek() not in {"", ")"}:
            if self.peek() == "|":
                self.at += 1; choices.append(self.b.sequence(sequence)); sequence = []
                continue
            atom = self.atom()
            modifier = self.peek()
            if modifier in {"?", "*", "+"}:
                self.at += 1
                atom = self.b.repeat(atom, int(modifier == "+"), 1 if modifier == "?" else None)
            elif modifier == "{":
                match = re.match(r"\{([0-9]+)(?:,([0-9]*))?\}", self.text[self.at:])
                if not match:
                    raise UnsupportedConstraint("invalid repetition")
                self.at += len(match[0])
                low = int(match[1])
                high = low if match[2] is None else int(match[2]) if match[2] else None
                atom = self.b.repeat(atom, low, high)
            sequence.append(atom)
        choices.append(self.b.sequence(sequence))
        return self.b.choice(choices)

    def parse(self):
        root = self.expression()
        if self.peek():
            raise UnsupportedConstraint("trailing expression data")
        return root


class Schema:
    def __init__(self, builder):
        self.b, self.character = builder, None

    def string_character(self):
        if self.character is not None:
            return self.character
        b = self.b
        def char(low, high):
            return b.chars(range(low, high + 1))
        continuation = char(0x80, 0xbf)
        hex_digit = b.chars(b"0123456789abcdefABCDEF")
        # Raw Unicode scalar values and paired surrogate escapes count as one
        # decoded character. Unpaired surrogate escapes are deliberately omitted.
        raw = [b.chars(set(range(0x20, 0x7f)) - {34, 92}),
               b.sequence([char(0xc2, 0xdf), continuation]),
               b.sequence([char(0xe0, 0xe0), char(0xa0, 0xbf), continuation]),
               b.sequence([char(0xe1, 0xec), continuation, continuation]),
               b.sequence([char(0xed, 0xed), char(0x80, 0x9f), continuation]),
               b.sequence([char(0xee, 0xef), continuation, continuation]),
               b.sequence([char(0xf0, 0xf0), char(0x90, 0xbf), continuation, continuation]),
               b.sequence([char(0xf1, 0xf3), continuation, continuation, continuation]),
               b.sequence([char(0xf4, 0xf4), char(0x80, 0x8f), continuation, continuation]),
               b.sequence([92, b.chars(b'"\\/bfnrt')])]
        lead = b.chars(b"0123456789aAbBcCeEfF")
        d = b.chars(b"dD")
        scalar = b.choice([b.sequence([lead, hex_digit, hex_digit, hex_digit]),
                           b.sequence([d, b.chars(b"01234567"), hex_digit, hex_digit])])
        pair = b.sequence([92, 117, d, b.chars(b"89aAbB"), hex_digit, hex_digit,
                           92, 117, d, b.chars(b"cCdDeEfF"), hex_digit, hex_digit])
        raw += [b.sequence([92, 117, scalar]), pair]
        self.character = b.choice(raw)
        return self.character

    def compile(self, schema, depth=0):
        if depth > 32 or not isinstance(schema, dict):
            raise UnsupportedConstraint("schema object with bounded nesting required")
        b = self.b
        annotations = {"title", "description", "$comment"}
        schema = {k: v for k, v in schema.items() if k not in annotations}
        if "$schema" in schema:
            if schema.pop("$schema") != "https://json-schema.org/draft/2020-12/schema":
                raise UnsupportedConstraint("only the declared 2020-12 subset is supported")
        if "enum" in schema or "const" in schema:
            key = "enum" if "enum" in schema else "const"
            if set(schema) - {key, "type"}:
                raise UnsupportedConstraint("enum/const supports only an optional type")
            values = schema[key] if key == "enum" else [schema[key]]
            if not isinstance(values, list) or not values or len(values) > 4096:
                raise UnsupportedConstraint("nonempty bounded enum required")
            def kind(value):
                return ("null" if value is None else "boolean" if type(value) is bool else
                        "integer" if type(value) is int or (type(value) is float and value.is_integer()) else "number" if type(value) is float else
                        "string" if isinstance(value, str) else "array" if isinstance(value, list) else "object")
            expected = schema.get("type")
            if "type" in schema and (not isinstance(expected, str) or expected not in {"null", "boolean", "integer", "number", "string", "array", "object"}):
                raise UnsupportedConstraint("enum type must be a supported type name")
            if expected is not None and any(kind(v) != expected and not (expected == "number" and kind(v) == "integer") for v in values):
                raise UnsupportedConstraint("enum contains a value outside its declared type")
            return b.choice(b.literal(_literal_json(v)) for v in values)
        kind = schema.get("type")
        permitted = {"object": {"properties", "required", "additionalProperties"},
                     "array": {"items", "minItems", "maxItems"},
                     "string": {"minLength", "maxLength"},
                     "integer": {"minimum", "maximum"}, "number": set(), "boolean": set(), "null": set()}
        if not isinstance(kind, str) or kind not in permitted or set(schema) - permitted[kind] - {"type"}:
            raise UnsupportedConstraint("unsupported schema type/keyword")
        if kind in {"boolean", "null"}:
            return b.choice(b.literal(v) for v in (("true", "false") if kind == "boolean" else ("null",)))
        if kind == "integer" and ("minimum" in schema or "maximum" in schema):
            low, high = schema.get("minimum"), schema.get("maximum")
            if type(low) is not int or type(high) is not int or not 0 <= high - low <= 4095:
                raise UnsupportedConstraint("integer ranges need both integral bounds and at most 4096 values")
            return b.choice(b.literal(str(i)) for i in range(low, high + 1))
        if kind in {"integer", "number"}:
            expression = r"-?(0|[1-9][0-9]*)"
            if kind == "number":
                expression += r"(\.[0-9]+)?([eE][+-]?[0-9]+)?"
            return Expression(expression, b).parse()
        if kind == "string":
            low = _integer(schema.get("minLength", 0), "minLength")
            high = schema.get("maxLength")
            if "maxLength" in schema:
                _integer(high, "maxLength")
            return b.sequence([34, b.repeat(self.string_character(), low, high), 34])
        if kind == "array":
            if "items" not in schema:
                raise UnsupportedConstraint("typed items required")
            item = self.compile(schema["items"], depth + 1)
            low = _integer(schema.get("minItems", 0), "minItems")
            high = schema.get("maxItems")
            if "maxItems" in schema:
                _integer(high, "maxItems")
                if high < low:
                    raise UnsupportedConstraint("inverted array bounds")
            if high == 0:
                return b.literal("[]")
            body = b.sequence([item, b.repeat(b.sequence([44, item]), max(0, low - 1), None if high is None else high - 1)])
            if not low:
                body = b.choice([body, b.literal("")])
            return b.sequence([91, body, 93])
        properties, required = schema.get("properties", {}), schema.get("required", [])
        if (not isinstance(properties, dict) or len(properties) > 128 or not isinstance(required, list)
                or any(not isinstance(k, str) or k not in properties for k in required)
                or len(set(required)) != len(required) or schema.get("additionalProperties") is not False):
            raise UnsupportedConstraint("bounded declared properties, unique required names and additionalProperties:false required")
        # Canonical property order avoids exponential subsets/permutations while
        # preserving every optional-property combination and all required fields.
        fields = [(b.literal(_literal_json(k) + ":"), self.compile(v, depth + 1), k in required)
                  for k, v in properties.items()]
        tails = {False: b.literal(""), True: b.literal("")}
        for key, value, required in reversed(fields):
            next_tails = {}
            for seen in (False, True):
                choices = [([44] if seen else []) + [key, value, tails[True]]]
                if not required:
                    choices.append([tails[seen]])
                next_tails[seen] = b.rule(choices)
            tails = next_tails
        return b.sequence([123, tails[False], 125])


def compile_language(kind, source):
    """Compile exact source text; unsupported syntax never silently relaxes it."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE:
        raise UnsupportedConstraint("bounded UTF8 constraint source required")
    b = Builder()
    try:
        if kind == "schema":
            start = Schema(b).compile(_json(source))
        elif kind == "regex":
            # All regexes use whole-byte-string matching; optional outer anchors.
            if source.startswith("^"):
                source = source[1:]
            if source.endswith("$"):
                backslashes = len(source[:-1]) - len(source[:-1].rstrip("\\"))
                if backslashes % 2 == 0:
                    source = source[:-1]
            start = Expression(source, b).parse()
        elif kind == "ebnf":
            definitions = []
            for line in source.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z_0-9-]*)\s*::=\s*(.*)", line)
                if not match:
                    raise UnsupportedConstraint("one name ::= expression per line required")
                definitions.append(match.groups())
            if len({name for name, _ in definitions}) != len(definitions):
                raise UnsupportedConstraint("duplicate rule")
            names = {name: b.new() for name, _ in definitions}
            if "root" not in names:
                raise UnsupportedConstraint("root rule required")
            for name, text in definitions:
                b.define(names[name], [(Expression(text, b, names).parse(),)])
            start = names["root"]
        else:
            raise UnsupportedConstraint("kind must be schema, regex or ebnf")
        return b.artifact(start)
    except (RecursionError, UnicodeError, json.JSONDecodeError, OverflowError) as exc:
        raise UnsupportedConstraint("invalid or excessive constraint input") from exc


@dataclass(frozen=True)
class CompiledConstraint:
    artifact: str
    tokenizer_identity: str
    pieces: tuple[bytes, ...]
    eos: int


@dataclass(frozen=True)
class CompileRequest:
    version: str
    kind: str
    source: str
    tokenizer_identity: str
    pieces: tuple[bytes, ...]
    eos: int

    @property
    def input_bytes(self):
        return sum(len(s.encode()) for s in (self.version, self.kind, self.source, self.tokenizer_identity)) + sum(map(len, self.pieces))


def make_compile_request(kind, source, *, tokenizer_identity, pieces, eos, version=VERSION):
    """Exact immutable identity shared by synchronous and live preparation caches."""
    if (type(version) is not str or not version or len(version.encode()) > 4096
            or type(kind) is not str or kind not in {'schema', 'regex', 'ebnf'}):
        raise UnsupportedConstraint("known language and bounded compiler version required")
    if type(tokenizer_identity) is not str or not tokenizer_identity or len(tokenizer_identity.encode()) > 4096:
        raise UnsupportedConstraint("bounded tokenizer identity required")
    pieces = tuple(islice(pieces, 1048577))
    if (not pieces or len(pieces) > 1048576 or type(eos) is not int or not 0 <= eos < len(pieces)
            or any(type(p) is not bytes for p in pieces) or pieces[eos]
            or sum(map(len, pieces)) > 67108864):
        raise UnsupportedConstraint("exact bounded byte vocabulary and empty EOS required")
    if type(source) is not str or len(source.encode()) > MAX_SOURCE:
        raise UnsupportedConstraint("bounded constraint source required")
    return CompileRequest(version, kind, source, tokenizer_identity, pieces, eos)


class CompilerCache:
    """Caller-owned bounded LRU. Keys bind exact source/options/version/vocabulary.

    No mutable matcher state is cached. A cache hit is not a shared request.
    max_bytes bounds retained source, vocabulary and artifact bytes, excluding
    Python object overhead; engines must also budget their native chart memory.
    """
    def __init__(self, max_entries=16, max_bytes=8388608):
        _integer(max_entries, "cache entries", 1024)
        _integer(max_bytes, "cache bytes", 1 << 30)
        self.max_entries, self.max_bytes, self.bytes = max_entries, max_bytes, 0
        self._entries = OrderedDict()

    def compile(self, kind, source, *, tokenizer_identity, pieces, eos):
        key = make_compile_request(kind, source, tokenizer_identity=tokenizer_identity, pieces=pieces, eos=eos)
        if key in self._entries:
            result, size = self._entries.pop(key)
            self._entries[key] = result, size
            return result
        result = CompiledConstraint(compile_language(kind, source), tokenizer_identity, key.pieces, eos)
        size = len(result.artifact.encode()) + key.input_bytes
        if self.max_entries and size <= self.max_bytes:
            while self._entries and (len(self._entries) >= self.max_entries or self.bytes + size > self.max_bytes):
                _, (_, old) = self._entries.popitem(last=False); self.bytes -= old
            self._entries[key] = result, size; self.bytes += size
        return result


if __name__ == "__main__":
    from pathlib import Path
    if len(sys.argv) != 4:
        raise SystemExit("usage: python -m crit_core.constraints.compiler schema|regex|ebnf SOURCE OUTPUT.grammar")
    Path(sys.argv[3]).write_text(compile_language(sys.argv[1], Path(sys.argv[2]).read_text(encoding="utf-8")),
                              encoding="ascii", newline="\n")
