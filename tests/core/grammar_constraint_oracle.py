"""Exercise Core-compiled artifacts in the native GrammarMatcher against independent standard-library laws."""
import itertools
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "python"))
from crit_core.constraints import ByteGrammar, PreparationService, matches  # noqa: E402


def run(executable, service):
    rng = random.Random(41379)
    total = 0

    def check(kind, source, corpus, predicate):
        nonlocal total
        corpus = list(dict.fromkeys(corpus))
        with tempfile.TemporaryDirectory(prefix="crit-grammar-") as folder:
            artifact = Path(folder) / "language.grammar"
            handle = service.request(kind, source, tokenizer_identity='oracle.byte.v1',
                                     pieces=tuple(bytes([i]) for i in range(256)) + (b'',), eos=256)
            try:
                while not handle.ready():
                    if handle.state != 'pending':
                        handle.result()  # Surface compilation failure/timeout.
                    time.sleep(.001)
                artifact.write_text(handle.result().artifact, encoding="ascii")
                grammar = ByteGrammar.load(handle.result().artifact)
            finally:
                handle.close()
            result = subprocess.run([executable, "grammar_corpus", str(artifact)],
                                    input="\n".join(raw.hex() for raw in corpus) + "\n",
                                    capture_output=True, text=True, encoding="ascii", check=True, timeout=30)
        answers = result.stdout.splitlines()
        if len(answers) != len(corpus):
            raise AssertionError(f"Expected {len(corpus)} answers, got {len(answers)}: {result.stderr}")
        for raw, actual in zip(corpus, answers):
            expected = "1" if predicate(raw) else "0"
            if actual != expected:
                raise AssertionError(f"{kind}={source}: native={actual}, oracle={expected}, bytes={raw!r}")
            # The host recognizer (crit_core.constraints.matches) must agree with the native matcher too.
            if ("1" if matches(grammar, raw) else "0") != actual:
                raise AssertionError(f"{kind}={source}: host recognizer differs from native={actual} for {raw!r}")
        total += len(corpus)

    alphabet = b'ab()01\\\x00\n'
    short = [bytes(row) for n in range(4) for row in itertools.product(alphabet, repeat=n)]
    short += [b"abc0", b"abc12", b"a0", b"123", b"c\xc3\xa9", b"\\", b"abc\n"]
    for expression in (r"a(b|c)*[0-9]{1,3}", r"(ab|a)?", r"[^a-c]{0,3}",
                       r"\x00|c\xc3\xa9", r"\d+\s?\w", r"^\\$", r".*", r"(a?)*"):
        reference = re.compile(expression.encode("ascii"))
        check("regex", expression, short, lambda raw: reference.fullmatch(raw) is not None)

    def balanced(raw):
        depth = 0
        for byte in raw:
            if byte not in b"()":
                return False
            depth += 1 if byte == 40 else -1
            if depth < 0:
                return False
        return depth == 0

    parentheses = [bytes(row) for n in range(9) for row in itertools.product(b"()", repeat=n)]
    check("ebnf", 'root ::= "(" root ")" root | ""', parentheses + short, balanced)
    check("ebnf", 'root ::= root "a" | ""', short, lambda raw: all(c == 97 for c in raw))

    def variants(values):
        rows = [b"", b"NaN", b"Infinity", b"01", b'"\xed\xa0\x80"', b'"\\uD800"', b'"\\uDC00"']
        for value in values:
            for ascii_only in (True, False):
                raw = json.dumps(value, ensure_ascii=ascii_only, separators=(",", ":")).encode("utf-8")
                rows.append(raw)
                rows.extend(raw[:n] for n in range(len(raw)))
                rows += [b" " + raw, raw + b" ", raw + b"x"]
                for _ in range(2):
                    at = rng.randrange(len(raw))
                    rows.append(raw[:at] + bytes([rng.choice(b'{}[],:"\\012true\x00')]) + raw[at + 1:])
        return rows

    def parse(raw, predicate, *, ordered=None, canonical=False):
        try:
            def pairs(items):
                if len(dict(items)) != len(items):
                    raise ValueError("duplicate keys")
                return dict(items)
            def bad_constant(_):
                raise ValueError("nonfinite constant")
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
            # The supported schema language deliberately emits compact JSON.
            quoted, escaped = False, False
            for ch in raw:
                if not quoted and ch in b" \t\r\n":
                    return False
                if escaped:
                    escaped = False
                elif quoted and ch == 92:
                    escaped = True
                elif ch == 34:
                    quoted = not quoted
            if canonical and raw != json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii"):
                return False
            if ordered is not None and isinstance(value, dict) and list(value) != [k for k in ordered if k in value]:
                return False
            return predicate(value)
        except (UnicodeError, ValueError, TypeError):
            return False

    def schema_test(schema, values, predicate, *, extra=(), ordered=None, canonical=False):
        check("schema", json.dumps(schema), variants(values) + list(extra),
              lambda raw: parse(raw, predicate, ordered=ordered, canonical=canonical))

    schema_test({"type": "integer", "minimum": -2, "maximum": 3},
                list(range(-4, 6)) + [None, True, "1", 1.0],
                lambda v: type(v) is int and -2 <= v <= 3)
    schema_test({"type": "string", "minLength": 1, "maxLength": 2},
                ["", "a", "ab", "abc", "é", "éé", "😀", "😀a", "\x00", "\n", " a", 1, None],
                lambda v: isinstance(v, str) and 1 <= len(v) <= 2 and all(not 0xd800 <= ord(c) <= 0xdfff for c in v),
                extra=[b'"\\u0041"', b'"\\uD83D\\uDE00"', b'"\\uD83D\\uDE00x"', b'"\\uD83D\\uDE00xy"',
                       b'"\\u0000"', b'"\\/"', b'"\\uD800x"', b'"\xc0\x80"'])
    schema_test({"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 3},
                [[], [True], [False, True], [False] * 3, [True] * 4, [1], [None], {}, None],
                lambda v: isinstance(v, list) and 1 <= len(v) <= 3 and all(type(x) is bool for x in v))
    schema_test({"type": "array", "items": {"type": "null"}},
                [[], [None], [None] * 6, [True], 1],
                lambda v: isinstance(v, list) and all(x is None for x in v))
    schema_test({"type": "array", "items": {"type": "null"}, "maxItems": 0},
                [[], [None], {}, ""], lambda v: v == [])
    schema = {"type": "object", "properties": {"x": {"type": "boolean"},
              "y": {"type": "array", "items": {"type": "null"}, "maxItems": 2}, "z": {"const": 7}},
              "required": ["y"], "additionalProperties": False}
    values = [{k: v for k, v, yes in zip(("x", "y", "z"), (True, [None], 7), present) if yes}
              for present in itertools.product((False, True), repeat=3)]
    values += [{"y": []}, {"y": [None] * 3}, {"y": [1]}, {"y": [], "other": 1},
               {"y": [], "x": True}, {"x": 1, "y": []}, {"y": [], "z": 8}, [], None]
    schema_test(schema, values, lambda v: isinstance(v, dict) and "y" in v and set(v) <= {"x", "y", "z"}
                and ("x" not in v or type(v["x"]) is bool) and isinstance(v["y"], list)
                and len(v["y"]) <= 2 and all(x is None for x in v["y"]) and ("z" not in v or v["z"] == 7),
                ordered=["x", "y", "z"], extra=[b'{"y":[],"y":[]}', b'{"y": [],}', b'{"y":[]}'])
    enum = ["é", True, None, {"x": [1, 2]}]
    schema_test({"enum": enum}, enum + [False, 0, 1, "e", [1, 2], {"x": [2]}],
                lambda v: any(type(v) is type(e) and v == e for e in enum), canonical=True)
    schema_test({"type": "object", "properties": {}, "additionalProperties": False},
                [{}, {"a": 1}, [], None], lambda v: type(v) is dict and not v)
    schema_test({"type": "number"}, [0, -1, 1.5, 1e-4, None, True, "1"],
                lambda v: type(v) in (int, float), extra=[b"1e999", b"-0", b"1e+3", b"1.", b"+1"])
    print(f"Compiled constraints match independent byte/JSON laws on {total} strings")


def main(executable):
    with PreparationService() as service:
        run(executable, service)


if __name__ == "__main__":
    main(sys.argv[1])
