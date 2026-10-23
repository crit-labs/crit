"""Core constraints (Python): compiler contract, byte vocabularies from tokenizers and the host grammar checks.

Native semantic comparisons run in Core's CTest oracles (tests/core/*_constraint_oracle.py)."""
import json
import unittest
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core/python"))

from crit_core.constraints import compiler as c  # noqa: E402


class CompilerTests(unittest.TestCase):
    def compile(self, schema):
        return c.compile_language("schema", json.dumps(schema))

    def test_supported_schemas_produce_bounded_data_only_artifacts(self):
        cases = [{"type": "string", "minLength": 1, "maxLength": 4},
                 {"type": "integer", "minimum": -2, "maximum": 5},
                 {"type": "number"}, {"type": "boolean"}, {"type": "null"},
                 {"type": "array", "items": {"type": "boolean"}, "maxItems": 0},
                 {"type": "object", "properties": {"x": {"enum": [1, 2]}, "y": {"const": "a"}},
                  "required": ["x"], "additionalProperties": False},
                 {"enum": [None, True, {"x": [1, 2]}, "caf\u00e9"]}]
        for schema in cases:
            with self.subTest(schema=schema):
                artifact = self.compile(schema)
                header, *rows = artifact.splitlines()
                name, rules, count, start = header.split()
                self.assertEqual(name, "crit-byte-grammar-v1")
                self.assertEqual(int(count), len(rows))
                self.assertLess(int(start), int(rules))
                for row in rows:
                    lhs, size, *rhs = map(int, row.split())
                    self.assertLess(lhs, int(rules))
                    self.assertEqual(size, len(rhs))
                    self.assertTrue(all(0 <= s < 256 + int(rules) for s in rhs))

    def test_unknown_or_unsupported_schema_keywords_never_relax_constraints(self):
        cases = [{"type": "string", "pattern": "a", "maxLength": 3}, {"type": "string", "format": "email"},
                 {"$ref": "#/$defs/x"}, {"oneOf": [{"type": "integer"}, {"type": "number"}]},
                 {"type": "number", "maximum": 1}, {"type": "object"},
                 {"type": "array"}, {"type": ["string", "null"], "items": {"type": "null"}},
                 {"type": ["string", "string"]}, {"anyOf": []}, {"anyOf": [{"type": "string"}], "type": "string"},
                 {"$ref": "http://example.invalid/schema"}, {"$ref": "#/$defs/x", "type": "string", "$defs": {"x": {"type": "string"}}},
                 {"type": "object", "properties": {"a": {"$defs": {}, "type": "null"}}, "additionalProperties": False},
                 {"allOf": [{"type": "string"}]}, {"not": {"type": "string"}},
                 {"type": "string", "pattern": "(?=a)"}, {"type": "string", "pattern": "a*?"},
                 {"type": "string", "pattern": "\\bword"}, {"type": "string", "pattern": "[\u00e9]"},
                 {"type": "string", "pattern": "\\xe9"},
                 {"type": "object", "properties": {}, "required": ["x"], "additionalProperties": False},
                 {"type": "string", "maxLength": True}, {"type": "string", "maxLength": None},
                 {"type": "array", "items": {"type": "null"}, "maxItems": None},
                 {"enum": [1], "type": None}, {"enum": ["\ud800"]},
                 {"type": "integer", "minimum": 0},
                 {"enum": ["a"], "type": "integer"}, {"const": "a", "minLength": 2},
                 {"type": "array", "items": {"type": "null"}, "minItems": 2, "maxItems": 1}]
        for schema in cases:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                self.compile(schema)

    def test_unions_type_lists_references_and_optional_properties(self):
        from crit_core.constraints import matches
        union = self.compile({"type": "object", "properties": {
            "v": {"anyOf": [{"type": "integer"}, {"type": ["string", "null"], "maxLength": 3}]},
            "o": {"type": "boolean"}}, "required": ["v"], "additionalProperties": False})
        for raw, ok in ((b'{"v":1}', True), (b'{"v":"abc"}', True), (b'{"v":"abcd"}', False), (b'{"v":null,"o":true}', True),
                        (b'{"o":true}', False), (b'{"v":1.5}', False), (b'{"v":1,"o":false}', True)):
            self.assertEqual(matches(union, raw), ok, raw)
        tree = self.compile({"$defs": {"node": {"type": "object", "properties": {
            "v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}},
            "required": ["v", "kids"], "additionalProperties": False}}, "$ref": "#/$defs/node"})
        self.assertTrue(matches(tree, b'{"v":1,"kids":[{"v":2,"kids":[]},{"v":3,"kids":[{"v":4,"kids":[]}]}]}'))
        self.assertFalse(matches(tree, b'{"v":1,"kids":[{"v":2}]}'))
        nested = self.compile({"type": "array", "items": {"$ref": "#"}, "maxItems": 2})
        self.assertTrue(matches(nested, b"[[],[[]]]"))
        self.assertFalse(matches(nested, b"[[],[],[]]"))
        with self.assertRaises(ValueError):   # a reference cycle that never produces a value
            self.compile({"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"})

    def test_one_of_only_with_provably_disjoint_branches(self):
        from crit_core.constraints import matches
        tagged = self.compile({"oneOf": [
            {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}}, "required": ["kind", "x"],
             "additionalProperties": False},
            {"type": "object", "properties": {"kind": {"const": "b"}, "y": {"type": "string"}}, "required": ["kind", "y"],
             "additionalProperties": False}]})
        self.assertTrue(matches(tagged, b'{"kind":"a","x":1}'))
        self.assertFalse(matches(tagged, b'{"kind":"a","y":"s"}'))
        for disjoint in ({"oneOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
                         {"oneOf": [{"const": "a"}, {"enum": ["b", 1]}]}, {"oneOf": [{"enum": [True]}, {"enum": [1]}]},
                         {"oneOf": [{"type": "boolean"}, {"$ref": "#/$defs/s"}], "$defs": {"s": {"type": "string"}}}):
            self.compile(disjoint)
        for overlapping in ({"oneOf": [{"enum": [1, "a"]}, {"enum": [1.0]}]}, {"oneOf": [{"type": "string"}, {"type": "string", "maxLength": 2}]},
                            {"oneOf": [{"$ref": "#/$defs/s"}, {"type": "string"}], "$defs": {"s": {"type": "string"}}},
                            {"oneOf": [{"type": "object", "properties": {"k": {"const": "a"}}, "required": [], "additionalProperties": False},
                                       {"type": "object", "properties": {"k": {"const": "b"}}, "required": [], "additionalProperties": False}]}):
            with self.subTest(schema=overlapping), self.assertRaises(ValueError):
                self.compile(overlapping)

    def test_patterns_match_the_decoded_string_with_search_semantics(self):
        import random
        import re
        from crit_core.constraints import matches
        rng = random.Random(4)
        e_acute, line_separator, no_break = chr(0xe9), chr(0x2028), chr(0xa0)
        alphabet = ["a", "b", "c", "1", "2", '"', "\\", "\n", "\r", " ", e_acute, line_separator, no_break, "x", "-", ".", "\x01"]
        # ECMA-262 (JSON Schema's dialect) as Python regexes: '.' excludes every line terminator, '$' anchors only at the
        # end, \d and \w are ASCII, \s includes the Unicode spaces.
        dot = "[^\n\r" + line_separator + chr(0x2029) + "]"
        space = "[\t\n\x0b\x0c\r " + "".join(chr(c) for c in (0xa0, 0x1680, *range(0x2000, 0x200b), 0x2028, 0x2029, 0x202f,
                                                               0x205f, 0x3000, 0xfeff)) + "]"
        cases = ((r"^a(b|c)*[0-9]{1,3}$", r"\Aa(b|c)*[0-9]{1,3}\Z"), (r"b.c", "b" + dot + "c"), (r"^[^a-c]+$", r"\A[^a-c]+\Z"),
                 (r"\d+", "[0-9]+"), (r"^.*$", r"\A" + dot + r"*\Z"), (r"a\.b", r"a\.b"), (r"^x?$", r"\Ax?\Z"), ('"', '"'),
                 (r"^\w+\s\w+$", r"\A[A-Za-z0-9_]+" + space + r"[A-Za-z0-9_]+\Z"), (r"\\", r"\\"), (e_acute, e_acute),
                 (r"^[\x00-\x1f]$", r"\A[\x00-\x1f]\Z"))
        for pattern, ecma in cases:
            artifact = self.compile({"type": "string", "pattern": pattern})
            reference = re.compile(ecma)
            for _ in range(250):
                text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
                # json.dumps(..., ensure_ascii=False) writes the canonical encoding the compiled pattern generates.
                encoded = json.dumps(text, ensure_ascii=False).encode("utf-8")
                self.assertEqual(matches(artifact, encoded), reference.search(text) is not None, (pattern, text))

    def test_schema_dialect_duplicates_nonfinite_and_compiler_limits(self):
        for source in ('{"type":"string","type":"integer"}', '{"enum":[NaN]}',
                       '{"$schema":"unknown","type":"string"}', '['):
            with self.assertRaises(ValueError):
                c.compile_language("schema", source)
        for schema in ({"type": "string", "maxLength": 1025},
                       {"type": "integer", "minimum": 0, "maximum": 5000}):
            with self.assertRaises(ValueError):
                self.compile(schema)
        with self.assertRaises(ValueError):
            c.compile_language("regex", "a" * (c.MAX_SOURCE + 1))

    def test_regex_and_recursive_ebnf_supported_and_invalid_constructs_rejected(self):
        for expression in (r"^a(b|c)*[0-9]{1,3}$", r"\x00|café", r"\d+\s?\w", ""):
            self.assertTrue(c.compile_language("regex", expression).startswith("crit-byte"))
        self.assertTrue(c.compile_language("ebnf", 'root ::= "(" root ")" root | ""').startswith("crit-byte"))
        self.assertTrue(c.compile_language("ebnf", 'root ::= item+\nitem ::= [a-z] # bytes').startswith("crit-byte"))
        for kind, source in (("regex", r"(?=a)a"), ("regex", r"(a)\1"), ("regex", "a{2000}"),
                             ("regex", "[]"), ("regex", "[^]"), ("regex", "[z-a]"), ("regex", r"\012"),
                             ("regex", "(" * 65), ("ebnf", 'root ::= missing'),
                             ("ebnf", 'root ::= root'), ("ebnf", 'x ::= "a"'),
                             ("ebnf", 'root ::= "a"\nroot ::= "b"')):
            with self.subTest(kind=kind, source=source), self.assertRaises(ValueError):
                c.compile_language(kind, source)

    def test_cache_binds_source_language_and_exact_tokenizer_bytes(self):
        cache = c.CompilerCache()
        args = dict(tokenizer_identity="v1", pieces=(b"a", b"b", b""), eos=2)
        a = cache.compile("regex", "a", **args)
        self.assertIs(cache.compile("regex", "a", **args), a)
        for change in ({"tokenizer_identity": "v2"}, {"pieces": (b"aa", b"b", b"")},
                       {"pieces": (b"a", b"", b"b"), "eos": 1}):
            self.assertIsNot(cache.compile("regex", "a", **(args | change)), a)
        self.assertIsNot(cache.compile("regex", "^a$", **args), a)
        self.assertIsNot(cache.compile("ebnf", 'root ::= "a"', **args), a)
        self.assertIsNot(c.CompilerCache().compile("regex", "a", **args), a)

    def test_cache_eviction_size_limit_and_failed_compilation_are_transactional(self):
        cache = c.CompilerCache(max_entries=1)
        args = dict(tokenizer_identity="v1", pieces=(b"a", b"b", b""), eos=2)
        a = cache.compile("regex", "a", **args)
        size = cache.bytes
        with self.assertRaises(ValueError):
            cache.compile("regex", "[", **args)
        self.assertEqual(cache.bytes, size)
        self.assertIs(cache.compile("regex", "a", **args), a)
        cache.compile("regex", "b", **args)
        self.assertIsNot(cache.compile("regex", "a", **args), a)
        tiny = c.CompilerCache(max_bytes=1)
        tiny.compile("regex", "a", **args)
        self.assertEqual(tiny.bytes, 0)
        for change in ({"eos": True}, {"pieces": (b"a", b"b", b"x")}, {"pieces": ("a", b"b", b"")}):
            with self.assertRaises(ValueError):
                cache.compile("regex", "a", **(args | change))


class VocabularyTests(unittest.TestCase):
    """The byte table comes from the tokenizer's vocabulary and decoder, independently of decode()."""
    def check(self, data, texts, eos_name):
        from crit_core.constraints import byte_vocabulary
        from crit_core.text import Tokenizer
        tokenizer = Tokenizer(data, special_tokens={"eos_token": eos_name})
        vocabulary = byte_vocabulary(tokenizer)
        self.assertEqual(len(vocabulary.pieces), tokenizer.vocab_size)
        self.assertEqual(vocabulary.pieces[tokenizer.eos_token_id], b"")
        for token in tokenizer.special_ids:
            self.assertEqual(vocabulary.pieces[token], b"")
        # A continuation's bytes: decoders may strip a space only at the start of the whole text.
        lead = tokenizer.encode("x", add_special_tokens=False)
        start = tokenizer.decode(lead).encode("utf-8")
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            whole = tokenizer.decode(lead + ids).encode("utf-8")
            self.assertTrue(whole.startswith(start))
            self.assertEqual(b"".join(vocabulary.pieces[i] for i in ids), whole[len(start):], text)
        self.assertEqual(byte_vocabulary(tokenizer).identity, vocabulary.identity)
        return tokenizer, vocabulary

    def test_byte_level_and_sentencepiece_pieces_match_decoded_text(self):
        import core_text_fixtures as fx
        texts = list(fx.CORPUS) + ["{\"reply\": \"Grüße 😀\"}", " leading", "<think>x</think>"]
        tokenizer, vocabulary = self.check(fx.byte_level_json(), texts, "<|eot_id|>")
        self.assertEqual(vocabulary.pieces[tokenizer.token_to_id("<think>")], b"<think>")
        tokenizer, vocabulary = self.check(fx.sentencepiece_json(), texts, "</s>")
        self.assertEqual(vocabulary.pieces[tokenizer.token_to_id("<0x41>")], b"A")

    def test_explicit_eos_unknown_decoders_and_identity(self):
        from crit_core.constraints import byte_vocabulary, from_pieces
        from crit_core.text import TextError, Tokenizer
        import core_text_fixtures as fx
        tokenizer = Tokenizer(fx.byte_level_json())
        with self.assertRaises(TextError):
            byte_vocabulary(tokenizer)  # no eos_token and no explicit id
        other = tokenizer.token_to_id("<|im_end|>")
        self.assertEqual(byte_vocabulary(tokenizer, other).eos, other)
        data = json.loads(fx.byte_level_json())
        data["decoder"] = {"type": "WordPiece", "prefix": "##", "cleanup": True}
        with self.assertRaises(TextError):
            byte_vocabulary(Tokenizer(json.dumps(data)), other)
        self.assertNotEqual(from_pieces((b"a", b""), 1).identity, from_pieces((b"a", b"", b""), 1).identity)
        for pieces, eos in (((b"a",), 0), ((), 0), (("a", b""), 1)):
            with self.assertRaises(TextError):
                from_pieces(pieces, eos)


class GrammarCheckTests(unittest.TestCase):
    """The host recognizer follows the native GrammarMatcher; its corpus oracle compares against the native one."""
    def test_regex_ebnf_and_schema_languages(self):
        import itertools
        import re
        from crit_core.constraints import ByteGrammar, GrammarState, matches
        alphabet = b"ab()01"
        short = [bytes(row) for n in range(5) for row in itertools.product(alphabet, repeat=n)]
        for expression in (r"a(b|c)*[0-9]{1,3}", r"(ab|a)?", r"[^a-c]{0,3}", r"(a?)*", r".*"):
            grammar = ByteGrammar.load(c.compile_language("regex", expression))
            reference = re.compile(expression.encode())
            for raw in short:
                self.assertEqual(matches(grammar, raw), reference.fullmatch(raw) is not None, (expression, raw))
        balanced = c.compile_language("ebnf", 'root ::= "(" root ")" root | ""')
        self.assertTrue(matches(balanced, b"(()())"))
        self.assertFalse(matches(balanced, b"(()"))
        schema = {"type": "object", "properties": {"x": {"type": "boolean"}, "n": {"type": "integer"}},
                  "required": ["x"], "additionalProperties": False}
        artifact = c.compile_language("schema", json.dumps(schema))
        for raw, ok in ((b'{"x":true}', True), (b'{"x":false,"n":-12}', True), (b'{"n":1,"x":true}', False),
                        (b'{"x": true}', False), (b'{"n":1}', False), (b'{"x":true}x', False)):
            self.assertEqual(matches(artifact, raw), ok, raw)
        state = GrammarState(ByteGrammar.load(artifact)).advance(b'{"x":t')
        self.assertIsNotNone(state)
        self.assertFalse(state.complete)
        self.assertIsNone(state.advance(b"x"))
        self.assertTrue(state.advance(b"rue}").complete)

    def test_artifact_limits_and_json_objects(self):
        from crit_core.constraints import ByteGrammar, GrammarError, GrammarState, json_object_matches
        for text in ("", "crit-byte-grammar-v0 1 1 0", "crit-byte-grammar-v1 1 1 0 0 1 256",
                     "crit-byte-grammar-v1 1 1 0 0 0 garbage", "crit-byte-grammar-v1 1 1 0 0 1 -1",
                     "crit-byte-grammar-v1 2 1 0 0 0"):
            with self.subTest(text=text), self.assertRaises(GrammarError):
                ByteGrammar.load(text)
        tight = GrammarState(ByteGrammar.load("crit-byte-grammar-v1 1 2 0\n0 0\n0 2 97 256\n"), states_per_byte=2)
        with self.assertRaises(GrammarError):
            tight.advance(b"a")
        for raw, ok in ((b' {"a":[1,2.5e3,null]}\n', True), (b"[]", False), (b'{"a":NaN}', False),
                        (b'{"a":"\xc3"}', False), (b'{"a":1}{}', False), (b'{"a":"\n"}', False)):
            self.assertEqual(json_object_matches(raw), ok, raw)

    def test_package_imports_only_the_standard_library(self):
        import subprocess
        code = ("import sys; before = set(sys.modules); sys.path.insert(0, sys.argv[1]); import crit_core.constraints; "
                "loaded = {n.partition('.')[0] for n in set(sys.modules) - before}; "
                "loaded -= set(sys.stdlib_module_names) | {'crit_core'}; sys.exit(', '.join(sorted(loaded)) or None)")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "core/python")],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
