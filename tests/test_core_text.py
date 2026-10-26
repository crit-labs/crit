"""Core text layer: tokenizer, chat templates, generation config, detokenizer, stop strings and streams (offline)."""
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import tokenizers
import tokenizers.decoders

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))

import core_text_fixtures as fx  # noqa: E402
from crit_core.files import CallableFiles, DirectoryFiles, MappingFiles  # noqa: E402
from crit_core.model import ConfigError, load_model_config  # noqa: E402
from crit_core.text import (ChatTemplate, ChatTemplates, GenerationConfig, IncrementalDetokenizer,  # noqa: E402
                            ModelTextError, StopMatcher, TemplateError, TextError, TextStream, TokenError, Tokenizer,
                            legacy_eos_ids, load_model_text, openai_messages, template_sources)
from crit_core.text import detokenize as detokenize_module, stops as stops_module  # noqa: E402
from crit_core.text.stream import TextDelta  # noqa: E402
import eval_v2_rules as v2  # noqa: E402  crit-eval-v2's stop_ids, the oracle legacy_eos_ids was ported from

FFFD = "�"
CLOCK = datetime(2026, 10, 15, 9, 30)
WINDOW_LIMIT = 5 + 2 * detokenize_module.HOLD_LIMIT  # context ids plus at most two runs without a clean boundary


def byte_tokenizer(**options):
    return Tokenizer.from_bytes(fx.byte_level_json(), special_tokens={"bos_token": "<|begin_of_text|>",
                                                                      "eos_token": "<|eot_id|>"}, **options)


def sp_tokenizer():
    return Tokenizer.from_bytes(fx.sentencepiece_json(), special_tokens={"bos_token": "<s>", "eos_token": "</s>"})


def straddle_tokenizer():
    return Tokenizer.from_bytes(fx.straddling_json(), special_tokens={"eos_token": "<|eot|>"})


def mapping(files):
    return MappingFiles({name: fx.file_bytes(value) for name, value in files.items()})


class LookAlike:
    """Has ModelFiles' read and names but is not one: sources of this kind are refused everywhere."""

    def __init__(self):
        self.source = mapping(fx.llama_files())

    def read(self, name, max_bytes=None):
        return self.source.read(name, max_bytes)

    def names(self, prefix=""):
        return self.source.names(prefix)


def first_stop(text, stops):
    """Brute force: (start, index) of the first completed stop, ties to the earliest start then the lowest index."""
    for end in range(1, len(text) + 1):
        found = [(end - len(stop), index) for index, stop in enumerate(stops)
                 if end >= len(stop) and text[end - len(stop):end] == stop]
        if found:
            return min(found)
    return None


def holdback(text, stops):
    """Brute force: length of the longest suffix of text that is a proper prefix of some stop."""
    best = 0
    for stop in stops:
        for size in range(1, min(len(stop) - 1, len(text)) + 1):
            if text.endswith(stop[:size]):
                best = max(best, size)
    return best


def chunks(rng, items, largest=4):
    position = 0
    while position < len(items):
        size = rng.randint(1, largest)
        yield items[position:position + size]
        position += size


class LazyImportTests(unittest.TestCase):
    def test_importing_the_package_and_the_stop_matcher_loads_neither_tokenizers_nor_jinja2(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import crit_core.text; "
                "from crit_core.text import StopMatcher, TextError, openai_messages, GenerationConfig; "
                "StopMatcher(['x']).push('abc'); "
                "print(sorted(name for name in ('tokenizers', 'jinja2') if name in sys.modules))")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "core/python")],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_exports_resolve_lazily_and_unknown_names_raise_attribute_error(self):
        import crit_core.text as text
        self.assertIs(text.Tokenizer, Tokenizer)
        self.assertIn("load_model_text", dir(text))
        with self.assertRaises(AttributeError):
            text.NoSuchName


class StopMatcherTests(unittest.TestCase):
    def test_the_first_completed_stop_wins_whatever_the_chunking(self):
        for pieces in (["abcd"], ["a", "b", "c", "d"], ["ab", "cd"], ["abc", "d"]):
            with self.subTest(pieces=pieces):
                matcher = StopMatcher(["abcd", "bc"])
                released = "".join(matcher.push(piece) for piece in pieces)
                self.assertEqual((released, matcher.stopped, matcher.stop_index, matcher.stop_offset), ("a", "bc", 1, 1))
                self.assertEqual(matcher.push("more"), "")
                self.assertEqual(matcher.finish(), "")

    def test_ties_at_the_same_end_go_to_the_earliest_start_then_the_lowest_index(self):
        matcher = StopMatcher(["bc", "abc", "abc"])
        self.assertEqual(matcher.push("xabcy"), "x")
        self.assertEqual((matcher.stopped, matcher.stop_index, matcher.stop_offset), ("abc", 1, 1))

    def test_only_the_longest_suffix_that_can_begin_a_stop_is_held_back(self):
        matcher = StopMatcher(["STOP", "END"])
        self.assertEqual(matcher.push("hello"), "hello")
        self.assertEqual(matcher.push(" S"), " ")
        self.assertEqual(matcher.held, "S")
        self.assertEqual(matcher.push("TA"), "STA")
        self.assertEqual(matcher.push("E"), "")
        self.assertEqual(matcher.held, "E")
        self.assertEqual(matcher.finish(), "E")
        self.assertIsNone(matcher.stopped)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            matcher.push("x")
        with self.assertRaisesRegex(RuntimeError, "finished"):
            matcher.finish()

    def test_random_chunkings_agree_with_a_brute_force_oracle(self):
        rng = random.Random(20261015)
        for case in range(3000):
            alphabet = "abc" if case % 3 else "ab日😀"
            stops = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 5))) for _ in range(rng.randint(0, 4))]
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            expected = first_stop(text, stops)
            matcher = StopMatcher(stops)
            released = ""
            fed = 0
            for piece in chunks(rng, text, 6):
                released += matcher.push(piece)
                fed += len(piece)
                if matcher.stopped is None:
                    self.assertEqual(len(released), fed - holdback(text[:fed], stops), (stops, text, fed))
            released += matcher.finish()
            with self.subTest(stops=stops, text=text):
                if expected is None:
                    self.assertEqual((released, matcher.stopped), (text, None))
                else:
                    start, index = expected
                    self.assertEqual((released, matcher.stop_index, matcher.stop_offset), (text[:start], index, start))
                    self.assertEqual(matcher.stopped, stops[index])

    def test_advance_serves_crit_serve_cumulative_text_with_minimal_holdback(self):
        matcher = StopMatcher(["STOP", "END"])
        self.assertEqual(matcher.advance("hello S"), "hello ")
        self.assertEqual(matcher.advance("hello ST"), "")
        self.assertEqual(matcher.advance("hello STOP and END"), "")
        self.assertEqual(matcher.stopped, "STOP")
        self.assertEqual(matcher.advance("hello STOP and END more", final=True), "")
        plain = StopMatcher([])
        self.assertEqual(plain.advance("abc"), "abc")
        self.assertEqual(plain.advance("abcd", final=True), "d")
        self.assertEqual(plain.advance("abcde"), "")
        held = StopMatcher(["XY"])
        self.assertEqual(held.advance("aX"), "a")
        self.assertEqual(held.advance("aX", final=True), "X")

    def test_advance_refuses_cumulative_text_that_does_not_extend_what_it_saw(self):
        matcher = StopMatcher(["STOP"])
        matcher.advance("hello ST")
        with self.assertRaisesRegex(TextError, "extend"):
            matcher.advance("hello")
        with self.assertRaisesRegex(TextError, "extend"):
            matcher.advance("hello SX more")

    def test_invalid_stop_lists_are_rejected(self):
        for stops, message in (("STOP", "not one string"), ([""], "non-empty"), ([1], "non-empty"),
                               (["x"] * 17, "at most 16"), (["é" * 129], "UTF-8 bytes"), (["\ud800"], "Unicode"),
                               (5, "list of strings")):
            with self.subTest(stops=stops), self.assertRaisesRegex(TextError, message):
                StopMatcher(stops)
        self.assertEqual(len(StopMatcher(["é" * 128]).stops), 1)
        with self.assertRaisesRegex(TextError, "string"):
            StopMatcher(["x"]).push(b"x")

    def test_an_adversarial_long_input_is_scanned_once(self):
        stop = "a" * (stops_module.MAX_STOP_BYTES - 1) + "b"
        matcher = StopMatcher([stop, "aab"])
        released = matcher.push("a" * 200000)
        self.assertEqual(len(released), 200000 - (len(stop) - 1))
        self.assertEqual(matcher.push("b"), "")
        self.assertEqual((matcher.stopped, matcher.stop_offset), (stop, 200000 - len(stop) + 1))
        short = StopMatcher([stop, "aac"])
        self.assertEqual(short.push("a" * 200000) + short.push("c"), "a" * (200000 - 2))
        self.assertEqual((short.stopped, short.stop_offset), ("aac", 200000 - 2))


class TokenizerTests(unittest.TestCase):
    def test_vocab_size_is_max_id_plus_one_and_ids_must_be_contiguous(self):
        for tokenizer in (byte_tokenizer(), sp_tokenizer()):
            vocab = tokenizer.get_vocab()
            self.assertEqual(tokenizer.vocab_size, max(vocab.values()) + 1)
            self.assertEqual(tokenizer.vocab_size, tokenizer.backend.get_vocab_size(with_added_tokens=True))
        with self.assertRaisesRegex(ModelTextError, "not contiguous"):
            Tokenizer.from_bytes(fx.gapped_json())

    def test_added_tokens_special_ids_and_bos_eos_ids(self):
        tokenizer = byte_tokenizer()
        think = tokenizer.token_to_id("<think>")
        self.assertFalse(tokenizer.added_tokens[think].special)
        self.assertEqual(tokenizer.added_tokens[think].content, "<think>")
        self.assertEqual(tokenizer.special_ids, frozenset(tokenizer.token_to_id(t) for t in fx.BYTE_SPECIALS))
        self.assertEqual(tokenizer.bos_token_id, tokenizer.token_to_id("<|begin_of_text|>"))
        self.assertEqual(tokenizer.eos_token_id, tokenizer.token_to_id("<|eot_id|>"))
        self.assertIsNone(Tokenizer.from_bytes(fx.byte_level_json()).bos_token_id)
        with self.assertRaisesRegex(TextError, "unknown special token"):
            Tokenizer.from_bytes(fx.byte_level_json(), special_tokens={"boss_token": "x"})

    def test_encode_requires_add_special_tokens_as_a_keyword(self):
        tokenizer = byte_tokenizer()
        with self.assertRaises(TypeError):
            tokenizer.encode("Hello")
        with self.assertRaises(TypeError):
            tokenizer.encode("Hello", True)
        plain = tokenizer.encode("Hello", add_special_tokens=False)
        self.assertEqual(tokenizer.encode("Hello", add_special_tokens=True), [tokenizer.bos_token_id] + plain)
        with self.assertRaisesRegex(TextError, "add_special_tokens"):
            tokenizer.encode("Hello", add_special_tokens=1)
        with self.assertRaisesRegex(TextError, "string"):
            tokenizer.encode(b"Hello", add_special_tokens=False)
        with self.assertRaisesRegex(TextError, "cannot encode"):
            tokenizer.encode("bad \ud800", add_special_tokens=False)

    def test_decode_keeps_special_tokens_by_default_and_validates_every_id(self):
        tokenizer = byte_tokenizer()
        ids = tokenizer.encode("<|begin_of_text|>Hi<think>", add_special_tokens=False)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        self.assertEqual(tokenizer.decode(ids), "<|begin_of_text|>Hi<think>")
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), "Hi<think>")
        for bad in (-1, tokenizer.vocab_size, True, 1.0, "3", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(TokenError, "position 1"):
                tokenizer.decode([5, bad])
        with self.assertRaisesRegex(TokenError, "sequence"):
            tokenizer.decode(5)
        self.assertEqual(tokenizer.validate(range(3)), [0, 1, 2])
        self.assertEqual(tokenizer.id_to_token(tokenizer.bos_token_id), "<|begin_of_text|>")
        with self.assertRaises(TokenError):
            tokenizer.id_to_token(tokenizer.vocab_size)
        self.assertIsNone(tokenizer.token_to_id("no such token text"))

    def test_split_special_tokens_uses_a_second_backend_and_never_mutates_the_shared_one(self):
        tokenizer = byte_tokenizer()
        text = "<|im_start|>user"
        parsed = tokenizer.encode(text, add_special_tokens=False)
        split = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
        self.assertEqual(parsed[0], tokenizer.token_to_id("<|im_start|>"))
        self.assertNotIn(tokenizer.token_to_id("<|im_start|>"), split)
        self.assertEqual(tokenizer.decode(split), text)
        self.assertFalse(tokenizer.backend.encode_special_tokens)
        self.assertEqual(tokenizer.encode(text, add_special_tokens=False), parsed)
        hardened = byte_tokenizer(split_special_tokens=True)
        self.assertEqual(hardened.encode(text, add_special_tokens=False), split)
        self.assertEqual(hardened.encode(text, add_special_tokens=False, split_special_tokens=False), parsed)

    def test_truncation_and_padding_saved_in_tokenizer_json_are_switched_off(self):
        tokenizer = Tokenizer.from_bytes(fx.truncating_json())
        text = "The quick brown fox jumps over the lazy dog."
        self.assertGreater(len(tokenizer.encode(text, add_special_tokens=False)), 4)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)), text)

    def test_special_tokens_come_from_strings_or_added_token_objects(self):
        files = mapping({"tokenizer.json": fx.byte_level_json(),
                         "tokenizer_config.json": {"bos_token": {"content": "<|begin_of_text|>", "lstrip": False},
                                                   "eos_token": "<|eot_id|>", "pad_token": None,
                                                   "added_tokens_decoder": {}},
                         "special_tokens_map.json": {"bos_token": "IGNORED"}})
        tokenizer = Tokenizer.from_files(files)
        self.assertEqual(dict(tokenizer.special_tokens), {"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"})
        self.assertNotIn("special_tokens_map.json", [record.path for record in files.records])

    def test_special_tokens_map_is_read_for_legacy_configs_and_takes_precedence(self):
        files = mapping({"tokenizer.json": fx.sentencepiece_json(),
                         "tokenizer_config.json": {"bos_token": "<s>", "eos_token": "<s>", "unk_token": "<unk>"},
                         "special_tokens_map.json": {"eos_token": {"content": "</s>"}, "unk_token": None,
                                                     "pad_token": "<unk>", "additional_special_tokens": ["x"]}})
        tokenizer = Tokenizer.from_files(files)
        self.assertEqual(dict(tokenizer.special_tokens), {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<unk>"})
        self.assertIn("special_tokens_map.json", [record.path for record in files.records])

    def test_malformed_tokenizer_files_raise_model_text_errors(self):
        cases = (({}, "tokenizer.json is missing"),
                 ({"tokenizer.json": "{not json"}, "not a valid tokenizer"),
                 ({"tokenizer.json": b"\xff\xfe"}, "not UTF-8"),
                 ({"tokenizer.json": fx.byte_level_json(), "tokenizer_config.json": {"bos_token": 5}}, "bos_token"),
                 ({"tokenizer.json": fx.byte_level_json(), "tokenizer_config.json": "[1]"}, "JSON object"),
                 ({"tokenizer.json": fx.byte_level_json(), "tokenizer_config.json": '{"a": 1, "a": 2}'}, "duplicate"))
        for files, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ModelTextError, message):
                Tokenizer.from_files(mapping(files))

    def test_token_bytes_read_the_vocabulary_of_byte_level_and_sentencepiece_models(self):
        for data in (fx.byte_level_json(), fx.sentencepiece_json()):
            tokenizer = Tokenizer(data)
            expected = fx.token_bytes(tokenizer, range(tokenizer.vocab_size), False)
            self.assertEqual(list(tokenizer.token_bytes()), expected)
            self.assertIs(tokenizer.token_bytes(), tokenizer.token_bytes())
            ids = tokenizer.encode("Grüße 😀 tides", add_special_tokens=False)
            self.assertIn("Grüße 😀 tides".encode(), b"".join(tokenizer.token_bytes()[t] for t in ids))


class TemplateTests(unittest.TestCase):
    def test_tojson_keeps_key_order_and_non_ascii_and_passes_indent_and_separators(self):
        template = ChatTemplate("{{ value | tojson }}|{{ value | tojson(indent=2) }}|"
                                "{{ value | tojson(separators=(',', ':')) }}|{{ value | tojson(sort_keys=true) }}")
        value = {"z": "<Grüße & 'quotes'>", "a": [1, 2]}
        expected = "|".join((json.dumps(value, ensure_ascii=False), json.dumps(value, ensure_ascii=False, indent=2),
                             json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                             json.dumps(value, ensure_ascii=False, sort_keys=True)))
        self.assertEqual(template.render([{"role": "user", "content": ""}], value=value), expected)

    def test_bos_token_comes_from_special_tokens_and_caller_variables_override_it(self):
        template = ChatTemplate(fx.GEMMA, special_tokens={"bos_token": "<s>", "eos_token": "</s>"})
        messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi "}]
        self.assertEqual(template.render(messages, add_generation_prompt=True),
                         "<s><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\n")
        self.assertTrue(template.render(messages, bos_token="[B]").startswith("[B]<start_of_turn>"))
        self.assertTrue(ChatTemplate(fx.GEMMA).render(messages).startswith("<start_of_turn>"))

    def test_loopcontrols_break_and_continue_work(self):
        messages = [{"role": "user", "content": "a"}, {"role": "skip", "content": "b"}, {"role": "user", "content": "c"},
                    {"role": "stop", "content": "d"}, {"role": "user", "content": "e"}]
        self.assertEqual(ChatTemplate(fx.LOOPS).render(messages), "a;c;")

    def test_generation_tag_records_assistant_character_spans(self):
        template = ChatTemplate(fx.LLAMA3, special_tokens={"bos_token": "<|begin_of_text|>"}, clock=None)
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there "},
                    {"role": "user", "content": "More"}, {"role": "assistant", "content": "Sure."}]
        spans = ["stale"]
        rendered = template.render(messages, assistant_spans=spans)
        self.assertEqual([rendered[start:end] for start, end in spans], ["Hello there<|eot_id|>", "Sure.<|eot_id|>"])
        self.assertEqual(template.render(messages), rendered)
        with self.assertRaisesRegex(TextError, "list"):
            template.render(messages, assistant_spans=())

    def test_spans_are_recorded_per_render_across_threads(self):
        template = ChatTemplate(fx.LLAMA3, special_tokens={"bos_token": "<|begin_of_text|>"}, clock=None)
        results, errors = {}, []

        def work(number):
            try:
                messages = [{"role": "user", "content": "q" * number}, {"role": "assistant", "content": "a" * number}]
                for _ in range(50):
                    spans = []
                    rendered = template.render(messages, assistant_spans=spans)
                    assert [rendered[s:e] for s, e in spans] == ["a" * number + "<|eot_id|>"], spans
                results[number] = True
            except Exception as error:  # reported below
                errors.append(error)
        threads = [threading.Thread(target=work, args=(number,)) for number in range(1, 9)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)

    def test_strftime_now_uses_the_injected_clock_and_is_undefined_without_one(self):
        messages = [{"role": "user", "content": "Hi"}]
        dated = ChatTemplate(fx.LLAMA3, clock=lambda: CLOCK).render(messages)
        self.assertIn("Today Date: 15 Oct 2026\n", dated)
        self.assertIn("Today Date: 01 Jan 2000\n", ChatTemplate(fx.LLAMA3, clock=None).render(messages))
        self.assertIn("Today Date: 02 Feb 2020\n", ChatTemplate(fx.LLAMA3, clock=None).render(messages,
                                                                                            date_string="02 Feb 2020"))
        self.assertEqual(ChatTemplate("{{ strftime_now('%Y') }}").render(messages), str(datetime.now().year))
        with self.assertRaisesRegex(TextError, "clock"):
            ChatTemplate("x", clock="today")

    def test_raise_exception_and_sandbox_violations_become_template_errors(self):
        gemma = ChatTemplate(fx.GEMMA, name="gemma", special_tokens={"bos_token": "<s>"})
        with self.assertRaises(TemplateError) as caught:
            gemma.render([{"role": "assistant", "content": "x"}])
        self.assertIn("Conversation roles must alternate user/assistant/user/assistant/...", str(caught.exception))
        self.assertEqual((caught.exception.template_name, caught.exception.lineno), ("gemma", 11))
        for source, message in (("{{ messages.append(1) }}", "sandbox"), ("{{ ''.__class__.__mro__ }}", "sandbox"),
                                ("{{ (1).__class__.__name__ }}", "sandbox"),
                                ("{{ missing | tojson }}", "not JSON serializable"),
                                ("{{ 1 / 0 }}", "ZeroDivisionError"), ("{{ missing.attribute }}", "undefined"),
                                ("{{ 'x' * 100000000 }}", "sandbox"), ("{{ 7 ** 100000 }}", "sandbox")):
            with self.subTest(source=source), self.assertRaisesRegex(TemplateError, message):
                ChatTemplate(source).render([{"role": "user", "content": "x"}])

    def test_the_render_limit_refuses_long_output_and_oversized_single_results(self):
        messages = [{"role": "user", "content": "x"}]
        with self.assertRaisesRegex(TemplateError, "rendered text exceeds 67108864 characters"):
            ChatTemplate("{% for i in range(100000) %}{{ 'x' * 1000 }}{% endfor %}").render(messages)
        for source, message in (("{{ (range(100000) | list) | join('x' * 1000) }}", "join exceeds"),
                                ("{{ '' | center(2 ** 62) }}", "center exceeds"),
                                ("{{ 'ab' | indent(2 ** 62, true) }}", "indent exceeds"),
                                ("{{ ('a' * 40000000) | replace('a', 'bb') }}", "replace exceeds"),
                                ("{% set ns = namespace(s='x' * 40000000) %}{{ ns.s + ns.s }}", "concatenation"),
                                ("{{ [0] * 100000000 }}", "repetition exceeds"),
                                ("{{ '%9999999999999999d' % 1 }}", "ran out of memory")):
            with self.subTest(source=source), self.assertRaisesRegex(TemplateError, message):
                ChatTemplate(source).render(messages)
        self.assertEqual(ChatTemplate("{{ [1, 2] | join('-') }}|{{ 'a' | center(3) }}|{{ 'a\nb' | indent(2) }}|"
                                      "{{ 'aXa' | replace('X', 'yy') }}|{{ 'a' + 'b' }}|{{ [1] + [2] }}|"
                                      "{{ [{'n': 'p'}, {'n': 'q'}] | join(',', attribute='n') }}").render(messages),
                         "1-2| a |a\n  b|ayya|ab|[1, 2]|p,q")

    def test_syntax_errors_name_the_template_and_line(self):
        with self.assertRaises(TemplateError) as caught:
            ChatTemplate("line one\n{% if x %}\n{% unknowntag %}\n", name="broken")
        self.assertEqual((caught.exception.template_name, caught.exception.lineno), ("broken", 3))
        self.assertIn("'broken'", str(caught.exception))
        self.assertIn("line 3", str(caught.exception))
        with self.assertRaisesRegex(TextError, "source"):
            ChatTemplate(None)

    def test_continue_final_message_cuts_string_content_at_the_open_end(self):
        messages = [{"role": "user", "content": "Write"}, {"role": "assistant", "content": "Once upon a "}]
        qwen = ChatTemplate(fx.QWEN3)
        self.assertEqual(qwen.render(messages, continue_final_message=True),
                         "<|im_start|>user\nWrite<|im_end|>\n<|im_start|>assistant\nOnce upon a ")
        llama = ChatTemplate(fx.LLAMA3, special_tokens={"bos_token": "<B>"}, clock=None)
        self.assertTrue(llama.render(messages, continue_final_message="content").endswith(
            "<|start_header_id|>assistant<|end_header_id|>\n\nOnce upon a"))
        self.assertEqual(messages[-1]["content"], "Once upon a ")

    def test_continue_final_message_handles_list_content_and_rejects_bad_combinations(self):
        gemma = ChatTemplate(fx.GEMMA, special_tokens={"bos_token": "<s>"})
        messages = [{"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello"}, {"type": "image"},
                                                      {"type": "text", "text": "and then "}]}]
        self.assertEqual(gemma.render(messages, continue_final_message=True),
                         "<s><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHelloand then")
        self.assertEqual(messages[-1]["content"][2]["text"], "and then ")
        cases = ((dict(add_generation_prompt=True, continue_final_message=True), "cannot be combined"),
                 (dict(continue_final_message=True, assistant_spans=[]), "cannot be combined"),
                 (dict(continue_final_message="reasoning"), "never mentions"),
                 (dict(continue_final_message=""), "field name"))
        for options, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(TextError, message):
                gemma.render(messages, **options)
        with self.assertRaisesRegex(TemplateError, "no text part"):
            gemma.render([{"role": "user", "content": [{"type": "image"}]}], continue_final_message=True)
        with self.assertRaisesRegex(TemplateError, "no \"content\""):
            gemma.render([{"role": "user"}], continue_final_message=True)
        with self.assertRaisesRegex(TemplateError, "does not appear"):
            ChatTemplate("{{ messages[0].content[:2] }}").render([{"role": "user", "content": "abcdef"}],
                                                                 continue_final_message=True)
        with self.assertRaisesRegex(TemplateError, "empty conversation"):
            gemma.render([])

    def test_render_validates_messages_tools_and_documents(self):
        template = ChatTemplate("{% for d in documents %}{{ d.title }};{% endfor %}{{ tools | length }}")
        self.assertEqual(template.render([{"role": "user", "content": ""}], documents=[{"title": "A"}, {"title": "B"}],
                                         tools=[{"type": "function"}]), "A;B;1")
        for options, message in ((dict(tools=[lambda: None]), r"tools\[0\]"), (dict(documents="doc"), "documents"),
                                 (dict(add_generation_prompt="yes"), "add_generation_prompt")):
            with self.subTest(options=options), self.assertRaisesRegex(TextError, message):
                template.render([{"role": "user", "content": ""}], **options)
        for messages in ("hi", {"role": "user"}, ["hi"]):
            with self.subTest(messages=messages), self.assertRaises(TextError):
                template.render(messages)

    def test_qwen_style_tool_prompt_keeps_insertion_order_and_thinking_variables(self):
        tools = [{"type": "function", "function": {"name": "grep", "description": "Find <p> in Grüße",
                                                   "parameters": {"z": 1, "a": 2}}}]
        messages = openai_messages([
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Find "}, {"type": "text", "text": "it"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x", "path": "é"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "found"}])
        rendered = ChatTemplate(fx.QWEN3).render(messages, tools=tools, add_generation_prompt=True, enable_thinking=False)
        self.assertIn('\n{"type": "function", "function": {"name": "grep", "description": "Find <p> in Grüße", '
                      '"parameters": {"z": 1, "a": 2}}}\n</tools>', rendered)
        self.assertIn('{"name": "grep", "arguments": {"pattern": "x", "path": "é"}}', rendered)
        self.assertIn("<|im_start|>user\nFind it<|im_end|>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertTrue(ChatTemplate(fx.QWEN3).render(messages, add_generation_prompt=True).endswith(
            "<|im_start|>assistant\n"))


class TemplateDiscoveryTests(unittest.TestCase):
    def test_template_files_replace_every_tokenizer_config_template(self):
        config = {"chat_template": [{"name": "default", "template": "config default"},
                                    {"name": "tool_use", "template": "config tools"}]}
        self.assertEqual(template_sources(mapping({"tokenizer_config.json": config})),
                         {"default": "config default", "tool_use": "config tools"})
        self.assertEqual(template_sources(mapping({"tokenizer_config.json": config, "chat_template.jinja": "file"})),
                         {"default": "file"})
        self.assertEqual(template_sources(mapping({"tokenizer_config.json": config, "chat_template.jinja": "file",
                                                   "additional_chat_templates/rag.jinja": "rag",
                                                   "additional_chat_templates/nested/x.jinja": "ignored",
                                                   "additional_chat_templates/notes.txt": "ignored"})),
                         {"default": "file", "rag": "rag"})
        self.assertEqual(template_sources(mapping({"tokenizer_config.json": {"chat_template": "single"}})),
                         {"default": "single"})
        self.assertEqual(template_sources(mapping({"tokenizer_config.json": {}})), {})

    def test_legacy_chat_template_json_supplies_the_default_and_conflicts_are_errors(self):
        legacy = {"chat_template": "legacy"}
        self.assertEqual(template_sources(mapping({"chat_template.json": legacy,
                                                   "tokenizer_config.json": {"chat_template": "config"}})),
                         {"default": "legacy"})
        self.assertEqual(template_sources(mapping({"chat_template.json": legacy, "chat_template.jinja": "legacy"})),
                         {"default": "legacy"})
        cases = (({"chat_template.json": legacy, "additional_chat_templates/tool_use.jinja": "t"}, "cannot be combined"),
                 ({"chat_template.json": legacy, "chat_template.jinja": "other"}, "different templates"),
                 ({"chat_template.jinja": "a", "additional_chat_templates/default.jinja": "b"}, "both define"),
                 ({"chat_template.json": {"chat_template": 1}}, "chat_template string"),
                 ({"tokenizer_config.json": {"chat_template": [{"name": "x"}]}}, r"chat_template\[0\]"),
                 ({"tokenizer_config.json": {"chat_template": [{"name": "x", "template": "a"},
                                                               {"name": "x", "template": "b"}]}}, "defined twice"),
                 ({"tokenizer_config.json": {"chat_template": 3}}, "string, a list"),
                 ({"chat_template.jinja": b"\xff"}, "not UTF-8"))
        for files, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ModelTextError, message):
                template_sources(mapping(files))

    def test_from_files_accepts_a_directory_path(self):
        with tempfile.TemporaryDirectory() as directory:
            fx.write_files(directory, fx.llama_files(**{"additional_chat_templates/tool_use.jinja": fx.QWEN3}))
            self.assertEqual(template_sources(directory), {"default": fx.LLAMA3, "tool_use": fx.QWEN3})
            self.assertEqual(list(ChatTemplates.from_files(directory, clock=None)), ["default", "tool_use"])
            self.assertEqual(Tokenizer.from_files(directory).bos_token_id, byte_tokenizer().bos_token_id)
        for call in (template_sources, Tokenizer.from_files, ChatTemplates.from_files):
            with self.subTest(call=call), self.assertRaisesRegex(ModelTextError, "not a directory"):
                call(ROOT / "no-such-model-directory")
            with self.subTest(call=call), self.assertRaisesRegex(TypeError, "directory path or .* not int$"):
                call(42)
            with self.subTest(call=call), self.assertRaisesRegex(TypeError, "directory path or"):
                call(LookAlike())   # an object with read and names that is not a ModelFiles

    def test_files_outside_the_template_folder_are_not_listed_or_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            fx.write_files(directory, fx.llama_files(**{"additional_chat_templates/rag.jinja": "rag",
                                                        "additional_chat_templates/nested/x.jinja": "ignored",
                                                        "other/deep/file.bin": b"\0"}))
            if os.name == "posix":   # names crit_core.files refuses, legal on POSIX file systems
                Path(directory, "notes\\draft.txt").write_text("unrelated")
                Path(directory, "additional_chat_templates", "odd\\name.jinja").write_text("skipped")
            self.assertEqual(template_sources(directory), {"default": fx.LLAMA3, "rag": "rag"})
            self.assertEqual(list(load_model_text(directory, clock=None).templates), ["default", "rag"])
        listed = CallableFiles(mapping(fx.llama_files()).read, names=["bad\\name", "additional_chat_templates/../x",
                                                                      "notes:x", "additional_chat_templates/CON.jinja"])
        self.assertEqual(template_sources(listed), {"default": fx.LLAMA3})
        listed = CallableFiles(mapping(fx.llama_files()).read, names=["additional_chat_templates/a.jinja"])
        with self.assertRaisesRegex(ModelTextError, "^additional_chat_templates/a.jinja is listed but missing$"):
            template_sources(listed)

        class Unlisted(MappingFiles):
            def _names(self):
                raise OSError("listing failed")

        with self.assertRaisesRegex(ModelTextError, "^cannot list additional_chat_templates/: listing failed$"):
            template_sources(Unlisted(mapping(fx.llama_files())._files))

    def test_selection_follows_explicit_name_then_tool_use_then_default(self):
        both = ChatTemplates.from_sources({"default": "d", "tool_use": "t", "rag": "r"})
        self.assertEqual(both.select().name, "default")
        self.assertEqual(both.select(tools=[]).name, "tool_use")
        self.assertEqual(both.select("rag", tools=[{}]).name, "rag")
        self.assertEqual(ChatTemplates.from_sources({"default": "d"}).select(tools=[{}]).name, "default")
        with self.assertRaisesRegex(ModelTextError, "no chat template named 'chat'; available: default, rag, tool_use"):
            both.select("chat")
        with self.assertRaisesRegex(ModelTextError, "no default; pass one of: rag, tool_use"):
            ChatTemplates.from_sources({"tool_use": "t", "rag": "r"}).select()
        self.assertEqual(ChatTemplates.from_sources({"tool_use": "t", "rag": "r"}).select(tools=[]).name, "tool_use")
        with self.assertRaisesRegex(ModelTextError, "has no chat template"):
            ChatTemplates().select()
        self.assertEqual(list(both), ["default", "rag", "tool_use"])


class MessagesTests(unittest.TestCase):
    def test_openai_messages_normalize_roles_content_and_tool_calls(self):
        source = [{"role": "developer", "content": "sys", "extra": 1},
                  {"role": "assistant", "content": None, "reasoning_content": "think",
                   "tool_calls": [{"id": "a", "type": "function",
                                   "function": {"name": "f", "arguments": '{"b": 1, "a": [2]}'}},
                                  {"id": "b", "type": "function", "function": {"name": "g", "arguments": {"x": 1}}}]},
                  {"role": "tool", "tool_call_id": "a", "name": "f", "content": [{"type": "text", "text": "ok"}]}]
        self.assertEqual(openai_messages(source), [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": {"b": 1, "a": [2]}}},
                            {"id": "b", "type": "function", "function": {"name": "g", "arguments": {"x": 1}}}],
             "reasoning_content": "think"},
            {"role": "tool", "content": "ok", "tool_call_id": "a", "name": "f"}])
        self.assertEqual(source[1]["tool_calls"][0]["function"]["arguments"], '{"b": 1, "a": [2]}')
        self.assertEqual(list(openai_messages(source)[1]["tool_calls"][0]["function"]["arguments"]), ["b", "a"])

    def test_invalid_messages_raise_text_errors_naming_the_field(self):
        call = {"type": "function", "function": {"name": "f", "arguments": "{bad"}}
        cases = (("x", "list"), ([5], r"messages\[0\] must be"), ([{"content": "x"}], r"messages\[0\]\.role"),
                 ([{"role": "user", "content": [{"type": "image_url"}]}], r"content\[0\] must be a text part"),
                 ([{"role": "user", "content": 5}], "content must be"),
                 ([{"role": "assistant", "tool_calls": [call]}], r"tool_calls\[0\]\.function\.arguments"),
                 ([{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}], "JSON object"),
                 ([{"role": "assistant", "tool_calls": [{"function": {}}]}], "name string"),
                 ([{"role": "assistant", "tool_calls": "x"}], "must be a list"),
                 ([{"role": "tool", "tool_call_id": 3}], "tool_call_id must be a string"))
        for messages, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(TextError, message):
                openai_messages(messages)


class GenerationTests(unittest.TestCase):
    def test_generation_config_file_wins_and_unset_stays_none(self):
        config = GenerationConfig.from_files(mapping(fx.llama_files()), byte_tokenizer())
        self.assertEqual((config.eos_token_ids, config.bos_token_id, config.pad_token_id), ((1, 4), 0, None))
        self.assertEqual(config.sampling_defaults(), {"do_sample": True, "temperature": 0.6, "top_p": 0.9})
        self.assertIsNone(config.top_k)
        self.assertEqual(config.raw["custom_key"], {"kept": True})
        self.assertEqual(config.stop_token_ids, (1, 4))
        self.assertEqual(GenerationConfig().sampling_defaults(), {})

    def test_without_generation_config_the_model_config_view_applies(self):
        files = mapping({"config.json": {"eos_token_id": None, "top_k": 20, "pad_token_id": None,
                                         "text_config": {"eos_token_id": [7, 8], "pad_token_id": 3, "top_k": 99,
                                                         "stop_strings": "<END>"}}})
        config = GenerationConfig.from_files(files)
        self.assertEqual((config.eos_token_ids, config.pad_token_id, config.top_k, config.stop_strings),
                         ((7, 8), 3, 20, ("<END>",)))
        self.assertEqual(dict(config.raw), {})
        decoder = mapping({"config.json": {"decoder": {"eos_token_id": 5}, "text_config": {"eos_token_id": 6}}})
        self.assertEqual(GenerationConfig.from_files(decoder).eos_token_ids, (5,))
        first_only = GenerationConfig.from_files(mapping({"config.json": {
            "decoder": {"eos_token_id": 5}, "text_config": {"pad_token_id": 3, "top_k": 7}}}))
        self.assertEqual((first_only.eos_token_ids, first_only.pad_token_id, first_only.top_k), ((5,), None, None))

    def test_eos_falls_back_to_the_model_config_and_stop_ids_are_an_ordered_union(self):
        files = mapping({"generation_config.json": {"temperature": 0.7},
                         "config.json": {"eos_token_id": 6, "temperature": 0.1,
                                         "text_config": {"eos_token_id": [6, 2]}}})
        tokenizer = byte_tokenizer()
        config = GenerationConfig.from_files(files, tokenizer)
        self.assertEqual(config.eos_token_ids, (6,))
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.stop_token_ids, (6, 2, tokenizer.eos_token_id))
        explicit = mapping({"generation_config.json": {"eos_token_id": []}, "config.json": {"eos_token_id": 6}})
        self.assertEqual(GenerationConfig.from_files(explicit).eos_token_ids, ())

    def test_legacy_eos_ids_match_eval_prepare_stop_ids_on_the_same_directories(self):
        layouts = (fx.llama_files(),
                   {"generation_config.json": {"eos_token_id": 9}, "config.json": {"eos_token_id": [3, 9]}},
                   {"config.json": {"eos_token_id": None, "text_config": {"eos_token_id": [12, 11]}}},
                   {"generation_config.json": {"eos_token_id": None, "text_config": {"eos_token_id": 4}}},
                   {"generation_config.json": {}, "config.json": {}},
                   {})
        for layout in layouts:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                fx.write_files(directory, layout)
                expected = v2.stop_ids(directory)
                self.assertEqual(legacy_eos_ids(directory), expected)
                self.assertEqual(legacy_eos_ids(DirectoryFiles(directory)), expected)
                if layout:
                    stop_ids = GenerationConfig.from_files(directory).stop_token_ids
                    self.assertLessEqual(set(expected), set(stop_ids))

    def test_invalid_generation_values_are_rejected(self):
        tokenizer = byte_tokenizer()
        cases = (({"eos_token_id": True}, "eos_token_id"), ({"eos_token_id": [1, -2]}, "eos_token_id"),
                 ({"eos_token_id": 10 ** 6}, "outside the tokenizer vocabulary"), ({"bos_token_id": [1]}, "one token"),
                 ({"temperature": -1}, "at least 0"), ({"top_p": 1.5}, "between 0 and 1"),
                 ({"repetition_penalty": 0}, "greater than 0"), ({"top_k": 2.5}, r"integer in \[0, 4294967295\]"),
                 ({"do_sample": 1}, "true, false"), ({"stop_strings": [""]}, "stop_strings"),
                 ({"min_p": "0.1"}, "finite number"))
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ModelTextError, message):
                GenerationConfig.from_files(mapping({"generation_config.json": values}), tokenizer)
        with self.assertRaisesRegex(ModelTextError, "text_config must be an object"):
            legacy_eos_ids(mapping({"config.json": {"text_config": [1]}}))
        with self.assertRaisesRegex(ModelTextError, "text_config must be an object"):
            GenerationConfig.from_files(mapping({"config.json": {"text_config": [1]}}))
        with self.assertRaisesRegex(ModelTextError, "^config.json: eos_token_id must be a token id"):
            legacy_eos_ids(mapping({"config.json": '{"eos_token_id": NaN}'}))


class _Unreadable:
    """A file whose bytes must not be read (it can still be stat-ed)."""

    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *details):
        self.handle.close()

    def fileno(self):
        return self.handle.fileno()

    def read(self, *details):
        raise AssertionError("read before the size check")


def unreadable_open(file, mode="r", *details, **options):
    return _Unreadable(open(file, mode, *details, **options))


class CappedReads(MappingFiles):
    """Records the max_bytes of every read."""

    def __init__(self, files):
        super().__init__({name: fx.file_bytes(value) for name, value in files.items()})
        self.caps = {}

    def read(self, name, max_bytes=None):
        self.caps[name] = max_bytes
        return super().read(name, max_bytes)


class NativeParityTests(unittest.TestCase):
    """The text layer reads generation_config.json and config.json with the rules of the native loaders."""

    def test_a_dangling_or_non_regular_generation_config_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fx.write_files(directory, fx.llama_files())
            (root / "generation_config.json").unlink()
            (root / "generation_config.json").mkdir()
            for call in (GenerationConfig.from_files, legacy_eos_ids, load_model_text):
                with self.subTest(call=call), self.assertRaisesRegex(
                        ModelTextError, "^generation_config.json: not a regular file$"):
                    call(directory)
            (root / "generation_config.json").rmdir()
            try:
                os.symlink(root / ".." / "blobs" / "missing-blob", root / "generation_config.json")
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"no symlinks here: {error}")
            for call in (GenerationConfig.from_files, legacy_eos_ids, load_model_text):   # Eval skips the file
                with self.subTest(call=call), self.assertRaisesRegex(ModelTextError,
                                                                     "^generation_config.json: open: "):
                    call(directory)
            for target in ("generation_config.json", "config.json/inner"):   # a loop, a target through a file
                (root / "generation_config.json").unlink()
                os.symlink(target, root / "generation_config.json")
                for call in (GenerationConfig.from_files, legacy_eos_ids, load_model_text):
                    with self.subTest(target=target, call=call), self.assertRaisesRegex(
                            ModelTextError, "^generation_config.json: open: "):
                        call(directory)

    def test_bos_and_pad_take_any_32_bit_id_and_eos_stays_a_vocabulary_id(self):
        tokenizer = byte_tokenizer()
        files = mapping({"generation_config.json": {"eos_token_id": 1, "bos_token_id": -(1 << 31), "pad_token_id": -1}})
        config = GenerationConfig.from_files(files, tokenizer)
        self.assertEqual((config.eos_token_ids, config.bos_token_id, config.pad_token_id), ((1,), -(1 << 31), -1))
        model_only = mapping({"config.json": {"eos_token_id": 1, "pad_token_id": -1, "vocab_size": 420}})
        self.assertEqual(GenerationConfig.from_files(model_only).pad_token_id, -1)
        cases = (({"bos_token_id": 1 << 31}, "bos_token_id must be a token id in \\[-2147483648, 2147483647\\]$"),
                 ({"pad_token_id": -(1 << 31) - 1}, "pad_token_id must be a token id in"),
                 ({"pad_token_id": 10 ** 6}, "pad_token_id 1000000 is outside the tokenizer vocabulary"),
                 ({"bos_token_id": 1.0}, "bos_token_id must be a token id"),
                 ({"eos_token_id": -1}, "eos_token_id must be a token id in \\[0, 2147483647\\] or a list of them$"),
                 ({"eos_token_id": [1, 1 << 31]}, "eos_token_id must be a token id in \\[0, 2147483647\\]"),
                 ({"eos_token_id": 1 << 64}, "eos_token_id: integer exceeds 64 bits$"),
                 ({"pad_token_id": -(1 << 63) - 1}, "pad_token_id: integer exceeds 64 bits$"),
                 ({"temperature": 1 << 64}, "temperature: integer exceeds 64 bits$"),
                 ({"top_k": 1 << 32}, "top_k must be an integer in \\[0, 4294967295\\]$"),
                 ({"max_new_tokens": -1}, "max_new_tokens must be an integer in \\[0, 4294967295\\]$"),
                 ({"max_new_tokens": 1 << 70}, "max_new_tokens: integer exceeds 64 bits$"),
                 ({"max_length": 1 << 32}, "max_length must be an integer in \\[0, 4294967295\\]$"),
                 ({"max_length": -1}, "max_length must be an integer in"),
                 ({"max_length": True}, "max_length must be an integer in"),
                 ({"max_length": 20.0}, "max_length must be an integer in"),
                 ({"max_length": 1 << 64}, "max_length: integer exceeds 64 bits$"))
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(ModelTextError,
                                                                     "^generation_config.json: " + message):
                GenerationConfig.from_files(mapping({"generation_config.json": values}), tokenizer)
        limits = GenerationConfig.from_files(mapping({"generation_config.json": {
            "eos_token_id": (1 << 31) - 1, "top_k": (1 << 32) - 1, "max_new_tokens": (1 << 32) - 1,
            "max_length": (1 << 32) - 1}}))
        self.assertEqual((limits.eos_token_ids, limits.top_k, limits.max_new_tokens, limits.max_length),
                         (((1 << 31) - 1,), (1 << 32) - 1, (1 << 32) - 1, (1 << 32) - 1))
        self.assertNotIn("max_length", limits.sampling_defaults())
        self.assertIsNone(GenerationConfig.from_files(mapping({"generation_config.json": {}})).max_length)
        model_only = GenerationConfig.from_files(mapping({"config.json": {"text_config": {"max_length": 20}}}))
        self.assertEqual(model_only.max_length, 20)   # from config.json as transformers reads it
        with self.assertRaisesRegex(ModelTextError, "^config.json: max_length must be an integer in"):
            GenerationConfig.from_files(mapping({"config.json": {"max_length": -1}}))
        with self.assertRaisesRegex(ModelTextError, "^config.json: eos_token_id must be a token id in \\[0, "):
            legacy_eos_ids(mapping({"config.json": {"eos_token_id": 1 << 31}}))

    def test_one_directory_loads_through_the_text_layer_and_the_model_config(self):
        layouts = ({"config.json": {"eos_token_id": 1, "pad_token_id": -1, "vocab_size": 420}},
                   {"generation_config.json": {"eos_token_id": 1, "pad_token_id": -1, "bos_token_id": -(1 << 31)},
                    "config.json": {"eos_token_id": 1, "pad_token_id": -1}},
                   {"config.json": json.dumps({"model_type": "mamba2", "eos_token_id": 1, "vocab_size": 420,
                                               "time_step_limit": [0.0, float("inf")], "x": float("nan")})},
                   {"generation_config.json": '{"eos_token_id": 1, "note": NaN, "range": [-Infinity, 1]}',
                    "config.json": {"eos_token_id": 1}})
        for layout in layouts:
            files = fx.llama_files()
            del files["generation_config.json"]
            files.update(layout)
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                fx.write_files(directory, files)
                model = load_model_text(directory, clock=None)
                self.assertEqual(model.generation.eos_token_ids, (1,))
                self.assertEqual(load_model_config(directory).eos_token_ids, (1,))
                self.assertEqual(legacy_eos_ids(directory), [1])
        self.assertEqual(model.generation.raw["note"] != model.generation.raw["note"], True)   # NaN kept, untyped
        nan_files = mapping({"generation_config.json": '{"eos_token_id": 1, "note": NaN}'})
        self.assertEqual(GenerationConfig.from_files(nan_files), GenerationConfig.from_files(nan_files))
        for text, message in (('{"temperature": NaN}', "temperature must be a finite number"),
                              ('{"top_p": Infinity}', "top_p must be a finite number"),
                              ('{"eos_token_id": [1, -Infinity]}', "eos_token_id must be a token id")):
            with self.subTest(text=text), self.assertRaisesRegex(ModelTextError, "^generation_config.json: " + message):
                GenerationConfig.from_files(mapping({"generation_config.json": text}))
        with self.assertRaisesRegex(ModelTextError, "^tokenizer_config.json: non-finite number NaN$"):
            load_model_text(mapping(fx.llama_files(**{"tokenizer_config.json": '{"model_max_length": NaN}'})))

    def test_a_null_text_config_reads_as_absent_everywhere(self):
        files = mapping({"generation_config.json": {"eos_token_id": 3, "text_config": None},
                         "config.json": {"eos_token_id": 5, "text_config": None}})
        self.assertEqual(legacy_eos_ids(files), [3, 5])
        self.assertEqual(GenerationConfig.from_files(files).stop_token_ids, (3, 5))
        self.assertEqual(load_model_config(files).eos_token_ids, (5,))

    def test_config_files_are_read_with_the_16_mib_cap_and_other_files_with_128_mib(self):
        files = CappedReads(fx.llama_files(**{"additional_chat_templates/rag.jinja": "rag"}))
        load_model_text(files, clock=None)
        expected = {name: 128 << 20 for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                                 "additional_chat_templates/rag.jinja", "chat_template.json")}
        expected.update({"generation_config.json": 16 << 20, "config.json": 16 << 20})
        self.assertEqual(files.caps, expected)
        files = fx.llama_files(**{"chat_template.json": {"chat_template": fx.LLAMA3}})
        del files["tokenizer_config.json"]["added_tokens_decoder"]   # the legacy layout reads special_tokens_map.json
        files = CappedReads(files)
        load_model_text(files, clock=None)
        del expected["additional_chat_templates/rag.jinja"]
        self.assertEqual(files.caps, {**expected, "special_tokens_map.json": 128 << 20})
        legacy = CappedReads({"config.json": {}})
        legacy_eos_ids(legacy)
        self.assertEqual(legacy.caps, {"generation_config.json": 16 << 20, "config.json": 16 << 20})
        with tempfile.TemporaryDirectory() as directory:
            root = fx.write_files(directory, fx.llama_files())
            with open(root / "generation_config.json", "wb") as handle:
                handle.truncate(400 << 20)   # sparse
            message = "^generation_config.json: 419430400 bytes exceeds the 16777216-byte limit$"
            with mock.patch("crit_core.files.open", unreadable_open, create=True):   # generation_config.json first
                for call in (GenerationConfig.from_files, legacy_eos_ids):
                    with self.subTest(call=call), self.assertRaisesRegex(ModelTextError, message):
                        call(directory)
            with self.assertRaisesRegex(ModelTextError, message):
                load_model_text(directory)
            (root / "generation_config.json").unlink()
            (root / "config.json").write_bytes(b'{"eos_token_id": 1}' + b" " * (16 << 20))   # over 16 MiB
            with self.assertRaisesRegex(ModelTextError, "^config.json: 16777235 bytes exceeds the 16777216-byte"):
                load_model_text(directory)
            with self.assertRaisesRegex(ConfigError, "16777235 bytes exceeds the 16777216-byte limit$"):
                load_model_config(directory)

    def test_malformed_values_raise_text_errors_naming_the_file(self):
        digits = "1" * 5000
        for name, text in (("tokenizer_config.json", '{"model_max_length": %s}' % digits),
                           ("generation_config.json", '{"eos_token_id": %s}' % digits),
                           ("config.json", '{"x": %s}' % digits),
                           ("tokenizer_config.json", '{"bos_token": "\\ud800"}'),
                           ("special_tokens_map.json", '{"eos_token": "\\udc00"}'),
                           ("generation_config.json", '{"stop_strings": ["\\ud800"]}'),
                           ("chat_template.json", '{"chat_template": "\\ud800"}')):
            files = fx.llama_files(**{name: text})
            if name == "special_tokens_map.json":
                del files["tokenizer_config.json"]["added_tokens_decoder"]   # the legacy layout reads the map
            with self.subTest(name=name, text=text[:40]), self.assertRaisesRegex(ModelTextError, f"^{name}: "):
                load_model_text(mapping(files))
        call = {"function": {"name": "f", "arguments": '{"a": %s}' % digits}}
        with self.assertRaisesRegex(TextError, r"^messages\[0\]\.tool_calls\[0\]\.function\.arguments: Exceeds"):
            openai_messages([{"role": "assistant", "tool_calls": [call]}])
        call = {"function": {"name": "f", "arguments": '{"a": "\\ud800"}'}}
        with self.assertRaisesRegex(TextError, "arguments: unpaired surrogate escape$"):
            openai_messages([{"role": "assistant", "tool_calls": [call]}])


def corpus_text(rng):
    return "".join(rng.choice(fx.CORPUS)[rng.randrange(8):][:rng.randrange(1, 40)] for _ in range(rng.randint(1, 3)))


def straddle_text(rng):
    """Runs of characters whose tokens end inside the next character, some longer than HOLD_LIMIT tokens."""
    return "".join(rng.choice(fx.STRADDLE_CORPUS) * rng.choice((1, 2, 5, 20, 40)) for _ in range(rng.randint(1, 4)))


TOKENIZERS = (("byte-level", byte_tokenizer, corpus_text), ("sentencepiece", sp_tokenizer, corpus_text),
              ("straddling", straddle_tokenizer, straddle_text))


def valid_streams(tokenizer, rng, count, text_of):
    """(prompt, output) id lists cut from encoded text at random token boundaries, so byte-fallback and byte-level
    characters are split between prompt and output; added tokens are inserted at character boundaries."""
    added = sorted(tokenizer.added_tokens)
    for _ in range(count):
        ids = tokenizer.encode(text_of(rng), add_special_tokens=False)
        if rng.random() < 0.3:
            for _ in range(rng.randint(1, 3)):
                cuts = [i for i in range(len(ids) + 1) if not tokenizer.decode(ids[:i]).endswith(FFFD)]
                ids.insert(rng.choice(cuts), rng.choice(added))
        cut = rng.randint(0, len(ids))
        yield ids[:cut], ids[cut:]


def byte_ids(tokenizer, data):
    """Byte-level token ids that spell `data` one byte at a time."""
    return [tokenizer.token_to_id(fx.BYTE_CHARS[byte]) for byte in data]


class DetokenizerTests(unittest.TestCase):
    def test_streamed_text_is_released_as_soon_as_characters_complete_for_random_sequences(self):
        for name, make, text_of in TOKENIZERS:
            tokenizer = make()
            rng = random.Random(name)
            split_prompts = 0
            for number, (prompt, output) in enumerate(valid_streams(tokenizer, rng, 400, text_of)):
                skip = number % 2 == 0
                whole = fx.ByteText(tokenizer, prompt + output, skip)
                self.assertEqual(whole.text, tokenizer.decode(prompt + output, skip_special_tokens=skip))
                prompt_text = whole.complete(len(prompt))
                split_prompts += not whole.boundary(len(prompt))
                decoder = IncrementalDetokenizer(tokenizer, prompt, skip_special_tokens=skip)
                streamed, pushed = "", 0
                for piece in chunks(rng, output):
                    streamed += decoder.push(piece)
                    pushed += len(piece)
                    self.assertEqual(streamed, whole.complete(len(prompt) + pushed)[len(prompt_text):],
                                     (name, prompt, output, pushed))
                    self.assertLessEqual(len(decoder._window), WINDOW_LIMIT)
                streamed += decoder.finish()
                with self.subTest(tokenizer=name, prompt=prompt, output=output, skip=skip):
                    self.assertEqual(streamed, whole.text[len(prompt_text):])
                    self.assertEqual(decoder.anomalies, 0)
                    self.assertEqual(decoder.emitted, len(streamed))
            self.assertGreater(split_prompts, 10, name)

    def test_long_runs_of_tokens_that_end_inside_a_character_stream_exactly(self):
        tokenizer = straddle_tokenizer()
        for text in ("γ" * 40, "λ" * 60 + "a", "д" * 33 + "😀" * 20, "λγ" * 30 + "Ελλάδα"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            longest = max(len(run) for run in "".join("x" if tokenizer.decode(ids[:k]).endswith(FFFD) else " "
                                                         for k in range(1, len(ids) + 1)).split())
            self.assertGreater(longest, detokenize_module.HOLD_LIMIT, text)
            decoder = IncrementalDetokenizer(tokenizer)
            streamed = ""
            for token in ids:
                streamed += decoder.push([token])
                self.assertNotIn(FFFD, streamed)
                self.assertLessEqual(len(decoder._window), WINDOW_LIMIT)
            self.assertEqual(streamed + decoder.finish(), text)
            self.assertEqual(decoder.anomalies, 0)
            prompt = ids[:7]
            decoder = IncrementalDetokenizer(tokenizer, prompt)
            whole = fx.ByteText(tokenizer, ids)
            self.assertEqual(decoder.push(ids[7:]) + decoder.finish(), text[len(whole.complete(7)):])

    def test_sentencepiece_continuations_keep_their_leading_space(self):
        tokenizer = sp_tokenizer()
        prompt = tokenizer.encode("Hello", add_special_tokens=False)
        output = tokenizer.encode("Hello world, this is", add_special_tokens=False)[len(prompt):]
        self.assertFalse(tokenizer.decode(output[:1]).startswith(" "))
        decoder = IncrementalDetokenizer(tokenizer, prompt)
        self.assertEqual("".join(decoder.push([token]) for token in output) + decoder.finish(), " world, this is")

    def test_a_prompt_ending_mid_character_continues_into_the_output(self):
        tokenizer = sp_tokenizer()
        ids = tokenizer.encode("x 日本", add_special_tokens=False)
        byte = tokenizer.token_to_id("<0xE6>")
        cut = ids.index(byte, ids.index(byte) + 1) + 1
        decoder = IncrementalDetokenizer(tokenizer, ids[:cut])
        self.assertEqual(decoder.push(ids[cut:cut + 1]), "")
        self.assertEqual(decoder.push(ids[cut + 1:]) + decoder.finish(), "本")
        empty = IncrementalDetokenizer(tokenizer, ids[:cut])
        self.assertEqual(empty.finish(), "")
        run = tokenizer.encode("ä日", add_special_tokens=False)   # one byte run: <0xC3> <0xA4> <0xE6> <0x97> <0xA5>
        self.assertTrue(tokenizer.decode(run[:-1]).endswith(FFFD * 4))
        decoder = IncrementalDetokenizer(tokenizer, run[:-1])
        self.assertEqual(decoder.push(run[-1:]) + decoder.finish(), "日")

    def test_a_prompt_ending_inside_a_character_is_not_repeated_in_the_output(self):
        tokenizer = byte_tokenizer()
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
        g_lead = tokenizer.token_to_id("g" + fx.BYTE_CHARS[0xC3])                           # b"g\xc3"
        space_lead = tokenizer.token_to_id("".join(fx.BYTE_CHARS[byte] for byte in b" \xf0\x9f"))
        greek = encode("Ελληνικά")
        self.assertIsNotNone(g_lead)
        self.assertIsNotNone(space_lead)
        cases = ((encode("Fuß") + [g_lead], byte_ids(tokenizer, b"\xa4") + encode("nger"), "änger"),
                 (encode("and emoji") + [space_lead], byte_ids(tokenizer, b"\x98\x80") + encode(" and more"),
                  "😀 and more"),
                 (greek[:2], greek[2:], "ληνικά"))
        for prompt, output, expected in cases:
            with self.subTest(expected=expected):
                self.assertTrue(tokenizer.decode(prompt).endswith(FFFD))
                self.assertEqual(tokenizer.decode(prompt).rstrip(FFFD) + expected, tokenizer.decode(prompt + output))
                decoder = IncrementalDetokenizer(tokenizer, prompt)
                self.assertEqual("".join(decoder.push([token]) for token in output) + decoder.finish(), expected)
                stream = TextStream(tokenizer, prompt)
                self.assertEqual("".join(stream.push([token]).text for token in output) + stream.finish().text,
                                 expected)

    def test_replacement_characters_the_output_leaves_in_place_stay_prompt_text(self):
        tokenizer = byte_tokenizer()
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
        literal = IncrementalDetokenizer(tokenizer, encode("ok �"))
        self.assertEqual(literal.push(encode(" more")) + literal.finish(), " more")
        invalid = encode("ok") + byte_ids(tokenizer, b"\xe6")   # an incomplete character the output makes invalid
        decoder = IncrementalDetokenizer(tokenizer, invalid)
        self.assertEqual(decoder.push(byte_ids(tokenizer, b"A")) + decoder.finish(), "A")
        unfinished = IncrementalDetokenizer(tokenizer, invalid)     # still incomplete when generation ends
        self.assertEqual(unfinished.push(byte_ids(tokenizer, b"\x97")) + unfinished.finish(), "")
        completed = IncrementalDetokenizer(tokenizer, invalid)
        self.assertEqual(completed.push(byte_ids(tokenizer, b"\x97\xa5!")) + completed.finish(), "日!")

    def test_a_prompt_whose_byte_run_ends_in_literal_replacement_characters_starts_its_window_at_a_character(self):
        tokenizer = sp_tokenizer()
        cases = (("hello 日��", "😀 ok", "😀", 4), ("x ���", "日", "日", 3), ("Grüße ���", "日本 ok", "日", 3),
                 ("Hello �😀", "ä ok", "ä", 2), ("Hello 日�本", "𝄞𝄞", "𝄞", 4))
        for prompt_text, output_text, stop, consumed in cases:
            with self.subTest(prompt=prompt_text):
                prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
                ids = tokenizer.encode(prompt_text + output_text, add_special_tokens=False)
                self.assertEqual(ids[:len(prompt)], prompt)
                output = ids[len(prompt):]
                self.assertTrue(all(tokenizer.backend.id_to_token(token).startswith("<0x") for token in output[:2]))
                decoder = IncrementalDetokenizer(tokenizer, prompt)
                self.assertEqual("".join(decoder.push([token]) for token in output) + decoder.finish(), output_text)
                stream = TextStream(tokenizer, prompt, stop_strings=[stop])
                deltas = [stream.push([token]) for token in output[:consumed]]
                self.assertEqual(("".join(delta.text for delta in deltas), deltas[-1].finish_reason, stream.consumed),
                                 ("", "stop", consumed))

    def test_a_prompt_cut_inside_a_literal_replacement_character_never_releases_prompt_text(self):
        tokenizer = sp_tokenizer()
        sp_bytes = lambda data: [tokenizer.token_to_id(f"<0x{byte:02X}>") for byte in data]  # noqa: E731
        hi, ok = (tokenizer.encode(text, add_special_tokens=False) for text in ("Hi", "x ok"))
        ok = ok[1:]   # "▁" "o" "k"
        # The prompt's run (д, U+FFFD, then the first two bytes of another) decodes as U+FFFD per byte; the output's
        # byte completes it: the output is that U+FFFD, and "д" stays prompt text.
        prompt = hi + sp_bytes("д�".encode() + b"\xef\xbf")
        output = sp_bytes(b"\xbd") + ok
        self.assertEqual(tokenizer.decode(prompt), "Hi" + FFFD * 7)
        self.assertEqual(tokenizer.decode(prompt + output), "Hiд" + FFFD * 2 + " ok")
        decoder = IncrementalDetokenizer(tokenizer, prompt)
        self.assertEqual("".join(decoder.push([token]) for token in output) + decoder.finish(), FFFD + " ok")
        stream = TextStream(tokenizer, prompt, stop_strings=["д"])
        text = "".join(stream.push([token]).text for token in output)
        self.assertEqual((text + stream.finish().text, stream.stop), (FFFD + " ok", None))
        # The prompt decode's U+FFFD per byte matches the output's literal U+FFFD by chance: still not prompt text.
        prompt = hi + sp_bytes("�".encode() + b"\xef")
        output = sp_bytes(b"\xbf\xbd" + "��".encode()) + ok
        self.assertTrue(tokenizer.decode(prompt + output).startswith(tokenizer.decode(prompt)))
        decoder = IncrementalDetokenizer(tokenizer, prompt)
        self.assertEqual("".join(decoder.push([token]) for token in output) + decoder.finish(), FFFD * 3 + " ok")
        # Byte-level: a U+FFFD that the prompt decode shows and the full decode keeps stays prompt text, also when the
        # output's bytes complete it as a literal U+FFFD (the decode cannot tell it from an incomplete character).
        tokenizer = byte_tokenizer()
        prompt = tokenizer.encode("xa", add_special_tokens=False) + byte_ids(tokenizer, b"\xef")
        output = byte_ids(tokenizer, b"\xbf\xbd") + tokenizer.encode("Ω yes", add_special_tokens=False)
        self.assertEqual(tokenizer.decode(prompt + output), "xa" + FFFD + "Ω yes")
        decoder = IncrementalDetokenizer(tokenizer, prompt)
        self.assertEqual("".join(decoder.push([token]) for token in output) + decoder.finish(), "Ω yes")

    def test_a_literal_replacement_character_merged_into_the_split_character_stays_prompt_text(self):
        tokenizer = straddle_tokenizer()
        merged = tokenizer.token_to_id(fx.BYTE_CHARS[0xBD] + fx.BYTE_CHARS[0xC3])   # ends a literal U+FFFD, starts "ä"
        self.assertIsNotNone(merged)
        prompt = tokenizer.encode("ga", add_special_tokens=False) + byte_ids(tokenizer, b"\xef\xbf") + [merged]
        output = byte_ids(tokenizer, b"\xa4") + tokenizer.encode("gg", add_special_tokens=False)
        self.assertEqual(tokenizer.decode(prompt), "ga" + FFFD * 2)
        self.assertEqual(tokenizer.decode(prompt + output), "ga" + FFFD + "ägg")
        decoder = IncrementalDetokenizer(tokenizer, prompt)
        self.assertEqual("".join(decoder.push([token]) for token in output) + decoder.finish(), "ägg")
        stream = TextStream(tokenizer, prompt, stop_strings=[FFFD])
        text = "".join(stream.push([token]).text for token in output)
        self.assertEqual((text + stream.finish().text, stream.stop), ("ägg", None))

    def test_finish_flushes_an_incomplete_tail_and_closes_the_decoder(self):
        tokenizer = sp_tokenizer()
        partial = [tokenizer.token_to_id("▁"), tokenizer.token_to_id("<0xE6>"), tokenizer.token_to_id("<0x97>")]
        decoder = IncrementalDetokenizer(tokenizer)
        self.assertEqual(decoder.push(partial), "")
        self.assertEqual(decoder.finish(), FFFD * 2)
        for call in (lambda: decoder.push([1]), decoder.finish):
            with self.assertRaisesRegex(RuntimeError, "text decoder is closed"):
                call()
        tokenizer = byte_tokenizer()
        decoder = IncrementalDetokenizer(tokenizer)
        self.assertEqual(decoder.push(tokenizer.encode("ab", add_special_tokens=False) + byte_ids(tokenizer, b"\xe6")),
                         "ab")
        self.assertEqual(decoder.finish(), FFFD)

    def test_invalid_ids_close_the_decoder(self):
        tokenizer = byte_tokenizer()
        with self.assertRaises(TokenError):
            IncrementalDetokenizer(tokenizer, [tokenizer.vocab_size])
        decoder = IncrementalDetokenizer(tokenizer)
        with self.assertRaisesRegex(TokenError, "position 1"):
            decoder.push([5, -1])
        with self.assertRaisesRegex(RuntimeError, "text decoder is closed"):
            decoder.push([5])
        with self.assertRaisesRegex(TextError, "context_tokens"):
            IncrementalDetokenizer(tokenizer, context_tokens=-1)

    def test_garbage_streams_stay_bounded(self):
        for name, make, _ in TOKENIZERS:
            tokenizer = make()
            rng = random.Random(3)
            for _ in range(50):
                decoder = IncrementalDetokenizer(tokenizer, [rng.randrange(tokenizer.vocab_size) for _ in range(8)])
                text = ""
                for _ in range(200):
                    text += decoder.push([rng.randrange(tokenizer.vocab_size)])
                    self.assertLessEqual(len(decoder._window), WINDOW_LIMIT, name)
                text += decoder.finish()
                self.assertEqual(len(text), decoder.emitted)
        tokenizer = sp_tokenizer()
        decoder = IncrementalDetokenizer(tokenizer)
        lead = tokenizer.token_to_id("<0xE6>")
        self.assertEqual(decoder.push([lead] * 15), "")
        self.assertEqual(decoder.push([lead]), FFFD * 13)   # invalid bytes are released, the last three still held
        self.assertEqual(decoder.finish(), FFFD * 3)
        self.assertEqual(decoder.anomalies, 0)

    def test_random_ids_stream_as_the_full_decode_unless_an_anomaly_is_counted(self):
        for name, make, _ in TOKENIZERS:
            tokenizer = make()
            rng = random.Random(name + " garbage")
            anomalous = 0
            for _ in range(1500):
                output = [rng.randrange(tokenizer.vocab_size) for _ in range(rng.randint(1, 40))]
                skip = rng.random() < 0.5
                decoder = IncrementalDetokenizer(tokenizer, skip_special_tokens=skip)
                streamed = "".join(decoder.push(piece) for piece in chunks(rng, output)) + decoder.finish()
                if decoder.anomalies:
                    anomalous += 1
                else:
                    self.assertEqual(streamed, tokenizer.decode(output, skip_special_tokens=skip), (name, output, skip))
            if name == "sentencepiece":
                self.assertGreater(anomalous, 0)

    def test_a_byte_run_that_invalidates_released_text_is_counted_as_an_anomaly(self):
        tokenizer = sp_tokenizer()
        ids = [tokenizer.token_to_id(piece) for piece in ("▁sind", "plus", "▁lazy", "▁Grüe", "<0x20>", "<0xD8>",
                                                          "<0x13>")]
        self.assertNotIn(None, ids)
        self.assertEqual(tokenizer.decode(ids), "sindplus lazy Grüe" + FFFD * 3)
        for context in (0, 5):   # the released " " stays in the window as context either way
            decoder = IncrementalDetokenizer(tokenizer, skip_special_tokens=False, context_tokens=context)
            streamed = "".join(decoder.push([token]) for token in ids) + decoder.finish()
            self.assertTrue(streamed.startswith("sindplus lazy Grüe "), context)
            self.assertEqual(decoder.anomalies, 1, context)

    def test_a_decoder_that_rewrites_released_text_never_repeats_it(self):
        raw = tokenizers.Tokenizer.from_str(fx.byte_level_json())
        raw.decoder = tokenizers.decoders.Sequence([tokenizers.decoders.ByteLevel(),
                                                    tokenizers.decoders.Replace("xgä", "Q")])
        tokenizer = Tokenizer.from_bytes(raw.to_str())
        ids = [tokenizer.token_to_id("x"), tokenizer.token_to_id("g" + fx.BYTE_CHARS[0xC3]),
               tokenizer.token_to_id(fx.BYTE_CHARS[0xA4])]
        self.assertEqual(tokenizer.decode(ids), "Q")
        decoder = IncrementalDetokenizer(tokenizer)
        self.assertEqual(decoder.push(ids[:2]), "xg")
        self.assertEqual(decoder.push(ids[2:]), "ä")   # the ids since the last anchor alone give "gä"
        self.assertEqual((decoder.finish(), decoder.anomalies), ("", 1))


def expected_stream(tokenizer, prompt, output, stops, stop_ids):
    """(text, consumed, stop, finish_reason) of a TextStream over valid ids, from byte offsets: after k output ids
    the released text is the output characters whose bytes all came from them; a stop string ends the reply at the
    first id after which that text contains it; a stop id flushes the incomplete tail as decode() shows it."""
    whole = fx.ByteText(tokenizer, prompt + output)
    prompt_text = whole.complete(len(prompt))
    for count in range(1, len(output) + 1):
        if output[count - 1] in stop_ids:
            text = flushed_text(tokenizer, whole, prompt, output, count - 1)
            found = first_stop(text, stops)
            if found is None:
                return text, count, output[count - 1], "stop"
            return text[:found[0]], count - 1, stops[found[1]], "stop"  # held text of earlier ids completed it
        text = whole.complete(len(prompt) + count)[len(prompt_text):]
        found = first_stop(text, stops)
        if found is not None:
            return text[:found[0]], count, stops[found[1]], "stop"
    return whole.text[len(prompt_text):], len(output), None, "length"


def flushed_text(tokenizer, whole, prompt, output, count):
    """All reply text when generation ends after `count` output ids: decode(prompt + ids) after the prompt text (the
    prompt's decode when it is still a prefix, else its complete characters); an incomplete SentencePiece byte run
    that began before text already released is decoded on its own."""
    prompt_text = whole.complete(len(prompt))
    released = whole.complete(len(prompt) + count)[len(prompt_text):]
    full = tokenizer.decode(prompt + output[:count], skip_special_tokens=True)
    for head in (tokenizer.decode(prompt, skip_special_tokens=True), prompt_text):
        if full.startswith(head + released):
            return full[len(head):]
    start = max((k for k in range(count + 1) if whole.boundary(len(prompt) + k)), default=0)
    return released + tokenizer.decode(output[start:count], skip_special_tokens=True)


class TextStreamTests(unittest.TestCase):
    def test_stop_ids_are_consumed_without_being_decoded(self):
        tokenizer = byte_tokenizer()
        output = tokenizer.encode("Hello world", add_special_tokens=False)
        stream = TextStream(tokenizer, stop_token_ids=[tokenizer.eos_token_id], skip_special_tokens=False)
        delta = stream.push(output + [tokenizer.eos_token_id, 5, 6])
        self.assertEqual(delta, TextDelta("Hello world", "stop", tokenizer.eos_token_id, len(output) + 1))
        self.assertEqual(stream.push([7]).consumed, 0)
        self.assertEqual(stream.consumed, len(output) + 1)
        self.assertEqual(stream.finish().text, "")
        with self.assertRaisesRegex(RuntimeError, "text stream is closed"):
            stream.push([1])

    def test_include_stop_str_with_a_stop_id_returns_the_text_without_the_id(self):
        tokenizer = byte_tokenizer()
        output = tokenizer.encode("Hello", add_special_tokens=False)
        stream = TextStream(tokenizer, stop_strings=["zz"], stop_token_ids=[tokenizer.eos_token_id],
                            include_stop_str=True)
        self.assertEqual(stream.push(output + [tokenizer.eos_token_id]),
                         TextDelta("Hello", "stop", tokenizer.eos_token_id, len(output) + 1))
        completing = next(k for k in range(1, len(output) + 1) if "ll" in tokenizer.decode(output[:k]))
        matched = TextStream(tokenizer, stop_strings=["ll"], include_stop_str=True)
        self.assertEqual(matched.push(output), TextDelta("Hell", "stop", "ll", completing))

    def test_stop_strings_across_token_boundaries_report_the_completing_token(self):
        cases = ((sp_tokenizer(), "Grüße aus München! Ça va?", ("aus M", "ü", "Ça", "va?")),
                 (byte_tokenizer(), "Привет, мир! Как дела? Ελληνικά", (", ", "Как ", "Ελ", "мир! К")),
                 (straddle_tokenizer(), "γγγγλλдд😀äga", ("γλ", "дд", "😀", "äg", "γγγγλλдд😀äga")))
        for tokenizer, text, stops in cases:
            output = tokenizer.encode(text, add_special_tokens=False)
            whole = fx.ByteText(tokenizer, output)
            self.assertEqual(whole.text, text)
            for stop in stops:
                with self.subTest(text=text, stop=stop):
                    completing = next(k for k in range(1, len(output) + 1) if stop in whole.complete(k))
                    for include in (False, True):
                        stream = TextStream(tokenizer, stop_strings=[stop, "never"], include_stop_str=include)
                        deltas = [stream.push([token]) for token in output]
                        self.assertEqual("".join(delta.text for delta in deltas),
                                         text[:text.index(stop)] + (stop if include else ""))
                        self.assertEqual(sum(delta.consumed for delta in deltas), completing)
                        self.assertEqual((deltas[completing - 1].finish_reason, deltas[completing - 1].stop),
                                         ("stop", stop))
                        self.assertEqual(stream.finish().finish_reason, "stop")

    def test_the_token_that_completes_a_stop_inside_a_character_is_the_last_consumed(self):
        tokenizer = byte_tokenizer()
        eos = tokenizer.eos_token_id
        russian = tokenizer.encode("Как дела?", add_special_tokens=False)
        self.assertEqual(tokenizer.decode(russian[:5]), "Как " + FFFD)   # the fifth token also starts "д"
        stream = TextStream(tokenizer, stop_strings=["Как "])
        deltas = [stream.push([token]) for token in russian]
        self.assertEqual([delta.consumed for delta in deltas], [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual((deltas[4].finish_reason, stream.consumed), ("stop", 5))
        greek = tokenizer.encode("Ελληνικά", add_special_tokens=False)
        stream = TextStream(tokenizer, stop_strings=["Ελ"], stop_token_ids=[eos])
        self.assertEqual(stream.push(greek[:2] + [eos]), TextDelta("", "stop", "Ελ", 2))
        sp = sp_tokenizer()
        partial = [sp.token_to_id("<0xE6>"), sp.token_to_id("<0x97>")]
        held = TextStream(sp, stop_strings=[FFFD], stop_token_ids=[sp.eos_token_id])
        self.assertEqual(held.push(partial + [sp.eos_token_id]), TextDelta("", "stop", FFFD, 2))
        plain = TextStream(sp, stop_token_ids=[sp.eos_token_id])
        self.assertEqual(plain.push(partial + [sp.eos_token_id]), TextDelta(FFFD * 2, "stop", sp.eos_token_id, 3))

    def test_random_streams_match_a_byte_level_oracle(self):
        for name, make, text_of in TOKENIZERS:
            tokenizer = make()
            rng = random.Random(name + " stream")
            stopped = 0
            for prompt, output in valid_streams(tokenizer, rng, 200, text_of):
                whole = fx.ByteText(tokenizer, prompt + output)
                text = whole.text[len(whole.complete(len(prompt))):]
                stops = [text[start:start + rng.randint(1, 4)] for start in
                         (rng.randrange(len(text)) for _ in range(rng.randint(0, 2)))] if text else []
                stops = [stop for stop in stops if stop] + ["\x00never"]
                stop_ids = [rng.choice(output)] if output and rng.random() < 0.3 else []
                expected = expected_stream(tokenizer, prompt, output, stops, stop_ids)
                stream = TextStream(tokenizer, prompt, stop_strings=stops, stop_token_ids=stop_ids)
                deltas = [stream.push(piece) for piece in chunks(rng, output)] + [stream.finish()]
                with self.subTest(tokenizer=name, prompt=prompt, output=output, stops=stops, stop_ids=stop_ids):
                    self.assertEqual(("".join(delta.text for delta in deltas), stream.consumed, stream.stop,
                                      deltas[-1].finish_reason), expected)
                    self.assertEqual(sum(delta.consumed for delta in deltas), stream.consumed)
                stopped += expected[3] == "stop"
            self.assertGreater(stopped, 50, name)

    def test_finish_reports_length_and_flushed_text_can_still_complete_a_stop(self):
        tokenizer = sp_tokenizer()
        partial = [tokenizer.token_to_id("<0xE6>"), tokenizer.token_to_id("<0x97>")]
        stream = TextStream(tokenizer, stop_strings=[FFFD])
        self.assertEqual(stream.push(partial), TextDelta("", None, None, 2))
        delta = stream.finish()
        self.assertEqual((delta.text, delta.finish_reason, delta.stop), ("", "stop", FFFD))
        plain = TextStream(tokenizer, stop_strings=["zz"])
        self.assertEqual(plain.push(tokenizer.encode("Hello z", add_special_tokens=False)).text, "Hello ")
        self.assertEqual(plain.finish(), TextDelta("z", "length", None, 0))

    def test_invalid_stream_input_closes_the_stream(self):
        tokenizer = byte_tokenizer()
        with self.assertRaises(TokenError):
            TextStream(tokenizer, stop_token_ids=[-1])
        with self.assertRaisesRegex(TextError, "include_stop_str"):
            TextStream(tokenizer, include_stop_str="yes")
        stream = TextStream(tokenizer)
        with self.assertRaises(TokenError):
            stream.push([tokenizer.vocab_size])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            stream.finish()


class ModelTextTests(unittest.TestCase):
    def test_load_model_text_from_a_path_mapping_files_and_callable_files(self):
        files = fx.llama_files()
        with tempfile.TemporaryDirectory() as directory:
            fx.write_files(directory, files)
            from_path = load_model_text(directory, clock=lambda: CLOCK)
            from_directory = load_model_text(DirectoryFiles(directory), clock=lambda: CLOCK)
        source = mapping(files)
        listed = CallableFiles(source.read, names=list(files))
        for model in (from_directory, load_model_text(source, clock=lambda: CLOCK),
                      load_model_text(listed, clock=lambda: CLOCK)):
            self.assertEqual(model.identity.sha256, from_path.identity.sha256)
            self.assertEqual(list(model.templates), ["default"])
            self.assertEqual(model.generation.eos_token_ids, (1, 4))
        with self.assertRaisesRegex(TypeError, "directory path or"):
            load_model_text(42)
        with self.assertRaisesRegex(TypeError, "directory path or"):
            load_model_text(LookAlike())
        with self.assertRaisesRegex(ModelTextError, "not a directory"):
            load_model_text(ROOT / "no-such-model-directory")
        with self.assertRaisesRegex(ModelTextError, "tokenizer.json is missing"):
            load_model_text(mapping({"config.json": {}}))

    def test_encode_chat_takes_bos_from_the_template_and_guards_against_two(self):
        model = load_model_text(mapping(fx.llama_files()), clock=lambda: CLOCK)
        messages = [{"role": "user", "content": "Hello world"}]
        ids = model.encode_chat(messages, add_generation_prompt=True)
        bos = model.tokenizer.bos_token_id
        self.assertEqual(ids[0], bos)
        self.assertNotEqual(ids[1], bos)
        self.assertEqual(ids, model.tokenizer.encode(model.render(messages, add_generation_prompt=True),
                                                     add_special_tokens=False))
        with self.assertRaisesRegex(TextError, "two BOS tokens"):
            model.encode_chat(messages, bos_token="<|begin_of_text|><|begin_of_text|>")
        doubled = load_model_text(mapping(fx.llama_files(**{"chat_template.jinja": "{{ bos_token + bos_token }}x"})))
        with self.assertRaisesRegex(TextError, "two BOS tokens"):
            doubled.encode_chat(messages)

    def test_encode_chat_keeps_the_template_control_tokens_when_the_tokenizer_splits_them(self):
        default = load_model_text(mapping(fx.llama_files()), clock=None)
        hardened = load_model_text(mapping(fx.llama_files()), clock=None, split_special_tokens=True)
        messages = [{"role": "user", "content": "Hi"}]
        ids = hardened.encode_chat(messages, add_generation_prompt=True)
        tokenizer = hardened.tokenizer
        self.assertEqual(ids, default.encode_chat(messages, add_generation_prompt=True))
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        for marker in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
            self.assertIn(tokenizer.token_to_id(marker), ids)
        rendered = hardened.render(messages, add_generation_prompt=True)
        self.assertNotIn(tokenizer.bos_token_id, tokenizer.encode(rendered, add_special_tokens=False))
        with self.assertRaisesRegex(TextError, "two BOS tokens"):
            hardened.encode_chat(messages, bos_token="<|begin_of_text|><|begin_of_text|>")

    def test_a_model_without_a_template_loads_and_render_raises(self):
        files = fx.llama_files()
        del files["chat_template.jinja"]
        model = load_model_text(mapping(files))
        self.assertEqual(len(model.templates), 0)
        with self.assertRaisesRegex(ModelTextError, "no chat template"):
            model.render([{"role": "user", "content": "x"}])
        self.assertEqual(model.tokenizer.decode(model.tokenizer.encode("ok", add_special_tokens=False)), "ok")

    def test_identity_records_exactly_the_files_read(self):
        files = fx.llama_files(**{"special_tokens_map.json": {"bos_token": "x"}, "model.safetensors": b"\0" * 64,
                                  "additional_chat_templates/tool_use.jinja": fx.QWEN3, "notes.txt": "unrelated"})
        source = mapping(files)
        source.read("notes.txt")  # read by someone else before loading: not part of the text identity
        model = load_model_text(source, clock=None)
        self.assertEqual([record.path for record in model.identity.files],
                         ["additional_chat_templates/tool_use.jinja", "chat_template.jinja", "config.json",
                          "generation_config.json", "tokenizer.json", "tokenizer_config.json"])
        record = model.identity.files[-2]
        self.assertEqual((record.bytes, len(record.sha256)), (len(files["tokenizer.json"].encode()), 64))
        self.assertEqual(set(model.identity.versions), {"crit_core", "jinja2", "tokenizers"})
        self.assertEqual(model.identity.row()["sha256"], model.identity.sha256)
        changed = load_model_text(mapping(dict(files, **{"chat_template.jinja": fx.LLAMA3 + " "})), clock=None)
        self.assertNotEqual(changed.identity.sha256, model.identity.sha256)
        self.assertEqual(model.render([{"role": "user", "content": "x"}], tools=[]), "<|im_start|>user\nx<|im_end|>\n")

    def test_text_stream_defaults_to_the_generation_stop_ids(self):
        model = load_model_text(mapping(fx.llama_files()))
        tokenizer = model.tokenizer
        prompt = model.encode_chat([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
        stream = model.text_stream(prompt)
        delta = stream.push(tokenizer.encode("Hello", add_special_tokens=False) + [4, 9])
        self.assertEqual((delta.text, delta.stop, delta.finish_reason), ("Hello", 4, "stop"))
        unstopped = model.text_stream(prompt, stop_token_ids=(), stop_strings=["lo"])
        self.assertEqual(unstopped.push(tokenizer.encode("Hello", add_special_tokens=False)).text, "Hel")


if __name__ == "__main__":
    unittest.main()
