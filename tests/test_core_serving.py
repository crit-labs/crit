"""crit_core.serving.ResidentChat and the package check that keeps resident adapters complete (offline)."""
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import QWEN3, llama_files, write_files  # noqa: E402
from test_core_markers import marked_tokenizer  # noqa: E402
from test_core_tools import BRACKET, XML_TEMPLATE  # noqa: E402
from crit.run.store import serving_gaps  # noqa: E402
from crit_api.engine import reply  # noqa: E402
from crit_api.logprobs import completion_logprobs  # noqa: E402
from crit_core.constraints import ByteGrammar, GrammarState, byte_vocabulary, matches  # noqa: E402
from crit_core.serving import RequestRejected, ResidentChat, capabilities  # noqa: E402
from test_core_reasoning import THINK_TEMPLATE  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
TEMPLATE = XML_TEMPLATE.replace("{%- for message in messages %}",
                                "{{ tools | tojson if tools else '' }}{%- for message in messages %}"
                                "{%- if message.role != 'assistant' %}{{ message.role }}:{{ message.content }};{%- endif %}", 1)


class Decoder(ResidentChat):
    sampling = ("temperature", "seed")

    def __init__(self, context, model):
        super().__init__(context, model, stop_ids=[1])
        self.calls = []

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        self.calls.append((list(ids), max_new_tokens, dict(sampling)))
        if cancelled():
            return None, {}
        reply = self.text.tokenizer.encode("Hi there", add_special_tokens=False)
        for token in reply + [1]:
            on_tokens([token])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


def request(**extra):
    return {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "limits": {"max_output_tokens": 8},
            "messages": [{"role": "user", "content": "hello"}], **extra}


class ResidentChatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": TEMPLATE}))
        self.plain = write_files(Path(temp.name) / "plain", llama_files())
        self.context = types.SimpleNamespace(configuration={"batch": 1}, limits={"kv_tokens": 4096})

    def test_decode_rounds_from_the_engine_timing_reach_the_telemetry(self):
        engine = Decoder(self.context, self.model)
        decode = engine.generate_tokens
        engine.generate_tokens = lambda *args: (lambda tokens, timing: (tokens, dict(timing, rounds=2)))(*decode(*args))
        self.assertEqual(engine.generate(request(), threading.Event())["telemetry"]["decode_rounds"], 2)
        engine.generate_tokens = decode
        self.assertNotIn("decode_rounds", engine.generate(request(), threading.Event())["telemetry"])

    def test_a_reply_ending_on_its_stop_token_at_the_cap_finishes_as_stop(self):
        engine = Decoder(self.context, self.model)
        n = len(engine.text.tokenizer.encode("Hi there", add_special_tokens=False)) + 1  # the reply and its stop token
        at_cap = request(limits={"max_output_tokens": n, "kv_tokens": 4096, "timeout_seconds": 60})
        found = engine.generate(at_cap, threading.Event())
        self.assertEqual((found["telemetry"]["output_tokens"], found["finish"]), (n, "stop"))
        self.assertEqual(reply(found, at_cap)[1]["finish"], "stop")
        self.assertEqual(engine.generate(request(), threading.Event())["finish"], "stop")
        decode = engine.generate_tokens
        engine.generate_tokens = lambda *args: (lambda tokens, timing: (tokens[:-1], timing))(*decode(*args))
        cut = request(limits={"max_output_tokens": n - 1})
        self.assertEqual(engine.generate(cut, threading.Event())["finish"], "length")

    def test_plain_decoding_and_draft_telemetry_reach_the_decoder_and_the_telemetry(self):
        engine = Decoder(self.context, self.model)
        seen = []

        def decode(ids, max_new_tokens, sampling, on_tokens, cancelled, plain=False):
            seen.append(plain)
            tokens, timing = Decoder.generate_tokens(engine, ids, max_new_tokens, sampling, on_tokens, cancelled)
            return tokens, dict(timing, rounds=2, accepted=[1, 0], verify_ms=1.25, draft_ms=0.5)
        engine.generate_tokens = decode
        with self.assertRaisesRegex(ValueError, "does not decode plainly"):
            engine.generate(request(decode="plain"), threading.Event())
        engine.plain = True
        found = engine.generate(request(decode="plain"), threading.Event())["telemetry"]
        self.assertEqual(seen, [True])
        self.assertEqual((found["accepted"], found["verify_ms"], found["draft_ms"]), ([1, 0], 1.25, 0.5))
        engine.generate(request(), threading.Event())
        self.assertEqual(seen, [True, False])
        with self.assertRaisesRegex(ValueError, "decode must be plain"):
            engine.generate(request(decode="greedy"), threading.Event())

    def test_capabilities_follow_the_template_and_the_decoder(self):
        engine = Decoder(self.context, self.model)
        self.assertEqual(capabilities(engine.text, Decoder.sampling),
                         ["chat-json-v1", "chat-resident-v1", "chat-fit-output-v1", "chat-stream-v1",
                          "chat-config-v1", "chat-continue-v1", "completion-v1", "chat-tokenize-v1", "chat-tools-v1",
                          "chat-tool-stream-v1", "chat-sampling-temperature-v1",
                          "chat-sampling-seed-v1"])
        self.assertNotIn("chat-tools-v1", capabilities(Decoder(self.context, self.plain).text))
        self.assertNotIn("chat-continue-v1", capabilities(engine.text, Decoder.sampling, continuation=False))
        self.assertNotIn("completion-v1", capabilities(engine.text, Decoder.sampling, completion=False))
        # A raw prompt reaches an images engine's generate_prepared without its inputs, so such an engine opts in.
        self.assertNotIn("completion-v1", capabilities(engine.text, images=True))
        self.assertIn("completion-v1", capabilities(engine.text, images=True, completion=True))
        self.assertNotIn("chat-tokenize-v1", capabilities(engine.text, Decoder.sampling, tokenize=False))
        with self.assertRaises(ValueError):
            capabilities(engine.text, ["warmth"])

    def test_a_final_assistant_message_is_continued_and_the_reply_is_only_what_follows(self):
        engine = Decoder(self.context, self.plain)
        messages = [{"role": "user", "content": "Write"}, {"role": "assistant", "content": "Once upon a "}]
        emitted = []
        response = engine.generate_stream(request(messages=messages, continue_final_message=True), threading.Event(),
                                          emitted.append)
        self.assertEqual((response["assistant"], "".join(emitted)), ("Hi there", "Hi there"))
        # The template leaves the message open where the generation prompt would start a new reply.
        self.assertEqual(engine.calls[-1][0], engine.text.encode_chat(messages, continue_final_message=True))
        self.assertTrue(engine.text.tokenizer.decode(engine.calls[-1][0]).endswith("Once upon a"))
        engine.generate(request(messages=messages[:1]), threading.Event())
        self.assertEqual(engine.calls[-1][0], engine.text.encode_chat(messages[:1], add_generation_prompt=True))
        generated = len(engine.calls)
        dropped = write_files(self.model.parent / "dropped", llama_files(**{"chat_template.jinja":
            "{% for m in messages %}{% if m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}"}))
        cases = ((engine, request(continue_final_message=True), "needs a final assistant message"),
                 (engine, request(messages=messages, continue_final_message=True, tools=TOOLS, tool_choice="none"),
                  "with tools"),
                 (Decoder(self.context, dropped), request(messages=messages, continue_final_message=True),
                  "template cannot continue the final message"))
        for adapter, refused, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(RequestRejected, reason):
                adapter.generate(refused, threading.Event())
        self.assertEqual(len(engine.calls), generated)  # refused before generating, so the engine stays loaded

    def test_reply_stream_sampling_tools_and_telemetry(self):
        engine = Decoder(self.context, self.model)
        emitted = []
        response = engine.generate_stream(request(sampling={"temperature": 0}), threading.Event(), emitted.append)
        self.assertEqual((response["assistant"], "".join(emitted)), ("Hi there", "Hi there"))
        self.assertEqual(engine.calls[-1][1:], (8, {"temperature": 0}))
        produced = len(engine.text.tokenizer.encode("Hi there", add_special_tokens=False)) + 1
        self.assertEqual(response["telemetry"]["output_tokens"], produced)
        self.assertEqual(response["telemetry"]["decode_tokens"], produced - 1)
        engine.generate(request(tools=TOOLS, tool_choice="auto"), threading.Event())
        self.assertIn("read_file", engine.text.tokenizer.decode(engine.calls[-1][0]))
        engine.generate(request(tools=TOOLS, tool_choice="none"), threading.Event())
        self.assertNotIn("read_file", engine.text.tokenizer.decode(engine.calls[-1][0]))
        self.assertEqual(engine.effective_configuration(), {"batch": 1})
        with self.assertRaisesRegex(ValueError, r"sampling field\(s\): top_p"):
            engine.generate(request(sampling={"top_p": .5}), threading.Event())
        generated = len(engine.calls)
        with self.assertRaisesRegex(ValueError, "exceeds the context") as raised:
            engine.generate(dict(request(), limits={"max_output_tokens": 5000}), threading.Event())
        # A refusal before generation, which crit serve answers with a 400 and the engine stays loaded.
        self.assertIsInstance(raised.exception, RequestRejected)
        self.assertEqual(raised.exception.record(), {"reason": "context_exceeded", "output_tokens": 5000,
                                                     "prompt_tokens": raised.exception.details["prompt_tokens"],
                                                     "capacity": 4096})
        self.assertEqual(len(engine.calls), generated)  # nothing was generated
        self.assertNotIn("max_output_tokens", engine.generate(request(), threading.Event()))
        stop = threading.Event()
        stop.set()
        with self.assertRaises(RuntimeError):
            engine.generate(request(), types.SimpleNamespace(is_set=stop.is_set, check=lambda: None))

    def test_a_fitted_request_decodes_the_room_its_prompt_leaves_and_names_that_cap(self):
        engine = Decoder(self.context, self.model)
        prompt = len(engine.prompt_ids(request()))
        fitted = dict(request(), limits={"max_output_tokens": 4095, "kv_tokens": 4096, "fit_output": True})
        response = engine.generate(fitted, threading.Event())
        self.assertEqual((engine.calls[-1][1], response["max_output_tokens"]), (4096 - prompt, 4096 - prompt))
        self.assertEqual(response["telemetry"]["prompt_tokens"], prompt)
        # A cap that already fits is kept; a prompt that fills the context alone is still refused before generating.
        kept = engine.generate(dict(fitted, limits=dict(fitted["limits"], max_output_tokens=8)), threading.Event())
        self.assertEqual((engine.calls[-1][1], kept["max_output_tokens"]), (8, 8))
        generated = len(engine.calls)
        with self.assertRaises(RequestRejected) as raised:
            engine.generate(dict(fitted, limits=dict(fitted["limits"], kv_tokens=prompt)), threading.Event())
        self.assertEqual(raised.exception.record(), {"reason": "context_exceeded", "prompt_tokens": prompt,
                                                     "output_tokens": 4095, "capacity": prompt})
        self.assertEqual(len(engine.calls), generated)


class TokenizeTests(unittest.TestCase):
    """chat-tokenize-v1: what crit serve's /tokenize, /detokenize, /apply-template and count_tokens read."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": TEMPLATE}))
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_render_gives_the_prompt_a_reply_prefills_and_tokenize_and_detokenize_invert_it(self):
        engine = Decoder(self.context, self.model)
        text, ids = engine.render(request(tools=TOOLS))
        engine.generate(request(tools=TOOLS), threading.Event())
        self.assertEqual(ids, engine.calls[-1][0])
        self.assertIn("read_file", text)
        self.assertEqual((engine.tokenize(text, False), engine.detokenize(ids)), (ids, text))
        plain = engine.tokenize("Hi there", False)
        self.assertEqual(engine.tokenize("Hi there", True), [engine.text.tokenizer.bos_token_id] + plain)
        self.assertNotIn("read_file", engine.render(request(tools=TOOLS, tool_choice="none"))[0])

    def test_invalid_input_and_images_are_refused_before_anything_runs(self):
        engine = Decoder(self.context, self.model)
        image = {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "auto"}}
        for call, expected in ((lambda: engine.detokenize([10**9]), "token id"),
                               (lambda: engine.render(request(messages=[{"role": "user", "content": 5}])), "content"),
                               (lambda: engine.render(request(messages=[{"role": "user", "content": [image]}])),
                                "vision processing")):
            with self.subTest(expected=expected), self.assertRaisesRegex(RequestRejected, expected):
                call()

    def test_render_takes_the_template_variables_and_the_continued_message_a_reply_does(self):
        # A template that reads enable_thinking in the assistant turn the request continues.
        template = ("{%- for m in messages %}{{ m.role }}: {% if m.role == 'assistant' and enable_thinking %}(thought) "
                    "{% endif %}{{ m.content }}\n{% endfor %}{%- if add_generation_prompt %}assistant: {% endif %}")
        model = write_files(self.model.parent / "turn", llama_files(**{"chat_template.jinja": template}))
        engine = Decoder(self.context, model)
        messages = [{"role": "user", "content": "Write"}, {"role": "assistant", "content": "Once upon a"}]
        for options, ending in (({"template_kwargs": {"enable_thinking": True}}, "assistant: (thought) Once upon a"),
                                ({}, "assistant: Once upon a")):
            with self.subTest(options=options):
                continued = request(messages=messages, continue_final_message=True, **options)
                text, ids = engine.render(continued)
                engine.generate(continued, threading.Event())
                self.assertEqual(ids, engine.calls[-1][0])
                self.assertTrue(text.endswith(ending), text)
        self.assertTrue(engine.render(request())[0].endswith("assistant: "))
        for refused in (request(messages=messages, continue_final_message=True, template_kwargs={"bos_token": "<s>"}),
                        request(continue_final_message=True)):
            with self.subTest(refused=refused), self.assertRaises(RequestRejected):
                engine.render(refused)


class XmlDecoder(Decoder):
    """A model whose chat template teaches a syntax crit registers: ResidentChat parses it with no package code."""
    reply = ("Let me read it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n"
             "</function>\n</tool_call>")

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        reply = self.text.tokenizer.encode(self.reply, add_special_tokens=False)
        for token in reply + [1]:
            on_tokens([token])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class TemplateToolFormatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": TEMPLATE}))
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_calls_in_the_templates_registered_format_come_back_parsed(self):
        engine = XmlDecoder(self.context, self.model)
        emitted = []
        response = engine.generate_stream(request(tools=TOOLS), threading.Event(), emitted.append)
        self.assertEqual(response["assistant"], "Let me read it.")
        self.assertEqual([c["function"] for c in response["tool_calls"]],
                         [{"name": "read_file", "arguments": '{"path": "a.py"}'}])
        self.assertEqual("".join(e for e in emitted if isinstance(e, str)), "Let me read it.")
        # The call streams from its name on: a header, then its arguments as the model writes them.
        deltas = [e["tool_calls"][0] for e in emitted if isinstance(e, dict)]
        self.assertGreater(len(deltas), 1)
        self.assertEqual({d["index"] for d in deltas}, {0})
        self.assertEqual((deltas[0]["type"], deltas[0]["function"]["name"]), ("function", "read_file"))
        self.assertTrue(all(set(d) == {"index", "function"} and set(d["function"]) == {"arguments"} for d in deltas[1:]))
        self.assertEqual("".join(d["function"]["arguments"] for d in deltas), '{"path": "a.py"}')
        for kept in (engine.generate(request(tools=TOOLS, tool_choice="none"), threading.Event()),
                     engine.generate(request(), threading.Event())):
            self.assertEqual(kept["assistant"], XmlDecoder.reply)
            self.assertNotIn("tool_calls", kept)
        with self.assertRaisesRegex(RequestRejected, "tool_choice must be auto or none"):
            engine.generate(request(tools=TOOLS, tool_choice="required"), threading.Event())


class CutDecoder(XmlDecoder):
    """Writes its reply up to max_new_tokens, or with `end` its first `end` tokens and its end token: a reply cut
    inside its call, or ended there."""
    end = None

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        reply = self.text.tokenizer.encode(self.reply, add_special_tokens=False)
        reply = reply[:max_new_tokens] if self.end is None else reply[:self.end] + [1]
        for token in reply:
            on_tokens([token])
        return reply, {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class CutCallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": TEMPLATE}))
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_a_reply_cut_inside_a_call_keeps_it_and_one_ended_there_fails(self):
        engine = CutDecoder(self.context, self.model)
        tokens = engine.text.tokenizer.encode(XmlDecoder.reply, add_special_tokens=False)
        named = next(n for n in range(len(tokens)) if "<function=read_file>" in engine.text.tokenizer.decode(tokens[:n]))
        for limit in (named, len(tokens) - 1):
            with self.subTest(limit=limit):
                emitted = []
                response = engine.generate_stream(request(tools=TOOLS, limits={"max_output_tokens": limit}),
                                                  threading.Event(), emitted.append)
                self.assertEqual(response["assistant"], "Let me read it.")
                [call] = response["tool_calls"]
                self.assertEqual(call["function"]["name"], "read_file")
                self.assertTrue(call["function"]["arguments"].startswith("{"))
                self.assertFalse(call["function"]["arguments"].endswith("}"))
                self.assertEqual("".join(e["tool_calls"][0]["function"]["arguments"] for e in emitted
                                         if isinstance(e, dict)), call["function"]["arguments"])
        engine.end = named
        with self.assertRaisesRegex(ValueError, "ended its reply inside its call to read_file"):
            engine.generate(request(tools=TOOLS, limits={"max_output_tokens": len(tokens)}), threading.Event())


class BracketDecoder(Decoder):
    """A package that parses its model's own call syntax (crit registers none for it)."""
    tool_format = BRACKET
    reply = 'Sure. [[call read_file {"path": "a.py"}]]'

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled):
        self.calls.append((list(ids), max_new_tokens, dict(sampling)))
        reply = self.text.tokenizer.encode(self.reply, add_special_tokens=False)
        for token in reply + [1]:
            on_tokens([token])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class PackageToolFormatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.plain = write_files(Path(temp.name) / "plain", llama_files())
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_the_package_parses_calls_and_declares_chat_tools(self):
        engine = BracketDecoder(self.context, self.plain)
        declared = capabilities(engine.text, BracketDecoder.sampling, tool_format=BracketDecoder.tool_format)
        self.assertIn("chat-tools-v1", declared)
        self.assertNotIn("chat-tool-choice-v1", declared)
        self.assertIn("chat-tool-stream-v1", declared)
        emitted = []
        response = engine.generate_stream(request(tools=TOOLS, tool_choice="auto"), threading.Event(), emitted.append)
        self.assertEqual(response["assistant"], "Sure.")
        self.assertEqual([c["function"]["name"] for c in response["tool_calls"]], ["read_file"])
        # Its format reads no name before the call closes, so the call arrives whole.
        self.assertEqual([e["tool_calls"][0]["function"] for e in emitted if isinstance(e, dict)],
                         [{"name": "read_file", "arguments": '{"path": "a.py"}'}])
        plain = engine.generate(request(tools=TOOLS, tool_choice="none"), threading.Event())
        self.assertEqual((plain["assistant"], "tool_calls" in plain), (BracketDecoder.reply, False))
        self.assertNotIn("tool_calls", engine.generate(request(), threading.Event()))
        with self.assertRaisesRegex(RequestRejected, "tool_choice must be auto or none"):
            engine.generate(request(tools=TOOLS, tool_choice="required"), threading.Event())


class TemplateKwargsTests(unittest.TestCase):
    """A request sets the thinking switch the chat template reads (chat-template-kwargs-v1) over the configured one."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": QWEN3}))
        self.context = types.SimpleNamespace(configuration={"chat_template_kwargs": {"enable_thinking": False}},
                                             limits={"kv_tokens": 4096})

    def test_a_request_sets_the_variables_the_template_reads_for_that_request_only(self):
        engine = Decoder(self.context, self.model)
        declared = capabilities(engine.text, Decoder.sampling)
        self.assertEqual([c for c in declared if c.startswith("chat-template-")],
                         ["chat-template-kwargs-v1", "chat-template-enable-thinking-v1"])
        prompt = lambda: engine.text.tokenizer.decode(engine.calls[-1][0])
        engine.generate(request(), threading.Event())
        self.assertTrue(prompt().endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))   # configured: off
        engine.generate(request(template_kwargs={"enable_thinking": True}), threading.Event())
        self.assertTrue(prompt().endswith("<|im_start|>assistant\n"))
        engine.generate(request(), threading.Event())
        self.assertTrue(prompt().endswith("</think>\n\n"))
        self.assertEqual(engine.template_variables, {"enable_thinking": False})
        generated = len(engine.calls)
        for refused in ({"reasoning_effort": "low"}, {"enable_thinking": 1}, {"bos_token": "<s>"}):
            with self.subTest(refused=refused), self.assertRaises(RequestRejected):
                engine.generate(request(template_kwargs=refused), threading.Event())
        self.assertEqual(len(engine.calls), generated)

    def test_without_its_own_kwargs_an_engine_serves_the_variables_recon_froze(self):
        frozen = {"recon": {"template_variables": {"enable_thinking": False}}}
        engine = Decoder(types.SimpleNamespace(configuration=frozen, limits={"kv_tokens": 4096}), self.model)
        self.assertEqual(engine.template_variables, {"enable_thinking": False})
        engine.generate(request(), threading.Event())
        self.assertTrue(engine.text.tokenizer.decode(engine.calls[-1][0]).endswith("</think>\n\n"))
        # The engine's own chat_template_kwargs win, an empty one included.
        own = Decoder(types.SimpleNamespace(configuration={**frozen, "chat_template_kwargs": {}}, limits={}), self.model)
        self.assertEqual(own.template_variables, {})

    def test_a_continued_final_message_renders_with_the_request_variables_too(self):
        # A template that reads enable_thinking in the assistant turn the request continues.
        template = ("{%- for m in messages %}{{ m.role }}: {% if m.role == 'assistant' and enable_thinking %}(thought) "
                    "{% endif %}{{ m.content }}\n{% endfor %}{%- if add_generation_prompt %}assistant: {% endif %}")
        model = write_files(self.model.parent / "turn", llama_files(**{"chat_template.jinja": template}))
        engine = Decoder(self.context, model)
        messages = [{"role": "user", "content": "Write"}, {"role": "assistant", "content": "Once upon a"}]
        prompt = lambda: engine.text.tokenizer.decode(engine.calls[-1][0])
        engine.generate(request(messages=messages, continue_final_message=True,
                                template_kwargs={"enable_thinking": True}), threading.Event())
        self.assertTrue(prompt().endswith("assistant: (thought) Once upon a"), prompt())
        engine.generate(request(messages=messages, continue_final_message=True), threading.Event())
        self.assertTrue(prompt().endswith("assistant: Once upon a"), prompt())   # configured: off
        generated = len(engine.calls)
        with self.assertRaises(RequestRejected):
            engine.generate(request(messages=messages, continue_final_message=True,
                                    template_kwargs={"bos_token": "<s>"}), threading.Event())
        self.assertEqual(len(engine.calls), generated)


class GrammarDecoder(XmlDecoder):
    """Keeps its reply to a grammar (grammars): when a request forces calls it writes `forced`, each token checked as
    a matcher over the tokenizer's byte vocabulary would allow it, and its end token only where the grammar ends."""
    grammars = True
    forced = "<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, grammar=None):
        self.grammar = grammar
        if grammar is None:
            return super().generate_tokens(ids, max_new_tokens, sampling, on_tokens, cancelled)
        reply = self.text.tokenizer.encode(self.forced, add_special_tokens=False)
        pieces, state = byte_vocabulary(self.text.tokenizer, 1).pieces, GrammarState(ByteGrammar.load(grammar))
        for token in reply:
            state = state.advance(pieces[token])
            if state is None:
                raise AssertionError(f"the grammar does not allow token {token}")
            on_tokens([token])
        if not state.complete:
            raise AssertionError("the grammar does not allow the end token")
        on_tokens([1])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class ForcedCallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def model(self, name, template=TEMPLATE, **files):
        return write_files(self.root / name, llama_files(**{"chat_template.jinja": template, **files}))

    def test_a_forced_choice_brings_the_decoder_the_grammar_of_its_calls(self):
        engine = GrammarDecoder(self.context, self.model("model"))
        self.assertIn("chat-tool-choice-v1", capabilities(engine.text, Decoder.sampling, grammars=True))
        self.assertNotIn("chat-tool-choice-v1", capabilities(engine.text, Decoder.sampling))
        call = [{"name": "read_file", "arguments": '{"path": "a.py"}'}]
        for choice in ("required", {"type": "function", "function": {"name": "read_file"}}):
            emitted = []
            response = engine.generate_stream(request(tools=TOOLS, tool_choice=choice), threading.Event(),
                                              emitted.append)
            self.assertEqual((response["assistant"], [c["function"] for c in response["tool_calls"]]), ("", call))
            self.assertTrue(emitted and all(set(e) == {"tool_calls"} for e in emitted))   # the call streams, no text
            # The free reply, text before its call, is outside the grammar; one or more calls are inside.
            self.assertFalse(matches(engine.grammar, XmlDecoder.reply.encode()))
            self.assertTrue(matches(engine.grammar, (GrammarDecoder.forced + "\n" + GrammarDecoder.forced).encode()))
        engine.generate(request(tools=TOOLS, tool_choice="required", parallel_tool_calls=False), threading.Event())
        self.assertFalse(matches(engine.grammar, (GrammarDecoder.forced * 2).encode()))
        self.assertEqual(engine.generate(request(tools=TOOLS), threading.Event())["assistant"], "Let me read it.")
        self.assertIsNone(engine.grammar)   # auto decodes freely

    def test_a_reasoning_the_prompt_opens_closes_before_the_call(self):
        thinking = TEMPLATE + "{% if add_generation_prompt %}<think>\n{% endif %}{# </think> #}"
        engine = GrammarDecoder(self.context, self.model("thinking", thinking))
        engine.forced = "Read a.py first.\n</think>\n\n" + GrammarDecoder.forced
        response = engine.generate(request(tools=TOOLS, tool_choice="required"), threading.Event())
        self.assertEqual((response["reasoning"], response["assistant"], len(response["tool_calls"])),
                         ("Read a.py first.", "", 1))
        self.assertFalse(matches(engine.grammar, GrammarDecoder.forced.encode()))
        # Reasoning markers held as special tokens cannot be written by a grammar, so such a reply cannot reach a call.
        special = marked_tokenizer(markers=("<think>", "</think>")).backend.to_str()
        engine = GrammarDecoder(self.context, self.model("special", thinking, **{"tokenizer.json": special}))
        with self.assertRaisesRegex(RequestRejected, "reasoning whose markers a grammar cannot write"):
            engine.generate(request(tools=TOOLS, tool_choice="required"), threading.Event())

    def test_calls_a_grammar_cannot_write_are_not_forced(self):
        special = marked_tokenizer().backend.to_str()   # <tool_call> as a special token
        bracket = BracketDecoder(self.context, write_files(self.root / "plain", llama_files()))
        bracket.grammars = True   # its own syntax, which crit does not write
        for engine in (GrammarDecoder(self.context, self.model("special", **{"tokenizer.json": special})), bracket):
            with self.subTest(format=engine.tool_format):
                self.assertNotIn("chat-tool-choice-v1", capabilities(engine.text, Decoder.sampling, grammars=True,
                                                                     tool_format=engine.tool_format))
                with self.assertRaisesRegex(RequestRejected, "tool_choice must be auto or none"):
                    engine.generate(request(tools=TOOLS, tool_choice="required"), threading.Event())


class LogprobDecoder(Decoder):
    """Reports every token's log-probability and its alternatives: ids t, t+1, ... one nat apart."""
    logprobs = True

    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, logprobs=None):
        reply = self.text.tokenizer.encode("Hi there", add_special_tokens=False)
        for token in reply + [1]:
            if logprobs is None:
                on_tokens([token])
            else:
                own = float("-inf") if token == reply[0] else 1e-7   # Core allows both: no probability, rounding
                on_tokens([token], [(own, [(token + k, -1.0 * k) for k in range(logprobs)])])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class LogprobTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "plain", llama_files())
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_each_generated_token_but_the_stop_token_comes_with_its_alternatives_and_bytes(self):
        engine = LogprobDecoder(self.context, self.model)
        self.assertIn("chat-logprobs-v1", capabilities(engine.text, LogprobDecoder.sampling, logprobs=True))
        self.assertNotIn("chat-logprobs-v1", capabilities(engine.text, LogprobDecoder.sampling))
        emitted = []
        response = engine.generate_stream(request(logprobs=2), threading.Event(), emitted.append)
        reply = engine.text.tokenizer.encode("Hi there", add_special_tokens=False)
        rows, named = response["logprobs"]["tokens"], response["logprobs"]["bytes"]
        self.assertEqual([row[0] for row in rows], reply)
        self.assertEqual([row[1] for row in rows], [-9999.0] + [0.0] * (len(reply) - 1))
        self.assertEqual(rows[0][2], [[reply[0], 0.0], [reply[0] + 1, -1.0]])
        self.assertEqual(b"".join(bytes.fromhex(named[str(row[0])]) for row in rows).decode().strip(), "Hi there")
        self.assertEqual(set(named), {str(t) for row in rows for t in (row[0], *(o for o, _ in row[2]))})
        self.assertEqual("".join(emitted), "Hi there")
        self.assertNotIn("logprobs", engine.generate(request(), threading.Event()))
        self.assertEqual(engine.generate(request(logprobs=0), threading.Event())["logprobs"]["tokens"][1][2], [])

    def test_missing_rows_and_engines_without_logprobs_fail(self):
        class Bare(LogprobDecoder):
            def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, logprobs=None):
                on_tokens([5])
                return [5], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}
        with self.assertRaisesRegex(ValueError, "without their log-probabilities"):
            Bare(self.context, self.model).generate(request(logprobs=1), threading.Event())
        with self.assertRaisesRegex(ValueError, "does not report log-probabilities"):
            Decoder(self.context, self.model).generate(request(logprobs=1), threading.Event())
        with self.assertRaisesRegex(ValueError, r"integer in \[0, 20\]"):
            LogprobDecoder(self.context, self.model).generate(request(logprobs=21), threading.Event())


class Thinker(Decoder):
    """Writes reasoning in the template's markers, then its reply; alternatives t, t+1, ... wrap at the vocabulary."""
    def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, logprobs=None):
        self.calls.append((list(ids), max_new_tokens, dict(sampling)))
        reply = self.text.tokenizer.encode("plan\n</think>\n\nHello there", add_special_tokens=False)
        for token in reply + [1]:
            alternatives = [((token + k) % self.text.tokenizer.vocab_size, -1.0 * k) for k in range(logprobs or 0)]
            on_tokens([token], None if logprobs is None else [(-0.5, alternatives)])
        return reply + [1], {"prefill_ms": 2.0, "decode_ms": 3.0, "first_token_ms": 2.5}


class Scorer(Thinker):
    """Reports log-probabilities and scores a prompt's positions (chat-score-v1): ids t, t+1, ... one nat apart."""
    logprobs = scores = True

    def score_tokens(self, ids, continuation, top_k, cancelled, prompt_rows):
        self.scored = (list(ids), list(continuation), prompt_rows)
        return [(-0.25, [(t + k, -1.0 * k) for k in range(top_k)]) for t in ids[1:]], [], {"prefill_ms": 1.0,
                                                                                          "decode_ms": None}


def raw(prompt, **extra):
    return {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "prompt": prompt,
            "limits": {"max_output_tokens": 64, "kv_tokens": 4096, "timeout_seconds": 10}, **extra}


class RawPromptTests(unittest.TestCase):
    """completion-v1: a raw prompt, its text or token ids, continued without the chat template."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = write_files(Path(temp.name) / "model", llama_files(**{"chat_template.jinja": THINK_TEMPLATE}))
        self.context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 4096})

    def test_text_is_encoded_with_one_bos_and_ids_pass_as_given(self):
        engine = Thinker(self.context, self.model)
        tokenizer = engine.text.tokenizer
        ids = tokenizer.encode("Hello world", add_special_tokens=True)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        for prompt in ({"text": "Hello world"}, {"text": "<|begin_of_text|>Hello world"}, {"ids": ids}):
            with self.subTest(prompt=prompt):
                response = engine.generate(raw(prompt), threading.Event())
                self.assertEqual(engine.calls[-1][0], ids)   # no template and a single BOS
                # The reply is all the text the model wrote: no reasoning split without the template.
                self.assertEqual(response["assistant"], "plan\n</think>\n\nHello there")
                self.assertNotIn("reasoning", response)
                self.assertNotIn("prompt", response)
        chat = engine.generate(request(), threading.Event())
        self.assertEqual((chat["assistant"], chat["reasoning"]), ("Hello there", "plan"))
        generated = len(engine.calls)
        for prompt in ({"ids": [5, tokenizer.vocab_size]}, {"text": "\ud800"}):
            with self.subTest(prompt=prompt), self.assertRaises(RequestRejected):
                engine.generate(raw(prompt), threading.Event())
        self.assertEqual(len(engine.calls), generated)

    def test_echo_returns_the_prompt_and_with_logprobs_its_scored_rows(self):
        engine = Scorer(self.context, self.model)
        ids = engine.text.tokenizer.encode("Hi there", add_special_tokens=True)
        self.assertEqual(engine.generate(raw({"text": "Hi there"}, echo=True), threading.Event())["prompt"], "Hi there")
        current = raw({"ids": ids}, echo=True, logprobs=2)
        response = engine.generate(current, threading.Event())
        self.assertEqual(response["prompt"], "<|begin_of_text|>Hi there")
        self.assertEqual(engine.scored, (ids, [], True))
        rows = response["logprobs"]["prompt"]
        self.assertEqual(rows[0], [ids[0], None, []])   # nothing precedes the first token
        self.assertEqual(rows[1:], [[t, -0.25, [[t, 0.0], [t + 1, -1.0]]] for t in ids[1:]])
        # crit serve checks the rows and gives the prompt's entries first.
        assistant, measured = reply(response, current)
        self.assertEqual(measured["prompt_tokens"], len(ids))
        self.assertEqual(len(assistant["logprobs"]), len(ids) + measured["output_tokens"])   # the stop token's too
        self.assertEqual((assistant["logprobs"][0]["token"], assistant["logprobs"][0]["logprob"]),
                         ("<|begin_of_text|>", None))
        self.assertEqual(assistant["prompt"], "<|begin_of_text|>Hi there")
        with self.assertRaisesRegex(RequestRejected, "chat-score-v1"):
            LogprobDecoder(self.context, self.model).generate(raw({"ids": ids}, echo=True, logprobs=1),
                                                               threading.Event())

    def test_a_stop_token_keeps_its_row_where_lm_eval_reads_one_past_the_prompt(self):
        class Ender(Scorer):
            def generate_tokens(self, ids, max_new_tokens, sampling, on_tokens, cancelled, logprobs=None):
                on_tokens([1], [(-0.5, [(1, -0.5)])])   # the model ends at once
                return [1], {"prefill_ms": 2.0, "decode_ms": None, "first_token_ms": 2.5}
        engine = Ender(self.context, self.model)
        tokenizer = engine.text.tokenizer
        context = tokenizer.encode("Question: 2+2? Answer:", add_special_tokens=True)
        continuation = tokenizer.encode(" 4", add_special_tokens=False)
        # lm-eval's local-completions sends echo, logprobs 1 and max_tokens 1 and sums token_logprobs[ctxlen:-1].
        current = raw({"ids": context + continuation}, echo=True, logprobs=1,
                      limits={"max_output_tokens": 1, "kv_tokens": 4096, "timeout_seconds": 10})
        response = engine.generate(current, threading.Event())
        self.assertEqual((response["assistant"], response["logprobs"]["tokens"]), ("", [[1, -0.5, [[1, -0.5]]]]))
        found = completion_logprobs(reply(response, current)[0]["logprobs"])
        self.assertEqual((found["tokens"][-1], found["token_logprobs"][-1]), ("<|end_of_text|>", -0.5))
        self.assertEqual(found["token_logprobs"][len(context):-1], [-0.25] * len(continuation))
        # A chat reply still leaves its stop token out.
        self.assertEqual(engine.generate(request(logprobs=1), threading.Event())["logprobs"]["tokens"], [])


class PackageGapTests(unittest.TestCase):
    def test_resident_packages_need_temperature_and_tools_when_the_template_teaches_them(self):
        with tempfile.TemporaryDirectory() as temp:
            tools = write_files(Path(temp) / "tools", llama_files(**{"chat_template.jinja": TEMPLATE}))
            plain = write_files(Path(temp) / "plain", llama_files())
            resident = ["chat-json-v1", "chat-resident-v1"]
            self.assertEqual(serving_gaps(["prefill", "decode"], tools), [])
            self.assertEqual(serving_gaps(resident + ["chat-sampling-temperature-v1"], plain), [])
            self.assertEqual(len(serving_gaps(resident, plain)), 1)
            gaps = serving_gaps(resident + ["chat-sampling-temperature-v1"], tools)
            self.assertEqual(len(gaps), 1)
            self.assertIn("chat-tools-v1", gaps[0])
            for parsed in ("chat-tools-v1", "chat-tool-template-v1"):   # the second: packages sealed before
                self.assertEqual(serving_gaps(resident + ["chat-sampling-temperature-v1", parsed], tools), [])
            self.assertEqual(serving_gaps(resident + ["chat-sampling-temperature-v1"], Path(temp) / "model.gguf"), [])


if __name__ == "__main__":
    unittest.main()
