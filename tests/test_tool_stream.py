"""Strict tools and incremental calls, including real resident IPC and HTTP SSE."""
import copy
import http.client
import json
import threading
import time
import unittest

import test_api
import test_chat_stream
from crit_api.options import (RequestError, STRICT_TOOLS_CAPABILITY, TOOLS_CAPABILITY, tool_options,
                              validate_tool_reply)

TOOLS = [{"type": "function", "function": {"name": "read_file", "strict": True, "parameters": {
    "type": "object", "properties": {"path": {"type": "string", "maxLength": 32}},
    "required": ["path"], "additionalProperties": False}}}]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}


class StrictSchemaTests(unittest.TestCase):
    def test_compiles_and_checks_arguments_and_rejects_unsupported_schemas(self):
        options = tool_options({"tools": TOOLS}, [TOOLS_CAPABILITY, STRICT_TOOLS_CAPABILITY])
        self.assertIn("crit-byte-grammar-v1", options["tool_grammars"]["read_file"])
        self.assertEqual(validate_tool_reply([CALL], options), [CALL])
        for arguments in ('{}', '{"path":2}', '{"path":"x","extra":1}', '{"path":"' + 'x' * 33 + '"}'):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_tool_reply([dict(CALL, function={"name": "read_file", "arguments": arguments})], options)
        # A call a cut reply ends with: its arguments so far must begin arguments the schema accepts.
        for arguments, valid in (('{"path":"a', True), ('{', True), ('{"pa', True), ('{"path":2', False),
                                 ('{"wrong"', False)):
            with self.subTest(arguments=arguments):
                call = dict(CALL, function={"name": "read_file", "arguments": arguments})
                if valid:
                    self.assertEqual(validate_tool_reply([call], options, truncated=True), [call])
                else:
                    with self.assertRaisesRegex(ValueError, "strict schema"):
                        validate_tool_reply([call], options, truncated=True)
                with self.assertRaises(ValueError):
                    validate_tool_reply([call], options)
        with self.assertRaises(RequestError):
            tool_options({"tools": TOOLS}, [TOOLS_CAPABILITY])
        for schema in ({"type": "object", "additionalProperties": True},
                       {"type": "object", "properties": {"path": {"type": "string", "pattern": "a+"}},
                        "additionalProperties": False}):
            tools = copy.deepcopy(TOOLS)
            tools[0]["function"]["parameters"] = schema
            with self.assertRaises(RequestError) as caught:
                tool_options({"tools": tools}, [TOOLS_CAPABILITY, STRICT_TOOLS_CAPABILITY])
            self.assertEqual(caught.exception.code, "unsupported_schema")


class ToolStreamContractTests(test_chat_stream.StreamContractTests):
    def setUp(self):
        super().setUp()
        self.request.update(tool_options({"tools": TOOLS}, [TOOLS_CAPABILITY, STRICT_TOOLS_CAPABILITY]))

    def test_interleaved_calls_preserve_exact_arguments(self):
        writer, text, deltas = self.writer(tools=True), [], []
        reader = self.reader(text, deltas.extend)
        calls = [CALL, dict(CALL, id="call_2")]
        for i, call in enumerate(calls):
            writer.emit({"tool_calls": [dict(call, index=i, function={"name": "read_file", "arguments": "{"})]})
        for i in (1, 0):
            writer.emit({"tool_calls": [{"index": i, "function": {"arguments": CALL["function"]["arguments"][1:]}}]})
        writer.finish({"assistant": "", "tool_calls": calls})
        for piece in self.sent:
            reader.apply(piece)
        rebuilt = {}
        for delta in deltas:
            index = delta["index"]
            rebuilt.setdefault(index, {"id": delta.get("id"), "name": delta["function"].get("name"), "arguments": ""})
            rebuilt[index]["arguments"] += delta["function"].get("arguments", "")
        self.assertEqual([r["arguments"] for r in rebuilt.values()], [CALL["function"]["arguments"]] * 2)
        self.assertEqual([r["id"] for r in rebuilt.values()], ["call_1", "call_2"])
        self.assertEqual(text, [])

    def test_bad_prefix_and_swallowed_callback_errors_fail_the_turn(self):
        writer = self.writer(tools=True)
        with self.assertRaisesRegex(ValueError, "strict tool schema"):
            writer.emit({"tool_calls": [dict(CALL, index=0, function={"name": "read_file", "arguments": '{"wrong"'})]})
        self.assertEqual(self.sent, [])
        with self.assertRaises(ValueError):
            writer.finish({"assistant": "", "tool_calls": [CALL]})

    def test_adapter_request_edits_cannot_disable_admitted_constraints(self):
        writer = self.writer(tools=True)
        self.request['tool_grammars'].clear()
        with self.assertRaisesRegex(ValueError, 'strict tool schema'):
            writer.emit({'tool_calls': [dict(CALL, index=0, function={'name': 'read_file', 'arguments': '{}'})]})

    def test_rewrites_missing_fragments_and_policy_violations_are_rejected(self):
        for delta in ({"index": True, "function": {}}, dict(CALL, index=1),
                      dict(CALL, index=0, function={"name": "unknown", "arguments": ""})):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                self.writer(tools=True).emit({"tool_calls": [delta]})
        writer = self.writer(tools=True)
        writer.emit({"tool_calls": [dict(CALL, index=0, function={"name": "read_file", "arguments": "{"})]})
        with self.assertRaisesRegex(ValueError, "differ"):
            writer.finish({"assistant": "", "tool_calls": [CALL]})
        reader = self.reader([])
        reader.apply(self.sent[0])
        for delta in (dict(CALL, index=0, id="rewritten"), {"index": 0, "function": {"arguments": '"x"}'}}):
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, "Later tool deltas|strict"):
                reader.apply(self.piece(tool_calls=[delta]))


ADAPTER = r'''
from pathlib import Path
class ToolEngine(Engine):
    def generate_stream(self, request, cancellation, emit):
        message = request['messages'][-1]['content']
        if not message.startswith('stream-tools:'):
            return super().generate_stream(request, cancellation, emit)
        data = json.loads(message[len('stream-tools:'):])
        calls = data['calls']
        for index, call in enumerate(calls):
            emit({'tool_calls': [dict(call, index=index, function={'name': call['function']['name'], 'arguments': ''})]})
        if data.get('hold'):
            root = Path(request['model_path']).parent
            (root / 'holding').touch()
            while not (root / 'release').exists():
                cancellation.check()
                time.sleep(.01)
        for index, call in enumerate(calls):
            for char in call['function']['arguments']:
                cancellation.check()
                emit({'tool_calls': [{'index': index, 'function': {'arguments': char}}]})
                time.sleep(.01)
        response = self.response(request, '')
        response['tool_calls'] = calls
        return response
def create_engine(context):
    return ToolEngine(context)
'''


class AdvancedToolTests(test_api.ApiTests):
    request_options = strict_tools = tool_stream = True
    adapter_suffix = ADAPTER

    def test_strict_schema_reaches_adapter_and_unsupported_schema_returns_400(self):
        response, value = self.chat('echo-request', tools=TOOLS)
        self.assertEqual(response.status, 200, value)
        self.assertIn('read_file', json.loads(value['choices'][0]['message']['content'])['tool_grammars'])
        response, value = self.chat('stream-tools:' + json.dumps({'calls': [CALL]}), tools=TOOLS)
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value['choices'][0]['message']['tool_calls'], [CALL])
        bad_call = dict(CALL, function={'name': 'read_file', 'arguments': '{}'})
        response, value = self.chat('stream-tools:' + json.dumps({'calls': [bad_call]}), tools=TOOLS,
                                   tool_choice='required', parallel_tool_calls=False)
        self.assertEqual(response.status, 500, value)
        bad = copy.deepcopy(TOOLS)
        bad[0]['function']['parameters']['additionalProperties'] = True
        response, value = self.chat(tools=bad)
        self.assertEqual((response.status, value['error']['code']), (400, 'unsupported_schema'))

    def test_sse_tool_header_arrives_before_generation_finishes_and_arguments_are_incremental(self):
        connection = http.client.HTTPConnection(*self.server.server_address[:2], timeout=15)
        self.addCleanup(connection.close)
        body = {'messages': [{'role': 'user', 'content': 'stream-tools:' + json.dumps({'calls': [CALL], 'hold': True})}],
                'tools': TOOLS, 'stream': True, 'stream_options': {'include_usage': True}}
        connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        events = []
        while not any(e['choices'][0]['delta'].get('tool_calls') for e in events):
            line = response.readline()
            if line.startswith(b'data: {'):
                events.append(json.loads(line[6:]))
        self.assertFalse((self.root / 'release').exists())
        self.assertEqual(self.served.status()['state'], 'generating')
        (self.root / 'release').touch()
        tail = response.read().decode()
        events += [json.loads(e[6:]) for e in tail.splitlines() if e.startswith('data: {')]
        deltas = [c['choices'][0]['delta']['tool_calls'][0] for c in events
                  if c.get('choices') and c['choices'][0]['delta'].get('tool_calls')]
        self.assertGreater(len(deltas), 2)
        self.assertEqual(sum('id' in d for d in deltas), 1)
        self.assertEqual(''.join(d['function'].get('arguments', '') for d in deltas), CALL['function']['arguments'])
        self.assertEqual(events[-2]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(events[-1]['usage']['completion_tokens'], 4)
        self.assertTrue(tail.endswith('data: [DONE]\n\n'))

    def test_a_call_the_output_limit_cuts_comes_back_as_written_and_finishes_as_length(self):
        cut = dict(CALL, function={'name': 'read_file', 'arguments': '{"path":"a'})
        message = 'stream-tools:' + json.dumps({'calls': [cut]})
        response, value = self.chat(message, tools=TOOLS, max_tokens=4)  # the fixture's replies are 4 tokens
        self.assertEqual(response.status, 200, value)
        self.assertEqual((value['choices'][0]['finish_reason'], value['choices'][0]['message']['tool_calls']),
                         ('length', [cut]))
        response, data = self.chat(message, tools=TOOLS, max_tokens=4, stream=True)
        events = [json.loads(e[6:]) for e in data.decode().splitlines() if e.startswith('data: {')]
        self.assertEqual([e['choices'][0]['finish_reason'] for e in events if e.get('choices')][-1], 'length')
        deltas = [d for e in events if e.get('choices') for d in e['choices'][0]['delta'].get('tool_calls', [])]
        self.assertEqual(''.join(d['function'].get('arguments', '') for d in deltas), cut['function']['arguments'])
        # A reply that was not cut and ends inside a call wrote a malformed call.
        response, value = self.chat(message, tools=TOOLS, max_tokens=30)
        self.assertEqual(response.status, 500, value)

    def test_cancelling_tool_stream_keeps_resident_model_and_private_state_is_reset(self):
        cancel, seen = threading.Event(), []
        def tools(delta):
            seen.extend(delta)
            cancel.set()
        options = tool_options({'tools': TOOLS}, self.served.spec()['capabilities'])
        status, assistant, measured = self.served.generate(
            [{'role': 'user', 'content': 'stream-tools:' + json.dumps({'calls': [CALL], 'hold': True})}],
            30, cancel, tool_options=options, on_tools=tools)
        self.assertEqual((status, assistant, measured), ('cancelled', None, None))
        self.assertTrue(seen)
        response, value = self.chat()
        self.assertEqual(response.status, 200, value)
        self.assertEqual(self.served.status()['engine_loads'], 1)

    def test_tool_consumer_exception_retires_producer_before_next_request(self):
        root, tree = self.served.runtime.root, self.served.runtime.tree
        self.addCleanup(self.served._retire, "tool consumer regression cleanup")
        error, seen = RuntimeError("tool consumer failed"), []

        def tools(delta):
            seen.extend(delta)
            raise error

        options = tool_options({'tools': TOOLS}, self.served.spec()['capabilities'])
        with self.assertRaises(RuntimeError) as caught:
            self.served.generate(
                [{'role': 'user', 'content': 'stream-tools:' + json.dumps({'calls': [CALL], 'hold': True})}],
                30, threading.Event(), tool_options=options, on_tools=tools)
        self.assertIs(caught.exception, error)
        self.assertTrue(seen)
        self.assertIsNone(self.served.runtime.tree)
        self.assertIsNotNone(tree.process.poll())
        response, value = self.chat('stream-tools:' + json.dumps({'calls': [CALL]}), tools=TOOLS)
        self.assertEqual(response.status, 200, value)
        self.assertEqual(value['choices'][0]['message']['tool_calls'], [CALL])
        self.assertNotEqual(self.served.runtime.root, root)

    def test_invalid_strict_prefix_is_sse_error_without_invalid_argument_delta(self):
        call = dict(CALL, function={'name': 'read_file', 'arguments': '{"wrong":1}'})
        response, data = self.chat('stream-tools:' + json.dumps({'calls': [call]}), tools=TOOLS, stream=True)
        self.assertEqual(response.status, 200)
        events = [json.loads(e[6:]) for e in data.decode().splitlines() if e.startswith('data: {')]
        self.assertTrue(any('error' in e for e in events))
        self.assertFalse(any(e.get('choices') and e['choices'][0]['finish_reason'] == 'tool_calls' for e in events))
        deltas = [d for e in events if e.get('choices') for d in e['choices'][0]['delta'].get('tool_calls', [])]
        self.assertNotIn('wrong', ''.join(d['function'].get('arguments', '') for d in deltas))
