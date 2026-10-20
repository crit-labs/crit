"""Real TCP disconnects against a synthetic resident adapter; no model or device."""
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

from test_chat_residency import ServeFixture
from crit_api.server import Api, Server


GATED_ADAPTER = r'''
from pathlib import Path

_generate = Engine.generate

def gated_generate(self, request, cancellation):
    message = request['messages'][-1]['content']
    if not message.startswith('gate:'):
        return _generate(self, request, cancellation)
    gate = Path(message[5:])
    self.mode = 'gate'
    self.private.append(request['request_id'])
    gate.with_suffix('.started').touch()
    try:
        while not gate.with_suffix('.release').exists():
            cancellation.check()
            time.sleep(.01)
        return self.response(request, 'released')
    finally:
        gate.with_suffix('.settled').touch()

Engine.generate = gated_generate
'''


class DisconnectTests(ServeFixture, unittest.TestCase):
    stream = True
    adapter_suffix = GATED_ADAPTER

    def setUp(self):
        super().setUp()
        self.served = self.engine(timeout=30)
        self.server = Server(('127.0.0.1', 0), Api(self.served))
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server_thread.join, 5)
        self.addCleanup(self.server.shutdown)
        self.gate = self.root / 'generation'
        # Release even on assertion failure so cleanup never waits for the request timeout.
        self.addCleanup(self.gate.with_suffix('.release').touch)

    def until(self, predicate, message, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail(message)

    def connect(self, stream=False, content=None):
        client = socket.create_connection(self.server.server_address, timeout=5)
        self.addCleanup(client.close)
        body = json.dumps({'messages': [{'role': 'user', 'content': content or 'gate:' + str(self.gate)}],
                           'stream': stream, 'max_tokens': 128}).encode()
        client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n'
                       b'Content-Type: application/json\r\nContent-Length: ' +
                       str(len(body)).encode() + b'\r\n\r\n' + body)
        if content is None:
            self.until(self.gate.with_suffix('.started').exists, 'Generation did not reach the gate')
        return client

    def reset(self, client):
        # Force RST only for the reset regressions; ordinary-client tests use close().
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                          struct.pack('HH' if os.name == 'nt' else 'ii', 1, 0))
        client.close()

    def http_request(self, stream=False, content=None):
        """Use ordinary HTTP client APIs, without changing TCP close behavior."""
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        self.addCleanup(connection.close)
        connection.request('POST', '/v1/chat/completions', json.dumps({
            'messages': [{'role': 'user', 'content': content or 'gate:' + str(self.gate)}],
            'stream': stream, 'max_tokens': 128,
        }), {'Content-Type': 'application/json'})
        response = None
        if stream:
            response = connection.getresponse()
            self.addCleanup(response.close)
            self.assertEqual(response.status, 200)
            # The role event is available even while the adapter has emitted no text.
            self.assertIn(b'"role": "assistant"', response.readline())
        if content is None:
            self.until(self.gate.with_suffix('.started').exists, 'Generation did not reach the gate')
        return connection, response

    def assert_cancelled_and_reusable(self, before):
        self.until(self.gate.with_suffix('.settled').exists,
                   'Disconnected generation kept running without cancellation')
        self.until(lambda: self.served.status()['state'] == 'ready',
                   'Cancelled generation did not reset the resident engine')
        after = self.ask(self.served)
        self.assertEqual((after['pid'], after['device'], after['generation']),
                         (before['pid'], before['device'], before['generation']))
        self.assertEqual(after['private'], 1)
        self.assertEqual(self.served.status()['engine_loads'], 1)
        self.assertEqual(list((self.served.runtime.root / 'requests').iterdir()), [])
        self.until(lambda: not any(t.name == 'crit-client-disconnect' for t in threading.enumerate()),
                   'Disconnect watcher outlived the request')

    def check_reset(self, stream):
        before = self.ask(self.served)
        with patch.object(self.server, 'handle_error') as server_error:
            client = self.connect(stream)
            self.reset(client)
            self.assert_cancelled_and_reusable(before)
            server_error.assert_not_called()

    def check_http_client_close(self, stream):
        before = self.ask(self.served)
        with patch.object(self.server, 'handle_error') as server_error:
            connection, response = self.http_request(stream)
            if response is not None:
                response.close()
            connection.close()
            self.assert_cancelled_and_reusable(before)
            server_error.assert_not_called()

    def test_http_client_close_cancels_nonstreaming_generation(self):
        self.check_http_client_close(False)

    def test_http_client_close_cancels_silent_streaming_generation(self):
        self.check_http_client_close(True)

    def test_reset_cancels_nonstreaming_generation_and_keeps_weights(self):
        self.check_reset(False)

    def test_reset_cancels_silent_streaming_generation_and_keeps_weights(self):
        self.check_reset(True)

    def check_curl_timeout(self, stream):
        curl = shutil.which('curl')
        if curl is None:
            self.skipTest('curl is required for the external HTTP client regression')
        before = self.ask(self.served)
        body = self.root / 'curl-request.json'
        body.write_text(json.dumps({
            'messages': [{'role': 'user', 'content': 'gate:' + str(self.gate)}],
            'stream': stream, 'max_tokens': 128,
        }), encoding='utf-8')
        output = self.root / 'curl-response.txt'
        address, port = self.server.server_address
        with patch.object(self.server, 'handle_error') as server_error:
            result = subprocess.run([
                curl, '--disable', '--silent', '--show-error', '--no-buffer',
                '--noproxy', '*', '--max-time', '2',
                '--header', 'Content-Type: application/json', '--data-binary', '@' + str(body),
                '--output', str(output), f'http://{address}:{port}/v1/chat/completions',
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 28, result.stderr)
            self.assertTrue(self.gate.with_suffix('.started').exists(),
                            'Client timed out before the adapter started')
            if stream:
                self.assertIn(b'"role": "assistant"', output.read_bytes())
            # The adapter stays gated indefinitely and the server timeout is 30 s.
            # Only client cancellation can settle this turn within the 5 s check.
            self.assert_cancelled_and_reusable(before)
            server_error.assert_not_called()

    def test_curl_timeout_cancels_nonstreaming_generation(self):
        self.check_curl_timeout(False)

    def test_curl_timeout_cancels_silent_streaming_generation(self):
        self.check_curl_timeout(True)

    def check_cancel_while_waiting(self, reset):
        before = self.ask(self.served)
        entered, finished = threading.Event(), threading.Event()
        generate = self.served.generate

        def observed(messages, *args, **kwargs):
            queued = messages[-1]['content'] == 'reply:abandoned'
            if queued:
                entered.set()
            try:
                return generate(messages, *args, **kwargs)
            finally:
                if queued:
                    finished.set()

        with patch.object(self.served, 'generate', side_effect=observed):
            active = self.connect()
            if reset:
                queued = self.connect(content='reply:abandoned')
            else:
                queued, _ = self.http_request(content='reply:abandoned')
            self.assertTrue(entered.wait(5), 'Second request did not enter the engine queue')
            if reset:
                self.reset(queued)
            else:
                queued.close()
            self.assertTrue(finished.wait(2), 'Cancelled request stayed blocked behind the active turn')
            self.assertFalse(self.gate.with_suffix('.settled').exists())
            self.assertEqual(self.served.status()['state'], 'generating')
            self.gate.with_suffix('.release').touch()
            response = http.client.HTTPResponse(active)
            response.begin()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())['choices'][0]['message']['content'], 'released')
        after = self.ask(self.served)
        # The gated turn does not increment the adapter's ordinary generation counter.
        self.assertEqual(after['calls'], before['calls'] + 1)
        self.assertEqual((after['pid'], after['device']), (before['pid'], before['device']))

    def test_reset_while_waiting_skips_generation_without_cancelling_the_active_turn(self):
        self.check_cancel_while_waiting(reset=True)

    def test_http_client_close_while_waiting_skips_generation_without_cancelling_the_active_turn(self):
        self.check_cancel_while_waiting(reset=False)

    def check_uncooperative_cancel(self, reset):
        before = self.ask(self.served)
        if reset:
            client = self.connect(content='hang')
        else:
            client, _ = self.http_request(content='hang')
        self.generating(self.served)
        retired_root = self.served.runtime.root
        if reset:
            self.reset(client)
        else:
            client.close()
        self.until(lambda: self.served.status()['state'] == 'unloaded',
                   'Uncooperative adapter was not retired after client cancellation')
        # Status can report process exit before file cleanup finishes. The next
        # turn must wait for settlement and start with a new runtime.
        after = self.ask(self.served)
        self.assertFalse((retired_root / 'models').exists())
        self.assertNotEqual(after['generation'], before['generation'])
        self.assertEqual(after['private'], 1)

    def test_reset_retires_an_adapter_that_ignores_cancellation_before_next_turn(self):
        self.check_uncooperative_cancel(reset=True)

    def test_http_client_close_retires_an_adapter_that_ignores_cancellation_before_next_turn(self):
        self.check_uncooperative_cancel(reset=False)

    def test_write_half_close_cancels_the_turn_under_the_serving_contract(self):
        before = self.ask(self.served)
        client = self.connect()
        client.shutdown(socket.SHUT_WR)
        self.assert_cancelled_and_reusable(before)

    def test_pipelined_request_bytes_are_left_for_the_http_handler(self):
        client = self.connect()
        client.sendall(b'GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
        time.sleep(.2)
        self.assertFalse(self.gate.with_suffix('.settled').exists())
        self.gate.with_suffix('.release').touch()
        # Read until the pipelined GET closes the connection, without an HTTPResponse
        # buffer hiding the next response from the test.
        wire = bytearray()
        while chunk := client.recv(65536):
            wire.extend(chunk)
        self.assertEqual(wire.count(b'HTTP/1.1 200 OK\r\n'), 2, wire)
        self.assertIn(b'"content": "released"', wire)
        self.assertIn(b'"status": "ok"', wire)
