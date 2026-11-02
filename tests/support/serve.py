"""A run with two sealed resident candidates, A (selected) and B, served by a ServedEngine (ServeFixture), and
the synthetic resident adapter their packages launch (ADAPTER)."""
import json
import threading
import time
from unittest.mock import patch

from .runs import RunFixture
from crit_api.engine import EngineError, ServedEngine

ADAPTER = r'''
import json
import os
import time
import uuid
from crit_api.models import DeviceAllocation

class Engine:
    def __init__(self, context):
        self.context = context
        self.weights = context.host('model.fixture')
        self.private = []
        self.calls = 0
        self.mode = ''
        key = dict(device='fake:0', representation='synthetic-weights', dtype='bytes',
                   quantization='none', layout='flat', shape=[len(self.weights)], abi='fixture-v1')
        def allocate():
            return DeviceAllocation({'id': uuid.uuid4().hex}, len(self.weights), lambda: None, lambda: None)
        self.lease = context.device(key, len(self.weights), allocate)
        print('loaded synthetic engine', flush=True)

    def generate(self, request, cancellation):
        self.mode = request['messages'][-1]['content']
        self.calls += 1
        if self.mode.startswith('tool:'):
            response = self.response(request, '')
            response['tool_calls'] = json.loads(self.mode[5:])
            return response
        if self.mode == 'echo-request':
            return self.response(request, json.dumps({k: request.get(k) for k in
                ('sampling', 'tools', 'tool_choice', 'parallel_tool_calls', 'messages', 'tool_grammars',
                 'template_kwargs', 'limits', 'continue_final_message')}))
        if self.mode == 'crash':
            os._exit(17)
        if self.mode.startswith('overflow:'):
            # A prompt of that many tokens; fitted (chat-fit-output-v1), the reply fills the rest of the context.
            from crit_core.serving import ContextExceeded
            prompt, limits = int(self.mode[len('overflow:'):]), request['limits']
            if limits.get('fit_output') and prompt < limits['kv_tokens']:
                return self.response(request, 'full', min(limits['max_output_tokens'], limits['kv_tokens'] - prompt),
                                     prompt_tokens=prompt)
            raise ContextExceeded(prompt, limits['max_output_tokens'], limits['kv_tokens'])
        if self.mode.startswith('reply:'):
            return self.response(request, self.mode[len('reply:'):])
        if self.mode == 'hang':
            time.sleep(30)
        if self.mode == 'slow':
            for _ in range(500):
                cancellation.check()
                time.sleep(.01)
        self.private.append(request['request_id'])
        data = dict(pid=os.getpid(), generation=self.context.pool.generation,
                    device=self.lease.buffer['id'], calls=self.calls, private=len(self.private),
                    history=len(request['messages']), roles=[m['role'] for m in request['messages']],
                    limit=request['limits']['max_output_tokens'], fit_output=request['limits'].get('fit_output'),
                    weights=bytes(self.weights).decode(), format=(request.get('response_format') or {}).get('type'))
        return self.response(request, json.dumps(data))

    def response(self, request, assistant, output_tokens=4, rounds=None, prompt_tokens=10):
        limits = request['limits']
        return dict(schema_version=1, record_kind='chat_response',
                    **{k: request[k] for k in ('request_id', 'candidate_id', 'package_sha256')},
                    assistant=assistant, telemetry=dict(prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                    prefill_tokens=prompt_tokens, prefill_ms=5, decode_tokens=output_tokens - 1, decode_ms=10, ttft_ms=6,
                    missing={}, **({'decode_rounds': rounds} if rounds else {})),
                    **({'max_output_tokens': min(limits['max_output_tokens'], limits['kv_tokens'] - prompt_tokens)}
                       if limits.get('fit_output') else {}))

    def generate_stream(self, request, cancellation, emit):
        message = request['messages'][-1]['content']
        if not message.startswith('words:'):
            response = self.generate(request, cancellation)
            if response['assistant']:
                emit(response['assistant'])
            return response
        self.mode = message
        pieces = []
        for word in message[len('words:'):].split(' '):
            cancellation.check()
            piece = word + ' '
            emit(piece)
            pieces.append(piece)
            time.sleep(.05)
        # Words decode as one verify round of up to three tokens after the first (a speculative engine's telemetry).
        output = min(len(pieces), request['limits']['max_output_tokens'])
        return self.response(request, ''.join(pieces), output, -(-(output - 1) // 3) if output > 1 else None)

    def reset(self):
        if self.mode == 'badreset':
            raise RuntimeError('synthetic drain failure')
        self.private.clear()

    def close(self):
        if self.mode == 'badclose':
            raise RuntimeError('synthetic detach failure')
        self.private.clear()

def create_engine(context):
    return Engine(context)
'''


class ServeFixture(RunFixture):
    """A run with two sealed resident candidates, A (selected) and B, served by a ServedEngine."""
    reported = None  # chat-config-v1 fixture: configuration the adapter reports as applied
    stream = False   # declare chat-stream-v1
    constraints = False  # declare chat-constraints-v1
    request_options = False  # sampling and tool fixtures
    strict_tools = False
    tool_stream = False
    tool_choice = True  # with request_options, declare chat-tool-choice-v1
    fit_output = False  # declare chat-fit-output-v1
    temperature = True  # declare chat-sampling-temperature-v1, which crit run package requires
    adapter_suffix = ''
    extra_capabilities = ()

    def package(self, candidate):
        package = self.root / ('resident-package-' + candidate['candidate_id'])
        package.mkdir(exist_ok=True)
        capabilities = ['prefill', 'decode', 'chat-json-v1', 'chat-resident-v1']
        capabilities.extend(self.extra_capabilities)
        adapter = ADAPTER
        if self.stream:
            capabilities.append('chat-stream-v1')
        if self.constraints:
            capabilities.append('chat-constraints-v1')
        if self.request_options:
            from crit_api.options import SAMPLING, sampling_capability
            capabilities += ['chat-tools-v1', *map(sampling_capability, sorted(SAMPLING))]
            capabilities += ['chat-tool-choice-v1'] if self.tool_choice else []
        elif self.temperature:
            capabilities.append('chat-sampling-temperature-v1')
        if self.strict_tools:
            capabilities.append('chat-tools-strict-v1')
        if self.tool_stream:
            capabilities.append('chat-tool-stream-v1')
        if self.fit_output:
            capabilities.append('chat-fit-output-v1')
        if self.reported is not None:
            capabilities.append('chat-config-v1')
            adapter += '\nEngine.effective_configuration = lambda self: %r\n' % (self.reported,)
        (package / 'engine.py').write_text(adapter + self.adapter_suffix)
        return self.store.package(self.run_id, candidate['candidate_id'], package,
            {'build': {'toolchain': {'fixture': 'host-python'}, 'dependencies': [], 'command': ['fixture']},
             'launch': ['engine.py'], 'capabilities': capabilities})

    def setUp(self):
        super().setUp()
        environment = patch.dict('os.environ', {'CRIT_RUNTIME_DIRECTORY': str(self.root / 'runtime-owner')})
        environment.start()
        self.addCleanup(environment.stop)
        candidate, *_ = self.measured()
        self.a = candidate['candidate_id']
        self.store.seal(self.run_id, self.a, 'completed')
        self.store.select(self.run_id, self.a)
        candidate, *_ = self.measured(label='Engine B', configuration={'batch': 2})
        self.b = candidate['candidate_id']
        self.store.seal(self.run_id, self.b, 'completed')

    def engine(self, open=True, **options):
        engine = ServedEngine(self.store.root, self.run_id, host_limit=1024, device_limit=1024, **options)
        self.addCleanup(engine.close)
        if open:
            engine.open()
        return engine

    def ask(self, engine, message='hello', candidate=None, limit=128):
        status, assistant, measured = engine.generate([{'role': 'user', 'content': message}], limit,
                                                      threading.Event(), candidate_id=candidate)
        self.assertEqual(status, 'completed')
        return json.loads(assistant)

    def generating(self, engine):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            if engine.status()['state'] == 'generating':
                return
            time.sleep(.02)
        self.fail('Synthetic resident generation did not start')

    def background(self, engine, message, stop, candidate=None):
        """Generate in a thread; returns a function that waits and gives (status, error)."""
        outcome = {}
        def run():
            try:
                outcome['status'] = engine.generate([{'role': 'user', 'content': message}], 128, stop,
                                                    candidate_id=candidate)[0]
            except EngineError as exc:
                outcome['error'] = exc
        thread = threading.Thread(target=run)
        thread.start()
        def wait():
            thread.join(10)
            self.assertFalse(thread.is_alive())
            return outcome.get('status'), outcome.get('error')
        return wait
