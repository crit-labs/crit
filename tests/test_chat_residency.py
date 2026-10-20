"""Resident lifetimes use synthetic model bytes and a fake device allocator only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from test_run_records import RunFixture
from crit_api.engine import EngineError, ServedEngine
from crit_api.host import Host
from crit_api.models import DeviceAllocation, ModelContext, ModelPool
from crit.process.ownership import RuntimeOwnership, group_alive, process_identity
from crit_api.residency import ResidentRuntime
from crit.process.tree import ProcessTree
from crit.run.records import file_sha, read_json, verification_scope, write_json

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
                ('sampling', 'tools', 'tool_choice', 'parallel_tool_calls', 'messages', 'tool_grammars')}))
        if self.mode == 'crash':
            os._exit(17)
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
                    limit=request['limits']['max_output_tokens'], weights=bytes(self.weights).decode(),
                    format=(request.get('response_format') or {}).get('type'))
        return self.response(request, json.dumps(data))

    def response(self, request, assistant, output_tokens=4):
        return dict(schema_version=1, record_kind='chat_response',
                    **{k: request[k] for k in ('request_id', 'candidate_id', 'package_sha256')},
                    assistant=assistant, telemetry=dict(prompt_tokens=10, output_tokens=output_tokens,
                    prefill_tokens=10, prefill_ms=5, decode_tokens=output_tokens - 1, decode_ms=10, ttft_ms=6, missing={}))

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
        return self.response(request, ''.join(pieces), min(len(pieces), request['limits']['max_output_tokens']))

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
        elif self.temperature:
            capabilities.append('chat-sampling-temperature-v1')
        if self.strict_tools:
            capabilities.append('chat-tools-strict-v1')
        if self.tool_stream:
            capabilities.append('chat-tool-stream-v1')
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


class ResidentTests(ServeFixture, unittest.TestCase):
    def test_already_cancelled_generation_does_not_dispatch_or_switch_candidates(self):
        engine = self.engine()
        before = self.ask(engine)
        stop = threading.Event()
        stop.set()
        commands = set((engine.runtime.root / 'accepted').iterdir())
        self.assertEqual(engine.generate([{'role': 'user', 'content': 'hello'}], 128, stop,
                                         candidate_id=self.b), ('cancelled', None, None))
        self.assertEqual(set((engine.runtime.root / 'accepted').iterdir()), commands)
        self.assertEqual(engine.status()['engine_loads'], 1)
        after = self.ask(engine)
        self.assertEqual(after['calls'], before['calls'] + 1)
        self.assertEqual(after['device'], before['device'])

    def test_load_progress_reports_actual_reads_and_allocation_then_reuse(self):
        engine = self.engine(open=False)
        engine.default = self.a
        host = Host(self.root / 'progress-host', 1024, 1024)
        try:
            with patch.object(host, 'status', wraps=host.status) as status:
                host.attach(engine.spec(self.a))
                states = [call.args[0] for call in status.call_args_list]
                progress = [call.kwargs['progress'] for call in status.call_args_list if 'progress' in call.kwargs]
                self.assertEqual(progress[0]['completed_bytes'], 0)
                self.assertEqual(progress[-1]['completed_bytes'], self.model.stat().st_size)
                self.assertLess(states.index('loading_model'), states.index('loading_weights'))
                self.assertEqual(states[-1], 'ready')
                self.assertIsNone(host.context.progress)
                generation = host.pool.generation
                status.reset_mock()
                host.attach(engine.spec(self.b))
                self.assertEqual(host.pool.generation, generation)
                self.assertNotIn('loading_weights', [call.args[0] for call in status.call_args_list])
                self.assertFalse(any('progress' in call.kwargs for call in status.call_args_list))
                setup = [call.kwargs for call in status.call_args_list if call.args[0] == 'loading_engine']
                self.assertTrue(all(row['model_reused'] for row in setup))
                self.assertEqual(setup[-1]['model_device_bytes'], self.model.stat().st_size)
        finally:
            host.handle({'action': 'unload', 'payload': {}})

    def test_a_a_b_a_reuses_model_and_device_but_resets_private_state(self):
        engine = self.engine()
        a = self.ask(engine)
        again = self.ask(engine)
        b = self.ask(engine, candidate=self.b)
        last = self.ask(engine, candidate=self.a)
        for result in (again, b, last):
            self.assertEqual((result['pid'], result['generation'], result['device']),
                             (a['pid'], a['generation'], a['device']))
            self.assertEqual(result['private'], 1)
        self.assertEqual([r['calls'] for r in (a, again, b, last)], [1, 2, 1, 1])
        self.assertEqual(engine.status()['engine_loads'], 3)
        self.assertEqual(engine.status()['device_bytes'], self.model.stat().st_size)
        root = engine.runtime.root
        engine.close()
        self.assertIsNone(process_identity(a['pid']))
        self.assertFalse((root / 'models').exists())
        self.assertFalse(engine.ownership.path.exists())
        restarted = self.ask(self.engine())
        self.assertNotEqual(restarted['generation'], a['generation'])

    def test_requests_are_stateless_and_carry_their_own_limit(self):
        engine = self.engine()
        messages = [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'one'},
                    {'role': 'assistant', 'content': 'two'}, {'role': 'user', 'content': 'hello'}]
        status, assistant, measured = engine.generate(messages, 77, threading.Event())
        data = json.loads(assistant)
        self.assertEqual((data['history'], data['roles'], data['limit']), (4, ['system', 'user', 'assistant', 'user'], 77))
        self.assertEqual(measured['output_tokens'], 4)
        self.assertEqual(self.ask(engine)['history'], 1)
        with self.assertRaisesRegex(EngineError, 'max_tokens'):
            engine.generate(messages, 8192, threading.Event())

    def test_cooperative_stop_and_switch_keep_the_resident_model(self):
        engine = self.engine()
        original = self.ask(engine)
        stop = threading.Event()
        wait = self.background(engine, 'slow', stop)
        self.generating(engine)
        stop.set()
        self.assertEqual(wait(), ('cancelled', None))
        self.assertEqual(self.ask(engine, candidate=self.b)['generation'], original['generation'])
        self.assertEqual(self.ask(engine, candidate=self.a)['generation'], original['generation'])

    def test_noncooperative_stop_timeout_crash_and_reset_failure_retire_host(self):
        engine = self.engine()
        for mode, expected in [('hang', 'cancelled'), ('hang', 'timed_out'), ('crash', 'failed'), ('badreset', 'failed')]:
            with self.subTest(mode=mode, expected=expected):
                engine.timeout = .4 if expected == 'timed_out' else 120
                stop = threading.Event()
                wait = self.background(engine, mode, stop)
                if expected == 'cancelled':
                    self.generating(engine)
                    stop.set()
                status, error = wait()
                if expected == 'failed':
                    self.assertIsNotNone(error)
                else:
                    self.assertEqual(status, expected)
                self.assertEqual(engine.status()['state'], 'unloaded')
                self.assertFalse((engine.runtime.root / 'models').exists())
                self.assertTrue(engine.ownership.path.exists())  # The server keeps runtime ownership.
                engine.timeout = 120
                self.ask(engine, candidate=self.b)

    def test_detach_failure_retires_host_before_next_engine_can_run(self):
        engine = self.engine()
        first = self.ask(engine, 'badclose')
        with self.assertRaisesRegex(EngineError, 'detach failure'):
            self.ask(engine, candidate=self.b)
        self.assertIsNone(process_identity(first['pid']))
        self.assertNotEqual(self.ask(engine, candidate=self.b)['generation'], first['generation'])

    def test_serving_blocks_other_clients_until_closed(self):
        engine = self.engine()
        self.ask(engine)
        other = self.engine(open=False)
        with self.assertRaisesRegex(ValueError, 'Another CRIT runtime'):
            other.open()
        with self.assertRaisesRegex(ValueError, 'still alive'):
            RuntimeOwnership().recover()
        engine.close()
        self.ask(self.engine())

    def test_resident_budget_and_source_mutation_do_not_publish_reply(self):
        # --verify-model serves a private copy and leaves the source writable; a change is still detected.
        engine = self.engine(open=False, verify_model=True)
        engine.runtime.host_limit = 1
        with self.assertRaisesRegex(ValueError, 'RAM budget'):
            engine.open()
        self.assertFalse(engine.ownership.path.exists())
        engine.runtime.host_limit = 1024
        engine.open()
        self.ask(engine)
        stop = threading.Event()
        wait = self.background(engine, 'slow', stop)
        self.generating(engine)
        self.model.write_text('changed source')
        stop.set()
        status, error = wait()
        self.assertIsNotNone(error)
        self.assertEqual(engine.status()['state'], 'unloaded')

    def test_changed_sealed_package_fails_the_request(self):
        engine = self.engine()
        self.ask(engine)
        (self.directory / 'candidates' / self.a / 'package/engine.py').write_text('# changed sealed file')
        with self.assertRaisesRegex(EngineError, 'invalid response'):
            self.ask(engine)

    def test_unchanged_sealed_package_is_not_rehashed_between_requests(self):
        package = self.directory / 'candidates' / self.a / 'package/engine.py'
        opens = []
        original = Path.open
        def counted(path, *args, **kwargs):
            if path == package and args and args[0] == 'rb':
                opens.append(path)
            return original(path, *args, **kwargs)
        engine = self.engine(open=False)
        with patch.object(Path, 'open', counted):
            engine.open()
            first = len(opens)
            self.ask(engine)
            self.ask(engine, 'again')
        self.assertGreaterEqual(first, 1)
        self.assertEqual(len(opens), first)

    def test_chat_config_v1_rejects_an_engine_that_ignored_its_configuration(self):
        engine = self.engine()
        frozen = {'batch': 2, 'drafter': 'fp8'}
        self.reported = {'batch': 2}
        candidate, *_ = self.measured(label='Drops a key', configuration=frozen)
        self.store.seal(self.run_id, candidate['candidate_id'], 'completed')
        with self.assertRaisesRegex(EngineError, 'ignored: drafter'):
            self.ask(engine, candidate=candidate['candidate_id'])
        self.reported = {'batch': 2, 'drafter': 'fp8', 'derived': True}
        candidate, *_ = self.measured(label='Applies all keys', configuration=frozen)
        self.store.seal(self.run_id, candidate['candidate_id'], 'completed')
        self.ask(engine, candidate=candidate['candidate_id'])

    def test_unknown_unsealed_and_non_resident_candidates_are_refused(self):
        engine = self.engine()
        with self.assertRaisesRegex(EngineError, 'Unknown candidate'):
            engine.spec('candidate-9999')
        unsealed, *_ = self.measured(label='Unsealed')
        with self.assertRaisesRegex(ValueError, 'sealed'):
            engine.spec(unsealed['candidate_id'])
        self.assertEqual([c['candidate_id'] for c in engine.candidates()], [self.a, self.b])


class LoadBoundTests(ServeFixture, unittest.TestCase):
    def test_load_bound_grows_with_the_model_unless_set(self):
        engine = self.engine(open=False)
        spec = engine.spec(self.a)
        size = sum(row['bytes'] for row in spec['model_manifest'])
        self.assertAlmostEqual(engine.load_seconds(spec), 600 + size / 200e6)
        verifying = self.engine(open=False, verify_model=True)
        self.assertAlmostEqual(verifying.load_seconds(verifying.spec(self.a)), 600 + size / 50e6)
        self.assertEqual(self.engine(open=False, load_timeout=1234).load_seconds(spec), 1234)


class LifecycleTests(ServeFixture, unittest.TestCase):
    def test_cleanup_failure_blocks_requests_and_other_clients_until_closed(self):
        engine = self.engine()
        with patch.object(ProcessTree, 'close', side_effect=OSError('synthetic retirement failure')):
            with self.assertRaises(EngineError):
                self.ask(engine, 'crash')
            self.assertEqual(engine.status()['state'], 'blocked')
            with self.assertRaisesRegex(EngineError, 'cleanup is unresolved'):
                self.ask(engine)
            with self.assertRaisesRegex(ValueError, 'Another CRIT runtime'):
                self.engine(open=False).open()
        engine.close()
        self.assertFalse(engine.ownership.path.exists())
        self.ask(self.engine())

    def test_failed_launch_binding_retains_unbound_guardian_until_cleanup(self):
        engine = self.engine(open=False)
        with patch.object(engine.ownership, 'bind', side_effect=OSError('synthetic bind failure')), \
                patch.object(ProcessTree, 'close', side_effect=OSError('synthetic close failure')):
            with self.assertRaises(OSError):
                engine.open()
            self.assertIsNotNone(engine.ownership.failed_tree)
            self.assertIsNone(read_json(engine.ownership.path)['process'])
        engine.close()
        self.assertFalse(engine.ownership.path.exists())


class EffectiveConfigurationTests(unittest.TestCase):
    def test_every_frozen_key_must_be_applied_unchanged(self):
        from crit_api.host import effective_configuration_problem
        class Reports:
            def __init__(self, value):
                self.effective_configuration = lambda: value
        frozen = {'chunk': 1024, 'native': True}
        self.assertIsNone(effective_configuration_problem(Reports({'chunk': 1024, 'native': True, 'extra': 1}), frozen))
        self.assertIn('ignored: native', effective_configuration_problem(Reports({'chunk': 1024}), frozen))
        self.assertIn('changed: chunk', effective_configuration_problem(Reports({'chunk': 512, 'native': True}), frozen))
        self.assertIn('dictionary', effective_configuration_problem(Reports([1]), frozen))
        self.assertIn('requires', effective_configuration_problem(object(), frozen))




class SnapshotCleanupTests(unittest.TestCase):
    def test_retired_windows_mapping_retries_then_preserves_persistent_failure(self):
        from crit.process import ownership as chat_ownership
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_id = 'a' * 32
            models = root / 'runtimes' / runtime_id / 'models'
            models.mkdir(parents=True)
            (models / 'weights').write_bytes(b'fixture')
            denied = PermissionError('Synthetic Windows mapping still retiring')
            denied.winerror = 32
            remove = chat_ownership.shutil.rmtree
            calls = []
            def transient(path):
                calls.append(path)
                if len(calls) == 1:
                    raise denied
                remove(path)
            with patch.object(chat_ownership.shutil, 'rmtree', side_effect=transient):
                chat_ownership.clean_snapshots(root, runtime_id)
            self.assertEqual(calls, [models, models])
            self.assertFalse(models.exists())
            models.mkdir()
            (models / 'weights').write_bytes(b'fixture')
            with patch.object(chat_ownership.shutil, 'rmtree', side_effect=denied), \
                    patch.object(chat_ownership.time, 'monotonic', side_effect=[0, 1]):
                with self.assertRaises(PermissionError):
                    chat_ownership.clean_snapshots(root, runtime_id)
            self.assertEqual((models / 'weights').read_bytes(), b'fixture')


class ModelPoolTests(unittest.TestCase):
    def test_read_progress_spans_chunks_and_files_and_is_throttled(self):
        folder = self.root / 'parts'
        folder.mkdir()
        manifest = []
        for name, size in [('one', 17 * 1024**2), ('two', 3)]:
            path = folder / name
            path.write_bytes(b'x' * size)
            manifest.append({'path': name, 'bytes': size, 'sha256': file_sha(path)})
        total = sum(row['bytes'] for row in manifest)
        for native in (False, True):
            with self.subTest(native=native):
                pool = ModelPool(self.root / str(native), total, 0)
                updates = []
                try:
                    clock = {'side_effect': (i * .3 for i in range(100))} if native else {'return_value': 1}
                    with patch('crit_api.models.time.monotonic', **clock):
                        pool.load(folder, manifest, native_files=native, progress=updates.append, verify=True)
                    amounts = [p['completed_bytes'] for p in updates]
                    self.assertEqual(amounts, sorted(amounts))
                    self.assertEqual((amounts[0], amounts[-1]), (0, total))
                    if native:
                        self.assertIn(8 * 1024**2, amounts)  # Progress appears during a slow read.
                    else:
                        # Fast reads publish boundaries, not one disk write per chunk.
                        self.assertEqual(amounts, [0, manifest[0]['bytes'], manifest[0]['bytes'], total])
                        self.assertEqual([p['file_index'] for p in updates], [1, 1, 2, 2])
                    self.assertTrue(all(p['file_count'] == 2 and p['total_bytes'] == total for p in updates))
                    with patch('crit_api.models.time.monotonic', side_effect=AssertionError('No reread on reuse')):
                        self.assertFalse(pool.load(folder, manifest, native_files=native, progress=updates.append,
                                                   verify=True))
                finally:
                    pool.close()
        self.pool.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'weights'
        self.source.write_bytes(b'four')
        self.manifest = [{'path': 'weights', 'bytes': 4, 'sha256': file_sha(self.source)}]
        self.pool = ModelPool(self.root / 'pool', 4, 8)
        self.pool.load(self.source, self.manifest, verify=True)  # a hashed private copy
        self.key = dict(device='fake:0', representation='all-weights', dtype='bytes', quantization='none',
                        layout='flat', shape=[4], abi='v1')
        self.events = []

    def allocation(self):
        self.events.append('allocate')
        return DeviceAllocation(object(), 4, lambda: self.events.append('wait'), lambda: self.events.append('free'))

    def test_native_file_lease_avoids_copy_and_holds_source_until_close(self):
        pool = ModelPool(self.root / 'native', 4, 8)
        pool.load(self.source, self.manifest, native_files=True)
        context = ModelContext(pool, {}, [])
        self.assertEqual(context.model_file('weights'), self.source)
        self.assertEqual(bytes(context.host('weights')), b'four')
        self.assertFalse(any(p.is_file() for p in pool.root.rglob('*')))
        if os.name == 'nt':
            with self.assertRaises(PermissionError):
                self.source.write_bytes(b'new!')
        context.close(); pool.close()
        self.source.write_bytes(b'new!')
        self.pool.close()

    def test_default_load_checks_sizes_and_maps_in_place_without_reading(self):
        pool = ModelPool(self.root / 'fast', 4, 8)
        with patch('crit_api.models.hashlib') as hashing:
            self.assertTrue(pool.load(self.source, self.manifest))
        hashing.sha256.assert_not_called()  # no model file is read or hashed by default
        context = ModelContext(pool, {}, [])
        self.assertEqual(bytes(context.host('weights')), b'four')
        self.assertEqual(context.model_file('weights'), self.source)
        self.assertFalse(any(p.is_file() for p in pool.root.rglob('*')))  # no private copy
        with patch('crit_api.models.file_sha', side_effect=AssertionError('no re-hash per read')):
            context.host('weights'); pool.verify_sources()
        if os.name == 'nt':
            with self.assertRaises(PermissionError):
                self.source.write_bytes(b'new!')  # the held handle denies writes while serving
        context.close(); pool.close()
        self.pool.close()

    def test_default_load_refuses_wrong_sizes_and_accepts_same_size_bytes_that_only_verify_catches(self):
        wrong_size = [{**self.manifest[0], 'bytes': 5}]
        pool = ModelPool(self.root / 'sizes', 8, 8)
        with self.assertRaisesRegex(ValueError, 'frozen manifest'):
            pool.load(self.source, wrong_size)
        other_bytes = [{**self.manifest[0], 'sha256': hashlib.sha256(b'five').hexdigest()}]
        self.assertTrue(pool.load(self.source, other_bytes))  # the documented cost of skipping the hash
        pool.close()
        with self.assertRaisesRegex(ValueError, 'frozen manifest'):
            pool.load(self.source, other_bytes, verify=True)
        self.pool.close()

    def test_backend_measured_allocation_is_charged_and_reused(self):
        first = self.pool.acquire_device(self.key, None, self.allocation)
        first.close()
        again = self.pool.acquire_device(self.key, None, self.allocation)
        self.assertEqual(self.pool.device_bytes, 4)
        self.assertEqual(self.events.count('allocate'), 1)
        again.close(); self.pool.close()

    def test_invalid_allocation_metadata_retains_owner_until_process_retirement(self):
        for declared in (None, 4):
            for actual in (None, -1, 0, True, 1.5, float('nan'), 2 ** 61):
                with self.subTest(declared=declared, actual=actual):
                    pool = ModelPool(self.root / 'invalid', 4, 8)
                    pool.load(self.source, self.manifest)
                    allocation = DeviceAllocation(object(), actual, lambda: None,
                                                  lambda: self.events.append('free'))
                    try:
                        with self.assertRaisesRegex(ValueError, 'actual device allocation bytes'):
                            pool.acquire_device(self.key, declared, lambda: allocation)
                        self.assertTrue(pool.blocked)
                        self.assertIs(next(iter(pool.device.values()))['allocation'], allocation)
                        with self.assertRaisesRegex(ValueError, 'retirement'):
                            pool.close()
                        with self.assertRaisesRegex(ValueError, 'healthy'):
                            pool.acquire_device(self.key, 4, self.allocation)
                        self.assertEqual(self.events, [])
                    finally:
                        # Fake owners have no external resources; release fixture mappings.
                        pool.device.clear(); pool.blocked = False; pool.close()
        self.pool.close()

    def test_mismatched_extent_charges_actual_bytes_and_foreign_owner_is_retained(self):
        for allocation, error, charged in (
                (DeviceAllocation(object(), 6, lambda: None, lambda: None), 'contract mismatch', 6),
                (object(), 'owned allocation', 0)):
            pool = ModelPool(self.root / 'mismatch', 4, 8)
            pool.load(self.source, self.manifest)
            try:
                with self.assertRaisesRegex(ValueError, error):
                    pool.acquire_device(self.key, 4, lambda: allocation)
                self.assertTrue(pool.blocked)
                self.assertIs(next(iter(pool.device.values()))['allocation'], allocation)
                self.assertEqual(pool.device_bytes, charged)
            finally:
                pool.device.clear(); pool.device_bytes = 0; pool.blocked = False; pool.close()
        self.pool.close()

    def test_host_snapshot_is_readonly_bounded_content_verified_and_source_checked(self):
        context = ModelContext(self.pool, {}, [])
        view = context.host('weights')
        self.assertEqual(bytes(view), b'four')
        with self.assertRaises(TypeError):
            view[0] = 1
        self.assertFalse(self.pool.load(self.source, self.manifest))
        with self.assertRaises(BufferError):
            self.pool.close()
        context.close()
        stamp = self.source.stat().st_mtime_ns
        self.source.write_bytes(b'new!')
        # Windows file times follow a coarse clock tick, so a rewrite right after setUp can keep the same time.
        os.utime(self.source, ns=(stamp + 10 ** 9, stamp + 10 ** 9))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.pool.verify_sources()
        self.pool.close()
        with self.assertRaisesRegex(ValueError, 'frozen manifest'):
            self.pool.load(self.source, self.manifest, verify=True)
        too_small = ModelPool(self.root / 'small', 3, 0)
        with self.assertRaisesRegex(ValueError, 'RAM budget'):
            too_small.load(self.source, self.manifest)

    def test_device_compatible_reuse_incompatible_capacity_and_generation(self):
        a = self.pool.acquire_device(self.key, 4, self.allocation)
        b = self.pool.acquire_device(self.key, 4, self.allocation)
        self.assertIs(a.buffer, b.buffer)
        self.assertEqual(self.events, ['allocate'])
        with self.assertRaisesRegex(ValueError, 'leases remain'):
            self.pool.close()
        a.close(); b.close()
        c = self.pool.acquire_device(self.key | {'layout': 'other'}, 4, self.allocation)
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            self.pool.acquire_device(self.key | {'device': 'fake:1'}, 4, self.allocation)
        with self.assertRaisesRegex(ValueError, 'physical extent'):
            self.pool.acquire_device(self.key, 5, self.allocation)
        c.close(); self.pool.close()
        self.assertEqual(self.events.count('allocate'), 2)
        self.assertEqual(self.events.count('free'), 2)
        with self.assertRaisesRegex(ValueError, 'closed, stale'):
            _ = a.buffer

    def test_device_inputs_join_the_key_by_content(self):
        overlay = self.root / 'drafter.bin'
        overlay.write_bytes(b'encoding one')
        a = self.pool.acquire_device(self.key, 4, self.allocation, inputs=[overlay])
        a.close()
        b = self.pool.acquire_device(self.key, 4, self.allocation, inputs=[overlay])  # same bytes: reused
        self.assertEqual(self.events.count('allocate'), 1)
        b.close()
        overlay.write_bytes(b'encoding two')  # same name and size, different bytes: a new allocation
        c = self.pool.acquire_device(self.key, 4, self.allocation, inputs=[overlay])
        self.assertEqual(self.events.count('allocate'), 2)
        c.close()
        with self.assertRaisesRegex(ValueError, 'not a readable file'):
            self.pool.acquire_device(self.key, 4, self.allocation, inputs=[self.root / 'missing.bin'])
        self.pool.close()

    def test_failed_fence_or_free_quarantines_owners_without_double_free(self):
        def fail():
            self.events.append('failure')
            raise OSError('synthetic fence/free failure')
        lease = self.pool.acquire_device(self.key, 4, lambda: DeviceAllocation(object(), 4, fail, fail))
        with self.assertRaises(OSError):
            lease.close()
        self.assertTrue(self.pool.blocked)
        self.assertEqual(self.pool.device_bytes, 4)
        with self.assertRaisesRegex(ValueError, 'retirement'):
            self.pool.close()
        self.assertEqual(self.events, ['failure'])
        # A second independent pool covers an ambiguous free after a valid fence.
        pool = ModelPool(self.root / 'pool2', 4, 4)
        pool.load(self.source, self.manifest)
        value = pool.acquire_device(self.key, 4, lambda: DeviceAllocation(object(), 4, lambda: None, fail))
        value.close()
        with self.assertRaises(OSError):
            pool.close()
        with self.assertRaisesRegex(ValueError, 'retirement'):
            pool.close()
        self.assertEqual(self.events, ['failure', 'failure'])
        # Tests own fake objects, so release mappings for portable fixture cleanup.
        for item in (self.pool, pool):
            item.blocked = False; item.device.clear(); item.close()

    def test_hash_cache_is_scoped_and_rechecks_changed_files(self):
        original = Path.open
        count = []
        def opened(path, *args, **kwargs):
            if path == self.source and args and args[0] == 'rb': count.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', opened):
            with verification_scope():
                first = file_sha(self.source)
                self.assertEqual(file_sha(self.source), first)
                self.source.write_bytes(b'next')
                self.assertNotEqual(file_sha(self.source), first)
            with verification_scope():
                file_sha(self.source)
        self.assertEqual(len(count), 4 if os.name == 'nt' else 3)
        self.pool.close()

    def test_a_verified_copy_keeps_serving_its_bytes_without_rehashing_the_source(self):
        frozen = self.pool.fingerprint(self.source)
        context = ModelContext(self.pool, {}, [])
        try:
            with patch.object(self.pool, 'fingerprint', return_value=frozen), \
                    patch('crit_api.models.file_sha', side_effect=AssertionError('no re-hash per read')):
                self.source.write_bytes(b'next')  # metadata that looks unchanged, as Windows can report
                self.pool.verify_sources()
                self.assertEqual(bytes(context.host('weights')), b'four')  # the private copy, verified at load
        finally:
            context.close()
            self.pool.close()

    @unittest.skipUnless(os.name == 'nt', 'Windows timestamp cache behavior')
    def test_windows_hash_rechecks_bytes_with_unchanged_metadata(self):
        original_stat = Path.stat
        frozen = self.source.stat()
        def unchanged(path, *args, **kwargs):
            return frozen if path == self.source else original_stat(path, *args, **kwargs)
        try:
            with patch.object(Path, 'stat', unchanged), verification_scope():
                first = file_sha(self.source)
                self.source.write_bytes(b'next')
                self.assertNotEqual(file_sha(self.source), first)
        finally:
            self.pool.close()


class RuntimeStatusTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import Mock
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_json(self.root / 'status.json', {'state': 'ready', 'host_bytes': 8, 'device_bytes': 4})
        self.runtime = ResidentRuntime(RuntimeOwnership(self.root), 16, 16)
        self.runtime.root = self.root
        self.runtime.tree = Mock()
        self.runtime.tree.process.poll.return_value = None

    def denied(self, code=32):
        error = PermissionError(13, 'status sharing denied')
        error.winerror = code
        return error

    def test_transient_windows_sharing_denial_returns_only_new_valid_status(self):
        expected = read_json(self.root / 'status.json')
        with patch('crit_api.residency.sys.platform', 'win32'), \
             patch('crit_api.residency.read_json', side_effect=[self.denied(), self.denied(5), self.denied(None), expected]) as reader, \
             patch('crit_api.residency.time.sleep') as sleep:
            self.assertEqual(self.runtime.status(), expected)
            self.assertEqual(reader.call_count, 4)
            self.assertEqual(sleep.call_count, 3)

    def test_persistent_windows_sharing_denial_is_bounded_and_propagated(self):
        denied = self.denied()
        with patch('crit_api.residency.sys.platform', 'win32'), \
             patch('crit_api.residency.read_json', side_effect=denied) as reader, \
             patch('crit_api.residency.time.monotonic', side_effect=[0, .3]), \
             patch('crit_api.residency.time.sleep') as sleep:
            with self.assertRaises(PermissionError) as raised:
                self.runtime.status()
            self.assertIs(raised.exception, denied)
            self.assertEqual(reader.call_count, 1)
            sleep.assert_not_called()

    def test_other_platform_permissions_and_invalid_status_never_retry(self):
        for platform, error in [('linux', self.denied()), ('win32', self.denied(13)),
                                ('win32', ValueError('invalid status'))]:
            with self.subTest(platform=platform, error=error), \
                 patch('crit_api.residency.sys.platform', platform), \
                 patch('crit_api.residency.read_json', side_effect=error) as reader, \
                 patch('crit_api.residency.time.sleep') as sleep:
                with self.assertRaises(type(error)):
                    self.runtime.status()
                self.assertEqual(reader.call_count, 1)
                sleep.assert_not_called()


class OwnershipReadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.owner = RuntimeOwnership(self.temp.name)
        self.owner.token = 'original-owner'

    def test_transient_access_failure_retries_without_using_cached_state(self):
        current = {'token': self.owner.token, 'state': 'blocked', 'error': 'retained work'}
        with patch('crit.process.ownership.read_json', side_effect=[PermissionError('sharing'), current]) as read, \
                patch('crit.process.ownership.time.monotonic', return_value=0), \
                patch('crit.process.ownership.time.sleep') as sleep:
            self.assertIs(self.owner.read_owned(), current)
        self.assertEqual(read.call_count, 2)
        sleep.assert_called_once_with(.005)

    def test_persistent_access_failure_is_bounded_and_keeps_ownership(self):
        failure = PermissionError('persistent access failure')
        with patch('crit.process.ownership.read_json', side_effect=failure) as read, \
                patch('crit.process.ownership.time.monotonic', side_effect=[0, .1, .25]), \
                patch('crit.process.ownership.time.sleep') as sleep:
            with self.assertRaises(PermissionError) as caught:
                self.owner.read_owned()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.owner.token, 'original-owner')
        self.assertEqual(read.call_count, 2)
        sleep.assert_called_once_with(.005)

    def test_changed_identity_after_retry_is_still_rejected(self):
        with patch('crit.process.ownership.read_json', side_effect=[PermissionError('sharing'), {'token': 'replacement'}]), \
                patch('crit.process.ownership.time.monotonic', return_value=0), \
                patch('crit.process.ownership.time.sleep'):
            with self.assertRaisesRegex(ValueError, 'ownership changed'):
                self.owner.read_owned()
        self.assertEqual(self.owner.token, 'original-owner')

    def test_missing_or_malformed_record_is_not_retried_or_treated_as_idle(self):
        for error in (FileNotFoundError('missing'), ValueError('malformed record')):
            with self.subTest(error=type(error).__name__), \
                    patch('crit.process.ownership.read_json', side_effect=error) as read, \
                    patch('crit.process.ownership.time.sleep') as sleep:
                with self.assertRaises(type(error)):
                    self.owner.read_owned()
                read.assert_called_once_with(self.owner.path)
                sleep.assert_not_called()


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process identity and group retirement')
class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.owner = RuntimeOwnership(self.root / 'owner')

    def test_competing_processes_and_live_owner_recovery_fail_closed(self):
        self.owner.acquire()
        self.addCleanup(self.owner.release)
        with self.assertRaises(ValueError):
            RuntimeOwnership(self.owner.directory).acquire()
        with self.assertRaisesRegex(ValueError, 'still alive'):
            RuntimeOwnership(self.owner.directory).recover()
        code = "import sys; sys.path.insert(0, sys.argv[1]); from crit.process.ownership import RuntimeOwnership; RuntimeOwnership(sys.argv[2]).acquire()"
        result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parents[1] / 'cli'), str(self.owner.directory)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'Another CRIT runtime', result.stderr)

    def test_parent_death_guardian_retires_engine_and_explicit_recovery_releases(self):
        source = Path(__file__).resolve().parents[1] / 'cli'
        code = r'''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from crit.process.ownership import RuntimeOwnership
from crit.process.tree import guarded_tree
root = Path(sys.argv[2])
owner = RuntimeOwnership(root / 'owner'); owner.acquire()
log = (root / 'engine.log').open('wb')
command = [sys.executable, '-B', '-c', "import os,time; from pathlib import Path; Path('pid').write_text(str(os.getpid())); time.sleep(60)"]
tree = guarded_tree(command, root, log, owner, root)
time.sleep(60)
'''
        client = subprocess.Popen([sys.executable, '-B', '-c', code, str(source), str(self.root)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            end = time.monotonic() + 5
            while not (self.root / 'pid').exists() and time.monotonic() < end:
                time.sleep(.02)
            self.assertTrue((self.root / 'pid').exists())
            pid = int((self.root / 'pid').read_text())
            value = read_json(self.owner.path)
            client.kill(); client.wait(timeout=5)
            end = time.monotonic() + 5
            while process_identity(pid) is not None and time.monotonic() < end:
                time.sleep(.02)
            self.assertIsNone(process_identity(pid))
            self.assertFalse(group_alive(value['process']['pid']))
            with self.assertRaises(ValueError):
                self.owner.acquire()
            self.assertEqual(self.owner.recover()['status'], 'recovered')
            self.assertEqual(len(list(self.owner.directory.glob('recovery-*.json'))), 1)
            self.owner.acquire(); self.owner.release()
        finally:
            if client.poll() is None: client.kill(); client.wait(timeout=5)
            client.stderr.close()
            if self.owner.path.exists(): self.owner.recover()

    def test_unbound_dead_owner_recovers_without_starting_engine(self):
        self.owner.acquire()
        self.owner.update(owner_pid=99999999, owner_identity='dead-fixture')
        self.assertEqual(RuntimeOwnership(self.owner.directory).recover()['status'], 'recovered')
        self.owner.token = None
