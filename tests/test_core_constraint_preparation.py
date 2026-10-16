"""Real compilation workers, single flight, bounded leases and independent users."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core/python"))

from crit_core.constraints import compiler, preparation  # noqa: E402


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.c, self.m = compiler, preparation
        self.args = dict(tokenizer_identity='tokenizer.v1', pieces=(b'a', b'b', b''), eos=2)
        self.services, self.gates = [], []

    def service(self, compile=None, version=None, **kw):
        compiler = SimpleNamespace(make_compile_request=self.c.make_compile_request,
                                   compile_language=compile or self.c.compile_language,
                                   CompiledConstraint=self.c.CompiledConstraint,
                                   VERSION=version or self.c.VERSION)
        service = self.m.PreparationService(compiler=compiler, **kw)
        self.services.append(service)
        return service

    def gated(self):
        gate, entered, calls = Event(), Event(), []
        self.gates.append(gate)
        def compile(kind, source):
            calls.append((kind, source))
            entered.set()
            if not gate.wait(3):
                raise RuntimeError('test producer was not released')
            return self.c.compile_language(kind, source)
        return compile, gate, entered, calls

    def await_ready(self, handle):
        deadline = time.monotonic() + 3
        while not handle.ready():
            if handle.state != 'pending' or time.monotonic() >= deadline:
                self.fail('preparation did not become ready: ' + handle.state)
            time.sleep(.001)
        return handle.result()

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        for service in self.services:
            service.close()

    def test_concurrent_single_flight_and_independent_cancellation(self):
        compile, gate, entered, calls = self.gated()
        service = self.service(compile)
        with ThreadPoolExecutor(max_workers=8) as clients:
            handles = list(clients.map(lambda _: service.request('regex', 'ab|a', **self.args), range(16)))
        self.assertTrue(entered.wait(1))
        self.assertEqual(len(calls), 1)
        with self.assertRaises(self.m.PreparationPending):
            handles[1].result()
        handles[0].cancel()
        self.assertEqual(handles[0].state, 'cancelled')
        self.assertEqual(service.stats['waiters'], 15)
        gate.set()
        artifact = self.await_ready(handles[1])
        for handle in handles[1:]:
            self.assertIs(self.await_ready(handle), artifact)
            handle.close()
        warm = service.request('regex', 'ab|a', **self.args)
        self.assertTrue(warm.ready())
        self.assertIs(warm.result(), artifact)
        self.assertEqual(len(calls), 1)
        warm.close()

    def test_queue_deadline_and_running_cancellation_keep_reserved_owners(self):
        now = [0.0]
        compile, gate, entered, calls = self.gated()
        service = self.service(compile, workers=1, max_pending=2, clock=lambda: now[0])
        a = service.request('regex', 'a', **self.args, timeout=1)
        self.assertTrue(entered.wait(1))
        b = service.request('regex', 'b', **self.args, timeout=3)
        with self.assertRaises(ValueError):
            service.request('regex', 'ab', **self.args)
        now[0] = 1
        self.assertEqual(a.state, 'timeout')
        self.assertEqual(service.stats['running'], 1)
        self.assertEqual(service.stats['pending'], 2)
        b.cancel()  # Queued job has no physical work and releases its slot.
        self.assertEqual(service.stats['pending'], 1)
        again = service.request('regex', 'a', **self.args)
        gate.set()
        self.await_ready(again)
        self.assertEqual(calls, [('regex', 'a')])
        again.close()

    def test_private_matcher_factory_and_lease_pin(self):
        service = self.service(max_entries=1)
        a = service.request('regex', 'ab|a', **self.args)
        b = service.request('regex', 'ab|a', **self.args)
        self.assertIs(self.await_ready(a), self.await_ready(b))
        # Factory/frontier isolation; actual CFG semantics are checked by native tests.
        def factory(compiled):
            return SimpleNamespace(compiled=compiled, frontier=[])
        first, second = a.new_matcher(factory), b.new_matcher(factory)
        first.matcher.frontier.append(0)
        self.assertEqual(second.matcher.frontier, [])
        self.assertIs(first.matcher.compiled, second.matcher.compiled)
        with self.assertRaises(ValueError):
            a.cancel()
        with self.assertRaises(ValueError):
            service.request('regex', 'b', **self.args)
        service.close()
        self.assertEqual(service.stats['entries'], 1)  # Live leases survive service shutdown.
        first.close()
        self.assertEqual(service.stats['waiters'], 1)
        second.close()
        self.assertEqual(service.stats['reserved_bytes'], 0)

    def test_identity_includes_language_source_version_tokenizer_vocabulary_eos(self):
        service = self.service(max_entries=16)
        initial = service.request('regex', 'a', **self.args)
        artifact = self.await_ready(initial)
        requests = [('regex', '^a$', self.args), ('ebnf', 'root ::= "a"', self.args),
                    ('regex', 'a', self.args | dict(tokenizer_identity='v2')),
                    ('regex', 'a', self.args | dict(pieces=(b'aa', b'b', b''))),
                    ('regex', 'a', self.args | dict(pieces=(b'a', b'', b'b'), eos=1))]
        for kind, source, args in requests:
            handle = service.request(kind, source, **args)
            self.assertIsNot(self.await_ready(handle), artifact)
            handle.close()
        v2 = self.c.make_compile_request('regex', 'a', **self.args, version='compiler-v2')
        v1 = self.c.make_compile_request('regex', 'a', **self.args)
        self.assertNotEqual(v1, v2)
        initial.close()

    def test_failed_compile_is_request_failure_and_retry_is_fresh(self):
        calls = []
        def compile(kind, source):
            calls.append(source)
            return self.c.compile_language(kind, source)
        service = self.service(compile)
        for _ in range(2):
            handle = service.request('regex', '[', **self.args)
            deadline = time.monotonic() + 2
            while handle.state == 'pending' and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertEqual(handle.state, 'failed')
            with self.assertRaises(self.m.PreparationFailed):
                handle.result()
        self.assertEqual(calls, ['[', '['])
        self.assertEqual(service.stats['entries'], 0)

    def test_waiter_output_and_reserved_byte_bounds(self):
        compile, gate, entered, _ = self.gated()
        service = self.service(compile, max_waiters=1)
        a = service.request('regex', 'a', **self.args)
        self.assertTrue(entered.wait(1))
        with self.assertRaises(ValueError):
            service.request('regex', 'a', **self.args)
        before = service.stats['reserved_bytes']
        a.cancel()
        self.assertEqual(service.stats['reserved_bytes'], before)
        gate.set()
        small = self.service(max_artifact_bytes=1, max_bytes=1000)
        bad = small.request('regex', 'a', **self.args)
        deadline = time.monotonic() + 2
        while bad.state == 'pending' and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertEqual(bad.state, 'failed')
        self.assertEqual(small.stats['reserved_bytes'], 0)
        tiny = self.service(max_bytes=1)
        with self.assertRaises(ValueError):
            tiny.request('regex', 'a', **self.args)
        self.assertEqual(tiny.stats['entries'], 0)

    def test_shutdown_joins_compile_and_factory_failure_keeps_artifact_safe(self):
        compile, gate, entered, _ = self.gated()
        service = self.service(compile)
        handle = service.request('regex', 'a', **self.args)
        self.assertTrue(entered.wait(1))
        closed = Event()
        thread = Thread(target=lambda: (service.close(), closed.set()))
        thread.start()
        self.assertFalse(closed.wait(.03))
        gate.set()
        thread.join(2)
        self.assertTrue(closed.is_set())
        self.assertEqual(service.stats['reserved_bytes'], 0)
        other = self.service()
        handle = other.request('regex', 'a', **self.args)
        self.await_ready(handle)
        with self.assertRaises(ValueError):
            handle.new_matcher(lambda _: None)
        self.assertTrue(handle.ready())
        handle.close()

    def test_cache_lru_and_live_bytes_are_bounded(self):
        calls = []
        def compile(kind, source):
            calls.append(source)
            return self.c.compile_language(kind, source)
        service = self.service(compile, max_entries=2)
        for source in ('a', 'b', 'a', 'ab', 'b'):
            handle = service.request('regex', source, **self.args)
            self.await_ready(handle)
            self.assertLessEqual(service.stats['entries'], 2)
            self.assertLessEqual(service.stats['reserved_bytes'], service.max_bytes)
            handle.close()
        self.assertEqual(calls, ['a', 'b', 'ab', 'b'])


if __name__ == '__main__':
    unittest.main()
