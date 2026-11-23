"""crit_plugins.host: the host's processors as one object. The plan says who runs where; the pool runs every task
of a batch once and serves whoever asks; a handoff's job, posted inside a captured graph, is answered by the host
on every replay (CUDA)."""
import ctypes
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "core" / "python"))

from crit_plugins import native  # noqa: E402
from crit_plugins.host import Host, current, memory_bytes  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

# A handoff's handler: payload [0] is the device's input, [1] the host's answer; an input of 0 fails.
SERVE = """
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
extern "C" int serve_triple(void* context, std::uint64_t, void* payload, std::uint64_t bytes, char* error, std::uint64_t n) {
    std::uint32_t in = 0;
    std::memcpy(&in, payload, 4);
    if(in == 0 || bytes < 8) {
        std::strncpy(error, "no input", n - 1);
        return 1;
    }
    if(in == 77) std::this_thread::sleep_for(std::chrono::milliseconds(600));   // a job that does not come back in time
    const std::uint32_t out = 3 * in + 1;
    std::memcpy(static_cast<char*>(payload) + 4, &out, 4);
    ++*static_cast<std::uint64_t*>(context);
    return 0;
}
"""


def buildable():
    try:
        native.compiler()
        return True
    except native.NativeUnavailable:
        return False


class PlanTests(unittest.TestCase):
    def test_the_first_core_is_the_services_and_each_other_core_has_one_worker(self):
        host = Host(cores=[0, 2, 4, 6])
        self.assertEqual(host.plan(), {"service": 0, "workers": [2, 4, 6]})
        self.assertEqual(host.plan(workers=2), {"service": 0, "workers": [2, 4]})
        self.assertEqual(host.plan(workers=5), {"service": 0, "workers": [2, 4, 6, -1, -1]})   # past the cores: unpinned
        self.assertEqual(host.plan(workers=3, pin=False), {"service": None, "workers": [-1, -1, -1]})
        self.assertEqual(Host(cores=[5]).plan(), {"service": 5, "workers": []})
        unknown = Host(cores=[]).plan()   # the OS names no cores: nothing is pinned
        self.assertIsNone(unknown["service"])
        self.assertEqual(set(unknown["workers"]) - {-1}, set())
        report = host.report()
        self.assertEqual(report["cores"], 4)
        self.assertEqual(set(memory_bytes()), {"total", "available", "binding"})
        self.assertNotIn("pool", report)   # nobody asked for the pool: no thread was started


@unittest.skipUnless(buildable(), "no C++ compiler for native code built at load")
class PoolTests(unittest.TestCase):
    def test_the_pool_runs_every_task_of_a_batch_once_and_is_one_for_everyone(self):
        host = Host()
        self.addCleanup(host.close)
        pool = host.pool(workers=3, pin=False)
        self.assertIs(host.pool(), pool)
        self.assertIs(host.pool(workers=7), pool)   # the pool as it is serves whoever asks later
        hits = [0] * 400
        for n in (1, 7, 400):
            hits[:n] = [0] * n
            pool.run(n, lambda i: hits.__setitem__(i, hits[i] + 1))
            self.assertEqual(hits[:n], [1] * n)

        def fails(i):
            if i == 7:
                raise ValueError("task 7")
        with self.assertRaisesRegex(ValueError, "task 7"):
            pool.run(10, fails)
        nested = []
        pool.run(4, lambda i: pool.run(3, lambda j: nested.append((i, j))))   # a task's own batch runs on its thread
        self.assertEqual(sorted(nested), [(i, j) for i in range(4) for j in range(3)])
        stats = host.report()["pool"]
        self.assertEqual((stats["workers"], stats["cpus"]), (3, []))
        self.assertGreaterEqual(stats["batches"], 5)
        entry, handle = pool.native
        self.assertTrue(entry and handle)

    def test_the_process_has_one_host_whose_cores_the_os_names(self):
        host = current()
        self.assertIs(current(), host)
        cores = host.cores
        if cores:   # Linux: the plan pins the service to the first core and a worker to each other one
            self.assertEqual(host.plan(), {"service": cores[0], "workers": cores[1:]})
            self.assertTrue(set(cores) <= os.sched_getaffinity(0))


@unittest.skipUnless(buildable() and torch is not None and torch.cuda.is_available(), "no CUDA device and compiler")
class HandoffTests(unittest.TestCase):
    def test_a_captured_post_and_join_is_answered_by_the_host_on_every_replay(self):
        host = current()
        try:
            joins = host.joins("cuda")
        except native.NativeUnavailable as error:
            self.skipTest(f"Core's handoff does not build here: {error}")
        self.assertIn(joins["choice"], ("stream_wait", "device_poll"))
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "serve.cpp"
            source.write_text(SERVE, encoding="utf-8")
            lib = native.library("crit_test_serve", [source])
            served = ctypes.c_uint64(0)
            handoff = host.handoff("cuda", "test", "triple", 2, 64, ctypes.cast(lib.serve_triple, ctypes.c_void_p).value,
                                   ctypes.addressof(served))
            self.addCleanup(handoff.close)
            self.assertEqual(handoff.join_kind, joins["choice"])
            self.assertEqual(host.report()["handoffs"]["test/triple"]["slots"], 2)
            # Mapped memory: the payload's device address is its host address too.
            payload = (ctypes.c_uint32 * 2).from_address(handoff.words(0)[3])
            graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            payload[0] = 1
            with torch.cuda.stream(stream):
                handoff.post(0)   # the join's path once outside the capture
                handoff.join(0)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                handoff.post(0)
                handoff.join(0)
                handoff.skip(1)   # a slot the device answers itself
                handoff.join(1)
            for value in (5, 6, 7):
                payload[0], payload[1] = value, 0
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(payload[1], 3 * value + 1)
            self.assertEqual((served.value, handoff.served(), handoff.failed()), (4, 4, False))
            payload[0] = 0   # the handler fails: the join still completes, and the handoff says so
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(handoff.failed())
            self.assertIn("no input", handoff.error())
            handoff.close()
            self.assertNotIn("handoffs", host.report())
            # A job that runs past the service's budget: the watch fails the handoff and the device stops waiting
            # long before the job returns.
            slow = host.handoff("cuda", "test", "slow", 1, 64, ctypes.cast(lib.serve_triple, ctypes.c_void_p).value,
                                ctypes.addressof(served))
            self.addCleanup(slow.close)
            self.addCleanup(host.job_budget, 20000)
            host.job_budget(100)
            self.assertEqual(host.report()["service"]["job_budget_ms"], 100)
            payload = (ctypes.c_uint32 * 2).from_address(slow.words(0)[3])
            payload[0], payload[1] = 77, 0
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                slow.post(0)
                slow.join(0)
            stream.synchronize()
            waited = time.perf_counter() - started
            self.assertLess(waited, 0.5)
            self.assertTrue(slow.failed())
            self.assertIn("job budget", slow.error())
            time.sleep(0.7)   # the job returns late: nothing counts it as served
            self.assertEqual(slow.served(), 0)
            slow.close()
            self.assertEqual(host.report()["native"]["crit_host_handoff"]["source"], "built")


if __name__ == "__main__":
    unittest.main()
