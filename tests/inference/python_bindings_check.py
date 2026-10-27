"""The python-bindings shared library through crit_inference.py on a CUDA device.

Every declared function loads; launches see their scalar and buffer arguments in order (greedy sampling through
packed Request and Params rows, grouped accepted counts); a refused launch raises CudaError with its code. Buffers
are pinned, mapped host memory, so the check needs no array library."""
import ctypes
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference/python-bindings/code"))
import crit_inference  # noqa: E402

LIBRARY = Path(sys.argv[1]).resolve()
INVALID_VALUE = 1


class PythonBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = crit_inference.Library(LIBRARY)
        cls.buffers = []

    @classmethod
    def tearDownClass(cls):
        for host in cls.buffers:
            cls.lib.host_free(host)

    def mapped(self, data):
        host, device = self.lib.host_alloc(max(len(data), 8))
        self.buffers.append(host)
        ctypes.memmove(host, data, len(data))
        return host, device

    def ints(self, host, count):
        return list(struct.unpack(f"<{count}i", ctypes.string_at(host, 4 * count)))

    def test_every_declared_function_is_exported(self):
        self.assertEqual(self.lib.abi_version(), crit_inference.ABI_VERSION)
        self.assertEqual(set(self.lib.signatures), set(crit_inference.signatures()))
        self.assertGreater(self.lib.kv_decode_partial_floats(4, 128, 8), 0)
        self.assertEqual(self.lib.packed_layout(3)["words"], 3 + 4)

    def test_greedy_rows_select_the_lowest_maximal_id(self):
        vocab, rows = 50, 2
        logits = [0.0] * (vocab * rows)
        logits[7] = logits[9] = 3.0          # tie: the lower id wins
        logits[vocab + 42] = 1.0
        _, device_logits = self.mapped(struct.pack(f"<{len(logits)}f", *logits))
        _, scratch = self.mapped(bytes(self.lib.ts_select_scratch_bytes(vocab, rows)))
        params, device_params = self.mapped(self.lib.params() * rows)
        out, device_out = self.mapped(bytes(4 * rows))
        self.lib.ts_select(device_logits, vocab, rows, vocab, device_params, scratch, device_out, 0)
        self.lib.stream_synchronize(0)
        self.assertEqual(self.ints(out, rows), [7, 42])
        offset = self.lib.ts_params_position_offset()
        self.assertEqual(struct.unpack_from("<Q", ctypes.string_at(params, len(self.lib.params())), offset)[0], 1)

        requests = self.lib.request() * rows
        _, device_request = self.mapped(requests)
        _, sample_scratch = self.mapped(bytes(4 * self.lib.ts_scratch_floats(vocab, rows)))
        sampled, device_sampled = self.mapped(bytes(4 * rows))
        self.lib.ts_sample(device_logits, vocab, vocab, rows, 1, device_request, None, 0, 0, 0, sample_scratch, None,
                           None, 0, device_sampled, None, None, 0, None, None, None, 0)
        self.lib.stream_synchronize(0)
        self.assertEqual(self.ints(sampled, rows), [7, 42])

    def test_grouped_accepted_counts_reach_each_group(self):
        _, accepted = self.mapped(struct.pack("<i", 2))
        pending, device_pending = self.mapped(struct.pack("<12i", *([-1] * 12)))
        self.lib.rr_record_accepted(accepted, 4, device_pending, 2, 3, 4, 0)
        self.lib.stream_synchronize(0)
        self.assertEqual(self.ints(pending, 12), [2, 2, -1, -1] * 3)
        self.lib.rr_record_accepted(accepted, 1, device_pending, 0, 0, 0, 0)
        self.lib.stream_synchronize(0)
        self.assertEqual(self.ints(pending, 1), [1])

    def test_a_refused_launch_raises_its_cuda_error(self):
        with self.assertRaises(crit_inference.CudaError) as raised:
            self.lib.ts_commit(None, 1, 50, None, 0, None, 0, 0)
        self.assertEqual(raised.exception.code, INVALID_VALUE)
        self.assertIn("crit_ts_commit", str(raised.exception))


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
