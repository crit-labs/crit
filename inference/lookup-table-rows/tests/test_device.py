"""lookup-table-rows' device form gives the stored rows byte for byte: from the device, from registered host memory,
from one or many files through the native reader (and its Python fallback); prefetched rows are served from the
hot-row cache and the others read; a lookup over files or host memory captures in a CUDA graph and replays; a buffer
set is reused only after its last lookup ran; a read that failed in a host function is never served again."""
import os
import tempfile
import unittest

from crit_plugins.testing import PluginTestCase

# Host memory regions (a table opened from a tensor, or a file mapped) need libc's mmap, which Windows does not
# have; files read by the readers and device tables work everywhere.
HOST_PLACES = ("host", "mapped")
NO_HOST_MEMORY = "host memory regions need libc's mmap, which Windows does not have"


class HostRowsTests(PluginTestCase):
    def setUp(self):
        self.torch = __import__("torch") if self.device() else None
        g = self.torch.Generator().manual_seed(3)
        self.table = self.torch.randint(0, 256, (1000, 90), dtype=self.torch.uint8, generator=g)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.paths = []
        for i, (first, last) in enumerate(((0, 300), (300, 301), (301, 1000))):
            path = os.path.join(self.folder.name, f"rows-{i}.bin")
            with open(path, "wb") as f:
                f.write(self.table[first:last].numpy().tobytes())
            self.paths.append(path)

    def plugin(self, **options):
        return self.assemble(options=options)["embeddings"]

    def ids(self, values):
        return self.torch.tensor(values, dtype=self.torch.int64, device=self.device())

    def test_every_place_gives_the_stored_rows(self):
        torch, device = self.torch, self.device()
        want = [5, 299, 300, 301, 999, 5, 0, 640]
        expected = self.table[want]
        for place in ("device", "host", "files", "mapped", "python"):
            with self.subTest(place=place):
                if place in HOST_PLACES and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                rows = self.plugin(native=place != "python")
                if place == "device":
                    rows.store(self.table.to(device))
                elif place == "host":
                    rows.open(self.table, 90, torch.uint8, device)
                elif place == "mapped":
                    rows.open(self.paths[2], 90, torch.uint8, device, mapped=True)
                else:
                    rows.open(self.paths, 90, torch.uint8, device)
                ids = want if place != "mapped" else [w - 301 for w in want if w >= 301]
                got = rows.lookup(self.ids(ids)).cpu()
                ref = expected if place != "mapped" else self.table[[w for w in want if w >= 301]]
                self.assertTrue(torch.equal(got, ref))
                with self.assertRaises(IndexError):
                    rows.lookup(self.ids([1000 if place != "mapped" else 699]))
                if place in ("files", "python"):
                    # The native reader where it built here (a compiler and Ninja); else the facts say why not.
                    facts = rows.facts()["embeddings"]
                    native = place == "files" and "native_unavailable" not in facts
                    self.assertEqual(facts["reader"], "native" if native else "python")

    def test_prefetched_rows_come_from_the_cache_and_the_rest_are_read(self):
        torch, device = self.torch, self.device()
        for native in (True, False):
            with self.subTest(native=native):
                rows = self.plugin(native=native, prefetch_rows=64)
                rows.open(self.paths, 90, torch.uint8, device)
                rows.prefetch([10, 11, 12, 700])
                rows.device_rows.drain()
                self.assertEqual(rows.telemetry()["table_rows"]["prefetched"], 4)
                got = rows.lookup(self.ids([10, 700, 13, 11, 13])).cpu()
                self.assertTrue(torch.equal(got, self.table[[10, 700, 13, 11, 13]]))
                stats = rows.telemetry()["table_rows"]
                self.assertEqual((stats["hits"], stats["reads"] - 4), (3, 1))   # 13 read, once

    @unittest.skipIf(os.name == "nt", NO_HOST_MEMORY)
    def test_tables_are_handles_and_nvfp4_rows_decode_to_their_values(self):
        torch, device = self.torch, self.device()
        g = torch.Generator().manual_seed(8)
        codes = torch.randint(0, 256, (500, 16), dtype=torch.uint8, generator=g)      # 32 values a row
        scales = torch.randint(0, 0x7F, (500, 2), dtype=torch.uint8, generator=g)      # E4M3, no NaN
        e2m1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

        def e4m3(byte):   # sign, 4 exponent bits (bias 7), 3 mantissa bits; exponent 0 subnormal
            sign = -1.0 if byte & 0x80 else 1.0
            exponent, mantissa = (byte >> 3) & 15, byte & 7
            return sign * (mantissa / 8 * 2.0 ** -6 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7))

        def row(r):
            out = []
            for k in range(32):
                nibble = (int(codes[r, k // 2]) >> (4 * (k % 2))) & 15
                value = e2m1[nibble & 7] * (-1.0 if nibble & 8 else 1.0)
                out.append(value * e4m3(int(scales[r, k // 16])) * 0.25)
            return out

        want = [3, 499, 0, 3]
        expected = torch.tensor([row(r) for r in want], dtype=torch.float32).bfloat16()
        paths = []
        for name, data in (("codes", codes), ("scales", scales)):
            path = os.path.join(self.folder.name, f"{name}.bin")
            with open(path, "wb") as f:
                f.write(data.numpy().tobytes())
            paths.append(path)
        rows = self.plugin(prefetch_rows=32)
        tokens = rows.store(self.table[:50].to(device))
        made = [rows.store(codes.to(device), format="nvfp4", scales=scales.to(device), scale=0.25),
                rows.open(codes, 32, torch.bfloat16, device, format="nvfp4", scales=scales, scale=0.25),
                rows.open(paths[0], 32, torch.bfloat16, device, format="nvfp4", scales=paths[1], scale=0.25)]
        for table in made:
            table.prefetch(want)
            got = table.lookup(self.ids(want)).cpu()
            self.assertEqual(got.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(got.view(torch.int16), expected.view(torch.int16)), table.mode)
        self.assertTrue(torch.equal(tokens.lookup(self.ids([4, 9])).cpu(), self.table[[4, 9]]))
        self.assertTrue(torch.equal(rows.lookup(self.ids([7])).cpu(), expected[:0].new_tensor([row(7)]).bfloat16()))
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            rows.open(codes, 24, torch.bfloat16, device, format="nvfp4", scales=scales)

    def test_a_lookup_captures_in_a_cuda_graph(self):
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        torch = self.torch
        for place in ("files", "host"):
            with self.subTest(place=place):
                if place in HOST_PLACES and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                rows = self.plugin(prefetch_rows=16)
                if place == "files":
                    rows.open(self.paths, 90, torch.uint8, "cuda")
                else:
                    rows.open(self.table, 90, torch.uint8, "cuda")
                ids = self.ids([1, 2, 3])
                rows.lookup(ids)                       # buffers of this size, outside the capture
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        out = rows.lookup(ids)
                        total = out.sum(1, dtype=torch.int64)
                for want in ([7, 300, 999], [0, 0, 512], [301, 302, 299]):
                    ids.copy_(torch.tensor(want))
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(out.cpu(), self.table[want]))
                    self.assertEqual(total.tolist(), self.table[want].sum(1, dtype=torch.int64).tolist())

    def test_host_ids_reuse_a_buffer_set_only_after_its_last_lookup_ran(self):
        if self.device() != "cuda":
            self.skipTest("lookups on a stream")
        torch = self.torch
        for place in ("files", "host"):
            with self.subTest(place=place):
                if place in HOST_PLACES and os.name == "nt":
                    self.skipTest(NO_HOST_MEMORY)
                rows = self.plugin(ring=2)
                rows.open(self.paths if place == "files" else self.table, 90, torch.uint8, "cuda")
                wants = [[i, 999 - i, 300 + i] for i in range(7)]
                torch.cuda._sleep(100_000_000)   # the stream runs late: every lookup is queued behind this
                got = [rows.lookup(torch.tensor(want)).clone() for want in wants]   # host ids, more than the ring
                torch.cuda.synchronize()
                for want, out in zip(wants, got):
                    self.assertTrue(torch.equal(out.cpu(), self.table[want]), want)

    def test_a_failed_read_is_never_served_as_rows(self):
        torch, device = self.torch, self.device()
        rows = self.plugin(native=True, prefetch_rows=0)
        rows.open(self.paths, 90, torch.uint8, device)
        if rows.facts()["embeddings"]["reader"] != "native":
            self.skipTest("the native row reader did not build")
        os.truncate(self.paths[2], 90 * 100)   # rows 401 and on are gone after the open
        if device == "cuda":
            first = rows.lookup(self.ids([5, 900]))   # a host function cannot raise: it zero-fills and marks
            torch.cuda.synchronize()
            self.assertEqual(first.cpu().sum().item(), 0)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                rows.lookup(self.ids([5]))
            with self.assertRaisesRegex(RuntimeError, "failed"):
                rows.device_rows.drain()
            self.assertGreaterEqual(rows.telemetry()["table_rows"]["failures"], 1)
        else:   # on the host the gather raises where it fails
            with self.assertRaises(RuntimeError):
                rows.lookup(self.ids([5, 900]))
        self.assertEqual(rows.round_problems(), [])
