"""Bounded transport, delayed physical readers and private bundle publication."""
from concurrent.futures import Future
import gc
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
import weakref

CODE = Path(__file__).resolve().parents[1] / "inference" / "checkpoint-storage" / "code"


def load(name):
    identifier = "crit_transfer_tests_" + name
    spec = importlib.util.spec_from_file_location(identifier, CODE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[identifier] = module
    spec.loader.exec_module(module)
    return module


t, b = load("transfers"), load("bundles")


class Backend:
    """Deferred physical work: no buffer access until the fixture completes it."""
    def __init__(self, values):
        self.values = values
        self.destinations = {name: bytearray(len(data)) for name, data in values.items()}
        self.reads, self.writes, self.calls = [], [], []
        self.extra_reader = None

    def read(self, window):
        receipt = window.record(Future())
        self.reads.append((window, receipt))
        self.calls.append(("read", window.part.name, window.offset, window.nbytes))
        return receipt

    def write(self, window):
        receipt = window.record(Future())
        if self.extra_reader is not None:
            window.track_work(self.extra_reader)
        self.writes.append((window, receipt))
        self.calls.append(("write", window.part.name, window.offset, window.nbytes))
        return receipt

    def finish_read(self, index=0):
        window, receipt = self.reads.pop(index)
        window.buffer[:] = self.values[window.part.name][window.offset:window.offset + window.nbytes]
        receipt.set_result(window.nbytes)

    def finish_write(self, index=0):
        window, receipt = self.writes.pop(index)
        assert window.buffer.readonly
        self.destinations[window.part.name][window.offset:window.offset + window.nbytes] = window.buffer
        receipt.set_result(window.nbytes)

    def finish(self):
        while self.reads:
            self.finish_read()
        while self.writes:
            self.finish_write()


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.retained = set(t.retained_transfers())

    def tearDown(self):
        # Fixture callbacks never launch real asynchronous I/O. This explicit
        # proof covers even intentionally lost/throwing completion receipts.
        for transfer in t.retained_transfers():
            if transfer not in self.retained:
                transfer.discard(lambda window: True)
        for restore in b.retained_restores():
            restore.discard(lambda operation: True)
        self.assertEqual(set(t.retained_transfers()), self.retained)

    def make(self, values=None, *, window=4, staging=8, windows=2, read=None, write=None, owners=()):
        values = values if values is not None else {"state": b"abcdefghijk"}
        backend = Backend(values)
        parts = tuple(t.TransferPart(name, len(data), hashlib.sha256(data).hexdigest())
                      for name, data in values.items())
        transfer = t.BoundedTransfer(parts, t.TransferLimits(window, staging, windows, 1 << 20),
                                     read or backend.read, write or backend.write, owners=owners)
        return transfer, backend

    def drain(self, transfer, backend):
        for _ in range(1000):
            if transfer.done():
                return transfer.result()
            transfer.pump(max_bytes=16, max_submissions=4)
            backend.finish()
        self.fail("transfer did not make bounded progress")

    def test_both_stages_obey_each_pump_quota_and_staging_backpressure(self):
        transfer, backend = self.make()
        self.assertEqual(transfer.pump(max_bytes=8, max_submissions=2), 8)
        self.assertEqual(transfer.staging_bytes, 8)
        for _ in range(3):
            self.assertEqual(transfer.pump(max_bytes=8, max_submissions=2), 0)
        backend.finish_read()
        before = len(backend.calls)
        self.assertEqual(transfer.pump(max_bytes=4, max_submissions=2), 4)
        self.assertEqual(backend.calls[before:], [("write", "state", 0, 4)])
        # A pending write still consumes its complete staging reservation.
        self.assertEqual(transfer.staging_bytes, 8)
        backend.finish_write()
        self.assertEqual(transfer.pump(max_bytes=4), 3)
        self.assertLessEqual(transfer.staging_bytes, 8)
        backend.finish()
        self.assertEqual(self.drain(transfer, backend), 11)
        self.assertEqual(backend.destinations["state"], b"abcdefghijk")
        self.assertEqual(transfer.peak_staging_bytes, 8)
        self.assertEqual(transfer.read_bytes, 11)
        self.assertEqual(transfer.staging_bytes, 0)

    def test_out_of_order_reads_are_hashed_in_order_and_empty_parts_advance(self):
        values = {"empty": b"", "a": b"abcdefghi", "b": b"xyz", "tail": b""}
        transfer, backend = self.make(values)
        transfer.pump(max_bytes=8, max_submissions=2)
        backend.finish_read(1)
        transfer.pump(max_bytes=8, max_submissions=2)
        self.assertEqual(backend.writes, [])
        backend.finish_read()
        transfer.pump(max_bytes=8, max_submissions=2)
        backend.finish_write(1)
        backend.finish_write()
        self.drain(transfer, backend)
        self.assertEqual(backend.destinations, values)

    def test_extra_graph_or_eager_reader_blocks_staging_reuse(self):
        for slots in (1, 2):
            with self.subTest(slots=slots):
                transfer, backend = self.make(window=4, staging=4, windows=slots)
                reader = Future()
                backend.extra_reader = reader
                transfer.pump(max_bytes=4)
                backend.finish_read()
                transfer.pump(max_bytes=4)
                backend.finish_write()
                self.assertEqual(transfer.pump(max_bytes=16, max_submissions=4), 0)
                self.assertEqual(transfer.staging_bytes, 4)
                with self.assertRaises(t.PendingTransfer):
                    transfer.result()
                reader.set_result(None)
                self.drain(transfer, backend)

    def test_cancellation_stops_new_work_but_holds_pending_source_and_destination(self):
        transfer, backend = self.make()
        transfer.pump(max_bytes=8, max_submissions=2)
        calls = list(backend.calls)
        transfer.cancel()
        with self.assertRaises(ValueError):
            transfer.pump(max_bytes=8)
        with self.assertRaises(t.PendingTransfer):
            transfer.discard()
        self.assertFalse(transfer.done())
        self.assertEqual(backend.calls, calls)
        self.assertEqual(transfer.staging_bytes, 8)
        backend.finish()
        self.assertTrue(transfer.done())
        with self.assertRaises(t.TransferFailure):
            transfer.result()
        transfer.discard()
        self.assertEqual(transfer.staging_bytes, 0)
        self.assertEqual(backend.destinations["state"], bytes(11))

    def test_failed_write_never_releases_pending_sibling(self):
        transfer, backend = self.make()
        transfer.pump(max_bytes=8, max_submissions=2)
        backend.finish()
        transfer.pump(max_bytes=8, max_submissions=2)
        _, failed = backend.writes.pop(0)
        failed.set_exception(OSError("disk full"))
        with self.assertRaises(t.TransferFailure):
            transfer.pump(max_bytes=8)
        self.assertFalse(transfer.done())
        with self.assertRaises(t.PendingTransfer):
            transfer.discard()
        backend.finish()
        self.assertTrue(transfer.done())
        transfer.discard()

    def test_unknown_submission_remains_retained_after_caller_loses_handle(self):
        class Owner:
            pass

        receipts, references = [], []

        def launch():
            owner = Owner()
            references.append(weakref.ref(owner))

            def broken(window):
                receipts.append(window.record(Future()))
                raise OSError("submission may have reached the device")

            transfer, _ = self.make(read=broken, owners=(owner,))
            references.append(weakref.ref(transfer))
            try:
                transfer.pump(max_bytes=8)
            except t.TransferFailure:
                pass

        launch()
        gc.collect()
        self.assertTrue(all(r() is not None for r in references))
        transfer = references[1]()
        receipts[0].set_result(4)
        self.assertFalse(transfer.done())  # Known receipt does not settle unknown work.
        for proof in (False, 1, "yes"):
            with self.assertRaises(t.PendingTransfer):
                transfer.discard(lambda window: proof)
        transfer.discard(lambda window: True)
        gc.collect()
        self.assertIsNone(references[0]())

    def test_unknown_fence_and_nonboolean_completion_require_explicit_proof(self):
        for result in ("throw", None, 1):
            class Receipt:
                def done(self):
                    if result == "throw":
                        raise OSError("lost fence")
                    return result

                def result(self):
                    return 4

            transfer, _ = self.make(read=lambda window: window.record(Receipt()))
            transfer.pump(max_bytes=4)
            self.assertFalse(transfer.done())
            with self.assertRaises(t.PendingTransfer):
                transfer.discard()
            transfer.discard(lambda window: True)

    def test_short_or_invalid_io_cannot_succeed(self):
        for value in (0, 3, 5, True, None):
            def short(window):
                receipt = window.record(Future())
                receipt.set_result(value)
                return receipt

            for phase in ("read", "write"):
                transfer, backend = self.make(**{phase: short})
                transfer.pump(max_bytes=4)
                backend.finish()
                if phase == "write":
                    transfer.pump(max_bytes=4)
                self.assertTrue(transfer.done())
                with self.assertRaises(t.TransferFailure):
                    transfer.result()
                transfer.discard()

    def test_corruption_is_detected_even_if_all_physical_writes_succeed(self):
        transfer, backend = self.make()
        backend.values["state"] = b"ABCDEFGHIJK"
        with self.assertRaises(t.TransferFailure):
            self.drain(transfer, backend)
        self.assertEqual(transfer.state, "failed")
        transfer.discard()

    def test_lost_registration_and_reentrant_callback_fail_closed(self):
        def no_registration(window):
            return Future()

        transfer, _ = self.make(read=no_registration)
        with self.assertRaises(t.TransferFailure):
            transfer.pump(max_bytes=4)
        with self.assertRaises(t.PendingTransfer):
            transfer.discard()
        transfer.discard(lambda window: True)

        def recursive(window):
            transfer.cancel()

        transfer, _ = self.make(read=recursive)
        with self.assertRaises(t.TransferFailure):
            transfer.pump(max_bytes=4)
        transfer.discard(lambda window: True)

    def test_large_file_stream_uses_bounded_windows_and_exact_tail(self):
        # Keep neither a component-sized read nor an in-memory output. The
        # expected digest is calculated independently while creating the file.
        block = bytes(range(251))
        with tempfile.TemporaryDirectory() as directory:
            source_path, target_path = Path(directory) / "source", Path(directory) / "private"
            digest = hashlib.sha256()
            with source_path.open("wb") as source:
                for _ in range(2000):
                    source.write(block)
                    digest.update(block)
            with source_path.open("rb", buffering=0) as source, target_path.open("w+b", buffering=0) as target:
                count = source_path.stat().st_size
                transfer = t.BoundedTransfer((t.TransferPart("state", count, digest.hexdigest()),),
                    t.TransferLimits(4096, 7000, 2, count),
                    lambda w: t.read_file_window(w, source), lambda w: t.write_file_window(w, target),
                    owners=(source, target))
                while not transfer.done():
                    sent = transfer.pump(max_bytes=8192, max_submissions=3)
                    self.assertLessEqual(sent, 8192)
                    self.assertLessEqual(transfer.staging_bytes, 7000)
                self.assertEqual(transfer.result(), count)
                self.assertEqual(target.seek(0, 2), count)
                target.seek(0)
                actual = hashlib.file_digest(target, "sha256").hexdigest()
                self.assertEqual(actual, digest.hexdigest())
                self.assertLessEqual(transfer.peak_staging_bytes, 7000)

    def test_truncated_file_and_write_error_never_report_success(self):
        def broken_write(window):
            class Broken(io.BytesIO):
                def write(self, data):
                    raise OSError("disk full")
            return t.write_file_window(window, Broken())

        for read, write in ((lambda w: t.read_file_window(w, io.BytesIO(b"ab")), None),
                            (None, broken_write)):
            transfer, backend = self.make(read=read, write=write)
            with self.assertRaises(t.TransferFailure):
                self.drain(transfer, backend)
            transfer.discard()

    def test_complete_bundle_publication_waits_for_every_window_and_sibling(self):
        values = {"attention": b"012345678", "recurrent": b"matrix", "convolution": b"abc"}
        description = b.BundleSpec("synthetic-layout", "complete-state", tuple(
            b.ComponentSpec(name, "opaque", 16) for name in values), 32)
        plan = b.decode_bundle(description, b.encode_bundle(description, values), prefix=b"after-token-9")
        transfer, backend = self.make(values)
        transaction = b.RestoreTransaction(plan)
        transaction.submit(lambda op: backend.destinations[op.component.name],
                           lambda op: op.record_copy(transfer))
        live = {"state": b"untouched"}
        with self.assertRaises(b.PendingRestore):
            transaction.publish()
        transfer.pump(max_bytes=8, max_submissions=2)
        backend.finish()
        with self.assertRaises(b.PendingRestore):
            transaction.publish()
        self.assertEqual(live, {"state": b"untouched"})
        self.drain(transfer, backend)
        restored = transaction.publish()
        self.assertEqual(dict(restored.destinations), values)
        self.assertEqual(restored.prefix, b"after-token-9")

    def test_budget_validation_precedes_any_backend_work(self):
        part = t.TransferPart("state", 10, hashlib.sha256(b"x" * 10).hexdigest())
        for limits in ((0, 8, 2, 10), (9, 8, 2, 10), (4, 8, 0, 10), (4, 8, 65, 10), (True, 8, 2, 10)):
            with self.assertRaises(ValueError):
                t.TransferLimits(*limits)
        with self.assertRaises(ValueError):
            t.BoundedTransfer((part,), t.TransferLimits(4, 8, 2, 9), lambda w: None, lambda w: None)
        transfer, backend = self.make()
        for quota in (-1, 1, True):
            with self.assertRaises(ValueError):
                transfer.pump(max_bytes=quota)
        self.assertEqual(transfer.pump(max_bytes=0), 0)
        self.assertEqual(backend.calls, [])
        transfer.discard()

        def oversized(value):
            for _ in range(65):
                yield value
            self.fail("bounded inventory validation consumed excess metadata")

        limits = t.TransferLimits(4, 8, 2, 10000)
        for parts, owners in ((oversized(part), ()), ((part,), oversized(object()))):
            with self.assertRaises(ValueError):
                t.BoundedTransfer(parts, limits, backend.read, backend.write, owners=owners)

    def test_retained_batch_limit_rejects_before_allocation_or_submission(self):
        transfers = []
        for _ in range(16):
            transfer, _ = self.make()
            transfer.pump(max_bytes=4)
            transfers.append(transfer)
        overflow, backend = self.make()
        with self.assertRaises(ValueError):
            overflow.pump(max_bytes=4)
        self.assertEqual(backend.calls, [])
        self.assertEqual(overflow.staging_bytes, 0)
        self.assertEqual(overflow.state, "created")
        # Every fixture backend is inert. Releasing one settled batch admits
        # exactly one replacement while the other batches stay retained.
        transfers[0].discard(lambda window: True)
        overflow.pump(max_bytes=4)
        self.assertEqual(len(t.retained_transfers()), 16)

    def test_read_dependency_prevents_hash_or_write_before_producer_settles(self):
        producer = Future()
        backend = Backend({"state": b"abcd"})

        def read(window):
            receipt = backend.read(window)
            window.track_work(producer)
            return receipt

        transfer, _ = self.make({"state": b"abcd"}, read=read, write=backend.write)
        transfer.pump(max_bytes=4)
        backend.finish_read()
        self.assertEqual(transfer.pump(max_bytes=8, max_submissions=2), 0)
        self.assertEqual(backend.writes, [])
        producer.set_result(None)
        self.drain(transfer, backend)
        self.assertEqual(backend.destinations["state"], b"abcd")


if __name__ == "__main__":
    unittest.main()
