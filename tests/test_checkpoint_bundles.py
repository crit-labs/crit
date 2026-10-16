"""Opaque bundle extent/identity and physical-owner restoration regressions."""
from concurrent.futures import Future
from dataclasses import FrozenInstanceError, replace
import gc
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "inference" / "checkpoint-storage" / "code"


def load(name):
    identifier = "crit_bundle_tests_" + name
    if identifier not in sys.modules:
        spec = importlib.util.spec_from_file_location(identifier, CODE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[identifier] = module
        spec.loader.exec_module(module)
    return sys.modules[identifier]


m = load("bundles")
storage = load("implementation")


class Owner:
    def __init__(self, nbytes):
        self.bytes = bytearray(nbytes)


def spec():
    return m.BundleSpec("verified-weights/config/representation-v1", "bundle-codec-v1", (
        m.ComponentSpec("attention", "kv-v1", 16),
        m.ComponentSpec("convolution", "conv-v1", 16),
        m.ComponentSpec("recurrent", "ssm-v1", 16),
    ), max_payload_bytes=32)


def values():
    return {"attention": b"abcde", "convolution": b"window", "recurrent": b"matrix"}


def mutate(encoded, change):
    _, nbytes = struct.unpack_from("<8sI", encoded)
    manifest = json.loads(encoded[12:12 + nbytes])
    change(manifest)
    altered = json.dumps(manifest, separators=(",", ":")).encode()
    return struct.pack("<8sI", b"CRITBND1", len(altered)) + altered + encoded[12 + nbytes:]


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.restores = set(m.retained_restores())
        self.writes = set(m.retained_writes())

    def tearDown(self):
        # All fixture backends are inert host mocks. Completing their manually
        # controlled futures and this proof settle all fixture physical work.
        for transaction in m.retained_restores():
            if transaction not in self.restores:
                for operation in transaction.operations:
                    for work in operation.work:
                        if not work.receipt.done():
                            work.receipt.set_result(None)
                transaction.discard(lambda operation: True)
        for lease in m.retained_writes():
            if lease not in self.writes:
                if not lease.receipt.done():
                    lease.receipt.set_result(None)
                try:
                    lease.retire()
                except RuntimeError:
                    pass
        self.assertEqual(set(m.retained_restores()), self.restores)
        self.assertEqual(set(m.retained_writes()), self.writes)

    def plan(self):
        return m.decode_bundle(spec(), m.encode_bundle(spec(), values()), prefix=b"semantic-prefix")

    def allocate(self, operation):
        return Owner(operation.component.nbytes)

    def copy(self, operation):
        receipt = operation.record_copy(Future())
        operation.destination.bytes[:] = operation.source
        receipt.set_result(None)
        return receipt

    def test_store_binds_complete_codecs_and_immutable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            description = spec()
            outer = storage.CheckpointStore(directory, description.fingerprint, description.codec, 8192)
            store = m.BundleStore(outer, description)
            store.put(b"prefix", values())
            plan = store.longest_prefix(b"prefix/more")
            self.assertEqual(plan.prefix, b"prefix")
            self.assertEqual({c.name: bytes(plan.source(c)) for c in plan.components}, values())
            self.assertTrue(all(plan.source(c).readonly for c in plan.components))
            with self.assertRaises(FrozenInstanceError):
                description.fingerprint = "changed"
            changed = values()
            changed["recurrent"] = b"different"
            with self.assertRaises(ValueError):
                store.put(b"prefix", changed)
            with self.assertRaises(ValueError):
                m.BundleStore(outer, replace(description, fingerprint="other-weights"))
            outer.codec = "different-codec"
            with self.assertRaises(ValueError):
                store.exact(b"prefix")

    def test_exact_required_set_and_size_before_allocation(self):
        description = spec()
        for wrong in ({"attention": b"a"}, {**values(), "extra": b"x"},
                      {**values(), "recurrent": b"x" * 17},
                      {**values(), "recurrent": bytearray(b"matrix")}):
            with self.subTest(values=tuple(wrong)):
                with self.assertRaises(ValueError):
                    m.encode_bundle(description, wrong)
        with self.assertRaises(ValueError):
            m.encode_bundle(replace(description, max_payload_bytes=1), values())
        with self.assertRaises(ValueError):
            m.BundleSpec("weights", "codec", (description.components[0], description.components[0]))

    def test_corrupt_missing_overlapping_and_codec_mismatch_rejected_before_allocate(self):
        encoded = m.encode_bundle(spec(), values())
        mutations = [
            lambda x: x["components"].pop(),
            lambda x: x["components"][1].update(offset=0),
            lambda x: x["components"][0].update(nbytes=-1),
            lambda x: x["components"][0].update(nbytes=True),
            lambda x: x["components"][0].update(nbytes=1 << 50),
            lambda x: x["components"][0].update(codec="wrong-codec"),
            lambda x: x["components"][0].update(name="unknown"),
            lambda x: x.update(fingerprint="different-model"),
        ]
        for change in mutations:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    m.decode_bundle(spec(), mutate(encoded, change))
        for wrong in (encoded[:-1], encoded + b"trailer", encoded[:-1] + bytes([encoded[-1] ^ 1]),
                      b"BADMAGIC" + encoded[8:], struct.pack("<8sI", b"CRITBND1", 65537)):
            with self.assertRaises(ValueError):
                m.decode_bundle(spec(), wrong)
        # Public plan constructors cannot bypass byte verification by inventing
        # descriptor lists that allocation trusts.
        forged = m.RestorePlan(spec(), b"prefix", encoded[:-1], ())
        with self.assertRaises(ValueError):
            m.RestoreTransaction(forged)

    def test_duplicate_manifest_field_is_rejected(self):
        data = b'{"format":"crit-components-v1","format":"crit-components-v1"}'
        with self.assertRaises(ValueError):
            m.decode_bundle(spec(), struct.pack("<8sI", b"CRITBND1", len(data)) + data)

    def test_publication_waits_for_every_component_and_read(self):
        transaction = m.RestoreTransaction(self.plan())
        receipts, reads = [], []

        def enqueue(operation):
            read = operation.track_work(Future())
            receipt = operation.record_copy(Future())
            receipts.append(receipt)
            reads.append(read)
            operation.destination.bytes[:] = operation.source
            return receipt

        transaction.submit(self.allocate, enqueue)
        for receipt in receipts:
            receipt.set_result(None)
        self.assertFalse(transaction.poll())
        with self.assertRaises(m.PendingRestore):
            transaction.publish()
        with self.assertRaises(m.PendingRestore):
            transaction.discard()
        for read in reads:
            read.set_result(None)
        restored = transaction.publish()
        self.assertEqual({name: bytes(owner.bytes) for name, owner in restored.destinations}, values())
        self.assertEqual(restored.prefix, b"semantic-prefix")
        self.assertNotIn(transaction, m.retained_restores())
        with self.assertRaises(ValueError):
            transaction.publish()

    def test_middle_copy_failure_never_publishes_and_retains_all_owners(self):
        transaction = m.RestoreTransaction(self.plan())
        calls, receipts = [], []
        retained_reader = Owner(1)

        def enqueue(operation):
            operation.retain(retained_reader)
            receipt = operation.record_copy(Future())
            calls.append(operation.component.name)
            receipts.append(receipt)
            if len(calls) == 2:
                # It may have submitted a copy before throwing. The registered
                # receipt and all later allocated owners must stay retained.
                raise RuntimeError("ambiguous middle-copy submission")
            return receipt

        with self.assertRaises(m.RestoreFailure) as failure:
            transaction.submit(self.allocate, enqueue)
        self.assertIs(failure.exception.transaction, transaction)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(op.destination is not None for op in transaction.operations))
        self.assertTrue(all(op.source.readonly for op in transaction.operations))
        self.assertTrue(any(retained_reader in op.owners for op in transaction.operations))
        with self.assertRaises(m.RestoreFailure):
            transaction.publish()
        receipts[0].set_result(None)
        with self.assertRaises(m.PendingRestore):
            transaction.discard()
        receipts[1].set_result(None)
        # A completed known receipt does not disprove unregistered work after
        # the throwing enqueue; an explicit backend settlement proof is needed.
        with self.assertRaises(m.PendingRestore):
            transaction.discard()
        transaction.discard(lambda operation: True)
        self.assertEqual(transaction.state, "discarded")

    def test_failed_receipt_and_pending_sibling_read_keep_bundle_private(self):
        transaction = m.RestoreTransaction(self.plan())
        receipts = []

        def enqueue(operation):
            receipt = operation.record_copy(Future())
            receipts.append(receipt)
            return receipt

        transaction.submit(self.allocate, enqueue)
        receipts[0].set_result(None)
        receipts[1].set_exception(RuntimeError("copy failed after physical settlement"))
        with self.assertRaises(m.RestoreFailure):
            transaction.publish()
        self.assertEqual(transaction.state, "failed")
        with self.assertRaises(m.PendingRestore):
            transaction.discard()
        receipts[2].set_result(None)
        transaction.discard()

    def test_unknown_submission_is_retained_even_if_caller_drops_transaction(self):
        receipts, references = [], []

        def create_and_forget():
            transaction = m.RestoreTransaction(self.plan())

            def allocate(operation):
                owner = Owner(operation.component.nbytes)
                references.append(weakref.ref(owner))
                return owner

            def enqueue(operation):
                read = operation.track_work(Future())
                receipts.append(read)
                raise RuntimeError("lost copy receipt")

            try:
                transaction.submit(allocate, enqueue)
            except m.RestoreFailure:
                pass

        create_and_forget()
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))
        transaction = next(t for t in m.retained_restores() if t not in self.restores)
        receipts[0].set_result(None)
        with self.assertRaises(m.PendingRestore):
            transaction.discard(lambda operation: False)
        transaction.discard(lambda operation: True)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_allocator_partial_owner_and_uncertain_settlement_are_retained(self):
        transaction = m.RestoreTransaction(self.plan())
        owner = Owner(4)

        def allocate(operation):
            operation.retain(owner)
            raise RuntimeError("partial allocator failure")

        with self.assertRaises(m.RestoreFailure):
            transaction.submit(allocate, self.copy)
        self.assertIn(owner, transaction.operations[0].owners)
        with self.assertRaises(m.PendingRestore):
            transaction.discard(lambda operation: "not a boolean proof")
        with self.assertRaises(m.PendingRestore):
            transaction.discard(lambda operation: (_ for _ in ()).throw(RuntimeError("fence failed")))
        transaction.discard(lambda operation: True)

    def test_identical_destination_owner_is_rejected_before_copy(self):
        transaction = m.RestoreTransaction(self.plan())
        shared = Owner(32)
        copies = []
        with self.assertRaises(m.RestoreFailure):
            transaction.submit(lambda operation: shared, lambda operation: copies.append(operation))
        self.assertEqual(copies, [])
        transaction.discard(lambda operation: True)

    def test_reentrant_backend_cannot_publish_during_submission(self):
        transaction = m.RestoreTransaction(self.plan())

        def enqueue(operation):
            with self.assertRaises(RuntimeError):
                transaction.publish()
            return self.copy(operation)

        transaction.submit(self.allocate, enqueue)
        transaction.publish()

    def test_split_fragments_share_exact_original_physical_write_lease(self):
        first, second = Future(), Future()
        lease = m.ActiveWriteLease(b"abcdefghijkl", first, max_bytes=12)
        another = m.ActiveWriteLease(b"ghijkl", second, max_bytes=6)
        prefix, tail = lease.fragment().split(4)
        middle, suffix = tail.split(8)
        self.assertIs(prefix.lease, suffix.lease)
        with self.assertRaises(AttributeError):
            lease.receipt = second
        self.assertEqual([bytes(f.source()) for f in (prefix, middle, suffix)], [b"abcd", b"efgh", b"ijkl"])
        with self.assertRaises(m.PendingRestore):
            lease.retire()
        second.set_result(None)
        another.retire()
        self.assertIsNotNone(lease.source)
        self.assertIn(lease, m.retained_writes())
        first.set_result(None)
        lease.retire()
        for fragment in (prefix, middle, suffix):
            with self.assertRaises(ValueError):
                fragment.source()
        with self.assertRaises(ValueError):
            lease.retire()

    def test_failed_settled_write_releases_but_unknown_fence_does_not(self):
        receipt = Future()
        lease = m.ActiveWriteLease(b"source", receipt)
        receipt.set_exception(RuntimeError("I/O failure, settled"))
        with self.assertRaises(RuntimeError):
            lease.retire()
        self.assertNotIn(lease, m.retained_writes())

        class Unknown:
            def done(self):
                raise RuntimeError("lost completion proof")

            def result(self):
                return None

        unknown = m.ActiveWriteLease(b"source", Unknown())
        with self.assertRaises(RuntimeError):
            unknown.retire()
        self.assertIn(unknown, m.retained_writes())
        # No actual I/O exists in this inert host fixture; an explicit proof
        # retires it while preserving the original receipt's immutable identity.
        with self.assertRaises(m.PendingRestore):
            unknown.retire(lambda original: False)
        unknown.retire(lambda original: True)


if __name__ == "__main__":
    unittest.main()
