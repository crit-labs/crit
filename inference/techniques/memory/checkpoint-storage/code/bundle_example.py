"""Run with python -B bundle_example.py; host ownership and byte-copy POC."""
from concurrent.futures import Future
from pathlib import Path
import tempfile

from implementation import CheckpointStore
from bundles import BundleSpec, BundleStore, ComponentSpec, RestoreTransaction
from transfers import BoundedTransfer, TransferLimits, TransferPart, read_file_window


def main():
    spec = BundleSpec("verified-weights/state-layout-v1", "opaque-bundle-v1", (
        ComponentSpec("attention", "attention-bytes-v1", 16),
        ComponentSpec("recurrent", "recurrent-bytes-v1", 16),
    ), max_payload_bytes=32)
    with tempfile.TemporaryDirectory() as directory:
        store = BundleStore(CheckpointStore(Path(directory), spec.fingerprint, spec.codec, 8192), spec)
        store.put(b"prefix", {"attention": b"keys", "recurrent": b"matrix"})
        transaction = RestoreTransaction(store.exact(b"prefix"))

        def copy(operation):
            receipt = operation.record_copy(Future())
            operation.destination[:] = operation.source
            receipt.set_result(None)
            return receipt

        transaction.submit(lambda operation: bytearray(operation.component.nbytes), copy)
        restored = transaction.publish()
        assert dict(restored.destinations) == {"attention": b"keys", "recurrent": b"matrix"}
        assert restored.prefix == b"prefix"

        # Pace file -> staging -> private destination copies through the same
        # restore transaction. The validated plan still owns its encoded payload;
        # bounded staging does not make that existing in-memory codec streaming.
        plan = store.exact(b"prefix")
        snapshot = Path(directory) / "leased-bundle"
        snapshot.write_bytes(plan.payload)
        components = {part.name: part for part in plan.components}
        destinations = {}
        with snapshot.open("rb", buffering=0) as source:
            def read(window):
                return read_file_window(window, source, base=components[window.part.name].offset)

            def write(window):
                receipt = window.record(Future())
                destination = destinations[window.part.name]
                destination[window.offset:window.offset + window.nbytes] = window.buffer
                receipt.set_result(window.nbytes)
                return receipt

            transfer = BoundedTransfer(
                tuple(TransferPart(c.name, c.nbytes, c.sha256) for c in plan.components),
                TransferLimits(window_bytes=3, staging_bytes=6, windows=2, total_bytes=32),
                read, write, owners=(source, destinations))
            transaction = RestoreTransaction(plan)

            def allocate(operation):
                destination = bytearray(operation.component.nbytes)
                destinations[operation.component.name] = destination
                return destination

            # One batch receipt makes every component wait for the complete
            # transfer, including all staging readers and component digests.
            try:
                transaction.submit(allocate, lambda operation: operation.record_copy(transfer))
                while not transfer.done():
                    transfer.pump(max_bytes=6, max_submissions=2)
                restored = transaction.publish()
                assert dict(restored.destinations) == {"attention": b"keys", "recurrent": b"matrix"}
            finally:
                # These callbacks are entirely synchronous: after they return
                # or throw, no physical reader remains. That is the proof below.
                # A native adapter must instead prove its actual backend drained.
                transfer.discard(lambda window: True)
                if transaction.state != "published":
                    transaction.discard(lambda operation: True)


if __name__ == "__main__":
    main()
