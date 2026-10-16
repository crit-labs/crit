"""Runnable metadata/capability POC with tiny synthetic local tensor files."""
import json
from pathlib import Path
import tempfile

from implementation import (
    AuditError, CapabilityRequirement, ModelSchema, PackedMatrix, TensorRole,
    audit_model, inspect_model, probe_proposer,
)


def run():
    # All payloads are zeros. This exercises metadata, never trained arithmetic.
    specs = (
        ("embedding.weight", "BF16", [4, 8], 64),
        ("mtp.e_proj.weight", "BF16", [8, 8], 128),
        ("expert.up.weight_packed", "I32", [2, 1], 8),
        ("expert.up.weight_scale", "F32", [2, 1], 8),
    )
    header, offset = {}, 0
    for name, dtype, shape, size in specs:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy.safetensors"
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(offset))
        inventory = inspect_model(path)
        schema = ModelSchema(
            roles=(TensorRole("embedding", "embedding.weight", (4, 8), "BF16"),
                   TensorRole("mtp_input", "mtp.e_proj.weight", (8, 8), "BF16"),
                   TensorRole("packed", "expert.up.weight_packed", (2, 1), "I32"),
                   TensorRole("scales", "expert.up.weight_scale", (2, 1), "F32")),
            capabilities=(CapabilityRequirement("mtp", ("embedding", "mtp_input")),),
            packed_matrices=(PackedMatrix("toy-signed-i4-lsb-g8-v1", "expert.up",
                                          "packed", "scales", 2, 16, 8, k_end=8),),
            reject_unexpected=True,
        )
        audit = audit_model(inventory, schema)
        calls = []

        def toy_proposer(prompt, budget):
            calls.append((prompt, budget))
            return ((prompt[-1] + 1) % 4,)

        probe = probe_proposer(audit, "mtp", toy_proposer, (1, 2), vocab_size=4)
        assert audit.capability_status["mtp"] == "metadata_ready"
        assert calls == [((1, 2), 1)] and probe.output == (3,)
        try:
            audit_model(inventory, ModelSchema((
                TensorRole("mtp_input", "mtp.e_proj.weight", (8, 8), "F16"),)))
        except AuditError:
            pass
        else:
            raise AssertionError("The actual BF16 header was accepted as F16")
        print("Header roles and local packed-scale geometry passed; toy proposer invoked once")
        print("Metadata readiness and probe invocation do not qualify model inference")


if __name__ == "__main__":
    run()
