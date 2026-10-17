"""Adversarial synthetic header/index/schema fixtures; no model payload loading."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "crit_test_model_import_audit", ROOT / "inference/model-import-audit/code/implementation.py")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class ImportAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def raw(self, header, payload=b"", name="model.safetensors", declared_length=None):
        data = header if isinstance(header, bytes) else json.dumps(header).encode("utf-8")
        path = self.root / name
        length = len(data) if declared_length is None else declared_length
        path.write_bytes(struct.pack("<Q", length) + data + payload)
        return path

    def tensors(self, specs, name="model.safetensors", metadata=None):
        sizes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4,
                 "I16": 2, "I64": 8, "U8": 1, "U16": 2, "U32": 4, "U64": 8,
                 "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "BOOL": 1}
        header, payload = {}, bytearray()
        if metadata is not None:
            header["__metadata__"] = metadata
        for key, dtype, shape in specs:
            count = 1
            for dim in shape:
                count *= dim
            length = count * sizes[dtype]
            begin = len(payload)
            payload.extend(bytes(length))
            header[key] = {"dtype": dtype, "shape": shape,
                           "data_offsets": [begin, len(payload)]}
        return self.raw(header, payload, name)

    def index(self, weights, metadata=None, extra=None):
        value = {"weight_map": weights}
        if metadata is not None:
            value["metadata"] = metadata
        value.update(extra or {})
        path = self.root / "model.safetensors.index.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def basic(self, dtype="BF16"):
        path = self.tensors((
            ("embedding.weight", "BF16", (4, 8)),
            ("mtp.e_proj.weight", dtype, (8, 8)),
        ), metadata={"claimed_dtype": "BF16"})
        schema = m.ModelSchema(
            (m.TensorRole("embedding", "embedding.weight", (4, 8), "BF16"),
             m.TensorRole("mtp", "mtp.e_proj.weight", (8, 8), "BF16")),
            (m.CapabilityRequirement("mtp", ("embedding", "mtp")),))
        return m.inspect_model(path), schema

    def packed(self, *, full=256, begin=0, end=128, group=128,
               data_shape=(2, 16), scale_shape=(2, 1), scale_key="expert.up.weight_scale",
               dtype="I32", bits=4):
        path = self.tensors((("expert.up.weight_packed", dtype, data_shape),
                             (scale_key, "F32", scale_shape)))
        matrix = m.PackedMatrix("engine-codec-v1", "expert.up", "data", "scale",
                                2, full, group, bits, begin, end)
        schema = m.ModelSchema(
            (m.TensorRole("data", "expert.up.weight_packed", data_shape, dtype),
             m.TensorRole("scale", scale_key, scale_shape, "F32")),
            packed_matrices=(matrix,))
        return m.inspect_model(path), schema

    def test_actual_header_bindings_and_immutable_results(self):
        inventory, schema = self.basic()
        audit = m.audit_model(inventory, schema)
        self.assertEqual(audit.bindings["mtp"].dtype, "BF16")
        self.assertEqual(audit.capability_status, {"mtp": "metadata_ready"})
        with self.assertRaises(TypeError):
            audit.bindings["mtp"] = inventory.tensors["embedding.weight"]
        with self.assertRaises(TypeError):
            inventory.tensors["new"] = inventory.tensors["embedding.weight"]

    def test_actual_wrong_dtype_rejects_bf16_metadata_claim(self):
        for dtype in ("F16", "F8_E4M3", "F8_E5M2"):
            with self.subTest(actual_dtype=dtype):
                inventory, schema = self.basic(dtype)
                with self.assertRaisesRegex(m.AuditError, "header dtype/shape"):
                    m.audit_model(inventory, schema)

    def test_byte_addressable_fp8_and_unsigned_headers_use_actual_storage_sizes(self):
        for dtype, size in (("F8_E4M3", 1), ("F8_E5M2", 1),
                            ("U16", 2), ("U32", 4), ("U64", 8)):
            with self.subTest(dtype=dtype):
                path = self.tensors((("a", dtype, (2, 3)),))
                inventory = m.inspect_model(path)
                audit = m.audit_model(inventory, m.ModelSchema((
                    m.TensorRole("typed", "a", (2, 3), dtype),)))
                self.assertEqual(audit.bindings["typed"].nbytes, 6 * size)
                self.assertEqual(audit.bindings["typed"].dtype, dtype)

    def test_missing_and_misnamed_mtp_reject_before_probe(self):
        inventory, schema = self.basic()
        for key in ("mtp.missing.weight", "model.mtp.e_proj.weight"):
            with self.subTest(key=key), self.assertRaisesRegex(m.AuditError, "Missing required"):
                m.audit_model(inventory, replace(schema, roles=(
                    schema.roles[0], replace(schema.roles[1], key=key))))

    def test_wrong_shape_and_unexpected_tensor_policy(self):
        inventory, schema = self.basic()
        with self.assertRaises(m.AuditError):
            m.audit_model(inventory, replace(schema, roles=(replace(schema.roles[0], shape=(8, 4)),)))
        with self.assertRaisesRegex(m.AuditError, "Unexpected"):
            m.audit_model(inventory, m.ModelSchema((schema.roles[0],), reject_unexpected=True))

    def test_disabled_capability_is_explicit_and_cannot_be_probed(self):
        inventory, schema = self.basic()
        audit = m.audit_model(inventory, replace(schema, capabilities=(
            replace(schema.capabilities[0], requested=False),)))
        self.assertEqual(audit.capability_status["mtp"], "not_requested")
        with self.assertRaises(m.AuditError):
            m.probe_proposer(audit, "mtp", lambda p, n: (1,), (0,), vocab_size=4)

    def test_capability_and_schema_declaration_validation(self):
        inventory, schema = self.basic()
        invalid = (
            replace(schema, roles=schema.roles + (schema.roles[0],)),
            replace(schema, capabilities=(m.CapabilityRequirement("mtp", ("missing",)),)),
            replace(schema, capabilities=(m.CapabilityRequirement("mtp", ()),)),
            replace(schema, capabilities=(m.CapabilityRequirement("mtp", ("mtp", "mtp")),)),
            replace(schema, capabilities=schema.capabilities * 2),
            replace(schema, roles=(replace(schema.roles[0], shape=(True, 8)),)),
            replace(schema, roles=(replace(schema.roles[0], dtype="F4"),)),
            replace(schema, reject_unexpected="yes"),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(m.AuditError):
                m.audit_model(inventory, candidate)

    def test_probe_invokes_requested_toy_proposer_and_keeps_status_separate(self):
        inventory, schema = self.basic()
        audit = m.audit_model(inventory, schema)
        calls = []
        def propose(prompt, budget):
            calls.append((prompt, budget))
            return (3,)
        probe = m.probe_proposer(audit, "mtp", propose, (1, 2), vocab_size=4)
        self.assertEqual(calls, [((1, 2), 1)])
        self.assertEqual((probe.output, probe.status), ((3,), "invoked_with_proposals"))
        self.assertEqual(audit.capability_status["mtp"], "metadata_ready")
        empty = m.probe_proposer(audit, "mtp", lambda p, n: (), (1,), vocab_size=4)
        self.assertEqual(empty.status, "invoked_without_proposals")

    def test_probe_rejects_invalid_inputs_outputs_and_propagates_hook_failure(self):
        inventory, schema = self.basic()
        audit = m.audit_model(inventory, schema)
        for output in ((4,), (-1,), (True,), (1, 2), iter((1,)), "1"):
            with self.subTest(output=output), self.assertRaises(m.AuditError):
                m.probe_proposer(audit, "mtp", lambda p, n: output, (1,), vocab_size=4)
        for prompt in ((4,), (True,), "1", (0,) * 4097):
            with self.subTest(prompt_type=type(prompt)), self.assertRaises(m.AuditError):
                m.probe_proposer(audit, "mtp", lambda p, n: (1,), prompt, vocab_size=4)
        for vocab, bound in ((0, 1), (4, 0), (True, 1), (4, 65)):
            with self.assertRaises(m.AuditError):
                m.probe_proposer(audit, "mtp", lambda p, n: (1,), (1,),
                                 vocab_size=vocab, max_proposals=bound)
        def broken(prompt, budget):
            raise RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            m.probe_proposer(audit, "mtp", broken, (1,), vocab_size=4)

    def test_multishard_index_reads_actual_header_names_and_total_size(self):
        self.tensors((("a", "BF16", (2,)),), "first.safetensors")
        self.tensors((("b", "I32", (3,)),), "second.safetensors")
        path = self.index({"a": "first.safetensors", "b": "second.safetensors"},
                          {"total_size": 16})
        inventory = m.inspect_model(path)
        self.assertEqual(set(inventory.tensors), {"a", "b"})
        self.assertEqual(len(inventory.shards), 2)
        self.assertEqual(inventory.tensors["b"].dtype, "I32")
        self.assertIsNotNone(inventory.index_sha256)

    def test_index_missing_undeclared_duplicate_and_wrong_shard_tensors(self):
        self.tensors((("a", "BF16", (2,)),), "first.safetensors")
        self.tensors((("b", "I32", (3,)),), "second.safetensors")
        maps = (
            {"a": "first.safetensors", "missing": "first.safetensors"},
            {"a": "second.safetensors", "b": "first.safetensors"},
            {"b": "first.safetensors"},
        )
        for weights in maps:
            with self.subTest(weights=weights), self.assertRaises(m.AuditError):
                m.inspect_model(self.index(weights))
        self.tensors((("a", "I32", (1,)),), "second.safetensors")
        with self.assertRaises(m.AuditError):
            m.inspect_model(self.index({"a": "first.safetensors", "b": "second.safetensors"}))

    def test_index_path_escape_and_portability_rejection(self):
        for name in ("../outside.safetensors", "/tmp/a.safetensors", "C:\\a.safetensors",
                     "sub/../a.safetensors", "a\\b.safetensors", "file:stream.safetensors",
                     "CON.safetensors", "a.gguf"):
            with self.subTest(name=name), self.assertRaises(m.AuditError):
                m.inspect_model(self.index({"a": name}))

    def test_static_symlink_rejection(self):
        outside = self.tensors((("a", "U8", (1,)),), "actual.safetensors")
        link = self.root / "link.safetensors"
        try:
            link.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Host does not permit symlink creation: {exc}")
        with self.assertRaises(m.AuditError):
            m.inspect_model(self.index({"a": "link.safetensors"}))

    def test_index_metadata_and_bounds(self):
        self.tensors((("a", "BF16", (2,)),), "first.safetensors")
        self.tensors((("b", "BF16", (2,)),), "second.safetensors")
        for metadata in ({"total_size": 99}, {"total_size": True}, {"total_size": None},
                         {"unknown": 1}, [], {"total_size": -1}):
            with self.subTest(metadata=metadata), self.assertRaises(m.AuditError):
                m.inspect_model(self.index({"a": "first.safetensors"}, metadata))
        for weights in ({}, [], {"a": None}):
            with self.assertRaises(m.AuditError):
                m.inspect_model(self.index(weights))
        with self.assertRaises(m.AuditError):
            m.inspect_model(self.index({"a": "first.safetensors"}, extra={"config": {}}))
        path = self.index({"a": "first.safetensors", "b": "second.safetensors"})
        for limits in (m.Limits(max_shards=1), m.Limits(max_tensors=1),
                       m.Limits(max_index_bytes=8), m.Limits(max_total_bytes=8)):
            with self.assertRaises(m.AuditError):
                m.inspect_model(path, limits)

    def test_duplicate_json_fields_rejected_at_every_level(self):
        headers = (
            b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}',
            b'{"a":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}',
            b'{"__metadata__":{"claim":"a","claim":"b"}}',
        )
        for header in headers:
            with self.subTest(header=header), self.assertRaises(m.AuditError):
                m.inspect_model(self.raw(header, b"\0\0"))
        path = self.root / "model.safetensors.index.json"
        path.write_text('{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}')
        with self.assertRaises(m.AuditError):
            m.inspect_model(path)

    def test_empty_scalar_and_out_of_order_records_are_supported(self):
        path = self.raw({
            "later": {"dtype": "BF16", "shape": [], "data_offsets": [1, 3]},
            "zero": {"dtype": "F32", "shape": [0, 100], "data_offsets": [1, 1]},
            "first": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        }, b"\0\0\0")
        inventory = m.inspect_model(path)
        self.assertEqual(inventory.tensors["later"].nbytes, 2)
        self.assertEqual(inventory.tensors["zero"].nbytes, 0)

    def test_truncated_huge_or_nonobject_headers_and_unsupported_containers(self):
        path = self.root / "model.safetensors"
        path.write_bytes(b"tiny")
        with self.assertRaises(m.AuditError):
            m.inspect_model(path)
        for header, length in ((b"{}", 0), (b"{}", 100), (b"{}", (1 << 64) - 1),
                               (b"[]", None), (b" {}", None), (b"{bad", None),
                               (b"{\xff}", None)):
            with self.subTest(header=header, length=length), self.assertRaises(m.AuditError):
                m.inspect_model(self.raw(header, declared_length=length))
        with self.assertRaises(m.AuditError):
            m.inspect_model(self.raw({}, name="model.gguf"))

    def test_nonfinite_float_boolean_negative_and_overflow_dimensions_rejected(self):
        for value in (True, -1, 1.0, float("nan"), float("inf"), 1 << 64):
            with self.subTest(value=value), self.assertRaises(m.AuditError):
                m.inspect_model(self.raw({"a": {"dtype": "U8", "shape": [value],
                                                 "data_offsets": [0, 0]}}))
        for shape in ((1 << 30, 1 << 30), (1,) * 9):
            with self.assertRaises(m.AuditError):
                m.inspect_model(self.raw({"a": {"dtype": "F64", "shape": shape,
                                                 "data_offsets": [0, 0]}}))

    def test_bad_offsets_overlap_holes_tail_and_storage_mismatch(self):
        record = {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}
        for offsets in ([True, 2], [-1, 1], [0, 3], [0, (1 << 64) - 1], [2, 0], [0], "02"):
            with self.subTest(offsets=offsets), self.assertRaises(m.AuditError):
                m.inspect_model(self.raw({"a": dict(record, data_offsets=offsets)}, b"\0\0"))
        for header, payload in (({"a": record, "b": record}, b"\0\0"),
                                ({"a": dict(record, data_offsets=[1, 3])}, b"\0\0\0"),
                                ({"a": record}, b"\0\0\0"),
                                ({"a": dict(record, dtype="F32")}, b"\0\0"),
                                ({"a": record}, b"\0")):
            with self.subTest(header=header), self.assertRaises(m.AuditError):
                m.inspect_model(self.raw(header, payload))

    def test_malformed_tensor_metadata_and_unsupported_dtype_rejected(self):
        for dtype in ("F4", "F6", "F8_UNKNOWN", [], "bad"):
            with self.subTest(dtype=dtype), self.assertRaises(m.AuditError):
                m.inspect_model(self.raw({"a": {"dtype": dtype, "shape": [1],
                                                 "data_offsets": [0, 1]}}, b"\0"))
        for metadata in ([], {"claim": True}, {"claim": {"dtype": "BF16"}},
                         {"claim": "\ud800"}):
            with self.subTest(metadata=repr(metadata)), self.assertRaises(m.AuditError):
                m.inspect_model(self.raw({"__metadata__": metadata}))
        for record in ([], {"dtype": "U8", "shape": [1]},
                       {"dtype": "U8", "shape": [1], "data_offsets": [0, 1], "other": 0}):
            with self.assertRaises(m.AuditError):
                m.inspect_model(self.raw({"a": record}, b"\0"))

    def test_limits_bound_files_names_rank_and_header_before_payload_read(self):
        path = self.tensors((("a", "BF16", (2,)),))
        for limits in (m.Limits(max_header_bytes=8), m.Limits(max_file_bytes=8),
                       m.Limits(max_dimension=1), m.Limits(max_total_bytes=8)):
            with self.assertRaises(m.AuditError):
                m.inspect_model(path, limits)
        path = self.tensors((("a", "U8", (1,)), ("b", "U8", (1,))))
        with self.assertRaises(m.AuditError):
            m.inspect_model(path, m.Limits(max_tensors=1))
        path = self.tensors((("long", "U8", (1,)),))
        with self.assertRaises(m.AuditError):
            m.inspect_model(path, m.Limits(max_name_bytes=3))
        for kwargs in ({"max_rank": 0}, {"max_shards": True}, {"max_file_bytes": 1 << 64}):
            with self.assertRaises(m.AuditError):
                m.Limits(**kwargs)

    def test_reader_is_cores_strict_reader_under_the_audit_policy(self):
        # Header-only reads and change detection are covered by Core's reader tests (test_core_model.py).
        path = self.tensors((("a", "BF16", (2,)),))
        inventory = m.inspect_model(path)
        self.assertIsInstance(inventory, m._safetensors.Checkpoint)
        self.assertEqual(inventory.index_policy, m._safetensors.IndexPolicy.STRICT)
        with self.assertRaises(m.AuditError) as caught:
            m.inspect_model(path, m.Limits(max_header_bytes=8))
        self.assertIsInstance(caught.exception.__cause__, m._safetensors.SafetensorsError)
        self.assertIsInstance(m.Limits(), m._safetensors.Limits)

    def test_aligned_packed_k_slices_and_final_group_word_tails(self):
        for options in ({}, {"begin": 128, "end": 256},
                        {"full": 135, "begin": 128, "end": 135,
                         "data_shape": (2, 1), "scale_shape": (2, 1)},
                        {"full": 17, "begin": 0, "end": 17, "group": 8,
                         "data_shape": (2, 3), "scale_shape": (2, 3)},
                        {"full": 9, "begin": 0, "end": 9, "group": 8,
                         "dtype": "U8", "bits": 8, "data_shape": (2, 9), "scale_shape": (2, 2)}):
            with self.subTest(options=options):
                inventory, schema = self.packed(**options)
                audit = m.audit_model(inventory, schema)
                self.assertEqual(len(audit.codecs), 1)

    def test_unsharded_scales_wrong_word_count_and_namespace_rejected(self):
        for options in ({"scale_shape": (2, 2)}, {"data_shape": (2, 32)},
                        {"scale_key": "expert.other.weight_scale"}):
            with self.subTest(options=options), self.assertRaises(m.AuditError):
                inventory, schema = self.packed(**options)
                m.audit_model(inventory, schema)

    def test_interior_partial_groups_words_and_out_of_range_slices_rejected(self):
        invalid = ({"begin": 1}, {"end": 127}, {"begin": 128, "end": 129},
                   {"full": 16, "group": 4, "begin": 4, "end": 8,
                    "data_shape": (2, 1), "scale_shape": (2, 1)},
                   {"end": 257}, {"begin": 128, "end": 128}, {"group": 0},
                   {"bits": 3}, {"full": True})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(m.AuditError):
                inventory, schema = self.packed(**options)
                m.audit_model(inventory, schema)


if __name__ == "__main__":
    unittest.main()
