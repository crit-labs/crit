"""Fresh generic host contract fixtures; no native/device/model execution."""
from concurrent.futures import Future
from dataclasses import replace
import hashlib
import json
import math
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


class DescriptionExamples(unittest.TestCase):
    def test_worked_examples_run_against_the_supplied_code(self):
        store = Path(__file__).resolve().parents[1] / "inference"
        for description in sorted(store.glob("*/README.md")):
            with self.subTest(technique=description.parent.name):
                blocks = re.findall(r"^```python\n(.*?)^```$",
                                    description.read_text(encoding="utf-8"), re.M | re.S)
                self.assertEqual(len(blocks), 1, "one runnable mechanism example per entry")
                result = subprocess.run([sys.executable, "-B", "-c", blocks[0]],
                                        cwd=description.parent / "code", capture_output=True,
                                        text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

ROOT = Path(__file__).resolve().parents[1]


def load(ident, name="implementation.py"):
    module_name = "crit_test_" + ident.replace("-", "_") + "_" + name.replace(".", "_")
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "inference" / ident / "code" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


from e4m3_vectors import e4m3_magnitude, e4m3_scale_vectors  # noqa: E402


class Receipt:
    def __init__(self, done=False):
        self.finished = done

    def done(self):
        return self.finished


class ExpertExecutionTests(unittest.TestCase):
    """Adversarial metadata/ownership fixtures; Future settlement is simulated."""
    def bundle(self, capacity=1, components=None):
        m = load("expert-residency")
        if components is None:
            components = (m.BundleComponent("expand", "packed-a", 3),
                          m.BundleComponent("contract", "packed-b", 5))
        return m, m.BundleDirectory(capacity, components, (object(),))

    def key(self, expert=1):
        return load("expert-residency").ExpertKey("weights-r1", 0, expert, "bundle-layout-r1")

    def enqueue(self, directory, ticket, receipt=None):
        m = load("expert-residency")
        receipt = Future() if receipt is None else receipt
        for component in directory.components:
            directory.submit(ticket, component, object(), lambda t, c, o:
                m.BundleReceipt(t, c.name, c.encoding, c.nbytes, receipt))
        return receipt

    def populate(self, directory, expert=1):
        ticket = directory.begin_upload(self.key(expert))
        receipt = self.enqueue(directory, ticket)
        receipt.set_result(None)
        directory.publish(ticket)
        return ticket

    def test_bundle_missing_or_pending_component_is_never_visible(self):
        m, directory = self.bundle()
        ticket = directory.begin_upload(self.key())
        first = Future()
        directory.submit(ticket, "expand", object(), lambda t, c, o:
            m.BundleReceipt(t, c.name, c.encoding, c.nbytes, first))
        first.set_result(None)
        with self.assertRaises(ValueError):
            directory.publish(ticket)
        self.assertEqual(directory.ready_slots(), {})
        self.assertIsNone(directory.acquire(self.key()))
        second = Future()
        directory.submit(ticket, "contract", object(), lambda t, c, o:
            m.BundleReceipt(t, c.name, c.encoding, c.nbytes, second))
        with self.assertRaises(ValueError):
            directory.publish(ticket)
        second.set_result(None)
        directory.publish(ticket)
        self.assertEqual(directory.ready_slots()[self.key()], (ticket.slot, ticket.generation))
        self.assertEqual(directory.slots[ticket.slot].copies, {})

    def test_bundle_overwrite_invalidates_old_content_and_failed_copy_stays_empty(self):
        m, directory = self.bundle()
        old = self.populate(directory)
        ticket = directory.begin_upload(self.key(2))
        self.assertIsNone(directory.acquire(self.key()))
        good, bad = Future(), Future()
        for name, future in (("expand", good), ("contract", bad)):
            directory.submit(ticket, name, object(), lambda t, c, o, f=future:
                m.BundleReceipt(t, c.name, c.encoding, c.nbytes, f))
        bad.set_exception(RuntimeError("copy failed"))
        with self.assertRaises(ValueError):
            directory.publish(ticket)
        self.assertEqual(len(directory.slots[0].copies), 2)
        good.set_result(None)
        with self.assertRaises(RuntimeError):
            directory.publish(ticket)
        self.assertEqual(directory.ready_slots(), {})
        self.assertGreater(ticket.generation, old.generation)
        with self.assertRaises(ValueError):
            directory.publish(old)

    def test_bundle_leases_protect_all_route_hits_before_miss_selection(self):
        m, directory = self.bundle(capacity=2)
        self.populate(directory, 1)
        self.populate(directory, 2)
        first, second = directory.acquire(self.key(1)), directory.acquire(self.key(2))
        with self.assertRaises(ValueError):
            directory.begin_upload(self.key(3))
        read = Future()
        with self.assertRaises(ValueError):
            directory.settle(first, read)
        with self.assertRaises(ValueError):
            directory.close()
        read.set_result(None)
        directory.settle(first, read)
        ticket = directory.begin_upload(self.key(3))
        self.assertEqual(ticket.slot, first.slot)
        self.assertEqual(directory.ready_slots()[self.key(2)][0], second.slot)
        with self.assertRaises(ValueError):
            directory.settle(first, read)
        with self.assertRaises(ValueError):
            directory.settle(replace(second), read)

    def test_bundle_rejects_duplicate_unknown_and_foreign_submission_before_enqueue(self):
        m, directory = self.bundle()
        _, other = self.bundle()
        ticket, foreign = directory.begin_upload(self.key()), other.begin_upload(self.key())
        calls = []
        def enqueue(t, c, o):
            calls.append(c.name)
            return m.BundleReceipt(t, c.name, c.encoding, c.nbytes, Future())
        for bad_ticket, component in ((foreign, "expand"), (ticket, "missing")):
            with self.assertRaises(ValueError):
                directory.submit(bad_ticket, component, object(), enqueue)
        self.assertEqual(calls, [])
        directory.submit(ticket, "expand", object(), enqueue)
        with self.assertRaises(ValueError):
            directory.submit(ticket, "expand", object(), enqueue)
        self.assertEqual(calls, ["expand"])
        with self.assertRaises(ValueError):
            directory.begin_upload(self.key())

    def test_bundle_failed_enqueue_keeps_sources_until_explicit_quiescence(self):
        _, directory = self.bundle()
        ticket, source = directory.begin_upload(self.key()), object()
        def partial(*_):
            raise RuntimeError("work submitted without a receipt")
        with self.assertRaises(RuntimeError):
            directory.submit(ticket, "expand", source, partial)
        self.assertIs(directory.slots[0].copies["expand"].source_owner, source)
        for quiesce in (None, lambda owners, receipts: False):
            with self.assertRaises(ValueError):
                directory.abort(ticket, quiesce)
        self.assertTrue(directory.backing_owners)
        observed = []
        directory.abort(ticket, lambda owners, receipts: observed.append((owners, receipts)) or True)
        self.assertIn(source, observed[0][0])
        self.assertEqual(directory.ready_slots(), {})
        directory.close()

    def test_bundle_prior_event_receipt_and_wrong_representation_cannot_publish(self):
        m = load("expert-residency")
        for fault in ("old_ticket", "encoding", "extent"):
            _, directory = self.bundle()
            previous = self.populate(directory)
            ticket = directory.begin_upload(self.key(2))
            settled = Future()
            settled.set_result(None)
            def wrong(t, c, o):
                receipt = m.BundleReceipt(t, c.name, c.encoding, c.nbytes, settled)
                return replace(receipt, **{"old_ticket": {"ticket": previous},
                    "encoding": {"encoding": "different-layout"}, "extent": {"nbytes": c.nbytes + 1}}[fault])
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                directory.submit(ticket, "expand", object(), wrong)
            with self.assertRaises(ValueError):
                directory.publish(ticket)
            with self.assertRaises(ValueError):
                directory.abort(ticket)
            directory.abort(ticket, lambda owners, receipts: True)

    def test_bundle_incomplete_abort_waits_for_known_copies(self):
        m, directory = self.bundle()
        ticket, future = directory.begin_upload(self.key()), Future()
        directory.submit(ticket, "expand", object(), lambda t, c, o:
            m.BundleReceipt(t, c.name, c.encoding, c.nbytes, future))
        with self.assertRaises(ValueError):
            directory.abort(ticket, lambda owners, receipts: True)
        future.set_result(None)
        directory.abort(ticket)
        with self.assertRaises(ValueError):
            directory.abort(ticket)

    def test_bundle_components_are_explicit_and_immutable(self):
        m = load("expert-residency")
        components = (m.BundleComponent("fused", "packed-fused", 7),
                      m.BundleComponent("bias", "float32", 4))
        _, directory = self.bundle(components=components)
        self.populate(directory)
        with self.assertRaises(TypeError):
            directory.components["extra"] = components[0]
        for bad in ((), components + (components[0],), (replace(components[0], nbytes=True),)):
            with self.assertRaises(ValueError):
                self.bundle(components=bad)

    def test_mixed_sources_keep_occurrences_and_exact_fallback_when_scratch_is_full(self):
        m = load("heterogeneous-expert-dispatch")
        routes = (m.Route(7, 1, .5), m.Route(3, 2, .25), m.Route(9, 2, .75), m.Route(4, 3, 1))
        plan = m.mixed_source_plan(routes, {1: m.ReadySlot(0, 8)}, 2, (1,))
        self.assertEqual([r.route.position for r in plan.routes], [7, 3, 9, 4])
        self.assertEqual([r.source for r in plan.routes], ["cache", "scratch", "scratch", "stock"])
        self.assertEqual(plan.demands, ((2, 1),))
        self.assertEqual(plan.routes[-1].index, 3)
        self.assertEqual(plan.routes[0].generation, 8)
        stock = m.mixed_source_plan(routes, {}, 0)
        self.assertEqual([r.index for r in stock.routes], [1, 2, 2, 3])

    def test_mixed_sources_reject_aliases_invalid_routes_and_fabricated_demand(self):
        m = load("heterogeneous-expert-dispatch")
        routes = (m.Route(0, 1, 1), m.Route(1, 2, 1))
        invalid = (({1: m.ReadySlot(0, 1), 2: m.ReadySlot(0, 2)}, ()),
                   ({1: m.ReadySlot(0, 1)}, (0,)), ({}, (1, 1)),
                   ({1: m.ReadySlot(0, 0)}, ()))
        for ready, scratch in invalid:
            with self.assertRaises(ValueError):
                m.mixed_source_plan(routes, ready, 2, scratch)
        for bad in ((), (routes[0], routes[0]), (replace(routes[0], weight=float("nan")),)):
            with self.assertRaises(ValueError):
                m.mixed_source_plan(bad, {}, 0)
        plan = m.mixed_source_plan(routes, {}, 2, (0, 1))
        with self.assertRaises(ValueError):
            replace(plan, demands=((1, 1), (2, 0))).validate()
        with self.assertRaises(ValueError):
            replace(plan, routes=(plan.routes[0], replace(plan.routes[1], index=0))).validate()

    def test_mixed_projection_quantizes_and_submits_once_with_weighted_occurrences(self):
        m = load("heterogeneous-expert-dispatch")
        routes = (m.Route(0, 1, .5), m.Route(1, 2, 2), m.Route(2, 1, .25))
        plan = m.mixed_source_plan(routes, {1: m.ReadySlot(0, 1)}, 1)
        calls = []
        def quantize(rows):
            calls.append("quantize")
            return tuple(rows)
        def project(encoded, p):
            calls.append("project")
            cache, stock = {0: (2, 3)}, {2: (4, 5)}
            return tuple((r.route.position, r.route.weight * sum(a*b for a, b in
                zip(encoded, (cache if r.source == "cache" else stock)[r.index]))) for r in p.routes)
        self.assertEqual(m.execute_mixed(plan, (1, 2), quantize, project), ((0, 4), (1, 28), (2, 2)))
        self.assertEqual(calls, ["quantize", "project"])

    def test_graph_read_of_a_bundle_blocks_replacement_until_settled(self):
        _, directory = self.bundle()
        ticket = self.populate(directory)
        use = directory.acquire(self.key())
        read = Future()  # A captured graph's replay receipt over the cached bundle.
        with self.assertRaises(ValueError):
            directory.begin_upload(self.key(2))
        read.set_result(None)
        directory.settle(use, read)
        next_ticket = directory.begin_upload(self.key(2))
        self.assertGreater(next_ticket.generation, ticket.generation)

    def test_failed_file_upload_keeps_staging_after_read_quiescence(self):
        m = load("ssd-expert-materialization")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.bin"
            path.write_bytes(b"payload")
            source = m.FileExpertSource(path, {1: m.ExpertRegion(0, 7, "packed", hashlib.sha256(b"payload").hexdigest())})
            observed = []
            def upload(encoding, data):
                observed.append(data)
                raise RuntimeError("submitted copy then failed")
            with self.assertRaises(RuntimeError):
                source.materialize(1, upload)
            source.quiesce()  # drains reads, not upload consumers
            self.assertIs(source.failed_uploads[0][0], observed[0])
            self.assertIs(source.failed_uploads[0][1], upload)
            with self.assertRaises(ValueError):
                source.recover_uploads(lambda retained: False)
            self.assertEqual(len(source.failed_uploads), 1)
            source.recover_uploads(lambda retained: True)
            self.assertEqual(source.failed_uploads, [])

class TechniqueTests(unittest.TestCase):
    def test_static_residency_accounts_fixed_state_aliases_and_headroom(self):
        m = load("heterogeneous-expert-dispatch")
        groups = [m.ResidencyGroup("early", 7), m.ResidencyGroup("late", 5)]
        plans = m.prefix_residency_plans(groups, 12, 4, 3)
        self.assertEqual(plans[0].host_keys, ("early",))
        self.assertEqual(plans[0].device_keys, ("late",))
        self.assertEqual(plans[0].device_bytes + plans[0].headroom_bytes, 12)
        expanded = [m.ResidencyGroup("early", 7), m.ResidencyGroup("late", 6)]
        self.assertEqual(m.prefix_residency_plans(expanded, 12, 4, 3)[0].host_keys,
                         ("early", "late"))
        with self.assertRaises(ValueError):
            m.prefix_residency_plans(groups, 6, 4, 3)
        with self.assertRaises(ValueError):
            m.prefix_residency_plans(groups + [groups[0]], 20, 0)
        with self.assertRaises(ValueError):
            m.prefix_residency_plans(groups, True, 0)
        self.assertEqual(m.prefix_residency_plans([], 0, 0)[0].device_bytes, 0)

    def test_artifact_extent_and_identity(self):
        m = load("artifact-loading")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.bin"
            path.write_bytes(b"abcd")
            region = m.TensorRegion("weights", 1, 2, (2,), "bytes", hashlib.sha256(b"bc").hexdigest())
            reader = m.ArtifactReader(path, [region])
            self.assertEqual(reader.read("weights"), b"bc")
            path.write_bytes(b"aXXd")
            with self.assertRaises(ValueError):
                reader.read("weights")
            with self.assertRaises(ValueError):
                m.ArtifactReader(path, [replace(region, nbytes=5)])
            with self.assertRaises(ValueError):
                m.ArtifactReader(path, [region, replace(region, name="alias")])

    def test_artifact_async_copy_keeps_both_endpoints_and_failure_owner(self):
        m = load("artifact-loading")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.bin"
            path.write_bytes(b"abc")
            reader = m.ArtifactReader(path, [m.TensorRegion("x", 0, 3, (3,), "bytes", hashlib.sha256(b"abc").hexdigest())])
            owner, receipt = object(), Receipt()
            result = reader.materialize("x", lambda n: owner, lambda a, b: receipt, lambda pair: None)
            self.assertIs(result.owner, owner)
            self.assertEqual(result.source_bytes, b"abc")
            self.assertIs(result.readiness, receipt)
            failed = []
            def fail(*_):
                raise RuntimeError("submitted then failed")
            with self.assertRaises(RuntimeError):
                reader.materialize("x", lambda n: owner, fail, failed.append)
            self.assertEqual(failed, [(owner, b"abc")])

    def test_quantized_signed_nibbles_odd_extent_and_projection(self):
        m = load("quantized-projections")
        weights = m.quantize([[-7, 7, 0]], group=3, bits=4)
        self.assertEqual(weights.codes, b"\x79\x00")
        self.assertEqual(m.dequantize(weights), ((-7, 7, 0),))
        self.assertEqual(m.project(weights, (1, 2, 3)), (7,))
        self.assertEqual(m.project(weights, (1, 2, 3), lambda w, a: ("backend",)), ("backend",))
        with self.assertRaises(ValueError):
            m.project(weights, (1,))

    def test_quantized_zero_groups_and_invalid_representation(self):
        m = load("quantized-projections")
        weights = m.quantize([[0, 0]], group=2)
        self.assertEqual(m.dequantize(weights), ((0, 0),))
        with self.assertRaises(ValueError):
            m.quantize([[float("nan"), 0]], group=2)
        with self.assertRaises(ValueError):
            replace(weights, codes=b"").validate()
        with self.assertRaises(ValueError):
            m.quantize([[1, 2, 3]], group=2)

    def test_external_scales_must_already_be_positive_represented_fp32(self):
        m = load("quantized-projections")
        for scale in (1e100, 1e-100, float('nan'), float('inf'), 0, -1, True, '1', 0.1):
            with self.subTest(scale=scale):
                weights = m.QuantizedRows(1, 1, 1, 8, b'\x01', (scale,))
                calls = []
                with self.assertRaisesRegex(ValueError, 'FP32'):
                    m.project(weights, (1,), lambda *args: calls.append(args))
                self.assertFalse(calls)
        for scale in (2 ** -149, (2 - 2 ** -23) * 2 ** 127, m.represented_scale(0.1)):
            weights = m.QuantizedRows(1, 1, 1, 8, b'\x01', (scale,))
            self.assertEqual(m.dequantize(weights), ((scale,),))
        self.assertEqual(m.quantize([[0.7]], group=1, bits=4).scales,
                         (m.represented_scale(0.7 / 7),))

    def test_attention_accepts_only_committed_prefix(self):
        m = load("attention-state")
        state = m.KVState(3, 2, 1)
        state.stage((1, 0), (2,))
        state.stage((0, 1), (10,))
        self.assertEqual(state.attend((0, 0), include_pending=True), (6,))
        state.commit(1)
        self.assertEqual(state.attend((0, 0)), (2,))
        self.assertEqual(state.pending, [])
        state.stage((1, 1), (8,))
        state.rollback()
        self.assertEqual(len(state.committed), 1)

    def test_attention_capacity_geometry_and_invalid_commit(self):
        m = load("attention-state")
        state = m.KVState(1, 1, 1)
        with self.assertRaises(ValueError):
            state.attend((1,))
        state.stage((1,), (2,))
        with self.assertRaises(ValueError):
            state.stage((2,), (3,))
        with self.assertRaises(ValueError):
            state.commit(2)
        self.assertEqual(len(state.pending), 1)

    def test_prefill_cursor_commit_rollback_and_empty_prompt(self):
        m = load("session-scheduling")
        cursor = m.PrefillCursor((1, 2, 3), 2)
        self.assertEqual(cursor.next_chunk(8), (1, 2))
        with self.assertRaises(ValueError):
            cursor.next_chunk(1)
        cursor.rollback()
        self.assertEqual(cursor.next_chunk(1), (1,))
        cursor.commit()
        self.assertEqual(cursor.next_chunk(8), (2, 3))
        cursor.commit()
        self.assertTrue(cursor.complete)
        empty = m.PrefillCursor((), 1)
        self.assertEqual(empty.next_chunk(1), ())
        empty.commit()
        self.assertTrue(empty.complete)

    def test_prefill_tail_folding_joins_a_short_remainder_to_the_previous_chunk(self):
        m = load("session-scheduling")
        self.assertEqual(m.chunk_plan(5000, 2048), [2048, 2048, 904])
        self.assertEqual(m.chunk_plan(4100, 2048, 256), [2048, 2052])
        self.assertEqual(m.chunk_plan(2048, 2048, 256), [2048])
        self.assertEqual(m.chunk_plan(0, 8, 2), [])
        cursor = m.PrefillCursor(tuple(range(10)), chunk_limit=4, fold_limit=2)
        self.assertEqual(len(cursor.next_chunk(8)), 4)
        cursor.commit()
        self.assertEqual(len(cursor.next_chunk(5)), 4)  # the budget does not allow folding the last 2
        cursor.rollback()
        self.assertEqual(len(cursor.next_chunk(8)), 6)  # 4 + the 2-token remainder
        cursor.commit()
        self.assertTrue(cursor.complete)
        for bad in ((-1, 4, 0), (10, 0, 0), (10, 4, -1)):
            with self.assertRaises(ValueError):
                m.chunk_plan(*bad)

    def test_progress_staging_waits_copies_and_charges(self):
        m = load("session-scheduling")
        borrowed, charges = [], []
        class Ready:
            def wait(self):
                borrowed.extend((4, 5))
        staged, owner = m.stage_progress(borrowed, 2, lambda n: charges.append(n) or n, Ready())
        borrowed.clear()
        self.assertEqual(staged.tokens, (4, 5))
        self.assertEqual(charges, [8])
        self.assertEqual(owner, 8)
        with self.assertRaises(ValueError):
            m.stage_progress((1, 2), 1, lambda n: None)

    def test_expert_eviction_requires_unpinned_settled_owners(self):
        m = load("expert-residency")
        releases = []
        cache = m.ExpertCache(4, releases.append)
        a, b = m.ExpertKey("r1", 0, 0, "int4"), m.ExpertKey("r1", 0, 1, "int4")
        lease = cache.acquire(a, 4, lambda key, n: object())
        with self.assertRaises(ValueError):
            cache.acquire(b, 4, lambda key, n: object())
        receipt = Receipt()
        old_owner, old_generation = lease.owner, lease.generation
        lease.close(receipt)
        with self.assertRaises(ValueError):
            cache.acquire(b, 4, lambda key, n: object())
        receipt.finished = True
        new = cache.acquire(b, 4, lambda key, n: object())
        self.assertEqual(releases, [old_owner])
        self.assertGreater(new.generation, old_generation)
        self.assertEqual(cache.live, 4)

    def test_expert_identity_extent_and_release_failure(self):
        m = load("expert-residency")
        def fail(owner):
            raise RuntimeError("release not settled")
        cache = m.ExpertCache(4, fail)
        a = m.ExpertKey("r1", 0, 0, "int4")
        lease = cache.acquire(a, 4, lambda key, n: object())
        with self.assertRaises(ValueError):
            cache.acquire(a, 3, lambda key, n: object())
        lease.close()
        with self.assertRaises(RuntimeError):
            cache.acquire(replace(a, expert=1), 4, lambda key, n: object())
        self.assertIn(a, cache.entries)
        self.assertEqual(cache.live, 4)

    def test_route_occurrences_survive_host_device_partition(self):
        m = load("heterogeneous-expert-dispatch")
        routes = (m.Route(0, 7, .5), m.Route(1, 7, .5), m.Route(2, 8, 1))
        device, host = m.partition(routes, {7}, 1)
        self.assertEqual(tuple(r.position for r in device), (0,))
        self.assertEqual(tuple(r.position for r in host), (1, 2))
        self.assertEqual(m.combine(routes, [(2, (3,)), (0, (1,)), (1, (2,))]), (6,))
        with self.assertRaises(ValueError):
            m.combine(routes, [(0, (1,)), (0, (2,)), (2, (3,))])
        with self.assertRaises(ValueError):
            m.combine(routes, [(0, (1,))])

    def test_async_receipts_keep_sources_until_complete(self):
        m = load("async-expert-contributions")
        group = m.ContributionGroup((0, 1))
        a, b = Future(), Future()
        group.accept(0, a, object())
        group.accept(1, b, object())
        a.set_result((1, 2))
        with self.assertRaises(ValueError):
            group.settle()
        self.assertEqual(len(group.accepted), 2)
        b.set_result((3, 4))
        self.assertEqual(group.settle(), (4, 6))
        self.assertEqual(group.accepted, {})

    def test_async_failure_and_abort_preserve_unsettled_sources(self):
        m = load("async-expert-contributions")
        group = m.ContributionGroup((0, 1))
        a, b = Future(), Future()
        group.accept(0, a, object())
        group.accept(1, b, object())
        a.set_exception(RuntimeError("failed contribution"))
        with self.assertRaises(ValueError):
            group.abort(lambda receipts: None)
        self.assertEqual(len(group.accepted), 2)
        b.set_result((2,))
        with self.assertRaises(RuntimeError):
            group.settle()
        self.assertTrue(group.completed)
        self.assertEqual(group.accepted, {})

    def test_file_expert_demand_prefetch_upload_and_quiescence(self):
        m = load("ssd-expert-materialization")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.bin"
            path.write_bytes(b"abcd")
            source = m.FileExpertSource(path, {1: m.ExpertRegion(1, 2, "packed", hashlib.sha256(b"bc").hexdigest())})
            class Executor:
                def submit(self, function, key):
                    receipt = Future()
                    receipt.set_result(function(key))
                    return receipt
            source.prefetch(1, Executor())
            owner, receipt = object(), Receipt()
            result = source.materialize(1, lambda encoding, data: (owner, receipt))
            self.assertIs(result.owner, owner)
            self.assertEqual(result.encoded_source, b"bc")
            source.quiesce()
            with self.assertRaises(ValueError):
                source.read(1)

    def test_file_expert_checksum_and_pending_failure_are_not_success(self):
        m = load("ssd-expert-materialization")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fixture.bin"
            path.write_bytes(b"abc")
            source = m.FileExpertSource(path, {1: m.ExpertRegion(0, 3, "packed", hashlib.sha256(b"abc").hexdigest())})
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                source.read(1)
            failed = Future()
            failed.set_exception(RuntimeError("read failed"))
            source.pending[1] = failed
            with self.assertRaises(RuntimeError):
                source.quiesce()
            self.assertTrue(source.closed)
            self.assertEqual(source.pending, {})

    def test_checkpoint_empty_payload_exact_bytes_and_longest_prefix(self):
        m = load("checkpoint-storage")
        with tempfile.TemporaryDirectory() as d:
            store = m.CheckpointStore(d, "state-v1", "opaque-v1")
            path = store.put(b"a\0", b"")
            self.assertEqual(store.exact(b"a\0").payload, b"")
            self.assertEqual(store.put(b"a\0", b""), path)
            store.put(b"a\0b", b"encoded")
            self.assertEqual(store.longest_prefix(b"a\0bc"), m.Checkpoint(b"a\0b", b"encoded"))
            self.assertIsNone(store.longest_prefix(b"other"))
            with self.assertRaises(ValueError):
                store.put(b"a\0", b"different state")

    def test_checkpoint_corruption_compatibility_and_capacity(self):
        m = load("checkpoint-storage")
        with tempfile.TemporaryDirectory() as d:
            store = m.CheckpointStore(d, "state-v1", "opaque-v1")
            path = store.put(b"a", b"payload")
            with self.assertRaises(ValueError):
                m.CheckpointStore(d, "state-v2", "opaque-v1").longest_prefix(b"a")
            data = path.read_bytes()
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with self.assertRaises(ValueError):
                store.exact(b"a")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.CheckpointStore(d, "state", "codec", max_bytes=60).put(b"a", b"b")
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_speculative_prefix_correction_bonus_stop_and_budget(self):
        m = load("mtp-decode")
        plan = m.verify((1, 2, 3), (1, 9, 4, 5), 4)
        self.assertEqual(plan, m.CommitPlan((1, 9), 1, 9))
        self.assertEqual(m.verify((1, 2), (1, 2, 3), 3).tokens, (1, 2, 3))
        self.assertEqual(m.verify((1, 2), (1, 2, 3), 1).tokens, (1,))
        self.assertEqual(m.verify((1, 2), (1, 2, 3), 3, stop_tokens={1}).tokens, (1,))
        self.assertEqual(m.verify((), (4,), 0).tokens, ())
        with self.assertRaises(ValueError):
            m.verify((1,), (1,), 2)

    def test_speculative_commit_updates_only_accepted_output(self):
        m = load("mtp-decode")
        plan = m.verify((1, 2), (1, 7, 8), 3)
        applied = []
        self.assertEqual(m.commit(plan, lambda tokens, accepted: applied.append((tokens, accepted))), (1, 7))
        self.assertEqual(applied, [((1, 7), 1)])

    def test_compact_rows_copy_before_commit_and_exact_charges(self):
        m = load("concurrent-request-execution")
        borrowed = [4, 5]
        rows = (m.Row(1, 1, borrowed, Receipt(True)), m.Row(2, 1, (7,), Receipt(False)),
                m.Row(3, 2, (9,), Receipt(True)))
        ready = m.compact_ready(rows, 2)
        self.assertEqual(tuple(r.request_id for r in ready), (1, 3))
        charges = []
        group = m.StagedGroup(ready, {1: 2, 3: 1}, lambda n: charges.append(n) or object())
        result = group.commit(lambda rows: borrowed.clear())
        self.assertEqual(result, ((1, 1, (4, 5)), (3, 2, (9,))))
        self.assertEqual(charges, [8, 4])
        with self.assertRaises(ValueError):
            group.commit(lambda rows: None)

    def test_failed_group_cannot_publish_or_commit_again(self):
        m = load("concurrent-request-execution")
        row = m.Row(1, 1, (2,), Receipt(True))
        group = m.StagedGroup([row], {1: 1}, lambda n: object())
        published = []
        def fail(rows):
            raise RuntimeError("state may already have changed")
        with self.assertRaises(RuntimeError):
            published.extend(group.commit(fail))
        self.assertEqual(published, [])
        self.assertTrue(group.owners)
        with self.assertRaises(ValueError):
            group.commit(lambda rows: None)
        with self.assertRaises(ValueError):
            m.StagedGroup([row], {1: 0}, lambda n: object())

    def test_retirement_is_receipt_based_and_keeps_neighbors_independent(self):
        m = load("concurrent-request-execution")
        receipt = Receipt()
        waiting = m.RetirementBarrier([object()], [receipt])
        neighbor = m.RetirementBarrier([object()], [Receipt(True)])
        self.assertFalse(waiting.poll())
        self.assertTrue(waiting.owners)
        self.assertTrue(neighbor.poll())
        receipt.finished = True
        self.assertTrue(waiting.poll())
        self.assertEqual(waiting.owners, [])

    def test_normalization_formula_modes_and_invalid_contracts(self):
        m = load("rmsnorm-bf16", "reference.py")
        self.assertEqual(m.normalize([[0, 0]], [1, 1], 1), ((0, 0),))
        plain = m.normalize([[1, 1]], [2, 2], 1)[0]
        offset = m.normalize([[1, 1]], [1, 1], 1, mode="unit_offset")[0]
        self.assertEqual(plain, offset)
        self.assertEqual(m.normalize([[1, 1]], [1, 1], 1, mode="gated", gate=[[0, 0]]), ((0, 0),))
        with self.assertRaises(ValueError):
            m.normalize([[1]], [1], 0)
        with self.assertRaises(ValueError):
            m.normalize([[1]], [1], 1, mode="gated")

    def test_normalization_rounding_orders(self):
        m = load("rmsnorm-bf16", "reference.py")
        self.assertEqual([m.bf16(v) for v in (1 + 2**-8, 1 + 3 * 2**-8, 2**-130 + 2**-140, 3.4e38)],
                         [1.0, 1 + 2**-6, 2**-130, float("inf")])
        # x * r = 1.3887 for x = 3: once, x * r * 1.5 = 2.0831 -> 2.078125; Llama-style,
        # bf16(x * r) = 1.390625 and 1.390625 * 1.5 = 2.0859375 ties to even -> 2.09375.
        single = m.normalize([[1, 2, 3]], [1.5] * 3, 1e-6, rounding="single")[0]
        first = m.normalize([[1, 2, 3]], [1.5] * 3, 1e-6, rounding="normalize_first")[0]
        self.assertEqual(single, (0.6953125, 1.390625, 2.078125))
        self.assertEqual(first, (0.6953125, 1.390625, 2.09375))
        self.assertEqual(m.normalize([[1, 2, 3]], [0.5] * 3, 1e-6, mode="unit_offset", rounding="normalize_first"), (first,))
        unrounded = m.normalize([[1, 2, 3]], [1.5] * 3, 1e-6)[0]
        self.assertEqual(tuple(m.bf16(v) for v in unrounded), single)
        with self.assertRaises(ValueError):
            m.normalize([[1]], [1], 1, rounding="llama")

    def test_split_partials_fold_in_fixed_order(self):
        m = load("rmsnorm-bf16", "reference.py")
        residual, partials = [[1e16, 1.0]], [[[1.0, 2.0]], [[-1e16, 3.0]]]
        updated, normalized = m.fold_and_normalize(residual, partials, [1, 1], 1e-6)
        # (1e16 + 1) - 1e16 rounds to 0 in this fixed order; another order gives 1.
        self.assertEqual(updated, ((0.0, 6.0),))
        self.assertEqual(normalized, m.normalize(updated, [1, 1], 1e-6))
        self.assertEqual(m.fold_partials([[2.0]], []), ((2.0,),))
        with self.assertRaises(ValueError):
            m.fold_partials([[1.0, 2.0]], [[[1.0]]])



class ChunkedPrefillTests(unittest.TestCase):
    def test_two_pass_chunks_equal_the_token_by_token_recurrence(self):
        import random
        code = ROOT / "inference" / "recurrent-state-recovery" / "code"
        sys.path.insert(0, str(code))
        try:
            m = load("recurrent-state-recovery", "chunked_prefill.py")
            rng = random.Random(5)
            for normalize in (True, False):
                geometry = m.Geometry(keys=5, values=3, normalize_keys=normalize, normalize_queries=normalize,
                                      query_scale=0.7)
                steps = [m.Step(tuple(rng.gauss(0, 1) for _ in range(5)), tuple(rng.gauss(0, 0.6) for _ in range(5)),
                                tuple(rng.gauss(0, 1) for _ in range(3)), -abs(rng.gauss(0, 0.3)), rng.uniform(0.05, 0.95))
                         for _ in range(13)]
                start = tuple(rng.gauss(0, 0.4) for _ in range(15))
                direct = m.sequential(start, steps, geometry)
                for chunk in (1, 4, 5, 13, 32):  # partial last chunks and a chunk longer than the prompt
                    self.assertLess(m.max_difference(direct, m.two_pass(start, steps, geometry, chunk)), 1e-11, chunk)
            with self.assertRaises(ValueError):
                m.two_pass(start, steps, geometry, 0)
        finally:
            sys.path.remove(str(code))

if __name__ == "__main__":
    unittest.main()


class RecordedSelectionTests(unittest.TestCase):
    def test_buckets_keys_and_table_validation(self):
        m = load("recorded-selection", "selection.py")
        self.assertEqual([m.bucket(w, 16) for w in (1, 16, 17, 1024)], [16, 16, 32, 1024])
        self.assertEqual(m.bucket(600, 16, cap=1000), 1000)
        for bad in ((0, 16, None), (5, 3, None), (1001, 16, 1000)):
            with self.assertRaises(ValueError):
                m.bucket(*bad)
        self.assertEqual(m.key("attn", (8, 128), 64), "attn:8x128:64")
        with self.assertRaises(ValueError):
            m.key("a:b", (1,), 1)
        for bad in ([], {"": 1}, {"k": -1}, {"k": True}, {'k"': 1}):
            with self.assertRaises(ValueError):
                m.validate(bad)

    def test_tuning_replay_fallback_and_stale_indices(self):
        m = load("recorded-selection", "selection.py")
        calls = []
        def time(i):
            calls.append(i)
            return {0: 2.0, 1: 1.0, 2: 1.0}.get(i)
        table = m.Table(tuning=True)
        self.assertEqual(table.choose("k:1:16", 4, time), (1, "tuned"))
        self.assertEqual(table.choose("k:1:16", 4, time), (1, "tuned"))
        self.assertEqual(calls, [0, 1, 2, 3])                      # timed once, unavailable 3 skipped
        self.assertEqual(table.choose("k:1:32", 2, lambda i: None), (0, "fallback"))
        self.assertEqual(table.export(), {"k:1:16": 1, "k:1:32": 0})
        replay = m.Table(table.export(), tuning=True)
        self.assertEqual(replay.choose("k:1:16", 4, lambda i: self.fail("timed")), (1, "recorded"))
        self.assertEqual(m.Table().choose("new:1:16", 9, lambda i: self.fail("timed")), (0, "fallback"))
        with self.assertRaises(ValueError):
            m.Table({"k:1:16": 4}).choose("k:1:16", 4)
        with self.assertRaises(ValueError):
            m.Table().choose("k", 0)
        shrunk = m.Table(tuning=True)                              # a kept choice beyond a shorter list
        self.assertEqual(shrunk.choose("k:1:16", 4, lambda i: 4.0 - i), (3, "tuned"))
        with self.assertRaises(ValueError):
            shrunk.choose("k:1:16", 2)
        self.assertEqual(shrunk.choose("k:1:16", 4), (3, "tuned"))
        for bad in ("", 'k"1', "k\\1"):                            # keys the exported table could not hold
            with self.assertRaises(ValueError):
                m.Table().choose(bad, 2)

    def test_differences_name_path_and_index(self):
        m = load("recorded-selection", "repeatability.py")
        self.assertIsNone(m.first_difference({"a": [1, 2]}, {"a": [1, 2]}))
        self.assertEqual(m.first_difference({"a": [1, 2, 3]}, {"a": [1, 2]}), {"path": "$.a", "reason": "length", "index": 2})
        self.assertEqual(m.first_difference([[1, 2], [3, 4]], [[1, 2], [3, 5]]), {"path": "$[1][1]", "reason": "value", "index": 1})
        self.assertEqual(m.first_difference({"a": 1}, {"b": 1})["reason"], "missing")
        self.assertEqual(m.first_difference([1], ["1"])["reason"], "type")

    def test_runner_uses_fresh_processes(self):
        m = load("recorded-selection", "repeatability.py")
        with tempfile.TemporaryDirectory() as scratch:
            same = [sys.executable, "-B", "-c", "import json, sys; json.dump({'t': [1, 2, 3]}, open(sys.argv[1], 'w'))", "{output}"]
            self.assertTrue(m.run(same, 3, 60)["identical"])
            counter = Path(scratch) / "count"
            differ = [sys.executable, "-B", "-c",
                      "import json, os, sys\np = sys.argv[2]\nn = int(open(p).read()) if os.path.exists(p) else 0\n"
                      "open(p, 'w').write(str(n + 1))\njson.dump({'t': [1, n, 3]}, open(sys.argv[1], 'w'))",
                      "{output}", str(counter)]
            report = m.run(differ, 2, 60)
            self.assertFalse(report["identical"])
            self.assertEqual(report["differences"][0]["path"], "$.t[1]")
            failing = [sys.executable, "-B", "-c", "import sys; sys.exit(3)", "{output}"]
            self.assertIn("failure", m.run(failing, 2, 60))
            report_path = Path(scratch) / "report.json"
            self.assertEqual(m.main(["--runs", "2", "--report", str(report_path), "--"] + same), 0)
            self.assertTrue(json.loads(report_path.read_text())["identical"])
            with self.assertRaises(ValueError):
                m.run([sys.executable, "-c", "pass"], 2, 60)


class HostMemoryAdmissionTests(unittest.TestCase):
    def tree(self, root, files):
        for path, text in files.items():
            target = Path(root) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def test_linux_observations_and_admission(self):
        m = load("host-memory-admission", "memory.py")
        with tempfile.TemporaryDirectory() as root:
            self.tree(root, {"proc/meminfo": "MemTotal: 1000 kB\nMemAvailable: 800 kB\nCommitLimit: 900 kB\nCommitted_AS: 600 kB\n",
                             "proc/sys/vm/overcommit_memory": "2\n", "proc/self/cgroup": "0::/app/worker\n",
                             "sys/fs/cgroup/app/worker/memory.max": "max\n", "sys/fs/cgroup/app/worker/memory.current": "1\n",
                             "sys/fs/cgroup/app/memory.max": "500000\n", "sys/fs/cgroup/app/memory.current": "450000\n",
                             "sys/fs/cgroup/app/memory.stat": "file 60000\nfile_dirty 15000\nfile_writeback 5000\n"})
            seen = m.observe_linux(root)
            self.assertEqual((seen.physical_available, seen.commit_available, seen.cgroup_available), (819200, 307200, 90000))
            self.assertTrue(m.admit(80000, seen, 10000).admitted)
            denied = m.admit(80001, seen, 10000)
            self.assertEqual((denied.admitted, denied.binding, denied.limit), (False, "cgroup", 80000))
            self.assertEqual(m.admit(0, seen, 90001).limit, 0)
            self.tree(root, {"proc/sys/vm/overcommit_memory": "0\n", "sys/fs/cgroup/app/memory.max": "max\n"})
            seen = m.observe_linux(root)
            self.assertIsNone(seen.commit_available)
            self.assertIsNone(seen.cgroup_available)
            self.assertEqual(len(seen.notes), 2)
            self.tree(root, {"proc/self/cgroup": "4:memory:/job\n1:cpu:/job\n",
                             "sys/fs/cgroup/memory/job/memory.usage_in_bytes": "700\n",
                             "sys/fs/cgroup/memory/job/memory.stat": "cache 100\ndirty 30\nwriteback 10\nhierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 360)
            # Hybrid host: an unlimited v2 tree and a v1 memory hierarchy; the v1 limit applies, as in Core.
            self.tree(root, {"proc/self/cgroup": "0::/app/worker\n4:memory:/job\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 360)
        with tempfile.TemporaryDirectory() as empty:
            unknown = m.observe_linux(empty)
            self.assertEqual(unknown.limits(), {})
            self.assertFalse(m.admit(1, unknown).admitted)
            self.assertTrue(m.admit(1, unknown, require_known=False).admitted)
        with self.assertRaises(ValueError):
            m.admit(-1, unknown)

    def test_cgroup_reclaimable_pages_and_memory_high(self):
        # The same cases as Core's host_memory_v2, host_memory_v1 and host_memory_malformed fixtures.
        m = load("host-memory-admission", "memory.py")
        gib = 1 << 30
        with tempfile.TemporaryDirectory() as root:
            app = "sys/fs/cgroup/app/"
            self.tree(root, {"proc/meminfo": "MemAvailable: 800 kB\n", "proc/self/cgroup": "0::/app/worker\n",
                             app + "memory.max": "500000\n", app + "memory.current": "450000\n",
                             app + "memory.stat": "file 60000\nshmem 50000\nfile_dirty 1000\nfile_writeback 0\n"})
            # tmpfs/shmem pages sit in file but are swap-backed: only the file LRUs, else file - shmem, are reclaimable.
            self.assertEqual(m.observe_linux(root).cgroup_available, 59000)
            self.tree(root, {app + "memory.stat": "file 60000\nshmem 50000\nactive_file 4000\ninactive_file 6000\n"
                                                  "file_dirty 1000\nfile_writeback 500\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 58500)
            self.tree(root, {app + "memory.max": "%d\n" % (8 * gib), app + "memory.current": "%d\n" % (8 * gib),
                             app + "memory.stat": "anon %d\nfile %d\nshmem %d\nactive_file 0\ninactive_file 0\n"
                                                  % (2 * gib, 6 * gib, 6 * gib)})
            seen = m.observe_linux(root)
            self.assertEqual(seen.cgroup_available, 0)
            denied = m.admit(5 * gib, seen)
            self.assertEqual((denied.admitted, denied.binding), (False, "cgroup"))
            # memory.high limits too: the effective limit is min(memory.max, memory.high).
            self.tree(root, {app + "memory.max": "max\n", app + "memory.high": "460000\n", app + "memory.current": "450000\n",
                             app + "memory.stat": "active_file 4000\ninactive_file 6000\nfile_dirty 1000\nfile_writeback 500\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 18500)
            self.tree(root, {app + "memory.max": "455000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 13500)
            self.tree(root, {app + "memory.max": "max\n", app + "memory.high": "max\n"})
            seen = m.observe_linux(root)
            self.assertIsNone(seen.cgroup_available)
            self.assertIn("no cgroup v2 memory limit", seen.notes)
            # Malformed counters count nothing as reclaimable, as in Core; malformed limits skip the level.
            self.tree(root, {app + "memory.max": "1000\n", app + "memory.high": "max\n", app + "memory.current": "400\n"})
            for stat in ("file 100\nfile_dirty lots\n", "file 100\nshmem x\n", "active_file 100\ninactive_file -1\n",
                         "active_file 100\nfile_writeback 1.5\n"):
                self.tree(root, {app + "memory.stat": stat})
                seen = m.observe_linux(root)
                self.assertEqual(seen.cgroup_available, 600)
                self.assertTrue(any("memory.stat malformed" in n for n in seen.notes))
            for name in ("memory.max", "memory.high", "memory.current"):
                self.tree(root, {app + "memory.max": "1000\n", app + "memory.high": "max\n", app + "memory.current": "400\n",
                                 app + name: "12z\n"})
                seen = m.observe_linux(root)
                self.assertIsNone(seen.cgroup_available)
                self.assertTrue(any("memory.max, memory.high or memory.current malformed" in n for n in seen.notes))
            # v1: shmem is in cache but not reclaimable; each key prefers its hierarchical total_ form.
            job = "sys/fs/cgroup/memory/job/"
            self.tree(root, {"proc/self/cgroup": "4:memory:/job\n", job + "memory.usage_in_bytes": "700\n",
                             job + "memory.stat": "cache 500\nshmem 300\ndirty 50\nhierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 450)
            self.tree(root, {job + "memory.stat": "cache 500\nshmem 300\ntotal_cache 900\ntotal_shmem 800\ndirty 50\n"
                                                  "hierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 350)
            self.tree(root, {job + "memory.stat": "cache 500\nshmem 300\ntotal_active_file 100\ntotal_inactive_file 60\n"
                                                  "inactive_file 1\ntotal_dirty 20\nhierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 440)
            self.tree(root, {job + "memory.stat": "hierarchical_memory_limit 100\ncache x\n", job + "memory.usage_in_bytes": "10\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 90)
            for stat, usage, note in (("hierarchical_memory_limit 1e9\n", "10\n", "hierarchical_memory_limit malformed"),
                                      ("hierarchical_memory_limit 100\n", "ten\n", "memory.usage_in_bytes malformed")):
                self.tree(root, {job + "memory.stat": stat, job + "memory.usage_in_bytes": usage})
                seen = m.observe_linux(root)
                self.assertIsNone(seen.cgroup_available)
                self.assertTrue(any(note in n for n in seen.notes))

    def test_live_observation_reports_physical_memory(self):
        m = load("host-memory-admission", "memory.py")
        seen = m.observe()
        if sys.platform.startswith("linux") or sys.platform == "win32":
            self.assertGreater(seen.physical_available, 0)


class Nvfp4CodecTests(unittest.TestCase):
    def test_codes_scales_ties_and_packing(self):
        m = load("quantized-projections", "nvfp4.py")
        self.assertEqual([m.e2m1_value(c) for c in range(8)], [0, .5, 1, 1.5, 2, 3, 4, 6])
        self.assertEqual(m.e2m1_value(15), -6)
        self.assertEqual((m.e4m3_value(0x7E), m.e4m3_value(0x38), m.e4m3_value(0x01)), (448, 1, 2 ** -9))
        self.assertTrue(math.isnan(m.e4m3_value(0x7F)))
        self.assertEqual([m.e4m3_encode(m.e4m3_value(b)) for b in range(0x7F)], list(range(0x7F)))
        self.assertEqual((m.e4m3_encode(1000), m.e4m3_encode(1.0625), m.e4m3_encode(1.1875)), (0x7E, 0x38, 0x3A))
        ties = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6, 7.0: 7, -0.25: 0, -5.0: 14, 0.3: 1}
        self.assertEqual({x: m.e2m1_encode(x) for x in ties}, ties)
        self.assertEqual(m.unpack(m.pack([1, 2, 15, 0]), 4), [1, 2, 15, 0])
        self.assertEqual(m.pack([1, 2]), bytes([0x21]))
        for bad in (lambda: m.e2m1_value(16), lambda: m.e2m1_encode(math.nan), lambda: m.quantize_row([1.0] * 15, 1.0),
                    lambda: m.Matrix(1, 16, b"\0" * 8, b"\0", 0.0), lambda: m.Matrix(1, 16, b"\0" * 7, b"\0", 1.0)):
            with self.assertRaises(ValueError):
                bad()

    def test_scale_encoder_vectors_and_every_scale_byte(self):
        m = load("quantized-projections", "nvfp4.py")
        for bits, code in e4m3_scale_vectors():
            x = __import__("struct").unpack("<f", bits.to_bytes(4, "little"))[0]
            self.assertEqual(m.e4m3_encode(x), code, f"{bits:#010x} ({x})")
        # Every byte a checkpoint can hold decodes to its fields' value: bit 7 the sign, 0x7F and 0xFF NaN.
        for b in range(0x80):
            low, high = m.e4m3_value(b), m.e4m3_value(b | 0x80)
            if b == 0x7F:
                self.assertTrue(math.isnan(low) and math.isnan(high))
            else:
                self.assertEqual(low, e4m3_magnitude(b), hex(b))
                self.assertEqual((high, math.copysign(1.0, high)), (-low, -1.0))

    def test_quantization_error_and_batched_projection(self):
        m = load("quantized-projections", "nvfp4.py")
        rng = __import__("random").Random(4)
        dense = [[rng.gauss(0, 1) for _ in range(32)] for _ in range(3)]
        dense[1][:16] = [0.0] * 16
        w = m.Matrix.quantize(dense)
        for n, row in enumerate(dense):
            decoded = w.row(n)
            for b in range(0, 32, 16):
                amax = max(abs(v) for v in row[b:b + 16]) or 1
                self.assertLessEqual(max(abs(a - c) for a, c in zip(row[b:b + 16], decoded[b:b + 16])) / amax, 0.25)
        self.assertEqual(w.row(1)[:16], [0.0] * 16)
        tokens = [[rng.gauss(0, 1) for _ in range(32)] for _ in range(2)]
        y = m.project(w, tokens)
        for t, x in enumerate(tokens):
            for n in range(3):
                self.assertAlmostEqual(y[t][n], math.fsum(a * b for a, b in zip(w.row(n), x)), places=9)
        with self.assertRaises(ValueError):
            m.project(w, [[0.0] * 31])


class RotatedKvTests(unittest.TestCase):
    def test_rotation_identities_and_outlier_fixture(self):
        m = load("rotated-kv-quantization", "rotation.py")
        s = m.signs(32, 3)
        self.assertEqual(s, m.signs(32, 3))
        self.assertTrue(all(v in (-1, 1) for v in s))
        rng = __import__("random").Random(2)
        x = [rng.gauss(0, 1) for _ in range(32)]
        y = [rng.gauss(0, 1) for _ in range(32)]
        self.assertLess(max(abs(a - b) for a, b in zip(m.fwht(m.fwht(x)), x)), 1e-12)
        self.assertAlmostEqual(math.fsum(a * b for a, b in zip(m.rotate(x, s), m.rotate(y, s))), math.fsum(a * b for a, b in zip(x, y)))
        codes, scales = m.quantize(x, 4)
        self.assertTrue(all(-7 <= c <= 7 for c in codes))
        self.assertLessEqual(max(abs(a - b) for a, b in zip(m.dequantize(codes, scales), x)), scales[0] / 2 + 1e-12)
        keys = [[rng.gauss(0, 1) * (25 if j == 3 else 1) for j in range(32)] for _ in range(12)]
        q = [rng.gauss(0, 0.1) for _ in range(32)]
        exact = m.attention(q, keys, keys)
        error = lambda out: sum((a - b) ** 2 for a, b in zip(out, exact))
        self.assertLess(error(m.compressed_attention(q, keys, keys, s)), error(m.compressed_attention(q, keys, keys)))
        for bad in (lambda: m.signs(24, 1), lambda: m.fwht([1.0] * 3), lambda: m.quantize([1.0] * 32, 6),
                    lambda: m.quantize([1.0] * 20), lambda: m.rotate(x, s[:16])):
            with self.assertRaises(ValueError):
                bad()

    def test_rotation_can_increase_quantization_error(self):
        m = load("rotated-kv-quantization", "rotation.py")
        dim = 128
        s = m.signs(dim, 0)
        for values in ([7.0, 1.0] + [0.0] * (dim - 2), [1.0] * dim):
            with self.subTest(values=values[:3]):
                # With one KV row, attention is exactly the stored value row.
                q = [0.125] * dim
                plain = m.compressed_attention(q, [values], [values], bits=4)
                rotated = m.compressed_attention(q, [values], [values], s, bits=4)
                self.assertEqual(plain, values)
                mse = math.fsum((a - b) ** 2 for a, b in zip(rotated, values)) / dim
                self.assertGreater(mse, 1e-6)
        sparse = [7.0, 1.0] + [0.0] * (dim - 2)
        self.assertLess(max(map(abs, m.rotate(sparse, s))), max(map(abs, sparse)))


class DeviceExpertCacheTests(unittest.TestCase):
    def test_clock_pins_hits_and_uses_the_prefetch_pool(self):
        m = load("device-expert-cache", "device_cache.py")
        d = m.ClockDirectory(slots=4, groups=2, experts=8, pool_slots=2)
        first = d.plan(0, [1, 2, 3])
        self.assertEqual((first.hits, first.misses), (0, 3))
        self.assertEqual(first.fills, [("store", 1, 0), ("store", 2, 1), ("store", 3, 2)])
        self.assertEqual(d.prefetch(1, [5, 6]), [(13, 0), (14, 1)])
        self.assertEqual(d.prefetch(0, [1, 4]), [(4, 0)])  # resident expert 1 is not fetched nor given a slot
        p = d.plan(1, [5, 2])
        self.assertEqual(p.fills, [(("pool", 0), 13, 3), ("store", 10, 0)])
        # pool slots go to nonresident predictions in order: resident 5 and the repeat take none
        self.assertEqual(d.prefetch(1, [5, 3, 3, 4, 6]), [(11, 0), (12, 1)])
        self.assertEqual(d.pool_tags[1], [11, 12])
        again = d.plan(0, [1, 2])
        self.assertEqual((again.hits, again.misses), (1, 1))
        # hits are pinned before admissions: a plan never evicts its own experts
        for _ in range(50):
            q = d.plan(1, [0, 1, 2, 3])
            self.assertEqual(len(set(q.slots)), 4)
        self.assertGreater(m.simulate(m.ClockDirectory(8, 1, 12), [(0, [i % 6, (i + 1) % 6]) for i in range(40)]), 0.5)
        for bad in (lambda: d.plan(0, [1, 1]), lambda: d.plan(2, [0]), lambda: d.plan(0, [8]),
                    lambda: d.plan(0, list(range(5))), lambda: m.ClockDirectory(0, 1, 1)):
            with self.assertRaises(ValueError):
                bad()



class TokenSamplingTests(unittest.TestCase):
    def test_greedy_nucleus_and_cores_draws(self):
        m = load("token-sampling", "sampling.py")
        # Core's Philox4x32-10 known answers (core/src/sampling/README.md).
        self.assertEqual(m.philox4x32(0, 0, 0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8))
        self.assertEqual(m.uniform(0, 0), 0xcc4fd1abc2d38 * 2.0 ** -53)
        top = (1 << 64) - 1
        self.assertEqual(m.philox4x32(top, top, 0xffffffff, 0xffffffff), (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd))
        self.assertEqual(m.uniform(top, top, 0xffffffff), m.uniform(top, top, 0xffffffff))
        row = [2.0, 1.0, 2.0, -1.0, 0.5]
        self.assertEqual(m.select(row), 0)
        self.assertEqual(m.select(row, 0.7, top_k=1, seed=5), 0)
        self.assertEqual(m.select(row, 0.7, top_p=1e-9, seed=5), 0)
        pool = m.nucleus(row, 0.7, top_p=0.9)
        self.assertEqual([t for t, _ in pool], [0, 2, 1])
        self.assertAlmostEqual(math.fsum(p for _, p in pool), 1.0)
        self.assertEqual(len(m.nucleus(row, 1.0, top_k=2)), 2)
        # crit::core::Sampler draws these for seed 1 at positions 0..7 (temperature 0.7, top_p 0.9).
        draws = [m.select(row, 0.7, 0.9, 0, seed=1, position=c) for c in range(8)]
        self.assertEqual(draws, [2, 2, 0, 2, 0, 0, 0, 2])
        self.assertNotEqual(draws, [m.select(row, 0.7, 0.9, 0, seed=1, position=c, stream=1) for c in range(8)])
        u = [m.uniform(7, c) for c in range(2000)]
        self.assertTrue(all(0.0 <= x < 1.0 for x in u))
        self.assertAlmostEqual(sum(u) / len(u), 0.5, delta=0.03)
        for bad in (lambda: m.select([]), lambda: m.nucleus(row, 0.0), lambda: m.nucleus(row, 1.0, top_p=1.5),
                    lambda: m.select([float("nan")]), lambda: m.uniform(-1, 0), lambda: m.uniform(0, 0, 1 << 32)):
            with self.assertRaises(ValueError):
                bad()
