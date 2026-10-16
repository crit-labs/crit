"""Portable measured-policy fixtures; no device arithmetic or speed qualification."""
from dataclasses import replace
from functools import lru_cache
import random
import unittest

from test_techniques import load

d = load("heterogeneous-expert-dispatch", "measured_dispatch.py")
r = load("heterogeneous-expert-dispatch")
c = load("device-expert-cache", "replay.py")
native_reference = load("device-expert-cache", "device_cache.py")
m = load("host-memory-admission", "memory.py")


class MeasuredDispatchTests(unittest.TestCase):
    def memory_costs(self, **changes):
        values = dict(identity=self.identity(), scratch=((0, 0), (1, 10), (2, 20), (3, 30)),
                      staging=((0, 0), (7, 3), (14, 6), (21, 9)),
                      experts=((3, 1, 40), (4, 1, 40), (5, 1, 40)),
                      upload_sources=((3, 0, 0), (4, 0, 0), (5, 0, 0)))
        values.update(changes)
        return d.HostMemoryCosts(**values)

    def test_memory_rejects_fast_cpu_split_before_ranking(self):
        admit = lambda **needs: m.admit_working_set(**needs, observation=m.Observation(30, 1000))
        result = self.plan(memory_costs=self.memory_costs(), admit_host=admit)
        self.assertEqual(result.best.host, ())
        self.assertEqual(result.candidates[1].host_memory_bytes, (11, 6, 40))
        self.assertFalse(result.candidates[1].memory_admitted)
        self.assertIn("physical", result.candidates[1].memory_reason)
        self.assertEqual(result.candidates[1].estimated_ms, 7.5)

    def test_file_working_set_does_not_consume_anonymous_commit(self):
        needs = self.memory_costs(scratch=((0, 0), (1, 0), (2, 0), (3, 0)),
                                 staging=((0, 0), (7, 0), (14, 0), (21, 0)),
                                 experts=((3, 0, 40), (4, 0, 40), (5, 0, 40)))
        admit = lambda **cost: m.admit_working_set(**cost, observation=m.Observation(100, 0))
        result = self.plan(memory_costs=needs, admit_host=admit)
        self.assertEqual([r.expert for r in result.best.host], [3])
        self.assertEqual(result.best.host_memory_bytes, (0, 0, 40))

    def test_unknown_memory_and_missing_costs_fail_closed(self):
        unknown = lambda **cost: m.admit_working_set(**cost, observation=m.Observation())
        self.assertIsNone(self.plan(memory_costs=self.memory_costs(), admit_host=unknown).best)
        admit = lambda **cost: m.admit_working_set(**cost, observation=m.Observation(1000))
        result = self.plan(memory_costs=self.memory_costs(scratch=((0, 0),)), admit_host=admit)
        self.assertEqual(result.best.host, ())
        self.assertEqual(result.candidates[1].memory_reason, "missing host memory cost")
        with self.assertRaises(ValueError):
            d.revalidate_memory(result.candidates[1], admit)

    def test_revalidation_uses_fresh_observation_and_preserves_exact_routes(self):
        observation = m.Observation(1000)
        admit = lambda **cost: m.admit_working_set(**cost, observation=observation, reserve=5)
        plan = self.plan(memory_costs=self.memory_costs(), admit_host=admit).best
        self.assertTrue(d.revalidate_memory(plan, admit))
        observation = m.Observation(30)
        self.assertFalse(d.revalidate_memory(plan, admit))
        self.assertEqual({r.position for r in plan.host + plan.device}, {1, 2, 7, 8})
        # The planner never invokes allocation/execution; admission failures do not drop routes.

    def test_ready_only_needs_no_additional_memory_or_host_expert_estimates(self):
        routes = (r.Route(0, 99, 1),)
        result = d.plan_dispatch(routes, {99}, {}, self.calibration(), identity=self.identity(),
                staging_budget_bytes=0, memory_costs=self.memory_costs(experts=()),
                admit_host=lambda **cost: m.admit_working_set(**cost, observation=m.Observation()))
        self.assertEqual(result.best.host_memory_bytes, (0, 0, 0))
        self.assertEqual(result.best.device, routes)

    def test_cpu_pages_deduplicate_experts_and_all_memory_domains_bind(self):
        costs = self.memory_costs()
        routes = (r.Route(0, 3, 1), r.Route(1, 3, 1))
        self.assertEqual(costs.requirement(routes, ((4, 7),)), (21, 3, 40))
        for observation, binding in ((m.Observation(1000, 23, 1000), "commit"),
                                     (m.Observation(1000, 1000, 63), "cgroup"),
                                     (m.Observation(63, 1000, 1000), "physical")):
            result = m.admit_working_set(anonymous_bytes=21, pinned_bytes=3, file_bytes=40,
                                         observation=observation)
            self.assertFalse(result.admitted)
            self.assertEqual(result.binding, binding)
        self.assertFalse(m.admit_working_set(anonymous_bytes=1, pinned_bytes=0, file_bytes=0,
                                            observation=m.Observation(None, 1000)).admitted)

    def test_invalid_memory_contracts_are_rejected(self):
        for change in ({"scratch": ((0, 1),)}, {"staging": ((0, 0), (0, 0))},
                       {"experts": ((3, -1, 1),)}, {"common_anonymous_bytes": True}):
            with self.assertRaises(ValueError):
                self.memory_costs(**change)
        with self.assertRaises(ValueError):
            self.plan(memory_costs=self.memory_costs())
        with self.assertRaises(ValueError):
            self.plan(memory_costs=self.memory_costs(identity=replace(self.identity(), workload="different")),
                      admit_host=lambda **_: True)
        with self.assertRaises(ValueError):
            self.plan(memory_costs=self.memory_costs(), admit_host=lambda **_: True)
        with self.assertRaises(ValueError):
            m.admit_working_set(anonymous_bytes=1, pinned_bytes=0, file_bytes=0, observation=m.Observation(-1))

    def test_upload_source_pages_and_loader_bytes_are_also_charged(self):
        costs = self.memory_costs(upload_sources=((3, 2, 20), (4, 2, 30), (5, 2, 40)))
        admit = lambda **needs: m.admit_working_set(**needs, observation=m.Observation(1000))
        plan = self.plan(memory_costs=costs, admit_host=admit).candidates[0]
        self.assertEqual(plan.host_memory_bytes, (6, 9, 90))
        missing = self.plan(memory_costs=self.memory_costs(upload_sources=()), admit_host=admit)
        self.assertTrue(all(not c.memory_admitted for c in missing.candidates[:-1]))
    def identity(self):
        return d.CalibrationKey("weights:1", "layout:2", "shape/batch/load:3",
                                "kernels/config:4", "cpu/device/links:5")

    def curve(self, points):
        return d.TimingCurve.from_samples({n: (time,) for n, time in points.items()})

    def calibration(self, **changes):
        profile = d.DispatchCalibration(
            self.identity(), self.curve({i: 4 * i for i in range(5)}),
            self.curve({7 * i: 2 * i for i in range(4)}),
            self.curve({i: i for i in range(5)}), (0.5,), "cpu-device")
        return replace(profile, **changes)

    def plan(self, *, calibration=None, **changes):
        options = dict(identity=self.identity(), staging_budget_bytes=21)
        options.update(changes)
        return d.plan_dispatch((r.Route(7, 99, 0.5), r.Route(2, 3, 1.0),
                                r.Route(8, 4, 0.25), r.Route(1, 5, 0.75)),
                               {99}, {3: 7, 4: 7, 5: 7},
                               calibration or self.calibration(), **options)

    def test_measured_overlap_selects_a_mixed_split_including_device_compute(self):
        decision = self.plan()
        self.assertEqual(decision.identity, self.identity())
        self.assertEqual([p.estimated_ms for p in decision.candidates], [10.5, 7.5, 8.5, 12.5])
        self.assertEqual([v.expert for v in decision.best.host], [3])
        self.assertEqual([v.expert for v in decision.best.device], [99, 4, 5])
        self.assertEqual(decision.best.uploads, ((4, 7), (5, 7)))
        serial = self.plan(calibration=self.calibration(overlap="serial"))
        self.assertEqual(serial.best.host, ())
        self.assertEqual(serial.best.estimated_ms, 10.5)

    def test_measurement_medians_are_snapshots_and_not_fastest_samples(self):
        samples = {0: [0], 1: [1, 4, 90]}
        curve = d.TimingCurve.from_samples(samples)
        samples[1][1] = 1000
        self.assertEqual(curve.at(1), 4)
        self.assertIsNone(curve.at(2))
        profile = self.calibration(cpu=curve)
        decision = self.plan(calibration=profile)
        self.assertEqual(decision.best.estimated_ms, 7.5)
        self.assertEqual(decision.candidates[2].missing_measurements, (("cpu", 2),))

    def test_stale_model_layout_workload_code_and_topology_are_rejected(self):
        for field in vars(self.identity()):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.plan(identity=replace(self.identity(), **{field: "changed"}))

    def test_sparse_curves_never_interpolate_or_invent_a_fallback_time(self):
        zero = self.curve({0: 0})
        decision = self.plan(calibration=self.calibration(cpu=zero, device=zero, upload=zero))
        self.assertIsNone(decision.best)
        self.assertEqual(len(decision.candidates), 4)
        self.assertTrue(all(p.estimated_ms is None and p.missing_measurements for p in decision.candidates))

    def test_byte_and_occurrence_limits_filter_before_selection(self):
        decision = self.plan(staging_budget_bytes=7)
        self.assertEqual([len(p.host) for p in decision.candidates], [2, 3])
        self.assertEqual(decision.best.staging_bytes, 7)
        self.assertEqual(self.plan(staging_budget_bytes=7, max_host_routes=1).candidates, ())
        self.assertIsNone(self.plan(max_device_routes=0).best)  # READY route stays device-side
        self.assertEqual(self.plan(max_host_routes=0).best.host, ())

    def test_repeated_experts_upload_once_and_keep_every_weighted_occurrence(self):
        routes = (r.Route(9, 4, 0.5), r.Route(1, 2, -0.25), r.Route(5, 4, 0.75))
        profile = self.calibration(upload=self.curve({0: 0, 5: 1, 13: 2, 18: 3}))
        decision = d.plan_dispatch(routes, set(), {4: 13, 2: 5}, profile,
                                   identity=self.identity(), staging_budget_bytes=18)
        self.assertEqual(decision.candidates[0].uploads, ((4, 13), (2, 5)))
        self.assertEqual([v.position for v in decision.candidates[1].host], [9, 5])
        expected = (sum(route.expert * route.weight for route in routes),)
        for plan in decision.candidates:
            # A deliberately reordered executor still combines by original occurrence.
            contributions = [(route.position, (route.expert * route.weight,))
                             for route in reversed(plan.device + plan.host)]
            self.assertEqual(r.combine(routes, contributions), expected)
            self.assertEqual(set(v.position for v in plan.host + plan.device), {9, 1, 5})
            self.assertEqual(len(plan.host + plan.device), 3)

    def test_custom_prefix_order_changes_membership_without_changing_route_order(self):
        plan = self.plan(miss_order=(5, 4, 3)).best
        self.assertEqual([v.position for v in plan.host], [1])
        self.assertEqual([v.position for v in plan.device], [7, 2, 8])
        self.assertEqual(plan.uploads, ((4, 7), (3, 7)))
        for order in ((3, 3, 5), (3, 4), (99, 4, 5), (3, 4, True)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                self.plan(miss_order=order)

    def test_ready_only_and_arbitrary_route_widths(self):
        # Wider than a warp, with no device-specific limits in this host policy.
        routes = tuple(r.Route(i * 3, 1000 + i, 1 / 65) for i in range(65))
        curve = self.curve({i: i for i in range(66)})
        profile = self.calibration(cpu=curve, device=curve, upload=self.curve({0: 0}))
        plan = d.plan_dispatch(routes, {v.expert for v in routes}, {}, profile,
                               identity=self.identity(), staging_budget_bytes=0).best
        self.assertEqual(plan.device, routes)
        self.assertEqual(plan.uploads, ())
        self.assertEqual(plan.estimated_ms, 65.5)

    def test_profiles_accept_nonlinear_curves_without_assuming_bandwidth_or_scaling(self):
        profile = self.calibration(cpu=self.curve({0: 0, 1: 30, 2: 1, 3: 90}),
                                   upload=self.curve({0: 0, 7: 0.1, 14: 40, 21: 20}),
                                   device=self.curve({0: 0, 1: 1, 2: 1, 3: 5, 4: 10}))
        self.assertEqual(self.plan(calibration=profile).best.estimated_ms, 1.6)

    def test_equal_estimates_are_stable_and_keep_the_first_feasible_split(self):
        profile = self.calibration(cpu=self.curve({0: 0, 1: 3, 2: 6, 3: 9}), overlap="serial")
        decision = self.plan(calibration=profile)
        self.assertTrue(all(p.estimated_ms == 10.5 for p in decision.candidates))
        self.assertEqual(decision.best.host, ())
        self.assertEqual(len(self.plan(calibration=profile, staging_budget_bytes=14).best.host), 1)

    def test_invalid_measurements_and_constraints_fail_closed(self):
        for points in ({0: [1]}, {1: [1]}, {0: [0], 1: []},
                       {0: [0], 1: [float("nan")]}, {0: [0], 1: [-1]},
                       {0: [0], 1: [float("inf")]}, {0: [0], True: [1]},
                       {0: [0], 1: [True]}, {0: [0], 1: [1e308, 1e308]}):
            with self.subTest(points=points), self.assertRaises(ValueError):
                d.TimingCurve.from_samples(points)
        with self.assertRaises(ValueError):
            d.TimingCurve(((0, (0,)), (0, (0,))))
        for changes in ({"common_samples_ms": ()}, {"common_samples_ms": (float("nan"),)},
                        {"overlap": "automatic"}, {"identity": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.calibration(**changes)
        for options in ({"staging_budget_bytes": -1}, {"max_host_routes": True},
                        {"max_device_routes": -2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)
        for routes, sizes in (((), {}), ((r.Route(0, 1, 1), r.Route(0, 2, 1)), {1: 7, 2: 7}),
                              ((r.Route(0, 1, float("nan")),), {1: 7}),
                              ((r.Route(0, 1, 1),), {}), ((r.Route(0, 1, 1),), {1: 0})):
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                d.plan_dispatch(routes, (), sizes, self.calibration(),
                                identity=self.identity(), staging_budget_bytes=21)
        with self.assertRaises(ValueError):
            d.plan_dispatch((r.Route(0, 1, 1),), (1, True), {}, self.calibration(),
                            identity=self.identity(), staging_budget_bytes=0)


class CacheReplayTests(unittest.TestCase):
    def test_capacity_changes_respect_live_pins_without_partial_eviction(self):
        trace = [c.Demand(("a", "b")), c.Pin("reader", ("a",)), c.Capacity(0),
                 c.Demand(("b",)), c.Release("reader"), c.Capacity(0), c.Demand(("c",))]
        for result in c.compare_policies(trace, self.bundles(), 16):
            self.assertEqual(result.capacity_rejections, 1)
            self.assertEqual((result.hits, result.misses, result.evictions), (1, 3, 2))
            self.assertEqual(result.final_capacity_bytes, 0)
            self.assertEqual(result.final_keys, ())
        grown = c.replay([c.Demand(("a",)), c.Capacity(16), c.Demand(("a", "b"))], self.bundles(), 0)
        self.assertEqual((grown.bypasses, grown.final_keys), (1, ("a", "b")))

    def test_aging_adapts_lfu_to_changed_working_set(self):
        trace = [c.Demand(("a",))] * 20 + [c.Demand(("b",)), c.Decay(0, 1),
                 c.Demand(("b",)), c.Demand(("c",)), c.Demand(("b",))]
        aged = c.replay(trace, self.bundles(), 16, policy="lfu")
        old = c.replay([v for v in trace if not isinstance(v, c.Decay)], self.bundles(), 16, policy="lfu")
        self.assertEqual(aged.final_keys, ("b", "c"))
        self.assertLess(aged.misses, old.misses)

    def test_prefill_reuse_and_recent_decode_protection_preserve_working_set(self):
        trace = [c.Demand(("a", "b")), c.Demand(("c", "d"), "prefill"), c.Demand(("a", "b"))]
        baseline = c.replay(trace, self.bundles(), 16)
        populated = c.replay(trace, self.bundles(), 16, prefill_populates=True)
        protected = c.replay(trace, self.bundles(), 16, prefill_populates=True, protect_decode_groups=1)
        self.assertEqual((baseline.hits, protected.hits, populated.hits), (2, 2, 0))
        self.assertEqual(baseline.bypasses, 2)
        self.assertEqual({s.phase for s in baseline.traffic}, {"prefill", "decode"})

    def test_speculative_outcomes_change_metrics_but_not_online_decisions(self):
        before = [c.Demand(("a",)), c.Demand(("b",), traffic="prefetch")]
        after = [c.Demand(("c",)), c.Demand(("b",))]
        accepted = before + [c.Demand(("b",), traffic="accepted-draft")] + after
        rejected = before + [c.Demand(("b",), traffic="rejected-draft")] + after
        for policy in ("lru", "lfu", "clock"):
            a, r = (c.replay(t, self.bundles(), 8, policy=policy) for t in (accepted, rejected))
            self.assertEqual((a.hits, a.misses, a.final_keys), (r.hits, r.misses, r.final_keys))
            self.assertEqual((a.wasted_prefetch_bytes, r.wasted_prefetch_bytes), (0, 8))
            self.assertEqual(sum(s.fetch_bytes for s in r.traffic), r.fetch_bytes)
        pending = c.replay([c.Demand(("b",), traffic="prefetch")], self.bundles(), 8)
        self.assertEqual((pending.unused_prefetch_bytes, pending.wasted_prefetch_bytes), (8, 0))

    def test_multiple_pin_owners_and_warmup_control_events(self):
        trace = [c.Demand(("a",)), c.Pin("one", ("a",)), c.Pin("two", ("a",)),
                 c.Release("one"), c.Capacity(0), c.Release("two"), c.Capacity(0), c.Demand(("b",))]
        result = c.replay(trace, self.bundles(), 8, warmup_groups=1)
        self.assertEqual((result.scored_groups, result.demands, result.capacity_rejections), (1, 1, 1))
        for trace in ([c.Pin("missing", ("a",))], [c.Release("missing")],
                      [c.Demand(("a",)), c.Pin("x", ("a",)), c.Pin("x", ("a",))],
                      [c.Decay(2, 1)], [c.Decay(1, 0)], [c.Capacity(True)],
                      [c.Demand(("a",), traffic="unknown")]):
            with self.assertRaises(ValueError):
                c.replay(trace, self.bundles(), 8)

    def test_event_free_replay_matches_explicit_decode_events(self):
        trace = (("a", "b"), ("c",), ("a",))
        self.assertEqual(c.compare_policies(trace, self.bundles(), 16),
                         c.compare_policies((c.Demand(t) for t in trace), self.bundles(), 16))
    def bundles(self, names="abcd", nbytes=8, miss_ms=2.0):
        return {name: c.Bundle(nbytes, miss_ms) for name in names}

    def test_group_pinning_bytes_and_measured_costs(self):
        bundles = {"a": c.Bundle(8, 1), "b": c.Bundle(8, 10), "c": c.Bundle(8, 100)}
        result = c.replay((("a", "b"), ("a", "c"), ("b", "c"), ("a",)), bundles, 16)
        self.assertEqual((result.demands, result.hits, result.misses), (7, 2, 5))
        self.assertEqual((result.fetch_bytes, result.estimated_miss_ms), (40, 122))
        self.assertEqual((result.admissions, result.bypasses, result.evictions), (5, 0, 3))
        self.assertEqual(result.final_keys, ("a", "c"))

    def test_no_partial_eviction_when_a_later_hit_pins_the_needed_space(self):
        bundles = {"a": c.Bundle(4), "b": c.Bundle(4), "c": c.Bundle(6)}
        for policy in c.POLICIES:
            with self.subTest(policy=policy):
                result = c.replay((("a", "b"), ("c", "a"), ("b",)), bundles, 8, policy=policy)
                self.assertEqual(result.final_keys, ("a", "b"))
                self.assertEqual((result.hits, result.misses, result.bypasses, result.evictions), (2, 3, 1, 0))
                self.assertEqual(result.fetch_bytes, 14)
                self.assertIsNone(result.estimated_miss_ms)

    def test_group_overflow_bypasses_exactly_without_evicting_its_own_fills(self):
        result = c.replay((("a", "b", "c"), ("c", "b")), self.bundles(), 16)
        self.assertEqual((result.hits, result.misses, result.bypasses), (1, 4, 1))
        self.assertEqual(result.final_keys, ("b", "c"))
        self.assertEqual(result.peak_resident_bytes, 16)

    def test_variable_bundles_evict_enough_bytes_and_oversized_ones_bypass(self):
        bundles = {"a": c.Bundle(3, 1), "b": c.Bundle(4, 2),
                   "c": c.Bundle(7, 4), "huge": c.Bundle(100, 10)}
        result = c.replay((("a",), ("b",), ("c",), ("huge",), ("c",)), bundles, 7)
        self.assertEqual((result.evictions, result.bypasses, result.hits), (2, 1, 1))
        self.assertEqual(result.fetch_bytes, 114)
        self.assertEqual(result.estimated_miss_ms, 17)
        self.assertEqual(result.final_keys, ("c",))

    def test_warmup_preserves_history_but_excludes_costs_and_deduplicates_routes(self):
        result = c.replay((("a", "a", "b"), ("b", "b", "c")), self.bundles(), 16, warmup_groups=1)
        self.assertEqual((result.scored_groups, result.demands, result.hits, result.misses), (1, 2, 1, 1))
        self.assertEqual((result.fetch_bytes, result.estimated_miss_ms), (8, 2))
        self.assertEqual(result.peak_resident_bytes, 16)
        unknown_warmup = c.replay((("a",), ("a",)), {"a": c.Bundle(8)}, 8, warmup_groups=1)
        self.assertEqual(unknown_warmup.estimated_miss_ms, 0)

    def test_second_touch_uses_prior_groups_and_resists_a_scan(self):
        catalog = self.bundles()
        result = c.replay((("a", "a"),), catalog, 8, admission="second-touch")
        self.assertEqual((result.demands, result.admissions, result.bypasses), (1, 0, 1))
        trace = (("a",), ("a",), ("b",), ("c",), ("d",), ("a",))
        scan = c.replay(trace, catalog, 8, admission="second-touch", warmup_groups=2)
        always = c.replay(trace, catalog, 8, warmup_groups=2)
        self.assertEqual((scan.hits, scan.misses, scan.bypasses), (1, 3, 3))
        self.assertEqual((always.hits, always.misses), (0, 4))

    def test_lfu_retains_frequency_across_groups(self):
        trace = tuple((k,) for k in "aabca")
        lru, lfu = c.compare_policies(trace, self.bundles(), 16, policies=("lru", "lfu"))
        self.assertEqual((lru.misses, lfu.misses), (4, 3))

    def test_zero_capacity_empty_trace_and_all_warmup(self):
        result = c.replay((("a", "a"), ("b",)), self.bundles(), 0)
        self.assertEqual((result.hits, result.misses, result.bypasses, result.fetch_bytes), (0, 2, 2, 16))
        self.assertEqual(result.peak_resident_bytes, 0)
        for trace, warmup in (((), 0), ((("a",),), 1)):
            result = c.replay(trace, self.bundles(), 8, warmup_groups=warmup)
            self.assertEqual((result.scored_groups, result.demands, result.estimated_miss_ms), (0, 0, 0))
            self.assertEqual(result.hit_rate, 0)

    def test_comparison_consumes_generators_once_and_is_repeatable(self):
        catalog = self.bundles()
        trace = (("a", "b"), ("b", "c"), ("a",))
        generator = (iter(group) for group in trace)
        results = c.compare_policies(generator, catalog, 16, warmup_groups=1)
        self.assertEqual(results, c.compare_policies(trace, catalog, 16, warmup_groups=1))
        self.assertEqual(tuple(result.policy for result in results), c.POLICIES)
        for result in results:
            self.assertEqual(result, c.replay(trace, catalog, 16, policy=result.policy, warmup_groups=1))
        self.assertEqual(catalog, self.bundles())

    def test_content_and_representation_identities_never_alias(self):
        keys = ("revision1/layer1/expert2/fp16", "revision2/layer1/expert2/fp16",
                "revision1/layer1/expert2/int8")
        result = c.replay((keys, keys), self.bundles(keys), 24)
        self.assertEqual((result.hits, result.misses), (3, 3))

    def test_clock_matches_the_existing_directory_on_fixed_slot_traces(self):
        rng = random.Random(302)
        for slots in (3, 7, 40):
            with self.subTest(slots=slots):
                directory = native_reference.ClockDirectory(slots, 3, 19)
                hits = misses = 0
                trace = []
                for _ in range(180):
                    group = rng.randrange(3)
                    experts = rng.sample(range(19), rng.randrange(1, min(slots, 19) + 1))
                    plan = directory.plan(group, experts)
                    hits += plan.hits
                    misses += plan.misses
                    trace.append(tuple(str(group * 19 + e) for e in experts))
                result = c.replay(trace, self.bundles((str(k) for k in range(57)), 16),
                                  slots * 16, policy="clock")
                self.assertEqual((result.hits, result.misses), (hits, misses))
                self.assertEqual(result.fetch_bytes, misses * 16)
                self.assertEqual(set(result.final_keys), {str(k) for k in directory.key_slot})

    def test_single_equal_size_hindsight_matches_exhaustive_mandatory_admission(self):
        rng = random.Random(103)
        for capacity in (1, 2, 3):
            for _ in range(35):
                sequence = tuple(rng.choice("abcd") for _ in range(11))

                @lru_cache(None)
                def optimum(index, retained):
                    if index == len(sequence):
                        return 0
                    key = sequence[index]
                    if key in retained:
                        return optimum(index + 1, retained)
                    if len(retained) < capacity:
                        return 1 + optimum(index + 1, frozenset((*retained, key)))
                    return 1 + min(optimum(index + 1, (retained - {old}) | {key}) for old in retained)

                result = c.replay(((k,) for k in sequence), self.bundles(), capacity * 8,
                                  policy="farthest-next-use")
                self.assertEqual(result.misses, optimum(0, frozenset()))

    def test_random_byte_budgets_account_for_every_demand(self):
        rng = random.Random(492)
        catalog = {str(i): c.Bundle(rng.randint(1, 11), i / 2) for i in range(14)}
        trace = [tuple(rng.choices(tuple(catalog), k=rng.randint(0, 17))) for _ in range(70)]
        for capacity in (0, 3, 10, 22, 60):
            for admission in c.ADMISSIONS:
                for result in c.compare_policies(trace, catalog, capacity, admission=admission, warmup_groups=10):
                    self.assertEqual(result.demands, sum(len(set(group)) for group in trace[10:]))
                    self.assertEqual(result.demands, result.hits + result.misses)
                    self.assertEqual(result.misses, result.admissions + result.bypasses)
                    self.assertLessEqual(result.peak_resident_bytes, capacity)
                    self.assertLessEqual(sum(catalog[k].nbytes for k in result.final_keys), capacity)

    def test_invalid_geometry_keys_measurements_and_policies(self):
        for size, cost in ((0, None), (-1, None), (True, None), (1, -1),
                           (1, float("nan")), (1, float("inf")), (1, True)):
            with self.subTest(size=size, cost=cost), self.assertRaises(ValueError):
                c.Bundle(size, cost)
        for options in ({"capacity_bytes": -1}, {"capacity_bytes": True}, {"warmup_groups": 2},
                        {"policy": "unknown"}, {"admission": "unknown"}):
            args = dict(capacity_bytes=8)
            args.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                c.replay((("a",),), self.bundles(), **args)
        for trace in (("a",), (("unknown",),), ((2,),)):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                c.replay(trace, self.bundles(), 8)
        for policies in ((), ("lru", "lru"), ("unknown",)):
            with self.subTest(policies=policies), self.assertRaises(ValueError):
                c.compare_policies((), {}, 0, policies=policies)


if __name__ == "__main__":
    unittest.main()
