"""Numerical and stateful fixtures for optional proposals and partitioned attention."""
import math
import random
import unittest
from unittest.mock import patch

from test_techniques import load


class PromptCopyTests(unittest.TestCase):
    def test_proposals_verify_and_only_accepted_tokens_enter_history(self):
        m, verifier = load("mtp-decode", "prompt_copy.py"), load("mtp-decode")
        index = m.PromptCopyIndex((1, 2, 3, 4, 1, 2), window=2, history_window=2)
        self.assertEqual(index.propose(), (3, 4, 1, 2))
        plan = verifier.verify(index.propose(), (3, 9, 0, 0, 0), remaining=4)
        self.assertEqual(plan.tokens, (3, 9))
        index.append(plan.tokens)
        self.assertEqual(index.tokens, (1, 2, 3, 4, 1, 2, 3, 9))
        index.rewind(6)
        self.assertEqual(index.propose(2), (3, 4))
        self.assertEqual(index.propose(2, stop_tokens=(3,)), (3,))
        self.assertEqual(index.propose(3, vocabulary=4), (3,))

    def test_newest_bucket_does_not_count_the_current_suffix(self):
        m = load("mtp-decode", "prompt_copy.py")
        index = m.PromptCopyIndex((1, 2, 3, 1, 2), window=2, history_window=2, max_matches=1)
        self.assertEqual(index.propose(1), (3,))
        index.append((4, 1, 2))
        self.assertEqual(index.propose(1), (4,))
        index.rewind(5)
        self.assertEqual(index.propose(1), (3,))
        class CollidingWindow(tuple):
            def __hash__(self):
                return 0
        # Force every indexed window into the same hash bucket; tokens still decide.
        with patch.object(m, "tuple", CollidingWindow, create=True):
            collided = m.PromptCopyIndex((1, 2, 3, 4, 8, 9, 6, 7, 1, 2),
                                        window=2, history_window=2)
            self.assertEqual(collided.propose(2), (3, 4))

    def test_append_and_rewind_match_a_fresh_index(self):
        m, rng = load("mtp-decode", "prompt_copy.py"), random.Random(82)
        index = m.PromptCopyIndex(window=3, history_window=3, max_matches=4, max_tokens=300)
        history = []
        for _ in range(150):
            if history and rng.random() < .25:
                count = rng.randrange(len(history) + 1)
                history = history[:count]
                index.rewind(count)
            else:
                new = tuple(rng.randrange(3) for _ in range(rng.randrange(1, 5)))
                history.extend(new)
                index.append(new)
            fresh = m.PromptCopyIndex(history, window=3, history_window=3, max_matches=4)
            self.assertEqual(index.tokens, tuple(history))
            self.assertEqual(index.propose(), fresh.propose())

    def test_longer_match_for_generated_history(self):
        m = load("mtp-decode", "prompt_copy.py")
        index = m.PromptCopyIndex((), window=2, history_window=4)
        index.append((7, 8, 1, 2, 9, 5, 6, 1, 2))
        self.assertEqual(index.propose(), ())
        index.append((7, 8, 1, 2))
        self.assertEqual(index.propose(1), (9,))

    def test_capacity_invalid_ids_and_rewind_do_not_mutate_context(self):
        m = load("mtp-decode", "prompt_copy.py")
        index = m.PromptCopyIndex((1, 2, 3), max_tokens=4)
        for tokens in ((4, 5), (-1,), (True,), (1 << 31,)):
            with self.assertRaises(ValueError):
                index.append(tokens)
            self.assertEqual(index.tokens, (1, 2, 3))
        for length in (-1, 4, True):
            with self.assertRaises(ValueError):
                index.rewind(length)
        self.assertEqual(index.propose(0), ())
        self.assertEqual(m.PromptCopyIndex((1, 2)).propose(), ())

    def test_batch_charges_padding_and_bonus_for_every_row(self):
        m = load("mtp-decode", "prompt_copy.py")
        a = m.PromptCopyIndex((1, 2, 3, 4, 5, 1, 2), window=2, history_window=2)
        b = m.PromptCopyIndex((9, 8, 7), window=2, history_window=2)
        plan = m.propose_batch((a, b), 12, bucket=4)
        self.assertEqual(plan.proposals, ((3, 4, 5, 1, 2), ()))
        self.assertEqual((plan.width, plan.verification_rows), (5, 12))
        limited = m.propose_batch((a, b), 10, bucket=4, remaining=(1, 0))
        self.assertEqual((limited.width, limited.verification_rows), (4, 10))
        with self.assertRaises(ValueError):
            m.propose_batch((a, b), 1)
        self.assertEqual(m.propose_batch((), 0).verification_rows, 0)
        stopped = m.propose_batch((a, a), 12, stop_tokens=iter((3,)))
        self.assertEqual(stopped.proposals, ((3,), (3,)))


class PartitionedAttentionTests(unittest.TestCase):
    def test_partition_is_bijective_balanced_and_respects_page_boundaries(self):
        m = load("partitioned-attention")
        for world in (1, 2, 3, 8):
            for count in (0, 1, 7, 16, 33, 257):
                shards = [m.SequencePartition(world, r, 4) for r in range(world)]
                restored = []
                for shard in shards:
                    positions = [shard.global_position(i) for i in range(shard.length(count))]
                    restored.extend(positions)
                    for p in positions:
                        page, row = shard.address(p)
                        self.assertEqual(page * 4 + row, shard.local(p))
                        self.assertLess(page, shard.pages(count))
                self.assertEqual(sorted(restored), list(range(count)))
                self.assertLessEqual(max(s.length(count) for s in shards)-min(s.length(count) for s in shards), 1)
        with self.assertRaises(ValueError):
            m.SequencePartition(2, 0).local(1)
        for args in ((0, 0), (2, 2), (True, 0)):
            with self.assertRaises(ValueError):
                m.SequencePartition(*args)

    def test_dense_and_sparse_causal_partials_match_direct_softmax(self):
        m, rng = load("partitioned-attention"), random.Random(51)
        rows = [(i, tuple(rng.uniform(-2, 2) for _ in range(3)),
                 tuple(rng.uniform(-3, 3) for _ in range(2))) for i in range(19)]
        for query_position in (0, 1, 6, 18):
            for selected in (None, (0, 3, 7, 12, 18)):
                query = tuple(rng.uniform(-1, 1) for _ in range(3))
                visible = [(k, v) for p, k, v in rows if p <= query_position and (selected is None or p in selected)]
                weights = [math.exp(sum(q * x for q, x in zip(query, k)) * .5) for k, _ in visible]
                expected = [sum(w * v[c] for w, (_, v) in zip(weights, visible)) / sum(weights) for c in range(2)]
                for world in (1, 2, 4, 23):
                    partials = [m.attention_partial(query, [row for row in rows if row[0] % world == rank],
                                value_width=2, query_position=query_position, scale=.5,
                                selected_positions=selected) for rank in range(world)]
                    merged = m.merge_partials(partials)
                    for a, b in zip(expected, merged.output):
                        self.assertAlmostEqual(a, b, places=13)
                    self.assertAlmostEqual(merged.log_normalizer, math.log(sum(weights)), places=13)

    def test_extreme_scores_empty_shards_and_nonfinite_inputs(self):
        m = load("partitioned-attention")
        empty = m.AttentionPartial((0., 0.), -math.inf)
        a = m.AttentionPartial((2., 4.), 10000.)
        b = m.AttentionPartial((10., 20.), 9999.)
        merged = m.merge_partials((a, b, empty))
        self.assertAlmostEqual(merged.output[0], (2 + 10 / math.e) / (1 + 1 / math.e))
        with self.assertRaises(ValueError):
            m.merge_partials((empty, empty))
        for lse in (math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                m.AttentionPartial((0.,), lse)
        with self.assertRaises(ValueError):
            m.AttentionPartial((1.,), -math.inf)
        with self.assertRaises(ValueError):
            m.merge_partials((a, m.AttentionPartial((1.,), 0.)))
        with self.assertRaises(ValueError):
            m.attention_partial((1.,), [(5, (math.nan,), (1.,))], value_width=1, query_position=0)

    def test_head_major_scatter_and_hierarchical_normalizers(self):
        m = load("partitioned-attention")
        shards = tuple(tuple(tuple(m.AttentionPartial((10 * rank + head + q,), float(rank))
                                   for head in range(4)) for q in range(3)) for rank in range(2))
        scattered = m.combine_and_scatter(shards)
        for rank in range(2):
            for local in range(2):
                for q in range(3):
                    head = rank * 2 + local
                    expected = m.merge_partials((shards[0][q][head], shards[1][q][head]))
                    self.assertEqual(scattered[rank][local][q], expected)
        parts = [m.AttentionPartial((float(r),), float(r)) for r in range(4)]
        hierarchical = m.merge_partials((m.merge_partials(parts[:2]), m.merge_partials(parts[2:])))
        self.assertAlmostEqual(hierarchical.output[0], m.merge_partials(parts).output[0])
        with self.assertRaises(ValueError):
            m.combine_and_scatter((shards[0], (shards[1][0][:3],)))
        # Matching widths across ranks are insufficient if another head differs.
        mixed = ((m.AttentionPartial((1.,), 0.), m.AttentionPartial((1., 2.), 0.)),)
        with self.assertRaises(ValueError):
            m.combine_and_scatter((mixed, mixed))

    def test_duplicate_positions_masks_and_immutable_outputs(self):
        m = load("partitioned-attention")
        row = (0, (1.,), (2.,))
        for rows, selected in (((row, row), None), ((row,), (0, 0))):
            with self.assertRaises(ValueError):
                m.attention_partial((1.,), rows, value_width=1, query_position=1, selected_positions=selected)
        p = m.attention_partial((1.,), (row,), value_width=1, query_position=0, selected_positions=())
        self.assertEqual(p, m.AttentionPartial((0.,), -math.inf))
        values = [2.]
        p = m.AttentionPartial(values, 0.)
        values[0] = 100.
        self.assertEqual(p.output, (2.,))


class Fp8KvReferenceTests(unittest.TestCase):
    def test_codes_round_trip_and_the_largest_element_maps_to_448(self):
        m = load("fp8-kv-attention", "reference.py")
        for code in range(256):
            value = m.decode(code)
            if not math.isnan(value) and value != 0.0:
                self.assertEqual(m.decode(m.encode(value)), value)
        self.assertEqual(m.encode(1000.0), m.encode(448.0))  # saturates
        codes, scale = m.quantize([0.5, -3.0, 1.25])
        self.assertAlmostEqual(scale, 3.0 / 448.0)
        self.assertEqual(m.decode(codes[1]) * scale, -3.0)

    def test_dequantized_values_stay_within_e4m3_precision(self):
        m, rng = load("fp8-kv-attention", "reference.py"), random.Random(5)
        vector = [rng.gauss(0.0, 1.0) for _ in range(256)]
        codes, scale = m.quantize(vector)
        for x, y in zip(vector, m.dequantize(codes, scale)):
            self.assertLessEqual(abs(x - y), abs(x) / 15 + scale * 0.002)
        self.assertEqual(m.quantize([0.0, 0.0])[1], 1.0)


class PagedAttentionReferenceTests(unittest.TestCase):
    def test_split_key_combine_equals_direct_softmax_over_paged_rows(self):
        m, rng = load("paged-attention", "reference.py"), random.Random(9)
        page_tokens, dim = 8, 4
        storage = [([rng.gauss(0, 1) for _ in range(dim)], [rng.gauss(0, 1) for _ in range(dim)]) for _ in range(512)]
        table = list(range(63, -1, -1))  # pages in reverse order
        queries = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(2)]
        for blocks in (1, 2, 5):
            got = m.attend(storage, page_tokens, table, 400, queries, 0.5, blocks)
            for i, q in enumerate(queries):
                position = 400 - 2 + i
                rows = [storage[m.physical_row(table, page_tokens, j)] for j in range(position + 1)]
                scores = [0.5 * sum(a * b for a, b in zip(q, key)) for key, _ in rows]
                top = max(scores)
                weights = [math.exp(s - top) for s in scores]
                want = [sum(w * value[d] for w, (_, value) in zip(weights, rows)) / sum(weights) for d in range(dim)]
                for x, y in zip(got[i], want):
                    self.assertAlmostEqual(x, y, places=12)

    def test_queries_need_their_own_keys(self):
        m = load("paged-attention", "reference.py")
        with self.assertRaises(ValueError):
            m.attend([([1.0], [1.0])] * 4, 4, [0], 1, [[1.0], [1.0]], 1.0)


class BatchedEngineReferenceTests(unittest.TestCase):
    def test_split_plan_ignores_rows_and_covers_k(self):
        m = load("batched-reference-engine", "reference.py")
        for n, k, sms in ((64, 64, 1), (128, 4096, 170), (11264, 2048, 170), (256, 640, 7)):
            splits, per = m.plan_gemm(n, k, sms)
            self.assertGreaterEqual(splits * per, k // 64)
            self.assertLess((splits - 1) * per, k // 64)
        with self.assertRaises(ValueError):
            m.plan_gemm(100, 64, 8)

    def test_decode_rows_come_first_and_logits_follow_phases(self):
        m = load("batched-reference-engine", "reference.py")
        layout, decode_rows, logits = m.arrange([("p", "prefill", 2, False), ("d", "decode", 1, False)])
        self.assertEqual((decode_rows, logits), (1, [0]))
        self.assertEqual(layout, [("d", "decode"), ("p", "prefill"), ("p", "prefill")])
        with self.assertRaises(ValueError):
            m.arrange([("d", "decode", 2, False)])


class GgufBlockReferenceTests(unittest.TestCase):
    # SHA-256 of the FP32 values of test_blocks(kind, 6, 0x5eed + len(kind)) as gguf 0.19.0's quants.dequantize returns them.
    DIGESTS = {"Q4_0": "52a944d0347e56eeb04f2dbd96e806d71f7ab3cdf242c1a88f92369aee862dc0",
               "Q8_0": "d0490b67f7527a093422496ba0ee467127deaf0710b10355ddada41f076be3f0",
               "Q4_K": "f0c481aeb5d28f121f3c2a9debd69aed8a87478857898010cd185a0ce64e8fc0",
               "Q5_K": "8572635116b72c2c501bcc25e82c2273e5d54cfb76d91072a1b7bc1538e5feda",
               "Q6_K": "1517224025493bd0dee5259d7a4018aaad62949ea072dfc18cba6fe638824d9b"}

    def test_decoded_blocks_equal_gguf_py(self):
        import hashlib
        import importlib.util
        import struct
        m = load("gguf-projections", "gguf_blocks.py")
        reference = None
        if importlib.util.find_spec("gguf") and importlib.util.find_spec("numpy"):
            import numpy
            from gguf import GGMLQuantizationType, quants
            reference = (numpy, GGMLQuantizationType, quants)
        for kind, digest in self.DIGESTS.items():
            data = m.test_blocks(kind, 6, 0x5eed + len(kind))
            values = m.dequantize(kind, data)
            self.assertEqual(hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest(), digest, kind)
            if reference:
                numpy, types, quants = reference
                size = m.FORMATS[kind][1]
                expected = quants.dequantize(numpy.frombuffer(data, dtype=numpy.uint8).reshape(-1, size), getattr(types, kind))
                self.assertEqual(expected.astype(numpy.float32).ravel().tobytes(), struct.pack(f"<{len(values)}f", *values), kind)

    def test_block_bounds_and_projection(self):
        m = load("gguf-projections", "gguf_blocks.py")
        with self.assertRaises(ValueError):
            m.dequantize("Q4_K", bytes(143))
        data = m.test_blocks("Q8_0", 2, 7)
        w = m.dequantize("Q8_0", data)
        x = [0.5] * 32
        self.assertEqual(m.project("Q8_0", data, 2, x), [sum(v * 0.5 for v in w[:32]), sum(v * 0.5 for v in w[32:])])


class DenseOffloadPlannerTests(unittest.TestCase):
    def test_candidates_rank_by_prediction_and_respect_the_budget(self):
        m = load("dense-layer-offload", "planner.py")
        rates = {"device_gbs": 1.0, "upload_gbs": 0.5, "host_gbs": 0.25, "host_layer_ns": 5.0}
        plans = m.candidates([10] * 8, 50, rates)
        self.assertEqual([p[0] for p in plans], ["stream", "host"])
        self.assertTrue(all(p[2] <= 50 for p in plans))
        self.assertEqual(m.candidates([10] * 8, 80, rates)[0][:2], ("resident", [m.DEVICE] * 8))
        sizes = [30, 10, 10]
        stream = [p for p in m.candidates(sizes, 40, rates) if p[0] == "stream"]
        # The largest streamed layer sets both slots: nothing else fits beside two 30-byte slots.
        self.assertFalse(stream)
        with self.assertRaises(ValueError):
            m.candidates([], 10, rates)


if __name__ == "__main__":
    unittest.main()
