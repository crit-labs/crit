"""device-speculative-round's Python driver (python_round.py) and adaptive width rule (width_policy.py): batched rounds
over a toy target with two different drafters decode exactly what plain greedy decoding does, at any widths, through
stops, output limits, truncations and row changes; replay and forced rounds follow a fixed text; sampled rounds keep
the target's distribution; and captured CUDA graphs give the eager rounds' tokens."""
import unittest

from test_techniques import load

try:
    import torch
except ImportError:   # the driver is torch's
    torch = None

V, P = 50, 1_000_003


def mix(state, token, position):
    return (state * 31 + token * 17 + position * 13 + 7) % P


class Toy:
    """A toy target: each row's state hashes every token it ingested at its position, and a column predicts its state
    modulo the vocabulary. Its deferred state is the state after each verify column; commit keeps the accepted one
    from the round's start. Two drafters: "features" catches up from the previous round's column states (as an MTP head
    from the target's hidden states) and is wrong at every seventh position; "ngram" proposes from the pending token
    alone and is mostly wrong."""

    def __init__(self, rows, widest, device, drafter="features"):
        long = dict(dtype=torch.long, device=device)
        self.state, self.start, self.drafter = torch.zeros(rows, **long), torch.zeros(rows, **long), drafter
        self.features = torch.zeros(rows, widest + 1, **long)
        self.verified = []   # the columns each verify saw, eager rounds only

    def prefill(self, slot, prompt):
        state = 0
        for position, token in enumerate(prompt):
            state = mix(state, token, position)
        self.state[slot] = state
        self.features[slot, 0] = state
        return state % V

    def draft(self, view):
        rows = view.rows
        if self.drafter == "ngram":
            for j in range(view.width):
                view.drafts[:, j] = (view.pending * 3 + j) % V
            return
        # Catch up: the target's state after the previous round's last emitted column is where drafting starts.
        source = (view.catch_up_count - 1).clamp(min=0)
        state = self.features[:rows].gather(1, source[:, None]).squeeze(1)
        token, position = view.pending, view.frontier
        for j in range(view.width):
            state = mix(state, token, position + j)
            guess = state % V
            token = torch.where((position + j) % 7 == 3, (guess + 1) % V, guess)
            view.drafts[:, j] = token

    def verify(self, view):
        rows = view.rows
        self.start[:rows] = self.state[:rows]
        columns = view.columns()
        if not torch.cuda.is_current_stream_capturing() if columns.is_cuda else True:
            self.verified.append(columns.tolist())
        state = self.state[:rows]
        for j in range(view.width + 1):
            state = mix(state, columns[:, j], view.frontier + j)
            self.features[:rows, j] = state
            view.predictions[:, j] = state % V

    def commit(self, view):
        rows = view.rows
        count = view.n_emitted
        kept = self.features[:rows].gather(1, (count - 1).clamp(min=0)[:, None]).squeeze(1)
        self.state[:rows] = torch.where(count > 0, kept, self.start[:rows])

    def snapshot(self):
        return self.state.clone(), self.features.clone()

    def restore(self, saved):
        self.state.copy_(saved[0])
        self.features.copy_(saved[1])


def plain(prompt, count, stops=()):
    """Plain greedy decoding of the toy target: `count` tokens, ending after a stop token."""
    state = 0
    for position, token in enumerate(prompt):
        state = mix(state, token, position)
    out, position = [state % V], len(prompt)
    while len(out) < count and out[-1] not in stops:
        state = mix(state, out[-1], position)
        out.append(state % V)
        position += 1
    return out


@unittest.skipIf(torch is None, "torch is not installed")
class PythonRoundTests(unittest.TestCase):
    def setUp(self):
        self.m = load("device-speculative-round", "python_round.py")
        self.policy = load("device-speculative-round", "width_policy.py")

    def decode(self, toy, rounds, prompts, counts, widths, mode="greedy", stops=(), continuations=None, truncate=None):
        """Decode every prompt in rows of one driver; widths: a list cycled per round, or a WidthPolicy."""
        outputs = []
        for slot, prompt in enumerate(prompts):
            first = toy.prefill(slot, prompt) if continuations is None else continuations[slot][0]
            rounds.start_row(slot, len(prompt), first, counts[slot] - 1, stops,
                             continuations[slot] if continuations else ())
            outputs.append([first])
        cycle = 0
        while any(len(o) < c and (not o or o[-1] not in stops) for o, c in zip(outputs, counts)):
            rows = len(prompts)
            if isinstance(widths, list):
                width = widths[cycle % len(widths)]
            else:
                width = widths.choose(range(rows))
            rounds.launch(rows, width, mode)
            found = rounds.read()
            if truncate and cycle in truncate:
                keep = [min(len(r["tokens"]), truncate[cycle]) for r in found]
                rounds.truncate(keep)
                found = [{**r, "tokens": r["tokens"][:k]} for r, k in zip(found, keep)]
            for row, result in enumerate(found):
                outputs[row] += result["tokens"]
                if not isinstance(widths, list) and result["tokens"]:
                    widths.observe(row, width, result["matched"])
            cycle += 1
            self.assertLess(cycle, 500)
        return outputs, cycle

    def test_batched_rounds_with_either_drafter_decode_plain_greedy_at_any_widths(self):
        prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11]]
        counts = [40, 23, 31, 9]
        for drafter in ("features", "ngram"):
            for widths in ([0], [1], [4], [3, 0, 6, 2, 5]):
                with self.subTest(drafter=drafter, widths=widths):
                    toy = Toy(4, 6, "cpu", drafter)
                    rounds = self.m.SpeculativeRounds(toy, 4, 6, "cpu")
                    outputs, _ = self.decode(toy, rounds, prompts, counts, widths)
                    self.assertEqual(outputs, [plain(p, c) for p, c in zip(prompts, counts)])

    def test_a_good_drafter_emits_several_tokens_per_round_and_the_policy_widens(self):
        toy = Toy(1, 8, "cpu")
        rounds = self.m.SpeculativeRounds(toy, 1, 8, "cpu")
        policy = self.policy.WidthPolicy([1, 2, 4, 8], base_ms=10.0, step_ms=0.5)
        outputs, cycles = self.decode(toy, rounds, [[3, 1, 4]], [200], policy)
        self.assertEqual(outputs, [plain([3, 1, 4], 200)])
        self.assertLess(cycles, 200 / 2.5)   # the drafter is right six positions in seven
        self.assertGreater(policy.choose([0]), 1)

    def test_stops_output_limits_and_inactive_rows(self):
        prompts = [[1, 2], [3, 4], [5]]
        reference = [plain(p, 60) for p in prompts]
        stop = reference[0][17]   # a token the first row predicts mid-round
        toy = Toy(3, 5, "cpu")
        rounds = self.m.SpeculativeRounds(toy, 3, 5, "cpu")
        outputs, _ = self.decode(toy, rounds, prompts, [60, 1, 12], [5], stops=[stop])
        self.assertEqual(outputs, [plain(p, c, [stop]) for p, c in zip(prompts, [60, 1, 12])])
        self.assertEqual(outputs[0][-1], stop)

    def test_a_truncation_keeps_fewer_tokens_and_decoding_continues_exactly(self):
        prompts = [[2, 7, 1], [8, 2]]
        toy = Toy(2, 4, "cpu")
        rounds = self.m.SpeculativeRounds(toy, 2, 4, "cpu")
        outputs, _ = self.decode(toy, rounds, prompts, [30, 30], [4], truncate={1: 1, 3: 2, 4: 0})
        self.assertEqual(outputs, [plain(p, 30) for p in prompts])

    def test_rows_move_and_release_between_rounds(self):
        toy = Toy(3, 3, "cpu")
        rounds = self.m.SpeculativeRounds(toy, 3, 3, "cpu")
        prompts = [[1], [2], [3]]
        out = [[toy.prefill(slot, p)] for slot, p in enumerate(prompts)]
        for slot, p in enumerate(prompts):
            rounds.start_row(slot, len(p), out[slot][0], 20)
        rounds.launch(3, 3)
        for row, r in enumerate(rounds.read()):
            out[row] += r["tokens"]
        # Row 0 finishes; row 2 moves into its slot with its target state, and two rows decode on.
        rounds.release(0)
        rounds.move(2, 0)
        toy.state[0], toy.features[0] = toy.state[2], toy.features[2]
        rounds.launch(2, 3)
        found = rounds.read()
        out[2] += found[0]["tokens"]
        out[1] += found[1]["tokens"]
        self.assertEqual(out[2], plain([3], len(out[2])))
        self.assertEqual(out[1], plain([2], len(out[1])))

    def test_replay_follows_a_fixed_continuation_and_forced_rounds_score_it_in_columns(self):
        prompt = [5, 6]
        greedy = plain(prompt, 30)
        text = greedy[:10] + [(t + 1) % V for t in greedy[10:]]   # departs from greedy decoding after ten tokens
        for mode in ("replay", "forced"):
            with self.subTest(mode=mode):
                toy = Toy(1, 4, "cpu")
                toy.prefill(0, prompt)
                rounds = self.m.SpeculativeRounds(toy, 1, 4, "cpu", continuation=40)
                outputs, cycles = self.decode(toy, rounds, [prompt], [30], [4], mode=mode, continuations=[text])
                self.assertEqual(outputs, [text[:30]])
                if mode == "forced":
                    self.assertEqual(cycles, 29 // 5 + 1)   # every round keeps its five columns
                    seen = [t for columns in toy.verified for t in columns[0]]
                    self.assertEqual(seen[:25], text[:25], "the verify columns are the text itself")

    def test_logprobs_are_packed_with_the_tokens(self):
        class Scored(Toy):
            def verify(self, view):
                super().verify(view)
                view.logprobs[:, :view.width + 1] = -view.predictions[:, :view.width + 1].float()
        toy = Scored(1, 3, "cpu")
        rounds = self.m.SpeculativeRounds(toy, 1, 3, "cpu", logprobs=True)
        rounds.start_row(0, 2, toy.prefill(0, [1, 2]), 10)
        rounds.launch(1, 3)
        found = rounds.read()[0]
        self.assertEqual(found["logprobs"], [-float(t) for t in found["tokens"]])

    def test_rejection_predictions_keep_the_target_distribution(self):
        generator = torch.Generator().manual_seed(7)
        rows, width = 60000, 2
        p = torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
        q = torch.tensor([[0.1, 0.1, 0.7, 0.1], [0.4, 0.2, 0.2, 0.2]])
        drafts = torch.stack([torch.multinomial(q[j], rows, replacement=True, generator=generator) for j in range(width)], 1)
        predictions = self.m.rejection_predictions(p.expand(rows, -1, -1), drafts, q.expand(rows, -1, -1),
                                                   torch.rand(rows, width, generator=generator),
                                                   torch.rand(rows, width + 1, generator=generator))
        first = torch.bincount(predictions[:, 0], minlength=4).float() / rows
        self.assertLess((first - p[0]).abs().max().item(), 0.01)
        # Where the first draft was kept, the second emitted token is again the target's at that column.
        kept = predictions[:, 0] == drafts[:, 0]
        second = torch.bincount(predictions[kept, 1], minlength=4).float() / kept.sum()
        self.assertLess((second - p[1]).abs().max().item(), 0.015)
        # A drafter whose distribution is the target's is always accepted.
        same = self.m.rejection_predictions(p[:2].expand(100, -1, -1)[:, :1].expand(100, 2, -1).contiguous(), drafts[:100, :1],
                                            p[:1].expand(100, -1, -1), torch.rand(100, 1, generator=generator),
                                            torch.rand(100, 2, generator=generator))
        self.assertTrue(bool((same[:, 0] == drafts[:100, 0]).all()))

    def test_sampled_rounds_accept_by_the_verify_hooks_rejection_predictions(self):
        """A sampled verify writes rejection-sampling predictions, which the driver accepts by its one rule: a drafter
        whose distribution is the target's keeps every draft, and the round emits them and one bonus draw."""
        m = self.m

        class Sampled(Toy):
            def verify(self, view):
                super().verify(view)
                cols = view.width + 1
                probs = torch.full((view.rows, cols, V), 1.0)
                probs.scatter_(2, view.predictions[:, :cols, None], 10.0)
                self.drafted = view.drafts[:, :view.width].tolist()
                view.predictions[:, :cols] = m.rejection_predictions(
                    probs, view.drafts[:, :view.width], probs[:, :view.width], torch.full((view.rows, view.width), 0.99),
                    torch.full((view.rows, cols), 0.5))
        toy = Sampled(2, 3, "cpu")
        rounds = m.SpeculativeRounds(toy, 2, 3, "cpu")
        for slot, prompt in enumerate([[1, 2, 3], [4]]):
            rounds.start_row(slot, len(prompt), toy.prefill(slot, prompt), 12)
        rounds.launch(2, 3, "sampled")
        for row, found in enumerate(rounds.read()):
            self.assertEqual(len(found["tokens"]), 4)
            self.assertEqual(found["tokens"][:3], toy.drafted[row])
            self.assertEqual(found["matched"], 3)

    def test_copies_override_only_rows_whose_match_is_long_and_their_draft_law_becomes_one_hot(self):
        drafts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        copies = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
        counts, matched = torch.tensor([3, 2, 3]), torch.tensor([6, 8, 3])
        probs = torch.full((3, 3, 40), 1 / 40)
        self.m.override_drafts(drafts, copies, counts, matched, 4, probs)
        self.assertEqual(drafts.tolist(), [[11, 12, 13], [21, 22, 6], [7, 8, 9]])
        self.assertEqual(probs[0, 1, 12].item(), 1.0)
        self.assertEqual(probs[0, 1].sum().item(), 1.0)
        self.assertAlmostEqual(probs[1, 2, 6].item(), 1 / 40)   # kept after a shorter copy: its own law
        self.assertAlmostEqual(probs[2, 0, 7].item(), 1 / 40)   # match below the threshold: untouched

    def test_copied_drafts_raise_acceptance_and_decoding_stays_plain_greedy(self):
        """Over the mostly wrong n-gram drafter, a row whose copies are long matches (here the true continuation, as
        a repeated passage gives) accepts every draft; a row whose match is short keeps its drafter. Both decode
        exactly what plain greedy decoding does."""
        m = self.m
        prompts, counts, width = [[3, 1, 4, 1, 5], [9, 2, 6]], [40, 40], 4
        truth = [plain(p, c + width) for p, c in zip(prompts, counts)]

        class Copying(Toy):
            def draft(self, view):
                super().draft(view)
                copies.apply(view)

        for matched, slower in ((8, False), (2, True)):
            toy = Copying(2, width, "cpu", drafter="ngram")
            copies = m.CopyDrafts(2, width, "cpu", threshold=4)
            rounds = m.SpeculativeRounds(toy, 2, width, "cpu")
            outputs = []
            for slot, prompt in enumerate(prompts):
                rounds.start_row(slot, len(prompt), toy.prefill(slot, prompt), counts[slot] - 1)
                outputs.append([truth[slot][0]])
            cycles, done = 0, [None, None]
            while any(len(o) < c for o, c in zip(outputs, counts)):
                for slot, out in enumerate(outputs):
                    copies.set(slot, truth[slot][len(out):len(out) + width], matched if slot == 0 else 2)
                rounds.launch(2, width)
                cycles += 1
                for slot, found in enumerate(rounds.read()):
                    outputs[slot] += found["tokens"]
                    if done[slot] is None and len(outputs[slot]) >= counts[slot]:
                        done[slot] = cycles
                self.assertLess(cycles, 100)
            for slot in range(2):
                self.assertEqual(outputs[slot], truth[slot][:counts[slot]])
            # Width + 1 tokens a round with every copy accepted; the n-gram drafter's rows take far more rounds.
            if not slower:
                self.assertEqual(done[0], -(-(counts[0] - 1) // (width + 1)))
            else:
                self.assertGreater(done[0], 20)
            self.assertGreater(done[1], 20)

    def test_copied_drafts_keep_the_target_distribution_in_sampled_rounds(self):
        generator = torch.Generator().manual_seed(11)
        rows, width = 60000, 2
        p = torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
        q = torch.tensor([[0.1, 0.1, 0.7, 0.1], [0.4, 0.2, 0.2, 0.2]])
        drafts = torch.stack([torch.multinomial(q[j], rows, replacement=True, generator=generator) for j in range(width)], 1)
        probs = q.expand(rows, -1, -1).clone()
        copies = torch.full((rows, width), 3)
        counts = torch.full((rows,), 1)
        matched = torch.where(torch.arange(rows) % 2 == 0, 8, 0)   # half the rows take the copied first draft
        self.m.override_drafts(drafts, copies, counts, matched, 4, probs)
        self.assertTrue(bool((drafts[0::2, 0] == 3).all()))
        predictions = self.m.rejection_predictions(p.expand(rows, -1, -1), drafts, probs,
                                                   torch.rand(rows, width, generator=generator),
                                                   torch.rand(rows, width + 1, generator=generator))
        for subset in (slice(0, None, 2), slice(1, None, 2)):
            first = torch.bincount(predictions[subset, 0], minlength=4).float() / (rows // 2)
            self.assertLess((first - p[0]).abs().max().item(), 0.015)
        # After an accepted copy, the kept neural draft is still judged by its own law: the target's again.
        kept = (predictions[0::2, 0] == 3)
        second = torch.bincount(predictions[0::2][kept, 1], minlength=4).float() / kept.sum()
        self.assertLess((second - p[1]).abs().max().item(), 0.03)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
    def test_captured_rounds_give_the_eager_rounds_tokens(self):
        prompts = [[1, 2, 3], [4, 5], [6]]
        counts = [37, 21, 29]
        results = {}
        for graphs in (False, True):
            toy = Toy(3, 6, "cuda")
            toy.reset_capture = lambda saved=toy.snapshot(), toy=toy: toy.restore(saved)
            rounds = self.m.SpeculativeRounds(toy, 3, 6, "cuda")
            for slot, prompt in enumerate(prompts):
                rounds.start_row(slot, len(prompt), toy.prefill(slot, prompt), counts[slot] - 1)
            if graphs:
                toy.reset_capture = lambda saved=toy.snapshot(), toy=toy: toy.restore(saved)
                for width in (0, 2, 6):
                    rounds.capture(3, width, "greedy")
            outputs = [[toy.state[slot].item() % V] for slot in range(3)]
            for cycle in range(40):
                rounds.launch(3, (0, 2, 6)[cycle % 3])
                for row, result in enumerate(rounds.read()):
                    outputs[row] += result["tokens"]
            results[graphs] = outputs
        self.assertEqual(results[True], results[False])
        self.assertEqual(results[False], [plain(p, c) for p, c in zip(prompts, counts)])

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
    def test_captured_rounds_apply_copied_drafts_staged_after_capture(self):
        """CopyDrafts.apply inside a captured round reads what set() staged in pinned memory before each replay."""
        m = self.m
        prompts, counts, width = [[1, 2, 3], [4, 5]], [33, 25], 4
        truth = [plain(p, c + width) for p, c in zip(prompts, counts)]
        results = {}
        for graphs in (False, True):
            copies = m.CopyDrafts(2, width, "cuda", threshold=4)

            class Copying(Toy):
                def draft(self, view):
                    super().draft(view)
                    copies.apply(view)
            toy = Copying(2, width, "cuda", drafter="ngram")
            rounds = m.SpeculativeRounds(toy, 2, width, "cuda")
            for slot, prompt in enumerate(prompts):
                rounds.start_row(slot, len(prompt), toy.prefill(slot, prompt), counts[slot] - 1)
            if graphs:
                toy.reset_capture = lambda saved=toy.snapshot(), toy=toy: toy.restore(saved)
                rounds.capture(2, width, "greedy")
            outputs = [[truth[slot][0]] for slot in range(2)]
            for _ in range(12):
                for slot, out in enumerate(outputs):
                    copies.set(slot, truth[slot][len(out):len(out) + width], 8 if slot == 0 else 1)
                rounds.launch(2, width)
                for row, result in enumerate(rounds.read()):
                    outputs[row] += result["tokens"]
            results[graphs] = outputs
        self.assertEqual(results[True], results[False])
        self.assertEqual(results[False][0][:33], truth[0][:33])
        self.assertEqual(len(results[False][0]), 33)   # every copy accepted: 1 + 8 rounds x 5 tokens, capped at 33


class WidthPolicyTests(unittest.TestCase):
    def setUp(self):
        self.m = load("device-speculative-round", "width_policy.py")

    def test_expected_tokens_follow_per_depth_acceptance_with_decay(self):
        policy = self.m.WidthPolicy([0, 1, 2, 4], decay=0.5, prior=0.5, prior_weight=0.0)
        policy.observe("a", 4, 4)
        self.assertEqual(policy.expected("a", 4), 5.0)
        policy.observe("a", 4, 1)   # the second draft was rejected, the third and fourth not tried
        # depth 1: (1 * .5 + 1) / (1 * .5 + 1) = 1; depth 2: (1 * .5 + 0) / (1 * .5 + 1) = 1/3; depths 3, 4 decayed only.
        self.assertAlmostEqual(policy.acceptance("a", 1), 1.0)
        self.assertAlmostEqual(policy.acceptance("a", 2), 1 / 3)
        self.assertAlmostEqual(policy.acceptance("a", 3), 1.0)
        self.assertAlmostEqual(policy.expected("a", 2), 1 + 1 + 1 / 3)
        with self.assertRaises(ValueError):
            policy.observe("a", 2, 3)

    def test_the_width_maximizes_tokens_per_ms_for_the_rows_together(self):
        policy = self.m.WidthPolicy([0, 2, 4, 8], prior=0.5, prior_weight=1.0, base_ms=13.8, step_ms=1.09)
        for _ in range(30):
            policy.observe("good", 8, 8)
            policy.observe("bad", 8, 0)
        self.assertEqual(policy.choose(["good"]), 8)
        self.assertEqual(policy.choose(["bad"]), 0)
        self.assertIn(policy.choose(["good", "bad"]), (2, 4, 8))
        # A costly draft step narrows the choice; measured rounds replace the configured costs.
        for width, ms in ((0, 10.0), (8, 90.0), (4, 50.0)):
            policy.observe_cost(width, 1, ms)
        self.assertAlmostEqual(policy.cost(2, 1), 30.0)
        self.assertEqual(policy.cost(2, 3), 13.8 + 2 * 1.09, "another batch size has no rounds measured yet")
        self.assertLess(policy.choose(["good"]), 8)
        policy.forget("good")
        self.assertAlmostEqual(policy.acceptance("good", 1), 0.5)


if __name__ == "__main__":
    unittest.main()
