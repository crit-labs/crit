"""Core's round driver over plugins (crit_plugins.rounds.Rounds): a toy model's verify over a RoundBatch, a toy
sampler's device predictions, a drafter drafting from the model's draft head and a pool that follows the round slots,
composed into one round. Its greedy rounds decode exactly what plain greedy decoding does at any widths, batched rows
as alone, through held rows, truncations and slot moves; it packs log-probabilities and each row's telemetry words; a
selection whose plugins lack their round facts is refused before any capture; and on a GPU its captured rounds give the
eager rounds' tokens and run as Core's native rounds, whose request facts count every round."""
import os
from pathlib import Path
import sys
import math
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

try:
    import torch
except ImportError:
    torch = None

V, P = 50, 1_000_003


def mix(state, token, position):
    return (state * 31 + token * 17 + position * 13 + 7) % P


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


class ToyModel:
    """A toy target in a model plugin's place: each slot's state hashes every token it took at its position, and a
    column's logits peak at its state modulo the vocabulary. verify reads the RoundBatch; commit_round keeps the state
    after column n_emitted - 1 from the round's start. It counts one routed expert read per column and one hit per
    row in counters of its own, which count_round adds into the round's (COUNTERS) and clears."""

    def __init__(self, rows, widest, device):
        long = dict(dtype=torch.long, device=device)
        self.state, self.start = torch.zeros(rows, **long), torch.zeros(rows, **long)
        self.features = torch.zeros(rows, widest + 1, **long)
        self.row_counts = torch.zeros(rows, 2, **long)   # as a device expert cache's row_counters: reads, hits
        self.device, self.batches = device, []

    def prefill(self, slot, prompt):
        state = 0
        for position, token in enumerate(prompt):
            state = mix(state, token, position)
        self.state[slot] = state
        self.features[slot, 0] = state
        return state % V

    def verify(self, view, batch):
        rows, cols = view.rows, view.width + 1
        self.batches.append((batch.rows, batch.width, batch.round))
        tokens, positions = batch.tokens_t.view(rows, cols), batch.positions_t.view(rows, cols)
        self.start[:rows] = self.state[:rows]
        state = self.state[:rows]
        for j in range(cols):
            state = mix(state, tokens[:, j], positions[:, j])
            self.features[:rows, j] = state
        logits = torch.zeros(rows, cols, V, dtype=torch.float32, device=self.device)
        logits.scatter_(2, (self.features[:rows, :cols] % V)[..., None], 10.0)
        self.row_counts[:rows, 0] += cols
        self.row_counts[:rows, 1] += 1
        return logits

    def count_round(self, view):
        if view.counter_names:
            view.counter("expert_reads").add_(self.row_counts[:view.rows, 0])
            view.counter("expert_hits").add_(self.row_counts[:view.rows, 1])
        self.row_counts[:view.rows].zero_()

    def commit_round(self, view):
        rows, count = view.rows, view.n_emitted
        kept = self.features[:rows].gather(1, (count - 1).clamp(min=0)[:, None]).squeeze(1)
        self.state[:rows] = torch.where(count > 0, kept, self.start[:rows])

    def move(self, source, target):
        self.state[target], self.features[target] = self.state[source], self.features[source]

    def snapshot(self):
        return self.state.clone(), self.features.clone(), self.start.clone()

    def restore(self, saved):
        for tensor, value in zip((self.state, self.features, self.start), saved):
            tensor.copy_(value)


class ToyHead:
    """The model's draft head: it catches up from the target's state after the previous round's last emitted column
    and is wrong at every seventh position."""

    def __init__(self, model):
        self.model, self.current = model, None

    def catch_up(self, view):
        source = (view.catch_up_count - 1).clamp(min=0)
        self.current = self.model.features[:view.rows].gather(1, source[:, None]).squeeze(1)

    def logits(self, depth, tokens, view):
        position = view.frontier + depth
        self.current = mix(self.current, tokens, position)
        guess = torch.where(position % 7 == 3, (self.current + 1) % V, self.current % V)
        out = torch.zeros(view.rows, V, dtype=torch.float32, device=tokens.device)
        return out.scatter_(1, guess[:, None], 1.0)


class ToyDrafter:
    """A chain drafter's device form (draft:device): one draft per depth from the head's argmax."""

    def __init__(self):
        self.commits = 0

    def draft(self, view, head, sampler):
        head.catch_up(view)
        token = view.pending
        for depth in range(view.width):
            token = head.logits(depth, token, view).argmax(-1)
            view.drafts[:, depth] = token

    def commit_round(self, view):
        self.commits += 1


class ToySampler:
    """The sampler's round form (rounds:sampler): greedy predictions, and each slot's staged rules."""

    def __init__(self):
        self.staged = {}

    def predictions_device(self, view, logits, draft_laws=None):
        return logits.argmax(-1)

    def stage(self, slot, params):
        self.staged[slot] = dict(params)


class ToyPool:
    """A pool that follows the round slots: what each slot holds, and its valid length after each round."""

    def __init__(self, rows, device, capacity=True):
        self.slots, self.lengths, self.commits = {}, torch.zeros(rows, dtype=torch.long, device=device), 0
        if not capacity:
            self.bind = lambda slot, sequence: self.slots.__setitem__(slot, (sequence, None))

    def bind(self, slot, sequence, capacity):
        self.slots[slot] = (sequence, capacity)

    def unbind(self, slot):
        self.slots.pop(slot, None)

    def commit_round(self, view):
        self.lengths[:view.rows] = view.frontier
        self.commits += 1


class ToyTree:
    """A tree verifier's accept: every row keeps nodes 0 and 2 (path [0, 2, -1] in a buffer of its own, which a
    captured round rewrites), emitting their predictions, within its allowance; a held row emits nothing."""

    def __init__(self, rows, device):
        self.path = torch.full((rows, 3), -1, dtype=torch.long, device=device)
        self.shape = torch.tensor([[0, 2, -1]], dtype=torch.long, device=device)

    def accept_device(self, view):
        rows = view.rows
        path = self.path[:rows, :view.width + 1]
        path.copy_(self.shape[:, :view.width + 1].expand(rows, -1))
        view.tree_path = path
        count = torch.minimum(torch.full_like(view.allowance, 2), view.allowance).clamp(min=0)
        count = torch.where(view.hold > 0, torch.zeros_like(count), count)
        view.n_emitted[:] = count
        view.n_matched[:] = (count - 1).clamp(min=0)
        along = view.predictions[:, :path.shape[1]].gather(1, path.clamp(min=0))
        column = torch.arange(path.shape[1], device=path.device)[None, :]
        view.emitted.fill_(-1)
        view.emitted[:, :path.shape[1]] = torch.where(column < count[:, None], along, torch.full_like(along, -1))


class TreePool(ToyPool):
    """A pool whose commit keeps a tree's accepted nodes: it reads view.tree_path at every commit."""

    def __init__(self, rows, device):
        super().__init__(rows, device)
        self.paths = []

    def commit_round(self, view):
        super().commit_round(view)
        self.paths.append(view.tree_path)


class FileTier:
    """A KV plugin that provides rounds:kv_cache, configured so that it cannot run inside a captured round."""

    def __init__(self, reasons):
        self.reasons = reasons

    def round_problems(self):
        return self.reasons


class Hooked:
    """An extra part of the round (a device expert cache): bookkeeping at each round's start, and a count of 100 hits
    a row added at its end, both inside the captured round."""

    def __init__(self, device):
        self.starts = torch.zeros(1, dtype=torch.long, device=device)

    def start_round(self, view):
        self.starts += 1

    def count_round(self, view):
        view.counter("expert_hits").add_(100)

    def reset_capture(self):
        self.starts.zero_()


class Parts:
    """A factory's parts as round_problems reads them: the plugin filling each slot, the facts it provides, and the
    plugin objects made so far (get(slot))."""

    def __init__(self, use, provides, objects=None):
        self.use, self.provides, self.objects = use, provides, dict(objects or {})

    def get(self, slot, default=None):
        return self.objects.get(slot, default)

    def slots(self):
        return list(self.use)

    def plugin(self, slot):
        return self.use.get(slot)

    def facts(self, slot):
        return tuple(self.provides.get(self.use.get(slot), ()))


ROUNDS_FACTS = {"toy-model": ["draft:head-device"], "toy-sampler": ["rounds:sampler"], "toy-drafter": ["draft:device"],
                "toy-kv": ["rounds:kv_cache"], "toy-attention": ["rounds:attention"],
                "toy-projections": ["rounds:projections"]}
COUNTERS = ("expert_reads", "expert_hits")
USE = {"model": "toy-model", "placement": "toy-placement", "sampler": "toy-sampler", "drafter": "toy-drafter",
       "kv_cache": "toy-kv", "attention": "toy-attention", "projections": "toy-projections"}


@unittest.skipIf(torch is None, "torch is not installed")
class RoundsTests(unittest.TestCase):
    def build(self, rows, widest, device="cpu", drafts=True, **options):
        from crit_plugins.rounds import Rounds
        model = ToyModel(rows, widest, device)
        self.model, self.head, self.drafter = model, ToyHead(model), ToyDrafter()
        self.sampler, self.pool = ToySampler(), ToyPool(rows, device)
        return Rounds(model, self.sampler, rows=rows, widest=widest, device=device,
                      drafter=self.drafter if drafts else None, head=self.head if drafts else None,
                      pools=[self.pool], parts=Parts(USE, ROUNDS_FACTS), counters=COUNTERS, **options)

    def decode(self, rounds, prompts, counts, widths, stops=(), hold=None, truncate=None):
        outputs = []
        for slot, prompt in enumerate(prompts):
            first = self.model.prefill(slot, prompt)
            rounds.bind(slot, ("sequence", slot), len(prompt), first, counts[slot] - 1, capacity=len(prompt) + counts[slot]
                        + rounds.widest + 1, stops=stops, params={"temperature": 0.0})
            outputs.append([first])
        cycle = 0
        while any(len(o) < c and (not o or o[-1] not in stops) for o, c in zip(outputs, counts)):
            width = widths[cycle % len(widths)]
            held = hold(cycle) if hold else ()
            rounds.launch(len(prompts), width, hold=held)
            found = rounds.read()
            if truncate and cycle in truncate:
                keep = [min(len(r["tokens"]), truncate[cycle]) for r in found]
                rounds.truncate(keep)
                found = [{**r, "tokens": r["tokens"][:k]} for r, k in zip(found, keep)]
            for row, result in enumerate(found):
                if row in held:
                    self.assertEqual(result["tokens"], [])
                outputs[row] += result["tokens"]
            cycle += 1
            self.assertLess(cycle, 500)
        return outputs, cycle

    def follow_rounds(self, follows, text, logprobs=0):
        """Rows of one greedy driver: a plain row, then a row per entry of `follows` following `text` (its first
        token the pending one, as a replay's prefill gives it). Returns each row's tokens, its tokens per round, its
        log-probability rows and the driver."""
        rounds = self.build(1 + len(follows), 3, continuation=64, logprobs=logprobs)
        prompts = [[5, 6]] + [[7, 8, 9]] * len(follows)
        outputs, counts, scores = [], [], []
        for slot, prompt in enumerate(prompts):
            first = self.model.prefill(slot, prompt)
            follow = follows[slot - 1] if slot else None
            pending = text[0] if follow else first
            rounds.bind(slot, ("sequence", slot), len(prompt), pending, len(text) - 1, capacity=128,
                        continuation=text if follow else (), follow=follow)
            outputs.append([pending])
            counts.append([])
            scores.append([])
        cycles = 0
        while any(len(o) < len(text) for o in outputs):
            rounds.launch(len(prompts), 3)
            for row, found in enumerate(rounds.read()):
                outputs[row] += found["tokens"]
                if found["tokens"]:
                    counts[row].append((len(found["tokens"]), found["matched"]))
                scores[row] += list(zip(found.get("logprobs", []), found.get("top", [])))
            cycles += 1
            self.assertLess(cycles, 100)
        return outputs, counts, scores, rounds

    def test_rows_follow_their_continuation_inside_a_greedy_round(self):
        # A replay row emits its text, its drafts (the model's own) accepted while they equal it; a forced row's text
        # is its drafts too, so it keeps width + 1 tokens a round; the plain row beside them decodes plain greedy.
        text = [11, 3, 29, 3, 7, 40, 2, 2, 19, 23, 5, 31, 8, 13, 17, 1]
        outputs, counts, scores, _ = self.follow_rounds(["replay", "forced"], text, logprobs=2)
        self.assertEqual(outputs[0], plain([5, 6], len(text)))
        self.assertEqual(outputs[1], text)
        self.assertEqual(outputs[2], text)
        self.assertEqual(counts[2], [(4, 3), (4, 3), (4, 3), (3, 2)], "a forced row keeps every column")
        self.assertTrue(all(n == matched + 1 for n, matched in counts[1]), counts[1])
        self.assertGreater(len(counts[1]), len(counts[2]), "the model's drafts rarely match the replayed text")
        # The followed text's log-probabilities under the model: the toy's logits peak at its own prediction.
        peak = 10.0 - math.log(math.exp(10.0) + V - 1)
        floor = -math.log(math.exp(10.0) + V - 1)
        self.assertEqual(len(scores[2]), len(text) - 1)
        for logprob, top in scores[2]:
            self.assertTrue(abs(logprob - peak) < 1e-4 or abs(logprob - floor) < 1e-4, logprob)
            self.assertAlmostEqual(top[0][1], peak, places=4)

    def test_a_forced_row_counts_in_neither_the_acceptance_nor_the_width_policy(self):
        text = list(range(20, 36))
        _, _, _, rounds = self.follow_rounds(["forced"], text)
        drafts = rounds.telemetry()["drafts"]
        _, _, _, alone = self.follow_rounds([], text)
        self.assertEqual(drafts["offered_by_depth"], alone.telemetry()["drafts"]["offered_by_depth"],
                         "only the plain row offered drafts")
        self.assertEqual(drafts["accepted_by_depth"], alone.telemetry()["drafts"]["accepted_by_depth"])

    def test_a_row_follows_only_a_continuation_the_driver_holds(self):
        from crit_plugins.rounds import Rounds
        rounds = self.build(1, 3)   # no continuation sizing
        with self.assertRaisesRegex(ValueError, "sized for continuations"):
            rounds.bind(0, "s", 2, 1, 4, capacity=16, continuation=[1], follow="replay")
        rounds = self.build(1, 3, continuation=8)
        with self.assertRaisesRegex(ValueError, "needs one"):
            rounds.bind(0, "s", 2, 1, 4, capacity=16, follow="forced")
        with self.assertRaisesRegex(ValueError, "one of replay, forced"):
            rounds.bind(0, "s", 2, 1, 4, capacity=16, continuation=[1, 2], follow="score")
        model = ToyModel(1, 2, "cpu")
        tree = Rounds(model, ToySampler(), rows=1, widest=2, device="cpu", drafter=ToyDrafter(), head=ToyHead(model),
                      verifier=ToyTree(1, "cpu"), pools=[ToyPool(1, "cpu")], continuation=8)
        with self.assertRaisesRegex(ValueError, "accepted by the chain"):
            tree.bind(0, "s", 2, 1, 4, capacity=16, continuation=[1, 2], follow="replay")

    def test_composed_greedy_rounds_decode_plain_greedy_at_any_widths_batched_as_alone(self):
        prompts, counts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11]], [40, 23, 31, 9]
        for widths in ([0], [1], [4], [3, 0, 6, 2, 5]):
            with self.subTest(widths=widths):
                rounds = self.build(4, 6)
                outputs, _ = self.decode(rounds, prompts, counts, widths)
                self.assertEqual(outputs, [plain(p, c) for p, c in zip(prompts, counts)])
                alone = []
                for prompt, count in zip(prompts, counts):
                    rounds = self.build(1, 6)
                    alone += self.decode(rounds, [prompt], [count], widths)[0]
                self.assertEqual(alone, outputs)
        self.assertTrue(all(is_round for _, _, is_round in self.model.batches))
        self.assertEqual(self.sampler.staged[0], {"temperature": 0.0})

    def test_a_good_head_emits_several_tokens_a_round_and_the_pool_follows_each_rows_frontier(self):
        rounds = self.build(2, 6)
        outputs, cycles = self.decode(rounds, [[3, 1, 4], [2, 7]], [120, 120], [6])
        self.assertEqual(outputs, [plain([3, 1, 4], 120), plain([2, 7], 120)])
        self.assertLess(cycles, 120 / 3)
        self.assertEqual(self.pool.lengths.tolist(), [3 + 119, 2 + 119])
        self.assertEqual(self.pool.commits, cycles)
        self.assertEqual(self.drafter.commits, cycles)
        self.assertEqual(self.pool.slots[0][0], ("sequence", 0))
        drafts = rounds.telemetry()["drafts"]
        self.assertEqual((drafts["rounds"], drafts["drafted"]), (2 * cycles, 2 * cycles * 6))
        self.assertEqual(drafts["offered_by_depth"], [2 * cycles] * 6)
        accepted = drafts["accepted_by_depth"]
        self.assertTrue(all(a >= b for a, b in zip(accepted, accepted[1:])), "a deeper draft is kept less often")
        self.assertEqual(sum(accepted) + drafts["rounds"], 2 * 119, "a row round emits its accepted drafts and one")

    def test_held_rows_emit_nothing_and_keep_their_state(self):
        prompts, counts = [[1, 2], [3, 4, 5], [6]], [30, 30, 30]
        rounds = self.build(3, 4)
        outputs, _ = self.decode(rounds, prompts, counts, [4, 2, 0], hold=lambda cycle: (1,) if cycle % 3 == 1 else ())
        self.assertEqual(outputs, [plain(p, c) for p, c in zip(prompts, counts)])
        self.assertEqual(rounds.hold.tolist(), [0, 0, 0], "a hold lasts one round")
        # A truncation of a round that held a row keeps that row's catch-up for the next round.
        rounds = self.build(3, 4)
        outputs, _ = self.decode(rounds, prompts, counts, [4], hold=lambda cycle: (1,) if cycle in (1, 4) else (),
                                 truncate={1: 1, 4: 2})
        self.assertEqual(outputs, [plain(p, c) for p, c in zip(prompts, counts)])

    def test_stops_truncations_and_output_limits(self):
        prompts = [[2, 7, 1], [8, 2]]
        stop = plain(prompts[0], 60)[17]
        rounds = self.build(2, 4)
        outputs, _ = self.decode(rounds, prompts, [60, 12], [4], stops=[stop], truncate={1: 1, 3: 2, 4: 0})
        self.assertEqual(outputs, [plain(p, c, [stop]) for p, c in zip(prompts, [60, 12])])

    def test_rows_move_between_slots_with_their_pool_bindings(self):
        rounds = self.build(3, 3)
        prompts = [[1], [2], [3]]
        out = [[self.model.prefill(slot, p)] for slot, p in enumerate(prompts)]
        for slot, p in enumerate(prompts):
            rounds.bind(slot, ("sequence", slot), len(p), out[slot][0], 20, capacity=32)
        rounds.launch(3, 3)
        for row, r in enumerate(rounds.read()):
            out[row] += r["tokens"]
        rounds.unbind(0)
        rounds.move(2, 0)
        self.model.move(2, 0)
        self.assertEqual(self.pool.slots, {0: (("sequence", 2), 32), 1: (("sequence", 1), 32)})
        self.assertEqual(set(rounds.bound), {0, 1})
        rounds.launch(2, 3)
        found = rounds.read()
        out[2] += found[0]["tokens"]
        out[1] += found[1]["tokens"]
        self.assertEqual(out[2], plain([3], len(out[2])))
        self.assertEqual(out[1], plain([2], len(out[1])))

    def test_a_pool_behind_the_loaders_proxy_binds_with_its_capacity(self):
        from crit_plugins.observe import Calls, Observed
        from crit_plugins.rounds import Rounds
        pool, calls = ToyPool(1, "cpu"), Calls()
        rounds = Rounds(ToyModel(1, 0, "cpu"), ToySampler(), rows=1, widest=0, device="cpu",
                        pools=[Observed(pool, "kv_cache", calls)])
        rounds.bind(0, "s", 2, 5, 3, capacity=9)
        self.assertEqual(pool.slots[0], ("s", 9))
        self.assertEqual(calls["kv_cache"], 1)

    def test_a_pool_with_fewer_round_rows_than_the_rounds_rows_is_refused_when_the_driver_is_built(self):
        from crit_plugins.rounds import Rounds
        pool = ToyPool(3, "cpu")
        pool.round_rows = 2      # a KV pool's round tables hold slots 0 and 1
        with self.assertRaisesRegex(ValueError, "rounds of 3 rows .* has 2 round rows"):
            Rounds(ToyModel(3, 0, "cpu"), ToySampler(), rows=3, widest=0, device="cpu", pools=[pool])
        pool.round_rows = 3
        pool.round_slots = __import__("torch").arange(3)   # another attribute of that name is the pool's own business
        Rounds(ToyModel(3, 0, "cpu"), ToySampler(), rows=3, widest=0, device="cpu", pools=[pool])
        pool.round_rows = __import__("torch").tensor(3)    # round_rows is a count, never a tensor
        with self.assertRaisesRegex(ValueError, "round_rows is the number of round rows"):
            Rounds(ToyModel(3, 0, "cpu"), ToySampler(), rows=3, widest=0, device="cpu", pools=[pool])

    def test_plain_rounds_need_no_drafter_and_a_pool_may_bind_without_capacity(self):
        from crit_plugins.rounds import Rounds
        model, sampler, pool = ToyModel(2, 0, "cpu"), ToySampler(), ToyPool(2, "cpu", capacity=False)
        self.model = model
        rounds = Rounds(model, sampler, rows=2, widest=0, device="cpu", pools=[pool],
                        parts=Parts({k: v for k, v in USE.items() if k != "drafter"}, ROUNDS_FACTS), counters=COUNTERS)
        self.assertEqual(rounds.problems, [])
        outputs, cycles = self.decode(rounds, [[5, 6], [7]], [15, 9], [0])
        self.assertEqual(outputs, [plain([5, 6], 15), plain([7], 9)])
        self.assertEqual(cycles, 14)
        self.assertEqual(pool.slots[1], (("sequence", 1), None))

    def test_the_block_carries_log_probabilities_and_each_rows_telemetry_words(self):
        rounds = self.build(2, 3, logprobs=2)
        for slot, prompt in enumerate([[1, 2], [3]]):   # row 0 asks: the round runs its log-probability form
            rounds.bind(slot, slot, len(prompt), self.model.prefill(slot, prompt), 10, capacity=16, logprobs=slot == 0)
        rounds.launch(2, 3)
        for found in rounds.read():
            count = len(found["tokens"])
            self.assertEqual((found["accepted"], found["expert_reads"], found["expert_hits"]), (count - 1, 4, 1))
            self.assertEqual(found["accepted"], found["matched"])
            self.assertEqual(len(found["logprobs"]), count)
            for token, logprob, top in zip(found["tokens"], found["logprobs"], found["top"]):
                self.assertEqual(top[0][0], token)
                self.assertAlmostEqual(top[0][1], logprob, places=5)
                self.assertGreater(logprob, -0.01)   # a logit of 10 over 49 of 0
                self.assertEqual(len(top), 2)

    def test_a_round_runs_its_log_probability_form_only_for_a_row_that_asks(self):
        # Rows that ask for nothing run the plain form (no log-probabilities packed); a row asking for them, or one
        # following its continuation, makes the round's form the log-probability one, and the plain rows beside it
        # decode the tokens they decode alone.
        prompts, counts = [[5, 6], [7, 8, 9], [4]], [14, 11, 9]
        rounds = self.build(3, 2, logprobs=2, continuation=16)
        alone, _ = self.decode(rounds, prompts, counts, [2])
        forms = rounds.telemetry()["rounds"]
        self.assertEqual((forms["forms"], forms["by_form"]["logprobs"]), (["plain", "logprobs"], 0))
        self.assertGreater(forms["by_form"]["plain"], 0)
        rounds = self.build(3, 2, logprobs=2, continuation=16)
        outputs, scored = [], []
        for slot, prompt in enumerate(prompts):
            first = self.model.prefill(slot, prompt)
            rounds.bind(slot, slot, len(prompt), first, counts[slot] - 1, capacity=32, logprobs=slot == 1)
            outputs.append([first])
        while any(len(o) < c for o, c in zip(outputs, counts)):
            rounds.launch(3, 2)
            found = rounds.read()
            scored.append(all("logprobs" in row for row in found))
            for row, result in enumerate(found):
                outputs[row] += result["tokens"]
            if len(outputs[1]) == counts[1] and 1 in rounds.bound:
                rounds.unbind(1)   # the request asking ends: its slot asks no longer
        self.assertEqual(outputs, alone, "batched with a row asking for log-probabilities, as alone")
        forms = rounds.telemetry()["rounds"]["by_form"]
        self.assertEqual(forms["logprobs"], scored.count(True))
        self.assertEqual(scored[:3], [True] * 3, "the row asking decodes in the log-probability form")
        self.assertEqual(scored[-1], False, "once it ended, the plain form")
        with self.assertRaisesRegex(ValueError, "built with logprobs"):
            self.build(1, 2).bind(0, 0, 2, 1, 4, capacity=16, logprobs=True)

    def test_a_selection_without_its_round_facts_is_refused_before_any_capture(self):
        from crit_plugins.rounds import Rounds, RoundsRefused, round_problems
        lacking = {**ROUNDS_FACTS, "toy-attention": [], "toy-drafter": []}
        problems = round_problems(Parts(USE, lacking))
        self.assertEqual(len(problems), 2)
        self.assertIn("toy-attention in attention does not provide rounds:attention", problems[0])
        self.assertIn("toy-drafter in drafter does not provide draft:device", problems[1])
        self.assertEqual(round_problems(Parts(USE, lacking), uses=["kv_cache", "projections"], drafts=False), [])
        trees = {**ROUNDS_FACTS, "toy-drafter": ["draft:device", "draft:tree"]}
        problems = round_problems(Parts(USE, trees))
        self.assertIn("verify:device", problems[0])
        self.assertIn("attention:tree", problems[1], "a tree is verified under ancestor masks")
        trees = {**trees, "toy-verifier": ["verify:device"], "toy-attention": ["rounds:attention", "attention:tree"]}
        self.assertEqual(round_problems(Parts({**USE, "verifier": "toy-verifier"}, trees)), [])
        model = ToyModel(1, 2, "cpu")
        rounds = Rounds(model, ToySampler(), rows=1, widest=2, device="cpu", drafter=ToyDrafter(), head=ToyHead(model),
                        parts=Parts(USE, lacking))
        self.assertIsNone(rounds.native_rounds("unused", [2]))
        with self.assertRaisesRegex(RoundsRefused, "toy-attention in attention"):
            rounds._refuse()
        # A refused selection still decodes eagerly, through the model's step().
        rounds.bind(0, 0, 2, model.prefill(0, [1, 2]), 5, capacity=10)
        rounds.launch(1, 2)
        tokens = [plain([1, 2], 1)[0]] + rounds.read()[0]["tokens"]
        self.assertEqual(tokens, plain([1, 2], len(tokens)))
        self.assertGreater(len(tokens), 1)

    def test_a_model_computes_its_inputs_from_the_tokens_before_each_column_in_a_device_ring(self):
        # The target mixes each token with the one before it (a 2-gram, as hashed n-gram embeddings take): the ring
        # gives every column its previous token inside the round, and keeps the accepted columns' tokens after it.
        from crit_plugins.rounds import Rounds, TokenRing

        def bigram(previous, token):
            return token * 7 + previous

        def reference(prompt, count, stops=()):
            state, seen = 0, [-1]
            for position, token in enumerate(prompt):
                state = mix(state, bigram(seen[-1], token), position)
                seen.append(token)
            out, position = [state % V], len(prompt)
            while len(out) < count:
                state = mix(state, bigram(seen[-1], out[-1]), position)
                seen.append(out[-1])
                out.append(state % V)
                position += 1
            return out

        class Bigram(ToyModel):
            def __init__(self, rows, widest, device, ring):
                super().__init__(rows, widest, device)
                self.ring = ring

            def prefill(self, slot, prompt):
                state, previous = 0, -1
                for position, token in enumerate(prompt):
                    state = mix(state, bigram(previous, token), position)
                    previous = token
                self.state[slot] = state
                self.features[slot, 0] = state
                return state % V

            def verify(self, view, batch):
                rows, cols = view.rows, view.width + 1
                tokens, positions = batch.tokens_t.view(rows, cols), batch.positions_t.view(rows, cols)
                pairs = self.ring.windows(tokens, 2)
                self.start[:rows] = self.state[:rows]
                state = self.state[:rows]
                for j in range(cols):
                    state = mix(state, bigram(pairs[:, j, 0], pairs[:, j, 1]), positions[:, j])
                    self.features[:rows, j] = state
                logits = torch.zeros(rows, cols, V, dtype=torch.float32, device=self.device)
                return logits.scatter_(2, (self.features[:rows, :cols] % V)[..., None], 10.0)

        prompts, counts = [[1, 2, 3], [4, 5, 6, 7, 8], [9]], [30, 21, 25]
        for widths, truncate in (([0], None), ([3], None), ([3, 0, 2], {1: 1, 4: 0})):
            ring = TokenRing(3, 4, "cpu")
            model = Bigram(3, 3, "cpu", ring)
            self.model = model
            rounds = Rounds(model, ToySampler(), rows=3, widest=3, device="cpu", drafter=ToyDrafter(),
                            head=ToyHead(model), pools=[ring], parts=Parts(USE, ROUNDS_FACTS))
            outputs = []
            for slot, prompt in enumerate(prompts):
                first = model.prefill(slot, prompt)
                rounds.bind(slot, slot, len(prompt), first, counts[slot] - 1, capacity=64, tokens=prompt + [first])
                outputs.append([first])
            cycle = 0
            while any(len(o) < c for o, c in zip(outputs, counts)):
                rounds.launch(3, widths[cycle % len(widths)])
                found = rounds.read()
                if truncate and cycle in truncate:
                    keep = [min(len(r["tokens"]), truncate[cycle]) for r in found]
                    rounds.truncate(keep)
                    found = [{**r, "tokens": r["tokens"][:k]} for r, k in zip(found, keep)]
                for row, result in enumerate(found):
                    outputs[row] += result["tokens"]
                cycle += 1
            with self.subTest(widths=widths, truncate=truncate):
                self.assertEqual(outputs, [reference(p, c) for p, c in zip(prompts, counts)])

    def test_a_native_round_adopted_after_a_held_round_truncates_without_the_old_hold(self):
        rounds = self.build(2, 3)
        for slot, prompt in enumerate([[1, 2, 3], [4, 5]]):
            rounds.bind(slot, slot, len(prompt), self.model.prefill(slot, prompt), 40, capacity=64)
        rounds.launch(2, 3, hold=(1,))   # a step() round that holds row 1
        rounds.read()
        self.assertIsNotNone(rounds._last_held)
        rounds._round(2, 3, "greedy", False)    # what a native round's replay leaves: the block, no read()
        rounds.adopt(2, 3)
        self.assertIsNone(rounds._last_held, "native rounds hold no row")
        rounds.truncate([1, 1])
        self.assertEqual(rounds.catch_up_count.tolist(), [1, 1])
        self.assertEqual(rounds.n_emitted.tolist(), [1, 1])

    def test_the_driver_advances_from_what_a_verifier_accepted_and_truncates_a_prefix_of_it(self):
        # A verifier that accepts another path than the predictions' prefix (as a tree's): the driver's state follows
        # its emitted tokens, and a truncation keeps a prefix of them.
        class PathVerifier:
            def accept_device(self, view):
                rows = view.rows
                view.n_emitted[:] = 2
                view.n_matched[:] = 1
                view.emitted.fill_(-1)
                view.emitted[:, 0] = 7
                view.emitted[:, 1] = 9 + torch.arange(rows)
        from crit_plugins.rounds import Rounds
        model, pool = ToyModel(2, 2, "cpu"), ToyPool(2, "cpu")
        rounds = Rounds(model, ToySampler(), rows=2, widest=2, device="cpu", drafter=ToyDrafter(), head=ToyHead(model),
                        verifier=PathVerifier(), pools=[pool])
        for slot, prompt in enumerate([[1, 2], [3]]):
            rounds.bind(slot, slot, len(prompt), model.prefill(slot, prompt), 10, capacity=32)
        rounds.launch(2, 2)
        found = rounds.read()
        self.assertEqual([r["tokens"] for r in found], [[7, 9], [7, 10]])
        self.assertEqual(rounds.pending.tolist(), [9, 10])
        self.assertEqual(rounds.frontier.tolist(), [4, 3])
        self.assertEqual(rounds.out_count.tolist(), [3, 3])
        self.assertEqual(rounds.catch_up[:, :2].tolist(), [[7, 9], [7, 10]])
        self.assertEqual(pool.lengths.tolist(), [4, 3])
        rounds.truncate([1, 2])
        self.assertEqual(rounds.pending.tolist(), [7, 10])
        self.assertEqual(rounds.emitted[:, :2].tolist(), [[7, -1], [7, 10]])
        self.assertEqual(pool.lengths.tolist(), [3, 3], "the pools commit again from the round's start")

    def test_hooks_draft_at_every_width_and_a_row_starts_at_its_output_count(self):
        from crit_plugins.rounds import SpeculativeRounds

        class Hooks(ToyModel):
            drafts = 0

            def draft(self, view):
                Hooks.drafts += 1

            def verify(self, view):
                view.predictions[:, :view.width + 1] = 3

            def commit(self, view):
                pass
        rounds = SpeculativeRounds(Hooks(1, 2, "cpu"), 1, 2, "cpu")
        rounds.start_row(0, 5, 1, 10, outputs=4)
        self.assertEqual(rounds.out_count.tolist(), [4])
        rounds.launch(1, 0)
        rounds.read()
        self.assertEqual(Hooks.drafts, 1, "a width-0 round calls the draft hook too")
        self.assertEqual(rounds.out_count.tolist(), [5])
        with self.assertRaises(ValueError):
            rounds.start_row(0, 5, 1, 10, outputs=0)

    def test_the_driver_reserves_seeds_and_takes_the_samplers_laws(self):
        from crit_plugins.rounds import Rounds
        calls = []

        class Sampler(ToySampler):
            def reserve(self, rows, widest, vocab, device, logprobs=0):
                calls.append(("sampler", rows, widest, vocab))

            def laws(self, rows, width):
                return ("laws", rows, width)

        class Drafter(ToyDrafter):
            def reserve(self, rows, widest, vocab, device):
                calls.append(("drafter", rows, widest, vocab))

            def seed(self, slot, tokens):
                calls.append(("seed", slot, list(tokens)))

        from crit_plugins.rounds import TokenRing
        model, ring = ToyModel(2, 3, "cpu"), TokenRing(2, 4, "cpu")
        rounds = Rounds(model, Sampler(), rows=2, widest=3, device="cpu", drafter=Drafter(), head=ToyHead(model),
                        pools=[ring], vocabulary=V, sampled=True)
        self.assertEqual(calls[:2], [("sampler", 2, 3, V), ("drafter", 2, 3, V)])
        self.assertEqual(rounds.view(2, 3, "sampled").draft_laws, ("laws", 2, 3))
        rounds.bind(1, "s", 3, 8, 5, capacity=16, tokens=[4, 5, 6, 8])
        self.assertEqual(calls[2], ("seed", 1, [4, 5, 6, 8]))
        self.assertEqual(ring.ring[1].tolist(), [-1, 4, 5, 6], "the ring keeps the tokens before the pending one")
        with self.assertRaises(ValueError):
            rounds.bind(0, "t", 3, 8, 5, capacity=16, tokens=[4, 5, 8])   # not positions 0 .. frontier
        with self.assertRaises(ValueError):
            Rounds(model, ToySampler(), rows=2, widest=3, device="cpu", sampled=True)   # no laws to draw drafts by

    def test_alternatives_rank_as_core_ranks_them_and_finished_rows_count_no_rounds(self):
        rounds = self.build(2, 2, logprobs=3)

        def tied(view, batch):
            logits = torch.zeros(view.rows, view.width + 1, V)
            logits[..., 9] = logits[..., 4] = logits[..., 30] = 5.0   # three equal best: ranked 4, 9, 30
            return logits
        self.model.verify = tied
        for slot, prompt in enumerate([[1, 2], [3]]):
            rounds.bind(slot, slot, len(prompt), self.model.prefill(slot, prompt), 8 if slot == 0 else 0, capacity=32,
                        logprobs=True)
        rounds.launch(2, 2)
        found = rounds.read()
        self.assertEqual([token for token, _ in found[0]["top"][0]], [4, 9, 30])
        self.assertEqual(found[1]["tokens"], [], "a row whose allowance is spent emits nothing")
        drafts = rounds.telemetry()["drafts"]
        self.assertEqual((drafts["rounds"], drafts["offered_by_depth"]), (1, [1, 1]))

    def test_log_probabilities_follow_a_trees_accepted_path(self):
        # A tree verifier emits the predictions of the nodes on its accepted path (view.tree_path): each emitted
        # token's log-probability is that node's, in emitted order, not the column's at the token's index.
        from crit_plugins.rounds import Rounds

        class Tree:
            def accept_device(self, view):
                path = torch.tensor([[0, 2, -1]]).expand(view.rows, -1)
                view.tree_path = path
                view.n_emitted[:] = 2
                view.n_matched[:] = 1
                view.emitted.fill_(-1)
                view.emitted[:, :3] = torch.where(path >= 0, view.predictions[:, :3].gather(1, path.clamp(min=0)), -1)

        def verify(view, batch):
            logits = torch.zeros(view.rows, view.width + 1, V)
            for column in range(view.width + 1):
                logits[:, column, 3 * (column + 1)] = 2.0 + column   # column c predicts 3(c + 1) at its own odds
            return logits
        model = ToyModel(1, 2, "cpu")
        model.verify = verify
        rounds = Rounds(model, ToySampler(), rows=1, widest=2, device="cpu", drafter=ToyDrafter(), head=ToyHead(model),
                        verifier=Tree(), pools=[ToyPool(1, "cpu")], logprobs=1)
        rounds.bind(0, 0, 2, model.prefill(0, [1, 2]), 8, capacity=16, logprobs=True)
        rounds.launch(1, 2)
        found = rounds.read()[0]
        self.assertEqual(found["tokens"], [3, 9])
        columns = verify(rounds.view(1, 2, "greedy"), None)[0].log_softmax(-1)
        self.assertAlmostEqual(found["logprobs"][0], float(columns[0, 3]), places=5)
        self.assertAlmostEqual(found["logprobs"][1], float(columns[2, 9]), places=5, msg="node 2's, not column 1's")
        self.assertEqual(found["top"][1][0][0], 9)

    def test_a_plugin_whose_configuration_keeps_it_out_of_a_round_is_refused_before_any_capture(self):
        from crit_plugins.manifest import PluginError
        from crit_plugins.rounds import Rounds, round_problems
        kv = FileTier(["its pages are read from a file tier"])
        parts = Parts(USE, ROUNDS_FACTS, {"kv_cache": kv, "attention": object()})
        refused = ["toy-kv in kv_cache cannot run inside a captured round as configured: its pages are read from a "
                   "file tier"]
        self.assertEqual(round_problems(parts), refused)
        model = ToyModel(1, 2, "cpu")
        rounds = Rounds(model, ToySampler(), rows=1, widest=2, device="cpu", drafter=ToyDrafter(), head=ToyHead(model),
                        pools=[ToyPool(1, "cpu")], parts=parts)
        self.assertEqual(rounds.problems, refused, "known when the driver is built")
        self.assertIsNone(rounds.native_rounds(None, [2]))
        kv.reasons = []
        self.assertEqual(round_problems(parts), [])
        # A slot the round does not call is not asked; an answer that is no list of reasons is the plugin's error.
        kv.reasons = ["unused here"]
        self.assertEqual(round_problems(parts, uses=["attention", "projections"], drafts=False), [])
        kv.reasons = None
        with self.assertRaisesRegex(PluginError, "toy-kv's round_problems"):
            round_problems(parts)
        # The loader's parts hand out counting proxies: the plugin is asked past them, which counts no call.
        from crit_plugins.observe import Calls, Observed
        calls = Calls()
        proxied = Parts(USE, ROUNDS_FACTS, {"kv_cache": Observed(FileTier(["a file tier"]), "kv_cache", calls)})
        self.assertEqual(len(round_problems(proxied)), 1)
        self.assertEqual(sum(calls.values()), 0)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
    def test_a_reason_given_after_the_driver_is_built_refuses_its_captures(self):
        from crit_plugins.rounds import Rounds, RoundsRefused
        kv = FileTier([])
        model = ToyModel(1, 2, "cuda")
        model.reset_capture = lambda saved=model.snapshot(), m=model: m.restore(saved)
        rounds = Rounds(model, ToySampler(), rows=1, widest=2, device="cuda", drafter=ToyDrafter(), head=ToyHead(model),
                        pools=[ToyPool(1, "cuda")], parts=Parts(USE, ROUNDS_FACTS, {"kv_cache": kv}))
        self.assertEqual(rounds.problems, [])
        kv.reasons = ["its pages are read from a file tier"]
        with self.assertRaisesRegex(RoundsRefused, "file tier"):
            rounds.capture(1, 2, "greedy")
        self.assertEqual(rounds.graphs, {})
        self.assertIsNone(rounds.native_rounds(None, [2]), "the model decodes through step()")
        kv.reasons = []
        rounds.capture(1, 2, "greedy")
        self.assertIn((1, 2, "greedy", False), rounds.graphs)

    def test_every_part_starts_and_counts_each_round_once(self):
        from crit_plugins.rounds import Rounds
        calls = []

        def hooked(part, name):
            part.start_round = lambda view: calls.append(("start", name))
            part.count_round = lambda view: calls.append(("count", name))
            return part
        model = ToyModel(1, 2, "cpu")
        sampler, drafter, head = ToySampler(), ToyDrafter(), ToyHead(model)
        pool, extra = ToyPool(1, "cpu"), type("Extra", (), {})()
        for part, name in ((sampler, "sampler"), (drafter, "drafter"), (head, "head"), (pool, "pool"),
                           (extra, "extra")):
            hooked(part, name)
        rounds = Rounds(model, sampler, rows=1, widest=2, device="cpu", drafter=drafter, head=head,
                        pools=[pool, pool], extra_parts=[extra], counters=COUNTERS)
        rounds.bind(0, 0, 2, model.prefill(0, [1, 2]), 8, capacity=16)
        rounds.launch(1, 2)
        rounds.read()
        names = ["sampler", "drafter", "head", "pool", "extra"]
        self.assertEqual([c for c in calls if c[0] == "start"], [("start", n) for n in names], "once each: the pool too")
        self.assertEqual([c for c in calls if c[0] == "count"], [("count", n) for n in names])
        self.assertLess(calls.index(("start", "extra")), calls.index(("count", "sampler")))

    def test_a_truncation_commits_again_over_the_rounds_tree_path(self):
        from crit_plugins.rounds import Rounds
        model, pool = ToyModel(1, 2, "cpu"), TreePool(1, "cpu")
        rounds = Rounds(model, ToySampler(), rows=1, widest=2, device="cpu", drafter=ToyDrafter(), head=ToyHead(model),
                        verifier=ToyTree(1, "cpu"), pools=[pool])
        rounds.bind(0, 0, 2, model.prefill(0, [1, 2]), 8, capacity=16)
        rounds.launch(1, 2)
        self.assertEqual(len(rounds.read()[0]["tokens"]), 2)
        rounds.truncate([1])
        self.assertEqual(len(pool.paths), 2)
        self.assertIsNotNone(pool.paths[1], "the truncation's view carries the tree path the verifier set")
        self.assertEqual(pool.paths[1].tolist(), [[0, 2, -1]])
        self.assertEqual((rounds.n_emitted.tolist(), rounds.frontier.tolist()), ([1], [3]))

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
    def test_a_captured_round_runs_every_parts_hooks_and_a_truncation_reads_its_tree(self):
        from crit_plugins.rounds import Rounds
        model, pool, extra = ToyModel(2, 2, "cuda"), TreePool(2, "cuda"), Hooked("cuda")
        model.reset_capture = lambda saved=model.snapshot(), m=model: m.restore(saved)
        rounds = Rounds(model, ToySampler(), rows=2, widest=2, device="cuda", drafter=ToyDrafter(), head=ToyHead(model),
                        verifier=ToyTree(2, "cuda"), pools=[pool], extra_parts=[extra], counters=COUNTERS)
        for slot, prompt in enumerate([[1, 2], [3]]):
            rounds.bind(slot, slot, len(prompt), model.prefill(slot, prompt), 20, capacity=32)
        rounds.capture(2, 2, "greedy")
        commits = len(pool.paths)
        for _ in range(3):
            rounds.launch(2, 2)
            found = rounds.read()
            self.assertEqual([f["expert_hits"] for f in found], [101, 101], "the model's hit and the extra part's 100")
            self.assertEqual([len(f["tokens"]) for f in found], [2, 2])
        self.assertEqual(len(pool.paths), commits, "replayed rounds run no Python")
        self.assertEqual(int(extra.starts.item()), 3, "start_round ran inside each replayed round")
        rounds.truncate([1, 2])
        self.assertEqual(pool.paths[-1].tolist(), [[0, 2, -1], [0, 2, -1]],
                         "the truncation's commit reads the captured round's tree path")
        self.assertEqual(rounds.n_emitted.tolist(), [1, 2])

    def test_a_sampler_or_drafter_that_reserves_needs_the_vocabulary(self):
        from crit_plugins.rounds import Rounds

        class Sampler(ToySampler):
            def reserve(self, rows, widest, vocab, device, logprobs=0):
                pass
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            Rounds(ToyModel(1, 0, "cpu"), Sampler(), rows=1, widest=0, device="cpu")

    def test_verify_returns_one_logits_row_per_column(self):
        rounds = self.build(1, 2)
        self.model.verify = lambda view, batch: torch.zeros(view.rows * (view.width + 1), V)
        rounds.bind(0, 0, 2, self.model.prefill(0, [1, 2]), 5, capacity=16)
        with self.assertRaisesRegex(ValueError, "logits of every column"):
            rounds.launch(1, 2)
        with self.assertRaisesRegex(ValueError, "a list"):
            from crit_plugins.rounds import Rounds
            Rounds(self.model, ToySampler(), rows=1, widest=2, device="cpu", pools={"kv": self.pool})

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
    def test_a_replayed_round_counts_as_the_calls_its_capture_made(self):
        # The loader counts each slot's calls through its proxies; a captured round makes none when it replays, so the
        # driver counts each replay as one call of every slot its capture called.
        from crit_plugins.observe import Calls
        from crit_plugins.rounds import Rounds
        calls = Calls()

        class Observed(Parts):
            observer = calls
        model = ToyModel(1, 2, "cuda")
        model.reset_capture = lambda saved=model.snapshot(), m=model: m.restore(saved)
        verify = model.verify

        def counted(view, batch):   # as a proxy counts the model's call to its projections plugin
            calls["projections"] += 1
            return verify(view, batch)
        model.verify = counted
        rounds = Rounds(model, ToySampler(), rows=1, widest=2, device="cuda", drafter=ToyDrafter(), head=ToyHead(model),
                        pools=[ToyPool(1, "cuda")], parts=Observed(USE, ROUNDS_FACTS))
        rounds.bind(0, 0, 2, model.prefill(0, [1, 2]), 8, capacity=16)
        rounds.capture(1, 2, "greedy")
        self.assertEqual(rounds.graph_calls[(1, 2, "greedy", False)], ["projections"])
        calls.reset()
        rounds.launch(1, 2)
        rounds.read()
        self.assertEqual(dict(calls), {"projections": 1})
        calls.stop()   # the loader's checks are done: replays count nothing
        rounds.launch(1, 2)
        rounds.read()
        self.assertEqual(dict(calls), {"projections": 1})

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
    def test_a_truncation_after_a_replayed_smaller_round_commits_that_rounds_batch(self):
        from crit_plugins.rounds import Rounds, TokenRing
        ring = TokenRing(2, 4, "cuda")
        model = ToyModel(2, 2, "cuda")
        plans = []

        class Placement:
            def plan(self, tensors, budget):
                plans.append(dict(tensors))
                return {name: "device" for name in tensors}

        class Planned(Parts):
            def get(self, slot):
                return Placement() if slot == "placement" else None
        rounds = Rounds(model, ToySampler(), rows=2, widest=2, device="cuda", drafter=ToyDrafter(), head=ToyHead(model),
                        pools=[ring], parts=Planned(USE, ROUNDS_FACTS))
        model.reset_capture = lambda saved=model.snapshot(), m=model: m.restore(saved)
        rounds.capture(1, 2, "greedy")
        rounds.capture(2, 2, "greedy")   # the batch's host attributes: 2 rows now
        self.assertTrue(plans and all(set(p) == {"rounds:captures"} for p in plans))
        self.assertEqual(sum(p["rounds:captures"] for p in plans), rounds.captured_bytes)
        prompt = [1, 2, 3]
        first = model.prefill(0, prompt)
        rounds.bind(0, 0, len(prompt), first, 20, capacity=64, tokens=prompt + [first])
        rounds.launch(1, 2)   # the (1, 2) graph: no eager fill
        found = rounds.read()
        rounds.truncate([1])
        self.assertEqual(ring.ring[0].tolist(), [1, 2, 3, first], "the ring took the round's first column alone")
        self.assertEqual(rounds.pending.tolist()[0], found[0]["tokens"][0])

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
    def test_captured_composed_rounds_give_the_eager_rounds_tokens(self):
        prompts, counts = [[1, 2, 3], [4, 5], [6]], [37, 21, 29]
        results = {}
        for graphs in (False, True):
            rounds = self.build(3, 6, device="cuda")
            if graphs:
                self.model.reset_capture = lambda saved=self.model.snapshot(), m=self.model: m.restore(saved)
                for width in (0, 2, 6):
                    rounds.capture(3, width, "greedy")
                self.assertEqual(self.pool.lengths.tolist(), [0, 0, 0], "the warm-ups hold every row")
            results[graphs] = self.decode(rounds, prompts, counts, [0, 2, 6])[0]
        self.assertEqual(results[True], results[False])
        self.assertEqual(results[False], [plain(p, c) for p, c in zip(prompts, counts)])


@unittest.skipIf(torch is None or not torch.cuda.is_available() or not os.environ.get("CRIT_CORE_STEP_LIBRARY"),
                 "a CUDA device and Core's step library")
class NativeLogprobReplayTests(unittest.TestCase):
    """Core's native rounds serve a request asking for log-probabilities (the driver's buffers, through Core's output
    path) and a replay (its row follows its continuation inside the round) beside plain requests: the same tokens and
    log-probabilities as the model's own step() rounds give, with their decode rows in native rounds."""

    TEXT = [11, 3, 29, 3, 7, 40, 2, 2, 19, 23, 5, 31, 8, 13, 17]

    def serve(self, native, requests=None):
        from crit_core.native import StepEngine, StepResult
        from crit_plugins.rounds import Rounds
        rows, widest, width = 3, 3, 3
        lse = math.log(math.exp(10.0) + V - 1)

        class Session:
            def __init__(self, request):
                self.request, self.prompt, self.output, self.slot = request, list(request.prompt), request.output_tokens, None

        class Model:
            def __init__(self):
                self.toy = ToyModel(rows, widest, "cuda")
                self.rounds = Rounds(self.toy, ToySampler(), rows=rows, widest=widest, device="cuda",
                                     drafter=ToyDrafter(), head=ToyHead(self.toy), pools=[ToyPool(rows, "cuda")],
                                     parts=Parts(USE, ROUNDS_FACTS), counters=COUNTERS, logprobs=2, continuation=64)
                self.toy.reset_capture = lambda saved=self.toy.snapshot(), m=self.toy: m.restore(saved)
                self.slots, self.decode_steps = [], 0

            def info(self):
                return {"implementation": "toy rounds with log-probabilities", "vocabulary": V, "context": 4096,
                        "max_concurrency": rows, "max_row_tokens": widest + 1, "decode_token_work": widest + 1,
                        "max_batch_tokens": 0, "round_counters": list(COUNTERS), "token_logprobs": True,
                        "max_top_logprobs": 2}

            def telemetry(self):
                return self.rounds.telemetry()

            def native_rounds(self, library):
                return self.rounds.native_rounds(library, [width] * rows) if native else None

            def open(self, request):
                return Session(request)

            def step(self, scheduled):
                results = {}
                for r in scheduled:
                    if r.prefill_complete:
                        continue
                    s = r.session
                    s.slot = len(self.slots)
                    self.slots.append(s)
                    first = self.toy.prefill(s.slot, s.prompt)
                    replay = list(s.request.replay or [])
                    pending = replay[0] if replay else first
                    self.rounds.bind(s.slot, s, len(s.prompt), pending, s.output - 1, capacity=len(s.prompt) + s.output
                                     + widest + 1, session=s, continuation=replay, follow="replay" if replay else None,
                                     logprobs=s.request.logprobs is not None)
                    rows_ = None
                    if s.request.logprobs is not None:   # the toy's logits peak at its prediction, the rest equal
                        alternative = 0 if first else 1
                        rows_ = [(10.0 - lse if pending == first else -lse, [(first, 10.0 - lse), (alternative, -lse)])]
                    results[id(s)] = StepResult(len(s.prompt), [pending], True, 0, 0, rows_)
                decode = [r for r in scheduled if r.prefill_complete]
                if decode:
                    self.decode_steps += 1
                    held = sorted(set(range(len(self.slots))) - {r.session.slot for r in decode})
                    self.rounds.launch(len(self.slots), width, hold=held)
                    found = self.rounds.read()
                    for r in decode:
                        row = found[r.session.slot]
                        asked = r.session.request.logprobs is not None
                        results[id(r.session)] = StepResult(0, row["tokens"], True, 0, 0,
                                                            list(zip(row["logprobs"], row["top"])) if asked else None,
                                                            width)
                return [results[id(r.session)] for r in scheduled]

            def commit(self, decisions):
                native_ = self.rounds.native
                if native_ is not None and native_.last[2]:
                    self.rounds.adopt(*native_.last[:2])

            def retire(self, session, finish):
                slot, last = session.slot, len(self.slots) - 1
                self.rounds.unbind(slot)
                if slot != last:
                    moved = self.slots[last]
                    self.rounds.move(last, slot)
                    self.toy.move(last, slot)
                    moved.slot = slot
                    self.slots[slot] = moved
                self.slots.pop()

        model = Model()
        engine = StepEngine(model)
        requests = requests or [([1, 2, 3], 20, {"logprobs": 2}), ([4, 5, 6, 7], 12, {}),
                                ([9], len(self.TEXT), {"logprobs": 2, "replay": self.TEXT})]
        try:
            def generate(prompt, count, options):
                rows = []
                tokens, _ = engine.generate(prompt, count, stop_tokens=[], on_tokens=lambda found, more=None: rows.extend(
                    more or []), **options)
                return list(tokens), rows
            with ThreadPoolExecutor(len(requests)) as pool:
                found = list(pool.map(lambda request: generate(*request), requests))
            self.telemetry = engine.telemetry()
        finally:
            engine.close()
        return found, model.decode_steps

    def test_a_plain_requests_native_rounds_launch_the_plain_capture(self):
        # The captured rounds count their own form: a plain request alone runs the plain graph only, and one asking
        # for log-probabilities the other.
        (found, _), = [self.serve(True, [([4, 5, 6, 7], 12, {})])]
        self.assertEqual(found[0][0], plain([4, 5, 6, 7], 12))
        by_form = self.telemetry["rounds"]["by_form"]
        self.assertEqual(by_form["logprobs"], 0)
        self.assertGreater(by_form["plain"], 0)
        self.assertEqual(self.telemetry["core"]["native_logprob_rounds"], 0)
        self.assertEqual(self.telemetry["core"]["native_rounds"], by_form["plain"])
        found, _ = self.serve(True, [([1, 2, 3], 9, {"logprobs": 2})])
        self.assertEqual(self.telemetry["rounds"]["by_form"]["plain"], 0)
        self.assertEqual(self.telemetry["core"]["native_logprob_rounds"], self.telemetry["core"]["native_rounds"])

    def test_native_rounds_serve_log_probabilities_and_replays(self):
        (python, python_steps), (native, native_steps) = self.serve(False), self.serve(True)
        self.assertEqual([tokens for tokens, _ in native], [plain([1, 2, 3], 20), plain([4, 5, 6, 7], 12), self.TEXT])
        self.assertEqual([tokens for tokens, _ in native], [tokens for tokens, _ in python])
        for (_, found), (_, expected) in zip(native, python):
            self.assertEqual(len(found), len(expected))
            for (logprob, top), (want, want_top) in zip(found, expected):
                self.assertAlmostEqual(logprob, want, places=4)
                self.assertEqual([t for t, _ in top], [t for t, _ in want_top])
        self.assertEqual(len(native[0][1]), 20, "every token of the request asking for log-probabilities")
        self.assertEqual(len(native[2][1]), len(self.TEXT), "and of the replay")
        self.assertLess(native_steps, python_steps / 3, (native_steps, python_steps))


@unittest.skipIf(torch is None or not torch.cuda.is_available() or not os.environ.get("CRIT_CORE_STEP_LIBRARY"),
                 "native rounds need CUDA and Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY)")
class NativeComposedRoundTests(unittest.TestCase):
    """The composed driver handed to Core: a toy step model whose decode rows all run as Core's native rounds, whose
    tokens equal plain greedy decoding, and whose results carry Core's facts of every round (accepted and drafted per
    round from the block's telemetry words, the expert reads and hits) with the model's request_facts merged in."""

    def serve(self, native, listen, policy=None):
        from crit_core.native import StepEngine, StepResult
        from crit_plugins.rounds import Rounds
        rows, widest, width = 3, 4, 3
        widths = (1, 2, 3, 4) if policy is not None else None

        class Session:
            def __init__(self, request):
                self.prompt, self.output, self.slot, self.step_rounds = list(request.prompt), request.output_tokens, None, 0

        class Model:
            def __init__(self):
                self.toy = ToyModel(rows, widest, "cuda")
                self.rounds = Rounds(self.toy, ToySampler(), rows=rows, widest=widest, device="cuda",
                                     drafter=ToyDrafter(), head=ToyHead(self.toy), pools=[ToyPool(rows, "cuda")],
                                     parts=Parts(USE, ROUNDS_FACTS), counters=COUNTERS, widths=widths,
                                     width_policy=policy)
                self.toy.reset_capture = lambda saved=self.toy.snapshot(), m=self.toy: m.restore(saved)
                self.slots, self.decode_steps = [], 0

            def info(self):
                return {"implementation": "toy composed rounds", "vocabulary": V, "context": 4096,
                        "max_concurrency": rows, "max_row_tokens": widest + 1, "decode_token_work": widest + 1,
                        "max_batch_tokens": 0, "round_counters": list(COUNTERS)}

            def native_rounds(self, library):
                return self.rounds.native_rounds(library, [width] * rows) if native else None

            def open(self, request):
                return Session(request)

            def step(self, scheduled):
                results = {}
                for r in scheduled:
                    if r.prefill_complete:
                        continue
                    s = r.session
                    s.slot = len(self.slots)
                    self.slots.append(s)
                    first = self.toy.prefill(s.slot, s.prompt)
                    self.rounds.bind(s.slot, s, len(s.prompt), first, s.output - 1, capacity=len(s.prompt) + s.output +
                                     widest + 1, session=s)
                    results[id(s)] = StepResult(len(s.prompt), [first], True)
                decode = [r for r in scheduled if r.prefill_complete]
                if decode:
                    self.decode_steps += 1
                    held = sorted(set(range(len(self.slots))) - {r.session.slot for r in decode})
                    now = self.rounds.width(len(self.slots)) if policy is not None else width
                    self.rounds.launch(len(self.slots), now, hold=held)
                    found = self.rounds.read()
                    for r in decode:
                        r.session.step_rounds += 1
                        results[id(r.session)] = StepResult(0, found[r.session.slot]["tokens"], True, drafted=now)
                return [results[id(r.session)] for r in scheduled]

            def commit(self, decisions):
                # The device's allowance ends each row where Core does (no stop tokens): nothing to truncate.
                native = self.rounds.native
                if native is not None and native.last[2]:
                    self.rounds.adopt(*native.last[:2])

            def request_facts(self, session):
                return {"verify_ms": 1.5, "expert_reads": 1000, "step_rounds": session.step_rounds}

            def retire(self, session, finish):
                slot, last = session.slot, len(self.slots) - 1
                self.rounds.unbind(slot)
                if slot != last:   # keep the decoding rows packed at the front
                    moved = self.slots[last]
                    self.rounds.move(last, slot)
                    self.toy.move(last, slot)
                    moved.slot = slot
                    self.slots[slot] = moved
                self.slots.pop()

        model = Model()
        engine = StepEngine(model)
        prompts, counts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11], [12, 13, 14]], [40, 23, 31, 9, 17]
        try:
            def generate(prompt, count):
                return engine.generate(prompt, count, stop_tokens=[], on_tokens=(lambda found: None) if listen else None)
            with ThreadPoolExecutor(len(prompts)) as pool:
                found = list(pool.map(lambda pc: generate(*pc), zip(prompts, counts)))
            telemetry = engine.telemetry()
        finally:
            engine.close()
        return [list(t) for t, _ in found], [r for _, r in found], model.decode_steps, prompts, counts, telemetry

    def test_native_composed_rounds_decode_plain_greedy_and_report_every_round(self):
        for native, listen in ((False, True), (True, False), (True, True)):
            with self.subTest(native=native, listen=listen):
                outputs, results, steps, prompts, counts, telemetry = self.serve(native, listen)
                self.assertEqual(outputs, [plain(p, c) for p, c in zip(prompts, counts)])
                if native:
                    self.assertLess(steps, 10, "every decode step whose rows sit in the slots runs natively")
                for tokens, result in zip(outputs, results):
                    self.assertEqual(result["rounds"], len(result["accepted"]))
                    self.assertEqual(len(tokens) - 1, sum(result["accepted"]) + result["rounds"])
                    self.assertEqual(result["drafted"], [3] * result["rounds"])
                    self.assertEqual(result["verify_ms"], 1.5)
                    native_rounds = result["rounds"] - result["step_rounds"]
                    # Each native round reads one expert per column (4) and hits once; the model adds 1000 reads.
                    self.assertEqual(result["expert_reads"], 1000 + 4 * native_rounds)
                    self.assertEqual(result.get("expert_hits", 0), native_rounds)
                self.assertGreater(telemetry["core"]["steps"], 0)

    def test_native_rounds_run_at_the_widths_a_policy_chooses_and_decode_plain_greedy(self):
        # The step policy over widths 1-4 (the toy head is wrong at every seventh position): Core reads each round's
        # width from the table the previous round copied, the tokens stay plain greedy, and each request's facts
        # name the width of every round it ran.
        from crit_plugins.rounds import StepWidth
        for native in (False, True):
            with self.subTest(native=native):
                outputs, results, steps, prompts, counts, _ = self.serve(native, False, StepWidth(start=3))
                self.assertEqual(outputs, [plain(p, c) for p, c in zip(prompts, counts)])
                if native:
                    self.assertLess(steps, 10)
                widths = {w for result in results for w in result["drafted"]}
                self.assertTrue(widths <= {1, 2, 3, 4} and len(widths) > 1, widths)
                for tokens, result in zip(outputs, results):
                    self.assertEqual(len(tokens) - 1, sum(result["accepted"]) + result["rounds"])


if __name__ == "__main__":
    unittest.main()
