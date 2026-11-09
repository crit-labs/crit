"""Drafting on the device: the library's drafter, verifier and sampler device forms (draft:device, verify:device,
rounds:sampler) give what their host forms give on the same rows, their commits are idempotent from the round's start,
and whole draft + verify + accept rounds over toy targets, run by Core's round driver (crit_plugins.rounds.Rounds) with
them, decode what a loop over the host forms decodes: chains, copies, hidden-state drafts, subsets and trees, greedy and
sampled, with held rows and truncations, eagerly and replayed from CUDA graphs."""
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))
from crit_plugins.assemble import Registry, assemble, load_module  # noqa: E402

try:
    import torch
except ImportError:   # the device forms are torch's
    torch = None

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
LIBRARY = ROOT / "inference"
HEAD = ["draft:head", "draft:head-device"]
GRAPHS = (False, True) if DEVICE == "cuda" else (False,)


def plugins(use, options=None, facts=()):
    return assemble({"use": use, "options": options or {}}, Registry.discover(LIBRARY), None, facts)


def view_of(rows, width, mode, widest=None, stops=4, allowance=1000):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView over its buffers)."""
    from crit_plugins.rounds import SpeculativeRounds
    widest = width if widest is None else widest
    view = SpeculativeRounds(None, rows, widest, DEVICE, stops=stops).view(rows, width, mode)
    view.allowance.fill_(allowance)
    return view


def mirror():
    return load_module(Registry.discover(LIBRARY)["token-sampling"]).sampling


class Head:
    """A toy draft head on the device and its host twin: after token t at depth d its logits are table[t] + bias[d]
    (FP32, computed alike on both), over `columns` (a draft vocabulary, d2t mapping it to target ids) or, with rows
    set (a subset drafter's), over those rows only. Path-independent, so a tree's frontier entries need no state."""

    def __init__(self, vocab, depths, seed, columns=None, d2t=None):
        g = torch.Generator().manual_seed(seed)
        columns = columns or vocab
        self.table = (torch.randn(vocab, columns, generator=g) * 2).float()
        self.bias = torch.randn(depths, columns, generator=g).float()
        self.device_table, self.device_bias = self.table.to(DEVICE), self.bias.to(DEVICE)
        self.d2t, self.rows = d2t, None
        self.caught = 0

    def like(self, target, weight=0.4):
        """The head, made to mostly agree with a target: the target's table plus `weight` times its own."""
        self.table = target.table + weight * self.table
        self.device_table = self.table.to(DEVICE)
        return self

    def catch_up(self, view):
        self.caught += 1

    def logits(self, depth, tokens, view, parents=None):
        found = self.device_table[tokens] + self.device_bias[depth]
        return found if self.rows is None else found[..., self.rows]

    def row(self, token, depth, rows=None):
        found = (self.table[token] + self.bias[depth]).tolist()
        return found if rows is None else [found[r] for r in rows]

    def host(self, pending):
        """head(paths, rows=None) for a sequence whose pending token is `pending` (paths after it, target ids)."""
        def head(paths, rows=None):
            return [self.row(p[-1] if p else pending, len(p), rows) for p in paths]
        head.d2t = self.d2t
        return head


# --- the device forms against their host forms on the same rows ---------------------------------------------------

@unittest.skipIf(torch is None, "torch is not installed")
class DeviceVerifierTests(unittest.TestCase):
    CASES = (("all", [4, 5, 6, 7, 8], 10, ()), ("first", [3, 5, 6, 7, 8], 10, ()), ("middle", [4, 5, 2, 7, 8], 10, ()),
             ("budget", [4, 5, 6, 7, 8], 2, ()), ("stop", [4, 5, 6, 7, 8], 10, (5,)), ("none", [4, 5, 6, 7, 8], 0, ()))

    def test_the_chain_verifier_accepts_on_the_device_what_its_host_form_accepts(self):
        verifier = plugins({"verifier": "chain-verifier"})["verifier"]
        drafts = [[4, 5, 6, 7]] * len(self.CASES) + [[2, -1, -1, -1], [4, 5, 6, 7]]
        predictions = [c[1] for c in self.CASES] + [[2, 3, 9, 9, 9], [4, 5, 6, 7, 8]]
        allowance = [c[2] for c in self.CASES] + [10, 10]
        stops = [c[3] for c in self.CASES] + [(), ()]
        rows = len(drafts)
        view = view_of(rows, 4, "greedy")
        view.drafts[:, :4] = torch.tensor(drafts)
        view.predictions[:, :5] = torch.tensor(predictions)
        view.allowance.copy_(torch.tensor(allowance))
        for r, s in enumerate(stops):
            if s:
                view.stops[r, :len(s)] = torch.tensor(s)
        view.hold[rows - 1] = 1   # the last row sits the round out
        frontier = view.frontier.clone()
        verifier.accept_device(view)
        for r in range(rows):
            count = int(view.n_emitted[r])
            host = verifier.accept({"tokens": [9] + drafts[r], "parents": list(range(-1, 4))}, predictions[r],
                                   allowance[r], stops[r])
            with self.subTest(row=r):
                if r == rows - 1:
                    self.assertEqual((count, view.emitted[r].tolist()), (0, [-1] * 5))
                    continue
                self.assertEqual(view.emitted[r, :count].tolist(), host["tokens"])
                self.assertTrue(all(t == -1 for t in view.emitted[r, count:].tolist()))
                self.assertEqual(int(view.n_matched[r]), min(host["accepted"], max(count - 1, 0)))
        self.assertTrue(torch.equal(view.frontier, frontier))   # the driver advances its own state

    @unittest.skipIf(DEVICE != "cuda", "CUDA graphs")
    def test_a_captured_accept_replays_with_new_predictions(self):
        verifier = plugins({"verifier": "chain-verifier"})["verifier"]
        rows, width = 4, 5
        view = view_of(rows, width, "greedy")
        verifier.accept_device(view)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verifier.accept_device(view)
        rng = random.Random(3)
        for _ in range(20):
            drafts = [[rng.randrange(4) for _ in range(width)] for _ in range(rows)]
            predictions = [[rng.randrange(4) for _ in range(width + 1)] for _ in range(rows)]
            allowance = [rng.randrange(0, 8) for _ in range(rows)]
            view.drafts[:, :width] = torch.tensor(drafts)
            view.predictions[:, :width + 1] = torch.tensor(predictions)
            view.allowance.copy_(torch.tensor(allowance))
            view.stops[:, 0] = 3
            graph.replay()
            for r in range(rows):
                host = verifier.accept({"tokens": [0] + drafts[r], "parents": list(range(-1, width))}, predictions[r],
                                       allowance[r], (3,))
                self.assertEqual(view.emitted[r, :int(view.n_emitted[r])].tolist(), host["tokens"])


@unittest.skipIf(torch is None, "torch is not installed")
class DeviceDrafterTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(11)

    def test_the_greedy_chain_drafts_what_its_host_form_drafts(self):
        assembly = plugins({"drafter": "greedy-head-drafter", "sampler": "token-sampling"}, facts=HEAD)
        drafter, sampler = assembly["drafter"], assembly["sampler"]
        vocab, rows = 50, 3
        head = Head(vocab, 6, 1)
        for width in (0, 1, 4, 6):
            view = view_of(rows, width, "greedy")
            pending = [self.rng.randrange(vocab) for _ in range(rows)]
            view.pending.copy_(torch.tensor(pending))
            drafter.draft(view, head, sampler)
            for r in range(rows):
                drafter.start(r, [pending[r]])
                host = drafter.propose(r, width, head.host(pending[r]))["tokens"][1:]
                self.assertEqual(view.drafts[r, :width].tolist(), host)
        self.assertEqual(head.caught, 4)

    def test_sampled_rounds_draw_drafts_from_the_heads_law_with_cores_draft_draws(self):
        assembly = plugins({"drafter": "greedy-head-drafter", "sampler": "token-sampling"}, facts=HEAD)
        drafter, sampler = assembly["drafter"], assembly["sampler"]
        host = mirror()
        vocab, rows, width = 50, 3, 4
        rules = [{"temperature": 0.0}, {"temperature": 0.9, "top_k": 8, "seed": 4}, {"temperature": 1.2, "top_p": 0.8,
                                                                                    "seed": 7, "stream": 2}]
        sampler.reserve(rows, width, vocab, DEVICE)
        for slot, rule in enumerate(rules):
            sampler.stage(slot, rule)
        head = Head(vocab, width, 2)
        view = view_of(rows, width, "sampled")
        view.draft_laws = sampler.laws(rows, width)   # the round driver's, in sampled rounds
        pending, out_count = [3, 9, 27], [5, 0, 40]
        view.pending.copy_(torch.tensor(pending))
        view.out_count.copy_(torch.tensor(out_count))
        drafter.draft(view, head, sampler)
        laws = view.draft_laws.as_dicts(rows, width)
        for r, rule in enumerate(rules):
            token = pending[r]
            for depth in range(width):
                law = sampler.law(head.row(token, depth), rule)
                if rule["temperature"] > 0:
                    u = host.uniform(rule.get("seed", 0), out_count[r] + depth, rule.get("stream", 0), host.DRAFT)
                    token = host.walk(law, u * math.fsum(p for _, p in law))
                else:
                    token = law[0][0]
                with self.subTest(row=r, depth=depth):
                    self.assertEqual(int(view.drafts[r, depth]), token)
                    self.assertEqual(sorted(laws[r][depth]), sorted(t for t, _ in law))

    def test_a_sampled_round_needs_the_samplers_draft_draws(self):
        drafter = plugins({"drafter": "greedy-head-drafter"}, facts=HEAD)["drafter"]

        class Predicts:   # a sampler with predictions_device alone
            def predictions_device(self, view, logits, draft_laws=None):
                return None
        with self.assertRaisesRegex(ValueError, "draft_device"):
            drafter.draft(view_of(1, 2, "sampled"), Head(10, 2, 3), Predicts())

    def emit(self, view, emitted):
        """The accept's output, as a round's commit reads it: emitted tokens, counts and the frontier after it."""
        view.emitted.fill_(-1)
        for r, e in enumerate(emitted):
            view.emitted[r, :len(e)] = torch.tensor(e)
        view.n_emitted.copy_(torch.tensor([len(e) for e in emitted]))

    def history_round(self, drafter, view, host_tokens, rng, width, vocab, truncate=False):
        """One round of the copy drafter on the device and the host: drafts compared, then a random emission (with
        `truncate`, first a longer one committed and then the shorter one committed again: a truncation)."""
        rows = view.rows
        drafter.draft(view, None, None)
        for r in range(rows):
            host = drafter.propose(r, width)["tokens"][1:]
            self.assertEqual(view.drafts[r, :width].tolist(), host + [-1] * (width - len(host)),
                             (r, host_tokens[r][-12:]))
        emitted = []
        for r in range(rows):
            k = rng.randrange(0, width + 1)
            drafts = [t for t in view.drafts[r, :width].tolist() if t >= 0][:k]
            emitted.append(drafts + [rng.randrange(vocab)])
        start = view.frontier.clone()
        if truncate:   # a longer emission committed first, then the round truncated to `emitted`
            longer = [e + [rng.randrange(vocab) for _ in range(width + 1 - len(e))] for e in emitted]
            self.emit(view, longer)
            view.frontier.copy_(start + view.n_emitted)
            drafter.commit_round(view)
        self.emit(view, emitted)
        view.frontier.copy_(start + view.n_emitted)
        view.pending.copy_(torch.tensor([e[-1] for e in emitted]))
        drafter.commit_round(view)
        for r, e in enumerate(emitted):
            drafter.observe(r, e)
            host_tokens[r] += e

    def copy_drafter(self, rows, width, vocab, **options):
        drafter = plugins({"drafter": "mtp-decode"}, {"mtp-decode": options}, facts=HEAD)["drafter"]
        drafter.reserve(rows, width, vocab, DEVICE)
        return drafter

    def start_rows(self, drafter, view, prompts):
        for r, prompt in enumerate(prompts):
            drafter.start(r, prompt)
            drafter.seed(r, prompt)
            view.frontier[r] = len(prompt) - 1
            view.pending[r] = prompt[-1]

    def test_copy_drafts_on_the_device_follow_the_host_index_through_truncations(self):
        for window, history in ((2, 4), (3, 6)):
            rows, width, vocab = 4, 5, 6
            drafter = self.copy_drafter(rows, width, vocab, window=window, history_window=history, capacity=512)
            rng = random.Random(window)
            prompts = [[rng.randrange(vocab) for _ in range(rng.randrange(3, 60))] for _ in range(rows)]
            tokens = [list(p) for p in prompts]
            view = view_of(rows, width, "greedy")
            self.start_rows(drafter, view, prompts)
            for round_ in range(40):
                self.history_round(drafter, view, tokens, rng, width, vocab, truncate=round_ % 3 == 1)

    def test_a_ring_that_wraps_keeps_its_last_positions_through_truncations(self):
        """Past the ring, the copies search the last capacity - (widest + 1) positions: the same proposals as a host
        index over those positions, whatever a truncated round had written past its frontier."""
        rows, width, vocab, capacity = 2, 3, 4, 70
        drafter = self.copy_drafter(rows, width, vocab, window=2, history_window=4, capacity=capacity)
        span = capacity - (width + 1)
        rng = random.Random(5)
        prompts = [[rng.randrange(vocab) for _ in range(40)] for _ in range(rows)]
        tokens = [list(p) for p in prompts]
        view = view_of(rows, width, "greedy")
        self.start_rows(drafter, view, prompts)
        prompt_copy = load_module(Registry.discover(LIBRARY)["mtp-decode"]).prompt_copy
        for round_ in range(60):
            drafter.draft(view, None, None)
            for r in range(rows):
                # What the ring searches: the last `span` positions, those before the prompt's end still prompt ones.
                first = max(0, len(tokens[r]) - span)
                window = tokens[r][first:]
                prompt_part = max(0, len(prompts[r]) - first)
                index = prompt_copy.PromptCopyIndex(window[:prompt_part], window=2, history_window=4, max_draft=64,
                                                    max_tokens=1 << 20)
                index.append(window[prompt_part:])
                expected = list(index.propose(width))
                self.assertEqual(view.drafts[r, :width].tolist(), expected + [-1] * (width - len(expected)),
                                 (round_, r))
            emitted = [[rng.randrange(vocab) for _ in range(rng.randrange(1, width + 2))] for _ in range(rows)]
            start = view.frontier.clone()
            self.emit(view, [e + [0] * (width + 1 - len(e)) for e in emitted])   # a longer round first
            view.frontier.copy_(start + view.n_emitted)
            drafter.commit_round(view)
            self.emit(view, emitted)                                            # then truncated
            view.frontier.copy_(start + view.n_emitted)
            view.pending.copy_(torch.tensor([e[-1] for e in emitted]))
            drafter.commit_round(view)
            for r in range(rows):
                tokens[r] += emitted[r]
        self.assertTrue(all(len(t) > capacity for t in tokens))

    def test_long_copies_override_the_heads_chain(self):
        vocab, rows, width = 6, 3, 4
        drafter = self.copy_drafter(rows, width, vocab, window=2, override=3, capacity=256)
        head = Head(vocab, width, 5)
        rng = random.Random(8)
        view = view_of(rows, width, "greedy")
        prompts = [[1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2], [rng.randrange(vocab) for _ in range(30)], [3, 3]]
        self.start_rows(drafter, view, prompts)
        drafter.draft(view, head, None)
        for r, prompt in enumerate(prompts):
            host = drafter.propose(r, width, head.host(prompt[-1]))["tokens"][1:]
            self.assertEqual(view.drafts[r, :width].tolist(), host)
        self.assertEqual(view.drafts[0, :width].tolist(), [3, 4, 5, 0])   # the repeating prompt's copy

    def test_sampled_copies_have_no_law(self):
        assembly = plugins({"drafter": "mtp-decode", "sampler": "token-sampling"},
                           {"mtp-decode": {"window": 2, "override": 3, "capacity": 256}}, facts=HEAD)
        drafter, sampler = assembly["drafter"], assembly["sampler"]
        vocab, rows, width = 6, 2, 4
        drafter.reserve(rows, width, vocab, DEVICE)
        sampler.reserve(rows, width, vocab, DEVICE)
        for slot in range(rows):
            sampler.stage(slot, {"temperature": 1.0, "seed": slot + 3})
        view = view_of(rows, width, "sampled")
        view.draft_laws = sampler.laws(rows, width)
        self.start_rows(drafter, view, [[1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2], [3, 3]])
        drafter.draft(view, Head(vocab, width, 4), sampler)
        laws = view.draft_laws.as_dicts(rows, width)
        self.assertEqual(view.drafts[0, :width].tolist(), [3, 4, 5, 0])
        self.assertTrue(all(law in ({}, None) for law in laws[0]))   # copies: no law (the target's own draw)
        self.assertTrue(all(law for law in laws[1]))                  # the head's drafts keep theirs

    def sure_head(self):
        class Sure(Head):
            def logits(self, depth, tokens, view, parents=None):
                row = [5.0, 0.0, 0.0, 0.0] if depth == 0 else [0.1, 0.0, 0.2, 0.0]
                return torch.tensor(row, device=DEVICE).expand(tokens.shape[0], 4).contiguous()

            def host(self, pending):
                def head(paths, rows=None):
                    return [[5.0, 0.0, 0.0, 0.0] if not p else [0.1, 0.0, 0.2, 0.0] for p in paths]
                head.d2t = self.d2t
                return head
        return Sure(14, 3, 0, columns=4, d2t=[10, 10, 10, 10])

    def test_hidden_state_drafts_map_and_cut_as_on_the_host_through_truncations(self):
        """The host test's head: a 4-token draft vocabulary mapped to 10..13, sure of its first draft only; the cut
        learns to drop the unsure drafts from the rounds' emitted tokens, on the device without the host, and a
        truncation's second commit counts the round once."""
        drafter = plugins({"drafter": "hidden-state-drafters"}, {"hidden-state-drafters": {"cut_target": 0.8}},
                          ["taps:hidden-states", *HEAD])["drafter"]
        rows, width = 2, 3
        drafter.reserve(rows, width, 14, DEVICE)
        head = self.sure_head()
        view = view_of(rows, width, "greedy")
        for r in range(rows):
            drafter.start(r, [4])
        seen = []
        for round_ in range(80):
            drafter.draft(view, head, None)
            device = view.drafts[:, :width].tolist()
            self.assertEqual(device, [drafter.propose(r, width, head.host(4))["tokens"][1:] for r in range(rows)],
                             round_)
            seen.append(device[0])
            self.emit(view, [[10, 12, 12, 11]] * rows)   # a longer emission committed first, then truncated
            drafter.commit_round(view)
            self.emit(view, [[10, 11]] * rows)
            drafter.commit_round(view)
            for r in range(rows):
                drafter.observe(r, [10, 11])
        self.assertEqual(seen[0], [10, 12, 12])
        self.assertEqual(seen[-1], [10, -1, -1])

    def cyclic_head(self):
        class Cyclic(Head):
            """After n tokens the best of 12 is (n * 5 + 2) % 12 (the host test's head)."""
            def logits(self, depth, tokens, view, parents=None):
                row = torch.zeros(12, device=DEVICE)
                row[(depth * 5 + 2) % 12] = 1.0
                found = row.expand(tokens.shape[0], 12)
                return found if self.rows is None else found[:, self.rows]

            def host(self, pending):
                def head(paths, rows=None):
                    full = [[1.0 if t == (len(p) * 5 + 2) % 12 else 0.0 for t in range(12)] for p in paths]
                    return full if rows is None else [[r[t] for t in rows] for r in full]
                return head
        return Cyclic(12, 3, 0)

    def test_subset_drafts_follow_the_emitted_counts_on_the_device_through_truncations(self):
        drafter = plugins({"drafter": "draft-vocabulary-subset"}, {"draft-vocabulary-subset": {"size": 4, "refresh": 2}},
                          ["draft:head-rows", *HEAD])["drafter"]
        width = 3
        drafter.reserve(1, width, 12, DEVICE)
        head = self.cyclic_head()
        view = view_of(1, width, "greedy")
        drafter.start(0, [3])
        for emitted in ([7, 7], [7], [1, 2, 3], [7, 0], [11], [7, 7, 7], [2]):
            drafter.draft(view, head, None)
            self.assertEqual(view.drafts[0, :width].tolist(), drafter.propose(0, width, head.host(3))["tokens"][1:])
            self.emit(view, [emitted + [7] * (width + 1 - len(emitted))])   # committed, then truncated
            drafter.commit_round(view)
            self.emit(view, [emitted])
            drafter.commit_round(view)
            drafter.observe(0, emitted)
        self.assertIn(7, view.drafts[0, :width].tolist())


# --- whole rounds through Core's round driver ------------------------------------------------------------------------

class Target:
    """A toy target on the device: after token t at position p its logits are table[t] + phase[p % 13] (FP32), so a
    verify column's logits are those of a one-token step; host() gives the same rows as lists."""

    def __init__(self, vocab, seed):
        g = torch.Generator().manual_seed(seed)
        self.table = (torch.randn(vocab, vocab, generator=g) * 2.5).float()
        self.phase = torch.randn(13, vocab, generator=g).float()
        self.device_table, self.device_phase = self.table.to(DEVICE), self.phase.to(DEVICE)

    def logits(self, tokens, positions):
        return self.device_table[tokens.clamp(min=0)] + self.device_phase[positions % 13]

    def host(self, token, position):
        return (self.table[token] + self.phase[position % 13]).tolist()

    def verify(self, view, batch):
        rows, cols = view.rows, view.width + 1
        return self.logits(batch.tokens_t, batch.positions_t).view(rows, cols, -1)


def chain_drafts(sampler, head, pending, width, rule, position):
    """A head chain drafted as the device drafts it: argmax for a greedy rule, else Core's draft draws from each
    depth's law at output position position + depth; returns (drafts, laws)."""
    host = mirror()
    drafts, laws, token = [], [], pending
    for depth in range(width):
        law = sampler.law(head.row(token, depth), rule)
        if rule.get("temperature", 0.0) > 0:
            u = host.uniform(rule.get("seed", 0), position + depth, rule.get("stream", 0), host.DRAFT)
            token = host.walk(law, u * math.fsum(p for _, p in law))
        else:
            token = law[0][0]
        drafts.append(token)
        laws.append(dict(law))
    return drafts, laws


def host_decode(assembly, target, head, prompts, firsts, counts, widths, rules, stops, truncated, options):
    """The same rounds over the host forms. Drafts: greedy-head-drafter's chain drawn as the device draws it (sampled
    rounds draw from the head's law), mtp-decode's host index (with `override` over that chain), the other drafters'
    propose (greedy rounds); the sampler's predictions with each column's law (None for a copy); the verifier's accept;
    truncated: {row: {round: kept}}."""
    sampler, verifier, drafter = assembly["sampler"], assembly["verifier"], assembly["drafter"]
    drafter_id = assembly.use["drafter"]
    out = []
    for r, prompt in enumerate(prompts):
        tokens, emitted, cycle = list(prompt) + [firsts[r]], [firsts[r]], 0
        rule = rules[r]
        sampled = rule.get("temperature", 0.0) > 0
        drafter.start(r, tokens)
        while len(emitted) < counts[r] and emitted[-1] not in stops:
            width = widths[r][cycle % len(widths[r])]
            pending, frontier, position = tokens[-1], len(tokens) - 1, len(emitted)
            laws = None
            if drafter_id == "greedy-head-drafter" or (drafter_id == "mtp-decode" and options.get("override")):
                drafts, laws = chain_drafts(sampler, head, pending, width, rule, position)
                if drafter_id == "mtp-decode" and width:
                    copied, matched = drafter.indices[r].propose_matched(width)
                    if matched >= options["override"]:
                        drafts[:len(copied)] = copied
                        laws[:len(copied)] = [None] * len(copied)
            else:
                drafts = drafter.propose(r, width, head.host(pending) if head else None)["tokens"][1:]
            columns = [pending] + drafts
            rows = [target.host(t, frontier + c) for c, t in enumerate(columns)]
            predictions = sampler.predictions(rows, drafts, dict(rule, position=position), laws if sampled else None)
            found = verifier.accept({"tokens": columns, "parents": list(range(-1, len(drafts)))}, predictions,
                                    counts[r] - len(emitted), stops)["tokens"]
            found = found[:truncated.get(r, {}).get(cycle, len(found))]
            drafter.observe(r, found)
            emitted += found
            tokens += found
            cycle += 1
        out.append(emitted)
    return out


@unittest.skipIf(torch is None, "torch is not installed")
class CoreRoundTests(unittest.TestCase):
    """crit_plugins.rounds.Rounds over the library's device forms (draft:device, the model's verify, rounds:sampler's
    predictions_device, verify:device's accept_device, every commit_round), with held rows and truncations, eager and
    captured, greedy and sampled, against the host-form loop."""

    def decode(self, graphs, mode, rules, widths, prompts, counts, stops=(5,), drafter="greedy-head-drafter",
               options=None, hold=None, truncate=None, head=None, facts=HEAD, vocab=40):
        from crit_plugins.rounds import Rounds
        widest = max(widths)
        assembly = plugins({"drafter": drafter, "verifier": "chain-verifier", "sampler": "token-sampling"},
                           {drafter: options or {}}, facts=facts)
        target = Target(vocab, 21)
        head = head or Head(vocab, widest, 22).like(target)
        rows = len(prompts)
        rounds = Rounds(target, assembly["sampler"], rows=rows, widest=widest, device=DEVICE,
                        drafter=assembly["drafter"], head=head, verifier=assembly["verifier"], stops=2,
                        vocabulary=vocab, sampled=mode == "sampled")
        if graphs:
            for width in sorted(set(widths)):
                rounds.capture(rows, width, mode)
        sampler = assembly["sampler"]
        firsts = [sampler.select([target.host(p[-1], len(p) - 1)], dict(rules[r], position=0))[0]
                  for r, p in enumerate(prompts)]
        for slot, (prompt, rule) in enumerate(zip(prompts, rules)):
            rounds.bind(slot, ("sequence", slot), len(prompt), firsts[slot], counts[slot] - 1,
                        len(prompt) + counts[slot] + widest + 1, stops=stops, params=rule,
                        tokens=list(prompt) + [firsts[slot]])
        outputs = [[f] for f in firsts]
        done = [len(o) >= c or o[-1] in stops for o, c in zip(outputs, counts)]
        ran, truncated = [[] for _ in prompts], [{} for _ in prompts]
        for cycle in range(300):
            if all(done):
                break
            held = [r for r in range(rows) if done[r] or (hold and hold(cycle, r))]
            width = widths[cycle % len(widths)]
            rounds.launch(rows, width, mode, hold=held)
            found = rounds.read()
            if truncate and truncate(cycle):
                keep = [max(len(f["tokens"]) - 1, 1) if r not in held else 0 for r, f in enumerate(found)]
                rounds.truncate(keep)
                for r in range(rows):
                    if r not in held and keep[r] < len(found[r]["tokens"]):
                        truncated[r][len(ran[r])] = keep[r]
                found = [{**f, "tokens": f["tokens"][:k]} for f, k in zip(found, keep)]
            for r, result in enumerate(found):
                if r in held:
                    self.assertEqual(result["tokens"], [])
                    continue
                ran[r].append(width)
                outputs[r] += result["tokens"]
                if outputs[r][-1] in stops or len(outputs[r]) >= counts[r]:
                    done[r] = True
        host = host_decode(assembly, target, head, prompts, firsts, counts, [w or [0] for w in ran], rules, stops,
                           dict(enumerate(truncated)), options or {})
        return outputs, host, rounds

    PROMPTS = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    SAMPLED = [{"temperature": 0.8, "top_k": 12, "seed": 3}, {"temperature": 0.0},
               {"temperature": 1.1, "top_p": 0.9, "seed": 17, "stream": 1}]

    def test_chain_rounds_decode_what_the_host_forms_decode(self):
        cases = (("greedy", [{"temperature": 0.0}] * 3, [0, 3, 5, 2], [30, 17, 25]),
                 ("sampled", self.SAMPLED, [4, 2, 0, 3], [26, 21, 24]))
        for mode, rules, widths, counts in cases:
            for graphs in GRAPHS:
                outputs, host, rounds = self.decode(graphs, mode, rules, widths, self.PROMPTS, counts,
                                                    hold=lambda cycle, r: cycle % 4 == 1 and r == 1,
                                                    truncate=lambda cycle: cycle % 5 == 3)
                with self.subTest(mode=mode, graphs=graphs):
                    self.assertEqual(outputs, host)
                    self.assertGreater(sum(rounds.telemetry()["drafts"]["accepted_by_depth"]), 0)

    def test_copy_rounds_decode_what_the_host_forms_decode(self):
        prompts, counts = [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2], [7, 8, 9, 7, 8, 9, 7], [11, 12]], [28, 22, 19]
        sampled = [{"temperature": 0.9, "top_k": 12, "seed": 3}, {"temperature": 1.1, "seed": 5},
                   {"temperature": 0.8, "top_p": 0.9, "seed": 7}]
        for mode, rules in (("greedy", [{"temperature": 0.0}] * 3), ("sampled", sampled)):
            for options in ({"window": 2}, {"window": 2, "override": 3}):
                for graphs in GRAPHS:
                    outputs, host, _ = self.decode(graphs, mode, rules, [4, 1, 3], prompts, counts, stops=(),
                                                   drafter="mtp-decode", options=dict(options, capacity=128),
                                                   hold=lambda cycle, r: cycle % 5 == 2 and r == 0,
                                                   truncate=lambda cycle: cycle % 4 == 3)
                    with self.subTest(mode=mode, options=options, graphs=graphs):
                        self.assertEqual(outputs, host)

    def test_hidden_state_and_subset_rounds_decode_plain_greedy(self):
        """Greedy rounds emit plain greedy decoding whatever the drafts: the hidden-state drafter's mapped and cut
        drafts and the subset drafter's, held and truncated, eager and captured."""
        vocab = 40
        target = Target(vocab, 21)
        hidden = Head(vocab, 5, 23, columns=20, d2t=[2 * i - i for i in range(20)])   # draft id i -> target 2i
        hidden.table = target.table[:, ::2] + 0.3 * hidden.table
        hidden.device_table = hidden.table.to(DEVICE)
        cases = (("hidden-state-drafters", {"cut_target": 0.6}, hidden, ["taps:hidden-states", *HEAD]),
                 ("draft-vocabulary-subset", {"size": 16, "refresh": 3}, None, ["draft:head-rows", *HEAD]))
        for drafter, options, head, facts in cases:
            for graphs in GRAPHS:
                outputs, host, _ = self.decode(graphs, "greedy", [{"temperature": 0.0}] * 3, [4, 2, 5], self.PROMPTS,
                                               [30, 25, 20], drafter=drafter, options=options, head=head, facts=facts,
                                               hold=lambda cycle, r: cycle % 4 == 2 and r == 2,
                                               truncate=lambda cycle: cycle % 6 == 4)
                with self.subTest(drafter=drafter, graphs=graphs):
                    self.assertEqual(outputs, host)


class MaskedTarget:
    """A toy target whose logits after a token read its whole path through an ancestor mask: the hidden state at a
    position is the sum of embed[token] * phase[position % 7] over the sequence up to it (prefix kept per row, plus
    the tree's ancestors), logits = hidden @ out. A tree verify with the mask equals one-token steps along any path, as
    an attention kernel with tree masks (attention:tree) gives. As the round driver's model, its commit keeps each
    row's accepted path from the round's start (idempotent: a truncation commits again)."""

    def __init__(self, vocab, seed, rows=1, hidden=16):
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.randn(vocab, hidden, generator=g, dtype=torch.float64).to(DEVICE)
        self.phase = torch.randn(7, hidden, generator=g, dtype=torch.float64).to(DEVICE)
        self.out = torch.randn(hidden, vocab, generator=g, dtype=torch.float64).to(DEVICE) * 3
        self.prefix = torch.zeros(rows, hidden, dtype=torch.float64, device=DEVICE)
        self.start = self.prefix.clone()
        self.nodes = self.positions = None

    def term(self, tokens, positions):
        return self.embed[tokens] * self.phase[positions % 7]

    def seed(self, slot, prompt):
        self.prefix[slot] = sum(self.term(torch.tensor(t, device=DEVICE), torch.tensor(q, device=DEVICE))
                                for q, t in enumerate(prompt[:-1]))

    def tree_logits(self, prefix, tokens, parents, depths, frontier):
        """prefix [rows][hidden] (the committed positions' sum), tokens [rows][N] node tokens (node 0 the pending
        one), parents [rows][N] (-1 for node 0), depths [rows][N]: each node's logits over its ancestor mask."""
        rows, nodes = tokens.shape
        mask = torch.eye(nodes, dtype=torch.float64, device=DEVICE).expand(rows, nodes, nodes).clone()
        index = torch.arange(rows, device=DEVICE)
        for i in range(1, nodes):   # ancestors of node i: its parent's ancestors and itself
            mask[:, i] = torch.maximum(mask[:, i], mask[index, parents[:, i].clamp(min=0)])
        terms = self.term(tokens.clamp(min=0), frontier[:, None] + depths)
        hidden = prefix[:, None, :] + mask @ terms                      # the masked "attention" over the tree
        return (hidden @ self.out).float()

    def verify(self, view, batch):
        rows, width = view.rows, view.width
        self.start[:rows].copy_(self.prefix[:rows])
        nodes = torch.cat([view.pending[:, None], view.drafts[:, :width]], 1)
        parents = torch.cat([torch.full((rows, 1), -1, device=DEVICE, dtype=torch.int64), view.tree_parents], 1)
        self.nodes, self.depths = nodes, view.tree_depths
        self.frontier = view.frontier.clone()
        return self.tree_logits(self.prefix[:rows], nodes, parents, view.tree_depths, view.frontier)

    def commit_round(self, view):
        """The root and the accepted nodes before the last emitted token are materialized: their terms join the
        prefix, from the round's start."""
        rows, nodes = view.rows, view.width + 1
        column = torch.arange(nodes, device=DEVICE)[None, :]
        kept = (column < view.n_emitted[:rows, None]) & (view.tree_path[:rows, :nodes] >= 0)
        path = view.tree_path[:rows, :nodes].clamp(min=0)
        depth = self.depths.gather(1, path)
        terms = self.term(self.nodes.gather(1, path), self.frontier[:, None] + depth)
        self.prefix[:rows] = self.start[:rows] + (terms * kept[..., None]).sum(1)

    def plain(self, prompt, count):
        hidden = sum(self.term(torch.tensor(t, device=DEVICE), torch.tensor(p, device=DEVICE))
                     for p, t in enumerate(prompt))
        out, position = [], len(prompt) - 1
        while len(out) < count:
            token = int(torch.argmax((hidden @ self.out).float()))
            out.append(token)
            position += 1
            hidden = hidden + self.term(torch.tensor(token, device=DEVICE), torch.tensor(position, device=DEVICE))
        return out


@unittest.skipIf(torch is None, "torch is not installed")
class DeviceTreeTests(unittest.TestCase):
    OPTIONS = {"depth": 3, "branch": 3, "beam": 2}

    def trees(self, options=None, rows=3, width=9, vocab=30):
        trees = plugins({"drafter": "draft-trees", "verifier": "draft-trees"}, {"draft-trees": options or self.OPTIONS},
                        HEAD)["drafter"]
        trees.reserve(rows, width, vocab, DEVICE)
        return trees

    def test_device_trees_are_the_host_trees(self):
        trees = self.trees()
        vocab, rows = 30, 3
        head = Head(vocab, 4, 31)
        rng = random.Random(5)
        for width in (1, 2, 5, 9):
            view = view_of(rows, width, "greedy", widest=9)
            pending = [rng.randrange(vocab) for _ in range(rows)]
            view.pending.copy_(torch.tensor(pending))
            trees.draft(view, head)
            for r in range(rows):
                trees.start(r, [pending[r]])
                host = trees.propose(r, width, head.host(pending[r]))
                count = len(host["tokens"]) - 1
                with self.subTest(width=width, row=r):
                    self.assertEqual(view.drafts[r, :count].tolist(), host["tokens"][1:])
                    self.assertEqual(view.tree_parents[r, :count].tolist(), host["parents"][1:])
                    self.assertTrue(all(t == -1 for t in view.drafts[r, count:width].tolist()))

    def test_the_device_walk_is_the_host_walk_and_predictions_follow_the_path(self):
        trees = self.trees(rows=5, width=7, vocab=6)
        vocab, rows, width = 6, 5, 7
        head = Head(vocab, 4, 9)
        rng = random.Random(2)
        for trial in range(30):
            view = view_of(rows, width, "greedy")
            pending = [rng.randrange(vocab) for _ in range(rows)]
            view.pending.copy_(torch.tensor(pending))
            trees.draft(view, head)
            tokens = torch.cat([view.pending[:, None], view.drafts[:, :width]], 1).tolist()
            parents = [[-1] + p for p in view.tree_parents.tolist()]
            # Predictions that follow one branch for a while, then differ.
            predictions = [[rng.randrange(vocab) for _ in range(width + 1)] for _ in range(rows)]
            for r in range(rows):
                v = 0
                for _ in range(rng.randrange(0, 4)):
                    children = [i for i in range(1, width + 1) if parents[r][i] == v and tokens[r][i] >= 0]
                    if not children:
                        break
                    child = rng.choice(children)
                    predictions[r][v] = tokens[r][child]
                    v = child
            allowance = [rng.randrange(0, 6) for _ in range(rows)]
            view.predictions[:, :width + 1] = torch.tensor(predictions)
            view.allowance.copy_(torch.tensor(allowance))
            view.stops[:, 0] = 1
            view.hold[rows - 1] = trial % 2
            trees.accept_device(view)
            for r in range(rows):
                kept = len([t for t in tokens[r] if t >= 0])
                proposal = {"tokens": tokens[r][:kept], "parents": parents[r][:kept]}
                host = trees.accept(proposal, predictions[r][:kept], allowance[r], (1,))
                count = int(view.n_emitted[r])
                with self.subTest(trial=trial, row=r):
                    if r == rows - 1 and trial % 2:
                        self.assertEqual(count, 0)   # held
                        continue
                    self.assertEqual(view.emitted[r, :count].tolist(), host["tokens"])
                    self.assertEqual(view.tree_path[r, :count].tolist(), host["path"][:count])
                    self.assertEqual(view.predictions[r, :count].tolist(), host["tokens"])   # in path order

    def test_tree_rounds_through_the_driver_decode_plain_greedy(self):
        """Core's round driver over draft-trees and a masked-attention toy target: plain greedy decoding with held
        rows and truncations, eager and captured (the tree's buffers are the verifier's own)."""
        from crit_plugins.rounds import Rounds
        vocab, width = 12, 6
        prompts, count = [[1, 2, 3], [4, 5, 6, 7, 8]], 30
        for graphs in GRAPHS:
            trees = plugins({"drafter": "draft-trees", "verifier": "draft-trees", "sampler": "token-sampling"},
                            {"draft-trees": {"depth": 3, "branch": 2, "beam": 2}}, HEAD)
            target = MaskedTarget(vocab, 4, rows=len(prompts))
            truth = [target.plain(p, count) for p in prompts]
            rounds = Rounds(target, trees["sampler"], rows=len(prompts), widest=width, device=DEVICE,
                            drafter=trees["drafter"], head=Head(vocab, 4, 13), verifier=trees["verifier"],
                            vocabulary=vocab)
            if graphs:
                rounds.capture(len(prompts), width, "greedy")
            outputs = []
            for slot, prompt in enumerate(prompts):
                target.seed(slot, prompt)
                rounds.bind(slot, slot, len(prompt) - 1, prompt[-1], count, len(prompt) + count + width + 1,
                            params={"temperature": 0.0}, tokens=prompt)
                outputs.append([])
            for cycle in range(100):
                if min(len(o) for o in outputs) >= count:
                    break
                held = [1] if cycle % 4 == 2 else []
                rounds.launch(len(prompts), width, "greedy", hold=held)
                found = rounds.read()
                if cycle % 5 == 3:
                    keep = [max(len(f["tokens"]) - 1, 1) if r not in held else 0 for r, f in enumerate(found)]
                    rounds.truncate(keep)
                    found = [{**f, "tokens": f["tokens"][:k]} for f, k in zip(found, keep)]
                for r, f in enumerate(found):
                    outputs[r] += f["tokens"]
            with self.subTest(graphs=graphs):
                self.assertEqual([o[:count] for o in outputs], truth)
                self.assertGreater(sum(rounds.telemetry()["drafts"]["accepted_by_depth"]), 0)


if __name__ == "__main__":
    unittest.main()
