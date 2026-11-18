"""Core's prompt cache (crit_plugins.prompt_cache): the ledger over a model's pools, and the toy engine that keeps its
ended requests' sequences with it, run by Core's loader on Core's concurrent runtime."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"):
    sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_core.engine import load  # noqa: E402
from crit_plugins.prompt_cache import PromptCache, shared  # noqa: E402
from support.engines import STEP_LIBRARY, write_package  # noqa: E402

VOCAB = 420   # the llama fixture's vocab_size, which the toy model reads from config.json


def successor(seen):
    h = 1469598103934665603
    for position, token in enumerate(seen):
        h = ((h ^ ((token + 1) * (position + 1))) * 1099511628211) % 2 ** 64
    return h % VOCAB


def oracle(prompt, output):
    rows, out = list(prompt), []
    for _ in range(output):
        out.append(successor(rows))
        rows.append(out[-1])
    return out


class Pool:
    """A pool as the ledger sees one: sequences with a length, forked whole and (a KV pool) kept to a prefix."""

    def __init__(self):
        self.lengths, self.calls = {}, []

    def open(self, sequence):
        assert sequence not in self.lengths
        self.lengths[sequence] = 0

    def close(self, sequence):
        self.calls.append(("close", sequence))
        del self.lengths[sequence]

    def fork(self, sequence, new):
        assert new not in self.lengths
        self.lengths[new] = self.lengths[sequence]
        self.calls.append(("fork", sequence, new))

    def keep(self, sequence, length):
        assert length <= self.lengths[sequence]
        self.lengths[sequence] = length

    def grow(self, sequence, length):   # the model's own writes
        self.lengths[sequence] = length


class LedgerTests(unittest.TestCase):
    def serve(self, cache, pools, sequence, prompt, output=(), allow=True, marks=()):
        reused = cache.open(sequence, prompt, allow)
        for position in marks:
            for pool in pools:
                pool.grow(sequence, position)
            cache.mark(sequence, position)
        tokens = list(prompt) + list(output)
        for pool in pools:
            pool.grow(sequence, len(tokens))
        cache.retire(sequence, tokens)
        return reused

    def test_a_prompt_that_begins_as_a_kept_sequence_reuses_the_shared_prefix(self):
        pool = Pool()
        cache = PromptCache([pool])
        self.assertEqual(self.serve(cache, [pool], 1, [5, 6, 7, 8], [9, 9]), 0)
        # The next turn: the first prompt, another reply in its place, more text. They share the first prompt.
        self.assertEqual(cache.open(2, [5, 6, 7, 8, 3, 3, 4]), 4)
        self.assertEqual(pool.lengths[2], 4)             # forked from the kept sequence and kept to the prefix
        self.assertEqual(pool.lengths[1], 6)             # which stays whole for another request
        self.assertEqual(shared([1, 2, 3], [1, 2, 4, 5]), 2)
        self.assertEqual(cache.telemetry()["prompt_cache"]["hits"], 1)

    def test_the_last_prompt_token_is_always_computed(self):
        pool = Pool()
        cache = PromptCache([pool])
        self.serve(cache, [pool], 1, [5, 6, 7, 8], [9])
        self.assertEqual(cache.open(2, [5, 6, 7, 8]), 3)      # the same prompt again: its last row gives the first token
        self.assertEqual(cache.open(3, [5, 6, 7, 8, 9]), 4)   # the reply it kept, as a prompt: all but the last

    def test_a_request_that_does_not_allow_the_cache_runs_as_without_one(self):
        pool = Pool()
        cache = PromptCache([pool])
        self.serve(cache, [pool], 1, [5, 6, 7, 8])
        self.assertEqual(self.serve(cache, [pool], 2, [5, 6, 7, 8, 9], allow=False), 0)
        self.assertNotIn(2, pool.lengths)                     # closed when it ended, not kept
        self.assertEqual(list(cache.kept), [1])
        self.assertFalse(cache.mark(2, 3))

    def test_the_longest_prefix_wins_and_an_extended_sequence_leaves(self):
        pool = Pool()
        cache = PromptCache([pool])
        self.serve(cache, [pool], 1, [1, 2, 3, 4])
        self.serve(cache, [pool], 2, [1, 2, 9, 9, 9])
        self.assertEqual(self.serve(cache, [pool], 3, [1, 2, 9, 9, 9, 7, 7]), 5)
        self.assertEqual(sorted(cache.kept), [1, 3])          # 2 is a prefix of 3: nothing it holds is lost
        self.assertNotIn(2, pool.lengths)

    def test_kept_sequences_leave_least_recently_used_first(self):
        pool = Pool()
        cache = PromptCache([pool], entries=2)
        for sequence in (1, 2):
            self.serve(cache, [pool], sequence, [sequence, 0, 0])
        cache.open(9, [1, 0, 0, 5])                           # 1 is used again
        self.serve(cache, [pool], 3, [3, 0, 0])
        self.assertEqual(sorted(cache.kept), [1, 3])
        self.assertTrue(cache.evict())
        self.assertEqual(list(cache.kept), [3])
        cache.clear()
        self.assertFalse(cache.evict())
        bounded = PromptCache([pool], tokens=5)
        self.serve(bounded, [pool], 11, [7, 0, 0])
        self.serve(bounded, [pool], 12, [8, 0, 0])
        self.assertEqual(list(bounded.kept), [12])            # six positions pass the five it may hold

    def test_states_resume_where_they_stood_at_a_sequences_end_or_a_mark(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state])
        self.serve(cache, [kv, state], 1, [1, 2, 3, 4, 5, 6], [7, 7], marks=(4, 6))
        # Shares six tokens: the state stood there, at the mark the model set when its prompt ended.
        self.assertEqual(cache.open(2, [1, 2, 3, 4, 5, 6, 8, 8]), 6)
        self.assertEqual((kv.lengths[2], state.calls[-1][2]), (6, 2))
        self.assertNotEqual(state.calls[-1][1], 1)            # forked from the mark, not from the sequence's end
        # Shares five: the mark at four is the last place the state stood.
        self.assertEqual(cache.open(3, [1, 2, 3, 4, 5, 0, 0]), 4)
        # Shares the whole kept sequence: its own end state.
        self.assertEqual(cache.open(4, [1, 2, 3, 4, 5, 6, 7, 7, 1]), 8)
        self.assertEqual(state.calls[-1][:2], ("fork", 1))
        # Shares three: no state stood that early, so nothing is reused.
        self.assertEqual(cache.open(5, [1, 2, 3, 0]), 0)

    def test_marks_are_bounded_and_go_with_their_sequence(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state], marks=2)
        self.serve(cache, [kv, state], 1, [1, 2, 3, 4, 5, 6], marks=(2, 4, 6))
        self.assertEqual(sorted(cache.kept[1].marks), [4, 6])
        held = set(state.lengths)
        cache.clear()
        self.assertEqual((kv.lengths, state.lengths), ({}, {}))
        self.assertEqual(len(held), 3)                        # the sequence and its two marks
        # A request that fails is closed with its marks.
        cache.open(2, [1, 2, 3])
        state.grow(2, 2)
        cache.mark(2, 2)
        cache.retire(2, None)
        self.assertEqual(state.lengths, {})

    def test_a_model_hands_its_own_state_over_where_it_stood_and_gets_it_back(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state])
        cache.open(1, [1, 2, 3, 4, 5, 6])
        for pool in (kv, state):
            pool.grow(1, 4)
        cache.mark(1, 4, payload="hidden@4")
        for pool in (kv, state):
            pool.grow(1, 8)
        cache.retire(1, [1, 2, 3, 4, 5, 6, 7, 7], payload="hidden@8")
        self.assertEqual((cache.open(2, [1, 2, 3, 4, 9, 9]), cache.resumed(2)), (4, "hidden@4"))
        self.assertIsNone(cache.resumed(2))                   # once
        self.assertEqual((cache.open(3, [1, 2, 3, 4, 5, 6, 7, 7, 0]), cache.resumed(3)), (8, "hidden@8"))
        self.assertEqual((cache.open(4, [9, 9, 9]), cache.resumed(4)), (0, None))
        # Without state pools a model whose payload stands at its marks resumes there only.
        kv = Pool()
        marked = PromptCache([kv], at_marks=True)
        marked.open(1, [1, 2, 3, 4, 5, 6])
        kv.grow(1, 4)
        self.assertTrue(marked.mark(1, 4, payload="p"))
        kv.grow(1, 6)
        marked.retire(1, [1, 2, 3, 4, 5, 6], payload="q")
        self.assertEqual((marked.open(2, [1, 2, 3, 4, 5, 0, 0]), marked.resumed(2)), (4, "p"))
        self.assertFalse(PromptCache([Pool()]).at_marks)

    def test_a_request_whose_output_the_model_did_not_see_resumes_at_its_marks_only(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state])
        cache.open(1, [1, 2, 3, 4])
        for pool in (kv, state):
            pool.grow(1, 4)
        cache.mark(1, 4)
        state.grow(1, 9)                                      # rounds decoded on: the state is past the prompt
        cache.retire(1, [1, 2, 3, 4], stands=False)           # the KV kept to the prompt, the state not there
        self.assertEqual(cache.open(2, [1, 2, 3, 4, 5, 5]), 4)
        self.assertNotEqual(state.calls[-1][1], 1)            # from the mark at the prompt's end, not the sequence
        self.assertNotIn(1, state.lengths)                    # its own state went when it ended
        cache.clear()
        self.assertEqual((kv.lengths, len(state.lengths)), ({2: 4}, 2))   # the open request's, and the mark it holds
        cache.retire(2, None)
        self.assertEqual(state.lengths, {})
        unmarked_kv = Pool()
        unmarked = PromptCache([unmarked_kv], states=[Pool()], marks=0)
        unmarked.open(1, [1, 2, 3])
        unmarked.retire(1, [1, 2, 3], stands=False)
        self.assertEqual(unmarked.open(2, [1, 2, 3, 4]), 0)   # nowhere its state stood
        self.assertNotIn(1, unmarked_kv.lengths)              # so it was not kept at all

    def test_marking_bounds_the_open_sequences_that_hold_a_state_copy(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state], marking=1)
        for sequence in (1, 2):
            cache.open(sequence, [sequence, 5, 5, 5])
            for pool in (kv, state):
                pool.grow(sequence, 4)
        self.assertTrue(cache.mark(1, 4))
        self.assertFalse(cache.mark(2, 4))                    # one open sequence holds marks already
        self.assertEqual(len(state.lengths), 3)               # two sequences and one mark
        cache.retire(1, [1, 5, 5, 5], stands=False)
        cache.retire(2, [2, 5, 5, 5], stands=False)
        self.assertEqual(list(cache.kept), [1])               # the unmarked one resumes nowhere: not kept
        cache.open(3, [3, 5, 5, 5])
        for pool in (kv, state):
            pool.grow(3, 4)
        self.assertTrue(cache.mark(3, 4))                     # the budget is free again

    def test_a_mark_takes_no_more_states_than_it_keeps_and_a_full_state_pool_refuses_it(self):
        class Bounded(Pool):
            def fork(self, sequence, new):
                if len(self.lengths) >= 2:
                    raise MemoryError("the state pool's sequences are all open")
                super().fork(sequence, new)

        kv, state = Pool(), Bounded()                         # room for the sequence and one mark
        cache = PromptCache([kv], states=[state], marks=1)
        cache.open(1, [1, 2, 3, 4, 5, 6])
        for position in (2, 4):
            for pool in (kv, state):
                pool.grow(1, position)
            self.assertTrue(cache.mark(1, position))          # the older mark left before the copy
        self.assertEqual(sorted(cache.live[1]), [4])
        self.assertFalse(cache.mark(1, 3))                    # older than the mark kept
        cache.open(2, [7, 7, 7])                              # without the cache's say: the pool is full now
        state.grow(2, 2)
        self.assertFalse(cache.mark(2, 2))                    # refused, and the request goes on
        self.assertEqual(cache.live[2], {})
        self.assertEqual(len(state.lengths), 3)
        # With a sequence kept, its states go for the open one's mark.
        cache.retire(2, None)
        for pool in (kv, state):
            pool.grow(1, 6)
        cache.retire(1, [1, 2, 3, 4, 5, 6], stands=False)     # kept by its mark at 4
        cache.open(3, [8, 8, 8, 8])
        for pool in (kv, state):
            pool.grow(3, 2)
        self.assertTrue(cache.mark(3, 2))
        self.assertEqual((list(cache.kept), cache.counted["evicted"]), ([], 1))

    def test_open_sequences_give_their_marks_back_when_the_model_is_short_of_states(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state], marks=1, marking=1)
        cache.open(1, [1, 2, 3, 4])
        for pool in (kv, state):
            pool.grow(1, 2)
        self.assertTrue(cache.mark(1, 2))
        self.assertFalse(cache.evict())                       # nothing kept yet
        self.assertTrue(cache.unmark())
        self.assertEqual(set(state.lengths), {1})             # the mark's copy went, the sequence stays open
        self.assertFalse(cache.unmark())
        for pool in (kv, state):
            pool.grow(1, 4)
        self.assertTrue(cache.mark(1, 4))                     # it marks again and is kept
        cache.retire(1, [1, 2, 3, 4], stands=False)
        self.assertEqual(cache.open(2, [1, 2, 3, 4, 5]), 4)

    def test_a_sequence_keeps_the_place_it_resumed_from_without_a_second_copy(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state], entries=1, marks=2)
        cache.open(1, [1, 2, 3, 4, 5, 6])
        for pool in (kv, state):
            pool.grow(1, 4)
        cache.mark(1, 4, payload="at 4")
        cache.retire(1, [1, 2, 3, 4, 5, 6], stands=False)
        # A turn that parts from the kept one after the mark: it holds the mark it resumed from.
        self.assertEqual((cache.open(2, [1, 2, 3, 4, 8, 8, 8]), cache.resumed(2)), (4, "at 4"))
        self.assertEqual(sorted(cache.live[2]), [4])
        cache.retire(2, [1, 2, 3, 4, 8, 8, 8], stands=False)   # it made no mark of its own, and is kept by that one
        self.assertEqual(list(cache.kept), [2])               # the first left (one entry), its mark did not
        self.assertEqual(len(state.lengths), 1)               # one state copy all along
        self.assertEqual((cache.open(3, [1, 2, 3, 4, 9, 9, 9, 9]), cache.resumed(3)), (4, "at 4"))
        for position in (5, 6, 7):                            # the two newest it makes, beside the one it holds
            for pool in (kv, state):
                pool.grow(3, position)
            cache.mark(3, position)
        self.assertEqual(sorted(cache.live[3]), [4, 6, 7])
        cache.unmark()                                        # the open sequence's marks go, the kept one's stays
        self.assertEqual(cache.live[3], {})
        self.assertEqual(sorted(cache.kept[2].marks), [4])
        cache.retire(3, None)
        cache.clear()
        self.assertEqual((kv.lengths, state.lengths), ({}, {}))

    def test_a_tail_marks_a_place_before_the_prompts_end_where_a_chunk_ends(self):
        kv, state = Pool(), Pool()
        cache = PromptCache([kv], states=[state], marks=2, tail=2)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        def prefill(sequence, *positions):
            for position in positions:
                for pool in (kv, state):
                    pool.grow(sequence, position)
                self.assertTrue(cache.mark(sequence, position))

        cache.open(1, prompt)
        self.assertEqual(cache.stop(1), 8)                    # a chunk ends where the tail begins
        prefill(1, 8, 10)                                     # marked there, and at the prompt's end
        cache.retire(1, prompt, stands=False)
        self.assertEqual(cache.open(2, prompt), 8)            # the same prompt again
        self.assertEqual(cache.stop(2), 8)
        cache.retire(2, None)
        self.assertEqual(cache.open(3, prompt[:9] + [0, 0, 0]), 8)   # a turn that ends otherwise than it began the next
        cache.retire(3, None)
        # A turn that repeats the kept prompt whole resumes at its end and has no stop: its conversation repeats its
        # prompts whole, so the step would buy nothing. It holds the end it resumed from for the turn sent again.
        turn = prompt + [11, 12, 13, 14]
        self.assertEqual(cache.open(4, turn), 10)
        self.assertIsNone(cache.stop(4))
        prefill(4, 14)
        cache.retire(4, turn, stands=False)
        self.assertEqual(sorted(cache.kept[4].marks), [8, 10, 14])   # with the mark of the turn it extends
        self.assertEqual(cache.open(5, turn), 10)             # sent again: from the end before, and whole again
        self.assertIsNone(cache.stop(5))
        cache.retire(5, None)
        self.assertEqual(cache.open(6, turn[:12] + [0, 0]), 10)   # its last message edited
        cache.retire(6, None)
        # The turn sent again and kept takes over the marks of the one it replaces: 8 stays for a turn that parts
        # before 10.
        cache.open(9, turn)
        prefill(9, 14)
        cache.retire(9, turn, stands=False)
        self.assertEqual((list(cache.kept), sorted(cache.kept[9].marks)), ([9], [8, 10, 14]))
        further = turn + [15, 16, 17]
        self.assertEqual(cache.open(10, further), 14)
        prefill(10, 17)
        cache.retire(10, further, stands=False)
        self.assertEqual(sorted(cache.kept[10].marks), [10, 14, 17])   # one more than it may make
        self.assertEqual(len(state.lengths), 3)
        cache.clear()
        self.assertEqual(state.lengths, {})
        # No stop for a request that may not use the cache, a prompt no longer than the tail or shorter than
        # tail_least, a cache that resumes anywhere, and a sequence past the marking budget.
        cache.open(7, prompt, allow=False)
        self.assertIsNone(cache.stop(7))
        cache.open(8, [1, 2])
        self.assertIsNone(cache.stop(8))
        kv, state = Pool(), Pool()
        long = PromptCache([kv], states=[state], tail=2, tail_least=12)
        long.open(1, prompt)
        long.open(2, [9] + prompt + [11])
        self.assertEqual((long.stop(1), long.stop(2)), (None, 10))
        anywhere = PromptCache([Pool()], tail=2)
        anywhere.open(1, prompt)
        self.assertIsNone(anywhere.stop(1))
        kv, state = Pool(), Pool()
        one = PromptCache([kv], states=[state], tail=2, marking=1)
        for sequence in (1, 2):
            one.open(sequence, prompt)
            for pool in (kv, state):
                pool.grow(sequence, 8)
        self.assertTrue(one.mark(1, 8))
        self.assertEqual((one.stop(1), one.stop(2)), (8, None))
        with self.assertRaises(ValueError):
            PromptCache([Pool()], tail=-1)

    def chunks(self, cache, pools, sequence, prompt, chunk, allow=True):
        """A model's prefill: chunks of `chunk` tokens that end at the cache's stops, each chunk's end marked."""
        done = cache.open(sequence, prompt, allow)
        reused, ends = done, []
        while done < len(prompt):
            stop = cache.stop(sequence, done)
            done = min(len(prompt), done + chunk, stop if stop is not None and stop > done else len(prompt))
            for pool in pools:
                pool.grow(sequence, done)
            cache.mark(sequence, done)
            ends.append(done)
        cache.retire(sequence, list(prompt))
        return reused, ends

    def test_spread_marks_thin_out_with_their_distance_so_a_long_prompt_resumes_near_a_change(self):
        kv, states = Pool(), Pool()
        cache = PromptCache([kv], states=[states], marks=6, spread=100)
        prompt = list(range(1000, 2000))                              # 1,000 tokens, marked every 50
        self.chunks(cache, [kv, states], 1, prompt, 50)
        # None closer than 100 tokens but the newest, and wider apart the further back.
        self.assertEqual(sorted(cache.kept[1].marks), [450, 650, 750, 850, 950, 1000])
        # The newest marks alone would be 750..1000: a prompt changed at token 700 resumed nowhere. Here at 650.
        changed = prompt[:700] + [7] * 300
        self.assertEqual(cache.open(2, changed), 650)
        cache.retire(2, None)
        self.assertEqual(len(states.lengths), 1 + 6)                  # the kept sequence's state and its marks
        plain = PromptCache([Pool()], states=[Pool()], marks=6)
        self.chunks(plain, plain.pools + plain.states, 1, prompt, 50)
        self.assertEqual(sorted(plain.kept[1].marks), [750, 800, 850, 900, 950, 1000])
        self.assertEqual(plain.open(2, changed), 0)

    def test_spread_is_one_rule_while_a_prompt_is_marked_and_once_it_is_kept(self):
        cache = PromptCache([Pool()], states=[Pool()], marks=4, spread=64)
        for step in (16, 48, 200):   # chunks under and over the spread
            seen = []
            for end in range(step, 4096, step):
                seen = cache._thin([*seen, end], 4)
                self.assertEqual(seen[-1], end)
                self.assertLessEqual(len(seen), 4)
            self.assertEqual(seen, cache._thin(range(step, 4096, step), 4))
            gaps = [b - a for a, b in zip(seen, seen[1:])]
            self.assertEqual(gaps, sorted(gaps, reverse=True))        # wider the further back
            self.assertTrue(all(gap >= 64 for gap in gaps[:-1]))
        self.assertEqual(cache._thin(range(16, 4096, 16), 4), [3472, 3984, 4048, 4080])

    def test_a_head_two_conversations_share_is_stopped_at_marked_and_outlives_them(self):
        kv, states = Pool(), Pool()
        cache = PromptCache([kv], states=[states], entries=2, tail=4, shared_least=8)
        head = list(range(100, 140))                                  # a system prompt and tools: 40 tokens
        self.assertEqual(self.chunks(cache, [kv, states], 1, head + [1] * 30, 32), (0, [32, 64, 66, 70]))
        # A second conversation shares the head, where the first holds no state: it stops there and marks it.
        reused, ends = self.chunks(cache, [kv, states], 2, head + [2] * 30, 32)
        self.assertEqual((reused, ends), (0, [32, 40, 66, 70]))
        self.assertTrue(cache.kept[2].marks[40].shared)
        self.assertEqual(cache.telemetry()["prompt_cache"]["shared_marks"], 1)
        # A third resumes at the head and holds the mark too; the conversations that made it may leave.
        self.assertEqual(self.chunks(cache, [kv, states], 3, head + [3] * 30, 32)[0], 40)
        self.assertEqual(self.chunks(cache, [kv, states], 4, head + [4] * 30, 32)[0], 40)
        self.assertEqual(self.chunks(cache, [kv, states], 5, head + [5] * 30, 32)[0], 40)
        self.assertEqual(len(cache.kept), 2)                          # entries 2: the conversations that made it left
        self.assertNotIn(1, cache.kept)
        # The shared mark is outside the count of marks: each sequence still keeps its newest two beside it.
        self.assertEqual(sorted(cache.kept[5].marks), [40, 66, 70])
        self.assertEqual(sum(mark.shared for mark in cache.kept[5].marks.values()), 1)

    def test_a_shared_place_needs_states_or_marks_and_a_gain(self):
        kv = Pool()
        anywhere = PromptCache([kv], shared_least=8)                  # KV alone is cut anywhere: nothing to learn
        self.serve(anywhere, [kv], 1, list(range(40)) + [1] * 9)
        self.assertEqual(anywhere.open(2, list(range(40)) + [2] * 9), 40)
        self.assertIsNone(anywhere.stop(2))
        kv, states = Pool(), Pool()
        cache = PromptCache([kv], states=[states], shared_least=8)
        self.chunks(cache, [kv, states], 1, list(range(40)) + [1] * 9, 100)
        cache.open(2, list(range(5)) + [2] * 40)                      # 5 shared tokens: under shared_least
        self.assertIsNone(cache.stop(2, 0))
        cache.retire(2, None)
        cache.open(3, list(range(40)) + [3] * 9)
        self.assertEqual(cache.stop(3, 0), 40)
        self.assertIsNone(cache.stop(3, 44))                          # a model that passed it gets no stop there
        cache.retire(3, None)
        self.assertEqual(cache.cuts, {})


@unittest.skipUnless(STEP_LIBRARY, "Core's crit_core_steps library is not built: run `crit core build`")
class EngineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = write_files(self.root / "model", llama_files(**{"config.json": {
            "model_type": "llama", "text_config": {"vocab_size": VOCAB}}}))

    def engine(self, drafting=False, **options):
        package = write_package(self.root / f"package-{len(list(self.root.iterdir()))}", drafting=drafting,
                                options={"toy-model": {"prompt_cache": 4, **options}})
        engine = load(package, self.model)
        self.addCleanup(engine.close)
        return engine

    def ask(self, engine, prompt, output=6, **options):
        tokens, result = engine.generate(prompt, output, stop_tokens=[], sampling={"temperature": 0.0}, **options)
        return tokens, result["timing"]

    def test_a_follow_up_turn_prefills_only_what_the_kept_turn_lacks_and_answers_as_fresh(self):
        engine = self.engine()
        self.assertTrue(engine.info()["prompt_cache"])
        first = [3 + i % 50 for i in range(40)]
        reply, timing = self.ask(engine, first, prompt_cache=True)
        self.assertEqual((reply, timing["reused_prompt_tokens"]), (oracle(first, 6), 0))
        # The next turn: the conversation so far (the prompt, the reply) and more.
        turn = first + reply + [9, 8, 7]
        again, timing = self.ask(engine, turn, prompt_cache=True)
        # The pool held the prompt and every output but the last, still pending when the request ended.
        self.assertEqual(timing["reused_prompt_tokens"], len(first) + len(reply) - 1)
        self.assertEqual(timing["prefill_tokens"], len(turn))
        self.assertEqual(again, oracle(turn, 6))
        # Another reply in the first one's place: the turns share the first prompt alone.
        other = first + [200, 201] + [9, 8, 7]
        answer, timing = self.ask(engine, other, prompt_cache=True)
        self.assertEqual((answer, timing["reused_prompt_tokens"]), (oracle(other, 6), len(first)))
        counted = engine.telemetry()["prompt_cache"]
        self.assertEqual((counted["hits"], counted["reused_tokens"]), (2, len(first) + len(reply) - 1 + len(first)))

    def test_a_request_prefills_everything_unless_it_allows_the_cache(self):
        engine = self.engine()
        prompt = [3 + i % 50 for i in range(40)]
        self.ask(engine, prompt, prompt_cache=True)
        for options in ({}, {"prompt_cache": False}, {"logprobs": 2}):
            reply, timing = self.ask(engine, prompt + [5], **options)
            self.assertEqual((reply, timing["reused_prompt_tokens"]), (oracle(prompt + [5], 6), 0))
        # A replay follows its own continuation and never reads the cache.
        reply, timing = self.ask(engine, prompt, replay=oracle(prompt, 6), prompt_cache=True)
        self.assertEqual(timing["reused_prompt_tokens"], 0)

    def test_an_engine_without_the_option_keeps_nothing(self):
        package = write_package(self.root / "plain")
        engine = load(package, self.model)
        self.addCleanup(engine.close)
        prompt = [3 + i % 50 for i in range(40)]
        self.ask(engine, prompt, prompt_cache=True)
        _, timing = self.ask(engine, prompt + [5], prompt_cache=True)
        self.assertEqual(timing["reused_prompt_tokens"], 0)
        self.assertNotIn("prompt_cache", engine.info())

    def test_a_drafting_engine_resumes_from_the_rows_it_kept(self):
        engine = self.engine(drafting=True)
        first = [3 + i % 50 for i in range(24)]
        reply, _ = self.ask(engine, first, output=7, prompt_cache=True)
        turn = first + reply + [9, 8]
        again, timing = self.ask(engine, turn, output=7, prompt_cache=True)
        self.assertEqual(again, oracle(turn, 7))
        self.assertGreaterEqual(timing["reused_prompt_tokens"], len(first))

    def test_requests_at_once_each_fork_the_kept_sequence(self):
        engine = self.engine(concurrency=4)
        first = [3 + i % 50 for i in range(32)]
        self.ask(engine, first, prompt_cache=True)
        turns = [first + [100 + i, 7] for i in range(4)]
        with ThreadPoolExecutor(4) as pool:
            found = list(pool.map(lambda turn: self.ask(engine, turn, prompt_cache=True), turns))
        for turn, (reply, timing) in zip(turns, found):
            self.assertEqual((reply, timing["reused_prompt_tokens"]), (oracle(turn, 6), len(first)))

    def test_a_cancelled_request_keeps_what_it_computed(self):
        engine = self.engine(step_ms=20)
        prompt = [3 + i % 50 for i in range(64)]
        seen = []
        tokens, _ = engine.generate(prompt, 50, stop_tokens=[], sampling={"temperature": 0.0}, prompt_cache=True,
                                    on_tokens=seen.extend, cancelled=lambda: len(seen) >= 2)
        self.assertIsNone(tokens)
        reply, timing = self.ask(engine, prompt + [5], prompt_cache=True)
        self.assertEqual(reply, oracle(prompt + [5], 6))
        self.assertGreaterEqual(timing["reused_prompt_tokens"], len(prompt))


if __name__ == "__main__":
    unittest.main()
