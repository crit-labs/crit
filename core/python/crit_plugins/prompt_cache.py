"""Core's prompt cache: sequences kept in the pools after their request ended, so a later prompt that begins as one of
them did prefills only the rest (a conversation's next turn, an agent's next tool step).

A model plugin keeps one PromptCache over its pools and calls it where it opened and closed sequences before:

    cache = PromptCache(pools=[kv_pool], states=[state_pool], entries=8)
    reused = cache.open(session.handle, request.prompt, allow=request.prompt_cache)   # in open(): 0, or the prefix reused
    ...                                              # prefill starts at `reused`; StepResult.reused_prompt_tokens says so
    step_item(row, len(prompt), frontier, stop=cache.stop(session.handle))   # with `tail`: a chunk ends before it
    cache.mark(session.handle, position)             # optional, where the state pools stand at `position` of the prompt
    cache.retire(session.handle, tokens)             # in retire(): the tokens whose rows the pools hold, or None to drop

It is a ledger: it touches no KV byte, only the pools' own open, fork, keep and close (crit_plugins.device KV_POOL, and
the same four on a recurrence slot's state pool). `pools` are cut at any position: a kept sequence is forked (its
pages shared until one of the two writes) and the fork kept to the reused prefix. `states` hold one state per
sequence, which a fork copies whole and nothing rolls back (recurrent state, a sliding window's ring), so they are
reused only where they stood: at the end of a kept sequence, or at a position the model marked while it prefilled.
With states the prefix reused is the longest such position the new prompt still shares. A sequence that resumed at a
mark holds that mark too, the same state copy and no second one, beside the `marks` it makes itself, so a
conversation keeps the place its last turn resumed from: a turn sent again with another last message resumes there.
A kept sequence that a newer one extends hands its marks on as it leaves.

Which marks a sequence keeps is the cache's rule, not the model's: the model marks every chunk's end. By default
the newest `marks` stay, so a prompt resumes near a kept one's end. With `spread` (a token count) they thin out with
their distance from the newest instead: no two closer than `spread` tokens but the newest, and past `marks` of
them the one goes whose neighbours lie closest for their distance from the newest, so the marks kept lie about
twice as far apart with each step back. A long prompt changed in its middle then resumes within a few times its
distance to the change, where the newest marks alone resume nowhere.

With `shared_least`, the cache also learns where conversations part. A prompt that shares more tokens with a kept
sequence than the place it can resume from (at least `shared_least` more: a system prompt and tool list that every
conversation of a client repeats) gets that place as a stop, so a chunk ends there and the model's mark of it is
a shared mark, kept beside the others and outside their count. The next prompt with that head resumes there, holds
the mark too, and so the place outlives the conversation that made it.

A later prompt rarely repeats a prompt to its last token: the same prompt sent again computes its last token anew, and
a chat template may end a prompt otherwise than it renders that turn in the next one (a thinking model's opening
tag). With `tail` the cache also marks a place that many tokens before a prompt's end: stop() is the position
there, which the model's prompt chunk does not pass (crit_core.batch step_item's stop), and the model marks it as
it commits that chunk, and the prompt's end as it commits the next. The shorter chunk costs the prompt a step, so
a prompt has no stop where the step would not pay: one that resumed at a kept prompt's end (its conversation
repeats its prompts whole, and it holds that place for the turn sent again), and one shorter than `tail_least`
(prefilled whole in about a step's time).

A model that keeps state of its own beside the pools (a hidden row its draft layer reads, say) hands it over where it
stood: mark(sequence, position, payload) and retire(sequence, tokens, payload), and resumed(sequence) gives back the
payload of the place open() resumed from. Such a model resumes at those places only (at_marks), as one with states
does. A request whose output the model did not see (Core's native rounds publish it) ends with its pools past what
the model can name: it keeps the pools to the tokens it knows and retires with stands False, so later prompts resume
at its marks and never at its end, and its own state in the state pools goes at once. A state is large where a KV page
is small (a recurrent model's is its every layer's), so `marking` bounds the open sequences that make marks at once:
one past it prefills unmarked, though it may still resume from a kept one. Marks need no state of their own in the
pool: a model may let them stand in the states no open request uses, and take them back when a request or a score
needs one, with evict() for the kept sequences' and then unmark() for the open ones'. A mark the state pool has no
room for takes the kept sequences', least recently used first, and is refused when none is left.

A request that does not allow the cache (Eval's timed requests, StepRequest.prompt_cache false) reuses nothing, marks
nothing and is not kept: it runs as on an engine without a cache. The last prompt token is never reused: its row gives
the first output token. Kept sequences leave least recently used first, past `entries` of them or `tokens` positions,
and on evict(), which a model calls when its admission runs short of pages and which a pool that offers
on_exhausted(callback) calls itself before it refuses a page; a kept sequence that a newer one extends leaves too.
Every method runs on Core's one engine worker, as the model's do, so it needs no lock.
"""
from collections import OrderedDict

SHARED = 2   # the shared marks a kept sequence holds, the longest


class _Mark:
    """A place the states stood: the state pools' copy of it (`name`, None without state pools) and the model's own
    state there. The sequences that resumed from it hold it with the one that made it; the copy goes with the last."""
    __slots__ = ("name", "payload", "holders", "end", "shared")

    def __init__(self, name, payload, end=False, shared=False):
        self.name, self.payload, self.holders, self.end = name, payload, 1, end   # end: it stood at its prompt's end
        self.shared = shared   # where prompts of other conversations part: kept outside the count of `marks`


class _Entry:
    __slots__ = ("sequence", "tokens", "marks", "end")

    def __init__(self, sequence, tokens, marks, end):
        # marks: {position: _Mark}; end: (payload,) where the states stand at the sequence's end, None where they do
        # not (the state pools then hold nothing of the sequence itself).
        self.sequence, self.tokens, self.marks, self.end = sequence, tokens, marks, end


def shared(a, b):
    """The length of the longest prefix two token lists share."""
    limit = min(len(a), len(b))
    return next((i for i in range(limit) if a[i] != b[i]), limit)


class PromptCache:
    """entries: the most sequences kept; tokens: the most positions they hold together (None: no bound); least: the
    shortest prefix worth reusing; marks: the most positions a sequence marks and keeps, the newest, beside the
    one it resumed from (0: resumed at a kept sequence's end only); at_marks: resume only where a sequence ended or was marked, also without state pools
    (a model whose own payload stands there); marking: the most open sequences that make marks at once (None: any);
    tail: a prompt chunk ends this many tokens before the prompt's end, a place marked beside the end (0: no such
    place); tail_least: the shortest prompt that has it; spread: the marks kept thin out with their distance from
    the newest, none but the newest closer than this many tokens to the one before it (0: the newest `marks`); shared_least: a place a prompt shares
    with a kept sequence this many tokens past where it resumes is stopped at and kept as a shared mark (0: none)."""

    def __init__(self, pools, states=(), entries=8, tokens=None, least=1, marks=2, at_marks=False, marking=None,
                 tail=0, tail_least=0, spread=0, shared_least=0):
        self.pools, self.states = list(pools), list(states)
        if not self.pools and not self.states:
            raise ValueError("a prompt cache keeps sequences of one pool or more")
        if entries < 1 or least < 1 or min(marks, tail, spread, shared_least) < 0 or (
                tokens is not None and tokens < 1):
            raise ValueError("a prompt cache keeps an entry or more, reuses a token or more and holds marks >= 0, "
                             "tail >= 0, spread >= 0, shared_least >= 0")
        self.entries, self.tokens, self.least, self.marks = int(entries), tokens, int(least), int(marks)
        self.at_marks = bool(at_marks or self.states)
        self.marking = None if marking is None else int(marking)
        self.tail, self.tail_least = int(tail), int(tail_least)
        self.spread, self.shared_least = int(spread), int(shared_least)
        self.cuts = {}                       # an open sequence -> the place it shares with a kept one, not yet marked
        self.kept = OrderedDict()            # sequence -> _Entry, least recently used first
        self.live = {}                       # an open sequence that may be kept: its marks so far, {position: _Mark}
        self.making = set()                  # the open sequences that made a mark (`marking` bounds them)
        self.lengths = {}                    # an open sequence that may be kept -> its prompt's length
        self.whole = set()                   # the open sequences that resumed at a kept prompt's end
        self.origin = {}                     # an open sequence -> the position of the mark it resumed from
        self.payloads = {}                   # an open sequence -> the payload of the place it resumed from
        self._marks = 0
        for pool in self.pools + self.states:
            if callable(getattr(pool, "on_exhausted", None)):
                pool.on_exhausted(self.evict)   # a kept sequence goes before a request is refused a page
        self.counted = {"hits": 0, "misses": 0, "reused_tokens": 0, "kept": 0, "evicted": 0, "shared_marks": 0}

    # -- a request's sequence --------------------------------------------------------------------------------------------
    def open(self, sequence, prompt, allow=True):
        """Opens `sequence` in every pool, from the kept sequence that shares the longest reusable prefix with `prompt`
        when the request allows it. Returns the prefix reused: the positions the pools already hold for it."""
        prompt = list(prompt)
        found, common = self._match(prompt) if allow else (None, 0)
        # A place this prompt shares with a kept sequence well past where it resumes: worth a mark of its own.
        if (self.shared_least and self.at_marks and self.marks
                and common - (found[1] if found else 0) >= self.shared_least):
            self.cuts[sequence] = common
        if found is None:
            for pool in self.pools + self.states:
                pool.open(sequence)
            if allow:
                self.live[sequence], self.lengths[sequence] = {}, len(prompt)
                self.counted["misses"] += 1
            return 0
        entry, length, source, payload, mark = found
        for pool in self.pools:
            pool.fork(entry.sequence, sequence)
            pool.keep(sequence, length)
        for pool in self.states:
            pool.fork(source, sequence)
        self.kept.move_to_end(entry.sequence)
        self.live[sequence], self.lengths[sequence] = {}, len(prompt)
        if mark is not None and self.marks:   # the place it resumed from stays one of its own
            mark.holders += 1
            self.live[sequence][length], self.origin[sequence] = mark, length
        if mark.end if mark is not None else self.at_marks:   # a kept prompt's end, or a kept sequence's
            self.whole.add(sequence)
        self.payloads[sequence] = payload
        self.counted["hits"] += 1
        self.counted["reused_tokens"] += length
        return length

    def resumed(self, sequence):
        """The payload of the place open() resumed `sequence` from (a mark's, a kept sequence's end's), once; None
        where it resumed from none or the model handed none over."""
        return self.payloads.pop(sequence, None)

    def stop(self, sequence, frontier=None):
        """The next position a prompt chunk of `sequence` does not pass, so that a chunk ends there and the model
        marks it as it commits (the model reports the rows it computed, and the rest is its next step's): first the
        place the prompt shares with a kept sequence (shared_least), until a chunk reached it (a mark at or past it,
        or `frontier`, the prompt tokens computed so far, where the model passes it), then the position `tail`
        tokens before the prompt's end. None without a tail, for a cache that resumes anywhere or holds no marks,
        for a sequence whose request did not allow the cache or that may not mark, for one that resumed at a kept
        prompt's end and for a prompt shorter than tail_least."""
        cut = self.cuts.get(sequence)
        if cut is not None and frontier is not None and frontier >= cut:
            del self.cuts[sequence]
            cut = None
        if cut is not None and sequence in self.live and self._may_mark(sequence):
            return cut
        length = self.lengths.get(sequence, 0)
        place = length - self.tail
        if (not self.tail or not self.at_marks or not self.marks or sequence not in self.live
                or sequence in self.whole or length < self.tail_least or place < self.least
                or not self._may_mark(sequence)):
            return None
        return place

    def mark(self, sequence, position, payload=None):
        """The state pools, and the model's own state `payload`, stand at `position` of `sequence` (after a committed
        prompt chunk): a copy of the pools' state is kept with it, where a later prompt sharing that many tokens
        resumes. Nothing for a cache that resumes anywhere (no states, not at_marks), without marks, or for a
        sequence whose request did not allow the cache. The marks it made past `marks` leave before the copy is
        made, so a sequence marking holds `marks` states of its own at most; a state pool without room for the
        copy takes the kept sequences', least recently used first, and with none left refuses the mark, not the
        request. Returns whether it marked."""
        marks = self.live.get(sequence)
        cut = self.cuts.get(sequence)
        if cut is not None and position >= cut:
            del self.cuts[sequence]   # a chunk reached the shared place: marked now, or passed
        if marks is None or not self.at_marks or not self.marks or position < self.least or position in marks:
            return False
        if not self._may_mark(sequence):
            return False   # enough open sequences make marks: this one prefills unmarked
        shared = position == cut
        if not shared:
            made = [found for found in marks if found != self.origin.get(sequence) and not marks[found].shared]
            newest = self._thin([*made, position], self.marks)
            if position not in newest:
                return False   # older than every mark kept
            for old in [found for found in made if found not in newest]:
                self._drop_mark(marks.pop(old))
        name = None
        if self.states:
            self._marks += 1
            name = ("prompt-cache-mark", self._marks)
            while True:
                copied = []
                try:
                    for pool in self.states:
                        pool.fork(sequence, name)
                        copied.append(pool)
                    break
                except MemoryError:
                    for pool in copied:
                        pool.close(name)
                    if not self.evict():   # a kept sequence's states for this one's, while there is one
                        return False
        marks[position] = _Mark(name, payload, position == self.lengths[sequence], shared)
        self.counted["shared_marks"] += shared
        self.making.add(sequence)
        return True

    def retire(self, sequence, tokens=None, payload=None, stands=True):
        """The request ended. tokens: the tokens whose rows the pools hold for it, in order (its prompt and the output
        it kept), which keeps the sequence for later prompts; None closes it (a failed request, state the model
        cannot vouch for). payload: the model's own state at the sequence's end. stands False: the state pools do not
        stand at the end of `tokens` (the model kept the pools to the tokens it knows), so later prompts resume at
        its marks only. A sequence whose request did not allow the cache is closed either way."""
        marks = self.live.pop(sequence, None)
        self.making.discard(sequence)
        self.whole.discard(sequence)
        self.origin.pop(sequence, None)
        self.lengths.pop(sequence, None)
        self.payloads.pop(sequence, None)
        self.cuts.pop(sequence, None)
        if marks is None or tokens is None or len(tokens) < self.least:
            for mark in (marks or {}).values():
                self._drop_mark(mark)
            for pool in self.pools + self.states:
                pool.close(sequence)
            return False
        tokens = tuple(tokens)
        # A kept sequence this one extends holds nothing this one lacks, but for the marks before its end: this one
        # takes them over, the newest, to one more than it may make itself.
        for other in [e for e in self.kept.values() if len(e.tokens) <= len(tokens)
                      and tokens[:len(e.tokens)] == e.tokens]:
            for position, mark in other.marks.items():
                if position not in marks:
                    mark.holders += 1
                    marks[position] = mark
            self._close(other)
        held = [p for p in marks if p <= len(tokens)]
        keep = set(self._thin([p for p in held if not marks[p].shared], self.marks + 1))
        keep |= set(sorted(p for p in held if marks[p].shared)[-SHARED:])   # the longest shared places
        for position in [p for p in marks if p not in keep]:
            self._drop_mark(marks.pop(position))
        if not stands:
            if self.at_marks and not marks:   # nowhere to resume it: nothing worth its pages
                for pool in self.pools + self.states:
                    pool.close(sequence)
                return False
            for pool in self.states:          # its own state stands nowhere a prompt resumes
                pool.close(sequence)
        self.kept[sequence] = _Entry(sequence, tokens, marks, (payload,) if stands else None)
        self.counted["kept"] += 1
        while len(self.kept) > self.entries or (self.tokens is not None and self.held() > self.tokens
                                                 and len(self.kept) > 1):
            self.evict()
        return True

    # -- room ------------------------------------------------------------------------------------------------------------
    def held(self):
        """The positions the kept sequences hold, each counted whole (shared pages are counted in each)."""
        return sum(len(entry.tokens) for entry in self.kept.values())

    def evict(self):
        """Closes the least recently used kept sequence. Returns whether there was one: a model short of pages calls it
        until its pool has room or it answers False."""
        if not self.kept:
            return False
        self._close(next(iter(self.kept.values())))
        self.counted["evicted"] += 1
        return True

    def unmark(self):
        """Drops the marks open sequences hold, their state copies with them (one a kept sequence holds too stays its).
        Returns whether there was one: a model short of states calls it once evict() has nothing left. Such a sequence
        is kept only if it marks again."""
        found = False
        for marks in self.live.values():
            while marks:
                self._drop_mark(marks.popitem()[1])
                found = True
        self.making.clear()
        self.origin.clear()
        return found

    def clear(self):
        while self.evict():
            pass

    def telemetry(self):
        return {"prompt_cache": {**self.counted, "entries": len(self.kept), "tokens": self.held()}}

    # -- inside ----------------------------------------------------------------------------------------------------------
    def _may_mark(self, sequence):
        return sequence in self.making or self.marking is None or len(self.making) < self.marking

    def _thin(self, positions, limit):
        """The marks kept of `positions`, `limit` at most. Without spread the newest. With it, as if the marks came
        one by one, oldest first: a mark that stood within `spread` of the one before it leaves when a newer one
        comes, and past `limit` the mark goes whose two gaps, joined, are the smallest for their distance from the
        newest. What stays lies about twice as far apart with each step back, whatever the chunks were, and the
        marks of a prompt still being prefilled are thinned by the same rule as a kept one's."""
        positions = sorted(positions)
        if not self.spread:
            return positions[-limit:] if limit else []
        kept = []
        for position in positions:
            if len(kept) >= 2 and kept[-1] - kept[-2] < self.spread:
                kept.pop()
            kept.append(position)
            while len(kept) > limit:
                def joined(index, newest=position):
                    before = kept[index - 1] if index else 0
                    return (kept[index + 1] - before) / (newest - before), -index
                del kept[min(range(len(kept) - 1), key=joined)]
        return kept

    def _match(self, prompt):
        """((entry, length, the sequence the state pools fork from, the payload there, the mark there or None) of
        the kept sequence with the longest reusable prefix of `prompt`, or None; the longest prefix `prompt` shares
        with a kept sequence, reusable or not). The last prompt token is always computed."""
        best, most = None, 0
        for entry in self.kept.values():
            common = min(shared(entry.tokens, prompt), len(prompt) - 1)
            most = max(most, common)
            if common < self.least:
                continue
            if not self.at_marks:
                length, source, payload, mark = common, None, None, None
            else:
                # States resume where they stood: the kept sequence's end, or a marked position.
                stood = {position: (found.name, found.payload, found) for position, found in entry.marks.items()}
                if entry.end is not None:
                    stood[len(entry.tokens)] = (entry.sequence, entry.end[0], None)
                length = max((p for p in stood if p <= common), default=0)
                source, payload, mark = stood.get(length, (None, None, None))
            if length >= self.least and (best is None or length > best[1]):
                best = (entry, length, source, payload, mark)
        return best, most

    def _close(self, entry):
        self.kept.pop(entry.sequence, None)
        for mark in entry.marks.values():
            self._drop_mark(mark)
        for pool in self.pools + (self.states if entry.end is not None else []):
            pool.close(entry.sequence)

    def _drop_mark(self, mark):
        mark.holders -= 1
        if not mark.holders and mark.name is not None:
            for pool in self.states:
                pool.close(mark.name)


__all__ = ["PromptCache", "shared"]
