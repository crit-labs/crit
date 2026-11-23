"""Core's general capture helper (crit_plugins.captures): a model plugin that decodes its own way, with no drafts, no
verify and its state in a layout of its own (no bind, unbind or commit_round), captures one graph per row count and
replays it: the replays give the eager pass's tokens and leave its state, the warm-ups and the capture change no state
(the parts' reset_capture), a selection lacking a slot's fact or giving a reason of its own runs eagerly and refuses a
capture, a replay counts as a call of each slot its capture called, the captures' memory is planned through the
placement plugin, and the graphs share the engine's pool where Core's loader made one. The logic runs on the CPU over
EagerCapture; the real CUDA capture is checked where a GPU is."""
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "core" / "python"))

try:
    import torch
except ImportError:  # pragma: no cover - torch is in crit's test environment
    torch = None

CUDA = torch is not None and torch.cuda.is_available()
P = (1 << 31) - 1   # every value stays an exact int64 below it
ROWS, VOCAB = 4, 101


class RingState:
    """The toy's own state layout: per row a ring of its last four tokens and a running hash, all at fixed addresses.
    Nothing of the round driver's pool protocol: a step writes the state in place, and reset_capture() puts back the
    state it held when the capture began (snapshot())."""

    def __init__(self, device):
        long = dict(dtype=torch.long, device=device)
        self.ring, self.hash, self.at = torch.zeros(ROWS, 4, **long), torch.zeros(ROWS, **long), torch.zeros(1, **long)
        self.saved = None
        self.resets = 0

    def snapshot(self):
        self.saved = [t.clone() for t in (self.ring, self.hash, self.at)]

    def reset_capture(self):
        self.resets += 1
        for tensor, saved in zip((self.ring, self.hash, self.at), self.saved):
            tensor.copy_(saved)


class OwnDecoder:
    """A model plugin's own decode step over `rows` rows: no drafts, no verify. Reads each row's token from the fixed
    input buffer, mixes it with the row's ring and hash, writes the next token into the fixed output buffer and steps
    the state in place. The projections slot's plugin (here a callable counted by the parts) does the mixing."""

    def __init__(self, device, project):
        self.state, self.project = RingState(device), project
        self.tokens = torch.zeros(ROWS, dtype=torch.long, device=device)
        self.next = torch.zeros(ROWS, dtype=torch.long, device=device)
        self.calls = 0

    def forward(self, rows):
        self.calls += 1
        s = self.state
        mixed = self.project((s.ring[:rows].sum(1) * 7 + s.hash[:rows] * 13 + self.tokens[:rows] * 31) % P)
        self.next[:rows] = mixed % VOCAB
        slot = s.at % 4
        s.ring[:rows].scatter_(1, slot.expand(rows).view(rows, 1), self.tokens[:rows].view(rows, 1))
        s.hash[:rows] = (s.hash[:rows] * 131 + self.tokens[:rows]) % P
        s.at += 1
        return self.next

    def decode(self, run, rows, first, steps):
        """Greedy decoding through run(rows), the step's inputs written into the fixed buffer before each."""
        self.tokens[:rows] = torch.tensor(first, device=self.tokens.device)
        out = []
        for _ in range(steps):
            produced = run(rows)[:rows].clone()
            out.append(produced.tolist())
            self.tokens[:rows] = produced
        return out


class Parts:
    """A factory's parts: each slot's plugin, its facts and objects, and optionally the loader's counter and context."""

    def __init__(self, use, provides, objects=None):
        self.use, self.provides, self.objects = use, provides, dict(objects or {})

    def slots(self):
        return list(self.use)

    def plugin(self, slot):
        return self.use.get(slot)

    def facts(self, slot):
        return tuple(self.provides.get(self.use.get(slot), ()))

    def get(self, slot, default=None):
        return self.objects.get(slot, default)


USE = {"model": "own-model", "placement": "toy-placement", "projections": "toy-projections", "sampler": "toy-sampler"}
FACTS = {"toy-projections": ["rounds:projections"]}


@unittest.skipIf(torch is None, "torch is not installed")
class CaptureTests(unittest.TestCase):
    def build(self, device, parts=None, backend=None, project=None, **options):
        from crit_plugins.captures import Captures, EagerCapture
        observer = getattr(parts, "observer", None)

        def counted(x):
            if observer is not None and observer.active:
                observer["projections"] += 1
            return (x * 3 + 1) % P
        model = OwnDecoder(device, project or counted)
        model.state.snapshot()
        captures = Captures(model.forward, device=device, keys=(1, 2, ROWS), parts=parts, uses=["projections"],
                            extra_parts=[model.state], backend=backend or EagerCapture(), **options)
        return model, captures

    def test_replays_give_the_eager_steps_tokens_and_leave_its_state(self):
        eager, _ = self.build("cpu")
        model, captures = self.build("cpu", parts=Parts(USE, FACTS))
        self.assertEqual(captures.prepare(), [1, 2, ROWS])
        self.assertEqual(model.state.resets, 4 * 3, "per key a reset before each of two warm-ups, one after them and "
                                                    "one after the capture")
        self.assertEqual(model.state.hash.tolist(), [0] * ROWS, "the warm-ups and the capture changed no state")
        for rows, first in ((ROWS, [3, 5, 7, 11]), (2, [1, 2])):
            expected = eager.decode(eager.forward, rows, first, 6)
            self.assertEqual(model.decode(captures.run, rows, first, 6), expected)
        self.assertEqual(model.state.ring.tolist(), eager.state.ring.tolist())
        self.assertEqual(model.state.hash.tolist(), eager.state.hash.tolist())
        output = captures.run(1)
        self.assertIs(output, captures.graphs[1][1], "a replay's output is the captured one, at its address")
        self.assertNotIn(3, captures.graphs)
        captures.run(3)
        counts = captures.telemetry()["captures"]["captures"]
        self.assertEqual((counts["replays"], counts["eager"]), (13, 1), "a key not captured runs eagerly")
        self.assertEqual(counts["keys"], ["1", "2", "4"])

    def test_a_selection_lacking_a_fact_or_giving_a_reason_runs_eagerly_and_refuses_a_capture(self):
        from crit_plugins.captures import CapturesRefused, capture_problems
        self.assertEqual(capture_problems(Parts(USE, FACTS), ["projections"]), [])
        self.assertEqual(capture_problems(Parts(USE, {"toy-projections": ["prompts:projections"]}), ["projections"]),
                         ["toy-projections in projections does not provide rounds:projections: its device form is "
                          "not known to run inside a captured graph"])
        self.assertEqual(capture_problems(Parts(USE, FACTS), ["norms"]),
                         ["no plugin fills norms, which the captured pass calls"])
        self.assertEqual(capture_problems(Parts(USE, {}), None)[0].split(":")[0],
                         "toy-projections in projections does not provide rounds", "by default every computing slot")

        class FileTier:
            def round_problems(self):
                return ["its rows are read from a file"]
        model, captures = self.build("cpu", parts=Parts(USE, FACTS, {"projections": FileTier()}))
        self.assertEqual(captures.prepare(), [])
        self.assertIn("as configured: its rows are read from a file", captures.problems[0])
        with self.assertRaises(CapturesRefused):
            captures.capture(1)
        self.assertEqual(len(captures.run(1)), ROWS)
        self.assertEqual((model.calls, captures.counts["eager"]), (1, 1), "the eager pass, uncaptured")

    def test_a_fact_of_the_plugins_choice_is_checked(self):
        from crit_plugins.prompt_graphs import prompt_fact
        _, captures = self.build("cpu", parts=Parts(USE, FACTS), fact=prompt_fact)
        self.assertEqual(captures.prepare(), [])
        self.assertIn("does not provide prompts:projections", captures.problems[0])

    def test_replays_count_the_slots_their_capture_called_and_plans_go_through_placement(self):
        from crit_plugins.captures import EagerCapture
        from crit_plugins.observe import Calls
        plans = []

        class Placement:
            def plan(self, tensors, budget):
                plans.append((dict(tensors), dict(budget)))
                return {name: "device" for name in tensors}

        class Grows(EagerCapture):
            def capture(self, device, warm, run, pool, observer=None):
                graph, output, called, _ = super().capture(device, warm, run, pool, observer)
                return graph, output, called, 1000
        parts = Parts(USE, FACTS, {"placement": Placement()})
        parts.observer = Calls()
        backend = Grows(free=5000)
        model, captures = self.build("cpu", parts=parts, backend=backend, name="own-decode")
        captures.prepare()
        self.assertEqual(captures.slot_calls[1], ["projections"])
        parts.observer.reset()
        model.decode(captures.run, 2, [4, 4], 5)
        self.assertEqual(parts.observer["projections"], 5, "each replay one call of the slot its capture called")
        self.assertEqual(backend.replays, 5)
        self.assertEqual([p[0] for p in plans], [{"own-decode:captures": 1000}] * 3)
        self.assertEqual(plans[0][1], {"device": 6000, "host": 0})
        self.assertEqual(captures.captured_bytes, 3000)

        class Off:
            def plan(self, tensors, budget):
                return {name: "host" for name in tensors}
        from crit_plugins.captures import CapturesRefused
        _, refused = self.build("cpu", parts=Parts(USE, FACTS, {"placement": Off()}), backend=Grows())
        with self.assertRaises(CapturesRefused):
            refused.capture(1)

    def test_keys_share_the_engines_pool_where_their_parts_carry_one(self):
        from crit_plugins.captures import EagerCapture
        pools = []

        class Recording(EagerCapture):
            def capture(self, device, warm, run, pool, observer=None):
                pools.append(pool)
                return super().capture(device, warm, run, pool, observer)
        parts = Parts(USE, FACTS)
        parts.context = types.SimpleNamespace(graphs="engine pool")
        _, captures = self.build("cpu", parts=parts, backend=Recording())
        captures.prepare()
        self.assertEqual(pools, ["engine pool"] * 3)
        pools.clear()
        _, own = self.build("cpu", parts=parts, backend=Recording(), share_pool=False)
        own.prepare()
        self.assertEqual(pools, ["captures, 1", "captures, 2", "captures, 4"])

    def test_names_and_warm_ups_are_checked(self):
        with self.assertRaises(ValueError):
            self.build("cpu", name="own:decode")
        with self.assertRaises(ValueError):
            self.build("cpu", warmups=-1)
        model, captures = self.build("cpu", warmups=0)
        captures.capture(1)
        self.assertEqual(model.calls, 1, "no warm-up: the capture's run alone")

    @unittest.skipIf(not CUDA, "no CUDA device")
    def test_cuda_graphs_replay_the_eager_steps_tokens(self):
        from crit_plugins.captures import CudaCapture
        eager, _ = self.build("cuda", project=lambda x: (x * 3 + 1) % P)
        model, captures = self.build("cuda", project=lambda x: (x * 3 + 1) % P, backend=CudaCapture(),
                                     share_pool=False)
        self.assertEqual(captures.prepare(), [1, 2, ROWS])
        calls = model.calls
        expected = eager.decode(eager.forward, ROWS, [3, 5, 7, 11], 6)
        self.assertEqual(model.decode(captures.run, ROWS, [3, 5, 7, 11], 6), expected)
        self.assertEqual(model.calls, calls, "every step replayed a graph: no forward call")
        self.assertEqual(model.state.hash.tolist(), eager.state.hash.tolist())
        self.assertGreaterEqual(captures.captured_bytes, 0)


if __name__ == "__main__":
    unittest.main()
