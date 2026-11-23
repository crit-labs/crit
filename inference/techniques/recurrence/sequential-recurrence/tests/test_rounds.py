"""sequential-recurrence's pool through Core's round driver (crit_plugins.rounds.Rounds) end to end: a toy model of two
gated-delta layers (each a causal convolution, then the recurrence, into a residual) decodes two sequences greedily
through rounds of verify windows whose drafts are wrong at every third position, eagerly and captured, with a held row
and a truncation, by every commit strategy, and emits exactly the tokens, states and convolution windows of decoding the
same sequences one token at a time through step() rows."""
from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

VOCAB, KEY_HEADS, HEADS, KEYS, VALUES, KERNEL = 48, 1, 2, 32, 32, 4
CHANNELS = 2 * KEY_HEADS * KEYS + HEADS * VALUES
GEOMETRY = {"keys": KEYS, "values": VALUES, "epsilon": 1e-6, "query_scale": KEYS ** -0.5, "normalize_keys": True,
            "normalize_queries": True}


class Toy:
    """Embedding, two layers of convolve + forward into a residual, a head: logits [tokens][vocab] FP32."""

    def __init__(self, torch, recurrence, pool, device):
        g = torch.Generator().manual_seed(41)
        r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(device)  # noqa: E731
        self.torch, self.recurrence, self.pool = torch, recurrence, pool
        self.embed = r(VOCAB, CHANNELS)
        self.conv = [r(CHANNELS, KERNEL, scale=0.5) for _ in range(2)]
        self.gates = [r(CHANNELS, 2 * HEADS, scale=0.2) for _ in range(2)]
        self.mix = [r(HEADS * VALUES, CHANNELS, scale=0.3) for _ in range(2)]
        self.head = r(CHANNELS, VOCAB, scale=2.0)

    def rows(self, x, matrix):
        """x @ matrix a row at a time: a row's bits whatever rows share the step (a GEMM's are not)."""
        return self.torch.cat([x[i:i + 1] @ matrix for i in range(x.shape[0])])

    def run(self, tokens, batch):
        torch = self.torch
        x = self.embed[tokens]
        t = x.shape[0]
        for layer in (0, 1):
            y = self.recurrence.convolve(layer, x, self.conv[layer], None, self.pool, batch)
            q, k, v = torch.split(y, [KEY_HEADS * KEYS, KEY_HEADS * KEYS, HEADS * VALUES], dim=-1)
            gate = self.rows(y, self.gates[layer])
            log_decay = -torch.nn.functional.softplus(gate[:, :HEADS])
            beta = torch.sigmoid(gate[:, HEADS:])
            out = self.recurrence.forward(layer, q.reshape(t, KEY_HEADS, KEYS), k.reshape(t, KEY_HEADS, KEYS),
                                          v.reshape(t, HEADS, VALUES), log_decay, beta, self.pool, batch, GEOMETRY)
            x = x + self.rows(out.reshape(t, -1), self.mix[layer])
        return self.rows(x, self.head).float()

    def verify(self, view, batch):
        return self.run(batch.tokens_t, batch).view(batch.rows, batch.width + 1, VOCAB)


class Greedy:
    def predictions_device(self, view, logits, draft_laws=None):
        return logits.argmax(-1)


class Oracle:
    """Drafts the serial decode's tokens, wrong at every third output position."""

    def __init__(self, torch, reference):
        self.torch, self.reference = torch, reference   # [rows][outputs] int64 on the device

    def draft(self, view, head, sampler):
        if not view.width:
            return
        torch = self.torch
        index = (view.out_count[:, None] + torch.arange(view.width, device=view.drafts.device)[None, :]).clamp(
            max=self.reference.shape[1] - 1)
        tokens = self.reference[:view.rows].gather(1, index)
        view.drafts[:, :view.width] = torch.where(index % 3 == 2, (tokens + 1) % VOCAB, tokens)


class Head:
    pass


class RoundTests(PluginTestCase):
    def setUp(self):
        import torch
        self.torch = torch
        if self.device() != "cuda":
            self.skipTest("rounds over CUDA graphs")
        g = torch.Generator().manual_seed(43)
        self.prompts = {"a": torch.randint(0, VOCAB, (9,), generator=g).tolist(),
                        "b": torch.randint(0, VOCAB, (13,), generator=g).tolist()}

    def prefill(self, toy, pool, sequence):
        tokens = self.prompts[sequence]
        pool.open(sequence)
        batch = StepBatch.build([(sequence, 0, len(tokens))], "cuda")
        logits = toy.run(self.torch.tensor(tokens, device="cuda"), batch)
        pool.accept(sequence, len(tokens))
        return int(logits[-1].argmax())

    def serial(self, plugin, outputs):
        """Each sequence decoded a token per step: (tokens, pool), the pool after every token but the last (which a
        round leaves pending too)."""
        pool = plugin.pool([0, 1], HEADS, KEYS, VALUES, "cuda", 4, 0, (CHANNELS, KERNEL), KEY_HEADS, 2)
        toy = Toy(self.torch, plugin, pool, "cuda")
        found = {}
        for s in "ab":
            tokens = [self.prefill(toy, pool, s)]
            while len(tokens) < outputs:
                position = len(self.prompts[s]) + len(tokens) - 1
                batch = StepBatch.build([(s, position, 1)], "cuda")
                tokens.append(int(toy.run(self.torch.tensor([tokens[-1]], device="cuda"), batch)[0].argmax()))
                pool.accept(s, 1)
            found[s] = tokens
        return found, pool

    def rounds(self, plugin, reference, outputs, width, capture, hold_at=None, truncate_at=None):
        from crit_plugins.rounds import Rounds
        torch = self.torch
        pool = plugin.pool([0, 1], HEADS, KEYS, VALUES, "cuda", 4, width, (CHANNELS, KERNEL), KEY_HEADS, 2)
        toy = Toy(torch, plugin, pool, "cuda")
        table = torch.tensor([reference[s] for s in "ab"], dtype=torch.int64, device="cuda")
        driver = Rounds(toy, Greedy(), rows=2, widest=width, device="cuda", drafter=Oracle(torch, table), head=Head(),
                        pools=[pool])
        tokens = {}
        for slot, s in enumerate("ab"):
            first = self.prefill(toy, pool, s)
            tokens[s] = [first]
            driver.bind(slot, s, len(self.prompts[s]), first, outputs - 1, len(self.prompts[s]) + outputs + width + 1)
        if capture:
            driver.capture(2, width, "greedy")
        launched = 0
        while any(len(tokens[s]) < outputs for s in "ab"):
            hold = (1,) if launched == hold_at else ()
            driver.launch(2, width, "greedy", hold=hold)
            found = driver.read()
            if launched == truncate_at:
                keep = [min(1, len(row["tokens"])) for row in found]
                driver.truncate(keep)
                found = [{**row, "tokens": row["tokens"][:k]} for row, k in zip(found, keep)]
            for s, row in zip("ab", found):
                tokens[s] += row["tokens"]
            launched += 1
            self.assertLess(launched, 4 * outputs)
        return tokens, pool, launched

    def test_rounds_decode_what_step_rows_decode_by_every_commit(self):
        torch = self.torch
        outputs, width = 14, 3
        for commit in ("deferred", "checkpoints", "recompute"):
            for kernels in (True, False):
                plugin = self.assemble(options={"commit": commit, "kernels": kernels})["recurrence"]
                if not kernels and self.assemble()["recurrence"].kernels("cuda", KEYS) is None:
                    continue   # the default is the torch formula already
                reference, serial = self.serial(plugin, outputs)
                for capture, hold_at, truncate_at in ((False, None, None), (True, 2, 3)):
                    with self.subTest(commit=commit, kernels=kernels, capture=capture):
                        tokens, pool, launched = self.rounds(plugin, reference, outputs, width, capture, hold_at,
                                                             truncate_at)
                        self.assertEqual(tokens, reference)
                        self.assertLess(launched, outputs - 1)   # drafts were kept
                        for s in "ab":
                            for layer in (0, 1):
                                self.assertTrue(torch.equal(pool.state(layer, s), serial.state(layer, s)))
                                self.assertTrue(torch.equal(pool.window(layer, s), serial.window(layer, s)))
