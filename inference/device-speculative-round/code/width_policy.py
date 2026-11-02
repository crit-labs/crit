"""Adaptive draft width per round for speculative decoding, as a small pure-Python policy.

Each row (a request) keeps its own per-depth acceptance: a_d, how often the d-th draft was accepted given the drafts
before it were, counted with exponential decay so recent rounds weigh most. A round of width K is expected to emit

    E(K) = 1 + a_1 + a_1 a_2 + ... + a_1 ... a_K

tokens for that row, and costs base + K * step milliseconds: one target pass over K + 1 columns plus K draft steps,
fitted from the rounds the engine measured (least squares with the same decay, per batch size). The policy picks the
width among the captured ones that emits the most tokens per millisecond for the rows that decode together, sum of
their E(K) over the round's cost. Until a size has rounds of two widths measured, the configured base and step costs
stand in (only their ratio matters).

For the Qwen3.8-27B MTP engine the rule's numbers were a 13.8 ms round plus 1.09 ms per draft step with 0.95 decay;
they illustrate, the engine measures its own.
"""


class WidthPolicy:
    def __init__(self, widths, decay=0.95, prior=0.7, prior_weight=1.0, base_ms=1.0, step_ms=0.1):
        widths = sorted({int(w) for w in widths})
        if not widths or widths[0] < 0:
            raise ValueError("draft widths are non-negative integers")
        if not 0 < decay <= 1 or not 0 <= prior <= 1 or prior_weight < 0 or base_ms <= 0 or step_ms < 0:
            raise ValueError("decay in (0, 1], prior in [0, 1], a positive base cost and a non-negative step cost")
        self.widths, self.decay, self.prior, self.prior_weight = widths, decay, prior, prior_weight
        self.base_ms, self.step_ms = base_ms, step_ms
        self.rows = {}    # row -> [[hits, trials] per depth]
        self.costs = {}   # batch size -> decayed sums for the least-squares fit of ms = base + step * K

    def observe(self, row, drafted, accepted):
        """A round of `row` drafted `drafted` tokens and the target accepted the first `accepted` of them."""
        if not 0 <= accepted <= drafted:
            raise ValueError("a round accepts at most the drafts it made")
        depths = self.rows.setdefault(row, [])
        while len(depths) < drafted:
            depths.append([0.0, 0.0])
        for counts in depths:
            counts[0] *= self.decay
            counts[1] *= self.decay
        for d in range(min(accepted + 1, drafted)):
            depths[d][1] += 1.0
            depths[d][0] += 1.0 if d < accepted else 0.0

    def acceptance(self, row, depth):
        """a_depth (1-based) for `row`: its decayed hits over trials, pulled toward the prior by prior_weight."""
        depths = self.rows.get(row) or []
        hits, trials = depths[depth - 1] if depth <= len(depths) else (0.0, 0.0)
        return (hits + self.prior * self.prior_weight) / (trials + self.prior_weight) if trials + self.prior_weight else self.prior

    def expected(self, row, width):
        """Tokens a round of `width` drafts is expected to emit for `row`."""
        total, chain = 1.0, 1.0
        for depth in range(1, width + 1):
            chain *= self.acceptance(row, depth)
            total += chain
        return total

    def observe_cost(self, width, rows, ms):
        """A measured round: `width` drafts for `rows` rows took `ms` milliseconds."""
        sums = self.costs.setdefault(rows, {"n": 0.0, "x": 0.0, "y": 0.0, "xx": 0.0, "xy": 0.0, "widths": set()})
        for key in ("n", "x", "y", "xx", "xy"):
            sums[key] *= self.decay
        sums["n"] += 1.0
        sums["x"] += width
        sums["y"] += ms
        sums["xx"] += width * width
        sums["xy"] += width * ms
        sums["widths"].add(width)

    def cost(self, width, rows=1):
        """A round's cost in ms at `width` for `rows` rows: the fitted line where two widths were measured, else the
        configured costs."""
        sums = self.costs.get(rows)
        if sums and len(sums["widths"]) >= 2:
            spread = sums["n"] * sums["xx"] - sums["x"] ** 2
            if spread > 0:
                step = max(0.0, (sums["n"] * sums["xy"] - sums["x"] * sums["y"]) / spread)
                base = (sums["y"] - step * sums["x"]) / sums["n"]
                if base > 0:
                    return base + step * width
        return self.base_ms + self.step_ms * width

    def choose(self, rows):
        """The width among the policy's that emits the most tokens per ms for `rows` (row ids) decoding together; the
        narrower on a tie."""
        rows = list(rows)
        if not rows:
            return self.widths[0]
        return max(self.widths, key=lambda w: (sum(self.expected(r, w) for r in rows) / self.cost(w, len(rows)), -w))

    def forget(self, row):
        """A finished row's acceptance is dropped."""
        self.rows.pop(row, None)
