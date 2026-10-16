"""Fixed-width catch-up for stateful drafters in a device round; host reference.

A stateful drafter keeps per-position state (for example its own attention
cache) conditioned on the target's per-position features. Before it proposes,
it must ingest the columns the previous round accepted. Only the device knows
how many that was, so a captured step processes a fixed number of columns and
masks them by position instead of resizing the launch.

Column c holds (feature[c], token[c]) at position frontier - emitted + c, where
feature[c] is the target's feature for verify column c of the previous round and
token[c] the c-th emitted token. Columns c >= emitted are junk and land at
positions >= frontier. Chain step j (j >= 1) writes position frontier + j - 1
before reading, so every junk position is overwritten before any read or lies
beyond what the round reads. The first draft comes from column emitted - 1.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CatchUp:
    positions: tuple        # drafter position of each catch-up column
    valid: tuple            # column c < emitted
    source: int             # column whose output seeds the first draft
    chain_positions: tuple  # positions of chain steps 1..draft_tokens-1


def catch_up_plan(frontier, emitted, width, draft_tokens):
    """Positions for one fixed-width catch-up plus the draft chain.

    `frontier` is the target's materialized count after the previous round,
    `emitted` that round's emitted count (0..width), `width` the fixed column
    count (draft width + 1 for a device round)."""
    if not isinstance(width, int) or width < 1 or not isinstance(draft_tokens, int) or draft_tokens < 1:
        raise ValueError("positive width and draft count required")
    if not isinstance(emitted, int) or not 0 <= emitted <= width:
        raise ValueError("emitted count must be within the catch-up width")
    if not isinstance(frontier, int) or frontier < emitted:
        raise ValueError("frontier must cover the emitted columns")
    return CatchUp(tuple(frontier - emitted + c for c in range(width)),
                   tuple(c < emitted for c in range(width)),
                   max(emitted - 1, 0),
                   tuple(frontier + j - 1 for j in range(1, draft_tokens)))


class PositionalDrafter:
    """Toy stateful drafter: per-position memory, causal reads. Each step stores
    (feature, token) at its position and returns a feature computed from every
    stored position up to and including its own; the draft token is derived from
    that feature. Writing before reading in each step is the invariant the
    fixed-width catch-up relies on."""

    def __init__(self, vocabulary):
        self.vocabulary, self.memory = vocabulary, {}

    def step(self, position, feature, token):
        self.memory[position] = (feature, token)
        acc = 7
        for p in range(position + 1):
            f, t = self.memory[p]
            acc = (acc * 1_000_003 + f * 31 + t) % 2_147_483_647
        return acc

    def token(self, feature):
        return feature % self.vocabulary

    def propose_fixed(self, plan, features, tokens, draft_tokens):
        """Fixed-width catch-up (junk columns included) followed by the chain."""
        outputs = [self.step(position, features[c], tokens[c]) for c, position in enumerate(plan.positions)]
        feature = outputs[plan.source]
        drafts = [self.token(feature)]
        for position in plan.chain_positions:
            feature = self.step(position, feature, drafts[-1])
            drafts.append(self.token(feature))
        return drafts[:draft_tokens]

    def propose_exact(self, frontier, emitted, features, tokens, draft_tokens):
        """Reference: ingest only the valid columns, then chain (emitted >= 1)."""
        if emitted < 1:
            raise ValueError("a round emits at least one token")
        for c in range(emitted):
            feature = self.step(frontier - emitted + c, features[c], tokens[c])
        drafts = [self.token(feature)]
        for j in range(1, draft_tokens):
            feature = self.step(frontier + j - 1, feature, drafts[-1])
            drafts.append(self.token(feature))
        return drafts
