"""Run directly: bounded queued prefetch, complete publication and owned reuse."""
import planning
import transfers
from tiers import MemoryTier, TierMove
from prefetch import PrefetchController, tier_prefetch


def main():
    slow, fast = MemoryTier(64), MemoryTier(64)
    slow.put('prefix', 'opaque-complete-state', b'state')
    identity = dict(fingerprint='weights+conditioning', codec='engine.v1', components={'kv', 'recurrence'})
    source = planning.PrefixSource('prefix', (1, 2, 3), **identity, ready=False)
    controller = PrefetchController(planning=planning, clock=lambda: 0)
    def prepare():
        move = TierMove(slow, fast, 'prefix', 'opaque-complete-state',
                        transfers.TransferLimits(2, 4, 2, 5), transport=transfers)
        return tier_prefetch(source, move, lambda: .1)
    controller.start('request', prepare, deadline=10)
    cost = planning.PrefillCost(8, 0, 1)
    assert controller.decide('request', (1, 2, 3, 4), cost, **identity).action == 'wait'
    for _ in range(20):
        controller.pump(max_bytes=2, max_submissions=1)
    decision = controller.decide('request', (1, 2, 3, 4), cost, **identity)
    assert decision.action == 'reuse' and decision.lease.data == b'state'
    # Engine codecs validate/restore this complete state into private request state.
    decision.lease.close()
    assert not controller.retained
    print('Complete prefetch reused through an owned generation lease')


if __name__ == '__main__':
    main()
