"""The sums' order of the matrices a sweep chose it for: {(weights, mma, N, K): splits of K}, mma "fp4" for the
block-scaled FP4 MMA (acts nvfp4_mma) and "bf16" for every other activation on mma.m16n8k16. A matrix's splits fix
the order of each of its outputs' sums, so they are chosen once, by shape, and shipped: every GPU and every tuning
reads the same table, and a row gets the same bits on any device. A matrix the table does not name takes
order_splits' rule (ORDER_CTAS).

Written by benchmarks/order_sweep.py, never by hand: for each shape it times every allowed split count (each split at
least two steps of K, at most MAX_SPLITS) on decode calls of 1, 2, 4, 8 and 16 rows and prompt calls of 512 and 2,048
rows and keeps the one whose worst ratio to the fastest count over those calls is least (one count a matrix for every
row count, since the bits must not follow the row count).

Sweep: NVIDIA GeForce RTX 5090, 2026-11-27, benchmarks/order_sweep.py."""

ORDERS = {
    ('fp8', 'bf16', 5120, 6144): 6,
    ('fp8', 'bf16', 16384, 5120): 2,
    ('nvfp4', 'bf16', 5120, 17408): 2,
    ('nvfp4', 'bf16', 5376, 21504): 4,
    ('nvfp4', 'bf16', 34816, 5120): 1,
    ('nvfp4', 'bf16', 43008, 5376): 1,
    ('nvfp4', 'fp4', 5120, 17408): 8,
    ('nvfp4', 'fp4', 5376, 21504): 10,
    ('nvfp4', 'fp4', 34816, 5120): 3,
    ('nvfp4', 'fp4', 43008, 5376): 1,
}
