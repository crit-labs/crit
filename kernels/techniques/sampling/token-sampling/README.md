Core's sampling law on the device: one draw per row of logits by Core's rules (temperature, top-k, top-p, min-p) and
Core's counter-based Philox draws, rejection-sampling verify predictions for speculative rounds, and log-probabilities.
crit's token-sampling plugin (`inference/token-sampling`) loads it for its captured decode rounds, built for the engine's
device with the engine's tuning or the default knobs; token-constraints loads it through the same code. Where it does not
build, the plugin's torch law serves (capture-safe, slower).

- **Core's bits**: a draw is the token Core's host sampler draws from the same row, rules and stream (the C++ suites
  `tests/inference/token_law_tests.cpp` and `token_sampling_tests.cpp` check it against Core's sampler).
- **Two paths**: an exact top-64 fast path (greedy rows, 1 <= top_k <= 64, and open rows whose kept set closes within
  the candidates), and the full-vocabulary law by prefix searches over the keys' digits for the rest.
- **Spread over blocks**: the full law runs a row's 1,024 threads as virtual threads over PARTS blocks of a
  cooperative launch, with the same per-thread sums and warp trees, so the same bits. It falls back to one block a row
  where the device runs no cooperative launch or the grid does not fit.
- **One launch set, no host synchronization**: a call at a fixed shape is captured in a CUDA graph.

**Built from the kernel core** (`kernels/core`): the radix keys and the digit search are `crit_kcore/radix.cuh`'s.
It includes Core's public headers (`"core_include": true`) for the Request and Philox definitions.

## Interface

```python
from crit_kernels.build import load      # a plugin: crit_plugins.kernels.load_kernel("token-sampling", folder=...)
ts = load("token-sampling")
requests = torch.cat([ts.pack_request(temperature, top_p, min_p, top_k, seed, stream, position) for ...]).cuda()
scratch = torch.empty(ts.scratch_bytes(vocab, rows), dtype=torch.uint8, device="cuda")
ts.sample(logits, vocab, requests, count, columns, position_base, purpose, out, sparse, dense, scratch, lp_n,
          chosen, top_ids, top_values, tree, tree_tokens)    # None for an absent buffer
ts.verify(logits, vocab, requests, count, columns, drafts, sparse, dense, predictions, scratch, lp_n, chosen,
          top_ids, top_values)
```

## Structures and knobs

| Structure | What | Exactness | Knobs |
| --- | --- | --- | --- |
| `fast-law` | stage blocks of 256 threads x 16 logits a row, each block's exact top 65 keys and FP64 weight partials, then a law block a row | row-exact | none: the block sizes set the FP64 sums' grouping |
| `full-law` | the full-vocabulary law in one block of 1,024 threads a row | row-exact | none |
| `spread-law` | the full law over PARTS blocks a row (cooperative launch) | exact, the bits of `full-law` | PARTS 8, 16, 32 (default 32) |

`CRIT_SAMPLING_ONE_BLOCK=1` takes `full-law`. The bench (`bench.py:Greedy`) times greedy rows against `torch.argmax`.

## Tests

`tests/test_token_sampling.py`: greedy rows are the argmax (ties to the lower id); the full law's draws are the same
bits under every PARTS and in one block; a captured draw replays the eager bits; a verify round of greedy drafts
predicts the target's draws; requests pack a 64-bit seed.
