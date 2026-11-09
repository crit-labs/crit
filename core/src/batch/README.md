# Batched steps

[batch.cpp](batch.cpp) implements [batch.h](../../include/crit/batch.h): the host
bookkeeping a batched engine needs to run many requests' rows as one step on a
`ConcurrentModel` with [mixed batches](../runtime/README.md#mixed-token-work-batches).
It lays out a step's rows, names the rows that produce logits and reserves pages per
request. The engine's arithmetic, state bytes and logits stay its own; the runtime does not call it.

## Step layout

`step_item(work, prompt_tokens, frontier, decode_rows)` turns a `WorkItem` into a
`StepItem`: a decode item (prefill complete) stages its pending token, or with
`decode_rows` above one a verify of `decode_rows - 1` drafts, cut to the item's
token work on mixed batches and never below one row, so a small grant falls back to
an ordinary step. A prefill item computes the prompt left after `frontier`, at most
its prefill grant and token work. Every decode or verify row produces logits; a
prompt chunk produces one logits row when it ends its prompt.

`StepLayout::arrange(items)` places decode items first, then prefill items, each
phase in input order:

| Accessor | Meaning |
| --- | --- |
| `placements()` | One `StepPlacement` per item in row order: rows `[first, first + rows)` and logits rows `[first_logit, first_logit + logits)`; `item` is its input index |
| `placement(i)` | The placement of input item `i` |
| `decode_rows()`, `decode_items()` | Rows `[0, decode_rows())` are the decode block |
| `logit_rows()` | The step row of every logits row, ascending: the gather index of one logits pass |

```cpp
std::vector<crit::core::StepItem> items;
for(const auto& work:group) items.push_back(crit::core::step_item(work, prompt_size(work), committed(work)));
layout.arrange(items);   // decode block, then prompt chunks; layout.logit_rows() feeds the logits pass
```

The engine's work then runs once over all `rows()` rows, so a step reads the model's
weights once however many requests it holds; the decode block and the prompt chunks
are two contiguous ranges it can compute by phase.

## Batch invariance

A request's tokens must not depend on the requests sharing its steps or on how its
prompt was chunked; batched output then equals serial output token for token, which
Eval's `repeatable` and `request_history` gates and the greedy exactness check rely
on. The rules for the engine's arithmetic on the step path:

- Work local to a row reads only that row, in a fixed order.
- A reduction a row takes part in is split by the operation's shape and the device,
  never by the row count; short tiles are filled with zero rows, not regrouped.
- Work that reads a sequence's earlier rows gives a row a result that depends only on
  its position and the rows it sees, in an order fixed by that position, so neither
  its step-mates nor prompt chunk boundaries change it.
- Greedy argmax picks the lowest id among equal maxima, as Core's `Sampler` does.
- A verify of K drafts computes each of its K + 1 columns as a one-token step
  computes its column (column invariance, [device-speculative-round](../../../inference/device-speculative-round/README.md#column-invariant-arithmetic)).

Column invariance serves only the greedy exactness check, which is a gate when a run
requires exactness. In quality mode `speed`, Eval passes an output difference
that parts at a near-tie ([forks](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#forks-in-quality-mode-speed)), so there every rule above is optional: a
kernel that breaks one (a split chosen by the row count, regrouped tiles, chunk
boundaries in its order) is fine when it measures faster.

## Page reservations

`PageReservations({pages, page_tokens, max_tokens, verify_rows})` is page-reserving
admission for a StateCache pool. `need(view)` covers every row a request can stage:
its prompt, its output and a verify's `verify_rows - 1` columns past the last output,
at most `max_tokens`, with no credit for prefix pages it may share. Call
`admission(view)` from `ConcurrentModel::admission`, `reserve(view)` in `open` and
`release(pages)` at retirement:

| `admission()` | When |
| --- | --- |
| `infeasible` | Empty prompt, or prompt plus output beyond `max_tokens` |
| `deferred` | The reserved pages leave fewer than the need |
| `ready` | Otherwise; `reserve()` refuses anything else with `logic_error` |

An admitted request never runs out of pages: a sequence's block table holds at most
the pages its rows need, shared and copied-on-write pages included, as long as the
engine retires released pages before beginning the next round (a synchronized step)
or keeps a pool margin for pages still retiring. Credit for shared prefix pages
would admit more requests but needs eviction or preemption when sharing ends
([incremental memory and recovery](../runtime/README.md#incremental-memory-and-recovery)).
`row_pages(rows, page_tokens)` is the page count of `rows` rows.

### Projected peak pages

Reserving every request's maximum admits too few when replies are shorter than their
caps: a request that finishes early frees its pages before a longer one peaks.
`peak_pages(demands, page_tokens, verify_rows)` bounds what the requests hold at once,
each a `PageDemand{held_rows, remaining_rows}` (its prompt, computed or not, and output
so far; the output rows it may still stage). With the requests sorted by remaining
rows, longest first, the k still running at step `remaining_k` hold at most
`(held_1 + ... + held_k + k * (remaining_k + verify_rows - 1 + page_tokens - 1)) /
page_tokens` pages; the bound is the largest of these, never above the sum of the
maxima. An engine admits a request when the bound over the active requests and the
candidate fits the pool.

It is a projection, not a reservation: it holds while every request stages at most one
row per step from now on. A request that advances faster (speculative rounds) or starts
late (a prompt still prefilling) can hold its pages past the step the bound assumes, so
admit by it only with a fallback when pages run out: Core's
[memory preemption](../runtime/README.md#incremental-memory-and-recovery) or a pool
margin. [`batch_peak_pages`](../../../tests/core/batch_tests.cpp) checks the bound
against a step-by-step simulation of 300 random lockstep batches, and shows its gain
and its limit: 16 requests holding 500 rows with 100 to 1,900 rows left need 1,504
pages by their maxima and at most 835 by the bound (the simulation's peak is 830), while
a short request that starts 480 steps late takes 36 pages where the bound says 33.

## Speculative rows

`SpeculativeRow(cache, sequence, pending)` runs one request's speculative rounds over
its StateCache sequence, so a batched engine verifies every request's drafts in the
same step and each request accepts and rolls back on its own:

| Call | Effect |
| --- | --- |
| `begin(drafts)` | Stages the pending token and the drafts as one round of K + 1 rows (cut to `max_round_rows` and `max_tokens`); a decode `StepItem` of K + 1 rows, all producing logits (`step_item(work, ..., K + 1)` cuts K to the token work) |
| `accept(predictions, remaining, stops)` | The target's prediction after each staged row decides the round by Core's greedy rule (`accept_greedy`): the matching drafts, then a correction or bonus, bounded by the output allowance and ending after an accepted stop |
| `commit(accepted)` | Keeps the first `accepted` rows (the pending token and `accepted - 1` emitted tokens) and makes the last kept token pending; the other rows roll back, their pages returning at the cache's next epoch. Zero discards the round. When the cache's commit throws and keeps the round pending (a throwing snapshot copy), the row keeps it staged |

Commit what Core accepted: it may keep fewer tokens than the round emitted, at a stop
the request's round did not know or the output limit. Draft widths can differ per
request in one step; rows of other requests never change a request's acceptance,
so with batch-invariant verify arithmetic the output equals plain greedy decoding.
Sampled verification (rejection sampling) keeps its draws in the engine: feed its
predictions to `accept` as the device round does
([device-speculative-round](../../../inference/device-speculative-round/README.md#sampled-requests)).

## Checks

[batch_tests.cpp](../../../tests/core/batch_tests.cpp) checks the layout (phase
order, placements, logits rows, verify columns, rejected items, a failed arrange
keeping the previous layout), WorkItem rows (grants, token work, prompt ends) and
reservations (deferral, release, infeasible requests, verify columns, the
`max_tokens` cap). A toy paged engine through a Core `Engine` stores its rows' tokens in
StateCache pages and computes each next token from every row a sequence sees through its
block table: nine requests decode the same tokens one at a time and six at once,
with steps mixing prompt chunks and decode rows, and a 20-page pool admits only two
of them at a time. Speculative rows keep the accepted rows' tokens and return the
rejected rows' pages, follow a Core trim, a spent allowance, a stop and the
`max_tokens` cut, and keep a round staged that a throwing snapshot copy left pending
in the cache, so discarding or committing it again ends it. A toy speculative engine through a Core `Engine` drafts one to four
tokens per request with deliberate mistakes and verifies them over its pages: one
request per step and six at once, every request decodes exactly the plain greedy
tokens, with rounds that accept several tokens, rounds that reject drafts, steps that
verify several requests' drafts and stops Core trims inside a round.
