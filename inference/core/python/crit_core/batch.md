# Batched steps

[batch.py](batch.py) is the host bookkeeping a model plugin's step model needs to run many requests' rows as one step
on Core's [step engine](../../src/runtime/README.md#step-engines) with
[mixed batches](../../src/runtime/README.md#mixed-token-work-batches). It lays out a step's rows, names the rows that
produce logits and reserves pages per request. The model's arithmetic, state bytes and logits stay its own; the
runtime does not call it.

## Step layout

`step_item(row, prompt_tokens, frontier, decode_rows=1)` turns a scheduled row (a `crit_core.native.StepRow`) into a
`StepItem`: a decode item (prefill complete) stages its pending token, or with `decode_rows` above one a verify of
`decode_rows - 1` drafts, cut to the row's token work on mixed batches and never below one row, so a small grant
falls back to an ordinary step. A prefill item computes the prompt left after `frontier`, at most its prefill grant
and token work, and stops early at `stop` when the model wants a chunk to end there (a prompt cache's mark). Every
decode or verify row produces logits; a prompt chunk produces one logits row when it ends its prompt.

`StepLayout(items)` places decode items first, then prefill items, each phase in input order:

| Attribute | Meaning |
| --- | --- |
| `placements` | One `Placement` per item in row order: rows `[first, first + rows)` and logits rows `[first_logit, first_logit + logits)`; `item` is its input index |
| `placement(i)` | The placement of input item `i` |
| `decode_rows`, `decode_items` | Rows `[0, decode_rows)` are the decode block |
| `logit_rows` | The step row of every logits row, ascending: the gather index of one logits pass |
| `read_rows` | The step row of every row whose last-layer output anything reads, ascending (each placement's last `read` rows) |

```python
items = [step_item(row, len(session.prompt), session.frontier) for row, session in rows]
layout = StepLayout(items)   # decode block, then prompt chunks; layout.logit_rows feeds the logits pass
```

Only the logits rows read the last layer's output, unless an item sets `readers` (a drafter's or MTP head's taps, a
hidden-state readout read every row). So the last layer needs its query, attention, output projection and MLP only on
`read_rows`: every other row stops at the layer's state writes (its keys and values, or its recurrent state), and a
prompt chunk that does not end its prompt stops there altogether. That saves about one layer in the model's layers of
a cold prompt's work. A model whose kernels run whole tiles takes the tile that holds the read rows, and one whose
last layer fuses the query with the key and value projections keeps them.

The model's work then runs once over all `rows` rows, so a step reads the model's weights once however many requests
it holds; the decode block and the prompt chunks are two contiguous ranges it can compute by phase.

## Batch invariance

A request's tokens must not depend on the requests sharing its steps or on how its prompt was chunked; batched output
then equals serial output token for token, which Eval's `repeatable` and `request_history` gates and the greedy
exactness check rely on. The rules for the model's arithmetic on the step path:

- Work local to a row reads only that row, in a fixed order.
- A reduction a row takes part in is split by the operation's shape and the device, never by the row count; short
  tiles are filled with zero rows, not regrouped.
- Work that reads a sequence's earlier rows gives a row a result that depends only on its position and the rows it
  sees, in an order fixed by that position, so neither its step-mates nor prompt chunk boundaries change it.
- Greedy argmax picks the lowest id among equal maxima.
- A verify of K drafts computes each of its K + 1 columns as a one-token step computes its column (column invariance,
  [device-speculative-round](../../../techniques/speculation/device-speculative-round/README.md#column-invariant-arithmetic)).

Column invariance serves only the greedy exactness check, which is a gate when a run requires exactness. In quality
mode `speed`, Eval passes an output difference that parts at a near-tie
([forks](https://github.com/crit-labs/crit-eval/blob/main/docs/MEASUREMENTS.md#forks-in-quality-mode-speed)), so there
every rule above is optional: a kernel that breaks one (a split chosen by the row count, regrouped tiles, chunk
boundaries in its order) is fine when it measures faster.

## Page reservations

`PageReservations(pages, page_tokens, max_tokens, verify_rows=1)` is page-reserving admission for a paged state
pool. `need(prompt_tokens, output_tokens)` covers every row a request can stage: its prompt, its output and a verify's
`verify_rows - 1` columns past the last output, at most `max_tokens`, with no credit for prefix pages it may share.
Call `admission()` from the step model's admission, `reserve()` when it opens the request and `release(pages)` at
retirement:

| `admission()` | When |
| --- | --- |
| `INFEASIBLE` | Empty prompt, or prompt plus output beyond `max_tokens` |
| `DEFERRED` | The reserved pages leave fewer than the need |
| `READY` | Otherwise; `reserve()` refuses anything else |

An admitted request never runs out of pages: a sequence's block table holds at most the pages its rows need, shared
and copied-on-write pages included, as long as the model retires released pages before beginning the next round or
keeps a pool margin for pages still retiring. Credit for shared prefix pages would admit more requests but needs
eviction or preemption when sharing ends. `row_pages(rows, page_tokens)` is the page count of `rows` rows.

## Checks

[test_core_batch.py](../../../../tests/test_core_batch.py) checks the layout (phase order, placements, logits and read
rows, verify columns, rejected items), scheduled rows (grants, token work, prompt ends) and reservations
(deferral, release, infeasible requests, verify columns, the `max_tokens` cap); the
[step engine check](../../../../tests/core/step_engine_check.py) runs a toy step model batched and serial on Core's
runtime and compares their tokens.
