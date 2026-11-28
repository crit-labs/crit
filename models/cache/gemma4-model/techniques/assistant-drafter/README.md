# assistant-drafter

A model technique of the [gemma4-model](../../README.md) package ([a family's own pieces](../../../README.md#a-familys-own-pieces)):
Gemma 4's companion MTP drafter, a `gemma4_assistant` checkpoint published beside the target in a folder of its own
(for example the assistant of a 31B target: four layers that hold queries only), as the model's device draft head.

| | |
| --- | --- |
| seed | the pending token `t` (position `P`, not yet in the target's KV) and `h`, the target's final-normed row of the column that predicted `t` |
| step | `x = pre_projection([embed(t) * sqrt(hidden), h])`; each layer a decoder layer between two norms scaled by its `layer_scalar`, its attention query-only: `q = rope(q_norm(q_proj(norm(x))))` at `P` over the target's keys and values of its last layer of the same type (sliding or full) before `P`; `out = norm(x)`, logits through its tied embedding, `h' = post_projection(out)` |
| chain | the next draft repeats the step at the same position with its draft token and `h'` |

The layout names it as its draft block with the companion's model_type, `"draft": {"block": "gemma4-assistant",
"companion": "gemma4_assistant"}`; the models core finds the folder (option `draft_folder`, a run's `--drafter`, else
the one beside the checkpoint that `fits()`: as wide a backbone, no ordered embeddings, a vocabulary that holds the
model's, the layer types it reads) and hands it over as `model.draft_companion`
([companion.py](../../../../core/model-template/code/companion.py)). The drafter only proposes: the target verifies
every draft, so its precision (its matrices at the draft encoding, `mtp_encode`) changes which drafts are kept,
never what a request emits. Where it reads a shared-KV layer of the target it calls that layer's `read()`.

`step()` is its arithmetic over the operations it is handed: the device head's (the projections slot's held
matrices, the norms slot's rotation, the attention slot over the target's pools) and the host reference's in the
package's tests, so both run the same code.

| File | What it holds |
| --- | --- |
| [entry.json](entry.json) | What it provides and reads |
| [code/assistant_drafter.py](code/assistant_drafter.py) | The drafter technique (`DRAFTERS`), its draft head, `step()` |

Tested in [tests/test_assistant.py](../../tests/test_assistant.py) against transformers' official
`Gemma4AssistantForCausalLM` on tiny random checkpoints: `step()` over FP32 host products gives the official
drafter's logits, next row and greedy chain from the same target keys, values and row, with the drafter as published
and under another spelling its layout's `names` map back; the engine finds the folder beside the checkpoint and plans
it as its draft head; on a GPU drafting from it replies the plain greedy tokens.

Promoted from the 2026-11-25 Gemma 4 build's engine (the 2026-11-24 build wrote the same), reworked to the library's
standard: the folder comes from the models core (it was a path in the engine's layout), its byte counts from the
formats' sizes (they were ratios), its arithmetic one function for the device head and the reference.
