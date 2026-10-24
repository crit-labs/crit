"""Faster forms of modeling code the reference would otherwise run as a Python loop per token.

transformers' qwen4_exp sparse-attention indexer (Qwen4ExpTextQSAIndexer.forward) selects, for every query token, the
top blocks of pooled keys among the complete blocks it sees, in a loop over the queries: on a 32K-token text that is
about 33,000 iterations of a host-synchronising nonzero, a pooling of every visible block, RoPE, a matmul and a top-k
per full-attention layer, which made the reference of Qwen3.8 Flash-Next take hours. `qsa_indexer_forward` computes
the same selection at once when every query sees a prefix of the keys (a causal mask without padding, as the
reference's chunks are): a block's pooled, normed and rotated key does not depend on the query, so every complete
block is pooled once; each query's scores are the same sums of ReLU'd head scores, masked to the blocks it sees, and
the same top-k picks its blocks. The selection is the same up to how floating-point ties at the top-k edge are broken,
which the loop leaves to the device as well. Any other mask runs transformers' own loop.

`exact_forms(model)` binds these forms to a model's modules whose forward is the code they were checked against
(by a hash of its source, so a changed transformers keeps its own) and returns their names.
"""
import hashlib
import inspect
import math
import sys
import types

# sha256 of the source of each forward a form replaces, as the transformers release the form was checked against
# (tests/test_eval_indexers.py) has it.
CHECKED = {
    "Qwen4ExpTextQSAIndexer": "cc3b0347832752a530a86a4b69e5147daea9b58d29f756e8f60a580bf519f093",   # transformers 5.20.2
}
QUERY_ROWS = 1024   # queries scored at once: their block scores take QUERY_ROWS x blocks floats per head


def source_hash(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def qsa_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Qwen4ExpTextQSAIndexer.forward with the per-query loop done at once (module docstring)."""
    import torch
    rope = sys.modules[type(self).__module__].apply_rotary_pos_emb
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q = rope(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    kv_length = attention_mask.shape[-1]
    counts = visible[:, 0].sum(-1)   # (batch, queries): every query sees a prefix of this many keys, or the loop runs
    positions = torch.arange(kv_length, device=visible.device)
    if not torch.equal(positions < counts[..., None], visible[:, 0]):
        return type(self).forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)

    ratio = self.compress_ratio
    blocks = kv_length // ratio
    selected = torch.zeros((batch_size, seq_length, kv_length), dtype=torch.bool, device=attention_mask.device)
    complete = counts // ratio                                    # each query's complete blocks
    # Its tail: the visible tokens after its complete blocks.
    selected |= (positions >= (complete * ratio)[..., None]) & (positions < counts[..., None])
    if blocks:
        starts = torch.arange(blocks, device=hidden_states.device) * ratio
        tokens = (starts[:, None] + torch.arange(ratio, device=hidden_states.device)).flatten()
        for batch in range(batch_size):
            groups = raw_keys[batch].index_select(0, tokens).view(blocks, ratio, self.index_head_dim)
            pooled = self.k_layernorm(groups.float().mean(dim=1).to(raw_keys.dtype))
            keys = rope(pooled.unsqueeze(1), cos=full_cos[batch].index_select(0, starts),
                        sin=full_sin[batch].index_select(0, starts)).squeeze(1).float()
            for first in range(0, seq_length, QUERY_ROWS):
                rows = slice(first, min(seq_length, first + QUERY_ROWS))
                queries = q[batch, rows].float()                 # (rows, heads, dim)
                scores = torch.zeros((queries.shape[0], blocks), dtype=torch.float32, device=queries.device)
                for head in range(queries.shape[1]):
                    scores += torch.relu(queries[:, head] @ keys.T)
                scores /= math.sqrt(self.index_head_dim)
                seen = complete[batch, rows]
                scores.masked_fill_(torch.arange(blocks, device=scores.device) >= seen[:, None], float("-inf"))
                take = min(self.block_topk, blocks)
                values, picked = scores.topk(take, dim=-1)
                chosen = torch.zeros((queries.shape[0], blocks), dtype=torch.bool, device=scores.device)
                chosen.scatter_(-1, picked, values > float("-inf"))   # a query with fewer blocks takes all it sees
                selected[batch, rows, :blocks * ratio] |= chosen.repeat_interleave(ratio, dim=-1)
    selected_token_mask = selected.unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)
    return selected_token_mask


FORMS = {"Qwen4ExpTextQSAIndexer": qsa_indexer_forward}


def exact_forms(model):
    """Bind FORMS to `model`'s modules whose class forward is the source they were checked against (CHECKED).
    Returns the class names bound."""
    bound = set()
    for module in model.modules():
        name = type(module).__name__
        if name in FORMS and source_hash(type(module).forward) == CHECKED[name]:
            module.forward = types.MethodType(FORMS[name], module)
            bound.add(name)
    return sorted(bound)
