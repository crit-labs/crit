# indexed-attention

Block-sparse attention: an indexer scores the complete blocks of positions before a token and the token reads its best ones within a budget, and its own block.

## How it works

Block-sparse softmax attention: an indexer picks the KV blocks each token reads.

The sequence's positions lie in blocks of `block` positions from position 0. A token at position p reads the
complete blocks an indexer picks for it, up to budget / block of them, and the block it lies in (its tail), so a
token with no more complete blocks than that reads every position up to its own: plain causal attention below
the budget. The indexer projects each token's row to index queries (several heads) and one raw index key; a
complete block's key is the mean of its positions' raw keys, normed and rotated at the block's first position; a
token scores a block by the sum over its index heads of relu(query . key) / sqrt(dim), queries normed and rotated
at the token's position, and keeps its highest-scoring blocks (crit's canonical choice: ties to the lower block).

Keys: the layers keep one pool of keys only, a row a position (`pools`): a position's raw key, and at a block's
last position, once the block is complete, the block's key in its place (the raw key of a block's last position
is read by nothing else). So the pool follows a sequence exactly as its KV does, through steps, rounds, forks and
truncations, and a block's key is rewritten whenever its last position is computed again.

Selection runs on crit's kernel sparse-indexer where it loads and takes the indexer's shape (the keys read in
place through their slots), else in torch a token at a time, so a token's blocks never depend on the tokens
beside it. A prompt step's tokens are scored in groups whose scores fit `score_bytes` (a token's scores are one
FP32 a block, and the choice works on a wider copy of them: a 4,096-token step over 50,000 blocks would hold
gigabytes at once), which changes no choice: each token's is its own. Where no sequence the engine serves can pass the budget (its KV pool holds no more blocks) the block is
its base: no index keys, no block lists. Otherwise every token's attention reads a block list, below the budget
too, so a row's arithmetic is one whatever shares its step. Config keys (the spec's `config` renames them):
indexer_n_heads, indexer_kv_heads (1), indexer_head_dim, indexer_budget, indexer_compress_ratio; the spec's
`indexer` is the indexer's tensor prefix under the attention's (indexer.).

## Use it

A layout names it as a mixer, an FFN or a `before` block (`indexed-attention`); the model plugin mounts [code/indexed_attention.py](code/indexed_attention.py) as its source `indexed_attention`, beside the sources it imports (attention). What it reads is in [entry.json](entry.json) (`reads`): the tensors after the block's prefix and the config keys, as its code names them.
