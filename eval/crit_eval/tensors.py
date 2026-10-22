"""What each stored tensor is to a text engine, decided in one place for the ceiling and the context limit.

    tensor_role(name, file_name) -> Role(kind, draft)

kind is one of
- other_modality: vision towers and their projectors, which a text engine does not load;
- embedding: token and hashed n-gram embedding tables, of which a pass gathers rows per token instead of reading
  the table;
- expert: mixture-of-experts expert weights, read sparsely;
- head: the output head;
- layer: everything else, read on every pass.
draft is true for draft (MTP) layers, in the checkpoint or in their own file beside it, whatever their kind.
GGUF files name tensors their own way (token_embd, blk.N.ffn_up_exps, output) and mark draft layers only by
their block number, so tensor_role takes the draft blocks the file's metadata declares (crit_eval.ceiling).

The rules are data (the tables below), checked in this order. A checkpoint that names its parts differently gets a
row here, never a branch in the callers.
"""
from typing import NamedTuple
import re

# Substrings of a tensor name that belong to another modality, and module prefixes that do (InternVL-style
# projectors of multimodal wrappers sit at the root as mlp1.*).
OTHER_MODALITY = ("visual", "vision")
OTHER_MODALITY_PREFIXES = ("mlp1.",)
EMBEDDING = re.compile(r"embed_tokens|(^|\.)embeddings\.weight$|(^|_)token_embd\.|(^|\.)ngram_embedding\.")
EXPERT = re.compile(r"\.experts\.|_exps\.")
HEAD = re.compile(r"lm_head|^output\.weight$")
DRAFT = "mtp"
GGUF_BLOCK = re.compile(r"blk\.(\d+)\.")


class Role(NamedTuple):
    kind: str
    draft: bool


def tensor_role(name, file_name="", draft_blocks=()):
    """The role of tensor `name` stored in file `file_name` (see the module docstring); draft_blocks: the GGUF
    block numbers that are draft layers."""
    if any(part in name for part in OTHER_MODALITY) or name.startswith(OTHER_MODALITY_PREFIXES):
        kind = "other_modality"
    elif EMBEDDING.search(name):
        kind = "embedding"
    elif EXPERT.search(name):
        kind = "expert"
    elif HEAD.search(name):
        kind = "head"
    else:
        kind = "layer"
    block = GGUF_BLOCK.match(name) if draft_blocks else None
    return Role(kind, DRAFT in name or DRAFT in file_name or bool(block and int(block[1]) in draft_blocks))
