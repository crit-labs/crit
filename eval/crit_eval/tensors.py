"""What each stored tensor is to a text engine, decided in one place for the ceiling and the context limit.

    tensor_role(name, file_name) -> Role(kind, draft)

kind is one of
- other_modality: vision towers and their projectors, which a text engine does not load;
- embedding: token embedding tables, of which a pass gathers one row per token instead of reading the table;
- expert: mixture-of-experts expert weights, read sparsely;
- head: the output head;
- layer: everything else, read on every pass.
draft is true for draft (MTP) layers, in the checkpoint or in their own file beside it, whatever their kind.

The rules are data (the tables below), checked in this order. A checkpoint that names its parts differently gets a
row here, never a branch in the callers.
"""
from typing import NamedTuple
import re

# Substrings of a tensor name that belong to another modality, and module prefixes that do (InternVL-style
# projectors of multimodal wrappers sit at the root as mlp1.*).
OTHER_MODALITY = ("visual", "vision")
OTHER_MODALITY_PREFIXES = ("mlp1.",)
EMBEDDING = re.compile(r"embed_tokens|(^|\.)embeddings\.weight$")
EXPERT = re.compile(r"\.experts\.")
HEAD = "lm_head"
DRAFT = "mtp"


class Role(NamedTuple):
    kind: str
    draft: bool


def tensor_role(name, file_name=""):
    """The role of tensor `name` stored in file `file_name` (see the module docstring)."""
    if any(part in name for part in OTHER_MODALITY) or name.startswith(OTHER_MODALITY_PREFIXES):
        kind = "other_modality"
    elif EMBEDDING.search(name):
        kind = "embedding"
    elif EXPERT.search(name):
        kind = "expert"
    elif HEAD in name:
        kind = "head"
    else:
        kind = "layer"
    return Role(kind, DRAFT in name or DRAFT in file_name)
