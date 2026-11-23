"""The blocks a composed model is made of: each the arithmetic of one kind of layer part over the other slots' plugins.

A block is made once per layer, `Block(model, layer, prefix, spec)`: `model` is the ComposedModel (its matrices,
norms, pools and the slots' plugins), `prefix` the layer's tensor prefix for the block ("model.layers.3.self_attn."),
`spec` the layout's entry for it. A mixer block (attention, gated-delta) has `pool`, the key of the pool its layer
state lives in ("kv", ("kv", window), "states"), and `forward(h, ctx)`; an FFN block (swiglu, moe) has
`forward(h, ctx)`. ctx is the step's Context: the batch, positions, row groups and each pool's slots. Every matmul goes
through ctx.mm (RowGroups: exact rows), every norm, rotary position and gated product through the norms slot, so a row's
bits never depend on the rows beside it.

A block reads what varies between checkpoints from the checkpoint and its config (biases, query/key norms, the
recurrence's projection layout, shared experts, the router's rule) and what a family fixes from the layout (the norm
offset, an attention output gate). An engine adds a kind of block by passing its class to compose() under a name its
layout uses (README: adding a block).

A block that keeps state of its own beside its layer's pool declares it before anything is made, so the model plans
its bytes with every other tensor: the class method `pools(model, spec, layers)` gives [{key, kind, layers, ...}] for
the layers that run it, kind "keys" (a pool of keys only, a row a position: heads, dim) or "window" (the last inputs
of a causal depthwise convolution: channels, kernel, dilation), which the model makes and follows through steps,
rounds and the prompt cache as it does its KV and state pools; `tables(model, prefix, spec)` gives {plan name: bytes}
for lookup tables it reads rows of, which may lie off the device. A `before` block (the layout's `before`) has
`forward(residual, ctx)` and returns the residual with its rows added."""


class Context:
    """One forward's step: batch (StepBatch, RoundBatch or PromptBatch), positions, row groups, the slots each pool
    reserved for the batch's tokens; ids, the tokens' ids where the forward has them (a draft layer's rows have
    none), and windows, each token's ids with the ones before it (model.py, _windows), for the blocks that read them;
    shared, what the blocks of one forward compute once for all of them (a step's block lists are every indexed
    layer's)."""

    def __init__(self, model, batch, groups, slots, ids=None, windows=None):
        self.model, self.batch, self.groups, self.slots = model, batch, groups, slots
        self.ids, self.windows, self.shared = ids, windows, {}
        self.residual = None   # the stream a layer's FFN output is added to (a residual that gives it: sandwich)
        self.positions = batch.positions_t
        self.prompt = bool(getattr(batch, "prompt", False)) or (not batch.round and any(p for _, _, p in groups))

    def mm(self, x, matrix, out_dtype=None):
        return self.model.rows.matmul(x, matrix, self.groups, out_dtype)

    def mm_each(self, x, matrix, out_dtype=None):
        """The matmul of x [tokens][n][in], n rows a token (its residual streams): [tokens][n][out], a prompt chunk's
        rows in one prompt call and one-token rows in calls of at most the invariant rows, as mm's."""
        tokens, n = x.shape[0], x.shape[1]
        step, groups = self.model.rows.decode_rows, []
        for first, end, prompt in self.groups:
            if prompt:
                groups.append((first * n, end * n, True))
            else:
                groups += [(at, min(at + step, end * n), False) for at in range(first * n, end * n, step)]
        return self.model.rows.matmul(x.reshape(tokens * n, -1), matrix, groups, out_dtype).view(tokens, n, -1)


def row_sum(x):
    """The sum of each last-dimension row of x in one fixed order whatever the row count: adjacent pairs added, then
    the pairs' sums, an odd value joining the next level. Elementwise adds only, where torch's own sum over a row
    takes another order in a call of other rows (the norms slot's reductions are the norms and the stream sums; this
    is a plain dot product's)."""
    import torch
    while x.shape[-1] > 1:
        n = x.shape[-1]
        pairs = x[..., 0:n - n % 2:2] + x[..., 1:n:2]
        x = torch.cat([pairs, x[..., n - 1:]], dim=-1) if n % 2 else pairs
    return x[..., 0]


def _tensor(model, name, required=True):
    if name in model.ck:
        return model.ck.get(name, model.device)
    if required:
        raise KeyError(f"model-template: the checkpoint has no {name}")
    return None


def sources():
    """{key: folder} of the sources the model plugin mounts (its manifest's `sources`, read from the plugin package
    this code is a source of, else from the plugin.json in that package's folder)."""
    import json
    from pathlib import Path
    import sys
    package = sys.modules[__package__.rsplit(".", 1)[0]]
    manifest = getattr(package, "__crit_manifest__", None)
    base = Path(package.__path__[0]) if getattr(package, "__path__", None) else None
    if manifest is None:
        manifest = json.loads((base / "plugin.json").read_text(encoding="utf-8"))
    return {key: (Path(source["folder"]) if isinstance(source, dict) else (base / source).resolve())
            for key, source in manifest.get("sources", {}).items()}


def core_folder():
    """The models core's folder (its plugin.json beside this code), or None where the code runs outside a package."""
    import sys
    from pathlib import Path
    package = sys.modules.get(__package__.rsplit(".", 1)[0])
    path = getattr(package, "__path__", None)
    return Path(path[0]) if path else None


def library():
    """(blocks, residuals, drafters): {name: class} of every model technique the plugin's sources mount. A technique's
    code is the source under a key, KEY.py in its folder, which gives BLOCKS, RESIDUALS or DRAFTERS (the draft blocks
    a layout's draft names beside the core's mtp; models/techniques/*/*: code/KEY.py and entry.json); the plugin is model-template, or an engine's own model plugin over compose() whose manifest
    mounts the techniques it composes the same way (README: an engine's own model plugin)."""
    import importlib
    parent = __package__.rsplit(".", 1)[0]
    blocks, residuals, drafters = {}, {}, {}
    for key, folder in sources().items():
        if key == __package__.rsplit(".", 1)[1] or not (folder / f"{key}.py").is_file():
            continue
        module = importlib.import_module(f"{parent}.{key}.{key}")
        for found, kind in ((blocks, "BLOCKS"), (residuals, "RESIDUALS"), (drafters, "DRAFTERS")):
            for name, cls in getattr(module, kind, {}).items():
                if name in found and found[name] is not cls:
                    raise ValueError(f"model-template: two techniques give the {kind.lower()[:-1]} {name}")
                found[name] = cls
    return blocks, residuals, drafters
