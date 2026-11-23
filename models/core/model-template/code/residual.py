"""How a composed model's blocks join its residual: the layout's `residual` names one of these.

    pre-norm   one residual stream: an RMSNorm before each block, the block's output added to the stream (the norm
               after a block's output and its add are one call of the norms slot), a final norm before the head
    sandwich   one residual stream with a norm on either side of each block: the stream's norm goes into the block,
               the block's output is normed again and then added, and the stream is scaled once a layer where the
               checkpoint stores a scalar for it; a final norm before the head
    streams    several residual streams (hyper-connections): each block reads one row mixed from the streams' norms by
               weights the streams compute themselves, and its output is added to every stream by weights of its
               own; a last mix, without a block, gives the row the head reads

A residual makes the per-layer parameters a decoder layer keeps beside its mixer and FFN (`layer`), runs the layers
over a step's rows (`forward`: the head's rows and the draft feature, what a draft layer reads of the target), and
runs the checkpoint's draft layer (`draft`, `draft_input`, `draft_keys`, `draft_forward`). Every reduction is the
norms slot's and every matmul goes through the step's row groups, so a row's bits never depend on the rows beside it.

The layers run one at a time over a state the residual defines: `begin(layers, x, ctx)` gives the state of the
embedding rows before the first layer, `advance(layers, i, state, ctx)` the state after layer i, `end(layers, state,
ctx)` the head's rows and the draft feature, and `forward` is the three in turn. The state is a tuple of tensors of
one row a token (None for what the first layer has not made yet), which is all a layer leaves the next: a prompt span
run layer by layer (passes.py) keeps each chunk's state between its layers and gets the bits of the chunk run through
every layer at once. `carry(itemsize)` is the state's bytes a token, for the memory such a span plans.

The residuals are model techniques (models/techniques/residuals), which blocks.library() finds among the plugin's
sources; a layout without a `residual` runs pre-norm.

A drafter technique that reads the target's rows on entry to some layer applications (its `taps`) gets them from any
residual through the functions below, which the core runs instead of the residual's own where the model has taps: the
state carries the tapped rows after the residual's own (one more tensor, so a layer-by-layer prompt keeps them as it
keeps the rest), and the feature is the tapped rows, then the head's rows. A residual gives its running row with
`row(state)` (the row a layer reads on entry: pre-norm's stream, sandwich's and block-attention's row); one that keeps
no single row (several streams) gives no row, and a drafter that taps refuses it.
"""
DEFAULT = "pre-norm"


def tapped_row(model, state):
    row = getattr(model.residual, "row", None)
    if row is None:
        raise ValueError(f"model-template: the drafter reads the target's rows on entry to layers "
                         f"{list(model.taps)}, and the {model.residual.name} residual keeps no single row")
    return row(state)


def begin(model, layers, x, ctx):
    """The residual's state before layer 0 (and no tapped rows yet, where the model has taps)."""
    state = model.residual.begin(layers, x, ctx)
    return tuple(state) + (None,) if model.taps else state


def advance(model, layers, i, state, ctx):
    """The state after layer i: the residual's, and the entry row of a tapped layer beside the rows before it."""
    if not model.taps:
        return model.residual.advance(layers, i, state, ctx)
    inner, taps = tuple(state[:-1]), state[-1]
    if i in model.taps:
        row = tapped_row(model, inner)
        taps = row if taps is None else model.torch.cat([taps, row], dim=-1)
    return tuple(model.residual.advance(layers, i, inner, ctx)) + (taps,)


def end(model, layers, state, ctx):
    """(h, feature): the head's rows, and the draft feature (with taps: the tapped rows, then the head's rows)."""
    if not model.taps:
        return model.residual.end(layers, state, ctx)
    h, _ = model.residual.end(layers, tuple(state[:-1]), ctx)
    return h, model.torch.cat([state[-1], h], dim=-1)


def forward(model, layers, x, ctx):
    """(h, feature) of the embedding rows x through every layer."""
    if not model.taps:
        return model.residual.forward(layers, x, ctx)
    state = begin(model, layers, x, ctx)
    for i in range(len(layers)):
        state = advance(model, layers, i, state, ctx)
    return end(model, layers, state, ctx)


def carry(model, itemsize):
    """The state's bytes a token: the residual's, and the tapped rows."""
    return model.residual.carry(itemsize) + len(model.taps) * model.hidden * itemsize


def _vector(model, name):
    return model.ck.get(name, model.device).float()
