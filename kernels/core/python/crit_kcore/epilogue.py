"""crit's kernel core for Triton: epilogue functions (kernels/core/README.md, "epilogue"): the activations kernels apply
to a finished FP32 value before they store it, one definition each."""
import triton
import triton.language as tl

# act(): 0 silu, 1 gelu (erf), 2 gelu_tanh, 3 sigmoid, 4 none (the value itself).
SILU, GELU, GELU_TANH, SIGMOID, IDENTITY = 0, 1, 2, 3, 4


@triton.jit
def sigmoid(x):
    """1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def act(g, ACT: tl.constexpr):
    """The activation ACT of g (codes above); gelu_tanh as g sigmoid(2u), finite for every u."""
    if ACT == 0:
        out = g * sigmoid(g)
    elif ACT == 1:
        out = 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))
    elif ACT == 2:
        inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        # 0.5 g (1 + tanh(u)) = g sigmoid(2u): finite for every u
        out = g * sigmoid(2.0 * inner)
    elif ACT == 3:
        out = sigmoid(g)
    else:
        out = g
    return out
