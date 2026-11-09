"""fused-norms in the norms slot: the host form is the slot's FP64 rows (code/rows.py). The device form takes, for each
call, a provider's kernel where the provider computes exactly the slot's function for that call, then the registry
kernel crit-labs/fused-norms (one launch per call), and where that kernel is not there (not published, no build for
this machine, no `kernels` package) torch-norms' arithmetic (code/torch_norms.py).

The provider is FlashInfer, where it is installed (option `flashinfer`): its rmsnorm and gemma_rmsnorm (RMSNorm with
weight, or 1 + weight, in the input's dtype) for an rms_norm or rms_norm_heads call without a residual, with a weight of
x's dtype and the output in x's dtype, and its silu_and_mul, gelu_and_mul and gelu_tanh_and_mul for a silu_mul or
gelu_mul of packed halves in BF16 or FP16 with the output in their dtype. Those keep the slot's contract (FP32 inside, one rounding, a row's bits whatever the row count,
capture), which the plugin's tests check through the plugin. Everything else is the registry kernel's: an FP32
residual beside 16-bit rows, a weight in another dtype or one per group, a chosen output dtype, rotary positions (the
provider's take queries and keys together, with frequencies from a base or a cache), gated norms, interleaved
or two-tensor products, sigmoid gates, L2 norms, stream sums and adds. FlashInfer's fused residual-add norm is not used: it normalizes the
unrounded sum, the slot the stored one.

The fused forms (norms:fused: rms_norm_rope_pair, rope_pair, stream_add_norm) are one launch of the kernel where the
separate calls take two; each output has the bits the plugin's separate calls give it, so a stream_add_norm whose norm
the provider would compute is the kernel's stream_add, then the provider's norm.

A route depends on dtypes, shapes, devices and options only, never on the row count, so a call site keeps one path
(a call on another device than the current one goes to the kernel, which launches on its tensors' device). The choice
of kernel or torch is made once, when the plugin is created, so a process never mixes the two."""
from pathlib import Path

from crit_plugins.kernels import registry_kernel

from .norms import rows, torch_norms

KERNEL = "crit-labs/fused-norms"


def _provider():
    try:
        import flashinfer
        import flashinfer.activation
        import flashinfer.norm
    except Exception:  # noqa: BLE001 - not installed, or not importable here: the kernel serves
        return None
    return flashinfer


class FusedNorms(rows.Rows):
    def __init__(self, kernel, provider):
        # A kernel build without every operation the plugin calls (one of the registry's version before an
        # operation was added) serves none of them: the fused forms keep the separate calls' bits only from one
        # source, so the plugin runs torch-norms' arithmetic throughout and the facts say why.
        self.lacking = [name for name in torch_norms.OPERATIONS if kernel is not None and not hasattr(kernel, name)]
        if self.lacking:
            kernel = None
        self.kernel, self.provider = kernel, provider
        self.source = kernel if kernel is not None else torch_norms
        for name in torch_norms.OPERATIONS:
            setattr(self, name, getattr(self.source, name))
        if provider is not None:
            import torch
            self._sixteen = (torch.bfloat16, torch.float16)
            self._current = torch.cuda.current_device
            self._rmsnorm, self._gemma = provider.norm.rmsnorm, provider.norm.gemma_rmsnorm
            self._products = {"silu": provider.activation.silu_and_mul, "none": provider.activation.gelu_and_mul,
                              "tanh": provider.activation.gelu_tanh_and_mul}
            self.rms_norm, self.rms_norm_heads = self._rms_norm, self._rms_norm_heads
            self.silu_mul, self.gelu_mul = self._silu_mul, self._gelu_mul
            self.stream_add_norm = self._stream_add_norm

    def _norm_route(self, x, weight, offset, out_dtype):
        return (weight is not None and x.is_cuda and x.device.index == self._current() and weight.dim() == 1
                and weight.dtype == x.dtype
                and offset in (0.0, 1.0) and (out_dtype is None or out_dtype == x.dtype) and 2 <= x.dim() <= 3
                and x.stride(-1) == 1 and weight.device == x.device and weight.stride(0) == 1)

    def _provider_norm(self, x, weight, eps, offset, out):
        if out is not None and (tuple(out.shape) != tuple(x.shape) or out.dtype != x.dtype
                                or (x.shape[-1] > 1 and out.stride(-1) != 1)):
            raise ValueError(f"out is {tuple(out.shape)} {out.dtype} (its last dimension contiguous); the result is "
                             f"{tuple(x.shape)} {x.dtype}")
        norm = self._gemma if offset else self._rmsnorm
        if out is None or out.is_contiguous():
            return norm(x, weight, float(eps), out=out)
        out.copy_(norm(x, weight, float(eps)))
        return out

    def _rms_norm(self, x, weight, eps, offset=0.0, residual=None, out_dtype=None, out=None):
        if residual is None and self._norm_route(x, weight, offset, out_dtype):
            return self._provider_norm(x, weight, eps, offset, out)
        return self.source.rms_norm(x, weight, eps, offset, residual, out_dtype, out)

    def _rms_norm_heads(self, x, weight, eps, offset=0.0, out_dtype=None, out=None):
        if self._norm_route(x, weight, offset, out_dtype):
            return self._provider_norm(x, weight, eps, offset, out)
        return self.source.rms_norm_heads(x, weight, eps, offset, out_dtype, out)

    def _stream_add_norm(self, streams, x, weights, weight, eps, offset=0.0, out_dtype=None, out=None):
        if self._norm_route(streams, weight, offset, out_dtype):
            self.source.stream_add(streams, x, weights)
            return self._provider_norm(streams, weight, eps, offset, out), streams
        return self.source.stream_add_norm(streams, x, weights, weight, eps, offset, out_dtype, out)

    def _product_route(self, x, up, layout, out_dtype, out):
        if not (up is None and layout == "halves" and x.is_cuda and x.device.index == self._current()
                and x.dtype in self._sixteen
                and (out_dtype is None or out_dtype == x.dtype) and x.dim() >= 1 and x.stride(-1) == 1
                and x.shape[-1] % 2 == 0):
            return False
        shape = (*x.shape[:-1], x.shape[-1] // 2)
        if out is not None and (tuple(out.shape) != shape or out.dtype != x.dtype
                                or (shape[-1] > 1 and out.stride(-1) != 1)):
            raise ValueError(f"out is {tuple(out.shape)} {out.dtype} (its last dimension contiguous); the result is "
                             f"{shape} {x.dtype}")
        return True

    def _product(self, kind, x, out):
        """The provider's product of packed halves. It reads and writes contiguous rows only: other layouts go through
        a contiguous copy (the same bits as the copy's own call)."""
        x = x if x.is_contiguous() else x.contiguous()
        if out is None or out.is_contiguous():
            return self._products[kind](x, out=out)
        out.copy_(self._products[kind](x))
        return out

    def _silu_mul(self, x, up=None, layout="halves", out_dtype=None, out=None):
        if self._product_route(x, up, layout, out_dtype, out):
            return self._product("silu", x, out)
        return self.source.silu_mul(x, up, layout, out_dtype, out)

    def _gelu_mul(self, x, up=None, layout="halves", approximate="none", out_dtype=None, out=None):
        if approximate in ("none", "tanh") and self._product_route(x, up, layout, out_dtype, out):
            return self._product(approximate, x, out)
        return self.source.gelu_mul(x, up, layout, approximate, out_dtype, out)

    def facts(self):
        """Which code serves the device form (the engine's facts carry it, beside the registry kernel's record)."""
        found = {"kernel": KERNEL if self.kernel is not None else "torch fallback",
                 "provider": "flashinfer" if self.provider is not None else None}
        if self.lacking:
            found["kernel_lacks"] = self.lacking
        return {"norms": found}


def create(context, options, parts):
    provider = _provider() if options["flashinfer"] else None
    return FusedNorms(registry_kernel(KERNEL, folder=Path(__file__).parent), provider)
