"""latent-attention's sweep bench (crit_kernels.sweep): the model's latent geometry (kv_lora_rank, qk_rope_head_dim,
num_attention_heads), one decode row of `rows` columns (a verify window; row 0 its first column) at the end of N
cached positions, against attention-state's float64 reference (latent_attention.py) within BF16 probability
rounding; FlashInfer's MLA wrapper as the yardstick. A configuration that cannot hold its tiles (shared memory) fails
its build or launch and the sweep drops it."""
import importlib.util
from pathlib import Path

import torch

PAGE = 64
LENGTH = 8192
_spec = importlib.util.spec_from_file_location(
    "latent_reference", Path(__file__).resolve().parents[4] / "inference" / "attention-state" / "code" /
    "latent_attention.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


def model_shapes(model):
    rank = model.get("kv_lora_rank") or 0
    rope = model.get("qk_rope_head_dim") or 0
    heads = model.get("num_attention_heads") or 0
    if not (rank and rope and heads) or rank & (rank - 1) or rope & (rope - 1):
        return []
    return [{"R": int(rank), "P": int(rope), "H": int(heads), "PAGE": PAGE, "N": LENGTH, "KIND": kind,
             "scale": float((model.get("qk_nope_head_dim") or rank) + rope) ** -0.5} for kind in (0, 1)]


class Window:
    structure = "window"
    lengths = None

    def __init__(self):
        self._shape, self._state, self._refs = None, {}, {}

    def shapes(self, model):
        return model_shapes(model)

    def _cache(self, shape, device):
        key = tuple(sorted(shape.items()))
        if key != self._shape:
            self._shape, self._state, self._refs = key, {}, {}
            R, P, H, n = int(shape["R"]), int(shape["P"]), int(shape["H"]), int(shape.get("N", LENGTH))
            g = torch.Generator(device=device).manual_seed(1)
            pages = -(-(n + 64) // PAGE)
            cache = (torch.randn(pages, PAGE, R + P, generator=g, device=device) * 0.3).to(torch.bfloat16)
            self._state["cache"], self._state["scales"] = cache, None
            if int(shape.get("KIND", 0)):   # the same rows in an E4M3 cache (write()'s rule)
                from crit_kernels.build import load
                fp8 = torch.empty(pages, PAGE, R + P, dtype=torch.float8_e4m3fn, device=device)
                scales = torch.empty(pages, PAGE, 2, device=device)
                rows = cache.view(-1, R + P)
                load("latent-attention").write(rows[:, :R], rows[:, R:], torch.arange(rows.shape[0], device=device),
                                               fp8, scales)
                self._state["cache"], self._state["scales"] = fp8, scales
            self._state["tables"] = torch.randperm(pages, generator=g, device=device).to(torch.int32)[None, :]
            self._state["q"] = (torch.randn(64, H, R + P, generator=g, device=device) * 0.3).to(torch.bfloat16)
        return self._state

    def inputs(self, shape, rows, device):
        s = self._cache(shape, device)
        n = int(shape.get("N", LENGTH))
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        return dict(shape=shape, rows=rows, q=s["q"][:rows], cache=s["cache"], scales=s["scales"], tables=s["tables"],
                    zero=zero,
                    first=torch.full((1,), n - rows, dtype=torch.int64, device=device), n=n)

    def run(self, module, st):
        shape = st["shape"]
        return module.attend(st["q"], st["cache"], st["tables"], st["zero"], st["zero"], st["first"], st["rows"], PAGE,
                             rank=int(shape["R"]), scale=shape.get("scale"), scales=st["scales"])

    def reference(self, st):
        key = (self.structure, st["rows"])
        if key not in self._refs:
            R, n, rows = int(st["shape"]["R"]), st["n"], st["rows"]
            flat = st["cache"].view(-1, st["cache"].shape[-1])
            slots = (st["tables"][0][torch.arange(n, device=flat.device) // PAGE].long() * PAGE
                     + torch.arange(n, device=flat.device) % PAGE)
            rows_ = flat[slots].double()
            if st["scales"] is not None:   # the stored values: codes times their part's scale
                sc = st["scales"].view(-1, 2)[slots].double()
                rows_ = torch.cat([rows_[:, :R] * sc[:, :1], rows_[:, R:] * sc[:, 1:]], dim=-1)
            seen = torch.arange(n, device=flat.device)[None, :] <= (n - rows + torch.arange(rows, device=flat.device))[:, None]
            scale = st["shape"].get("scale") or (st["cache"].shape[-1]) ** -0.5
            out, _ = ref.latent_attention(st["q"][:, :, :R], st["q"][:, :, R:], rows_[:, :R], rows_[:, R:], seen, scale)
            self._refs[key] = out
        return self._refs[key]

    def check(self, got, want, st):
        err = float((got.double() - want).abs().max())
        return None if err < 8e-3 + 2e-2 * float(want.abs().max()) else f"max error {err:.3g}"

    def first_row(self, out):
        return out[0]

    def bytes(self, shape, rows):
        return int(shape.get("N", LENGTH)) * (int(shape["R"]) + int(shape["P"])) * 2

    def yardstick(self, st):
        if st["scales"] is not None:
            return None   # the yardstick reads 16-bit caches
        try:
            import flashinfer
        except ImportError:
            return None
        R = int(st["shape"]["R"])
        cache, tables, rows, n = st["cache"], st["tables"], st["rows"], st["n"]
        device = cache.device
        ckv, kpe = cache[..., :R].contiguous(), cache[..., R:].contiguous()
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(torch.empty(128 << 20, dtype=torch.uint8, device=device))
        used = -(-n // PAGE)
        wrapper.plan(torch.tensor([0, rows], dtype=torch.int32, device=device),
                     torch.tensor([0, used], dtype=torch.int32, device=device), tables[0][:used].contiguous(),
                     torch.tensor([n], dtype=torch.int32, device=device), int(st["shape"]["H"]), R,
                     cache.shape[-1] - R, PAGE, rows > 1, st["shape"].get("scale") or cache.shape[-1] ** -0.5,
                     torch.bfloat16, torch.bfloat16)
        qn, qp = st["q"][:, :, :R].contiguous(), st["q"][:, :, R:].contiguous()
        return lambda: wrapper.run(qn, qp, ckv, kpe)


class Prompt(Window):
    """The prompt reader (structure prompt): a chunk of `rows` tokens at the end of N cached positions, against the
    same float64 reference; FlashInfer's MLA wrapper with causal prompt rows as the yardstick."""
    structure = "prompt"

    def _queries(self, shape, rows, device):
        if self._state.get("prompt_rows", 0) < rows:
            g = torch.Generator(device=device).manual_seed(2)
            R, P, H = int(shape["R"]), int(shape["P"]), int(shape["H"])
            self._state["pq"] = (torch.randn(rows, H, R + P, generator=g, device=device) * 0.3).to(torch.bfloat16)
            self._state["prompt_rows"] = rows
        return self._state["pq"][:rows]

    def inputs(self, shape, rows, device):
        st = super().inputs(shape, 1, device)
        st.update(rows=rows, q=self._queries(shape, rows, device))
        return st

    def run(self, module, st):
        shape = st["shape"]
        return module.prefill(st["q"], st["cache"], st["tables"][0], st["n"] - st["rows"], PAGE, rank=int(shape["R"]),
                              scale=shape.get("scale"), scales=st["scales"])

    def bytes(self, shape, rows):
        return int(shape.get("N", LENGTH)) * (int(shape["R"]) + int(shape["P"])) * 2
