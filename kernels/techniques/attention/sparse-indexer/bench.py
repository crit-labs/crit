"""sparse-indexer's sweep bench (crit_kernels.sweep): the model's indexer geometry (index_n_heads, index_head_dim,
index_topk) over N compressed keys (a context of 4 N), tokens near its end, row t seeded by t alone; scores against
attention-state's float64 reference (sparse_index.py) within FP32 accumulation of the 16-bit products, picks against
its picks (where they differ, the chosen scores within that error of the k-th best); torch as the yardstick."""
import importlib.util
from pathlib import Path

import torch

KEYS = 16384
_spec = importlib.util.spec_from_file_location(
    "sparse_reference", Path(__file__).resolve().parents[4] / "inference" / "attention-state" / "code" /
    "sparse_index.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
CHECKED = 8


def model_shapes(model):
    heads, dim, k = model.get("index_n_heads"), model.get("index_head_dim"), model.get("index_topk")
    if not (heads and dim and k) or heads & (heads - 1) or dim not in (64, 128, 256):
        return []
    return [{"H": int(heads), "D": int(dim), "K": int(k), "N": KEYS, "KIND": kind} for kind in (0, 1)]


def _rows(rows, h, d, n, device):
    """Token t's query, weights and end, seeded by t alone (a decode token near the end of the keys)."""
    q, w, e = [], [], []
    for t in range(rows):
        g = torch.Generator().manual_seed(1000 + t)
        q.append(torch.randn(h, d, generator=g) * 0.5)
        w.append(torch.randn(h, generator=g) * h ** -0.5)
        e.append(n - int(torch.randint(0, 64, (1,), generator=g)))
    return (torch.stack(q).to(device).bfloat16(), torch.stack(w).to(device).float(),
            torch.tensor(e, dtype=torch.int32, device=device))


class Scores:
    structure = "scores"

    def __init__(self):
        self._keys, self._rows, self._refs = {}, {}, {}

    def shapes(self, model):
        return model_shapes(model)

    def valid(self, knobs):
        return True

    def _keyset(self, shape, device):
        d, n, kind = int(shape["D"]), int(shape.get("N", KEYS)), int(shape.get("KIND", 0))
        key = (d, n, kind)
        if key not in self._keys:
            g = torch.Generator().manual_seed(7)
            k = (torch.randn(n, d, generator=g) * 0.5).to(device).bfloat16()
            if kind:
                scales = (k.float().abs().amax(1) / 448.0).clamp_min(1e-30)
                codes = (k.float() / scales[:, None]).to(torch.float8_e4m3fn)
                self._keys[key] = (codes, scales, codes.float() * scales[:, None])
            else:
                self._keys[key] = (k, None, k.float())
        return self._keys[key]

    def inputs(self, shape, rows, device):
        h, d, n = int(shape["H"]), int(shape["D"]), int(shape.get("N", KEYS))
        codes, scales, values = self._keyset(shape, device)
        at = (h, d, n)
        if at not in self._rows or self._rows[at][0].shape[0] < rows:
            self._rows[at] = _rows(max(rows, 1), h, d, n, device)
        q, w, e = (x[:rows] for x in self._rows[at])
        return {"q": q, "keys": codes, "scales": scales, "values": values, "w": w, "ends": e, "K": int(shape["K"]),
                "out": torch.empty(rows, n, dtype=torch.float32, device=device), "key": (tuple(sorted(shape.items())),)}

    def run(self, module, s):
        return module.scores(s["q"], s["keys"], s["w"], s["ends"], key_scales=s["scales"], out=s["out"])

    def reference(self, s):
        rows = min(s["q"].shape[0], CHECKED)
        key = s["key"] + (rows,)
        if key not in self._refs:
            d = s["q"].shape[-1]
            self._refs[key] = (ref.sparse_scores(s["q"][:rows], s["values"], s["w"][:rows], s["ends"][:rows], d ** -0.5),
                               ref.sparse_scores(s["q"][:rows].abs(), s["values"].abs(), s["w"][:rows].abs(),
                                                 s["ends"][:rows], d ** -0.5))
        return self._refs[key]

    def check(self, got, want, s):
        value, magnitude = want
        rows = value.shape[0]
        g = got[:rows].double()
        finite = torch.isfinite(value)
        if not torch.equal(finite, torch.isfinite(g)):
            return "-inf where the reference has a score, or a score where it has -inf"
        bad = int(((g - value).abs() > magnitude * 1e-4 + 1e-6)[finite].sum())
        return None if bad == 0 else f"{bad} scores beyond FP32 accumulation of the 16-bit products"

    def first_row(self, out):
        return out[0]

    def yardstick(self, s):
        q, keys, w = s["q"], s["values"].to(s["q"].dtype), s["w"]
        d = q.shape[-1]

        def call():
            prod = torch.matmul(q, keys.t())              # [T][H][N]
            return (torch.relu(prod).float() * (w * d ** -0.5)[..., None]).sum(1)
        return call

    def bytes(self, shape, rows):
        n, d, h = int(shape.get("N", KEYS)), int(shape["D"]), int(shape["H"])
        return n * d * (1 if int(shape.get("KIND", 0)) else 2) + rows * h * d * 2 + rows * n * 4


class TopK:
    structure = "topk"
    nan_rule = "contained"   # a NaN ranks below every number: its row's picks stay positions

    def __init__(self):
        self._scores, self._refs = {}, {}

    def shapes(self, model):
        return [dict(s, KIND=0) for s in model_shapes(model)][:1]

    def valid(self, knobs):
        return True

    def inputs(self, shape, rows, device):
        n, k = int(shape.get("N", KEYS)), int(shape["K"])
        at = (n,)
        if at not in self._scores or self._scores[at][0].shape[0] < rows:
            s, e = [], []
            for t in range(max(rows, 1)):
                g = torch.Generator().manual_seed(2000 + t)
                row = torch.randn(n, generator=g)
                row[torch.randint(0, n, (n // 8,), generator=g)] = 0.0     # ties, as ReLU leaves them
                s.append(row)
                e.append(n - int(torch.randint(0, 64, (1,), generator=g)) if t % 7 else int(k // 2))
            self._scores[at] = (torch.stack(s).to(device), torch.tensor(e, dtype=torch.int32, device=device))
        sc, e = (x[:rows] for x in self._scores[at])
        return {"s": sc, "ends": e, "K": k, "out": torch.empty(rows, k, dtype=torch.int32, device=device),
                "key": (n, k)}

    def run(self, module, s):
        return module.topk(s["s"], s["ends"], s["K"], out=s["out"])

    def reference(self, s):
        rows = min(s["s"].shape[0], CHECKED)
        key = s["key"] + (rows,)
        if key not in self._refs:
            self._refs[key] = ref.sparse_index(s["s"][:rows], s["ends"][:rows], s["K"])
        return self._refs[key]

    def check(self, got, want, s):
        rows = want.shape[0]
        if not torch.equal(got[:rows].long(), want):
            return "picks differ from the reference's (k best, ties to the lower position, ascending, then -1)"
        return None

    def first_row(self, out):
        return out[0]

    def yardstick(self, s):
        sc, e, k = s["s"], s["ends"], s["K"]
        n = sc.shape[1]

        def call():
            masked = sc.masked_fill(torch.arange(n, device=sc.device)[None, :] >= e[:, None].long(), float("-inf"))
            return torch.sort(torch.topk(masked, k, dim=1).indices, dim=1).values
        return call

    def bytes(self, shape, rows):
        return rows * int(shape.get("N", KEYS)) * 4 * 5
