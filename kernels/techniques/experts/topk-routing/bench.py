"""topk-routing's sweep bench (crit_kernels.sweep): the model's router shape (experts, top) and rule, FP32 logits whose
row i is seeded by i alone. No knob changes what a row computes, so a configuration must give the untuned build's ids
and weights bit for bit (the build tests/test_topk_routing.py checks against an FP64 reference of the rule)."""
import torch

from crit_kernels import build as kbuild


def _logits(rows, experts, device):
    return torch.stack([torch.randn(experts, generator=torch.Generator().manual_seed(1000 + i)) * 3
                        for i in range(rows)]).to(device)


class WarpRows:
    structure = "warp-rows"
    # A NaN ranks below every number and a row whose weights come out NaN gets weights 0 (README: junk positions of a
    # captured decode round must still route to valid experts without poisoning their sums).
    nan_rule = "contained"

    def __init__(self):
        self._logits, self._bias, self._references, self._default = {}, {}, {}, None

    def shapes(self, model):
        experts = model.get("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
        top = model.get("num_experts_per_tok", "moe_top_k", "top_k")
        if not experts or not top:
            return []
        sigmoid = str(model.get("scoring_func", default="softmax")).lower() == "sigmoid"
        bias = bool(model.get("topk_method", default="") == "noaux_tc" or sigmoid)
        return [{"experts": int(experts), "top": int(top), "softmax": int(not sigmoid), "before": 1,
                 "renormalize": int(bool(model.get("norm_topk_prob", default=sigmoid))), "bias": int(bias)}]

    def inputs(self, shape, rows, device):
        experts, top = int(shape["experts"]), int(shape["top"])
        key = (experts, rows)
        if key not in self._logits:   # one set of logits a shape and row count, kept for every configuration
            self._logits[key] = _logits(rows, experts, device)
        bias = None
        if int(shape.get("bias", 0)):
            if experts not in self._bias:
                g = torch.Generator().manual_seed(experts)
                self._bias[experts] = ((torch.rand(experts, generator=g) - 0.5) * 0.1).to(device)
            bias = self._bias[experts]
        ids = torch.empty(rows, top, dtype=torch.int64, device=device)
        weights = torch.empty(rows, top, dtype=torch.float32, device=device)
        rule = dict(softmax=bool(int(shape.get("softmax", 1))), before=bool(int(shape.get("before", 0))),
                    renormalize=bool(int(shape.get("renormalize", 0))))
        return {"logits": self._logits[key], "top": top, "bias": bias, "rule": rule, "ids": ids, "weights": weights,
                "key": (experts, top, rows, tuple(sorted(rule.items())), bias is not None)}

    def run(self, module, s):
        return module.route(s["logits"], s["top"], s["bias"], ids=s["ids"], weights=s["weights"], **s["rule"])

    def reference(self, s):
        if s["key"] not in self._references:
            if self._default is None:
                self._default = kbuild.load("topk-routing")   # the untuned build plugins load
            ids, weights = self._default.route(s["logits"], s["top"], s["bias"], **s["rule"])
            self._references[s["key"]] = (ids.clone(), weights.clone())
        return self._references[s["key"]]

    def check(self, got, want, s):
        if not torch.equal(got[0], want[0]):
            return "ids differ from the untuned build's"
        if not torch.equal(got[1].view(torch.int32), want[1].view(torch.int32)):
            return "weights differ from the untuned build's bits"
        return None

    def yardstick(self, s):
        """The torch ops an engine without this kernel runs: softmax or sigmoid scores, the bias, the selection, the
        picked weights' softmax or renormalization; the selection by torch.topk below 1,024 rows and by FlashInfer's
        row top-k (flashinfer.top_k, sorted) from 1,024 rows, the faster of the two at each (RTX 5090, 512 experts:
        torch 24 us against 91 at 1 row, 210 against 153 at 4,096). No library fuses the whole rule: FlashInfer's
        fused_topk_deepseek always limits groups (one group of all experts is rejected for top > 1)."""
        logits, top, bias, rule = s["logits"], s["top"], s["bias"], s["rule"]
        top_k = None
        if logits.shape[0] >= 1024:
            try:
                from flashinfer import top_k
            except ImportError:
                top_k = None

        def pick(v):
            return top_k(v, top, sorted=True)[1] if top_k else torch.topk(v, top, dim=-1).indices

        def call():
            if rule["softmax"] and not rule["before"]:
                ids = pick(logits if bias is None else logits + bias)
                return ids, logits.gather(1, ids).softmax(-1)
            scores = logits.softmax(-1) if rule["softmax"] else logits.sigmoid()
            ids = pick(scores if bias is None else scores + bias)
            weights = scores.gather(1, ids)
            return ids, weights / weights.sum(-1, keepdim=True) if rule["renormalize"] else weights
        return call

    def first_row(self, out):
        ids, weights = out
        return torch.cat((ids[0].contiguous().view(torch.uint8), weights[0].contiguous().view(torch.uint8)))

    def bytes(self, shape, rows):
        experts, top = int(shape["experts"]), int(shape["top"])
        return rows * experts * 4 + rows * top * 12
