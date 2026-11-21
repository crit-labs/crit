"""token-sampling's sweep and gates bench: greedy rows through the fast path, against torch.argmax (Core's rank order
takes the lower id on ties, as torch's argmax takes the first). The draws of sampled rows follow Core's Philox stream,
which the token-sampling plugin's tests and the C++ suites (tests/inference/token_law_tests.cpp,
token_sampling_tests.cpp) check against Core's own sampler."""
import torch


def vocab_shapes(model):
    vocab = model.get("vocab_size")
    return [{"vocab": int(vocab)}] if vocab else []


class Greedy:
    structure = "fast-law"
    default = None   # the untuned build: the sweep sets it, else loaded on first use

    def module(self):
        if self.default is None:
            from crit_kernels import build as kbuild
            self.default = kbuild.load("token-sampling")
        return self.default

    def shapes(self, model):
        return vocab_shapes(model)

    def inputs(self, shape, rows, device):
        ts = self.module()
        vocab = int(shape["vocab"])
        logits = torch.stack([torch.randn(vocab, generator=torch.Generator().manual_seed(1000 + r))
                              for r in range(rows)]).to(device)
        requests = torch.cat([ts.pack_request(0.0, seed=r) for r in range(rows)]).to(device)
        return {"logits": logits, "requests": requests, "rows": rows, "vocab": vocab,
                "out": torch.empty(rows, dtype=torch.int32, device=device),
                "scratch": torch.empty(ts.scratch_bytes(vocab, rows), dtype=torch.uint8, device=device)}

    def run(self, module, state):
        module.sample(state["logits"], state["vocab"], state["requests"], state["rows"], 1, 0, 0, state["out"], None,
                      None, state["scratch"], 0, None, None, None, None, None)
        return state["out"]

    def reference(self, state):
        return torch.argmax(state["logits"], dim=1).to(torch.int32)

    def check(self, got, want, state):
        if torch.equal(got, want):
            return None
        bad = int((got != want).sum())
        return f"{bad} of {got.numel()} greedy rows away from the argmax"

    def first_row(self, out):
        return out[:1]

    def bytes(self, shape, rows):
        return rows * int(shape["vocab"]) * 4

    def yardstick(self, state):
        logits = state["logits"]
        return lambda: torch.argmax(logits, dim=1)
