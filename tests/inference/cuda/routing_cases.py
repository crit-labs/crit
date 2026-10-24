"""Routing cases for the CUDA kernel, with expectations from the host reference routing.py.

`python routing_cases.py OUT` writes them for crit_inference_cuda_tests --routing OUT. The format is
whitespace-separated: the case count, then per case the Routing fields, experts, top and tokens,
the FP32 logits, the bias when present, and the expected IDs and weights."""
from dataclasses import replace
import importlib.util
import random
import struct
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "crit_routing_reference",
    Path(__file__).resolve().parents[3] / "inference" / "heterogeneous-expert-dispatch" / "code" / "routing.py")
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)

SOFTMAX_BEFORE = r.Routing(score_before_selection=True)
BIASED_SIGMOID = r.Routing(scoring="sigmoid", score_before_selection=True, correction_bias=True, renormalize=True,
                           renormalize_epsilon=1e-20)
# (routing, experts, top, tokens, logit generator); parameterizations follow tests/test_expert_routing.py.
CASES = (
    (r.Routing(), 19, 5, 8, "normal"),                                          # softmax over selected logits
    (replace(SOFTMAX_BEFORE, renormalize=True), 128, 8, 7, "normal"),           # renormalized top-k
    (SOFTMAX_BEFORE, 60, 4, 3, "normal"),                                       # unrenormalized top-k
    (replace(BIASED_SIGMOID, groups=8, selected_groups=4, group_score_top=2, scaling=2.5), 256, 8, 8, "normal"),
    (replace(BIASED_SIGMOID, scaling=2.827), 384, 8, 5, "normal"),             # one group
    (replace(SOFTMAX_BEFORE, groups=8, selected_groups=3, scaling=16.0), 160, 6, 4, "normal"),
    (r.Routing(scoring="sigmoid"), 16, 1, 6, "normal"),                         # sigmoid after selection
    (replace(SOFTMAX_BEFORE, correction_bias=True, renormalize=True, renormalize_floor=1e-12), 64, 6, 2, "normal"),
    (replace(SOFTMAX_BEFORE, correction_bias=True, scaling=6.0), 1024, 32, 2, "normal"),
    (r.Routing(groups=32, selected_groups=8, group_score_top=3), 1024, 32, 1, "normal"),
    (r.Routing(groups=8, selected_groups=2, group_score_top=2), 32, 6, 3, "ties"),
    (replace(BIASED_SIGMOID, groups=4, selected_groups=1, group_score_top=1), 8, 2, 2, "ties"),
)


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def render(seed=7):
    rng = random.Random(seed)
    out = [str(len(CASES))]
    for routing, experts, top, tokens, kind in CASES:
        if kind == "ties":
            logits = [f32(float((e % 3) == 0)) for _ in range(tokens) for e in range(experts)]
        else:
            logits = [f32(rng.gauss(0, 2)) for _ in range(tokens * experts)]
        bias = [f32(rng.gauss(0, 0.1)) if kind != "ties" else 0.0 for _ in range(experts)]
        out.append(" ".join(str(v) for v in (
            r.SCORINGS.index(routing.scoring), int(routing.score_before_selection), int(routing.correction_bias),
            routing.groups, routing.selected_groups, routing.group_score_top, int(routing.renormalize),
            repr(f32(routing.renormalize_epsilon)), repr(f32(routing.renormalize_floor)), repr(f32(routing.scaling)),
            experts, top, tokens)))
        out.append(" ".join(repr(v) for v in logits))
        if routing.correction_bias:
            out.append(" ".join(repr(v) for v in bias))
        ids, weights = [], []
        for t in range(tokens):
            chosen, values = r.route(logits[t * experts:(t + 1) * experts], top, replace(
                routing, renormalize_epsilon=f32(routing.renormalize_epsilon),
                renormalize_floor=f32(routing.renormalize_floor), scaling=f32(routing.scaling)),
                bias if routing.correction_bias else None)
            ids += chosen
            weights += values
        out.append(" ".join(str(v) for v in ids))
        out.append(" ".join(repr(v) for v in weights))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(render(), encoding="utf-8")
