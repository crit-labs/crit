# Roadmap

What crit is working toward, by theme. Each item is a GitHub issue labeled by area and priority; issues labeled
[`bounty`](https://github.com/crit-labs/crit/issues?q=is%3Aissue+is%3Aopen+label%3Abounty) are paid on a fixed
done-check ([template](.github/ISSUE_TEMPLATE/bounty.yml)). The issues are the source of truth: this page only groups
them.

## Decode rounds

Captured rounds that keep the GPU busy and cover every request.

- Close the idle gap between captured rounds ([#35](https://github.com/crit-labs/crit/issues/35))
- Constrained requests in a captured round ([#36](https://github.com/crit-labs/crit/issues/36))
- A faster speculative verify pass ([#43](https://github.com/crit-labs/crit/issues/43))

## Prompt speed

Prompt GEMMs and long context near the hardware's rate, without giving up exact summation where a mode needs it.

- NVFP4 W4A4 prompt GEMMs ([#39](https://github.com/crit-labs/crit/issues/39))
- The exact 16-bit prompt tile ([#40](https://github.com/crit-labs/crit/issues/40))
- FP16-accumulate prompt GEMMs ([#41](https://github.com/crit-labs/crit/issues/41))
- The long-context prompt reader ([#42](https://github.com/crit-labs/crit/issues/42))

## Quantized formats

Every GGUF block format read in its stored form, at bandwidth.

- Ternary TQ1_0 and TQ2_0 ([#53](https://github.com/crit-labs/crit/issues/53), bounty)
- Low-bit IQ decode speed ([#52](https://github.com/crit-labs/crit/issues/52), bounty)
- Q4_K and Q6_K prompt GEMMs ([#45](https://github.com/crit-labs/crit/issues/45))
- exl3-rows at 16 rows ([#44](https://github.com/crit-labs/crit/issues/44))

## Attention

Library attention for every common layer kind, tested against a float64 reference.

- Keys derived from cached values ([#47](https://github.com/crit-labs/crit/issues/47), bounty)
- Library interfaces that do not assume the library's layouts ([#57](https://github.com/crit-labs/crit/issues/57))

## Building engines faster

Shorter build loops and clearer records.

- Ablations and Evals that redo only what changed ([#38](https://github.com/crit-labs/crit/issues/38))
- GPU time by kernel family per token ([#48](https://github.com/crit-labs/crit/issues/48))
- Real kernel templates from `crit kernels new` ([#46](https://github.com/crit-labs/crit/issues/46))
- Smoke: rounding or fault ([#49](https://github.com/crit-labs/crit/issues/49)), limits relative to the checkpoint
  ([#60](https://github.com/crit-labs/crit/issues/60))
- Run records: unreachable rungs ([#50](https://github.com/crit-labs/crit/issues/50)), field hints
  ([#51](https://github.com/crit-labs/crit/issues/51)), skipped reference tests
  ([#61](https://github.com/crit-labs/crit/issues/61)), retroactive plans ([#62](https://github.com/crit-labs/crit/issues/62))

## Reliability

- The draft-head re-encode memory peak ([#37](https://github.com/crit-labs/crit/issues/37))
- Engine commands and the process guard on Microsoft Store Python ([#3](https://github.com/crit-labs/crit/issues/3))

## Portability

- A backend-neutral kernel core ([#55](https://github.com/crit-labs/crit/issues/55))
- Multi-GPU ([#56](https://github.com/crit-labs/crit/issues/56))
- macOS ([#2](https://github.com/crit-labs/crit/issues/2))

## Serving and measurement

- Serving a sealed external candidate ([#54](https://github.com/crit-labs/crit/issues/54))
- Recorded traffic as run cases ([#58](https://github.com/crit-labs/crit/issues/58))
- Energy per token ([#59](https://github.com/crit-labs/crit/issues/59))
