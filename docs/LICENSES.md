# Licensing and third-party notices

Original crit source, including Core, Inference, Kernels integration code, Eval and the CLI,
is offered under the [Apache License, Version 2.0](../LICENSE), SPDX identifier
`Apache-2.0`. It permits commercial and noncommercial use, modification and
distribution subject to its terms, including its copyright and patent grants.

When distributing original crit code or adaptations, including Inference code copied
into an engine, retain the license and attribution notices and identify changes in
modified files. Project attribution is included in `LICENSE`. Source snapshots
include it, Python wheels and source distributions retain it in their metadata, and
installed Core SDKs include it at `share/licenses/crit/`.

The repository bundles no third-party source files. crit Eval's repository holds the DeepSeek V4
chat template crit renders that family with, a port of DeepSeek's DeepSeek V4 encoder, and the
encoder's golden tests, released under the MIT License and kept there with its notice
([crit-eval's DeepSeek V4 fixtures](https://github.com/crit-labs/crit-eval/blob/main/tests/fixtures/deepseek_v4/NOTICE.md)).
External models, kernels and dependencies have separate terms; retain their identities, licenses
and notices with each engine candidate ([Building engines](BUILDING_ENGINES.md),
[Kernels](../kernels/README.md)).
