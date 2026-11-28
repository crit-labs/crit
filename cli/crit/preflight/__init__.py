"""crit's library on a real checkpoint's tensors, before an agent builds on it: the preflight and its parts.

A build starts from the library's plugins at their defaults, so a toolkit bug the checkpoint's geometry exposes (a head
width no kernel takes, a cache sized for another vocabulary, a pool that evicts what a prompt cache keeps) is met
first by the agent, at the price of its candidates. The preflight meets it first, in minutes and without an agent:
`crit run ... preflight` cuts the run's own checkpoint (cut: the first layer of every kind, the draft layer and the
lookup tables, every kept tensor byte for byte, hard-linked where a shard is kept whole), packages a candidate of the
library's plugins at their defaults for it (candidate), serves it as crit run serve --check serves a candidate and
runs the scenarios its facts make meaningful (plan, scenarios), and keeps the report in the run folder (records).

The release-time stress test (tests/stress.py) is the same code over the checkpoints crit targets, at fixed pressures
and in two tiers; it is the check before a release, the preflight the check before a build.

| Module | What it holds |
| --- | --- |
| cut | layer-cut checkpoints (python -m crit.preflight.cut SOURCE TARGET) |
| candidate | the library-defaults selection for a checkpoint, and a run whose one candidate packages it |
| scenarios | the chat client, the prompts sized by the engine's tokenizer, the report and the scenarios of a load |
| plan | which loads and scenarios a run's checkpoint asks, every size from the run |
| records | the report's key (crit's code and the checkpoint's revision), its files and the run's newest one |
| run | crit run ... preflight: cut, build, plan, serve every load under the GPU lock within a time budget |
"""
