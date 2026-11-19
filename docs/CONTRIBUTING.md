# Contributing

Contributions are licensed under the [Apache License, Version 2.0](../LICENSE)
(section 5); contributors keep their copyright. Third-party code keeps its own terms:
identify its source, confirm it may be distributed and don't rename it to hide its
origin. See [licensing](LICENSES.md).

Every change must meet the three criteria in [what belongs in crit](CONTRIBUTION_CRITERIA.md):
it improves the platform measurably, it does not overfit to one model or machine, and it is
valuable to almost any model on any hardware backend.

## How pull requests land

crit's `main` only receives release commits: each release is one squash commit, "crit X.Y.Z", whose description
is that version's [CHANGELOG](../CHANGELOG.md) section. So a pull request is never merged as it is. When a change is
accepted, a maintainer applies it to the next release as one focused commit, credits the author in its CHANGELOG
line (and, if they want, with a `Co-authored-by:` line in the release commit), and closes the pull request once that
release is out, naming the version it shipped in.

- Open an issue first for anything larger than a small fix, and link it from the pull request (`Closes #N`).
- Show the evidence the [criteria](CONTRIBUTION_CRITERIA.md) ask for: the tests you ran with their output, and for
  speed changes the measurement before and after, with the GPU named.
- Issues labeled `bounty` that are assigned to someone are reserved for them; claims and pull requests from anyone
  else are closed.
- Pull requests without evidence, or ones that change code only to touch it, are closed without review.

## Code boundaries

| Folder | Responsibility |
| --- | --- |
| `core/` | C++ request lifecycle, scheduling, resources and retirement, with its text, file and structured-output libraries; the `crit_core` and `crit_plugins` Python packages (`python/`) |
| `inference/` | Reusable technique descriptions, contracts and code |
| `kernels/` | Kernel inspection/retention, the `crit_kernels` package behind `crit kernels`, integration examples |
| `models/` | Weight transforms (PTQ), their recipes and records, the `crit_models` package behind `crit models` |
| `eval/` | Common evaluation suite, preparation, reference scoring and comparisons |
| `cli/` | The `crit` command, run records, process trees and runtime ownership |
| `api/` | `crit serve`, the OpenAI-compatible API |
| `tests/` | Host and native fixtures |
| `docs/` | Usage and interface reference |

Model-specific code, including Eval adapters for real models, belongs in engine
projects. Techniques and execution policies stay optional; Inference entries follow the
[authoring contract](../inference/README.md). Changes to Eval's suite, protocol,
reference or timing must keep a comparable identity and state their effect on existing
results. Keep caches, builds, credentials and run data outside product folders, and
never run `git clean` on a workspace holding retained runs.

## Check a change

Update docs and examples with behavior; docs describe what ships. The host suite needs
the pinned dependencies CI installs:

```sh
python -m pip install tokenizers==0.23.3 jinja2==3.1.6
python -m pip install -e core/python -e .
crit core build
python -B -m unittest discover -s tests -v
```

`-e core/python` matters: crit's version pin on `crit-core` is met by the release on PyPI too, and a checkout's CLI
on a released Core fails in confusing ways, so crit refuses to run that way. `crit core build` builds Core's runtime,
`crit_core_steps`, into `build/core-steps`, where Core's loader finds it (`CRIT_CORE_STEP_LIBRARY` names another).
`python -B tests/run_parallel.py` runs the same tests as CI does, several processes at once: one a module, or one
a test class where [tests/durations.json](../tests/durations.json) splits a module. `--shard I/N` runs one of CI's
shards, filled by those recorded times.

CI installs no transformers, so the fixtures that compare against transformers' own code (expert routing, RoPE
tables) skip there. They run with torch and a transformers in crit's supported range, `>=5.20,<6` (the `reference`
extra's), installed.

Run the tests in `tests/` that cover the changed behavior. Host fixtures don't qualify
device arithmetic: native, quality and performance claims need their own tests.

### Real checkpoints

No CI runner has a GPU or a real checkpoint, and the tests' small random checkpoints hold none of the formats
published checkpoints store their tensors in (an FP8 n-gram table, a draft layer's experts in FP8 blocks beside NVFP4
decoder experts). Two checks cover them:

- **Checkpoint manifests** (CPU, in CI): [tests/manifests](../tests/manifests) keeps the safetensors headers and
  configs of every checkpoint crit targets, and `tests/test_checkpoint_manifests.py` loads model-template over each
  one without its weights ([tests/support/dry_plan.py](../tests/support/dry_plan.py)), failing on a tensor the load
  cannot read or that no part of the model reads. Add a target's manifest with
  `python tests/support/manifests.py MODEL_FOLDER`, which reads only the headers.
- **The stress test** (a GPU and the checkpoints, on a machine that has them): `python tests/stress.py` serves
  layer-cut copies of real checkpoints ([tests/support/layer_cut.py](../tests/support/layer_cut.py): a few decoder
  layers, every tensor byte for byte, and a twelve-layer cut where an error that grows with depth shows) as
  `crit run serve --check` serves a candidate of the library's plugins at their defaults, in both quality modes and at four device memory pressures made by a VRAM ballast
  (`CRIT_VRAM_BALLAST`, [crit_plugins.memory](../core/python/crit_plugins/memory.py)), with short, 32k and
  full-context prompts, a cancel inside a prefill, a reopen at a lower context, drafting from the draft layer and
  through the drafter seam, crit smoke and crit-eval's gate pre-check repeated for its spread. A ladder of prompt
  lengths fails where a longer prompt is prefilled at a lower rate than a shorter one (a cost its length does not
  ask for). The full tier ends with one load of a whole checkpoint at the context engine builds ask for, every gate
  kept: the first load's expert store, the device memory left at the full context and drift at the model's own
  depth show only there. What does not change
  with the scenario (Core's build, the runs and their packages, crit-eval's data, expert-offload's store, the
  plugin tests) is built once per commit and cached; it writes `stress.json` and `stress.md`, a scenario a row.

Builds and releases start only from a commit where both pass ([Building engines](BUILDING_ENGINES.md)).

## Source snapshots

[CI](../.github/workflows/ci.yml) runs the host tests on Windows and Linux with Python 3.11
and 3.14, in shards, and the native CMake tests once per OS and compiler, builds independent SDK consumers
and installs the wheel from a clean snapshot:

```sh
mkdir -p build/package-source
git archive HEAD | tar -x -C build/package-source
python -B -c "import os; from setuptools.build_meta import build_sdist, build_wheel; os.chdir('build/package-source'); build_sdist('../dist'); build_wheel('../dist')"
python -m pip install build/dist/crit-0.21.0-py3-none-any.whl
```

The `crit` wheel holds `crit`, `crit_api`, `crit_kernels`, `crit_mcp`, `crit_models` and `crit_smoke`, and
Inference, the plugin library, as package data at `crit/library/inference`: every technique's and base plugin's
folder whole, without Inference's native `CMakeLists.txt` and `examples/`, which build against a checkout's `core/`.
Core's `crit_core` and `crit_plugins` are the `crit-core` wheel (`core/python`), and crit Eval is its own
distribution, `crit-eval`. crit depends on `tokenizers`, `jinja2`, `crit-core` and `crit-eval`. Core's native SDK is
a separate CMake project, and Eval's suite data ship in the source checkout. crit finds its library in the checkout it
runs from (`inference/`, installed with `-e .` or not), else in the copy its wheel ships, so an engine's plugins are
built from `python -m pip install .` as from a checkout: `crit plugins list|check|test|add` and `crit inference` act
on the installed library, and a crit without it says its install is broken. `crit run` keeps its records against a
crit checkout (`--project`) and, from a crit installed without one, in any folder: the run's brief then names crit's
repository at the installed release's tag (crit X.Y.Z is published from vX.Y.Z) for the guides, and runs stay outside
the folder crit is installed in. Serving a built engine needs neither, since a package carries its plugins.
[tests/test_wheel_install.py](../tests/test_wheel_install.py) installs the wheel from a snapshot, runs the library
from it and keeps a run of the toy engine with it, outside any checkout, up to a built candidate.
