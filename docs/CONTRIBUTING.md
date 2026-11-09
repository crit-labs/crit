# Contributing

Contributions are licensed under the [Apache License, Version 2.0](../LICENSE)
(section 5); contributors keep their copyright. Third-party code keeps its own terms:
identify its source, confirm it may be distributed and don't rename it to hide its
origin. See [licensing](LICENSES.md).

Every change must meet the three criteria in [what belongs in crit](CONTRIBUTION_CRITERIA.md):
it improves the platform measurably, it does not overfit to one model or machine, and it is
valuable to almost any model on any CUDA hardware.

## Code boundaries

| Folder | Responsibility |
| --- | --- |
| `core/` | C++ request lifecycle, scheduling, resources and retirement; text, sampling, state cache, file and model libraries; the `crit_core` Python package (`python/`) |
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
python -m pip install -e .
python -B -m unittest discover -s tests -v
```

`python -B tests/run_parallel.py` runs the same tests one module per process, several at once, as CI does.

CI installs no transformers, so the fixtures that compare against transformers' own code (expert routing, RoPE
tables) skip there. They run with torch and a transformers in crit's supported range, `>=5.20,<6` (the `reference`
extra's), installed.

Run the tests in `tests/` that cover the changed behavior. Host fixtures don't qualify
device arithmetic: native, quality and performance claims need their own tests.

## Source snapshots

[CI](../.github/workflows/ci.yml) runs the host tests on Windows and Linux with Python 3.11
and 3.14 and the native CMake tests once per OS, builds independent SDK consumers and installs the wheel from
a clean snapshot:

```sh
mkdir -p build/package-source
git archive HEAD | tar -x -C build/package-source
python -B -c "import os; from setuptools.build_meta import build_sdist, build_wheel; os.chdir('build/package-source'); build_sdist('../dist'); build_wheel('../dist')"
python -m pip install build/dist/crit-0.17.2-py3-none-any.whl
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
