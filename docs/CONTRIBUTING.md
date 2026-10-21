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

Run the tests in `tests/` that cover the changed behavior. Host fixtures don't qualify
device arithmetic: native, quality and performance claims need their own tests.

## Source snapshots

[CI](../.github/workflows/ci.yml) runs host and native CMake tests on Windows and Linux
with Python 3.11 and 3.14, builds independent SDK consumers and installs the wheel from
a clean snapshot:

```sh
mkdir -p build/package-source
git archive HEAD | tar -x -C build/package-source
python -B -c "import os; from setuptools.build_meta import build_sdist, build_wheel; os.chdir('build/package-source'); build_sdist('../dist'); build_wheel('../dist')"
python -m pip install build/dist/crit-0.3.0-py3-none-any.whl
```

The wheel holds `crit`, `crit_api`, `crit_eval`, `crit_kernels`, `crit_models` and `crit_core` and
depends on `tokenizers` and `jinja2`. Core's native SDK is a separate CMake project;
Inference and Eval's suite data ship in the source checkout.
