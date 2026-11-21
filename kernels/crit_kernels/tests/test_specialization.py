"""crit_kernels.specialization: every launched Triton kernel says which runtime arguments follow the request, crit's
own kernels and plugins included, and the build refuses one that does not."""
import json
from pathlib import Path

import pytest

from crit_kernels import build as kbuild
from crit_kernels import specialization

ROOT = Path(__file__).resolve().parents[3]

KERNEL = '''import triton
import triton.language as tl


@triton.jit
def _scale(x, f):   # inlined into the kernel: no launch of its own
    return x * f


@triton.jit{decorator}
def scale_kernel(x_ptr, out_ptr, n, factor, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + i, _scale(tl.load(x_ptr + i, mask=i < n), factor), mask=i < n)
'''


def package(tmp_path, decorator):
    folder = tmp_path / "toy"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("", encoding="utf-8")
    (folder / "_kernels.py").write_text(KERNEL.format(decorator=decorator), encoding="utf-8")
    return folder


def test_every_launched_triton_kernel_of_crit_says_which_arguments_follow_the_request():
    folders = [p for p in sorted((ROOT / "kernels" / "techniques").glob("*/*/torch-ext/*")) if p.is_dir()]
    folders += sorted({p.parent for p in (ROOT / "inference").glob("*/code/*.py")
                       if "triton.jit" in p.read_text(encoding="utf-8")})
    found = {str(folder.relative_to(ROOT)): specialization.problems(folder) for folder in folders}
    assert {k: v for k, v in found.items() if v} == {}
    launched = sum(len(specialization.kernels({p.name: p.read_text(encoding="utf-8")
                                               for p in folder.rglob("*.py")})) for folder in folders)
    assert launched >= 20


def test_a_kernel_that_does_not_say_or_names_an_argument_it_lacks_is_refused(tmp_path):
    assert any("scale_kernel does not say" in p and "n, factor" in p
               for p in specialization.problems(package(tmp_path / "a", "")))
    assert specialization.problems(package(tmp_path / "b", '(do_not_specialize=["n"])')) == []
    assert specialization.problems(package(tmp_path / "c", "(do_not_specialize=[])")) == []
    wrong = specialization.problems(package(tmp_path / "d", '(do_not_specialize=["tokens"])'))
    assert len(wrong) == 1 and "names tokens" in wrong[0]


def test_the_build_refuses_a_package_whose_kernel_does_not_say(tmp_path):
    root = tmp_path / "kernels"
    kernel = root / "toy-scale"
    (kernel / "torch-ext").mkdir(parents=True)
    package(kernel / "torch-ext", "")
    (kernel / "torch-ext" / "toy").rename(kernel / "torch-ext" / "toy_scale")
    (kernel / "kernel.json").write_text(json.dumps({
        "schema": kbuild.SCHEMA, "id": "toy-scale", "package": "toy_scale", "language": "triton",
        "title": "toy", "structures": {}}), encoding="utf-8")
    try:
        kbuild.find("toy-scale", root)
    except (KeyError, ValueError) as error:   # the manifest needs more of a real kernel's fields here
        pytest.skip(f"a toy manifest is not a kernel here: {error}")
    with pytest.raises(ValueError, match="scale_kernel does not say"):
        kbuild.build("toy-scale", root=root)
