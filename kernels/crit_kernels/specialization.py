"""Which runtime arguments of a Triton kernel follow the request: every launched kernel says so.

Triton specializes a runtime argument it is not told to leave alone: an integer equal to 1 becomes a constant, one
divisible by 16 says so, a pointer says it is 16-byte aligned, and each mix is a kernel of its own. An argument that
follows the request (a token count, a key count, a position, a split count, the stride of a buffer made per call)
then compiles a new kernel the first time a request brings a new mix, whatever the load warmed: the request waits
seconds and crit run serve --check fails compiles_after_load. An argument the model fixes (a head dim, a weight's
stride, a pool's table width) may stay specialized, which keeps the loads its divisibility allows vectorized.

So every launched kernel (a @triton.jit function no other kernel of its package calls) states the arguments that
follow the request in do_not_specialize, an empty list where none does, and a pointer to a slice at an offset the
request sets (a request's token ends, a row of a per-call buffer) in do_not_specialize_on_alignment; crit_kernels.build refuses a package whose
kernel does not, or names an argument it does not have. The rule is about classifying, which only the kernel's
author can do from its launcher; Core's compile watch (crit_plugins.compiles) names the argument whose
specialization changed when a classification is wrong.

    problems(folder)     # ["<file>: <kernel> ...", ...] for a package folder's Python files
"""
import ast
from pathlib import Path


def _jit(node):
    """The node's @triton.jit decorator (a Name, Attribute or Call), or None."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if ast.unparse(target) in ("triton.jit", "jit"):
            return decorator
    return None


def _named(decorator, keyword_name="do_not_specialize"):
    """The names in a decorator's do_not_specialize (or another keyword's list), or None where it has none."""
    if not isinstance(decorator, ast.Call):
        return None
    for keyword in decorator.keywords:
        if keyword.arg == keyword_name:
            if not isinstance(keyword.value, (ast.List, ast.Tuple)):
                return [ast.unparse(keyword.value)]
            return [element.value if isinstance(element, ast.Constant) else ast.unparse(element)
                    for element in keyword.value.elts]
    return None


def _runtime(node):
    """The kernel's runtime parameters: those not annotated constexpr."""
    return [a.arg for a in node.args.args if not (a.annotation is not None and "constexpr" in ast.unparse(a.annotation))]


def kernels(sources):
    """{name: (file, the FunctionDef)} of the launched kernels among sources ({file: text}): @triton.jit functions no
    other @triton.jit function of these sources calls."""
    found, called = {}, set()
    for file, text in sources.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and _jit(node) is not None:
                found[node.name] = (file, node)
    for name, (_, node) in found.items():
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                callee = ast.unparse(sub.func).split(".")[-1]
                if callee != name:
                    called.add(callee)
    return {name: value for name, value in found.items() if name not in called}


def problems(folder):
    """What a package's launched kernels leave unsaid (the module's description): a kernel without
    do_not_specialize, or one naming an argument it does not have. [] where every one says."""
    folder = Path(folder)
    sources = {p.relative_to(folder).as_posix(): p.read_text(encoding="utf-8") for p in sorted(folder.rglob("*.py"))
               if "__pycache__" not in p.parts}
    out = []
    for name, (file, node) in sorted(kernels(sources).items()):
        runtime = _runtime(node)
        named = _named(_jit(node))
        if named is None:
            out.append(f"{file}: {name} does not say which of its runtime arguments follow the request "
                       f"({', '.join(runtime) or 'none'}): list them in @triton.jit(do_not_specialize=[...]), [] "
                       "where none does")
            continue
        aligned = _named(_jit(node), "do_not_specialize_on_alignment") or []
        unknown = [n for n in named + aligned if n not in runtime]
        if unknown:
            out.append(f"{file}: {name}'s do_not_specialize names {', '.join(map(str, unknown))}, which are not its "
                       "runtime arguments")
    return out
