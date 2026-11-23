"""Each model technique's entry.json declares what its code reads (`reads`), and nothing its code does not read.

A technique reads its tensors, its config keys and its layout fields under its own names (models/techniques/README.md,
docs/CONTRIBUTION_CRITERIA.md#model-neutral-code); its entry.json lists them, and `crit models inspect` matches
the tensors and config keys against a checkpoint. This test reads each technique's code (ast: nothing is imported)
and derives the three lists:

- reads.tensors: each tensor name its code reads (the core's _tensor and _vector, the matrices' load, together, bias,
  has, rows, format, expert and stacked_expert, the checkpoint's get, nbytes, shape and headers, `name in ck`, and a
  residual's anchor() and names()), from where its own spelling starts: after the part its caller gives (the block's
  prefix) or a layout field without a default gives. A layout field read with a default is spelled as that default,
  a layer's or an expert's number as N.
- reads.config: each config.json key its code reads by name (model.setting(KEY), model.config.get(KEY), the layout's
  `defaults` by key), a key inside a config entry it reads as ENTRY.KEY (rope_parameters.factor), inside an entry a
  layout field names as *.KEY. What it reads through the parsed config's typed fields (config.moe, config.mla,
  config.linear, config.sliding_window, ...) is the config reader's (crit_core.model.config, crit_models.config).
- reads.layout: each layout field its code reads (its spec's and the layout's own), with those of its spec that a
  models-core helper it hands the spec to reads (kv_shape: head_dim, kv_heads), but the core's own fields of every
  block (block, prefix, kind).

Values are followed through the code: module and local constants, loops and comprehensions over them, the functions
and methods of its own files called with the values given them, closures, and the attributes it sets on self. A
technique's class built on another technique's (needs_sources) reads that one's names under that one's entry.

    python tests/test_technique_reads.py           the check
    python tests/test_technique_reads.py --write   each entry.json's reads rewritten from its code
"""
import ast
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TECHNIQUES = ROOT / "models/techniques"
CORE = ROOT / "models/core/model-template/code"
KINDS = ("tensors", "config", "layout")

OUT, NUM = "\x00", "\x01"   # in a name: a part its caller or a layout gives; a number
CORE_FIELDS = {"block", "prefix", "kind"}   # the fields the models core reads or sets on every block's spec
MATRICES = {"load": (0,), "together": (0,), "bias": (0,), "has": (0,), "rows": (0,), "format": (0,),
            "expert": (0, 1, 2), "stacked_expert": (0, 1)}
CHECKPOINT = {"get": (0,), "nbytes": (0,), "shape": (0,)}
NAMED = {"_tensor": 1, "_vector": 1}   # the core's readers of one tensor: (model, name)
RETURNS_NAMES = ("anchor", "names")    # a residual's methods that give the core tensor names
PASS = {"tuple", "list", "sorted", "set", "frozenset", "reversed", "dict", "str", "int", "float", "bool"}
UNK = frozenset({("unk",)})
EMPTY = frozenset()
LIMIT = 4096        # the values an expression may take before the reader gives up on it
EXPAND = 64         # the iterations of a loop the reader follows one by one
DEPTH = 12          # nested calls it follows


def _v(*items):
    return frozenset(items)


def _union(*values):
    found = frozenset().union(*values)
    if len(found) > LIMIT:
        raise RuntimeError(f"an expression takes more than {LIMIT} values: the reader cannot follow it")
    return found


def _known(text):
    return OUT not in text and NUM not in text


def _piece(item):
    """An item's text inside a name being built."""
    if item[0] == "s":
        return item[1]
    if item[0] == "i":
        return str(item[1])
    return NUM if item[0] == "n" else OUT


def _string(text):
    return ("s", re.sub(OUT + "+", OUT, text))


def _members(value):
    """([Value], finite): each element of a sequence in order (a literal tuple's, a dict's keys, a known string's
    characters), or one value for any element where the count is not known."""
    found, finite = [], True
    for item in value:
        kind = item[0]
        if kind == "seq":
            found += list(item[1])
        elif kind == "map":
            found += [key for key, _ in item[1]]
            if item[2]:
                found.append(UNK)
                finite = False
        elif kind == "s" and _known(item[1]):
            found += [_v(("s", c)) for c in item[1]]
        else:
            found.append(item[1] if kind == "each" else UNK)
            finite = False
    return found, finite


def _at(value, index):
    """The element `index` of a sequence value (nothing where a literal one is shorter)."""
    found = set()
    for item in value:
        if item[0] == "seq":
            if -len(item[1]) <= index < len(item[1]):
                found |= item[1][index]
        elif item[0] == "each":
            found |= item[1]
        else:
            found.add(("unk",))
    return frozenset(found)


def _truth(value):
    """True or False where every value the expression may take is that, else None."""
    found = set()
    for item in value:
        if item[0] == "b":
            found.add(item[1])
        elif item[0] == "none":
            found.add(False)
        elif item[0] == "i":
            found.add(bool(item[1]))
        elif item[0] == "s" and _known(item[1]):
            found.add(bool(item[1]))
        else:
            return None
    return found.pop() if len(found) == 1 else None


def _is_checkpoint(node):
    return (isinstance(node, ast.Name) and node.id == "ck") or (isinstance(node, ast.Attribute) and node.attr == "ck")


def _static(fn):
    return any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in getattr(fn, "decorator_list", ()))


def _private(name):
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def tensor_name(text):
    """A tensor name as entry.json lists it, from the name the code builds: from where its own spelling starts, N for
    a number between dots; None where nothing of the code's own is left."""
    tail = text.rsplit(OUT, 1)[-1].replace(NUM, "N")
    tail = re.sub(r"(?<![^.])\d+(?![^.])", "N", tail)
    return tail if re.fullmatch(r"[A-Za-z][\w.]*\w", tail) and ".." not in tail else None


def core_helpers():
    """{function: {fields}} of the models core's functions that take a block's spec (a parameter `spec`): the spec
    fields each reads by name."""
    found = {}
    for path in sorted(CORE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    any(a.arg == "spec" for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
                fields = set()
                for sub in ast.walk(node):
                    key = None
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get" \
                            and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "spec" and sub.args:
                        key = sub.args[0]
                    elif isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == "spec":
                        key = sub.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        fields.add(key.value)
                found[node.name] = found.get(node.name, set()) | fields
    return found


class Scope:
    """The names a function (or a comprehension) binds: binders[name] = [(visible from line, top, binder)], top for
    an unconditional assignment in the function's own body, which hides the ones before it."""

    alive = []   # every scope made, so that no id the reader's memos key by is reused

    def __init__(self, parent, cls=None, flow=False):
        self.parent, self.cls, self.flow, self.binders = parent, cls, flow, {}
        Scope.alive.append(self)

    def bind(self, name, binder, line=0, top=False):
        self.binders.setdefault(name, []).append((line, top, binder))


class Reader:
    """What one technique's code reads (`reads()`), each code file a module of its own."""

    def __init__(self, files, helpers=None):
        self.helpers = core_helpers() if helpers is None else helpers
        self.found = {kind: set() for kind in KINDS}
        self.memo, self.comps, self.calls, self.funcs, self.func_ids = {}, {}, {}, {}, {}
        self.canonical, self.classes, self.depth = {}, {}, 0
        self.modules = []
        for path in files:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
            module = Scope(None)
            self._collect(module, tree.body, top=False)
            self.modules.append((tree, module))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    self._class(node, module)

    # -- binding ---------------------------------------------------------------------------------------------------
    def _class(self, node, module):
        methods = {n.name: n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        attrs = {}
        body = Scope(module)
        self._collect(body, [n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                      top=False)
        for name, binders in body.binders.items():
            attrs.setdefault(name, []).extend((b, body) for _, _, b in binders)
        self.classes[node.name] = {"node": node, "module": module, "methods": methods, "attrs": attrs,
                                   "bases": [b.id for b in node.bases if isinstance(b, ast.Name)]}
        for fn in methods.values():
            if _static(fn) or not (fn.args.posonlyargs + fn.args.args):
                continue
            me = (fn.args.posonlyargs + fn.args.args)[0].arg
            for sub in ast.walk(fn):
                if isinstance(sub, (ast.Assign, ast.AnnAssign)) and getattr(sub, "value", None) is not None:
                    for target in (sub.targets if isinstance(sub, ast.Assign) else [sub.target]):
                        self._self_target(attrs, me, fn, target, sub.value, ())

    def _self_target(self, attrs, me, fn, target, value, path):
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == me:
            attrs.setdefault(target.attr, []).append((("unpack", value, path) if path else ("expr", value), fn))
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not path and isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
                for t, v in zip(target.elts, value.elts):
                    self._self_target(attrs, me, fn, t, v, ())
            else:
                for i, t in enumerate(target.elts):
                    self._self_target(attrs, me, fn, t, value, path + (i,))

    def _target(self, scope, target, value, line, top, path=()):
        if isinstance(target, ast.Name):
            scope.bind(target.id, ("unpack", value, path) if path else ("expr", value), line, top and not path)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if not path and isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
                for t, v in zip(target.elts, value.elts):
                    self._target(scope, t, v, line, top)
            else:
                for i, t in enumerate(target.elts):
                    self._target(scope, t, value, line, top, path + (i,))

    def _loop_target(self, scope, target, iterable, line, path=()):
        if isinstance(target, ast.Name):
            scope.bind(target.id, ("iter", iterable, path), line)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, t in enumerate(target.elts):
                self._loop_target(scope, t, iterable, line, path + (i,))

    def _collect(self, scope, statements, top):
        """The binders of a body's statements (not of the functions, classes and comprehensions inside it)."""
        for s in statements:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope.bind(s.name, ("func", s), s.lineno)
            elif isinstance(s, ast.ClassDef):
                scope.bind(s.name, ("class", s), s.lineno)
            elif isinstance(s, ast.Assign):
                for target in s.targets:
                    self._target(scope, target, s.value, s.end_lineno, top)
            elif isinstance(s, ast.AnnAssign) and s.value is not None:
                self._target(scope, s.target, s.value, s.end_lineno, top)
            elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
                scope.bind(s.target.id, ("expr", s.value), s.end_lineno)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                self._loop_target(scope, s.target, s.iter, s.iter.end_lineno)
                self._collect(scope, s.body + s.orelse, False)
            elif isinstance(s, (ast.If, ast.While)):
                self._collect(scope, s.body + s.orelse, False)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    if item.optional_vars is not None:
                        self._target(scope, item.optional_vars, ast.Constant(None), s.lineno, False)
                self._collect(scope, s.body, top)
            elif isinstance(s, ast.Try):
                for handler in s.handlers:
                    if handler.name:
                        scope.bind(handler.name, ("value", UNK), handler.lineno)
                    self._collect(scope, handler.body, False)
                self._collect(scope, s.body + s.orelse + s.finalbody, False)
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for alias in s.names:
                    scope.bind((alias.asname or alias.name).split(".")[0], ("value", UNK), s.lineno)
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and \
                    isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "append" and \
                    isinstance(s.value.func.value, ast.Name) and s.value.args:
                scope.bind(s.value.func.value.id, ("append", s.value.args[0]), s.end_lineno)

    def _func(self, node, parent, cls, me=None):
        key = (id(node), id(parent), me)
        if key not in self.func_ids:
            self.func_ids[key] = len(self.funcs)
            self.funcs[len(self.funcs)] = (node, parent, cls, me)
        return _v(("func", self.func_ids[key]))

    def _scope_of(self, fn, parent, cls, values):
        scope = Scope(parent, cls, flow=True)
        for name, value in values.items():
            scope.bind(name, ("value", value), 0, True)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._collect(scope, fn.body, True)
        return scope

    def _free(self, fn, module, cls):
        """A function's scope with its parameters unknown: self (cls) its class, spec the layout's entry, defaults
        the layout's defaults (config values by key)."""
        if id(fn) not in self.canonical:
            values = {}
            params = fn.args.posonlyargs + fn.args.args
            for i, a in enumerate(params + fn.args.kwonlyargs):
                if cls is not None and i == 0 and not _static(fn):
                    values[a.arg] = _v(("self", cls))
                elif a.arg == "spec":
                    values[a.arg] = _v(("spec",))
                elif a.arg == "defaults":
                    values[a.arg] = _v(("entry", ()))
                else:
                    values[a.arg] = UNK
            for a in (fn.args.vararg, fn.args.kwarg):
                if a is not None:
                    values[a.arg] = UNK
            self.canonical[id(fn)] = self._scope_of(fn, module, cls, values)
        return self.canonical[id(fn)]

    # -- names -----------------------------------------------------------------------------------------------------
    def _lookup(self, name, scope, line):
        """The value of a name where it is read (line), or None where nothing binds it (a builtin)."""
        while scope is not None:
            entries = scope.binders.get(name)
            if entries:
                if scope.flow:
                    entries = [e for e in entries if e[0] < line]
                    tops = [i for i, e in enumerate(entries) if e[1]]
                    entries = entries[tops[-1]:] if tops else entries
                return _union(*(self._binder(b, scope) for _, _, b in entries))
            scope = scope.parent
        return None

    def _binder(self, binder, scope):
        kind = binder[0]
        if kind == "value":
            return binder[1]
        if kind == "expr":
            return self.ev(binder[1], scope)
        if kind == "unpack":
            value = self.ev(binder[1], scope)
            for i in binder[2]:
                value = _at(value, i)
            return value
        if kind == "iter":
            value = _union(*_members(self.ev(binder[1], scope))[0])
            for i in binder[2]:
                value = _at(value, i)
            return value
        if kind == "append":
            return _v(("seq", (self.ev(binder[1], scope),)))
        if kind == "func":
            return self._func(binder[1], scope, scope.cls)
        return _v(("class", binder[1].name)) if binder[1].name in self.classes else UNK

    def _member(self, item, name):
        """An attribute of self, a class or super(): what its class sets it to, or its method (bound)."""
        kind, cls = item
        start = self.classes[cls]["bases"] if kind == "super" else [cls]
        me = ("self", cls) if kind in ("self", "super") else None
        found, seen, todo = set(), set(), [c for c in start if c in self.classes]
        while todo:
            c = todo.pop(0)
            if c in seen:
                continue
            seen.add(c)
            info = self.classes[c]
            if name in info["methods"]:
                fn = info["methods"][name]
                found |= self._func(fn, info["module"], c, me if not _static(fn) else None)
            for binder, owner in info["attrs"].get(name, ()):
                if isinstance(owner, Scope):
                    found |= self._binder(binder, owner)
                else:
                    found |= self._binder(binder, self._free(owner, info["module"], c))
            todo += [b for b in info["bases"] if b in self.classes]
        return frozenset(found) or UNK

    # -- reads -----------------------------------------------------------------------------------------------------
    def _config(self, path):
        if path and path[-1] != "*":
            self.found["config"].add(".".join(path))

    def _read(self, base, key, default, subscript):
        """base[key] or base.get(key, default), recording the layout fields and config keys it reads."""
        keys = [item[1] for item in key if item[0] == "s" and _known(item[1])]
        ints = [item[1] for item in key if item[0] == "i"]
        unknown = any(not (item[0] == "s" and _known(item[1])) and item[0] != "i" for item in key)
        found = set()
        for item in base:
            kind = item[0]
            if kind in ("spec", "layout"):
                self.found["layout"] |= set(keys)
                found |= default if default is not None else UNK
            elif kind in ("config", "entry"):
                path = () if kind == "config" else item[1]
                for k in keys:
                    self._config(path + (k,))
                    found.add(("entry", path + (k,)))
                if unknown:
                    found.add(("entry", path + ("*",)))
                if ints:
                    found.add(("unk",))
                found |= default or EMPTY
            elif kind == "map":
                pairs, rest = item[1], item[2]
                if unknown:
                    found |= _union(*(value for _, value in pairs))
                    found |= self._read(rest, UNK, default, subscript) if rest else EMPTY
                for k in [("s", k) for k in keys] + [("i", n) for n in ints]:
                    hit = [value for names, value in pairs if k in names]
                    if hit:
                        found |= _union(*hit)
                    elif rest:
                        found |= self._read(rest, _v(k), default, subscript)
                    elif default is not None:
                        found |= default
                    elif not subscript:
                        found.add(("none",))
            elif kind in ("seq", "each"):
                for n in ints:
                    found |= _at(_v(item), n)
                if keys or unknown:
                    found |= _union(*_members(_v(item))[0])
            else:
                found.add(("unk",))
                found |= default or EMPTY
        return frozenset(found)

    def _tensors(self, value):
        for item in value:
            if item[0] == "s":
                name = tensor_name(item[1])
                if name:
                    self.found["tensors"].add(name)
            elif item[0] == "seq":
                for element in item[1]:
                    self._tensors(element)
            elif item[0] == "each":
                self._tensors(item[1])

    # -- evaluation ------------------------------------------------------------------------------------------------
    def ev(self, node, scope):
        """The values an expression may take (frozenset of items), recording what it reads."""
        key = (id(node), id(scope))
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = EMPTY   # a name read while it is being computed adds nothing new
        value = self._ev(node, scope)
        self.memo[key] = value
        return value

    def _ev(self, node, scope):
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return _v(("b", value))
            if isinstance(value, int):
                return _v(("i", value))
            if isinstance(value, str):
                return _v(("s", value))
            return _v(("none",)) if value is None else UNK
        if isinstance(node, ast.Name):
            found = self._lookup(node.id, scope, node.lineno)
            return UNK if found is None else found
        if isinstance(node, ast.JoinedStr):
            texts = {""}
            for part in node.values:
                options = {part.value} if isinstance(part, ast.Constant) else \
                    {_piece(item) for item in self.ev(part.value, scope)} or {OUT}
                texts = {a + b for a in texts for b in options}
                if len(texts) > LIMIT:
                    raise RuntimeError("an f-string takes too many values: the reader cannot follow it")
            return frozenset(_string(t) for t in texts)
        if isinstance(node, ast.BinOp):
            left, right = self.ev(node.left, scope), self.ev(node.right, scope)
            found = set()
            for a in left:
                for b in right:
                    if a[0] in ("i", "n") and b[0] in ("i", "n"):
                        found.add(("n",))
                    elif not isinstance(node.op, ast.Add):
                        found.add(("unk",))
                    elif a[0] == "seq" and b[0] == "seq":
                        found.add(("seq", a[1] + b[1]))
                    elif a[0] in ("seq", "each") and b[0] in ("seq", "each"):
                        found.add(("each", _union(*_members(_v(a))[0], *_members(_v(b))[0])))
                    elif a[0] in ("seq", "each", "map") or b[0] in ("seq", "each", "map"):
                        found.add(("unk",))
                    else:
                        found.add(_string(_piece(a) + _piece(b)))
            return _union(found)
        if isinstance(node, ast.UnaryOp):
            value = self.ev(node.operand, scope)
            if isinstance(node.op, ast.USub):
                return frozenset(("i", -item[1]) if item[0] == "i" else ("n",) for item in value)
            if isinstance(node.op, ast.Not):
                truth = _truth(value)
                return UNK if truth is None else _v(("b", not truth))
            return UNK
        if isinstance(node, ast.BoolOp):
            return _union(*(self.ev(v, scope) for v in node.values))
        if isinstance(node, ast.IfExp):
            truth = _truth(self.ev(node.test, scope))
            if truth is None:
                return _union(self.ev(node.body, scope), self.ev(node.orelse, scope))
            return self.ev(node.body if truth else node.orelse, scope)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return _v(("seq", tuple(self.ev(e.value if isinstance(e, ast.Starred) else e, scope) for e in node.elts)))
        if isinstance(node, ast.Dict):
            pairs, rest = [], set()
            for k, value in zip(node.keys, node.values):
                if k is None:
                    for item in self.ev(value, scope):
                        if item[0] == "map":
                            pairs += list(item[1])
                            rest |= item[2] or EMPTY
                        else:
                            rest.add(item)
                else:
                    pairs.append((self.ev(k, scope), self.ev(value, scope)))
            return _v(("map", tuple(pairs), frozenset(rest) or None))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            scopes, positional = self._comprehension(node, scope)
            if isinstance(node, ast.DictComp):
                return _v(("map", tuple((self.ev(node.key, s), self.ev(node.value, s)) for s in scopes), None))
            if positional:
                return _v(("seq", tuple(self.ev(node.elt, s) for s in scopes)))
            return _v(("each", _union(*(self.ev(node.elt, s) for s in scopes))))
        if isinstance(node, ast.Lambda):
            return self._func(node, scope, scope.cls)
        if isinstance(node, ast.Attribute):
            if node.attr == "config":
                return _v(("config",))
            if node.attr == "layout":
                return _v(("layout",))
            found = set()
            for item in self.ev(node.value, scope):
                if item[0] in ("self", "super", "class") and item[1] in self.classes:
                    found |= self._member(item, node.attr)
                else:
                    found.add(("unk",))
            return frozenset(found)
        if isinstance(node, ast.Subscript):
            base = self.ev(node.value, scope)
            if isinstance(node.slice, ast.Slice):
                bounds = [None if b is None else self.ev(b, scope) for b in
                          (node.slice.lower, node.slice.upper, node.slice.step)]
                bounds = [None if b is None else (next(iter(b))[1] if len(b) == 1 and next(iter(b))[0] == "i"
                                                  else "?") for b in bounds]
                found = set()
                for item in base:
                    if "?" in bounds:
                        found.add(item if item[0] in ("seq", "each") else ("unk",))
                    elif item[0] == "s":
                        found.add(_string(item[1][slice(*bounds)]))
                    elif item[0] == "seq":
                        found.add(("seq", item[1][slice(*bounds)]))
                    else:
                        found.add(item if item[0] == "each" else ("unk",))
                return frozenset(found)
            return self._read(base, self.ev(node.slice, scope), None, True)
        if isinstance(node, ast.Call):
            return self._call(node, scope)
        if isinstance(node, ast.NamedExpr):
            return self.ev(node.value, scope)
        return UNK

    def _comprehension(self, node, scope):
        """(the scopes its element is computed in, whether they follow the iterations in order): one scope an
        iteration where the loops run over known sequences, one for all of them otherwise."""
        key = (id(node), id(scope))
        if key not in self.comps:
            scopes, positional = [scope], True
            for gen in node.generators:
                following = []
                for outer in scopes:
                    members, finite = _members(self.ev(gen.iter, outer))
                    if not finite or len(members) > EXPAND:
                        members, positional = [_union(*members)], False
                    for member in members:
                        inner = Scope(outer, outer.cls)
                        self._bind_value(inner, gen.target, member)
                        following.append(inner)
                scopes = following
                positional = positional and not gen.ifs
            self.comps[key] = (scopes, positional and len(node.generators) == 1)
        return self.comps[key]

    def _bind_value(self, scope, target, value):
        if isinstance(target, ast.Name):
            scope.bind(target.id, ("value", value))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, t in enumerate(target.elts):
                self._bind_value(scope, t, _at(value, i))

    def _call(self, node, scope):
        f = node.func
        if isinstance(f, ast.Name):
            bound = self._lookup(f.id, scope, f.lineno)
            if bound is None:
                return self._builtin(f.id, node, scope)
            return self._invoke(bound, node, scope)
        if not isinstance(f, ast.Attribute):
            return UNK
        receiver = self.ev(f.value, scope)
        found = set()
        own = frozenset(item for item in receiver if item[0] in ("self", "super", "class") and item[1] in self.classes)
        if own:
            for item in own:
                found |= self._invoke(self._member(item, f.attr), node, scope)
        others = receiver - own
        if not others:
            return frozenset(found)
        args = [self.ev(a, scope) for a in node.args if not isinstance(a, ast.Starred)]
        if f.attr == "setting" and args:
            found |= self._read(_v(("config",)), args[0], None, True)
        elif f.attr == "get" and args:
            default = args[1] if len(args) > 1 else next(
                (self.ev(k.value, scope) for k in node.keywords if k.arg == "default"), None)
            found |= self._read(others, args[0], default, False)
        elif f.attr == "pop" and args:
            found |= self._read(others, args[0], args[1] if len(args) > 1 else None, True)
        elif f.attr in ("items", "values", "keys"):
            for item in others:
                if item[0] == "map":   # what a merged unknown dict adds: one more row, unknown
                    rows = [{"items": _v(("seq", (k, value))), "values": value, "keys": k}[f.attr]
                            for k, value in list(item[1]) + ([(UNK, UNK)] if item[2] else [])]
                    found.add(("seq", tuple(rows)))
                else:
                    found.add(("each", _v(("seq", (UNK, UNK))) if f.attr == "items" else UNK))
        else:
            if f.attr in self.helpers and any(("spec",) in a or any(i[0] == "map" and i[2] and ("spec",) in i[2]
                                                                    for i in a) for a in args):
                self.found["layout"] |= self.helpers[f.attr]
            found.add(("unk",))
        return frozenset(found)

    def _builtin(self, name, node, scope):
        args = [self.ev(a, scope) for a in node.args if not isinstance(a, ast.Starred)]
        if name == "range":
            return _v(("each", _v(("n",))))
        if name == "len" and args:   # a known string's (a list may grow as the code runs: a number)
            return frozenset(("i", len(item[1])) if item[0] == "s" and _known(item[1]) else ("n",)
                             for item in args[0])
        if name in PASS:
            return args[0] if args else UNK
        if name == "enumerate" and args:
            return _v(("each", _v(("seq", (_v(("n",)), _union(*_members(args[0])[0]))))))
        if name == "zip" and args:
            lists = [_members(a) for a in args]
            if all(finite for _, finite in lists) and len({len(m) for m, _ in lists}) == 1:
                return _v(("seq", tuple(_v(("seq", tuple(column))) for column in zip(*(m for m, _ in lists)))))
            return _v(("each", _v(("seq", tuple(_union(*m) for m, _ in lists)))))
        if name == "getattr" and len(args) >= 2:
            found = set()
            names = [item[1] for item in args[1] if item[0] == "s" and _known(item[1])]
            for item in args[0]:
                if item[0] in ("self", "class") and item[1] in self.classes and names:
                    for n in names:
                        found |= self._member(item, n)
                else:
                    found.add(("unk",))
            return frozenset(found) | (args[2] if len(args) > 2 else EMPTY)
        if name == "super":
            return _v(("super", scope.cls)) if scope.cls in self.classes else UNK
        return UNK

    def _invoke(self, value, node, scope):
        found = set()
        for item in value:
            if item[0] == "func":
                found |= self._inline(item[1], node, scope)
            else:
                found.add(("unk",))
        return frozenset(found)

    def _inline(self, fid, node, scope):
        """A call of a function of the technique's own files: its body read with the values given it, and what it
        returns."""
        fn, parent, cls, me = self.funcs[fid]
        params = fn.args.posonlyargs + fn.args.args
        values, offset = {}, 0
        if me is not None and params:
            values[params[0].arg], offset = _v(me), 1
        defaults = [None] * (len(params) - len(fn.args.defaults)) + list(fn.args.defaults)
        for i, a in enumerate(node.args):
            if isinstance(a, ast.Starred):
                break
            if offset + i < len(params):
                values[params[offset + i].arg] = self.ev(a, scope)
        names = {a.arg for a in params + fn.args.kwonlyargs}
        for k in node.keywords:
            if k.arg in names:
                values[k.arg] = self.ev(k.value, scope)
        for a, d in list(zip(params, defaults)) + list(zip(fn.args.kwonlyargs, fn.args.kw_defaults)):
            if a.arg not in values:
                values[a.arg] = self.ev(d, parent) if d is not None else UNK
        for a in (fn.args.vararg, fn.args.kwarg):
            if a is not None:
                values[a.arg] = UNK
        key = (fid, tuple(sorted(values.items(), key=lambda kv: kv[0])))
        if key in self.calls:
            return self.calls[key]
        if self.depth >= DEPTH:
            return UNK
        self.calls[key] = EMPTY
        callee = self._scope_of(fn, parent, cls, values)
        self.depth += 1
        try:
            result = self._scan(fn, callee)
        finally:
            self.depth -= 1
        self.calls[key] = result
        return result

    # -- the walk --------------------------------------------------------------------------------------------------
    def _scan(self, fn, scope):
        """Read a function's body in a scope: its sinks and reads recorded; the values it returns."""
        returns = []
        body = [fn.body] if isinstance(fn, ast.Lambda) else fn.body
        for statement in body:
            self._visit(statement, scope, fn, returns)
        if isinstance(fn, ast.Lambda):
            return self.ev(fn.body, scope)
        return _union(*returns) if returns else _v(("none",))

    def _visit(self, node, scope, fn, returns):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return   # its body is read where it is called
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            self._visit(node.generators[0].iter, scope, fn, returns)
            scopes, _ = self._comprehension(node, scope)
            parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
            parts += [n for g in node.generators[1:] for n in [g.iter]] + [i for g in node.generators for i in g.ifs]
            for inner in scopes:
                for part in parts:
                    self._visit(part, inner, fn, returns)
            return
        if isinstance(node, ast.Return) and node.value is not None:
            value = self.ev(node.value, scope)
            returns.append(value)
            if getattr(fn, "name", None) in RETURNS_NAMES:
                self._tensors(value)
        elif isinstance(node, ast.Call):
            self._sink_call(node, scope)
            self.ev(node, scope)
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "headers" and \
                    _is_checkpoint(node.value.value):
                self._tensors(self.ev(node.slice, scope))
            self.ev(node, scope)
        elif isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and _is_checkpoint(right):
                    self._tensors(self.ev(node.left, scope))
        for child in ast.iter_child_nodes(node):
            self._visit(child, scope, fn, returns)

    def _sink_call(self, node, scope):
        f, indices = node.func, ()
        if isinstance(f, ast.Name) and f.id in NAMED:
            bound = self._lookup(f.id, scope, f.lineno)
            if not (bound and any(item[0] == "func" for item in bound)):
                indices = (NAMED[f.id],)
        elif isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Attribute) and f.value.attr == "matrices":
                indices = MATRICES.get(f.attr, ())
            elif _is_checkpoint(f.value):
                indices = CHECKPOINT.get(f.attr, ())
        for i in indices:
            if i < len(node.args):
                self._tensors(self.ev(node.args[i], scope))

    def reads(self):
        """{tensors, config, layout}: sorted lists of what the code reads (module docstring)."""
        for tree, module in self.modules:
            referenced = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
                {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            roots = [(fn, None) for fn in tree.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))]
            roots += [(fn, cls) for cls, info in self.classes.items() if info["module"] is module
                      for fn in info["methods"].values()]
            for fn, cls in roots:
                if not _private(fn.name) or fn.name not in referenced:
                    self._scan(fn, self._free(fn, module, cls))
        found = dict(self.found)
        found["layout"] = found["layout"] - CORE_FIELDS
        return {kind: sorted(found[kind]) for kind in KINDS}


def code_reads(folder, helpers=None):
    """What a technique's code (FOLDER/code/*.py) reads: {tensors, config, layout}."""
    return Reader(sorted((Path(folder) / "code").glob("*.py")), helpers).reads()


def entries():
    return sorted(TECHNIQUES.glob("*/*/entry.json"))


def problems():
    """[problem]: entries whose reads differ from what their code reads."""
    found, helpers = [], core_helpers()
    for path in entries():
        entry = json.loads(path.read_text(encoding="utf-8"))
        name = path.relative_to(ROOT).as_posix()
        have = entry.get("reads")
        if not isinstance(have, dict) or sorted(have) != sorted(KINDS):
            found.append(f"{name}: reads is {{{', '.join(KINDS)}}}, each a sorted list")
            continue
        want = code_reads(path.parent, helpers)
        for kind in KINDS:
            listed = have[kind]
            if listed != sorted(set(listed)):
                found.append(f"{name}: reads.{kind} is sorted, without repeats")
            lacking, extra = sorted(set(want[kind]) - set(listed)), sorted(set(listed) - set(want[kind]))
            if lacking:
                found.append(f"{name}: reads.{kind} lacks {lacking}, which its code reads")
            if extra:
                found.append(f"{name}: reads.{kind} declares {extra}, which its code never reads")
    return found


SAMPLE = '''
from ..template.blocks import _tensor

NAMES = {"size": "sample_size", "width": "sample_width"}


class Sample:
    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        self.at = spec.get("inner", "inner.")
        self.size = m.setting(spec.get("size_key", "sample_count"))
        found = {key: m.setting(name) for key, name in {**NAMES, **(spec.get("config") or {})}.items()}
        rope = m.config.get("rope_parameters") or {}
        self.factor = rope.get("factor")
        self.heads, self.dim = m.kv_shape(layer, spec)
        self.w = self._mix(prefix + self.at, inject=False)
        self.u = self._mix(prefix + "outer.")
        if spec.get("kind") == "x" and found:
            self.v = _tensor(m, prefix + spec["named"] + "tail.weight")

    def _mix(self, p, inject=True):
        bases = [p + "down"] + ([p + "inject"] if inject else [])
        def get(name):
            return self.m.ck.get(p + name, "cpu")
        return [get(f"layers.{i}.{n}") for i in range(2) for n in ("a", "b")], self.m.matrices.load(bases, "x")
'''


class TechniqueReads(unittest.TestCase):
    def test_each_entry_declares_what_its_code_reads_and_no_more(self):
        found = problems()
        self.assertEqual(found, [], "\n".join(found) + "\n(python tests/test_technique_reads.py --write rewrites "
                                                        "each entry.json's reads from its code)")

    def test_the_reader_follows_defaults_tables_calls_and_closures(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "code").mkdir()
            (Path(folder) / "code" / "sample.py").write_text(SAMPLE, encoding="utf-8")
            found = code_reads(folder, {"kv_shape": {"head_dim", "kv_heads"}})
        self.assertEqual(found["tensors"], sorted([
            "inner.down", "outer.down", "outer.inject", "tail.weight",
            "inner.layers.N.a", "inner.layers.N.b", "outer.layers.N.a", "outer.layers.N.b"]))
        self.assertEqual(found["config"], ["rope_parameters", "rope_parameters.factor", "sample_count",
                                           "sample_size", "sample_width"])
        self.assertEqual(found["layout"], ["config", "head_dim", "inner", "kv_heads", "named", "size_key"])

    def test_a_name_starts_where_the_code_s_own_spelling_does(self):
        self.assertEqual(tensor_name(OUT + "layers.0.mlp." + NUM + ".up_proj"), "layers.N.mlp.N.up_proj")
        self.assertIsNone(tensor_name(OUT + ".weight"))
        self.assertIsNone(tensor_name("ple." + OUT))


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        helpers = core_helpers()
        for entry_path in entries():
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            entry["reads"] = code_reads(entry_path.parent, helpers)
            entry_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(entry_path.relative_to(ROOT).as_posix())
    else:
        unittest.main()
