"""crit's shared code is model-neutral: it names no model, reads no one checkpoint's spelling, and every model
technique is engine-tested on tiny random checkpoints in two checkpoint layouts.

Read from the source tree (nothing is imported or built), over crit's shared code: Inference (Core, the techniques,
the starters), Kernels, the models core and the model techniques, the CLI, the API, Eval, the MCP server, smoke and
tools. Every text file there counts (tracked and new, whatever its suffix or none: .py, .cu, .inc, .in, .toml, a
LICENSE), but Markdown: docs may name a model as an example or as the source of a measured number
(docs/CONTRIBUTION_CRITERIA.md); code may not, its comments included. A binary file holds no text to read. In a JSON
manifest the identifiers count (keys and string values without whitespace), not its prose.

Model names are data: they appear only in models/cache (the finished model plugins: their plugin.json, layout.json,
README and tests) and in test fixtures. Exempt are the files of any tests/ folder, and the modules of
crit_models.testing (the harness a model plugin's tests run on, an installed package) that no other shared code
imports: a module of it that a file outside tests/ imports, directly or through another module of the package (whose
__init__ runs with it), is runtime code and is read like any other.

Four rules:

1. No model name. A model family's name (FAMILIES, and the first word of every model_type a models/cache plugin
   serves) starts no word of a shared file, whatever follows it, in any case: `LlamaForCausalLM`, `deepseekv3`,
   `isQwen` and `ChatGLM` each name one. A word starts after anything but a letter, or at a capital after a small
   letter. SHAPED names are a family's only in their shape (`phi-3`, `Yi-34B`, `Falcon3`); NOT_FAMILIES are words that
   begin with a family's name and name none (`gptq`, `coherent`), listed one by one. A name no English word holds
   (ANYWHERE: `llama`, `qwen`, `deepseek`, ...) counts inside a word too (`codellama`), and ABBREVIATIONS are a family's
   short names (`dsv3`).
   The only exception is an identifier a published format or library defines and crit does not choose (a config
   value such as a rope type, a GGUF metadata value, a library's function name), listed per file in
   tests/support/model_neutral.json as {identifier: {"field": the field it is a value of, "in": [definitions],
   "as": "module."}}. It counts as listed only where it stands as the whole of a string literal ("llama3" or
   'llama3'), or, with "as", as that module's attribute (fi.gemma_rmsnorm); and in a Python file only inside the
   source of the definitions "in" names (an assignment to that name, a def or a class of it: the table or the
   function that reads the field). Anywhere else in the file, a comment or another comparison, the same word is a hit.
   A Python file's entry names its definitions; each entry must still occur as listed.
2. No checkpoint's own spelling. The tensor names a models/cache layout renames (the left side of its `names` rules:
   the checkpoint's spellings that the layout maps to the techniques' vocabulary) appear in no shared code file: a
   technique reads the names its entry declares, and a layout maps a checkpoint onto them.
3. Two checkpoint layouts. Each model plugin of models/cache declares in tests/checkpoints.json the checkpoint
   layouts its engine tests serve on tiny random checkpoints against the reference ({layout: ["path::Class", ...]}:
   safetensors under transformers' names, the same model written as GGUF, under another spelling that names rules map
   back), at least two, each named class defined in its file and listed once, one of them a
   crit_models.testing.Respelled class (a GGUF is read back under the techniques' names before they see it); and each model technique
   (models/techniques/CATEGORY/ID) is composed by a layout whose plugin does. A technique one model uses today is
   fine: what it must be is built so another model's layout can use it, which rules 1 and 2 and these tests hold.

4. A family's own code stays in its package. A finished model plugin may hold code only its family uses (a model
   technique of its own under techniques/ID, laid out as models/techniques' are; a tool format), built from crit's
   interfaces and engine-tested as a library technique is: its manifest's sources stay inside its folder or point into
   shared code, its code names no other family package, its techniques are its manifest's own sources, give names no
   library technique and no other family gives, and are composed by its own layout only. A piece a second family
   needs moves to models/techniques (or inference/techniques) under its general name in the same change. No shared
   file imports a family package: no shared manifest's source lies in models/cache, and no shared file names a
   models/cache package.

No debt and no allowlist of names: a shared file that names a model fails, wherever it is, and the fix is to move the
name to the model plugin's layout or a test fixture, or to describe the shape or kind instead (or, for a format's or
library's own identifier, to list it with the field it is a value of).
"""
import ast
import functools
import json
from pathlib import Path
import re
import subprocess
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tests/support/model_neutral.json"

# crit's shared code: every folder but the finished model plugins (models/cache), the docs and the records.
SHARED = ("inference", "kernels", "models/core", "models/techniques", "cli", "api", "eval", "mcp", "smoke", "tools")
# crit_models.testing: test fixtures that may name the models they build, while only tests import them.
PYTHON_ROOT = ROOT / "models/core/python"
FIXTURES = PYTHON_ROOT / "crit_models/testing"
# Model families beside the ones models/cache serves (their model_types' first words are added to these).
FAMILIES = ("baichuan", "cohere", "dbrx", "deepseek", "ernie", "exaone", "flash-next", "flashnext", "gemma",
            "glm", "gpt", "granite", "hermes", "hunyuan", "inkling", "internlm", "internvl", "jamba", "joyai", "kimi",
            "llama", "megrez", "minicpm", "minimax", "mistral", "mixtral", "nanbeige", "nemotron", "olmo", "olmoe",
            "qwen", "smollm", "stablelm", "starcoder")
# Families whose name is a family's only in one shape (a math symbol, a syllable, a bird otherwise): the pattern from
# the word's start, matched in any case but a (?-i:...) part.
SHAPED = {"falcon": r"falcon(?=[0-9_-]| [0-9]|(?-i:[A-Z]))", "phi": r"phi(?=[0-9-])", "yi": r"yi(?=[-_])"}
# Words that begin with a family's name and name no model: a quantization format, English words.
NOT_FAMILIES = ("coherence", "coherency", "coherent", "gptq")
# Names no English word or common identifier holds: they count anywhere inside a word (codellama, tinyllama).
ANYWHERE = ("baichuan", "deepseek", "exaone", "gemma", "hunyuan", "inkling", "internlm", "internvl", "llama",
            "minicpm", "mistral", "mixtral", "nanbeige", "nemotron", "qwen", "smollm", "stablelm", "starcoder")
# A family's short names, from a word's start in any case: dsv3, ds-v3.
ABBREVIATIONS = {"deepseek": r"ds-?v[0-9]"}
START = r"(?:(?<![A-Za-z])|(?<=[a-z])(?=[A-Z]))"   # a word's start: after a non-letter, or a capital after a small one
IDENTIFIER = r"(?<![A-Za-z0-9_-]){}(?![A-Za-z0-9_-])"


def relative(path):
    return path.relative_to(ROOT).as_posix()


def text_of(path):
    """A file's text to read (None for a binary file); of a JSON document its identifiers, one quoted a line."""
    data = path.read_bytes()
    if b"\0" in data:
        return None
    text = data.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    if path.suffix == ".json":
        try:
            return "\n".join(f'"{i}"' for i in json_identifiers(json.loads(text)))
        except ValueError:
            return text
    return text


def json_identifiers(value):
    """The keys of a JSON document and its string values without whitespace: its identifiers, not its prose."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from json_identifiers(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_identifiers(item)
    elif isinstance(value, str) and not re.search(r"\s", value):
        yield value


def _parse(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ast.parse(text)
        except (SyntaxError, ValueError):
            return None


def _fixture_imports(path, text):
    """The files of crit_models.testing a Python file's imports load (with the package's __init__)."""
    tree, targets, found = _parse(text), [], set()
    for node in ast.walk(tree) if tree else ():
        if isinstance(node, ast.Import):
            targets += [PYTHON_ROOT.joinpath(*alias.name.split(".")) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = path.parents[node.level - 1] if node.level else PYTHON_ROOT
            base = base.joinpath(*node.module.split(".")) if node.module else base
            targets += [base] + [base / alias.name for alias in node.names]
    for target in targets:
        for file in (target.with_suffix(".py"), target / "__init__.py"):
            if FIXTURES in file.parents and file.is_file():
                found |= {file, FIXTURES / "__init__.py"}
    return found


def _is_test(name):
    return "tests" in name.split("/")[:-1]


@functools.lru_cache(maxsize=None)
def _listed():
    """The files under the shared folders (tracked and new, not ignored), but Markdown."""
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *SHARED],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    return tuple(ROOT / name for name in sorted(set(listed))
                 if not name.endswith(".md") and (ROOT / name).is_file())


@functools.lru_cache(maxsize=None)
def exempt_fixtures():
    """The files of crit_models.testing that no shared file outside tests/ loads (through the package included)."""
    fixtures = {path for path in _listed() if FIXTURES in path.parents}
    runtime = [path for path in _listed() if path.suffix == ".py" and path not in fixtures
               and not _is_test(relative(path))]
    loaded = set().union(*(_fixture_imports(path, text_of(path)) for path in runtime))
    frontier = loaded
    while frontier:
        frontier = set().union(*(_fixture_imports(path, text_of(path)) for path in frontier)) - loaded
        loaded |= frontier
    return frozenset(fixtures - loaded)


@functools.lru_cache(maxsize=None)
def _shared():
    return tuple((path, text) for path in _listed()
                 if not _is_test(relative(path)) and path not in exempt_fixtures()
                 for text in [text_of(path)] if text is not None)


def shared_files():
    """The shared files the guard reads: every text file under the shared folders but Markdown, tests/ and the
    fixtures only tests load."""
    return [path for path, _ in _shared()]


def model_plugins():
    return sorted(p.parent for p in (ROOT / "models/cache").glob("*/plugin.json"))


def families():
    """FAMILIES and the first word of each model_type a models/cache plugin serves."""
    words = set(FAMILIES)
    for folder in model_plugins():
        manifest = json.loads((folder / "plugin.json").read_text(encoding="utf-8"))
        for need in manifest.get("needs", []):
            for fact in need if isinstance(need, list) else [need]:
                if fact.startswith("model_type:"):
                    words.add(re.match(r"[a-z]+", fact.split(":", 1)[1]).group(0))
    return sorted(words)


def family_pattern(words):
    """(regex, [word or None]): group i matches words[i] at a word's start whatever follows, in any case, a NOT_FAMILIES
    word (None) winning at its start, an ABBREVIATIONS pattern at a word's start, and an ANYWHERE name inside a word
    too; the longest first where one word holds another."""
    alternatives = [(None, re.escape(word)) for word in sorted(NOT_FAMILIES, key=len, reverse=True)]
    for word in sorted(set(words) | set(SHAPED), key=len, reverse=True):
        alternatives.append((word, SHAPED.get(word, re.escape(word))))
    alternatives += [(word, pattern) for word, pattern in sorted(ABBREVIATIONS.items())]
    inside = [(word, re.escape(word)) for word in sorted(ANYWHERE, key=len, reverse=True)]
    regex = (START + "(?i:" + "|".join(f"({pattern})" for _, pattern in alternatives) + ")|(?i:" +
             "|".join(f"({pattern})" for _, pattern in inside) + ")")
    return re.compile(regex), [word for word, _ in alternatives + inside]


def _segments(pattern):
    """The literal dotted parts of a names rule's side: groups, anchors and back-references dropped."""
    pattern = re.sub(r"\(\?[!=<][^)]*\)", "", pattern)
    pattern = re.sub(r"\\\d", "", pattern)
    pattern = re.sub(r"\([^)]*\)", "", pattern.replace("\\.", ".").replace("^", "").replace("$", ""))
    return {part.strip("_") for part in pattern.split(".") if part.strip("_")}


def _gguf_spellings():
    """GGUF's own tensor parts (the models core's table of the convention): a published format's names, which a
    checkpoint may happen to share."""
    tree = ast.parse((ROOT / "models/core/model-template/code/gguf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "COMMON" for t in node.targets):
            return set(ast.literal_eval(node.value))
    return set()


def checkpoint_spellings():
    """The tensor-name parts a models/cache layout renames and no layout renames to: one checkpoint's own spellings
    (with an underscore and at least five characters, so a plain word such as `attn` or `embed` is not one)."""
    raw, canonical = set(), set()
    for folder in model_plugins():
        layout = folder / "layout.json"
        if layout.is_file():
            for rule in json.loads(layout.read_text(encoding="utf-8")).get("names", []):
                raw |= _segments(rule[0])
                canonical |= _segments(rule[1])
    return sorted(s for s in raw - canonical - _gguf_spellings() if "_" in s and len(s) >= 5)


def definitions(text):
    """{name: [(start, end)]}: the source spans (whole lines) of a Python file's assignments to a name, its defs and
    its classes, at any depth."""
    tree = _parse(text)
    if tree is None:
        return {}
    starts = [0] + [m.end() for m in re.finditer("\n", text)] + [len(text)]
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [getattr(t, "id", getattr(t, "attr", None)) for t in targets]
        else:
            continue
        for name in filter(None, names):
            found.setdefault(name, []).append((starts[node.lineno - 1], starts[min(node.end_lineno, len(starts) - 1)]))
    return found


def listed_spans(path, text, entries):
    """{identifier: [(start, end)]}: where a file's listed identifiers stand as listed (the whole of a string literal,
    or the "as" module's attribute; in a Python file inside the definitions "in" names)."""
    if not entries:
        return {}
    scopes = definitions(text) if path.suffix == ".py" else None
    found = {}
    for identifier, entry in entries.items():
        entry = entry if isinstance(entry, dict) else {}
        if scopes is None:
            within = [(0, len(text))]
        else:
            within = [span for name in entry.get("in", []) for span in scopes.get(name, [])]
        if "as" in entry:
            form = r"(?<![\w.]){}({})(?!\w)".format(re.escape(entry["as"]), re.escape(identifier))
        else:
            form = r"([\"'])(?:{})\1".format(re.escape(identifier))
        spans = []
        for match in re.finditer(form, text):
            span = match.span(1) if "as" in entry else (match.start() + 1, match.end() - 1)
            if any(start <= span[0] and span[1] <= end for start, end in within):
                spans.append(span)
        found[identifier] = spans
    return found


def _counts(regex, words, masks):
    """{word: {file: occurrences}} of each word in the shared code, a file's listed identifiers (as listed) aside."""
    found = {}
    for path, text in _shared():
        allowed = [s for spans in listed_spans(path, text, masks.get(relative(path), {})).values() for s in spans]
        for match in regex.finditer(text):
            word = words[match.lastindex - 1]
            if word is None or any(start <= match.start() < end for start, end in allowed):
                continue
            counts = found.setdefault(word, {})
            counts[relative(path)] = counts.get(relative(path), 0) + 1
    return found


def _layout_blocks(layout):
    """The block, residual and drafter names a layout composes (auto FFN: the core's moe and swiglu)."""
    names = {spec.get("block") for spec in layout.get("mixers", {}).values()}
    ffn = layout.get("ffn", {"block": "auto"})
    names |= {ffn.get("block"), ffn.get("routed"), ffn.get("dense")}
    if ffn.get("block") == "auto":
        names |= {ffn.get("routed", "moe"), ffn.get("dense", "swiglu")}
    names.add(layout.get("residual", {}).get("block", "pre-norm"))
    names.add(layout.get("draft", {}).get("block"))
    names |= {spec.get("block") for spec in layout.get("before", [])}
    return names - {None, "auto"}


CHECKPOINTS = "tests/checkpoints.json"   # a model plugin's checkpoint layouts and the engine tests that serve them


def checkpoint_layouts(folder):
    """({layout: [the test classes found]}, [problems]) of a model plugin's tests/checkpoints.json."""
    path = folder / CHECKPOINTS
    if not path.is_file():
        return {}, [f"{relative(folder)}: no {CHECKPOINTS} (the checkpoint layouts its engine tests serve)"]
    data, found, problems = json.loads(path.read_text(encoding="utf-8")), {}, []
    if data.get("schema") != "crit-model-checkpoints-v1" or not isinstance(data.get("layouts"), dict):
        return {}, [f"{relative(path)}: schema crit-model-checkpoints-v1 with layouts {{layout: [path::Class]}}"]
    respelled, listed = False, set()
    for layout, tests in data["layouts"].items():
        for test in tests:
            file, _, name = test.partition("::")
            source = ROOT / file
            text = source.read_text(encoding="utf-8") if source.is_file() else ""
            defined = re.search(rf"^class {re.escape(name)}\((.*?)\):", text, re.M | re.S) if name else None
            if defined is None:
                problems.append(f"{relative(path)}: {layout}: {test} is no test class")
            elif test in listed:
                problems.append(f"{relative(path)}: {test} serves two layouts")
            else:
                listed.add(test)
                found.setdefault(layout, []).append(test)
                respelled = respelled or "Respelled" in defined.group(1)
    if len(found) < 2:
        problems.append(f"{relative(path)}: its engine tests serve {len(found)} checkpoint layout(s), not two")
    if found and not respelled:
        # A GGUF is read back under the techniques' names before any technique sees it: only a spelling the layout's
        # names rules map back shows that the techniques read through the layout.
        problems.append(f"{relative(path)}: no layout is a crit_models.testing.Respelled class (a spelling the names "
                        "rules map back)")
    return found, problems


def untested_techniques():
    """[problems]: model plugins whose checkpoint layouts are short, and techniques no such plugin's layout composes."""
    problems, tested = [], set()
    for folder in model_plugins():
        layout = folder / "layout.json"
        if not layout.is_file():
            continue
        found, wrong = checkpoint_layouts(folder)
        problems += wrong
        if len(found) >= 2:
            tested |= _layout_blocks(json.loads(layout.read_text(encoding="utf-8")))
    for entry in sorted((ROOT / "models/techniques").glob("*/*/entry.json")) + sorted(
            (ROOT / "models/cache").glob("*/" + FAMILY_TECHNIQUES)):
        technique = json.loads(entry.read_text(encoding="utf-8"))
        provided = {name for names in technique.get("provides", {}).values() for name in names}
        if not provided & tested:
            problems.append(f"{technique['id']}: no model plugin's layout composes it with engine tests in two "
                            "checkpoint layouts")
    return problems


# A family's own pieces (models/cache/README.md): code in its finished model plugin that only that family uses.
FAMILY_TECHNIQUES = "techniques/*/entry.json"   # a family's techniques, laid out as models/techniques' are
CODE_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".h", ".inc", ".in")


def _plugin_folders(root):
    return sorted(p.parent for p in (Path(root) / "models/cache").glob("*/plugin.json"))


def _technique_names(entry):
    return {name for names in entry.get("provides", {}).values() for name in names}


def _own_code(folder):
    """A family package's code files: every file under it but its tests, Markdown and JSON."""
    return [path for path in sorted(Path(folder).rglob("*")) if path.is_file() and path.suffix in CODE_SUFFIXES
            and "tests" not in path.relative_to(folder).parts[:-1] and "__pycache__" not in path.parts]


def _package_pattern(ids):
    """A package id as written (gemma4-model) or as a module name (gemma4_model), as a whole word."""
    return re.compile("|".join(rf"(?<![\w-]){re.escape(i)}(?![\w-])|(?<!\w){re.escape(i.replace('-', '_'))}(?!\w)"
                               for i in sorted(ids, key=len, reverse=True)) or r"(?!)")


def family_problems(root=ROOT):
    """[problems] of the rule that keeps a family's own code in its finished model plugin (models/cache/<family>): its
    manifest's sources resolve inside its own folder or into shared code, never into another family's package; its
    code names no other family package; each of its techniques (techniques/ID/entry.json) is mounted by its own
    manifest from its own folder, gives names no library technique and no other family gives, and is composed by its
    own layout and by no other family's: a piece a second family needs moves to models/techniques (or
    inference/techniques) under its general name, in the same change."""
    root = Path(root)
    cache = (root / "models/cache").resolve()
    folders = _plugin_folders(root)
    library = {}
    for path in sorted((root / "models/techniques").glob("*/*/entry.json")):
        for name in _technique_names(json.loads(path.read_text(encoding="utf-8"))):
            library[name] = path.parent.relative_to(root).as_posix()
    problems, given, layouts = [], {}, {}
    for folder in folders:
        where = folder.relative_to(root).as_posix()
        manifest = json.loads((folder / "plugin.json").read_text(encoding="utf-8"))
        layout = folder / "layout.json"
        layouts[folder.name] = (_layout_blocks(json.loads(layout.read_text(encoding="utf-8")))
                                if layout.is_file() else set())
        sources = manifest.get("sources", {})
        for key, source in sources.items():
            target = (folder / source).resolve()
            if target.is_relative_to(cache) and not target.is_relative_to(folder.resolve()):
                problems.append(f"{where}: source {key} is {source}, code of another family's package; a piece two "
                                "families use is shared code (models/techniques, inference/techniques)")
        others = _package_pattern([f.name for f in folders if f != folder])
        for path in _own_code(folder):
            for match in others.finditer(path.read_text(encoding="utf-8", errors="ignore")):
                problems.append(f"{path.relative_to(root).as_posix()} names the package {match.group(0)}: a family's "
                                "code uses its own package and shared code only")
        for entry_path in sorted(folder.glob(FAMILY_TECHNIQUES)):
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            name = entry_path.relative_to(root).as_posix()
            key, code = entry.get("source"), entry_path.parent / "code"
            if key not in sources or (folder / sources[key]).resolve() != code.resolve():
                problems.append(f"{name}: the package's manifest mounts it as its source {key!r} "
                                f"(sources: {{{key}: {code.relative_to(folder).as_posix()}}})")
            for technique in sorted(_technique_names(entry)):
                if technique in library:
                    problems.append(f"{name}: {technique} is the library technique {library[technique]}'s name")
                if technique in given:
                    problems.append(f"{name}: {technique} is given by {given[technique]} too")
                given[technique] = name
            if not _technique_names(entry) & layouts[folder.name]:
                problems.append(f"{name}: its package's layout composes none of what it provides (a family's piece "
                                "serves its family)")
    for plugin, blocks in sorted(layouts.items()):
        for technique in sorted(blocks):
            owner = given.get(technique)
            if owner is not None and not owner.startswith(f"models/cache/{plugin}/"):
                problems.append(f"models/cache/{plugin}: its layout composes {technique}, which {owner} gives: a piece "
                                "a second family needs moves to models/techniques under its general name")
    return problems


def shared_family_imports():
    """[problems]: shared code that imports a family package: a shared manifest's source in models/cache, or a shared
    file naming a models/cache package (by its id, or its id as a module name)."""
    cache = (ROOT / "models/cache").resolve()
    pattern = _package_pattern([folder.name for folder in model_plugins()])
    problems = []
    for path, text in _shared():
        if path.name == "plugin.json":
            for key, source in json.loads(path.read_text(encoding="utf-8")).get("sources", {}).items():
                if (path.parent / source).resolve().is_relative_to(cache):
                    problems.append(f"{relative(path)}: source {key} is a family package's code ({source})")
        for match in pattern.finditer(text):
            problems.append(f"{relative(path)} names the family package {match.group(0)}")
    return problems


def measured(identifiers):
    spellings = checkpoint_spellings()
    spelled = re.compile("|".join(IDENTIFIER.format(f"({re.escape(w)})")
                                  for w in sorted(spellings, key=len, reverse=True)) or r"(?!)")
    return {"families": _counts(*family_pattern(families()), identifiers),
            "spellings": _counts(spelled, sorted(spellings, key=len, reverse=True), identifiers)}


def allowlist():
    return json.loads(ALLOWLIST.read_text(encoding="utf-8"))


class ModelNeutral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.allowed = allowlist()
        cls.found = measured(cls.allowed)

    def _named(self, kind, what):
        return [f"{path} names {word!r} {count} times: shared code names no {what}; put it in the model plugin's "
                "layout (models/cache) or a test fixture, or describe the shape or kind"
                for word, files in sorted(self.found[kind].items()) for path, count in sorted(files.items())]

    def test_shared_code_names_no_model(self):
        self.assertEqual(self._named("families", "model family or model type"), [])

    def test_shared_code_reads_no_checkpoints_own_spelling(self):
        self.assertTrue(checkpoint_spellings() or not any(
            json.loads((f / "layout.json").read_text(encoding="utf-8")).get("names")
            for f in model_plugins() if (f / "layout.json").is_file()))
        self.assertEqual(self._named("spellings", "tensor spelling of one checkpoint"), [])

    def test_each_model_technique_is_engine_tested_in_two_checkpoint_layouts(self):
        self.assertEqual(untested_techniques(), [])

    def test_a_familys_own_code_stays_in_its_package(self):
        self.assertEqual(family_problems(), [])

    def test_no_shared_code_imports_a_family_package(self):
        self.assertEqual(shared_family_imports(), [])

    def test_the_family_rule_sees_a_piece_out_of_its_package(self):
        # Two family packages: one mounts the other's code and names it, its technique is not mounted by its
        # manifest and takes a library technique's name, and the other family's layout composes it.
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models/techniques/mixers/attention").mkdir(parents=True)
            (root / "models/techniques/mixers/attention/entry.json").write_text(
                json.dumps({"provides": {"blocks": ["attention"]}}), encoding="utf-8")
            one, two = root / "models/cache/one-model", root / "models/cache/two-model"
            (one / "techniques/own/code").mkdir(parents=True)
            (two / "code").mkdir(parents=True)
            files = {one / "techniques/own/entry.json": {"source": "own",
                                                         "provides": {"blocks": ["own-block", "attention"]}},
                     one / "plugin.json": {"sources": {"theirs": "../two-model/code"}},
                     one / "layout.json": {"mixers": {"full_attention": {"block": "own-block"}}},
                     two / "plugin.json": {"sources": {}},
                     two / "layout.json": {"mixers": {"full_attention": {"block": "own-block"}}}}
            for path, data in files.items():
                path.write_text(json.dumps(data), encoding="utf-8")
            (one / "techniques/own/code/own.py").write_text("from two_model import x\n", encoding="utf-8")
            found = "\n".join(family_problems(root))
        for wanted in ("source theirs is ../two-model/code", "names the package two_model",
                       "mounts it as its source 'own'", "attention is the library technique",
                       "two-model: its layout composes own-block"):
            self.assertIn(wanted, found)

    def test_each_identifier_is_a_listed_formats_value_that_still_occurs(self):
        problems = []
        for path, identifiers in sorted(self.allowed.items()):
            file = ROOT / path
            text = (text_of(file) if file.is_file() else "") or ""
            scopes = definitions(text) if file.suffix == ".py" else {}
            spans = listed_spans(file, text, identifiers)
            for identifier, entry in sorted(identifiers.items()):
                entry = entry if isinstance(entry, dict) else {}
                if not entry.get("field"):
                    problems.append(f"{path}: {identifier!r} names no format field it is a value of")
                if set(entry) - {"field", "in", "as"}:
                    problems.append(f"{path}: {identifier!r} holds keys beside field, in and as")
                if file.suffix == ".py" and not entry.get("in"):
                    problems.append(f"{path}: {identifier!r} names no definition it stands in (in)")
                if file.suffix != ".py" and "in" in entry:
                    problems.append(f"{path}: {identifier!r}: only a Python file's entry names definitions")
                for name in entry.get("in", []):
                    if name not in scopes:
                        problems.append(f"{path}: {identifier!r}: no definition {name!r} in the file")
                if not spans[identifier]:
                    problems.append(f"{path} no longer holds {identifier!r} as listed: remove the entry")
        self.assertEqual(problems, [])

    def test_the_family_rule_sees_a_name_whatever_follows(self):
        regex, words = family_pattern(families())

        def named(text):
            return [words[m.lastindex - 1] for m in regex.finditer(text) if words[m.lastindex - 1]]
        for text, word in (("LlamaForCausalLM", "llama"), ("MistralForCausalLM", "mistral"),
                           ("deepseekv3", "deepseek"), ("isQwen", "qwen"), ("ChatGLM", "glm"), ("is_qwen", "qwen"),
                           ("NemotronH_Omni", "nemotron"), ("InternVL", "internvl"), ("Phi-3", "phi"),
                           ("phi4", "phi"), ("Yi-34B", "yi"), ("yi_6b", "yi"), ("OLMoE", "olmoe"),
                           ("CohereForAI", "cohere"), ("falcon-7b", "falcon"), ("FalconMamba", "falcon"),
                           ("Falcon 40B", "falcon"), ("codellama", "llama"), ("TinyLlama", "llama"),
                           ("dsv3", "deepseek"), ("DS-V3", "deepseek"), ("myqwen2", "qwen")):
            self.assertEqual(named(text), [word], text)
        for text in ("gptq", "GPTQ_TILE", "coherent", "phi", "phase", "yield", "mayi_", "biglm", "skimit",
                     "an eagle and a falcon"):
            self.assertEqual(named(text), [], text)

    def test_the_guard_sees_the_shared_code(self):
        files = {relative(f) for f in shared_files()}
        self.assertIn("models/core/model-template/code/model.py", files)
        for suffix in (".inc", ".in", ".toml", "LICENSE"):
            self.assertTrue(any(f.endswith(suffix) for f in files), suffix)
        self.assertFalse(any(f.startswith("models/cache/") or "/tests/" in f or f.endswith(".md") for f in files))
        self.assertIn("inkling", families())
        # crit_models.testing is exempt while only tests load it
        self.assertIn(FIXTURES / "__init__.py", exempt_fixtures())
        self.assertEqual(_fixture_imports(ROOT / "cli/x.py", "from crit_models.testing import gguf_checkpoint"),
                         {FIXTURES / "__init__.py", FIXTURES / "gguf_checkpoint.py"})


if __name__ == "__main__":
    unittest.main()
