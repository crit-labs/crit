"""crit's commands as MCP tools, read from the commands' own argument parsers so the two never drift apart.

Each subcommand becomes one tool (`crit run plan` is `crit_run_plan`) whose input schema lists the subcommand's
options and those of the commands above it; a call turns the tool's arguments back into the command line."""
import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
import uuid

from crit.cli import FAMILIES as COMMANDS

# The command families, in the order tools/list shows them, and the module each one's parser comes from.
FAMILIES = ("run", "eval", "pull", "serve", "kernels", "models", "inference")
MODULES = {name: module for name, module, _ in COMMANDS} | {"inference": "crit.inference"}
# Options a call may leave out that the server fills from its own settings: --project for every family that has
# it, the run root for crit run, and crit serve's workspace (the folder of run-* directories, the same run root).
DEFAULTED = {"project": "project", ("run", "root"): "root", ("serve", "workspace"): "root"}

# What each tool does to the machine, as MCP tool annotations: a client runs a read-only tool without asking and
# confirms a destructive one first. Read-only tools only read files, records, logs or the device. Destructive
# tools end something or cannot be undone: a stopped run or job, a sealed or selected candidate, a recovered lease.
# Every other tool adds records or files (a plan, a build, an Eval result, a quantized checkpoint) and removes
# nothing. Open-world tools reach the internet (crit-hub, Hugging Face); the rest stay on this machine and its
# localhost. A tool this table does not name, such as a subcommand of a newer crit Eval, is taken as destructive and
# open-world, the protocol's own defaults, until it is named here.
READ_ONLY = {"crit_run_show", "crit_run_next", "crit_eval_context", "crit_eval_status", "crit_eval_journal", "crit_eval_compare",
             "crit_kernels_list", "crit_models_plan", "crit_models_verify", "crit_inference_list",
             "crit_inference_search", "crit_inference_show", "crit_inference_check"}
DESTRUCTIVE = {"crit_run_stop", "crit_run_seal", "crit_run_select", "crit_run_recover"}
ADDITIVE = {"crit_run_create", "crit_run_resolve", "crit_run_plan", "crit_run_check", "crit_run_benchmark",
            "crit_run_candidate", "crit_run_recon", "crit_run_capture", "crit_run_build", "crit_run_package",
            "crit_run_reserve", "crit_run_settle", "crit_run_measure", "crit_run_eval", "crit_run_serve",
            "crit_run_ablate", "crit_run_kernels", "crit_run_resume", "crit_run_cost", "crit_run_replay", "crit_eval_run",
            "crit_eval_precheck", "crit_eval_probe", "crit_eval_prepare", "crit_eval_reference", "crit_eval_fetch",
            "crit_eval_pack", "crit_eval_pin", "crit_eval_ab", "crit_eval_kernels", "crit_eval_workbench", "crit_pull",
            "crit_serve", "crit_kernels_new", "crit_kernels_build", "crit_kernels_sweep", "crit_models_ptq", "crit_models_drafter",
            "crit_models_eval"}
OPEN_WORLD = {"crit_pull", "crit_eval_fetch", "crit_eval_run", "crit_run_eval"}


def annotations(name, title):
    """NAME's MCP tool annotations, with TITLE."""
    known = name in READ_ONLY or name in DESTRUCTIVE or name in ADDITIVE
    return {"title": title, "readOnlyHint": name in READ_ONLY,
            "destructiveHint": name in DESTRUCTIVE or not known, "openWorldHint": name in OPEN_WORLD or not known}


class _Parsed(Exception):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser


def parser_of(main, *arguments):
    """The parser MAIN builds, caught as it starts parsing: no command runs and nothing prints."""
    original = argparse.ArgumentParser.parse_args

    def caught(self, *_, **__):
        raise _Parsed(self)

    argparse.ArgumentParser.parse_args = caught
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(*arguments)
    except _Parsed as parsed:
        return parsed.parser
    finally:
        argparse.ArgumentParser.parse_args = original
    raise ValueError(f"{main.__module__}.{main.__name__} returned without parsing its arguments")


def family_parser(name):
    """The parser of one command family; ImportError when its package is not installed."""
    if name not in FAMILIES:
        raise ValueError(f"Unknown command family {name!r}; known: {', '.join(FAMILIES)}")
    module = importlib.import_module(MODULES[name])
    # crit's own families expose parser(); crit Eval builds its parser inside main, caught as it starts parsing.
    return module.parser() if hasattr(module, "parser") else parser_of(module.main, [])


class Option:
    """One argparse action as a tool property."""

    def __init__(self, action, parser):
        self.action = action
        self.flag = next((s for s in action.option_strings if s.startswith("--")), None) or \
            (action.option_strings[0] if action.option_strings else None)
        self.name = self.flag.lstrip("-").replace("-", "_") if self.flag else action.dest
        self.positional = not action.option_strings
        self.switch = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                          argparse._StoreConstAction, argparse._AppendConstAction))
        self.counted = isinstance(action, argparse._CountAction)
        self.repeated = isinstance(action, argparse._AppendAction)
        self.many = action.nargs in ("+", "*", argparse.REMAINDER) or (isinstance(action.nargs, int) and action.nargs > 1)
        # nargs="?" with a const: the option alone (true) or with a value.
        self.bare = action.nargs == "?" and action.option_strings and action.const is not None
        # A JSON record read from a file: a call may give the object itself.
        self.record = self.name == "record" and action.type is Path
        self.required = action.required or (self.positional and action.nargs not in ("?", "*", argparse.REMAINDER))
        self.description = _help(action, parser)

    def schema(self):
        scalar = {int: "integer", float: "number"}.get(self.action.type, "string")
        item = {"type": scalar}
        if self.action.choices is not None and not isinstance(self.action.choices, dict):
            item["enum"] = list(self.action.choices)
        if self.switch:
            result = {"type": "boolean"}
        elif self.counted:
            result = {"type": "integer", "minimum": 0}
        elif self.record:
            result = {"type": ["object", "array", "string"]}
        elif self.repeated or self.many:
            result = {"type": "array", "items": item}
        elif self.bare:
            result = {"type": ["boolean", scalar]}
        else:
            result = item
        text = self.description
        if self.record:
            text = (text + ". " if text else "") + "A JSON object (written to a file for the command) or a JSON file's path"
        elif self.bare:
            text = (text + ". " if text else "") + f"true gives {self.flag} alone"
        if text:
            result["description"] = text
        default = self.action.default
        if not self.switch and default not in (None, argparse.SUPPRESS, []) and isinstance(default, (str, int, float)):
            result["default"] = default
        return result

    def arguments(self, value, scratch):
        """The command-line words for VALUE."""
        if value is None or value is False:
            return []
        if self.switch:
            if value is not True:
                raise ValueError(f"{self.name} is a switch: true or false")
            return [self.flag]
        if self.counted:
            return [self.flag] * int(value)
        if self.bare and value is True:
            return [self.flag]
        if self.record and isinstance(value, (dict, list)):
            scratch.mkdir(parents=True, exist_ok=True)
            path = scratch / f"record-{uuid.uuid4().hex}.json"
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
            value = str(path)
        values = value if (self.repeated or self.many) and isinstance(value, list) else [value]
        words = [json.dumps(v) if isinstance(v, (dict, list)) else _word(v, self.name) for v in values]
        if self.positional:
            return words
        if self.many:
            return [self.flag, *words]
        # --option=value keeps a value that starts with a dash a value.
        return [f"{self.flag}={word}" for word in words]


def _word(value, name):
    if value is True or isinstance(value, (dict, list)) or value is None:
        raise ValueError(f"{name} takes a value, not {json.dumps(value)}")
    return str(value)


def _help(action, parser):
    if not action.help or action.help == argparse.SUPPRESS:
        return ""
    try:
        return parser._get_formatter()._expand_help(action)
    except (KeyError, TypeError, ValueError):
        return action.help


def _actions(parser):
    """PARSER's own options and positionals, and its subcommands' action if it has one."""
    options, commands = [], None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            commands = action
        elif not isinstance(action, (argparse._HelpAction, argparse._VersionAction)) and action.help != argparse.SUPPRESS:
            options.append(Option(action, parser))
    return options, commands


class Tool:
    """One crit command line: its words before the options of each level, and those options."""

    def __init__(self, family, levels, description):
        self.family = family
        self.levels = levels  # [(word or None, [Option])]: the family's own level first, then each subcommand
        words = [family] + [word for word, _ in levels if word]
        self.command = "crit " + " ".join(words)
        self.name = "crit_" + "_".join(w.replace("-", "_") for w in words)
        self.description = description
        self.options = {}
        for word, options in levels:
            for option in options:
                if option.name in self.options:  # a subcommand's option named like one above it
                    option.name = f"{word}_{option.name}"
                self.options[option.name] = option
        # Options are shared with the other subcommands of a level, so what this one requires is its own.
        self.required = [name for name, option in self.options.items() if option.required]
        if family == "run" and levels[-1][0] != "create" and "id" in self.options:
            self.required.append("id")  # every action but create names an existing run

    def definition(self, wait):
        properties = {name: option.schema() for name, option in self.options.items()}
        properties["wait_seconds"] = {"type": "number", "minimum": 0, "default": wait, "description":
                                      "Seconds to wait for the command to end before returning its job to wait on "
                                      "with crit_job_wait (the command keeps running)"}
        schema = {"type": "object", "properties": properties, "additionalProperties": False}
        if self.required:
            schema["required"] = self.required
        return {"name": self.name, "title": self.command, "description": self.description,
                "inputSchema": schema, "annotations": annotations(self.name, self.command)}

    def argv(self, arguments, defaults, scratch):
        """The command line, without crit itself: the family, then each level's words and options."""
        unknown = sorted(set(arguments) - set(self.options))
        if unknown:
            raise ValueError(f"{self.command} has no option {', '.join(unknown)}; it takes "
                             f"{', '.join(self.options) or 'none'}")
        missing = [name for name in self.required if arguments.get(name) is None and not defaults.get(name)]
        if missing:
            raise ValueError(f"{self.command} needs {', '.join(missing)}")
        words = [self.family]
        for word, options in self.levels:
            if word:
                words.append(word)
            # Positionals first: an option taking an optional value cannot swallow them.
            for option in sorted(options, key=lambda o: not o.positional):
                value = arguments.get(option.name)
                if value is None:
                    value = defaults.get(option.name)
                words += option.arguments(value, scratch)
        return words


def tools(families=FAMILIES):
    """{name: Tool} for the families whose packages import, and {family: reason} for those that do not."""
    found, missing = {}, {}
    for family in families:
        try:
            parser = family_parser(family)
        except ImportError as error:
            missing[family] = str(error)
            continue
        for tool in _walk(family, parser, [], parser.description or ""):
            found[tool.name] = tool
    return found, missing


def _walk(family, parser, levels, description, word=None):
    options, commands = _actions(parser)
    levels = levels + [(word, options)]
    if commands is None:
        yield Tool(family, levels, " ".join((description or parser.description or "").split()))
        return
    helps = {choice.dest: choice.help for choice in commands._choices_actions}
    for name, subparser in commands.choices.items():
        yield from _walk(family, subparser, levels, helps.get(name) or subparser.description or "", name)


def defaults_for(tool, settings):
    """The server's settings for the options TOOL has: {option: value}."""
    result = {}
    for key, setting in DEFAULTED.items():
        family, name = key if isinstance(key, tuple) else (None, key)
        if name in tool.options and family in (None, tool.family) and settings.get(setting) is not None:
            result[name] = str(settings[setting])
    return result
