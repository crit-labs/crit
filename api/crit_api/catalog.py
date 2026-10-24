"""Named sealed engines from several runs, sharing one resident runtime."""
from fnmatch import fnmatchcase
from pathlib import Path
import re

from crit.run.records import keys, read_json, text
from crit.run.store import ROOT, RunStore
from .engine import EngineError, ServedEngine

# Names clients send: Ollama tags (name:tag), gateway paths (vendor/name), Vertex versions (name@date); * is a wildcard.
ALIAS = r"[A-Za-z0-9*][A-Za-z0-9._:/@*-]{0,127}"


class ModelCatalog(ServedEngine):
    def __init__(self, path, *, project=ROOT, **options):
        path = Path(path).resolve()
        value = read_json(path)
        keys(value, {"models"}, {"default", "fallback"})
        if not isinstance(value["models"], list) or not value["models"]:
            raise ValueError("Model catalog requires a nonempty models list")
        self.bindings, self.aliases, self.display_names = {}, {}, {}
        for row in value["models"]:
            keys(row, {"name", "workspace", "run"}, {"candidate", "aliases", "display_name"})
            name = row["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
                raise ValueError("Model names must be 1–128 letters, digits, dots, underscores or hyphens")
            if name in self.bindings:
                raise ValueError("Duplicate model name: " + name)
            aliases = row.get("aliases", [])
            if not isinstance(aliases, list) or not all(isinstance(a, str) and re.fullmatch(ALIAS, a) for a in aliases):
                raise ValueError("Model aliases must be a list of 1–128 letters, digits, dots, underscores, hyphens, "
                                 "colons, slashes or @, with * matching any characters")
            for alias in aliases:
                if alias in self.aliases:
                    raise ValueError("Duplicate model alias: " + alias)
                self.aliases[alias] = name
            if "display_name" in row:
                self.display_names[name] = text(row["display_name"], "Model display_name", 160)
            if not isinstance(row["workspace"], str) or not row["workspace"]:
                raise ValueError("Model workspace must be a path")
            store = RunStore(path.parent / row["workspace"], project)
            view = store.get(row["run"])
            candidate = row.get("candidate") or view["state"]["selected_candidate_id"]
            if not candidate:
                raise ValueError("Model " + name + " needs a selected candidate or an explicit candidate")
            self.bindings[name] = (store, row["run"], candidate)
        clash = next((alias for alias in self.aliases if alias in self.bindings), None)
        if clash:
            raise ValueError("Model alias is also a model name: " + clash)
        default = value.get("default", next(iter(self.bindings)))
        if not isinstance(default, str) or default not in self.bindings:
            raise ValueError("Catalog default must name a configured model")
        if "fallback" in value and (not isinstance(value["fallback"], str) or value["fallback"] not in self.bindings):
            raise ValueError("Catalog fallback must name a configured model")
        self.fallback = value.get("fallback")
        store, run_id, _ = self.bindings[default]
        super().__init__(store.root, run_id, default, project=project, **options)

    def _verify(self, selector):
        if selector not in self.bindings:
            raise EngineError("Unknown model: " + str(selector), 404)
        store, run_id, candidate = self.bindings[selector]
        return super()._verify(candidate, store=store, run_id=run_id) | {"selector": selector}

    def candidates(self, refresh=False):
        return [{"candidate_id": name, "label": self.spec(name)["label"], "display_name": self.display_names.get(name)}
                for name in self.bindings]

    def model_id(self, candidate_id=None):
        return candidate_id or self.default

    def resolve_model(self, model):
        """A configured name, else the name of the alias it equals, else of the matching alias pattern with the most
        characters besides * (the earlier on a tie), else the fallback; without one an unknown name is a 404."""
        if model is None:
            return self.default
        if model in self.bindings:
            return model
        if model in self.aliases:
            return self.aliases[model]
        found = [(len(alias) - alias.count("*"), name) for alias, name in self.aliases.items()
                 if "*" in alias and fnmatchcase(model, alias)]
        if found:
            return max(found, key=lambda match: match[0])[1]
        if self.fallback is None:
            raise EngineError("Unknown model " + model + "; GET /v1/models lists the configured models", 404)
        return self.fallback
