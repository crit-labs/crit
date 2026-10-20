"""Named sealed engines from several runs, sharing one resident runtime."""
from pathlib import Path
import re

from crit.run.records import keys, read_json
from crit.run.store import ROOT, RunStore
from .engine import EngineError, ServedEngine


class ModelCatalog(ServedEngine):
    def __init__(self, path, *, project=ROOT, **options):
        path = Path(path).resolve()
        value = read_json(path)
        keys(value, {"models"}, {"default"})
        if not isinstance(value["models"], list) or not value["models"]:
            raise ValueError("Model catalog requires a nonempty models list")
        self.bindings = {}
        for row in value["models"]:
            keys(row, {"name", "workspace", "run"}, {"candidate"})
            name = row["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
                raise ValueError("Model names must be 1–128 letters, digits, dots, underscores or hyphens")
            if name in self.bindings:
                raise ValueError("Duplicate model name: " + name)
            if not isinstance(row["workspace"], str) or not row["workspace"]:
                raise ValueError("Model workspace must be a path")
            store = RunStore(path.parent / row["workspace"], project)
            view = store.get(row["run"])
            candidate = row.get("candidate") or view["state"]["selected_candidate_id"]
            if not candidate:
                raise ValueError("Model " + name + " needs a selected candidate or an explicit candidate")
            self.bindings[name] = (store, row["run"], candidate)
        default = value.get("default", next(iter(self.bindings)))
        if not isinstance(default, str) or default not in self.bindings:
            raise ValueError("Catalog default must name a configured model")
        store, run_id, _ = self.bindings[default]
        super().__init__(store.root, run_id, default, project=project, **options)

    def _verify(self, selector):
        if selector not in self.bindings:
            raise EngineError("Unknown model: " + str(selector), 404)
        store, run_id, candidate = self.bindings[selector]
        return super()._verify(candidate, store=store, run_id=run_id) | {"selector": selector}

    def candidates(self, refresh=False):
        return [{"candidate_id": name, "label": self.spec(name)["label"]} for name in self.bindings]

    def model_id(self, candidate_id=None):
        return candidate_id or self.default

    def resolve_model(self, model):
        if model is None:
            return self.default
        if model not in self.bindings:
            raise EngineError("Unknown model " + model + "; GET /v1/models lists the configured models", 404)
        return model
