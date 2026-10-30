"""crit serve --engine OWNER/NAME: an engine crit pull installed, served as a run's sealed candidate is."""
from crit.pull.store import InstallStore, engine_id
from crit.run.store import ROOT

from .engine import CAPABILITY, EngineError, ServedEngine, package_signature
from .options import serving_capabilities
from .residency import CAPABILITY as RESIDENT_CAPABILITY


class InstalledEngine(ServedEngine):
    """One installed engine (crit.pull.store) behind the API, under its <owner>/<name>; any requested model name
    reaches it, as a single run's default does."""
    def __init__(self, store, engine, *, project=ROOT, **options):
        self.installs = InstallStore(store)
        self.installs.get(engine_id(engine))  # an unknown engine fails before the runtime starts
        super().__init__(None, "installed", engine, project=project, **options)

    def selected(self):
        return self.requested_candidate

    def _verify(self, candidate_id, **_):
        if candidate_id != self.requested_candidate:
            raise EngineError("Unknown model: " + str(candidate_id), 404)
        spec = self.installs.spec(candidate_id)
        if not {CAPABILITY, RESIDENT_CAPABILITY} <= set(spec["capabilities"]):
            raise ValueError(f"{candidate_id}'s package declares no resident chat adapter ({CAPABILITY}, "
                             f"{RESIDENT_CAPABILITY}), so crit serve cannot load it")
        return spec | dict(capabilities=serving_capabilities(spec["capabilities"]),
                           signature=package_signature(spec["package_path"]), verify_model=self.verify_model)

    def candidates(self, refresh=False):
        return [{"candidate_id": self.requested_candidate, "label": self.spec(self.requested_candidate)["label"]}]

    def model_id(self, candidate_id=None):
        return self.requested_candidate

    def resolve_model(self, model):
        return self.requested_candidate
