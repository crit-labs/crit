"""A companion drafter: a draft model published in a folder of its own beside the checkpoint (its config.json names its
own model_type), which a model plugin's layout names as its draft block's source instead of a draft layer stored in
the checkpoint.

    "draft": {"block": DRAFTER, "companion": MODEL_TYPE, "names": [[pattern, replacement], ...]}

DRAFTER is a drafter technique (a model technique's DRAFTERS, the library's or the plugin's own); MODEL_TYPE the
companion's config.json `model_type`; `names` the rules that rename the companion's tensors to the names the drafter
reads, as a layout's `names` do the checkpoint's. The folder is the plugin's option `draft_folder` where it names one
(a run's `--drafter`), else the one folder near the checkpoint that holds such a companion and that the drafter
technique says fits the model (its `fits(config, shape)`): a folder beside the checkpoint's own (a sibling, or a
sibling's subfolder: the snapshot folders a download keeps), looked for first beside the checkpoint, then one level
up. Two that fit at the same level are refused, naming them: the option names the one meant.

The drafter technique reads the companion through `model.draft_companion` (Companion: its folder, its config, its
tensors' headers and reads under the names' rules) and states which of the target's state it reads (its taps, and in
its own words the KV it reads), as any drafter technique does; the models core loads it as the model's device draft
head through the same facts as a draft layer stored in the checkpoint (draft:head-device, drafter_kind,
draft_bytes_per_step, draft_width)."""
import json
from pathlib import Path


def _config(folder):
    path = Path(folder) / "config.json"
    if not path.is_file():
        return None
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


class Companion:
    """A companion drafter's folder: its config (`config`, and `text`: its text_config where it nests one), its
    tensors (`ck`: a Checkpoint of its safetensors under the layout's draft `names`) and their stored bytes."""

    def __init__(self, folder, names=None):
        self.folder = Path(folder)
        self.config = _config(self.folder) or {}
        self.text = self.config.get("text_config") or self.config
        self.names = names
        self._ck = None

    @property
    def ck(self):
        """The companion's tensors (opened at the first read: a dry resolve reads the config alone)."""
        if self._ck is None:
            from .checkpoint import Checkpoint
            self._ck = Checkpoint(self.folder, self.names)
        return self._ck

    def stored_bytes(self):
        """Every tensor's stored bytes."""
        return sum(self.ck.nbytes(name) for name in self.ck.names())

    def close(self):
        if self._ck is not None:
            self._ck.close()
            self._ck = None


def _folders(base):
    try:
        return sorted(p for p in Path(base).iterdir() if p.is_dir())
    except OSError:
        return []


def _near(checkpoint):
    """The folders to look in, nearest first: [[folders]] beside the checkpoint's folder (its siblings and their
    subfolders), then one level up; never the checkpoint's own folder or one inside it."""
    checkpoint = Path(checkpoint).resolve()
    levels = []
    for base in (checkpoint.parent, checkpoint.parent.parent):
        found = [folder for sibling in _folders(base) for folder in [sibling] + _folders(sibling)
                 if not folder.resolve().is_relative_to(checkpoint)]
        levels.append(found)
    return levels


def find(spec, checkpoint, given, drafter, shape):
    """The Companion the layout's draft names (`spec`: its draft entry), or None where none is found: the folder
    `given` (the option draft_folder; ValueError where it holds no companion of the model type, or one the drafter
    technique says does not fit), else the one near `checkpoint` (the model's folder, None where the load has none)
    that holds one and fits. drafter: the drafter technique's class, whose `fits(config, shape)` gives a problem (a
    line of text) or None."""
    model_type = spec["companion"]
    fits = getattr(drafter, "fits", None)

    def problem(config):
        if (config or {}).get("model_type") != model_type:
            return f"holds no {model_type} checkpoint (config.json model_type)"
        return fits(config, shape) if callable(fits) else None

    if given:
        folder = Path(given).expanduser()
        wrong = problem(_config(folder))
        if wrong:
            raise ValueError(f"model-template: the draft folder {folder} {wrong}")
        return Companion(folder, spec.get("names"))
    if checkpoint is None:
        return None
    for folders in _near(checkpoint):
        fit = [folder for folder in folders if not problem(_config(folder))]
        if len(fit) > 1:
            raise ValueError(f"model-template: {len(fit)} {model_type} drafters fit the model beside its checkpoint "
                             f"({', '.join(str(f) for f in fit)}): option draft_folder names the one to draft from")
        if fit:
            return Companion(fit[0], spec.get("names"))
    return None
