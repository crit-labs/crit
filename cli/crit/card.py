"""crit-hub engine cards as crit reads them: an engine folder's card.yaml and card.lock.json, for crit pull, crit hub
report and crit run create --from-engine. crit-hub's own tools write and check the cards; crit reads what it needs and
checks the source pin every reader relies on (an https repository and a full commit)."""
from pathlib import Path
import re

from crit.records import read_json

CARD = "card.yaml"
LOCK = "card.lock.json"
COMMIT = re.compile(r"[0-9a-f]{40}")
REPOSITORY = re.compile(r"https://[A-Za-z0-9.-]+(/[A-Za-z0-9_.~-]+)+/?")
# A folder in the engine's repository: relative, forward slashes, no segment that is empty, . or .. or starts with -.
SOURCE_PATH = re.compile(r"(?!-)(?!\.\.?(/|$))[A-Za-z0-9_.-]+(/(?!-)(?!\.\.?(/|$))[A-Za-z0-9_.-]+)*/?")
NAME = re.compile(r"[A-Za-z0-9_.-]+")
LOCK_BYTES = 16 << 20   # a lock lists every file of every checkpoint the engine was measured on


def yaml_module():
    """PyYAML, which reading a card needs (crit-hub's own tools use it)."""
    try:
        import yaml
    except ImportError:
        raise ValueError("Reading a crit-hub card needs PyYAML (python -m pip install pyyaml), which crit-hub's own tools use") from None
    return yaml


def read_card(path):
    """The card at `path` (an engine folder's card.yaml) as an object."""
    path = Path(path)
    card = yaml_module().safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        raise ValueError(f"{path} must hold an object")
    return card


def read_folder(folder):
    """An engine folder's card and lock, both objects."""
    folder = Path(folder)
    card_path, lock_path = folder / CARD, folder / LOCK
    if not card_path.is_file() or not lock_path.is_file():
        raise ValueError(f"{folder} is not a crit-hub engine folder: it needs {CARD} and {LOCK}")
    card, lock = read_card(card_path), read_json(lock_path, max_bytes=LOCK_BYTES)
    if not isinstance(lock, dict):
        raise ValueError(f"{lock_path} must hold an object")
    return card, lock


def check_source(card, where):
    """The card's source pin: an https repository and a full 40-character commit revision; `where` names the card in the
    error."""
    source = card.get("source")
    if (not isinstance(source, dict) or not REPOSITORY.fullmatch(str(source.get("repository", "")))
            or not COMMIT.fullmatch(str(source.get("revision", "")))):
        raise ValueError(f"{where}: source needs an https repository and a full 40-character commit revision")
    path = source.get("path")
    if path not in (None, "", ".") and not (isinstance(path, str) and SOURCE_PATH.fullmatch(path)):
        raise ValueError(f"{where}: source path {path!r} must be a folder in the repository, a relative path such as "
                         "engines/my-engine")
    return source
