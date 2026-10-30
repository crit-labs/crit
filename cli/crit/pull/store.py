"""Installed engines: crit-hub engines that `crit pull` built and installed, which `crit serve --engine OWNER/NAME`
serves. Runs stay the builder's record; installs are the user's.

A store (default PROJECT/engines) holds one folder per engine and the checkpoints the engines run:

    <owner>/<name>/install.json      what was installed: the card's identity, the pinned source, the checkpoint, the
                                     package's digest, the limits it serves at and the requirements check
    <owner>/<name>/card.json         the card as crit pull read it (from a crit-hub checkout or index.json)
    <owner>/<name>/card.lock.json    its lock: every checkpoint file and the sealed run's provenance
    <owner>/<name>/package/          the engine's source as built, with manifest.json (its files, launch, capabilities)
    <owner>/<name>/logs/build.log    the build's output
    models/<org>--<repo>@<revision>/ a checkpoint, shared by the engines that run it

No record holds an absolute path: every location is relative to the store, so a store moved to another folder or
machine still serves.
"""
from pathlib import Path
import re

from crit.run.records import digest, keys, number, read_json, safe_path, text, verify_inventory

ENGINE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
INSTALL = "install.json"
MANIFEST = "manifest.json"
# The adapter module of an engine exported from crit (crit-engines <name>/engine/adapter.py).
ENTRY = "adapter.py"


def engine_id(value):
    """An installed engine's name, <owner>/<name> as crit-hub names it."""
    if not isinstance(value, str) or not ENGINE.fullmatch(value) or any(p in (".", "..") for p in value.split("/")):
        raise ValueError(f"Name the engine as <owner>/<name>, as crit-hub does (crit-labs/nvfp4-gemv-int8kv), not {value!r}")
    return value


def checkpoint_folder(repository, revision):
    """A checkpoint's folder in the store, relative to it."""
    return f"models/{repository.replace('/', '--')}@{revision}"


class InstallStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().absolute()
        if self.root.resolve() != self.root:
            raise ValueError("The engine store must not traverse symlinks or reparse points")
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, engine):
        return safe_path(self.root, engine_id(engine))

    def get(self, engine):
        """The engine's install record."""
        path = self.folder(engine) / INSTALL
        if not path.is_file():
            installed = ", ".join(row["engine"] for row in self.list()) or "none"
            raise ValueError(f"{engine} is not installed in {self.root} (installed: {installed}); crit pull {engine} "
                             "installs it")
        value = read_json(path)
        keys(value, {"schema_version", "record_kind", "engine", "title", "installed_at", "card", "source", "checkpoint",
                     "package", "limits", "requirements", "build"}, {"crit_version"})
        if value["schema_version"] != 1 or value["record_kind"] != "engine_install" or value["engine"] != engine:
            raise ValueError(f"{path} is not {engine}'s install record")
        return value

    def list(self):
        """Every installed engine: its name, title, checkpoint and when it was installed."""
        rows = []
        for path in sorted(self.root.glob("*/*/" + INSTALL)):
            engine = f"{path.parent.parent.name}/{path.parent.name}"
            try:
                value = read_json(path)
                rows.append({"engine": engine, "title": value.get("title"), "installed_at": value.get("installed_at"),
                             "checkpoint": (value.get("checkpoint") or {}).get("model")})
            except ValueError as exc:
                rows.append({"engine": engine, "error": str(exc)})
        return rows

    def spec(self, engine):
        """What crit serve loads: the engine's verified package, its checkpoint and the limits it was installed at,
        in the form a run's sealed candidate takes (crit_api.engine.ServedEngine._verify)."""
        install = self.get(engine)
        folder = self.folder(engine)
        package = safe_path(folder, "package")
        manifest = read_json(package / MANIFEST)
        keys(manifest, {"schema_version", "record_kind", "engine", "files", "launch", "capabilities", "build",
                        "configuration_sha256"})
        if digest(manifest) != install["package"]["sha256"]:
            raise ValueError(f"{engine}'s package manifest is not the one it was installed with; crit pull it again")
        verify_inventory(package, manifest["files"], exclude={MANIFEST})
        configuration = read_json(package / "configuration.json")
        if digest(configuration) != manifest["configuration_sha256"]:
            raise ValueError(f"{engine}'s configuration.json changed since it was installed")
        checkpoint = install["checkpoint"]
        model = safe_path(self.root, checkpoint["folder"])
        if not model.is_dir():
            raise ValueError(f"{engine}'s checkpoint {checkpoint['repository']}@{checkpoint['revision']} is missing "
                             f"from {self.root / 'models'}; crit pull {engine} fetches it again")
        lock = read_json(folder / "card.lock.json")
        rows = lock["checkpoints"][checkpoint["model"]]["files"]
        if digest(rows) != checkpoint["files_sha256"]:
            raise ValueError(f"{engine}'s card.lock.json is not the one it was installed with")
        limits = install["limits"]
        keys(limits, {"kv_tokens", "vram_headroom_bytes", "ram_headroom_bytes"})
        number(limits["kv_tokens"], "installed context", 2, 1 << 24, integer=True)
        return dict(run_id="installed", candidate_id=engine, selector=engine, package_sha256=digest(manifest),
                    package_path=str(package), model_path=str(model), model_manifest=rows, configuration=configuration,
                    launch=manifest["launch"], capabilities=manifest["capabilities"], limits=dict(limits),
                    label=text(install["title"], "engine title", 400))
