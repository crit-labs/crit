"""Host-side stores a plugin builds from the checkpoint (a large MoE's experts packed in host memory, say), kept between
the engines one process builds for one model, as crit serve's model pool keeps the model's mapped files and device
weights.

A plugin asks for a store by name and by the key of everything its bytes depend on (the checkpoint, the layout, the
code that packs it) through its context (crit_core.engine.LoadContext.keep): the store kept under that name with that
key comes back as it is, and only a store not kept yet is built. Kept stores belong to the keeper, never to the
engine: the engine's close leaves them, and the keeper closes them when its model unloads (crit_api.models.ModelPool)
or when told to (crit_core.engine.eval_release). One store a name: a store asked under a name with another key, by an
engine built with other options, replaces the one kept there, which is closed first, so two versions of a store that
fills host memory are never held at once.
"""
import hashlib
import json


def key_digest(key):
    """A store's key as its sha256: any JSON value, objects compared by content, other values by their text."""
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


class Kept:
    """The stores kept for one model: name -> {key, value, close, nbytes, users, builds}."""

    def __init__(self):
        self.entries = {}

    def keep(self, name, key, build, close=None, nbytes=None):
        """The store kept as `name` under `key`, or build() it and keep it (close(value) frees it later; nbytes, its
        host bytes, is reported). A kept store under another key is closed first; one an engine still holds is not
        replaced (ValueError). Each call is one user until release(name)."""
        name, digest = str(name), key_digest(key)
        entry = self.entries.get(name)
        if entry is not None and entry["key"] == digest:
            entry["users"] += 1
            entry["reused"] += 1
            return entry["value"]
        if entry is not None:
            if entry["users"]:
                raise ValueError(f"the kept store {name} is held by an engine under another key: two engines of one "
                                 "process ask the same store of different contents")
            self._close(name)
        value = build()
        self.entries[name] = {"key": digest, "value": value, "close": close, "users": 1, "reused": 0,
                              "nbytes": None if nbytes is None else int(nbytes)}
        return value

    def release(self, name):
        """An engine that asked for `name` is closed: the store stays kept."""
        entry = self.entries.get(str(name))
        if entry is not None and entry["users"] > 0:
            entry["users"] -= 1

    def _close(self, name):
        entry = self.entries.pop(name)
        if entry["close"] is not None:
            entry["close"](entry["value"])

    def nbytes(self, name):
        """The host bytes of the store kept as `name` that no engine holds (0 for none): memory the next engine's
        plugin has, whether it gets the store back or replaces it."""
        entry = self.entries.get(str(name))
        return entry["nbytes"] or 0 if entry is not None and not entry["users"] else 0

    def drop(self, name):
        """Close the store kept as `name` where no engine holds it (an engine that will not ask for it frees its
        memory first)."""
        entry = self.entries.get(str(name))
        if entry is not None and not entry["users"]:
            self._close(str(name))

    def held(self):
        """The names of the kept stores an engine holds."""
        return sorted(name for name, entry in self.entries.items() if entry["users"])

    def close(self):
        """Close every kept store. Refused while an engine holds one."""
        held = self.held()
        if held:
            raise ValueError("cannot free kept stores an engine still holds: " + ", ".join(held))
        failure = None
        for name in list(self.entries):
            try:
                self._close(name)
            except Exception as error:   # the others are closed all the same
                failure = failure or error
        if failure is not None:
            raise failure

    @property
    def host_bytes(self):
        return sum(entry["nbytes"] or 0 for entry in self.entries.values())

    def report(self):
        """name -> {nbytes, users, reused}: what is kept, by whom, and how often a build was saved."""
        return {name: {key: entry[key] for key in ("nbytes", "users", "reused")}
                for name, entry in self.entries.items()}
