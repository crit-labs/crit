"""A tuning swept on one GPU is not applied on a GPU of another compute capability where it is found implicitly (an
engine's tuning folder): the kernel builds with its default knobs, recorded as built with the reason. Without a GPU."""
import json

from crit_kernels import build as kbuild
from crit_plugins import kernels as pk

SWEPT = {"schema": kbuild.TUNING_SCHEMA, "kernel": "decode-gemv", "structures": {},
         "device": "NVIDIA GeForce RTX 5090", "capability": "12.0"}


def test_a_tuning_from_another_capability_is_foreign():
    reason = kbuild.foreign_tuning(SWEPT, capability="8.9")
    assert reason and "RTX 5090" in reason and "12.0" in reason and "8.9" in reason


def test_a_tuning_from_this_capability_or_without_one_is_taken():
    assert kbuild.foreign_tuning(SWEPT, capability="12.0") is None
    assert kbuild.foreign_tuning(dict(SWEPT, capability=None), capability="8.9") is None
    assert kbuild.foreign_tuning({k: v for k, v in SWEPT.items() if k != "capability"}, capability="8.9") is None


def test_no_device_to_ask_takes_the_tuning(monkeypatch):
    monkeypatch.setattr(kbuild, "device_capability", lambda: None)
    assert kbuild.foreign_tuning(SWEPT) is None


def _load(monkeypatch, tmp_path, capability):
    (tmp_path / "decode-gemv.json").write_text(json.dumps(SWEPT), encoding="utf-8")
    given = {}

    def load(name, tuning=None):
        given["tuning"] = tuning
        return object()

    monkeypatch.setattr(kbuild, "device_capability", lambda: capability)
    monkeypatch.setattr(kbuild, "load", load)
    monkeypatch.setattr(pk, "declared", lambda folder, name: ({"name": name, "fallback": "torch"}, "toy"))
    monkeypatch.setattr(kbuild, "_foreign_warned", set())
    pk.forget()
    pk.use_tuning(tmp_path)
    try:
        assert pk.load_kernel("decode-gemv", folder=tmp_path) is not None
        return given["tuning"], pk.loaded()["decode-gemv"]
    finally:
        pk.use_tuning(None)
        pk.forget()


def test_the_plugin_loader_builds_another_gpus_tuning_with_the_default_knobs(monkeypatch, tmp_path, capsys):
    tuning, record = _load(monkeypatch, tmp_path, "8.9")
    assert tuning is None
    assert record["source"] == "built" and "RTX 5090" in record["reason"] and "tuning" not in record
    assert "default knobs" in capsys.readouterr().err


def test_the_plugin_loader_applies_this_gpus_tuning(monkeypatch, tmp_path):
    tuning, record = _load(monkeypatch, tmp_path, "12.0")
    assert tuning["capability"] == "12.0"
    assert record["source"] == "tuned" and "reason" not in record
