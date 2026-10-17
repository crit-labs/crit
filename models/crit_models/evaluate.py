"""Run a transformed checkpoint in an engine through Eval, against the source model's prepared data and reference."""
from pathlib import Path

from crit.run.records import file_sha, now, read_json, write_json

from .ptq import verify

RECORD_FIELDS = {"schema_version", "record_kind", "crit_version", "recipe", "recipe_sha256", "seed", "source",
                 "calibration", "plan", "output", "errors", "cost", "environment", "started_at", "ended_at"}


def load_record(path):
    record = read_json(Path(path))
    if (not isinstance(record, dict) or record.get("record_kind") != "models_transform"
            or record.get("schema_version") != 1 or set(record) != RECORD_FIELDS):
        raise ValueError(f"{path} is not a models_transform record")
    return record


def evaluate(record_path, adapter, data, out, evaluation_path, config=None, parts=None, device=None, baseline=None,
             quality_budget=None, exactness=None, context_tokens=None, label=None, log=print):
    """Verify the output against its record, check that `data` was prepared from the source checkpoint (so Eval's
    quality reference is the original model), run Eval on the output and write a models_evaluation record."""
    from crit_eval import load_json, machine_settings
    from crit_eval.prepare import PREPARED
    from crit_eval.reference import REFERENCE
    from crit_eval.runner import PARTS, run
    record_path, data, evaluation_path = Path(record_path), Path(data), Path(evaluation_path).absolute()
    record = load_record(record_path)
    if evaluation_path.exists():
        raise ValueError(f"{evaluation_path} already exists")
    findings = [f for f in verify(record) if not f.startswith("source")]
    if findings:
        raise ValueError("; ".join(findings))
    parts = tuple(parts or PARTS)
    if not (data / PREPARED).is_file():
        raise ValueError(f"{data} holds no {PREPARED}: prepare it from the source checkpoint first "
                         f"(crit eval prepare --model {record['source']['path']} --data {data})")
    prepared = load_json(data / PREPARED)
    if prepared["model"]["sha256"] != record["source"]["eval_model_sha256"]:
        raise ValueError(f"{data} was prepared from another model than the source checkpoint; Eval must compare "
                         "the output with the original model's prepared data and reference")
    if "quality" in parts and not (data / REFERENCE).is_file():
        raise ValueError(f"{data} holds no {REFERENCE}: build it from the source checkpoint first "
                         f"(crit eval reference --model {record['source']['path']} --data {data})")
    settings = machine_settings()
    reserve = settings.get("context_reserve_gib")
    result = run(str(adapter), record["output"]["path"], data, out, config or {}, parts, label=label, log=log,
                 device=device, baseline=baseline, quality_budget=quality_budget, exactness=exactness,
                 context_tokens=context_tokens, vram_headroom_bytes=round(settings.get("vram_headroom_gib", 1.0) * (1 << 30)),
                 context_reserve_bytes=round(reserve * (1 << 30)) if reserve is not None else None)
    status = result["status"]
    kpis = status["kpis"]
    evaluation = {
        "schema_version": 1, "record_kind": "models_evaluation",
        "transform": {"path": str(record_path.absolute()), "sha256": file_sha(record_path),
                      "output_manifest_sha256": record["output"]["manifest_sha256"]},
        "adapter": {"path": str(Path(adapter).absolute()), "sha256": file_sha(Path(adapter))},
        "configuration": result["engine"]["configuration"], "parts": list(parts),
        "data": {"path": str(data.absolute()), "prepared_sha256": file_sha(data / PREPARED),
                 "reference_sha256": file_sha(data / REFERENCE) if (data / REFERENCE).is_file() else None,
                 "model_sha256": prepared["model"]["sha256"]},
        "result": {"path": str(Path(out).absolute()), "sha256": file_sha(Path(out))},
        "summary": {"gates": {g["name"]: g["state"] for g in status["gates"]}, "quality": kpis.get("quality"),
                    "decode_score": kpis["decode"]["score"], "prefill_score": kpis["prefill"]["score"],
                    "ttft_ms": kpis["ttft"]["score_ms"]},
        "recorded_at": now(),
    }
    write_json(evaluation_path, evaluation, immutable=True)
    return evaluation
