"""Bind an Eval result of an engine serving a transformed checkpoint to its transform record, with its quality replies
scored on the source checkpoint."""
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


def evaluate(record_path, data, result_path, evaluation_path, baseline=None, log=print):
    """Verify the output against its record, check that `data` was prepared from the source checkpoint (so Eval's
    quality reference is the original model: crit eval reference --model SOURCE --data DATA before the round) and
    that the result at `result_path` (crit eval run --url on the engine serving the output, stopped since) was
    measured on it, and write a models_evaluation record."""
    from crit_eval import load_json
    from crit_eval.prepare import PREPARED
    record_path, data, result_path = Path(record_path), Path(data), Path(result_path)
    evaluation_path = Path(evaluation_path).absolute()
    record = load_record(record_path)
    if evaluation_path.exists():
        raise ValueError(f"{evaluation_path} already exists")
    findings = [f for f in verify(record) if not f.startswith("source")]
    if findings:
        raise ValueError("; ".join(findings))
    if not (data / PREPARED).is_file():
        raise ValueError(f"{data} holds no {PREPARED}: prepare it from the source checkpoint first "
                         f"(crit eval prepare --model {record['source']['path']} --data {data})")
    prepared = load_json(data / PREPARED)
    if prepared["model"]["sha256"] != record["source"]["eval_model_sha256"]:
        raise ValueError(f"{data} was prepared from another model than the source checkpoint; Eval must compare "
                         "the output with the original model's prepared data and reference")
    result = load_json(result_path)
    if (result.get("model_sha256"), result.get("suite_sha256")) != (prepared["model"]["sha256"], prepared["suite_sha256"]):
        raise ValueError(f"{result_path} was measured on other prepared data than {data}")
    if not result.get("server"):
        raise ValueError(f"{result_path} measured an adapter's engine: measure the engine serving the output with "
                         f"crit eval run --url ... --model {record['output']['path']} --data {data}")
    if result.get("aborted") or "status" not in result:
        raise ValueError(f"{result_path} is no finished Eval round")
    status = result["status"]
    kpis = status["kpis"]
    decode = kpis.get("decode_sampled") or kpis.get("decode_greedy") or kpis.get("decode") or {}
    # crit-eval's quality KPIs, perplexity and kl_divergence (earlier results: one quality KPI).
    quality = ({"perplexity": kpis.get("perplexity"), "kl_divergence": kpis.get("kl_divergence")}
               if "perplexity" in kpis else kpis.get("quality"))
    evaluation = {
        "schema_version": 1, "record_kind": "models_evaluation",
        "transform": {"path": str(record_path.absolute()), "sha256": file_sha(record_path),
                      "output_manifest_sha256": record["output"]["manifest_sha256"]},
        "server": {"url": result["server"], "model": result["engine"]["name"]},
        "data": {"path": str(data.absolute()), "prepared_sha256": file_sha(data / PREPARED),
                 "model_sha256": prepared["model"]["sha256"]},
        "result": {"path": str(result_path.absolute()), "sha256": file_sha(result_path)},
        "summary": {"gates": {g["name"]: g["state"] for g in status["gates"]}, "quality": quality,
                    "decode_score": decode.get("score"), "prefill_score": kpis["prefill"]["score"],
                    "ttft_ms": kpis["ttft"]["score_ms"]},
        "recorded_at": now(),
    }
    write_json(evaluation_path, evaluation, immutable=True)
    return evaluation
