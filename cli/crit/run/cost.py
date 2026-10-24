"""Build cost: what a run took in time, agent tokens and API price, per candidate.

`crit run --id RUN cost --transcript PATH [--transcript PATH ...] [--agent NAME] [--prices JSON]` reads the coding
agent's session transcripts, attributes every model request to the candidate that was being built when it was
made, prices the tokens at the model's list prices (or the table passed with --prices) and writes `cost.json` at
the run root. The spans are the run's own events: a candidate's span runs from the previous candidate's seal (the
run's creation for the first) to its own seal, so a candidate's row is what it cost to reach it; requests after
the last seal are the selection and handoff. Time is wall time between those events, tokens come from the
transcripts, and the price is the agent's API-equivalent at the recorded rates: a fact about this run, never a
goal.

Transcript formats, detected per file: Claude Code session JSONL (one `assistant` line per response with
`message.usage`), Codex rollout JSONL (`event_msg` lines whose payload is a `token_count`), and a plain JSONL of
`{"timestamp", "model", "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"}`
for any other agent.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from .records import digest, file_sha, now, read_json, record, safe_path, text, timestamp, write_json

COST_JSON = "cost.json"
TOKEN_KINDS = ("input", "cache_read", "cache_write_5m", "cache_write_1h", "output")
# USD per million tokens at Anthropic's list prices (October 2026): input, cache reads, cache writes with a 5-minute
# and a 1-hour lifetime, output. Pass --prices to add a model or change a rate; the record keeps the table it used.
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "cache_read": 0.20, "cache_write_5m": 5.0, "cache_write_1h": 8.0, "output": 20.0},
    "claude-opus-5": {"input": 5.0, "cache_read": 0.50, "cache_write_5m": 6.25, "cache_write_1h": 10.0, "output": 25.0},
    "claude-sonnet-5-5": {"input": 2.0, "cache_read": 0.20, "cache_write_5m": 2.5, "cache_write_1h": 4.0, "output": 10.0},
    "claude-sonnet-5": {"input": 2.0, "cache_read": 0.20, "cache_write_5m": 2.5, "cache_write_1h": 4.0, "output": 10.0},
    "claude-haiku-5-5": {"input": 0.10, "cache_read": 0.01, "cache_write_5m": 0.125, "cache_write_1h": 0.20, "output": 0.50},
    "claude-fable-5-1": {"input": 10.0, "cache_read": 0.25, "cache_write_5m": 12.5, "cache_write_1h": 20.0, "output": 50.0},
}
PRICES_SOURCE = "Anthropic API list prices, October 2026 (USD per million tokens)"
MAX_TRANSCRIPT_BYTES = 4 * 1024 * 1024 * 1024


def _when(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name}: timestamp missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{name}: unreadable timestamp {value!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _count(value, name):
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: token counts must be non-negative integers")
    return value


def _request(at, model, tokens, key):
    return {"at": at, "model": model, "key": key, **{kind: tokens.get(kind, 0) for kind in TOKEN_KINDS}}


def _claude_code(lines, name):
    """One request per API response: Claude Code writes a line per content block, each repeating the response's
    usage, so responses are counted once by request id (the message id when there is none)."""
    requests = {}
    for number, line in lines:
        if line.get("type") != "assistant" or not isinstance(line.get("message"), dict):
            continue
        message = line["message"]
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        key = line.get("requestId") or message.get("id") or f"{name}:{number}"
        creation = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else None
        if creation:
            writes = {"cache_write_5m": _count(creation.get("ephemeral_5m_input_tokens"), name),
                      "cache_write_1h": _count(creation.get("ephemeral_1h_input_tokens"), name)}
        else:
            # Without the lifetime split the API's default 5-minute cache is assumed.
            writes = {"cache_write_5m": _count(usage.get("cache_creation_input_tokens"), name), "cache_write_1h": 0}
        found = _request(_when(line.get("timestamp"), f"{name}:{number}"), message.get("model"),
                         {"input": _count(usage.get("input_tokens"), name), "cache_read": _count(usage.get("cache_read_input_tokens"), name),
                          "output": _count(usage.get("output_tokens"), name), **writes}, key)
        earlier = requests.get(key)
        if earlier is None or found["output"] > earlier["output"]:
            requests[key] = found if earlier is None else {**found, "at": earlier["at"]}
    return list(requests.values())


def _codex(lines, name):
    """Codex rollouts report each response in a token_count event's last_token_usage; input_tokens there includes
    the cached tokens, which are split out. The model comes from the latest turn context."""
    requests, model = [], None
    for number, line in lines:
        payload = line.get("payload") if isinstance(line.get("payload"), dict) else {}
        kind = payload.get("type")
        if kind == "turn_context" and isinstance(payload.get("model"), str):
            model = payload["model"]
        elif kind == "thread_settings_applied" and isinstance(payload.get("thread_settings"), dict):
            model = payload["thread_settings"].get("model", model)
        elif kind == "session_meta" and isinstance(payload.get("model"), str):
            model = payload["model"]
        if line.get("type") != "event_msg" or kind != "token_count":
            continue
        info = payload.get("info")
        usage = info.get("last_token_usage") if isinstance(info, dict) else None
        if not isinstance(usage, dict):
            continue
        cached = _count(usage.get("cached_input_tokens"), name)
        requests.append(_request(_when(line.get("timestamp"), f"{name}:{number}"), model,
                                 {"input": max(0, _count(usage.get("input_tokens"), name) - cached), "cache_read": cached,
                                  "cache_write_5m": _count(usage.get("cache_write_input_tokens"), name), "cache_write_1h": 0,
                                  "output": _count(usage.get("output_tokens"), name)}, f"{name}:{number}"))
    return requests


def _plain(lines, name):
    requests = []
    for number, line in lines:
        if "input_tokens" not in line or "timestamp" not in line:
            continue
        requests.append(_request(_when(line["timestamp"], f"{name}:{number}"), line.get("model"),
                                 {"input": _count(line.get("input_tokens"), name), "cache_read": _count(line.get("cache_read_input_tokens"), name),
                                  "cache_write_5m": _count(line.get("cache_creation_input_tokens"), name), "cache_write_1h": 0,
                                  "output": _count(line.get("output_tokens"), name)}, f"{name}:{number}"))
    return requests


def _format(lines):
    for _, line in lines:
        if line.get("type") == "assistant" and isinstance(line.get("message"), dict) and "usage" in line["message"]:
            return "claude-code"
        if line.get("type") in {"session_meta", "event_msg", "response_item", "turn_context"} and isinstance(line.get("payload"), dict):
            return "codex"
        if "input_tokens" in line and "timestamp" in line:
            return "plain"
    raise ValueError("no model requests found: expected a Claude Code session, a Codex rollout or lines with "
                     "timestamp, model, input_tokens and output_tokens")


READERS = {"claude-code": _claude_code, "codex": _codex, "plain": _plain}


def read_transcript(path):
    """(facts, requests) of one transcript file: its path, bytes, sha256, format and request count, and every model
    request as {at, model, input, cache_read, cache_write_5m, cache_write_1h, output}."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    if path.stat().st_size > MAX_TRANSCRIPT_BYTES:
        raise ValueError(f"{path} is larger than the transcript limit")
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except ValueError:
                continue  # a torn or non-JSON line never hides the rest of the session
            if isinstance(value, dict):
                lines.append((number, value))
    found = _format(lines)
    requests = READERS[found](lines, path.name)
    facts = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha(path), "format": found,
             "requests": len(requests)}
    return facts, requests


def price_table(override=None):
    """The list prices with the user's --prices applied: a JSON object {model: {input, cache_read, cache_write_5m,
    cache_write_1h, output}} in USD per million tokens, or a path to one."""
    table = {model: dict(rates) for model, rates in PRICES.items()}
    source = PRICES_SOURCE
    if override:
        value = override
        if isinstance(override, (str, Path)):
            path = Path(override)
            value = read_json(path) if path.is_file() else json.loads(override)
            source += f"; overridden by {path if path.is_file() else '--prices'}"
        if not isinstance(value, dict):
            raise ValueError("--prices must be an object of models to rates")
        for model, rates in value.items():
            text(model, "price model")
            if not isinstance(rates, dict) or set(rates) - set(TOKEN_KINDS):
                raise ValueError(f"--prices {model}: rates are any of " + ", ".join(TOKEN_KINDS))
            for kind, rate in rates.items():
                if type(rate) not in (int, float) or rate < 0:
                    raise ValueError(f"--prices {model}.{kind}: a non-negative number")
            table[model] = {**table.get(model, {kind: 0.0 for kind in TOKEN_KINDS}), **rates}
    return table, source


def rates_for(model, table):
    """The rates of a model, also when the request names a dated or suffixed variant of a priced model."""
    if not isinstance(model, str):
        return None
    if model in table:
        return table[model]
    candidates = [name for name in table if model.startswith(name + "-") or model.startswith(name + "@")]
    return table[max(candidates, key=len)] if candidates else None


def _empty():
    return {"requests": 0, "tokens": {kind: 0 for kind in TOKEN_KINDS} | {"total": 0}, "usd": 0.0, "unpriced_requests": 0,
            "models": {}}


def _add(bucket, request, table):
    bucket["requests"] += 1
    for kind in TOKEN_KINDS:
        bucket["tokens"][kind] += request[kind]
        bucket["tokens"]["total"] += request[kind]
    rates = rates_for(request["model"], table)
    model = request["model"] if isinstance(request["model"], str) else "unknown"
    bucket["models"][model] = bucket["models"].get(model, 0) + 1
    if rates is None:
        bucket["unpriced_requests"] += 1
        bucket.setdefault("unpriced_models", [])
        if model not in bucket["unpriced_models"]:
            bucket["unpriced_models"].append(model)
        return
    bucket["usd"] += sum(request[kind] * rates.get(kind, 0.0) for kind in TOKEN_KINDS) / 1_000_000


def _finish(bucket):
    """A bucket with requests nobody can price has no USD figure, only the priced part and why."""
    if bucket["unpriced_requests"]:
        bucket["usd_priced"] = round(bucket["usd"], 4)
        bucket["usd"] = None
        bucket["usd_note"] = (f"{bucket['unpriced_requests']} requests name a model without a price "
                              f"({', '.join(bucket['unpriced_models'])}); pass --prices to include them")
    else:
        bucket["usd"] = round(bucket["usd"], 4)
    return bucket


def _hours(seconds):
    return round(seconds / 3600, 3)


def spans(root, run, candidates, until):
    """Each candidate's build span from the run's events: previous seal (or run creation) to its seal, or open."""
    created = _when(run["created_at"], "run created_at")
    sealed = {}
    for path in sorted(safe_path(root, "events").glob("event-*.json")):
        event = read_json(path)
        if event.get("event_type") == "candidate_sealed" and event.get("candidate_id"):
            sealed.setdefault(event["candidate_id"], _when(event["recorded_at"], path.name))
    rows, start = [], created
    for candidate in sorted(candidates, key=lambda c: c["ordinal"]):
        end = sealed.get(candidate["candidate_id"])
        rows.append({"candidate_id": candidate["candidate_id"], "label": candidate["label"], "from": start,
                     "until": end if end is not None else until, "sealed": end is not None})
        if end is not None:
            start = end
    return created, rows, start


def account(root, run, candidates, transcripts, agent=None, prices=None, until=None):
    """The run_cost record: per candidate, the wall time from the previous seal to its own, the agent's requests and
    tokens in that span and their API-equivalent price, with running totals from the run's first event."""
    if agent is not None:
        text(agent, "agent", 200)
    table, source = price_table(prices)
    facts, requests = [], []
    for path in transcripts:
        found, rows = read_transcript(path)
        facts.append(found)
        requests.extend(rows)
    if not requests:
        raise ValueError("the transcripts hold no model requests")
    requests.sort(key=lambda r: r["at"])
    stopped = None
    for path in sorted(safe_path(root, "events").glob("event-*.json")):
        event = read_json(path)
        if event.get("event_type") == "run_stopped":
            stopped = _when(event["recorded_at"], path.name)
    close = (_when(until, "until") if until else stopped) or datetime.now(timezone.utc)
    created, rows, after_start = spans(root, run, candidates, close)
    buckets = {row["candidate_id"]: _empty() for row in rows}
    before, selection, total = _empty(), _empty(), _empty()
    for request in requests:
        at = request["at"]
        if at < created:
            _add(before, request, table)
            continue
        if at >= close:
            continue
        _add(total, request, table)
        row = next((r for r in rows if r["from"] <= at < r["until"]), None)
        if row is not None:
            _add(buckets[row["candidate_id"]], request, table)
        elif at >= after_start:
            _add(selection, request, table)
    cumulative_tokens, cumulative_usd, cumulative_incomplete = 0, 0.0, False
    candidates_out = []
    for row in rows:
        bucket = _finish(buckets[row["candidate_id"]])
        seconds = round((row["until"] - row["from"]).total_seconds())
        cumulative_tokens += bucket["tokens"]["total"]
        if bucket["usd"] is None or cumulative_incomplete:
            cumulative_incomplete = True
        else:
            cumulative_usd += bucket["usd"]
        candidates_out.append({
            **{key: row[key] for key in ("candidate_id", "label", "sealed")},
            "from": row["from"].isoformat().replace("+00:00", "Z"), "until": row["until"].isoformat().replace("+00:00", "Z"),
            "seconds": seconds, **bucket,
            "cumulative": {"seconds": round((row["until"] - created).total_seconds()),
                           "hours": _hours((row["until"] - created).total_seconds()),
                           "tokens": cumulative_tokens, "usd": None if cumulative_incomplete else round(cumulative_usd, 4)}})
    run_seconds = round((close - created).total_seconds())
    value = record("run_cost", run["run_id"], recorded_at=now(), agent=agent,
                   transcripts=facts, prices={"source": source, "usd_per_million_tokens": table},
                   run={"from": created.isoformat().replace("+00:00", "Z"), "until": close.isoformat().replace("+00:00", "Z"),
                        "until_source": "given" if until else ("run_stopped" if stopped else "now"),
                        "seconds": run_seconds, "hours": _hours(run_seconds), **_finish(total)},
                   candidates=candidates_out, selection=_finish(selection), before_run=_finish(before))
    return value


def write(root, run, candidates, transcripts, agent=None, prices=None, until=None):
    value = account(root, run, candidates, transcripts, agent, prices, until)
    path = safe_path(root, COST_JSON)
    write_json(path, value)
    return value


def load(root):
    """The run's cost.json, or None."""
    path = safe_path(root, COST_JSON)
    if not path.is_file():
        return None
    value = read_json(path)
    if value.get("record_kind") != "run_cost":
        raise ValueError(f"{COST_JSON} is not a run_cost record")
    timestamp(value["recorded_at"], "cost recorded_at")
    return value


def built(value):
    """The card-sized summary of a run_cost record: hours, agent tokens, API-equivalent USD and the agent."""
    return {"hours": value["run"]["hours"], "agent_tokens": value["run"]["tokens"]["total"], "api_usd": value["run"]["usd"],
            "agent": value["agent"], "candidates": len(value["candidates"]), "cost_sha256": digest(value)}
