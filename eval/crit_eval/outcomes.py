"""Bounded OpenAI-style single-choice SSE outcomes and fixed-window accounting.

The caller owns transport, request scheduling, time measurement and cancellation.
This validates reported usage; it cannot prove a server's tokenizer or quality.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
import re
from collections.abc import Iterable


def _integer(value, label):
    if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
        raise ValueError(f"{label} must be a bounded nonnegative integer")
    return value


def _seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite nonnegative seconds")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite nonnegative seconds") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite nonnegative seconds")
    return float(value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(_):
    raise ValueError("Nonfinite JSON number")


def _bounded_snapshot(values, maximum):
    if isinstance(values, (str, bytes)):
        raise ValueError("Cohort collections must contain records or request IDs")
    result = []
    for value in values:
        if len(result) == maximum:
            raise ValueError("Cohort iterable exceeds its bound")
        result.append(value)
    return tuple(result)


class _TransportFailure(Exception):
    pass


def _transport_bytes(chunks):
    try:
        iterator = iter(chunks)
    except Exception as error:
        raise _TransportFailure(type(error).__name__) from None
    while True:
        try:
            value = next(iterator)
        except StopIteration:
            return
        except Exception as error:
            raise _TransportFailure(type(error).__name__) from None
        yield value


@dataclass(frozen=True)
class StreamOutcome:
    successful: bool
    completion_tokens: int | None
    finish_reason: str | None
    errors: tuple[str, ...]


def validate_completion(http_status: int, chunks: Iterable[bytes], *,
                        allowed_finishes=frozenset({"stop", "length", "tool_calls"}),
                        max_frame_bytes=1024 * 1024, max_stream_bytes=16 * 1024 * 1024):
    """Validate LF/CRLF SSE, n=1, finish, DONE and final completion usage.

    Request ``stream_options.include_usage`` from compatible servers. Missing
    usage is an unqualified outcome, never an assumed token count. Bytes after
    DONE may contain comments/blank lines, but no further data events. JSON and
    SSE bounds include usage/error frames. A disconnect or malformed/truncated
    frame returns a failed outcome. This function does not close the transport.
    """
    if type(http_status) is not int or not 100 <= http_status <= 599:
        raise ValueError("Invalid HTTP status")
    _integer(max_frame_bytes, "max_frame_bytes")
    _integer(max_stream_bytes, "max_stream_bytes")
    if not 0 < max_frame_bytes <= max_stream_bytes or not allowed_finishes or any(
            not isinstance(value, str) or not value or value in {"error", "cancelled"}
            for value in allowed_finishes):
        raise ValueError("Invalid stream policy or bounds")
    errors = [] if http_status == 200 else [f"HTTP status {http_status}"]
    finish, usage, done = None, None, False
    pending, data_lines, total, frame_bytes = bytearray(), [], 0, 0

    def event():
        nonlocal finish, usage, done
        if not data_lines:
            return
        value = b"\n".join(data_lines).decode("utf-8", errors="strict")
        if done:
            raise ValueError("Data event after DONE")
        if value == "[DONE]":
            done = True
            return
        payload = json.loads(value, object_pairs_hook=_object, parse_constant=_invalid_constant)
        if not isinstance(payload, dict):
            raise ValueError("Completion event must be an object")
        if "error" in payload or payload.get("type") == "error":
            raise ValueError("In-band server error")
        if "choices" not in payload or not isinstance(payload["choices"], list):
            raise ValueError("Completion event lacks choices")
        choices = payload["choices"]
        if len(choices) > 1:
            raise ValueError("Only a single completion choice is supported")
        for choice in choices:
            if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
                raise ValueError("Invalid choice index")
            if finish is not None:
                raise ValueError("Choice event after finish")
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in allowed_finishes:
                    raise ValueError("Finish reason does not satisfy the workload policy")
                finish = reason
        if payload.get("usage") is not None:
            if finish is None:
                raise ValueError("Completion usage arrived before final finish")
            reported = payload["usage"]
            if not isinstance(reported, dict):
                raise ValueError("Invalid usage object")
            count = _integer(reported.get("completion_tokens"), "completion_tokens")
            if usage is not None and usage != count:
                raise ValueError("Conflicting completion usage")
            usage = count

    try:
        for chunk in _transport_bytes(chunks):
            if not isinstance(chunk, bytes):
                raise ValueError("Transport must yield bytes")
            total += len(chunk)
            if total > max_stream_bytes:
                raise ValueError("Stream byte bound exceeded")
            pending.extend(chunk)
            while True:
                split = pending.find(b"\n")
                if split < 0:
                    break
                line = bytes(pending[:split])
                del pending[:split + 1]
                frame_bytes += split + 1
                if frame_bytes > max_frame_bytes:
                    raise ValueError("SSE frame byte bound exceeded")
                if line.endswith(b"\r"):
                    line = line[:-1]
                if b"\r" in line:
                    raise ValueError("Only LF or CRLF framing is supported")
                if not line:
                    event()
                    data_lines.clear()
                    frame_bytes = 0
                elif not line.startswith(b":"):
                    field, _, value = line.partition(b":")
                    if field == b"data":
                        data_lines.append(value[1:] if value.startswith(b" ") else value)
                    elif field == b"event" and value.strip() == b"error":
                        raise ValueError("In-band SSE error event")
            if frame_bytes + len(pending) > max_frame_bytes:
                raise ValueError("SSE frame byte bound exceeded")
        if pending or data_lines or frame_bytes:
            raise ValueError("Truncated SSE frame")
    except _TransportFailure as error:
        errors.append(f"Transport failed: {error}")
    except (ValueError, UnicodeError, RecursionError) as error:
        errors.append(str(error))
    except Exception as error:
        errors.append(f"Validation failed: {type(error).__name__}")
    if not done:
        errors.append("Missing DONE event")
    if finish is None:
        errors.append("Missing finish reason")
    if usage is None:
        errors.append("Missing actual completion token usage")
    return StreamOutcome(not errors, usage, finish, tuple(errors))


def telemetry_count(raw: str | None):
    """Unavailable telemetry is None; only an explicit integral sample is zero.

    A caller extracts a declared metric/series first. This is an observation,
    not proof of server exclusivity between polls.
    """
    if raw is None:
        return None
    if not isinstance(raw, str) or len(raw) > 64 or re.fullmatch(
            r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", raw.strip()) is None:
        raise ValueError("Invalid telemetry sample")
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError("Invalid telemetry sample") from error
    if not value.is_finite() or value < 0 or value > (1 << 63) - 1 or value != value.to_integral_value():
        raise ValueError("Telemetry sample is not a finite count")
    return _integer(int(value), "telemetry")


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    started: float
    finished: float
    outcome: StreamOutcome


@dataclass(frozen=True)
class BenchmarkOutcome:
    qualified: bool
    errors: tuple[str, ...]
    successful_requests: int
    successful_completion_tokens: int
    wall_seconds: float
    completion_tokens_per_second: float | None
    peak_concurrency: int
    external_activity_observation: int | None


def validate_cohort(expected_ids, records, *, wall_started, wall_finished,
                    minimum_peak=1, process_exit_code=0, external_activity=None):
    """Use one fixed measured wall interval including every expected request.

    A failed/missing/duplicate request, bad process exit or insufficient measured
    concurrency disqualifies the aggregate rate. Successful counts remain visible
    for diagnosis. Zero-duration requests count as completed but not concurrent.
    """
    expected = _bounded_snapshot(expected_ids, 10000)
    records = _bounded_snapshot(records, 10000)
    if not 0 < len(expected) <= 10000 or len(records) > 10000:
        raise ValueError("Cohort exceeds bounds or is empty")
    if any(not isinstance(key, str) or not 0 < len(key) <= 128 for key in expected) or len(set(expected)) != len(expected):
        raise ValueError("Expected request IDs must be bounded and unique")
    start, end = _seconds(wall_started, "wall start"), _seconds(wall_finished, "wall finish")
    if end <= start:
        raise ValueError("Measured wall interval must be positive")
    _integer(minimum_peak, "minimum peak")
    if not 1 <= minimum_peak <= len(expected) or type(process_exit_code) is not int:
        raise ValueError("Invalid concurrency or process status")
    if external_activity is not None:
        _integer(external_activity, "external activity")
    errors, seen, events = [], set(), []
    successes = tokens = 0
    for record in records:
        if not isinstance(record, RequestRecord) or not isinstance(record.outcome, StreamOutcome):
            raise ValueError("Expected owned request/outcome records")
        if record.request_id not in expected or record.request_id in seen:
            errors.append("Unexpected or duplicate request ID")
            continue
        seen.add(record.request_id)
        began, ended = _seconds(record.started, "request start"), _seconds(record.finished, "request finish")
        if not start <= began <= ended <= end:
            errors.append(f"{record.request_id}: request outside measured wall interval")
            continue
        if ended > began:
            events.extend(((began, 1), (ended, -1)))
        if record.outcome.successful and not record.outcome.errors and record.outcome.completion_tokens is not None:
            tokens += _integer(record.outcome.completion_tokens, "completion tokens")
            successes += 1
        else:
            errors.append(f"{record.request_id}: completion failed validation")
    if seen != set(expected):
        errors.append("Expected requests are missing")
    current = peak = 0
    for _, change in sorted(events):  # Finish (-1) precedes start (1) at a tie.
        current += change
        peak = max(peak, current)
    if peak < minimum_peak:
        errors.append("Measured concurrency below workload requirement")
    if process_exit_code != 0:
        errors.append(f"Benchmark process exited with {process_exit_code}")
    duration = end - start
    return BenchmarkOutcome(not errors, tuple(errors), successes, tokens, duration,
                            tokens / duration if not errors else None, peak, external_activity)
