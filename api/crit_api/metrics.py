"""Serving metrics in the Prometheus text exposition format (GET /metrics): counters since the server started and
the last request's telemetry, from what crit serve already measures."""
import threading
import time

OUTCOMES = ("completed", "error", "cancelled", "timed_out")


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = {}       # (dialect, outcome) -> count
        self.in_flight = 0
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self.last = {}           # the last completed request's rates

    def begin(self):
        with self.lock:
            self.in_flight += 1

    def end(self, dialect, outcome, measured=None, seconds=None):
        """Record one finished request: its dialect (openai, anthropic, ollama), outcome (OUTCOMES), the engine's
        telemetry when it completed, and its wall time."""
        with self.lock:
            self.in_flight = max(0, self.in_flight - 1)
            key = (dialect, outcome if outcome in OUTCOMES else "error")
            self.requests[key] = self.requests.get(key, 0) + 1
            if seconds is not None:
                self.seconds += seconds
            if measured:
                self.prompt_tokens += measured.get("prompt_tokens", 0)
                self.completion_tokens += measured.get("output_tokens", 0)
                if measured.get("prefill_tokens") is not None:
                    self.cached_prompt_tokens += measured["prompt_tokens"] - measured["prefill_tokens"]
                self.last = {key: measured[key] for key in ("prefill_tps", "decode_tps", "ttft_ms", "tokens_per_round")
                             if measured.get(key) is not None}

    def render(self, engine_status):
        """The exposition text. `engine_status` is ServedEngine.status(): the loaded candidate and state."""
        with self.lock:
            lines = []

            def metric(name, kind, help_text, samples):
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {kind}")
                for labels, value in samples:
                    label = ("{" + ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items()) + "}") if labels else ""
                    lines.append(f"{name}{label} {_number(value)}")

            metric("crit_serve_requests_total", "counter", "Chat requests finished, by API dialect and outcome.",
                   [({"dialect": d, "outcome": o}, n) for (d, o), n in sorted(self.requests.items())])
            metric("crit_serve_requests_in_flight", "gauge", "Chat requests being generated or queued.", [({}, self.in_flight)])
            metric("crit_serve_prompt_tokens_total", "counter", "Prompt tokens of completed requests.", [({}, self.prompt_tokens)])
            metric("crit_serve_cached_prompt_tokens_total", "counter",
                   "Prompt tokens the engine served from its prompt cache instead of computing.", [({}, self.cached_prompt_tokens)])
            metric("crit_serve_completion_tokens_total", "counter", "Completion tokens of completed requests.", [({}, self.completion_tokens)])
            metric("crit_serve_request_seconds_total", "counter", "Wall seconds spent in finished chat requests.", [({}, self.seconds)])
            metric("crit_serve_uptime_seconds", "gauge", "Seconds since the server started.", [({}, time.time() - self.started)])
            for key, name, help_text in (("prefill_tps", "crit_serve_last_prefill_tokens_per_second", "Prefill rate of the last completed request."),
                                         ("decode_tps", "crit_serve_last_decode_tokens_per_second", "Decode rate of the last completed request."),
                                         ("ttft_ms", "crit_serve_last_ttft_milliseconds", "Time to first token of the last completed request."),
                                         ("tokens_per_round", "crit_serve_last_tokens_per_round", "Output tokens per target pass of the last completed request.")):
                if key in self.last:
                    metric(name, "gauge", help_text, [({}, self.last[key])])
            engine = engine_status.get("engine") or {}
            state = engine_status.get("state", "unknown")
            metric("crit_serve_engine_loaded", "gauge", "1 while an engine is loaded, with its run and candidate.",
                   [({"run": str(engine.get("run_id", "")), "candidate": str(engine.get("candidate_id", "")), "state": str(state)},
                     1 if engine.get("candidate_id") else 0)])
            return "\n".join(lines) + "\n"


def _escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _number(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))
