"""Serving metrics in the Prometheus text exposition format (GET /metrics): counters, gauges and fixed-bucket
histograms since the server started and the last request's telemetry, from what crit serve already measures."""
import bisect
import threading
import time

OUTCOMES = ("completed", "rejected", "error", "cancelled", "timed_out")
SECONDS = (.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500)
TOKEN_SECONDS = (.0005, .001, .0025, .005, .0075, .01, .015, .02, .03, .05, .075, .1, .25, .5, 1, 2.5)
TOKENS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000)


class Histogram:
    """Observations counted in fixed buckets by their upper bounds."""
    def __init__(self, bounds):
        self.bounds = bounds
        self.counts = [0] * (len(bounds) + 1)  # the last counts observations above every bound
        self.sum = 0

    def observe(self, value):
        self.counts[bisect.bisect_left(self.bounds, value)] += 1
        self.sum += value

    def lines(self, name):
        """Its samples: the cumulative bucket counts, then the sum and count."""
        found, total = [], 0
        for bound, count in zip((*map(_number, self.bounds), "+Inf"), self.counts):
            total += count
            found.append(f'{name}_bucket{{le="{bound}"}} {total}')
        return found + [f"{name}_sum {_number(self.sum)}", f"{name}_count {total}"]


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = {}       # (dialect, outcome) -> count
        self.waiting = 0
        self.running = 0
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self.last = {}           # the last completed request's rates
        self.queue = Histogram(SECONDS)
        self.ttft = Histogram(SECONDS)
        self.per_token = Histogram(TOKEN_SECONDS)
        self.latency = Histogram(SECONDS)
        self.prompts = Histogram(TOKENS)
        self.completions = Histogram(TOKENS)

    def begin(self):
        """Count a chat or score request waiting for the engine; returns its timing for dispatched() and end()."""
        with self.lock:
            self.waiting += 1
        return {"begun": time.perf_counter(), "dispatched": None}

    def count(self, dialect, outcome):
        """Record a request answered before it ran, such as one that failed validation."""
        with self.lock:
            self._count(dialect, outcome)

    def dispatched(self, timing):
        """The request's command went to the engine, after admission and any engine load: it is running."""
        with self.lock:
            if timing["dispatched"] is None:
                timing["dispatched"] = time.perf_counter()
                self.waiting, self.running = max(0, self.waiting - 1), self.running + 1
                self.queue.observe(timing["dispatched"] - timing["begun"])

    def end(self, dialect, outcome, measured=None, timing=None):
        """Record one finished request: its dialect (openai, responses, anthropic, ollama, crit for scores), outcome
        (OUTCOMES), the engine's telemetry when it completed, and its timing from begin()."""
        with self.lock:
            self._count(dialect, outcome)
            if timing is not None:
                if timing["dispatched"] is None:
                    self.waiting = max(0, self.waiting - 1)
                else:
                    self.running = max(0, self.running - 1)
                seconds = time.perf_counter() - timing["begun"]
                self.seconds += seconds
                self.latency.observe(seconds)
            if measured:
                self.prompt_tokens += measured.get("prompt_tokens", 0)
                self.completion_tokens += measured.get("output_tokens", 0)
                if measured.get("prefill_tokens") is not None:
                    self.cached_prompt_tokens += measured["prompt_tokens"] - measured["prefill_tokens"]
                self.last = {key: measured[key] for key in ("prefill_tps", "decode_tps", "ttft_ms", "tokens_per_round")
                             if measured.get(key) is not None}
                self.prompts.observe(measured.get("prompt_tokens", 0))
                self.completions.observe(measured.get("output_tokens", 0))
                if measured.get("ttft_ms") is not None:
                    self.ttft.observe(measured["ttft_ms"] / 1000)
                if measured.get("decode_tokens") and measured.get("decode_ms") is not None:
                    self.per_token.observe(measured["decode_ms"] / 1000 / measured["decode_tokens"])

    def _count(self, dialect, outcome):
        key = (dialect, outcome if outcome in OUTCOMES else "error")
        self.requests[key] = self.requests.get(key, 0) + 1

    def render(self, served):
        """The exposition text. `served` is the ServedEngine: its status (the loaded candidate and state) and its
        load counts."""
        engine_status = served.status()
        with self.lock:
            lines = []

            def metric(name, kind, help_text, samples=()):
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {kind}")
                for labels, value in samples:
                    label = ("{" + ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items()) + "}") if labels else ""
                    lines.append(f"{name}{label} {_number(value)}")

            metric("crit_serve_requests_total", "counter", "Chat and score requests finished, by API dialect and outcome.",
                   [({"dialect": d, "outcome": o}, n) for (d, o), n in sorted(self.requests.items())])
            metric("crit_serve_requests_in_flight", "gauge", "Chat and score requests running or queued.",
                   [({}, self.running + self.waiting)])
            metric("crit_serve_requests_running", "gauge", "Chat and score requests the engine is running.", [({}, self.running)])
            metric("crit_serve_requests_waiting", "gauge", "Chat and score requests queued for the engine or for its load.",
                   [({}, self.waiting)])
            metric("crit_serve_prompt_tokens_total", "counter", "Prompt tokens of completed chat requests.", [({}, self.prompt_tokens)])
            metric("crit_serve_cached_prompt_tokens_total", "counter",
                   "Prompt tokens the engine served from its prompt cache instead of computing.", [({}, self.cached_prompt_tokens)])
            metric("crit_serve_completion_tokens_total", "counter", "Completion tokens of completed chat requests.", [({}, self.completion_tokens)])
            metric("crit_serve_request_seconds_total", "counter", "Wall seconds spent in chat and score requests after validation.", [({}, self.seconds)])
            metric("crit_serve_uptime_seconds", "gauge", "Seconds since the server started.", [({}, time.time() - self.started)])
            for name, help_text, found in (
                    ("crit_serve_request_queue_time_seconds",
                     "Seconds chat and score requests waited for the engine to start them: queued and during an engine load.", self.queue),
                    ("crit_serve_time_to_first_token_seconds",
                     "The engine's time to first token of completed chat requests, from its admission of the request.", self.ttft),
                    ("crit_serve_time_per_output_token_seconds",
                     "The engine's decode seconds per token after the first, of completed chat requests.", self.per_token),
                    ("crit_serve_e2e_request_latency_seconds",
                     "Wall seconds of chat and score requests after validation, every outcome.", self.latency),
                    ("crit_serve_request_prompt_tokens", "Prompt tokens of completed chat requests.", self.prompts),
                    ("crit_serve_request_generation_tokens", "Completion tokens of completed chat requests.", self.completions)):
                metric(name, "histogram", help_text)
                lines.extend(found.lines(name))
            for key, name, help_text in (("prefill_tps", "crit_serve_last_prefill_tokens_per_second", "Prefill rate of the last completed request."),
                                         ("decode_tps", "crit_serve_last_decode_tokens_per_second", "Decode rate of the last completed request."),
                                         ("ttft_ms", "crit_serve_last_ttft_milliseconds", "Time to first token of the last completed request."),
                                         ("tokens_per_round", "crit_serve_last_tokens_per_round", "Output tokens per target pass of the last completed request.")):
                if key in self.last:
                    metric(name, "gauge", help_text, [({}, self.last[key])])
            metric("crit_serve_engine_loads_total", "counter",
                   "Engine loads completed: at start, after a retired host and for each model switch.", [({}, served.loads)])
            metric("crit_serve_model_switches_total", "counter",
                   "Engine loads that replaced another model's engine.", [({}, served.switches)])
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
