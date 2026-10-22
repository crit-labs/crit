"""How long Eval waits for one request to an engine behind a server, from what the engine already served.

Eval has no fixed time budget: its work scales with the model and the device. Each request is instead bounded by
the rates this engine measured in the same attempt (protocol requests.timeout): a prompt of n tokens is expected to
prefill in at most t * max(1, n / m)^2 for a prompt of m tokens that prefilled in t (prefill cost grows at most
quadratically with length), and each output token at a context of c to decode in at most d * max(1, c / k) for a
token that decoded in d at a context of k (a decode step grows at most linearly with the context). The tightest
such bound over the observations counts; the request may take slack times it plus the protocol's floor, and the
floor alone before anything was measured. A request past its bound means the engine hung: Eval stops (Hung).
"""


class Hung(Exception):
    """A request took longer than its bound (RequestDeadline): the engine stopped making progress."""


class RequestDeadline:
    def __init__(self, rule):
        self.floor, self.slack = float(rule["floor_seconds"]), float(rule["slack"])
        self.prefills = []   # (prompt tokens, seconds)
        self.decodes = []    # (context tokens, seconds per output token)

    def observe(self, prompt_tokens, prefill_seconds, output_tokens=0, decode_seconds=None):
        """One request served: its prompt prefilled in prefill_seconds; its output tokens after the first decoded in
        decode_seconds."""
        if prompt_tokens > 0 and prefill_seconds > 0:
            self.prefills.append((int(prompt_tokens), float(prefill_seconds)))
        if output_tokens > 1 and decode_seconds and decode_seconds > 0:
            self.decodes.append((int(prompt_tokens) + int(output_tokens), float(decode_seconds) / (output_tokens - 1)))

    def expected(self, prompt_tokens, output_tokens=0):
        """Upper bound of the request's seconds from the observations, None before any."""
        if not self.prefills:
            return None
        prefill = min(t * max(1.0, prompt_tokens / m) ** 2 for m, t in self.prefills)
        context = prompt_tokens + output_tokens
        step = min(d * max(1.0, context / k) for k, d in self.decodes) if self.decodes else prefill / max(1, prompt_tokens)
        return prefill + max(0, output_tokens - 1) * step

    def seconds(self, prompt_tokens, output_tokens=0):
        """How long the request may take."""
        expected = self.expected(prompt_tokens, output_tokens)
        return self.floor + (self.slack * expected if expected is not None else 0.0)
