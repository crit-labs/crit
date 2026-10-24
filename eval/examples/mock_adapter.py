"""Minimal eval-v1 adapter without a GPU, used by tests and as a template for real engines.

It "generates" a fixed token pattern, so every harness path can be exercised on any host. Sampled requests draw
from a fixed distribution over eight candidates per position.
"""
import math
import random
import time

# The mock model's distribution at every sampled position: raw log-probabilities of eight candidates, likeliest first.
CANDIDATES = [math.log(p) for p in (0.35, 0.2, 0.14, 0.1, 0.08, 0.06, 0.04, 0.03)]


def processed(logprobs, sampling):
    """Temperature, then top_k and top_p, as a sampler applies them; probabilities aligned with logprobs."""
    temperature = float(sampling.get("temperature") or 1.0)
    keep = list(range(len(logprobs)))[:int(sampling["top_k"])] if sampling.get("top_k") else list(range(len(logprobs)))
    weights = {i: math.exp(logprobs[i] / temperature) for i in keep}
    total = sum(weights.values())
    probs = {i: w / total for i, w in weights.items()}
    if sampling.get("top_p") is not None and float(sampling["top_p"]) < 1:
        mass, prefix = 0.0, []
        for i in keep:
            prefix.append(i)
            mass += probs[i]
            if mass >= float(sampling["top_p"]):
                break
        probs = {i: probs[i] / mass for i in prefix}
    return [probs.get(i, 0.0) for i in range(len(logprobs))]


class MockEngine:
    def __init__(self, context):
        self.config = {"draft_tokens": 4 if context["speculative"] else 0, **context["config"]}
        # Size the KV cache and any length-dependent tables for context_tokens when the harness gives it.
        self.capacity = int(context.get("context_tokens") or 8192)
        if self.config.get("max_context"):
            # Test knob: an engine that holds less than it is asked for.
            self.capacity = min(self.capacity, int(self.config["max_context"]))
        # Test knob: an engine that cannot be built below a context.
        if self.capacity < int(self.config.get("fails_below_context", 0)):
            raise RuntimeError("cannot be built below the context it was tuned for")
        self.calls = 0
        self.longest = 0  # the longest prompt served so far

    def info(self):
        return {"name": "mock", "configuration": self.config, "max_context": self.capacity,
                "speculative": self.config["draft_tokens"] > 0,
                # Optional headroom inputs: draft width and bytes one full model pass reads.
                "draft_width": self.config["draft_tokens"], "bytes_per_pass": self.config.get("bytes_per_pass"),
                "draft_bytes_per_step": self.config.get("draft_bytes_per_step"),
                "kv_bytes_per_token": self.config.get("kv_bytes_per_token"),
                # Optional: replay(prompt, continuation, on_tokens) decodes a pinned continuation.
                "replay": bool(self.config.get("replay", True)),
                # Optional: generate_plain(prompt, max_new_tokens, on_tokens) decodes plain greedy on this
                # speculative engine, so the exactness check opens no second engine.
                "plain": bool(self.config.get("plain", True)) and self.config["draft_tokens"] > 0}

    def _rounds(self, tokens, on_tokens, plain=False):
        step = 1 if plain else self.config["draft_tokens"] + 1
        # Test knob: an engine that gets slower when it is opened at a smaller context.
        pause = 0.0005 + (0.005 if self.capacity < int(self.config.get("slow_below_context", 0)) else 0.0)
        if self.config.get("disturbed"):
            # Test knob: a device something else is using: decoding takes 0.1 s, every other request twice that.
            pause += 0.1 * (1 + self.calls % 2) / -(-len(tokens) // step)
        for i in range(0, len(tokens), step):
            time.sleep(pause)
            on_tokens(tokens[i:i + step])
        rounds = -(-len(tokens) // step)
        # Optional: drafts accepted in each round, for the speed-of-light projection.
        out = {"tokens": tokens, "rounds": rounds, "accepted": [step - 1] * rounds}
        if self.config.get("report_parts"):
            # Optional: engine-measured time in verify passes and draft steps over the request, in ms.
            out.update(verify_ms=rounds * float(self.config.get("verify_ms_per_round", 0.4)),
                       draft_ms=rounds * (step - 1) * float(self.config.get("draft_ms_per_step", 0.02)))
        return out

    def generate(self, prompt, max_new_tokens, on_tokens):
        # Test knob: an engine that runs out of memory beyond a context it nevertheless claims.
        if len(prompt) + max_new_tokens > int(self.config.get("fails_above_context", 1 << 62)):
            raise RuntimeError("out of device memory")
        # Test knob: an engine whose greedy text changes from one request to the next.
        shift = self.calls if self.config.get("unrepeatable") else 0
        # Test knob: an engine that keeps stale state once it has served a prompt this long.
        if self.longest >= int(self.config.get("drifts_after_prompt", 1 << 62)):
            shift += 1
        self.calls += 1
        self.longest = max(self.longest, len(prompt))
        tokens = [(sum(prompt) + i + shift) % 1000 for i in range(max_new_tokens)]
        # Test knob: ids at or above facts_from are facts; the reply repeats them, unless the prompt is longer
        # than forgets_above.
        if "facts_from" in self.config and len(prompt) <= int(self.config.get("forgets_above", 1 << 62)):
            facts = [t for t in prompt if t >= int(self.config["facts_from"])]
            tokens = (facts + tokens)[:max_new_tokens]
        return self._rounds(tokens, on_tokens)

    def generate_plain(self, prompt, max_new_tokens, on_tokens):
        """One token per target pass, no drafting: the same greedy text."""
        return self._rounds([(sum(prompt) + i) % 1000 for i in range(max_new_tokens)], on_tokens, plain=True)

    def generate_sampled(self, prompt, max_new_tokens, on_tokens, sampling, logprobs=None):
        """Optional (crit-eval-v7): a request sampled at `sampling` (temperature, top_p, top_k, min_p, seed). With
        logprobs n, return each token's log-probability and n alternatives as "logprobs". This mock does not
        speculate under sampling: one token per pass."""
        rng = random.Random(int(sampling.get("seed", 0)))
        probs = processed(CANDIDATES, sampling)
        tokens, rows = [], []
        for i in range(max_new_tokens):
            base = (sum(prompt) + 10 * i) % 990
            # Test knob: a sampler that takes the likeliest token whenever a greedy draft would match.
            pick = 0 if self.config.get("greedy_sampling") and rng.random() < 0.6 else rng.choices(range(8), weights=probs)[0]
            tokens.append(base + pick)
            rows.append({"token": base + pick, "logprob": CANDIDATES[pick],
                         "top": [[base + j, lp] for j, lp in enumerate(CANDIDATES)][:logprobs or 0]})
        out = self._rounds(tokens, on_tokens, plain=True)
        if logprobs:
            out["logprobs"] = rows
        return out

    def replay(self, prompt, continuation, on_tokens):
        """Decode `continuation` as if it were the engine's own greedy output (drafts accepted while
        they match it, corrections taken from it)."""
        return self._rounds(list(continuation), on_tokens)

    def close(self):
        pass


def create_eval_engine(context):
    return MockEngine(context)
