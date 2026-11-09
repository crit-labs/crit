"""A small client for the endpoints the smoke tests read from a served engine (crit serve): the model card, the
engine's tokenizer, fixed-text scores and greedy chat replies with their log-probabilities. Rows come back as
(token, logprob, ((token, logprob), ...ranked alternatives)), the shape of crit's score rows."""
import json
import urllib.error
import urllib.request

SCORE = "chat-score-v1"
LOGPROBS = "chat-logprobs-v1"
TOKENIZE = "chat-tokenize-v1"
PLAIN = "chat-plain-v1"
CONCURRENT = "chat-concurrent-v1"


class Unsupported(Exception):
    """The engine does not offer what a test needs."""


class Client:
    def __init__(self, url, model=None, api_key=None, timeout=300):
        url = url.rstrip("/")
        self.url = url[:-3] if url.endswith("/v1") else url
        self.model, self.timeout = model, timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self._card = None

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, headers=self.headers,
                                         method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                message = (json.loads(raw).get("error") or {}).get("message") or raw[:300]
            except ValueError:
                message = raw[:300]
            if error.code == 404:
                raise Unsupported(f"{path}: {message}") from None
            raise ValueError(f"{path} answered {error.code}: {message}") from None

    def card(self):
        """The served model's card from /v1/models: the one named, else the default, else the loaded one."""
        if self._card is None:
            cards = self.call("/v1/models").get("data") or []
            if not cards:
                raise ValueError("/v1/models lists no model")
            named = [c for c in cards if self.model and c.get("id") == self.model]
            self._card = (named or [c for c in cards if c.get("default")] or [c for c in cards if c.get("loaded")]
                          or cards)[0]
        return self._card

    def capabilities(self):
        return set(self.card().get("capabilities") or ())

    def facts(self):
        return self.card().get("engine_facts") or {}

    def context(self):
        return self.card().get("context_length") or self.card().get("max_model_len")

    def concurrency(self):
        return self.card().get("max_concurrent_requests") or 1

    def _model(self):
        return self.model or self.card().get("id") or "default"

    def tokenize(self, text, special=True):
        return list(self.call("/tokenize", {"model": self._model(), "prompt": text,
                                            "add_special_tokens": special})["tokens"])

    def render(self, messages):
        """The ids a chat request with these messages prefills, as the engine renders them."""
        return list(self.call("/tokenize", {"model": self._model(), "messages": messages})["tokens"])

    def score(self, prompt, continuation=(), top=5, prompt_rows=True):
        """(prompt rows, continuation rows): one row per prompt position after the first, from one prefill, and one
        per continuation token, decoded after it with every step forced to the continuation."""
        data = self.call("/v1/crit/score", {"model": self._model(), "prompt_ids": list(prompt),
                                            "continuation_ids": list(continuation), "top_logprobs": top,
                                            "prompt_rows": prompt_rows})
        return [_row(r) for r in data.get("prompt") or []], [_row(r) for r in data.get("continuation") or []]

    def chat(self, messages, max_tokens=32, top=5, plain=False):
        """A greedy reply's (token ids, rows, crit_telemetry): every generated token with its log-probability and
        alternatives, reasoning tokens included, the stop token left out."""
        body = {"model": self._model(), "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                "logprobs": True, "top_logprobs": top}
        if plain:
            body["crit"] = {"decode": "plain"}
        data = self.call("/v1/chat/completions", body)
        entries = ((data["choices"][0].get("logprobs") or {}).get("content")) or []
        rows = [(e["crit_token_id"], e["logprob"],
                 tuple((a["crit_token_id"], a["logprob"]) for a in e.get("top_logprobs") or ())) for e in entries]
        return [r[0] for r in rows], rows, data.get("crit_telemetry") or {}


def _row(row):
    token, logprob, alternatives = row
    return token, logprob, tuple((t, p) for t, p in alternatives)
