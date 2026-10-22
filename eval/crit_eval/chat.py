"""Eval's transport for an engine behind an OpenAI-compatible server: `crit eval run --url URL`.

A chat API takes messages and returns text. Every prompt Eval sends is chat messages that prepare froze next to its
token ids, and the server applies its own chat template. A long-context or retrieval prompt is its frame's messages
with the document between the frame's head and tail decoded into them. Prompt and output lengths are the server's
usage counts; the reply's text, encoded with the model's tokenizer, stands in for its token ids in the repeat,
history and retrieval checks. No standard field turns end tokens off, so a reply that ends before the case's output
length is measured on the tokens it has (protocol speed.served).

Quality texts go to crit serve's score call as token ids (chat-score-v1, POST /v1/crit/score). The honesty
check's replies come with logprobs and top_logprobs (protocol quality.honesty); each reported token is mapped to
the model's id by its bytes, so the same tokens can be scored again after exactly the prompt the server rendered:
its usage must count the prompt ids the model's chat template renders.

The server must stream usage (stream_options.include_usage) and prefill every timed request: a reply whose usage
reports cached prompt tokens fails, since a prompt cache would make prefill and time to first token meaningless.
Requests judged on their text alone (retrieval) may be served from a prompt cache (prompt_cache), and their
cached tokens are recorded. A key for
the server goes in the CRIT_SERVER_KEY environment variable.

Each request may take as long as the engine's own earlier requests say it should (crit_eval.deadline, protocol
requests.timeout); one that takes longer raises Hung.
"""
import json
import os
import socket
import time
import urllib.error
import urllib.request

from . import EVAL_DIR, load_json
from .deadline import Hung, RequestDeadline
from .prepare import token_ids, with_document


class Server:
    """An engine behind a chat API at `url`, opened as an adapter opens one (create_eval_engine). model_name: the
    model to request; without it, the one /models lists."""

    def __init__(self, url, prepared, model_name=None):
        self.url, self.model_name = url.rstrip("/"), model_name
        self.timeout_rule = load_json(EVAL_DIR / "protocol.json")["requests"]["timeout"]
        # The server is reached directly: a proxy would add its own latency to every token.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        key = os.environ.get("CRIT_SERVER_KEY")
        self.headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        cases = prepared["speed"] + prepared.get("heldout", []) + prepared.get("quality", []) + prepared.get("followup", [])
        self.prompts = {tuple(case["prompt_ids"]): (case["messages"], case.get("tools")) for case in cases}
        self.frames = [(p["head_ids"], p["tail_ids"], p["messages"], p.get("tools"))
                       for p in (prepared.get("context"), prepared.get("retrieval")) if p]
        context = prepared.get("context")
        if context and context.get("retrieval_tail_ids"):
            self.frames.append((context["head_ids"], context["retrieval_tail_ids"], context["retrieval_messages"],
                                context.get("tools")))
        # The longest tail first: frames that share a head and end alike are told apart by their whole tail.
        self.frames.sort(key=lambda frame: -len(frame[1]))

    def get(self, path, timeout):
        request = urllib.request.Request(self.url + path, headers=self.headers)
        with _bounded(timeout, path), self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def post(self, path, body, timeout):
        return self.post_timed(path, body, timeout)[0]

    def post_timed(self, path, body, timeout):
        """(the response, {"seconds": until it was read, "response_bytes", "parse_seconds"})."""
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers=self.headers)
        started = time.monotonic()
        with _bounded(timeout, path), self.opener.open(request, timeout=timeout) as response:
            raw = response.read()
        read = time.monotonic()
        value = json.loads(raw)
        return value, {"seconds": read - started, "response_bytes": len(raw), "parse_seconds": time.monotonic() - read}

    def events(self, path, body, timeout):
        """The streamed events of a request; `timeout` bounds the wait for each."""
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers=self.headers)
        with _bounded(timeout, path), self.opener.open(request, timeout=timeout) as response:
            for line in response:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                yield json.loads(data)

    def create_eval_engine(self, context):
        return ChatEngine(self, context)


class _bounded:
    """Turns a request's socket timeout into Hung."""

    def __init__(self, timeout, path):
        self.timeout, self.path = timeout, path

    def __enter__(self):
        return self

    def __exit__(self, kind, error, _traceback):
        timed_out = isinstance(error, (socket.timeout, TimeoutError)) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, (socket.timeout, TimeoutError)))
        if timed_out:
            raise Hung(f"{self.path} gave no answer within {self.timeout:,.0f} s") from error
        return False


def load_text(model):
    """The model's tokenizer and chat template, loaded as prepare loads them."""
    from crit_core.model import open_model
    from crit_core.text import load_model_text
    return load_model_text(open_model(model).files, clock=None)


def ids_by_bytes(tokenizer):
    """{bytes: token id} for mapping the tokens a server reports to the model's ids (bytes from
    Tokenizer.token_bytes). Where ids share their bytes, an ordinary piece wins over a SentencePiece byte piece and
    both over an added token. A token that reads as U+FFFD maps to nothing: servers show a token that ends inside a
    character that way, so its bytes are lost."""
    pieces = tokenizer.token_bytes()

    def rank(token):
        piece = tokenizer.id_to_token(token)
        return (token in tokenizer.added_tokens, len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"))
    out = {}
    for token in sorted(range(len(pieces)), key=rank):
        if pieces[token] and pieces[token] != "\ufffd".encode():
            out.setdefault(pieces[token], token)
    return out


def cached_tokens(usage):
    """The prompt tokens a reply's usage reports as served from a prompt cache."""
    return int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def prefilled(server, usage):
    """Fail a reply whose usage reports prompt tokens served from a cache."""
    cached = cached_tokens(usage)
    if cached:
        raise RuntimeError(f"{server.url} served {cached} prompt tokens from its cache: turn its prompt cache off")


def reply_text(delta):
    """What one streamed delta adds to the reply: every text field but the role, and tool-call names and arguments."""
    parts = [value for key, value in delta.items() if key != "role" and isinstance(value, str)]
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        parts += [function.get("name") or "", function.get("arguments") or ""]
    return "".join(parts)


class ChatEngine:
    def __init__(self, server, context):
        self.server = server
        if not context.get("context_tokens"):
            raise SystemExit("an engine behind a server needs --context: the context it was started with")
        self.max_context = int(context["context_tokens"])
        self.deadline = RequestDeadline(server.timeout_rule)
        listed = [m.get("id") for m in server.get("/models", self.deadline.seconds(0)).get("data") or []]
        if server.model_name and server.model_name not in listed:
            raise SystemExit(f"{server.url}/models does not list {server.model_name}: it lists {', '.join(listed) or 'none'}")
        if not server.model_name and len(listed) != 1:
            raise SystemExit(f"{server.url}/models lists {', '.join(listed) or 'no model'}: name one with --served-model")
        self.model = server.model_name or listed[0]
        self.text = load_text(context["model"])
        self.by_bytes = None  # ids_by_bytes, built for the first quality reply

    def info(self):
        return {"name": self.model, "configuration": {"url": self.server.url, "model": self.model},
                "max_context": self.max_context, "speculative": False, "served": True}

    def messages(self, prompt):
        """The messages and tools a prompt renders from: a speed or held-out prompt's own, or a frame's with the
        document between its head and tail decoded into them."""
        found = self.server.prompts.get(tuple(prompt))
        if found is not None:
            return found
        for head, tail, messages, tools in self.server.frames:  # the curve, retrieval and shared-prefix frames
            end = len(prompt) - len(tail)
            if end > len(head) and prompt[:len(head)] == head and prompt[end:] == tail:
                return with_document(messages, self.text.tokenizer.decode(prompt[len(head):end])), tools
        raise RuntimeError("the prompt is not one the prepared data sends to a server")

    def generate(self, prompt, max_new_tokens, on_tokens, prompt_cache=False):
        """A greedy reply; with prompt_cache it may be served from the server's prompt cache (a request judged on
        its text alone), and served records the cached prompt tokens."""
        prompt = list(prompt)
        messages, tools = self.messages(prompt)
        body = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}, **({"tools": tools} if tools else {})}
        pieces, usage, finish, telemetry = [], None, None, None
        started = time.monotonic()
        first = last = None
        for event in self.server.events("/chat/completions", body, self.deadline.seconds(len(prompt), max_new_tokens)):
            telemetry = event.get("crit_telemetry") or telemetry
            if event.get("error"):
                # A server that fails mid-stream sends its error as an event; say so instead of "no usage".
                error = event["error"]
                raise RuntimeError(f"{self.server.url} streamed an error: "
                                   f"{error.get('message') if isinstance(error, dict) else error}")
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                piece = reply_text(choice.get("delta") or {})
                if piece:
                    last = time.monotonic()
                    first = first or last
                    pieces.append(piece)
                    on_tokens(token_ids(self.text, piece))
        if not usage or not usage.get("completion_tokens"):
            raise RuntimeError(f"{self.server.url} streamed no usage: it must support stream_options.include_usage")
        if not prompt_cache:
            prefilled(self.server, usage)
        if not pieces:
            raise RuntimeError(f"{self.server.url} streamed no reply text")
        self.deadline.observe(usage["prompt_tokens"], first - started, usage["completion_tokens"], last - first)
        # crit serve reports output tokens per target pass after prefill when the engine counts its rounds.
        per_round = (telemetry or {}).get("tokens_per_round")
        rounds = (round(usage["completion_tokens"] / per_round)
                  if isinstance(per_round, (int, float)) and per_round > 0 else None)
        return {"tokens": token_ids(self.text, "".join(pieces)), "rounds": rounds,
                "served": {"prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                           "finish_reason": finish, "cached_tokens": cached_tokens(usage)}}

    def reply(self, prompt, max_new_tokens, top_k):
        """A quality prompt's greedy reply with the log-probabilities the server reports for each token (protocol
        quality.rule): {"tokens", "logprobs", "top": [[id, logprob], ...] per token, "finish_reason",
        "prompt_tokens"}. An alternative whose bytes are no token of the model is left out of "top"."""
        prompt = list(prompt)
        messages, tools = self.messages(prompt)
        url = self.server.url
        response, timing = self.server.post_timed("/chat/completions", {
            "model": self.model, "messages": messages, "max_tokens": max_new_tokens, "temperature": 0,
            "logprobs": True, "top_logprobs": top_k, **({"tools": tools} if tools else {})},
            self.deadline.seconds(len(prompt), max_new_tokens))
        # A whole reply's time bounds its prefill from above.
        self.deadline.observe(len(prompt), timing["seconds"])
        usage = response.get("usage") or {}
        if usage.get("prompt_tokens") != len(prompt):
            raise RuntimeError(f"{url} rendered the prompt as {usage.get('prompt_tokens')} tokens where the model's chat "
                               f"template renders {len(prompt)}: serve the model's own chat template, thinking off")
        prefilled(self.server, usage)
        choice = (response.get("choices") or [{}])[0]
        entries, finish = (choice.get("logprobs") or {}).get("content"), choice.get("finish_reason")
        if not entries:
            raise RuntimeError(f"{url} returned no logprobs: it must support logprobs and top_logprobs")
        if self.by_bytes is None:
            self.by_bytes = ids_by_bytes(self.text.tokenizer)
        tokens, logprobs, top = [], [], []
        for position, entry in enumerate(entries):
            data = entry.get("bytes")
            if data is None:
                raise RuntimeError(f"{url} reports no bytes for reply token {position} ({entry.get('token')!r}): "
                                   "Eval maps each reply token to the model's ids by its bytes")
            if not data and position == len(entries) - 1 and finish == "stop":
                break  # the end token, which a server may report as no text
            token = self.by_bytes.get(bytes(data))
            if token is None:
                raise RuntimeError(f"{url} reports reply token {position} as {bytes(data)!r}, which is no token of the model")
            alternatives = entry.get("top_logprobs") or []
            if len(alternatives) < top_k:
                raise RuntimeError(f"{url} returned {len(alternatives)} top_logprobs for reply token {position}, not {top_k}")
            row = {}
            for alternative in alternatives:
                other = self.by_bytes.get(bytes(alternative.get("bytes") or b""))
                if other is not None:
                    row.setdefault(other, float(alternative["logprob"]))
            tokens.append(token)
            logprobs.append(float(entry["logprob"]))
            top.append([[other, value] for other, value in row.items()])
        if not tokens:
            raise RuntimeError(f"{url} returned an empty reply")
        return {"tokens": tokens, "logprobs": logprobs, "top": top, "finish_reason": finish,
                "prompt_tokens": usage["prompt_tokens"], "timing": timing}

    def score(self, prompt, continuation, top_k, prompt_rows=True):
        """Fixed text scored through the engine's serving paths (crit serve's chat-score-v1, POST /v1/crit/score):
        {"prompt": rows for prompt[1:] from its prefill, "continuation": rows for each continuation token from its
        decode forced to the continuation (none with prompt_rows false), "timing"}, rows {"logprob",
        "top": [[id, logprob], ...]}; timing: the
        request's seconds, the engine's own prefill_ms and decode_ms (crit_telemetry), the response's bytes and the
        seconds parsing it took."""
        try:
            value, timing = self.server.post_timed(
                "/crit/score", {"model": self.model, "prompt_ids": list(prompt), "continuation_ids": list(continuation),
                                "top_logprobs": top_k, "prompt_rows": prompt_rows},
                self.deadline.seconds(len(prompt), len(continuation)))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code == 404:
                raise RuntimeError(f"{self.server.url} scores no fixed text: serve the engine with crit serve and a "
                                   "package declaring chat-score-v1") from exc
            raise RuntimeError(f"{self.server.url} failed to score fixed text ({exc.code}): {detail}") from exc
        self.deadline.observe(len(prompt) + len(continuation), timing["seconds"])
        converted = time.monotonic()
        rows = {}
        for part, tokens in (("prompt", list(prompt)[1:] if prompt_rows else []), ("continuation", list(continuation))):
            found = value.get(part)
            if not isinstance(found, list) or [row[0] for row in found] != tokens:
                raise RuntimeError(f"{self.server.url} scored other {part} tokens than it was sent")
            if any(len(row[2]) != top_k for row in found):
                raise RuntimeError(f"{self.server.url} returned other than {top_k} alternatives per {part} token")
            rows[part] = [{"logprob": float(row[1]), "top": [[int(t), float(v)] for t, v in row[2]]} for row in found]
        engine = value.get("crit_telemetry") or {}
        rows["timing"] = {**timing, "parse_seconds": timing["parse_seconds"] + time.monotonic() - converted,
                          "engine_prefill_ms": engine.get("prefill_ms"), "engine_decode_ms": engine.get("decode_ms")}
        return rows

    def close(self):
        pass
