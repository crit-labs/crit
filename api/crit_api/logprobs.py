"""Token log-probabilities of a reply (chat-logprobs-v1): the request's options, the response record's bound and the
OpenAI logprobs content built from what the adapter reports (crit_core.serving.ResidentChat), also in the shape of
OpenAI's legacy completions (completion-v1)."""
import codecs
import re

from crit.run.records import MAX_JSON_BYTES, keys
from crit_core.serving import LOGPROBS_CAPABILITY, MIN_LOGPROB, TOP_LOGPROBS

from .options import MAX_CHAT_RECORD, RequestError, require

# Response record bytes per token and alternative beyond the reply itself: one indented [id, logprob] pair and the
# token's entry in the bytes table.
ROW_BYTES = 1024
# Resident commands whose result carries the engine's response record: a chat turn, a score (chat-score-v1), whose
# rows hold [id, logprob] pairs without a bytes table, and a tokenize (chat-tokenize-v1), at most MAX_CHAT_RECORD.
RESPONDING = ("generate", "score", "tokenize")
SCORE_PAIR_BYTES = 64
# An echoed prompt's text in the response record: a JSON-escaped character takes at most 6 bytes per UTF-8 byte, and
# a token's text at most the 1024 bytes the bytes table allows.
ESCAPED_BYTES, TOKEN_BYTES = 6, 1024


def logprob_options(body, capabilities):
    """The alternatives per token a request wants log-probabilities with (logprobs, top_logprobs), or None."""
    wanted, top = body.get("logprobs"), body.get("top_logprobs")
    if wanted is not None and type(wanted) is not bool:
        raise RequestError("logprobs must be a boolean", param="logprobs")
    if top is not None and (type(top) is not int or not 0 <= top <= TOP_LOGPROBS):
        raise RequestError(f"top_logprobs must be an integer from 0 to {TOP_LOGPROBS}", param="top_logprobs")
    if not wanted:
        if top:
            raise RequestError("top_logprobs requires logprobs true", param="top_logprobs")
        return None
    require(capabilities, LOGPROBS_CAPABILITY, "logprobs")
    return top or 0


def response_bytes(request):
    """The largest response record a request may produce."""
    if request.get("record_kind") == "tokenize_request":
        return MAX_CHAT_RECORD
    if request.get("record_kind") == "score_request":
        positions = len(request["prompt_ids"]) + len(request["continuation_ids"])
        return MAX_JSON_BYTES + positions * (request["top_logprobs"] + 2) * SCORE_PAIR_BYTES
    size = MAX_JSON_BYTES
    if request.get("logprobs") is not None:
        size += request["limits"]["max_output_tokens"] * (request["logprobs"] + 1) * ROW_BYTES
    if request.get("echo"):
        prompt = request["prompt"]
        size += ESCAPED_BYTES * (len(prompt["text"].encode("utf-8")) if "text" in prompt
                                 else TOKEN_BYTES * len(prompt["ids"]))
        if request.get("logprobs") is not None:
            size += request["limits"]["kv_tokens"] * (request["logprobs"] + 1) * ROW_BYTES
    return size


def logprob_entries(value, request):
    """OpenAI's logprobs content from a response's "logprobs": {"tokens": [[id, logprob, [[id, logprob], ...]],
    ...], "bytes": {id: hex}}, one row per generated token but stop tokens (a raw prompt's, completion-v1, keep
    them), each with exactly the requested number of distinct alternatives, ranked, and every id named by its
    bytes. Each entry and alternative keeps its id as crit_token_id. An echoed prompt's rows are its "prompt"
    (prompt_entries)."""
    keys(value, {"tokens", "bytes"} | ({"prompt"} if request.get("echo") else set()))
    rows = value["tokens"]
    if not isinstance(rows, list) or len(rows) > request["limits"]["max_output_tokens"]:
        raise ValueError("Reply logprobs need at most one row per output token")
    return _entries(rows, _named(value["bytes"]), request["logprobs"])


def prompt_entries(value, request):
    """The entries of an echoed prompt (completion-v1) from a response's logprobs "prompt": one row per prompt
    token, from the engine's score path, the first [id, null, []] since nothing precedes it, which becomes an entry
    with logprob and top_logprobs None. A prompt of token ids must be the one the request gave."""
    rows = value["prompt"]
    if (not isinstance(rows, list) or not rows or len(rows) > request["limits"]["kv_tokens"]
            or not isinstance(rows[0], list) or len(rows[0]) != 3 or rows[0][1:] != [None, []]):
        raise ValueError("Reply prompt logprobs are one row per prompt token, the first [token, null, []]")
    named = _named(value["bytes"])
    first = dict(_entries([[rows[0][0], 0, []]], named, 0)[0], logprob=None, top_logprobs=None)
    entries = [first] + _entries(rows[1:], named, request["logprobs"])
    if "ids" in request["prompt"] and [row[0] for row in rows] != request["prompt"]["ids"]:
        raise ValueError("Reply prompt logprobs name other tokens than the prompt")
    return entries


def _named(table):
    if not isinstance(table, dict):
        raise ValueError("Reply logprobs need the bytes of their tokens")
    named = {}
    for key, data in table.items():
        if (not re.fullmatch(r"0|[1-9][0-9]{0,9}", key) or not isinstance(data, str)
                or not re.fullmatch(r"(?:[0-9a-f]{2}){0,1024}", data)):
            raise ValueError("Reply logprob bytes map token ids to lowercase hex")
        named[int(key)] = bytes.fromhex(data)
    return named


def _entries(rows, named, alternatives):
    def entry(pair):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Reply logprobs pair each token with its log-probability")
        token, logprob = pair
        if type(token) is not int or token not in named:
            raise ValueError("Reply logprobs name a token without its bytes")
        if type(logprob) not in (int, float) or not MIN_LOGPROB <= logprob <= 0:
            raise ValueError(f"Reply log-probabilities must be numbers in [{MIN_LOGPROB}, 0]")
        data = named[token]
        return {"token": data.decode("utf-8", "replace"), "logprob": float(logprob), "bytes": list(data),
                "crit_token_id": token}

    content = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or not isinstance(row[2], list):
            raise ValueError("Reply logprob rows are [token, logprob, alternatives]")
        if len(row[2]) != alternatives:
            raise ValueError(f"Reply logprob rows need {alternatives} alternatives")
        top = [entry(pair) for pair in row[2]]
        if any(a["logprob"] < b["logprob"] for a, b in zip(top, top[1:])) or len({p[0] for p in row[2]}) != len(top):
            raise ValueError("Reply logprob alternatives must be distinct and ranked")
        content.append(dict(entry(row[:2]), top_logprobs=top))
    return content


def completion_logprobs(entries):
    """OpenAI's legacy completions logprobs (/v1/completions) for logprob_entries' entries: each token's text, its
    log-probability, its alternatives as {text: logprob} (the first of those with the same text kept; None where an
    echoed prompt starts) and its offset in the characters the tokens' bytes spell, which is the choice's text when
    the tokens spell it; a token ending inside a character is at that character's offset."""
    decoder = codecs.getincrementaldecoder("utf-8")("ignore")
    offsets, offset, top = [], 0, []
    for entry in entries:
        offsets.append(offset)
        offset += len(decoder.decode(bytes(entry["bytes"])))
        found = None if entry["top_logprobs"] is None else {}
        for pair in entry["top_logprobs"] or ():
            found.setdefault(pair["token"], pair["logprob"])
        top.append(found)
    return {"tokens": [e["token"] for e in entries], "token_logprobs": [e["logprob"] for e in entries],
            "top_logprobs": top, "text_offset": offsets}
