"""Token log-probabilities of a reply (chat-logprobs-v1): the request's options, the response record's bound and the
OpenAI logprobs content built from what the adapter reports (crit_core.serving.ResidentChat)."""
import re

from crit.run.records import MAX_JSON_BYTES, keys
from crit_core.serving import LOGPROBS_CAPABILITY, MIN_LOGPROB, TOP_LOGPROBS

from .options import RequestError, require

# Response record bytes per token and alternative beyond the reply itself: one indented [id, logprob] pair and the
# token's entry in the bytes table.
ROW_BYTES = 1024


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
    if request.get("logprobs") is None:
        return MAX_JSON_BYTES
    return MAX_JSON_BYTES + request["limits"]["max_output_tokens"] * (request["logprobs"] + 1) * ROW_BYTES


def logprob_entries(value, request):
    """OpenAI's logprobs content from a response's "logprobs": {"tokens": [[id, logprob, [[id, logprob], ...]],
    ...], "bytes": {id: hex}}, one row per generated token but stop tokens, each with exactly the requested number
    of distinct alternatives, ranked, and every id named by its bytes."""
    keys(value, {"tokens", "bytes"})
    rows, table = value["tokens"], value["bytes"]
    if not isinstance(rows, list) or len(rows) > request["limits"]["max_output_tokens"]:
        raise ValueError("Reply logprobs need at most one row per output token")
    if not isinstance(table, dict):
        raise ValueError("Reply logprobs need the bytes of their tokens")
    named = {}
    for key, data in table.items():
        if (not re.fullmatch(r"0|[1-9][0-9]{0,9}", key) or not isinstance(data, str)
                or not re.fullmatch(r"(?:[0-9a-f]{2}){0,1024}", data)):
            raise ValueError("Reply logprob bytes map token ids to lowercase hex")
        named[int(key)] = bytes.fromhex(data)

    def entry(pair):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Reply logprobs pair each token with its log-probability")
        token, logprob = pair
        if type(token) is not int or token not in named:
            raise ValueError("Reply logprobs name a token without its bytes")
        if type(logprob) not in (int, float) or not MIN_LOGPROB <= logprob <= 0:
            raise ValueError(f"Reply log-probabilities must be numbers in [{MIN_LOGPROB}, 0]")
        data = named[token]
        return {"token": data.decode("utf-8", "replace"), "logprob": float(logprob), "bytes": list(data)}

    content = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or not isinstance(row[2], list):
            raise ValueError("Reply logprob rows are [token, logprob, alternatives]")
        if len(row[2]) != request["logprobs"]:
            raise ValueError(f"Reply logprob rows need {request['logprobs']} alternatives")
        top = [entry(pair) for pair in row[2]]
        if any(a["logprob"] < b["logprob"] for a, b in zip(top, top[1:])) or len({p[0] for p in row[2]}) != len(top):
            raise ValueError("Reply logprob alternatives must be distinct and ranked")
        content.append(dict(entry(row[:2]), top_logprobs=top))
    return content
