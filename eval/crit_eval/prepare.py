"""Freeze the suite as token ids for one model: `crit eval prepare --model DIR --data DIR`.

Every engine evaluated against the same data directory gets exactly the same prompts, rendered once with the
model's own chat template: an adapter's engine their token ids, an engine behind a server the chat messages
frozen next to them (crit_eval.chat). Tokenizer, chat template, stop ids and vocabulary come from Core's text
layer (crit_core.text), which needs the `tokenizers` and `jinja2` packages (preparation only).

Long prompts (suite/long.json) are frozen the same way: the long quality prompts at each of the protocol's
prompt lengths, and the pieces a run assembles long-context and retrieval prompts from once it knows the
required context (head, filler and fact token ids, and the text of every token for matching the answers).
"""
import hashlib
from itertools import count
import json
from pathlib import Path

from . import SUITE_DIR, protocol_version, write_json
from .ceiling import model_ceiling

PREPARED = "prepared.json"
LONG = "long.json"
SENTINEL = "\u0000DOCUMENT\u0000"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(model):
    """Hash of the files that define tokens and shapes, plus weight file sizes.
    Weight bytes are not hashed here: engines and crit run records already do that."""
    model = Path(model)
    parts = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json", "model.safetensors.index.json"):
        if (model / name).is_file():
            parts[name] = file_sha256(model / name)
    # The weight files beside the config and those the index lists, which may sit in a subfolder (tensors/...).
    names = {p.name for p in model.glob("*.safetensors")}
    index = model / "model.safetensors.index.json"
    if index.is_file():
        names |= {name for name in json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()
                  if isinstance(name, str) and (model / name).is_file()}
    parts["weight_files"] = {name: (model / name).stat().st_size for name in sorted(names)}
    blob = json.dumps(parts, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(blob).hexdigest(), "files": parts}


def suite_identity():
    digest = hashlib.sha256()
    # By name parts, case and all: Windows paths sort without case, which would give the suite another identity there.
    for path in sorted(SUITE_DIR.rglob("*"), key=lambda path: path.relative_to(SUITE_DIR).parts):
        if path.is_file():
            digest.update(path.relative_to(SUITE_DIR).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def check_prepared(prepared, protocol, data):
    """Data prepared under the same major protocol version from the current suite runs under any minor version."""
    found, current = prepared.get("protocol"), protocol["protocol"]
    try:
        same_major = protocol_version(found)[0] == protocol_version(current)[0]
    except ValueError:
        same_major = False
    if not same_major:
        raise SystemExit(f"{data} was prepared for {found}, not {current}: "
                         "prepare it again (crit eval prepare) into a new data directory")
    if prepared.get("suite_sha256") != suite_identity():
        raise SystemExit(f"{data} was prepared from another suite than {current}'s: "
                         "prepare it again (crit eval prepare) into a new data directory")


# `model` below is a crit_core.text.ModelText. Templates render in non-thinking mode where they support it.

def prompt_ids(model, messages, tools=None):
    """A generation prompt's token ids (ModelText.encode_chat)."""
    return model.encode_chat(messages, tools=tools, add_generation_prompt=True, enable_thinking=False)


def chat_text(model, messages, generation_prompt, tools=None):
    return model.render(messages, tools=tools, add_generation_prompt=generation_prompt, enable_thinking=False)


def token_ids(model, text):
    """Rendered text as token ids, encoded as ModelText.encode_chat encodes it."""
    return model.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=False)


def speed_messages(case, document=None):
    """The case's messages with content_file entries read from the suite and its tool definitions.
    document, when given, replaces {document} in the inline message contents (not in file contents)."""
    messages = []
    for message in case["messages"]:
        message = dict(message)
        if document is not None and "{document}" in (message.get("content") or ""):
            message["content"] = message["content"].replace("{document}", document)
        if "content_file" in message:
            message["content"] = (SUITE_DIR / message.pop("content_file")).read_text(encoding="utf-8")
        messages.append(message)
    tools = json.loads((SUITE_DIR / case["tools"]).read_text(encoding="utf-8")) if "tools" in case else None
    return messages, tools


def document_frame(model, messages, tools=None):
    """A generation prompt around a document: the token ids before and after SENTINEL in the messages' render."""
    head, tail = chat_text(model, messages, True, tools).split(SENTINEL)
    return token_ids(model, head), token_ids(model, tail)


def with_document(messages, document):
    return [{**m, "content": m["content"].replace(SENTINEL, document)} if SENTINEL in (m.get("content") or "") else m
            for m in messages]


def speed_prompt(model, case):
    """A speed case's prompt as (token ids, messages, tools): the messages and tools hold the text the ids render
    from, a long-context document cut to fit."""
    if "document" not in case:
        messages, tools = speed_messages(case)
        return prompt_ids(model, messages, tools), messages, tools
    # Exact-length long-context prompt: template head + document tokens cut to fit + template tail.
    # The document goes where an inline message content holds {document}.
    messages, tools = speed_messages(case, SENTINEL)
    head_ids, tail_ids = document_frame(model, messages, tools)
    document = (SUITE_DIR / case["document"]).read_text(encoding="utf-8")
    need = case["prompt_tokens"] - len(head_ids) - len(tail_ids)
    document_ids = token_ids(model, document)
    if need <= 0 or len(document_ids) < need:
        raise ValueError(f"{case['id']}: document has {len(document_ids)} tokens, needs {need}")
    return (head_ids + document_ids[:need] + tail_ids, with_document(messages, model.tokenizer.decode(document_ids[:need])),
            tools)


def curve_case(speed):
    """The speed case the long-context curve extends: the longest, which holds a document."""
    case = max(speed["cases"], key=lambda c: c.get("prompt_tokens", 0))
    if "document" not in case:
        raise ValueError(f"{case['id']}: the longest speed case has no document for the long-context curve to extend")
    return case


def curve_pieces(model, case):
    """What a run assembles long-context curve prompts from (context.curve_prompt): the case's token ids before
    and after its document, and the document followed by a paragraph break."""
    messages, tools = speed_messages(case, SENTINEL)
    head_ids, tail_ids = document_frame(model, messages, tools)
    document = (SUITE_DIR / case["document"]).read_text(encoding="utf-8")
    return {"case": case["id"], "head_ids": head_ids, "tail_ids": tail_ids,
            "document_ids": token_ids(model, document) + token_ids(model, "\n\n"),
            "messages": api_messages(messages), "tools": tools}


def speed_case(model, case):
    return speed_prompt(model, case)[0]


def heldout_messages(prompt):
    return [{"role": "user", "content": prompt["content"]}]


def api_messages(messages):
    """Suite messages in the chat API's form, as an engine behind a server is sent them (crit_eval.chat): tool-call
    arguments as JSON text, each tool result paired with its call by id."""
    out, ids, pending = [], count(), []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = [{"id": f"call_{next(ids)}", "type": "function",
                                      "function": {"name": call["function"]["name"],
                                                   "arguments": json.dumps(call["function"]["arguments"])}}
                                     for call in message["tool_calls"]]
            pending += [call["id"] for call in message["tool_calls"]]
        elif message["role"] == "tool":
            message["tool_call_id"] = pending.pop(0)
        out.append(message)
    return out


def speed_entry(model, case, output_tokens):
    ids, messages, tools = speed_prompt(model, case)
    return {"id": case["id"], "kind": case["kind"], "prompt_ids": ids, "messages": api_messages(messages),
            "tools": tools, "output_tokens": output_tokens}


def filler_ids(model, long):
    """The long-prompt filler: the suite files long.json names, joined by blank lines, as token ids."""
    return token_ids(model, "\n\n".join((SUITE_DIR / name).read_text(encoding="utf-8") for name in long["filler"]))


def repeated(ids, length):
    """The first `length` tokens of `ids` repeated as often as needed."""
    copies = -(-length // len(ids))
    return (list(ids) * copies)[:length]


def quality_prompt(model, prompt_id, set_name, messages):
    return {"id": prompt_id, "set": set_name, "prompt_ids": prompt_ids(model, messages), "messages": api_messages(messages)}


def long_quality_messages(model, conversation, preamble, filler, prompt_tokens):
    """A quality conversation whose generation prompt renders to about `prompt_tokens` tokens: long.json's preamble,
    holding filler cut to fit, goes before its first user message. The filler goes in as text, so the prompt's ids
    are its render's (prompt_ids), as a server renders it; the text can re-encode a token or two off the cut."""
    messages = [dict(m) for m in conversation["messages"]]
    first = next(i for i, m in enumerate(messages) if m["role"] == "user")
    messages[first]["content"] = preamble.replace("{document}", SENTINEL) + messages[first]["content"]
    head_ids, tail_ids = document_frame(model, messages)
    need = prompt_tokens - len(head_ids) - len(tail_ids)
    if need <= 0:
        raise ValueError(f"{conversation['id']}: {prompt_tokens} prompt tokens leave no room for the filler")
    return with_document(messages, model.tokenizer.decode(repeated(filler, need)))


def token_strings(model, count):
    """The text of each token id below `count` decoded on its own; None where it is not ASCII (part of a
    character, or non-Latin text). The retrieval answers are ASCII, so joining these spells them exactly."""
    out = []
    for token in range(count):
        try:
            text = model.tokenizer.decode([token])
        except Exception:  # an id the tokenizer cannot decode
            text = None
        out.append(text if isinstance(text, str) and text.isascii() else None)
    return out


def retrieval_messages(rule):
    return [{"role": "system", "content": rule["system"]},
            {"role": "user", "content": rule["user"].replace("{document}", SENTINEL)}]


def retrieval_pieces(model, long, filler):
    """What a run assembles retrieval prompts from (context.retrieval_prompt): the rendered prompt's token ids
    before and after the filler, the filler, each fact's token ids with its answer, and token_strings."""
    rule = long["retrieval"]
    head_ids, tail_ids = document_frame(model, retrieval_messages(rule))
    return {"head_ids": head_ids, "tail_ids": tail_ids, "messages": retrieval_messages(rule), "filler_ids": list(filler),
            "facts": [{"id": f["id"], "answer": f["answer"], "ids": token_ids(model, "\n\n" + f["text"] + "\n\n")}
                      for f in rule["facts"]],
            "token_text": token_strings(model, model.tokenizer.vocab_size)}


def quality_prompts(model, quality, long, filler, rule):
    """protocol quality: the suite's conversations as one set, then the long ones at each prompt length, a set each."""
    prompts = [quality_prompt(model, c["id"], "conversations", c["messages"]) for c in quality["conversations"]]
    by_id = {c["id"]: c for c in quality["conversations"]}
    for length in rule["long"]["prompt_tokens"]:
        for conversation_id in rule["long"]["conversations"]:
            messages = long_quality_messages(model, by_id[conversation_id], long["quality"]["preamble"], filler, length)
            prompts.append(quality_prompt(model, f"{conversation_id}@{length}", f"long-{length}", messages))
    return prompts


def prepare(model, data, speed_tokens=None):
    from crit_core.files import DirectoryFiles
    from crit_core.text import legacy_eos_ids, load_model_text
    protocol = json.loads((SUITE_DIR.parent / "protocol.json").read_text(encoding="utf-8"))
    files = DirectoryFiles(model)
    # No clock: templates use their own fallback date, so renders are reproducible.
    text = load_model_text(files, clock=None)
    speed = json.loads((SUITE_DIR / "speed.json").read_text(encoding="utf-8"))
    quality = json.loads((SUITE_DIR / "quality.json").read_text(encoding="utf-8"))
    long = json.loads((SUITE_DIR / LONG).read_text(encoding="utf-8"))
    filler = filler_ids(text, long)
    prepared = {
        "protocol": protocol["protocol"],
        "suite_sha256": suite_identity(),
        "model": model_identity(model),
        "stop_ids": legacy_eos_ids(files),
        # Token ids that exist; head rows beyond this are padding and excluded from scoring.
        "valid_vocab": text.tokenizer.vocab_size,
        "speed": [speed_entry(text, c, speed_tokens or protocol["speed"]["output_tokens"]) for c in speed["cases"]],
        "quality": quality_prompts(text, quality, long, filler, protocol["quality"]),
        # Arithmetic and KV bytes per token for the prefill and long-context decode floors.
        "ceiling": model_ceiling(model),
    }
    # Held-out prompts: outside the speed suite, never a tuning target (protocol "heldout").
    heldout = json.loads((SUITE_DIR / "heldout.json").read_text(encoding="utf-8"))
    prepared["heldout"] = [{"id": p["id"], "language": p["language"], "kind": p["kind"],
                            "prompt_ids": prompt_ids(text, heldout_messages(p)), "messages": heldout_messages(p),
                            "output_tokens": protocol["heldout"]["output_tokens"]} for p in heldout["prompts"]]
    prepared["retrieval"] = retrieval_pieces(text, long, filler)
    prepared["context"] = curve_pieces(text, curve_case(speed))
    Path(data).mkdir(parents=True, exist_ok=True)
    write_json(Path(data) / PREPARED, prepared)
    return prepared
