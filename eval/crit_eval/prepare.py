"""Freeze the suite as token ids for one model: `crit eval prepare --model DIR --data DIR`.

Engines never see text. Every engine evaluated against the same data directory receives
exactly the same prompt and corpus token ids, rendered once with the model's own chat
template. Tokenizer, chat template, stop ids and vocabulary come from Core's text layer
(crit_core.text), which needs the `tokenizers` and `jinja2` packages (preparation only).
"""
import hashlib
import json
from pathlib import Path

from . import SUITE_DIR, write_json
from .ceiling import model_ceiling

PREPARED = "prepared.json"


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
    for path in sorted(SUITE_DIR.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(SUITE_DIR).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


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


def speed_case(model, case):
    if "document" not in case:
        messages, tools = speed_messages(case)
        return prompt_ids(model, messages, tools)
    # Exact-length long-context prompt: template head + document tokens cut to fit + template tail.
    # The document goes where an inline message content holds {document}.
    sentinel = "\u0000DOCUMENT\u0000"
    messages, tools = speed_messages(case, sentinel)
    rendered = chat_text(model, messages, True, tools)
    head, tail = rendered.split(sentinel)
    head_ids, tail_ids = token_ids(model, head), token_ids(model, tail)
    document = (SUITE_DIR / case["document"]).read_text(encoding="utf-8")
    need = case["prompt_tokens"] - len(head_ids) - len(tail_ids)
    document_ids = token_ids(model, document)
    if need <= 0 or len(document_ids) < need:
        raise ValueError(f"{case['id']}: document has {len(document_ids)} tokens, needs {need}")
    return head_ids + document_ids[:need] + tail_ids


def quality_conversation(model, conversation):
    messages = conversation["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{conversation['id']}: the scored turn must be the final assistant message")
    prefix = prompt_ids(model, messages[:-1])
    full = token_ids(model, chat_text(model, messages, False).rstrip("\n"))
    if full[:len(prefix)] != prefix:
        raise ValueError(f"{conversation['id']}: the generation prompt is not a token prefix of the conversation")
    return full, len(prefix)


def prepare(model, data, speed_tokens=None):
    from crit_core.files import DirectoryFiles
    from crit_core.text import legacy_eos_ids, load_model_text
    protocol = json.loads((SUITE_DIR.parent / "protocol.json").read_text(encoding="utf-8"))
    files = DirectoryFiles(model)
    # No clock: templates use their own fallback date, so renders are reproducible.
    text = load_model_text(files, clock=None)
    speed = json.loads((SUITE_DIR / "speed.json").read_text(encoding="utf-8"))
    quality = json.loads((SUITE_DIR / "quality.json").read_text(encoding="utf-8"))
    prepared = {
        "protocol": protocol["protocol"],
        "suite_sha256": suite_identity(),
        "model": model_identity(model),
        "stop_ids": legacy_eos_ids(files),
        # Token ids that exist; head rows beyond this are padding and excluded from scoring.
        "valid_vocab": text.tokenizer.vocab_size,
        "speed": [{"id": c["id"], "kind": c["kind"], "prompt_ids": speed_case(text, c),
                   "output_tokens": speed_tokens or protocol["speed"]["output_tokens"]} for c in speed["cases"]],
        "quality": [],
        # Arithmetic and KV bytes per token for the prefill and long-context decode floors.
        "ceiling": model_ceiling(model),
    }
    for conversation in quality["conversations"]:
        ids, first = quality_conversation(text, conversation)
        prepared["quality"].append({"id": conversation["id"], "ids": ids, "first_target": first})
    # Held-out prompts: outside the speed suite, never a tuning target (protocol "heldout").
    heldout = json.loads((SUITE_DIR / "heldout.json").read_text(encoding="utf-8"))
    prepared["heldout"] = [{"id": p["id"], "language": p["language"], "kind": p["kind"],
                            "prompt_ids": prompt_ids(text, [{"role": "user", "content": p["content"]}]),
                            "output_tokens": protocol["heldout"]["output_tokens"]} for p in heldout["prompts"]]
    Path(data).mkdir(parents=True, exist_ok=True)
    write_json(Path(data) / PREPARED, prepared)
    return prepared
