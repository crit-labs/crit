"""Freeze the suite as token ids for one model: `crit eval prepare --model DIR --data DIR`.

Every engine evaluated against the same data directory gets exactly the same prompts, rendered once with the
model's own chat template: an adapter's engine their token ids, an engine behind a server the chat messages
frozen next to them (crit_eval.chat). Tokenizer, chat template, stop ids and vocabulary come from Core's text
layer (crit_core.text), which needs the `tokenizers` and `jinja2` packages (preparation only).

Every quality text is a conversation's generation prompt followed by its written answer (continuation_ids, cut to
the protocol's continuation_tokens), which the engine and the reference score at every position.

Long prompts (suite/long.json) are frozen the same way: the long quality prompts at each of the protocol's
prompt lengths, and the pieces a run assembles long-context and retrieval prompts from once it knows the
required context (head, filler and fact token ids, and the text of every token for matching the answers).
"""
import hashlib
from itertools import count
import json
from pathlib import Path

from crit_core.model import open_model
from crit_core.model.gguf import PROJECTOR

from . import SUITE_DIR, protocol_version, write_json
from .ceiling import model_ceiling

PREPARED = "prepared.json"
LONG = "long.json"
SENTINEL = "\u0000DOCUMENT\u0000"


def model_identity(model):
    """Hash of the files that define tokens and shapes, plus weight file sizes. The files are the model's own or,
    for a GGUF without them, the ones its metadata derives (crit_core.model.open_model).
    Weight bytes are not hashed here: engines and crit run records already do that."""
    source = open_model(model)
    files = source.files
    parts = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if name in source.derived or (source.folder is not None and (source.folder / name).is_file()):
            parts[name] = hashlib.sha256(files.read(name)).hexdigest()
    folder = source.folder
    if source.format in ("torch", "onnx"):
        # The files the weights are in: the state dicts, or the graph and its external data files.
        names = {}
        for shard in source.weight_files(drafts=True):
            for path in {shard.path, *(tensor.shard for tensor in shard.tensors)}:
                names[path.name] = path.stat().st_size
    elif folder is None:   # a lone weight file (and the other parts of its split set)
        names = {shard.path.name: shard.size for shard in source.weight_files(drafts=True)}
    else:
        # The weight files beside the config and those the index lists, which may sit in a subfolder
        # (tensors/...), or the GGUF parts (not a multimodal projector beside them).
        found = {p.name for p in folder.glob("*.safetensors")}
        found |= {p.name for p in folder.glob("*.gguf") if not PROJECTOR.search(p.stem)}
        index = folder / "model.safetensors.index.json"
        if index.is_file():
            found |= {name for name in json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()
                      if isinstance(name, str) and (folder / name).is_file()}
        names = {name: (folder / name).stat().st_size for name in found}
    parts["weight_files"] = dict(sorted(names.items()))
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


def quality_prompt(model, prompt_id, set_name, messages, continuation, tokens):
    """A quality text: the conversation's generation prompt (prompt_ids, and its messages for a server), followed by
    its written answer encoded as the model's tokens, cut to `tokens` (continuation_ids)."""
    return {"id": prompt_id, "set": set_name, "prompt_ids": prompt_ids(model, messages),
            "continuation_ids": token_ids(model, continuation)[:tokens], "messages": api_messages(messages)}


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


def followup_prompts(model, followup, filler, prompt_tokens):
    """protocol followup: every turn's prompt, the conversation so far with the suite's fixed assistant replies and
    the turn's user message; the first user message holds the filler, cut so the first turn renders to about
    prompt_tokens tokens."""
    first = [{"role": "system", "content": followup["system"]},
             {"role": "user", "content": followup["turns"][0]["user"].replace("{document}", SENTINEL)}]
    head, tail = document_frame(model, first)
    need = prompt_tokens - len(head) - len(tail)
    if need < 1:
        raise ValueError(f"followup: {prompt_tokens} prompt tokens cannot hold its first turn")
    document = model.tokenizer.decode(repeated(filler, need))
    messages = with_document(first, document)
    prompts = []
    for number, turn in enumerate(followup["turns"], 1):
        if number > 1:
            messages = messages + [{"role": "assistant", "content": turn["assistant"]},
                                   {"role": "user", "content": turn["user"]}]
        prompts.append({"turn": number, "prompt_ids": prompt_ids(model, messages), "messages": api_messages(messages)})
    return prompts


def retrieval_messages(rule, user="user"):
    return [{"role": "system", "content": rule["system"]},
            {"role": "user", "content": rule[user].replace("{document}", SENTINEL)}]


def retrieval_pieces(model, long, filler):
    """What a run assembles retrieval prompts from (context.retrieval_prompt): the rendered prompt's token ids
    before and after the filler, the filler, each fact's token ids with its answer, and token_strings. The
    long-context curve's prompts (context.curve_prompt) end with curve_tail_ids instead, its request asking for the
    facts and then for a summary of the archive (curve_messages), after the same head."""
    rule = long["retrieval"]
    head_ids, tail_ids = document_frame(model, retrieval_messages(rule))
    curve_head, curve_tail = document_frame(model, retrieval_messages(rule, "curve_user"))
    if curve_head != head_ids:
        raise ValueError("long.json's retrieval user and curve_user must open alike: the chat template renders their "
                         "prompts with another start")
    return {"head_ids": head_ids, "tail_ids": tail_ids, "messages": retrieval_messages(rule), "filler_ids": list(filler),
            "curve_tail_ids": curve_tail, "curve_messages": api_messages(retrieval_messages(rule, "curve_user")),
            "facts": [{"id": f["id"], "answer": f["answer"], "ids": token_ids(model, "\n\n" + f["text"] + "\n\n")}
                      for f in rule["facts"]],
            "token_text": token_strings(model, model.tokenizer.vocab_size)}


def quality_prompts(model, quality, long, filler, rule):
    """protocol quality: the suite's conversations as one set, then the long ones at each prompt length, a set each."""
    tokens = rule["continuation_tokens"]
    prompts = [quality_prompt(model, c["id"], "conversations", c["messages"], c["continuation"], tokens)
               for c in quality["conversations"]]
    by_id = {c["id"]: c for c in quality["conversations"]}
    for length in rule["long"]["prompt_tokens"]:
        for conversation_id in rule["long"]["conversations"]:
            messages = long_quality_messages(model, by_id[conversation_id], long["quality"]["preamble"], filler, length)
            prompts.append(quality_prompt(model, f"{conversation_id}@{length}", f"long-{length}", messages,
                                          by_id[conversation_id]["continuation"], tokens))
    return prompts


def sampled_settings(generation):
    """protocol sampled_speed.sampling: the model's own sampling defaults (a crit_core.text GenerationConfig) when it
    samples by default, temperature 1.0 with no filters when it does not. Penalties stay neutral."""
    defaults = generation.sampling_defaults()
    if not defaults.get("do_sample"):
        return {"temperature": 1.0}
    settings = {"temperature": float(defaults.get("temperature", 1.0))}
    for key in ("top_p", "top_k", "min_p"):
        if defaults.get(key) is not None:
            settings[key] = defaults[key]
    if settings["temperature"] <= 0:
        settings = {"temperature": 1.0}
    return settings


SAMPLING_KEYS = {"temperature": (float, 0.0, 100.0), "top_p": (float, 0.0, 1.0), "top_k": (int, 1, 1 << 20),
                 "min_p": (float, 0.0, 1.0)}


def sampling_override(value):
    """A user's sampling for the sampled suite in place of the model's own: {"temperature": > 0 (required), "top_p":
    (0, 1], "top_k": >= 1, "min_p": [0, 1)}, or the text "temperature=0.6,top_p=0.95,top_k=20". Penalties stay
    neutral. Raises ValueError on anything else."""
    if isinstance(value, str):
        pairs = [part.split("=", 1) for part in value.split(",") if part.strip()]
        if not all(len(pair) == 2 for pair in pairs):
            raise ValueError("sampling requires name=value pairs separated by commas, e.g. temperature=0.6,top_p=0.95")
        value = {name.strip(): number.strip() for name, number in pairs}
    if not isinstance(value, dict) or set(value) - set(SAMPLING_KEYS) or "temperature" not in value:
        raise ValueError(f"sampling takes temperature (required) and any of top_p, top_k, min_p; got {value}")
    settings = {}
    for name, raw in value.items():
        kind, low, high = SAMPLING_KEYS[name]
        try:
            number = kind(raw) if not isinstance(raw, bool) else None
        except (TypeError, ValueError):
            number = None
        if kind is int and number is not None and float(raw) != number:
            number = None
        if number is None or not low <= number <= high or (name in ("temperature", "top_p") and number <= 0) or (
                name == "min_p" and number >= 1):
            raise ValueError(f"sampling {name} {raw!r} is out of range")
        settings[name] = number
    return dict(sorted(settings.items(), key=lambda item: list(SAMPLING_KEYS).index(item[0])))


# Every sampling field a sampled request names, at the value that leaves the distribution as the model's own
# settings make it: filters off and penalties neutral (top_k 0 keeps every token).
NEUTRAL_PENALTIES = {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
NEUTRAL_SAMPLING = {"top_p": 1.0, "top_k": 0, "min_p": 0.0, **NEUTRAL_PENALTIES}
# Every greedy request: temperature zero and the penalties neutral, which an engine's served defaults (a model card's
# presence penalty, say) would otherwise change, sending the request through another sampler.
GREEDY_REQUEST = {"temperature": 0, **NEUTRAL_PENALTIES}


def sampled_request(settings, seed):
    """The sampling fields of one sampled request (protocol sampled_speed): the prepared settings, every filter they
    leave unset and every penalty at its neutral value, and the seed, so no engine default changes the distribution
    (an engine applies its own defaults to the fields a request leaves out)."""
    return {"temperature": float(settings["temperature"]), **NEUTRAL_SAMPLING,
            **{key: settings[key] for key in ("top_p", "top_k", "min_p") if settings.get(key) is not None},
            "seed": int(seed)}


def unapplied(request, supported):
    """The fields of a sampled request an engine that applies `supported` cannot honour: those whose value is not
    neutral (temperature always counts; a missing seed only costs repeatability). Fields at their neutral value may
    be left out for such an engine."""
    return sorted(key for key, value in request.items()
                  if key not in supported and key != "seed"
                  and (key not in NEUTRAL_SAMPLING or value != NEUTRAL_SAMPLING[key]))


def sampling_source(generation):
    """Where the model's sampled-suite settings come from: its generation_config.json when it samples by default with a
    positive temperature, else the protocol's default (temperature 1.0, no filters)."""
    defaults = generation.sampling_defaults()
    if defaults.get("do_sample") and float(defaults.get("temperature", 1.0)) > 0:
        return "generation_config"
    return "protocol_default"


def prepare(model, data, speed_tokens=None, sampling=None):
    """sampling: the user's override of the sampled suite's settings (sampling_override); None takes the model's."""
    from crit_core.text import legacy_eos_ids, load_model_text
    protocol = json.loads((SUITE_DIR.parent / "protocol.json").read_text(encoding="utf-8"))
    files = open_model(model).files
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
        # The sampling the sampled speed suite and the sampling check run at (protocol sampled_speed.sampling), and
        # where it came from: the model's generation_config.json, the protocol default, or the user's override.
        "sampling": sampling_override(sampling) if sampling is not None else sampled_settings(text.generation),
        "sampling_source": "override" if sampling is not None else sampling_source(text.generation),
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
    followup = json.loads((SUITE_DIR / "followup.json").read_text(encoding="utf-8"))
    prepared["followup"] = followup_prompts(text, followup, filler, protocol["followup"]["prompt_tokens"])
    Path(data).mkdir(parents=True, exist_ok=True)
    write_json(Path(data) / PREPARED, prepared)
    return prepared
