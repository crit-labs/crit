"""The fixed quality texts and their reference scores (protocol quality), read without torch.

Every quality text is its prompt_ids followed by its continuation_ids (crit_eval.prepare). `crit eval reference
--model --data` scores each text at every position once and stores the scores under the data (STORE): one
safetensors file per text (`target` [n - 1] float64, the log-probability of each id after the first; `top_ids`
[n - 1, top_k] int64 and `top_logprobs` [n - 1, top_k] float64, the reference's ranked top_k) and index.json,
written last, with the identity they were scored under (the reference's, top_k, valid_vocab and the sha256 of
every text's ids) and each file's sha256. A run reads them here, in pure Python.
"""
import array
import hashlib
import json
import struct
import sys
from pathlib import Path

STORE = "reference"
DTYPES = {"F64": "d", "F32": "f", "I64": "q", "I32": "i"}


def texts(prepared):
    """{text id: (ids, long)} of the prepared quality texts; the long ones use the reference's chunked attention."""
    return {t["id"]: (list(t["prompt_ids"]) + list(t["continuation_ids"]), t["set"] != "conversations")
            for t in prepared["quality"]}


def identity(texts_by_id, top_k, valid_vocab):
    """What the stored scores depend on besides the reference itself."""
    return {"top_k": top_k, "valid_vocab": valid_vocab,
            "texts": {text_id: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                      for text_id, (ids, _) in texts_by_id.items()}}


def read_safetensors(data):
    """{name: (shape, values)} of a safetensors file's bytes, values flat in row-major order."""
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + size])
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        values = array.array(DTYPES[info["dtype"]])
        values.frombytes(data[8 + size + begin:8 + size + end])
        if sys.byteorder != "little":
            values.byteswap()
        out[name] = (info["shape"], values)
    return out


def text_file(text_id):
    return f"text-{hashlib.sha256(text_id.encode()).hexdigest()[:16]}.safetensors"


def safetensors(tensors, metadata=None):
    """The bytes of a safetensors file of {name: (dtype, shape, values)}, values flat and row-major."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, values) in tensors.items():
        data = array.array(DTYPES[dtype], values)
        if sys.byteorder != "little":
            data.byteswap()
        blob = data.tobytes()
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs)


def write(data, identity_value, scores):
    """Store scores ({text id: (target [n - 1], top ids [n - 1][k], top logprobs [n - 1][k])}, sequences of
    numbers) under prepared data folder `data`: one file per text, then index.json with `identity_value` and each
    file's sha256. Returns the bytes written."""
    folder = Path(data) / STORE
    folder.mkdir(parents=True, exist_ok=True)
    files, size = {}, 0
    for text_id, (target, top, values) in scores.items():
        k = len(top[0]) if len(top) else 0
        raw = safetensors({"target": ("F64", [len(target)], target),
                           "top_ids": ("I64", [len(top), k], [t for row in top for t in row]),
                           "top_logprobs": ("F64", [len(values), k], [v for row in values for v in row])},
                          {"text": text_id})
        (folder / text_file(text_id)).write_bytes(raw)
        files[text_id] = [text_file(text_id), hashlib.sha256(raw).hexdigest()]
        size += len(raw)
    temporary = folder / "index.json.tmp"
    temporary.write_text(json.dumps({"identity": identity_value, "files": files}, indent=1), encoding="utf-8")
    temporary.replace(folder / "index.json")
    return size


def load(data, expected):
    """The reference scores stored under prepared data folder `data` whose identity holds every key of
    `expected`: {text id: [{"logprob", "top_ids", "top_logprobs"} per position after the first]}; None when the
    store is missing or another identity's. A file whose bytes differ from its recorded sha256 is refused."""
    folder = Path(data) / STORE
    index = folder / "index.json"
    if not index.is_file():
        return None
    stored = json.loads(index.read_text(encoding="utf-8"))
    found = stored.get("identity") or {}
    if any(found.get(key) != value for key, value in expected.items()):
        return None
    out = {}
    for text_id, (name, sha) in stored["files"].items():
        raw = (folder / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit(f"{folder / name}: its bytes differ from the sha256 recorded in index.json")
        tensors = read_safetensors(raw)
        (count,), target = tensors["target"]
        (_, k), ids = tensors["top_ids"]
        _, values = tensors["top_logprobs"]
        out[text_id] = [{"logprob": target[i], "top_ids": list(ids[i * k:(i + 1) * k]),
                         "top_logprobs": list(values[i * k:(i + 1) * k])} for i in range(count)]
    return out
