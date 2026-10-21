"""Calibration data: a JSONL file of {"text": ...} documents, cut into fixed token windows by the recipe's seed.

Calibration text must be unrelated to Eval's suite: the quality references and held-out prompts judge the result,
so a document that contains suite text is rejected (a 64-character excerpt check over the whole suite).
"""
import json
from pathlib import Path
import random
import re

from crit.run.records import digest, file_sha

MAX_BYTES = 1 << 30
EXCERPT = 64


def read_documents(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path}: calibration data must be a regular JSONL file under 1 GiB")
    documents = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from None
            if not isinstance(value, dict) or not isinstance(value.get("text"), str) or not value["text"].strip():
                raise ValueError(f'{path}:{number}: each line must be an object with nonempty "text"')
            documents.append(value["text"])
    if not documents:
        raise ValueError(f"{path}: no documents")
    return documents


def _normal(text):
    return re.sub(r"\s+", " ", text).strip()


def suite_texts():
    """Every text Eval's suite scores or prompts with, or None when the suite is not on this host."""
    try:
        from crit_eval import SUITE_DIR
    except ImportError:
        return None
    if not SUITE_DIR.is_dir():
        return None
    texts = []

    def walk(value):
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for path in sorted(SUITE_DIR.rglob("*")):
        if path.suffix == ".json":
            walk(json.loads(path.read_text(encoding="utf-8")))
        elif path.is_file():
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return [t for t in (_normal(t) for t in texts) if len(t) >= EXCERPT]


def suite_overlap(documents, texts):
    """Documents (by index) that contain an excerpt of a suite text: its start, middle or end."""
    excerpts = {t[i:i + EXCERPT] for t in texts for i in (0, (len(t) - EXCERPT) // 2, len(t) - EXCERPT)}
    found = []
    for index, document in enumerate(documents):
        normal = _normal(document)
        if any(e in normal for e in excerpts):
            found.append(index)
    return found


def windows(documents, encode, separator, samples, sequence_tokens, seed):
    """Shuffle documents with the seed, join their tokens (separator between documents) and cut the stream into
    `samples` windows of `sequence_tokens`. Raises when the data is too short."""
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    stream, need = [], samples * sequence_tokens
    for index in order:
        stream += encode(documents[index]) + separator
        if len(stream) >= need:
            break
    if len(stream) < need:
        raise ValueError(f"calibration data holds {len(stream)} tokens; the recipe needs {samples} x {sequence_tokens}")
    return [stream[k * sequence_tokens:(k + 1) * sequence_tokens] for k in range(samples)]


def load_calibration(path, model, recipe):
    """Token windows for a gptq recipe and the facts its record keeps (file hash, windows hash, overlap check)."""
    from crit_core.model import open_model
    from crit_core.text import legacy_eos_ids, load_model_text
    documents = read_documents(path)
    texts = suite_texts()
    if texts is not None:
        overlap = suite_overlap(documents, texts)
        if overlap:
            raise ValueError(f"{path}: {len(overlap)} documents contain Eval suite text (first: document "
                             f"{overlap[0] + 1}); calibrate on data unrelated to Eval")
    files = open_model(model).files
    text = load_model_text(files, clock=None)
    stops = legacy_eos_ids(files)
    settings = recipe["calibration"]
    ids = windows(documents, lambda t: text.tokenizer.encode(t, add_special_tokens=False, split_special_tokens=True),
                  stops[:1], settings["samples"], settings["sequence_tokens"], recipe["seed"])
    facts = {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": file_sha(Path(path)),
             "documents": len(documents), "samples": settings["samples"], "sequence_tokens": settings["sequence_tokens"],
             "separator_ids": stops[:1], "windows_sha256": digest(ids),
             "suite_overlap": "checked" if texts is not None else "unchecked: Eval's suite is not on this host"}
    return ids, facts
