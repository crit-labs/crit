"""crit run capture: a case's chat request written so it renders to the workload's prompt tokens.

The request renders through the run's model's chat template with the template variables candidates serve (recon's
frozen template_variables) under the request's own chat_template_kwargs, exactly as resolve checks it
(measure.check_case_prompts). Its user message is text the caller gives, else a fixed neutral passage, cut to the
longest prefix of whole words that fits and then padded one short word at a time to the exact count.
"""
import json

from .records import file_sha, ident, load_run, safe_path

# Neutral filler for a case whose prompt content does not matter, only its length.
PASSAGE = ("The notes describe a small water utility that is moving its maintenance logs from paper to a shared "
           "database. Crews inspect pumps, valves and reservoirs on fixed schedules, record what they find, and "
           "open repair orders that the operations manager approves before parts are bought. Winter brings frozen "
           "valves and burst mains; summer brings pump overheating and low reservoir levels. The new system must "
           "keep every inspection with its date, crew and asset, show overdue checks, and let the manager approve "
           "orders from a phone. ")
INSTRUCTION = "Summarize the key requirements for the new system as a numbered list."
PADS = (" ok", " a", " x", " the", ".")


def capture(store, run_id, case_id, text=None, instruction=INSTRUCTION, system=None, template_kwargs=None,
            inputs=None):
    """Write captured/CASE/request.json; returns its manifest row, the prompt tokens and the variables it renders
    with. `inputs`: a resolve payload file whose case_manifests entry for the case is set to the new row."""
    from crit_core.model import open_model
    from crit_core.text import load_model_text
    from . import recon
    root = store.directory(run_id)
    run = load_run(root)
    case = next((c for c in run["spec"]["cases"] if c["case_id"] == ident(case_id, "case")), None)
    if case is None:
        raise ValueError(f"The run has no case {case_id}; its cases: " + ", ".join(c["case_id"] for c in run["spec"]["cases"]))
    if (root / "inputs.json").exists():
        raise ValueError("The run's inputs are resolved: captured cases are frozen")
    frozen, _ = recon.load(root, run)
    if frozen is None:
        raise ValueError("Run crit run ... recon first: the case renders with the template variables candidates serve")
    if template_kwargs is not None and not isinstance(template_kwargs, dict):
        raise ValueError("--template-kwargs must be a JSON object")
    variables = {**frozen["template_variables"], **(template_kwargs or {})}
    target = case["workload"]["prompt_tokens"]
    model = load_model_text(open_model(run["spec"]["model_path"]).files, clock=None)
    words = (text if text is not None else PASSAGE * (1 + 4 * target // len(PASSAGE.split()))).split()
    if not words:
        raise ValueError("The case text is empty")

    def request(content):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": content + ("\n\n" + instruction if instruction else "")}]
        return {"messages": messages, **({"chat_template_kwargs": template_kwargs} if template_kwargs else {})}

    def count(content):
        return len(model.encode_chat(request(content)["messages"], add_generation_prompt=True, **variables))

    if count("") > target:
        raise ValueError(f"{case['case_id']}: the template, instruction and system message alone render to "
                         f"{count('')} prompt tokens, the workload has {target}")
    low, high = 0, len(words)
    if count(" ".join(words)) < target and text is not None:
        raise ValueError(f"{case['case_id']}: the given text renders to {count(' '.join(words))} prompt tokens, the "
                         f"workload has {target}; give a longer text")
    while low < high:  # the most whole words that fit
        middle = (low + high + 1) // 2
        if count(" ".join(words[:middle])) <= target:
            low = middle
        else:
            high = middle - 1
    base = " ".join(words[:low])
    content = None
    for pad in PADS:
        padded, n = base, count(base)
        while n < target:
            padded += pad
            n = count(padded)
        if n == target:
            content = padded
            break
    if content is None:
        raise ValueError(f"{case['case_id']}: no padding reaches exactly {target} prompt tokens; give --text")
    path = safe_path(root, f"captured/{case['case_id']}/request.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request(content), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    row = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_sha(path)}
    if inputs is not None:
        payload = json.loads(inputs.read_text(encoding="utf-8")) if inputs.is_file() else {}
        payload.setdefault("case_manifests", {})[case["case_id"]] = [row]
        inputs.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return {"case_id": case["case_id"], "row": row, "prompt_tokens": target, "template_variables": variables}
