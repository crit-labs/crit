"""The general corpus crit names for residency profiles (corpus.json), and fetching it.

corpus.json lists public datasets unrelated to crit-eval's suite: web text in seven
languages and code in three to tune on, and held-out sources (test splits, languages
and a programming language the tuning never sees, rows far from the tuning ones) to
check on. A profile built from its tuning sources and checked on its held-out ones
(residency_profile.build_profile with corpus=describe(...)) meets AGENTS.md's rule for
content-dependent choices without a corpus of the build's own.

fetch reads the rows through the Hugging Face dataset viewer's rows API, 100 a page,
standard library only, and caches each source as JSON; it records each dataset's
revision, so a profile names exactly what it was built from. The engine then runs its
own forward over each text (teacher-forced, cut to max_chars) and logs one plan per
routed layer and token: the traces build_profile and held_out_check take.
"""
import json
from pathlib import Path
import urllib.parse
import urllib.request

MANIFEST = Path(__file__).with_name("corpus.json")
ROWS = "https://datasets-server.huggingface.co/rows"
DATASETS = "https://huggingface.co/api/datasets/"
PAGE = 100


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "device-expert-cache-corpus":
        raise ValueError("not a residency profile corpus")
    names = [s["name"] for s in manifest["tuning"] + manifest["held_out"]]
    if len(set(names)) != len(names) or len(manifest["held_out"]) < 2:
        raise ValueError("source names must be distinct, with at least two held-out sources")
    return manifest


def _get(url, open_url):
    with open_url(url) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch(manifest, cache, part="tuning", open_url=urllib.request.urlopen):
    """{source name: {"texts": [...], "revision": dataset sha}} for the sources of `part` ("tuning" or "held_out"),
    each text cut to the manifest's max_chars. A source already cached under `cache` is read from there."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out, revisions = {}, {}
    for source in manifest[part]:
        path = cache / f"{source['name']}.json"
        if path.is_file():
            out[source["name"]] = json.loads(path.read_text(encoding="utf-8"))
            continue
        dataset = source["dataset"]
        if dataset not in revisions:
            revisions[dataset] = _get(DATASETS + urllib.parse.quote(dataset, safe="/"), open_url).get("sha")
        texts = []
        for offset in range(source["offset"], source["offset"] + source["rows"], PAGE):
            query = urllib.parse.urlencode({"dataset": dataset, "config": source["config"], "split": source["split"],
                                            "offset": offset, "length": min(PAGE, source["offset"] + source["rows"] - offset)})
            page = _get(f"{ROWS}?{query}", open_url)
            if "rows" not in page:
                raise RuntimeError(f"{source['name']}: the rows API returned {page.get('error', 'no rows')}")
            texts += [row["row"][source["field"]][:manifest["max_chars"]] for row in page["rows"]]
        out[source["name"]] = {"texts": texts, "revision": revisions[dataset]}
        path.write_text(json.dumps(out[source["name"]]), encoding="utf-8")
    return out


def describe(manifest, fetched):
    """The corpus statement a profile records: every source with its dataset, split, rows and revision."""
    by_name = {s["name"]: s for s in manifest["tuning"] + manifest["held_out"]}
    return "crit residency corpus: " + "; ".join(
        f"{name} {by_name[name]['dataset']}/{by_name[name]['config']}/{by_name[name]['split']} "
        f"rows {by_name[name]['offset']}+{len(value['texts'])} @ {(value.get('revision') or 'unknown')[:12]}"
        for name, value in fetched.items())
