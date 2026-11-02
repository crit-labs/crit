"""crit bench OWNER/NAME: measure an engine crit pull installed with crit Eval on this machine, and print its card's
stats beside the ones measured here.

It readies crit Eval's data for the installed checkpoint in --data when the folder holds none (a published reference
package for the checkpoint when there is one, else crit eval prepare, with the sampling the card measured at), serves
the engine through crit serve as `crit serve --engine` does, and runs crit Eval over its chat API at the context it
was installed at: the speed suite, and quality where the data holds the reference's scores for the checkpoint
(--reference scores them first, which reads the whole checkpoint). The result file (--out) is what `crit hub report`
adds to the card, as a reproduction of the GPU and model it measured or a new result.
"""
import argparse
import datetime
from pathlib import Path
import re
import sys
import threading

from crit.cli import default_project
from crit.records import GIB, read_json
from crit.run.served import QUALITY, SERVED
from crit.run.store import ROOT

from .cli import device_probe, log
from .store import InstallStore, engine_id

# The six headline stats, as crit-hub's cards name them.
STATS = ("decode_tok_s", "prefill_tok_s", "ttft_ms", "ppl", "kld_mean", "context_tokens")


class InstalledEngines:
    """crit Eval's engines for an installed engine (crit_eval.run.runner.run takes it as its adapter): the engine
    served through crit serve at the context it was installed at, in one resident host for the whole run."""

    def __init__(self, installs, engine, prepared, kv_tokens):
        self.installs, self.engine, self.prepared, self.kv_tokens = installs, engine, prepared, kv_tokens
        self.server = f"crit serve --engine {engine}"
        self.current = None   # (InstalledEngine, HTTP server) once opened

    def create_eval_engine(self, context):
        from crit_api.installed import InstalledEngine
        from crit_api.server import Api, Server
        from crit_eval.run import chat
        if context.get("context_tokens") not in (None, self.kv_tokens):
            raise ValueError(f"{self.engine} serves at the {self.kv_tokens:,} tokens of context it was installed at, not "
                             f"{context['context_tokens']:,}; crit pull --context installs it at another")
        if self.current is None:
            engine = InstalledEngine(self.installs.root, self.engine)
            log(f"crit bench: loading {self.engine} at {self.kv_tokens:,} tokens of context")
            engine.open()
            server = Server(("127.0.0.1", 0), Api(engine))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.current = (engine, server)
        engine, server = self.current
        served = chat.Server("http://127.0.0.1:%d/v1" % server.server_address[1], self.prepared,
                             engine.model_id()).create_eval_engine(context)
        served.close = lambda: None   # one host serves every engine Eval opens; stop() ends it
        return served

    def stop(self):
        if self.current is None:
            return
        engine, server = self.current
        self.current = None
        server.shutdown()
        server.server_close()
        engine.close()


def card_result(card, gpu, model):
    """The card's result for this GPU and model, else its result for the model on another GPU, else None."""
    results = [r for r in card.get("results") or [] if r.get("model") == model]
    return next((r for r in results if r.get("gpu") == gpu), results[0] if results else None)


def gpu_of(card, device):
    """The card's id for this machine's GPU: one of its own ids the device's name holds, else the name without its
    vendor (crit-hub's tools/add_result.py derives it the same way)."""
    name = re.sub(r"[^a-z0-9]", "", str(device or "").lower())
    known = sorted((r["gpu"] for r in card.get("results") or [] if re.sub(r"[^a-z0-9]", "", r["gpu"]) in name), key=len)
    if known:
        return known[-1]
    for vendor in ("nvidiageforce", "nvidia", "amdradeon", "amd", "intel", "apple"):
        if name.startswith(vendor):
            return name[len(vendor):]
    return name or None


def worst(result, field):
    """The worst path's quality figure (perplexity over the served paths where the result has them), as crit-hub reads
    a result file."""
    sets = {n: s for n, s in ((result.get("quality") or {}).get("sets") or {}).items() if isinstance(s, dict)}
    if field == "ppl_ratio" and any(n.endswith("/served") for n in sets):
        sets = {n: s for n, s in sets.items() if n.endswith("/served")}
    values = [s.get(field) for s in sets.values() if isinstance(s.get(field), (int, float))]
    return max(values) if values else None


def measured_stats(result):
    """A crit Eval result's six headline stats, as crit-hub's cards state them: the sampled decode where it measured
    one, prefill and time to first token, the worst served perplexity ratio and mean KL divergence, and the context."""
    sampled = (result.get("sampled_speed") or {}).get("decode_score")
    speed = result.get("speed") or {}
    ppl = worst(result, "ppl_ratio")
    return {"decode_tok_s": sampled if sampled is not None else speed.get("decode_score"),
            "prefill_tok_s": speed.get("prefill_score"), "ttft_ms": speed.get("ttft_score_ms"),
            "ppl": {"vs_reference": ppl} if ppl is not None else None, "kld_mean": worst(result, "kld_mean"),
            "context_tokens": (result.get("context") or {}).get("tokens")}


def shown(key, value):
    if value is None:
        return "-"
    if isinstance(value, list):
        return "-".join(shown(key, v) for v in value)
    if key == "ppl":
        value = value.get("vs_reference", value.get("absolute"))
        return f"{(value - 1) * 100:+.1f}%" if value is not None else "-"
    if key == "kld_mean":
        return f"{value:.3g}"
    if key == "context_tokens":
        return f"{value:,}"
    return f"{value:,.1f}" if key == "decode_tok_s" else f"{value:,.0f}"


def comparison(card_stats, stats, label):
    """The card's stats and the measured ones side by side, one line per stat."""
    names = {"decode_tok_s": "decode tok/s", "prefill_tok_s": "prefill tok/s", "ttft_ms": "time to first token ms",
             "ppl": "perplexity vs reference", "kld_mean": "mean KL divergence", "context_tokens": "context tokens"}
    lines = [f"{'':<26}{label:>18}{'this machine':>16}"]
    for key in STATS:
        lines.append(f"{names[key]:<26}{shown(key, (card_stats or {}).get(key)):>18}{shown(key, stats.get(key)):>16}")
    return "\n".join(lines)


def ready_data(model, data, sampling=None):
    """Prepared data for the checkpoint in `data`: kept when it holds some, else a published reference package for
    the checkpoint (prepared data and reference scores), else crit eval prepare at the card's sampling."""
    from crit_eval.cli import main as eval_main
    from crit_eval.quality.packages import fetch, source
    from crit_eval.quality.prepare import PREPARED
    if (data / PREPARED).is_file():
        return "kept"
    data.mkdir(parents=True, exist_ok=True)
    if sampling is None:
        try:
            if fetch(model, data, source(), log):
                return "published package"
        except (OSError, ValueError) as exc:
            log(f"crit bench: no published reference package ({exc}); preparing the data here")
    arguments = ["prepare", "--model", str(model), "--data", str(data)]
    if sampling:
        arguments += ["--sampling", ",".join(f"{k}={v}" for k, v in sampling.items())]
    if eval_main(arguments):
        raise ValueError(f"crit eval prepare failed for {model}")
    return "prepared"


def bench(engine, store, data, out=None, project=ROOT, device=None, parts=None, stage=None, budget_minutes=None,
          reference=False, quality_mode=None):
    """Measure the installed engine; returns (result, the card's stats it compares with, their label)."""
    from crit_core.machine import engine_lock
    from crit_eval import load_json
    from crit_eval.quality.fixed import STORE
    from crit_eval.quality.prepare import PREPARED
    from crit_eval.run.runner import run as run_eval
    from crit.run.cli import load_settings
    engine = engine_id(engine)
    spec = store.spec(engine)   # the package and checkpoint as installed, verified
    install, folder = store.get(engine), store.folder(engine)
    card = read_json(folder / "card.json")
    model, model_id = Path(spec["model_path"]), install["checkpoint"]["model"]
    probe = device_probe(project, store, device)
    gpu = gpu_of(card, probe.get("device"))
    stated = card_result(card, gpu, model_id)
    sampling = (stated or {}).get("sampling")
    data = Path(data).expanduser().absolute()
    how = ready_data(model, data, (sampling or {}).get("settings") if (sampling or {}).get("source") == "override" else None)
    log(f"crit bench: crit Eval data in {data} ({how})")
    from crit_eval.run.runner import PARTS
    parts = tuple(parts or ("speed", "quality"))
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        raise ValueError(f"crit Eval has no part {', '.join(unknown)}: {', '.join(PARTS)}")
    missing = [c for c in SERVED if c not in spec["capabilities"]]
    if missing:
        raise ValueError(f"{engine}'s package declares no {', '.join(missing)}: crit Eval measures it through crit serve")
    if "quality" in parts and [c for c in QUALITY if c not in spec["capabilities"]]:
        log(f"crit bench: {engine} cannot score fixed text ({', '.join(QUALITY)}): measuring speed only")
        parts = tuple(p for p in parts if p != "quality")
    if "quality" in parts and not (data / STORE / "index.json").is_file():
        if reference:
            from crit_eval.quality.reference import score_data
            log("crit bench: scoring the reference on the checkpoint (crit eval reference)")
            score_data(model, data, log)
        else:
            log(f"crit bench: {data} holds no reference scores for the checkpoint: measuring speed only (--reference "
                "scores them first, which reads the whole checkpoint)")
            parts = tuple(p for p in parts if p != "quality")
    settings = load_settings(project)
    kv_tokens = install["limits"]["kv_tokens"]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out).expanduser().absolute() if out else folder / "bench" / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = quality_mode or (stated or {}).get("quality_mode") or "preserve"
    engines = InstalledEngines(store, engine, load_json(data / PREPARED), kv_tokens)
    source = install["source"]
    try:
        with engine_lock(f"crit bench {engine}"):
            result = run_eval(engines, str(model), data, out, {}, parts, label=f"{engine} @ {source['revision'][:12]}",
                              device=probe, context_tokens=kv_tokens, quality_mode=mode, stage=stage,
                              budget_seconds=budget_minutes * 60 if budget_minutes else None,
                              vram_headroom_bytes=round(settings.get("vram_headroom_gib", 1.0) * GIB),
                              identity={"engine": engine, "package_sha256": spec["package_sha256"]}, log=log)
    finally:
        engines.stop()
    label = f"card ({stated['gpu']})" if stated else "card"
    return result, (stated or {}).get("stats"), label, out


def parser():
    """The arguments of crit bench (crit mcp offers its tools from them)."""
    parser = argparse.ArgumentParser(prog="crit bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", metavar="OWNER/NAME", help="An engine crit pull installed")
    parser.add_argument("--data", type=Path, required=True,
                        help="crit Eval's data for the installed checkpoint: readied here when the folder holds none")
    parser.add_argument("--out", type=Path, help="The result file (default: the engine's bench/ folder in the store)")
    parser.add_argument("--store", type=Path, help="Engine store (default: PROJECT/engines)")
    parser.add_argument("--project", type=Path, default=default_project(),
                        help="crit repository, whose settings.toml gives the device and headroom (default: this checkout)")
    parser.add_argument("--device", type=Path, help="A crit eval probe result (default: settings.toml's device, else "
                                                     "the store's, else a probe run now)")
    parser.add_argument("--parts", nargs="+", metavar="PART", help="crit Eval's parts (default: speed and quality)")
    parser.add_argument("--stage", choices=("1", "2", "3", "full"),
                        help="How far crit Eval measures (default: as far as its budget allows; full: every part)")
    parser.add_argument("--budget-minutes", type=float, help="crit Eval's budget for a staged run")
    parser.add_argument("--reference", action="store_true",
                        help="Score the reference on the checkpoint first when the data holds no scores, so quality is "
                             "measured too (reads the whole checkpoint)")
    parser.add_argument("--quality-mode", choices=("preserve", "as_quantized"),
                        help="The quality mode (default: the card's result's)")
    return parser


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        store = InstallStore(args.store or args.project / "engines")
        stage = args.stage if args.stage in (None, "full") else int(args.stage)
        result, card_stats, label, out = bench(args.engine, store, args.data, args.out, args.project, args.device,
                                               args.parts, stage, args.budget_minutes, args.reference, args.quality_mode)
    except (ValueError, OSError, KeyError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    print(comparison(card_stats, measured_stats(result), label))
    print(f"wrote {out}")
    if result.get("aborted"):
        log(f"crit bench: crit Eval stopped ({result['aborted']['reason']}); the result holds what it measured")
        return 1
    log(f"crit bench: add it to the card with: crit hub report {engine_id(args.engine)} {out} --hub <crit-hub checkout> "
        "--by <your GitHub account>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
