"""crit smoke: run the smoke tests against a served engine and print one line per test."""
import argparse
import json
from pathlib import Path

from .client import Client
from .suite import TESTS, run


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="crit smoke",
        description="Check a served engine's quality components in seconds: each test compares two ways the engine "
                    "computes the same thing (a prefix and the whole prompt, decode and prefill, a reply and its "
                    "score, a request before and after another one) or a property any working model has.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="crit serve's address (default %(default)s)")
    parser.add_argument("--model", help="Model or candidate to test (default: the server's default)")
    parser.add_argument("--api-key", help="Key crit serve asks for")
    parser.add_argument("--only", action="append", choices=[name for name, _, _ in TESTS], metavar="TEST",
                        help="Run only this test (repeatable)")
    parser.add_argument("--strict", action="store_true",
                        help="Tolerances for an engine that keeps one arithmetic on every path (preserve); the "
                             "default fits engines in quality mode speed")
    parser.add_argument("--long", type=int, default=4096,
                        help="Longest prompt the recall test sends, capped by the context (default %(default)s)")
    parser.add_argument("--timeout", type=float, default=120, help="Seconds per request (default %(default)s)")
    parser.add_argument("--budget", type=float, default=300,
                        help="Seconds the suite may run before it stops as failed: an engine that needs longer is too "
                             "slow or slows down as it serves (default %(default)s)")
    parser.add_argument("--out", type=Path, help="Write the items as JSON here")
    args = parser.parse_args(argv)
    client = Client(args.url, args.model, args.api_key, args.timeout)
    items = run(client, set(args.only or ()), "strict" if args.strict else "loose", args.long, budget=args.budget)
    total = sum(i["seconds"] for i in items)
    counts = {s: sum(i["state"] == s for i in items) for s in ("passed", "failed", "skipped")}
    print(f"{counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped in {total:.1f}s")
    if args.out:
        args.out.write_text(json.dumps({"url": args.url, "profile": "strict" if args.strict else "loose",
                                        "items": items}, indent=2) + "\n", encoding="utf-8")
    return 1 if counts["failed"] else 0
