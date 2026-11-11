"""Concurrent requests end to end on the CPU: the toy engine's plugin package (the plugin engine example), which Core's
loader runs on Core's concurrent runtime (crit_core_steps), served by crit serve four requests at once, measured by
`crit run measure` at concurrency 4 and by crit Eval's concurrency part. Both see the aggregate grow with the requests in flight and a peak
of as many requests as clients, and every batched reply equals the serial one. crit_eval is the one on PYTHONPATH (an
installed one, else the checkout beside this one)."""
import os
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Core's runtime, which the resident host's loader opens the toy model on.
os.environ["CRIT_CORE_STEP_LIBRARY"] = str(Path(sys.argv[1]).resolve())
sys.path.insert(0, str(ROOT / "core" / "python"))
sys.path.append(str(ROOT.parent / "crit-eval"))
import crit_eval  # noqa: E402,F401  imported before crit's fixtures put the checkout beside this one first on sys.path

sys.path.insert(0, str(ROOT / "tests"))
from test_run_measure import MeasureFixture  # noqa: E402
from support.engines import selection  # noqa: E402
from support.runs import with_plugins  # noqa: E402
from crit.run import measure  # noqa: E402
from crit_api.engine import ServedEngine  # noqa: E402
from crit_api.server import Api, Server  # noqa: E402
from crit_eval import EVAL_DIR, load_json  # noqa: E402
from crit_eval.run.chat import ChatEngine, Server as ChatServer, load_text  # noqa: E402
from crit_eval.run.concurrency import run_concurrency  # noqa: E402

# The package: the toy engine's plugins (the plugin engine example), whose model's next token is a hash of every token
# its row has seen, so a row decodes the same tokens in a batch as alone, and whose every step takes 4 ms however many
# rows it holds, as a memory-bound decode step does. Four rows a step, no requests waiting beyond them.
OPTIONS = {"toy-model": {"step_ms": 4, "prefill_chunk": 256, "concurrency": 4, "pending": 0}}
VOCAB = {f"t{i}": i for i in range(64)}


def words(seed, count):
    return " ".join(f"t{(7 * seed + 3 * j + j * j) % 64}" for j in range(count))


class ConcurrencyEndToEnd(MeasureFixture, unittest.TestCase):
    # One token per word: the workload's 64 prompt tokens are 64 words.
    content = words(0, 64)
    model_files = {
        "tokenizer.json": {"version": "1.0", "added_tokens": [], "normalizer": None,
                           "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
                           "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
                           "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": "t0"}},
        "tokenizer_config.json": {"chat_template": "{{ messages[-1]['content'] }}", "eos_token": None,
                                  "bos_token": None},
        "chat_template.jinja": "{{ messages[-1]['content'] }}",
        "config.json": {"vocab_size": 64, "eos_token_id": None},
        "generation_config.json": {"eos_token_id": []},
    }

    def setUp(self):
        super().setUp()
        # A case of four requests at once: 4 x (64 + 32) tokens of KV within the run's 8,192.
        self.view = self.store.create(self.data | {"workload": {
            "prompt_tokens": 64, "output_tokens": 32, "concurrency": 4, "repetitions": 2, "warmup_runs": 0}})
        self.run, self.run_id = self.view["run"], self.view["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)
        self.store.recon(self.run_id)

    def served(self):
        record = self.candidate(configuration={"plugins": selection(options=OPTIONS)})
        candidate = record["candidate_id"]
        package = with_plugins(self.root / ("package-" + candidate))
        manifest = self.store.package(self.run_id, candidate, package,
                                      {"build": {"toolchain": {"fixture": "host-python"}, "dependencies": [],
                                                 "command": ["fixture"]}})
        self.checks(record, manifest)   # recorded before measure, as check --from-eval records them
        return candidate

    def serve(self, candidate):
        engine = ServedEngine(self.store.root, self.run_id, candidate_id=candidate, host_limit=1 << 30,
                              device_limit=1 << 30)
        self.addCleanup(engine.close)
        engine.open()
        server = Server(("127.0.0.1", 0), Api(engine))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        return "http://%s:%d/v1" % server.server_address

    def test_measure_and_eval_see_requests_run_at_once_and_equal_their_serial_replies(self):
        candidate = self.served()

        # crit run measure: each repetition sends the case's request four times at once.
        (record,) = measure.measure(self.store, self.run_id, candidate)
        decode, aggregate = record["metrics"]["decode_tps"]["samples"], record["metrics"]["aggregate_tps"]["samples"]
        self.assertEqual((len(decode), len(aggregate)), (2 * 4, 2))
        self.assertEqual(record["extensions"]["concurrency"],
                         {"clients": 4, "peak_concurrency": [4, 4], "max_concurrent_requests": 4})
        self.assertEqual(record["extensions"]["output_tokens"], [32] * 8)
        # Four requests share each step: together they deliver well over one request's tokens per second.
        self.assertGreater(min(aggregate), 2 * sorted(decode)[4])
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

        # crit Eval's concurrency part against the same package behind crit serve, which serves sealed candidates.
        self.store.seal(self.run_id, candidate, "completed")
        protocol = load_json(EVAL_DIR / "protocol.json")
        text = load_text(str(self.model))
        cases = lambda kind, seeds, lengths: [  # noqa: E731
            {"id": f"{kind}-{seed}", "messages": messages, "prompt_ids": text.encode_chat(messages)}
            for seed, count in zip(seeds, lengths)
            for messages in [[{"role": "user", "content": words(seed, count)}]]]
        prepared = {"speed": cases("speed", (1, 2, 3, 4), (9, 17, 33, 50)),
                    "heldout": cases("heldout", (5, 6), (12, 40))}
        chat = ChatEngine(ChatServer(self.serve(candidate), prepared),
                          {"context_tokens": 8192, "model": str(self.model)})
        output = protocol["concurrency"]["output_tokens"]
        # The speed part's serial greedy replies, which the batched ones are compared with.
        serial = {case["id"]: {"status": "measured",
                               "tokens": chat.generate(case["prompt_ids"], output, lambda _: None)["tokens"]}
                  for case in prepared["speed"]}
        logged = []
        found = run_concurrency(chat, chat.info(), prepared, protocol, {"speed": {"cases": serial}}, 8192,
                                logged.append)
        self.assertEqual((found["status"], found["max_concurrent_requests"], found["planned"]),
                         ("measured", 4, [1, 2, 4]))
        levels = found["levels"]
        for clients in (1, 2, 4):
            level = levels[str(clients)]
            self.assertTrue(level["qualified"], level["errors"])
            self.assertEqual((level["requests"], level["completion_tokens"]), (4 * clients, 4 * clients * output))
            self.assertEqual(level["peak_concurrency"], clients)
            self.assertGreater(level["consistency"]["compared"], 0)
            self.assertEqual(level["consistency"]["fraction"], 1.0, level["consistency"]["differences"])
            self.assertIsNotNone(level["ttft_ms_p50"])
        self.assertGreater(levels["4"]["aggregate_tps"], 2 * levels["1"]["aggregate_tps"])
        self.assertTrue(any(line.startswith("concurrency 4:") and "replies equal the serial ones" in line
                            for line in logged), logged)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
