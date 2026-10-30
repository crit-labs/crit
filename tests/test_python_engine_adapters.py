"""A Python engine with NativeEngine's interface gets crit_core.native's stock eval-v1 and chat-resident-v1 adapters."""
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"): sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_core import native  # noqa: E402
from crit_core.native import NativeError, Scores, eval_engine, resident_engine  # noqa: E402


class ToyEngine:
    """Greedy successor model: after token t comes t + 1 (mod 400), with log-probability -0.5 and one alternative."""

    def __init__(self, configuration):
        self.configuration, self.calls, self.closed = configuration, [], False

    def info(self):
        return {"implementation": "toy", "context": 4096, "stop_tokens": [1], "speculative": False,
                "sampling": {"temperature": True}, "replay": True, "token_logprobs": True, "max_top_logprobs": 20,
                "query_scoring": True, "score_paths": ["prefill"], "native_libraries": ["toy_kernels.so"],
                "kernel_work": [{"match": "toy", "term": "matmul"}], "prompt_activations": {"nvfp4": "bf16"}}

    def generate(self, prompt, output_tokens, *, stop_tokens=None, sampling=None, replay=None, plain=False,
                 logprobs=None, on_tokens=None, cancelled=None):
        self.calls.append({"prompt": prompt, "stop_tokens": stop_tokens, "sampling": sampling, "replay": replay,
                           "plain": plain, "logprobs": logprobs})
        tokens, last = [], prompt[-1]
        for i in range(output_tokens):
            token = replay[i] if replay is not None else (last + 1) % 400
            tokens.append(token)
            if on_tokens:
                if logprobs is None:
                    on_tokens([token])
                else:
                    on_tokens([token], [(-0.5, [(token, -0.5), (0, -2.0)][:logprobs])])
            last = token
            if stop_tokens is None and token == 1:
                break
        return tokens, {"finish": "length", "timing": {"prefill_tokens": len(prompt), "reused_prompt_tokens": 0,
                                                      "prefill_ms": 1.0, "decode_ms": 2.0, "first_token_ms": 1.5,
                                                      "decode_steps": len(tokens)}}

    def score(self, ids, first_target, path="", queries=None, top_k=0, packed=False):
        targets = ids[first_target:]
        return Scores([-0.5] * len(targets), None, None,
                      [[(t, -0.5), (0, -2.0)][:top_k] for t in targets] if top_k else None)

    def close(self):
        self.closed = True


class PythonEngineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.opened = []

    def open(self, configuration):
        engine = ToyEngine(configuration)
        self.opened.append(engine)
        return engine

    def test_eval_adapter(self):
        engine = eval_engine({"model": "/m", "config": {"width": 3}, "context_tokens": 2048, "speculative": False},
                             self.open, defaults={"width": 1, "mode": "x"})
        toy = self.opened[0]
        self.assertEqual(toy.configuration, {"width": 3, "mode": "x", "model": "/m", "speculative": False,
                                             "context_tokens": 2048})
        info = engine.info()
        self.assertEqual((info["name"], info["max_context"], info["native_libraries"], info["kernel_work"]),
                         ("toy", 4096, ["toy_kernels.so"], [{"match": "toy", "term": "matmul"}]))
        out = engine.generate([5, 6], 3, lambda tokens: None)
        self.assertEqual((out["tokens"], out["prefill_ms"]), ([7, 8, 9], 1.0))
        self.assertEqual((toy.calls[-1]["stop_tokens"], toy.calls[-1]["sampling"]), ([], {"temperature": 0.0}))
        self.assertEqual(engine.replay([5, 6], [1, 2], lambda tokens: None)["tokens"], [1, 2])
        sampled = engine.generate_sampled([5], 2, lambda tokens: None, {"temperature": 0.7, "top_p": 1.0}, logprobs=1)
        self.assertEqual(sampled["logprobs"][0], {"token": 6, "logprob": -0.5, "top": [[6, -0.5]]})
        self.assertEqual(toy.calls[-1]["sampling"], {"temperature": 0.7})   # top_p is not a field it applies
        rows = engine.score([5, 6], [7], 2)
        self.assertEqual(rows["prompt"], [{"logprob": -0.5, "top": [[6, -0.5], [0, -2.0]]}])
        self.assertEqual(rows["continuation"], [{"logprob": -0.5, "top": [[7, -0.5], [0, -2.0]]}])
        engine.close()
        self.assertTrue(toy.closed)
        with self.assertRaisesRegex(NativeError, "engine is closed"):
            engine.generate([5], 1, lambda tokens: None)

    def test_resident_adapter(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        model = write_files(Path(temp.name) / "model", llama_files())
        context = types.SimpleNamespace(configuration={"width": 2}, limits={"kv_tokens": 1024},
                                        model_file=lambda name: str(model / name))
        engine = resident_engine(context, self.open)
        self.assertEqual((engine.capacity, engine.stop_ids, engine.sampling, engine.logprobs, engine.scores),
                         (1024, {1}, ("temperature",), True, True))
        self.assertEqual(self.opened[0].configuration["context_tokens"], 1024)
        reply = engine.generate({"request_id": "r", "candidate_id": "c", "package_sha256": "p",
                                 "limits": {"max_output_tokens": 4}, "sampling": {"temperature": 0.0},
                                 "messages": [{"role": "user", "content": "hi"}]}, threading.Event())
        self.assertEqual((reply["telemetry"]["output_tokens"], reply["telemetry"]["prefill_ms"]), (4, 1.0))
        prompt, continuation, _ = engine.score_tokens([5, 6], [7], 2, lambda: False, True)
        self.assertEqual((len(prompt), continuation), (1, [(-0.5, [(7, -0.5), (0, -2.0)])]))
        engine.close()
        self.assertTrue(self.opened[0].closed)

    def test_close_frees_an_engine_whose_parts_refer_to_each_other(self):
        import weakref
        engine = eval_engine({"model": "/m"}, self.open)
        toy = self.opened.pop()
        toy.drafter = types.SimpleNamespace(model=toy)   # a reference cycle, as a drafter and its model make
        freed = weakref.ref(toy)
        del toy
        import gc
        gc.disable()   # only close() may collect
        try:
            engine.close()
            self.assertIsNone(freed())
        finally:
            gc.enable()

    def test_a_cancelled_request_never_reaches_the_engine(self):
        engine = native.PythonEngine(self.open, {})
        self.assertEqual(engine.generate([1], 2, cancelled=lambda: True), (None, {"cancelled": True}))
        self.assertEqual(self.opened[0].calls, [])

    def test_requests_run_one_at_a_time_unless_the_engine_serves_several(self):
        inside, peak, lock = [0], [0], threading.Lock()

        class Slow(ToyEngine):
            def generate(self, prompt, output_tokens, **options):
                with lock:
                    inside[0] += 1
                    peak[0] = max(peak[0], inside[0])
                threading.Event().wait(0.02)
                with lock:
                    inside[0] -= 1
                return [], {}

        engine = native.PythonEngine(Slow, {})
        threads = [threading.Thread(target=engine.generate, args=([1], 1)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(peak[0], 1)


if __name__ == "__main__":
    unittest.main()
