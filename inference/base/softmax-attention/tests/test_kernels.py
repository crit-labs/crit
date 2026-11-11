"""softmax-attention's page-reading paths equal its gathered-rows SDPA path in a step: crit's split-KV and prompt
readers over every page type (where they load; the default, no library kernel), FlashInfer's over 16-bit and FP8 pages
with option flashinfer auto (when FlashInfer is installed: a yardstick),
the torch fallbacks otherwise; a row's result does not depend on the rows beside it; a decode step over step buffers
replays from a CUDA graph."""
from crit_plugins import StepBatch
from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}
HQ, HKV, D = 8, 2, 128


def _flashinfer():
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


class PagedKernelTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("the page-reading kernels run on CUDA")
        self.torch = __import__("torch")

    def engine(self, dtype, page, step_rows=0, int8_qk=False, decode="flashinfer", flashinfer="off"):
        state = {"page": page, "step_rows": step_rows, "step_tokens": 512}
        kv = self.assemble(USE, use_options={"attention-state": state})["kv_cache"]
        fast = self.assemble(USE, options={"int8_qk": int8_qk, "decode": decode, "flashinfer": flashinfer},
                             use_options={"attention-state": state})["attention"]
        plain = self.assemble(USE, options={"kernels": "torch"}, use_options={"attention-state": state})["attention"]
        pool = kv.pool([0], HKV, D, 4096, "cuda", dtype)
        if dtype == "fp8":
            pool.set_scales(0, 0.05, 0.04)
        return pool, fast, plain

    def write(self, pool, rows, seed):
        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        batch = StepBatch.build(rows, "cuda")
        slots = pool.reserve(batch)
        keys = torch.randn(batch.tokens, HKV, D, generator=g).cuda()
        values = torch.randn(batch.tokens, HKV, D, generator=g).cuda()
        pool.write(0, slots, keys, values)
        self.written = keys, values
        return batch

    def expected_kind(self, fast, dtype, step_rows, decode, flashinfer):
        if _flashinfer() and dtype in ("bf16", "fp16", "fp8") and not step_rows and decode == "flashinfer" and                 flashinfer == "auto":
            return "flashinfer"
        return "split" if fast.device_form.kernels is not None else "torch"

    def compare(self, dtype, page, step_rows=0, int8_qk=False, bound=0.02, decode="flashinfer", flashinfer="off"):
        torch = self.torch
        pool, fast, plain = self.engine(dtype, page, step_rows, int8_qk, decode, flashinfer)
        probe = torch.zeros(1, HQ, D, dtype=torch.bfloat16, device="cuda")
        self.assertEqual((fast.device_form.reader(pool, probe), plain.device_form.reader(pool, probe)),
                         (self.expected_kind(fast, dtype, step_rows, decode, flashinfer), "torch"))
        for s in "abc":
            pool.open(s)
        self.write(pool, [("a", 0, 300), ("b", 0, 70)], 1)
        steps = [[("a", 300, 1), ("b", 70, 1), ("c", 0, 1)],                     # decode rows
                 [("a", 301, 1), ("b", 71, 40), ("c", 1, 130)],                  # decode + prompt chunks
                 [("a", 302, 6), ("b", 111, 1), ("c", 131, 3)]]                  # verify windows
        for seed, rows in enumerate(steps, 2):
            batch = self.write(pool, rows, seed)
            keys, values = self.written
            q = torch.randn(batch.tokens, HQ, D, generator=torch.Generator().manual_seed(seed)).cuda()
            q = q.to(torch.bfloat16)
            found, want = fast.forward(0, q, pool, batch), plain.forward(0, q, pool, batch)
            error = float((found.float() - want.float()).abs().max())
            self.assertLess(error, bound, (dtype, page, rows))
            for r in range(batch.rows):
                sequence, first, count, position = batch.row(r)
                alone = StepBatch.build([(sequence, position, count)], "cuda")
                # A step of this row alone: its keys and values written again (the same codes; a prompt row's kept
                # fresh for its attention as in the step).
                pool.write(0, pool.reserve(alone), keys[first:first + count], values[first:first + count])
                single = fast.forward(0, q[first:first + count], pool, alone)
                self.assertTrue(self.torch.equal(single, found[first:first + count]), (dtype, sequence))

    def test_16_bit_and_fp8_pages_read_as_sdpa_does(self):
        self.compare("bf16", 16)
        self.compare("fp8", 64)
        self.compare("fp16", 48)

    def test_no_library_kernel_by_default_and_flashinfer_on_request(self):
        form = self.assemble(USE)["attention"].device_form
        self.assertEqual(form._wrappers, {})            # option flashinfer off: nothing of FlashInfer is made
        self.assertIsNone(form.flashinfer_workspace)
        if _flashinfer():                               # the yardstick, on request: the same results within bounds
            self.compare("bf16", 16, flashinfer="auto")
            self.compare("fp8", 64, flashinfer="auto")

    def test_row_scaled_pages_read_as_sdpa_does(self):
        for dtype in ("int8", "fp8:row"):
            self.compare(dtype, 64)
            self.compare(dtype, 16)
            self.compare(dtype, 48)
        if self.assemble(USE)["attention"].device_form.kernels is not None:
            # INT8 queries: one more rounding of each query, within a few hundredths of the exact attention here.
            self.compare("int8", 64, int8_qk=True, bound=0.06)

    def test_flashinfer_plans_a_mixed_step_once_whatever_its_layers(self):
        if not _flashinfer():
            self.skipTest("FlashInfer is not installed")
        torch = self.torch
        state = {"page": 16}
        kv = self.assemble(USE, use_options={"attention-state": state})["kv_cache"]
        attention = self.assemble(USE, options={"decode": "split", "flashinfer": "auto"},
                                  use_options={"attention-state": state})["attention"]
        plans = []
        for kind, wrapper in attention.device_form._wrappers.items():
            def counted(*args, _plan=wrapper.plan, _kind=kind, **kwargs):
                plans.append(_kind)
                return _plan(*args, **kwargs)
            wrapper.plan = counted
        pool = kv.pool([0, 1, 2], HKV, D, 4096, "cuda", "bf16")
        for s in "ab":
            pool.open(s)
        batch = StepBatch.build([("a", 0, 40), ("b", 0, 1)], "cuda")    # a prompt row and a window row
        slots = pool.reserve(batch)
        g = torch.Generator().manual_seed(4)
        q = torch.randn(batch.tokens, HQ, D, generator=g).cuda().to(torch.bfloat16)
        for layer in range(3):
            pool.write(layer, slots, torch.randn(batch.tokens, HKV, D, generator=g).cuda(),
                       torch.randn(batch.tokens, HKV, D, generator=g).cuda())
            attention.forward(layer, q, pool, batch)
        # The window row's metadata and the prompt row's plan are each made at the step's first layer only.
        self.assertEqual(plans, [128])

    def test_the_split_reader_serves_every_page_type_on_request(self):
        for dtype in ("bf16", "fp8"):
            self.compare(dtype, 16, decode="split")

    def test_a_capture_that_would_plan_or_copy_from_the_host_is_refused(self):
        torch = self.torch
        pool, fast, plain = self.engine("bf16", 16)        # no step buffers: a step's rows are planned on the host
        for s in "ab":
            pool.open(s)
        self.write(pool, [("a", 0, 100), ("b", 0, 30)], 1)
        batch = self.write(pool, [("a", 100, 1), ("b", 30, 1)], 2)
        q = torch.randn(2, HQ, D, generator=torch.Generator().manual_seed(3)).cuda().to(torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "host"):
            with torch.cuda.graph(graph):
                fast.forward(0, q, pool, batch)

    def test_a_pool_with_step_buffers_reads_every_page_type_as_sdpa_does(self):
        for dtype in ("bf16", "fp8", "fp8:row", "int8"):
            self.compare(dtype, 64, step_rows=8)

    def test_a_decode_step_over_step_buffers_replays_from_a_cuda_graph(self):
        torch = self.torch
        for dtype in ("bf16", "int8"):
            pool, fast, plain = self.engine(dtype, 64, step_rows=4)
            for s in "ab":
                pool.open(s)
            self.write(pool, [("a", 0, 200), ("b", 0, 90)], 1)
            batch = self.write(pool, [("a", 200, 1), ("b", 90, 1)], 2)
            q = torch.randn(2, HQ, D, generator=torch.Generator().manual_seed(3)).cuda().to(torch.bfloat16)
            fast.forward(0, q, pool, batch)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fast.forward(0, q, pool, batch)
            for step in range(3):   # the next steps: reserve() and the KV writes outside the graph, then a replay
                batch = self.write(pool, [("a", 201 + step, 1), ("b", 91 + step, 1)], 10 + step)
                graph.replay()
                self.assertTrue(torch.equal(out, fast.forward(0, q, pool, batch)), (dtype, step))
                error = float((out.float() - plain.forward(0, q, pool, batch).float()).abs().max())
                self.assertLess(error, 0.02, (dtype, step))
