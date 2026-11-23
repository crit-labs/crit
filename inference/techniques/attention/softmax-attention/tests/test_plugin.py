import math

from crit_plugins.testing import cases_for
from crit_plugins.testing import PluginTestCase

CASES = cases_for(__file__)

USE = {"kv_cache": "attention-state"}


class SoftmaxAttentionTests(PluginTestCase):
    def test_it_has_the_attention_interface(self):
        self.assertImplements(self.assemble(USE))

    def test_one_visible_row_returns_its_value_and_masks_hold(self):
        parts = self.assemble(USE)
        kv, attention = parts["kv_cache"], parts["attention"]
        kv.open(1, 2, 2, 8)
        kv.stage(1, [[1.0, 0.0], [0.0, 1.0]], [[3.0, 4.0], [5.0, 6.0]])
        kv.commit(1)
        self.assertEqual(attention.attend(1, [[0.3, -0.2]], [0]), [[3.0, 4.0]])
        both = attention.attend(1, [[1.0, 0.0]], [1])[0]
        weight = 1 / (1 + math.exp(-1 / math.sqrt(2)))
        self.assertAlmostEqual(both[0], weight * 3 + (1 - weight) * 5)

    def test_the_attention_cases_run(self):
        out = CASES["attention"](self.assemble(USE))
        self.assertEqual([len(out[k]) for k in ("decode", "prefill", "scaled", "pending")], [1, 4, 2, 3])

    def test_memory_declares_exactly_what_the_device_form_allocates(self):
        from crit_plugins.assemble import load_module
        module = load_module(self.registry()["softmax-attention"])
        base = {"kernels": "auto", "workspace_mib": 4, "flashinfer": "auto", "flashinfer_mib": 16, "decode": "split",
                "prompt_reader": "prefill"}
        cuda = self.device() == "cuda"
        for change in ({}, {"decode": "flashinfer"}, {"flashinfer": "off"}, {"kernels": "torch"},
                       {"prompt_reader": "decode"}):
            options = dict(base, **change)
            declared = module.memory(None, options)
            torch = __import__("torch") if cuda else None
            if cuda:
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
            # attention-state's fresh buffer is its own declaration: here none, so only the attention's are counted
            form = self.assemble(USE, options=options, use_options={"attention-state": {"fresh_mib": 0}})[
                "attention"].device_form
            if not cuda:
                self.assertEqual(declared, {"device": 0, "host": 0})
                continue
            torch.cuda.synchronize()
            self.assertEqual(torch.cuda.memory_allocated() - before, declared["device"], change)
            pinned = sum(w._pin_memory_int_workspace_buffer.numel() for w in form._wrappers.values())
            self.assertEqual(pinned, declared["host"], change)
            if change.get("kernels") == "torch":
                self.assertEqual(declared, {"device": 0, "host": 0})
            form.close()

    def test_prompt_reader_auto_follows_the_level_the_engine_asks_of_its_prompt_rows(self):
        from crit_plugins import kernels
        self.addCleanup(kernels.use_exactness, None)

        def made(asked, **options):
            kernels.use_exactness(asked and {"decode": "row-exact", "prompt": asked})
            return self.assemble(USE, options=options)["attention"]
        # crit's kernel reads the pages: reader split's own prompt reader keeps a token's bits at every chunk length
        kernel = made(None).device_form.kernels is not None and self.device() == "cuda"
        for asked, reader in ((None, "prefill"), ("repeatable", "prefill"),
                              ("row-exact", "prefill" if kernel else "decode"), ("exact", "decode"),
                              ("declared", "decode")):
            self.assertEqual(made(asked).device_form.prompt_reader, reader, asked)
        self.assertEqual(made("row-exact", reader="flash").device_form.prompt_reader, "decode")   # a flash prompt chunk
        # What the rows keep, for Core to compare with what is asked.
        self.assertEqual(made("exact").exactness(), {"decode": "row-exact", "prompt": "exact"})
        self.assertEqual(made(None).exactness(), {"decode": "row-exact", "prompt": "row-exact"} if kernel else {})
        self.assertEqual(made(None, reader="flash").exactness(),
                         {"decode": "row-exact", "prompt": "repeatable"} if kernel else {})
        # An option set to one reader is that reader, whatever is asked.
        self.assertEqual(made("exact", prompt_reader="prefill").device_form.prompt_reader, "prefill")
        self.assertEqual(made("repeatable", prompt_reader="decode").device_form.prompt_reader, "decode")
        with self.assertRaisesRegex(ValueError, "prompt_reader decode \\(auto, for the exact prompt rows the "
                                                "configuration's exactness asks\\).*int8_qk"):
            made("exact", int8_qk=True)

    def test_its_facts_say_how_this_gpu_reads_wide_windows(self):
        attention = self.assemble(USE)["attention"]
        facts = attention.facts()
        self.assertEqual(facts["served_vs_scored"], "within the readers' rounding")   # prompt_reader prefill
        if attention.device_form.kernels is None or self.device() != "cuda":
            self.assertNotIn("split_kv_wide_windows", facts)
        else:
            self.assertRegex(facts["split_kv_wide_windows"], r"^(64-row 4-warp|32-row 2-warp) tiles")
