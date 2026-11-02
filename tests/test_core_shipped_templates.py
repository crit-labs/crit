"""Chat templates packages give Core for model families whose checkpoints carry none (crit_core.registry
chat_templates, by config.json model_type): the text layer renders a model's own template, else a given one, else the
registered one, which then joins the model's identity (offline). crit-eval tests DeepSeek V4's, the one crit ships."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"): sys.path.insert(0, str(ROOT / _folder))

import core_text_fixtures as fx  # noqa: E402
from crit_core import registry  # noqa: E402
from crit_core.files import MappingFiles  # noqa: E402
from crit_core.text import (ChatTemplate, ModelTextError, TextError, chat_template_sources, load_model_text,  # noqa: E402
                            shipped_template)

FAMILY = "crit_test_family"   # a model_type no package gives a template: the test registers its own
TEMPLATE = ("{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
            "{% if add_generation_prompt %}<reply>{% endif %}")
_folder = None


def setUpModule():
    global _folder
    _folder = tempfile.TemporaryDirectory()
    path = Path(_folder.name) / "family.jinja"
    path.write_text(TEMPLATE, encoding="utf-8")
    registry.register("chat_templates", {FAMILY: path, "crit_test_broken": 4})


def tearDownModule():
    _folder.cleanup()


def mapping(files):
    return MappingFiles({name: fx.file_bytes(value) for name, value in files.items()})


def model_files(model_type=FAMILY):
    """A small model without a chat template whose config.json names `model_type`."""
    files = fx.llama_files()
    del files["chat_template.jinja"]
    files["config.json"] = {"model_type": model_type, "eos_token_id": 1}
    return files


class ShippedTemplateTests(unittest.TestCase):
    def test_a_registered_template_by_model_type(self):
        self.assertEqual(shipped_template(FAMILY), ("family.jinja", TEMPLATE))
        for model_type in ("crit_test_other", None, 4):
            self.assertIsNone(shipped_template(model_type))
        with self.assertRaisesRegex(TextError, r"registered for 'crit_test_broken' is not a path \(int\)"):
            shipped_template("crit_test_broken")

    def test_a_checkpoint_without_a_template_renders_the_registered_one(self):
        text = load_model_text(mapping(model_files()), clock=None)
        self.assertEqual((text.template_origin, list(text.templates)), ("crit", ["default"]))
        sha256 = ChatTemplate(TEMPLATE, clock=None).sha256
        self.assertEqual(text.templates["default"].sha256, sha256)
        self.assertEqual(text.render([{"role": "user", "content": "Hi"}], add_generation_prompt=True),
                         "<|begin_of_text|><user>Hi<reply>")
        # The template is no file of the model's, so the identity names it.
        self.assertEqual(text.identity.row()["template"], {"origin": "crit", "sha256": {"default": sha256}})
        self.assertEqual(chat_template_sources(mapping(model_files()))[2], "family.jinja")
        other = load_model_text(mapping(model_files("crit_test_other")), clock=None)
        self.assertEqual((other.template_origin, len(other.templates), other.identity.template), (None, 0, None))
        with self.assertRaisesRegex(ModelTextError, "no chat template"):
            other.render([{"role": "user", "content": "Hi"}])

    def test_a_given_template_renders_a_model_without_one_and_never_replaces_the_models_own(self):
        given = "{{ bos_token }}{% for m in messages %}[{{ m.content }}]{% endfor %}"
        text = load_model_text(mapping(model_files()), clock=None, chat_template=given)
        self.assertEqual(text.template_origin, "given")
        self.assertEqual(text.render([{"role": "user", "content": "Hi"}]), "<|begin_of_text|>[Hi]")
        self.assertNotEqual(text.identity.sha256, load_model_text(mapping(model_files()), clock=None).identity.sha256)
        own = mapping(fx.llama_files())
        with self.assertRaisesRegex(ModelTextError, "carries its own chat template"):
            load_model_text(own, clock=None, chat_template=given)
        with self.assertRaisesRegex(TextError, "template source string"):
            chat_template_sources(own, chat_template=b"bytes")
        # A model's own template keeps the identity it had: no template entry joins it.
        text = load_model_text(mapping(fx.llama_files()), clock=None)
        self.assertEqual((text.template_origin, text.identity.template), ("model", None))
        self.assertNotIn("template", text.identity.row())


if __name__ == "__main__":
    unittest.main()
