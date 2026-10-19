"""crit_eval.prepare: the scored turn of a template whose generation prompt holds more than a history turn."""
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import prepare  # noqa: E402

EXTRA = "<think></think>"   # what this template's generation prompt adds and its history turns strip


def render(messages, tools=None, add_generation_prompt=False, enable_thinking=False):
    text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" if m["role"] != "assistant"
                   else f"<assistant>{m['content']}</assistant>" for m in messages)
    return text + ("<assistant>" + EXTRA if add_generation_prompt else "")


class ReplyStartTests(unittest.TestCase):
    def setUp(self):
        self.model = types.SimpleNamespace(render=render)

    def test_a_reply_that_begins_like_the_prompts_extra_text_keeps_its_characters(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<thanks>"}]
        rendered = prepare.chat_text(self.model, messages, False)
        prompt = prepare.chat_text(self.model, messages[:-1], True)
        opening = prepare.reply_opening(self.model, messages)
        self.assertEqual(opening, "<user>hi</user><assistant>")
        self.assertEqual(prepare.generated_part(rendered, prompt, opening), "<thanks></assistant>")
        # Matching characters alone would have eaten the reply's leading "<th".
        self.assertEqual(prepare.generated_part(rendered, prompt), "anks></assistant>")

    def test_a_template_that_rewrites_the_content_falls_back_to_the_shared_text(self):
        model = types.SimpleNamespace(render=lambda messages, **options: render(
            [dict(m, content=m["content"].replace("\u0000", "")) for m in messages], **options))
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        self.assertIsNone(prepare.reply_opening(model, messages))


if __name__ == "__main__":
    unittest.main()
