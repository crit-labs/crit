"""chat-tool-template-v1, declared by packages sealed before ResidentChat parsed calls itself: crit serve reads it as
chat-tools-v1 (and chat-tool-stream-v1 when the package streams), with tool_choice auto or none."""
import unittest

import test_api  # noqa: F401  (puts the checkout's packages on sys.path)
from test_core_tools import TOOLS
from crit_api.options import (RequestError, TOOL_STREAM_CAPABILITY, TOOL_TEMPLATE_CAPABILITY, TOOLS_CAPABILITY,
                              serving_capabilities, tool_options)


class EarlierTemplateToolsTests(unittest.TestCase):
    def test_the_earlier_capability_offers_tools_and_tool_streams_but_only_auto_or_none(self):
        self.assertEqual(serving_capabilities(["chat-json-v1", TOOL_TEMPLATE_CAPABILITY]),
                         ["chat-json-v1", TOOL_TEMPLATE_CAPABILITY, TOOLS_CAPABILITY])
        streamed = serving_capabilities(["chat-stream-v1", TOOL_TEMPLATE_CAPABILITY])
        self.assertEqual(streamed[-2:], [TOOLS_CAPABILITY, TOOL_STREAM_CAPABILITY])
        self.assertEqual(serving_capabilities(["chat-json-v1"]), ["chat-json-v1"])
        for choice in ("auto", "none"):
            self.assertEqual(tool_options({"tools": TOOLS, "tool_choice": choice}, streamed)["tool_choice"], choice)
        for choice in ("required", {"type": "function", "function": {"name": "list"}}):
            with self.assertRaises(RequestError) as caught:
                tool_options({"tools": TOOLS, "tool_choice": choice}, streamed)
            self.assertEqual((caught.exception.param, caught.exception.code), ("tool_choice", "unsupported_parameter"))


if __name__ == "__main__":
    unittest.main()
