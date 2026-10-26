# DeepSeek V4 encoder goldens

The eight files beside this notice are unmodified copies of the test fixtures of DeepSeek's official DeepSeek V4
encoder (`encoding_dsv4.py`, with `test_encoding_dsv4.py` and `tests/`, which DeepSeek publishes with the DeepSeek V4
checkpoints in place of a chat template). `test_output_N.txt` is what `encode_messages` renders for
`test_input_N.json`: cases 1 to 3 in thinking mode (case 1 with the input's `tools` on its first message), case 4 in
chat mode. `tests/test_core_shipped_templates.py` checks that crit's port,
`core/python/crit_core/text/templates/deepseek-v4.jinja`, renders them exactly.

| File | sha256 |
| --- | --- |
| test_input_1.json | 10e0c074c977c3a80daab758af28219c6b1c2bd7f3f5cf2890c84b361cc32897 |
| test_input_2.json | c44ae0db20fafff38a6021e8068d7ed6e28605d76cafd20be80b03398509f447 |
| test_input_3.json | 37bf8ef95e0411ea5f411be0b02fbafec7363438b6ccefddca0c52ec9aeaf69a |
| test_input_4.json | c45bbd0a1b7a2f75033d8db4ba74ee5b7653bd01114045a52f79d3b387663465 |
| test_output_1.txt | 9b366d9d2eac842a6e890594aac0b58648e5623717202b33497afadf03e26540 |
| test_output_2.txt | ca66b01a1ac3a204bb032c928fb607d4877171e998a50f7f89f39fa821b75665 |
| test_output_3.txt | b3b1cd8748b7b90d3c6be6da3f786f12e4d70be073bd445ea162dfad4dc01a64 |
| test_output_4.txt | 60e1643840ba9e4aeede450feb7b0498fa66ee24e4a939d48855ce04ec6fc375 |

They come with `encoding_dsv4.py` sha256 bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0
and `test_encoding_dsv4.py` sha256 c2bc54c4c934f5c64096bd9c555efa7d1ddf179c1eff58f01ceb2dcd60adcf28.

## License

DeepSeek releases DeepSeek V4, its encoder and these fixtures under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
