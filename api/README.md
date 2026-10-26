# crit API

`crit_api` is `crit serve`: it loads one sealed candidate engine, keeps it resident and answers
HTTP requests until stopped. Any OpenAI client can use the engine, and the same engine also speaks
the Anthropic Messages, OpenAI Responses and Ollama APIs. `crit run ... eval` measures every
candidate through this server, so what Eval reports is what clients get. The full reference is
[Serving](../docs/SERVING.md).

```sh
python -m pip install -e .        # from the crit checkout
crit serve --workspace /path/to/runs
# Serving RUN_ID/CANDIDATE_ID at http://127.0.0.1:8000/v1
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
reply = client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hello"}])
```

## Endpoints

| Endpoint | What it does |
| --- | --- |
| `POST /v1/chat/completions` | Chat, JSON or streamed: tool calls, reasoning, structured output, log-probabilities, stop strings |
| `POST /v1/completions` | Raw-prompt completions, for engines that declare them |
| `POST /v1/messages` | The Anthropic Messages API, so Claude Code needs only a base URL |
| `POST /v1/responses` | The OpenAI Responses API, statelessly, for clients such as Codex |
| `/api/chat`, `/api/generate`, `/api/tags`, ... | The Ollama API, for local chat UIs |
| `GET /v1/models`, `GET /health`, `GET /metrics` | Served models and their context, readiness, Prometheus metrics |
| `POST /v1/crit/score`, `/tokenize`, `/v1/crit/abort`, `/v1/crit/load`, `/v1/crit/unload` | Fixed-text scores, the engine's tokenizer, cancelling a request, loading and unloading weights |

## Common options

| Option | Purpose |
| --- | --- |
| `--run ID`, `--candidate ID` | Which run and candidate to serve (default: the only run, its selected candidate) |
| `--models PATH` | Serve several named models from several runs on one server |
| `--host`, `--port` | Listen address, default `127.0.0.1:8000` |
| `--api-key KEY`, `--ssl-certfile PATH` | Require a bearer key; serve HTTPS |
| `--idle-unload SECONDS` | Free the GPU when idle; the next request loads the engine again |
| `--write-launchers` | Write `serve.cmd` and `serve.sh` with the same options, then exit |

`crit serve --help` lists every option.

## How it is built

The server uses only the standard library and `crit_core`. A supervisor ([residency.py](crit_api/residency.py))
runs the engine in its own resident host process ([host.py](crit_api/host.py)) and talks to it over one
authenticated loopback channel ([channel.py](crit_api/channel.py)). The engine's process tree is retired on
stop, timeout, client death or a failed request, the server keeps running, and the next request loads a fresh
engine. Requests are admitted up to the engine's capacity
and queued with bounds ([admission.py](crit_api/admission.py)); each dialect is one module over the same engine
([server.py](crit_api/server.py), [anthropic.py](crit_api/anthropic.py), [responses.py](crit_api/responses.py),
[ollama.py](crit_api/ollama.py)). Engines plug in through the
[adapter contract](../docs/SERVING.md#adapter-contract).
