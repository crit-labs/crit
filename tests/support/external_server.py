"""A tiny external engine for crit run's external candidates: an OpenAI-compatible chat server on the standard library.

Its tokens are characters. The greedy reply is REPLY cut to max_tokens, one piece per token, --delay-ms apart
(--fast halves it); stop strings cut it. --render chars counts the prompt as the characters of the messages' contents,
--render text as Eval's test tokenizer renders it ('<role>content' per message, then '<a>'). It writes its own process
id, and with --child the id of a child process it starts and leaves running, one per line to --pids, so a test can
check the whole process tree is gone once crit stops it, and its context (--context) to --pids too.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPLY = "tides rise and fall " * 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--render", choices=("chars", "text"), default="chars")
    parser.add_argument("--delay-ms", type=float, default=2.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--context", type=int)
    parser.add_argument("--pids")
    parser.add_argument("--exit", type=int, help="Exit with this code before serving")
    args = parser.parse_args()
    if args.exit is not None:
        print("toy engine: refusing to start", flush=True)
        raise SystemExit(args.exit)
    lines = [f"server {os.getpid()}", f"context {args.context}", f"fast {args.fast}"]
    if args.child:
        child = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(1)"])
        lines.append(f"child {child.pid}")
    if args.pids:
        with open(args.pids, "a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    delay = args.delay_ms / 1000 / (2 if args.fast else 1)

    def prompt_tokens(messages):
        if args.render == "chars":
            return sum(len(m.get("content") or "") for m in messages)
        return len("".join("<%s>%s" % (m["role"], m.get("content") or "") for m in messages) + "<a>")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def send(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self.send(200, {"object": "list", "data": [{"id": "toy", "object": "model"}]})
            elif self.path == "/health":
                self.send(200, {"status": "ok", "max_concurrent_requests": 4})
            else:
                self.send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path != "/v1/chat/completions":
                self.send(404, {"error": {"message": f"{self.path} is not served"}})
                return
            reply, finish = REPLY[:int(body.get("max_tokens") or 16)], "length"
            stops = body.get("stop") or []
            for stop in [stops] if isinstance(stops, str) else stops:
                if stop and stop in reply:
                    reply, finish = reply[:reply.find(stop)], "stop"
            usage = {"prompt_tokens": prompt_tokens(body["messages"]), "completion_tokens": len(reply),
                     "total_tokens": prompt_tokens(body["messages"]) + len(reply)}
            if not body.get("stream"):
                time.sleep(delay * len(reply))
                self.send(200, {"id": "toy", "object": "chat.completion", "model": "toy", "usage": usage,
                                "choices": [{"index": 0, "finish_reason": finish,
                                             "message": {"role": "assistant", "content": reply}}]})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def event(value):
                self.wfile.write(b"data: " + json.dumps(value).encode() + b"\n\n")
                self.wfile.flush()
            try:
                for piece in reply:
                    time.sleep(delay)
                    event({"id": "toy", "object": "chat.completion.chunk", "model": "toy",
                           "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                event({"id": "toy", "object": "chat.completion.chunk", "model": "toy",
                       "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                event({"id": "toy", "object": "chat.completion.chunk", "model": "toy", "choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass   # the client went away: the request is cancelled

    print(f"toy engine: serving on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
