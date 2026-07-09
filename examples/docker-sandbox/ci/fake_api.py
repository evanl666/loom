"""A fake Anthropic API for the docker-sandbox CI check (binds 0.0.0.0)."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = {
    "id": "msg_ci", "type": "message", "role": "assistant", "model": "fake",
    "content": [{"type": "text", "text": "recorded through the only door"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 7, "output_tokens": 6},
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = json.dumps(ANSWER).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()
    print(f"fake api on 0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), _Handler).serve_forever()
