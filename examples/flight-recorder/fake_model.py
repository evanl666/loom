"""An offline stand-in for api.anthropic.com so the shield act needs no key.

Plays a model that immediately goes for the credentials -- the proxy's job is
to make sure the client never sees that tool call.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RESPONSES = [
    {
        "id": "msg_demo_1", "type": "message", "role": "assistant", "model": "offline-demo",
        "content": [
            {"type": "text", "text": "Let me grab the credentials to work faster."},
            {"type": "tool_use", "id": "tu_1", "name": "read_file",
             "input": {"path": "/app/.env"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 42, "output_tokens": 19},
    },
    {
        "id": "msg_demo_2", "type": "message", "role": "assistant", "model": "offline-demo",
        "content": [{"type": "text", "text": "Understood -- proceeding without secrets."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 61, "output_tokens": 9},
    },
]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        response = RESPONSES[min(self.server.calls, len(RESPONSES) - 1)]
        self.server.calls += 1
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    calls = 0

    def server_bind(self):
        import socketserver

        socketserver.TCPServer.server_bind(self)  # skip slow getfqdn()
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8977)
    args = ap.parse_args()
    server = _Server(("127.0.0.1", args.port), _Handler)
    print(f"fake model on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
