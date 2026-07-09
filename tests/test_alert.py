"""loom alert: fleet thresholds with webhook."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from loom import Agent, tool
from loom.alert import evaluate, post_webhook
from loom.cli import main
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def get_customer(id: int) -> str:
    "lookup"
    return "Jane"


@tool
def send_email(to: str) -> str:
    "email"
    return "sent"


def _corpus(tmp_path):
    # a PII -> email run (the leak-path metric) that also fails
    r = Agent(model=ScriptedProvider([
        ModelResponse(tool_calls=[ToolCall("t", "get_customer", {"id": 1})], stop_reason="tool_use"),
        ModelResponse(tool_calls=[ToolCall("t2", "send_email", {"to": "x"})], stop_reason="tool_use"),
        ModelResponse(text="done"),
    ]), tools=[get_customer, send_email]).run("go")
    d = r.to_dict()
    d["stop_reason"] = "budget"
    (tmp_path / "leak.loom.json").write_text(json.dumps(d))
    Agent(model=ScriptedProvider([ModelResponse(text="42")])).run("q").save(
        str(tmp_path / "clean.loom.json"))
    return str(tmp_path)


def test_evaluate_breaches_and_ordering(tmp_path):
    import glob
    d = _corpus(tmp_path)
    results, metrics = evaluate(glob.glob(d + "/*.loom.json"),
                                {"alerts": [{"metric": "failure_rate", "max": 10},
                                            {"metric": "pii_to_comm_paths", "max": 0},
                                            {"metric": "blocked_actions", "max": 5}]})
    assert metrics["pii_to_comm_paths"] == 1
    breached = [r["metric"] for r in results if r["breached"]]
    assert set(breached) == {"failure_rate", "pii_to_comm_paths"}
    assert results[0]["breached"] is True          # breached first


def test_unknown_metric_is_a_breach(tmp_path):
    import glob
    d = _corpus(tmp_path)
    results, _ = evaluate(glob.glob(d + "/*.loom.json"),
                          {"alerts": [{"metric": "nope", "max": 1}]})
    assert results[0]["breached"] and "unknown metric" in results[0]["error"]


def test_webhook_posts_slack_payload(tmp_path):
    got = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ok = post_webhook(f"http://127.0.0.1:{srv.server_address[1]}/",
                      [{"metric": "failure_rate", "value": 33, "max": 10}], runs=3)
    srv.shutdown()
    assert ok and "loom alert" in got["body"]["text"] and "failure_rate" in got["body"]["text"]


def test_cli_alert_gates(tmp_path, capsys):
    d = _corpus(tmp_path)
    assert main(["alert", d, "--max", "failure_rate=10"]) == 1
    assert "🚨" in capsys.readouterr().out
    assert main(["alert", d, "--max", "failure_rate=90"]) == 0
