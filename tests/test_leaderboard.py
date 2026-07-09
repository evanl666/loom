"""loom leaderboard: rank agents by safety/cost/risk."""

from loom import Agent, tool
from loom.cli import main
from loom.leaderboard import compute_leaderboard, leaderboard_text
from loom.providers import ModelResponse, ScriptedProvider, ToolCall


@tool
def Read(file_path: str) -> str:
    "read"
    return "data"


@tool
def issue_refund(amount: int, order_id: str) -> str:
    "refund"
    return "ok"


def _board(tmp_path):
    for i in range(3):
        Agent(model=ScriptedProvider([
            ModelResponse(tool_calls=[ToolCall("t", "Read", {"file_path": "a.py"})],
                          stop_reason="tool_use"),
            ModelResponse(text="done", usage={"input_tokens": 500, "output_tokens": 50}),
        ]), tools=[Read, issue_refund]).run("go").save(
            str(tmp_path / "safe" / f"r{i}.loom.json") if (tmp_path / "safe").mkdir(
                parents=True, exist_ok=True) or True else "")
    for i in range(3):
        Agent(model=ScriptedProvider([
            ModelResponse(tool_calls=[ToolCall("t", "issue_refund",
                                               {"amount": 500, "order_id": "A"})],
                          stop_reason="tool_use"),
            ModelResponse(text="done", usage={"input_tokens": 3000, "output_tokens": 200}),
        ]), tools=[Read, issue_refund]).run("go").save(
            str(tmp_path / "risky" / f"r{i}.loom.json") if (tmp_path / "risky").mkdir(
                parents=True, exist_ok=True) or True else "")
    return str(tmp_path)


def test_leaderboard_ranks_safest_first(tmp_path):
    rows = compute_leaderboard(_board(tmp_path))
    assert [r["agent"] for r in rows] == ["safe", "risky"]
    assert rows[0]["safety"] == 100 and rows[0]["risky_rate"] == 0
    assert rows[1]["safety"] < 100 and rows[1]["risky_rate"] == 100
    assert rows[1]["cost"] > rows[0]["cost"]
    assert "safest: safe" in leaderboard_text(rows)


def test_cli_leaderboard_html(tmp_path, capsys):
    d = _board(tmp_path)
    out = tmp_path / "board.html"
    assert main(["leaderboard", d, "--html", str(out)]) == 0
    page = out.read_text()
    assert "Agent safety leaderboard" in page and "safe" in page and "risky" in page


def test_leaderboard_html_escapes_agent_name():
    from loom.leaderboard import leaderboard_html

    html = leaderboard_html([{"agent": "<script>alert(1)</script>", "runs": 1,
                              "safety": 50, "cost": 10, "risky_rate": 0,
                              "blocked": 0, "failure_rate": 0}])
    assert "<script>alert(1)</script>" not in html      # escaped, no XSS
    assert "&lt;script&gt;" in html
