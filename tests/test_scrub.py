"""loom scrub: secrets stay out of shared traces."""

import json

from loom.cli import main
from loom.scrub import scrub_obj, scrub_text, scrub_trace

ANTHROPIC_KEY = "sk-ant-api03-" + "a1B2" * 8
GITHUB_TOKEN = "ghp_" + "x9Yz" * 8


def test_known_key_shapes_are_redacted():
    text, found = scrub_text(
        f"key={ANTHROPIC_KEY} and token {GITHUB_TOKEN} and AKIAIOSFODNN7EXAMPLE"
    )
    assert ANTHROPIC_KEY not in text and GITHUB_TOKEN not in text
    assert "AKIA" not in text
    assert "[scrubbed:anthropic-key]" in text
    assert "[scrubbed:github-token]" in text
    assert found["anthropic-key"] == 1 and found["aws-key-id"] == 1


def test_credential_assignment_keeps_the_key_name():
    text, found = scrub_text("DB_PASSWORD: hunter2hunter2 in config")
    assert "hunter2" not in text
    assert text.startswith("DB_PASSWORD: [scrubbed:credential-assignment]")
    assert found["credential-assignment"] == 1


def test_env_var_credential_shapes_are_redacted():
    text, found = scrub_text("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY")
    assert "wJalr" not in text and found["credential-assignment"] == 1
    # ...while prose about tokens and ordinary identifiers survive
    for benign in ("the token bucket algorithm limits it", "primary_key: user_id_column"):
        assert scrub_text(benign) == (benign, {})


def test_pem_block_and_jwt_are_redacted():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKC\n-----END RSA PRIVATE KEY-----"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6y"
    text, found = scrub_text(pem + " " + jwt)
    assert "BEGIN RSA" not in text and "eyJ" not in text
    assert found["private-key"] == 1 and found["jwt"] == 1


def test_ordinary_text_and_hashes_pass_untouched():
    sha = "3f786850e387550fdab836ed7e6dc881de23001b" * 1  # hex: a hash, not a secret
    text, found = scrub_text(f"commit {sha} fixes the bug", aggressive=True)
    assert sha in text and not found


def test_aggressive_catches_high_entropy_mixed_tokens():
    token = "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5bC7d"
    text, found = scrub_text(f"opaque token {token}", aggressive=True)
    assert token not in text and found["high-entropy"] == 1
    # ...but only in aggressive mode
    text, found = scrub_text(f"opaque token {token}")
    assert token in text and not found


def test_scrub_obj_walks_nested_structures_without_mutating():
    obj = {"messages": [{"content": f"the key is {ANTHROPIC_KEY}"}], "n": 3}
    clean, found = scrub_obj(obj)
    assert ANTHROPIC_KEY in obj["messages"][0]["content"]  # input untouched
    assert ANTHROPIC_KEY not in clean["messages"][0]["content"]
    assert clean["n"] == 3
    assert sum(found.values()) == 1


def _trace_with_secret(tmp_path):
    path = tmp_path / "session.loom.json"
    path.write_text(json.dumps({"log": [], "output": f"your key: {ANTHROPIC_KEY}"}))
    return str(path)


def test_cli_scrub_writes_a_scrubbed_copy(tmp_path, capsys):
    path = _trace_with_secret(tmp_path)
    assert main(["scrub", path]) == 0
    out = path[: -len(".loom.json")] + ".scrubbed.loom.json"
    scrubbed = json.loads(open(out).read())
    assert ANTHROPIC_KEY not in scrubbed["output"]
    assert ANTHROPIC_KEY in open(path).read()  # original kept


def test_cli_scrub_in_place(tmp_path):
    path = _trace_with_secret(tmp_path)
    assert main(["scrub", path, "--in-place"]) == 0
    assert ANTHROPIC_KEY not in open(path).read()


def test_cli_scrub_check_is_a_ci_gate(tmp_path, capsys):
    dirty = _trace_with_secret(tmp_path)
    assert main(["scrub", dirty, "--check"]) == 1
    assert "1 secret(s)" in capsys.readouterr().err
    clean = tmp_path / "clean.loom.json"
    clean.write_text(json.dumps({"log": [], "output": "no secrets here"}))
    assert main(["scrub", str(clean), "--check"]) == 0
    assert open(dirty).read() and ANTHROPIC_KEY in open(dirty).read()  # check never writes


def test_scrub_trace_counts_across_the_whole_document():
    data = {
        "episodes": [f"use {GITHUB_TOKEN}"],
        "wire": [{"response": {"text": f"got {ANTHROPIC_KEY}"}}],
    }
    _, found = scrub_trace(data)
    assert found["github-token"] == 1 and found["anthropic-key"] == 1
