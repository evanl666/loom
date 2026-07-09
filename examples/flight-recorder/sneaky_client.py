"""A minimal agent client, API-shaped like any real one.

It does whatever the model says: if the response carries a tool call, it
"executes" it. Run it through ``loom record --deny 'read_file(*.env*)'`` and
the dangerous call never reaches this code -- the model's response arrives
rewritten, with a readable notice where the tool call used to be.
"""

import json
import os
import urllib.request

BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")


def call_model(messages):
    req = urllib.request.Request(
        BASE + "/v1/messages",
        data=json.dumps({"model": "offline-demo", "max_tokens": 512,
                         "messages": messages}).encode(),
        headers={"content-type": "application/json", "x-api-key": "demo-key"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


messages = [{"role": "user", "content": "Tidy up the repo, whatever it takes."}]
response = call_model(messages)

for block in response["content"]:
    if block["type"] == "text":
        print(f"model says: {block['text']}")
    elif block["type"] == "tool_use":
        print(f"EXECUTING {block['name']}({json.dumps(block['input'])})  <- unshielded!")
