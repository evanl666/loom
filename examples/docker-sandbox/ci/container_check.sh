#!/usr/bin/env bash
# Prove `loom record --container` records a real containerized agent.
# The agent runs inside Docker; the API is routed to the host proxy; the
# exchange lands as a loom trace on the host.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf ctr && mkdir ctr

# A fake upstream on the host (the "API"), reachable from the container via
# the proxy only.
python3 fake_api.py --port 9100 > ctr/fake.log 2>&1 &
FAKE=$!
trap 'kill $FAKE 2>/dev/null || true' EXIT
sleep 0.5

# The agent: a tiny inline python that talks to $ANTHROPIC_BASE_URL. It runs
# INSIDE the container; loom sets that env to the host proxy.
cat > ctr/agent.py <<'PY'
import json, os, urllib.request
url = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages"
req = urllib.request.Request(url,
    data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
    headers={"content-type": "application/json"}, method="POST")
print(json.loads(urllib.request.urlopen(req, timeout=15).read())["content"][0]["text"])
PY

# The proxy runs on the host and reaches the host fake api via loopback; the
# CONTAINER reaches the proxy via host.docker.internal (loom sets that env
# inside the container automatically).
loom record --container --container-image python:3.12-slim \
    --save ctr/session.loom.json \
    --target http://127.0.0.1:9100 \
    -- python /workspace/ctr/agent.py

python3 - <<'PY'
import json
d = json.load(open("ctr/session.loom.json"))
assert d["recorded_via"] == "proxy", d.get("recorded_via")
assert d["output"] == "recorded through the only door", d["output"]
print("container record verified:", len(d["log"]), "effect(s)")
PY
echo "loom record --container: agent ran in Docker, exchange recorded on the host"
