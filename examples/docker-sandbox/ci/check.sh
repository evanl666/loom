#!/usr/bin/env bash
# Stand up the jail, run the agent's checks, assert the trace landed.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf traces && mkdir traces

code=0
docker compose -f ci-compose.yml up \
    --abort-on-container-exit --exit-code-from agent || code=$?

if [ "$code" -ne 0 ]; then
    echo "--- proxy logs (for the postmortem):"
    docker compose -f ci-compose.yml logs proxy || true
fi
docker compose -f ci-compose.yml down -v --remove-orphans > /dev/null 2>&1 || true
[ "$code" -eq 0 ] || exit "$code"

# The exchange that went through the door is on disk as a loom trace.
python3 - << 'EOF'
import json
data = json.load(open("traces/session.loom.json"))
assert data["recorded_via"] == "proxy", data.get("recorded_via")
assert data["output"] == "recorded through the only door", data["output"]
print("trace recorded:", len(data["log"]), "effect(s)")
EOF
echo "docker sandbox verified: egress blocked, proxy recorded"
